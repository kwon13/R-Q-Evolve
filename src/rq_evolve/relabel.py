"""Read a child's SKILL off the child, instead of trusting what it declared.

WHY. The archive's SKILL coordinate is whatever stage 1 wrote into the source,
and stage 1 is told which cell to aim at (``target_cell_injection``), so what it
writes is an instruction echo. Measured against a blind reference labelling of
60 candidates (gpt-5.6-luna, shown the family and the generator, shown no
existing label): the declared SKILL is one the problem actually requires 37.5%
of the time, and is the primary one 28.1% of the time. stage 2's own
``INFERRED_SKILL`` is barely better at 40.6% / 34.4%. Two thirds of the MAP's
skill axis is therefore noise, and cells fill with problems that do not belong
in them.

WHAT THIS DOES INSTEAD. Ask ONE binary question per skill -- "does the shortest
clean solution turn on X?" -- over the family AND the generator, recover P(YES)
from the logprob of a single token, subtract a per-skill offset, and take the
argmax. Measured on the same 32 well-posed references:

    declared SKILL (what ships)                     0.375 / 0.281
    stage-2 INFERRED_SKILL                          0.406 / 0.344
    binary x8, greedy YES/NO, ties broken at random 0.641 / 0.551
    binary x8, P(YES) + per-skill offset            0.812 / 0.750
    random                                          0.176 / 0.125
                              (required-set basis / primary-skill basis)

THREE THINGS THAT LOOK OPTIONAL AND ARE NOT.

*Binary, not eight-way.* Asking the same model to pick one of eight labels in a
single call, reading the full distribution off the letter's logprobs, scores
0.125 -- chance. Same model, same context, same information; only the question
shape differs, and it costs a factor of 6.5.

*The generator, not just the statement.* With the code hidden the same probe
drops from AUC 0.722 to 0.642 against the declared label. Seeing the code makes
the model REJECT wrong skills better -- P(YES) on a decoy falls 0.375 -> 0.293
while P(YES) on the declared skill does not move.

*The offset, not a z-score.* The judge's per-skill bias is a shift, not a
scale: casework sits 2.26 logits below construction. Subtracting a per-skill
mean buys +0.03; dividing by a per-skill standard deviation as well LOSES
0.125. Half-shrinking the offset (0.5x) beats the full subtraction on small
pools, which is what an outer iteration has.

The skill block goes LAST in the user turn on purpose: the eight calls then
share the system prompt, the family and the code as a prefix, so they cost one
prefill and eight single-token decodes. Measured on the step-160 policy, 32
candidates x 8 skills = 256 calls took 3.98 s at concurrency 32 -- 4.8% of the
82.5 s that iteration's evolve phase already spends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .concepts import SKILLS, GROUPS

SYSTEM_PROMPT_FILE = "relabel_system_prompt.txt"
USER_PROMPT_FILE = "relabel_user_prompt.txt"


@dataclass
class LabelOffsets:
    """Per-label logit offsets, carried across iterations as an EWMA.

    One outer iteration judges at most ``inner_iteration_batch_size`` children,
    and a mean over that few items is itself noisy: resampling the 220-item
    corpus, a 32-item centring pool scores 0.681 +/- 0.027 against 0.719 at 220.
    Half-shrinking recovers most of it (0.737 +/- 0.016) because the estimator
    errs toward no correction rather than toward a wrong one.

    Cold start is ``offset = 0``, i.e. plain argmax over P(YES), which measures
    0.719 on the full pool. The estimator degrades to something that works.
    """

    alpha: float = 0.3
    shrink: float = 0.5
    min_items: int = 8
    mean: dict[str, float] = field(default_factory=dict)
    count: dict[str, int] = field(default_factory=dict)

    def observe(self, scores: dict[str, float]) -> None:
        """Fold one child's eight logits into the running means.

        Called for EVERY judged child, including ones that later fail
        verification: the offsets describe the judge, not the survivors.
        """
        for skill, logit in scores.items():
            n = self.count.get(skill, 0)
            prev = self.mean.get(skill, logit)
            self.mean[skill] = prev + self.alpha * (logit - prev) if n else logit
            self.count[skill] = n + 1

    def correct(self, scores: dict[str, float]) -> dict[str, float]:
        return {
            skill: logit - self.shrink * self.mean.get(skill, 0.0)
            if self.count.get(skill, 0) >= self.min_items
            else logit
            for skill, logit in scores.items()
        }

    def to_dict(self) -> dict:
        return {"mean": dict(self.mean), "count": dict(self.count)}

    @classmethod
    def from_dict(cls, payload: dict | None) -> "LabelOffsets":
        out = cls()
        if payload:
            out.mean = {str(k): float(v) for k, v in (payload.get("mean") or {}).items()}
            out.count = {str(k): int(v) for k, v in (payload.get("count") or {}).items()}
        return out


def logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def p_yes_from_logprobs(top_logprobs: list[dict]) -> float | None:
    """Two-way probability from one token's logprobs.

    With ``allowed_token_ids`` restricted to the YES and NO ids this is exact.
    Without it the call still works: the YES-initial and NO-initial mass is
    summed out of whatever top-k came back and renormalised, which is what the
    offline measurements used.
    """
    yes = no = 0.0
    for entry in top_logprobs or []:
        token = str(entry.get("token", "")).strip().upper()
        if not token:
            continue
        weight = math.exp(float(entry.get("logprob", -math.inf)))
        if token.startswith("Y"):
            yes += weight
        elif token.startswith("N"):
            no += weight
    total = yes + no
    return (yes / total) if total > 0 else None


def choose_label(
    p_yes: dict[str, float], offsets: LabelOffsets
) -> tuple[str | None, float, dict[str, float]]:
    """(label, margin, corrected logits). ``label`` is None if nothing scored.

    ``margin`` is the gap to the runner-up in corrected logits. It is recorded
    rather than acted on: a low-margin child is filed in its argmax cell like
    any other, but the number is there for a later rule that wants to treat a
    coin-flip differently from a decision.
    """
    scored = {s: logit(p) for s, p in p_yes.items() if p is not None}
    if not scored:
        return None, 0.0, {}
    corrected = offsets.correct(scored)
    ranked = sorted(corrected.items(), key=lambda kv: kv[1], reverse=True)
    margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 0.0
    return ranked[0][0], float(margin), corrected


def all_skills() -> tuple[str, ...]:
    return tuple(SKILLS)


def all_groups() -> tuple[str, ...]:
    return tuple(GROUPS)


SkillOffsets = LabelOffsets
GroupOffsets = LabelOffsets
choose_skill = choose_label
choose_group = choose_label
