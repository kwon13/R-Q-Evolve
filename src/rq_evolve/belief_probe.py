"""Belief/desire attribution as a falsifiable, Python-checkable experiment.

Motivation
----------
The v5 plan asked the planner for 22 fields and got behaviour descriptions back:
``failure_summary`` said "the solver incorrectly adds the equations" when the
addition was arithmetically correct -- the actual error was that the solver
*believed* adding two equations eliminates a variable and therefore dropped a
term. A behaviour description records that the trace went wrong; it cannot
record why the agent thought it was right, so nothing downstream could target
the belief.

Design
------
Under the intentional stance an attribution earns its keep by *predicting*
behaviour, not by sounding mentalistic. So each hypothesis here carries a Python
function computing the answer a solver holding it would produce. That makes the
attribution falsifiable without an LLM judge:

    correct answer  != belief-predicted answer   -> the probe can discriminate
    observed answer == belief-predicted answer   -> the attribution is confirmed

The registry owns the hypotheses, the probe each one is falsified by, and the
prediction. The planner therefore makes exactly one judgement that needs natural
language understanding -- *which* hypothesis the wrong trace exhibits -- and
quotes the span showing it. Everything else is derived and verified in Python,
because delegating more than one judgement to a small model reliably produced
echoed defaults rather than analysis.

A desire is not identified by a wrong answer but by what the solver is willing
to spend, so desire hypotheses expose ``response_signature`` instead: a check
over the response text rather than the final integer.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

BELIEF_SCHEMA_VERSION = 6

# One propositional attitude attributed to the solver.
Kind = str  # "belief" | "desire"


@dataclass(frozen=True, slots=True)
class BeliefHypothesis:
    """A falsifiable attribution plus the probe that tests it.

    ``predicted_wrong_answer`` returns the integer a solver holding this belief
    would report for one instance, or ``None`` when the instance cannot
    discriminate. ``response_signature`` detects a desire from the shape of the
    derivation instead of its result.
    """

    hypothesis_id: str
    kind: Kind
    proposition: str
    omitted_precondition: str
    probe_variant: str
    probe_rationale: str
    predicted_wrong_answer: Callable[[Mapping[str, Any]], int | None] | None = None
    response_signature: Callable[[str], bool] | None = None
    # How much out-of-sample support this route has. Recorded for reporting and
    # deliberately kept out of ``to_payload`` -- showing the planner which
    # option is already validated would tell it the answer and destroy the very
    # comparison the catalog exists to run.
    evidence_status: str = "untested"
    # How many seeds must be able to tell the route apart from the truth.
    #
    # A count, not a fraction. The unit of evidence is the *instance*: a route is
    # a claim about a functional form, and one instance only ever checks a single
    # predicted number, which a solver can hit by luck (1/7 on seven residues).
    # Several instances make the route predict several different numbers, and
    # only then does "follows the rule" separate from "said 3 by chance". Extra
    # rollouts on one instance re-check the same prediction and add nothing.
    #
    # A fraction was tried and was wrong: this family's discrimination rate is
    # 62% overall but swings from 20% to 90% across ten-seed blocks, so a
    # fractional bar rejects sound probes on block luck -- it gated out all ten
    # in_depth candidates of one run. The discriminating seeds are fixed by the
    # instances alone, before any rollout is read, so restricting scoring to them
    # is not selection on the outcome.
    min_discriminating_seeds: int = 3

    def to_payload(self) -> dict[str, Any]:
        """Prompt-safe view. The planner never sees the prediction function."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "kind": self.kind,
            "proposition": self.proposition,
            "omitted_precondition": self.omitted_precondition,
        }


# ---------------------------------------------------------------------------
# Predictions: what a solver holding the belief would actually compute.
# ---------------------------------------------------------------------------


def _linear_unweighted_sum_then_divide(data: Mapping[str, Any]) -> int | None:
    """Adds the equations as-is, then correctly divides by the multiplier.

    Discovered by mining the recorded rollouts rather than authored: pooled over
    both in_depth conditions this route matched 6 of 47 coherent errors against a
    permutation-null expectation of 1.6 (p = 0.002). It is silent when the
    division is not whole, which is why the hypothesis declares a minimum
    discriminating-seed count instead of requiring every seed.
    """
    rhs = data.get("rhs")
    multiplier = data.get("aggregate_multiplier")
    if not isinstance(rhs, (list, tuple)) or len(rhs) != 2 or not multiplier:
        return None
    total = int(rhs[0]) + int(rhs[1])
    multiplier = int(multiplier)
    if multiplier == 0 or total % multiplier != 0:
        return None
    return total // multiplier


def _linear_undivided(data: Mapping[str, Any]) -> int | None:
    """Finds the right combination but never divides by the multiplier."""
    rhs = data.get("rhs")
    weight = data.get("combination_weight")
    if not isinstance(rhs, (list, tuple)) or len(rhs) != 2 or weight is None:
        return None
    return int(weight) * int(rhs[0]) + int(rhs[1])


def _modular_inversion_skipped(data: Mapping[str, Any]) -> int | None:
    """Reduces correctly to ``m * (x+y+z) = rhs[0] + rhs[1]``, then stops.

    The only route to survive out-of-sample testing. On greedy decoding of 200
    fresh instances of the ``balanced`` variant it matched 22 of 66 coherent
    errors against a permutation-null expectation of 9.6 -- p = 0.0002, and
    p = 0.0015 after Bonferroni over all fifteen confirmatory tests.

    The trace evidence rules out the obvious reading. Only 1 of the 22 matching
    responses simply adds the congruences; 21 run an elimination. What they share
    is arriving at the combined right-hand side (the literal value of
    ``rhs[0] + rhs[1]`` appears in 22/22, against 68% of other errors and 74% of
    correct answers) and then not applying the multiplier's inverse ("inverse" is
    mentioned in 45% of them versus 68% of correct answers). So the route is a
    dropped *final* step, not a wrong opening move.
    """
    rhs = data.get("rhs")
    modulus = data.get("modulus")
    if not isinstance(rhs, (list, tuple)) or len(rhs) != 2 or not modulus:
        return None
    return (int(rhs[0]) + int(rhs[1])) % int(modulus)


def _modular_integer_division(data: Mapping[str, Any]) -> int | None:
    """Treats division modulo p as ordinary integer division."""
    rhs = data.get("rhs")
    modulus = data.get("modulus")
    multiplier = data.get("multiplier")
    if (
        not isinstance(rhs, (list, tuple))
        or len(rhs) != 2
        or not modulus
        or not multiplier
    ):
        return None
    return ((int(rhs[0]) + int(rhs[1])) // int(multiplier)) % int(modulus)


_ASSIGNMENT = re.compile(
    r"\b([xyz])\s*=\s*(-?\d+)\b(?!\s*[+\-*/])",
    re.IGNORECASE,
)


def asserts_individual_values(response: str) -> bool:
    """Detect a solver committing to unique x, y, z on an underdetermined system.

    This is the desire signature: the system has a free direction, so any claim
    of unique individual values is unfounded regardless of the final answer. It
    is read off the derivation, not the result, because a desire shows up in what
    the solver is willing to spend rather than in what it concludes.
    """
    return len({match.group(1).lower() for match in _ASSIGNMENT.finditer(response or "")}) >= 3


LINEAR_HYPOTHESES: tuple[BeliefHypothesis, ...] = (
    BeliefHypothesis(
        hypothesis_id="unweighted_sum_then_divide",
        kind="belief",
        proposition=(
            "The two printed equations may be added as they stand, and "
            "dividing that sum by the aggregate multiplier gives x + y + z."
        ),
        omitted_precondition=(
            "Only a *weighted* combination has coefficient vector equal to a "
            "multiple of (1, 1, 1); the plain sum does not, so dividing it is "
            "dividing the wrong quantity."
        ),
        probe_variant="asymmetric_combination",
        probe_rationale=(
            "Forces a hidden row weight of magnitude at least 2, which is "
            "exactly the factor this route drops, so the mistake is maximally "
            "separated from the correct answer."
        ),
        predicted_wrong_answer=_linear_unweighted_sum_then_divide,
        # Provisional. On greedy decoding it matched 3 of 26 coherent errors
        # against a chance expectation of 0.5 -- significant within the linear
        # family (Bonferroni 0.041) but not across all fifteen confirmatory
        # tests (0.206), and three observations is too few to rest on. Its
        # specificity is the encouraging part: 3 hits on asymmetric_combination
        # and zero on both balanced and heavy_division, which is exactly where
        # a dropped row weight should and should not show up.
        evidence_status="provisional_underpowered",
    ),
    BeliefHypothesis(
        hypothesis_id="division_step_skipped",
        kind="belief",
        proposition=(
            "Once the right combination of equations is found, its right-hand "
            "side is already x + y + z."
        ),
        omitted_precondition=(
            "The combination equals the aggregate multiplier times x + y + z, "
            "so it must still be divided by that multiplier."
        ),
        probe_variant="heavy_division",
        probe_rationale=(
            "Raises the aggregate multiplier to 4..7 so an undivided answer is "
            "far from the correct one and easy to tell apart."
        ),
        predicted_wrong_answer=_linear_undivided,
        evidence_status="refuted_out_of_sample",
    ),
    BeliefHypothesis(
        hypothesis_id="full_solution_required",
        kind="desire",
        proposition=(
            "An aggregate question must be answered by first pinning down x, y "
            "and z individually."
        ),
        omitted_precondition=(
            "The system is rank deficient, so individual values do not exist; "
            "only the aggregate is determined."
        ),
        probe_variant="balanced",
        probe_rationale=(
            "Every instance leaves a free direction, so any derivation that "
            "commits to unique individual values reveals the preference."
        ),
        response_signature=asserts_individual_values,
        evidence_status="untested",
    ),
)


MODULAR_HYPOTHESES: tuple[BeliefHypothesis, ...] = (
    BeliefHypothesis(
        hypothesis_id="inversion_step_skipped",
        kind="belief",
        proposition=(
            "Once the congruences are combined into a single statement about "
            "x + y + z, its right-hand side is the answer."
        ),
        omitted_precondition=(
            "The combined statement is about the common multiplier *times* "
            "x + y + z, so the multiplier's modular inverse must still be "
            "applied to it."
        ),
        probe_variant="balanced",
        probe_rationale=(
            "Modulus 7 keeps the elimination short enough that the solver "
            "reliably reaches the combined right-hand side, which is where this "
            "route diverges. The harder modulus-11 variant fails earlier -- 36% "
            "of its greedy responses give no answer at all -- so it cannot "
            "expose a dropped final step."
        ),
        predicted_wrong_answer=_modular_inversion_skipped,
        evidence_status="validated_out_of_sample",
    ),
    BeliefHypothesis(
        hypothesis_id="inverse_as_ordinary_division",
        kind="belief",
        proposition=(
            "Dividing by the common multiplier modulo p is ordinary division."
        ),
        omitted_precondition=(
            "Division modulo a prime is multiplication by the modular inverse, "
            "which does not agree with integer division."
        ),
        probe_variant="hard_inverse",
        probe_rationale=(
            "A large multiplier makes integer division and modular inversion "
            "disagree on almost every instance."
        ),
        predicted_wrong_answer=_modular_integer_division,
        evidence_status="refuted_out_of_sample",
    ),
    BeliefHypothesis(
        hypothesis_id="full_solution_required",
        kind="desire",
        proposition=(
            "An aggregate question must be answered by first pinning down x, y "
            "and z individually."
        ),
        omitted_precondition=(
            "The congruence system is underdetermined, so individual residues "
            "are not unique; only the aggregate is."
        ),
        probe_variant="balanced",
        probe_rationale=(
            "Every instance has at least two full solutions, so committing to "
            "unique residues reveals the preference."
        ),
        response_signature=asserts_individual_values,
        evidence_status="untested",
    ),
)


HYPOTHESIS_CATALOG: dict[str, tuple[BeliefHypothesis, ...]] = {
    "linear_system_aggregate": LINEAR_HYPOTHESES,
    "modular_linear_system_aggregate": MODULAR_HYPOTHESES,
}


def hypotheses_for(generator_family: str) -> tuple[BeliefHypothesis, ...]:
    return HYPOTHESIS_CATALOG.get(generator_family, ())


def get_hypothesis(
    generator_family: str,
    hypothesis_id: str,
) -> BeliefHypothesis | None:
    for hypothesis in hypotheses_for(generator_family):
        if hypothesis.hypothesis_id == hypothesis_id:
            return hypothesis
    return None


def hypothesis_slot(hypothesis_id: str) -> int | None:
    """Stable, collision-free sub-cell index for the archive's diversity axis.

    Hashing the id into a small number of slots collides -- with three
    hypotheses in four slots two of them shared a niche, which silently merges
    two distinct attributions into one cell and defeats the axis. The catalog
    position is deterministic and unique instead.
    """
    for hypotheses in HYPOTHESIS_CATALOG.values():
        for index, hypothesis in enumerate(hypotheses):
            if hypothesis.hypothesis_id == hypothesis_id:
                return index
    return None


def catalog_payload(generator_family: str) -> list[dict[str, Any]]:
    """The closed choice set shown to the planner."""
    return [
        hypothesis.to_payload() for hypothesis in hypotheses_for(generator_family)
    ]


# ---------------------------------------------------------------------------
# Plan validation: two fields carry meaning, and both are checkable.
# ---------------------------------------------------------------------------

BELIEF_PLAN_FIELDS: tuple[str, ...] = (
    "schema_version",
    "operator",
    "attributed_hypothesis",
    "evidence_quote",
)


def _normalize(text: str) -> str:
    return " ".join(str(text).split()).lower()


def validate_belief_plan(
    plan: Mapping[str, Any],
    *,
    operator: str,
    generator_family: str,
    wrong_trace: str | None,
) -> list[str]:
    """Check the planner's two judgements; everything else is derived.

    ``evidence_quote`` must appear verbatim in the wrong trace. That single
    substring check is what separates an attribution grounded in the evidence
    from a plausible-sounding invention, and it costs nothing to run.
    """
    errors: list[str] = []
    try:
        schema_version = int(plan.get("schema_version", 0) or 0)
    except (TypeError, ValueError):
        schema_version = 0
    if schema_version != BELIEF_SCHEMA_VERSION:
        errors.append(f"schema_version must be {BELIEF_SCHEMA_VERSION}")
    if plan.get("operator") != operator:
        errors.append(
            f"operator must be {operator!r}, got {plan.get('operator')!r}"
        )
    unexpected = sorted(set(plan) - set(BELIEF_PLAN_FIELDS))
    for name in unexpected:
        errors.append(f"unexpected plan field: {name}")

    hypothesis_id = plan.get("attributed_hypothesis")
    valid_ids = [h.hypothesis_id for h in hypotheses_for(generator_family)]
    if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
        errors.append("attributed_hypothesis must be a non-empty string")
    elif hypothesis_id not in valid_ids:
        errors.append(
            f"unknown attributed_hypothesis {hypothesis_id!r}; choose one of "
            + ", ".join(valid_ids)
        )

    quote = plan.get("evidence_quote")
    if wrong_trace:
        if not isinstance(quote, str) or not quote.strip():
            errors.append(
                "evidence_quote must quote the wrong trace when one is supplied"
            )
        elif _normalize(quote) not in _normalize(wrong_trace):
            errors.append(
                "evidence_quote must appear verbatim in the wrong trace; "
                "paraphrase is not evidence"
            )
    elif quote is not None and str(quote).strip():
        errors.append(
            "evidence_quote must be null when no wrong trace is supplied"
        )
    return errors


# ---------------------------------------------------------------------------
# Diagnosticity: can this probe tell the attribution apart from correctness?
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProbeDiagnosticity:
    """Per-seed check that the probe can actually discriminate."""

    valid: bool
    hypothesis_id: str
    kind: Kind
    per_seed: tuple[Mapping[str, Any], ...] = ()
    reasons: tuple[str, ...] = ()
    # Seeds whose instance lets the route be told apart from the truth. Fixed by
    # the instances alone, so scoring may restrict to them without selecting on
    # the observed answers.
    discriminating_seeds: tuple[int, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "hypothesis_id": self.hypothesis_id,
            "kind": self.kind,
            "per_seed": [dict(entry) for entry in self.per_seed],
            "reasons": list(self.reasons),
            "discriminating_seeds": list(self.discriminating_seeds),
        }


def check_probe_diagnosticity(
    hypothesis: BeliefHypothesis,
    instances: Mapping[int, tuple[Mapping[str, Any], str]],
) -> ProbeDiagnosticity:
    """Require the belief's prediction to differ from the correct answer.

    A probe on which the attributed belief and the truth coincide cannot confirm
    or refute anything, so it is rejected before any rollout is spent. Desire
    hypotheses are scored from the derivation instead and are diagnostic by
    construction on these families, which are all underdetermined.
    """
    per_seed: list[Mapping[str, Any]] = []
    reasons: list[str] = []

    if hypothesis.predicted_wrong_answer is None:
        for seed in sorted(instances):
            per_seed.append(
                {
                    "seed": seed,
                    "correct_answer": int(instances[seed][1]),
                    "belief_predicted_answer": None,
                    "discriminates": True,
                    "basis": "response_signature",
                }
            )
        return ProbeDiagnosticity(
            valid=bool(per_seed),
            hypothesis_id=hypothesis.hypothesis_id,
            kind=hypothesis.kind,
            per_seed=tuple(per_seed),
            discriminating_seeds=tuple(sorted(instances)),
        )

    discriminating: list[int] = []
    for seed in sorted(instances):
        data, answer = instances[seed]
        correct = int(answer)
        predicted = hypothesis.predicted_wrong_answer(data)
        discriminates = predicted is not None and predicted != correct
        if discriminates:
            discriminating.append(seed)
        per_seed.append(
            {
                "seed": seed,
                "correct_answer": correct,
                "belief_predicted_answer": predicted,
                "discriminates": discriminates,
                "basis": "predicted_answer",
            }
        )

    total = len(per_seed)
    required = int(hypothesis.min_discriminating_seeds)
    valid = len(discriminating) >= required
    if not valid:
        reasons.append(
            f"only {len(discriminating)} of {total} seeds discriminate; this "
            f"route needs at least {required} distinct predictions to be "
            "separable from chance. Widen the evaluation seed set rather than "
            "filtering instances, which would change the problem family."
        )
        for entry in per_seed:
            if entry["discriminates"]:
                continue
            seed = entry["seed"]
            if entry["belief_predicted_answer"] is None:
                reasons.append(f"seed={seed}: prediction not computable")
            else:
                reasons.append(
                    f"seed={seed}: route predicts "
                    f"{entry['belief_predicted_answer']}, which equals the "
                    "correct answer, so this seed cannot discriminate"
                )
    return ProbeDiagnosticity(
        valid=valid,
        hypothesis_id=hypothesis.hypothesis_id,
        kind=hypothesis.kind,
        per_seed=tuple(per_seed),
        reasons=tuple(reasons),
        discriminating_seeds=tuple(discriminating),
    )


# ---------------------------------------------------------------------------
# Outcome: did the attribution predict what the solver actually did?
# ---------------------------------------------------------------------------


def score_belief_attribution(
    hypothesis: BeliefHypothesis,
    instances: Mapping[int, tuple[Mapping[str, Any], str]],
    rollouts: Mapping[int, list[Mapping[str, Any]]],
    *,
    discriminating_seeds: Any = None,
) -> dict[str, Any]:
    """Measure attribution accuracy from the *modal* rollout answer per instance.

    R_Q and attribution are different statistics of the same distribution, so
    they need different estimators from the same rollouts. R_Q is
    ``s(1-s)*H`` and needs the spread, which is why the solver samples at
    temperature 1. An attribution is a claim about what the policy
    systematically does, which is the centre; sampling noise is not a belief.
    Measured on this task, 68% of the temperature-1 errors on one variant are
    answers the same model gets right under greedy decoding, so scoring every
    rollout drowns any route in noise -- which is what two null results were.

    Taking one modal answer per instance recovers the centre from the spread
    already collected: no extra generation, no change to pipeline temperature,
    and R_Q and the attribution are by construction measured on the same
    instances and the same rollouts. It also matches the unit of evidence, since
    a route is a claim about a functional form across instances rather than
    within one.

    ``rollouts[seed]`` holds dicts with ``predicted_answer``, ``correct`` and
    ``mean_negative_logprob``. Confidence is reported alongside because a belief
    predicts *confidently* wrong, while a skill deficit predicts hesitation --
    correctness alone cannot separate the two.
    """
    total = 0
    wrong = 0
    hits = 0
    hit_conf: list[float] = []
    other_wrong_conf: list[float] = []
    correct_conf: list[float] = []
    modal_support: list[float] = []

    allowed = (
        None if discriminating_seeds is None else {int(s) for s in discriminating_seeds}
    )
    for seed, entries in rollouts.items():
        if seed not in instances:
            continue
        # A seed whose instance cannot separate the route from the truth carries
        # no evidence either way, so it is excluded rather than counted as a miss.
        if allowed is not None and int(seed) not in allowed:
            continue
        if not entries:
            continue
        data, answer = instances[seed]
        correct = int(answer)
        predicted = (
            hypothesis.predicted_wrong_answer(data)
            if hypothesis.predicted_wrong_answer is not None
            else None
        )

        # Collapse this instance's rollouts to the single answer the policy
        # settles on. Ties break toward the first-seen answer, which is
        # deterministic given the recorded rollout order.
        counts: dict[str, int] = {}
        for entry in entries:
            key = str(entry.get("predicted_answer"))
            counts[key] = counts.get(key, 0) + 1
        modal_key = max(counts, key=lambda k: counts[k])
        modal_support.append(counts[modal_key] / len(entries))
        entry = next(
            item
            for item in entries
            if str(item.get("predicted_answer")) == modal_key
        )

        total += 1
        confidence = entry.get("mean_negative_logprob")
        if bool(entry.get("correct")):
            if isinstance(confidence, (int, float)):
                correct_conf.append(float(confidence))
            continue
        wrong += 1
        if hypothesis.response_signature is not None:
            matched = hypothesis.response_signature(str(entry.get("response", "")))
        else:
            try:
                matched = (
                    predicted is not None
                    and int(str(entry.get("predicted_answer")).strip()) == predicted
                )
            except (TypeError, ValueError):
                matched = False
        if matched:
            hits += 1
            if isinstance(confidence, (int, float)):
                hit_conf.append(float(confidence))
        elif isinstance(confidence, (int, float)):
            other_wrong_conf.append(float(confidence))

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "hypothesis_id": hypothesis.hypothesis_id,
        "kind": hypothesis.kind,
        "evidence_status": hypothesis.evidence_status,
        # One observation per instance, not per rollout.
        "scoring_basis": "modal_answer_per_instance",
        "mean_modal_support": (
            sum(modal_support) / len(modal_support) if modal_support else None
        ),
        "scored_seeds": (
            None if allowed is None else sorted(allowed & set(rollouts))
        ),
        "num_instances_scored": total,
        "num_rollouts": total,
        "num_wrong": wrong,
        "num_attribution_hits": hits,
        # Share of *errors* the attribution explains. Undefined with no errors.
        "attribution_hit_rate": (hits / wrong) if wrong else None,
        "mean_neg_logprob_on_hits": mean(hit_conf),
        "mean_neg_logprob_on_other_errors": mean(other_wrong_conf),
        "mean_neg_logprob_on_correct": mean(correct_conf),
        # Confidently wrong in the predicted way is the belief signature.
        "hits_more_confident_than_correct": (
            mean(hit_conf) is not None
            and mean(correct_conf) is not None
            and mean(hit_conf) < mean(correct_conf)
        ),
    }


@dataclass(frozen=True, slots=True)
class Eligibility:
    """Gate-then-select: eligibility is boolean, fitness stays R_Q alone.

    A weighted sum of R_Q and attribution score is deliberately avoided. Any
    weighting either lets a generically hard problem outrank the mental-state
    objective, or rewards contrived problems that maximize attribution hits, so
    the two are kept as a filter and an ordering rather than blended.
    """

    eligible: bool
    valid: bool
    diagnostic: bool
    attribution_supported: bool
    reasons: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "valid": self.valid,
            "diagnostic": self.diagnostic,
            "attribution_supported": self.attribution_supported,
            "reasons": list(self.reasons),
        }


def evaluate_eligibility(
    *,
    family_semantics_valid: bool,
    family_semantics_reasons: Any = (),
    diagnosticity: ProbeDiagnosticity | None,
    plan_errors: Any = (),
) -> Eligibility:
    """Eligible(x, h) = Valid(x) and Diagnostic(x, h) and AttributionSupported(h).

    ``AttributionSupported`` checks only the *internal* consistency of the
    attribution -- the hypothesis is in the catalog, and the quote is verbatim
    when a trace was supplied. It deliberately does not ask whether a trace
    existed, because a gate that only the reasoning condition can fail would put
    a condition-dependent filter in front of a shared archive.
    """
    reasons: list[str] = []
    valid = bool(family_semantics_valid)
    for reason in tuple(family_semantics_reasons):
        reasons.append(f"valid: {reason}")

    diagnostic = diagnosticity is not None and diagnosticity.valid
    if diagnosticity is None:
        reasons.append("diagnostic: no diagnosticity check was run")
    else:
        for reason in diagnosticity.reasons:
            reasons.append(f"diagnostic: {reason}")

    supported = not tuple(plan_errors)
    for reason in tuple(plan_errors):
        reasons.append(f"attribution: {reason}")

    return Eligibility(
        eligible=valid and diagnostic and supported,
        valid=valid,
        diagnostic=diagnostic,
        attribution_supported=supported,
        reasons=tuple(reasons),
    )


def attribution_niche_key(
    concept_type: str | None,
    hypothesis_id: str | None,
) -> str:
    """Archive niche label: one cell per (concept_type, attributed attitude).

    The attribution supplies a new exploration axis while R_Q still decides the
    champion inside each cell, so the search cannot collapse onto a single
    belief and keep re-generating problems for it.
    """
    return f"{concept_type or 'unknown'}::{hypothesis_id or 'none'}"


def build_belief_plan_prompt(
    *,
    parent_problem: str,
    generator_family: str,
    operator: str,
    wrong_trace: str | None,
    correct_trace: str | None = None,
    template_dir: Any = None,
) -> str:
    """Render the single-judgement planning prompt.

    Both conditions receive the identical hypothesis catalog and instructions.
    The only difference is whether the traces are present, which is exactly the
    manipulation under test: attribution from evidence versus from priors.
    """
    import json as _json
    from pathlib import Path as _Path
    from string import Template as _Template

    directory = _Path(
        template_dir
        if template_dir is not None
        else _Path(__file__).resolve().parents[2] / "prompt_templates"
    )
    template = (directory / "belief_plan.txt").read_text(encoding="utf-8")

    if wrong_trace:
        blocks = ["Wrong reasoning trace:\n" + wrong_trace.strip()]
        if correct_trace:
            blocks.append("Correct reasoning trace:\n" + correct_trace.strip())
        evidence_block = "\n\n".join(blocks)
        quote_instruction = '"<exact span copied from the wrong trace>"'
    else:
        evidence_block = (
            "No reasoning traces are available. Choose the hypothesis a solver "
            "of this problem is most likely to hold."
        )
        quote_instruction = "null"

    return _Template(template).safe_substitute(
        parent_problem=parent_problem.strip(),
        evidence_block=evidence_block,
        hypothesis_catalog=_json.dumps(
            catalog_payload(generator_family),
            ensure_ascii=False,
            indent=2,
        ),
        operator=operator,
        evidence_quote_instruction=quote_instruction,
    )


__all__ = [
    "BELIEF_PLAN_FIELDS",
    "BELIEF_SCHEMA_VERSION",
    "BeliefHypothesis",
    "Eligibility",
    "attribution_niche_key",
    "evaluate_eligibility",
    "HYPOTHESIS_CATALOG",
    "LINEAR_HYPOTHESES",
    "MODULAR_HYPOTHESES",
    "ProbeDiagnosticity",
    "asserts_individual_values",
    "build_belief_plan_prompt",
    "catalog_payload",
    "check_probe_diagnosticity",
    "get_hypothesis",
    "hypotheses_for",
    "score_belief_attribution",
    "validate_belief_plan",
]
