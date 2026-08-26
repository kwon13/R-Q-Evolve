import collections
import hashlib
import json
import math
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .archive import MAPElitesArchive, StructuralInspirationSelection
from .ast_contract import check_generator_contract, check_problem_text
from .backends import EvolutionBackend, RolloutRecord
from .code_utils import (
    answer_is_bare_draw,
    answer_leaks_in_every_instance,
    extract_generator_code,
    lint_generator_source,
    lint_mutation_generator_source,
    lint_problem_instance,
    set_label_declarations,
)
from .concepts import SKILLS, GROUPS, validate_label_decl
from .config import EvolutionConfig, TrainingDataConfig
from .dataset import (
    DynamicProblemDataset,
    build_replay_training_examples,
    build_training_examples,
)
from .openai_evaluator import (
    EvaluatorRuntimeError,
    OpenAIEvaluatorConfig,
    evaluate_messages_with_openai,
)
from .reward import grader_stats
from .program import ProblemInstance, ProblemProgram
from .replay import PreviousRQScoreboard, RolloutReplayBuffer
from .seed_stream import SeedStream
from .prompts import (
    build_relabel_task,
    build_relabel_group_task,
    MutationTask,
    MUTATION_OP,
    build_judge_messages,
    build_judge_task,
    build_family_task,
    build_generator_task,
    parse_declared_labels,
    parse_family_plan,
    judge_accepts,
    parse_judge_verdict,
)
from .relabel import SkillOffsets, GroupOffsets, choose_skill, choose_group, logit
from .scoring import RQResult, compute_rq_program, is_frontier, score_seed
from .solver_trace import clean_and_grade_solver_rollout


# A rejected child is discarded the moment its report is written, so without
# this the source behind "execute failed at seed=0" is gone. Across six runs
# that left ~7,000 rejections whose programs cannot be recovered, and no way to
# measure a new gate's false-positive rate against the population it rejects.
_MAX_LOGGED_SOURCE_CHARS = 8000


@dataclass(slots=True)
class CandidateReport:
    status: str
    op: str
    child_id: str | None = None
    rq_score: float = 0.0
    s_hat: float = 0.0
    u_score: float = 0.0
    reason: str | None = None
    source_code: str | None = None
    ast_findings: list[str] = field(default_factory=list)
    # Reproductive ancestry and context provenance are logged for every
    # outcome, including stage-1 parse failures. Without them only champions
    # could be audited and the donor's effect on the rejection funnel would be
    # unknowable.
    parent_id: str | None = None
    inspiration: dict | None = None


def _relabel_p_yes(pair, reply, yes_id) -> float | None:
    """P(YES) from the sampled token's logprob, else from the decoded text.

    ``pair`` is the ``(token id, logprob)`` the backend recorded for this task.
    With only two allowed tokens and greedy decoding the sampled side's
    probability fixes the other's, so one logprob is the whole distribution --
    which is why the caller asks for one token and one logprob, not a top-k.
    """
    if pair is not None and yes_id is not None:
        try:
            token_id, logp = pair
            p = math.exp(float(logp))
            if 0.0 <= p <= 1.0:
                return p if int(token_id) == yes_id else 1.0 - p
        except (TypeError, ValueError, OverflowError):
            pass
    if reply:
        head = str(reply).strip().lstrip("`*_\"' \t\n").upper()
        if head.startswith("YES"):
            return 1.0
        if head.startswith("NO"):
            return 0.0
    return None


def _logged_source(source: str | None) -> str | None:
    if source is None:
        return None
    return source[:_MAX_LOGGED_SOURCE_CHARS]


def _ast_findings(program: "ProblemProgram | None") -> list[str]:
    if program is None:
        return []
    verdict = program.metadata.get("ast_contract") or {}
    return list(verdict.get("findings") or [])


def _report_context(task: "MutationTask | None") -> dict:
    if task is None:
        return {"parent_id": None, "inspiration": None}
    parent = getattr(task, "parent", None)
    provenance = dict(getattr(task, "provenance", {}) or {})
    inspiration = provenance.get("structural_inspiration")
    return {
        "parent_id": getattr(parent, "program_id", None),
        "inspiration": dict(inspiration) if inspiration is not None else None,
    }


def _child_metadata(task: "MutationTask") -> dict:
    """Persistent ancestry/plan metadata for a generated child."""
    metadata = {
        "op": task.op,
        "lineage_root_id": task.parent.lineage_root_id(),
    }
    provenance = dict(task.provenance or {})
    if provenance.get("structural_inspiration") is not None:
        metadata["structural_inspiration"] = dict(provenance["structural_inspiration"])
    if provenance.get("family_plan") is not None:
        metadata["family_plan"] = dict(provenance["family_plan"])
    return metadata


def _record_inspiration_copy_gate(
    task: "MutationTask", child: "ProblemProgram", verdict: dict
) -> None:
    """Attach donor-copy telemetry to both the report path and child snapshot."""
    provenance = dict(task.provenance or {})
    inspiration = dict(provenance.get("structural_inspiration") or {})
    inspiration["copy_gate"] = dict(verdict)
    provenance["structural_inspiration"] = inspiration
    task.provenance = provenance
    if "structural_inspiration" in child.metadata:
        child.metadata["structural_inspiration"] = dict(inspiration)


@dataclass
class RQEvolver:
    """Owns one archive, one backend, and one dynamic training dataset."""

    archive: MAPElitesArchive
    backend: EvolutionBackend
    evolution_config: EvolutionConfig = field(default_factory=EvolutionConfig)
    training_config: TrainingDataConfig = field(default_factory=TrainingDataConfig)
    dataset: DynamicProblemDataset = field(default_factory=DynamicProblemDataset)
    used_seeds: dict[str, set[int]] = field(default_factory=dict)
    # The never-reused seed source every evaluation draws from. Persisted with
    # the archive so a resumed run does not re-issue seeds the pre-restart run
    # already graded on -- which would reopen exactly the overfitting hole that
    # fresh seeds close.
    seed_stream: SeedStream = field(default_factory=SeedStream)
    # Per-skill logit offsets for relabelling, carried across iterations
    # and persisted with the archive so a resume does not restart cold.
    skill_offsets: SkillOffsets = field(default_factory=SkillOffsets)
    group_offsets: GroupOffsets = field(default_factory=GroupOffsets)
    # This iteration's re-scoring rollouts, kept so the solver update trains on
    # them instead of paying for a second sampling pass over the same programs.
    replay: RolloutReplayBuffer = field(default_factory=RolloutReplayBuffer)
    # Which elites train is decided by PAST scores; what they train on is this
    # iteration's rollouts. Scoring and selecting on the same draw would keep
    # whichever elite's measurement noise landed high.
    previous_rq: PreviousRQScoreboard = field(default_factory=PreviousRQScoreboard)
    # program_id -> why it was rejected, for every child that ever failed a gate.
    #
    # Mutation is a near-deterministic function of (prompt, parent) and parents
    # are drawn with replacement from a small archive, so the same source comes
    # back repeatedly: one program was regenerated 13 times in 32 slots, and 34%
    # of a two-iteration probe went on children already rejected. Re-running them
    # cannot teach us anything -- the judge is deterministic at temperature 0, so
    # a repeat is guaranteed the same verdict -- while each one still costs a
    # 5-seed execution and a judge call. Keyed by program_id, which IS
    # md5(source_code), so a hit means byte-identical code.
    #
    # Rejections only. Accepted children live in the archive, which has its own
    # behaviour/template duplicate check.
    rejected_children: dict[str, str] = field(default_factory=dict)

    events: list[dict] = field(default_factory=list)
    # Candidate reports from the most recent run_outer_iteration, exposed so the
    # sampler can persist them to the per-step evolution log.
    last_reports: list[CandidateReport] = field(default_factory=list)
    # Number of refresh_dataset() calls so far. Only used to vary the seed of the
    # select_random_order ablation shuffle across outer iterations (reproducible).
    dataset_refresh_count: int = 0
    # Per-champion frontier decision from the most recent refresh_dataset():
    # {program_id, s_hat, rq_score, decision: in_frontier | s_hat_out_of_range}.
    # Persisted into evolution_log.jsonl so the learnability/curriculum filter
    # is observable (which champions fed the training set, and why not).
    last_frontier: list[dict] = field(default_factory=list)
    current_iteration: int = -1
    # Separate from the process-global RNG so enabling donor context does not
    # change baseline parent draws or few-shot rotation. Persisted on disk and
    # in verl's data checkpoint for exact resume at a weight-aligned point.
    inspiration_draw_count: int = 0
    mutation_prompt_draw_count: int = 0
    search_draw_count: int = 0

    def load_seed_programs(self, seed_dir: str | Path) -> list[ProblemProgram]:
        programs: list[ProblemProgram] = []
        for path in sorted(Path(seed_dir).glob("*.py")):
            program = ProblemProgram.from_file(path, generation=0)
            # Each authored seed begins one primary-parent lineage. Descendants
            # inherit this id; structural inspirations never create a second
            # ancestry edge.
            program.metadata.setdefault("lineage_root_id", program.program_id)
            inst, reason = self.verify_program(program)
            if inst is None:
                self.events.append(
                    {
                        "event": "seed_rejected",
                        "file": path.name,
                        "reason": reason,
                    }
                )
                continue
            programs.append(program)
            self.events.append(
                {
                    "event": "seed_loaded",
                    "file": path.name,
                    "program_id": program.program_id,
                    "problem": inst.problem,
                }
            )
        return programs

    def initialize_archive(self, seed_dir: str | Path) -> int:
        """Load seeds, evaluate them once, and place them in MAP-Elites."""
        inserted = 0
        for program in self.load_seed_programs(seed_dir):
            inst, reason = self.verify_program(program)
            if inst is None:
                self.events.append(
                    {
                        "event": "seed_verify_failed",
                        "program_id": program.program_id,
                        "reason": reason,
                    }
                )
                continue
            result = self.evaluate_programs([program], store_replay=True)[0]
            if result is None:
                continue
            # Bootstrap counts as iteration -1. Without it every seed would be
            # "first scored this iteration" at t=0, the previous-score filter would
            # exclude all of them, and the first training batch would be empty.
            self.previous_rq.record(program.program_id, -1, result.rq_score)
            if self.archive.try_insert(
                program=program,
                u_value=result.u_score,
                rq_score=result.rq_score,
            ):
                inserted += 1
        self.refresh_dataset()
        return inserted

    def verify_program(
        self,
        program: ProblemProgram,
        n_seeds: int | None = None,
    ) -> tuple[ProblemInstance | None, str | None]:
        """Multi-seed execution and cheap mathematical sanity checks."""
        n = n_seeds or self.evolution_config.verify_seeds
        source_errors = lint_generator_source(program.source_code)
        mode = self.evolution_config.ast_contract
        is_generated_mutation = program.metadata.get("op") == MUTATION_OP
        if is_generated_mutation:
            # Determinism only: a seeded `rng`, no direct `random.*` calls, and
            # sorted() around dict/set iteration, so a program yields the same
            # instance in the verifier and in every later rollout process.
            #
            # Every notation flag stays OFF. MAX_ATTEMPTS = 200, the canonical
            # `instance_data` assignment and the `str(sympy.Integer(answer))`
            # wrapper were contracts a registered family compiler satisfied by
            # construction; free-form generation does not. Enforcing them here
            # rejected mathematically sound programs on shape alone -- observed
            # on 20/20 candidates for instance_data and on 9 of 14 rejections
            # for the notation pair. Execution-level checks carry the guarantee
            # instead: every verified seed must yield an integer answer below,
            # and the judge gates the labels before the archive.
            source_errors.extend(
                lint_mutation_generator_source(
                    program.source_code,
                    require_assert=False,
                    reject_trivial_assert=False,
                    reject_unbounded_sampling=False,
                    require_answer_routes=False,
                    require_canonical_instance_data=False,
                    require_mechanical_shape=False,
                )
            )
            # The structural contract on the child's own cross-check. This is a
            # dataflow predicate, not one of the shape predicates above, and it
            # lives in its own module for that reason: the flags overhead are
            # all OFF and must stay that way, and a single ON flag among six OFF
            # ones invites the next reader to flip them as a batch.
            #
            # The verdict is always recorded so a shadow run costs no plumbing;
            # only "enforce" turns it into a rejection.
            if mode != "off":
                findings = check_generator_contract(program.source_code)
                program.metadata["ast_contract"] = {
                    "verdict": "flagged" if findings else "clean",
                    "findings": [str(f) for f in findings],
                }
                if findings and mode == "enforce":
                    source_errors.extend(str(f) for f in findings)
            # The statement/code decoupling the AST contract does not cover:
            # `answer` being a sampled name unchanged. 21% of the 4B run's
            # champions are built this way, and the assert cannot catch it
            # because both routes trivially yield the same draw.
            bare = answer_is_bare_draw(program.source_code)
            if bare:
                source_errors.append(bare)
        if source_errors:
            return None, "; ".join(source_errors[:3])

        group = program.declared_group()
        skill = program.declared_skill()
        label_errors = validate_label_decl(group, skill)
        if label_errors:
            return None, "; ".join(label_errors)
        # Resolve the two axis labels into metadata once, so a program keeps
        # them after an archive round-trip without re-parsing its source.
        program.metadata["group"] = group
        program.metadata["skill"] = skill

        first: ProblemInstance | None = None
        # statement -> the answers it was rendered with. A set would only
        # answer "does the program vary?"; the mapping additionally proves
        # ill-posedness when it does not vary the way it should.
        seen_problems: dict[str, set[str]] = {}
        rendered: list[ProblemInstance] = []
        for seed in range(n):
            inst = program.execute(seed=seed)
            if inst is None:
                # Name the failure. "execute failed" was the single largest
                # rejection reason in a run where 58% of candidates died here,
                # and it cannot distinguish the child's own AssertionError --
                # its cross-check working as designed -- from broken code.
                why = program.last_execution_error or "unknown"
                return None, f"execute failed at seed={seed}: {why}"
            instance_errors = lint_problem_instance(inst)
            if instance_errors:
                return None, "; ".join(instance_errors[:3])
            if is_generated_mutation and mode == "enforce":
                # A statement that names its own technique makes the declared
                # SKILL untrue: the reasoning is quoted, not forced. Needs the
                # rendered text, so it cannot live with the source rules.
                handed_over = check_problem_text(inst.problem)
                if handed_over:
                    return None, str(handed_over[0])
            if (
                is_generated_mutation
                and re.fullmatch(r"-?\d+", inst.answer.strip()) is None
            ):
                # The static `str(sympy.Integer(answer))` contract is checked on
                # the source above; this takes the same guarantee from the
                # executed output, which the source text cannot prove.
                return None, (
                    "generated answer must be a base-10 integer string: "
                    f"{inst.answer!r} at seed={seed}"
                )
            if not _answer_parseable(inst.answer):
                return None, f"answer is not parseable: {inst.answer!r}"
            first = first or inst
            rendered.append(inst)
            key = " ".join(inst.problem.split())
            answers = seen_problems.setdefault(key, set())
            answers.add(str(inst.answer).strip())
            if len(answers) > 1:
                # The one ill-posedness result that is a PROOF rather than a
                # heuristic: the same question, word for word, graded against
                # two different answers. Whatever the statement determines, it
                # is not the value being scored.
                #
                # The old check -- at least two distinct statements over n seeds
                # -- cannot see it. Champion 3cedc80f7798 of the 4B run renders
                # three distinct statements across five seeds and so passed,
                # while "Let n = 1 be a positive integer. Determine the number
                # of ways to partition n into distinct prime factors." appeared
                # twice with answers 1 and 2.
                return None, (
                    "the same problem text is graded against two answers "
                    f"({sorted(answers)}): the statement does not determine it"
                )

        if n > 1 and len(seen_problems) <= 1:
            return None, "program does not vary its visible problem across seeds"
        # verify_program just rendered seeds 0..n-1 of this program, and the
        # judge (when enabled) reads the seed-0 instance. Move the stream past
        # them so the candidate's FIRST scoring draw is genuinely fresh.
        #
        # Without this the comment above `draw_instances` -- "A candidate is
        # graded on n FRESH seeds, not on the seed-0 instance the judge saw" --
        # is false: SeedStream starts every new program_id at 0 and nothing
        # else advances it, so `take` returns 0 and the candidate is admitted
        # on the one instance every structural check already ran against. It is
        # visible in the persisted cursors: every champion born after iteration
        # 1 has cursor == (snapshots it appears in) + exactly 1.
        self.seed_stream.reserve_through(program.program_id, n - 1)
        if is_generated_mutation and mode == "enforce":
            # Needs every rendered instance, so it cannot live in the per-seed
            # loop above: one coincidental appearance of the answer in the text
            # is common, the same one on all n seeds is the answer being handed
            # to the solver. 25% of the 4B run's champions leak this way.
            leaked = answer_leaks_in_every_instance(rendered)
            if leaked:
                return None, leaked
        return first, None

    def run_outer_iteration(self, outer_iteration: int) -> dict:
        self.current_iteration = int(outer_iteration)
        attempted = 0
        inserted = 0
        reports: list[CandidateReport] = []
        # Judge telemetry for this outer iteration only. Accumulated across the
        # inner batches because the gate runs once per batch, and reset here so
        # a wandb series reads per-iteration rather than cumulative.
        self.judge_tally = {
            "reached": 0,
            "agreed": 0,
            "group_agreed": 0,
            "skill_agreed": 0,
            "label_mismatch": 0,
            "failed_closed": 0,
            "skill_none": 0,
            "group_none": 0,
        }
        self.judge_skill_counts: dict[str, int] = {}
        # Rollouts are on-policy for exactly one update; carrying them across an
        # iteration would need an importance-ratio correction this design does
        # not have, so the buffer starts empty every time.
        self.replay.begin_iteration(outer_iteration)

        # Push current actor weights into vLLM ONCE for the whole evolve phase.
        # Evolve runs no optimizer step, so weights are static throughout:
        # reevaluate + every inner batch reuse the resident model (no per-session
        # re-push). The backend's begin_session no longer pushes weights when the
        # rollout is resident (sleep mode off).
        self.backend.sync_weights()
        # Ablation: with re-measurement off, a champion keeps the R_Q it was
        # admitted with and is never rescored or evicted. It is the archive's
        # dominant sink -- across three 4B arms the eviction/insertion ratio sat
        # at 0.86-0.94, so 268 insertions over 77 iterations left only +19 net
        # champions and half of them lived 3 iterations or fewer. Turning it off
        # asks whether the MAP is a curriculum that has to track the policy or a
        # conveyor belt paying for itself.
        if self.evolution_config.reevaluate_champions:
            self.reevaluate_champions()

        # Freeze the population whose CURRENT rollouts may train this update
        # before mutation changes the archive.  Selection is based on scores
        # known before those rollouts (PreviousRQScoreboard), the current s_hat is
        # only a zero-advantage gate, and the payload is this iteration's replay.
        # A child inserted below therefore becomes eligible at the next outer
        # iteration instead of displacing a measured incumbent from the batch
        # that is about to update theta_t -> theta_{t+1}.
        training_pool = list(self.archive.champions())

        batch_size = self.evolution_config.inner_iteration_batch_size
        for start in range(0, self.evolution_config.inner_iterations, batch_size):
            current_batch = min(
                batch_size,
                self.evolution_config.inner_iterations - start,
            )
            batch_reports = self.inner_iteration_batch(current_batch)
            reports.extend(batch_reports)
            attempted += sum(1 for r in batch_reports if r.status != "no_parent")
            inserted += sum(1 for r in batch_reports if r.status == "inserted")

        self.refresh_dataset(training_champions=training_pool)
        stats = self.archive.stats()
        frontier_in = sum(
            1 for f in self.last_frontier if f["decision"] == "in_frontier"
        )
        # Dispersion is a diagnostic, never a fitness term (rewarding
        # consistency would hand a free maximum to a seed-ignoring generator),
        # but it is the signal that says whether a champion's R_Q is an average
        # over comparable instances or over a trivial/impossible mixture.
        dispersions = [
            float(c.metadata.get("dispersion", 0.0))
            for c in self.archive.champions()
            if "dispersion" in (c.metadata or {})
        ]
        dispersion_metrics = (
            {
                "mean_dispersion": sum(dispersions) / len(dispersions),
                "max_dispersion": max(dispersions),
            }
            if dispersions
            else {}
        )
        # Replay accounting. `replay_degenerate_frac` is the share of the batch
        # that can produce no gradient at all (all-correct or all-wrong groups
        # self-neutralise under the per-instance LOO baseline), so it is the
        # honest measure of how much of the update is wasted.
        replay_metrics = self.replay.stats()
        tally = getattr(self, "judge_tally", {}) or {}
        reached = tally.get("reached", 0)
        judge_metrics = {f"judge_{k}": v for k, v in tally.items()}
        if reached:
            judge_metrics["judge_agree_rate"] = tally["agreed"] / reached
            judge_metrics["judge_group_agree_rate"] = tally["group_agreed"] / reached
            judge_metrics["judge_skill_agree_rate"] = tally["skill_agreed"] / reached
            judge_metrics["judge_skill_none_rate"] = tally["skill_none"] / reached
        # Distinct SKILLs the judge actually emitted, excluding "none".
        judge_metrics["judge_skill_vocabulary"] = len(
            [k for k in getattr(self, "judge_skill_counts", {}) if k != "none"]
        )
        # Why candidates died, not just how many. accept_rate alone cannot tell
        # "the Evolver writes code that does not run" from "the judge refuses
        # the labels", and those need opposite fixes.
        status_counts: dict[str, int] = {}
        for report in reports:
            key = f"status_{report.status}"
            status_counts[key] = status_counts.get(key, 0) + 1

        inspiration_reports = [
            report for report in reports if report.inspiration is not None
        ]
        attached_inspirations = [
            report
            for report in inspiration_reports
            if bool((report.inspiration or {}).get("attached"))
        ]
        omitted_reasons = collections.Counter(
            str((report.inspiration or {}).get("omitted_reason"))
            for report in inspiration_reports
            if not bool((report.inspiration or {}).get("attached"))
        )
        selection_tiers = collections.Counter(
            str((report.inspiration or {}).get("selection_tier"))
            for report in attached_inspirations
        )
        inspiration_metrics = {
            "inspiration_enabled": int(self.evolution_config.structural_inspiration),
            "inspiration_attempted": len(inspiration_reports),
            "inspiration_attached": len(attached_inspirations),
            "inspiration_attached_rate": (
                len(attached_inspirations) / len(inspiration_reports)
                if inspiration_reports
                else 0.0
            ),
            "inspiration_unique_donors": len(
                {
                    str((report.inspiration or {}).get("program_id"))
                    for report in attached_inspirations
                }
            ),
            "inspiration_inserted": sum(
                1 for report in attached_inspirations if report.status == "inserted"
            ),
            **{
                f"inspiration_omitted_{reason}": count
                for reason, count in omitted_reasons.items()
            },
            **{
                f"inspiration_tier_{tier}": count
                for tier, count in selection_tiers.items()
            },
        }
        copy_checked = [
            report
            for report in attached_inspirations
            if bool(((report.inspiration or {}).get("copy_gate") or {}).get("checked"))
        ]
        copy_rejected = [
            report
            for report in copy_checked
            if bool(((report.inspiration or {}).get("copy_gate") or {}).get("rejected"))
        ]
        copy_rejection_reasons = collections.Counter(
            str(((report.inspiration or {}).get("copy_gate") or {}).get("reason"))
            for report in copy_rejected
        )
        inspiration_metrics.update(
            {
                "inspiration_copy_checked": len(copy_checked),
                "inspiration_copy_rejected": len(copy_rejected),
                "inspiration_copy_rejected_rate": (
                    len(copy_rejected) / len(copy_checked) if copy_checked else 0.0
                ),
                "inspiration_cross_descriptor_rate": (
                    selection_tiers.get("cross_lineage_cross_descriptor", 0)
                    / len(attached_inspirations)
                    if attached_inspirations
                    else 0.0
                ),
                **{
                    f"inspiration_copy_rejected_{reason}": count
                    for reason, count in copy_rejection_reasons.items()
                },
            }
        )

        grader_metrics = {
            # How often grading had to be killed. Non-zero means math_verify met
            # an input it cannot finish (`51!!` -> factorial(factorial(51))) and
            # the answer was scored non-match without wedging the run. Silence
            # here used to mean the same thing had leaked a thread instead.
            **{f"grader_{k}": v for k, v in grader_stats().items()},
        }

        # Why inserts were refused, split by cause. The novelty gates and
        # ordinary cell competition used to be one indistinguishable bucket.
        insert_rejects = collections.Counter(
            e.get("reason") for e in self.events if e.get("event") == "insert_rejected"
        )
        insert_metrics = {f"insert_rejected_{k}": v for k, v in insert_rejects.items()}
        insert_metrics["insert_rejected"] = sum(insert_rejects.values())

        # SKILL relabelling. `relabel_changed` is the one to watch: the declared
        # label agreed with a reference only 32.4% of the time, so a run where
        # this stays near zero is not one where the Evolver got the labels
        # right -- it is one where the relabeller never reached the model.
        relabel_metrics = dict(getattr(self, "_relabel_stats", {}) or {})
        if relabel_metrics.get("relabel_decisions"):
            relabel_metrics["relabel_change_rate"] = (
                relabel_metrics["relabel_changed"]
                / relabel_metrics["relabel_decisions"]
            )
        # Mean YES-bias per skill, the quantity the offsets subtract out. A
        # skill drifting far from the rest is one the prompt over-accepts.
        relabel_metrics["relabel_offsets"] = {
            k: round(v, 4) for k, v in self.skill_offsets.mean.items()
        }
        relabel_metrics["relabel_group_offsets"] = {
            k: round(v, 4) for k, v in self.group_offsets.mean.items()
        }

        result = {
            "outer_iteration": outer_iteration,
            "attempted": attempted,
            "inserted": inserted,
            "accept_rate": inserted / attempted if attempted else 0.0,
            "dataset_size": len(self.dataset),
            "frontier_in": frontier_in,
            "frontier_out": len(self.last_frontier) - frontier_in,
            "group_size": self.evolution_config.group_size,
            "train_batch_target": self.evolution_config.train_batch_target,
            **dispersion_metrics,
            **replay_metrics,
            **judge_metrics,
            **grader_metrics,
            **insert_metrics,
            **inspiration_metrics,
            **relabel_metrics,
            **status_counts,
            **stats,
        }
        self.last_reports = reports
        self.events.append({"event": "outer_iteration_done", **result})
        return result

    def inner_iteration_batch(self, batch_size: int) -> list[CandidateReport]:
        """Run one batch: mutate -> verify -> evaluate -> rollout -> report.

        ``entries`` is a ``list[dict]`` where an entry's *state is which key it
        holds* (there is no status field). Each entry moves through three shapes:

          * ``{"task", "child", "inst"}``   -- verified child, alive for rollouts;
          * ``{"_retry": {...}}``           -- parsed but failed verify; eligible
                                               for one Reflexion self-fix;
          * ``{"report": CandidateReport}`` -- terminal; carries the outcome.

        Flow: mutate() -> split into the three shapes -> _resolve_retries
        (_retry -> child | report) -> _apply_judge (drops mislabelled children
        -> report) -> generate_rollouts on survivors -> each child scored and
        archived into a terminal CandidateReport. See docs/PIPELINE.md
        ("Evolution Candidate State") for the diagram and full status vocabulary.
        """
        parents: list[ProblemProgram] = []
        targets: list[tuple | None] = []
        for _ in range(batch_size):
            search_rng = self._next_search_rng()
            parent = self.archive.sample_parent(rng=search_rng)
            if parent is None:
                return [CandidateReport(status="no_parent", op="none")]
            parents.append(parent)

            if self.evolution_config.target_cell_injection:
                # Relabelling updates metadata without rewriting source (the
                # program id is the source hash). Read the resolved labels or a
                # resumed child would be mutated from its stale declaration
                # while the MAP correctly stored it in a different cell.
                parent_group = parent.get_group()
                parent_skill = parent.get_skill()

                if parent_group and parent_skill:
                    mutation_strategy = search_rng.choice(
                        ["mutate_skill", "mutate_group"]
                    )

                    if mutation_strategy == "mutate_skill":
                        targets.append(
                            (
                                parent_group,
                                None,
                                mutation_strategy,
                                parent_group,
                                parent_skill,
                            )
                        )
                    else:
                        targets.append(
                            (
                                None,
                                parent_skill,
                                mutation_strategy,
                                parent_group,
                                parent_skill,
                            )
                        )
                else:
                    target_group, target_skill = self.archive.sample_target_cell(
                        rng=search_rng
                    )
                    targets.append(
                        (target_group, target_skill, "mutate_both", None, None)
                    )
            else:
                targets.append(None)

        # Draw donors only AFTER every primary parent/target draw is complete,
        # and with an independent deterministic RNG. Inspiration therefore
        # changes only the prompt context -- it cannot shift the baseline arm's
        # subsequent parent choices through Python's global RNG state.
        inspirations = self._sample_structural_inspirations(parents)

        # One vLLM wake for the whole batch: the mutation generate and every
        # solver generate run while vLLM is awake; entropy (actor forward) is
        # deferred until after end_session (vLLM asleep) inside finalize_rollouts.
        self.backend.begin_session()
        try:
            if not self.evolution_config.two_stage_mutation:
                raise RuntimeError(
                    "single-stage mutation was removed along with its prompt "
                    "templates (mutation_*_prompt.txt); set "
                    "evolution.two_stage_mutation: true"
                )
            tasks, outputs, mutation_strategies = self._mutate_in_two_stages(
                parents, targets, inspirations
            )
            entries: list[dict] = []
            # Repeats also happen WITHIN one batch: 32 parents are drawn with
            # replacement from a ~10-cell archive and mutation is near
            # deterministic, so a measured 11 of 32 slots in one iteration were
            # the same few programs. self.rejected_children cannot catch those --
            # it is written when reports are finalized, once the batch is over --
            # so the first occurrence in a batch claims the id and the rest are
            # reported against it. They would have produced the same verdict:
            # the source is byte-identical and the judge is deterministic.
            in_flight: set[str] = set()
            for task, output, mutation_strategy in zip(
                tasks, outputs, mutation_strategies
            ):
                child, inst, reason, source = self._make_child_from_output(
                    task, output, in_flight=in_flight
                )
                # A cross-lineage context can move either mathematical axis,
                # even when the scheduler asked stage 1 to preserve one. Never
                # trust that held label on an inspiration-attached child.
                if child is not None and task.inspiration_donor is not None:
                    mutation_strategy = "mutate_both"
                if child is not None:
                    child.metadata["mutation_strategy"] = mutation_strategy
                if child is not None and child.program_id in in_flight:
                    # The source is already known viable, but copy status is a
                    # property of this proposal's (child, assigned donor) pair.
                    # Record it before collapsing the repeated source so every
                    # attached donor assignment remains visible in telemetry.
                    self._check_structural_inspiration_copy(task, child)
                    entries.append(
                        {
                            "report": CandidateReport(
                                status="already_rejected",
                                op=task.op,
                                child_id=child.program_id,
                                reason="duplicate of an earlier candidate in this batch",
                                **_report_context(task),
                            )
                        }
                    )
                elif child is not None and child.program_id in self.rejected_children:
                    entries.append(
                        {
                            "report": CandidateReport(
                                status="already_rejected",
                                op=task.op,
                                child_id=child.program_id,
                                reason=reason,
                                **_report_context(task),
                            )
                        }
                    )
                elif inst is not None:
                    copy_verdict = self._check_structural_inspiration_copy(task, child)
                    if copy_verdict and copy_verdict["rejected"]:
                        entries.append(
                            {
                                "report": CandidateReport(
                                    status="inspiration_copy_rejected",
                                    op=task.op,
                                    child_id=child.program_id,
                                    reason=str(copy_verdict["reason"]),
                                    source_code=_logged_source(child.source_code),
                                    ast_findings=_ast_findings(child),
                                    **_report_context(task),
                                )
                            }
                        )
                        continue
                    # A donor-copy verdict is relative to (child, donor), not
                    # to child source alone. Claim the batch-wide source id only
                    # after that pair passes, so the same source can still be
                    # judged against a different assigned donor later in this
                    # batch.
                    in_flight.add(child.program_id)
                    entries.append({"task": task, "child": child, "inst": inst})
                elif source is not None:
                    in_flight.add(child.program_id)
                    # Parses but failed verification -> eligible for one
                    # self-fix. Keep the raw output: it becomes the assistant
                    # turn in the multi-turn fix prompt.
                    # ``source`` and ``child_id`` ride INSIDE the payload on
                    # purpose. _resolve_retries writes e["report"] without
                    # calling e.clear(), so a top-level "child" key would
                    # survive into _apply_judge's selector and desync the
                    # zip(to_eval, grouped) that attributes rollout groups.
                    entries.append(
                        {
                            "_retry": {
                                "task": task,
                                "output": output,
                                "reason": reason,
                                "source": source,
                                "child_id": child.program_id if child else None,
                                "ast_findings": (
                                    list(
                                        (child.metadata.get("ast_contract") or {}).get(
                                            "findings", []
                                        )
                                    )
                                    if child
                                    else []
                                ),
                            }
                        }
                    )
                else:
                    entries.append(
                        {
                            "report": CandidateReport(
                                status=("mutation_failed" if not output else "no_code"),
                                op=task.op,
                                reason=reason,
                                **_report_context(task),
                            )
                        }
                    )

            # One-shot Reflexion self-fix: show the model its rejected program +
            # reason and re-verify. Runs inside the open vLLM session so the extra
            # generate reuses the already-awake rollout worker.
            self._resolve_retries(entries)

            # Final gate: the judge re-derives GROUP and SKILL from the
            # seed-0 problem alone and must land on both declared labels.
            # Runs BEFORE solver rollouts are spent, in the same open session.
            # Read each surviving child's descriptors off the child, replacing the
            # label stage 1 was told to write. Inside the same open session and
            # before rollouts are spent: it needs the policy and costs one
            # prefill per child. See relabel.py -- the declared label is one the
            # problem actually requires 37.5% of the time; this is 81.2%.
            # Inspiration may move both axes. Relabel it BEFORE an optional
            # judge so the judge compares against the independent readback,
            # not stage 1's self-declaration. The legacy arm keeps its cheaper
            # judge-first order and avoids relabelling candidates the judge
            # already rejects.
            if self.evolution_config.structural_inspiration:
                self._apply_relabel(entries)
                self._apply_judge(entries)
            else:
                self._apply_judge(entries)
                self._apply_relabel(entries)

            to_eval = [e for e in entries if "child" in e]
            # A candidate is graded on n FRESH seeds, not on the seed-0 instance
            # the judge saw: a generator that special-cases the instance it is
            # scored on is otherwise indistinguishable from an honest one.
            for e in to_eval:
                e["eval_instances"] = self.draw_instances(e["child"])
            pending = self.backend.generate_rollouts(
                [inst for e in to_eval for inst in e["eval_instances"]],
                n_rollouts=self.evolution_config.group_size,
            )
        finally:
            self.backend.end_session()

        grouped = self.backend.finalize_rollouts(pending)
        rollouts_by_child: dict[int, list] = {}
        cursor = 0
        for e in to_eval:
            take = len(e["eval_instances"])
            rollouts_by_child[id(e["child"])] = list(
                zip(e["eval_instances"], grouped[cursor : cursor + take])
            )
            cursor += take

        reports: list[CandidateReport] = []
        for entry in entries:
            if "report" in entry:
                reports.append(entry["report"])
                continue
            task, child, inst = entry["task"], entry["child"], entry["inst"]
            scored = rollouts_by_child.get(id(child), [])
            stats = []
            for eval_inst, rollouts in scored:
                stat = self._seed_stat(child, eval_inst, rollouts)
                if stat is not None:
                    stats.append(stat)
            if not stats:
                # Every seed lost its rollouts (timeout / worker error / stale).
                # Report the dominant reason instead of letting it masquerade as
                # an ordinary s_hat_zero.
                reasons: dict[str, int] = {}
                for _, rollouts in scored:
                    for r in rollouts:
                        key = r.reject_reason or "unknown"
                        reasons[key] = reasons.get(key, 0) + 1
                reports.append(
                    CandidateReport(
                        status="rollout_failed",
                        op=task.op,
                        child_id=child.program_id,
                        reason=(
                            max(reasons, key=reasons.get) if reasons else "no seeds"
                        ),
                        source_code=_logged_source(child.source_code),
                        ast_findings=_ast_findings(child),
                        **_report_context(task),
                    )
                )
                continue
            result = self._store_result(child, compute_rq_program(stats))
            # An R_Q of 0 no longer ends the candidate's run. It still cannot
            # take a cell from a scoring champion (competition is a strict `>`)
            # and it still contributes no training examples (the frontier band
            # in dataset.py drops p=0 and p=1), but it can hold an otherwise
            # empty cell and be drawn as a mutation parent from there.
            inserted = self._try_insert_with_telemetry(
                program=child,
                u_value=result.u_score,
                rq_score=result.rq_score,
            )
            if inserted:
                # The candidate was measured under theta_t.  Recording that
                # score now makes it a PAST score at t+1, so a new child waits
                # exactly one update before it can train (not two).
                self.previous_rq.record(
                    child.program_id, self.current_iteration, result.rq_score
                )
            if inserted:
                status = "inserted"
            elif result.s_hat <= 0.0:
                status = "s_hat_zero"
            elif result.rq_score <= 0.0:
                status = "rq_zero"
            else:
                status = "rejected_non_elite"
            reports.append(
                CandidateReport(
                    status=status,
                    op=task.op,
                    child_id=child.program_id,
                    rq_score=result.rq_score,
                    s_hat=result.s_hat,
                    u_score=result.u_score,
                    source_code=_logged_source(child.source_code),
                    ast_findings=_ast_findings(child),
                    **_report_context(task),
                )
            )
        self._memoize_rejections(reports)
        return reports

    def _next_search_rng(self) -> random.Random:
        """One restart-stable RNG for a reproductive proposal."""
        material = (
            f"{self.evolution_config.search_seed}:{self.search_draw_count}"
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        self.search_draw_count += 1
        return random.Random(seed)

    def _check_structural_inspiration_copy(
        self, task: MutationTask, child: ProblemProgram
    ) -> dict | None:
        donor = getattr(task, "inspiration_donor", None)
        if donor is None:
            return None
        verdict = self.archive.compare_with_structural_inspiration(child, donor)
        _record_inspiration_copy_gate(task, child, verdict)
        return verdict

    # Only rejections that are a property of the SOURCE, and so cannot come out
    # differently later. The code either runs or it does not; the judge is
    # deterministic at temperature 0 and sees only the seed-0 problem.
    #
    # Deliberately excluded: s_hat_zero, rq_zero, rejected_non_elite and
    # rollout_failed. Those are verdicts about the CURRENT policy and the
    # CURRENT occupant of the cell -- a program nobody can solve today is
    # exactly the program a stronger policy should get another look at, and one
    # that lost its cell should be reconsidered when the incumbent changes.
    # Memoizing them would quietly make the archive monotone in a way the
    # design does not intend.
    _DETERMINISTIC_REJECTIONS = frozenset(
        {
            "verify_failed",
            "no_code",
            "mutation_failed",
            "judge_rejected",
            "judge_input_too_large",
        }
    )

    def _memoize_rejections(self, reports: list[CandidateReport]) -> None:
        for report in reports:
            if report.child_id and report.status in self._DETERMINISTIC_REJECTIONS:
                self.rejected_children.setdefault(
                    report.child_id, f"{report.status}: {report.reason}"
                )

    def _sample_structural_inspirations(
        self, parents: list[ProblemProgram]
    ) -> list[StructuralInspirationSelection | None]:
        """One cross-lineage donor per parent, or ``None`` when disabled.

        The SHA-derived local seeds are stable across Python processes (unlike
        ``hash()``) and the draw counter is checkpointed. Each proposal gets a
        distinct draw even when the same parent appears repeatedly in a batch.
        """
        cfg = self.evolution_config
        if not cfg.structural_inspiration:
            return [None] * len(parents)

        selections: list[StructuralInspirationSelection] = []
        for parent in parents:
            material = (
                f"{cfg.structural_inspiration_seed}:"
                f"{self.inspiration_draw_count}:{parent.program_id}"
            ).encode("utf-8")
            local_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
            self.inspiration_draw_count += 1
            selections.append(
                self.archive.sample_structural_inspiration(
                    parent,
                    rng=random.Random(local_seed),
                    max_template_chars=cfg.structural_inspiration_max_chars,
                    selection_strategy=cfg.structural_inspiration_selection,
                )
            )
        return selections

    def _mutate_in_two_stages(self, parents, targets=None, inspirations=None):
        """Problem first, program second: two calls instead of one.

        The two stages are PIPELINED per parent when the backend supports it
        (``mutate_pipelined``): a parent's stage-2 generator call goes out the
        moment its own stage-1 plan lands, rather than after the slowest plan
        in the batch. Both stages are also output-capped
        (``evolution.family_max_output_tokens`` / ``generator_max_output_tokens``)
        -- uncapped, each stage ran to data.max_response_length and a 32-parent
        batch paid its worst sample twice per iteration.

        Returns ``(tasks, outputs)`` in the shape the single-call path returns,
        so everything downstream -- retries, judging, scoring, reporting -- is
        untouched. A parent whose stage-1 reply does not parse yields an empty
        output, which ``_make_child_from_output`` already reports as a failed
        mutation.

        The labels come from stage 1 and are stapled onto the program here
        rather than asked of stage 2, because the ONE thing every variant of
        this prompt got wrong was the file's tail: whatever the parent's last
        two lines contained is what the child's contained, including nothing.

        Stage 2 is TOLD the target SKILL and is asked to build a family whose
        shortest solution turns on it. It used to be asked the opposite -- to
        re-derive the label blind, so the caller could reject a mismatch -- and
        that gate is gone. It was answering the wrong question: whether the
        model can name what it wrote, not whether what it wrote demands the
        skill. It agreed 32.4% of the time against a reference, and most of its
        recoverable vocabulary was the four skills its own worked examples
        demonstrate. The cell coordinate now comes from :meth:`_apply_relabel`,
        which reads the finished program, so nothing downstream needs stage 2
        to guess.
        """
        cfg = self.evolution_config
        rotate = cfg.rotate_few_shots
        targets = targets or [None] * len(parents)
        inspirations = inspirations or [None] * len(parents)
        if len(inspirations) != len(parents):
            raise ValueError(
                "one structural-inspiration selection is required per parent"
            )
        prompt_rngs: list[tuple[random.Random, random.Random]] = []
        for parent in parents:
            draw = self.mutation_prompt_draw_count
            self.mutation_prompt_draw_count += 1

            def _prompt_rng(stage: str) -> random.Random:
                material = (
                    f"{cfg.mutation_prompt_seed}:{draw}:" f"{parent.program_id}:{stage}"
                ).encode("utf-8")
                seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
                return random.Random(seed)

            prompt_rngs.append((_prompt_rng("family"), _prompt_rng("generator")))
        family_tasks = [
            build_family_task(
                parent,
                temperature=cfg.code_temperature,
                top_p=cfg.code_top_p,
                max_output_tokens=cfg.family_max_output_tokens,
                target_cell=target,
                rotate_shots=rotate,
                rng=prompt_rngs[index][0],
                inspiration_template=(selection.template if selection else None),
                inspiration_donor=(selection.donor if selection else None),
                provenance=(
                    {"structural_inspiration": dict(selection.provenance)}
                    if selection is not None
                    else None
                ),
            )
            for index, (parent, target, selection) in enumerate(
                zip(parents, targets, inspirations)
            )
        ]

        def _stage_two(index: int, family_reply: str | None):
            """Stage-1 reply -> (plan, stage-2 task), or (None, None)."""
            plan = parse_family_plan(family_reply)
            if plan is None:
                return None, None
            provenance = dict(family_tasks[index].provenance)
            provenance["family_plan"] = {
                key: plan[key]
                for key in (
                    "STRUCTURAL MUTATION",
                    "CHILD FAMILY",
                    "WHY FINITE",
                    "GROUP",
                    "SKILL",
                )
                if key in plan
            }
            inspiration = provenance.get("structural_inspiration")
            if inspiration is not None:
                inspiration = dict(inspiration)
                inspiration["claimed_transfer"] = plan.get("STRUCTURAL MUTATION", "")
                provenance["structural_inspiration"] = inspiration
            return plan, build_generator_task(
                parents[index],
                plan,
                temperature=cfg.generator_temperature,
                top_p=cfg.generator_top_p,
                max_output_tokens=cfg.generator_max_output_tokens,
                rotate_shots=rotate,
                rng=prompt_rngs[index][1],
                provenance=provenance,
                inspiration_donor=family_tasks[index].inspiration_donor,
            )

        plans: list[dict | None] = [None] * len(parents)
        gen_task_by_i: dict[int, object] = {}
        gen_reply_by_i: dict[int, str | None] = {}

        # Preferred path: no barrier between the stages. A parent's stage-2 call
        # is submitted the moment ITS stage-1 reply lands, so the batch never
        # waits for the slowest plan before any generator starts. The blocking
        # two-call path below is the fallback for backends without it (tests,
        # dry runs) and is otherwise identical.
        pipelined = (
            family_tasks
            and hasattr(self.backend, "mutate_pipelined")
            and getattr(self.backend, "supports_pipelined_mutation", lambda: False)()
        )
        if pipelined:
            stage_two_tasks: dict[int, object] = {}

            def _build_stage_two(index: int, family_reply: str | None):
                plan, task = _stage_two(index, family_reply)
                plans[index] = plan
                if task is not None:
                    stage_two_tasks[index] = task
                return task

            _family_replies, gen_reply_by_i = self.backend.mutate_pipelined(
                family_tasks, _build_stage_two
            )
            gen_task_by_i = stage_two_tasks
        elif family_tasks:
            family_replies = self.backend.mutate(family_tasks)
            gen_tasks = []
            order = []
            for index, reply in enumerate(family_replies):
                plan, task = _stage_two(index, reply)
                plans[index] = plan
                if task is not None:
                    order.append(index)
                    gen_tasks.append(task)
            gen_task_by_i = dict(zip(order, gen_tasks))
            gen_replies = self.backend.mutate(gen_tasks) if gen_tasks else []
            gen_reply_by_i = dict(zip(order, gen_replies))

        self.stage_one_parsed = sum(1 for plan in plans if plan is not None)

        tasks = list(family_tasks)
        outputs: list[str | None] = [None] * len(parents)
        mutation_strategies = ["mutate_both"] * len(parents)
        for i, plan in enumerate(plans):
            if plan is None:
                continue
            # Preserve stage-2 provenance even when its reply is empty or
            # malformed. The report must still identify the family plan and
            # assigned donor, and a nonempty malformed reply is ``no_code`` --
            # not an indistinguishable "empty model output".
            tasks[i] = gen_task_by_i[i]
            reply = gen_reply_by_i.get(i)
            outputs[i] = reply
            source = extract_generator_code(reply or "")
            if source is None:
                continue

            target = targets[i] if targets else None
            mutation_strategy = "mutate_both"
            if target is not None and len(target) >= 5:
                _, _, mutation_strategy, parent_group, parent_skill = target
                if mutation_strategy == "mutate_skill":
                    final_group = parent_group
                    final_skill = plan.get("SKILL")
                elif mutation_strategy == "mutate_group":
                    final_group = plan.get("GROUP")
                    final_skill = parent_skill
                else:
                    final_group = plan.get("GROUP")
                    final_skill = plan.get("SKILL")
            else:
                final_group = plan.get("GROUP")
                final_skill = plan.get("SKILL")

            if final_group is None:
                final_group = "UNKNOWN"
            if final_skill is None:
                final_skill = "UNKNOWN"

            outputs[i] = set_label_declarations(source, final_group, final_skill)
            mutation_strategies[i] = mutation_strategy

        return tasks, outputs, mutation_strategies

    def _make_child_from_output(
        self,
        task: MutationTask,
        output: str | None,
        in_flight: set[str] | None = None,
    ):
        """Extract -> build -> verify a child from one model output.

        Returns ``(child, inst, reason, source)``. On success ``inst`` is the
        verified instance; on failure ``inst`` is None and ``source`` is the
        parsed program (None if the output had no parseable ``generate``).
        """
        if not output:
            return None, None, "empty model output", None
        source = extract_generator_code(output)
        if source is None:
            return None, None, "no parseable generate() in output", None
        if self.evolution_config.two_stage_mutation:
            # Stage 2's reply was already extracted and had stage 1's labels
            # stapled on; re-extracting would take the fenced block back out of
            # it and lose them again.
            source = output

        # The contract puts GROUP / SKILL in PART 1 and ends the python block
        # at `return problem, str(answer)`, so the extracted program carries no
        # labels of its own. Put them back from the prose. A label line after
        # `return` sits past the strongest completion boundary in the file and
        # was simply dropped -- 15 of 24 children on one probe -- whereas one
        # decided in PART 1, with the solution still in view, gets written.
        # Missing or out-of-vocabulary labels are left alone here so that
        # verify_program reports them through validate_label_decl as before.
        group, skill = parse_declared_labels(output)
        if group and skill:
            source = set_label_declarations(source, group, skill)

        metadata = _child_metadata(task)
        child = ProblemProgram(
            source_code=source,
            parent_id=task.parent.program_id,
            generation=task.parent.generation + 1,
            metadata=metadata,
        )
        # Before execution, not after: a repeat costs 5 sandbox runs and a judge
        # call, and the verdict is already known. `source=None` in the return
        # also keeps it out of the self-fix retry, which would re-derive the
        # same program from the same reason.
        if in_flight is not None and child.program_id in in_flight:
            # Same batch, same source. Checked here rather than in the caller so
            # the repeat skips the 5-seed execution too, not just the judge.
            return child, None, "duplicate of an earlier candidate in this batch", None
        memo = self.rejected_children.get(child.program_id)
        if memo is not None:
            return child, None, memo, None
        inst, reason = self.verify_program(child)
        return child, inst, reason, source

    def _resolve_retries(self, entries: list[dict]) -> None:
        """Collapse every ``_retry`` entry into its verify_failed report.

        The self-fix round went with the single-stage prompts: ``build_fix_task``
        replayed the original mutation conversation, and that conversation no
        longer exists in the two-stage pipeline. ``fix_retry`` therefore cannot
        be honoured -- refusing loudly beats silently skipping the fix the
        config asked for.
        """
        targets = [e for e in entries if "_retry" in e]
        if not targets:
            return
        if self.evolution_config.fix_retry:
            raise RuntimeError(
                "fix_retry was removed along with the single-stage mutation "
                "prompts; set evolution.fix_retry: false"
            )
        for e in targets:
            info = e.pop("_retry")
            task = info["task"]
            e["report"] = CandidateReport(
                status="verify_failed",
                op=task.op,
                child_id=info.get("child_id"),
                reason=info["reason"],
                source_code=_logged_source(info.get("source")),
                ast_findings=list(info.get("ast_findings") or []),
                **_report_context(task),
            )

    def _apply_relabel(self, entries: list[dict]) -> None:
        """Blindly re-read every descriptor axis the mutation was free to move.

        ``mutate_skill`` probes all eight skills, ``mutate_group`` probes all six
        groups, and ``mutate_both`` probes both. The last case is the label-free
        structural-inspiration arm: trusting stage 1 on GROUP there would make
        only half of its supposedly blind MAP coordinate independent.

        Silent no-op when ``relabel_skill`` is off. Falls back to the decoded
        text when the backend cannot return logprobs -- that is the greedy
        variant, measured at 0.641 against this one's 0.812, and it is what runs
        if patches/verl_agent_loop_sampling.py is missing its logprobs key.
        """
        self._relabel_stats = {}
        if not self.evolution_config.relabel_skill:
            return
        live = [e for e in entries if "child" in e and e.get("inst") is not None]
        if not live:
            return
        skills = list(SKILLS)
        groups = list(GROUPS)
        yes_id, no_id = self._relabel_token_ids()
        allowed = [yes_id, no_id] if yes_id is not None and no_id is not None else None

        tasks = []
        task_offsets = (
            []
        )  # To keep track of which tasks belong to which entry and label type

        for i, e in enumerate(live):
            strategy = e["child"].metadata.get("mutation_strategy", "mutate_both")
            if strategy in ("mutate_skill", "mutate_both"):
                for skill in skills:
                    tasks.append(
                        build_relabel_task(
                            parent=e["task"].parent,
                            child_family=e["inst"].problem,
                            child_source=e["child"].source_code,
                            skill=skill,
                            allowed_token_ids=allowed,
                        )
                    )
                    task_offsets.append((i, "skill", skill))
            if strategy in ("mutate_group", "mutate_both"):
                for group in groups:
                    tasks.append(
                        build_relabel_group_task(
                            parent=e["task"].parent,
                            child_family=e["inst"].problem,
                            child_source=e["child"].source_code,
                            group=group,
                            allowed_token_ids=allowed,
                        )
                    )
                    task_offsets.append((i, "group", group))

        replies = self.backend.mutate(tasks)
        pairs = getattr(self.backend, "last_mutation_logprobs", None) or []

        changed = agreed = 0
        margins: list[float] = []

        # Group replies by entry and label type
        entry_results = collections.defaultdict(lambda: collections.defaultdict(dict))
        for idx, (i, ltype, label) in enumerate(task_offsets):
            p = _relabel_p_yes(
                pairs[idx] if idx < len(pairs) else None,
                replies[idx] if idx < len(replies) else None,
                yes_id,
            )
            if p is not None:
                entry_results[i][ltype][label] = p

        for i, e in enumerate(live):
            results = entry_results[i]
            strategy = e["child"].metadata.get("mutation_strategy", "mutate_both")

            if "skill" in results:
                p_yes = results["skill"]
                self.skill_offsets.observe({s: logit(v) for s, v in p_yes.items()})
                picked, margin, _ = choose_skill(p_yes, self.skill_offsets)
                if picked is not None:
                    declared = e["child"].metadata.get("skill")
                    margins.append(margin)
                    if picked == declared:
                        agreed += 1
                    else:
                        changed += 1
                    e["child"].metadata["skill"] = picked
                    e["child"].metadata["skill_declared"] = declared
                    e["child"].metadata["skill_margin"] = round(float(margin), 4)

            if "group" in results:
                p_yes = results["group"]
                self.group_offsets.observe({g: logit(v) for g, v in p_yes.items()})
                picked, margin, _ = choose_group(p_yes, self.group_offsets)
                if picked is not None:
                    declared = e["child"].metadata.get("group")
                    margins.append(margin)
                    if picked == declared:
                        agreed += 1
                    else:
                        changed += 1
                    e["child"].metadata["group"] = picked
                    e["child"].metadata["group_declared"] = declared
                    e["child"].metadata["group_margin"] = round(float(margin), 4)

        self._relabel_stats = {
            "relabel_children": len(live),
            "relabel_decisions": changed + agreed,
            "relabel_changed": changed,
            "relabel_agreed": agreed,
            "relabel_margin_mean": (
                round(sum(margins) / len(margins), 4) if margins else 0.0
            ),
        }

    def _relabel_token_ids(self) -> tuple[int | None, int | None]:
        """Token ids for a leading YES / NO, or (None, None) without a tokenizer."""
        cached = getattr(self, "_relabel_ids", None)
        if cached is not None:
            return cached
        ids: tuple[int | None, int | None] = (None, None)
        tok = getattr(self.backend, "tokenizer", None)
        if tok is not None:
            try:
                y = tok.encode("YES", add_special_tokens=False)
                n = tok.encode("NO", add_special_tokens=False)
                if y and n:
                    ids = (int(y[0]), int(n[0]))
            except Exception:
                ids = (None, None)
        self._relabel_ids = ids
        return ids

    def _apply_judge(self, entries: list[dict]) -> None:
        """Label gate: the judge must independently reach both declared labels.

        For each ``{"task","child","inst"}`` entry the judge sees ONLY the seed-0
        problem text and the supplied answer -- never the source, the declared
        GROUP/SKILL, or the parent. It runs its own validity gates, reconstructs
        the shortest human solution, and returns a GROUP and a SKILL. The child
        survives only when both agree with what it declared.

        That is the whole point of the redesign. The Evolver labels its own
        output, and a label it invented is unfalsifiable until something else
        derives one from the visible problem. A child whose statement says one
        thing and whose label says another does not merely score badly -- it is
        filed in the wrong cell, so the MAP's coordinates stop meaning anything
        and both the parent sampler and the coverage metric read a fiction.

        With ``evaluator_provider=policy`` one batched ``mutate`` runs inside the
        already-open vLLM session; with ``openai`` the same messages go to the
        Responses API. Judge configuration or runtime failures raise immediately
        -- continuing would silently admit unlabelled children.
        """
        if not self.evolution_config.use_evaluator:
            return
        targets = [e for e in entries if "child" in e]
        if not targets:
            return
        targets = self._drop_oversized_judge_inputs(targets)
        if not targets:
            return
        if self.evolution_config.evaluator_provider == "openai":
            outputs = self._run_openai_judge(targets)
        else:
            judge_tasks = [
                build_judge_task(
                    e["child"],
                    e["inst"].problem,
                    e["inst"].answer,
                    temperature=self.evolution_config.judge_temperature,
                    top_p=self.evolution_config.judge_top_p,
                    rubric_file=self.evolution_config.judge_rubric,
                )
                for e in targets
            ]
            try:
                outputs = self.backend.mutate(judge_tasks)
            except Exception as exc:
                raise EvaluatorRuntimeError(
                    "Judge backend failed; aborting R_Q-Evolve instead of "
                    f"continuing with an invalid curriculum: {exc}"
                ) from exc
        for e, output in zip(targets, outputs):
            if isinstance(output, Exception):
                raise EvaluatorRuntimeError(
                    "Judge call failed; aborting R_Q-Evolve instead of "
                    f"discarding the candidate and continuing: {output}"
                ) from output
            child = e["child"]
            verdict = parse_judge_verdict(output or "")
            accepted, reason = judge_accepts(
                verdict,
                child.get_group(),
                child.get_skill(),
            )
            # Recorded either way: the disagreements are the measurement that
            # says whether the Evolver's self-labelling is trustworthy at all.
            child.metadata["judge"] = {
                "accepted": accepted,
                "reason": reason,
                **verdict.to_dict(),
            }
            tally = getattr(self, "judge_tally", None)
            if tally is not None:
                tally["reached"] += 1
                tally["agreed"] += int(accepted)
                tally["group_agreed"] += int(verdict.group == child.get_group())
                tally["skill_agreed"] += int(verdict.skill == child.get_skill())
                tally["skill_none"] += int(verdict.skill is None)
                tally["group_none"] += int(verdict.group is None)
                if not accepted:
                    key = (
                        "failed_closed"
                        if verdict.group is None or verdict.skill is None
                        else "label_mismatch"
                    )
                    tally[key] += 1
                # Which SKILLs the judge is willing to emit at all. A label it
                # never returns is a label no child can be archived under, so
                # this bounds reachable coverage independently of how often the
                # Evolver is right -- measured at 3 of 8 for one judge and 6 of
                # 8 for another over the same corpus.
                name = verdict.skill or "none"
                self.judge_skill_counts[name] = self.judge_skill_counts.get(name, 0) + 1
            if accepted:
                continue
            task = e["task"]
            e.clear()
            e["report"] = CandidateReport(
                status="judge_rejected",
                op=task.op,
                child_id=child.program_id,
                reason=reason,
                source_code=_logged_source(child.source_code),
                ast_findings=_ast_findings(child),
                **_report_context(task),
            )

    def _drop_oversized_judge_inputs(self, targets: list[dict]) -> list[dict]:
        """Convert candidates with an over-budget judge prompt into reports.

        The budget is read off the backend's own context window when it exposes
        one, so this tracks ``rollout.max_model_len`` instead of duplicating it.
        Without a window to read, nothing is dropped: refusing to guess is
        better than silently rejecting valid children on an invented limit.

        Batched generation fails as a unit, so one oversized prompt would
        otherwise raise for a whole batch that has nothing wrong with it.
        """
        budget = getattr(self.backend, "max_prompt_chars", None)
        if budget is None:
            window = getattr(self.backend, "max_model_len", None)
            if not window:
                return targets
            # Reserve room for the verdict itself, then convert the token
            # window to characters at a deliberately conservative ratio -- 2
            # chars/token is below anything real text produces, so the guard
            # trips early rather than one token too late.
            budget = max(0, int(window) - 512) * 2
        entry_rubric = self.evolution_config.judge_rubric
        kept: list[dict] = []
        for entry in targets:
            task = build_judge_task(
                entry["child"],
                entry["inst"].problem,
                entry["inst"].answer,
                rubric_file=entry_rubric,
            )
            size = sum(len(m.get("content", "")) for m in task.messages or [])
            if size <= budget:
                kept.append(entry)
                continue
            mutation_task, child = entry["task"], entry["child"]
            self.events.append(
                {
                    "event": "judge_input_too_large",
                    "program_id": child.program_id,
                    "op": mutation_task.op,
                    "prompt_chars": size,
                    "budget_chars": budget,
                }
            )
            entry.clear()
            entry["report"] = CandidateReport(
                status="judge_input_too_large",
                op=mutation_task.op,
                child_id=child.program_id,
                reason=(
                    f"judge prompt is {size} chars, over the {budget}-char "
                    "context budget; dropping this candidate instead of "
                    "failing the batch"
                ),
                source_code=_logged_source(child.source_code),
                ast_findings=_ast_findings(child),
                **_report_context(mutation_task),
            )
        return kept

    def _run_openai_judge(self, targets: list[dict]) -> list[str | Exception]:
        cfg = OpenAIEvaluatorConfig(
            model=self.evolution_config.evaluator_model,
            reasoning_effort=self.evolution_config.evaluator_reasoning_effort,
            timeout_s=self.evolution_config.evaluator_timeout_s,
            max_output_tokens=self.evolution_config.evaluator_max_output_tokens,
        )

        def judge_one(e: dict) -> str | Exception:
            messages = build_judge_messages(
                e["inst"].problem,
                e["inst"].answer,
                rubric_file=self.evolution_config.judge_rubric,
            )
            try:
                return evaluate_messages_with_openai(messages, cfg)
            except Exception as exc:
                return exc

        max_workers = min(self.evolution_config.evaluator_concurrency, len(targets))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(judge_one, targets))

    def _score_from_rollouts(
        self,
        program: ProblemProgram,
        rollouts: list[RolloutRecord],
        *,
        instance: ProblemInstance | None = None,
    ) -> RQResult:
        """One instance's rollouts -> a one-seed R_Q. Kept for single-instance
        callers; the program-level path is :meth:`evaluate_program`."""
        stat = self._seed_stat(
            program, instance, rollouts, seed=int(getattr(instance, "seed", 0) or 0)
        )
        return self._store_result(program, compute_rq_program([stat] if stat else []))

    def _seed_stat(
        self,
        program: ProblemProgram,
        instance: ProblemInstance | None,
        rollouts: list[RolloutRecord],
    ) -> "SeedStat | None":
        """Collapse one instance's m rollouts into its (s, L, U) triple.

        Rejected rollouts (timeout / stale / worker error) leave the estimate
        entirely -- they were never drawn from the policy, so counting them as
        failures would understate s. A seed whose rollouts were ALL rejected
        returns None and is dropped from the seed set rather than scored as 0:
        one transient infra failure must not look like an unsolvable instance.
        """
        accepted = [
            r for r in rollouts if getattr(r, "status", "accepted") == "accepted"
        ]
        if not accepted:
            return None
        flags = [
            (
                clean_and_grade_solver_rollout(record, instance)[2]
                if instance is not None
                else bool(record.correct)
            )
            for record in accepted
        ]
        return score_seed(
            seed=int(getattr(instance, "seed", 0) or 0),
            correct_flags=flags,
            rollout_entropies=[r.entropy for r in accepted],
        )

    @staticmethod
    def _store_result(program: ProblemProgram, result: RQResult) -> RQResult:
        program.s_hat = result.s_hat
        program.u_score = result.u_score
        program.rq_score = result.rq_score
        program.metadata["dispersion"] = result.dispersion
        program.metadata["num_seeds"] = result.num_seeds
        return result

    def draw_instances(
        self,
        program: ProblemProgram,
        n_seeds: int | None = None,
    ) -> list[ProblemInstance]:
        """Execute ``program`` on n FRESH seeds from its never-reused stream.

        Seeds it cannot execute are skipped, not retried at a lower seed: the
        stream only moves forward, so a program that fails on some seeds is
        simply graded on fewer instances rather than on a re-drawn set.
        """
        n = int(n_seeds or 1)
        instances: list[ProblemInstance] = []
        for seed in self.seed_stream.take(program.program_id, n):
            inst = program.execute(seed=seed)
            if inst is not None:
                instances.append(inst)
        return instances

    def evaluate_programs(
        self,
        programs: list[ProblemProgram],
        *,
        store_replay: bool = False,
        instance_counts: list[int] | None = None,
    ) -> list[RQResult | None]:
        """Score each program on ONE fresh instance x G rollouts, one session.

        ``instance_counts`` may ask for more than one instance per program. The
        extras exist only to fill the training batch when the frontier is
        smaller than it (see :meth:`_allocate_instances`); **R_Q is always the
        first instance alone**, so the fitness of a champion that filled three
        slots means the same thing as one that filled one. All the instances go
        to replay, because every rollout that was paid for should produce a
        gradient.

        Every program's instances go into ONE flat rollout batch and are
        regrouped afterwards, so the evaluation costs the same number of
        wake/sleep cycles regardless of how the instances are distributed.

        With ``store_replay`` the rollouts are kept in :attr:`replay` instead of
        being discarded, and become the solver's training batch. That is only
        correct for the re-scoring pass: those rollouts come from the same
        theta_t that the following update starts from, so training on them is
        on-policy. Candidate scoring passes ``False`` -- a child inserted this
        iteration has no previous score yet and does not train until the next one.
        """
        if not programs:
            return []
        counts = instance_counts or [1] * len(programs)
        per_program: list[list[ProblemInstance]] = [
            self.draw_instances(p, n_seeds=k) for p, k in zip(programs, counts)
        ]
        flat = [inst for group in per_program for inst in group]
        if not flat:
            return [None] * len(programs)

        self.backend.begin_session()
        try:
            pending = self.backend.generate_rollouts(
                flat, n_rollouts=self.evolution_config.group_size
            )
        finally:
            self.backend.end_session()
        grouped = self.backend.finalize_rollouts(pending)
        payloads = getattr(pending, "payloads", None) or []

        results: list[RQResult | None] = []
        cursor = 0
        for program, instances in zip(programs, per_program):
            stats = []
            for instance in instances:
                rollouts = grouped[cursor] if cursor < len(grouped) else []
                cursor += 1
                stat = self._seed_stat(program, instance, rollouts)
                if stat is not None:
                    stats.append(stat)
                    if store_replay:
                        self.replay.store(
                            program.program_id,
                            instance,
                            rollouts,
                            payload=(
                                payloads[cursor - 1]
                                if cursor - 1 < len(payloads)
                                else None
                            ),
                        )
            if not stats:
                # Every seed lost its rollouts. Scoring this as 0 would zero the
                # program's s_hat/u_score in place and let one transient infra
                # failure evict a champion from its true niche.
                self.events.append(
                    {
                        "event": "eval_rollout_failed",
                        "program_id": program.program_id,
                        "seeds_drawn": len(instances),
                    }
                )
                results.append(None)
                continue
            # The FIRST instance is the measurement; any others were drawn to
            # fill the batch. Scoring on all of them would make R_Q depend on
            # how many slots a champion happened to be given.
            results.append(self._store_result(program, compute_rq_program(stats[:1])))
        return results

    def _allocate_instances(self, champions: list[ProblemProgram]) -> list[int]:
        """One fresh instance each, then extras by the previous raw R_Q.

        The trainer needs ``train_batch_target`` prompts and the frontier is
        routinely smaller than that (median 18 against a target of 16-32 over
        the 8B run), so the shortfall is covered by drawing further FRESH seeds
        from the highest-R_Q champions rather than by shrinking the batch.
        Extra seeds are extra problems, not extra rollouts on the same problem
        -- LILO found scaling the number of levels more effective than scaling
        rollouts per level.

        Ranked by the same previous raw R_Q used to build the training batch.
        ``program.rq_score`` is deliberately not the ranking key: it is only
        overwritten by the current re-score before the batch is built, whereas
        ``PreviousRQScoreboard`` preserves exactly the t-1 value without an EWMA.

        THE SHORTFALL IS COUNTED OVER FRONTIER CHAMPIONS, NOT ALL OF THEM, and
        that distinction is the whole point of this function. Only a champion
        with 0 < s_hat < 1 contributes a training row -- ``is_frontier`` in
        dataset.py drains the rest -- so comparing the target against
        ``len(champions)`` asks the wrong question. Measured on the 4B run: the
        archive reached 48 champions while only 18-22 were on the frontier, so
        ``target(32) <= n(48)`` held, every champion got exactly one instance,
        and the training set came out at 19 rows against a 32-prompt batch.
        VerlDynamicDataset is built with ``min_size=train_batch_size``
        (verl_adapter.py), so the missing 13 rows were filled by wrapping onto
        rows already in the batch -- the same instance, and under replay the
        same stored rollouts, counted twice in the update. Over iterations 100+
        that was 41% of every batch, peaking at 75% when the frontier fell to 8.
        Allocating against the frontier count closes it: extras go to frontier
        champions until the batch is full of distinct instances.
        """
        n = len(champions)
        counts = [1] * n
        target = int(self.evolution_config.train_batch_target)
        if n == 0:
            return counts
        low, high = self.evolution_config.frontier_s_hat_range
        frontier = [
            i
            for i in range(n)
            if is_frontier(float(getattr(champions[i], "s_hat", 0.0) or 0.0), low, high)
        ]
        # Nothing on the frontier yet (bootstrap, or every champion degenerate):
        # fall back to the old whole-population behaviour rather than refusing
        # to allocate, so the batch is still filled with something.
        pool = frontier or list(range(n))
        if target <= len(pool):
            return counts
        past_scores = {
            i: self.previous_rq.selection_score(
                champions[i].program_id, self.current_iteration
            )
            for i in pool
        }
        # Bootstrap / a pre-scoreboard checkpoint has no past-only value at
        # all.  Only in that all-missing state may the already-stored score seed
        # the first batch.  Once any history exists, a missing program is a new
        # child and must not receive priority before its next iteration.
        if all(score is None for score in past_scores.values()):
            rank_score = {
                i: float(getattr(champions[i], "rq_score", 0.0) or 0.0) for i in pool
            }
        else:
            rank_score = {
                i: (float(score) if score is not None else float("-inf"))
                for i, score in past_scores.items()
            }
        order = sorted(pool, key=lambda i: rank_score[i], reverse=True)
        for j in range(target - len(pool)):
            counts[order[j % len(order)]] += 1
        return counts

    def reevaluate_champions(self) -> None:
        """Refresh champion scores under the current backend.

        One vLLM wake/sleep for the whole champion set (was one per champion).
        """
        champions = list(self.archive.champions())
        if not champions:
            return
        # Degenerate champions (s_hat in {0, 1}) are held in the archive but not
        # necessarily re-measured. They cannot enter the training batch -- the
        # frontier band drains them in dataset.py -- so under
        # replay_training_batch their rollouts are generated and discarded.
        # See EvolutionConfig.reevaluate_degenerate_every for the budget this
        # recovers and, importantly, for what skipping them costs at n=1.
        every = int(self.evolution_config.reevaluate_degenerate_every)
        if every != 1:
            low, high = self.evolution_config.frontier_s_hat_range
            due = every > 0 and (self.current_iteration % every == 0)
            if not due:
                skipped = [
                    c
                    for c in champions
                    if not is_frontier(
                        float(getattr(c, "s_hat", 0.0) or 0.0), low, high
                    )
                ]
                if skipped:
                    champions = [c for c in champions if c not in skipped]
                    self.events.append(
                        {
                            "event": "degenerate_reevaluation_skipped",
                            "iteration": self.current_iteration,
                            "skipped": len(skipped),
                            "rescored": len(champions),
                        }
                    )
        if not champions:
            return
        # Fresh seeds every re-scoring. A program that degenerates on only an
        # eps-fraction of its seed space is caught with probability
        # 1-(1-eps)^n per re-evaluation, so tail-overfitted generators leave the
        # archive within a few iterations instead of surviving on a lucky seed.
        results = self.evaluate_programs(
            champions,
            store_replay=True,
            instance_counts=self._allocate_instances(champions),
        )

        for champion, result in zip(champions, results):
            if result is None:
                # all rollouts rejected (transient timeout/worker error) --
                # keep the champion's previous scores and niche untouched.
                continue
            # Every champion is rescored against the same weights before any of
            # them moves, so re-binning cannot depend on the order of this loop.
            # Recorded BEFORE the archive moves, so the score is attributed to
            # the iteration that measured it regardless of what happens to the
            # champion's cell afterwards.
            self.previous_rq.record(
                champion.program_id, self.current_iteration, result.rq_score
            )
            self.archive.remove_program(champion.program_id)
            # A champion whose rescore comes back at R_Q = 0 is reinserted, not
            # dropped. Removing it emptied the cell and left the grid with no
            # record that the region had ever been reached; keeping it means the
            # cell reports what is actually the best generator found for it.
            inserted = self._try_insert_with_telemetry(
                program=champion,
                u_value=result.u_score,
                rq_score=result.rq_score,
                source="champion_reevaluation",
            )
            if not inserted:
                self.events.append(
                    {
                        "event": "champion_removed_after_reevaluation",
                        "program_id": champion.program_id,
                        "reason": "lost_target_bin_competition",
                        "s_hat": result.s_hat,
                        "rq_score": result.rq_score,
                    }
                )

    def _try_insert_with_telemetry(
        self,
        *,
        program: ProblemProgram,
        u_value: float,
        rq_score: float,
        source: str = "mutation",
    ) -> bool:
        target_cell = self.archive.target_cell(program)
        incumbent = (
            self.archive.grid[target_cell].champion if target_cell is not None else None
        )
        inserted = self.archive.try_insert(
            program=program,
            u_value=u_value,
            rq_score=rq_score,
        )
        if not inserted:
            # WHY it was refused. try_insert distinguishes six rejections --
            # unlabelled, seed_variation, duplicate_behavior, duplicate_template,
            # near_duplicate_template, structural_duplicate -- and losing a cell
            # to a higher-R_Q incumbent is a seventh. All seven were reported as
            # one bare "rejected_non_elite" with no reason, so the novelty gates
            # could not be told apart from ordinary cell competition: 176 such
            # rejections in the 65-iteration run, none attributable.
            #
            # This is the most expensive rejection in the pipeline. try_insert
            # takes rq_score, so the child has already paid for verification,
            # relabelling and its full rollout budget by the time it gets here.
            self.events.append(
                {
                    "event": "insert_rejected",
                    "source": source,
                    "reason": (program.metadata or {}).get(
                        "archive_status", "lost_cell_contest"
                    ),
                    "program_id": program.program_id,
                    "rq_score": float(rq_score),
                    "target_cell": list(target_cell) if target_cell else None,
                }
            )
        if (
            inserted
            and incumbent is not None
            and incumbent.program_id != program.program_id
        ):
            self.events.append(
                {
                    "event": "champion_replaced",
                    "source": source,
                    "target_cell": list(target_cell),
                    "target_labels": list(self.archive.cell_labels(target_cell)),
                    "incoming_program_id": program.program_id,
                    "incoming_rq": float(rq_score),
                    "evicted_program_id": incumbent.program_id,
                    "evicted_rq": float(incumbent.rq_score),
                }
            )
        return inserted

    def refresh_dataset(
        self,
        *,
        warmup: bool = False,
        training_champions: list[ProblemProgram] | None = None,
    ) -> None:
        # ``training_champions`` is captured immediately after incumbent
        # re-scoring and before mutation.  Keeping it separate from the live
        # archive makes the temporal contract explicit: children created in
        # iteration t may train from t+1, while every row used now has a replay
        # group sampled under the current theta_t.
        champions = (
            list(training_champions)
            if training_champions is not None
            else list(self.archive.champions())
        )
        # Record each champion's frontier decision (the learnability filter that
        # decides which problems feed training) with the SAME predicate
        # build_training_examples uses -- observability only, no behavior change.
        low, high = self.evolution_config.frontier_s_hat_range
        self.last_frontier = [
            {
                "program_id": champion.program_id,
                "s_hat": round(float(champion.s_hat), 4),
                "rq_score": round(float(champion.rq_score), 6),
                "decision": (
                    "in_frontier"
                    if is_frontier(champion.s_hat, low, high)
                    else "s_hat_out_of_range"
                ),
            }
            for champion in champions
        ]
        if self.training_config.replay_training_batch:
            replay_budget = (
                self.training_config.training_budget
                if self.training_config.training_budget is not None
                else self.evolution_config.train_batch_target
            )
            examples = build_replay_training_examples(
                champions,
                replay=self.replay,
                previous_rq=self.previous_rq,
                iteration=self.current_iteration,
                frontier_s_hat_range=self.evolution_config.frontier_s_hat_range,
                training_budget=replay_budget,
                warmup=warmup,
            )
            if not examples and not warmup:
                # Nothing has a prior measurement to be selected on. That is the
                # same situation as bootstrap -- a cold resume, or an archive
                # that turned over completely -- and the honest response is to
                # fall back to the current scores rather than hand the trainer
                # an empty dataloader. The lag re-engages the moment any
                # champion carries history again.
                if champions and all(
                    self.previous_rq.selection_score(
                        c.program_id, self.current_iteration
                    )
                    is None
                    for c in champions
                ):
                    self.events.append(
                        {
                            "event": "replay_warmup_fallback",
                            "iteration": self.current_iteration,
                            "champions": len(champions),
                        }
                    )
                    examples = build_replay_training_examples(
                        champions,
                        replay=self.replay,
                        previous_rq=self.previous_rq,
                        iteration=self.current_iteration,
                        frontier_s_hat_range=(
                            self.evolution_config.frontier_s_hat_range
                        ),
                        training_budget=replay_budget,
                        warmup=True,
                    )
            if not examples and not warmup and champions:
                # The frontier admitted nobody: every champion currently reads
                # degenerate (s_hat exactly 0 or 1). Handing the trainer an
                # empty dataloader ends the run -- verl raises IndexError out of
                # VerlDynamicDataset and there is no recovery path. A run died
                # this way at iteration 65 with 29 of 30 champions degenerate.
                #
                # Training on them produces no gradient (RLOO advantages are
                # identically 0 when a group scores alike), so this costs one
                # step and changes nothing. It keeps the loop alive so the next
                # re-evaluation can pull champions back onto the band, which is
                # what actually recovers: 20.3% of degenerate readings are
                # followed by a live one at the very next measurement.
                self.events.append(
                    {
                        "event": "frontier_empty_fallback",
                        "iteration": self.current_iteration,
                        "champions": len(champions),
                    }
                )
                examples = build_replay_training_examples(
                    champions,
                    replay=self.replay,
                    previous_rq=self.previous_rq,
                    iteration=self.current_iteration,
                    frontier_s_hat_range=self.evolution_config.frontier_s_hat_range,
                    training_budget=replay_budget,
                    warmup=True,
                    allow_degenerate=True,
                )
            # The dataset is modulo-padded up to train_batch_size. Under replay
            # that repeats IDENTICAL responses rather than resampling them, so a
            # short batch double-counts the same rollouts. Harmless but worth
            # seeing: it shrinks as the archive fills.
            budget = replay_budget
            if examples and budget and len(examples) < budget:
                self.events.append(
                    {
                        "event": "replay_batch_short",
                        "iteration": self.current_iteration,
                        "examples": len(examples),
                        "requested": budget,
                    }
                )
            self.dataset.update(examples)
            self.dataset_refresh_count += 1
            return
        examples = build_training_examples(
            champions,
            instances_per_program=self.training_config.instances_per_program,
            training_budget=self.training_config.training_budget,
            frontier_s_hat_range=self.evolution_config.frontier_s_hat_range,
            used_seeds=self.used_seeds,
            strict_anti_reuse=self.training_config.strict_anti_reuse,
            select_lowest_rq_first=self.training_config.select_lowest_rq_first,
            select_random_order=self.training_config.select_random_order,
            # Vary the shuffle per refresh so the random ablation isn't frozen to
            # one ordering across outer iterations, while staying reproducible.
            select_random_seed=(
                self.training_config.select_random_seed + self.dataset_refresh_count
            ),
            select_ignores_uncertainty=self.evolution_config.select_ignores_uncertainty,
            select_ignores_variance=self.evolution_config.select_ignores_variance,
        )
        self.dataset.update(examples)
        self.dataset_refresh_count += 1

    _USED_SEEDS_FILE = "rq_used_seeds.json"
    _ITERATION_FILE = "rq_iteration.json"

    def save_state(self, directory: str | Path, iteration: int | None = None) -> None:
        """Persist the MAP-Elites archive + used_seeds for restart.

        The verl weight checkpoint does NOT include the archive, so without this
        a resumed run restarts from a seed-only grid and loses every evolved
        champion. Called once per outer iteration (after evolution) so the
        latest archive is always on disk.

        ``archive.json`` is the latest snapshot (overwritten each call, used for
        resume). When ``iteration`` is given a versioned copy
        ``archive_iter{iteration}.json`` is also written so the per-step evolution
        trajectory is recoverable.
        """
        import shutil

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.archive.save(directory)
        if iteration is not None:
            shutil.copyfile(
                directory / "archive.json",
                directory / f"archive_iter{int(iteration)}.json",
            )
        used = {pid: sorted(seeds) for pid, seeds in self.used_seeds.items()}
        (directory / self._USED_SEEDS_FILE).write_text(
            json.dumps(
                {
                    "strict_anti_reuse": self.training_config.strict_anti_reuse,
                    "instances_per_program": self.training_config.instances_per_program,
                    "used_seeds": used,
                    "seed_cursor": self.seed_stream.to_dict(),
                    "previous_rq_scores": self.previous_rq.to_dict(),
                    "rejected_children": self.rejected_children,
                    # Per-skill YES-bias, learned across iterations. A cold
                    # start re-learns it from the first batch alone, and the
                    # picker is worst exactly while the means are noisiest.
                    "skill_offsets": self.skill_offsets.to_dict(),
                    "group_offsets": self.group_offsets.to_dict(),
                    "inspiration_draw_count": self.inspiration_draw_count,
                    "mutation_prompt_draw_count": self.mutation_prompt_draw_count,
                    "search_draw_count": self.search_draw_count,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (directory / self._ITERATION_FILE).write_text(
            json.dumps(
                {"current_iteration": self.current_iteration},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def append_evolution_log(
        self,
        directory: str | Path,
        iteration: int,
        metrics: dict,
        reports: list[CandidateReport] | None = None,
    ) -> None:
        """Append one JSON line per outer iteration to ``evolution_log.jsonl``.

        Unlike the archive (latest snapshot only), this is append-only, so the
        full evolution trajectory is preserved: per-iteration metrics plus every
        candidate report. ``CandidateReport.status`` is one of: no_parent,
        mutation_failed, no_code, verify_failed, judge_rejected,
        judge_input_too_large, inspiration_copy_rejected, rollout_failed,
        s_hat_zero, rq_zero, inserted, rejected_non_elite (each with op, rq_score, s_hat,
        u_score; see docs/PIPELINE.md "Evolution Candidate State").

        Each report also carries ``source_code`` (truncated at
        ``_MAX_LOGGED_SOURCE_CHARS``) and ``ast_findings``. Only champions land
        in the archive snapshots, so without those two a rejected child's
        program is unrecoverable and a gate's false-positive rate cannot be
        measured against the population it rejects. ``reports`` defaults to
        ``self.last_reports``.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        reports = self.last_reports if reports is None else reports
        record = {
            "iteration": int(iteration),
            "metrics": metrics,
            "reports": [asdict(r) for r in reports],
            "frontier": self.last_frontier,
        }
        with (directory / "evolution_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_state(self, directory: str | Path) -> bool:
        """Restore archive + used_seeds written by :meth:`save_state`.

        Returns True if a snapshot was found and loaded (the caller should then
        skip seed bootstrapping), False otherwise. The dataset is refreshed from
        the restored archive so training can resume immediately.
        """
        directory = Path(directory)
        if not (directory / "archive.json").exists():
            return False
        n_champions = self.archive.load(directory)
        if n_champions == 0:
            # A snapshot written before the GROUP x SKILL migration carries no
            # SKILL label, so ``archive.load`` drops every champion and hands
            # back an empty grid. Reporting that as a successful resume made the
            # caller skip seed bootstrapping, and the trainer then died on its
            # first batch with "VerlDynamicDataset is empty". Nothing to resume
            # is the same as no snapshot. Returning before ``used_seeds`` is
            # restored matters too: those seeds belong to the run that produced
            # the unusable archive, and carrying them over would retire seeds
            # the bootstrap is about to need.
            return False
        missing_prompt_draw_count = True
        missing_search_draw_count = True
        seeds_file = directory / self._USED_SEEDS_FILE
        if seeds_file.exists():
            payload = json.loads(seeds_file.read_text(encoding="utf-8"))
            self.used_seeds = {
                pid: set(seeds) for pid, seeds in payload.get("used_seeds", {}).items()
            }
            self.previous_rq = PreviousRQScoreboard.from_dict(
                # Pre-migration archives used the key ``lagged_scores``.
                # Accept it for offline inspection; changed training semantics
                # still require a fresh run rather than checkpoint resume.
                payload.get("previous_rq_scores")
                or payload.get("lagged_scores"),
            )
            self.rejected_children = dict(payload.get("rejected_children") or {})
            offsets = payload.get("skill_offsets")
            if offsets:
                self.skill_offsets = SkillOffsets.from_dict(offsets)
            group_offsets = payload.get("group_offsets")
            if group_offsets:
                self.group_offsets = GroupOffsets.from_dict(group_offsets)
            self.inspiration_draw_count = int(
                payload.get("inspiration_draw_count", self.inspiration_draw_count)
            )
            if "mutation_prompt_draw_count" in payload:
                self.mutation_prompt_draw_count = int(
                    payload["mutation_prompt_draw_count"]
                )
                missing_prompt_draw_count = False
            if "search_draw_count" in payload:
                self.search_draw_count = int(payload["search_draw_count"])
                missing_search_draw_count = False
            cursor = payload.get("seed_cursor")
            if cursor:
                self.seed_stream = SeedStream.from_dict(cursor)
            else:
                # A snapshot from before the stream existed records WHICH seeds
                # were emitted, not how far the stream ran. Resume one past the
                # largest so nothing is issued twice.
                self.seed_stream = SeedStream.from_used_seeds(self.used_seeds)
        iteration_file = directory / self._ITERATION_FILE
        if iteration_file.exists():
            payload = json.loads(iteration_file.read_text(encoding="utf-8"))
            self.current_iteration = int(
                payload.get("current_iteration", self.current_iteration)
            )
        if missing_prompt_draw_count:
            # Every successful primary-parent draw increments total_selections
            # exactly once, so it is the exact legacy proposal counter.
            self.mutation_prompt_draw_count = self.archive.total_selections
        if missing_search_draw_count:
            self.search_draw_count = self.archive.total_selections
        self.refresh_dataset()
        self.events.append({"event": "archive_restored", "champions": n_champions})
        return True


def _answer_parseable(answer: str) -> bool:
    try:
        from sympy import sympify

        sympify(str(answer).replace("^", "**"))
        return True
    except Exception:
        try:
            float(answer)
            return True
        except Exception:
            return False
