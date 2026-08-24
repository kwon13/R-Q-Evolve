import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .archive import MAPElitesArchive
from .ast_contract import check_generator_contract, check_problem_text
from .backends import EvolutionBackend, RolloutRecord
from .code_utils import (
    extract_generator_code,
    lint_generator_source,
    lint_mutation_generator_source,
    lint_problem_instance,
    set_label_declarations,
)
from .concepts import validate_label_decl
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
from .program import ProblemInstance, ProblemProgram
from .replay import LaggedScoreboard, RolloutReplayBuffer
from .seed_stream import SeedStream
from .prompts import (
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


def _logged_source(source: str | None) -> str | None:
    if source is None:
        return None
    return source[:_MAX_LOGGED_SOURCE_CHARS]


def _ast_findings(program: "ProblemProgram | None") -> list[str]:
    if program is None:
        return []
    verdict = program.metadata.get("ast_contract") or {}
    return list(verdict.get("findings") or [])


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
    # This iteration's re-scoring rollouts, kept so the solver update trains on
    # them instead of paying for a second sampling pass over the same programs.
    replay: RolloutReplayBuffer = field(default_factory=RolloutReplayBuffer)
    # Which elites train is decided by PAST scores; what they train on is this
    # iteration's rollouts. Scoring and selecting on the same draw would keep
    # whichever elite's measurement noise landed high.
    lagged: LaggedScoreboard = field(default_factory=LaggedScoreboard)
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

    def __post_init__(self) -> None:
        self.lagged.ewma_alpha = float(self.training_config.lagged_selection_ewma)
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

    def load_seed_programs(self, seed_dir: str | Path) -> list[ProblemProgram]:
        programs: list[ProblemProgram] = []
        for path in sorted(Path(seed_dir).glob("*.py")):
            program = ProblemProgram.from_file(path, generation=0)
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
            # "first scored this iteration" at t=0, the lagged filter would
            # exclude all of them, and the first training batch would be empty.
            self.lagged.record(program.program_id, -1, result.rq_score)
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
        seen_problems: set[str] = set()
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
            seen_problems.add(" ".join(inst.problem.split()))

        if n > 1 and len(seen_problems) <= 1:
            return None, "program does not vary its visible problem across seeds"
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

        self.refresh_dataset()
        stats = self.archive.stats()
        frontier_in = sum(1 for f in self.last_frontier if f["decision"] == "in_frontier")
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
        for _ in range(batch_size):
            parent = self.archive.sample_parent()
            if parent is None:
                return [CandidateReport(status="no_parent", op="none")]
            parents.append(parent)

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
            tasks, outputs = self._mutate_in_two_stages(parents)
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
            for task, output in zip(tasks, outputs):
                child, inst, reason, source = self._make_child_from_output(
                    task, output, in_flight=in_flight
                )
                if child is not None and child.program_id in in_flight:
                    entries.append(
                        {
                            "report": CandidateReport(
                                status="already_rejected",
                                op=task.op,
                                child_id=child.program_id,
                                reason="duplicate of an earlier candidate in this batch",
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
                            )
                        }
                    )
                elif inst is not None:
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
                                "ast_findings": list(
                                    (child.metadata.get("ast_contract") or {}).get(
                                        "findings", []
                                    )
                                )
                                if child
                                else [],
                            }
                        }
                    )
                else:
                    entries.append(
                        {
                            "report": CandidateReport(
                                status=(
                                    "mutation_failed" if not output else "no_code"
                                ),
                                op=task.op,
                                reason=reason,
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
            self._apply_judge(entries)

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
                )
            )
        self._memoize_rejections(reports)
        return reports

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
            if (
                report.child_id
                and report.status in self._DETERMINISTIC_REJECTIONS
            ):
                self.rejected_children.setdefault(
                    report.child_id, f"{report.status}: {report.reason}"
                )

    def _mutate_in_two_stages(self, parents):
        """Problem first, program second: two batched calls instead of one.

        Returns ``(tasks, outputs)`` in the shape the single-call path returns,
        so everything downstream -- retries, judging, scoring, reporting -- is
        untouched. A parent whose stage-1 reply does not parse yields an empty
        output, which ``_make_child_from_output`` already reports as a failed
        mutation.

        The labels come from stage 1 and are stapled onto the program here
        rather than asked of stage 2, because the ONE thing every variant of
        this prompt got wrong was the file's tail: whatever the parent's last
        two lines contained is what the child's contained, including nothing.
        """
        family_tasks = [
            build_family_task(
                parent,
                temperature=self.evolution_config.code_temperature,
                top_p=self.evolution_config.code_top_p,
            )
            for parent in parents
        ]
        family_replies = self.backend.mutate(family_tasks) if family_tasks else []

        plans = [parse_family_plan(reply) for reply in family_replies]
        self.stage_one_parsed = sum(1 for plan in plans if plan is not None)

        live = [(i, parents[i], plan) for i, plan in enumerate(plans) if plan]
        gen_tasks = [
            build_generator_task(
                parent, plan,
                temperature=self.evolution_config.generator_temperature,
                top_p=self.evolution_config.generator_top_p,
            )
            for _, parent, plan in live
        ]
        gen_replies = self.backend.mutate(gen_tasks) if gen_tasks else []

        tasks = list(family_tasks)
        outputs: list[str | None] = [None] * len(parents)
        for (i, _parent, plan), reply in zip(live, gen_replies):
            source = extract_generator_code(reply or "")
            if source is None:
                continue
            outputs[i] = set_label_declarations(source, plan["GROUP"], plan["SKILL"])
            # The task carried downstream is the one whose reply a self-fix
            # would have to repair: the generator call, not the family call.
            tasks[i] = gen_tasks[live.index((i, _parent, plan))]
        return tasks, outputs

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

        metadata = {"op": task.op}  # MUTATION_OP; the judge decides the labels
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
            )

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
                self.judge_skill_counts[name] = (
                    self.judge_skill_counts.get(name, 0) + 1
                )
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
        stat = self._seed_stat(program, instance, rollouts, seed=
                               int(getattr(instance, "seed", 0) or 0))
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
            clean_and_grade_solver_rollout(record, instance)[2]
            if instance is not None
            else bool(record.correct)
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
        iteration has no lagged score yet and does not train until the next one.
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
        """One fresh instance each, then extra instances to the best champions.

        The trainer needs ``train_batch_target`` prompts and the frontier is
        routinely smaller than that (median 18 against a target of 16-32 over
        the 8B run), so the shortfall is covered by drawing further FRESH seeds
        from the highest-R_Q champions rather than by shrinking the batch.
        Extra seeds are extra problems, not extra rollouts on the same problem
        -- LILO found scaling the number of levels more effective than scaling
        rollouts per level.

        Ranked by the score each champion already carries, i.e. the previous
        iteration's measurement. That is the same lag the batch selection uses:
        ranking on the rollouts about to be trained on would condition the
        sample on its own measurement noise.
        """
        n = len(champions)
        counts = [1] * n
        target = int(self.evolution_config.train_batch_target)
        if n == 0 or target <= n:
            return counts
        order = sorted(
            range(n),
            key=lambda i: float(getattr(champions[i], "rq_score", 0.0) or 0.0),
            reverse=True,
        )
        for j in range(target - n):
            counts[order[j % n]] += 1
        return counts

    def reevaluate_champions(self) -> None:
        """Refresh champion scores under the current backend.

        One vLLM wake/sleep for the whole champion set (was one per champion).
        """
        champions = list(self.archive.champions())
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
            self.lagged.record(
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
            self.archive.grid[target_cell].champion
            if target_cell is not None
            else None
        )
        inserted = self.archive.try_insert(
            program=program,
            u_value=u_value,
            rq_score=rq_score,
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

    def refresh_dataset(self, *, warmup: bool = False) -> None:
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
            for champion in self.archive.champions()
        ]
        if self.training_config.replay_training_batch:
            examples = build_replay_training_examples(
                self.archive.champions(),
                replay=self.replay,
                lagged=self.lagged,
                iteration=self.current_iteration,
                frontier_s_hat_range=self.evolution_config.frontier_s_hat_range,
                training_budget=self.training_config.training_budget,
                warmup=warmup,
            )
            if not examples and not warmup:
                # Nothing has a prior measurement to be selected on. That is the
                # same situation as bootstrap -- a cold resume, or an archive
                # that turned over completely -- and the honest response is to
                # fall back to the current scores rather than hand the trainer
                # an empty dataloader. The lag re-engages the moment any
                # champion carries history again.
                champions = list(self.archive.champions())
                if champions and all(
                    self.lagged.selection_score(c.program_id, self.current_iteration)
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
                        lagged=self.lagged,
                        iteration=self.current_iteration,
                        frontier_s_hat_range=(
                            self.evolution_config.frontier_s_hat_range
                        ),
                        training_budget=self.training_config.training_budget,
                        warmup=True,
                    )
            # The dataset is modulo-padded up to train_batch_size. Under replay
            # that repeats IDENTICAL responses rather than resampling them, so a
            # short batch double-counts the same rollouts. Harmless but worth
            # seeing: it shrinks as the archive fills.
            budget = self.training_config.training_budget
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
            self.archive.champions(),
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
                    "lagged_scores": self.lagged.to_dict(),
                    "rejected_children": self.rejected_children,
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
        judge_input_too_large, rollout_failed, s_hat_zero, rq_zero,
        inserted, rejected_non_elite (each with op, rq_score, s_hat,
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
        seeds_file = directory / self._USED_SEEDS_FILE
        if seeds_file.exists():
            payload = json.loads(seeds_file.read_text(encoding="utf-8"))
            self.used_seeds = {
                pid: set(seeds)
                for pid, seeds in payload.get("used_seeds", {}).items()
            }
            self.lagged = LaggedScoreboard.from_dict(
                payload.get("lagged_scores"),
                ewma_alpha=float(self.training_config.lagged_selection_ewma),
            )
            self.rejected_children = dict(payload.get("rejected_children") or {})
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
        self.refresh_dataset()
        self.events.append(
            {"event": "archive_restored", "champions": n_champions}
        )
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
