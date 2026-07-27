import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .archive import MAPElitesArchive
from .backends import EvolutionBackend, RolloutRecord
from .code_utils import (
    extract_generator_code,
    lint_generator_source,
    lint_metacognitive_generator_source,
    lint_problem_instance,
)
from .concepts import validate_concept_decl
from .config import EvolutionConfig, MetacognitionConfig, TrainingDataConfig
from .dataset import DynamicProblemDataset, build_training_examples
from .metacognition import (
    adaptive_depth_probability,
    collect_planning_evidence,
    compute_meta_progress,
    progress_context,
    select_reasoning_evidence,
    update_operator_ema,
)
from .openai_evaluator import (
    EvaluatorRuntimeError,
    OpenAIEvaluatorConfig,
    evaluate_messages_with_openai,
)
from .program import ProblemInstance, ProblemProgram
from .prompts import (
    MutationTask,
    build_evaluator_messages,
    build_evaluator_task,
    build_fix_task,
    build_metacognitive_plan_task,
    build_mutation_task,
    build_planned_mutation_task,
    parse_evaluator_verdict,
    parse_mutation_plan,
)
from .scoring import RQResult, compute_rq_full, is_frontier


@dataclass(slots=True)
class CandidateReport:
    status: str
    op: str
    child_id: str | None = None
    rq_score: float = 0.0
    p_hat: float = 0.0
    uncertainty: float = 0.0
    reason: str | None = None
    plan_id: str | None = None
    plan_status: str = "legacy"


@dataclass
class RQEvolver:
    """Owns one archive, one backend, and one dynamic training dataset."""

    archive: MAPElitesArchive
    backend: EvolutionBackend
    evolution_config: EvolutionConfig = field(default_factory=EvolutionConfig)
    metacognition_config: MetacognitionConfig = field(
        default_factory=MetacognitionConfig
    )
    training_config: TrainingDataConfig = field(default_factory=TrainingDataConfig)
    dataset: DynamicProblemDataset = field(default_factory=DynamicProblemDataset)
    used_seeds: dict[str, set[int]] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    # Candidate reports from the most recent run_outer_iteration, exposed so the
    # sampler can persist them to the per-step evolution log.
    last_reports: list[CandidateReport] = field(default_factory=list)
    # Number of refresh_dataset() calls so far. Only used to vary the seed of the
    # select_random_order ablation shuffle across outer iterations (reproducible).
    dataset_refresh_count: int = 0
    # Per-champion frontier decision from the most recent refresh_dataset():
    # {program_id, p_hat, rq_score, decision: in_frontier | p_hat_out_of_range}.
    # Persisted into evolution_log.jsonl so the learnability/curriculum filter
    # is observable (which champions fed the training set, and why not).
    last_frontier: list[dict] = field(default_factory=list)
    last_meta_progress: dict = field(default_factory=dict)
    operator_ema: dict[str, float] = field(default_factory=dict)
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
            result = self.evaluate_instance(program, inst)
            if result is None or result.rq_score <= 0.0:
                continue
            if self.archive.try_insert(
                program=program,
                h_value=result.uncertainty,
                problem_text=inst.problem,
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
        if self.metacognition_config.enabled and program.metadata.get("mutation_plan"):
            source_errors.extend(
                lint_metacognitive_generator_source(
                    program.source_code,
                    require_assert=self.metacognition_config.require_assert,
                    reject_trivial_assert=(
                        self.metacognition_config.reject_trivial_assert
                    ),
                    reject_unbounded_sampling=(
                        self.metacognition_config.reject_unbounded_sampling
                    ),
                )
            )
        if source_errors:
            return None, "; ".join(source_errors[:3])

        concept_type = program.declared_concept_type()
        concept_group = program.declared_concept_group()
        concept_errors = validate_concept_decl(concept_type, concept_group)
        if concept_errors:
            return None, "; ".join(concept_errors)
        program.metadata["concept_type"] = concept_type
        program.metadata["concept_group"] = concept_group

        first: ProblemInstance | None = None
        seen_pairs: set[tuple[str, str]] = set()
        for seed in range(n):
            inst = program.execute(seed=seed)
            if inst is None:
                return None, f"execute failed at seed={seed}"
            instance_errors = lint_problem_instance(inst)
            if instance_errors:
                return None, "; ".join(instance_errors[:3])
            if (
                self.metacognition_config.enabled
                and program.metadata.get("mutation_plan")
                and re.fullmatch(r"-?\d+", inst.answer.strip()) is None
            ):
                return None, (
                    "planned generator answer must be a base-10 integer string: "
                    f"{inst.answer!r}"
                )
            if not _answer_parseable(inst.answer):
                return None, f"answer is not parseable: {inst.answer!r}"
            first = first or inst
            seen_pairs.add((inst.problem.strip(), inst.answer.strip()))

        if n > 1 and len(seen_pairs) <= 1:
            return None, "program does not vary across seeds"
        return first, None

    def run_outer_iteration(self, outer_iteration: int) -> dict:
        self.current_iteration = int(outer_iteration)
        attempted = 0
        inserted = 0
        reports: list[CandidateReport] = []

        # Push current actor weights into vLLM ONCE for the whole evolve phase.
        # Evolve runs no optimizer step, so weights are static throughout:
        # reevaluate + every inner batch reuse the resident model (no per-session
        # re-push). The backend's begin_session no longer pushes weights when the
        # rollout is resident (sleep mode off).
        self.backend.sync_weights()
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
        result = {
            "outer_iteration": outer_iteration,
            "attempted": attempted,
            "inserted": inserted,
            "accept_rate": inserted / attempted if attempted else 0.0,
            "dataset_size": len(self.dataset),
            "frontier_in": frontier_in,
            "frontier_out": len(self.last_frontier) - frontier_in,
            **stats,
        }
        if self.last_meta_progress:
            result["meta_delta_p_global"] = float(
                (self.last_meta_progress.get("global_progress") or {}).get(
                    "delta_p", 0.0
                )
            )
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
        (_retry -> child | report) -> _apply_evaluator (drops incoherent children
        -> report) -> generate_rollouts on survivors -> each child scored and
        archived into a terminal CandidateReport. See docs/PIPELINE.md
        ("Evolution Candidate State") for the diagram and full status vocabulary.
        """
        requests: list[tuple[str, ProblemProgram]] = []
        for _ in range(batch_size):
            parent = self.archive.sample_parent()
            if parent is None:
                return [CandidateReport(status="no_parent", op="none")]

            op = self._sample_operator(parent)
            requests.append((op, parent))

        # One vLLM wake for the whole batch: the mutation generate and every
        # solver generate run while vLLM is awake; entropy (actor forward) is
        # deferred until after end_session (vLLM asleep) inside finalize_rollouts.
        self.backend.begin_session()
        try:
            prepared = self._prepare_mutation_tasks(requests)
            tasks = [item for item in prepared if isinstance(item, MutationTask)]
            outputs = self.backend.mutate(tasks)
            entries: list[dict] = []
            output_iter = iter(outputs)
            for item in prepared:
                if isinstance(item, CandidateReport):
                    entries.append({"report": item})
                    continue
                task = item
                output = next(output_iter, None)
                child, inst, reason, source = self._make_child_from_output(
                    task, output
                )
                if inst is not None:
                    entries.append({"task": task, "child": child, "inst": inst})
                elif source is not None:
                    # parses but failed verification -> eligible for one self-fix.
                    # Keep the RAW output: it becomes the assistant turn in the
                    # multi-turn fix prompt (so it is not re-quoted in the user turn).
                    entries.append(
                        {"_retry": {"task": task, "output": output, "reason": reason}}
                    )
                else:
                    status = "mutation_failed" if not output else "no_code"
                    entries.append(
                        {
                            "report": CandidateReport(
                                status=status,
                                op=task.op,
                                plan_id=task.plan_id,
                                plan_status=task.plan_status,
                            )
                        }
                    )

            # One-shot Reflexion self-fix: show the model its rejected program +
            # reason and re-verify. Runs inside the open vLLM session so the extra
            # generate reuses the already-awake rollout worker.
            self._resolve_retries(entries)

            # Final coherence gate: drop verified children whose seed-0 problem
            # the evaluator marks INVALID, BEFORE solver rollouts are spent on
            # them. Runs in the same open vLLM session as mutate.
            self._apply_evaluator(entries)

            to_eval = [e for e in entries if "child" in e]
            pending = self.backend.generate_rollouts(
                [e["inst"] for e in to_eval],
                n_rollouts=self.evolution_config.num_rollouts,
            )
        finally:
            self.backend.end_session()

        grouped = self.backend.finalize_rollouts(pending)
        rollouts_by_child = {
            id(e["child"]): rollouts for e, rollouts in zip(to_eval, grouped)
        }

        reports: list[CandidateReport] = []
        for entry in entries:
            if "report" in entry:
                reports.append(entry["report"])
                continue
            task, child, inst = entry["task"], entry["child"], entry["inst"]
            rollouts = rollouts_by_child.get(id(child), [])
            accepted = [
                r for r in rollouts if getattr(r, "status", "accepted") == "accepted"
            ]
            if rollouts and not accepted:
                # Every rollout for this child was rejected (timeout / worker
                # error / stale / ...): report the dominant reason explicitly
                # instead of letting it masquerade as an ordinary p_hat_zero.
                reasons: dict[str, int] = {}
                for r in rollouts:
                    key = r.reject_reason or "unknown"
                    reasons[key] = reasons.get(key, 0) + 1
                reports.append(
                    CandidateReport(
                        status="rollout_failed",
                        op=task.op,
                        child_id=child.program_id,
                        reason=max(reasons, key=reasons.get),
                        plan_id=task.plan_id,
                        plan_status=task.plan_status,
                    )
                )
                continue
            result = self._score_from_rollouts(child, rollouts, instance=inst)
            if result.rq_score <= 0.0:
                reports.append(
                    CandidateReport(
                        status=(
                            "p_hat_zero"
                            if result.p_hat <= 0.0
                            else "rq_zero"
                        ),
                        op=task.op,
                        child_id=child.program_id,
                        rq_score=result.rq_score,
                        p_hat=result.p_hat,
                        uncertainty=result.uncertainty,
                        plan_id=task.plan_id,
                        plan_status=task.plan_status,
                    )
                )
                continue

            inserted = self._try_insert_with_telemetry(
                program=child,
                h_value=result.uncertainty,
                problem_text=inst.problem,
                rq_score=result.rq_score,
            )
            reports.append(
                CandidateReport(
                    status="inserted" if inserted else "rejected_non_elite",
                    op=task.op,
                    child_id=child.program_id,
                    rq_score=result.rq_score,
                    p_hat=result.p_hat,
                    uncertainty=result.uncertainty,
                    plan_id=task.plan_id,
                    plan_status=task.plan_status,
                )
            )
        return reports

    def _prepare_mutation_tasks(
        self,
        requests: list[tuple[str, ProblemProgram]],
    ) -> list[MutationTask | CandidateReport]:
        """Build legacy tasks or run one batched monitoring/planning pass.

        Planning reuses reasoning evidence collected by the existing R_Q
        rollouts. It adds no Solver rollout; only the mutation model performs one
        additional generation before code generation.
        """
        if not self.metacognition_config.enabled:
            return [build_mutation_task(op, parent) for op, parent in requests]

        tokenizer = getattr(self.backend, "tokenizer", None)
        champions = list(self.archive.champions())
        plan_inputs: list[tuple[str, ProblemProgram, list[dict]]] = []
        prepared: list[MutationTask | CandidateReport | None] = []

        for op, parent in requests:
            evidence = collect_planning_evidence(
                parent,
                op,
                champions,
                total_tokens=(
                    self.metacognition_config.monitoring_total_trace_tokens
                ),
                tokenizer=tokenizer,
            )
            if self.metacognition_config.require_contrast_pair and len(evidence) < 2:
                if self.metacognition_config.fallback_to_legacy_mutation:
                    task = build_mutation_task(op, parent)
                    task.plan_status = "fallback_no_evidence"
                    prepared.append(task)
                else:
                    prepared.append(
                        CandidateReport(
                            status="mutation_failed",
                            op=op,
                            reason="metacognitive planning requires two evidence traces",
                            plan_status="missing_evidence",
                        )
                    )
                continue
            plan_inputs.append((op, parent, evidence))
            prepared.append(None)

        if not plan_inputs:
            return [item for item in prepared if item is not None]

        plan_tasks = [
            build_metacognitive_plan_task(
                op,
                parent,
                evidence=evidence,
                meta_progress=progress_context(self.last_meta_progress, parent),
                max_output_tokens=(
                    self.metacognition_config.plan_max_output_tokens
                ),
            )
            for op, parent, evidence in plan_inputs
        ]
        plan_outputs = self.backend.mutate(plan_tasks)
        planned_iter = iter(
            (
                plan_input,
                plan_outputs[index] if index < len(plan_outputs) else None,
            )
            for index, plan_input in enumerate(plan_inputs)
        )

        finalized: list[MutationTask | CandidateReport] = []
        for item in prepared:
            if item is not None:
                finalized.append(item)
                continue
            (op, parent, _), output = next(planned_iter)
            inherited_move = ""
            if op == "in_breadth":
                inherited_move = str(
                    (
                        (parent.metadata or {}).get("mutation_plan") or {}
                    ).get("target_reasoning_move", "")
                )
            plan, reason = parse_mutation_plan(
                output,
                op,
                required_target_reasoning_move=inherited_move,
            )
            if plan is not None:
                task = build_planned_mutation_task(op, parent, plan)
                finalized.append(task)
                self.events.append(
                    {
                        "event": "mutation_plan_created",
                        "program_id": parent.program_id,
                        "op": op,
                        "plan_id": task.plan_id,
                    }
                )
                continue
            self.events.append(
                {
                    "event": "mutation_plan_failed",
                    "program_id": parent.program_id,
                    "op": op,
                    "reason": reason,
                }
            )
            if (
                self.metacognition_config.fallback_to_legacy_mutation
                and not inherited_move
            ):
                task = build_mutation_task(op, parent)
                task.plan_status = "fallback_invalid_plan"
                finalized.append(task)
            else:
                finalized.append(
                    CandidateReport(
                        status="mutation_failed",
                        op=op,
                        reason=f"metacognitive plan rejected: {reason}",
                        plan_status="invalid_plan",
                    )
                )
        return finalized

    def _make_child_from_output(
        self,
        task: MutationTask,
        output: str | None,
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
        child = ProblemProgram(
            source_code=source,
            parent_id=task.parent.program_id,
            generation=task.parent.generation + 1,
            metadata={
                "op": task.op,
                "plan_status": task.plan_status,
                **(
                    {
                        "mutation_plan": task.mutation_plan,
                        "plan_id": task.plan_id,
                    }
                    if task.mutation_plan is not None
                    else {}
                ),
            },
        )
        inst, reason = self.verify_program(child)
        if inst is not None:
            reason = self._validate_mutation_contract(task, child)
            if reason is not None:
                inst = None
        return child, inst, reason, source

    @staticmethod
    def _validate_mutation_contract(
        task: MutationTask,
        child: ProblemProgram,
    ) -> str | None:
        """Enforce the structural part of a valid metacognitive plan."""
        if task.mutation_plan is None:
            return None
        if task.op == "in_depth":
            if child.get_concept_group() != task.parent.get_concept_group():
                return (
                    "planned in-depth mutation changed CONCEPT_GROUP: "
                    f"{task.parent.get_concept_group()!r} -> "
                    f"{child.get_concept_group()!r}"
                )
            if child.get_concept_type() != task.parent.get_concept_type():
                return (
                    "planned in-depth mutation changed CONCEPT_TYPE: "
                    f"{task.parent.get_concept_type()!r} -> "
                    f"{child.get_concept_type()!r}"
                )
        elif (
            task.op == "in_breadth"
            and child.get_concept_group() == task.parent.get_concept_group()
        ):
            return (
                "planned in-breadth mutation must change CONCEPT_GROUP from "
                f"{task.parent.get_concept_group()!r}"
            )
        return None

    def _resolve_retries(self, entries: list[dict]) -> None:
        """Finalize every ``_retry`` entry into a success or a report entry.

        With ``fix_retry`` enabled, each verify-failed child gets ONE self-fix
        round (one batched ``mutate`` over all retryable entries); survivors
        become ``{"task","child","inst"}`` and are tagged ``fixed_after_retry``.
        With it disabled, the originals collapse straight to a verify_failed
        report so no entry is ever left dangling.
        """
        targets = [e for e in entries if "_retry" in e]
        if not targets:
            return
        enabled = self.evolution_config.fix_retry
        if enabled:
            fix_tasks = [
                build_fix_task(
                    e["_retry"]["task"], e["_retry"]["output"], e["_retry"]["reason"]
                )
                for e in targets
            ]
            outputs = self.backend.mutate(fix_tasks)
        else:
            outputs = [None] * len(targets)

        for e, output in zip(targets, outputs):
            info = e.pop("_retry")
            task = info["task"]
            if not enabled:
                e["report"] = CandidateReport(
                    status="verify_failed",
                    op=task.op,
                    reason=info["reason"],
                    plan_id=task.plan_id,
                    plan_status=task.plan_status,
                )
                continue
            child, inst, reason, _ = self._make_child_from_output(task, output)
            if inst is not None:
                child.metadata["fixed_after_retry"] = True
                e["task"], e["child"], e["inst"] = task, child, inst
            else:
                e["report"] = CandidateReport(
                    status="verify_failed",
                    op=task.op,
                    child_id=child.program_id if child else "",
                    reason=f"[after fix] {reason}",
                    plan_id=task.plan_id,
                    plan_status=task.plan_status,
                )

    def _apply_evaluator(self, entries: list[dict]) -> None:
        """LLM coherence gate over every verified child's seed-0 problem.

        For each ``{"task","child","inst"}`` entry the evaluator judges whether
        the seed-0 problem statement is internally coherent (using
        ``EVALUATOR_SYSTEM_PROMPT`` + the ``evaluator.txt`` shots). A child marked
        INVALID is converted in place to an ``evaluator_rejected`` report, so it
        is discarded before solver rollouts and never reaches the archive.

        With ``evaluator_provider=policy``, one batched ``mutate`` over all
        targets runs inside the already-open vLLM session. With
        ``evaluator_provider=openai``, the same evaluator messages are sent to
        the OpenAI Responses API using ``evaluator_model``. Evaluator
        configuration or runtime failures raise immediately; they are never
        converted into candidate reports because continuing would invalidate
        the resulting curriculum.
        """
        if not self.evolution_config.use_evaluator:
            return
        targets = [e for e in entries if "child" in e]
        if not targets:
            return
        if self.evolution_config.evaluator_provider == "openai":
            outputs = self._run_openai_evaluator(targets)
        else:
            eval_tasks = [
                build_evaluator_task(
                    e["child"],
                    e["inst"].problem,
                    mutation_plan=getattr(
                        e["task"],
                        "mutation_plan",
                        None,
                    ),
                )
                for e in targets
            ]
            try:
                outputs = self.backend.mutate(eval_tasks)
            except Exception as exc:
                raise EvaluatorRuntimeError(
                    "Evaluator backend failed; aborting R_Q-Evolve instead of "
                    f"continuing with an invalid curriculum: {exc}"
                ) from exc
        for e, output in zip(targets, outputs):
            if isinstance(output, Exception):
                raise EvaluatorRuntimeError(
                    "Evaluator call failed; aborting R_Q-Evolve instead of "
                    f"discarding the candidate and continuing: {output}"
                ) from output
            is_valid, reason = parse_evaluator_verdict(output or "")
            if is_valid:
                continue
            task, child = e["task"], e["child"]
            e.clear()
            e["report"] = CandidateReport(
                status="evaluator_rejected",
                op=task.op,
                child_id=child.program_id,
                reason=reason,
                plan_id=getattr(task, "plan_id", None),
                plan_status=getattr(task, "plan_status", "legacy"),
            )

    def _run_openai_evaluator(self, targets: list[dict]) -> list[str | Exception]:
        cfg = OpenAIEvaluatorConfig(
            model=self.evolution_config.evaluator_model,
            reasoning_effort=self.evolution_config.evaluator_reasoning_effort,
            timeout_s=self.evolution_config.evaluator_timeout_s,
            max_output_tokens=self.evolution_config.evaluator_max_output_tokens,
        )

        def evaluate_one(e: dict) -> str | Exception:
            messages = build_evaluator_messages(
                e["inst"].problem,
                mutation_plan=getattr(
                    e["task"],
                    "mutation_plan",
                    None,
                ),
            )
            try:
                return evaluate_messages_with_openai(messages, cfg)
            except Exception as exc:
                return exc

        max_workers = min(self.evolution_config.evaluator_concurrency, len(targets))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(evaluate_one, targets))

    def _score_from_rollouts(
        self,
        program: ProblemProgram,
        rollouts: list[RolloutRecord],
        *,
        instance: ProblemInstance | None = None,
    ) -> RQResult:
        # Streamed rollouts carry a status: rejected samples (timeout / stale /
        # overlong / ...) leave the p_hat estimate entirely -- they are logged
        # with a reason but never scored. Legacy records default to "accepted".
        accepted = [
            r for r in rollouts if getattr(r, "status", "accepted") == "accepted"
        ]
        flags = [r.correct for r in accepted]
        uncertainty = (
            sum(r.entropy for r in accepted) / len(accepted)
            if accepted
            else 0.0
        )
        result = compute_rq_full(flags, uncertainty)
        program.p_hat = result.p_hat
        program.h_score = result.uncertainty
        program.rq_score = result.rq_score
        program.fitness = result.rq_score
        if self.metacognition_config.enabled and instance is not None:
            evidence = select_reasoning_evidence(
                rollouts,
                program=program,
                instance=instance,
                iteration=self.current_iteration,
                max_tokens=self.metacognition_config.trace_storage_max_tokens,
                tokenizer=getattr(self.backend, "tokenizer", None),
            )
            if evidence:
                # This is telemetry attached to the single live archive entry,
                # not a second archive and not an additional Solver rollout.
                program.metadata["reasoning_evidence"] = [
                    asdict(item) for item in evidence
                ]
        return result

    def evaluate_instances(
        self,
        programs: list[ProblemProgram],
        instances: list[ProblemInstance],
    ) -> list[RQResult | None]:
        """Score a batch of (program, instance) pairs with ONE vLLM wake/sleep.

        All solver rollouts are generated while vLLM is awake; vLLM is slept
        once, then entropies (actor forward) are computed against the freed
        memory. Replaces N per-instance wake_up cycles with a single one.

        A group whose rollouts were ALL rejected (streaming timeout / worker
        error / staleness) yields ``None`` instead of an RQResult: scoring it
        would zero the program's p_hat/h_score IN PLACE and let a single
        transient infra failure evict a champion from its true niche (the
        stale-p_hat archive-pollution failure mode). Callers must skip None.
        """
        if not instances:
            return []
        self.backend.begin_session()
        try:
            pending = self.backend.generate_rollouts(
                instances, n_rollouts=self.evolution_config.num_rollouts
            )
        finally:
            self.backend.end_session()
        grouped = self.backend.finalize_rollouts(pending)
        results: list[RQResult | None] = []
        for program, instance, rollouts in zip(programs, instances, grouped):
            accepted = [
                r for r in rollouts if getattr(r, "status", "accepted") == "accepted"
            ]
            if rollouts and not accepted:
                reasons = sorted({r.reject_reason or "unknown" for r in rollouts})
                self.events.append(
                    {
                        "event": "eval_rollout_failed",
                        "program_id": program.program_id,
                        "reasons": reasons,
                    }
                )
                print(
                    f"[RQ-Evolve] eval skipped for {program.program_id}: all "
                    f"{len(rollouts)} rollouts rejected ({reasons}); keeping "
                    f"previous scores"
                )
                results.append(None)
                continue
            results.append(
                self._score_from_rollouts(
                    program,
                    rollouts,
                    instance=instance,
                )
            )
        return results

    def evaluate_instance(
        self,
        program: ProblemProgram,
        instance: ProblemInstance,
    ) -> RQResult | None:
        return self.evaluate_instances([program], [instance])[0]

    def reevaluate_champions(self) -> None:
        """Refresh champion scores under the current backend.

        One vLLM wake/sleep for the whole champion set (was one per champion).
        """
        pairs: list[tuple[ProblemProgram, ProblemInstance]] = []
        for champion in list(self.archive.champions()):
            inst = champion.execute(seed=0)
            if inst is None:
                continue
            pairs.append((champion, inst))
        if not pairs:
            return
        pre_scores = {
            champion.program_id: {
                "p_hat": float(champion.p_hat),
                "concept_group": champion.get_concept_group(),
                "concept_type": champion.get_concept_type(),
                "operator": (champion.metadata or {}).get("op", "seed"),
            }
            for champion, _instance in pairs
        }
        results = self.evaluate_instances(
            [p for p, _ in pairs], [i for _, i in pairs]
        )

        if self.metacognition_config.enabled:
            progress = compute_meta_progress(
                pre_scores,
                [program for program, _instance in pairs],
                results,
                iteration=self.current_iteration,
            )
            self.last_meta_progress = progress.to_dict()
            if self.metacognition_config.adaptive_operator:
                self.operator_ema = update_operator_ema(
                    self.operator_ema,
                    self.last_meta_progress,
                    alpha=self.metacognition_config.operator_ema_alpha,
                )
            self.events.append(
                {
                    "event": "meta_progress_measured",
                    **self.last_meta_progress,
                    "operator_ema": dict(self.operator_ema),
                }
            )

        for (champion, inst), result in zip(pairs, results):
            if result is None:
                # all rollouts rejected (transient timeout/worker error) --
                # keep the champion's previous scores and niche untouched.
                continue
            # Delta-p above was measured on the fixed pre-update cohort. Only
            # now may a champion move to a new H cell, replace another champion,
            # or disappear when R_Q has reached zero.
            self.archive.remove_program(champion.program_id)
            if result.rq_score <= 0.0:
                self.events.append(
                    {
                        "event": "champion_removed_after_reevaluation",
                        "program_id": champion.program_id,
                        "reason": "rq_zero",
                        "p_hat": result.p_hat,
                    }
                )
                continue
            inserted = self._try_insert_with_telemetry(
                program=champion,
                h_value=result.uncertainty,
                problem_text=inst.problem,
                rq_score=result.rq_score,
                source="champion_reevaluation",
            )
            if not inserted:
                self.events.append(
                    {
                        "event": "champion_removed_after_reevaluation",
                        "program_id": champion.program_id,
                        "reason": "lost_target_bin_competition",
                        "p_hat": result.p_hat,
                        "rq_score": result.rq_score,
                    }
                )

    def _try_insert_with_telemetry(
        self,
        *,
        program: ProblemProgram,
        h_value: float,
        problem_text: str,
        rq_score: float,
        source: str = "mutation",
    ) -> bool:
        target_cell = self.archive.target_cell(
            program,
            h_value=h_value,
            problem_text=problem_text,
        )
        incumbent = self.archive.grid[target_cell].champion
        inserted = self.archive.try_insert(
            program=program,
            h_value=h_value,
            problem_text=problem_text,
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
                    "incoming_program_id": program.program_id,
                    "incoming_rq": float(rq_score),
                    "evicted_program_id": incumbent.program_id,
                    "evicted_rq": float(incumbent.rq_score),
                }
            )
        return inserted

    def refresh_dataset(self) -> None:
        # Record each champion's frontier decision (the learnability filter that
        # decides which problems feed training) with the SAME predicate
        # build_training_examples uses -- observability only, no behavior change.
        low, high = self.evolution_config.frontier_p_hat_range
        self.last_frontier = [
            {
                "program_id": champion.program_id,
                "p_hat": round(float(champion.p_hat), 4),
                "rq_score": round(float(champion.rq_score), 6),
                "decision": (
                    "in_frontier"
                    if is_frontier(champion.p_hat, low, high)
                    else "p_hat_out_of_range"
                ),
            }
            for champion in self.archive.champions()
        ]
        examples = build_training_examples(
            self.archive.champions(),
            instances_per_program=self.training_config.instances_per_program,
            training_budget=self.training_config.training_budget,
            frontier_p_hat_range=self.evolution_config.frontier_p_hat_range,
            n_h_bins=self.archive.n_h_bins,
            n_div_bins=self.archive.n_div_bins,
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
    _METACOGNITION_FILE = "rq_metacognition.json"

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
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (directory / self._METACOGNITION_FILE).write_text(
            json.dumps(
                {
                    "last_meta_progress": self.last_meta_progress,
                    "operator_ema": self.operator_ema,
                    "current_iteration": self.current_iteration,
                },
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
        mutation_failed, no_code, verify_failed, evaluator_rejected,
        rollout_failed, p_hat_zero, rq_zero, inserted,
        rejected_non_elite (each with op, rq_score, p_hat, uncertainty; see
        docs/PIPELINE.md "Evolution Candidate State"). ``reports`` defaults to
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
            "meta_progress": self.last_meta_progress,
            "operator_ema": self.operator_ema,
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
        seeds_file = directory / self._USED_SEEDS_FILE
        if seeds_file.exists():
            payload = json.loads(seeds_file.read_text(encoding="utf-8"))
            self.used_seeds = {
                pid: set(seeds)
                for pid, seeds in payload.get("used_seeds", {}).items()
            }
        metacognition_file = directory / self._METACOGNITION_FILE
        if metacognition_file.exists():
            payload = json.loads(metacognition_file.read_text(encoding="utf-8"))
            self.last_meta_progress = dict(
                payload.get("last_meta_progress") or {}
            )
            self.operator_ema = {
                str(key): float(value)
                for key, value in (payload.get("operator_ema") or {}).items()
            }
            self.current_iteration = int(
                payload.get("current_iteration", self.current_iteration)
            )
        self.refresh_dataset()
        self.events.append(
            {"event": "archive_restored", "champions": n_champions}
        )
        return True

    def _sample_operator(self, parent: ProblemProgram | None = None) -> str:
        del parent  # reserved for future per-concept controllers
        depth_probability = self.evolution_config.in_depth_ratio
        if (
            self.metacognition_config.enabled
            and self.metacognition_config.adaptive_operator
        ):
            depth_probability = adaptive_depth_probability(
                self.operator_ema,
                fallback=depth_probability,
                min_probability=(
                    self.metacognition_config.operator_min_probability
                ),
            )
        roll = random.random()
        if roll < depth_probability:
            return "in_depth"
        return "in_breadth"


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
