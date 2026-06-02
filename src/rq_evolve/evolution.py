import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .archive import MAPElitesArchive
from .backends import EvolutionBackend, RolloutRecord
from .code_utils import (
    extract_generator_code,
    lint_generator_source,
    lint_problem_instance,
)
from .concepts import validate_concept_decl
from .config import EvolutionConfig, TrainingDataConfig
from .dataset import DynamicProblemDataset, build_training_examples
from .program import ProblemInstance, ProblemProgram
from .prompts import build_fix_task, build_mutation_task
from .scoring import RQResult, compute_rq_full


@dataclass(slots=True)
class CandidateReport:
    status: str
    op: str
    child_id: str | None = None
    rq_score: float = 0.0
    p_hat: float = 0.0
    uncertainty: float = 0.0
    reason: str | None = None


@dataclass
class RQEvolver:
    """Owns one archive, one backend, and one dynamic training dataset."""

    archive: MAPElitesArchive
    backend: EvolutionBackend
    evolution_config: EvolutionConfig = field(default_factory=EvolutionConfig)
    training_config: TrainingDataConfig = field(default_factory=TrainingDataConfig)
    dataset: DynamicProblemDataset = field(default_factory=DynamicProblemDataset)
    used_seeds: dict[str, set[int]] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    # Candidate reports from the most recent run_outer_iteration, exposed so the
    # sampler can persist them to the per-step evolution log.
    last_reports: list[CandidateReport] = field(default_factory=list)

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
            if result.p_hat <= 0.0:
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
            if not _answer_parseable(inst.answer):
                return None, f"answer is not parseable: {inst.answer!r}"
            first = first or inst
            seen_pairs.add((inst.problem.strip(), inst.answer.strip()))

        if n > 1 and len(seen_pairs) <= 1:
            return None, "program does not vary across seeds"
        return first, None

    def run_outer_iteration(self, outer_iteration: int) -> dict:
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
        result = {
            "outer_iteration": outer_iteration,
            "attempted": attempted,
            "inserted": inserted,
            "accept_rate": inserted / attempted if attempted else 0.0,
            "dataset_size": len(self.dataset),
            **stats,
        }
        self.last_reports = reports
        self.events.append({"event": "outer_iteration_done", **result})
        return result

    def inner_iteration_batch(self, batch_size: int) -> list[CandidateReport]:
        tasks = []
        for _ in range(batch_size):
            parent = self.archive.sample_parent()
            if parent is None:
                return [CandidateReport(status="no_parent", op="none")]

            op = self._sample_operator()
            parent_b = None
            if op == "crossover":
                parent, parent_b = self.archive.sample_two_parents()
                if parent is None or parent_b is None:
                    op = "in_depth"
                    parent = self.archive.sample_parent()
                if parent is None:
                    return [CandidateReport(status="no_parent", op="none")]
            tasks.append(build_mutation_task(op, parent, parent_b))

        # One vLLM wake for the whole batch: the mutation generate and every
        # solver generate run while vLLM is awake; entropy (actor forward) is
        # deferred until after end_session (vLLM asleep) inside finalize_rollouts.
        self.backend.begin_session()
        try:
            outputs = self.backend.mutate(tasks)
            entries: list[dict] = []
            for task, output in zip(tasks, outputs):
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
                    entries.append({"report": CandidateReport(status=status, op=task.op)})

            # One-shot Reflexion self-fix: show the model its rejected program +
            # reason and re-verify. Runs inside the open vLLM session so the extra
            # generate reuses the already-awake rollout worker.
            self._resolve_retries(entries)

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
            result = self._score_from_rollouts(
                child, rollouts_by_child.get(id(child), [])
            )
            if result.p_hat <= 0.0:
                reports.append(
                    CandidateReport(
                        status="p_hat_zero",
                        op=task.op,
                        child_id=child.program_id,
                        rq_score=result.rq_score,
                        p_hat=result.p_hat,
                        uncertainty=result.uncertainty,
                    )
                )
                continue

            inserted = self.archive.try_insert(
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
                )
            )
        return reports

    def _make_child_from_output(self, task, output):
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
            parent_id=_parent_id(task),
            generation=max(
                task.parent.generation,
                task.parent_b.generation if task.parent_b else 0,
            )
            + 1,
            metadata={"op": task.op},
        )
        inst, reason = self.verify_program(child)
        return child, inst, reason, source

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
                    status="verify_failed", op=task.op, reason=info["reason"]
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
                )

    def _score_from_rollouts(
        self,
        program: ProblemProgram,
        rollouts: list[RolloutRecord],
    ) -> RQResult:
        flags = [r.correct for r in rollouts]
        uncertainty = (
            sum(r.entropy for r in rollouts) / len(rollouts)
            if rollouts
            else 0.0
        )
        result = compute_rq_full(flags, uncertainty)
        program.p_hat = result.p_hat
        program.h_score = result.uncertainty
        program.rq_score = result.rq_score
        program.fitness = result.rq_score
        return result

    def evaluate_instances(
        self,
        programs: list[ProblemProgram],
        instances: list[ProblemInstance],
    ) -> list[RQResult]:
        """Score a batch of (program, instance) pairs with ONE vLLM wake/sleep.

        All solver rollouts are generated while vLLM is awake; vLLM is slept
        once, then entropies (actor forward) are computed against the freed
        memory. Replaces N per-instance wake_up cycles with a single one.
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
        return [
            self._score_from_rollouts(program, rollouts)
            for program, rollouts in zip(programs, grouped)
        ]

    def evaluate_instance(
        self,
        program: ProblemProgram,
        instance: ProblemInstance,
    ) -> RQResult:
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
        results = self.evaluate_instances(
            [p for p, _ in pairs], [i for _, i in pairs]
        )
        for (champion, inst), result in zip(pairs, results):
            self.archive.try_insert(
                program=champion,
                h_value=result.uncertainty,
                problem_text=inst.problem,
                rq_score=result.rq_score,
            )

    def refresh_dataset(self) -> None:
        examples = build_training_examples(
            self.archive.champions(),
            instances_per_program=self.training_config.instances_per_program,
            training_budget=self.training_config.training_budget,
            frontier_p_hat_range=self.evolution_config.frontier_p_hat_range,
            n_h_bins=self.archive.n_h_bins,
            n_div_bins=self.archive.n_div_bins,
            used_seeds=self.used_seeds,
            strict_anti_reuse=self.training_config.strict_anti_reuse,
        )
        self.dataset.update(examples)

    _USED_SEEDS_FILE = "rq_used_seeds.json"

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
        candidate report (status = inserted / rejected_non_elite / verify_failed
        / p_hat_zero / mutation_failed / no_code, with op, rq_score, p_hat,
        uncertainty). ``reports`` defaults to ``self.last_reports``.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        reports = self.last_reports if reports is None else reports
        record = {
            "iteration": int(iteration),
            "metrics": metrics,
            "reports": [asdict(r) for r in reports],
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
        self.refresh_dataset()
        self.events.append(
            {"event": "archive_restored", "champions": n_champions}
        )
        return True

    def _sample_operator(self) -> str:
        roll = random.random()
        if roll < self.evolution_config.crossover_ratio:
            return "crossover"
        if roll < self.evolution_config.crossover_ratio + self.evolution_config.in_depth_ratio:
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


def _parent_id(task) -> str:
    if task.parent_b is not None:
        return f"{task.parent.program_id}x{task.parent_b.program_id}"
    return task.parent.program_id

