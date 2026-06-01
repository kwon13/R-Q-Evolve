"""Framework-free R_Q-Evolve orchestration loop."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
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
from .prompts import build_mutation_task
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

        outputs = self.backend.mutate(tasks)
        reports: list[CandidateReport] = []
        for task, output in zip(tasks, outputs):
            if not output:
                reports.append(CandidateReport(status="mutation_failed", op=task.op))
                continue
            source = extract_generator_code(output)
            if source is None:
                reports.append(CandidateReport(status="no_code", op=task.op))
                continue

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
            if inst is None:
                reports.append(
                    CandidateReport(
                        status="verify_failed",
                        op=task.op,
                        child_id=child.program_id,
                        reason=reason,
                    )
                )
                continue

            result = self.evaluate_instance(child, inst)
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

    def evaluate_instance(
        self,
        program: ProblemProgram,
        instance: ProblemInstance,
    ) -> RQResult:
        groups = self.backend.rollout(
            [instance],
            n_rollouts=self.evolution_config.num_rollouts,
        )
        rollouts: list[RolloutRecord] = groups[0] if groups else []
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

    def reevaluate_champions(self) -> None:
        """Refresh champion scores under the current backend."""
        for champion in list(self.archive.champions()):
            inst = champion.execute(seed=0)
            if inst is None:
                continue
            result = self.evaluate_instance(champion, inst)
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

    def save_state(self, directory: str | Path) -> None:
        """Persist the MAP-Elites archive + used_seeds for restart.

        The verl weight checkpoint does NOT include the archive, so without this
        a resumed run restarts from a seed-only grid and loses every evolved
        champion. Called once per outer iteration (after evolution) so the
        latest archive is always on disk.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.archive.save(directory)
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

