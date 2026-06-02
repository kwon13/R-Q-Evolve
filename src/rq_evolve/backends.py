"""Backend interfaces for mutation and solver evaluation."""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Protocol

from .program import ProblemInstance
from .prompts import MutationTask


@dataclass(slots=True)
class RolloutRecord:
    response: str
    predicted_answer: str | None
    correct: bool
    entropy: float


@dataclass
class PendingRollouts:
    """Carrier between the generate (vLLM awake) and entropy (vLLM asleep) phases.

    ``generate_rollouts`` returns one of these without computing entropy so the
    expensive actor forward can be deferred until after vLLM has been slept,
    keeping the whole phase to a single cumem wake_up. Mock backends fill
    ``grouped`` eagerly; the verl backend stashes ``full_batch`` + ``decoded``.
    """

    instances: list[ProblemInstance]
    n_rollouts: int
    grouped: list[list[RolloutRecord]] | None = None
    full_batch: object | None = None
    decoded: list[str] = field(default_factory=list)


class EvolutionBackend(Protocol):
    """Everything that depends on an LLM or inference engine."""

    def mutate(self, tasks: list[MutationTask]) -> list[str | None]:
        """Return generated Python source, one per task."""

    def rollout(
        self,
        instances: list[ProblemInstance],
        n_rollouts: int,
    ) -> list[list[RolloutRecord]]:
        """Return G solver rollouts for each problem instance."""

    def sync_weights(self) -> None:
        """Push current policy weights into the inference engine once per phase."""

    def begin_session(self) -> None:
        """Wake the inference engine once for a batch of generate calls."""

    def end_session(self) -> None:
        """Sleep the inference engine once at the end of the phase."""

    def generate_rollouts(
        self,
        instances: list[ProblemInstance],
        n_rollouts: int,
    ) -> PendingRollouts:
        """Generate solver rollouts (no entropy) inside an open session."""

    def finalize_rollouts(self, pending: PendingRollouts) -> list[list[RolloutRecord]]:
        """Assemble grouped records, computing entropy after the engine slept."""


class MockEvolutionBackend:
    """Deterministic, no-model backend for pipeline development.

    This lets you implement archive logic, verification, scoring, and dataset
    refresh before wiring vLLM / OpenAI / verl into the system.
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.counter = 0

    def mutate(self, tasks: list[MutationTask]) -> list[str | None]:
        outputs: list[str | None] = []
        for task in tasks:
            self.counter += 1
            if task.op == "in_breadth":
                outputs.append(_geometry_template(self.counter))
            elif task.op == "crossover":
                outputs.append(_modular_template(self.counter))
            else:
                outputs.append(_quadratic_template(self.counter))
        return outputs

    def rollout(
        self,
        instances: list[ProblemInstance],
        n_rollouts: int,
    ) -> list[list[RolloutRecord]]:
        results: list[list[RolloutRecord]] = []
        for inst in instances:
            local = random.Random(_stable_int(inst.problem + inst.answer))
            difficulty = 0.25 + min(len(inst.problem), 400) / 800.0
            pass_prob = max(0.1, min(0.9, 1.0 - difficulty))
            rows: list[RolloutRecord] = []
            for _ in range(n_rollouts):
                correct = local.random() < pass_prob
                pred = inst.answer if correct else _nearby_wrong_answer(inst.answer, local)
                entropy = 0.5 + difficulty * 3.0 + local.random()
                rows.append(
                    RolloutRecord(
                        response=f"Reasoning omitted in mock. \\boxed{{{pred}}}",
                        predicted_answer=pred,
                        correct=correct,
                        entropy=entropy,
                    )
                )
            results.append(rows)
        return results

    def sync_weights(self) -> None:
        pass

    def begin_session(self) -> None:
        pass

    def end_session(self) -> None:
        pass

    def generate_rollouts(
        self,
        instances: list[ProblemInstance],
        n_rollouts: int,
    ) -> PendingRollouts:
        return PendingRollouts(
            instances=list(instances),
            n_rollouts=int(n_rollouts),
            grouped=self.rollout(instances, n_rollouts),
        )

    def finalize_rollouts(self, pending: PendingRollouts) -> list[list[RolloutRecord]]:
        if pending.grouped is not None:
            return pending.grouped
        return [[] for _ in pending.instances]


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL)


def extract_boxed(text: str) -> str | None:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def _nearby_wrong_answer(answer: str, rng: random.Random) -> str:
    try:
        return str(int(answer) + rng.choice([-2, -1, 1, 2]))
    except ValueError:
        return "0" if answer.strip() != "0" else "1"


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _quadratic_template(index: int) -> str:
    return f'''import random


def generate(seed):
    rng = random.Random(seed + {index})
    a = rng.randint(2, 9)
    b = rng.randint(2, 9)
    s = a + b
    p = a * b
    problem = (
        f"The roots of x^2 - {{s}}x + {{p}} = 0 are r and t. "
        f"Find r^2 + t^2."
    )
    answer = s * s - 2 * p
    return problem, str(answer)


CONCEPT_GROUP = "algebra"
CONCEPT_TYPE = "algebra.quadratic"
'''


def _geometry_template(index: int) -> str:
    return f'''import random


def generate(seed):
    rng = random.Random(seed + {index})
    width = rng.randint(4, 12)
    height = rng.randint(3, 10)
    scale = rng.randint(2, 5)
    problem = (
        f"A rectangle has width {{width}} and height {{height}}. "
        f"It is enlarged by scale factor {{scale}}. What is the new area?"
    )
    answer = width * height * scale * scale
    return problem, str(answer)


CONCEPT_GROUP = "geometry"
CONCEPT_TYPE = "geometry.area_scale"
'''


def _modular_template(index: int) -> str:
    return f'''import random


def generate(seed):
    rng = random.Random(seed + {index})
    modulus = rng.choice([7, 11, 13, 17, 19])
    x = rng.randint(2, modulus - 2)
    a = rng.randint(2, modulus - 2)
    b = (a * x) % modulus
    problem = (
        f"Find the least positive integer x such that {{a}}x is congruent "
        f"to {{b}} modulo {{modulus}}."
    )
    return problem, str(x)


CONCEPT_GROUP = "number_theory"
CONCEPT_TYPE = "number_theory.modular_arithmetic"
'''

