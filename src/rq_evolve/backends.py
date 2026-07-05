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
    keeping the whole phase to a single cumem wake_up. The verl backend stashes
    ``full_batch`` + ``decoded`` for finalization after generation.
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


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL)


def extract_boxed(text: str) -> str | None:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None
