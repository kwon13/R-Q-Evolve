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
    # --- async-RL metadata (defaulted: legacy call sites unaffected) --------
    # accepted | rejected; rejected records carry reject_reason and are
    # excluded from p_hat / R_Q scoring (never trained on silently).
    status: str = "accepted"
    reject_reason: str | None = None
    # Policy/adapter version at GENERATION time (PolicyVersionTracker.stamp()),
    # -1 = unknown (legacy path without a tracker).
    policy_version: int = -1
    adapter_version: int = -1
    global_step: int = -1
    source_checkpoint: str = ""
    # Wall-clock submit/finish of the chunk this sample rode in.
    ts_start: float = 0.0
    ts_end: float = 0.0
    response_tokens: int = 0


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
    # Streaming path: per-chunk results (records already verified/filtered by
    # the scheduler's consumers; entropy still pending). When set,
    # finalize_rollouts runs the deferred entropy pass over the chunks' batches
    # and backfills RolloutRecord.entropy instead of using full_batch/decoded.
    chunk_results: list | None = None


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
