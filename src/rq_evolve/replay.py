"""Re-scoring rollouts, kept so the solver update can train on them directly.

Every outer iteration already rolls the current solver out over each elite's n
fresh seeds to recompute R_Q. Those rollouts were thrown away and an identical
sampling pass was run again for the gradient step -- the same policy, on the
same programs, at twice the cost. Storing them removes the second pass and, more
importantly, three mismatches with it:

  * the distribution fitness describes and the one the solver trains on become
    the same rollouts, not two draws from the same programs;
  * nothing goes stale between scoring and training;
  * every instance trained on is an instance that was measured -- a tail
    instance the evaluation never saw cannot reach the batch.

Programs are ranked by the current reassessment's fitness. The training pool
is captured before mutation; new children join it on the next iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .backends import RolloutRecord
from .program import ProblemInstance


@dataclass(slots=True)
class ReplayGroup:
    """One instance and the m rollouts the solver produced for it."""

    program_id: str
    instance: ProblemInstance
    rollouts: list[RolloutRecord]
    # The backend's own generation payload for this group, when it exposes one.
    # Carrying it is what lets the trainer skip re-sampling: the text alone is
    # not enough for a gradient step, which needs the response token ids.
    payload: Any = None
    # Stable within a program's replay buffer, including across dataset
    # padding/shuffling. The iteration prevents a seed-0 row from resolving
    # to the same slot in a later iteration's rollouts.
    group_id: str | None = None

    @property
    def size(self) -> int:
        return len(self.rollouts)

    @property
    def degenerate(self) -> bool:
        """All-correct or all-wrong: every LOO advantage in this group is 0.

        Such a group contributes nothing to the update. It is not dropped here
        -- the per-instance baseline already neutralises it -- but it is
        reported, because a batch made mostly of these is a wasted step.
        """
        if not self.rollouts:
            return True
        correct = [bool(r.correct) for r in self.rollouts]
        return all(correct) or not any(correct)


@dataclass
class RolloutReplayBuffer:
    """This iteration's re-scoring rollouts, grouped by program.

    Cleared at the start of every outer iteration: rollouts are on-policy for
    exactly one update, and reusing them across an iteration would need an
    off-policy correction the design does not carry.
    """

    groups: dict[str, list[ReplayGroup]] = field(default_factory=dict)
    iteration: int = -1

    def begin_iteration(self, iteration: int) -> None:
        self.groups = {}
        self.iteration = int(iteration)

    def store(
        self,
        program_id: str,
        instance: ProblemInstance,
        rollouts: list[RolloutRecord],
        payload: Any = None,
    ) -> None:
        accepted = [
            r for r in rollouts if getattr(r, "status", "accepted") == "accepted"
        ]
        if not accepted:
            return
        groups = self.groups.setdefault(program_id, [])
        groups.append(
            ReplayGroup(
                program_id=program_id,
                instance=instance,
                rollouts=accepted,
                payload=payload,
                group_id=f"{self.iteration}:{len(groups)}",
            )
        )

    def get(self, program_id: str) -> list[ReplayGroup]:
        return self.groups.get(program_id, [])

    def has(self, program_id: str) -> bool:
        return bool(self.groups.get(program_id))

    def stats(self) -> dict[str, float | int]:
        all_groups = [g for gs in self.groups.values() for g in gs]
        if not all_groups:
            return {"replay_programs": 0, "replay_groups": 0, "replay_rollouts": 0}
        degenerate = sum(1 for g in all_groups if g.degenerate)
        return {
            "replay_programs": len(self.groups),
            "replay_groups": len(all_groups),
            "replay_rollouts": sum(g.size for g in all_groups),
            "replay_degenerate_groups": degenerate,
            # The share of the batch that can produce no gradient at all. Under
            # a per-instance LOO baseline these are self-neutralising, so this
            # is the honest measure of how much of the step is wasted.
            "replay_degenerate_frac": degenerate / len(all_groups),
        }
