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

**Selection is deliberately one step behind.** Scoring and training on the same
rollouts conditions the training sample on the selection event: an elite whose
measurement noise happened to land high is the one kept, which biases the
update even though each individual baseline is unbiased. Choosing WHICH elites
train from the exact score measured in the previous iteration, while training
on THIS iteration's rollouts, breaks that coupling at first order and costs one
iteration of delay -- no extra rollouts and no temporal averaging. See
``PreviousRQScoreboard``.
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
        self.groups.setdefault(program_id, []).append(
            ReplayGroup(
                program_id=program_id,
                instance=instance,
                rollouts=accepted,
                payload=payload,
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


@dataclass
class PreviousRQScoreboard:
    """The immediately previous raw R_Q for each program.

    ``record`` is called with each iteration's fresh score; ``selection_score``
    returns the most recent score known BEFORE the current iteration. Scores
    from different policies are never averaged: R_Q is defined relative to one
    current policy, so an EWMA would change the target into temporal frontierness.
    A program first scored this iteration returns None and joins training in the
    next iteration.
    """

    # program_id -> (iteration last recorded, raw score there, raw score before it).
    #
    # The third slot is what makes the lag work regardless of call order. Within
    # one outer iteration the re-scoring records THIS iteration's score before
    # the training set is built, so "everything strictly before iteration t" is
    # no longer readable from the current value alone -- it has already been
    # overwritten. Keeping the pre-update value means selection can ask for the
    # state as of the end of iteration t-1 whether or not t has been recorded.
    history: dict[str, tuple[int, float, float | None]] = field(default_factory=dict)

    def record(self, program_id: str, iteration: int, rq_score: float) -> None:
        previous = self.history.get(program_id)
        iteration = int(iteration)
        value = float(rq_score)
        if previous is None:
            before = None
        elif previous[0] >= iteration:
            # Re-recorded within the same iteration: the pre-iteration state is
            # unchanged, only the current estimate is replaced.
            before = previous[2]
        else:
            before = previous[1]
        self.history[program_id] = (iteration, value, before)

    def selection_score(self, program_id: str, iteration: int) -> float | None:
        """The program's raw R_Q as of the end of iteration ``iteration - 1``.

        None means "not eligible yet": either the program has never been scored,
        or its first score is the one taken this iteration. Both are the same
        answer -- it trains from the next iteration on.
        """
        entry = self.history.get(program_id)
        if entry is None:
            return None
        recorded_at, value, before = entry
        return value if recorded_at < int(iteration) else before

    def to_dict(self) -> dict[str, list]:
        return {
            pid: [it, val, before]
            for pid, (it, val, before) in self.history.items()
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "PreviousRQScoreboard":
        board = cls()
        for pid, entry in (payload or {}).items():
            try:
                before = float(entry[2]) if len(entry) > 2 and entry[2] is not None else None
                board.history[str(pid)] = (int(entry[0]), float(entry[1]), before)
            except (TypeError, ValueError, IndexError):
                continue
        return board
