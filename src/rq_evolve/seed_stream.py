"""A monotone, never-reused seed source, one cursor per program.

Every evaluation draws FRESH seeds. A fixed evaluation set (the old design
always scored seed 0) lets a generator overfit the instances it is graded on --
a branch that returns a calibrated instance exactly when ``z == 0`` is cheap to
write and impossible to detect from the score. Unpredictable seeds are the
integrity device; for an honest program a fresh draw costs only O(D/n)
estimation noise.

The cursor is per program and persists across restarts, so a resumed run does
not re-issue seeds the pre-restart run already graded on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SeedStream:
    """Hands out each program's next unused seeds and remembers where it stopped."""

    cursor: dict[str, int] = field(default_factory=dict)
    seed_refresh: bool = True

    def take(self, program_id: str, count: int) -> list[int]:
        """The next ``count`` seeds for ``program_id``, consuming them.

        When ``seed_refresh`` is False, cursor does not advance and all seeds
        are issued as 0.
        """
        if count <= 0:
            return []
        if not self.seed_refresh:
            return [0] * count
        start = int(self.cursor.get(program_id, 0))
        self.cursor[program_id] = start + count
        return list(range(start, start + count))

    def peek(self, program_id: str) -> int:
        """The seed ``take`` would issue next, without consuming it."""
        if not self.seed_refresh:
            return 0
        return int(self.cursor.get(program_id, 0))

    def reserve_through(self, program_id: str, seed: int) -> None:
        """Advance the cursor past ``seed`` if it has not already passed it.

        Lets a pre-existing record of consumed seeds (an archive written before
        this stream existed) be folded in without re-issuing them.
        """
        if not self.seed_refresh:
            return
        self.cursor[program_id] = max(int(self.cursor.get(program_id, 0)), int(seed) + 1)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {pid: int(n) for pid, n in self.cursor.items() if n}
        if not self.seed_refresh:
            data["__seed_refresh__"] = False
        return data

    @classmethod
    def from_dict(cls, payload: dict | None, seed_refresh: bool | None = None) -> "SeedStream":
        if not payload:
            return cls(seed_refresh=True if seed_refresh is None else seed_refresh)
        refresh = (
            bool(payload.get("__seed_refresh__", True))
            if seed_refresh is None
            else bool(seed_refresh)
        )
        cursor = {
            str(k): int(v)
            for k, v in payload.items()
            if not str(k).startswith("__")
        }
        return cls(cursor=cursor, seed_refresh=refresh)

    @classmethod
    def from_used_seeds(
        cls, used: dict[str, list[int] | set[int]] | None, seed_refresh: bool = True
    ) -> "SeedStream":
        """Build a cursor from the legacy ``used_seeds`` set-per-program record.

        The old format tracked WHICH seeds were emitted, not how far the stream
        had advanced. The safe reconstruction is one past the largest seed ever
        used: re-issuing a lower unused seed would be legal, but only at the
        cost of having to persist the holes forever.
        """
        stream = cls(seed_refresh=seed_refresh)
        if not seed_refresh:
            return stream
        for pid, seeds in (used or {}).items():
            if seeds:
                stream.cursor[str(pid)] = int(max(seeds)) + 1
        return stream
