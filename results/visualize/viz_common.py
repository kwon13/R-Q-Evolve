"""Shared, deterministic data handling for the R_Q-Evolve figures."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


OPERATORS = ("seed", "in_depth", "in_breadth")
_OP_TIE_ORDER = {op: rank for rank, op in enumerate(OPERATORS)}
_ITER_RE = re.compile(r"archive_iter(\d+)\.json$")


def snapshot_iteration(path: Path) -> int:
    match = _ITER_RE.search(path.name)
    if not match:
        raise ValueError(f"not an archive snapshot: {path}")
    return int(match.group(1))


def load_snapshots(
    archive: Path, max_outer_iteration: int | None = None
) -> list[tuple[int, dict]]:
    paths = sorted(archive.glob("archive_iter*.json"), key=snapshot_iteration)
    if max_outer_iteration is not None:
        paths = [
            path for path in paths
            if snapshot_iteration(path) <= max_outer_iteration
        ]
    if not paths:
        raise FileNotFoundError(f"no archive_iter*.json snapshots under {archive}")
    return [
        (snapshot_iteration(path), json.loads(path.read_text(encoding="utf-8")))
        for path in paths
    ]


def operator_of(program: dict) -> str:
    """Return origin operator without treating missing child metadata as a seed."""
    if int(program.get("generation", 0) or 0) == 0:
        return "seed"
    op = str((program.get("metadata") or {}).get("op") or "")
    if op not in OPERATORS:
        raise ValueError(
            f"program {program.get('program_id', '<unknown>')} has generation "
            f"{program.get('generation')} but invalid/missing operator {op!r}"
        )
    return op


def concept_type_of(program: dict) -> str | None:
    value = (program.get("metadata") or {}).get("concept_type")
    return str(value) if value else None


def inserted_report_order(log_path: Path) -> dict[str, tuple[int, int]]:
    """Map inserted child IDs to their chronological report position."""
    order: dict[str, tuple[int, int]] = {}
    if not log_path.exists():
        return order
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        record = json.loads(raw)
        iteration = int(record.get("iteration", -1))
        for index, report in enumerate(record.get("reports", [])):
            child_id = report.get("child_id")
            if report.get("status") == "inserted" and child_id:
                order.setdefault(str(child_id), (iteration, index))
    return order


def family_history(
    snapshots: list[tuple[int, dict]], log_path: Path
) -> tuple[dict[str, set[int]], dict[str, tuple[int, str]]]:
    """Return family presence and deterministic first-introduction attribution.

    A bootstrap seed always predates mutations. For multiple mutated programs
    first seen in the same snapshot, evolution-log report order breaks the tie.
    Remaining ties use generation, operator name, and program ID, never JSON
    champion-list order.
    """
    presence: dict[str, set[int]] = defaultdict(set)
    first_candidates: dict[str, tuple[int, list[dict]]] = {}
    insertion_order = inserted_report_order(log_path)

    for iteration, payload in snapshots:
        by_type: dict[str, list[dict]] = defaultdict(list)
        for champion in payload.get("champions", []):
            concept_type = concept_type_of(champion)
            if not concept_type:
                continue
            operator_of(champion)  # validate while loading
            presence[concept_type].add(iteration)
            by_type[concept_type].append(champion)
        for concept_type, candidates in by_type.items():
            if concept_type not in first_candidates:
                first_candidates[concept_type] = (iteration, candidates)

    first: dict[str, tuple[int, str]] = {}
    for concept_type, (iteration, candidates) in first_candidates.items():
        def tie_key(program: dict) -> tuple:
            op = operator_of(program)
            program_id = str(program.get("program_id") or "")
            report_position = insertion_order.get(
                program_id, (10**12, 10**12)
            )
            return (
                0 if op == "seed" else 1,
                report_position,
                int(program.get("generation", 0) or 0),
                _OP_TIE_ORDER[op],
                program_id,
            )

        winner = min(candidates, key=tie_key)
        first[concept_type] = (iteration, operator_of(winner))
    return dict(presence), first


def contiguous_ranges(iterations: set[int]) -> list[tuple[int, int]]:
    """Split observed iterations into true contiguous presence intervals."""
    values = sorted(iterations)
    if not values:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            ranges.append((start, previous))
            start = value
        previous = value
    ranges.append((start, previous))
    return ranges
