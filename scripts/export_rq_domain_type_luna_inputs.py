#!/usr/bin/env python3
"""Render the R-Q 8B early/middle/final windows for a shared Luna audit.

The training rollout log stores ``(iteration, program_id, instance_seed)`` but
not the rendered problem text.  This script reconstructs the exact generator
source for every logged program, executes it with the recorded seed in the
repository sandbox, and writes one audit JSONL per non-overlapping window.

The latest contiguous sample block is retained for a resumed iteration, which
matches ``plot_rq_problem_domain_type.py``.  Multiple solver rollouts from the
same generated instance are collapsed to one row.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rq_evolve.program import ProblemProgram  # noqa: E402


WINDOWS = {
    "early_steps_000_032": (-1, 32),
    "middle_steps_097_128": (97, 128),
    "final_steps_225_255": (225, 255),
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def register_source(
    sources: dict[Any, str], key: Any, source_code: Any, origin: str
) -> None:
    if not key or not source_code:
        return
    value = str(source_code)
    previous = sources.get(key)
    if previous is not None and previous != value:
        raise ValueError(f"Conflicting source for {key}: existing vs {origin}")
    sources[key] = value


def collect_sources(
    archive_dir: Path,
) -> tuple[dict[str, str], dict[int, dict[str, str]], dict[tuple[int, str], str]]:
    seed_sources: dict[str, str] = {}
    snapshots: dict[int, dict[str, str]] = {}
    candidate_sources: dict[tuple[int, str], str] = {}

    seed_dir = ROOT / "seed_programs_domain_type"
    for path in sorted(seed_dir.glob("*.py")):
        program = ProblemProgram.from_file(path)
        register_source(seed_sources, program.program_id, program.source_code, str(path))

    for path in archive_dir.glob("archive_iter*.json"):
        try:
            snapshot_iteration = int(path.stem.replace("archive_iter", ""))
        except ValueError:
            continue
        state: dict[str, str] = {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        for field in ("champions", "structural_donors"):
            for record in payload.get(field) or []:
                register_source(
                    state,
                    str(record.get("program_id") or ""),
                    record.get("source_code"),
                    f"{path}:{field}",
                )
        snapshots[snapshot_iteration] = state

    for row in read_jsonl(archive_dir / "evolution_log.jsonl"):
        iteration = int(row.get("iteration", -1))
        for report in row.get("reports") or []:
            child_id = str(report.get("child_id") or "")
            if not child_id:
                continue
            register_source(
                candidate_sources,
                (iteration, child_id),
                report.get("source_code"),
                f"evolution_log:iteration={row.get('iteration')}",
            )
    return seed_sources, snapshots, candidate_sources


def repair_truncated_candidate_source(source_code: str) -> str:
    """Replace a log-truncated trusted ``generate`` wrapper with a small renderer.

    Evolution reports cap ``source_code`` at 8,000 characters.  The complete
    ``build_instance`` function, template parts, and helper renderer precede
    ``generate``; only the generic trusted tail is truncated.  Archive
    snapshots retain the full source for surviving candidates.  For candidates
    that did not survive, rebuild only that generic tail from the persisted
    template contract instead of dropping their already-logged rollout rows.
    """

    marker = "\ndef generate(seed):"
    if len(source_code) < 8000 or marker not in source_code:
        return source_code
    prefix = source_code.split(marker, 1)[0]
    renderer = r'''

def generate(seed):
    rng = random.Random(seed)
    payload = build_instance(rng)
    if not isinstance(payload, (tuple, list)) or len(payload) != 3:
        return None
    parameters, answer, check = payload
    problem_parts = []
    for part_kind, part_value in __rq_template_parts:
        if part_kind == "literal":
            problem_parts.append(part_value)
        else:
            problem_parts.append(__rq_render_parameter(parameters[part_value]))
    problem = "".join(problem_parts)
    if __rq_mode == "boolean":
        answer_text = "Yes" if answer is True else "No" if answer is False else str(answer)
    elif __rq_mode == "set":
        values = list(answer) if isinstance(answer, (set, frozenset, list, tuple)) else [answer]
        answer_text = r"\{" + ",".join(sorted(str(value) for value in values)) + r"\}"
    else:
        answer_text = str(answer)
    return problem, answer_text
'''
    return prefix + renderer


def resolve_source(
    iteration: int,
    program_id: str,
    seed_sources: dict[str, str],
    snapshots: dict[int, dict[str, str]],
    candidate_sources: dict[tuple[int, str], str],
) -> tuple[str | None, str]:
    # A champion replay uses the previous snapshot even if a same-ID candidate
    # was also proposed in this iteration. A newly admitted candidate has its
    # full canonical source in the current snapshot. Only non-surviving
    # candidates need the report's truncated source repaired below.
    if iteration >= 2:
        prior = snapshots.get(iteration - 1, {}).get(program_id)
        if prior is not None:
            return prior, "prior_iteration_snapshot"
    current = snapshots.get(iteration, {}).get(program_id)
    if current is not None:
        return current, "current_iteration_snapshot"
    candidate = candidate_sources.get((iteration, program_id))
    if candidate is not None:
        return repair_truncated_candidate_source(candidate), "repaired_candidate_log"
    seed = seed_sources.get(program_id)
    if seed is not None:
        return seed, "manual_seed"
    return None, "missing"


def latest_blocks(samples_path: Path) -> tuple[list[tuple[int, dict]], dict[int, int]]:
    rows: list[tuple[int, dict]] = []
    latest: dict[int, int] = {}
    previous: int | None = None
    block = -1
    for row in read_jsonl(samples_path):
        iteration = int(row["iteration"])
        if iteration != previous:
            block += 1
            previous = iteration
        rows.append((block, row))
        latest[iteration] = block
    return rows, latest


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "rq_output" / "rq_evolve_8b_domain_type_35cell_8gpu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "analysis"
            / "rq_evolve_8b_domain_type_35cell_8gpu"
            / "luna_domain_type"
        ),
    )
    args = parser.parse_args()

    archive_candidates = (args.run_dir / "rq_archive", args.run_dir / "rq_output")
    archive_dir = next(
        (path for path in archive_candidates if (path / "rollout_samples.jsonl").is_file()),
        archive_candidates[-1],
    )
    samples_path = archive_dir / "rollout_samples.jsonl"
    seed_sources, snapshots, candidate_sources = collect_sources(archive_dir)
    sample_rows, latest = latest_blocks(samples_path)

    selected: dict[str, list[tuple[int, str, int]]] = {name: [] for name in WINDOWS}
    seen: set[tuple[int, str, int]] = set()
    for block, row in sample_rows:
        iteration = int(row["iteration"])
        if block != latest[iteration]:
            continue
        window = next(
            (name for name, (start, end) in WINDOWS.items() if start <= iteration <= end),
            None,
        )
        if window is None:
            continue
        key = (iteration, str(row["program_id"]), int(row["instance_seed"]))
        if key in seen:
            continue
        seen.add(key)
        selected[window].append(key)

    missing = sorted(
        {
            (iteration, program_id)
            for keys in selected.values()
            for iteration, program_id, _ in keys
            if resolve_source(
                iteration, program_id, seed_sources, snapshots, candidate_sources
            )[0]
            is None
        }
    )
    if missing:
        raise ValueError(
            f"Missing source code for {len(missing)} iteration/program pairs: {missing[:20]}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    programs: dict[tuple[str, str], ProblemProgram] = {}
    summary: dict[str, Any] = {
        "run_dir": str(args.run_dir.resolve()),
        "archive_dir": str(archive_dir.resolve()),
        "samples": str(samples_path.resolve()),
        "windows": {},
        "source_programs": {
            "manual_seeds": len(seed_sources),
            "snapshot_states": len(snapshots),
            "same_iteration_candidates": len(candidate_sources),
        },
        "render_failures": [],
    }

    for window, keys in selected.items():
        exported: list[dict[str, Any]] = []
        failures = Counter()
        resolutions = Counter()
        for iteration, program_id, seed in keys:
            source_code, resolution = resolve_source(
                iteration, program_id, seed_sources, snapshots, candidate_sources
            )
            assert source_code is not None
            source_hash = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
            program_key = (program_id, source_hash)
            program = programs.get(program_key)
            if program is None:
                program = ProblemProgram(source_code=source_code, program_id=program_id)
                programs[program_key] = program
            resolutions[resolution] += 1
            instance = program.execute(seed=seed)
            if instance is None:
                failures[program.last_execution_error or "unknown"] += 1
                summary["render_failures"].append(
                    {
                        "window": window,
                        "iteration": iteration,
                        "program_id": program_id,
                        "instance_seed": seed,
                        "error": program.last_execution_error,
                    }
                )
                continue
            exported.append(
                {
                    "source": "rq_evolve",
                    "round": iteration,
                    "item_id": f"rq:{window}:i{iteration}:{program_id}:s{seed}",
                    "problem": instance.problem,
                    "answer": instance.answer,
                    "program_id": program_id,
                    "instance_seed": seed,
                    "iteration": iteration,
                }
            )

        path = args.output_dir / f"{window}_input.jsonl"
        write_jsonl(path, exported)
        summary["windows"][window] = {
            "iteration_range": list(WINDOWS[window]),
            "logged_unique_instances": len(keys),
            "rendered_instances": len(exported),
            "render_failures": sum(failures.values()),
            "source_resolution_counts": dict(sorted(resolutions.items())),
            "input": str(path.resolve()),
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(
            f"{window}: logged={len(keys)} rendered={len(exported)} "
            f"failures={sum(failures.values())}",
            flush=True,
        )

    summary_path = args.output_dir / "export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
