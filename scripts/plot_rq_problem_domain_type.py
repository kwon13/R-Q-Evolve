#!/usr/bin/env python3
"""Plot the cumulative Domain x Problem Type distribution of R-Q problems.

One problem is one unique ``(iteration, program_id, instance_seed)`` group in
``rollout_samples.jsonl``.  The ten (or otherwise repeated) solver rollouts for
that problem are therefore counted once.  The bootstrap iteration ``-1`` is
shown as round 0.

Program descriptors are resolved at the time the problem was used:

* newly evaluated children use that iteration's final archive-placement label;
* incumbent programs use the preceding iteration's archive snapshot; and
* bootstrap/manual seed programs use their persisted seed descriptors.

This matters because a source-identical child can be independently relabelled
when it is proposed again in a later iteration.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DISPLAY_DOMAINS = {
    "algebra": "Algebra",
    "geometry": "Geometry",
    "number_theory": "Number Theory",
    "discrete_mathematics": "Discrete Mathematics",
    "applied_mathematics": "Applied Mathematics",
    "calculus": "Calculus",
    "precalculus": "Precalculus",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            yield value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_snapshots(
    archive_dir: Path,
    end_round: int,
) -> tuple[
    list[str],
    list[str],
    dict[int, dict[str, tuple[str, str]]],
    dict[str, tuple[str, str]],
    dict[str, Any],
]:
    snapshots: dict[int, dict[str, tuple[str, str]]] = {}
    manual_seeds: dict[str, tuple[str, str]] = {}
    latest_meta: dict[str, Any] | None = None

    for round_number in range(1, end_round + 1):
        path = archive_dir / f"archive_iter{round_number}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload["meta"]
        domains = list(meta["domain_labels"])
        problem_types = list(meta["problem_type_labels"])
        latest_meta = meta
        state: dict[str, tuple[str, str]] = {}
        for champion in payload.get("champions") or []:
            label = (
                domains[int(champion["niche_domain"])],
                problem_types[int(champion["niche_problem_type"])],
            )
            program_id = str(champion["program_id"])
            state[program_id] = label
            if not str(champion.get("parent_id") or ""):
                previous = manual_seeds.get(program_id)
                if previous is not None and previous != label:
                    raise ValueError(
                        f"manual seed {program_id} changed label: {previous} -> {label}"
                    )
                manual_seeds[program_id] = label
        snapshots[round_number] = state

    if latest_meta is None:
        raise FileNotFoundError(
            f"no archive_iter*.json snapshots through round {end_round} in {archive_dir}"
        )
    return (
        list(latest_meta["domain_labels"]),
        list(latest_meta["problem_type_labels"]),
        snapshots,
        manual_seeds,
        latest_meta,
    )


def load_candidate_labels(
    path: Path,
    end_round: int,
) -> dict[tuple[int, str], tuple[str, str]]:
    labels: dict[tuple[int, str], tuple[str, str]] = {}
    for row in read_jsonl(path):
        iteration = int(row.get("iteration", -1))
        if iteration > end_round:
            break
        for report in row.get("reports") or []:
            if not isinstance(report, dict) or not report.get("child_id"):
                continue
            decision = report.get("archive_decision") or {}
            placement = decision.get("placement_labels")
            if not isinstance(placement, list) or len(placement) != 2:
                continue
            key = (iteration, str(report["child_id"]))
            label = (str(placement[0]), str(placement[1]))
            previous = labels.get(key)
            if previous is not None and previous != label:
                raise ValueError(
                    f"candidate {key} has conflicting placement labels: "
                    f"{previous} -> {label}"
                )
            labels[key] = label
    return labels


def cumulative_counts(
    *,
    samples_path: Path,
    start_round: int,
    end_round: int,
    snapshots: dict[int, dict[str, tuple[str, str]]],
    manual_seeds: dict[str, tuple[str, str]],
    candidate_labels: dict[tuple[int, str], tuple[str, str]],
) -> tuple[
    Counter[tuple[str, str]],
    Counter[int],
    Counter[str],
    set[str],
]:
    counts: Counter[tuple[str, str]] = Counter()
    round_counts: Counter[int] = Counter()
    resolution_counts: Counter[str] = Counter()
    program_ids: set[str] = set()
    seen: set[tuple[int, str, int]] = set()
    missing: list[tuple[int, str]] = []

    for row in read_jsonl(samples_path):
        raw_iteration = int(row["iteration"])
        display_round = 0 if raw_iteration == -1 else raw_iteration
        if display_round < start_round or display_round > end_round:
            continue
        program_id = str(row["program_id"])
        instance_seed = int(row["instance_seed"])
        problem_key = (raw_iteration, program_id, instance_seed)
        if problem_key in seen:
            continue
        seen.add(problem_key)
        program_ids.add(program_id)

        candidate_key = (raw_iteration, program_id)
        if raw_iteration >= 1 and candidate_key in candidate_labels:
            label = candidate_labels[candidate_key]
            resolution = "same_round_candidate_placement"
        elif raw_iteration >= 2 and program_id in snapshots.get(raw_iteration - 1, {}):
            label = snapshots[raw_iteration - 1][program_id]
            resolution = "prior_round_archive_snapshot"
        elif program_id in manual_seeds:
            label = manual_seeds[program_id]
            resolution = "manual_seed_descriptor"
        else:
            missing.append((raw_iteration, program_id))
            continue

        counts[label] += 1
        round_counts[display_round] += 1
        resolution_counts[resolution] += 1

    if missing:
        examples = ", ".join(f"({it}, {pid})" for it, pid in missing[:10])
        raise ValueError(
            f"could not resolve {len(missing)} problem descriptors; examples: {examples}"
        )
    if not seen:
        raise ValueError(f"no problem instances found for rounds {start_round}-{end_round}")
    return counts, round_counts, resolution_counts, program_ids


def plot_map(
    *,
    counts: Counter[tuple[str, str]],
    domains: list[str],
    problem_types: list[str],
    start_round: int,
    end_round: int,
    total: int,
    occupied: int,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import numpy as np

    grid = np.array(
        [[counts[(domain, problem_type)] for problem_type in problem_types] for domain in domains],
        dtype=float,
    )
    masked_grid = np.ma.masked_less_equal(grid, 0)
    fig, ax = plt.subplots(figsize=(12.2, 7.2))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#f7f7f8")
    vmax = max(float(grid.max()), 1.0)
    image = ax.imshow(
        masked_grid,
        cmap=cmap,
        norm=colors.LogNorm(vmin=1, vmax=vmax),
        aspect="auto",
    )
    for row_index, domain in enumerate(domains):
        for col_index, problem_type in enumerate(problem_types):
            count = int(grid[row_index, col_index])
            normalized = image.norm(max(count, 1))
            color = "white" if count > 0 and normalized < 0.72 else "#111318"
            ax.text(
                col_index,
                row_index,
                str(count),
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=color,
            )

    ax.set_xticks(
        range(len(problem_types)),
        [value.replace("_", " ").title() for value in problem_types],
    )
    ax.set_yticks(
        range(len(domains)),
        [DISPLAY_DOMAINS.get(value, value.replace("_", " ").title()) for value in domains],
    )
    ax.set_xlabel("Computational problem type", fontsize=12, labelpad=12)
    ax.set_ylabel("Mathematical domain", fontsize=12, labelpad=12)
    ax.set_xticks(np.arange(-0.5, len(problem_types), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(domains), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    scope_label = (
        f"cumulative rounds {start_round}–{end_round}"
        if start_round == 0
        else f"newly added rounds {start_round}–{end_round}"
    )
    ax.set_title(
        "R-Q Evolve: Domain × Deterministic Problem Type\n"
        f"{scope_label} · unique problems n={total} · "
        f"occupied {occupied}/35 cells",
        fontsize=15,
        pad=16,
    )
    fig.text(
        0.5,
        0.018,
        "Count unit: unique (round, program, instance seed) · "
        "Domain: manual seed or local policy label · "
        "Problem type: deterministic statement-and-verifier contract",
        ha="center",
        fontsize=9.3,
        color="#454954",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Problem count (log scale)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    stem = f"rq_problem_domain_type_rounds_{start_round:03d}_{end_round:03d}"
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="R-Q run directory containing rq_archive/",
    )
    parser.add_argument("--start-round", type=int, default=0)
    parser.add_argument("--end-round", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.start_round < 0:
        parser.error("--start-round must be non-negative")
    if args.end_round < args.start_round:
        parser.error("--end-round must be at least --start-round")

    run_dir = args.run_dir.expanduser().resolve()
    archive_dir = run_dir / "rq_archive"
    samples_path = archive_dir / "rollout_samples.jsonl"
    evolution_path = archive_dir / "evolution_log.jsonl"
    for path in (samples_path, evolution_path):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir.parents[1]
        / "analysis"
        / run_dir.name
        / "problem_domain_type"
        / f"rounds_{args.start_round:03d}_{args.end_round:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    domains, problem_types, snapshots, manual_seeds, meta = load_snapshots(
        archive_dir, args.end_round
    )
    candidate_labels = load_candidate_labels(evolution_path, args.end_round)
    counts, round_counts, resolution_counts, program_ids = cumulative_counts(
        samples_path=samples_path,
        start_round=args.start_round,
        end_round=args.end_round,
        snapshots=snapshots,
        manual_seeds=manual_seeds,
        candidate_labels=candidate_labels,
    )
    total = sum(counts.values())
    occupied = sum(
        counts[(domain, problem_type)] > 0
        for domain in domains
        for problem_type in problem_types
    )

    cells = [
        {
            "domain": domain,
            "problem_type": problem_type,
            "count": counts[(domain, problem_type)],
        }
        for domain in domains
        for problem_type in problem_types
    ]
    with (output_dir / "domain_type_counts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("domain", "problem_type", "count"))
        writer.writeheader()
        writer.writerows(cells)

    summary = {
        "run_dir": str(run_dir),
        "round_range": [args.start_round, args.end_round],
        "distribution_scope": (
            "cumulative" if args.start_round == 0 else "newly_added_interval"
        ),
        "bootstrap_log_iteration": -1,
        "bootstrap_display_round": 0,
        "count_unit": "unique (raw_iteration, program_id, instance_seed)",
        "unique_problem_instances": total,
        "unique_program_ids": len(program_ids),
        "occupied_cells": occupied,
        "possible_cells": len(domains) * len(problem_types),
        "round_problem_counts": {
            str(round_number): round_counts[round_number]
            for round_number in range(args.start_round, args.end_round + 1)
        },
        "descriptor_resolution_counts": dict(sorted(resolution_counts.items())),
        "domain_authority": meta.get("domain_authority"),
        "domain_labeling_method": meta.get("domain_labeling_method"),
        "domain_labeling_ruleset_sha256": meta.get("domain_labeling_ruleset_sha256"),
        "problem_type_authority": meta.get("problem_type_authority"),
        "problem_type_ruleset": meta.get("problem_type_ruleset"),
        "problem_type_ruleset_sha256": meta.get("problem_type_ruleset_sha256"),
        "rollout_samples_sha256": sha256_file(samples_path),
        "evolution_log_sha256": sha256_file(evolution_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_map(
        counts=counts,
        domains=domains,
        problem_types=problem_types,
        start_round=args.start_round,
        end_round=args.end_round,
        total=total,
        occupied=occupied,
        output_dir=output_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
