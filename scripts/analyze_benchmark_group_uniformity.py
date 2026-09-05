#!/usr/bin/env python3
"""Compare RQ-Evolve and R-Zero uniformly across benchmark-derived groups.

The seven external math benchmarks are treated as one held-out problem pool,
while benchmark identity is retained for auditing.  AIME/AMC x32 rows are
clustered by original problem before aggregation, so every distinct problem has
unit weight.  Domain/type labels are produced separately by
``classify_rzero_domain_type.py`` and reused for every method/checkpoint.

Typical workflow:

  python scripts/analyze_benchmark_group_uniformity.py --prepare-only
  python scripts/classify_rzero_domain_type.py \
    --input analysis/benchmark_group_uniformity/benchmark_manifest.jsonl \
    --output-dir analysis/benchmark_group_uniformity/classification \
    --source all --dataset-label "seven-benchmark unique problem pool"
  python scripts/analyze_benchmark_group_uniformity.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PROBLEM_TYPE_SOURCE = ROOT / "src" / "rq_evolve" / "problem_type.py"
PROBLEM_TYPE_RULESET = "computational-output-contract-v1"

# Frozen public taxonomy from src/rq_evolve/concepts.py.  Keeping this analysis
# script independent of rq_evolve.__init__ avoids importing training-only
# dependencies such as OmegaConf just to read result JSONL files.
DOMAINS = (
    "algebra",
    "geometry",
    "number_theory",
    "discrete_mathematics",
    "applied_mathematics",
    "calculus",
    "precalculus",
)
PROBLEM_TYPES = ("decision", "search", "counting", "optimization", "function")


def problem_type_ruleset_sha256() -> str:
    return hashlib.sha256(PROBLEM_TYPE_SOURCE.read_bytes()).hexdigest()


DEFAULT_RQ_ROOT = (
    ROOT / "rq_output" / "prv_rq_evolve_8b_domain_type_35cell_8gpu"
)
DEFAULT_RZERO_ROOT = Path(
    "/data1/yhoon113/R-Zero/results/ablation/"
    "qwen3-8b-base_rzero/checkpoints"
)
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "benchmark_group_uniformity"
DEFAULT_RQ_COVERAGE = (
    ROOT
    / "analysis"
    / "rq_evolve_8b_domain_type_35cell_8gpu"
    / "luna_domain_type"
    / "final_steps_225_255"
    / "domain_type_counts.csv"
)
DEFAULT_RZERO_COVERAGE = (
    ROOT
    / "analysis"
    / "rzero_domain_type"
    / "gpt-5.6-luna__none__0434b5c0715c_round8_full_unique6242"
    / "domain_type_counts.csv"
)

BENCHMARKS = (
    "aime24",
    "aime25",
    "amc23",
    "gsm8k",
    "math500",
    "minerva_math",
    "olympiadbench",
)
STEPS = (32, 64, 96, 128, 160, 192, 224, 256)
METHODS = ("R-Zero", "RQ-Evolve")

DOMAIN_DISPLAY = {
    "algebra": "Algebra",
    "geometry": "Geometry",
    "number_theory": "Number Theory",
    "discrete_mathematics": "Discrete Mathematics",
    "applied_mathematics": "Applied Mathematics",
    "calculus": "Calculus",
    "precalculus": "Precalculus",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_cell_counts(path: Path) -> dict[tuple[str, str], int]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = {
        (str(row["domain"]), str(row["problem_type"])): int(row["count"])
        for row in rows
    }
    expected = {(domain, problem_type) for domain in DOMAINS for problem_type in PROBLEM_TYPES}
    if set(counts) != expected:
        missing = sorted(expected - set(counts))
        extra = sorted(set(counts) - expected)
        raise ValueError(f"invalid coverage grid {path}: missing={missing}, extra={extra}")
    return counts


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def result_path(root: Path, step: int, benchmark: str, filename: str) -> Path:
    return root / f"global_step_{step}" / "eval" / benchmark / filename


def prepare_manifest(reference_root: Path, output_path: Path) -> list[dict[str, Any]]:
    """Write one row per distinct benchmark problem from a reference checkpoint."""

    manifest: list[dict[str, Any]] = []
    seen_global: set[str] = set()
    for benchmark in BENCHMARKS:
        path = result_path(reference_root, STEPS[-1], benchmark, "details.jsonl")
        seen_benchmark: set[str] = set()
        for row in read_jsonl(path):
            problem = str(row.get("problem") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if not problem or problem in seen_benchmark:
                continue
            seen_benchmark.add(problem)
            if problem in seen_global:
                raise ValueError(
                    f"cross-benchmark duplicate found in {benchmark}: {fingerprint(problem)}"
                )
            seen_global.add(problem)
            manifest.append(
                {
                    "item_id": f"{benchmark}:{fingerprint(problem)}",
                    "source": benchmark,
                    "benchmark": benchmark,
                    "problem": problem,
                    "answer": answer,
                }
            )
    write_jsonl(output_path, manifest)
    return manifest


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        problem = str(row.get("question") or row.get("problem") or "").strip()
        if problem:
            labels[problem] = row
    return labels


def summary_entry(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    benchmarks = summary.get("benchmarks") or {}
    if len(benchmarks) != 1:
        raise ValueError(f"expected exactly one benchmark in {path}")
    return next(iter(benchmarks.values()))


def load_item_scores(
    roots: dict[str, Path], labels: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cluster raw repeats into one mean score per distinct problem."""

    item_rows: list[dict[str, Any]] = []
    degraded: list[dict[str, Any]] = []
    reference_items: set[tuple[str, str]] | None = None
    for method, root in roots.items():
        for step in STEPS:
            step_items: set[tuple[str, str]] = set()
            for benchmark in BENCHMARKS:
                details = result_path(root, step, benchmark, "details.jsonl")
                summary = result_path(root, step, benchmark, "summary.json")
                entry = summary_entry(summary)
                is_degraded = bool(entry.get("gpt_recheck_degraded", False))
                if is_degraded:
                    degraded.append(
                        {
                            "method": method,
                            "step": step,
                            "benchmark": benchmark,
                            "gpt_calls": entry.get("gpt_calls"),
                            "gpt_calls_failed": entry.get("gpt_calls_failed"),
                        }
                    )

                grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for raw in read_jsonl(details):
                    problem = str(raw.get("problem") or "").strip()
                    if problem:
                        grouped[problem].append(raw)

                for problem, repeats in grouped.items():
                    label = labels.get(problem)
                    if label is None:
                        raise ValueError(
                            f"missing classification for {benchmark}:{fingerprint(problem)}"
                        )
                    answers = {str(row.get("answer") or "") for row in repeats}
                    if len(answers) != 1:
                        raise ValueError(
                            f"conflicting answers for {benchmark}:{fingerprint(problem)}"
                        )
                    step_items.add((benchmark, problem))
                    item_rows.append(
                        {
                            "method": method,
                            "step": step,
                            "benchmark": benchmark,
                            "item_id": f"{benchmark}:{fingerprint(problem)}",
                            "problem_hash": fingerprint(problem),
                            "raw_repeats": len(repeats),
                            "score": sum(float(row.get("score", 0.0)) for row in repeats)
                            / len(repeats),
                            "domain": label.get("domain"),
                            "domain_confidence": label.get("domain_confidence"),
                            "problem_type": label.get("problem_type"),
                            "problem_type_confidence": label.get(
                                "problem_type_confidence"
                            ),
                            "valid_step": not is_degraded,
                        }
                    )
            if reference_items is None:
                reference_items = step_items
            elif step_items != reference_items:
                missing = len(reference_items - step_items)
                extra = len(step_items - reference_items)
                raise ValueError(
                    f"item mismatch for {method} step {step}: missing={missing}, extra={extra}"
                )
    return item_rows, degraded


def grouped_metrics(
    item_rows: list[dict[str, Any]], min_cell_n: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cell_groups: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    axis_groups: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)

    for row in item_rows:
        if not row["valid_step"]:
            continue
        domain = row.get("domain")
        problem_type = row.get("problem_type")
        domain_ok = domain in DOMAINS and row.get("domain_confidence") == "high"
        type_ok = problem_type in PROBLEM_TYPES
        method = str(row["method"])
        step = int(row["step"])
        score = float(row["score"])
        if domain_ok:
            axis_groups[(method, step, "domain", str(domain))].append(score)
        if type_ok:
            axis_groups[(method, step, "problem_type", str(problem_type))].append(score)
        if domain_ok and type_ok:
            cell_groups[(method, step, str(domain), str(problem_type))].append(score)

    cell_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for step in STEPS:
            for domain in DOMAINS:
                for problem_type in PROBLEM_TYPES:
                    scores = cell_groups.get((method, step, domain, problem_type), [])
                    cell_rows.append(
                        {
                            "method": method,
                            "step": step,
                            "domain": domain,
                            "problem_type": problem_type,
                            "n": len(scores),
                            "accuracy": sum(scores) / len(scores) if scores else None,
                            "supported": len(scores) >= min_cell_n,
                        }
                    )

    axis_rows: list[dict[str, Any]] = []
    for key, scores in sorted(axis_groups.items()):
        method, step, axis, group = key
        axis_rows.append(
            {
                "method": method,
                "step": step,
                "axis": axis,
                "group": group,
                "n": len(scores),
                "accuracy": sum(scores) / len(scores),
            }
        )
    return cell_rows, axis_rows


def bottom_mean(values: list[float], fraction: float = 0.2) -> float:
    count = max(1, math.ceil(len(values) * fraction))
    return sum(sorted(values)[:count]) / count


def uniformity_metrics(
    cell_rows: list[dict[str, Any]], degraded: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    bad_method_steps = {(str(r["method"]), int(r["step"])) for r in degraded}
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for step in STEPS:
            supported = [
                float(row["accuracy"])
                for row in cell_rows
                if row["method"] == method
                and row["step"] == step
                and row["supported"]
                and row["accuracy"] is not None
            ]
            valid = (method, step) not in bad_method_steps and bool(supported)
            mean = sum(supported) / len(supported) if valid else None
            variance = (
                sum((value - mean) ** 2 for value in supported) / len(supported)
                if valid and mean is not None
                else None
            )
            rows.append(
                {
                    "method": method,
                    "step": step,
                    "valid": valid,
                    "supported_cells": len(supported),
                    "macro_cell_accuracy": mean,
                    "worst_cell_accuracy": min(supported) if valid else None,
                    "bottom_20pct_cell_accuracy": bottom_mean(supported) if valid else None,
                    "cell_accuracy_std": math.sqrt(variance)
                    if variance is not None
                    else None,
                    "macro_minus_worst": mean - min(supported)
                    if valid and mean is not None
                    else None,
                }
            )
    return rows


def comparison_rows(
    cell_rows: list[dict[str, Any]],
    item_rows: list[dict[str, Any]],
    step: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    import numpy as np

    lookup = {
        (str(row["method"]), str(row["domain"]), str(row["problem_type"])): row
        for row in cell_rows
        if int(row["step"]) == step
    }
    item_lookup = {
        (str(row["method"]), str(row["item_id"])): row
        for row in item_rows
        if int(row["step"]) == step and row["valid_step"]
    }
    paired_deltas: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (method, item_id), rq_row in item_lookup.items():
        if method != "RQ-Evolve":
            continue
        rz_row = item_lookup.get(("R-Zero", item_id))
        if rz_row is None:
            continue
        domain = rq_row.get("domain")
        problem_type = rq_row.get("problem_type")
        if (
            domain in DOMAINS
            and rq_row.get("domain_confidence") == "high"
            and problem_type in PROBLEM_TYPES
        ):
            paired_deltas[(str(domain), str(problem_type))].append(
                float(rq_row["score"]) - float(rz_row["score"])
            )

    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for problem_type in PROBLEM_TYPES:
            rq = lookup[("RQ-Evolve", domain, problem_type)]
            rz = lookup[("R-Zero", domain, problem_type)]
            rq_acc = rq["accuracy"]
            rz_acc = rz["accuracy"]
            supported = bool(rq["supported"] and rz["supported"])
            deltas = np.asarray(
                paired_deltas.get((domain, problem_type), []), dtype=float
            )
            ci_low: float | None = None
            ci_high: float | None = None
            if supported and deltas.size:
                indices = rng.integers(
                    0, deltas.size, size=(bootstrap_replicates, deltas.size)
                )
                means = deltas[indices].mean(axis=1)
                ci_low, ci_high = (
                    float(value) for value in np.quantile(means, [0.025, 0.975])
                )
            if ci_low is not None and ci_low > 0:
                inference = "improved"
            elif ci_high is not None and ci_high < 0:
                inference = "regressed"
            elif supported:
                inference = "uncertain"
            else:
                inference = "insufficient_n"
            rows.append(
                {
                    "step": step,
                    "domain": domain,
                    "problem_type": problem_type,
                    "n": int(rq["n"]),
                    "supported": supported,
                    "rzero_accuracy": rz_acc,
                    "rq_evolve_accuracy": rq_acc,
                    "delta": float(rq_acc) - float(rz_acc)
                    if rq_acc is not None and rz_acc is not None
                    else None,
                    "delta_ci_low": ci_low,
                    "delta_ci_high": ci_high,
                    "paired_bootstrap_inference": inference,
                }
            )
    return rows


def axis_comparison_rows(
    item_rows: list[dict[str, Any]],
    step: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    min_group_n: int = 20,
) -> list[dict[str, Any]]:
    import numpy as np

    item_lookup = {
        (str(row["method"]), str(row["item_id"])): row
        for row in item_rows
        if int(row["step"]) == step and row["valid_step"]
    }
    groups: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for (method, item_id), rq_row in item_lookup.items():
        if method != "RQ-Evolve":
            continue
        rz_row = item_lookup.get(("R-Zero", item_id))
        if rz_row is None:
            continue
        pair = (float(rz_row["score"]), float(rq_row["score"]))
        domain = rq_row.get("domain")
        if domain in DOMAINS and rq_row.get("domain_confidence") == "high":
            groups[("domain", str(domain))].append(pair)
        problem_type = rq_row.get("problem_type")
        if problem_type in PROBLEM_TYPES:
            groups[("problem_type", str(problem_type))].append(pair)

    rng = np.random.default_rng(bootstrap_seed + 1)
    rows: list[dict[str, Any]] = []
    ordered = (
        [("domain", value) for value in DOMAINS]
        + [("problem_type", value) for value in PROBLEM_TYPES]
    )
    for axis, group in ordered:
        pairs = np.asarray(groups.get((axis, group), []), dtype=float)
        n = len(pairs)
        supported = n >= min_group_n
        rz_acc = float(pairs[:, 0].mean()) if n else None
        rq_acc = float(pairs[:, 1].mean()) if n else None
        delta = rq_acc - rz_acc if n and rq_acc is not None and rz_acc is not None else None
        ci_low: float | None = None
        ci_high: float | None = None
        if supported:
            paired_deltas = pairs[:, 1] - pairs[:, 0]
            indices = rng.integers(0, n, size=(bootstrap_replicates, n))
            means = paired_deltas[indices].mean(axis=1)
            ci_low, ci_high = (
                float(value) for value in np.quantile(means, [0.025, 0.975])
            )
        if ci_low is not None and ci_low > 0:
            inference = "improved"
        elif ci_high is not None and ci_high < 0:
            inference = "regressed"
        elif supported:
            inference = "uncertain"
        else:
            inference = "insufficient_n"
        rows.append(
            {
                "step": step,
                "axis": axis,
                "group": group,
                "n": n,
                "supported": supported,
                "rzero_accuracy": rz_acc,
                "rq_evolve_accuracy": rq_acc,
                "delta": delta,
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
                "paired_bootstrap_inference": inference,
            }
        )
    return rows


def support_sensitivity_rows(comparison: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in (10, 20, 30, 50, 75, 100):
        selected = [
            row
            for row in comparison
            if int(row["n"]) >= threshold and row["delta"] is not None
        ]
        deltas = [float(row["delta"]) for row in selected]
        rq_values = [float(row["rq_evolve_accuracy"]) for row in selected]
        rz_values = [float(row["rzero_accuracy"]) for row in selected]
        rows.append(
            {
                "min_cell_n": threshold,
                "cells": len(selected),
                "improved": sum(value > 0 for value in deltas),
                "tied": sum(value == 0 for value in deltas),
                "regressed": sum(value < 0 for value in deltas),
                "improvement_fraction": sum(value > 0 for value in deltas) / len(deltas),
                "macro_cell_delta": sum(deltas) / len(deltas),
                "bottom_20pct_delta": bottom_mean(rq_values) - bottom_mean(rz_values),
                "worst_cell_difference": min(rq_values) - min(rz_values),
            }
        )
    return rows


def add_generation_coverage(
    comparison: list[dict[str, Any]],
    rq_counts: dict[tuple[str, str], int],
    rzero_counts: dict[tuple[str, str], int],
    coverage_min_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in comparison:
        key = (str(row["domain"]), str(row["problem_type"]))
        rq_count = rq_counts[key]
        rzero_count = rzero_counts[key]
        rq_covered = rq_count >= coverage_min_count
        rzero_covered = rzero_count >= coverage_min_count
        if rq_covered and rzero_covered:
            category = "shared_covered"
        elif rq_covered:
            category = "rq_only_expansion"
        elif rzero_covered:
            category = "rzero_only"
        else:
            category = "both_empty"
        rows.append(
            {
                **row,
                "rzero_generation_count": rzero_count,
                "rq_evolve_generation_count": rq_count,
                "coverage_min_count": coverage_min_count,
                "generation_coverage": category,
            }
        )
    return rows


def coverage_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    categories = (
        "shared_covered",
        "rq_only_expansion",
        "rzero_only",
        "both_empty",
    )
    summary: list[dict[str, Any]] = []
    for category in categories:
        selected = [
            row
            for row in rows
            if row["supported"] and row["generation_coverage"] == category
        ]
        if not selected:
            continue
        rz_accuracy = sum(float(row["rzero_accuracy"]) for row in selected) / len(selected)
        rq_accuracy = sum(float(row["rq_evolve_accuracy"]) for row in selected) / len(selected)
        summary.append(
            {
                "generation_coverage": category,
                "cells": len(selected),
                "improved_cells": sum(float(row["delta"]) > 0 for row in selected),
                "tied_cells": sum(float(row["delta"]) == 0 for row in selected),
                "regressed_cells": sum(float(row["delta"]) < 0 for row in selected),
                "rzero_macro_cell_accuracy": rz_accuracy,
                "rq_evolve_macro_cell_accuracy": rq_accuracy,
                "macro_cell_delta": rq_accuracy - rz_accuracy,
            }
        )
    return summary


def plot_final_map(rows: list[dict[str, Any]], output_dir: Path, min_cell_n: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import numpy as np

    lookup = {(r["domain"], r["problem_type"]): r for r in rows}
    matrices = []
    for field in ("rzero_accuracy", "rq_evolve_accuracy", "delta"):
        matrices.append(
            np.array(
                [
                    [
                        (
                            float(lookup[(domain, problem_type)][field])
                            if lookup[(domain, problem_type)]["supported"]
                            and lookup[(domain, problem_type)][field] is not None
                            else np.nan
                        )
                        for problem_type in PROBLEM_TYPES
                    ]
                    for domain in DOMAINS
                ]
            )
        )

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.7), constrained_layout=True)
    titles = ("R-Zero accuracy", "RQ-Evolve accuracy", "RQ-Evolve − R-Zero")
    cmaps = ("YlGnBu", "YlGnBu", "RdBu_r")
    norms = (
        colors.Normalize(vmin=0, vmax=1),
        colors.Normalize(vmin=0, vmax=1),
        colors.TwoSlopeNorm(vmin=-0.35, vcenter=0, vmax=0.35),
    )
    images = []
    for panel, (ax, matrix, title, cmap_name, norm) in enumerate(
        zip(axes, matrices, titles, cmaps, norms)
    ):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#e6e8ec")
        image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        images.append(image)
        for i, domain in enumerate(DOMAINS):
            for j, problem_type in enumerate(PROBLEM_TYPES):
                row = lookup[(domain, problem_type)]
                if not row["supported"]:
                    text = f"n={row['n']}\ninsufficient"
                    color = "#6b7078"
                else:
                    value = matrix[i, j]
                    marker = (
                        "*"
                        if panel == 2
                        and row["paired_bootstrap_inference"] in {"improved", "regressed"}
                        else ""
                    )
                    text = (
                        f"{value:+.1%}{marker}\nn={row['n']}"
                        if panel == 2
                        else f"{value:.1%}\nn={row['n']}"
                    )
                    rgba = image.cmap(image.norm(value))
                    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    color = "white" if luminance < 0.47 else "#111318"
                ax.text(j, i, text, ha="center", va="center", fontsize=8.2, color=color)
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xticks(
            range(len(PROBLEM_TYPES)),
            [value.replace("_", " ").title() for value in PROBLEM_TYPES],
            rotation=30,
            ha="right",
        )
        ax.set_yticks(range(len(DOMAINS)), [DOMAIN_DISPLAY[d] for d in DOMAINS])
        ax.set_xticks(np.arange(-0.5, len(PROBLEM_TYPES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(DOMAINS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(images[0], ax=axes[:2], fraction=0.025, pad=0.02, label="Accuracy")
    fig.colorbar(images[2], ax=axes[2], fraction=0.046, pad=0.02, label="Accuracy difference")
    fig.suptitle(
        f"Seven-benchmark held-out pool · step 256 · supported cells n ≥ {min_cell_n}\n"
        "* paired bootstrap 95% CI excludes zero",
        fontsize=16,
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"step256_domain_type_performance_map.{suffix}", dpi=220)
    plt.close(fig)


def plot_delta_bars(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    supported = sorted(
        (row for row in rows if row["supported"] and row["delta"] is not None),
        key=lambda row: float(row["delta"]),
    )
    labels = [
        f"{DOMAIN_DISPLAY[str(row['domain'])]} × {str(row['problem_type']).replace('_', ' ').title()} (n={row['n']})"
        for row in supported
    ]
    values = [100.0 * float(row["delta"]) for row in supported]
    errors_low = [
        100.0 * (float(row["delta"]) - float(row["delta_ci_low"]))
        for row in supported
    ]
    errors_high = [
        100.0 * (float(row["delta_ci_high"]) - float(row["delta"]))
        for row in supported
    ]
    colors = ["#2878b5" if value >= 0 else "#c84e52" for value in values]
    height = max(5.0, 0.42 * len(values) + 1.8)
    fig, ax = plt.subplots(figsize=(11.5, height))
    bars = ax.barh(range(len(values)), values, color=colors, alpha=0.9)
    ax.errorbar(
        values,
        range(len(values)),
        xerr=[errors_low, errors_high],
        fmt="none",
        ecolor="#20242a",
        elinewidth=1,
        capsize=2.5,
    )
    ax.axvline(0, color="#30343b", linewidth=1)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Accuracy difference (percentage points)")
    ax.set_title(
        "Step 256 supported-cell improvements: RQ-Evolve − R-Zero\n"
        "error bars = paired problem bootstrap 95% CI"
    )
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(
            value + (0.25 if value >= 0 else -0.25),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )
    fig.tight_layout()
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"step256_supported_cell_deltas.{suffix}", dpi=220)
    plt.close(fig)


def plot_coverage_performance(
    rows: list[dict[str, Any]],
    coverage_summary: list[dict[str, Any]],
    output_dir: Path,
    min_cell_n: int,
    coverage_min_count: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    category_labels = {
        "shared_covered": "Shared covered",
        "rq_only_expansion": "RQ-only expansion",
        "rzero_only": "R-Zero only",
        "both_empty": "Both empty",
    }
    category_colors = {
        "shared_covered": "#3274a1",
        "rq_only_expansion": "#e1812c",
        "rzero_only": "#9c755f",
        "both_empty": "#9da3aa",
    }
    supported = sorted(
        (row for row in rows if row["supported"] and row["delta"] is not None),
        key=lambda row: float(row["delta"]),
    )
    labels = [
        f"{DOMAIN_DISPLAY[str(row['domain'])]} × "
        f"{str(row['problem_type']).replace('_', ' ').title()} (n={row['n']})"
        for row in supported
    ]
    values = np.asarray([100.0 * float(row["delta"]) for row in supported])
    errors_low = np.asarray(
        [
            100.0 * (float(row["delta"]) - float(row["delta_ci_low"]))
            for row in supported
        ]
    )
    errors_high = np.asarray(
        [
            100.0 * (float(row["delta_ci_high"]) - float(row["delta"]))
            for row in supported
        ]
    )

    fig = plt.figure(figsize=(18.5, 10.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(2.55, 1.0), wspace=0.12)
    ax = fig.add_subplot(grid[0, 0])
    summary_ax = fig.add_subplot(grid[0, 1])

    for index, (row, value, low, high) in enumerate(
        zip(supported, values, errors_low, errors_high)
    ):
        category = str(row["generation_coverage"])
        color = category_colors[category]
        marker = "D" if row["paired_bootstrap_inference"] == "improved" else "o"
        ax.errorbar(
            value,
            index,
            xerr=np.asarray([[low], [high]]),
            fmt=marker,
            markersize=7.2,
            markerfacecolor=color,
            markeredgecolor="#20242a",
            markeredgewidth=0.65,
            ecolor=color,
            elinewidth=1.7,
            capsize=3,
            zorder=3,
        )
        ax.text(
            value + (0.55 if value >= 0 else -0.55),
            index,
            f"{value:+.1f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9.5,
            color="#20242a",
        )
    ax.axvline(0, color="#30343b", linewidth=1.1)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Accuracy difference (percentage points)")
    ax.set_title("Cell-level difference with paired 95% confidence intervals", pad=12)
    ax.grid(axis="x", alpha=0.2)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=category_colors[key],
            markeredgecolor="#20242a",
            markersize=8,
            label=value,
        )
        for key, value in category_labels.items()
        if any(row["generation_coverage"] == key for row in supported)
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#20242a",
            markersize=7,
            label="95% CI excludes zero",
        )
    )
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=9.5)

    ordered_summary = [
        row
        for category in (
            "shared_covered",
            "rq_only_expansion",
            "rzero_only",
            "both_empty",
        )
        for row in coverage_summary
        if row["generation_coverage"] == category
    ]
    x = np.arange(len(ordered_summary), dtype=float)
    width = 0.34
    rz_values = [100.0 * float(row["rzero_macro_cell_accuracy"]) for row in ordered_summary]
    rq_values = [100.0 * float(row["rq_evolve_macro_cell_accuracy"]) for row in ordered_summary]
    rz_bars = summary_ax.bar(
        x - width / 2,
        rz_values,
        width,
        label="R-Zero",
        color="#c44e52",
        alpha=0.9,
    )
    rq_bars = summary_ax.bar(
        x + width / 2,
        rq_values,
        width,
        label="RQ-Evolve",
        color="#3274a1",
        alpha=0.9,
    )
    for left, right, row in zip(rz_bars, rq_bars, ordered_summary):
        top = max(left.get_height(), right.get_height())
        summary_ax.text(
            (left.get_x() + left.get_width() / 2 + right.get_x() + right.get_width() / 2)
            / 2,
            top + 2.0,
            f"Δ {100.0 * float(row['macro_cell_delta']):+.1f} pp\n"
            f"{row['improved_cells']}/{row['cells']} cells ↑",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )
    summary_ax.set_xticks(
        x,
        [
            f"{category_labels[str(row['generation_coverage'])]}\n"
            f"({row['cells']} cells)"
            for row in ordered_summary
        ],
        rotation=14,
        ha="right",
    )
    summary_ax.set_ylim(0, 100)
    summary_ax.set_ylabel("Macro-cell accuracy (%)")
    summary_ax.set_title("Performance by generation coverage", pad=12)
    summary_ax.grid(axis="y", alpha=0.2)
    summary_ax.legend(frameon=False, loc="upper left")

    fig.suptitle(
        "Step 256 benchmark performance by final-stage generation coverage\n"
        f"benchmark-supported cells n ≥ {min_cell_n} · generation covered if count ≥ "
        f"{coverage_min_count} · R-Zero Round 8 vs RQ-Evolve steps 225–255",
        fontsize=16,
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(
            output_dir / f"step256_performance_by_generation_coverage.{suffix}",
            dpi=220,
        )
    plt.close(fig)


def plot_trajectory(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = (
        ("macro_cell_accuracy", "Macro-cell accuracy"),
        ("worst_cell_accuracy", "Worst supported-cell accuracy"),
        ("bottom_20pct_cell_accuracy", "Bottom-20% cell accuracy"),
        ("cell_accuracy_std", "Across-cell standard deviation"),
    )
    colors = {"R-Zero": "#c84e52", "RQ-Evolve": "#2878b5"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True)
    for ax, (field, title) in zip(axes.flat, fields):
        for method in METHODS:
            selected = [row for row in rows if row["method"] == method]
            xs = [int(row["step"]) for row in selected]
            ys = [
                100.0 * float(row[field]) if row["valid"] and row[field] is not None else math.nan
                for row in selected
            ]
            ax.plot(xs, ys, marker="o", linewidth=2, label=method, color=colors[method])
        ax.set_title(title)
        ax.set_ylabel("Percent")
        ax.grid(alpha=0.22)
    for ax in axes[-1, :]:
        ax.set_xlabel("Global step")
        ax.set_xticks(STEPS)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Performance uniformity across supported Domain × Problem Type cells")
    fig.tight_layout()
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"uniformity_trajectory.{suffix}", dpi=220)
    plt.close(fig)


def build_summary(
    manifest: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    comparison: list[dict[str, Any]],
    axis_comparison: list[dict[str, Any]],
    coverage_summary: list[dict[str, Any]],
    uniformity: list[dict[str, Any]],
    degraded: list[dict[str, Any]],
    min_cell_n: int,
) -> dict[str, Any]:
    domain_high = sum(
        row.get("domain") in DOMAINS and row.get("domain_confidence") == "high"
        for row in labels.values()
    )
    type_mapped = sum(row.get("problem_type") in PROBLEM_TYPES for row in labels.values())
    joint = sum(
        row.get("domain") in DOMAINS
        and row.get("domain_confidence") == "high"
        and row.get("problem_type") in PROBLEM_TYPES
        for row in labels.values()
    )
    supported = [row for row in comparison if row["supported"]]
    deltas = [float(row["delta"]) for row in supported if row["delta"] is not None]
    final_uniformity = {
        str(row["method"]): row
        for row in uniformity
        if int(row["step"]) == STEPS[-1]
    }
    axis_summary: dict[str, Any] = {}
    for axis in ("domain", "problem_type"):
        selected = [
            row for row in axis_comparison if row["axis"] == axis and row["supported"]
        ]
        axis_summary[axis] = {
            "supported_groups": len(selected),
            "improved": sum(float(row["delta"]) > 0 for row in selected),
            "tied": sum(float(row["delta"]) == 0 for row in selected),
            "regressed": sum(float(row["delta"]) < 0 for row in selected),
            "significantly_improved": sum(
                row["paired_bootstrap_inference"] == "improved" for row in selected
            ),
            "significantly_regressed": sum(
                row["paired_bootstrap_inference"] == "regressed" for row in selected
            ),
        }
    return {
        "analysis": "benchmark group-wise performance uniformity",
        "benchmarks": list(BENCHMARKS),
        "steps": list(STEPS),
        "unique_problem_rows": len(manifest),
        "domain_high_confidence_rows": domain_high,
        "problem_type_classified_rows": type_mapped,
        "joint_mapped_rows": joint,
        "joint_mapped_fraction": joint / len(manifest) if manifest else 0.0,
        "problem_type_ruleset": PROBLEM_TYPE_RULESET,
        "problem_type_ruleset_sha256": problem_type_ruleset_sha256(),
        "min_supported_cell_n": min_cell_n,
        "supported_cells": len(supported),
        "supported_cells_improved": sum(value > 0 for value in deltas),
        "supported_cells_tied": sum(value == 0 for value in deltas),
        "supported_cells_regressed": sum(value < 0 for value in deltas),
        "supported_cells_significantly_improved": sum(
            row["paired_bootstrap_inference"] == "improved" for row in supported
        ),
        "supported_cells_significantly_regressed": sum(
            row["paired_bootstrap_inference"] == "regressed" for row in supported
        ),
        "supported_cells_statistically_uncertain": sum(
            row["paired_bootstrap_inference"] == "uncertain" for row in supported
        ),
        "supported_cell_improvement_fraction": (
            sum(value > 0 for value in deltas) / len(deltas) if deltas else None
        ),
        "supported_cell_mean_delta": sum(deltas) / len(deltas) if deltas else None,
        "supported_cell_worst_delta": min(deltas) if deltas else None,
        "final_uniformity": final_uniformity,
        "final_axis_comparison": axis_summary,
        "final_generation_coverage_comparison": coverage_summary,
        "generation_coverage_rule": (
            "count >= 1; R-Zero round 8 versus RQ-Evolve steps 225-255"
        ),
        "degraded_evaluations": degraded,
        "weighting": (
            "one unit per distinct problem; x32 AIME/AMC repeats averaged within problem; "
            "cell metrics macro-averaged with no benchmark-size weighting"
        ),
        "support_rule": (
            "joint Domain x Problem Type metrics require high-confidence domain, "
            f"classified problem type, and n >= {min_cell_n} distinct problems"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rq-root", type=Path, default=DEFAULT_RQ_ROOT)
    parser.add_argument("--rzero-root", type=Path, default=DEFAULT_RZERO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--rq-coverage", type=Path, default=DEFAULT_RQ_COVERAGE)
    parser.add_argument("--rzero-coverage", type=Path, default=DEFAULT_RZERO_COVERAGE)
    parser.add_argument("--coverage-min-count", type=int, default=1)
    parser.add_argument("--min-cell-n", type=int, default=20)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=271828)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.min_cell_n < 1:
        parser.error("--min-cell-n must be positive")
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")
    if args.coverage_min_count < 1:
        parser.error("--coverage-min-count must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "benchmark_manifest.jsonl"
    manifest = prepare_manifest(args.rq_root, manifest_path)
    print(f"[uniformity] manifest: {len(manifest)} unique problems -> {manifest_path}")
    if args.prepare_only:
        return

    labels_path = args.labels or args.output_dir / "classification" / "labels.jsonl"
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"classification labels not found: {labels_path}\n"
            "Run scripts/classify_rzero_domain_type.py as shown in this script's docstring."
        )
    labels = load_labels(labels_path)
    roots = {"R-Zero": args.rzero_root, "RQ-Evolve": args.rq_root}
    item_rows, degraded = load_item_scores(roots, labels)
    cell_rows, axis_rows = grouped_metrics(item_rows, args.min_cell_n)
    uniformity = uniformity_metrics(cell_rows, degraded)
    comparison = comparison_rows(
        cell_rows,
        item_rows,
        STEPS[-1],
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    axis_comparison = axis_comparison_rows(
        item_rows,
        STEPS[-1],
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    sensitivity = support_sensitivity_rows(comparison)
    coverage_comparison = add_generation_coverage(
        comparison,
        read_cell_counts(args.rq_coverage),
        read_cell_counts(args.rzero_coverage),
        args.coverage_min_count,
    )
    coverage_summary = coverage_summary_rows(coverage_comparison)

    write_jsonl(args.output_dir / "item_scores.jsonl", item_rows)
    write_csv(
        args.output_dir / "cell_metrics.csv",
        cell_rows,
        ["method", "step", "domain", "problem_type", "n", "accuracy", "supported"],
    )
    write_csv(
        args.output_dir / "axis_metrics.csv",
        axis_rows,
        ["method", "step", "axis", "group", "n", "accuracy"],
    )
    write_csv(
        args.output_dir / "uniformity_metrics.csv",
        uniformity,
        [
            "method",
            "step",
            "valid",
            "supported_cells",
            "macro_cell_accuracy",
            "worst_cell_accuracy",
            "bottom_20pct_cell_accuracy",
            "cell_accuracy_std",
            "macro_minus_worst",
        ],
    )
    write_csv(
        args.output_dir / "step256_cell_comparison.csv",
        comparison,
        [
            "step",
            "domain",
            "problem_type",
            "n",
            "supported",
            "rzero_accuracy",
            "rq_evolve_accuracy",
            "delta",
            "delta_ci_low",
            "delta_ci_high",
            "paired_bootstrap_inference",
        ],
    )
    write_csv(
        args.output_dir / "step256_axis_comparison.csv",
        axis_comparison,
        [
            "step",
            "axis",
            "group",
            "n",
            "supported",
            "rzero_accuracy",
            "rq_evolve_accuracy",
            "delta",
            "delta_ci_low",
            "delta_ci_high",
            "paired_bootstrap_inference",
        ],
    )
    write_csv(
        args.output_dir / "support_threshold_sensitivity.csv",
        sensitivity,
        [
            "min_cell_n",
            "cells",
            "improved",
            "tied",
            "regressed",
            "improvement_fraction",
            "macro_cell_delta",
            "bottom_20pct_delta",
            "worst_cell_difference",
        ],
    )
    write_csv(
        args.output_dir / "step256_coverage_cell_comparison.csv",
        coverage_comparison,
        [
            "step",
            "domain",
            "problem_type",
            "n",
            "supported",
            "rzero_accuracy",
            "rq_evolve_accuracy",
            "delta",
            "delta_ci_low",
            "delta_ci_high",
            "paired_bootstrap_inference",
            "rzero_generation_count",
            "rq_evolve_generation_count",
            "coverage_min_count",
            "generation_coverage",
        ],
    )
    write_csv(
        args.output_dir / "step256_coverage_summary.csv",
        coverage_summary,
        [
            "generation_coverage",
            "cells",
            "improved_cells",
            "tied_cells",
            "regressed_cells",
            "rzero_macro_cell_accuracy",
            "rq_evolve_macro_cell_accuracy",
            "macro_cell_delta",
        ],
    )
    summary = build_summary(
        manifest,
        labels,
        comparison,
        axis_comparison,
        coverage_summary,
        uniformity,
        degraded,
        args.min_cell_n,
    )
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    plot_final_map(comparison, args.output_dir, args.min_cell_n)
    plot_delta_bars(comparison, args.output_dir)
    plot_coverage_performance(
        coverage_comparison,
        coverage_summary,
        args.output_dir,
        args.min_cell_n,
        args.coverage_min_count,
    )
    plot_trajectory(uniformity, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
