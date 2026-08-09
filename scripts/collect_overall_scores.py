#!/usr/bin/env python3
"""Build one math+general model comparison table, including RQ Evolve."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

MATH = [
    "math500",
    "gsm8k",
    "amc23",
    "aime24",
    "aime25",
    "minerva_math",
    "olympiadbench",
]
GENERAL = ["supergpqa", "mmlupro", "bbeh"]
COLUMNS = [
    "MODEL",
    "AVG",
    "M-AVG",
    "G-AVG",
    "MATH",
    "GSM8K",
    "AMC",
    "AIME24",
    "AIME25",
    "MINERVA",
    "OLYMPIAD",
    "SUPER-GPQA",
    "MMLU-PRO",
    "BBEH",
]


def _read_entry(path: Path, benchmark: str) -> dict[str, Any]:
    summary_path = path / benchmark / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload["benchmarks"][benchmark]


def _load_standalone(model_dir: Path) -> dict[str, Any]:
    math: dict[str, float] = {}
    general: dict[str, float] = {}
    degraded: set[str] = set()
    for benchmark in MATH:
        entry = _read_entry(model_dir / "eval", benchmark)
        math[benchmark] = 100.0 * float(entry["pass_at_1"])
        if entry.get("gpt_recheck_degraded"):
            degraded.add(benchmark)
    for benchmark in GENERAL:
        entry = _read_entry(model_dir / "eval_general", benchmark)
        general[benchmark] = 100.0 * float(entry["pass_at_1"])
    return {
        "model": model_dir.name,
        "math": math,
        "general": general,
        "degraded": degraded,
        "provenance": "standalone benchmark summaries",
    }


def _load_curated(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    math = {name: float(payload["math"][name]) for name in MATH}
    general = {name: float(payload["general"][name]) for name in GENERAL}
    row = {
        "model": str(payload["model"]),
        "math": math,
        "general": general,
        "degraded": set(),
        "provenance": str(payload.get("provenance") or "curated result"),
        "checkpoint_step": int(payload["checkpoint_step"]),
    }
    expected = payload.get("expected_averages") or {}
    computed = _averages(row)
    for key in ("AVG", "M-AVG", "G-AVG"):
        if key in expected and abs(float(expected[key]) - computed[key]) > 0.011:
            raise ValueError(
                f"{path}: expected {key}={expected[key]}, computed={computed[key]:.4f}"
            )
    return row


def _averages(row: dict[str, Any]) -> dict[str, float]:
    math_values = [row["math"][name] for name in MATH]
    general_values = [row["general"][name] for name in GENERAL]
    return {
        "M-AVG": sum(math_values) / len(math_values),
        "G-AVG": sum(general_values) / len(general_values),
        "AVG": (sum(math_values) + sum(general_values))
        / (len(math_values) + len(general_values)),
    }


def _cells(row: dict[str, Any]) -> list[float | str]:
    averages = _averages(row)
    return [
        row["model"],
        averages["AVG"],
        averages["M-AVG"],
        averages["G-AVG"],
        *[row["math"][name] for name in MATH],
        *[row["general"][name] for name in GENERAL],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-bench-root", type=Path, default=Path("rq_output/model_bench")
    )
    parser.add_argument(
        "--curated-rq",
        type=Path,
        default=Path("rq_output/model_bench/rq_evolve_step160_curated.json"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("rq_output/model_bench/scores_overall"),
    )
    args = parser.parse_args()

    root = args.model_bench_root.expanduser().resolve()
    curated = _load_curated(args.curated_rq.expanduser().resolve())
    standalone_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "eval").is_dir() and (path / "eval_general").is_dir()
    )
    rows = [curated, *[_load_standalone(path) for path in standalone_dirs]]

    prefix = args.output_prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for row in rows:
            cells = _cells(row)
            writer.writerow([cells[0], *[f"{float(value):.2f}" for value in cells[1:]]])
    tsv_path = prefix.with_suffix(".tsv")
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(COLUMNS)
        for row in rows:
            cells = _cells(row)
            writer.writerow([cells[0], *[f"{float(value):.2f}" for value in cells[1:]]])

    md_lines = [
        "# Overall benchmark scores",
        "",
        (
            "AVG is the equal-weight mean over all ten benchmarks; M-AVG is "
            "the seven-math-benchmark mean; G-AVG is the three-general-benchmark mean."
        ),
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|---|" + "---:|" * (len(COLUMNS) - 1),
    ]
    any_degraded = False
    for row in rows:
        cells = _cells(row)
        formatted = []
        benchmark_order = [*MATH, *GENERAL]
        for index, value in enumerate(cells[1:]):
            marker = ""
            if index >= 3:
                benchmark = benchmark_order[index - 3]
                if benchmark in row["degraded"]:
                    marker = "*"
                    any_degraded = True
            formatted.append(f"{float(value):.2f}{marker}")
        label = str(cells[0])
        if row is curated:
            label += f" (step {row['checkpoint_step']})"
        md_lines.append(f"| {label} | " + " | ".join(formatted) + " |")
    md_lines.extend(
        [
            "",
            (
                f"RQ Evolve step {curated['checkpoint_step']} uses the "
                f"user-confirmed curated row: {curated['provenance']}."
            ),
        ]
    )
    if any_degraded:
        md_lines.extend(
            [
                "",
                (
                    "`*` indicates a stored standalone result whose GPT re-check "
                    "was degraded; that cell is effectively the pre-GPT score."
                ),
            ]
        )
    md_path = prefix.with_suffix(".md")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[scores] wrote {md_path}")
    print(f"[scores] wrote {csv_path}")
    print(f"[scores] wrote {tsv_path}")


if __name__ == "__main__":
    main()
