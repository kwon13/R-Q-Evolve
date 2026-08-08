#!/usr/bin/env python3
"""Plot two-panel ablations on the fixed 480-problem Evolve benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_RUNS = {
    "full": "rq_evolve_base_4b",
    "flat": "rq_evolve_4b_ablate_flat",
    "noreeval": "rq_evolve_4b_ablate_noreeval",
    "nounc": "rq_evolve_4b_ablate_nounc",
    "novar": "rq_evolve_4b_ablate_novar",
}


def _load_scores(results_dir: Path, max_step: int) -> tuple[str, dict[int, float]]:
    scores: dict[int, float] = {}
    hashes: set[str] = set()
    for summary_path in results_dir.glob("global_step_*/summary.json"):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        step = int(payload["global_step"])
        if step > max_step:
            continue
        if int(payload["num_examples"]) != 480:
            raise ValueError(
                f"{summary_path} has {payload['num_examples']} examples, not 480"
            )
        scores[step] = float(payload["score_percent"])
        hashes.add(str(payload.get("benchmark_sha256") or ""))
    if not scores:
        raise FileNotFoundError(f"no checkpoint summaries under {results_dir}")
    if len(hashes) != 1 or "" in hashes:
        raise ValueError(f"mixed or missing benchmark hashes under {results_dir}")
    return next(iter(hashes)), scores


def _load_math_benchmark_scores(scores_path: Path, step: int) -> dict[str, float]:
    """Read the final, post-recheck pass@1 table from a run's scores.md."""

    lines = scores_path.read_text(encoding="utf-8").splitlines()
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith("## pass@1 (final")
        ),
        None,
    )
    if start is None:
        raise ValueError(f"missing final pass@1 table in {scores_path}")
    header: list[str] | None = None
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue
        if cells and cells[0] == str(step):
            assert header is not None
            values = {name: float(value) for name, value in zip(header[1:], cells[1:])}
            benchmark_values = [value for name, value in values.items() if name != "AVG"]
            computed_average = sum(benchmark_values) / len(benchmark_values)
            if abs(computed_average - values["AVG"]) > 0.02:
                raise ValueError(
                    f"AVG mismatch in {scores_path}: table={values['AVG']}, "
                    f"computed={computed_average}"
                )
            return values
    raise ValueError(f"step {step} missing from final pass@1 table in {scores_path}")


def _write_tables(
    output_dir: Path,
    steps: list[int],
    series: dict[str, dict[int, float]],
    benchmark_scores: dict[str, dict[str, float]],
) -> None:
    labels = {
        "full": "R-Q-Evolve (Full 4B)",
        "flat": "Flat sampling",
        "noreeval": "Without reevaluation",
        "nounc": "Without uncertainty",
        "novar": "Without variance",
    }
    csv_path = output_dir / "ablation_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["global_step", *[labels[key] for key in series]])
        for step in steps:
            writer.writerow([step, *[f"{series[key][step]:.6f}" for key in series]])

    lines = [
        "# Evolved Performance Ablation (480 problems)",
        "",
        "| global step | " + " | ".join(labels[key] for key in series) + " |",
        "|---:|" + "---:|" * len(series),
    ]
    for step in steps:
        lines.append(
            f"| {step} | "
            + " | ".join(f"{series[key][step]:.2f}%" for key in series)
            + " |"
        )
    full_end = series["full"][steps[-1]]
    lines.extend(
        [
            "",
            f"## Difference from Full 4B at step {steps[-1]}",
            "",
            "| ablation | EPS | delta vs. Full |",
            "|---|---:|---:|",
        ]
    )
    for key in ("flat", "noreeval", "nounc", "novar"):
        value = series[key][steps[-1]]
        lines.append(f"| {labels[key]} | {value:.2f}% | {value - full_end:+.2f}%p |")
    benchmark_names = [
        name for name in benchmark_scores["full"] if name != "AVG"
    ]
    lines.extend(
        [
            "",
            f"## Standard Math Benchmarks at step {steps[-1]}",
            "",
            (
                "Final pass@1 after the stored R-Zero-aligned GPT-4o re-check. "
                "AVG is the macro average over the seven benchmark columns."
            ),
            "",
            "| method | " + " | ".join([*benchmark_names, "AVG"]) + " |",
            "|---|" + "---:|" * (len(benchmark_names) + 1),
        ]
    )
    for key in series:
        values = benchmark_scores[key]
        lines.append(
            f"| {labels[key]} | "
            + " | ".join(f"{values[name]:.2f}%" for name in [*benchmark_names, "AVG"])
            + " |"
        )
    (output_dir / "ablation_scores.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def plot(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_root = args.run_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    series: dict[str, dict[int, float]] = {}
    benchmark_scores: dict[str, dict[str, float]] = {}
    hashes: set[str] = set()
    for key, default_run in DEFAULT_RUNS.items():
        run_name = getattr(args, f"{key}_run") or default_run
        digest, scores = _load_scores(
            run_root / run_name / args.results_name, args.max_step
        )
        hashes.add(digest)
        series[key] = scores
        benchmark_scores[key] = _load_math_benchmark_scores(
            run_root / run_name / args.benchmark_scores_name, args.max_step
        )
    if len(hashes) != 1:
        raise ValueError(f"runs use different benchmark hashes: {sorted(hashes)}")

    step_sets = {tuple(sorted(values)) for values in series.values()}
    if len(step_sets) != 1:
        raise ValueError(f"runs have different checkpoint steps: {sorted(step_sets)}")
    steps = list(next(iter(step_sets)))
    if steps[-1] != args.max_step:
        raise ValueError(
            f"last common checkpoint is {steps[-1]}, expected {args.max_step}"
        )
    benchmark_columns = {tuple(values) for values in benchmark_scores.values()}
    if len(benchmark_columns) != 1:
        raise ValueError(
            f"runs contain different standard benchmark columns: {benchmark_columns}"
        )
    _write_tables(output_dir, steps, series, benchmark_scores)

    colors = {
        "full": "#2ca02c",
        "flat": "#ff7f0e",
        "noreeval": "#1f77b4",
        "nounc": "#d62728",
        "novar": "#9467bd",
    }
    labels = {
        "full": "R-Q-Evolve (Full 4B)",
        "flat": "Flat sampling",
        "noreeval": "Without reevaluation",
        "nounc": "Without uncertainty",
        "novar": "Without variance",
    }
    panels = [
        ("Ablation: Sampling & Reevaluation", ["noreeval", "flat", "full"]),
        ("Ablation: R_Q Score Components", ["nounc", "novar", "full"]),
    ]
    bar_panels = [
        ["full", "flat", "noreeval"],
        ["full", "nounc", "novar"],
    ]

    all_values = [value for values in series.values() for value in values.values()]
    lower = max(0.0, min(all_values) - 1.3)
    upper = min(100.0, max(all_values) + 1.8)
    fig = plt.figure(figsize=(25, 7.6))
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=[3.0, 1.25, 3.0, 1.25],
        wspace=0.30,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 2])]
    axes[1].sharey(axes[0])
    bar_axes = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 3])]
    tail_step = steps[-1] + max(4, (steps[-1] - steps[-2]) // 4)
    for ax, (title, keys) in zip(axes, panels):
        for key in keys:
            values = [series[key][step] for step in steps]
            linewidth = 3.3 if key == "full" else 2.8
            zorder = 4 if key == "full" else 3
            ax.step(
                [*steps, tail_step],
                [*values, values[-1]],
                where="post",
                color=colors[key],
                linewidth=linewidth,
                label=labels[key],
                zorder=zorder,
            )
            ax.annotate(
                f"{values[-1]:.2f}",
                xy=(tail_step, values[-1]),
                xytext=(-7, 10 if key == "full" else -15),
                textcoords="offset points",
                ha="right",
                va="bottom" if key == "full" else "top",
                color=colors[key],
                fontsize=10.5,
                weight="bold",
            )
        ax.set_title(title, fontsize=20, weight="bold", pad=14)
        ax.set_xlabel("Global Training Step (Saved Models)", fontsize=14, weight="bold")
        ax.set_xticks(steps)
        ax.set_xlim(steps[0], tail_step)
        ax.set_ylim(lower, upper)
        ax.grid(True, linestyle="--", alpha=0.28)
        ax.legend(loc="lower right", fontsize=11.5, framealpha=0.94)
        ax.tick_params(labelsize=11.5)
    axes[0].set_ylabel(
        "Evolved Performance Score (%)", fontsize=15, weight="bold"
    )
    axes[1].tick_params(axis="y", labelleft=False)

    bar_labels = [
        ["Full 4B", "Flat", "w/o reeval."],
        [
            "Full $R_Q$\n($L \\times U$)",
            "$L$ only\n($U = 1$)",
            "$U$ only\n($L = 1$)",
        ],
    ]
    base_math_average = float(args.base_math_average)
    benchmark_values = [scores["AVG"] for scores in benchmark_scores.values()]
    benchmark_lower = min(45.0, min([*benchmark_values, base_math_average]) - 1.0)
    benchmark_upper = max(54.5, max(benchmark_values) + 1.2)
    for ax, keys, tick_labels in zip(bar_axes, bar_panels, bar_labels):
        values = [benchmark_scores[key]["AVG"] for key in keys]
        positions = list(range(len(keys)))
        ax.bar(
            positions,
            values,
            color=["#0b7db8", "#929292", "#929292"],
            width=0.62,
            zorder=3,
        )
        ax.scatter(
            positions,
            values,
            s=24,
            color="#2b2b2b",
            edgecolors="none",
            zorder=5,
        )
        for position, value in zip(positions, values):
            ax.text(
                position,
                value + 0.25,
                f"{value:.2f}",
                va="bottom",
                ha="center",
                fontsize=11,
                color="#2b2b2b",
                weight="bold",
            )
        ax.axhline(
            base_math_average,
            color="#c47a2c",
            linestyle="--",
            linewidth=1.8,
            zorder=4,
        )
        ax.text(
            1.04,
            base_math_average,
            f"Base\n({base_math_average:.2f})",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=10.5,
            color="#b76817",
            weight="bold",
            clip_on=False,
        )
        ax.set_xticks(positions, tick_labels)
        ax.set_xlim(-0.55, len(keys) - 0.45)
        ax.set_ylim(benchmark_lower, benchmark_upper)
        ax.set_yticks([46, 48, 50, 52, 54])
        ax.set_ylabel("Average Math Score", fontsize=12, weight="bold")
        ax.set_title(
            f"Benchmark AVG @ Step {steps[-1]}",
            fontsize=15,
            weight="bold",
            pad=14,
        )
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=9.5)
        ax.tick_params(axis="y", labelsize=10.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(
        "R-Q-Evolve 4B Ablation on the Fixed 480-Problem Benchmark",
        fontsize=23,
        weight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        -0.015,
        (
            "Standard benchmark AVG: Math500 · GSM8K · AMC23 · AIME24 · "
            "AIME25 · Minerva Math · OlympiadBench"
        ),
        ha="center",
        fontsize=11,
        color="#444444",
    )
    png_path = output_dir / "evolved_performance_ablation_480.png"
    svg_path = output_dir / "evolved_performance_ablation_480.svg"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[EPB] wrote {png_path}")
    print(f"[EPB] wrote {svg_path}")
    print(f"[EPB] wrote {output_dir / 'ablation_scores.md'}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("rq_output"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rq_output/evolved_performance_480_ablation"),
    )
    parser.add_argument("--results-name", default="evolved_performance_480_v1")
    parser.add_argument("--benchmark-scores-name", default="scores.md")
    parser.add_argument(
        "--base-math-average",
        type=float,
        default=47.45,
        help="base-model seven-benchmark average shown as the dashed baseline",
    )
    parser.add_argument("--max-step", type=int, default=128)
    for key, default_run in DEFAULT_RUNS.items():
        parser.add_argument(
            f"--{key}-run",
            help=f"run directory name under --run-root (default: {default_run})",
        )
    return parser


if __name__ == "__main__":
    plot(build_argparser().parse_args())
