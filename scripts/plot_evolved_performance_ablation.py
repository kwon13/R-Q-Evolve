#!/usr/bin/env python3
"""Plot two-panel ablations on the fixed 480-problem Evolve benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
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

ARM_LABELS = {
    "full": "R-Q-Evolve (Full 4B)",
    "flat": "Flat archive (no MAP bins)",
    "noreeval": "Without reevaluation",
    "nounc": "Without uncertainty",
    "novar": "Without variance",
}

# Okabe-Ito, matching build_separate_detailed_evolution_figures.py so this figure
# sits next to the others in the paper without a palette change.
ARM_COLORS = {
    "full": "#009E73",
    "flat": "#E69F00",
    "noreeval": "#0072B2",
    "nounc": "#D55E00",
    "novar": "#CC79A7",
}

C_TEXT = "#20242A"
C_MUTED = "#65758B"
C_GRID = "#D5DBE3"
C_BAR_HIGHLIGHT = "#0072B2"
C_BAR_PLAIN = "#B9C0CA"
C_BASELINE = "#B7791F"


def configure_style() -> None:
    """The paper's shared rcParams (see build_separate_detailed_evolution_figures)."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

# One entry per figure column pair: the line panel's title and draw order, the
# bar panel's order, and that bar panel's tick labels. "full" is named
# differently in the two panels because the second contrasts the R_Q factors
# rather than the runs. A panel whose arms are not all selected is dropped, so
# the same definition serves the five-arm 4B figure and a partial set such as
# the three 8B runs.
PANELS = (
    (
        "Archive structure",
        ("noreeval", "flat", "full"),
        ("full", "flat", "noreeval"),
        {"full": "Full", "flat": "Flat", "noreeval": "w/o re-eval"},
    ),
    (
        "$R_Q$ components",
        ("nounc", "novar", "full"),
        ("full", "nounc", "novar"),
        {"full": "Full", "nounc": "$L$ only", "novar": "$U$ only"},
    ),
)


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


SECTION_HEADERS = {
    "final": "## pass@1 (final",
    "pre-gpt": "## pass@1 (pre-GPT",
}


def _load_math_benchmark_scores(
    scores_path: Path, step: int, section: str = "final"
) -> dict[str, float]:
    """Read one pass@1 table from a run's scores.md.

    ``section`` selects the post-recheck table ("final") or the math_verify-only
    one ("pre-gpt"). The pre-GPT table is the honest choice whenever the arms
    being compared did not all get a working GPT-4o re-check — mixing the two
    moves an arm by ~12 points on its own (see scripts/rerun_gpt_recheck.py).
    """

    header_prefix = SECTION_HEADERS[section]
    lines = scores_path.read_text(encoding="utf-8").splitlines()
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith(header_prefix)
        ),
        None,
    )
    if start is None:
        raise ValueError(f"missing {section} pass@1 table in {scores_path}")
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
            # The pre-GPT table annotates its AVG cell with the flip count,
            # e.g. "43.93 (+693)"; keep only the number.
            values = {
                name: float(value.split()[0])
                for name, value in zip(header[1:], cells[1:])
            }
            benchmark_values = [value for name, value in values.items() if name != "AVG"]
            computed_average = sum(benchmark_values) / len(benchmark_values)
            if abs(computed_average - values["AVG"]) > 0.02:
                raise ValueError(
                    f"AVG mismatch in {scores_path}: table={values['AVG']}, "
                    f"computed={computed_average}"
                )
            return values
    raise ValueError(f"step {step} missing from {section} pass@1 table in {scores_path}")


def _write_tables(
    output_dir: Path,
    steps: list[int],
    series: dict[str, dict[int, float]],
    benchmark_scores: dict[str, dict[str, float]],
    labels: dict[str, str],
    section: str = "final",
) -> None:
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
            f"## Difference from {labels['full']} at step {steps[-1]}",
            "",
            "| ablation | EPS | delta vs. Full |",
            "|---|---:|---:|",
        ]
    )
    for key in series:
        if key == "full":
            continue
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
                if section == "final"
                else "pass@1 from math_verify only, before any GPT-4o re-check. "
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

    configure_style()

    run_root = args.run_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    arms = [key.strip() for key in args.arms.split(",") if key.strip()]
    unknown = [key for key in arms if key not in DEFAULT_RUNS]
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; choose from {list(DEFAULT_RUNS)}")
    if "full" not in arms:
        raise ValueError("--arms must include 'full' (every delta is measured against it)")
    labels = dict(ARM_LABELS)
    if args.full_label:
        labels["full"] = args.full_label

    series: dict[str, dict[int, float]] = {}
    benchmark_scores: dict[str, dict[str, float]] = {}
    hashes: set[str] = set()
    for key in arms:
        default_run = DEFAULT_RUNS[key]
        run_name = getattr(args, f"{key}_run") or default_run
        digest, scores = _load_scores(
            run_root / run_name / args.results_name, args.max_step
        )
        hashes.add(digest)
        series[key] = scores
        benchmark_scores[key] = _load_math_benchmark_scores(
            run_root / run_name / args.benchmark_scores_name,
            args.max_step,
            args.benchmark_scores_section,
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
    _write_tables(
        output_dir,
        steps,
        series,
        benchmark_scores,
        labels,
        args.benchmark_scores_section,
    )

    colors = ARM_COLORS
    selected = set(arms)
    active = [entry for entry in PANELS if set(entry[1]) <= selected]
    if not active:
        raise ValueError(
            f"arms {arms} do not fill any panel; each panel needs all of "
            + " or ".join(str(list(entry[1])) for entry in PANELS)
        )
    suffix = f" ({args.size_label})" if args.size_label else ""
    panels = [(title + suffix, list(line_arms)) for title, line_arms, _, _ in active]
    bar_panels = [list(bar_arms) for _, _, bar_arms, _ in active]
    bar_labels = [[ticks[key] for key in bar_arms] for _, _, bar_arms, ticks in active]

    all_values = [value for values in series.values() for value in values.values()]
    lower = max(0.0, min(all_values) - 1.3)
    upper = min(100.0, max(all_values) + 1.8)
    # Sized for \includegraphics[width=\textwidth] at the paper's 9 pt base font,
    # not for viewing at 100%.
    fig = plt.figure(figsize=(3.9 * len(active), 2.9))
    grid = fig.add_gridspec(
        1,
        2 * len(active),
        width_ratios=[3.0, 1.45] * len(active),
        wspace=0.46,
    )
    axes = [fig.add_subplot(grid[0, 2 * i]) for i in range(len(active))]
    for ax in axes[1:]:
        ax.sharey(axes[0])
    bar_axes = [fig.add_subplot(grid[0, 2 * i + 1]) for i in range(len(active))]
    tail_step = steps[-1] + max(4, (steps[-1] - steps[-2]) // 4)
    for ax, (title, keys) in zip(axes, panels):
        for key in keys:
            values = [series[key][step] for step in steps]
            linewidth = 1.8 if key == "full" else 1.4
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
        # End labels stay INSIDE the axes. At the paper's figure width there is
        # no margin to the right, so anything placed outside lands on the bar
        # panel next door. Arms finishing within a hair of each other are pushed
        # apart from the bottom up.
        ends = sorted(((series[key][steps[-1]], key) for key in keys))
        gap = 0.075 * (upper - lower)
        placed: list[float] = []
        for value, key in ends:
            target = value if not placed else max(value, placed[-1] + gap)
            placed.append(target)
            ax.annotate(
                f"{value:.1f}",
                xy=(steps[-1], target),
                xytext=(3, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                color=colors[key],
                fontsize=6.5,
                weight="bold",
                zorder=6,
                bbox=dict(boxstyle="square,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.82),
            )
        ax.set_title(title, weight="bold", pad=6, color=C_TEXT)
        ax.set_xlabel("Global training step")
        ax.set_xticks(steps)
        ax.set_xlim(steps[0], tail_step)
        ax.set_ylim(lower, upper)
        ax.grid(True, linestyle="-", linewidth=0.5, color=C_GRID)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.95,
                  borderpad=0.35, handlelength=1.4, labelspacing=0.3)
        ax.tick_params(colors=C_TEXT)
    axes[0].set_ylabel("Evolved performance score (%)")
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    base_math_average = float(args.base_math_average)
    benchmark_values = [scores["AVG"] for scores in benchmark_scores.values()]
    # Derived from the data rather than pinned to the 4B scale, so an 8B run
    # (whose AVG sits near 60) does not render as three bars in a corner.
    benchmark_lower = min([*benchmark_values, base_math_average]) - 1.0
    benchmark_upper = max(benchmark_values) + 1.2
    for ax, keys, tick_labels in zip(bar_axes, bar_panels, bar_labels):
        values = [benchmark_scores[key]["AVG"] for key in keys]
        positions = list(range(len(keys)))
        ax.bar(
            positions,
            values,
            color=[C_BAR_HIGHLIGHT] + [C_BAR_PLAIN] * (len(keys) - 1),
            width=0.66,
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
                fontsize=7,
                color=C_TEXT,
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
            fontsize=6.5,
            color=C_BASELINE,
            weight="bold",
            clip_on=False,
        )
        ax.set_xticks(positions, tick_labels, rotation=35, ha="right",
                      rotation_mode="anchor")
        ax.set_xlim(-0.55, len(keys) - 0.45)
        ax.set_ylim(benchmark_lower, benchmark_upper)
        first_tick = 2 * math.ceil(benchmark_lower / 2)
        ax.set_yticks(list(range(first_tick, int(benchmark_upper) + 1, 2)))
        ax.set_ylabel("Math average (%)")
        ax.set_title(f"Step {steps[-1]}", pad=6, color=C_TEXT)
        ax.grid(True, axis="y", linestyle="-", linewidth=0.5, color=C_GRID)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=6.5, colors=C_TEXT, pad=1)
        ax.tick_params(axis="y", colors=C_TEXT)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    # Both default to empty: a figure title that restates the caption, and a
    # footnote listing the benchmarks, are caption material.
    if args.suptitle:
        fig.suptitle(args.suptitle, fontsize=21, weight="bold", y=1.01)
    if args.footnote:
        fig.text(0.5, -0.015, args.footnote, ha="center", fontsize=10.5,
                 color="#444444")
    pdf_path = output_dir / "evolved_performance_ablation_480.pdf"
    png_path = output_dir / "evolved_performance_ablation_480.png"
    svg_path = output_dir / "evolved_performance_ablation_480.svg"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png_path, dpi=450, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"[EPB] wrote {pdf_path}")
    print(f"[EPB] wrote {svg_path}")
    print(f"[EPB] wrote {png_path}")
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
        "--benchmark-scores-section",
        choices=sorted(SECTION_HEADERS),
        default="final",
        help=(
            "which scores.md table to read: 'final' (post GPT-4o re-check) or "
            "'pre-gpt' (math_verify only). Use pre-gpt when the arms did not "
            "all get a working re-check"
        ),
    )
    parser.add_argument(
        "--base-math-average",
        type=float,
        default=47.45,
        help="base-model seven-benchmark average shown as the dashed baseline",
    )
    parser.add_argument("--max-step", type=int, default=128)
    parser.add_argument(
        "--arms",
        default=",".join(DEFAULT_RUNS),
        help=(
            "comma-separated arms to plot (must include 'full'); a panel is "
            "drawn only when all of its arms are selected"
        ),
    )
    parser.add_argument(
        "--full-label",
        help=f"legend label for the 'full' arm (default: {ARM_LABELS['full']!r})",
    )
    parser.add_argument(
        "--suptitle",
        default="",
        help="figure title; empty (the default) leaves it to the caption",
    )
    parser.add_argument(
        "--footnote",
        default="",
        help="line under the figure; empty (the default) leaves it to the caption",
    )
    parser.add_argument(
        "--size-label",
        default="",
        help="model scale appended to each panel title, e.g. 8B. Without it the "
        "4B and 8B figures are indistinguishable once the title is dropped",
    )
    for key, default_run in DEFAULT_RUNS.items():
        parser.add_argument(
            f"--{key}-run",
            help=f"run directory name under --run-root (default: {default_run})",
        )
    return parser


if __name__ == "__main__":
    plot(build_argparser().parse_args())
