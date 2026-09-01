#!/usr/bin/env python3
"""Build the current-run R-Q-Evolve versus R-Zero performance/efficiency figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


STEPS = (32, 64, 96, 128, 160, 192, 224, 256)
RQ_MATH = (55.45, 54.12, 53.25, 54.80, 55.35, 55.56, 56.23, 55.86)
RZERO_MATH = (53.18, 54.18, 54.79, 54.31, 52.48, 52.60, 52.39, 50.89)

# R-Q: W&B run tcxnymom, summary._runtime = 90,884.5957 seconds.
# R-Zero: audited start/end markers, 2026-08-13 20:34:06 to
# 2026-08-17 10:14:14 = 85 h 40 m 08 s.
RQ_HOURS = 90884.595719401 / 3600
RZERO_HOURS = 85 + 40 / 60 + 8 / 3600

RQ_COLOR = "#c6538c"
RZERO_COLOR = "#087bb5"


def trapezoid_mean(values: tuple[float, ...]) -> float:
    widths = [STEPS[index + 1] - STEPS[index] for index in range(len(STEPS) - 1)]
    area = sum(
        width * (values[index] + values[index + 1]) / 2
        for index, width in enumerate(widths)
    )
    return area / (STEPS[-1] - STEPS[0])


def format_2(value: float) -> str:
    """Format human-facing values with stable half-up-equivalent rounding."""
    return f"{value + 1e-9:.2f}"


def write_csv(output_dir: Path) -> Path:
    path = output_dir / "rq8b_rzero_performance_efficiency.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "step", "math_average", "wall_clock_hours"),
        )
        writer.writeheader()
        for method, scores, hours in (
            ("R-Q-Evolve", RQ_MATH, RQ_HOURS),
            ("R-Zero", RZERO_MATH, RZERO_HOURS),
        ):
            for step, score in zip(STEPS, scores):
                writer.writerow(
                    {
                        "method": method,
                        "step": step,
                        "math_average": score,
                        "wall_clock_hours": f"{hours:.6f}",
                    }
                )
    return path


def render(output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    import matplotlib.pyplot as plt

    fig, (trajectory_axis, time_axis) = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.1),
        gridspec_kw={"width_ratios": (1.55, 0.85), "wspace": 0.32},
    )

    for label, values, color, marker in (
        ("R-Q-Evolve", RQ_MATH, RQ_COLOR, "o"),
        ("R-Zero", RZERO_MATH, RZERO_COLOR, "s"),
    ):
        trajectory_axis.plot(
            STEPS,
            values,
            color=color,
            marker=marker,
            markersize=4.1,
            linewidth=1.9,
            label=label,
        )
    trajectory_axis.set_title("A  Mathematical benchmark", loc="left", weight="bold")
    trajectory_axis.set_xlabel("Global solver step")
    trajectory_axis.set_ylabel("7-task math average (%)")
    trajectory_axis.set_xticks(STEPS)
    trajectory_axis.set_xlim(26, 262)
    trajectory_axis.set_ylim(49.8, 57.0)
    trajectory_axis.grid(axis="both", color="#dbe3ec", linewidth=0.7, linestyle=":")
    trajectory_axis.legend(frameon=False, loc="lower left", fontsize=8.0)
    reported_index = STEPS.index(224)
    trajectory_axis.axvline(
        224,
        color="#9ca3af",
        linewidth=0.8,
        linestyle="--",
        alpha=0.70,
        zorder=0,
    )
    trajectory_axis.scatter(
        [224],
        [RQ_MATH[reported_index]],
        marker="*",
        s=82,
        facecolor="#fff7fb",
        edgecolor=RQ_COLOR,
        linewidth=1.2,
        zorder=5,
    )
    bars = time_axis.bar(
        ("R-Q-Evolve", "R-Zero"),
        (RQ_HOURS, RZERO_HOURS),
        color=(RQ_COLOR, RZERO_COLOR),
        width=0.60,
    )
    time_axis.set_title("B  End-to-end wall-clock", loc="left", weight="bold")
    time_axis.set_ylabel("Hours")
    time_axis.set_ylim(0, 96)
    time_axis.grid(axis="y", color="#dbe3ec", linewidth=0.7, linestyle=":")
    time_axis.set_axisbelow(True)
    for bar, hours in zip(bars, (RQ_HOURS, RZERO_HOURS)):
        time_axis.text(
            bar.get_x() + bar.get_width() / 2,
            hours + 2.2,
            f"{hours:.2f} h",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )
    time_axis.text(
        0.50,
        0.73,
        f"{RZERO_HOURS / RQ_HOURS:.2f}× faster\n"
        f"{100 * (1 - RQ_HOURS / RZERO_HOURS):.1f}% less wall time",
        transform=time_axis.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        color=RQ_COLOR,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 1.5},
    )

    for axis in (trajectory_axis, time_axis):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    stem = output_dir / "rq8b_rzero_performance_efficiency"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{stem}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{stem}.png", dpi=450, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data1/yhoon113/Private & Shared 5/paper_ready_ablation"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(args.output_dir)
    stem = render(args.output_dir)
    print(f"data: {csv_path}")
    print(f"figure: {stem}.pdf/.svg/.png")


if __name__ == "__main__":
    main()
