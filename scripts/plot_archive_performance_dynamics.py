#!/usr/bin/env python3
"""Plot breadth-to-within-cell archive dynamics and downstream performance.

The archive update denominator is mutation candidates with ``status=inserted``
in ``evolution_log.jsonl``.  Empty-cell insertions have no incumbent in the
logged final-admission decision; occupied-cell updates have an incumbent and
therefore replace the current champion.  Champion re-evaluations are excluded
from both categories.

The math trajectories are the fixed-checkpoint evaluations reported for the
domain_type_35cell 8-GPU runs.  Archive activity and occupancy are always read
from the corresponding raw run directories.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


CHECKPOINT_STEPS = np.array([32, 64, 96, 128, 160, 192, 224, 256])
SNAPSHOT_ITERATIONS = np.array([32, 64, 96, 128, 160, 192, 224, 255])

# Fixed-checkpoint seven-task math averages.  Source table:
# /data1/yhoon113/Private & Shared 5/결과 3ce4cec6dd2780a9b855f76784ee9631.md
MATH_AVG = {
    "8B": np.array([55.45, 54.12, 53.25, 54.80, 55.35, 55.56, 56.23, 55.86]),
    "4B": np.array([51.34, 49.96, 50.15, 51.41, 51.39, 50.78, 50.19, 52.29]),
}
BASE_MATH_AVG = {"8B": 49.74, "4B": 47.05}

# Color-blind-safe, print-friendly encodings.
C_BREADTH = "#0072B2"
C_WITHIN = "#E69F00"
C_PERF = "#3B4CC0"
C_ARCHIVE = "#00897B"
C_GRID = "#D9DDE3"
C_TEXT = "#202124"
C_SECONDARY = "#6B7280"


@dataclass(frozen=True)
class WindowStats:
    end_step: int
    attempted: int
    accepted: int
    breadth: int
    within_cell: int

    @property
    def breadth_share(self) -> float:
        return self.breadth / self.accepted if self.accepted else 0.0

    @property
    def within_share(self) -> float:
        return self.within_cell / self.accepted if self.accepted else 0.0

    @property
    def breadth_intensity(self) -> float:
        return self.breadth / self.attempted if self.attempted else 0.0

    @property
    def within_intensity(self) -> float:
        return self.within_cell / self.attempted if self.attempted else 0.0


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "analysis" / "archive_performance_dynamics",
    )
    return parser.parse_args()


def run_dir(repo_root: Path, model: str) -> Path:
    size = model.lower()
    return (
        repo_root
        / "rq_output"
        / f"rq_evolve_{size}_domain_type_35cell_8gpu"
        / "rq_output"
    )


def read_windows(path: Path) -> list[WindowStats]:
    buckets = [dict(attempted=0, accepted=0, breadth=0, within_cell=0) for _ in range(8)]
    with (path / "evolution_log.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            iteration = int(record["iteration"])
            bucket_index = min((iteration - 1) // 32, 7)
            bucket = buckets[bucket_index]
            bucket["attempted"] += int(record["metrics"].get("attempted", 0))

            inserted_in_record = 0
            for report in record.get("reports", []):
                if report.get("status") != "inserted":
                    continue
                inserted_in_record += 1
                bucket["accepted"] += 1
                decision = report.get("archive_decision") or {}
                if not decision.get("accepted", False):
                    raise ValueError(
                        f"inserted report lacks accepted final admission at iteration {iteration}"
                    )
                if decision.get("incumbent_program_id") is None:
                    bucket["breadth"] += 1
                else:
                    bucket["within_cell"] += 1

            metric_inserted = int(record["metrics"].get("inserted", 0))
            if inserted_in_record != metric_inserted:
                raise ValueError(
                    f"insert count mismatch at iteration {iteration}: "
                    f"reports={inserted_in_record}, metrics={metric_inserted}"
                )

    windows = []
    for end_step, bucket in zip(CHECKPOINT_STEPS, buckets, strict=True):
        if bucket["breadth"] + bucket["within_cell"] != bucket["accepted"]:
            raise ValueError(f"archive update partition failed for window ending {end_step}")
        windows.append(WindowStats(end_step=int(end_step), **bucket))
    return windows


def read_occupancy(path: Path) -> tuple[int, np.ndarray]:
    """Return pre-iteration seed occupancy and checkpoint live occupancy."""
    with (path / "evolution_log.jsonl").open(encoding="utf-8") as handle:
        iteration_one = json.loads(next(handle))
    iteration_one_children = {
        report.get("child_id")
        for report in iteration_one.get("reports", [])
        if report.get("status") == "inserted"
    }
    with (path / "archive_iter1.json").open(encoding="utf-8") as handle:
        archive_one = json.load(handle)
    initial_cells = {
        (int(champion["niche_domain"]), int(champion["niche_problem_type"]))
        for champion in archive_one["champions"]
        if champion["program_id"] not in iteration_one_children
    }

    occupied = []
    for iteration in SNAPSHOT_ITERATIONS:
        with (path / f"archive_iter{iteration}.json").open(encoding="utf-8") as handle:
            archive = json.load(handle)
        occupied.append(int(archive["meta"]["stats"]["num_champions"]))
    return len(initial_cells), np.asarray(occupied, dtype=int)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "axes.edgecolor": C_TEXT,
            "axes.labelcolor": C_TEXT,
            "xtick.color": C_TEXT,
            "ytick.color": C_TEXT,
            "text.color": C_TEXT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "grid.color": C_GRID,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_panel_label(ax: plt.Axes, label: str, *, y: float = 1.08) -> None:
    ax.text(
        -0.13,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10.2,
        fontweight="bold",
        va="top",
        ha="left",
    )


def panel_composition(
    ax: plt.Axes, stats: dict[str, list[WindowStats]]
) -> None:
    x = np.arange(len(CHECKPOINT_STEPS), dtype=float)
    width = 0.28
    offsets = {"8B": -0.17, "4B": 0.17}
    hatches = {"8B": None, "4B": "////"}
    alphas = {"8B": 1.0, "4B": 0.68}

    for model in ("8B", "4B"):
        breadth = np.array([row.breadth_share for row in stats[model]]) * 100
        within = np.array([row.within_share for row in stats[model]]) * 100
        xpos = x + offsets[model]
        ax.bar(
            xpos,
            breadth,
            width=width,
            color=C_BREADTH,
            edgecolor="white" if model == "8B" else C_TEXT,
            linewidth=0.45,
            alpha=alphas[model],
            hatch=hatches[model],
            zorder=3,
        )
        ax.bar(
            xpos,
            within,
            width=width,
            bottom=breadth,
            color=C_WITHIN,
            edgecolor="white" if model == "8B" else C_TEXT,
            linewidth=0.45,
            alpha=alphas[model],
            hatch=hatches[model],
            zorder=3,
        )
        for xp, value in zip(xpos, breadth, strict=True):
            if value >= 5.0:
                label_y = value * (0.62 if model == "8B" else 0.36)
                ax.text(
                    xp,
                    max(label_y, 2.2),
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    color="white" if model == "8B" else C_TEXT,
                    fontsize=6.2,
                    fontweight="bold" if model == "8B" else "normal",
                    zorder=5,
                )

    ax.set_title("Archive update composition", loc="left", y=1.19, pad=3)
    ax.set_ylabel("Accepted archive updates (%)")
    ax.set_xlabel("Training step (32-step window end)")
    ax.set_xticks(x, CHECKPOINT_STEPS)
    ax.set_ylim(0, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y")
    event_legend = ax.legend(
        handles=[
            Patch(facecolor=C_BREADTH, edgecolor="none", label="Breadth: empty-cell insertion"),
            Patch(facecolor=C_WITHIN, edgecolor="none", label="Within-cell refinement"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        frameon=False,
        handlelength=1.2,
        borderaxespad=0,
    )
    ax.add_artist(event_legend)
    ax.legend(
        handles=[
            Patch(facecolor="#C7CBD1", edgecolor="none", label="8B"),
            Patch(facecolor="white", edgecolor=C_TEXT, hatch="////", label="4B"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        frameon=False,
        ncol=2,
        handlelength=1.2,
        columnspacing=0.8,
        borderaxespad=0,
    )
    add_panel_label(ax, "(a)", y=1.27)


def panel_intensity(ax: plt.Axes, stats: dict[str, list[WindowStats]]) -> None:
    styles = {
        "8B": dict(ls="-", marker="o", alpha=1.0, lw=1.8, ms=4.0),
        "4B": dict(ls="--", marker="s", alpha=0.62, lw=1.55, ms=3.6),
    }
    for model in ("8B", "4B"):
        breadth = np.array([row.breadth_intensity for row in stats[model]]) * 100
        within = np.array([row.within_intensity for row in stats[model]]) * 100
        ax.plot(
            CHECKPOINT_STEPS,
            breadth,
            color=C_BREADTH,
            markerfacecolor="white" if model == "4B" else C_BREADTH,
            markeredgecolor=C_BREADTH,
            **styles[model],
        )
        ax.plot(
            CHECKPOINT_STEPS,
            within,
            color=C_WITHIN,
            markerfacecolor="white" if model == "4B" else C_WITHIN,
            markeredgecolor=C_WITHIN,
            **styles[model],
        )

    ax.set_title("Archive activity intensity", loc="left", y=1.19, pad=3)
    ax.set_ylabel("Insertions / attempted mutations (%)")
    ax.set_xlabel("Training step (32-step window end)")
    ax.set_xticks(CHECKPOINT_STEPS)
    ax.set_xlim(25, 263)
    ax.set_ylim(-0.08, 2.65)
    ax.set_yticks(np.arange(0, 2.6, 0.5))
    ax.grid(axis="y")

    event_legend = ax.legend(
        handles=[
            Line2D([0], [0], color=C_BREADTH, lw=2, label="Breadth intensity"),
            Line2D([0], [0], color=C_WITHIN, lw=2, label="Within-cell intensity"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        frameon=False,
        handlelength=1.7,
        borderaxespad=0.25,
    )
    ax.add_artist(event_legend)
    ax.legend(
        handles=[
            Line2D([0], [0], color=C_TEXT, lw=1.7, marker="o", ms=3.8, label="8B"),
            Line2D([0], [0], color=C_SECONDARY, lw=1.5, ls="--", marker="s", ms=3.5, label="4B"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        frameon=False,
        ncol=2,
        handlelength=1.7,
        columnspacing=0.8,
        borderaxespad=0.25,
    )
    add_panel_label(ax, "(b)", y=1.27)


def add_phase_shading(ax: plt.Axes) -> None:
    phases = [
        (0, 32, "Initial\nacquisition", "#F2F4F7"),
        (32, 96, "Breadth-to-depth\ntransition", "#EAF2F8"),
        (96, 256, "Within-cell refinement", "#F5F1E8"),
    ]
    for lo, hi, label, color in phases:
        ax.axvspan(lo, hi, color=color, alpha=0.82, lw=0, zorder=0)
        ax.text(
            (lo + hi) / 2,
            0.975,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.0,
            color=C_SECONDARY,
            linespacing=1.05,
        )
    for boundary in (32, 96):
        ax.axvline(boundary, color="#AEB4BC", lw=0.7, ls=(0, (2, 2)), zorder=1)


def panel_dynamics(
    ax: plt.Axes,
    initial_occupancy: dict[str, int],
    occupancy: dict[str, np.ndarray],
) -> None:
    add_phase_shading(ax)
    ax_right = ax.twinx()
    ax_right.spines["right"].set_visible(True)
    ax_right.spines["right"].set_color(C_ARCHIVE)
    ax_right.tick_params(axis="y", colors=C_ARCHIVE)
    ax_right.yaxis.label.set_color(C_ARCHIVE)

    all_steps = np.concatenate(([0], CHECKPOINT_STEPS))
    for model, line_style, marker, alpha, width in (
        ("8B", "-", "o", 1.0, 2.15),
        ("4B", "--", "s", 0.48, 1.55),
    ):
        math = np.concatenate(([BASE_MATH_AVG[model]], MATH_AVG[model]))
        occupied = np.concatenate(([initial_occupancy[model]], occupancy[model]))
        ax.plot(
            all_steps,
            math,
            color=C_PERF,
            ls=line_style,
            marker=marker,
            ms=4.3 if model == "8B" else 3.6,
            lw=width,
            alpha=alpha,
            markerfacecolor=C_PERF if model == "8B" else "white",
            markeredgecolor=C_PERF,
            zorder=5,
        )
        ax_right.plot(
            all_steps,
            occupied,
            color=C_ARCHIVE,
            ls=line_style,
            marker=marker,
            ms=4.0 if model == "8B" else 3.4,
            lw=1.9 if model == "8B" else 1.45,
            alpha=alpha,
            markerfacecolor=C_ARCHIVE if model == "8B" else "white",
            markeredgecolor=C_ARCHIVE,
            zorder=4,
        )

    # Descriptive annotations only; no causal language.
    ax.text(37, 55.78, "8B: 55.45", color=C_PERF, fontsize=7.0)
    ax.annotate(
        "53.25",
        xy=(96, MATH_AVG["8B"][2]),
        xytext=(105, 52.25),
        color=C_PERF,
        fontsize=7.0,
        arrowprops=dict(arrowstyle="-", color=C_PERF, lw=0.7),
    )
    ax.annotate(
        "56.23",
        xy=(224, MATH_AVG["8B"][6]),
        xytext=(205, 56.75),
        color=C_PERF,
        fontsize=7.0,
        arrowprops=dict(arrowstyle="-", color=C_PERF, lw=0.7),
    )

    ax.set_title("Archive–performance dynamics", loc="left", pad=9)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Math benchmark AVG", color=C_PERF)
    ax.tick_params(axis="y", colors=C_PERF)
    ax.spines["left"].set_color(C_PERF)
    ax.set_xlim(0, 256)
    ax.set_xticks(all_steps, ["Base", *[str(step) for step in CHECKPOINT_STEPS]])
    ax.set_ylim(45.7, 57.7)
    ax.set_yticks(np.arange(46, 58, 2))
    ax.grid(axis="y")

    ax_right.set_ylabel("Occupied archive cells (of 35)")
    ax_right.set_ylim(4.5, 30.5)
    ax_right.set_yticks([5, 10, 15, 20, 25, 30])

    ax.legend(
        handles=[
            Line2D([0], [0], color=C_PERF, lw=2.0, label="Math AVG"),
            Line2D([0], [0], color=C_ARCHIVE, lw=2.0, label="Occupied cells"),
            Line2D([0], [0], color=C_TEXT, lw=1.8, marker="o", ms=3.8, label="8B"),
            Line2D([0], [0], color=C_SECONDARY, lw=1.5, ls="--", marker="s", ms=3.5, label="4B"),
        ],
        loc="lower right",
        frameon=False,
        ncol=2,
        handlelength=2.0,
        columnspacing=1.0,
        borderaxespad=0.3,
    )
    add_panel_label(ax, "(c)")


def write_data(
    output_path: Path,
    stats: dict[str, list[WindowStats]],
    occupancy: dict[str, np.ndarray],
) -> None:
    fields = [
        "model",
        "window_start",
        "window_end",
        "attempted_mutations",
        "accepted_mutations",
        "breadth_insertions",
        "within_cell_replacements",
        "breadth_share",
        "within_cell_share",
        "breadth_intensity",
        "within_cell_intensity",
        "math_average",
        "occupied_cells",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in ("8B", "4B"):
            for index, row in enumerate(stats[model]):
                writer.writerow(
                    {
                        "model": model,
                        "window_start": index * 32 + 1,
                        "window_end": row.end_step if row.end_step < 256 else 255,
                        "attempted_mutations": row.attempted,
                        "accepted_mutations": row.accepted,
                        "breadth_insertions": row.breadth,
                        "within_cell_replacements": row.within_cell,
                        "breadth_share": f"{row.breadth_share:.8f}",
                        "within_cell_share": f"{row.within_share:.8f}",
                        "breadth_intensity": f"{row.breadth_intensity:.8f}",
                        "within_cell_intensity": f"{row.within_intensity:.8f}",
                        "math_average": f"{MATH_AVG[model][index]:.2f}",
                        "occupied_cells": int(occupancy[model][index]),
                    }
                )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    stats: dict[str, list[WindowStats]] = {}
    occupancy: dict[str, np.ndarray] = {}
    initial_occupancy: dict[str, int] = {}
    for model in ("8B", "4B"):
        path = run_dir(args.repo_root, model)
        stats[model] = read_windows(path)
        initial_occupancy[model], occupancy[model] = read_occupancy(path)

    figure = plt.figure(figsize=(7.35, 6.85))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.08],
        hspace=0.58,
        wspace=0.31,
        left=0.085,
        right=0.925,
        top=0.91,
        bottom=0.085,
    )
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, :])

    panel_composition(ax_a, stats)
    panel_intensity(ax_b, stats)
    panel_dynamics(ax_c, initial_occupancy, occupancy)

    stem = args.output_dir / "breadth_to_within_cell_archive_performance"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.035)
    figure.savefig(
        stem.with_suffix(".png"),
        dpi=500,
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(figure)
    write_data(stem.with_suffix(".csv"), stats, occupancy)

    print(f"Wrote {stem}.{{pdf,svg,png,csv}}")


if __name__ == "__main__":
    main()
