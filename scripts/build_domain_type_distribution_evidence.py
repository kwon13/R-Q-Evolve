#!/usr/bin/env python3
"""Build a paper-ready, Luna-audited domain-by-problem-type comparison.

Both methods use the exact same GPT-5.6-luna domain prompt and the same frozen
statement-only problem-type rules. R-Q-Evolve inputs are non-overlapping local
windows rendered from the recorded program/seed pairs; R-Zero inputs are the
corresponding round 1/4/8 full pools. Every panel is normalized by its own
number of jointly mapped problems, so the figure compares distributional
concentration rather than raw generation volume.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

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
DOMAIN_LABELS = (
    "Algebra",
    "Geometry",
    "Number Theory",
    "Discrete Math",
    "Applied Math",
    "Calculus",
    "Precalculus",
)
TYPE_LABELS = ("Decision", "Search", "Counting", "Optimization", "Function")


@dataclass(frozen=True)
class Panel:
    method: str
    stage: str
    interval: str
    counts: dict[tuple[str, str], int]
    sources: tuple[Path, ...]


def read_counts(path: Path) -> dict[tuple[str, str], int]:
    rows: dict[tuple[str, str], int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(row["domain"], row["problem_type"])] = int(row["count"])
    expected = {(domain, problem_type) for domain in DOMAINS for problem_type in PROBLEM_TYPES}
    missing = expected - rows.keys()
    if missing:
        raise ValueError(f"{path} is missing cells: {sorted(missing)}")
    return rows


def rq_luna_path(window: str) -> Path:
    return (
        ROOT
        / "analysis"
        / "rq_evolve_8b_domain_type_35cell_8gpu"
        / "luna_domain_type"
        / window
        / "domain_type_counts.csv"
    )


def rzero_path(round_index: int) -> Path:
    pattern = str(
        ROOT
        / "analysis"
        / "rzero_domain_type"
        / f"*_round{round_index}_full_unique*"
        / "domain_type_counts.csv"
    )
    matches = [Path(path) for path in glob.glob(pattern)]
    if len(matches) != 1:
        raise ValueError(f"Expected one R-Zero round-{round_index} grid, found {matches}")
    return matches[0]


def build_panels() -> list[list[Panel]]:
    early_path = rq_luna_path("early_steps_000_032")
    middle_path = rq_luna_path("middle_steps_097_128")
    final_path = rq_luna_path("final_steps_225_255")
    rq = [
        Panel(
            "R-Q-Evolve",
            "Early",
            "steps 0–32",
            read_counts(early_path),
            (early_path,),
        ),
        Panel(
            "R-Q-Evolve",
            "Middle",
            "steps 97–128",
            read_counts(middle_path),
            (middle_path,),
        ),
        Panel(
            "R-Q-Evolve",
            "Final",
            "steps 225–255",
            read_counts(final_path),
            (final_path,),
        ),
    ]
    rz = []
    for stage, round_index in zip(("Early", "Middle", "Final"), (1, 4, 8)):
        path = rzero_path(round_index)
        rz.append(
            Panel("R-Zero", stage, f"round {round_index}", read_counts(path), (path,))
        )
    return [rq, rz]


def statistics(panel: Panel) -> dict[str, float | int | str]:
    values = list(panel.counts.values())
    total = sum(values)
    if total <= 0:
        raise ValueError(f"Empty panel: {panel}")
    probabilities = [value / total for value in values if value > 0]
    entropy = -sum(value * math.log(value) for value in probabilities)
    ranked = sorted(values, reverse=True)
    top_cell, top_count = max(panel.counts.items(), key=lambda item: item[1])
    return {
        "method": panel.method,
        "stage": panel.stage,
        "interval": panel.interval,
        "classified_n": total,
        "occupied_cells": sum(value > 0 for value in values),
        "shannon_entropy_nats": entropy,
        "normalized_entropy": entropy / math.log(len(DOMAINS) * len(PROBLEM_TYPES)),
        "effective_cells": math.exp(entropy),
        "top1_share": top_count / total,
        "top5_share": sum(ranked[:5]) / total,
        "top_cell": f"{top_cell[0]}/{top_cell[1]}",
        "source_paths": " | ".join(str(path.resolve()) for path in panel.sources),
    }


def write_summary(panels: list[list[Panel]], output_dir: Path) -> Path:
    rows = [statistics(panel) for method_panels in panels for panel in method_panels]
    path = output_dir / "domain_type_distribution_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def render(panels: list[list[Panel]], output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import numpy as np

    shares: list[list[np.ndarray]] = []
    stats: list[list[dict[str, float | int | str]]] = []
    for method_panels in panels:
        share_row = []
        stat_row = []
        for panel in method_panels:
            total = sum(panel.counts.values())
            share_row.append(
                np.asarray(
                    [
                        [
                            panel.counts[(domain, problem_type)] / total
                            for problem_type in PROBLEM_TYPES
                        ]
                        for domain in DOMAINS
                    ],
                    dtype=float,
                )
            )
            stat_row.append(statistics(panel))
        shares.append(share_row)
        stats.append(stat_row)

    fig = plt.figure(figsize=(13.8, 7.3), facecolor="white")
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=(1, 1, 1, 0.045),
        left=0.135,
        right=0.965,
        bottom=0.09,
        top=0.91,
        wspace=0.08,
        hspace=0.26,
    )
    axes = [[fig.add_subplot(grid[row, col]) for col in range(3)] for row in range(2)]
    colorbar_axis = fig.add_subplot(grid[:, 3])
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#f3f4f6")
    norm = colors.LogNorm(vmin=1e-4, vmax=0.8)
    image = None

    for row, method_panels in enumerate(panels):
        for col, panel in enumerate(method_panels):
            ax = axes[row][col]
            share = shares[row][col]
            stat = stats[row][col]
            image = ax.imshow(
                np.ma.masked_less_equal(share, 0),
                cmap=cmap,
                norm=norm,
                aspect="auto",
                interpolation="nearest",
            )
            for domain_index in range(len(DOMAINS)):
                for type_index in range(len(PROBLEM_TYPES)):
                    value = share[domain_index, type_index]
                    if value == 0:
                        label = "–"
                        color = "#9ca3af"
                    elif value < 0.001:
                        label = "<0.1"
                        color = "#111827"
                    else:
                        label = f"{100 * value:.1f}"
                        color = "white" if norm(value) > 0.58 else "#111827"
                    ax.text(
                        type_index,
                        domain_index,
                        label,
                        ha="center",
                        va="center",
                        fontsize=7.4,
                        fontweight="semibold" if value >= 0.05 else "normal",
                        color=color,
                    )

            ax.set_title(
                f"{panel.stage} · {panel.interval}\n"
                f"N={stat['classified_n']:,}  ·  "
                f"effective cells={stat['effective_cells']:.1f}/35  ·  "
                f"top-1={100 * stat['top1_share']:.1f}%",
                fontsize=9.0,
                pad=7,
            )
            ax.set_xticks(range(len(PROBLEM_TYPES)), TYPE_LABELS)
            ax.set_yticks(range(len(DOMAINS)), DOMAIN_LABELS)
            ax.set_xticks(np.arange(-0.5, len(PROBLEM_TYPES), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(DOMAINS), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.0)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.tick_params(axis="both", which="major", length=0)
            if row == 0:
                ax.tick_params(labelbottom=False)
            else:
                ax.tick_params(axis="x", labelrotation=22, labelsize=7.6)
            if col:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(panel.method, fontsize=11.5, fontweight="bold", labelpad=23)
            for spine in ax.spines.values():
                spine.set_visible(False)

    assert image is not None
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Share within panel (%)", fontsize=9.0)
    ticks = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{100 * tick:g}" for tick in ticks])
    colorbar.ax.tick_params(labelsize=7.5)

    stem = output_dir / "domain_type_distribution_normalized_8b"
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
    panels = build_panels()
    summary = write_summary(panels, args.output_dir)
    stem = render(panels, args.output_dir)
    print(f"summary: {summary}")
    print(f"figure: {stem}.pdf/.svg/.png")


if __name__ == "__main__":
    main()
