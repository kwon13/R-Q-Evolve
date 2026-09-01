#!/usr/bin/env python3
"""Create a paper-ready 2 x 3 R-Q Evolve versus R-Zero MAP panel."""

from __future__ import annotations

import argparse
import csv
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
    "Discrete Mathematics",
    "Applied Mathematics",
    "Calculus",
    "Precalculus",
)
TYPE_LABELS = ("Decision", "Search", "Counting", "Optimization", "Function")


def read_grid(path: Path):
    import numpy as np

    counts: dict[tuple[str, str], int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            counts[(row["domain"], row["problem_type"])] = int(row["count"])
    expected = {(domain, problem_type) for domain in DOMAINS for problem_type in PROBLEM_TYPES}
    missing = expected - counts.keys()
    if missing:
        raise ValueError(f"{path} is missing cells: {sorted(missing)}")
    return np.asarray(
        [[counts[(domain, problem_type)] for problem_type in PROBLEM_TYPES] for domain in DOMAINS],
        dtype=float,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis" / "rq8b_vs_rzero_domain_type" / "early_middle_final",
    )
    args = parser.parse_args()

    rq_base = (
        ROOT
        / "analysis"
        / "rq_evolve_8b_domain_type_35cell_8gpu"
        / "problem_domain_type"
    )
    rz_base = ROOT / "analysis" / "rzero_domain_type"
    panels = (
        (
            "R-Q Evolve",
            (
                ("0–32", rq_base / "rounds_000_032" / "domain_type_counts.csv"),
                ("0–128", rq_base / "rounds_000_128" / "domain_type_counts.csv"),
                ("0–255", rq_base / "rounds_000_255" / "domain_type_counts.csv"),
            ),
        ),
        (
            "R-Zero",
            (
                (
                    "Round 1",
                    rz_base
                    / "gpt-5.6-luna__none__0434b5c0715c_round1_full_unique4079"
                    / "domain_type_counts.csv",
                ),
                (
                    "Round 4",
                    rz_base
                    / "gpt-5.6-luna__none__0434b5c0715c_round4_full_unique5799"
                    / "domain_type_counts.csv",
                ),
                (
                    "Round 8",
                    rz_base
                    / "gpt-5.6-luna__none__0434b5c0715c_round8_full_unique6242"
                    / "domain_type_counts.csv",
                ),
            ),
        ),
    )

    grids = [[read_grid(path) for _, path in row] for _, row in panels]

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import numpy as np

    vmax = max(float(grid.max()) for row in grids for grid in row)
    norm = colors.LogNorm(vmin=1, vmax=vmax)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#f2f3f5")

    fig = plt.figure(figsize=(14.2, 7.5))
    grid_spec = fig.add_gridspec(
        2,
        4,
        width_ratios=(1, 1, 1, 0.045),
        left=0.13,
        right=0.965,
        bottom=0.09,
        top=0.95,
        wspace=0.08,
        hspace=0.13,
    )
    axes = np.empty((2, 3), dtype=object)
    for row_index in range(2):
        for col_index in range(3):
            shared = axes[0, 0] if (row_index, col_index) != (0, 0) else None
            axes[row_index, col_index] = fig.add_subplot(
                grid_spec[row_index, col_index],
                sharex=shared,
                sharey=shared,
            )
    colorbar_axis = fig.add_subplot(grid_spec[:, 3])
    images = []
    for row_index, ((method, row_panels), row_grids) in enumerate(zip(panels, grids)):
        for col_index, ((stage_label, _), grid) in enumerate(zip(row_panels, row_grids)):
            ax = axes[row_index, col_index]
            image = ax.imshow(
                np.ma.masked_less_equal(grid, 0),
                cmap=cmap,
                norm=norm,
                aspect="auto",
                interpolation="nearest",
            )
            images.append(image)
            for domain_index in range(len(DOMAINS)):
                for type_index in range(len(PROBLEM_TYPES)):
                    count = int(grid[domain_index, type_index])
                    if count == 0:
                        text_color = "#1a1d23"
                    else:
                        text_color = "white" if norm(count) < 0.58 else "#111318"
                    ax.text(
                        type_index,
                        domain_index,
                        f"{count}",
                        ha="center",
                        va="center",
                        fontsize=8.7,
                        fontweight="semibold",
                        color=text_color,
                    )

            ax.set_title(stage_label, fontsize=10.5, pad=7)
            ax.set_xticks(range(len(PROBLEM_TYPES)), TYPE_LABELS)
            ax.set_yticks(range(len(DOMAINS)), DOMAIN_LABELS)
            ax.set_xticks(np.arange(-0.5, len(PROBLEM_TYPES), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(DOMAINS), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.15)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.tick_params(axis="both", which="major", length=2.5, width=0.7)
            if row_index == 0:
                ax.tick_params(labelbottom=False)
            else:
                ax.tick_params(axis="x", labelsize=8.5)
            if col_index != 0:
                ax.tick_params(labelleft=False)
            else:
                ax.tick_params(axis="y", labelsize=8.7)
                ax.set_ylabel(method, fontsize=11.5, fontweight="semibold", labelpad=28)

    colorbar = fig.colorbar(images[-1], cax=colorbar_axis)
    colorbar.set_label("Count", fontsize=9.5)
    colorbar.ax.tick_params(labelsize=8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "rq8b_rzero_domain_type_early_middle_final"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"outputs: {stem}.pdf, {stem}.png")


if __name__ == "__main__":
    main()
