#!/usr/bin/env python3
"""Plot a 7 x 5 Omni-MATH Domain x Computational Problem Type map."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DOMAIN_ORDER = (
    "Algebra",
    "Geometry",
    "Number Theory",
    "Discrete Mathematics",
    "Applied Mathematics",
    "Calculus",
    "Precalculus",
)
TYPE_ORDER = ("decision", "search", "counting", "optimization", "function")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--view",
        choices=("expanded", "single_domain"),
        default="expanded",
        help="Expanded counts include every top-level membership of multi-domain rows.",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import numpy as np

    count_key = f"{args.view}_count"
    values: dict[tuple[str, str], int] = {}
    with args.counts.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values[(row["domain"], row["problem_type"])] = int(row[count_key])

    grid = np.array(
        [
            [values.get((domain, problem_type), 0) for problem_type in TYPE_ORDER]
            for domain in DOMAIN_ORDER
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(11.6, 6.8))
    cmap = plt.cm.viridis.copy()
    cmap.set_under("#e7e7eb")
    image = ax.imshow(
        grid,
        cmap=cmap,
        norm=colors.LogNorm(vmin=1, vmax=max(grid.max(), 1)),
        aspect="auto",
    )

    for row_index, domain in enumerate(DOMAIN_ORDER):
        for col_index, problem_type in enumerate(TYPE_ORDER):
            count = int(grid[row_index, col_index])
            normalized = image.norm(max(count, 1))
            text_color = "white" if count > 0 and normalized < 0.72 else "#111318"
            ax.text(
                col_index,
                row_index,
                str(count),
                ha="center",
                va="center",
                color=text_color,
                fontsize=13,
                fontweight="bold",
            )

    ax.set_xticks(range(len(TYPE_ORDER)), [label.replace("_", " ") for label in TYPE_ORDER])
    ax.set_yticks(range(len(DOMAIN_ORDER)), DOMAIN_ORDER)
    ax.tick_params(axis="x", labelsize=11, pad=8)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_xlabel("Computational problem type", fontsize=12, labelpad=12)
    ax.set_ylabel("Omni-MATH top-level domain", fontsize=12, labelpad=12)
    ax.set_xticks(np.arange(-0.5, len(TYPE_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DOMAIN_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    view_text = (
        "expanded multi-domain memberships"
        if args.view == "expanded"
        else "exactly-one-domain problems only"
    )
    ax.set_title(
        "Omni-MATH: Domain × Computational Problem Type\n"
        f"4,428 problems · {view_text} · cell text = pilot-labeled count",
        fontsize=15,
        pad=16,
    )
    fig.text(
        0.5,
        0.018,
        "All 35 cells remain valid MAP coordinates; counts show Omni-MATH "
        "frequency only · statement-only abstentions excluded",
        ha="center",
        fontsize=10,
        color="#454954",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Problem count (log scale)", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
