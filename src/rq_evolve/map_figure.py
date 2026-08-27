"""Render the live DOMAIN x PROBLEM_TYPE archive as one figure, for wandb.

The scalar coverage number says how many of the 35 cells are occupied. It
cannot say WHICH, and that is the thing the two axes were introduced to make
visible: "we only ever work in two domains" and "we only ever generate two
output contracts" produce the same coverage but describe different failures.
Every cell is drawn; there is deliberately no supported-cell mask.

Kept out of ``archive.py`` so importing the archive never pulls in matplotlib,
and every entry point here degrades to None rather than raising: a logging
backend that cannot draw must not take a training run down.
"""

from __future__ import annotations

from typing import Any

from .concepts import DOMAINS, PROBLEM_TYPES

# The empty-cell colour. Deliberately not white: an unfilled niche and a niche
# holding an R_Q of exactly 0 are different states and must not look alike.
_EMPTY = "#1b1b1f"


def _figure_backend():
    """matplotlib with a headless backend, or None if it is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def render_archive_figure(
    archive: Any,
    *,
    iteration: int,
    new_cells: set[tuple[int, int]] | None = None,
    stats: dict | None = None,
):
    """One heatmap: rows are DOMAINs, columns are PROBLEM_TYPEs.

    Cell colour is the champion's R_Q; the annotation carries R_Q over s_hat, so
    a cell that is bright because the problem is hard-but-solvable is
    distinguishable from one that is bright because u_score is high. Cells
    filled for the first time this iteration get a red border, which turns the
    per-iteration image sequence into a record of where the search actually
    moved.

    Returns a matplotlib Figure the caller must close, or None when matplotlib
    is missing or the archive cannot be read.
    """
    plt = _figure_backend()
    if plt is None:
        return None
    try:
        import numpy as np
    except Exception:
        return None

    try:
        champions = list(archive.champions())
    except Exception:
        return None

    grid = np.full((len(DOMAINS), len(PROBLEM_TYPES)), np.nan)
    s_hat = np.full((len(DOMAINS), len(PROBLEM_TYPES)), np.nan)
    for champion in champions:
        cell = archive.program_to_cell(champion)
        if cell is None:
            continue
        domain_bin, type_bin = cell
        rq = float(getattr(champion, "rq_score", 0.0) or 0.0)
        if np.isnan(grid[domain_bin, type_bin]) or rq > grid[domain_bin, type_bin]:
            grid[domain_bin, type_bin] = rq
            s_hat[domain_bin, type_bin] = float(getattr(champion, "s_hat", 0.0) or 0.0)

    finite = grid[~np.isnan(grid)]
    vmax = float(finite.max()) if finite.size and finite.max() > 0 else 1.0

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(_EMPTY)
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)

    new_cells = new_cells or set()
    for domain_bin in range(len(DOMAINS)):
        for type_bin in range(len(PROBLEM_TYPES)):
            value = grid[domain_bin, type_bin]
            if not np.isnan(value):
                ax.text(
                    type_bin,
                    domain_bin,
                    f"{value:.2f}\n{s_hat[domain_bin, type_bin]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value < vmax * 0.6 else "black",
                )
            if (domain_bin, type_bin) in new_cells:
                ax.add_patch(
                    plt.Rectangle(
                        (type_bin - 0.5, domain_bin - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#e8453c",
                        lw=2.2,
                    )
                )

    ax.set_xticks(range(len(PROBLEM_TYPES)))
    ax.set_xticklabels(PROBLEM_TYPES, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(DOMAINS)))
    ax.set_yticklabels(DOMAINS, fontsize=8)
    ax.set_xlabel("PROBLEM TYPE")
    ax.set_ylabel("DOMAIN")

    stats = stats or {}
    headline = (
        f"MAP-Elites — iteration {iteration}   "
        f"champions {stats.get('num_champions', len(champions))}/"
        f"{len(DOMAINS) * len(PROBLEM_TYPES)}   "
        f"coverage {float(stats.get('coverage', 0.0)) * 100:.0f}%   "
        f"domain {float(stats.get('domain_coverage', 0.0)) * 100:.0f}%   "
        "type "
        f"{float(stats.get('problem_type_coverage', 0.0)) * 100:.0f}%"
    )
    subtitle = "cell text: R_Q over ŝ   ·   red border: filled this iteration"
    ax.set_title(f"{headline}\n{subtitle}", fontsize=9)

    bar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("R_Q of the cell's champion", fontsize=8)
    fig.tight_layout()
    return fig


def occupied_cells(archive: Any) -> set[tuple[int, int]]:
    """Cells holding a champion right now, for the next call's ``new_cells``."""
    try:
        cells = (archive.program_to_cell(c) for c in archive.champions())
        return {c for c in cells if c is not None}
    except Exception:
        return set()
