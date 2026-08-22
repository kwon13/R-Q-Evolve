"""Render the live GROUP x SKILL archive as one figure, for wandb.

The scalar coverage number says how many of the 48 cells are occupied. It
cannot say WHICH, and that is the thing the two axes were introduced to make
visible: "we only ever work in two domains" and "we only ever exercise two
reasoning moves" produce the same coverage and need opposite fixes. A picture
per outer iteration is the cheapest way to see the difference while a run is
still going.

Kept out of ``archive.py`` so importing the archive never pulls in matplotlib,
and every entry point here degrades to None rather than raising: a logging
backend that cannot draw must not take a training run down.
"""

from __future__ import annotations

from typing import Any

from .concepts import GROUPS, SKILLS

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
    """One heatmap of the archive: rows are GROUPs, columns are SKILLs.

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

    grid = np.full((len(GROUPS), len(SKILLS)), np.nan)
    s_hat = np.full((len(GROUPS), len(SKILLS)), np.nan)
    for champion in champions:
        cell = archive.program_to_cell(champion)
        if cell is None:
            continue
        g, s = cell
        rq = float(getattr(champion, "rq_score", 0.0) or 0.0)
        if np.isnan(grid[g, s]) or rq > grid[g, s]:
            grid[g, s] = rq
            s_hat[g, s] = float(getattr(champion, "s_hat", 0.0) or 0.0)

    finite = grid[~np.isnan(grid)]
    vmax = float(finite.max()) if finite.size and finite.max() > 0 else 1.0

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(_EMPTY)
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)

    new_cells = new_cells or set()
    for g in range(len(GROUPS)):
        for s in range(len(SKILLS)):
            value = grid[g, s]
            if not np.isnan(value):
                ax.text(
                    s, g, f"{value:.2f}\n{s_hat[g, s]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if value < vmax * 0.6 else "black",
                )
            if (g, s) in new_cells:
                ax.add_patch(
                    plt.Rectangle(
                        (s - 0.5, g - 0.5), 1, 1,
                        fill=False, edgecolor="#e8453c", lw=2.2,
                    )
                )

    ax.set_xticks(range(len(SKILLS)))
    ax.set_xticklabels(SKILLS, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(GROUPS)))
    ax.set_yticklabels(GROUPS, fontsize=8)
    ax.set_xlabel("SKILL")
    ax.set_ylabel("GROUP")

    stats = stats or {}
    headline = (
        f"MAP-Elites — iteration {iteration}   "
        f"champions {stats.get('num_champions', len(champions))}/"
        f"{len(GROUPS) * len(SKILLS)}   "
        f"coverage {float(stats.get('coverage', 0.0)) * 100:.0f}%   "
        f"group {float(stats.get('group_coverage', 0.0)) * 100:.0f}%   "
        f"skill {float(stats.get('skill_coverage', 0.0)) * 100:.0f}%"
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
