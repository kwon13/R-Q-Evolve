#!/usr/bin/env python
"""Draw a run's MAP-Elites archive as it stands, from its rq_archive snapshots.

Two figures, because the two questions have different forms:

  archive_map.png   the GROUP x SKILL grid at the latest snapshot -- WHICH cells
                    are occupied and how good their champion is. Rendered by the
                    run's own ``rq_evolve.map_figure``, so it is the same picture
                    the trainer logs to wandb rather than a second dialect of it.
  archive_progress  coverage and R_Q against outer iteration. Separate panels,
                    not one twin-axis plot: a fraction and a fitness share no
                    scale, and overlaying them invents a crossing point that
                    means nothing.

Usable mid-run: it reads whatever archive_iter*.json exist.

    python scripts/plot_live_archive.py --run-dir rq_output/rq_evolve_8b_8gpu
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.archive import MAPElitesArchive  # noqa: E402
from rq_evolve.map_figure import occupied_cells, render_archive_figure  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="run directory holding rq_archive/")
    p.add_argument("--out-dir", default="", help="default: analysis/<run name>/")
    p.add_argument("--stem", default="archive")
    return p.parse_args()


def iter_no(path: Path) -> int:
    m = re.search(r"archive_iter(\d+)\.json$", path.name)
    return int(m.group(1)) if m else -1


def load_archive(path: Path) -> MAPElitesArchive:
    archive = MAPElitesArchive()
    archive.load(path)
    return archive


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    snap_dir = run_dir / "rq_archive"
    snaps = sorted(snap_dir.glob("archive_iter*.json"), key=iter_no)
    if not snaps:
        raise SystemExit(f"no archive_iter*.json under {snap_dir}")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "analysis" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- the grid, at the latest snapshot ---------------------------------
    latest, prev = snaps[-1], (snaps[-2] if len(snaps) > 1 else None)
    archive = load_archive(latest)
    stats = json.loads(latest.read_text(encoding="utf-8"))["meta"].get("stats", {})
    new_cells = (
        occupied_cells(archive) - occupied_cells(load_archive(prev))
        if prev is not None
        else set()
    )
    fig = render_archive_figure(
        archive, iteration=iter_no(latest), new_cells=new_cells, stats=stats
    )
    if fig is None:
        raise SystemExit("matplotlib unavailable: no figure rendered")
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{args.stem}_map.{ext}", dpi=200, bbox_inches="tight")
    fig.clf()

    # --- the trajectory ----------------------------------------------------
    rows = []
    for snap in snaps:
        meta = json.loads(snap.read_text(encoding="utf-8"))["meta"].get("stats", {})
        rows.append({"iteration": iter_no(snap), **meta})
    (out_dir / f"{args.stem}_progress.json").write_text(
        json.dumps(rows, indent=1), encoding="utf-8"
    )
    progress_figure(rows, out_dir, args.stem)

    print(f"[plot] {len(snaps)} snapshots (iter {iter_no(snaps[0])}-{iter_no(latest)})")
    print(f"[plot] latest: champions {stats.get('num_champions')}/48, "
          f"coverage {float(stats.get('coverage', 0)) * 100:.1f}%, "
          f"mean R_Q {float(stats.get('mean_rq', 0)):.4f}, "
          f"max R_Q {float(stats.get('max_rq', 0)):.4f}, "
          f"newly filled this iteration {len(new_cells)}")
    print(f"[plot] -> {out_dir}")
    return 0


def progress_figure(rows: list[dict], out_dir: Path, stem: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 200, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    # One hue per entity, fixed order, never recycled; text stays ink-coloured.
    C_CELL, C_GROUP, C_SKILL = "#2b6cb0", "#7a8b99", "#b08968"
    C_MEAN, C_MAX = "#2b6cb0", "#c1666b"

    it = [r["iteration"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))

    for key, colour, label in (
        ("coverage", C_CELL, "cells (of 48)"),
        ("group_coverage", C_GROUP, "GROUPs"),
        ("skill_coverage", C_SKILL, "SKILLs"),
    ):
        ax1.plot(it, [100 * float(r.get(key, 0)) for r in rows], lw=2,
                 marker="o", ms=4, color=colour, label=label)
    ax1.set_ylim(0, 104)
    ax1.set_xlabel("outer iteration")
    ax1.set_ylabel("coverage (%)")
    ax1.set_title("what the grid reaches", fontsize=9.5)
    ax1.legend(fontsize=7.5, frameon=False)

    ax2.plot(it, [float(r.get("max_rq", 0)) for r in rows], lw=2, marker="o",
             ms=4, color=C_MAX, label="best cell")
    ax2.plot(it, [float(r.get("mean_rq", 0)) for r in rows], lw=2, marker="o",
             ms=4, color=C_MEAN, label="mean over occupied cells")
    ax2.set_xlabel("outer iteration")
    ax2.set_ylabel(r"$R_Q$")
    ax2.set_title("champion fitness", fontsize=9.5)
    ax2.legend(fontsize=7.5, frameon=False)

    for ax in (ax1, ax2):
        ax.set_xticks(it)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}_progress.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
