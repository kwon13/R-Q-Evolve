#!/usr/bin/env python
"""The archive grid twice: as the programs label themselves, and as GPT reads them.

The MAP axes are the programs' own GROUP/SKILL constants, so coverage is a claim
the generator makes about itself. This redraws the same champions on the same
grid using an independent zero-shot judge that sees ONLY the rendered problem
and its reference answer -- never the source, never the declared labels -- and
puts the two side by side. A cell count that collapses between the panels is
label inflation, not reach.

Judge output comes from run_grid_eval_final.py (labels/rq_unique_programs.jsonl).
The judge may answer "other": a ninth column, because forcing an out-of-vocabulary
problem into one of the eight would manufacture the agreement being measured.

    python scripts/plot_archive_gpt_vs_declared.py \
        --run-dir rq_output/rq_evolve_8b_8gpu \
        --labels analysis/concept_grid_8b_8gpu/<model>__<effort>__<hash>/labels/rq_unique_programs.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.concepts import GROUPS, SKILLS  # noqa: E402

OTHER = "other"
COLS = [*SKILLS, OTHER]
_EMPTY = "#1b1b1f"          # same as map_figure: an empty cell is not white
_INVALID_EDGE = "#e8453c"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--labels", required=True, help="rq_unique_programs.jsonl from the judge")
    p.add_argument("--iteration", type=int, default=0, help="0 = latest snapshot")
    p.add_argument(
        "--cumulative",
        action="store_true",
        help=(
            "union every program that appears in any snapshot, not the latest "
            "snapshot alone. A cell then reports the best R_Q ever recorded for "
            "it, so the panel answers 'where did the search ever reach' rather "
            "than 'where does it stand now'."
        ),
    )
    p.add_argument("--out-dir", default="")
    p.add_argument("--stem", default="archive_gpt_vs_declared")
    return p.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def snapshot(run_dir: Path, iteration: int) -> tuple[Path, int]:
    snaps = sorted(
        (run_dir / "rq_archive").glob("archive_iter*.json"),
        key=lambda q: int(re.search(r"iter(\d+)", q.name).group(1)),
    )
    if not snaps:
        raise SystemExit(f"no snapshots under {run_dir}/rq_archive")
    if iteration:
        for s in snaps:
            if int(re.search(r"iter(\d+)", s.name).group(1)) == iteration:
                return s, iteration
        raise SystemExit(f"no snapshot for iteration {iteration}")
    return snaps[-1], int(re.search(r"iter(\d+)", snaps[-1].name).group(1))


def union_champions(run_dir: Path, max_iter: int) -> list[dict]:
    """Every program that ever held a cell, keyed by program_id.

    Snapshots are states, not history: a champion can be replaced and its cell
    refilled, so the union is strictly larger than any one snapshot. Each
    program keeps the best R_Q it was ever recorded with -- re-evaluation moves
    the number, and the cumulative panel is about reach, not the current score.
    """
    best: dict[str, dict] = {}
    for snap in sorted(
        (run_dir / "rq_archive").glob("archive_iter*.json"),
        key=lambda q: int(re.search(r"iter(\d+)", q.name).group(1)),
    ):
        if int(re.search(r"iter(\d+)", snap.name).group(1)) > max_iter:
            break
        for champ in json.loads(snap.read_text(encoding="utf-8"))["champions"]:
            pid = champ["program_id"]
            prior = best.get(pid)
            if prior is None or float(champ.get("rq_score") or 0.0) > float(
                prior.get("rq_score") or 0.0
            ):
                best[pid] = champ
    return list(best.values())


def grid_from(champions: list[dict], cell_of) -> tuple[np.ndarray, np.ndarray, int]:
    """Best-R_Q grid, an invalid-flag grid, and how many cells got filled."""
    rq = np.full((len(GROUPS), len(COLS)), np.nan)
    invalid = np.zeros((len(GROUPS), len(COLS)), dtype=bool)
    for champ in champions:
        cell = cell_of(champ)
        if cell is None:
            continue
        g, s = cell
        value = float(champ.get("rq_score") or 0.0)
        if np.isnan(rq[g, s]) or value > rq[g, s]:
            rq[g, s] = value
            invalid[g, s] = champ.get("_gpt_valid") is False
    return rq, invalid, int((~np.isnan(rq)).sum())


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    snap, iteration = snapshot(run_dir, args.iteration)
    if args.cumulative:
        champions = union_champions(run_dir, iteration)
        scope = f"cumulative over iterations 1-{iteration}"
    else:
        champions = json.loads(snap.read_text(encoding="utf-8"))["champions"]
        scope = f"snapshot at outer iteration {iteration}"

    judged = {}
    for line in resolve(args.labels).read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        pid = str(row.get("source_id") or "").strip()
        if pid:
            judged[pid] = row
    hit = sum(1 for c in champions if c["program_id"] in judged)
    print(f"[plot] iteration {iteration}: {len(champions)} champions, "
          f"{hit} carry a judge label ({len(judged)} judged in total)")

    gi = {g: i for i, g in enumerate(GROUPS)}
    ci = {c: i for i, c in enumerate(COLS)}

    for champ in champions:
        row = judged.get(champ["program_id"])
        champ["_gpt_valid"] = None if row is None else bool(row.get("valid"))
        champ["_gpt"] = row

    def declared_cell(champ):
        meta = champ.get("metadata") or {}
        g, s = meta.get("group"), meta.get("skill")
        if g in gi and s in ci:
            return gi[g], ci[s]
        return None

    def gpt_cell(champ):
        row = champ.get("_gpt")
        if row is None:
            return None
        g = str(row.get("group") or "").strip()
        s = str(row.get("skill") or "").strip() or OTHER
        if g not in gi:
            return None
        return gi[g], ci.get(s, ci[OTHER])

    declared = grid_from(champions, declared_cell)
    gpt = grid_from(champions, gpt_cell)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "analysis" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    figure(declared, gpt, scope, len(champions), out_dir, args.stem)

    n_invalid = sum(1 for c in champions if c["_gpt_valid"] is False)
    print(f"[plot] cells occupied -- declared {declared[2]}, judge {gpt[2]}")
    print(f"[plot] champions the judge calls invalid: {n_invalid}/{hit}")
    print(f"[plot] -> {out_dir}/{args.stem}.png")
    return 0


def figure(declared, gpt, scope, n_champ, out_dir: Path, stem: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 200, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    # One shared scale, so a cell's colour means the same thing in both panels.
    vmax = max(
        float(np.nanmax(declared[0])) if np.isfinite(declared[0]).any() else 0.0,
        float(np.nanmax(gpt[0])) if np.isfinite(gpt[0]).any() else 0.0,
    ) or 1.0
    cmap = plt.cm.viridis.copy()   # matches src/rq_evolve/map_figure.py
    cmap.set_bad(_EMPTY)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.3))
    panels = (
        (axes[0], declared, "declared by the program", "GROUP/SKILL constants in the source"),
        (axes[1], gpt, "read by the judge", "zero-shot, problem text only"),
    )
    for ax, (rq, invalid, filled), title, sub in panels:
        im = ax.imshow(rq, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
        for g in range(len(GROUPS)):
            for s in range(len(COLS)):
                v = rq[g, s]
                if np.isnan(v):
                    continue
                ax.text(s, g, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v < vmax * 0.6 else "black")
                if invalid[g, s]:
                    ax.add_patch(plt.Rectangle((s - 0.5, g - 0.5), 1, 1, fill=False,
                                               edgecolor=_INVALID_EDGE, lw=2.0))
        ax.set_xticks(range(len(COLS)))
        ax.set_xticklabels(COLS, rotation=32, ha="right", fontsize=7.5)
        ax.get_xticklabels()[-1].set_style("italic")
        ax.set_yticks(range(len(GROUPS)))
        ax.set_yticklabels(GROUPS, fontsize=7.5)
        ax.set_xlabel("SKILL")
        ax.set_title(f"{title}   —   {filled}/48 cells\n{sub}", fontsize=9)
    axes[0].set_ylabel("GROUP")

    bar = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.015)
    bar.set_label(r"$R_Q$ of the cell's champion", fontsize=8)
    fig.suptitle(
        f"MAP-Elites, {scope} — {n_champ} programs, same set on both panels"
        "   ·   red border: judge marked the cell's champion invalid",
        fontsize=9.5, y=1.02,
    )
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
