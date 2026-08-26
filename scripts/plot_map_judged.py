#!/usr/bin/env python
"""The MAP grid as an independent reader fills it, over the cells the archive claims.

The archive's coordinates are the programs' own GROUP/SKILL constants (or, once
relabelling is on, the relabeller's pick). Either way the grid is a claim the
pipeline makes about itself, and coverage counts claims. This redraws the SAME
champions on the SAME grid using a zero-shot judge that saw only the rendered
problem and its reference answer, and lays the claim over the reading:

    purple fill   how many champions the judge put in that cell
    blue outline  the cell the ARCHIVE claims

Cells the archive claims that the judge never fills are label inflation. The
judge may answer "other" -- a ninth column, because forcing an out-of-vocabulary
problem into one of the eight manufactures the agreement being measured.

READ THE ROWS, NOT THE COLUMNS, unless the caveat line says otherwise. The line
is computed, not asserted: it scores the judge against the hand-written seeds in
this very archive, where the labels are ours and known. A judge that cannot
recover a seed's SKILL is not measuring the archive's SKILL either -- it is
measuring the distance between two taxonomies that share eight words.

    python scripts/plot_map_judged.py \
        --archive rq_output/<run>/rq_archive/archive.json \
        --readback rq_output/gate_experiment/<readback>.json \
        --label <key inside results> --out rq_output/<run>/map_judged.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.concepts import GROUPS, SKILLS  # noqa: E402

OTHER = "OTHER"
COLS = [*SKILLS, OTHER]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--readback", required=True, type=Path)
    p.add_argument("--label", default=None, help="key inside readback['results']")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--title", default=None)
    return p.parse_args()


def load(args):
    champs = json.load(open(args.archive))["champions"]
    by_id = {c["program_id"]: c for c in champs}

    payload = json.load(open(args.readback))
    results = payload["results"]
    key = args.label or next(iter(results))
    rows = results[key]

    claimed: set[tuple[int, int]] = set()
    for c in champs:
        g, s = c.get("niche_group", -1), c.get("niche_skill", -1)
        if 0 <= g < len(GROUPS) and 0 <= s < len(SKILLS):
            claimed.add((g, s))

    counts = np.zeros((len(GROUPS), len(COLS)), dtype=int)
    unplaced = 0
    seed_g = seed_s = seed_n = 0
    for r in rows:
        c = by_id.get(r["program_id"])
        if c is None:
            continue
        group, skill = r.get("group"), r.get("skill")
        if group not in GROUPS:
            # The judge is allowed to refuse a GROUP; there is no row for that.
            unplaced += 1
        else:
            col = SKILLS.index(skill) if skill in SKILLS else COLS.index(OTHER)
            counts[GROUPS.index(group), col] += 1

        # Seed control: a hand-written seed never went through relabelling.
        md = c.get("metadata") or {}
        if not md.get("skill_declared") and int(c.get("generation") or 0) == 0:
            seed_n += 1
            seed_g += group == r.get("old_group")
            seed_s += skill == r.get("old_skill")

    return by_id, claimed, counts, unplaced, len(rows), key, (seed_g, seed_s, seed_n)


def figure(claimed, counts, unplaced, n, key, seeds, out: Path, title: str | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    skill_cols = counts[:, : len(SKILLS)]
    judged_cells = int((skill_cols > 0).sum())
    other_total = int(counts[:, COLS.index(OTHER)].sum())
    total_cells = len(GROUPS) * len(SKILLS)

    fig, ax = plt.subplots(figsize=(1.55 * len(COLS) + 4.5, 1.05 * len(GROUPS) + 2.6))
    vmax = max(1, int(counts.max()))
    ax.imshow(counts, aspect="auto", cmap="Purples", vmin=0, vmax=vmax)

    for g in range(len(GROUPS)):
        for c in range(len(COLS)):
            v = int(counts[g, c])
            if v:
                ax.text(c, g, str(v), ha="center", va="center", fontsize=13,
                        fontweight="bold",
                        color="white" if v > 0.6 * vmax else "#15151a")
    for (g, s) in sorted(claimed):
        ax.add_patch(Rectangle((s - 0.5, g - 0.5), 1, 1, fill=False,
                               edgecolor="#1f77ff", linewidth=2.2, zorder=4))
    # The archive has no OTHER, so that column can never carry a claim.
    ax.axvline(len(SKILLS) - 0.5, color="#555", linewidth=2.0, zorder=5)

    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels(COLS, rotation=30, ha="right")
    ax.set_yticks(range(len(GROUPS)))
    ax.set_yticklabels(GROUPS)
    ax.set_xlabel("SKILL")
    ax.set_ylabel("GROUP")
    ax.set_xticks(np.arange(-0.5, len(COLS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(GROUPS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)

    sg, ss, sn = seeds
    caveat = (
        f"CAVEAT: this reader scores {sg}/{sn} on the hand-written seeds' GROUP "
        f"but {ss}/{sn} on their SKILL, so trust the rows, not the columns."
        if sn else
        "CAVEAT: no hand-written seed survives in this archive, so the reader is uncalibrated here."
    )
    head = title or f"{n} champions ({key}) placed by GPT instead of by their own labels"
    line2 = (
        f"archive claims {len(claimed)}/{total_cells} cells "
        f"(coverage {len(claimed)/total_cells:.3f})   ->   "
        f"GPT lands them in {judged_cells}/{total_cells} cells "
        f"(coverage {judged_cells/total_cells:.3f}), "
        f"plus {other_total} it will not label at all"
    )
    if unplaced:
        line2 += f"; {unplaced} it will not give a GROUP"
    ax.set_title(f"{head}\n{line2}\n{caveat}", fontsize=12, loc="left", pad=14)

    ax.legend(
        handles=[
            Patch(facecolor=plt.get_cmap("Purples")(0.72),
                  label=f"champions GPT placed here (n={n})"),
            Patch(facecolor="none", edgecolor="#1f77ff", linewidth=2.2,
                  label=f"cell the ARCHIVE claims ({len(claimed)}/{total_cells})"),
        ],
        loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=140)
    print(f"[plot] -> {out}")
    print(f"        archive {len(claimed)}/{total_cells}  ->  GPT {judged_cells}/{total_cells}"
          f"  | OTHER {other_total} | no GROUP {unplaced} | seeds GROUP {sg}/{sn} SKILL {ss}/{sn}")


def main() -> int:
    a = parse_args()
    _by_id, claimed, counts, unplaced, n, key, seeds = load(a)
    figure(claimed, counts, unplaced, n, key, seeds, a.out, a.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
