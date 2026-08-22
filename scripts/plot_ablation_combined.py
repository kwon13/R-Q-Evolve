#!/usr/bin/env python3
"""One ablation figure for both scales: evolution only, scores in a table.

The per-scale figures paired each trajectory panel with a bar panel of standard
math averages. At the paper's text width that is eight panels in a row and the
bars end up ~1.2 inches wide with rotated tick labels. Splitting the two jobs
fixes it: the figure carries the trajectories, where the shape matters and the
4B/8B rows can be compared at a glance, and the table carries the end scores,
where the exact numbers matter.

Note what this means for reading the figure alone: the archive-structure arms
(flat, no-reeval) are nearly indistinguishable from the full run *in the
trajectories* -- their cost shows up only in the standard math average, i.e.
only in the table. That is the finding, not a defect of the plot, but the two
must be presented together.

    python scripts/plot_ablation_combined.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_ablation", HERE / "plot_evolved_performance_ablation.py"
)
_ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ab)

RQ = HERE.parent
ARMS = ("full", "flat", "noreeval", "nounc", "novar")
SCALES = {
    "4B": {
        "root": RQ / "results",
        "runs": {
            "full": "rq_evolve_base_4b",
            "flat": "rq_evolve_4b_ablate_flat",
            "noreeval": "rq_evolve_4b_ablate_noreeval",
            "nounc": "rq_evolve_4b_ablate_nounc",
            "novar": "rq_evolve_4b_ablate_novar",
        },
        "base_math": 47.45,
    },
    "8B": {
        "root": RQ / "rq_output",
        "runs": {
            "full": "rq_evolve_base_8b",
            "flat": "rq_evolve_8b_ablate_flat",
            "noreeval": "rq_evolve_8b_ablate_noreeval",
            "nounc": "rq_evolve_8b_ablate_nounc",
            "novar": "rq_evolve_8b_ablate_novar",
        },
        "base_math": 49.45,
    },
}
COLUMNS = (
    ("Archive structure", ("noreeval", "flat", "full")),
    ("$R_Q$ components", ("nounc", "novar", "full")),
)
LABELS = {
    "full": "R-Q-Evolve (full)",
    "flat": "Flat archive",
    "noreeval": "Without reevaluation",
    "nounc": "Without uncertainty ($U=1$)",
    "novar": "Without variance ($L=1$)",
}
MAX_STEP = 128


def load(scale: str) -> dict:
    spec = SCALES[scale]
    eps, math = {}, {}
    for arm, run in spec["runs"].items():
        _, scores = _ab._load_scores(
            spec["root"] / run / "evolved_performance_480_v1", MAX_STEP
        )
        eps[arm] = scores
        math[arm] = _ab._load_math_benchmark_scores(
            spec["root"] / run / "scores.md", MAX_STEP, "final"
        )
    return {"eps": eps, "math": math, "base_math": spec["base_math"]}


def build_figure(data: dict[str, dict], outdir: Path) -> Path:
    _ab.configure_style()
    steps = sorted(next(iter(data["4B"]["eps"].values())))
    tail = steps[-1] + max(10, (steps[-1] - steps[-2]) // 2)

    fig, axes = plt.subplots(
        2, 2, figsize=(5.5, 3.8), sharex=True,
        gridspec_kw=dict(hspace=0.22, wspace=0.12),
    )
    handles: dict[str, object] = {}

    for row, scale in enumerate(("4B", "8B")):
        values = data[scale]["eps"]
        # one y-range per row: 4B and 8B live on different levels, so a shared
        # scale would flatten whichever row is smaller
        flat_values = [v for arm in ARMS for v in values[arm].values()]
        lo, hi = min(flat_values) - 1.5, max(flat_values) + 1.5
        for col, (title, keys) in enumerate(COLUMNS):
            ax = axes[row][col]
            for arm in keys:
                series = [values[arm][s] for s in steps]
                line = ax.step(
                    [*steps, tail], [*series, series[-1]], where="post",
                    color=_ab.ARM_COLORS[arm],
                    linewidth=1.7 if arm == "full" else 1.3,
                    zorder=4 if arm == "full" else 3,
                    label=LABELS[arm],
                )[0]
                handles.setdefault(arm, line)
            ax.set_xlim(steps[0], tail)
            ax.set_ylim(lo, hi)
            ax.set_xticks(steps)
            ax.grid(True, linestyle="-", linewidth=0.5, color=_ab.C_GRID)
            ax.set_axisbelow(True)
            ax.tick_params(colors=_ab.C_TEXT)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 0:
                ax.set_title(title, weight="bold", pad=5, color=_ab.C_TEXT)
            if row == 1:
                ax.set_xlabel("Global training step")
            if col == 1:
                ax.tick_params(axis="y", labelleft=False)
        axes[row][0].set_ylabel(scale, weight="bold", color=_ab.C_TEXT,
                                labelpad=8)

    fig.supylabel("Evolved performance score (%)", fontsize=9,
                  color=_ab.C_TEXT, x=0.005)

    ordered = [handles[a] for a in ARMS if a in handles]
    fig.legend(
        ordered, [h.get_label() for h in ordered],
        loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=3,
        frameon=False, fontsize=7, handlelength=1.6, columnspacing=1.3,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / "fig_ablation_evolution_4b_8b"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=450 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"[ablation] {stem}.pdf/.svg/.png")
    return stem


def write_table(data: dict[str, dict], outdir: Path) -> None:
    rows = []
    for scale in ("4B", "8B"):
        eps, math = data[scale]["eps"], data[scale]["math"]
        for arm in ARMS:
            rows.append({
                "model_size": scale,
                "arm": arm,
                "label": LABELS[arm],
                "eps_128": round(eps[arm][MAX_STEP], 2),
                "eps_delta": round(eps[arm][MAX_STEP] - eps["full"][MAX_STEP], 2),
                "math_avg_128": round(math[arm]["AVG"], 2),
                "math_delta": round(math[arm]["AVG"] - math["full"]["AVG"], 2),
            })

    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "ablation_combined.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Ablation at step 128, both scales",
        "",
        "EPS is the fixed 480-problem benchmark; Math is the macro average over the",
        "seven standard math benchmarks after the GPT-4o re-check. Deltas are against",
        "the full run at the same scale.",
        "",
        "| | | EPS | ΔEPS | Math | ΔMath |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        head = f"**{row['model_size']}**" if row["arm"] == "full" else ""
        delta_eps = "—" if row["arm"] == "full" else f"{row['eps_delta']:+.2f}"
        delta_math = "—" if row["arm"] == "full" else f"{row['math_delta']:+.2f}"
        lines.append(
            f"| {head} | {row['label']} | {row['eps_128']:.2f} | {delta_eps} "
            f"| {row['math_avg_128']:.2f} | {delta_math} |"
        )
    lines += [
        "",
        "Base model math average: "
        + ", ".join(f"{s} {SCALES[s]['base_math']:.2f}" for s in ("4B", "8B"))
        + ".",
        "",
    ]
    (outdir / "ablation_combined.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[ablation] {outdir / 'ablation_combined.md'}")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", type=Path,
        default=Path("/data1/yhoon113/RQ_Evolve_Final_Figures/figures"),
    )
    parser.add_argument(
        "--tabledir", type=Path,
        default=Path("/data1/yhoon113/RQ_Evolve_Final_Figures/tables"),
    )
    args = parser.parse_args()

    data = {scale: load(scale) for scale in SCALES}
    build_figure(data, args.outdir)
    write_table(data, args.tabledir)


if __name__ == "__main__":
    main()
