"""Side-by-side comparison of two R_Q-Evolve result directories."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from viz_common import load_snapshots


COLORS = {"prev": "#6f6c67", "new": "#2a78d6"}


def load_run(path: Path) -> dict:
    archive = path / "rq_archive"
    snapshots = dict(load_snapshots(archive))
    logs = [
        json.loads(line) for line in (archive / "evolution_log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    metrics = {int(row["metrics"]["outer_iteration"]): row["metrics"] for row in logs}
    return {"path": path, "snapshots": snapshots, "metrics": metrics}


def score_at_step(path: Path, step: int = 32) -> dict:
    scores = path / "scores.md"
    if not scores.exists():
        return {}
    headers = []
    in_final = False
    for line in scores.read_text().splitlines():
        if line.startswith("## pass@1 (final"):
            in_final = True
        elif in_final and line.startswith("## pass@1 (pre-GPT"):
            break
        elif in_final and line.startswith("| step |"):
            headers = [x.strip() for x in line.strip().strip("|").split("|")]
        elif in_final and headers and re.match(rf"^\|\s*{step}\s*\|", line):
            values = [x.strip() for x in line.strip().strip("|").split("|")]
            return {name: float(value) for name, value in zip(headers[1:], values[1:])}
    return {}


def archive_matrix(snapshot: dict, n_h: int, n_div: int) -> tuple[np.ndarray, int]:
    matrix = np.full((n_h, n_div), np.nan)
    collisions = 0
    for champion in snapshot.get("champions", []):
        row = int(champion.get("niche_h", -1))
        col = int(champion.get("niche_div", -1))
        if 0 <= row < n_h and 0 <= col < n_div:
            score = float(champion.get("rq_score", 0.0))
            if np.isfinite(matrix[row, col]):
                collisions += 1
                matrix[row, col] = max(matrix[row, col], score)
            else:
                matrix[row, col] = score
    return matrix, collisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prev", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-step", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    runs = {"prev": load_run(args.prev), "new": load_run(args.new)}
    common_snapshot_iters = set.intersection(
        *(set(run["snapshots"]) for run in runs.values())
    )
    if not common_snapshot_iters:
        raise ValueError("prev and new have no common archive iteration")
    common_end = max(common_snapshot_iters)
    common_snapshot = {name: run["snapshots"][common_end] for name, run in runs.items()}
    shapes = {
        name: (
            int(snapshot.get("meta", {}).get("n_h_bins", 10)),
            int(snapshot.get("meta", {}).get("n_div_bins", 6)),
        )
        for name, snapshot in common_snapshot.items()
    }
    if len(set(shapes.values())) != 1:
        raise ValueError(f"archive grid shapes differ at iter {common_end}: {shapes}")
    n_h, n_div = next(iter(shapes.values()))
    groups = {}
    group_conflicts = {}
    for run in runs.values():
        for snapshot in run["snapshots"].values():
            for champion in snapshot.get("champions", []):
                div = int(champion.get("niche_div", -1))
                group = (champion.get("metadata") or {}).get("concept_group", "")
                if div in groups and group and groups[div] != group:
                    group_conflicts.setdefault(div, set()).update((groups[div], group))
                elif group:
                    groups.setdefault(div, group)
    if group_conflicts:
        raise ValueError(f"concept-group/bin mapping is inconsistent: {group_conflicts}")
    group_labels = [groups.get(i) or f"bin {i}" for i in range(n_div)]
    matrix_results = {
        name: archive_matrix(snapshot, n_h, n_div)
        for name, snapshot in common_snapshot.items()
    }
    matrices = {name: result[0] for name, result in matrix_results.items()}
    collisions = {name: result[1] for name, result in matrix_results.items()}
    finite_scores = [
        float(value) for matrix in matrices.values()
        for value in matrix[np.isfinite(matrix)]
    ]
    vmax = max(finite_scores, default=0.0)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.grid": True,
        "grid.color": "#e8e7e3", "axes.axisbelow": True,
    })
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    for name, run in runs.items():
        xs = sorted(it for it in run["metrics"] if it <= common_end)
        axes[0, 0].plot(xs, [100 * run["metrics"][it]["coverage"] for it in xs],
                        lw=2.2, color=COLORS[name], label=name)
        axes[0, 1].plot(xs, [run["metrics"][it]["num_champions"] for it in xs],
                        lw=2.2, color=COLORS[name], label=name)
        axes[0, 2].plot(xs, [run["metrics"][it]["mean_rq"] for it in xs],
                        lw=2.2, color=COLORS[name], label=f"{name} mean")
        axes[0, 2].plot(xs, [run["metrics"][it]["max_rq"] for it in xs],
                        lw=1.2, ls="--", color=COLORS[name], alpha=.8,
                        label=f"{name} max")
    axes[0, 0].set_title("A · Archive coverage")
    axes[0, 0].set_ylabel("coverage (%)")
    axes[0, 1].set_title("B · Champion count")
    axes[0, 2].set_title("C · Archive RQ")
    for ax in axes[0]:
        ax.set_xlabel("outer iteration")
        ax.axvline(common_end, color="#aaa7a1", ls=":", lw=1)
        ax.legend(frameon=False, fontsize=8)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#191919")
    for col, name in enumerate(["prev", "new"]):
        ax = axes[1, col]
        image = ax.imshow(matrices[name], origin="lower", aspect="auto",
                          cmap=cmap, vmin=0, vmax=max(vmax, 1e-9))
        for row in range(n_h):
            for div in range(n_div):
                if np.isfinite(matrices[name][row, div]):
                    ax.text(div, row, f"{matrices[name][row, div]:.3f}",
                            ha="center", va="center", color="white", fontsize=6.5)
        ax.set_xticks(range(n_div))
        ax.set_xticklabels(group_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("difficulty bin")
        collision_note = (
            f" · {collisions[name]} coordinate collision(s), max shown"
            if collisions[name] else ""
        )
        ax.set_title(
            f"{'D' if col == 0 else 'E'} · {name} archive at iter "
            f"{common_end}{collision_note}",
            fontsize=10,
        )
    fig.colorbar(image, ax=[axes[1, 0], axes[1, 1]], fraction=.025, pad=.02,
                 label="rq_score")

    ax = axes[1, 2]
    prev_scores = score_at_step(args.prev, args.eval_step)
    new_scores = score_at_step(args.new, args.eval_step)
    benchmarks = [
        name for name in ["math500", "gsm8k", "amc23", "aime24", "aime25",
                          "minerva_math", "olympiadbench", "AVG"]
        if name in prev_scores and name in new_scores
    ]
    x = np.arange(len(benchmarks))
    width = .38
    ax.bar(x - width/2, [prev_scores[b] for b in benchmarks], width,
           color=COLORS["prev"], label="prev")
    ax.bar(x + width/2, [new_scores[b] for b in benchmarks], width,
           color=COLORS["new"], label="new")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("pass@1 (%)")
    ax.set_title(f"F · Downstream evaluation at step {args.eval_step}")
    ax.legend(frameon=False)

    fig.suptitle(
        f"R_Q-Evolve run comparison · common evolution horizon: iter 1–{common_end}",
        x=.04, ha="left", fontsize=15, fontweight="bold",
    )
    fig.subplots_adjust(left=.06, right=.96, bottom=.1, top=.91, wspace=.28, hspace=.35)
    png = args.output / "prev_vs_new_comparison.png"
    pdf = args.output / "prev_vs_new_comparison.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"saved {png}")
    print(f"saved {pdf}")


if __name__ == "__main__":
    main()
