"""Render MAP-Elites archive snapshots as PNG frames and an animated GIF."""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


def iteration(path: Path) -> int:
    return int(re.search(r"iter(\d+)", path.name).group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rq-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-ms", type=int, default=220)
    parser.add_argument("--max-outer-iteration", type=int, default=None)
    args = parser.parse_args()

    archive = args.rq_output / "rq_archive"
    files = sorted(archive.glob("archive_iter*.json"), key=iteration)
    if args.max_outer_iteration is not None:
        files = [path for path in files if iteration(path) <= args.max_outer_iteration]
    if not files:
        raise FileNotFoundError(f"No archive_iter*.json under {archive}")
    snapshots = [(iteration(path), json.loads(path.read_text())) for path in files]
    args.output.mkdir(parents=True, exist_ok=True)

    meta = snapshots[-1][1].get("meta", {})
    n_h = int(meta.get("n_h_bins", 10))
    n_div = int(meta.get("n_div_bins", 6))
    h_lo, h_hi = meta.get("h_range", [0.1, 0.6])
    groups = {}
    global_max = 0.0
    for _, data in snapshots:
        for champion in data.get("champions", []):
            groups.setdefault(
                int(champion.get("niche_div", -1)),
                (champion.get("metadata") or {}).get("concept_group", ""),
            )
            global_max = max(global_max, float(champion.get("rq_score", 0.0)))
    labels = [groups.get(i) or f"bin {i}" for i in range(n_div)]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#191919")
    previous_cells = set()
    frames = []

    for frame_number, (it, data) in enumerate(snapshots, 1):
        matrix = np.full((n_h, n_div), np.nan)
        occupied = set()
        for champion in data.get("champions", []):
            row = int(champion.get("niche_h", -1))
            col = int(champion.get("niche_div", -1))
            if 0 <= row < n_h and 0 <= col < n_div:
                matrix[row, col] = float(champion.get("rq_score", 0.0))
                occupied.add((row, col))
        newly_filled = occupied - previous_cells
        previous_cells = occupied
        stats = data.get("meta", {}).get("stats", {})

        fig, ax = plt.subplots(figsize=(9.2, 8.8))
        image = ax.imshow(
            matrix, origin="lower", aspect="auto", cmap=cmap,
            vmin=0.0, vmax=max(global_max, 1e-9),
        )
        for row, col in occupied:
            value = matrix[row, col]
            ax.text(col, row, f"{value:.3f}", color="white", fontsize=7.5,
                    ha="center", va="center")
        for row, col in newly_filled:
            ax.add_patch(patches.Rectangle(
                (col - 0.5, row - 0.5), 1, 1,
                fill=False, edgecolor="red", linewidth=2.5,
            ))
        h_edges = np.linspace(float(h_lo), float(h_hi), n_h + 1)
        ax.set_yticks(range(n_h))
        ax.set_yticklabels([
            f"{h_edges[i]:.2f}–{h_edges[i + 1]:.2f}" for i in range(n_h)
        ])
        ax.set_xticks(range(n_div))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_xlabel("Concept group (diversity axis)")
        ax.set_ylabel("Difficulty bin (h_score)")
        ax.set_title(
            f"MAP-Elites Archive · {args.rq_output.name} · iter {it} "
            f"({frame_number}/{len(snapshots)})\n"
            f"champions={stats.get('num_champions', len(occupied))}  "
            f"coverage={100 * float(stats.get('coverage', len(occupied)/(n_h*n_div))):.1f}%  "
            f"mean_rq={float(stats.get('mean_rq', 0)):.4f}  "
            f"max_rq={float(stats.get('max_rq', 0)):.4f}\n"
            f"insertions={stats.get('total_insertions', '–')}  "
            f"replacements={stats.get('total_replacements', '–')}  "
            "(red box = newly filled niche)",
            fontsize=11,
        )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("rq_score (fitness)")
        fig.tight_layout()
        frame_path = args.output / f"frame_{frame_number:03d}.png"
        fig.savefig(frame_path, dpi=130)
        plt.close(fig)
        frames.append(frame_path)

    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frames]
    gif_path = args.output / "map_evolution.gif"
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=False,
    )
    for image in images:
        image.close()
    print(f"saved {len(frames)} frames and {gif_path}")


if __name__ == "__main__":
    main()
