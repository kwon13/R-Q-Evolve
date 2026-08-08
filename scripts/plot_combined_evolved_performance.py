#!/usr/bin/env python3
"""Combine Seed-ID and Structural-OOD trajectories without new inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.evolved_performance import load_evolution_events  # noqa: E402


def _load_trajectory(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoints = payload.get("checkpoints") or []
    if not checkpoints:
        raise ValueError(f"trajectory has no checkpoints: {path}")
    steps = [int(row["global_step"]) for row in checkpoints]
    if len(steps) != len(set(steps)):
        raise ValueError(f"trajectory has duplicate global steps: {path}")
    return payload


def _concept_map(row: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    concepts: dict[tuple[str, str], dict[str, Any]] = {}
    for values in (row.get("per_program") or {}).values():
        key = (
            str(values.get("group") or "unknown"),
            str(values.get("skill") or "unknown"),
        )
        if key in concepts:
            raise ValueError(f"multiple programs share concept {key} in one benchmark")
        concepts[key] = values
    return concepts


def combine_trajectories(
    id_payload: dict[str, Any],
    ood_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    id_by_step = {
        int(row["global_step"]): row for row in id_payload["checkpoints"]
    }
    ood_by_step = {
        int(row["global_step"]): row for row in ood_payload["checkpoints"]
    }
    if set(id_by_step) != set(ood_by_step):
        raise ValueError(
            "Seed-ID and OOD trajectories have different checkpoint sets: "
            f"ID={sorted(id_by_step)}, OOD={sorted(ood_by_step)}"
        )

    rows: list[dict[str, Any]] = []
    best = float("-inf")
    for step in sorted(id_by_step):
        id_row = id_by_step[step]
        ood_row = ood_by_step[step]
        id_concepts = _concept_map(id_row)
        ood_concepts = _concept_map(ood_row)
        if set(id_concepts) != set(ood_concepts):
            raise ValueError(
                f"concept sets differ at step {step}: "
                f"ID={sorted(id_concepts)}, OOD={sorted(ood_concepts)}"
            )

        id_score = float(id_row["score_percent"])
        ood_score = float(ood_row["score_percent"])
        balanced = 0.5 * (id_score + ood_score)
        best = max(best, balanced)
        concepts = []
        for group, skill in sorted(id_concepts):
            id_values = id_concepts[(group, skill)]
            ood_values = ood_concepts[(group, skill)]
            id_accuracy = 100.0 * float(id_values["accuracy"])
            ood_accuracy = 100.0 * float(ood_values["accuracy"])
            concepts.append(
                {
                    "group": group,
                    "skill": skill,
                    "seed_id_accuracy_percent": id_accuracy,
                    "structural_ood_v2_accuracy_percent": ood_accuracy,
                    "balanced_accuracy_percent": 0.5
                    * (id_accuracy + ood_accuracy),
                }
            )
        rows.append(
            {
                "global_step": step,
                "seed_id_score_percent": id_score,
                "structural_ood_v2_score_percent": ood_score,
                "balanced_score_percent": balanced,
                "best_balanced_score_percent": best,
                "seed_id_correct": int(id_row["correct"]),
                "seed_id_examples": int(id_row["num_examples"]),
                "structural_ood_v2_correct": int(ood_row["correct"]),
                "structural_ood_v2_examples": int(ood_row["num_examples"]),
                "combined_correct": int(id_row["correct"])
                + int(ood_row["correct"]),
                "combined_examples": int(id_row["num_examples"])
                + int(ood_row["num_examples"]),
                "cumulative_inner_iterations": int(
                    id_row["cumulative_inner_iterations"]
                ),
                "concepts": concepts,
            }
        )
    return rows


def _write_reports(
    output_dir: Path,
    id_payload: dict[str, Any],
    ood_payload: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "definition": (
            "Balanced Evolve Performance = 0.5 * Seed-ID EPS + "
            "0.5 * Structural-OOD-v2 EPS"
        ),
        "seed_id_benchmark_sha256": id_payload["benchmark_sha256"],
        "structural_ood_v2_benchmark_sha256": ood_payload["benchmark_sha256"],
        "checkpoints": rows,
    }
    (output_dir / "combined_trajectory.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Balanced Evolve Performance",
        "",
        (
            "`Balanced = 0.5 × Seed-ID EPS + 0.5 × Structural-OOD-v2 EPS`. "
            "Both sets contain 240 balanced examples, so this is also pooled "
            "accuracy over 480 problems."
        ),
        "",
        "| global step | Seed-ID | OOD-v2 | Balanced | best Balanced | correct |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['global_step']} | {row['seed_id_score_percent']:.2f}% | "
            f"{row['structural_ood_v2_score_percent']:.2f}% | "
            f"{row['balanced_score_percent']:.2f}% | "
            f"{row['best_balanced_score_percent']:.2f}% | "
            f"{row['combined_correct']}/{row['combined_examples']} |"
        )

    concept_order = [
        (item["group"], item["skill"]) for item in rows[0]["concepts"]
    ]
    lines.extend(
        [
            "",
            "## Balanced Concept Scores",
            "",
            "Each cell is the 50:50 average of the corresponding Seed-ID and OOD-v2 concept accuracies.",
            "",
            "| global step | "
            + " | ".join(f"{group}/{skill}" for group, skill in concept_order)
            + " |",
            "|---:|" + "---:|" * len(concept_order),
        ]
    )
    for row in rows:
        by_concept = {
            (item["group"], item["skill"]): item for item in row["concepts"]
        }
        lines.append(
            f"| {row['global_step']} | "
            + " | ".join(
                f"{by_concept[key]['balanced_accuracy_percent']:.1f}%"
                for key in concept_order
            )
            + " |"
        )
    (output_dir / "combined_scores.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def plot(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    id_path = args.id_results_dir.expanduser().resolve() / "trajectory.json"
    ood_path = args.ood_results_dir.expanduser().resolve() / "trajectory.json"
    output_dir = args.output_dir.expanduser().resolve()
    id_payload = _load_trajectory(id_path)
    ood_payload = _load_trajectory(ood_path)
    rows = combine_trajectories(id_payload, ood_payload)
    _write_reports(output_dir, id_payload, ood_payload, rows)

    steps = [row["global_step"] for row in rows]
    id_scores = [row["seed_id_score_percent"] for row in rows]
    ood_scores = [row["structural_ood_v2_score_percent"] for row in rows]
    balanced = [row["balanced_score_percent"] for row in rows]
    best = [row["best_balanced_score_percent"] for row in rows]
    cumulative_inner = [row["cumulative_inner_iterations"] for row in rows]

    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.fill_between(
        steps,
        id_scores,
        ood_scores,
        color="#aeb8c6",
        alpha=0.12,
        label="ID–OOD transfer gap",
        zorder=1,
    )
    ax.plot(
        steps,
        id_scores,
        color="#1756d1",
        linestyle="-.",
        marker="o",
        linewidth=2.2,
        markersize=6,
        label="Seed-ID EPS (240)",
        zorder=4,
    )
    ax.plot(
        steps,
        ood_scores,
        color="#7b2cbf",
        linestyle=":",
        marker="s",
        linewidth=2.5,
        markersize=6,
        label="Structural-OOD v2 EPS (240)",
        zorder=4,
    )
    ax.plot(
        steps,
        balanced,
        color="#087f5b",
        linestyle="-",
        marker="D",
        linewidth=3.2,
        markersize=6.5,
        label="Balanced ID + OOD-v2 (480)",
        zorder=6,
    )
    ax.step(
        steps,
        best,
        where="post",
        color="#d62728",
        linewidth=2.4,
        label="Best balanced so far",
        zorder=3,
    )
    for index, (step, score) in enumerate(zip(steps, balanced)):
        offset = 9 if index % 2 == 0 else -15
        ax.annotate(
            f"{score:.1f}",
            xy=(step, score),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if offset > 0 else "top",
            fontsize=8,
            color="#075b43",
            weight="bold",
            zorder=7,
        )

    ax.set_xlabel("Global Training Step (Saved Models)", fontsize=15, weight="bold")
    ax.set_ylabel("Performance Score (%)", fontsize=15, weight="bold")
    ax.set_xlim(min(steps), max(steps))
    low = min([*id_scores, *ood_scores, *balanced])
    high = max([*id_scores, *ood_scores, *balanced])
    pad = max(4.0, 0.08 * (high - low))
    ax.set_ylim(max(0.0, low - pad), min(100.0, high + pad))
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    events = [
        event
        for event in load_evolution_events(args.id_results_dir.parent)
        if min(steps) <= event.global_step <= max(steps)
    ]
    if events:
        evolution_steps = [min(steps)] + [event.global_step for event in events]
        evolution_steps.append(max(steps))
        evolution_inner = [0] + [event.cumulative_inner for event in events]
        evolution_inner.append(evolution_inner[-1])
    else:
        evolution_steps = steps
        evolution_inner = cumulative_inner
    ax2.step(
        evolution_steps,
        evolution_inner,
        where="post",
        color="#f39c12",
        linestyle="--",
        linewidth=2.0,
        label="Cumulative inner evolutions",
        zorder=2,
    )
    ax2.set_ylim(0, max(evolution_inner) * 1.08)
    ax2.set_ylabel(
        "Cumulative Inner Iterations (Evaluated Proposals)",
        fontsize=13,
        color="#b36b00",
        weight="bold",
    )
    ax2.tick_params(axis="y", colors="#b36b00")

    title = args.title or (
        "R-Q-Evolve — ID + Structural-OOD v2 Performance Evolution "
        "(480 problems)"
    )
    ax.set_title(title, fontsize=21, weight="bold", pad=18)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        framealpha=0.94,
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))

    png_path = output_dir / "combined_evolved_performance.png"
    svg_path = output_dir / "combined_evolved_performance.svg"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[EPB-COMBINED] wrote {png_path}")
    print(f"[EPB-COMBINED] wrote {svg_path}")
    print(f"[EPB-COMBINED] wrote {output_dir / 'combined_scores.md'}")
    print(f"[EPB-COMBINED] wrote {output_dir / 'combined_trajectory.json'}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id-results-dir", type=Path, required=True)
    parser.add_argument("--ood-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title")
    return parser


if __name__ == "__main__":
    plot(build_argparser().parse_args())
