#!/usr/bin/env python3
"""Plot checkpoint EPS with cumulative inner and active outer evolution."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.evolved_performance import (  # noqa: E402
    build_concept_change_rows,
    build_prominent_high_rq_rows,
    evolution_state_at_step,
    load_evolution_events,
    write_concept_change_report,
    write_prominent_high_rq_report,
)

_RESULT_RE = re.compile(r"global_step_(\d+)$")
_GROUP_SHORT = {
    "algebra": "Alg",
    "combinatorics": "Comb",
    "geometry": "Geo",
    "inequality": "Ineq",
    "number_theory": "NT",
    "sequence": "Seq",
}
_SKILL_SHORT = {
    "casework": "Case",
    "contradiction": "Contra",
    "counting": "Count",
    "extremal_principle": "Ext",
    "induction": "Ind",
    "invariant": "Inv",
    "transformation": "Trans",
}


def _short_concept(group: str, skill: str) -> str:
    return (
        f"{_GROUP_SHORT.get(group, group)}/"
        f"{_SKILL_SHORT.get(skill, skill)}"
    )


def _load_results(results_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for path in results_dir.glob("global_step_*/summary.json"):
        match = _RESULT_RE.fullmatch(path.parent.name)
        if not match:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["global_step"] = int(payload.get("global_step", match.group(1)))
        # Summaries produced before concept-delta plotting contain per-program
        # accuracies but not GROUP/SKILL. The corresponding immutable details
        # rows still carry both benchmark labels, so enrich in-place and avoid
        # an expensive checkpoint re-evaluation solely for plotting metadata.
        per_program = payload.get("per_program") or {}
        missing = {
            name
            for name, values in per_program.items()
            if not values.get("group") or not values.get("skill")
        }
        details_path = path.parent / "details.jsonl"
        if missing and details_path.is_file():
            with details_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    detail = json.loads(line)
                    name = str(detail.get("program_name") or "")
                    if name not in missing:
                        continue
                    per_program[name]["group"] = str(
                        detail.get("group") or "unknown"
                    )
                    per_program[name]["skill"] = str(
                        detail.get("skill") or "unknown"
                    )
                    missing.remove(name)
                    if not missing:
                        break
        results.append(payload)
        hashes.add(str(payload.get("benchmark_sha256", "")))
    if not results:
        raise FileNotFoundError(f"no global_step_*/summary.json under {results_dir}")
    if len(hashes) != 1 or "" in hashes:
        raise ValueError(f"results mix benchmark hashes: {sorted(hashes)}")
    return sorted(results, key=lambda row: int(row["global_step"]))


def _write_tables(
    results_dir: Path,
    results: list[dict[str, Any]],
    events,
) -> list[dict[str, Any]]:
    best = float("-inf")
    rows: list[dict[str, Any]] = []
    for result in results:
        step = int(result["global_step"])
        score = float(result["score_percent"])
        best = max(best, score)
        outer, cumulative_inner, cumulative_inserted = evolution_state_at_step(
            events, step
        )
        rows.append(
            {
                "global_step": step,
                "score_percent": score,
                "best_score_percent": best,
                "correct": int(result["correct"]),
                "num_examples": int(result["num_examples"]),
                "active_outer_iteration": outer,
                "cumulative_inner_iterations": cumulative_inner,
                "cumulative_insertions": cumulative_inserted,
                "per_program": result.get("per_program", {}),
                "per_concept": result.get("per_concept", {}),
            }
        )
    (results_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "benchmark_sha256": results[0]["benchmark_sha256"],
                "checkpoints": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    md = [
        "# Evolved Performance Score",
        "",
        f"Benchmark SHA256: `{results[0]['benchmark_sha256']}`",
        "",
        "| global step | EPS (%) | best (%) | correct | active outer | cumulative inner | inserted |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        outer = (
            "bootstrap"
            if row["active_outer_iteration"] is None
            else str(row["active_outer_iteration"])
        )
        md.append(
            f"| {row['global_step']} | {row['score_percent']:.2f} | "
            f"{row['best_score_percent']:.2f} | {row['correct']}/"
            f"{row['num_examples']} | {outer} | "
            f"{row['cumulative_inner_iterations']} | "
            f"{row['cumulative_insertions']} |"
        )

    table_key = "per_concept" if rows[0].get("per_concept") else "per_program"
    program_names = sorted(rows[0][table_key])
    for row in rows[1:]:
        if set(row[table_key]) != set(program_names):
            raise ValueError(
                f"checkpoint {table_key} sets differ while writing scores.md"
            )
    md.extend(
        [
            "",
            "## Concept Scores by Checkpoint",
            "",
            (
                "Each cell is accuracy on the fixed benchmark examples for "
                "that seed-program concept."
            ),
            "",
        ]
    )
    concept_labels = []
    for name in program_names:
        values = rows[0][table_key][name]
        concept_labels.append(
            f"{values.get('group') or 'unknown'}/"
            f"{values.get('skill') or 'unknown'}"
        )
    md.append("| global step | " + " | ".join(concept_labels) + " |")
    md.append("|---:|" + "---:|" * len(program_names))
    for row in rows:
        concept_scores = [
            f"{100.0 * float(row[table_key][name]['accuracy']):.1f}%"
            for name in program_names
        ]
        md.append(
            f"| {row['global_step']} | " + " | ".join(concept_scores) + " |"
        )
    (results_dir / "scores.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return rows


def plot(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = args.run_dir.expanduser().resolve()
    results_dir = (
        args.results_dir.expanduser().resolve()
        if args.results_dir
        else run_dir / "evolved_performance"
    )
    results = _load_results(results_dir)
    events = load_evolution_events(run_dir)
    rows = _write_tables(results_dir, results, events)
    high_rq_rows = build_prominent_high_rq_rows(
        run_dir,
        quantile=args.rq_quantile,
        max_count=args.max_rq_annotations,
    )
    write_prominent_high_rq_report(
        results_dir,
        high_rq_rows,
        quantile=args.rq_quantile,
    )
    concept_changes = build_concept_change_rows(results)
    write_concept_change_report(results_dir, concept_changes)

    steps = [row["global_step"] for row in rows]
    scores = [row["score_percent"] for row in rows]
    best = [row["best_score_percent"] for row in rows]

    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.plot(
        steps,
        scores,
        color="#1756d1",
        linestyle="-.",
        marker="o",
        linewidth=2.2,
        markersize=6,
        label="Checkpoint EPS",
        zorder=4,
    )
    ax.scatter(
        steps,
        scores,
        color="black",
        s=28,
        label="Saved-model evals",
        zorder=5,
    )
    ax.step(
        steps,
        best,
        where="post",
        color="red",
        linewidth=2.8,
        label="Best EPS so far",
        zorder=3,
    )
    ax.set_xlabel("Global Training Step (Saved Models)", fontsize=15, weight="bold")
    ax.set_ylabel("Evolved Performance Score (%)", fontsize=15, weight="bold")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(min(steps), max(steps) if max(steps) > min(steps) else min(steps) + 1)
    score_pad = max(3.0, (max(scores) - min(scores)) * 0.18)
    ax.set_ylim(max(0.0, min(scores) - score_pad), min(100.0, max(scores) + score_pad))

    ax2 = ax.twinx()
    visible_events = [event for event in events if event.global_step <= max(steps)]
    if visible_events:
        evo_x = [min(steps)] + [event.global_step for event in visible_events] + [max(steps)]
        evo_y = [0] + [event.cumulative_inner for event in visible_events]
        evo_y.append(evo_y[-1])
        ax2.step(
            evo_x,
            evo_y,
            where="post",
            color="#f39c12",
            linestyle="--",
            linewidth=2.0,
            label="Cumulative inner evolutions",
            zorder=2,
        )
        ax2.set_ylim(0, max(evo_y) * 1.08)
    ax2.set_ylabel(
        "Cumulative Inner Iterations (Evaluated Proposals)",
        fontsize=13,
        color="#b36b00",
        weight="bold",
    )
    ax2.tick_params(axis="y", colors="#b36b00")

    score_by_step = dict(zip(steps, scores))

    # Saved-model callouts describe only how fixed benchmark concepts changed.
    # They intentionally contain no generated-problem or R_Q information.
    for index, change in enumerate(concept_changes):
        checkpoint_step = int(change["checkpoint_step"])
        if checkpoint_step not in score_by_step:
            continue
        changed = sorted(
            (
                item
                for item in change.get("concepts") or []
                if float(item["delta_pp"]) != 0.0
            ),
            key=lambda item: (-abs(float(item["delta_pp"])), item["program_name"]),
        )[:2]
        if not changed:
            label = "All labels unchanged"
        else:
            label = "\n".join(
                f"{_short_concept(item['group'], item['skill'])}: "
                f"{item['previous_accuracy_percent']:.1f}→"
                f"{item['accuracy_percent']:.1f} "
                f"({item['delta_pp']:+.1f}%p)"
                for item in changed
            )
        offset = 46 if index % 2 == 0 else -54
        if score_by_step[checkpoint_step] >= max(scores) - 1.0:
            offset = -58
        horizontal_offset = -48 if checkpoint_step == max(steps) else 0
        ax.annotate(
            label,
            xy=(checkpoint_step, score_by_step[checkpoint_step]),
            xytext=(horizontal_offset, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if offset > 0 else "top",
            fontsize=7.4,
            color="#123d73",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "#eef5ff",
                "edgecolor": "#5b7fa8",
                "alpha": 0.94,
            },
            arrowprops={"arrowstyle": "-", "color": "#333333", "lw": 0.7},
            zorder=6,
        )

    # High-R_Q callouts belong to the evolution timeline, not to checkpoints.
    # Their Y position is cumulative inner evaluations; R_Q is printed only as
    # event metadata so it is not confused with either chart axis.
    visible_high_rq = [
        row
        for row in high_rq_rows
        if min(steps) <= int(row["emerged_global_step"]) <= max(steps)
        and row.get("cumulative_inner") is not None
    ]
    if visible_high_rq:
        ax2.scatter(
            [row["emerged_global_step"] for row in visible_high_rq],
            [row["cumulative_inner"] for row in visible_high_rq],
            marker="D",
            s=42,
            color="#7a4a00",
            edgecolors="white",
            linewidths=0.7,
            label="Prominent high-R_Q events",
            zorder=7,
        )
    for index, row in enumerate(visible_high_rq):
        label = (
            f"R_Q={row['rq_score']:.1f}\n"
            f"{_short_concept(row['group'], row['skill'])}"
        )
        offset = 25 if index % 2 == 0 else -31
        horizontal_offset = -30 if row["emerged_global_step"] == max(steps) else 0
        ax2.annotate(
            label,
            xy=(row["emerged_global_step"], row["cumulative_inner"]),
            xytext=(horizontal_offset, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if offset > 0 else "top",
            fontsize=7.2,
            color="#6a3f00",
            bbox={
                "boxstyle": "round,pad=0.24",
                "facecolor": "#fff5dc",
                "edgecolor": "#b37a19",
                "alpha": 0.95,
            },
            arrowprops={"arrowstyle": "-", "color": "#8a5a0a", "lw": 0.7},
            zorder=8,
        )

    title = args.title or (
        "R-Q-Evolve — Seed-ID Performance Evolution "
        f"({results[0]['num_examples']} problems)"
    )
    ax.set_title(title, fontsize=22, weight="bold", pad=18)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="lower right",
        framealpha=0.92,
        fontsize=11,
    )
    fig.tight_layout()

    png = results_dir / "evolved_performance.png"
    svg = results_dir / "evolved_performance.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"[EPB] wrote {png}")
    print(f"[EPB] wrote {svg}")
    print(f"[EPB] wrote {results_dir / 'trajectory.json'}")
    print(f"[EPB] wrote {results_dir / 'scores.md'}")
    print(f"[EPB] wrote {results_dir / 'high_rq_problems.md'}")
    print(f"[EPB] wrote {results_dir / 'concept_score_changes.md'}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--title")
    parser.add_argument(
        "--max-rq-annotations",
        "--max-problem-annotations",
        dest="max_rq_annotations",
        type=int,
        default=8,
        help="maximum prominent high-R_Q concept callouts on the plot",
    )
    parser.add_argument(
        "--rq-quantile",
        type=float,
        default=0.90,
        help="retain this upper quantile of per-outer R_Q maxima (default: 0.90)",
    )
    return parser


if __name__ == "__main__":
    plot(build_argparser().parse_args())
