#!/usr/bin/env python3
"""Download, stitch, and plot 4B domain-type W&B training dynamics.

The production Standard and Reverse-U experiments were continued in new W&B
runs.  This script encodes the checkpoint lineage explicitly instead of
selecting a run by display name (which is ambiguous for the Standard arm).

Example:
    python scripts/plot_wandb_domain_type_training_dynamics.py \
        --env-file /data1/yhoon113/.env
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import wandb


ENTITY = "yhoon113-None"
PROJECT = "rq_evolve"
WANDB_PATH = f"{ENTITY}/{PROJECT}"

STEP_KEY = "training/global_step"
ENTROPY_KEY = "actor/entropy"
GRAD_NORM_KEY = "actor/grad_norm"
HISTORY_KEYS = [STEP_KEY, ENTROPY_KEY, GRAD_NORM_KEY, "_step", "_timestamp"]


@dataclass(frozen=True)
class Segment:
    run_id: str
    first_step: int
    last_step: int
    role: str


@dataclass(frozen=True)
class Arm:
    key: str
    display_name: str
    wandb_name: str
    color: str
    segments: tuple[Segment, ...]
    allowed_missing_steps: tuple[int, ...] = ()


ARMS = (
    Arm(
        key="standard_8gpu",
        display_name="Standard (8 GPUs)",
        wandb_name="qwen3_4b_domain_type_35cell_8gpu",
        color="#1768AC",
        segments=(
            Segment("ssdiszis", 1, 160, "final-lineage source run"),
            Segment("ghxzxptj", 161, 256, "resume from checkpoint step 160"),
        ),
    ),
    Arm(
        key="reverse_u_4gpu",
        display_name="Reverse-U (4 GPUs)",
        wandb_name="qwen3_4b_domain_type_reverse_u_35cell_4gpu",
        color="#D1495B",
        segments=(
            Segment("wrqdvu08", 1, 96, "source run before restart"),
            Segment("5ixajkvo", 97, 144, "resume from checkpoint step 96"),
        ),
        # The step-96 checkpoint exists, but no step-96 metric row was logged.
        allowed_missing_steps=(96,),
    ),
    Arm(
        key="no_u_4gpu",
        display_name="No-U (4 GPUs)",
        wandb_name="qwen3_4b_domain_type_no_u_35cell_4gpu",
        color="#2A9D8F",
        segments=(
            Segment("n3sojj6k", 1, 128, "single source run"),
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional .env containing WANDB_API_KEY; the key is never printed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/wandb_domain_type_training_dynamics"),
    )
    parser.add_argument("--max-step", type=int, default=128)
    parser.add_argument("--smooth-window", type=int, default=7)
    return parser.parse_args()


def load_wandb_key(env_file: Path | None) -> None:
    """Load only WANDB_API_KEY from an env file without echoing its value."""
    if os.environ.get("WANDB_API_KEY") or env_file is None:
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != "WANDB_API_KEY":
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ["WANDB_API_KEY"] = value
        return
    raise RuntimeError(f"WANDB_API_KEY was not found in {env_file}")


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def fetch_segment(api: wandb.Api, arm: Arm, segment: Segment) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run = api.run(f"{WANDB_PATH}/{segment.run_id}")
    if run.name != arm.wandb_name:
        raise RuntimeError(
            f"run {segment.run_id} has name {run.name!r}, expected {arm.wandb_name!r}"
        )

    rows: list[dict[str, Any]] = []
    for history_row in run.scan_history(keys=HISTORY_KEYS, page_size=1_000):
        raw_step = history_row.get(STEP_KEY)
        if raw_step is None:
            continue
        step = int(raw_step)
        if not segment.first_step <= step <= segment.last_step:
            continue
        entropy = finite_float(history_row.get(ENTROPY_KEY))
        grad_norm = finite_float(history_row.get(GRAD_NORM_KEY))
        if entropy is None and grad_norm is None:
            continue
        rows.append(
            {
                "arm": arm.key,
                "display_name": arm.display_name,
                "step": step,
                "actor_entropy": entropy,
                "actor_grad_norm": grad_norm,
                "source_run_id": run.id,
                "source_run_url": run.url,
                "source_wandb_step": history_row.get("_step"),
                "source_timestamp": history_row.get("_timestamp"),
            }
        )

    metadata = {
        "run_id": run.id,
        "run_name": run.name,
        "url": run.url,
        "state": run.state,
        "created_at": run.created_at,
        "segment_first_step": segment.first_step,
        "segment_last_step": segment.last_step,
        "role": segment.role,
        "downloaded_metric_rows": len(rows),
    }
    return rows, metadata


def centered_rolling_by_contiguous_segment(series: pd.Series, window: int) -> pd.Series:
    present = series.notna()
    groups = present.ne(present.shift(fill_value=False)).cumsum()
    smoothed = series.groupby(groups).transform(
        lambda part: part.rolling(window=window, center=True, min_periods=1).mean()
    )
    return smoothed.where(present)


def fetch_and_stitch(
    api: wandb.Api, max_step: int, smooth_window: int
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, list[int]]]:
    stitched_frames: list[pd.DataFrame] = []
    source_runs: list[dict[str, Any]] = []
    missing_by_arm: dict[str, list[int]] = {}

    for arm in ARMS:
        rows_by_step: dict[int, dict[str, Any]] = {}
        for segment in arm.segments:
            rows, metadata = fetch_segment(api, arm, segment)
            source_runs.append({"arm": arm.key, **metadata})
            for row in rows:
                step = int(row["step"])
                if step in rows_by_step:
                    raise RuntimeError(
                        f"overlapping lineage segments for {arm.key} at step {step}"
                    )
                rows_by_step[step] = row

        ordered_rows: list[dict[str, Any]] = []
        for step in range(1, max_step + 1):
            row = rows_by_step.get(step)
            if row is None:
                row = {
                    "arm": arm.key,
                    "display_name": arm.display_name,
                    "step": step,
                    "actor_entropy": None,
                    "actor_grad_norm": None,
                    "source_run_id": None,
                    "source_run_url": None,
                    "source_wandb_step": None,
                    "source_timestamp": None,
                }
            ordered_rows.append(row)

        frame = pd.DataFrame(ordered_rows)
        missing = frame.loc[
            frame[["actor_entropy", "actor_grad_norm"]].isna().any(axis=1), "step"
        ].astype(int).tolist()
        unexpected = sorted(set(missing) - set(arm.allowed_missing_steps))
        if unexpected:
            raise RuntimeError(f"unexpected missing steps for {arm.key}: {unexpected}")
        missing_by_arm[arm.key] = missing

        frame["actor_entropy_smooth"] = centered_rolling_by_contiguous_segment(
            frame["actor_entropy"], smooth_window
        )
        frame["actor_grad_norm_smooth"] = centered_rolling_by_contiguous_segment(
            frame["actor_grad_norm"], smooth_window
        )
        stitched_frames.append(frame)

    return pd.concat(stitched_frames, ignore_index=True), source_runs, missing_by_arm


def discover_candidate_runs(api: wandb.Api) -> list[dict[str, Any]]:
    filters = {"$or": [{"display_name": arm.wandb_name} for arm in ARMS]}
    runs = list(api.runs(WANDB_PATH, filters=filters, per_page=100))
    return [
        {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "created_at": run.created_at,
            "url": run.url,
            "selected_for_lineage": any(
                run.id == segment.run_id for arm in ARMS for segment in arm.segments
            ),
        }
        for run in sorted(runs, key=lambda item: item.created_at)
    ]


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.65,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_metrics(data: pd.DataFrame, max_step: int, smooth_window: int, output_dir: Path) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 6.8), sharex="col")
    metrics = (
        ("actor_entropy", "actor_entropy_smooth", "Actor entropy"),
        ("actor_grad_norm", "actor_grad_norm_smooth", "Actor gradient norm"),
    )

    standard = data[data["arm"] == "standard_8gpu"].set_index("step")
    comparison_arms = [arm for arm in ARMS if arm.key != "standard_8gpu"]

    for column, (_raw_key, smooth_key, y_label) in enumerate(metrics):
        absolute_ax = axes[0, column]
        delta_ax = axes[1, column]

        # Light alternating round bands make the 32-step structure visible
        # without competing with the curves.
        for round_index, first_step in enumerate(range(1, max_step + 1, 32), start=1):
            last_step = min(first_step + 31, max_step)
            if round_index % 2 == 0:
                for ax in (absolute_ax, delta_ax):
                    ax.axvspan(first_step, last_step, color="#6C757D", alpha=0.045, zorder=0)

        for arm in ARMS:
            frame = data[data["arm"] == arm.key]
            absolute_ax.plot(
                frame["step"],
                frame[smooth_key],
                color=arm.color,
                linewidth=2.6,
                zorder=3,
            )

        standard_values = standard[smooth_key]
        max_abs_delta = 0.0
        for arm in comparison_arms:
            frame = data[data["arm"] == arm.key].set_index("step")
            delta = frame[smooth_key] - standard_values
            delta_ax.plot(
                delta.index,
                delta,
                color=arm.color,
                linewidth=2.25,
                zorder=3,
            )
            delta_ax.fill_between(
                delta.index,
                0,
                delta,
                where=delta.notna(),
                color=arm.color,
                alpha=0.10,
                interpolate=False,
                zorder=1,
            )
            finite_delta = delta.dropna().abs()
            if not finite_delta.empty:
                max_abs_delta = max(max_abs_delta, float(finite_delta.max()))

        delta_limit = max_abs_delta * 1.12 if max_abs_delta > 0 else 1.0
        delta_ax.set_ylim(-delta_limit, delta_limit)
        delta_ax.axhline(0, color="#343A40", linewidth=1.1, alpha=0.8, zorder=2)

        for boundary in range(32, max_step + 1, 32):
            for ax in (absolute_ax, delta_ax):
                ax.axvline(
                    boundary,
                    color="#6C757D",
                    linestyle=":",
                    linewidth=0.85,
                    alpha=0.65,
                    zorder=1,
                )

        absolute_ax.set_title(y_label, fontweight="semibold", pad=9)
        absolute_ax.set_ylabel("Absolute value")
        delta_ax.set_ylabel("Difference vs Standard")
        delta_ax.set_xlabel("Training global step")
        for ax in (absolute_ax, delta_ax):
            ax.margins(x=0.01)
            ax.set_xlim(1, max_step)
            ax.set_xticks([1] + list(range(32, max_step + 1, 32)))

    if max_step >= 97:
        for ax in axes.flat:
            ax.axvline(96.5, color="#D1495B", linestyle="--", linewidth=1.0, alpha=0.8)
        axes[0, 0].annotate(
            "Reverse-U resumed",
            xy=(96.5, axes[0, 0].get_ylim()[1]),
            xytext=(4, -5),
            textcoords="offset points",
            ha="left",
            va="top",
            color="#A12D3D",
            fontsize=8.5,
        )

    fig.suptitle(f"R-Q Evolve 4B: Entropy and Gradient-Norm Dynamics", y=0.992, fontsize=16)
    fig.text(
        0.5,
        0.948,
        f"Steps 1–{max_step} · centered {smooth_window}-step mean · lower panels show signed differences",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#495057",
    )

    legend_handles = [
        Line2D([0], [0], color=arm.color, linewidth=2.3, label=arm.display_name)
        for arm in ARMS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=3,
        columnspacing=1.7,
        handlelength=2.5,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.865), h_pad=1.35, w_pad=2.0)
    stem = output_dir / "domain_type_entropy_grad_norm_steps_001_128"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.max_step < 1:
        raise ValueError("--max-step must be positive")
    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be positive")
    load_wandb_key(args.env_file)
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is not set; pass --env-file or export it")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=180)
    data, source_runs, missing_by_arm = fetch_and_stitch(
        api, args.max_step, args.smooth_window
    )
    candidates = discover_candidate_runs(api)

    csv_path = output_dir / "stitched_training_metrics_steps_001_128.csv"
    data.to_csv(csv_path, index=False, float_format="%.10g")
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wandb_path": WANDB_PATH,
        "max_step": args.max_step,
        "metric_keys": {
            "step": STEP_KEY,
            "entropy": ENTROPY_KEY,
            "gradient_norm": GRAD_NORM_KEY,
        },
        "smoothing": {
            "method": "centered rolling arithmetic mean within contiguous segments",
            "window_steps": args.smooth_window,
            "raw_values_are_also_plotted": True,
        },
        "source_runs": source_runs,
        "missing_steps": missing_by_arm,
        "candidate_runs_with_matching_names": candidates,
        "merge_policy": (
            "Use only the explicitly declared checkpoint lineage. Segment ranges are "
            "non-overlapping; failed/restarted runs outside that lineage are retained "
            "in the manifest but excluded from the plotted data. Missing metrics are "
            "not interpolated."
        ),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plot_metrics(data, args.max_step, args.smooth_window, output_dir)

    print(f"wrote {csv_path}")
    print(f"wrote {output_dir / 'run_manifest.json'}")
    print(f"wrote figure files under {output_dir}")
    print(f"missing steps: {missing_by_arm}")


if __name__ == "__main__":
    main()
