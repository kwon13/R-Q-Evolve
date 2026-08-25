#!/usr/bin/env python3
"""Background daemon to automatically merge older FSDP checkpoints to HF format
and clean up their heavy actor/ directories (saving ~40GB per checkpoint).

Rule:
- The LATEST step (e.g. step 64 while step 64 is newest) is KEPT INTACT for resume.
- Any PREVIOUS step (e.g. step 32 when step 64 exists):
    1. If `hf_merged` does not exist, merge FSDP shards to HF format.
    2. Once `hf_merged` is successfully verified, remove the heavy `actor/` directory.

Usage:
    python scripts/auto_merge_checkpoints.py \
        --ckpt_dir rq_output/rq_evolve_4b_8gpu \
        --interval 60

Or run detached under nohup:
    nohup python scripts/auto_merge_checkpoints.py \
        --ckpt_dir rq_output/rq_evolve_4b_8gpu \
        > rq_output/auto_merge.log 2>&1 &
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_step_dirs(ckpt_root: Path) -> list[tuple[int, Path]]:
    """Find all global_step_X directories sorted by step ascending."""
    steps = []
    if not ckpt_root.exists():
        return steps
    for p in ckpt_root.iterdir():
        if p.is_dir() and p.name.startswith("global_step_"):
            m = re.match(r"^global_step_(\d+)$", p.name)
            if m:
                steps.append((int(m.group(1)), p))
    steps.sort(key=lambda t: t[0])
    return steps


def is_merged_valid(hf_dir: Path) -> bool:
    """Check if hf_merged contains valid safetensors and config."""
    if not hf_dir.is_dir():
        return False
    safetensors = list(hf_dir.glob("*.safetensors"))
    config = hf_dir / "config.json"
    return len(safetensors) > 0 and config.exists()


def merge_step(step_dir: Path) -> bool:
    """Run merge_fsdp_to_hf.py for a step."""
    actor_dir = step_dir / "actor"
    hf_dir = step_dir / "hf_merged"

    if not actor_dir.is_dir():
        return False

    print(f"[{time.strftime('%X')}] [auto-merge] Merging {actor_dir} -> {hf_dir} ...", flush=True)
    merge_script = ROOT / "scripts" / "merge_fsdp_to_hf.py"

    cmd = [
        sys.executable,
        str(merge_script),
        "--ckpt_dir", str(actor_dir),
        "--out_dir", str(hf_dir),
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        print(f"[{time.strftime('%X')}] [auto-merge] ERROR merging {step_dir.name}:\n{res.stdout}", flush=True)
        return False

    print(f"[{time.strftime('%X')}] [auto-merge] SUCCESS: {step_dir.name} merged to {hf_dir}", flush=True)
    return True


def clean_actor(step_dir: Path) -> None:
    """Safely remove actor/ directory after verifying hf_merged."""
    actor_dir = step_dir / "actor"
    hf_dir = step_dir / "hf_merged"

    if actor_dir.is_dir() and is_merged_valid(hf_dir):
        print(f"[{time.strftime('%X')}] [auto-merge] Removing heavy actor dir {actor_dir} ...", flush=True)
        shutil.rmtree(actor_dir)
        print(f"[{time.strftime('%X')}] [auto-merge] Cleaned {actor_dir} (~40GB reclaimed)", flush=True)


def process_once(ckpt_root: Path) -> None:
    """Inspect directory and merge/clean eligible checkpoints."""
    steps = parse_step_dirs(ckpt_root)
    if not steps:
        return

    # If only 1 step exists, it is the newest and active; do not touch.
    # If >= 2 steps exist, all except the newest can be merged and cleaned.
    latest_step, _ = steps[-1]

    for step_num, step_dir in steps[:-1]:
        actor_dir = step_dir / "actor"
        hf_dir = step_dir / "hf_merged"

        # 1. Merge if not merged yet
        if actor_dir.is_dir() and not is_merged_valid(hf_dir):
            success = merge_step(step_dir)
            if success:
                clean_actor(step_dir)
        # 2. If already merged but actor/ still exists, clean it
        elif actor_dir.is_dir() and is_merged_valid(hf_dir):
            clean_actor(step_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="rq_output/rq_evolve_4b_8gpu", help="Path to checkpoint root")
    ap.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    args = ap.parse_args()

    ckpt_root = Path(args.ckpt_dir).resolve()
    print(f"[{time.strftime('%X')}] [auto-merge] Watching {ckpt_root} every {args.interval}s...", flush=True)

    while True:
        try:
            process_once(ckpt_root)
        except Exception as e:
            print(f"[{time.strftime('%X')}] [auto-merge] Unexpected error: {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
