#!/usr/bin/env python3
"""Supervisor for parallel runs:
- Monitors no_u on GPUs 0-3 (PID 2525210)
- Monitors random_cell on GPUs 4-7 (PID 2533914)
When each completes step 256:
- Ensures clean shutdown and merges step 256 to HF format.
- Parses in-training validation metrics to find the top step + step 256.
- Automatically launches evaluation on its respective GPUs.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
SUPERVISOR_LOG = LOGS_DIR / "monitor_and_eval_parallel.log"
PYTHON_BIN = Path("/data1/yhoon113/miniforge3/envs/vllm-g4/bin/python")

BENCHES = ["math500", "amc23", "aime24", "aime25", "minerva_math", "olympiadbench"]


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(SUPERVISOR_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_pid_tree(pid: int, sig: signal.Signals = signal.SIGTERM) -> None:
    try:
        os.kill(pid, sig)
    except OSError:
        pass


def merge_step_256(ckpt_root: Path) -> None:
    actor_dir = ckpt_root / "global_step_256" / "actor"
    hf_dir = ckpt_root / "global_step_256" / "hf_merged"
    if (hf_dir / "config.json").exists() and list(hf_dir.glob("*.safetensors")):
        log(f"Step 256 is already merged: {hf_dir}")
        return

    if not actor_dir.exists():
        log(f"Warning: {actor_dir} not found for step 256 merge")
        return

    log(f"Merging step 256: {actor_dir} -> {hf_dir} ...")
    cmd = [
        str(PYTHON_BIN),
        str(ROOT / "scripts" / "merge_fsdp_to_hf.py"),
        "--ckpt_dir", str(actor_dir),
        "--out_dir", str(hf_dir),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode == 0:
        log(f"SUCCESS: Step 256 merged to {hf_dir}")
    else:
        log(f"ERROR: Step 256 merge failed:\n{res.stdout}")


def cleanup_auto_merge_for_run(run_dir_name: str) -> None:
    res = subprocess.run(
        ["pgrep", "-f", f"auto_merge_checkpoints.py.*{run_dir_name}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pids = [p.strip() for p in res.stdout.strip().splitlines() if p.strip()]
    for p in pids:
        try:
            pid = int(p)
            log(f"Terminating auto_merge process (PID {pid}) for {run_dir_name}...")
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def parse_best_steps_from_train_log(log_path: Path) -> tuple[int, int, dict[int, float]]:
    results: dict[int, float] = {}
    if not log_path.exists():
        log(f"Warning: train log {log_path} not found; falling back to steps 224, 256")
        return 224, 256, {}

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"^step:(\d+)\s+-\s+", line)
            if not m:
                continue
            step = int(m.group(1))
            if "val-core/math500/acc/mean@1" in line:
                scores = []
                for b in BENCHES:
                    pat = rf"val-core/{b}/acc/mean@1:np\.float64\(([^)]+)\)"
                    bm = re.search(pat, line)
                    if bm:
                        scores.append(float(bm.group(1)))
                if scores:
                    results[step] = sum(scores) / len(scores)

    if not results:
        log(f"No validation scores found in {log_path}; using default steps 224, 256")
        return 224, 256, {}

    sorted_steps = sorted(results.keys(), key=lambda s: results[s], reverse=True)
    best_step = sorted_steps[0]
    second_best = sorted_steps[1] if len(sorted_steps) > 1 else 256

    log(f"=== [{log_path.name}] In-Training Validation Summary (AVG pass@1) ===")
    for s in sorted(results.keys()):
        log(f"  Step {s:3d}: {results[s]*100:6.2f}%")
    log(f"Top steps selected: Best={best_step} ({results[best_step]*100:.2f}%), Second={second_best} ({results.get(second_best,0)*100:.2f}%)")

    return best_step, second_best, results


def monitor_and_eval_run(name: str, train_pid: int, ckpt_dir: Path, log_dir: Path, gpus: str) -> None:
    log(f"[{name}] Starting monitor thread for PID {train_pid} on GPUs {gpus}...")
    latest_log = log_dir / "latest.log"
    step_256_reached_time: float | None = None

    while True:
        if not is_pid_alive(train_pid):
            log(f"[{name}] Training process PID {train_pid} has exited.")
            break

        if latest_log.exists():
            try:
                with open(latest_log, "rb") as f:
                    f.seek(max(0, f.seek(0, 2) - 20480))
                    tail_text = f.read().decode("utf-8", errors="ignore")

                if "Final validation metrics:" in tail_text or "256/256 [18:" in tail_text or "step:256 - " in tail_text:
                    if step_256_reached_time is None:
                        step_256_reached_time = time.time()
                        log(f"[{name}] [Hang Prevention] Step 256 / Final validation detected!")
                    else:
                        elapsed = time.time() - step_256_reached_time
                        if elapsed > 90:
                            log(f"[{name}] [Hang Prevention] Process PID {train_pid} lingered {elapsed:.0f}s after Step 256 finish. Terminating cleanly...")
                            kill_pid_tree(train_pid, signal.SIGTERM)
                            time.sleep(10)
                            if is_pid_alive(train_pid):
                                kill_pid_tree(train_pid, signal.SIGKILL)
                            break
            except Exception:
                pass

        time.sleep(30)

    time.sleep(15)
    cleanup_auto_merge_for_run(ckpt_dir.name)

    # Merge step 256
    merge_step_256(ckpt_dir)

    # Determine steps to evaluate
    resolved_train_log = latest_log.resolve() if latest_log.is_symlink() else latest_log
    best_step, second_best, _ = parse_best_steps_from_train_log(resolved_train_log)

    eval_steps = [best_step, 256] if best_step != 256 else [second_best, 256]
    eval_steps_str = ",".join(str(s) for s in sorted(eval_steps))
    log(f"[{name}] Running benchmark evaluation on steps: {eval_steps_str} on GPUs {gpus}...")

    eval_log = log_dir / f"eval_post_train_steps_{eval_steps_str.replace(',', '_')}.log"
    env = dict(os.environ)
    env["BASE"] = str(ckpt_dir)
    env["STEPS_LIST"] = eval_steps_str
    env["GPU_LIST"] = gpus

    with open(eval_log, "w", encoding="utf-8") as out_f:
        eval_proc = subprocess.run(
            ["bash", str(ROOT / "scripts" / "eval_steps_fanout.sh")],
            cwd=ROOT,
            env=env,
            stdout=out_f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if eval_proc.returncode == 0:
        log(f"[{name}] SUCCESS: Evaluation completed for steps {eval_steps_str}!")
        scores_file = ckpt_dir / "scores.md"
        if scores_file.exists():
            log(f"[{name}] Final Scores:\n{scores_file.read_text(encoding='utf-8')}")
    else:
        log(f"[{name}] ERROR: Evaluation failed, check {eval_log}")

    log(f"[{name}] ALL WORK COMPLETE!")


def main() -> None:
    log("Parallel supervisor started.")

    t1 = threading.Thread(
        target=monitor_and_eval_run,
        args=(
            "no_u",
            2525210,
            ROOT / "rq_output" / "rq_evolve_4b_domain_type_no_u_35cell_4gpu",
            LOGS_DIR / "rq_evolve_4b_domain_type_no_u_35cell_4gpu",
            "0,1,2,3",
        ),
    )
    t2 = threading.Thread(
        target=monitor_and_eval_run,
        args=(
            "random_cell",
            2533914,
            ROOT / "rq_output" / "rq_evolve_4b_domain_type_random_cell_35cell_4gpu",
            LOGS_DIR / "rq_evolve_4b_domain_type_random_cell_35cell_4gpu",
            "4,5,6,7",
        ),
    )

    t1.start()
    t2.start()

    t1.join()
    t2.join()
    log("ALL RUNS (no_u, random_cell) FINISHED!")


if __name__ == "__main__":
    main()
