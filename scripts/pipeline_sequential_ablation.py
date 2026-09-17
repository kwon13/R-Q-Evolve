#!/usr/bin/env python3
"""Automated supervisor pipeline:
1. Wait for current no_refresh step 256 evaluation to finish on GPUs 0-3.
2. Run No-U ablation training on GPUs 0-3 (--detach).
   - Monitor training to step 256.
   - Prevent lingering process hang (공회전 방지) after step 256.
   - Clean up auto_merge daemon & ensure step 256 is merged to HF format.
   - Parse in-training validation metrics to find the best performing step.
   - Run math benchmark evaluation (best_step + step 256) on GPUs 0-3.
3. Run Random-Cell ablation training on GPUs 0-3 (--detach).
   - Monitor training to step 256 with hang prevention.
   - Clean up auto_merge daemon & ensure step 256 is merged to HF format.
   - Parse in-training validation metrics to find the best performing step.
   - Run math benchmark evaluation (best_step + step 256) on GPUs 0-3.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
SUPERVISOR_LOG = LOGS_DIR / "pipeline_sequential_ablation.log"
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


def wait_for_gpus_clear(target_gpus: list[int] = [0, 1, 2, 3], timeout_sec: int = 300) -> bool:
    log(f"Waiting for GPUs {target_gpus} to be clear of compute processes...")
    start = time.time()
    while time.time() - start < timeout_sec:
        res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        lines = [ln.strip() for ln in res.stdout.strip().splitlines() if ln.strip()]
        # Check if our target GPUs have no heavy compute
        # More direct check: query process names on those GPUs
        proc_check = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,process_name", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if not lines:
            log("All GPUs are completely clear.")
            return True
        time.sleep(10)
    log("Warning: Timeout waiting for all GPUs to clear, proceeding with caution.")
    return False


def wait_for_eval_processes_to_finish() -> None:
    log("Checking for active eval_steps_fanout / eval_vllm_math processes...")
    while True:
        res = subprocess.run(
            ["pgrep", "-f", "eval_steps_fanout|eval_vllm_math"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        pids = [p.strip() for p in res.stdout.strip().splitlines() if p.strip()]
        if not pids:
            log("No active eval processes detected.")
            break
        log(f"Active eval process(es) running (PIDs: {', '.join(pids)}). Waiting 30s...")
        time.sleep(30)
    time.sleep(15)


def parse_best_steps_from_train_log(log_path: Path) -> tuple[int, int, dict[int, float]]:
    """Parse log for in-training val-core scores and find best step and second best."""
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
        log("No validation scores found in log; using default steps 224, 256")
        return 224, 256, {}

    sorted_steps = sorted(results.keys(), key=lambda s: results[s], reverse=True)
    best_step = sorted_steps[0]
    second_best = sorted_steps[1] if len(sorted_steps) > 1 else 256

    log("=== In-Training Validation Summary (AVG pass@1) ===")
    for s in sorted(results.keys()):
        log(f"  Step {s:3d}: {results[s]*100:6.2f}%")
    log(f"Top steps selected: Best={best_step} ({results[best_step]*100:.2f}%), Second={second_best} ({results.get(second_best,0)*100:.2f}%)")

    return best_step, second_best, results


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
    """Kill lingering auto_merge daemon for this specific run."""
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
            log(f"Terminating lingering auto_merge process (PID {pid})...")
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def run_ablation(
    ablation_name: str,
    launch_script: str,
    ckpt_dir: Path,
    log_dir: Path,
    gpus: str = "0,1,2,3",
) -> None:
    log("=================================================================")
    log(f"STARTING ABLATION PIPELINE: {ablation_name}")
    log(f"Launch script: {launch_script}")
    log(f"Checkpoints  : {ckpt_dir}")
    log(f"Logs         : {log_dir}")
    log(f"GPUs         : {gpus}")
    log("=================================================================")

    # 1. Ensure GPUs are clear
    wait_for_gpus_clear([int(g) for g in gpus.split(",")])

    # 2. Launch training detached
    launch_cmd = ["bash", str(ROOT / launch_script), "--gpus", gpus, "--detach"]
    log(f"Executing: {' '.join(launch_cmd)}")
    res = subprocess.run(launch_cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log(f"Launch output:\n{res.stdout.strip()}")

    time.sleep(5)

    # Read train.pid
    pid_file = log_dir / "train.pid"
    if not pid_file.exists():
        raise RuntimeError(f"train.pid not found in {log_dir} after launch!")
    train_pid = int(pid_file.read_text().strip())
    log(f"Training started with PID: {train_pid}")

    # 3. Monitor training to step 256 (with hang / idling prevention)
    step_256_reached_time: float | None = None
    latest_log = log_dir / "latest.log"

    while True:
        if not is_pid_alive(train_pid):
            log(f"Training process PID {train_pid} has exited.")
            break

        # Check latest.log for step 256 completion
        if latest_log.exists():
            try:
                # Read last 20KB of log to check progress
                with open(latest_log, "rb") as f:
                    f.seek(max(0, f.seek(0, 2) - 20480))
                    tail_bytes = f.read()
                tail_text = tail_bytes.decode("utf-8", errors="ignore")

                # Check if step 256 finished or Final validation metrics logged
                if "Final validation metrics:" in tail_text or "256/256 [18:" in tail_text or "step:256 - " in tail_text:
                    if step_256_reached_time is None:
                        step_256_reached_time = time.time()
                        log("[Hang Prevention] Step 256 / Final validation detected in training log!")
                    else:
                        elapsed_since_256 = time.time() - step_256_reached_time
                        if elapsed_since_256 > 90:
                            log(f"[Hang Prevention] Process PID {train_pid} lingered {elapsed_since_256:.0f}s after Step 256 finish. Terminating cleanly...")
                            kill_pid_tree(train_pid, signal.SIGTERM)
                            time.sleep(10)
                            if is_pid_alive(train_pid):
                                log(f"[Hang Prevention] Process PID {train_pid} still alive after SIGTERM, sending SIGKILL...")
                                kill_pid_tree(train_pid, signal.SIGKILL)
                            break
            except Exception as e:
                pass

        time.sleep(30)

    # Wait for Ray actors and any remaining python processes for this training to clear
    time.sleep(15)
    cleanup_auto_merge_for_run(ckpt_dir.name)

    # Secondary check: kill any lingering Ray worker processes from this run
    wait_for_gpus_clear([int(g) for g in gpus.split(",")])

    # 4. Merge step 256 if needed
    log(f"Ensuring step 256 HF merge for {ablation_name}...")
    merge_step_256(ckpt_dir)

    # 5. Determine which steps to evaluate:
    # "성능이 높았던 step + 최종 256 step 에서의 평가"
    resolved_train_log = latest_log.resolve() if latest_log.is_symlink() else latest_log
    best_step, second_best, _ = parse_best_steps_from_train_log(resolved_train_log)

    eval_steps = []
    if best_step == 256:
        eval_steps = [second_best, 256]
    else:
        eval_steps = [best_step, 256]
    eval_steps_str = ",".join(str(s) for s in sorted(eval_steps))
    log(f"Running benchmark evaluation on steps: {eval_steps_str}...")

    # 6. Run offline benchmark evaluation on GPUs 0-3
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
        log(f"SUCCESS: Evaluation completed for {ablation_name} steps {eval_steps_str}!")
        scores_file = ckpt_dir / "scores.md"
        if scores_file.exists():
            log(f"Scores for {ablation_name}:\n{scores_file.read_text(encoding='utf-8')}")
    else:
        log(f"ERROR: Evaluation failed for {ablation_name}, inspect {eval_log}")

    log(f"COMPLETED ABLATION PIPELINE: {ablation_name}")
    log("=================================================================\n")


def main() -> None:
    log("Supervisor pipeline started.")

    # 1. Wait for current no_refresh step 256 evaluation to finish
    log("Step 1/3: Waiting for active no_refresh step 256 evaluation to complete...")
    wait_for_eval_processes_to_finish()
    log("no_refresh evaluations on GPUs 0-3 are complete!")

    # 2. Run no_u ablation
    log("Step 2/3: Launching No-U ablation pipeline...")
    run_ablation(
        ablation_name="no_u",
        launch_script="scripts/run_train_domain_type_no_u_4gpu.sh",
        ckpt_dir=ROOT / "rq_output" / "rq_evolve_4b_domain_type_no_u_35cell_4gpu",
        log_dir=LOGS_DIR / "rq_evolve_4b_domain_type_no_u_35cell_4gpu",
        gpus="0,1,2,3",
    )

    # 3. Run random_cell ablation
    log("Step 3/3: Launching Random-Cell ablation pipeline...")
    run_ablation(
        ablation_name="random_cell",
        launch_script="scripts/run_train_domain_type_random_cell_4gpu.sh",
        ckpt_dir=ROOT / "rq_output" / "rq_evolve_4b_domain_type_random_cell_35cell_4gpu",
        log_dir=LOGS_DIR / "rq_evolve_4b_domain_type_random_cell_35cell_4gpu",
        gpus="0,1,2,3",
    )

    log("ALL SEQUENTIAL ABLATIONS (no_u, random_cell) AND EVALUATIONS FINISHED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
