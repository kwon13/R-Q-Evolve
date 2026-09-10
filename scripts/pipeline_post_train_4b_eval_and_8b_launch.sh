#!/usr/bin/env bash
# Automated pipeline:
# 1. Wait for 4B Epsilon-greedy training (PID 1454525 / global_step_256) to finish.
# 2. Verify / merge any remaining checkpoints (steps 32..256) to hf_merged format.
# 3. Run full Math benchmark evaluation on all steps (7 benchmarks, R-Zero aligned with GPT-4o recheck).
# 4. Run full General benchmark evaluation on all steps (MMLU-Pro, SuperGPQA, BBEH).
# 5. Clear and verify GPUs.
# 6. Launch 8B Epsilon-greedy training on 8 GPUs (--profile a100 --model-size 8b --detach).
set -uo pipefail

RQ="/data1/yhoon113/R-Q-Evolve"
cd "$RQ"

BASE_4B="$RQ/rq_output/rq_evolve_4b_domain_type_epsilon_0.25_35cell_8gpu_a100"
LOG_DIR="$RQ/logs"
mkdir -p "$LOG_DIR"
PIPELINE_LOG="$LOG_DIR/pipeline_post_train_4b_eval_and_8b_launch.log"
TRAIN_PID="1454525"
STEPS=(32 64 96 128 160 192 224 256)
STEPS_STR="32,64,96,128,160,192,224,256"
GPUS="0,1,2,3,4,5,6,7"

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg"
  echo "$msg" >> "$PIPELINE_LOG"
}

log "================================================================="
log "[pipeline] Starting automated post-training supervisor pipeline."
log "[pipeline] Target 4B run: $BASE_4B"
log "[pipeline] Target steps: $STEPS_STR"
log "[pipeline] Log file: $PIPELINE_LOG"
log "================================================================="

# -----------------------------------------------------------------------------
# Stage 1: Wait for 4B training to finish
# -----------------------------------------------------------------------------
log "[Stage 1] Monitoring 4B training process (PID $TRAIN_PID)..."
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 60
done

# Secondary check: ensure python training process has truly exited
while pgrep -f "train_with_verl.py.*configs/rq_evolve_4b_8gpu_domain_type_epsilon.yaml" >/dev/null 2>&1; do
  log "[Stage 1] Training PID exited, waiting for remaining trainer cleanup..."
  sleep 30
done

log "[Stage 1] 4B training process completed."
sleep 30

# -----------------------------------------------------------------------------
# Stage 2: Merge any unmerged checkpoints (especially global_step_256)
# -----------------------------------------------------------------------------
log "[Stage 2] Verifying hf_merged checkpoints for steps: ${STEPS[*]}..."
source /data1/yhoon113/miniforge3/etc/profile.d/conda.sh
set +u
conda activate vllm-g4
set -u
export CUDA_HOME=/data1/yhoon113/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export LD_PRELOAD="${CONDA_PREFIX}/lib/libgomp.so.1"
PY="${CONDA_PREFIX}/bin/python"

for s in "${STEPS[@]}"; do
  ckpt="$BASE_4B/global_step_$s"
  hf="$ckpt/hf_merged"
  if [[ -f "$hf/config.json" ]] && compgen -G "$hf/*.safetensors" >/dev/null 2>&1; then
    log "[Stage 2] Step $s: already merged ($hf)"
  else
    if [[ -d "$ckpt/actor" ]]; then
      log "[Stage 2] Step $s: merging $ckpt/actor -> $hf ..."
      mkdir -p "$ckpt/eval/logs"
      "$PY" "$RQ/scripts/merge_fsdp_to_hf.py" --ckpt_dir "$ckpt/actor" --out_dir "$hf" >> "$PIPELINE_LOG" 2>&1
      log "[Stage 2] Step $s: merge complete."
    else
      log "[Stage 2] WARNING: Neither $hf nor $ckpt/actor exists for step $s!"
    fi
  fi
done

# Wait for auto_merge daemon to finish any in-flight task
sleep 20

# -----------------------------------------------------------------------------
# Stage 3: Wait for GPUs to be completely idle
# -----------------------------------------------------------------------------
log "[Stage 3] Checking GPU status before evaluation..."
while true; do
  active_compute=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -v "^$" | wc -l)
  if [[ "$active_compute" -eq 0 ]]; then
    log "[Stage 3] All GPUs are clear of compute processes."
    break
  fi
  log "[Stage 3] $active_compute compute processes still active. Waiting 30s..."
  sleep 30
done

# -----------------------------------------------------------------------------
# Stage 4: Run Math Benchmark Evaluation on all steps
# -----------------------------------------------------------------------------
log "[Stage 4] Starting Math evaluation (7 benchmarks x ${#STEPS[@]} steps)..."
log "[Stage 4] Benchmarks: math500, gsm8k, amc23, aime24, aime25, minerva_math, olympiadbench"
BASE="$BASE_4B" GPU_LIST="$GPUS" STEPS_LIST="$STEPS_STR" bash "$RQ/scripts/eval_steps_fanout.sh" >> "$PIPELINE_LOG" 2>&1

if [[ -f "$BASE_4B/scores.md" ]]; then
  log "[Stage 4] Math evaluation successfully completed! Scores written to $BASE_4B/scores.md"
else
  log "[Stage 4] WARNING: $BASE_4B/scores.md not found. Check logs for details."
fi

sleep 30

# -----------------------------------------------------------------------------
# Stage 5: Run General Benchmark Evaluation on all steps
# -----------------------------------------------------------------------------
log "[Stage 5] Starting General evaluation (3 benchmarks x ${#STEPS[@]} steps)..."
log "[Stage 5] Benchmarks: mmlupro, supergpqa, bbeh (MAX_SAMPLES=1000)"
BASE="$BASE_4B" GPU_LIST="$GPUS" STEPS_LIST="$STEPS_STR" MAX_SAMPLES=1000 bash "$RQ/scripts/eval_general_fanout.sh" >> "$PIPELINE_LOG" 2>&1

if [[ -f "$BASE_4B/scores_general.md" ]]; then
  log "[Stage 5] General evaluation successfully completed! Scores written to $BASE_4B/scores_general.md"
else
  log "[Stage 5] WARNING: $BASE_4B/scores_general.md not found. Check logs for details."
fi

sleep 30

# -----------------------------------------------------------------------------
# Stage 6: Final GPU Cleanup
# -----------------------------------------------------------------------------
log "[Stage 6] Cleaning up GPUs for 8B training launch..."
# Ensure all background evaluation processes have exited
pkill -f "eval_vllm_math.py" 2>/dev/null || true
pkill -f "eval_general_vllm.py" 2>/dev/null || true
sleep 15

while true; do
  active_compute=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -v "^$" | wc -l)
  if [[ "$active_compute" -eq 0 ]]; then
    log "[Stage 6] All 8 GPUs verified 100% idle."
    break
  fi
  log "[Stage 6] $active_compute compute processes still active. Waiting 15s..."
  sleep 15
done

# -----------------------------------------------------------------------------
# Stage 7: Launch 8B Epsilon Training
# -----------------------------------------------------------------------------
log "[Stage 7] Launching 8B Epsilon training run on 8 GPUs..."
log "[Stage 7] Command: bash scripts/run_train_domain_type_epsilon_8gpu.sh --profile a100 --model-size 8b --gpus $GPUS --detach"

bash "$RQ/scripts/run_train_domain_type_epsilon_8gpu.sh" \
  --profile a100 \
  --model-size 8b \
  --gpus "$GPUS" \
  --detach >> "$PIPELINE_LOG" 2>&1

log "[Stage 7] 8B training launched in detached mode!"
log "================================================================="
log "[pipeline] ALL POST-TRAIN AND 8B LAUNCH STAGES COMPLETE!"
log "================================================================="
