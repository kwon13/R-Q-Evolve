#!/usr/bin/env bash
# 1. Run eval_models_fanout.sh across 8 GPUs for all checkpoints in rq_evolve_4b_8gpu
# 2. Upon completion, immediately start the 4B 8GPU U=1 ablation training run.
set -uo pipefail

ROOT=/data1/yhoon113/R-Q-Evolve
cd "$ROOT"

LOGDIR="$ROOT/rq_output/pipeline_eval_models_and_ablate_logs"
mkdir -p "$LOGDIR"
MAIN_LOG="$LOGDIR/eval_models_then_ablate_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MAIN_LOG"; }

# Gather all available hf_merged models
MODELS_CSV=$(find "$ROOT/rq_output/rq_evolve_4b_8gpu" -maxdepth 2 -name 'hf_merged' | sort -V | tr '\n' ',' | sed 's/,$//')

log "=== STEP 1: Starting eval_models_fanout.sh on 8 GPUs ==="
log "Evaluating models: $MODELS_CSV"

MODELS="$MODELS_CSV" GPU_LIST="0,1,2,3,4,5,6,7" SUITES="${SUITES:-math,general}" \
  bash scripts/eval_models_fanout.sh 2>&1 | tee -a "$MAIN_LOG"
EVAL_STATUS=${PIPESTATUS[0]}

if [[ $EVAL_STATUS -ne 0 ]]; then
  log "WARNING: eval_models_fanout exited with status $EVAL_STATUS."
else
  log "Evaluation completed successfully! Results written to $ROOT/rq_output/model_bench"
fi

log "Waiting 15s for GPU memory to fully release..."
sleep 15

# Clean any leftover ray/vllm sessions
source /data1/yhoon113/miniforge3/etc/profile.d/conda.sh
conda activate vllm
ray stop --force 2>/dev/null || true
sleep 5

log "=== STEP 2: Starting 4B 8GPU U=1 Ablation Training ==="
bash scripts/run_train_8gpu_4b_ablate_nounc.sh | tee -a "$MAIN_LOG"

log "Pipeline finished! Ablation run is now running in background."
