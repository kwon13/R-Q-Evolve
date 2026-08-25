#!/usr/bin/env bash
# Watch the current main training run (PID 2880874 or current active run)
# and immediately launch the U=1 ablation run (run_train_8gpu_4b_ablate_nounc.sh)
# when the main run finishes.
set -uo pipefail

ROOT=/data1/yhoon113/R-Q-Evolve
cd "$ROOT"
mkdir -p rq_output

MAIN_PID_FILE="$ROOT/rq_output/rq_evolve_4b_8gpu_logs/run.pid"
WAITLOG="$ROOT/rq_output/wait_and_run_ablate_nounc.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WAITLOG"; }

MAIN_PID=$(cat "$MAIN_PID_FILE" 2>/dev/null || echo "2880874")

log "=== Automated Ablation Runner Started ==="
log "Watching main training run PID: $MAIN_PID"

while kill -0 "$MAIN_PID" 2>/dev/null; do
  sleep 30
done

log "Main training process (PID $MAIN_PID) has finished!"
log "Waiting 20s for GPU resources to fully release..."
sleep 20

# Clean up any leftover ray processes
source /data1/yhoon113/miniforge3/etc/profile.d/conda.sh
conda activate vllm
ray stop --force 2>/dev/null || true
sleep 5

log "Starting 4B 8GPU U=1 Ablation run (run_train_8gpu_4b_ablate_nounc.sh)..."
bash scripts/run_train_8gpu_4b_ablate_nounc.sh | tee -a "$WAITLOG"

log "Ablation run launched successfully! Check rq_output/rq_evolve_4b_8gpu_ablate_nounc_logs/latest.log"
