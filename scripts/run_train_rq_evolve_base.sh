#!/bin/bash
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-online}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Capture ALL driver stdout+stderr to a timestamped logfile (still shown live in
# the terminal via tee). Without this the run's output exists ONLY on the
# terminal, so any hang/crash message ("Try again" retry loops, tracebacks)
# vanishes when the pane scrolls and post-mortem is impossible. pipefail makes
# the script's exit status reflect python's, not tee's.
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/rq_evolve_base_$(date +%Y%m%d_%H%M%S).log"
echo "[run] logging to $LOG"
set -o pipefail
python scripts/train_with_verl.py \
  --config configs/rq_evolve_base.yaml 2>&1 | tee "$LOG"
