#!/bin/bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-online}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/rq_evolve_base_$(date +%Y%m%d_%H%M%S).log"
echo "[run] logging to $LOG"
set -o pipefail
python scripts/train_with_verl.py \
  --config configs/rq_evolve_4b_base.yaml 2>&1 | tee "$LOG"
