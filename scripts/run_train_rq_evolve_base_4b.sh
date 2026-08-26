#!/bin/bash
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,6,7}"
export WANDB_MODE="${WANDB_MODE:-online}"
# The GPUs are shared and nothing reserves them, so the free pool shrinks
# without warning. Expandable segments let the allocator grow a block in place
# instead of needing one contiguous free region, which is what the OOM at step
# 5 actually failed on: 2.73 GiB was free but the 3.30 GiB request wanted it in
# one piece.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

mkdir -p "$ROOT/logs"
# Distinct from the 8B script's prefix: both used rq_evolve_base_* and the
# two runs' logs were indistinguishable after the fact.
LOG="$ROOT/logs/rq_evolve_4b_$(date +%Y%m%d_%H%M%S).log"
CONFIG="${1:-configs/rq_evolve_4b_4gpu.yaml}"
echo "[run] config : $CONFIG"
echo "[run] logging: $LOG"
set -o pipefail
python scripts/train_with_verl.py \
  --config "$CONFIG" 2>&1 | tee "$LOG"

# bash scripts/run_train_rq_evolve_base_4b.sh configs/rq_evolve_4b_4gpu.yaml
