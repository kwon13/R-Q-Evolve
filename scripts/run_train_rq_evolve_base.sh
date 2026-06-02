#!/bin/bash
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-online}"
export RQ_EXP_NAME="${RQ_EXP_NAME:-qwen3_4b_base_rq_evolve_base}"
export RQ_LOCAL_DIR="${RQ_LOCAL_DIR:-./rq_output/rq_evolve_base}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

echo "[run] ablation=crossover_only exp=$RQ_EXP_NAME"

exec python scripts/train_with_verl.py \
  --config configs/rq_evolve_base.yaml
