#!/bin/bash
# ABLATION run: crossover-only mutation (crossover_ratio=1.0, in_depth_ratio=0.0).
# Same RL/verl settings as run_train_grpo_h_g8_fix.sh; only the RQ mutation mix
# differs. RQ_EXP_NAME / RQ_LOCAL_DIR separate the wandb run and checkpoint dir
# (incl. rq_archive) from the base and the other ablation.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE="${WANDB_MODE:-online}"
export RQ_MODEL_PATH="${RQ_MODEL_PATH:-Qwen/Qwen3-4B-Base}"
export RQ_EXP_NAME="${RQ_EXP_NAME:-qwen3_4b_base_grpo_h_g8_crossover_only}"
export RQ_LOCAL_DIR="${RQ_LOCAL_DIR:-./rq_output/verl_ckpt_grpo_h_g8_crossover_only}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PY="python"
if [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
fi

echo "[run] ablation=crossover_only exp=$RQ_EXP_NAME"
"$PY" scripts/train_with_verl.py --print-verl-env

exec "$PY" scripts/train_with_verl.py \
  --config configs/rq_evolve_grpo_h_g8_fix_crossover.yaml
