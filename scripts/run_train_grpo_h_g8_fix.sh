#!/bin/bash
# R-Q-Evolve training launcher, ported from
#   evo-sample/scripts/run_train_grpo_h_g8_fix.sh
#
# Runs the standard pip-installed verl (RayPPOTrainer) with the R_Q-Evolve
# EvolvingSampler. Each verl *epoch* triggers one RQ outer iteration
# (archive re-eval -> mutation -> R_Q scoring -> dataset refresh) at the start
# of the epoch, then verl does the GRPO solver update on the refreshed dataset.
#
#   total_epochs: 100         (== evo-sample total_outer_iterations)
#   total_training_steps: 256 (== evo-sample max_steps, hard cap)
#
# NOTE on semantics vs evo-sample: EasyR1's RQEvolveTrainer ran a fixed number
# of solver steps *per* outer iteration. Here the per-epoch step count is
# instead len(dataset)/train_batch_size. The evolution cadence (once per outer
# iteration) is preserved; the inner step budget is governed by the dataset
# size and the global total_training_steps cap.
set -euo pipefail

# evo-sample used 4 GPUs (n_gpus_per_node: 4)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE="${WANDB_MODE:-online}"
# Model: defaults to Qwen/Qwen3-4B-Base (same as evo-sample). Override to a
# local HF checkpoint path if desired.
export RQ_MODEL_PATH="${RQ_MODEL_PATH:-Qwen/Qwen3-4B-Base}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Prefer the project venv's python (its verl is what actually trains).
PY="python"
if [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
fi

# WANDB_API_KEY must be exported (or in your shell env) when WANDB_MODE=online.
# To run without Weights & Biases:  WANDB_MODE=disabled ./scripts/run_train_grpo_h_g8_fix.sh

echo "[run] resolving verl runtime ..."
"$PY" scripts/train_with_verl.py --print-verl-env

echo "[run] starting training ..."
exec "$PY" scripts/train_with_verl.py \
  --config configs/rq_evolve_grpo_h_g8_fix.yaml
