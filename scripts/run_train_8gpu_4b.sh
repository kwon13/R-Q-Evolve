#!/usr/bin/env bash
# One training arm on all eight GPUs.
#
#   bash scripts/run_train_8gpu.sh configs/rq_evolve_8b_8gpu.yaml
#   bash scripts/run_train_8gpu.sh configs/rq_evolve_4b_8gpu.yaml
#
# Runs detached under nohup and tails nothing: the point of these arms is to be
# watched live in wandb while other work continues in the shell. Log directory
# and pid file are named after the config, so two arms never overwrite each
# other's bookkeeping.
set -uo pipefail

ROOT=/data1/yhoon113/R-Q-Evolve
CONFIG=${1:-${CONFIG:-configs/rq_evolve_4b_8gpu.yaml}}
cd "$ROOT"
[ -f "$CONFIG" ] || { echo "[8gpu] no such config: $CONFIG" >&2; exit 1; }
NAME=$(basename "$CONFIG" .yaml)
LOGDIR=${LOGDIR:-$ROOT/rq_output/${NAME}_logs}
mkdir -p "$LOGDIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

set +u
source /data1/yhoon113/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-vllm}" || { echo "[8gpu] conda activate failed" >&2; exit 1; }
export CUDA_HOME=/data1/yhoon113/cuda-12.8
export PATH=/data1/yhoon113/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/data1/yhoon113/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/data1/yhoon113/miniforge3/envs/vllm/lib/libgomp.so.1
set -u

# OOMs during weight sync, minutes in, with a traceback that blames the wrong
# thing. Refuse to start instead.
leaked=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
if [[ -n "$leaked" ]]; then
  echo "[8gpu] GPUs are not idle -- kill these first:" >&2
  echo "$leaked" >&2
  exit 1
fi

# Set here, not inherited: a stray WANDB_MODE=offline left over from a smoke
# shell would log this run to disk only, and nobody finds out until they look.
export WANDB_MODE=${WANDB_MODE:-online}
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$ROOT/.env" ]; then
  export WANDB_API_KEY=$(grep -E '^WANDB_API_KEY=' "$ROOT/.env" | cut -d= -f2-)
fi
[ -z "${WANDB_API_KEY:-}" ] && { echo "[8gpu] no WANDB_API_KEY; aborting" >&2; exit 1; }

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/${NAME}_$STAMP.log"
echo "[8gpu] config    : $CONFIG"
echo "[8gpu] model     : $(grep -oP '^\s+path:\s*\K\S+' "$CONFIG" | head -1)"
echo "[8gpu] gpus      : $CUDA_VISIBLE_DEVICES"
echo "[8gpu] wandb     : $WANDB_MODE"
echo "[8gpu] log       : $LOG"

nohup python scripts/train_with_verl.py --config "$CONFIG" > "$LOG" 2>&1 &
echo "$!" | tee "$LOGDIR/run.pid" | xargs -I{} echo "[8gpu] pid       : {}"
ln -sfn "$LOG" "$LOGDIR/latest.log"
