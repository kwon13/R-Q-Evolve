#!/usr/bin/env bash
# 4B 8GPU Ablation Arm: No Uncertainty (U := 1, R_Q = s(1-s))
#
#   bash scripts/run_train_8gpu_4b_ablate_nounc.sh
#
# Runs detached under nohup. Log directory and pid file are named after the config.
set -uo pipefail

ROOT=/data1/yhoon113/R-Q-Evolve
CONFIG=${1:-${CONFIG:-configs/rq_evolve_4b_8gpu_ablate_nounc.yaml}}
cd "$ROOT"
[ -f "$CONFIG" ] || { echo "[8gpu-ablate] no such config: $CONFIG" >&2; exit 1; }
NAME=$(basename "$CONFIG" .yaml)
LOGDIR=${LOGDIR:-$ROOT/rq_output/${NAME}_logs}
mkdir -p "$LOGDIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set +u
source /data1/yhoon113/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-vllm}" || { echo "[8gpu-ablate] conda activate failed" >&2; exit 1; }
export CUDA_HOME=/data1/yhoon113/cuda-12.8
export PATH=/data1/yhoon113/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/data1/yhoon113/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/data1/yhoon113/miniforge3/envs/vllm/lib/libgomp.so.1
set -u

# verl aborts on a device-count mismatch rather than quietly using fewer GPUs.
want=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | grep -c .)
have=$(grep -oP '^\s+n_gpus_per_node:\s*\K\d+' "$CONFIG")
if [[ "$want" != "$have" ]]; then
  echo "[8gpu-ablate] CUDA_VISIBLE_DEVICES names $want devices, $CONFIG sets $have" >&2
  exit 1
fi

# A killed run can leak a worker still holding ~40 GiB; refuse to start instead.
leaked=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
if [[ -n "$leaked" ]]; then
  echo "[8gpu-ablate] GPUs are not idle -- kill these first:" >&2
  echo "$leaked" >&2
  exit 1
fi

# Set here, not inherited
export WANDB_MODE=${WANDB_MODE:-online}
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$ROOT/.env" ]; then
  export WANDB_API_KEY=$(grep -E '^WANDB_API_KEY=' "$ROOT/.env" | cut -d= -f2-)
fi
[ -z "${WANDB_API_KEY:-}" ] && { echo "[8gpu-ablate] no WANDB_API_KEY; aborting" >&2; exit 1; }

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/${NAME}_$STAMP.log"
echo "[8gpu-ablate] config    : $CONFIG"
echo "[8gpu-ablate] model     : $(grep -oP '^\s+path:\s*\K\S+' "$CONFIG" | head -1)"
echo "[8gpu-ablate] gpus      : $CUDA_VISIBLE_DEVICES"
echo "[8gpu-ablate] wandb     : $WANDB_MODE"
echo "[8gpu-ablate] log       : $LOG"

nohup python scripts/train_with_verl.py --config "$CONFIG" > "$LOG" 2>&1 &
echo "$!" | tee "$LOGDIR/run.pid" | xargs -I{} echo "[8gpu-ablate] pid       : {}"
ln -sfn "$LOG" "$LOGDIR/latest.log"
