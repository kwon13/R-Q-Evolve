#!/usr/bin/env bash
# The single 4B training arm on all eight GPUs.
#
#   bash scripts/run_train_4b_8gpu.sh
#
# Runs detached under nohup and tails nothing: the point of this arm is to be
# watched live in wandb while other work continues in the shell.
set -uo pipefail

ROOT=/data1/yhoon113/R-Q-Evolve
CONFIG=${CONFIG:-configs/rq_evolve_4b_8gpu.yaml}
LOGDIR=${LOGDIR:-$ROOT/rq_output/train_4b_8gpu_logs}
cd "$ROOT"
mkdir -p "$LOGDIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# sm_120 (RTX PRO 6000 Blackwell) needs the cu128 stack; azr-bw's cu126 build
# dies with "no kernel image is available". Activate rather than prepend PATH so
# ninja/nvcc come with it -- verl JIT-compiles at startup.
# `set -u` off across the activate: the env's binutils hook reads ADDR2LINE and
# friends before setting them, so an unset-variable abort here is conda's bug,
# not ours.
set +u
source /data1/yhoon113/miniforge3/etc/profile.d/conda.sh
conda activate azr-bw-blackwell || { echo "[4b-8gpu] conda activate failed" >&2; exit 1; }
set -u

# verl aborts on a device-count mismatch rather than quietly using fewer GPUs,
# but it aborts AFTER Ray and the model shards are up, which costs two minutes.
want=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | grep -c .)
have=$(grep -oP '^\s+n_gpus_per_node:\s*\K\d+' "$CONFIG")
if [[ "$want" != "$have" ]]; then
  echo "[4b-8gpu] CUDA_VISIBLE_DEVICES names $want devices, $CONFIG sets $have" >&2
  exit 1
fi

# A killed run can leak a worker still holding ~40 GiB; the next launch then
# OOMs during weight sync, minutes in, with a traceback that blames the wrong
# thing. Refuse to start instead.
leaked=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
if [[ -n "$leaked" ]]; then
  echo "[4b-8gpu] GPUs are not idle -- kill these first:" >&2
  echo "$leaked" >&2
  exit 1
fi

# Set here, not inherited: a stray WANDB_MODE=offline left over from a smoke
# shell would log this run to disk only, and nobody finds out until they look.
export WANDB_MODE=${WANDB_MODE:-online}
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$ROOT/.env" ]; then
  export WANDB_API_KEY=$(grep -E '^WANDB_API_KEY=' "$ROOT/.env" | cut -d= -f2-)
fi
[ -z "${WANDB_API_KEY:-}" ] && { echo "[4b-8gpu] no WANDB_API_KEY; aborting" >&2; exit 1; }

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/train_4b_8gpu_$STAMP.log"
echo "[4b-8gpu] config    : $CONFIG"
echo "[4b-8gpu] gpus      : $CUDA_VISIBLE_DEVICES"
echo "[4b-8gpu] wandb     : $WANDB_MODE"
echo "[4b-8gpu] log       : $LOG"

nohup python scripts/train_with_verl.py --config "$CONFIG" > "$LOG" 2>&1 &
echo "$!" | tee "$LOGDIR/train_4b_8gpu.pid" | xargs -I{} echo "[4b-8gpu] pid       : {}"
ln -sfn "$LOG" "$LOGDIR/latest.log"
