#!/bin/bash
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-online}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# The training entry point calls bare `python`, and the project .venv has no
# torch. Put the vllm env first so the script works from any shell.
export PATH=/data1/yhoon113/miniforge3/envs/vllm/bin:$PATH

# configs/rq_evolve_octo_8b.yaml sets n_gpus_per_node: 8; verl aborts on a
# mismatch rather than quietly using fewer devices.
want=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | grep -c .)
have=$(grep -oP '^\s+n_gpus_per_node:\s*\K\d+' configs/rq_evolve_octo_8b.yaml)
if [[ "$want" != "$have" ]]; then
  echo "CUDA_VISIBLE_DEVICES names $want devices, config sets $have" >&2
  exit 1
fi

# Capture ALL driver stdout+stderr to a timestamped logfile (still shown live in
# the terminal via tee). Without this the run's output exists ONLY on the
# terminal, so any hang/crash message ("Try again" retry loops, tracebacks)
# vanishes when the pane scrolls and post-mortem is impossible. pipefail makes
# the script's exit status reflect python's, not tee's.
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/rq_evolve_octo_8b_$(date +%Y%m%d_%H%M%S).log"
echo "[run] logging to $LOG"
set -o pipefail
python scripts/train_with_verl.py \
  --config configs/rq_evolve_octo_8b.yaml 2>&1 | tee "$LOG"
