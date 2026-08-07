#!/bin/bash
# Launch one R_Q ablation arm.
#
#   ARM=novar GPUS=0,1,2,3 bash scripts/run_train_ablation.sh
#   ARM=nounc GPUS=4,5,6,7 bash scripts/run_train_ablation.sh
#
# The two arms are the same config as rq_evolve_4b_base.yaml with one factor
# dropped from the champion-selection priority:
#   novar -> s(1-s) := 1, rank by H alone
#   nounc -> H := 1,      rank by s(1-s) alone
# The control is the existing 256-step run's global_step_128.
#
# n_gpus_per_node is 4 in both configs, so GPUS must name exactly four devices;
# verl aborts on a mismatch rather than silently using fewer.
set -uo pipefail

ARM="${ARM:?set ARM=novar or ARM=nounc}"
GPUS="${GPUS:?set GPUS=0,1,2,3}"
CFG="configs/rq_evolve_4b_ablate_${ARM}.yaml"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
[[ -f "$CFG" ]] || { echo "no such config: $CFG" >&2; exit 1; }

want=$(tr ',' '\n' <<< "$GPUS" | grep -c .)
have=$(grep -oP '^\s+n_gpus_per_node:\s*\K\d+' "$CFG")
if [[ "$want" != "$have" ]]; then
  echo "GPUS names $want devices but $CFG sets n_gpus_per_node: $have" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export WANDB_MODE="${WANDB_MODE:-online}"
export PATH=/data1/yhoon113/miniforge3/envs/vllm/bin:$PATH

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/rq_ablate_${ARM}_$(date +%Y%m%d_%H%M%S).log"
echo "[run] arm=$ARM  gpus=$GPUS  config=$CFG"
echo "[run] logging to $LOG"
set -o pipefail
python scripts/train_with_verl.py --config "$CFG" 2>&1 | tee "$LOG"
