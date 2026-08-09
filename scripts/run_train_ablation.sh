#!/bin/bash
# Launch one R_Q ablation arm.
#
#   ARM=novar GPUS=0,1,2,3 bash scripts/run_train_ablation.sh
#   ARM=nounc GPUS=4,5,6,7 bash scripts/run_train_ablation.sh
#   SIZE=8b ARM=novar GPUS=0,1,2,3,4,5,6,7 bash scripts/run_train_ablation.sh
#
# The two arms are the same config as the size's base run with one factor
# dropped from the champion-selection priority:
#   novar -> s(1-s) := 1, rank by H alone
#   nounc -> H := 1,      rank by s(1-s) alone
# The control is the base run's own global_step_128.
#
# SIZE picks the config pair: 4b -> rq_evolve_4b_ablate_*.yaml (n_gpus 4, so the
# two arms fit side by side on 8 devices), 8b -> rq_evolve_8b_ablate_*.yaml
# (n_gpus 8, so the arms have to run one after the other). GPUS must name
# exactly n_gpus_per_node devices; verl aborts on a mismatch rather than
# silently using fewer.
set -uo pipefail

ARM="${ARM:?set ARM=novar or ARM=nounc}"
GPUS="${GPUS:?set GPUS=0,1,2,3}"
# Copied out of SIZE immediately: conda's binutils activate hook exports SIZE as
# the path to x86_64-conda-linux-gnu-size, so anything reading $SIZE after the
# activation below gets a toolchain path instead of "4b"/"8b".
RQ_SIZE="${SIZE:-4b}"
CFG="configs/rq_evolve_${RQ_SIZE}_ablate_${ARM}.yaml"

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

# An idle GPU is not a free GPU. A finished-looking run keeps its Ray workers
# alive (total_epochs is 10000, so "the checkpoints I wanted exist" is not the
# same as "it stopped"), and between two of its phases every card reads 9 MiB.
# Two arms launched into that gap both died of CUDA OOM an hour later when the
# older run woke up. Check for a live trainer, not for free memory.
others=$(pgrep -u "$USER" -f 'train_with_verl' | grep -v "^$$\$" | wc -l)
if [[ "$others" -gt 0 ]] && [[ "${IGNORE_RUNNING:-0}" != "1" ]]; then
  echo "$others train_with_verl process(es) already running:" >&2
  pgrep -u "$USER" -af 'train_with_verl' | cut -c1-110 >&2
  echo "Stop them first, or set IGNORE_RUNNING=1 if they use other GPUs." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export WANDB_MODE="${WANDB_MODE:-online}"
# Training runs in the cu128 env (sm_120). The old standalone `vllm` env this
# line used to point at no longer exists, so PATH silently kept whatever python
# was already active; activate the real env instead. The binutils activate hook
# reads $ADDR2LINE before setting it, which is fatal under `set -u`.
CONDA_ENV="${CONDA_ENV:-azr-bw-blackwell}"
source /data1/yhoon113/miniforge3/etc/profile.d/conda.sh
set +u
conda activate "$CONDA_ENV"
set -u

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/rq_ablate_${RQ_SIZE}_${ARM}_$(date +%Y%m%d_%H%M%S).log"
echo "[run] arm=$ARM  size=$RQ_SIZE  gpus=$GPUS  config=$CFG"
echo "[run] logging to $LOG"
set -o pipefail
python scripts/train_with_verl.py --config "$CFG" 2>&1 | tee "$LOG"
