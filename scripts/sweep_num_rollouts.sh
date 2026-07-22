#!/bin/bash
# Sweep evolution.num_rollouts (the p_hat denominator) holding everything else fixed.
#
#   bash scripts/sweep_num_rollouts.sh              # arms: 5 20 40
#   bash scripts/sweep_num_rollouts.sh 5 10 20 40   # include a fresh nr10 baseline
#
# Arms run SEQUENTIALLY: each one takes all 8 GPUs. Each arm gets its own
# generated config (configs/generated/) and output dir (rq_output/rq_evolve_4b_nr<N>/),
# with a config_used.yaml snapshot next to its artifacts.
#
# An arm whose output dir already holds global_step_128 is skipped, so a killed
# sweep can be re-run to pick up where it stopped.

set -uo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-online}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then
  ARMS=(40)
fi

mkdir -p "$ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_LOG="$ROOT/logs/sweep_num_rollouts_${STAMP}.log"

echo "[sweep] arms: ${ARMS[*]}" | tee "$SWEEP_LOG"
echo "[sweep] runtime scales with num_rollouts: the evolve phase generates" | tee -a "$SWEEP_LOG"
echo "[sweep] num_rollouts x (problems per iter) responses. nr40 is ~4x nr10." | tee -a "$SWEEP_LOG"
echo "" | tee -a "$SWEEP_LOG"

for N in "${ARMS[@]}"; do
  TAG="nr${N}"
  OUT="$ROOT/rq_output/rq_evolve_4b_${TAG}"

  if [ -d "$OUT/global_step_128" ]; then
    echo "[sweep] $TAG: already has global_step_128 -> skip" | tee -a "$SWEEP_LOG"
    continue
  fi

  CFG="$(python scripts/make_sweep_config.py --num-rollouts "$N" --tag "$TAG")"
  if [ ! -f "$CFG" ]; then
    echo "[sweep] $TAG: config generation FAILED -> abort" | tee -a "$SWEEP_LOG"
    exit 1
  fi

  LOG="$ROOT/logs/rq_evolve_4b_${TAG}_${STAMP}.log"
  echo "[sweep] === $TAG (num_rollouts=$N) ===" | tee -a "$SWEEP_LOG"
  echo "[sweep]   config : $CFG" | tee -a "$SWEEP_LOG"
  echo "[sweep]   output : $OUT" | tee -a "$SWEEP_LOG"
  echo "[sweep]   log    : $LOG" | tee -a "$SWEEP_LOG"
  echo "[sweep]   start  : $(date '+%F %T')" | tee -a "$SWEEP_LOG"

  set -o pipefail
  python scripts/train_with_verl.py --config "$CFG" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}

  echo "[sweep]   end    : $(date '+%F %T')  (exit $RC)" | tee -a "$SWEEP_LOG"
  if [ "$RC" -ne 0 ]; then
    # Don't burn GPU-hours on the remaining arms if this one died (OOM, actor
    # crash, ...) -- the sweep is re-runnable and will skip what finished.
    echo "[sweep] $TAG FAILED (exit $RC) -> stopping sweep" | tee -a "$SWEEP_LOG"
    exit "$RC"
  fi
  echo "" | tee -a "$SWEEP_LOG"

  # Ray/vLLM can leave workers holding GPU memory; give them a moment to drain
  # before the next arm reserves the devices.
  sleep 30
done

echo "[sweep] all arms done. summary log: $SWEEP_LOG" | tee -a "$SWEEP_LOG"
