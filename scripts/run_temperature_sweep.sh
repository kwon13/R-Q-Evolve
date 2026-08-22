#!/usr/bin/env bash
# Mutation-temperature sweep for the free-form operator + judge pipeline.
#
# One arm per temperature, one GPU each, run concurrently. Every arm
# bootstraps the same seed corpus and then takes 32 mutation steps in batches
# of 8, so parents are resampled four times and a child can become a parent.
#
#   bash scripts/run_temperature_sweep.sh
#
# Read the result with: python scripts/summarize_temperature_sweep.py
set -uo pipefail

ROOT=/data1/yhoon113/R-Q-Evolve
MODEL=${MODEL:-/data1/yhoon113/qwen3-8b-base}
CONFIG=$ROOT/configs/rq_evolve_judge_sweep.yaml
OUTROOT=${OUTROOT:-$ROOT/rq_output/temp_sweep}
LOGDIR=$OUTROOT/logs
mkdir -p "$LOGDIR"

TEMPS=(0.3 0.5 0.7)
GPUS=(2 3 4)

cd "$ROOT"
pids=()
for i in "${!TEMPS[@]}"; do
  t=${TEMPS[$i]}
  gpu=${GPUS[$i]}
  echo "[sweep] arm temperature=$t on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu nohup python scripts/sample_evolve_vllm.py \
    --config "$CONFIG" \
    --model "$MODEL" \
    --steps 32 --batch 8 --rollouts 8 \
    --gpu-util 0.4 --enforce-eager \
    --code-temperature "$t" \
    --out "$OUTROOT/t$t" \
    > "$LOGDIR/t$t.log" 2>&1 &
  pids+=($!)
done

echo "[sweep] ${#pids[@]} arms running: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do
  wait "$p" || { echo "[sweep] pid $p exited nonzero"; fail=1; }
done
echo "[sweep] done (fail=$fail)"
exit $fail
