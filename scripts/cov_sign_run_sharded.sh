#!/bin/bash
# Generic sharded cov_sign run: generation + entropy across 7 GPUs, then merge
# and analyse.  GPU 0 is left alone (another session's engine lives there).
#   cov_sign_run_sharded.sh <checkpoint> <out_dir> <label> [instances_from]
set -u
CKPT="$1"; OUT="$2"; LABEL="$3"; REUSE="${4:-}"
ROOT=/data1/yhoon113/R-Q-Evolve
PY=/data1/yhoon113/miniforge3/envs/azr-bw-blackwell/bin/python
GPUS=(1 2 3 4 5 6 7)
N=${#GPUS[@]}
GPU_UTIL=0.40
export RQ_COV_SIGN_SRC=/tmp/claude-1024/-data1-yhoon113/57ee35db-92d4-4e2a-bd8c-a249be500e3c/scratchpad/src_at_1023
cd "$ROOT"
TAG=$(basename "$OUT")
rm -rf "$OUT"; mkdir -p "$OUT"
EXTRA=""
[ -n "$REUSE" ] && EXTRA="--instances-from $REUSE"

echo "[pipe] === $TAG: generation, $N shards, ckpt $CKPT ==="
pids=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY scripts/cov_sign_generate.py \
    --g 32 --shard $i --num-shards $N --gpu-util $GPU_UTIL \
    --checkpoint "$CKPT" --out-dir "$OUT/shard_$i" $EXTRA \
    --archive-glob "$ROOT/rq_output/rq_evolve_base_8b/rq_archive/archive_iter*.json" \
    > "$ROOT/logs/${TAG}_gen_$i.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo "[pipe] FATAL: $TAG generation shard failed"; exit 1; }
echo "[pipe] === $TAG: generation done ==="

echo "[pipe] === $TAG: entropy, $N shards ==="
pids=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY scripts/cov_sign_entropy.py \
    --checkpoint "$CKPT" --out-dir "$OUT/shard_$i" \
    > "$ROOT/logs/${TAG}_ent_$i.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo "[pipe] FATAL: $TAG entropy shard failed"; exit 1; }
echo "[pipe] === $TAG: entropy done ==="

$PY scripts/cov_sign_merge_shards.py --out-dir "$OUT" || exit 1
$PY scripts/cov_sign_analyze.py --out-dir "$OUT" --label "$LABEL" \
  > "$ROOT/logs/${TAG}_analyze.log" 2>&1 || exit 1
echo "[pipe] $TAG DONE"
