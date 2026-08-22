#!/bin/bash
# Union population across 7 GPUs. GPU 0 is left alone (a resident vLLM engine
# from another session lives there). Each shard is an independent, complete
# cov_sign run over a disjoint slice; cov_sign_merge_shards.py concatenates.
set -u
ROOT=/data1/yhoon113/R-Q-Evolve
PY=/data1/yhoon113/miniforge3/envs/azr-bw-blackwell/bin/python
UNI=$ROOT/analysis/rq_evolve_base_8b/cov_sign_union
GPUS=(1 2 3 4 5 6 7)
N=${#GPUS[@]}
GPU_UTIL=0.40
export RQ_COV_SIGN_SRC=/tmp/claude-1024/-data1-yhoon113/57ee35db-92d4-4e2a-bd8c-a249be500e3c/scratchpad/src_at_1023
cd "$ROOT"
rm -rf "$UNI"; mkdir -p "$UNI"

echo "[pipe] === union: generation, $N shards, gpu_util $GPU_UTIL ==="
pids=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY scripts/cov_sign_generate.py \
    --g 32 --shard $i --num-shards $N --gpu-util $GPU_UTIL \
    --out-dir "$UNI/shard_$i" \
    --archive-glob "$ROOT/rq_output/rq_evolve_base_8b/rq_archive/archive_iter*.json" \
    > "$ROOT/logs/cov_sign_union_gen_$i.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo "[pipe] FATAL: a generation shard failed"; exit 1; }
echo "[pipe] === union: generation done ==="

echo "[pipe] === union: entropy, $N shards ==="
pids=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY scripts/cov_sign_entropy.py \
    --out-dir "$UNI/shard_$i" > "$ROOT/logs/cov_sign_union_ent_$i.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo "[pipe] FATAL: an entropy shard failed"; exit 1; }
echo "[pipe] === union: entropy done ==="

$PY scripts/cov_sign_merge_shards.py --out-dir "$UNI" || exit 1
$PY scripts/cov_sign_analyze.py --out-dir "$UNI" --label "all champions ever archived" > "$ROOT/logs/cov_sign_union_analyze.log" 2>&1 || exit 1
echo "[pipe] UNION DONE"
