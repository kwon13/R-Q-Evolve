#!/bin/bash
# Resume the base run: shard 5's entropy pass died on the JSONL reader bug
# (a raw U+2028 in one response), the other six shards are complete.
set -u
ROOT=/data1/yhoon113/R-Q-Evolve
PY=/data1/yhoon113/miniforge3/envs/azr-bw-blackwell/bin/python
OUT=$ROOT/analysis/rq_evolve_base_8b/cov_sign_union_base
export RQ_COV_SIGN_SRC=/tmp/claude-1024/-data1-yhoon113/57ee35db-92d4-4e2a-bd8c-a249be500e3c/scratchpad/src_at_1023
cd "$ROOT"
echo "[pipe] === base: entropy shard 5 retry ==="
CUDA_VISIBLE_DEVICES=1 $PY scripts/cov_sign_entropy.py \
  --checkpoint /data1/yhoon113/qwen3-8b-base --out-dir "$OUT/shard_5" \
  > "$ROOT/logs/cov_sign_union_base_ent_5.log" 2>&1 || { echo "[pipe] FATAL retry"; exit 1; }
echo "[pipe] === base: entropy done ==="
$PY scripts/cov_sign_merge_shards.py --out-dir "$OUT" || exit 1
$PY scripts/cov_sign_analyze.py --out-dir "$OUT" --label "RL init (base), same 182 problems" \
  > "$ROOT/logs/cov_sign_union_base_analyze.log" 2>&1 || exit 1
echo "[pipe] cov_sign_union_base DONE"

bash scripts/cov_sign_run_sharded.sh \
  "$ROOT/rq_output/rq_evolve_base_8b/global_step_128/hf_merged" \
  "$ROOT/analysis/rq_evolve_base_8b/cov_sign_union_step128" \
  "step 128, same 182 problems" "$ROOT/analysis/rq_evolve_base_8b/cov_sign_union/instances.json" || exit 1
echo "[pipe] SERIES DONE"
