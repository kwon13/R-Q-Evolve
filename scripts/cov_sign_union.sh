#!/bin/bash
# Union population (every program that ever held a MAP cell) for the signed
# covariance experiment. RQ_COV_SIGN_SRC pins rq_evolve to the module state the
# 27-champion run used: src/rq_evolve as of 10:20 today, with prompts loaded
# from the bytecode compiled then (the working-tree prompts.py was edited at
# 10:27 into a state that does not parse). Verified: identical prompt token ids
# on all 27 shared problems.
set -u
ROOT=/data1/yhoon113/R-Q-Evolve
PY=/data1/yhoon113/miniforge3/envs/azr-bw-blackwell/bin/python
UNI=$ROOT/analysis/rq_evolve_base_8b/cov_sign_union
export RQ_COV_SIGN_SRC=/tmp/claude-1024/-data1-yhoon113/57ee35db-92d4-4e2a-bd8c-a249be500e3c/scratchpad/src_at_1023
cd "$ROOT"

echo "[pipe] === union: generation ==="
$PY scripts/cov_sign_generate.py --g 32 --out-dir "$UNI" \
  --archive-glob "$ROOT/rq_output/rq_evolve_base_8b/rq_archive/archive_iter*.json" || exit 1
echo "[pipe] === union: entropy ==="
$PY scripts/cov_sign_entropy.py --out-dir "$UNI" || exit 1
echo "[pipe] === union: analysis ==="
$PY scripts/cov_sign_analyze.py --out-dir "$UNI" --label "all champions ever archived" || exit 1
echo "[pipe] UNION DONE"
