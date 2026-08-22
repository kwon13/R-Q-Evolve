#!/bin/bash
# Paired re-scoring of the SAME 182 problems at the RL init (base) and at an
# intermediate checkpoint, so the sign distribution has a trajectory rather than
# a single point. Instances (and their prompt token ids) come from the step-256
# run verbatim -- the runs differ only in the policy.
set -u
ROOT=/data1/yhoon113/R-Q-Evolve
REF=$ROOT/analysis/rq_evolve_base_8b/cov_sign_union/instances.json
cd "$ROOT"
bash scripts/cov_sign_run_sharded.sh \
  /data1/yhoon113/qwen3-8b-base \
  "$ROOT/analysis/rq_evolve_base_8b/cov_sign_union_base" \
  "RL init (base), same 182 problems" "$REF" || exit 1
bash scripts/cov_sign_run_sharded.sh \
  "$ROOT/rq_output/rq_evolve_base_8b/global_step_128/hf_merged" \
  "$ROOT/analysis/rq_evolve_base_8b/cov_sign_union_step128" \
  "step 128, same 182 problems" "$REF" || exit 1
echo "[pipe] SERIES DONE"
