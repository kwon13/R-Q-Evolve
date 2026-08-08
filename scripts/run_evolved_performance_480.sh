#!/usr/bin/env bash
# Evaluate one fixed 480-item Evolve benchmark: 240 Seed-ID + 240 Structural-OOD v2.
# The output uses the original single-EPS performance figure.
set -euo pipefail

RQ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data1/yhoon113/miniforge3/envs/vllm/bin/python}"
BASE="${BASE:-$RQ/rq_output/rq_evolve_base_4b}"
ID_BENCH_DIR="${ID_BENCH_DIR:-$RQ/benchmarks/evolved_performance_seed_id_v1}"
OOD_BENCH_DIR="${OOD_BENCH_DIR:-$RQ/benchmarks/evolved_performance_structural_ood_v2}"

export BASE PY
export BENCH_DIR="${BENCH_DIR:-$RQ/benchmarks/evolved_performance_480_v1}"
export RESULTS_DIR="${RESULTS_DIR:-$BASE/evolved_performance_480_v1}"
export BENCHMARK_NAME="${BENCHMARK_NAME:-evolved_performance_480_v1}"
export PREBUILT_BENCHMARK=1
export OVERLAP_AUDIT=0
export PLOT_TITLE="${PLOT_TITLE:-R-Q-Evolve — Evolved Performance Evolution (480 problems)}"

MERGE_ARGS=(
  "$RQ/scripts/merge_evolved_performance_benchmarks.py"
  --component "id=$ID_BENCH_DIR"
  --component "ood_v2=$OOD_BENCH_DIR"
  --output-dir "$BENCH_DIR"
  --benchmark-name "$BENCHMARK_NAME"
)
if [[ "${FORCE_BENCH:-0}" == "1" ]]; then
  MERGE_ARGS+=(--force)
fi
"$PY" "${MERGE_ARGS[@]}"

exec bash "$RQ/scripts/run_evolved_performance.sh"
