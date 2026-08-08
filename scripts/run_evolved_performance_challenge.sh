#!/usr/bin/env bash
# Evaluate the fixed Structural-OOD v2 benchmark on the base model and every
# saved checkpoint. This reuses the generic Evolved Performance pipeline but
# keeps benchmark data and results separate from the Seed-ID anchor.
set -uo pipefail

RQ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${BASE:-$RQ/rq_output/rq_evolve_base_4b}"

export BASE
export SEED_DIR="${SEED_DIR:-$RQ/challenge_seed_programs/structural_ood_v2}"
export BENCH_DIR="${BENCH_DIR:-$RQ/benchmarks/evolved_performance_structural_ood_v2}"
export RESULTS_DIR="${RESULTS_DIR:-$BASE/evolved_performance_structural_ood_v2}"
export BENCHMARK_NAME="${BENCHMARK_NAME:-evolved_performance_structural_ood_v2}"
export SEED_START="${SEED_START:-3000000}"
export EXAMPLES_PER_PROGRAM="${EXAMPLES_PER_PROGRAM:-40}"
export OVERLAP_AUDIT="${OVERLAP_AUDIT:-0}"
export PLOT_TITLE="${PLOT_TITLE:-R-Q-Evolve — Structural-OOD Performance Evolution (240 problems)}"

exec bash "$RQ/scripts/run_evolved_performance.sh"
