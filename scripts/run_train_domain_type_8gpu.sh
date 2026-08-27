#!/usr/bin/env bash
# Launch the fresh 35-cell DOMAIN x PROBLEM_TYPE run on eight GPUs.
set -euo pipefail

ROOT=/data1/yhoon113/R-Q-Evolve
CONFIG=configs/rq_evolve_4b_8gpu_domain_type.yaml
RUN_DIR=$ROOT/rq_output/rq_evolve_4b_domain_type_35cell_8gpu

cd "$ROOT"
[ -f "$CONFIG" ] || { echo "[domain-type] no such config: $CONFIG" >&2; exit 1; }

if [ -d "$RUN_DIR" ] && [ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  echo "[domain-type] refusing to reuse non-empty run directory: $RUN_DIR" >&2
  echo "[domain-type] choose a new config identity and output directory for a fresh run" >&2
  exit 1
fi

seed_count=$(find seed_programs_domain_type -maxdepth 1 -type f -name '*.py' | wc -l)
if [ "$seed_count" -ne 7 ]; then
  echo "[domain-type] expected exactly 7 diagonal seed programs, found $seed_count" >&2
  exit 1
fi

export CONFIG
exec bash scripts/run_train_8gpu.sh "$CONFIG"
