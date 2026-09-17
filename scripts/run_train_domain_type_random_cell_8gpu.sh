#!/usr/bin/env bash
# Launch the score-free random-cell ablation on eight selected GPUs.
#
# Usage (from an activated training environment):
#   bash scripts/run_train_domain_type_random_cell_8gpu.sh --gpus 0,1,2,3,4,5,6,7
#   bash scripts/run_train_domain_type_random_cell_8gpu.sh --gpus 0,1,2,3,4,5,6,7 --detach
#   bash scripts/run_train_domain_type_random_cell_8gpu.sh --gpus 0,1,2,3,4,5,6,7 --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RQ_DOMAIN_TYPE_CONFIG="configs/rq_evolve_4b_8gpu_domain_type_random_cell.yaml"
export RQ_EXPECTED_RQ_FITNESS_MODE="standard"
export RQ_EXPECTED_ADMISSION_STRATEGY="random"
export RQ_EXPECTED_ARCHIVE_ADMISSION_STRATEGY="random"
export RQ_EXPECTED_RANDOM_REPLACE_PROBABILITY="0.5"
export RQ_EXPECTED_TRAINING_RANDOM_ORDER="true"
exec bash "$SCRIPT_DIR/run_train_domain_type_8gpu.sh" "$@"
