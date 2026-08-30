#!/usr/bin/env bash
# Launch the No-U ablation: R_Q = L (the U multiplier is fixed to one).
#
# Usage:
#   bash scripts/run_train_domain_type_no_u_8gpu.sh --gpus 0,1,2,3,4,5,6,7
#   bash scripts/run_train_domain_type_no_u_8gpu.sh --gpus 0,1,2,3,4,5,6,7 --detach
#   bash scripts/run_train_domain_type_no_u_8gpu.sh --gpus 0,1,2,3,4,5,6,7 --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RQ_DOMAIN_TYPE_CONFIG="configs/rq_evolve_4b_8gpu_domain_type_no_u.yaml"
export RQ_EXPECTED_RQ_FITNESS_MODE="no_u"
exec bash "$SCRIPT_DIR/run_train_domain_type_8gpu.sh" "$@"
