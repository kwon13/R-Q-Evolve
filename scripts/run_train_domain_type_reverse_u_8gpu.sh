#!/usr/bin/env bash
# Launch the Reverse-U ablation: R_Q = L * (2 - U).
#
# Usage:
#   bash scripts/run_train_domain_type_reverse_u_8gpu.sh --gpus 0,1,2,3,4,5,6,7
#   bash scripts/run_train_domain_type_reverse_u_8gpu.sh --gpus 0,1,2,3,4,5,6,7 --detach
#   bash scripts/run_train_domain_type_reverse_u_8gpu.sh --gpus 0,1,2,3,4,5,6,7 --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RQ_DOMAIN_TYPE_CONFIG="configs/rq_evolve_4b_8gpu_domain_type_reverse_u.yaml"
export RQ_EXPECTED_RQ_FITNESS_MODE="reverse_u"
export RQ_EXPECTED_REVERSE_U_CONSTANT="2.0"
exec bash "$SCRIPT_DIR/run_train_domain_type_8gpu.sh" "$@"
