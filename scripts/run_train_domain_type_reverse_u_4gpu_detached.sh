#!/usr/bin/env bash
# Launch Reverse-U, R_Q = L * (2 - U), detached on physical GPUs 0-3.
#
# Usage (from an activated training environment):
#   bash scripts/run_train_domain_type_reverse_u_4gpu_detached.sh
#   bash scripts/run_train_domain_type_reverse_u_4gpu_detached.sh --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RQ_DOMAIN_TYPE_CONFIG="configs/rq_evolve_4b_4gpu_domain_type_reverse_u.yaml"
export RQ_EXPECTED_RQ_FITNESS_MODE="reverse_u"
export RQ_EXPECTED_REVERSE_U_CONSTANT="2.0"
exec bash "$SCRIPT_DIR/run_train_domain_type_4gpu.sh" \
  --gpus 0,1,2,3 --detach "$@"
