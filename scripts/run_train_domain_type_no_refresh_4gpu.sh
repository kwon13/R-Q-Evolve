#!/usr/bin/env bash
# Launch the No-refresh ablation on 4 GPUs: seed_refresh=false (always evaluate and train on seed 0).
#
# Usage:
#   bash scripts/run_train_domain_type_no_refresh_4gpu.sh --gpus 4,5,6,7
#   bash scripts/run_train_domain_type_no_refresh_4gpu.sh --gpus 4,5,6,7 --detach
#   bash scripts/run_train_domain_type_no_refresh_4gpu.sh --gpus 4,5,6,7 --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RQ_DOMAIN_TYPE_CONFIG="configs/rq_evolve_4b_4gpu_domain_type_no_refresh.yaml"
exec bash "$SCRIPT_DIR/run_train_domain_type_4gpu.sh" "$@"
