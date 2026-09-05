#!/usr/bin/env bash
# Fresh R_Q + epsilon-admission run; all other method settings inherit Standard.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_SIZE=4b
PROFILE=a100
EPSILON=0.25
MODEL_PATH=""
FORWARD_ARGS=()

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_train_domain_type_epsilon_8gpu.sh [options]
  --gpus LIST             Eight distinct GPU indices (default: CUDA_VISIBLE_DEVICES or 0-7)
  --epsilon FLOAT         Admission probability in [0, 1] (default: 0.25)
  --model-size 4b|8b      Model size (default: 4b)
  --profile a100|rtxpro6000  Existing hardware profile (default: a100; RTX requires 8b)
  --model-path PATH      Override the profile's local model path or Hugging Face ID
  --detach               Start training under nohup and print the log/PID locations
  --dry-run              Resolve and validate config without loading a model or launching workers
  --help                 Print this help

Fresh runs only. Each model/profile/epsilon has a separate output and W&B identity.
Activate the server's existing training environment before a real launch.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epsilon|--model-size|--profile|--model-path|--gpus)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || {
        echo "[epsilon-8gpu] $1 requires a value" >&2; exit 2;
      }
      case "$1" in
        --epsilon) EPSILON="$2" ;;
        --model-size) MODEL_SIZE="$2" ;;
        --profile) PROFILE="$2" ;;
        --model-path) MODEL_PATH="$2" ;;
        --gpus) FORWARD_ARGS+=(--gpus "$2") ;;
      esac
      shift 2
      ;;
    --detach|--dry-run) FORWARD_ARGS+=("$1"); shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "[epsilon-8gpu] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$MODEL_SIZE:$PROFILE" in
  4b:a100) CONFIG="configs/rq_evolve_4b_8gpu_domain_type_epsilon.yaml" ;;
  8b:a100) CONFIG="configs/rq_evolve_8b_8gpu_domain_type_epsilon_a100.yaml" ;;
  8b:rtxpro6000) CONFIG="configs/rq_evolve_8b_8gpu_domain_type_epsilon_rtxpro6000.yaml" ;;
  *) echo "[epsilon-8gpu] unsupported model/profile: $MODEL_SIZE / $PROFILE" >&2; exit 2 ;;
esac

if ! EPSILON="$(python3 - "$EPSILON" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit("[epsilon-8gpu] --epsilon must be a finite number in [0, 1]")
if not math.isfinite(value) or not 0.0 <= value <= 1.0:
    raise SystemExit("[epsilon-8gpu] --epsilon must be a finite number in [0, 1]")
print(str(value if value else 0.0))
PY
)"; then
  exit 2
fi

# Own every method/identity override: an old shell from a different arm must
# not silently change this fresh epsilon run. nohup inherits these exports.
export RQ_DOMAIN_TYPE_CONFIG="$CONFIG"
export RQ_ADMISSION_EPSILON="$EPSILON"
export RQ_EXPECTED_ADMISSION_STRATEGY=epsilon_greedy
export RQ_EXPECTED_ADMISSION_EPSILON="$EPSILON"
export RQ_EXPECTED_RQ_FITNESS_MODE=standard
export RQ_EXPECTED_REVERSE_U_CONSTANT=2.0
export RQ_EXPECTED_RESUME_MODE=disable
unset RQ_EPSILON_MODEL_PATH
if [[ -n "$MODEL_PATH" ]]; then
  export RQ_EPSILON_MODEL_PATH="$MODEL_PATH"
fi

exec bash "$SCRIPT_DIR/run_train_domain_type_8gpu.sh" "${FORWARD_ARGS[@]+${FORWARD_ARGS[@]}}"
