#!/usr/bin/env bash
# Launch the 8B DOMAIN x PROBLEM_TYPE run detached on RTX PRO 6000 GPUs 0-7.
#
# Usage:
#   bash scripts/run_train_domain_type_8b_rtxpro6000_8gpu_detached.sh
#   bash scripts/run_train_domain_type_8b_rtxpro6000_8gpu_detached.sh --dry-run
#
# Override the Blackwell conda environment only when the target server uses a
# different name:
#   CONDA_ENV=my-blackwell-env bash scripts/run_train_domain_type_8b_rtxpro6000_8gpu_detached.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

GPUS="0,1,2,3,4,5,6,7"
DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --help|-h)
      sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "[8b-rtxpro6000] unknown option: $arg" >&2
      exit 2
      ;;
  esac
done

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "[8b-rtxpro6000] nvidia-smi is required" >&2
  exit 1
}

GPU_COUNT=0
while IFS=',' read -r index name memory_mib; do
  index="${index//[[:space:]]/}"
  name="${name#${name%%[![:space:]]*}}"
  name="${name%${name##*[![:space:]]}}"
  memory_mib="${memory_mib//[[:space:]]/}"
  [[ "$name" == *"RTX PRO 6000 Blackwell Server Edition"* ]] || {
    echo "[8b-rtxpro6000] GPU $index is not an RTX PRO 6000 Blackwell Server Edition: $name" >&2
    exit 1
  }
  (( memory_mib >= 96000 )) || {
    echo "[8b-rtxpro6000] GPU $index has only ${memory_mib} MiB; expected a 96-GB card" >&2
    exit 1
  }
  GPU_COUNT=$((GPU_COUNT + 1))
done < <(nvidia-smi --id="$GPUS" \
  --query-gpu=index,name,memory.total --format=csv,noheader,nounits)

[[ "$GPU_COUNT" == "8" ]] || {
  echo "[8b-rtxpro6000] expected eight GPUs, detected $GPU_COUNT" >&2
  exit 1
}

# A dry-run validates hardware/config even while another future job occupies
# the box. A real launch refuses any selected GPU with a live compute process.
if ! $DRY_RUN; then
  BUSY="$(nvidia-smi --id="$GPUS" \
    --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "$BUSY" ]]; then
    echo "[8b-rtxpro6000] GPUs 0-7 are not idle:" >&2
    echo "$BUSY" >&2
    exit 1
  fi
fi


export RQ_DOMAIN_TYPE_CONFIG="configs/rq_evolve_8b_8gpu_domain_type_rtxpro6000.yaml"
export RQ_EXPECTED_RQ_FITNESS_MODE="standard"
export RQ_EXPECTED_REVERSE_U_CONSTANT="2.0"
exec bash "$SCRIPT_DIR/run_train_domain_type_8gpu.sh" \
  --gpus "$GPUS" --detach "$@"
