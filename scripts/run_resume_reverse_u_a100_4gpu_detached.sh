#!/usr/bin/env bash
# Resume the transferred Reverse-U step-96 checkpoint on A100 GPUs 0-3.
# The four-rank checkpoint cannot be loaded directly by an eight-rank worker.
#
# Usage on 41_80GB_8:
#   bash scripts/run_resume_reverse_u_a100_4gpu_detached.sh
#   CONDA_ENV=my-verl-env bash scripts/run_resume_reverse_u_a100_4gpu_detached.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

GPUS="4,5,6,7"
RUN_DIR="$ROOT/rq_output/rq_evolve_4b_domain_type_reverse_u_35cell_4gpu"
DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --help|-h)
      sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "[resume-reverse-u-a100] unknown option: $arg" >&2; exit 2 ;;
  esac
done

bash "$SCRIPT_DIR/verify_reverse_u_handoff_checkpoint.sh" "$RUN_DIR" 96 4

GPU_COUNT=0
while IFS=',' read -r index name memory_mib; do
  index="${index//[[:space:]]/}"
  name="${name#${name%%[![:space:]]*}}"
  name="${name%${name##*[![:space:]]}}"
  memory_mib="${memory_mib//[[:space:]]/}"
  if [[ "$name" != *"A100"* ]] || (( memory_mib < 80000 )); then
    if $DRY_RUN; then
      echo "[resume-reverse-u-a100] dry-run warning: GPU $index is $name (${memory_mib} MiB)"
    else
      echo "[resume-reverse-u-a100] GPU $index is not an 80-GB A100: $name (${memory_mib} MiB)" >&2
      exit 1
    fi
  fi
  GPU_COUNT=$((GPU_COUNT + 1))
done < <(nvidia-smi --id="$GPUS" \
  --query-gpu=index,name,memory.total --format=csv,noheader,nounits)
[[ "$GPU_COUNT" == "4" ]] || { echo "[resume-reverse-u-a100] expected four GPUs" >&2; exit 1; }

if ! $DRY_RUN; then
  BUSY="$(nvidia-smi --id="$GPUS" --query-compute-apps=pid,used_memory \
    --format=csv,noheader 2>/dev/null || true)"
  [[ -z "$BUSY" ]] || {
    echo "[resume-reverse-u-a100] GPUs 0-3 are busy:" >&2
    echo "$BUSY" >&2
    exit 1
  }
fi

set +u
source /data1/yhoon113/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-vllm}" || {
  echo "[resume-reverse-u-a100] conda activation failed: ${CONDA_ENV:-vllm}" >&2
  exit 1
}
set -u

export RQ_DOMAIN_TYPE_CONFIG="configs/rq_evolve_4b_4gpu_domain_type_reverse_u_resume_a100.yaml"
export RQ_EXPECTED_RQ_FITNESS_MODE="reverse_u"
export RQ_EXPECTED_REVERSE_U_CONSTANT="2.0"
export RQ_EXPECTED_RESUME_MODE="auto"
exec bash "$SCRIPT_DIR/run_train_domain_type_4gpu.sh" \
  --gpus "$GPUS" --detach "$@"
