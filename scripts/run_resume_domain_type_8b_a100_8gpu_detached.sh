#!/usr/bin/env bash
# Resume the transferred 8B step-160 checkpoint on all eight 80-GB A100 GPUs.
#
# Usage on 41_80GB_8:
#   bash scripts/run_resume_domain_type_8b_a100_8gpu_detached.sh
#   CONDA_ENV=my-verl-env bash scripts/run_resume_domain_type_8b_a100_8gpu_detached.sh
#   bash scripts/run_resume_domain_type_8b_a100_8gpu_detached.sh --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

GPUS="0,1,2,3,4,5,6,7"
RUN_DIR="$ROOT/rq_output/rq_evolve_8b_domain_type_35cell_8gpu_rtxpro6000"
CONFIG="configs/rq_evolve_8b_8gpu_domain_type_a100_resume.yaml"
DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --help|-h)
      sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "[resume-8b-a100] unknown option: $arg" >&2; exit 2 ;;
  esac
done

bash "$SCRIPT_DIR/verify_8b_domain_type_handoff_checkpoint.sh" "$RUN_DIR" 160 8

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "[resume-8b-a100] nvidia-smi is required" >&2
  exit 1
}

GPU_COUNT=0
while IFS=',' read -r index name memory_mib; do
  index="${index//[[:space:]]/}"
  name="${name#${name%%[![:space:]]*}}"
  name="${name%${name##*[![:space:]]}}"
  memory_mib="${memory_mib//[[:space:]]/}"
  if [[ "$name" != *"A100"* ]] || (( memory_mib < 80000 )); then
    if $DRY_RUN; then
      echo "[resume-8b-a100] dry-run warning: GPU $index is $name (${memory_mib} MiB)"
    else
      echo "[resume-8b-a100] GPU $index is not an 80-GB A100: $name (${memory_mib} MiB)" >&2
      exit 1
    fi
  fi
  GPU_COUNT=$((GPU_COUNT + 1))
done < <(nvidia-smi --id="$GPUS" \
  --query-gpu=index,name,memory.total --format=csv,noheader,nounits)
[[ "$GPU_COUNT" == "8" ]] || {
  echo "[resume-8b-a100] expected eight GPUs, detected $GPU_COUNT" >&2
  exit 1
}

if ! $DRY_RUN; then
  BUSY="$(nvidia-smi --id="$GPUS" --query-compute-apps=pid,used_memory \
    --format=csv,noheader 2>/dev/null || true)"
  [[ -z "$BUSY" ]] || {
    echo "[resume-8b-a100] GPUs 0-7 are busy:" >&2
    echo "$BUSY" >&2
    exit 1
  }
fi

if python3 -c 'import omegaconf, verl' >/dev/null 2>&1; then
  echo "[resume-8b-a100] environment : using active Python environment"
else
  set +u
  source /data1/yhoon113/miniforge3/etc/profile.d/conda.sh
  set -u
  if [[ -n "${CONDA_ENV:-}" ]]; then
    ENV_CANDIDATES=("$CONDA_ENV")
  else
    ENV_CANDIDATES=(vllm azr)
  fi
  SELECTED_ENV=""
  for candidate in "${ENV_CANDIDATES[@]}"; do
    if conda run -n "$candidate" python -c 'import omegaconf, verl' \
      >/dev/null 2>&1; then
      SELECTED_ENV="$candidate"
      break
    fi
  done
  [[ -n "$SELECTED_ENV" ]] || {
    echo "[resume-8b-a100] no usable training environment found; set CONDA_ENV" >&2
    exit 1
  }
  set +u
  conda activate "$SELECTED_ENV"
  set -u
  echo "[resume-8b-a100] environment : conda $SELECTED_ENV"
fi

# Resolve inheritance and refuse launch unless both A100 log-prob paths are 2.
mapfile -t RESUME_VALUES < <(python3 - "$CONFIG" <<'PY'
import sys
sys.path.insert(0, "src")
from omegaconf import OmegaConf
from rq_evolve.config import load_raw_config

cfg = load_raw_config(sys.argv[1])
for path in (
    "verl_config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu",
    "verl_config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
    "verl_config.trainer.n_gpus_per_node",
    "verl_config.trainer.resume_mode",
):
    print(OmegaConf.select(cfg, path))
PY
)
[[ "${RESUME_VALUES[0]:-}" == "2" ]] || {
  echo "[resume-8b-a100] ref log-prob micro-batch must be 2" >&2; exit 1;
}
[[ "${RESUME_VALUES[1]:-}" == "2" ]] || {
  echo "[resume-8b-a100] rollout log-prob micro-batch must be 2" >&2; exit 1;
}
[[ "${RESUME_VALUES[2]:-}" == "8" ]] || {
  echo "[resume-8b-a100] checkpoint requires eight workers" >&2; exit 1;
}
[[ "${RESUME_VALUES[3]:-}" == "auto" ]] || {
  echo "[resume-8b-a100] resume_mode must be auto" >&2; exit 1;
}

export RQ_DOMAIN_TYPE_CONFIG="$CONFIG"
export RQ_EXPECTED_RQ_FITNESS_MODE="standard"
export RQ_EXPECTED_REVERSE_U_CONSTANT="2.0"
export RQ_EXPECTED_RESUME_MODE="auto"
exec bash "$SCRIPT_DIR/run_train_domain_type_8gpu.sh" \
  --gpus "$GPUS" --detach "$@"
