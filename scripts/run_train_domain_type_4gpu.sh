#!/usr/bin/env bash
# Launch the fresh 35-cell DOMAIN x PROBLEM_TYPE run on four selected GPUs.
#
# Usage (from an activated training environment):
#   bash scripts/run_train_domain_type_4gpu.sh --gpus 0,1,2,3
#   bash scripts/run_train_domain_type_4gpu.sh --gpus 0,1,2,3 --detach
#   bash scripts/run_train_domain_type_4gpu.sh --gpus 0,1,2,3 --dry-run
#
# Checkpoints are written every 32 steps.  A companion daemon converts every
# previous FSDP actor checkpoint to hf_merged/ and retains the newest actor/
# intact as a full recovery artifact without consuming unbounded disk.  This
# fresh-run launcher itself deliberately keeps automatic resume disabled.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

CONFIG="configs/rq_evolve_4b_4gpu_domain_type.yaml"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
DETACH=false
DRY_RUN=false

usage() {
  sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      [[ $# -ge 2 ]] || { echo "[domain-type-4gpu] --gpus requires a value" >&2; exit 2; }
      GPUS="$2"
      shift 2
      ;;
    --detach)
      DETACH=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[domain-type-4gpu] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -f "$CONFIG" ]] || { echo "[domain-type-4gpu] missing config: $CONFIG" >&2; exit 1; }

if ! RESOLVED="$(python3 - "$CONFIG" 2>&1 <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "src")
from omegaconf import OmegaConf
from rq_evolve.config import load_raw_config

cfg = load_raw_config(sys.argv[1])

def get(path):
    value = OmegaConf.select(cfg, path)
    if value is None and path != "verl_config.trainer.max_actor_ckpt_to_keep":
        raise SystemExit(f"missing effective config value: {path}")
    return value

output = Path(str(get("verl_config.trainer.default_local_dir"))).expanduser()
if not output.is_absolute():
    output = (Path.cwd() / output).resolve()

keep = get("verl_config.trainer.max_actor_ckpt_to_keep")
print(output)
print(get("verl_config.trainer.n_gpus_per_node"))
print(get("verl_config.trainer.save_freq"))
print("null" if keep is None else keep)
print(get("verl_config.trainer.total_training_steps"))
print(get("verl_config.actor_rollout_ref.model.path"))
print(get("evolution.seed_programs_dir"))
print(str(get("evolution.target_cell_injection")).lower())
print(str(get("evolution.relabel_skill")).lower())
print(str(get("evolution.structural_inspiration")).lower())
print(str(get("verl_config.trainer.resume_mode")).lower())
PY
)"; then
  echo "[domain-type-4gpu] config resolution failed; activate the training environment:" >&2
  echo "$RESOLVED" >&2
  exit 1
fi

mapfile -t VALUES <<< "$RESOLVED"
CKPT_DIR="${VALUES[0]}"
EXPECTED_GPUS="${VALUES[1]}"
SAVE_FREQ="${VALUES[2]}"
MAX_KEEP="${VALUES[3]}"
TOTAL_STEPS="${VALUES[4]}"
MODEL_PATH="${VALUES[5]}"
SEED_DIR="${VALUES[6]}"
TARGET_INJECTION="${VALUES[7]}"
RELABEL_SKILL="${VALUES[8]}"
STRUCTURAL_INSPIRATION="${VALUES[9]}"
RESUME_MODE="${VALUES[10]}"

[[ "$EXPECTED_GPUS" == "4" ]] || { echo "[domain-type-4gpu] config must request four GPUs" >&2; exit 1; }
[[ "$SAVE_FREQ" == "32" ]] || { echo "[domain-type-4gpu] save_freq must be 32" >&2; exit 1; }
[[ "$MAX_KEEP" == "null" ]] || { echo "[domain-type-4gpu] max_actor_ckpt_to_keep must be null" >&2; exit 1; }
[[ "$SEED_DIR" == "seed_programs_domain_type" ]] || { echo "[domain-type-4gpu] wrong seed directory: $SEED_DIR" >&2; exit 1; }
[[ "$TARGET_INJECTION" == "false" ]] || { echo "[domain-type-4gpu] targeted mutation must be disabled" >&2; exit 1; }
[[ "$RELABEL_SKILL" == "false" ]] || { echo "[domain-type-4gpu] legacy skill relabelling must be disabled" >&2; exit 1; }
[[ "$STRUCTURAL_INSPIRATION" == "false" ]] || { echo "[domain-type-4gpu] donor context must be disabled for the clean run" >&2; exit 1; }
[[ "$RESUME_MODE" == "disable" ]] || { echo "[domain-type-4gpu] resume_mode must be disable" >&2; exit 1; }

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
[[ "${#GPU_IDS[@]}" == "$EXPECTED_GPUS" ]] || {
  echo "[domain-type-4gpu] --gpus must name exactly four devices, got: $GPUS" >&2
  exit 2
}
declare -A SEEN_GPUS=()
for gpu in "${GPU_IDS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "[domain-type-4gpu] invalid GPU id: $gpu" >&2; exit 2; }
  [[ -z "${SEEN_GPUS[$gpu]:-}" ]] || { echo "[domain-type-4gpu] duplicate GPU id: $gpu" >&2; exit 2; }
  SEEN_GPUS[$gpu]=1
done

seed_count=$(find "$SEED_DIR" -maxdepth 1 -type f -name '*.py' | wc -l)
[[ "$seed_count" == "7" ]] || {
  echo "[domain-type-4gpu] expected exactly 7 diagonal seed programs, found $seed_count" >&2
  exit 1
}

if [[ -d "$CKPT_DIR" ]] && [[ -n "$(find "$CKPT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "[domain-type-4gpu] refusing to reuse non-empty run directory: $CKPT_DIR" >&2
  echo "[domain-type-4gpu] choose a new config identity and output directory for a fresh run" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_KEY="$(basename "$CKPT_DIR")"
LOG_DIR="$ROOT/logs/$RUN_KEY"
TRAIN_LOG="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
MERGE_LOG="$LOG_DIR/auto_merge.log"
MERGE_PID_FILE="$LOG_DIR/auto_merge.pid"
TRAIN_PID_FILE="$LOG_DIR/train.pid"

echo "[domain-type-4gpu] config      : $CONFIG"
echo "[domain-type-4gpu] model       : $MODEL_PATH"
echo "[domain-type-4gpu] output      : $CKPT_DIR"
echo "[domain-type-4gpu] GPUs        : $CUDA_VISIBLE_DEVICES"
echo "[domain-type-4gpu] steps/save  : $TOTAL_STEPS / every $SAVE_FREQ"
echo "[domain-type-4gpu] descriptors : 7 DOMAIN x 5 PROBLEM_TYPE"
echo "[domain-type-4gpu] mutation    : untargeted; local admission only"

if $DRY_RUN; then
  echo "[domain-type-4gpu] dry-run complete; no process started"
  exit 0
fi

mkdir -p "$CKPT_DIR" "$LOG_DIR"

MERGE_PID=""
if [[ -f "$MERGE_PID_FILE" ]]; then
  CANDIDATE_PID="$(tr -dc '0-9' < "$MERGE_PID_FILE")"
  if [[ -n "$CANDIDATE_PID" ]] && kill -0 "$CANDIDATE_PID" 2>/dev/null; then
    MERGE_PID="$CANDIDATE_PID"
  fi
fi
if [[ -z "$MERGE_PID" ]]; then
  MERGE_PID="$(pgrep -f "auto_merge_checkpoints.py.*${RUN_KEY}" | head -1 || true)"
fi

if [[ -n "$MERGE_PID" ]]; then
  echo "[domain-type-4gpu] auto-merge  : already running (PID $MERGE_PID)"
else
  nohup python3 scripts/auto_merge_checkpoints.py \
    --ckpt_dir "$CKPT_DIR" --interval 60 \
    >> "$MERGE_LOG" 2>&1 &
  MERGE_PID=$!
  echo "$MERGE_PID" > "$MERGE_PID_FILE"
  sleep 2
  if ! kill -0 "$MERGE_PID" 2>/dev/null; then
    echo "[domain-type-4gpu] auto-merge exited; inspect $MERGE_LOG" >&2
    exit 1
  fi
  echo "[domain-type-4gpu] auto-merge  : started (PID $MERGE_PID)"
fi

echo "[domain-type-4gpu] merge log   : $MERGE_LOG"
echo "[domain-type-4gpu] train log   : $TRAIN_LOG"

if $DETACH; then
  nohup python3 scripts/train_with_verl.py --config "$CONFIG" \
    > "$TRAIN_LOG" 2>&1 &
  TRAIN_PID=$!
  echo "$TRAIN_PID" > "$TRAIN_PID_FILE"
  ln -sfn "$TRAIN_LOG" "$LOG_DIR/latest.log"
  echo "[domain-type-4gpu] training    : started detached (PID $TRAIN_PID)"
  echo "[domain-type-4gpu] follow      : tail -f $LOG_DIR/latest.log"
else
  echo "[domain-type-4gpu] training    : starting in foreground"
  set -o pipefail
  python3 scripts/train_with_verl.py --config "$CONFIG" 2>&1 | tee "$TRAIN_LOG"
fi
