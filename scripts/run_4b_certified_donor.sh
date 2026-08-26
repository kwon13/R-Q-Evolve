#!/usr/bin/env bash
# Minimal runner for the certified structural-donor v2 arm.
#
# Usage:
#   conda activate azr-bw-blackwell
#   bash scripts/run_4b_certified_donor.sh --gpus 0,1,4,6
#   bash scripts/run_4b_certified_donor.sh --gpus 0,1,4,6 --detach
#
# This intentionally skips the heavyweight launcher's GPU/W&B/download checks.
# It validates only method/checkpoint invariants, starts auto-merge, and trains.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

CONFIG="configs/rq_evolve_4b_4gpu_structural_inspiration_v2.yaml"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
DETACH=false
DRY_RUN=false

usage() {
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      [[ $# -ge 2 ]] || { echo "[donor-v2] --gpus requires a value" >&2; exit 2; }
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
      echo "[donor-v2] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -f "$CONFIG" ]] || { echo "[donor-v2] missing config: $CONFIG" >&2; exit 1; }

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
print(get("verl_config.trainer.save_freq"))
print("null" if keep is None else keep)
print(get("verl_config.trainer.total_training_steps"))
print(get("verl_config.actor_rollout_ref.model.path"))
print(str(get("evolution.structural_inspiration")).lower())
print(str(get("evolution.structural_inspiration_require_certified_donor")).lower())
print(str(get("evolution.structural_inspiration_require_positive_rq")).lower())
print(get("evolution.structural_inspiration_max_token_jaccard"))
print(str(get("evolution.use_evaluator")).lower())
PY
)"; then
  echo "[donor-v2] config resolution failed; activate the training environment:" >&2
  echo "$RESOLVED" >&2
  exit 1
fi

mapfile -t VALUES <<< "$RESOLVED"
CKPT_DIR="${VALUES[0]}"
SAVE_FREQ="${VALUES[1]}"
MAX_KEEP="${VALUES[2]}"
TOTAL_STEPS="${VALUES[3]}"
MODEL_PATH="${VALUES[4]}"
INSPIRATION_ENABLED="${VALUES[5]}"
CERTIFIED_REQUIRED="${VALUES[6]}"
POSITIVE_RQ_REQUIRED="${VALUES[7]}"
JACCARD_MAX="${VALUES[8]}"
USE_EVALUATOR="${VALUES[9]}"

[[ "$SAVE_FREQ" == "32" ]] || { echo "[donor-v2] save_freq must be 32" >&2; exit 1; }
[[ "$MAX_KEEP" == "null" ]] || { echo "[donor-v2] max_actor_ckpt_to_keep must be null" >&2; exit 1; }
[[ "$INSPIRATION_ENABLED" == "true" ]] || { echo "[donor-v2] donor is not enabled" >&2; exit 1; }
[[ "$CERTIFIED_REQUIRED" == "true" ]] || { echo "[donor-v2] certified donor gate is off" >&2; exit 1; }
[[ "$POSITIVE_RQ_REQUIRED" == "true" ]] || { echo "[donor-v2] positive-R_Q gate is off" >&2; exit 1; }
[[ "$JACCARD_MAX" == "0.45" ]] || { echo "[donor-v2] Jaccard threshold is not 0.45" >&2; exit 1; }
[[ "$USE_EVALUATOR" == "false" ]] || { echo "[donor-v2] evaluator/self-judge must be disabled" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPUS"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_KEY="$(basename "$CKPT_DIR")"
LOG_DIR="$ROOT/logs/$RUN_KEY"
TRAIN_LOG="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
MERGE_LOG="$LOG_DIR/auto_merge.log"
MERGE_PID_FILE="$LOG_DIR/auto_merge.pid"
TRAIN_PID_FILE="$LOG_DIR/train.pid"

echo "[donor-v2] config      : $CONFIG"
echo "[donor-v2] model       : $MODEL_PATH"
echo "[donor-v2] output      : $CKPT_DIR"
echo "[donor-v2] GPUs        : $CUDA_VISIBLE_DEVICES"
echo "[donor-v2] steps/save  : $TOTAL_STEPS / every $SAVE_FREQ"
echo "[donor-v2] evaluator   : disabled (no API/model judge)"
echo "[donor-v2] donor gate  : manual seed certification AND R_Q>0"
echo "[donor-v2] copy gate   : token Jaccard < $JACCARD_MAX"

if $DRY_RUN; then
  echo "[donor-v2] dry-run complete; no process started"
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
  echo "[donor-v2] auto-merge  : already running (PID $MERGE_PID)"
else
  nohup python3 scripts/auto_merge_checkpoints.py \
    --ckpt_dir "$CKPT_DIR" --interval 60 \
    >> "$MERGE_LOG" 2>&1 &
  MERGE_PID=$!
  echo "$MERGE_PID" > "$MERGE_PID_FILE"
  sleep 2
  if ! kill -0 "$MERGE_PID" 2>/dev/null; then
    echo "[donor-v2] auto-merge exited; inspect $MERGE_LOG" >&2
    exit 1
  fi
  echo "[donor-v2] auto-merge  : started (PID $MERGE_PID)"
fi

echo "[donor-v2] merge log   : $MERGE_LOG"
echo "[donor-v2] train log   : $TRAIN_LOG"

if $DETACH; then
  nohup python3 scripts/train_with_verl.py --config "$CONFIG" \
    > "$TRAIN_LOG" 2>&1 &
  TRAIN_PID=$!
  echo "$TRAIN_PID" > "$TRAIN_PID_FILE"
  ln -sfn "$TRAIN_LOG" "$LOG_DIR/latest.log"
  echo "[donor-v2] training    : started detached (PID $TRAIN_PID)"
  echo "[donor-v2] follow      : tail -f $LOG_DIR/latest.log"
else
  echo "[donor-v2] training    : starting in foreground"
  set -o pipefail
  python3 scripts/train_with_verl.py --config "$CONFIG" 2>&1 | tee "$TRAIN_LOG"
fi
