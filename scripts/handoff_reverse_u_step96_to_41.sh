#!/usr/bin/env bash
# Wait for a complete step-96 save, stop only this training driver/merge daemon,
# then transfer the exact four-rank checkpoint and required source to 41_80GB_8.
# This script does NOT shut down either server and does NOT start the remote run.
#
# Usage:
#   bash scripts/handoff_reverse_u_step96_to_41.sh
#   bash scripts/handoff_reverse_u_step96_to_41.sh --dry-run
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_KEY="rq_evolve_4b_domain_type_reverse_u_35cell_4gpu"
RUN_DIR="$ROOT/rq_output/$RUN_KEY"
LOG_DIR="$ROOT/logs/$RUN_KEY"
STEP=96
REMOTE_USER="yhoon113"
REMOTE_HOST="210.125.181.41"
REMOTE_PORT="54329"
REMOTE_ROOT="/data1/yhoon113/R-Q-Evolve"
MODEL_DIR="/data1/yhoon113/qwen3-4b-base"
REMOTE="$REMOTE_USER@$REMOTE_HOST"
CONTROL_PATH="${TMPDIR:-/tmp}/rq_handoff_${UID}_%C"
SSH=(ssh -p "$REMOTE_PORT" -o ControlMaster=auto -o ControlPersist=3600 \
  -o ControlPath="$CONTROL_PATH" -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
RSYNC_SSH="ssh -p $REMOTE_PORT -o ControlMaster=auto -o ControlPersist=3600 -o ControlPath=$CONTROL_PATH -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
elif [[ $# -gt 0 ]]; then
  echo "[handoff] unknown option: $1" >&2
  exit 2
fi

echo "[handoff] source      : $RUN_DIR/global_step_$STEP"
echo "[handoff] destination : $REMOTE:$REMOTE_ROOT/rq_output/$RUN_KEY"
echo "[handoff] resume      : exact 4-rank state on target GPUs 0-3"
if $DRY_RUN; then
  echo "[handoff] dry-run complete; no waiting, stopping, or transfer performed"
  exit 0
fi

# Authenticate before waiting so a password prompt cannot appear unattended
# only after the source job has already been stopped. The shared SSH control
# connection is retained for the subsequent rsync calls.
echo "[handoff] authenticating to $REMOTE (password may be requested once)"
"${SSH[@]}" "$REMOTE" true

echo "[handoff] waiting for latest_checkpointed_iteration.txt == $STEP"
while true; do
  latest="$(tr -dc '0-9' < "$RUN_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || true)"
  if [[ "$latest" == "$STEP" ]]; then
    break
  fi
  if [[ -n "$latest" ]] && (( latest > STEP )); then
    echo "[handoff] checkpoint advanced to $latest; refusing an ambiguous handoff" >&2
    exit 1
  fi
  sleep 5
done

bash scripts/verify_reverse_u_handoff_checkpoint.sh "$RUN_DIR" "$STEP" 4

TRAIN_PID="$(tr -dc '0-9' < "$LOG_DIR/train.pid" 2>/dev/null || true)"
if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
  TRAIN_CMD="$(ps -o args= -p "$TRAIN_PID" 2>/dev/null || true)"
  [[ "$TRAIN_CMD" == *"train_with_verl.py"*"rq_evolve_4b_4gpu_domain_type_reverse_u.yaml"* ]] || {
    echo "[handoff] PID $TRAIN_PID does not match the Reverse-U driver: $TRAIN_CMD" >&2
    exit 1
  }
  echo "[handoff] stopping training driver PID $TRAIN_PID after completed step $STEP"
  kill -TERM "$TRAIN_PID"
  for _ in $(seq 1 120); do
    kill -0 "$TRAIN_PID" 2>/dev/null || break
    sleep 1
  done
  kill -0 "$TRAIN_PID" 2>/dev/null && {
    echo "[handoff] driver did not stop after 120s; no transfer performed" >&2
    exit 1
  }
fi

MERGE_PID="$(tr -dc '0-9' < "$LOG_DIR/auto_merge.pid" 2>/dev/null || true)"
if [[ -n "$MERGE_PID" ]] && kill -0 "$MERGE_PID" 2>/dev/null; then
  MERGE_CMD="$(ps -o args= -p "$MERGE_PID" 2>/dev/null || true)"
  [[ "$MERGE_CMD" == *"auto_merge_checkpoints.py"*"$RUN_KEY"* ]] || {
    echo "[handoff] PID $MERGE_PID does not match this run's merge daemon" >&2
    exit 1
  }
  echo "[handoff] stopping auto-merge PID $MERGE_PID"
  kill -TERM "$MERGE_PID"
  for _ in $(seq 1 30); do
    kill -0 "$MERGE_PID" 2>/dev/null || break
    sleep 1
  done
fi

# The latest pointer is written only after all rank shards and data.pt finish.
# Check stability after the writer and merge daemon have stopped, so the copy
# is a fixed snapshot rather than a directory changing under rsync.
snapshot_a="$(mktemp)"
snapshot_b="$(mktemp)"
trap 'rm -f "$snapshot_a" "$snapshot_b"' EXIT
find "$RUN_DIR/global_step_$STEP" -type f -printf '%P %s\n' | sort > "$snapshot_a"
sleep 10
find "$RUN_DIR/global_step_$STEP" -type f -printf '%P %s\n' | sort > "$snapshot_b"
cmp -s "$snapshot_a" "$snapshot_b" || {
  echo "[handoff] checkpoint files are still changing; rerun after they settle" >&2
  exit 1
}

echo "[handoff] checking SSH destination and free space"
"${SSH[@]}" "$REMOTE" "mkdir -p '$REMOTE_ROOT' '$REMOTE_ROOT/rq_output/$RUN_KEY' '$REMOTE_ROOT/logs/$RUN_KEY'"
SOURCE_BYTES="$(du -sb "$RUN_DIR/global_step_$STEP" "$RUN_DIR/rq_archive" | awk '{s += $1} END {print s}')"
REMOTE_HAS_MODEL="$("${SSH[@]}" "$REMOTE" \
  "if test -s '$MODEL_DIR/config.json'; then echo 1; else echo 0; fi")"
if [[ "$REMOTE_HAS_MODEL" != "1" ]]; then
  [[ -s "$MODEL_DIR/config.json" ]] || {
    echo "[handoff] base model is missing on both source and destination: $MODEL_DIR" >&2
    exit 1
  }
  MODEL_BYTES="$(du -sb "$MODEL_DIR" | awk '{print $1}')"
  SOURCE_BYTES=$((SOURCE_BYTES + MODEL_BYTES))
fi
REMOTE_FREE="$("${SSH[@]}" "$REMOTE" \
  "df -PB1 '$REMOTE_ROOT' | tail -n 1 | tr -dc '0-9'")"
(( REMOTE_FREE > SOURCE_BYTES + SOURCE_BYTES / 5 )) || {
  echo "[handoff] target free space is insufficient: need >$((SOURCE_BYTES + SOURCE_BYTES / 5)), have $REMOTE_FREE" >&2
  exit 1
}

echo "[handoff] syncing source/config/prompt files (secrets and outputs excluded)"
rsync -a --info=progress2 -e "$RSYNC_SSH" \
  --exclude='__pycache__/' --exclude='*.pyc' \
  "$ROOT/src" "$ROOT/patches" "$ROOT/scripts" "$ROOT/configs" \
  "$ROOT/prompt_templates" "$ROOT/seed_programs_domain_type" \
  "$ROOT/pyproject.toml" \
  "$REMOTE:$REMOTE_ROOT/"

if [[ "$REMOTE_HAS_MODEL" != "1" ]]; then
  echo "[handoff] target base model is absent; syncing qwen3-4b-base"
  "${SSH[@]}" "$REMOTE" "mkdir -p '$MODEL_DIR'"
  rsync -a --partial --append-verify --info=progress2 -e "$RSYNC_SSH" \
    "$MODEL_DIR/" "$REMOTE:$MODEL_DIR/"
else
  echo "[handoff] target base model already exists; skipping model transfer"
fi

echo "[handoff] syncing global_step_$STEP (resumable)"
rsync -a --partial --append-verify --info=progress2 -e "$RSYNC_SSH" \
  "$RUN_DIR/global_step_$STEP/" \
  "$REMOTE:$REMOTE_ROOT/rq_output/$RUN_KEY/global_step_$STEP/"

echo "[handoff] syncing archive and source log"
rsync -a --partial --append-verify --info=progress2 -e "$RSYNC_SSH" \
  "$RUN_DIR/rq_archive/" \
  "$REMOTE:$REMOTE_ROOT/rq_output/$RUN_KEY/rq_archive/"
if [[ -e "$LOG_DIR/latest.log" ]]; then
  rsync -aL -e "$RSYNC_SSH" "$LOG_DIR/latest.log" \
    "$REMOTE:$REMOTE_ROOT/logs/$RUN_KEY/source_server_until_step96.log"
fi

# Publish the latest pointer last, after every large file has arrived.
rsync -a -e "$RSYNC_SSH" "$RUN_DIR/latest_checkpointed_iteration.txt" \
  "$REMOTE:$REMOTE_ROOT/rq_output/$RUN_KEY/latest_checkpointed_iteration.txt"

echo "[handoff] verifying checkpoint on target"
"${SSH[@]}" "$REMOTE" \
  "cd '$REMOTE_ROOT' && bash scripts/verify_reverse_u_handoff_checkpoint.sh 'rq_output/$RUN_KEY' '$STEP' 4"

echo "[handoff] transfer complete; source server may now be powered off"
echo "[handoff] remote resume command:"
echo "  ssh -p $REMOTE_PORT $REMOTE 'cd $REMOTE_ROOT && bash scripts/run_resume_reverse_u_a100_4gpu_detached.sh'"
