#!/usr/bin/env bash
# Wait for the complete 8B step-160 save, stop its source driver/merge daemon,
# then transfer the exact eight-rank checkpoint and live R-Q state to 41_80GB_8.
# This script does not shut down either server and does not start the remote run.
#
# Usage on the RTX PRO 6000 source server:
#   bash scripts/handoff_8b_domain_type_step160_to_41.sh
#   bash scripts/handoff_8b_domain_type_step160_to_41.sh --dry-run
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_KEY="rq_evolve_8b_domain_type_35cell_8gpu_rtxpro6000"
RUN_DIR="$ROOT/rq_output/$RUN_KEY"
LOG_DIR="$ROOT/logs/$RUN_KEY"
STEP=160
WORLD_SIZE=8
REMOTE_USER="yhoon113"
REMOTE_HOST="210.125.181.41"
REMOTE_PORT="54329"
REMOTE_ROOT="/data1/yhoon113/R-Q-Evolve"
MODEL_DIR="/data1/yhoon113/qwen3-8b-base"
REMOTE="$REMOTE_USER@$REMOTE_HOST"
CONTROL_PATH="${TMPDIR:-/tmp}/rq_8b_handoff_${UID}_%C"
SSH=(ssh -p "$REMOTE_PORT" -o ControlMaster=auto -o ControlPersist=3600 \
  -o ControlPath="$CONTROL_PATH" -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
RSYNC_SSH="ssh -p $REMOTE_PORT -o ControlMaster=auto -o ControlPersist=3600 -o ControlPath=$CONTROL_PATH -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
elif [[ $# -gt 0 ]]; then
  echo "[handoff-8b] unknown option: $1" >&2
  exit 2
fi

echo "[handoff-8b] source      : $RUN_DIR/global_step_$STEP"
echo "[handoff-8b] destination : $REMOTE:$REMOTE_ROOT/rq_output/$RUN_KEY"
echo "[handoff-8b] resume      : exact 8-rank state on A100 GPUs 0-7"
echo "[handoff-8b] log-prob MB : ref=2, rollout=2 per GPU"
if $DRY_RUN; then
  echo "[handoff-8b] dry-run complete; no waiting, stopping, or transfer performed"
  exit 0
fi

for command_name in ssh rsync; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "[handoff-8b] required command is missing: $command_name" >&2
    exit 1
  }
done

# Authenticate before waiting so an unattended handoff never stops training
# and only then discovers that the destination requires interactive login.
echo "[handoff-8b] authenticating to $REMOTE (password may be requested once)"
"${SSH[@]}" "$REMOTE" true

echo "[handoff-8b] waiting for latest_checkpointed_iteration.txt == $STEP"
while true; do
  latest="$(tr -dc '0-9' < "$RUN_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || true)"
  if [[ "$latest" == "$STEP" ]]; then
    break
  fi
  if [[ -n "$latest" ]] && (( latest > STEP )); then
    echo "[handoff-8b] checkpoint advanced to $latest; refusing an ambiguous handoff" >&2
    exit 1
  fi
  sleep 5
done

bash scripts/verify_8b_domain_type_handoff_checkpoint.sh \
  "$RUN_DIR" "$STEP" "$WORLD_SIZE"

TRAIN_PID="$(tr -dc '0-9' < "$LOG_DIR/train.pid" 2>/dev/null || true)"
if [[ -z "$TRAIN_PID" ]] || ! kill -0 "$TRAIN_PID" 2>/dev/null; then
  TRAIN_PID="$(pgrep -f \
    "train_with_verl.py.*rq_evolve_8b_8gpu_domain_type_rtxpro6000.yaml" \
    | head -1 || true)"
fi
if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
  TRAIN_CMD="$(ps -o args= -p "$TRAIN_PID" 2>/dev/null || true)"
  [[ "$TRAIN_CMD" == *"train_with_verl.py"*"rq_evolve_8b_8gpu_domain_type_rtxpro6000.yaml"* ]] || {
    echo "[handoff-8b] PID $TRAIN_PID does not match the 8B RTX PRO driver: $TRAIN_CMD" >&2
    exit 1
  }
  echo "[handoff-8b] stopping training driver PID $TRAIN_PID after completed step $STEP"
  kill -TERM "$TRAIN_PID"
  for _ in $(seq 1 120); do
    kill -0 "$TRAIN_PID" 2>/dev/null || break
    sleep 1
  done
  kill -0 "$TRAIN_PID" 2>/dev/null && {
    echo "[handoff-8b] driver did not stop after 120s; no transfer performed" >&2
    exit 1
  }
fi

MERGE_PID="$(tr -dc '0-9' < "$LOG_DIR/auto_merge.pid" 2>/dev/null || true)"
if [[ -n "$MERGE_PID" ]] && kill -0 "$MERGE_PID" 2>/dev/null; then
  MERGE_CMD="$(ps -o args= -p "$MERGE_PID" 2>/dev/null || true)"
  [[ "$MERGE_CMD" == *"auto_merge_checkpoints.py"*"$RUN_KEY"* ]] || {
    echo "[handoff-8b] PID $MERGE_PID does not match this run's merge daemon" >&2
    exit 1
  }
  echo "[handoff-8b] stopping auto-merge PID $MERGE_PID"
  kill -TERM "$MERGE_PID"
  for _ in $(seq 1 30); do
    kill -0 "$MERGE_PID" 2>/dev/null || break
    sleep 1
  done
fi

# Verify that neither the checkpoint nor live R-Q state changes after the
# writers have stopped. data.pt carries the exact step-160 MAP, while the live
# archive is also transferred for logs, analysis, and the adapter pre-load.
snapshot_a="$(mktemp)"
snapshot_b="$(mktemp)"
trap 'rm -f "$snapshot_a" "$snapshot_b"' EXIT
find "$RUN_DIR/global_step_$STEP" "$RUN_DIR/rq_archive" \
  -type f -printf '%p %s\n' | sort > "$snapshot_a"
sleep 10
find "$RUN_DIR/global_step_$STEP" "$RUN_DIR/rq_archive" \
  -type f -printf '%p %s\n' | sort > "$snapshot_b"
cmp -s "$snapshot_a" "$snapshot_b" || {
  echo "[handoff-8b] checkpoint/archive files are still changing; rerun after they settle" >&2
  exit 1
}

echo "[handoff-8b] checking destination and free space"
"${SSH[@]}" "$REMOTE" \
  "mkdir -p '$REMOTE_ROOT' '$REMOTE_ROOT/rq_output/$RUN_KEY' '$REMOTE_ROOT/logs/$RUN_KEY'"
REMOTE_LATEST="$("${SSH[@]}" "$REMOTE" \
  "tr -dc '0-9' < '$REMOTE_ROOT/rq_output/$RUN_KEY/latest_checkpointed_iteration.txt' 2>/dev/null || true")"
if [[ -n "$REMOTE_LATEST" && "$REMOTE_LATEST" != "$STEP" ]]; then
  echo "[handoff-8b] target run already points to step $REMOTE_LATEST, expected empty or $STEP" >&2
  exit 1
fi

SOURCE_BYTES="$(du -sb "$RUN_DIR/global_step_$STEP" "$RUN_DIR/rq_archive" \
  | awk '{s += $1} END {print s}')"
REMOTE_HAS_MODEL="$("${SSH[@]}" "$REMOTE" \
  "if test -s '$MODEL_DIR/config.json'; then echo 1; else echo 0; fi")"
if [[ "$REMOTE_HAS_MODEL" != "1" ]]; then
  [[ -s "$MODEL_DIR/config.json" ]] || {
    echo "[handoff-8b] base model is missing on both source and destination: $MODEL_DIR" >&2
    exit 1
  }
  MODEL_BYTES="$(du -sb "$MODEL_DIR" | awk '{print $1}')"
  SOURCE_BYTES=$((SOURCE_BYTES + MODEL_BYTES))
fi
REMOTE_FREE="$("${SSH[@]}" "$REMOTE" \
  "df -PB1 '$REMOTE_ROOT' | awk 'NR == 2 {print \$4}'")"
[[ "$REMOTE_FREE" =~ ^[0-9]+$ ]] || {
  echo "[handoff-8b] could not determine target free space: $REMOTE_FREE" >&2
  exit 1
}
(( REMOTE_FREE > SOURCE_BYTES + SOURCE_BYTES / 5 )) || {
  echo "[handoff-8b] target free space is insufficient: need >$((SOURCE_BYTES + SOURCE_BYTES / 5)), have $REMOTE_FREE" >&2
  exit 1
}

echo "[handoff-8b] syncing source/config/prompt files (secrets and outputs excluded)"
rsync -a --info=progress2 -e "$RSYNC_SSH" \
  --exclude='__pycache__/' --exclude='*.pyc' \
  "$ROOT/src" "$ROOT/patches" "$ROOT/scripts" "$ROOT/configs" \
  "$ROOT/prompt_templates" "$ROOT/seed_programs_domain_type" \
  "$ROOT/pyproject.toml" \
  "$REMOTE:$REMOTE_ROOT/"

if [[ "$REMOTE_HAS_MODEL" != "1" ]]; then
  echo "[handoff-8b] target base model is absent; syncing qwen3-8b-base"
  "${SSH[@]}" "$REMOTE" "mkdir -p '$MODEL_DIR'"
  rsync -a --partial --append-verify --info=progress2 -e "$RSYNC_SSH" \
    "$MODEL_DIR/" "$REMOTE:$MODEL_DIR/"
else
  echo "[handoff-8b] target base model already exists; skipping model transfer"
fi

echo "[handoff-8b] syncing global_step_$STEP (resumable)"
rsync -a --partial --append-verify --info=progress2 -e "$RSYNC_SSH" \
  "$RUN_DIR/global_step_$STEP/" \
  "$REMOTE:$REMOTE_ROOT/rq_output/$RUN_KEY/global_step_$STEP/"

echo "[handoff-8b] syncing archive and source log"
rsync -a --partial --append-verify --info=progress2 -e "$RSYNC_SSH" \
  "$RUN_DIR/rq_archive/" \
  "$REMOTE:$REMOTE_ROOT/rq_output/$RUN_KEY/rq_archive/"
if [[ -e "$LOG_DIR/latest.log" ]]; then
  rsync -aL -e "$RSYNC_SSH" "$LOG_DIR/latest.log" \
    "$REMOTE:$REMOTE_ROOT/logs/$RUN_KEY/source_server_until_step160.log"
fi

# Publish the latest pointer only after every large checkpoint file arrives.
rsync -a -e "$RSYNC_SSH" "$RUN_DIR/latest_checkpointed_iteration.txt" \
  "$REMOTE:$REMOTE_ROOT/rq_output/$RUN_KEY/latest_checkpointed_iteration.txt"

echo "[handoff-8b] verifying checkpoint on target"
"${SSH[@]}" "$REMOTE" \
  "cd '$REMOTE_ROOT' && bash scripts/verify_8b_domain_type_handoff_checkpoint.sh 'rq_output/$RUN_KEY' '$STEP' '$WORLD_SIZE'"

echo "[handoff-8b] transfer complete; source server may now be powered off"
echo "[handoff-8b] remote resume command:"
echo "  ssh -p $REMOTE_PORT $REMOTE 'cd $REMOTE_ROOT && bash scripts/run_resume_domain_type_8b_a100_8gpu_detached.sh'"
