#!/usr/bin/env bash
# Verify the exact eight-rank step-160 checkpoint used for the 8B server handoff.
# Usage: bash scripts/verify_8b_domain_type_handoff_checkpoint.sh [RUN_DIR] [STEP] [WORLD_SIZE]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-$ROOT/rq_output/rq_evolve_8b_domain_type_35cell_8gpu_rtxpro6000}"
STEP="${2:-160}"
WORLD_SIZE="${3:-8}"
CKPT="$RUN_DIR/global_step_$STEP"
ACTOR="$CKPT/actor"
LATEST="$RUN_DIR/latest_checkpointed_iteration.txt"

fail() {
  echo "[verify-8b-handoff] ERROR: $*" >&2
  exit 1
}

[[ "$STEP" =~ ^[0-9]+$ ]] || fail "step must be a non-negative integer: $STEP"
[[ "$WORLD_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "world size must be positive: $WORLD_SIZE"
[[ -d "$CKPT" ]] || fail "missing checkpoint directory: $CKPT"
[[ -s "$LATEST" ]] || fail "missing latest checkpoint pointer: $LATEST"
LATEST_STEP="$(tr -dc '0-9' < "$LATEST")"
[[ "$LATEST_STEP" == "$STEP" ]] || fail "latest checkpoint is $LATEST_STEP, expected $STEP"
[[ -s "$CKPT/data.pt" ]] || fail "missing dataloader/MAP state: $CKPT/data.pt"
[[ -s "$ACTOR/fsdp_config.json" ]] || fail "missing actor FSDP metadata"
[[ -s "$ACTOR/huggingface/config.json" ]] || fail "missing actor Hugging Face metadata"
[[ -s "$RUN_DIR/rq_archive/archive.json" ]] || fail "missing live MAP archive"
[[ -s "$RUN_DIR/rq_archive/rq_iteration.json" ]] || fail "missing evolution iteration state"

for ((rank = 0; rank < WORLD_SIZE; rank++)); do
  for kind in model optim extra_state; do
    file="$ACTOR/${kind}_world_size_${WORLD_SIZE}_rank_${rank}.pt"
    [[ -s "$file" ]] || fail "missing or empty shard: $file"
  done
done

for kind in model optim extra_state; do
  count="$(find "$ACTOR" -maxdepth 1 -type f \
    -name "${kind}_world_size_${WORLD_SIZE}_rank_*.pt" | wc -l)"
  [[ "$count" == "$WORLD_SIZE" ]] || fail "$kind shard count is $count, expected $WORLD_SIZE"
done

echo "[verify-8b-handoff] checkpoint : global_step_$STEP"
echo "[verify-8b-handoff] world size : $WORLD_SIZE"
echo "[verify-8b-handoff] actor size : $(du -sh "$ACTOR" | awk '{print $1}')"
echo "[verify-8b-handoff] archive    : $(du -sh "$RUN_DIR/rq_archive" | awk '{print $1}')"
echo "[verify-8b-handoff] status     : complete"
