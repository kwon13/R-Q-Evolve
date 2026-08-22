#!/usr/bin/env bash
# Two 4B arms that differ in ONE thing: whether U enters the fitness.
#
#   rq_evolve_4b_base          select_ignores_uncertainty=false  -> R_Q = s(1-s)U
#   rq_evolve_4b_ablate_nounc  select_ignores_uncertainty=true   -> R_Q = s(1-s)
#
# Everything else is identical: n=5 x m=2 fitness on fresh seeds, replay batch,
# RLOO, policy judge, 256 steps, test_freq 32, one checkpoint retained.
# 4 GPUs each, so both fit on the one 8-GPU node.
#
#   bash scripts/run_u_ablation.sh
set -uo pipefail

ROOT=/data1/yhoon113/R-Q-Evolve
LOGDIR=${LOGDIR:-$ROOT/rq_output/u_ablation_logs}
mkdir -p "$LOGDIR"
cd "$ROOT"

# Set here, not inherited. These are real 256-step runs and they are meant to be
# watched live; a stray WANDB_MODE=offline from a smoke-test shell would leave
# them logging to disk only, which is not discovered until someone looks.
export WANDB_MODE=${WANDB_MODE:-online}
if [ "$WANDB_MODE" != "online" ]; then
  echo "[u-ablation] WARNING: WANDB_MODE=$WANDB_MODE (not online)"
fi
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$ROOT/.env" ]; then
  export WANDB_API_KEY=$(grep -E '^WANDB_API_KEY=' "$ROOT/.env" | cut -d= -f2-)
fi
[ -z "${WANDB_API_KEY:-}" ] && { echo "[u-ablation] no WANDB_API_KEY; aborting"; exit 1; }
echo "[u-ablation] wandb mode: $WANDB_MODE"

launch() {  # name, config, gpus
  local name=$1 config=$2 gpus=$3
  echo "[u-ablation] $name on GPUs $gpus"
  CUDA_VISIBLE_DEVICES=$gpus nohup python scripts/train_with_verl.py \
      --config "$config" > "$LOGDIR/$name.log" 2>&1 &
  echo "$!" > "$LOGDIR/$name.pid"
}

launch withU   configs/rq_evolve_4b_base.yaml         0,1,2,3
sleep 60   # stagger: two Ray heads racing to bind ports is the one avoidable failure
launch noU     configs/rq_evolve_4b_ablate_nounc.yaml 4,5,6,7

echo "[u-ablation] launched; logs in $LOGDIR"
