#!/usr/bin/env bash
# One 4B training arm on all eight GPUs, SHARING them with another tenant.
#
#   bash scripts/run_train_8gpu_4b_shared.sh
#   GPUS=4,5,6,7 bash scripts/run_train_8gpu_4b_shared.sh configs/rq_evolve_4b_4gpu.yaml
#
# Differs from run_train_8gpu_4b.sh in exactly two ways, both about not owning
# the machine:
#
#  1. The idle check is a HEADROOM check. Refusing to start unless the GPUs are
#     empty is right when they should be empty; here they will not be, and the
#     question is whether what is left fits. The budget below is measured, not
#     assumed -- see configs/rq_evolve_4b_8gpu_shared.yaml for the derivation.
#
#  2. It starts the auto-merge daemon and refuses to run without it. save_freq
#     32 over 256 steps is 8 checkpoints; an unmerged actor/ is 47 GB, so the
#     run needs ~376 GB it does not have (243 GB free at time of writing).
#     auto_merge_checkpoints.py collapses each previous checkpoint to an ~8 GB
#     hf_merged and reclaims the actor/, which brings the steady state to about
#     110 GB. The 2026-08-23 run lost its step-32/64/96 weights precisely
#     because that merge was nobody's job.
set -uo pipefail

ROOT=/data1/yhoon113/R-Q-Evolve
CONFIG=${1:-${CONFIG:-configs/rq_evolve_4b_8gpu_shared.yaml}}
cd "$ROOT"
[ -f "$CONFIG" ] || { echo "[shared] no such config: $CONFIG" >&2; exit 1; }
NAME=$(basename "$CONFIG" .yaml)
LOGDIR=${LOGDIR:-$ROOT/rq_output/${NAME}_logs}
mkdir -p "$LOGDIR"

# GPUS names the devices; the config's n_gpus_per_node says how many there must
# be. A mismatch is a silent disaster -- verl would shard over a rank count the
# memory budget below was never computed for -- so it is checked, not assumed.
NGPU=$(grep -oP '^\s+n_gpus_per_node:\s*\K[0-9]+' "$CONFIG" | head -1)
GPUS=${GPUS:-$(seq -s, 0 $(( ${NGPU:-8} - 1 )))}
export CUDA_VISIBLE_DEVICES="$GPUS"
GPU_COUNT=$(awk -F, '{print NF}' <<< "$GPUS")
if [ "$GPU_COUNT" != "${NGPU:-8}" ]; then
  echo "[shared] ABORT: GPUS=$GPUS is $GPU_COUNT devices but the config asks for ${NGPU}." >&2
  exit 1
fi

# ---------------------------------------------------------------- headroom ---
# Measured on an idle 80.0 GiB A100 with this repo's own settings:
#
#   vLLM footprint (GiB) = gpu_memory_utilization x 80 + 3.6
#       0.38 -> 34.0 (KV 21.5 GiB / 156,704 tok)   [measured]
#       0.30 -> 27.6 (KV 15.2 GiB / 110,896 tok)   [measured]
#
#   verl trainer torch reserved = 41.2 GiB at ppo_micro_batch_size_per_gpu 4,
#       constant across all 243 steps of the 2026-08-24 run
#       (perf/max_memory_reserved_gb). Halving the micro-batch is ESTIMATED to
#       free 4-5 GiB -- verl reports one figure, not a per-term breakdown -- so
#       TRAINER_GIB below is an estimate and deliberately conservative. Override
#       it if you measure otherwise.
UTIL=$(grep -oP '^\s+gpu_memory_utilization:\s*\K[0-9.]+' "$CONFIG" | head -1)
MICRO=$(grep -oP '^\s+ppo_micro_batch_size_per_gpu:\s*\K[0-9]+' "$CONFIG" | head -1)
OPT_OFFLOAD=$(grep -oP '^\s+optimizer_offload:\s*\K\w+' "$CONFIG" | head -1)
# Two measured anchors, both verl's perf/max_memory_reserved_gb:
#
#   8 ranks, micro 4, no offload   41.2 GiB   flat across all 243 steps
#                                              (2026-08-24 run)
#   4 ranks, micro 2, offload      34.2 GiB   step 1 of the 2026-08-25 run
#
# The derivation below reproduces the first by construction and lands 1.5 GiB
# UNDER the second, so it carries a flat +2.0 pad. Padding rather than fitting
# a coefficient: one measurement is not enough to fit to, and the failure mode
# of being low here is an OOM in backward several minutes into a run.
TRAINER_GIB=${TRAINER_GIB:-$(python3 -c "
base = 41.2                                   # measured: 8 ranks, micro 4
base += (8.0 / ${NGPU:-8} - 1.0) * 8.0        # FSDP replica terms scale with 1/ranks
if ${MICRO:-4} <= 2: base -= 4.5              # activations + logits halve
if '${OPT_OFFLOAD:-false}'.lower() == 'true': base -= 12.0 * (8.0 / ${NGPU:-8}) / 2.0
base += 2.0                                   # see the anchors above
print(round(base, 1))")}
NEED=$(python3 -c "print(round(${UTIL:-0.38}*80 + 3.6 + $TRAINER_GIB + 0.5, 1))")

echo "[shared] config    : $CONFIG"
echo "[shared] budget    : vLLM $(python3 -c "print(round(${UTIL:-0.38}*80+3.6,1))") GiB (util ${UTIL})"
echo "[shared]             + trainer ~${TRAINER_GIB} GiB (${NGPU} ranks, micro=${MICRO}, opt_offload=${OPT_OFFLOAD:-false}, ESTIMATE)"
echo "[shared]             + context 0.5 GiB   =>  needs ${NEED} GiB free per GPU"

blocked=0
while IFS=, read -r idx used total; do
  used=$(echo "$used" | tr -dc '0-9'); total=$(echo "$total" | tr -dc '0-9')
  free_gib=$(python3 -c "print(round(($total-$used)/1024, 1))")
  ok=$(python3 -c "print('OK ' if $free_gib >= $NEED else 'SHORT')")
  printf '[shared]   gpu %s: %6.1f GiB free   %s\n' "$idx" "$free_gib" "$ok"
  [ "$ok" = "SHORT" ] && blocked=1
done < <(nvidia-smi --id="$GPUS" --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | tr -d ' ')

if [ "$blocked" = "1" ]; then
  cat >&2 <<MSG
[shared] ABORT: at least one GPU is short of ${NEED} GiB.
[shared] This run will OOM in backward, minutes in, not at startup -- which is
[shared] why this check exists. Options, in order of preference:
[shared]   * wait for the co-tenant, then use configs/rq_evolve_4b_8gpu.yaml
[shared]   * drop gpu_memory_utilization further (0.16 -> 16.4 GiB, KV ~39k tok)
[shared]   * actor.fsdp_config.optimizer_offload: true (~6 GiB, RAM has room)
[shared] Set TRAINER_GIB=<n> to override the trainer estimate if you have measured it.
MSG
  exit 1
fi

# ------------------------------------------------------------------- disk ---
FREE_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
echo "[shared] disk      : ${FREE_GB} GB free on $ROOT"
if [ "$FREE_GB" -lt 150 ]; then
  echo "[shared] ABORT: under 150 GB free. One unmerged checkpoint is 47 GB and" >&2
  echo "[shared] the latest one is never merged (it is the resume point)." >&2
  exit 1
fi

set +u
source /data1/yhoon113/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-vllm}" || { echo "[shared] conda activate failed" >&2; exit 1; }
export CUDA_HOME=/data1/yhoon113/cuda-12.8
export PATH=/data1/yhoon113/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/data1/yhoon113/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/data1/yhoon113/miniforge3/envs/vllm/lib/libgomp.so.1
set -u

export WANDB_MODE=${WANDB_MODE:-online}
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$ROOT/.env" ]; then
  export WANDB_API_KEY=$(grep -E '^WANDB_API_KEY=' "$ROOT/.env" | cut -d= -f2-)
fi
[ -z "${WANDB_API_KEY:-}" ] && { echo "[shared] no WANDB_API_KEY; aborting" >&2; exit 1; }

CKPT_DIR=$(grep -oP '^\s+default_local_dir:\s*\K\S+' "$CONFIG" | head -1)
CKPT_DIR=${CKPT_DIR:-./rq_output/${NAME}}
mkdir -p "$CKPT_DIR"

# --------------------------------------------------------------- automerge ---
if pgrep -f "auto_merge_checkpoints.py.*$(basename "$CKPT_DIR")" >/dev/null; then
  echo "[shared] automerge : already running for $CKPT_DIR"
else
  nohup python scripts/auto_merge_checkpoints.py \
    --ckpt_dir "$CKPT_DIR" --interval 60 \
    > "$LOGDIR/auto_merge.log" 2>&1 &
  echo "$!" > "$LOGDIR/auto_merge.pid"
  sleep 3
  if kill -0 "$(cat "$LOGDIR/auto_merge.pid")" 2>/dev/null; then
    echo "[shared] automerge : started pid $(cat "$LOGDIR/auto_merge.pid") -> $LOGDIR/auto_merge.log"
  else
    echo "[shared] ABORT: auto-merge daemon died immediately; see $LOGDIR/auto_merge.log" >&2
    exit 1
  fi
fi

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/${NAME}_$STAMP.log"
echo "[shared] model     : $(grep -oP '^\s+path:\s*\K\S+' "$CONFIG" | head -1)"
echo "[shared] gpus      : $CUDA_VISIBLE_DEVICES"
echo "[shared] wandb     : $WANDB_MODE"
echo "[shared] log       : $LOG"

nohup python scripts/train_with_verl.py --config "$CONFIG" > "$LOG" 2>&1 &
echo "$!" | tee "$LOGDIR/run.pid" | xargs -I{} echo "[shared] pid       : {}"
ln -sfn "$LOG" "$LOGDIR/latest.log"
echo "[shared] watch     : tail -f $LOGDIR/latest.log"
