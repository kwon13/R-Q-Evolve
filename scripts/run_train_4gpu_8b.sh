#!/usr/bin/env bash
# One 8B training arm on a SUBSET of the GPUs, sharing the box with another
# tenant.
#
#   bash scripts/run_train_4gpu_8b.sh
#   GPUS=0,1,2,3 bash scripts/run_train_4gpu_8b.sh configs/rq_evolve_8b_4gpu.yaml
#
# Sibling of run_train_8gpu.sh (which is the right launcher when all eight are
# ours) and of run_train_8gpu_4b_shared.sh (same idea, 4B, and calibrated for
# the A100 host's `vllm` env). This one differs from BOTH in three ways:
#
#  1. The idle check is a HEADROOM check on the named GPUs only. Refusing to
#     start unless every GPU on the box is empty -- what run_train_8gpu.sh does
#     -- is right when they should be empty; here half of them will not be, and
#     the only question is whether what is left on OUR half fits.
#
#  2. Card size is READ, not assumed. run_train_8gpu_4b_shared.sh hardcodes an
#     80.0 GiB A100 in its arithmetic. These are 95.6 GiB Blackwell cards, so
#     vLLM at the same gpu_memory_utilization holds ~8 GiB more than that
#     script would predict. Getting this wrong makes the check pass on a run
#     that then OOMs in backward, minutes in -- which is the exact failure the
#     check exists to prevent.
#
#  3. It starts the auto-merge daemon and refuses to run without it. save_freq
#     32 over 256 steps is 8 checkpoints and an unmerged 8B actor/ is 92 GB, so
#     the run would need ~736 GB against 491 GB free. auto_merge_checkpoints.py
#     collapses each PREVIOUS checkpoint to a ~16 GB hf_merged and reclaims its
#     actor/. The 2026-08-23 8B run lost its step-32/64/96 weights precisely
#     because that merge was nobody's job.
set -uo pipefail

ROOT=/data1/yhoon113/R-Q-Evolve
CONFIG=${1:-${CONFIG:-configs/rq_evolve_8b_4gpu.yaml}}
cd "$ROOT"
[ -f "$CONFIG" ] || { echo "[4gpu] no such config: $CONFIG" >&2; exit 1; }
NAME=$(basename "$CONFIG" .yaml)
LOGDIR=${LOGDIR:-$ROOT/rq_output/${NAME}_logs}
mkdir -p "$LOGDIR"

# GPUS names the devices; the config's n_gpus_per_node says how many there must
# be. A mismatch is a silent disaster -- verl would shard over a rank count the
# memory budget was never computed for -- and verl only notices AFTER Ray and
# the model shards are up, two minutes in. Check it here.
NGPU=$(grep -oP '^\s+n_gpus_per_node:\s*\K[0-9]+' "$CONFIG" | head -1)
GPUS=${GPUS:-$(seq -s, 0 $(( ${NGPU:-4} - 1 )))}
export CUDA_VISIBLE_DEVICES="$GPUS"
GPU_COUNT=$(awk -F, '{print NF}' <<< "$GPUS")
if [ "$GPU_COUNT" != "${NGPU:-4}" ]; then
  echo "[4gpu] ABORT: GPUS=$GPUS is $GPU_COUNT devices but $CONFIG asks for ${NGPU}." >&2
  exit 1
fi

# ---------------------------------------------------------------- headroom ---
# The one MEASURED term is the trainer's torch reserved figure: 40.4 GiB, held
# flat across all 128 steps of the 2026-08-23 8-rank micro=2 8B run
# (perf/max_memory_reserved_gb in rq_output/rq_evolve_8b_8gpu_logs). Everything
# else below is derived from it, and derived numbers are why TRAINER_GIB is
# overridable: read perf/max_memory_reserved_gb off this run's first few steps
# and set it for real.
#
#   vLLM footprint (GiB) ~ gpu_memory_utilization x CARD_GIB + 3.6 overhead
#
# CARD_GIB comes from nvidia-smi, so this holds on 80 GiB A100s and 95.6 GiB
# Blackwells alike.
UTIL=$(grep -oP '^\s+gpu_memory_utilization:\s*\K[0-9.]+' "$CONFIG" | head -1)
MICRO=$(grep -oP '^\s+ppo_micro_batch_size_per_gpu:\s*\K[0-9]+' "$CONFIG" | head -1)
OPT_OFFLOAD=$(grep -oP '^\s+optimizer_offload:\s*\K\w+' "$CONFIG" | head -1)
CARD_MIB=$(nvidia-smi --id="${GPUS%%,*}" --query-gpu=memory.total --format=csv,noheader,nounits | tr -dc '0-9')
CARD_GIB=$(python3 -c "print(round($CARD_MIB/1024, 1))")

# Qwen3-8B: 8.19e9 params. Per-rank FSDP replica terms, in GiB:
#   params bf16 16.4/R, grads bf16 16.4/R, AdamW m+v fp32 65.5/R, master fp32 32.8/R
# optimizer_offload moves m, v and master (98.3/R) to host RAM.
TRAINER_GIB=${TRAINER_GIB:-$(python3 -c "
R = ${NGPU:-4}
base = 40.4                                     # MEASURED: 8 ranks, micro 2
base += (16.4 / R) - (16.4 / 8)                 # params bf16
base += (16.4 / R) - (16.4 / 8)                 # grads bf16
base += (98.3 / R) - (98.3 / 8)                 # AdamW m,v + master fp32
if '${OPT_OFFLOAD:-false}'.lower() == 'true':
    base -= 98.3 / R                            # those three go to host RAM
if ${MICRO:-2} > 2: base += 4.5 * (${MICRO:-2} / 2 - 1)   # activations + logits
print(round(base, 1))")}
VLLM_GIB=$(python3 -c "print(round(${UTIL:-0.45}*$CARD_GIB + 3.6, 1))")
NEED=$(python3 -c "print(round($VLLM_GIB + $TRAINER_GIB + 0.5, 1))")

echo "[4gpu] config    : $CONFIG"
echo "[4gpu] card      : ${CARD_GIB} GiB"
echo "[4gpu] budget    : vLLM ${VLLM_GIB} GiB (util ${UTIL} x ${CARD_GIB} + 3.6)"
echo "[4gpu]             + trainer ~${TRAINER_GIB} GiB (${NGPU} ranks, micro=${MICRO}, opt_offload=${OPT_OFFLOAD:-false}, ESTIMATE)"
echo "[4gpu]             + context 0.5 GiB   =>  needs ${NEED} GiB free per GPU"

blocked=0
while IFS=, read -r idx used total; do
  used=$(echo "$used" | tr -dc '0-9'); total=$(echo "$total" | tr -dc '0-9')
  free_gib=$(python3 -c "print(round(($total-$used)/1024, 1))")
  ok=$(python3 -c "print('OK ' if $free_gib >= $NEED else 'SHORT')")
  printf '[4gpu]   gpu %s: %6.1f GiB free   %s\n' "$idx" "$free_gib" "$ok"
  [ "$ok" = "SHORT" ] && blocked=1
done < <(nvidia-smi --id="$GPUS" --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | tr -d ' ')

if [ "$blocked" = "1" ]; then
  cat >&2 <<MSG
[4gpu] ABORT: at least one of GPUs $GPUS is short of ${NEED} GiB.
[4gpu] This run would OOM in backward, minutes in, not at startup -- which is
[4gpu] why this check exists. Options, in order of preference:
[4gpu]   * wait for the co-tenant, then use configs/rq_evolve_8b_8gpu.yaml on all eight
[4gpu]   * drop gpu_memory_utilization (0.34 frees ~10.5 GiB, KV cache halves)
[4gpu]   * actor.fsdp_config.param_offload: true (~4 GiB/rank, costs a transfer per micro-step)
[4gpu] Set TRAINER_GIB=<n> to override the trainer estimate if you have measured it.
[4gpu]
[4gpu] Compute processes currently on the box -- a killed run leaves a worker
[4gpu] holding tens of GiB, and it is yours to kill, not a co-tenant's:
MSG
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader |
    while IFS=, read -r pid mem; do
      pid=$(echo "$pid" | tr -dc '0-9')
      printf '[4gpu]   pid %-8s %-12s %s\n' "$pid" "$(echo "$mem" | xargs)" \
        "$(ps -o user=,args= -p "$pid" 2>/dev/null | cut -c1-90)"
    done >&2
  exit 1
fi

# --------------------------------------------------------------- host RAM ---
# With optimizer_offload the fp32 m/v/master live in system memory: 98.3 GiB
# spread over the ranks, i.e. 98.3 GiB total however many ranks there are.
# That is the scarce resource on this box, not VRAM -- a co-tenant already
# holds ~364 GiB. If it does not fit, the OOM killer takes a worker mid-update
# and the traceback blames Ray.
if [ "${OPT_OFFLOAD:-false}" = "true" ]; then
  RAM_AVAIL=$(free -g | awk '/^Mem:/{print $7}')
  # 260, not the 140 this first shipped with. 140 was an estimate and it was
  # WRONG: the 2026-08-25 attempt passed the check at 239 GiB available and
  # Ray's memory monitor still killed two workers at step 1. The generation
  # backend died with it, all 32 mutations returned empty output, and the run
  # ended on `VerlDynamicDataset is empty`. 98 GiB of offloaded fp32 state is
  # only part of the total: four WorkerDicts each stage the full 16 GiB
  # checkpoint on the host during load, and the AgentLoopWorkers and object
  # store sit on top of that.
  RAM_NEED=${RAM_NEED:-260}
  echo "[4gpu] host ram  : ${RAM_AVAIL} GiB available, need ~${RAM_NEED} GiB (optimizer_offload)"
  if [ "$RAM_AVAIL" -lt "$RAM_NEED" ]; then
    cat >&2 <<MSG
[4gpu] ABORT: ${RAM_AVAIL} GiB of host RAM available, under ${RAM_NEED} GiB.
[4gpu] optimizer_offload is the wrong trade on this box. Ray kills workers at
[4gpu] 95% of total system memory, and the co-tenant already holds most of it.
[4gpu] Set actor.fsdp_config.optimizer_offload: false and pay on the GPU
[4gpu] instead -- gpu_memory_utilization 0.30 is what configs/rq_evolve_8b_4gpu.yaml
[4gpu] does, and GPUs 0-3 have the room.
MSG
    exit 1
  fi
fi

# ------------------------------------------------------------------- disk ---
FREE_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
echo "[4gpu] disk      : ${FREE_GB} GB free on $ROOT"
if [ "$FREE_GB" -lt 200 ]; then
  echo "[4gpu] ABORT: under 200 GB free. One unmerged 8B checkpoint is 92 GB and" >&2
  echo "[4gpu] the latest one is never merged (it is the resume point)." >&2
  exit 1
fi

# -------------------------------------------------------------------- env ---
# sm_120 (RTX PRO 6000 Blackwell) needs the cu128 stack; azr-bw's cu126 build
# dies with "no kernel image is available". Activate rather than prepend PATH
# so ninja/nvcc come with it -- verl JIT-compiles at startup. This is the env
# the 2026-08-23 8B run used; the 4B launchers' `vllm` env is the A100 host's
# and does not exist here.
# `set -u` off across the activate: the env's binutils hook reads ADDR2LINE and
# friends before setting them, so an unset-variable abort here is conda's bug.
set +u
source /data1/yhoon113/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-azr-bw-blackwell}" || { echo "[4gpu] conda activate ${CONDA_ENV:-azr-bw-blackwell} failed" >&2; exit 1; }
set -u

# Set here, not inherited: a stray WANDB_MODE=offline left over from a smoke
# shell would log this run to disk only, and nobody finds out until they look.
export WANDB_MODE=${WANDB_MODE:-online}
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$ROOT/.env" ]; then
  export WANDB_API_KEY=$(grep -E '^WANDB_API_KEY=' "$ROOT/.env" | cut -d= -f2-)
fi
[ -z "${WANDB_API_KEY:-}" ] && { echo "[4gpu] no WANDB_API_KEY; aborting" >&2; exit 1; }

CKPT_DIR=$(grep -oP '^\s+default_local_dir:\s*\K\S+' "$CONFIG" | head -1)
CKPT_DIR=${CKPT_DIR:-./rq_output/${NAME}}
mkdir -p "$CKPT_DIR"

# --------------------------------------------------------------- automerge ---
if pgrep -f "auto_merge_checkpoints.py.*$(basename "$CKPT_DIR")" >/dev/null; then
  echo "[4gpu] automerge : already running for $CKPT_DIR"
else
  nohup python scripts/auto_merge_checkpoints.py \
    --ckpt_dir "$CKPT_DIR" --interval 60 \
    > "$LOGDIR/auto_merge.log" 2>&1 &
  echo "$!" > "$LOGDIR/auto_merge.pid"
  sleep 3
  if kill -0 "$(cat "$LOGDIR/auto_merge.pid")" 2>/dev/null; then
    echo "[4gpu] automerge : started pid $(cat "$LOGDIR/auto_merge.pid") -> $LOGDIR/auto_merge.log"
  else
    echo "[4gpu] ABORT: auto-merge daemon died immediately; see $LOGDIR/auto_merge.log" >&2
    exit 1
  fi
fi

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/${NAME}_$STAMP.log"
echo "[4gpu] model     : $(grep -oP '^\s+path:\s*\K\S+' "$CONFIG" | head -1)"
echo "[4gpu] gpus      : $CUDA_VISIBLE_DEVICES"
echo "[4gpu] wandb     : $WANDB_MODE"
echo "[4gpu] log       : $LOG"

nohup python scripts/train_with_verl.py --config "$CONFIG" > "$LOG" 2>&1 &
echo "$!" | tee "$LOGDIR/run.pid" | xargs -I{} echo "[4gpu] pid       : {}"
ln -sfn "$LOG" "$LOGDIR/latest.log"
echo "[4gpu] watch     : tail -f $LOGDIR/latest.log"
