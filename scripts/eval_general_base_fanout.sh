#!/usr/bin/env bash
# General-domain reasoning eval (MMLU-Pro, SuperGPQA, BBEH) for already-HF
# models — the baselines, not the training checkpoints.
#
# This is to eval_general_fanout.sh what analysis/eval_base_model_fanout.sh is
# to eval_steps_fanout.sh: no FSDP merge stage, the model dir goes straight to
# vLLM. It differs from that math sibling in taking a LIST of models, because
# the baselines are always compared as a set and 3 benchmarks alone do not fill
# an 8-GPU pool.
#
# Layout matches the checkpoint eval so collect_general_scores.py works
# unchanged: results land in $OUTROOT/global_step_0/eval_general/<bench>/, where
# step 0 means "the model as given".
#
# Knobs are copied from eval_general_fanout.sh so a baseline row is comparable
# to a checkpoint row: greedy, max_tokens 8192, max_model_len 12000, and a
# stratified 1000-question sample per benchmark at seed 42. MAX_SAMPLES=0 runs
# the full ~42k splits and takes days.
set -uo pipefail

RQ=/data1/yhoon113/R-Q-Evolve
EVAL_SCRIPT="$RQ/scripts/eval_general_vllm.py"

# Same cu128 env as the checkpoint fan-outs — see eval_steps_fanout.sh for why
# the old standalone `vllm` env / cuda-12.8 paths are gone, and why the binutils
# activate hook has to run without `set -u`.
CONDA_ENV="${CONDA_ENV:-azr-bw-blackwell}"
source /data1/yhoon113/miniforge3/etc/profile.d/conda.sh
set +u
conda activate "$CONDA_ENV"
set -u
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
PY="${PY:-$CONDA_PREFIX/bin/python}"

# Comma-separated model dirs (or hub ids). Override with MODELS=/a,/b.
IFS=',' read -ra MODELS <<< "${MODELS:-/data1/yhoon113/Spiral-Qwen3-4B-Multi-Env,/data1/yhoon113/Spiral-Qwen3-8B-Multi-Env,/data1/yhoon113/qwen3-4b-base,/data1/yhoon113/qwen3-8b-base}"
IFS=',' read -ra GPUS <<< "${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra BENCHES <<< "${BENCH_LIST:-mmlupro,supergpqa,bbeh}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
MAXTOK="${MAXTOK:-8192}"
MAXMODELLEN="${MAXMODELLEN:-12000}"
OUTDIR_NAME="${OUTDIR_NAME:-eval_general}"

export PYTHONPATH="$RQ:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
cd "$RQ"

ts() { date +%H:%M:%S; }

outroot_for() {  # <model> -> the eval_<name> folder this model writes into
  echo "$RQ/rq_output/eval_$(basename "$1")"
}

# ---- 0. sanity: a local path must actually hold weights ---------------------
for m in "${MODELS[@]}"; do
  if [[ "$m" == /* || "$m" == .* || "$m" == ~* ]]; then
    if [[ ! -f "$m/config.json" ]]; then
      echo "[$(ts)] ERROR: $m/config.json not found — is this an HF model dir?" >&2
      exit 1
    fi
  else
    echo "[$(ts)] $m is a hub id, not a local dir — letting vLLM resolve it"
  fi
done

echo "[$(ts)] [plan] models: ${MODELS[*]}"
echo "[$(ts)] [plan] benches: ${BENCHES[*]}   max_samples=$MAX_SAMPLES   gpus: ${GPUS[*]}"

# ---- 1. job list ------------------------------------------------------------
JOBS=()
for m in "${MODELS[@]}"; do
  root="$(outroot_for "$m")"
  for b in "${BENCHES[@]}"; do
    # Resume-friendly, same as the checkpoint fan-out: a finished benchmark is
    # not re-run unless FORCE=1.
    if [[ -f "$root/global_step_0/$OUTDIR_NAME/$b/summary.json" ]] \
       && [[ "${FORCE:-0}" != "1" ]]; then
      echo "[$(ts)] [skip] $(basename "$m") $b (summary exists; FORCE=1 to redo)"
      continue
    fi
    JOBS+=("$m|$b")
  done
done
echo "[$(ts)] [plan] ${#JOBS[@]} jobs"
[[ ${#JOBS[@]} -eq 0 ]] && { echo "[$(ts)] nothing to do"; exit 0; }

run_job() {  # <gpu> <model> <bench>
  local gpu="$1" m="$2" b="$3"
  local root; root="$(outroot_for "$m")"
  local out="$root/global_step_0/$OUTDIR_NAME/$b"
  mkdir -p "$out" "$root/global_step_0/$OUTDIR_NAME/logs"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$EVAL_SCRIPT" \
    --model "$m" --tokenizer "$m" \
    --benchmark "$b" --output_dir "$out" \
    --max_tokens "$MAXTOK" --max_model_len "$MAXMODELLEN" \
    --temperature 0.0 --top_p 1.0 \
    --tensor_parallel_size 1 --gpu_memory_utilization 0.85 \
    --dtype bfloat16 --enforce_eager --no_tqdm \
    --max_samples "$MAX_SAMPLES" --sample_seed "$SAMPLE_SEED" \
    >"$root/global_step_0/$OUTDIR_NAME/logs/${b}.log" 2>&1
}

# ---- 2. worker pool ---------------------------------------------------------
declare -A PID_INFO   # pid -> "gpu|model|bench"
FREE=("${GPUS[@]}")
ok=0; fail=0
reap() {
  wait -n 2>/dev/null
  for pid in "${!PID_INFO[@]}"; do
    kill -0 "$pid" 2>/dev/null && continue
    IFS='|' read -r g mdl nm <<< "${PID_INFO[$pid]}"
    if wait "$pid"; then
      ok=$((ok+1)); echo "[$(ts)] [done] $(basename "$mdl") $nm (gpu $g) OK"
    else
      fail=$((fail+1))
      echo "[$(ts)] [done] $(basename "$mdl") $nm (gpu $g) FAIL -> $(outroot_for "$mdl")/global_step_0/$OUTDIR_NAME/logs/${nm}.log"
    fi
    unset "PID_INFO[$pid]"; FREE+=("$g")
    return
  done
}
for job in "${JOBS[@]}"; do
  IFS='|' read -r m b <<< "$job"
  while [[ ${#FREE[@]} -eq 0 ]]; do reap; done
  gpu="${FREE[0]}"; FREE=("${FREE[@]:1}")
  echo "[$(ts)] [launch] $(basename "$m") $b -> gpu $gpu"
  run_job "$gpu" "$m" "$b" & PID_INFO[$!]="$gpu|$m|$b"
done
while [[ ${#PID_INFO[@]} -gt 0 ]]; do reap; done
echo "[$(ts)] [pool] done  ok=$ok fail=$fail"

# ---- 3. aggregate, one table per model --------------------------------------
for m in "${MODELS[@]}"; do
  root="$(outroot_for "$m")"
  # The aggregator writes scores_general.md itself and echoes it — do not
  # redirect stdout here or the two writers race on the same file.
  BASE="$root" OUTDIR_NAME="$OUTDIR_NAME" "$PY" "$RQ/scripts/collect_general_scores.py" \
    >/dev/null 2>&1 \
    && echo "[$(ts)] wrote $root/scores_general.md" \
    || echo "[$(ts)] WARN: aggregation failed for $(basename "$m")"
done
echo "[$(ts)] ALL DONE"
