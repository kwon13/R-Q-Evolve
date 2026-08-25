#!/usr/bin/env bash
# Evaluate R-Q-Evolve checkpoints on general-domain reasoning benchmarks.
#
# The math sibling of this script is eval_steps_fanout.sh and the two share a
# shape: merge FSDP shards to HF once per step, then fan (step x benchmark)
# jobs across a GPU pool, one vLLM per GPU at tp=1. Anything already merged by
# the math run is reused, so running both costs one merge.
#
# Benchmarks are ports of R-Zero/evaluation/eval_{mmlupro,supergpqa,bbeh}.py
# (see scripts/eval_general_vllm.py for what was kept and what was changed).
#
# SIZE WARNING. The full splits are ~42k questions:
#     mmlupro 12,032 | supergpqa 26,529 | bbeh 4,520
# At 8192 max tokens across 8 checkpoints that is ~340k long generations --
# days of GPU time. MAX_SAMPLES defaults to a stratified 1,000 per benchmark
# per step, which is ~24k generations total and finishes overnight. Set
# MAX_SAMPLES=0 for the full splits, and expect it to take days.
set -uo pipefail

RQ=/data1/yhoon113/R-Q-Evolve
EVAL_SCRIPT="$RQ/scripts/eval_general_vllm.py"
# Same cu128 env as the math sibling — see eval_steps_fanout.sh for why the old
# standalone `vllm` env / cuda-12.8 paths are gone.
CONDA_ENV="${CONDA_ENV:-vllm}"
source /data1/yhoon113/miniforge3/etc/profile.d/conda.sh
# See eval_steps_fanout.sh: the binutils activate hook trips `set -u`.
set +u
conda activate "$CONDA_ENV"
set -u
export CUDA_HOME=/data1/yhoon113/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/data1/yhoon113/miniforge3/envs/vllm/lib/libgomp.so.1
PY="${PY:-$CONDA_PREFIX/bin/python}"
BASE="${BASE:-$RQ/rq_output/rq_evolve_4b_8gpu}"

# Default: every global_step_N under $BASE, numerically ordered.
if [[ -n "${STEPS_LIST:-}" ]]; then
  IFS=',' read -ra STEPS <<< "$STEPS_LIST"
else
  mapfile -t STEPS < <(
    find "$BASE" -maxdepth 1 -name 'global_step_*' -type d -printf '%f\n' 2>/dev/null \
      | sed 's/^global_step_//' | sort -n
  )
fi
if [[ ${#STEPS[@]} -eq 0 ]]; then
  echo "no global_step_* checkpoints under $BASE" >&2; exit 1
fi

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

echo "[$(ts)] [plan] steps: ${STEPS[*]}"
echo "[$(ts)] [plan] benches: ${BENCHES[*]}   max_samples=$MAX_SAMPLES   gpus: ${GPUS[*]}"

# ---- 1. FSDP -> HF merge, reusing whatever the math run already produced ----
mpids=()
for s in "${STEPS[@]}"; do
  ckpt="$BASE/global_step_$s"; hf="$ckpt/hf_merged"
  mkdir -p "$ckpt/$OUTDIR_NAME/logs"
  if [[ -f "$hf/config.json" ]] && compgen -G "$hf/*.safetensors" >/dev/null; then
    echo "[$(ts)] [merge] step $s: reusing $hf"; continue
  fi
  echo "[$(ts)] [merge] step $s: $ckpt/actor -> $hf"
  ( "$PY" "$RQ/scripts/merge_fsdp_to_hf.py" --ckpt_dir "$ckpt/actor" --out_dir "$hf" \
      >"$ckpt/$OUTDIR_NAME/logs/merge.log" 2>&1 ) &
  mpids+=($!)
done
mfail=0
for p in "${mpids[@]:-}"; do [[ -n "$p" ]] && { wait "$p" || mfail=$((mfail+1)); }; done
if [[ $mfail -gt 0 ]]; then echo "[$(ts)] [merge] FAILED ($mfail)"; exit 1; fi

# ---- 2. job list ------------------------------------------------------------
JOBS=()
for s in "${STEPS[@]}"; do
  for b in "${BENCHES[@]}"; do
    # Skip a job whose summary already exists: this script is long enough that
    # resuming after an interrupt matters more than re-running for freshness.
    if [[ -f "$BASE/global_step_$s/$OUTDIR_NAME/$b/summary.json" ]] \
       && [[ "${FORCE:-0}" != "1" ]]; then
      echo "[$(ts)] [skip] step $s $b (summary exists; FORCE=1 to redo)"; continue
    fi
    JOBS+=("$s|$b")
  done
done
echo "[$(ts)] [plan] ${#JOBS[@]} jobs"
[[ ${#JOBS[@]} -eq 0 ]] && { echo "[$(ts)] nothing to do"; exit 0; }

run_job() {  # <gpu> <step> <bench>
  local gpu="$1" s="$2" b="$3"
  local out="$BASE/global_step_$s/$OUTDIR_NAME/$b"
  local hf="$BASE/global_step_$s/hf_merged"
  mkdir -p "$out" "$BASE/global_step_$s/$OUTDIR_NAME/logs"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$EVAL_SCRIPT" \
    --model "$hf" --tokenizer "$hf" \
    --benchmark "$b" --output_dir "$out" \
    --max_tokens "$MAXTOK" --max_model_len "$MAXMODELLEN" \
    --temperature 0.0 --top_p 1.0 \
    --tensor_parallel_size 1 --gpu_memory_utilization 0.85 \
    --dtype bfloat16 --enforce_eager --no_tqdm \
    --max_samples "$MAX_SAMPLES" --sample_seed "$SAMPLE_SEED" \
    >"$BASE/global_step_$s/$OUTDIR_NAME/logs/${b}.log" 2>&1
}

# ---- 3. worker pool ---------------------------------------------------------
declare -A PID_INFO
FREE=("${GPUS[@]}")
ok=0; fail=0
reap() {
  wait -n 2>/dev/null
  for pid in "${!PID_INFO[@]}"; do
    kill -0 "$pid" 2>/dev/null && continue
    IFS='|' read -r g st nm <<< "${PID_INFO[$pid]}"
    if wait "$pid"; then ok=$((ok+1)); echo "[$(ts)] [done] step $st $nm (gpu $g) OK"
    else fail=$((fail+1)); echo "[$(ts)] [done] step $st $nm (gpu $g) FAIL -> $OUTDIR_NAME/logs/${nm}.log"; fi
    unset "PID_INFO[$pid]"; FREE+=("$g")
    return
  done
}
for job in "${JOBS[@]}"; do
  IFS='|' read -r s b <<< "$job"
  while [[ ${#FREE[@]} -eq 0 ]]; do reap; done
  gpu="${FREE[0]}"; FREE=("${FREE[@]:1}")
  echo "[$(ts)] [launch] step $s $b -> gpu $gpu"
  run_job "$gpu" "$s" "$b" & PID_INFO[$!]="$gpu|$s|$b"
done
while [[ ${#PID_INFO[@]} -gt 0 ]]; do reap; done
echo "[$(ts)] [pool] done  ok=$ok fail=$fail"

# ---- 4. aggregate -----------------------------------------------------------
BASE="$BASE" OUTDIR_NAME="$OUTDIR_NAME" "$PY" "$RQ/scripts/collect_general_scores.py" \
  && echo "[$(ts)] wrote $BASE/scores_general.md" \
  || echo "[$(ts)] WARN: aggregation failed"
echo "[$(ts)] ALL DONE"
