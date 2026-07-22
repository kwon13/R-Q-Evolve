#!/usr/bin/env bash
# Evaluate R-Q-Evolve checkpoints on the 7 R-Zero math benchmarks (incl. gsm8k).
#
# Pipeline per step:  FSDP shards (actor/) --merge--> hf_merged/ --eval--> eval/
# Parallelism: (step x benchmark) = 12 jobs fanned out across GPUS as a worker
# pool (one vLLM instance per GPU, tp=1). Modeled on evo-sample's
# run_eval_all_steps_parallel.sh + R-Q-Evolve's eval_step32_fanout.sh.
#
# Settings: greedy (temp=0.0, n=1), max_tokens=4096, max_model_len=12000,
# AMC/AIME ×32-inflated. Grading + GPT-4o re-check are R-Zero-aligned (see
# EVAL_SCRIPT below) — all knobs fixed to match R-Zero/evaluation.
set -uo pipefail

REPO=/data1/yhoon113/R-Q-Evolve          # merge_fsdp_to_hf.py lives here
RQ=/data1/yhoon113/R-Q-Evolve
EVAL_SCRIPT="$RQ/scripts/eval_vllm_math.py"   # R-Zero-aligned eval (math_eval loaders)
# Blackwell sm_120: use the vllm cu128 env (cuda-12.8) — azr-bw is cu126 -> "no kernel image"
source /data1/yhoon113/miniforge3/etc/profile.d/conda.sh
conda activate vllm
export CUDA_HOME=/data1/yhoon113/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
PY="${PY:-/data1/yhoon113/miniforge3/envs/vllm/bin/python}"
BASE="${BASE:-/data1/yhoon113/R-Q-Evolve/rq_output/rq_evolve_4b_nr5}"
# Steps to evaluate. Override with STEPS_LIST=32,64 (comma-separated).
IFS=',' read -ra STEPS <<< "${STEPS_LIST:-32,64,96,128}"
# IFS=',' read -ra STEPS <<< "${STEPS_LIST:-160,192}"
IFS=',' read -ra GPUS <<< "${GPU_LIST:-0,1,2,3,4,5,6,7}"
# R-Zero/evaluation/generate.py uses max_tokens=4096; match it for parity.
MAXTOK=4096
MAXMODELLEN=8192
# GPT-4o re-check (R-Zero results_recheck.py port). Default ON for R-Zero parity;
# set GPT_RECHECK=0 to score with math_verify only. Reads OPENAI_API_KEY from
# $RQ/.env (loaded by the eval script; values never printed).
GPT_RECHECK="${GPT_RECHECK:-1}"
# Optional: restrict to a subset of benchmarks (comma-separated names), e.g.
# BENCH_LIST=gsm8k to re-run only gsm8k. Empty = all SPECS below.
BENCH_FILTER="${BENCH_LIST:-}"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
cd "$REPO"

# name=hf_id:split
# R-Zero/evaluation/evaluate.bash evaluates 7 tasks (incl. gsm8k). hf_id here is
# cosmetic — the eval script maps NAME -> source via rq_evolve.math_eval.
declare -a SPECS=(
  "math500=test-time-compute/test_MATH:test"
  "gsm8k=openai/gsm8k:test"
  "amc23=test-time-compute/test_amc23:test"
  "aime24=test-time-compute/test_aime24:test"
  "aime25=test-time-compute/aime_2025:test"
  "minerva_math=test-time-compute/test_minerva_math:test"
  "olympiadbench=test-time-compute/test_olympiadbench:test"
)

ts() { date +%H:%M:%S; }

# ---- 1. FSDP -> HF merge (CPU) for each step, in parallel --------------------
echo "[$(ts)] [merge] starting FSDP->HF merges (CPU)"
mpids=()
for s in "${STEPS[@]}"; do
  ckpt="$BASE/global_step_$s"
  hf="$ckpt/hf_merged"
  mlog="$ckpt/eval/logs/merge.log"
  mkdir -p "$ckpt/eval/logs"
  if [[ -f "$hf/config.json" ]] && compgen -G "$hf/*.safetensors" >/dev/null; then
    echo "[$(ts)] [merge] step $s: reusing existing $hf"
    continue
  fi
  echo "[$(ts)] [merge] step $s: $ckpt/actor -> $hf  (log: $mlog)"
  ( "$PY" "$REPO/scripts/merge_fsdp_to_hf.py" \
        --ckpt_dir "$ckpt/actor" --out_dir "$hf" >"$mlog" 2>&1 ) &
  mpids+=($!)
done
mfail=0
for p in "${mpids[@]}"; do wait "$p" || mfail=$((mfail+1)); done
if [[ $mfail -gt 0 ]]; then
  echo "[$(ts)] [merge] FAILED ($mfail). Check eval/logs/merge.log"; exit 1
fi
echo "[$(ts)] [merge] all merges done"

# ---- 2. build job list: (step, spec) ---------------------------------------
JOBS=()
for s in "${STEPS[@]}"; do
  for spec in "${SPECS[@]}"; do
    bname="${spec%%=*}"
    if [[ -n "$BENCH_FILTER" ]] && [[ ",$BENCH_FILTER," != *",$bname,"* ]]; then continue; fi
    JOBS+=("$s|$spec")
  done
done
echo "[$(ts)] [plan] ${#JOBS[@]} jobs across GPUs: ${GPUS[*]}${BENCH_FILTER:+  (benches: $BENCH_FILTER)}"

run_job() {  # <gpu> <step> <spec>
  local gpu="$1" s="$2" spec="$3"
  local name="${spec%%=*}"
  local out="$BASE/global_step_$s/eval/$name"
  local hf="$BASE/global_step_$s/hf_merged"
  mkdir -p "$out" "$BASE/global_step_$s/eval/logs"
  local recheck_flag="--gpt_recheck"
  [[ "$GPT_RECHECK" == "0" ]] && recheck_flag="--no_gpt_recheck"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$EVAL_SCRIPT" \
    --model "$hf" --tokenizer "$hf" --config "" \
    --output_dir "$out" --benchmark "$spec" \
    --max_tokens "$MAXTOK" --temperature 0.0 --top_p 1.0 --n 1 \
    --tensor_parallel_size 1 --gpu_memory_utilization 0.85 \
    --max_model_len "$MAXMODELLEN" --dtype bfloat16 \
    --inflate_x32 --enforce_eager \
    $recheck_flag \
    >"$BASE/global_step_$s/eval/logs/${name}.log" 2>&1
}

# ---- 3. worker pool --------------------------------------------------------
declare -A PID_INFO   # pid -> "gpu|step|name"
FREE=("${GPUS[@]}")
ok=0; fail=0
reap() {
  wait -n 2>/dev/null
  for pid in "${!PID_INFO[@]}"; do
    kill -0 "$pid" 2>/dev/null && continue
    IFS='|' read -r g st nm <<< "${PID_INFO[$pid]}"
    if wait "$pid"; then ok=$((ok+1)); echo "[$(ts)] [done] step $st $nm (gpu $g) OK";
    else fail=$((fail+1)); echo "[$(ts)] [done] step $st $nm (gpu $g) FAIL -> eval/logs/${nm}.log"; fi
    unset "PID_INFO[$pid]"; FREE+=("$g")
    return
  done
}
for job in "${JOBS[@]}"; do
  IFS='|' read -r s spec <<< "$job"
  while [[ ${#FREE[@]} -eq 0 ]]; do reap; done
  gpu="${FREE[0]}"; FREE=("${FREE[@]:1}")
  name="${spec%%=*}"
  echo "[$(ts)] [launch] step $s $name -> gpu $gpu"
  run_job "$gpu" "$s" "$spec" & PID_INFO[$!]="$gpu|$s|$name"
done
while [[ ${#PID_INFO[@]} -gt 0 ]]; do reap; done
echo "[$(ts)] [pool] done  ok=$ok fail=$fail"

# ---- 4. aggregate per step -------------------------------------------------
for s in "${STEPS[@]}"; do
  echo "=========================================================="
  echo "global_step_$s  (max_tokens=$MAXTOK, greedy)  pass@1"
  OUT="$BASE/global_step_$s/eval" "$PY" - <<'PY'
import json, os
out=os.environ["OUT"]
order=["math500","gsm8k","amc23","aime24","aime25","minerva_math","olympiadbench"]
accs=[]
for name in order:
    p=os.path.join(out,name,"summary.json")
    if not os.path.isfile(p): print(f"  {name:14s}  (no summary)"); continue
    s=json.load(open(p)); b=s.get("benchmarks",{}).get(name,{})
    a=b.get("pass_at_1",0.0); n=b.get("num_examples",0)
    accs.append(a); print(f"  {name:14s}  pass@1={a*100:6.2f}%   n={n}")
if accs: print(f"  {f'AVG({len(accs)})':14s}  {sum(accs)/len(accs)*100:6.2f}%")
PY
done
echo "=========================================================="
# Write the per-step x benchmark markdown table to $BASE/scores.md.
"$PY" "$RQ/analysis/collect_scores.py" "$BASE" >/dev/null 2>&1 \
  && echo "[$(ts)] wrote $BASE/scores.md" \
  || echo "[$(ts)] WARN: collect_scores.py failed"
echo "[$(ts)] ALL DONE"
