#!/usr/bin/env bash
# Evaluate a single HF model (or hub ID) on the 7 R-Zero math benchmarks.
set -uo pipefail

RQ=/data1/yhoon113/R-Q-Evolve
EVAL_SCRIPT="$RQ/scripts/eval_vllm_math.py"

# Conda & CUDA 환경 설정 (eval_steps_fanout.sh와 동일)
CONDA_ENV="${CONDA_ENV:-vllm}"
source /data1/yhoon113/miniforge3/etc/profile.d/conda.sh
set +u
conda activate "$CONDA_ENV"
set -u
export CUDA_HOME=/data1/yhoon113/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/data1/yhoon113/miniforge3/envs/vllm/lib/libgomp.so.1
PY="${PY:-$CONDA_PREFIX/bin/python}"

# 1. 모델 및 GPU 설정 (환경변수로 오버라이드 가능)
MODEL="${MODEL:-/data1/yhoon113/INFUSER-Qwen3-8B-base}"
IFS=',' read -ra GPUS <<< "${GPU_LIST:-0,1,2,3,4,5,6,7}"

# 출력 경로 (기존 collect_scores.py 호환을 위해 global_step_0 하위로 저장)
MODEL_NAME="$(basename "$MODEL")"
OUT_ROOT="${OUT_ROOT:-$RQ/rq_output/eval_${MODEL_NAME}}"
OUT_BASE="$OUT_ROOT/global_step_0"

MAXTOK="${MAXTOK:-4096}"
MAXMODELLEN="${MAXMODELLEN:-8192}"
GPT_RECHECK="${GPT_RECHECK:-1}"
BENCH_FILTER="${BENCH_LIST:-${BENCH_FILTER:-}}"

export PYTHONPATH="$RQ:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
cd "$RQ"

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

# 2. 작업 목록 구성
JOBS=()
for spec in "${SPECS[@]}"; do
  bname="${spec%%=*}"
  if [[ -n "$BENCH_FILTER" ]] && [[ ",$BENCH_FILTER," != *",$bname,"* ]]; then continue; fi
  
  # 이미 완료된 벤치마크는 skip (재실행하려면 FORCE=1)
  if [[ -f "$OUT_BASE/eval/$bname/summary.json" ]] && [[ "${FORCE:-0}" != "1" ]]; then
    echo "[$(ts)] [skip] $bname (summary exists; FORCE=1 to redo)"
    continue
  fi
  JOBS+=("$spec")
done

echo "[$(ts)] [plan] Model: $MODEL"
echo "[$(ts)] [plan] ${#JOBS[@]} jobs across GPUs: ${GPUS[*]}"
[[ ${#JOBS[@]} -eq 0 ]] && { echo "[$(ts)] nothing to do"; exit 0; }

run_job() {  # <gpu> <spec>
  local gpu="$1" spec="$2"
  local name="${spec%%=*}"
  local out="$OUT_BASE/eval/$name"
  mkdir -p "$out" "$OUT_BASE/eval/logs"
  
  local recheck_flag="--gpt_recheck"
  [[ "$GPT_RECHECK" == "0" ]] && recheck_flag="--no_gpt_recheck"
  
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$EVAL_SCRIPT" \
    --model "$MODEL" --tokenizer "$MODEL" --config "" \
    --output_dir "$out" --benchmark "$spec" \
    --max_tokens "$MAXTOK" --temperature 0.0 --top_p 1.0 --n 1 \
    --tensor_parallel_size 1 --gpu_memory_utilization 0.85 \
    --max_model_len "$MAXMODELLEN" --dtype bfloat16 \
    --inflate_x32 --enforce_eager \
    $recheck_flag \
    >"$OUT_BASE/eval/logs/${name}.log" 2>&1
}

# 3. GPU Worker Pool 실행
declare -A PID_INFO
FREE=("${GPUS[@]}")
ok=0; fail=0
reap() {
  wait -n 2>/dev/null
  for pid in "${!PID_INFO[@]}"; do
    kill -0 "$pid" 2>/dev/null && continue
    IFS='|' read -r g nm <<< "${PID_INFO[$pid]}"
    if wait "$pid"; then
      ok=$((ok+1)); echo "[$(ts)] [done] $nm (gpu $g) OK"
    else
      fail=$((fail+1)); echo "[$(ts)] [done] $nm (gpu $g) FAIL -> $OUT_BASE/eval/logs/${nm}.log"
    fi
    unset "PID_INFO[$pid]"; FREE+=("$g")
    return
  done
}

for spec in "${JOBS[@]}"; do
  while [[ ${#FREE[@]} -eq 0 ]]; do reap; done
  gpu="${FREE[0]}"; FREE=("${FREE[@]:1}")
  name="${spec%%=*}"
  echo "[$(ts)] [launch] $name -> gpu $gpu"
  run_job "$gpu" "$spec" & PID_INFO[$!]="$gpu|$name"
done
while [[ ${#PID_INFO[@]} -gt 0 ]]; do reap; done
echo "[$(ts)] [pool] done  ok=$ok fail=$fail"

# 4. 결과 출력 및 Markdown 정리
echo "=========================================================="
echo "$MODEL (max_tokens=$MAXTOK, greedy) pass@1"
OUT="$OUT_BASE/eval" "$PY" - <<'PY'
import json, os
out = os.environ["OUT"]
order = ["math500","gsm8k","amc23","aime24","aime25","minerva_math","olympiadbench"]
accs = []
for name in order:
    p = os.path.join(out, name, "summary.json")
    if not os.path.isfile(p):
        print(f"  {name:14s}  (no summary)")
        continue
    s = json.load(open(p))
    b = s.get("benchmarks", {}).get(name, {})
    a = b.get("pass_at_1", 0.0)
    n = b.get("num_examples", 0)
    accs.append(a)
    print(f"  {name:14s}  pass@1={a*100:6.2f}%   n={n}")
if accs:
    print(f"  {f'AVG({len(accs)})':14s}  {sum(accs)/len(accs)*100:6.2f}%")
PY
echo "=========================================================="

"$PY" "$RQ/scripts/collect_scores.py" "$OUT_ROOT" >/dev/null 2>&1 \
  && echo "[$(ts)] wrote $OUT_ROOT/scores.md" \
  || echo "[$(ts)] WARN: collect_scores.py failed"
echo "[$(ts)] ALL DONE"
