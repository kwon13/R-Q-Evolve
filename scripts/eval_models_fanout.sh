#!/usr/bin/env bash
# Evaluate plain HF model directories on the math and general-domain suites.
#
# The sibling fan-outs (eval_steps_fanout.sh, eval_general_fanout.sh) walk
# global_step_*/actor and merge FSDP shards first. These models are already HF
# checkpoints, so there is nothing to merge; only the job list differs.
#
#   MODELS=/path/a,/path/b bash scripts/eval_models_fanout.sh
#   MODELS=... SUITES=math GPU_LIST=0,1,2,3 bash scripts/eval_models_fanout.sh
#
# A model directory whose safetensors do not match its index is skipped rather
# than evaluated half-loaded -- one of these was still downloading when the run
# was requested.
set -uo pipefail

RQ=/data1/yhoon113/R-Q-Evolve
source /data1/yhoon113/miniforge3/etc/profile.d/conda.sh
conda activate vllm
export CUDA_HOME=/data1/yhoon113/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
PY="${PY:-/data1/yhoon113/miniforge3/envs/vllm/bin/python}"

IFS=',' read -ra MODEL_DIRS <<< "${MODELS:?set MODELS=/path/a,/path/b}"
IFS=',' read -ra GPUS <<< "${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra SUITES <<< "${SUITES:-math,general}"
OUT_ROOT="${OUT_ROOT:-$RQ/rq_output/model_bench}"

GPU_UTIL="${GPU_UTIL:-0.45}"
MATH_MAXTOK="${MATH_MAXTOK:-4096}"      # R-Zero parity
MATH_MAXLEN="${MATH_MAXLEN:-8192}"
GEN_MAXTOK="${GEN_MAXTOK:-8192}"
GEN_MAXLEN="${GEN_MAXLEN:-12000}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"      # general suite; 0 = full splits
GPT_RECHECK="${GPT_RECHECK:-1}"

export PYTHONPATH="$RQ:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
cd "$RQ"

declare -a MATH_SPECS=(
  "math500=test-time-compute/test_MATH:test"
  "gsm8k=openai/gsm8k:test"
  "amc23=test-time-compute/test_amc23:test"
  "aime24=test-time-compute/test_aime24:test"
  "aime25=test-time-compute/aime_2025:test"
  "minerva_math=test-time-compute/test_minerva_math:test"
  "olympiadbench=test-time-compute/test_olympiadbench:test"
)
GENERAL_BENCHES=(mmlupro supergpqa bbeh)

ts() { date +%H:%M:%S; }

# ---- 1. readiness: every shard the index names must be present --------------
READY=()
for m in "${MODEL_DIRS[@]}"; do
  name=$(basename "$m")
  if [[ ! -f "$m/config.json" ]]; then
    echo "[$(ts)] [skip] $name: no config.json"; continue
  fi
  if ! "$PY" - "$m" <<'CHECK'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
idx = p / "model.safetensors.index.json"
if idx.is_file():
    want = set(json.load(idx.open())["weight_map"].values())
    have = {f.name for f in p.glob("*.safetensors")}
    sys.exit(0 if want <= have else 1)
sys.exit(0 if any(p.glob("*.safetensors")) else 1)
CHECK
  then
    echo "[$(ts)] [skip] $name: shards incomplete (still downloading?)"; continue
  fi
  READY+=("$m")
done
[[ ${#READY[@]} -eq 0 ]] && { echo "[$(ts)] no ready models"; exit 1; }
echo "[$(ts)] [plan] models: $(for m in "${READY[@]}"; do basename "$m"; done | tr '\n' ' ')"

# ---- 2. job list: (model, suite, benchmark) ---------------------------------
JOBS=()
for m in "${READY[@]}"; do
  name=$(basename "$m")
  for suite in "${SUITES[@]}"; do
    if [[ "$suite" == "math" ]]; then
      for spec in "${MATH_SPECS[@]}"; do
        b="${spec%%=*}"
        [[ -f "$OUT_ROOT/$name/eval/$b/summary.json" && "${FORCE:-0}" != "1" ]] && continue
        JOBS+=("$m|math|$spec")
      done
    else
      for b in "${GENERAL_BENCHES[@]}"; do
        [[ -f "$OUT_ROOT/$name/eval_general/$b/summary.json" && "${FORCE:-0}" != "1" ]] && continue
        JOBS+=("$m|general|$b")
      done
    fi
  done
done
echo "[$(ts)] [plan] ${#JOBS[@]} jobs across GPUs: ${GPUS[*]}  (gpu_util=$GPU_UTIL)"
[[ ${#JOBS[@]} -eq 0 ]] && { echo "[$(ts)] nothing to do"; exit 0; }

run_job() {  # <gpu> <model> <suite> <spec-or-bench>
  local gpu="$1" model="$2" suite="$3" what="$4"
  local name; name=$(basename "$model")
  local b logdir
  if [[ "$suite" == "math" ]]; then
    b="${what%%=*}"; logdir="$OUT_ROOT/$name/eval/logs"
    mkdir -p "$OUT_ROOT/$name/eval/$b" "$logdir"
    local flag="--gpt_recheck"; [[ "$GPT_RECHECK" == "0" ]] && flag="--no_gpt_recheck"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$RQ/scripts/eval_vllm_math.py" \
      --model "$model" --tokenizer "$model" --config "" \
      --output_dir "$OUT_ROOT/$name/eval/$b" --benchmark "$what" \
      --max_tokens "$MATH_MAXTOK" --temperature 0.0 --top_p 1.0 --n 1 \
      --tensor_parallel_size 1 --gpu_memory_utilization "$GPU_UTIL" \
      --max_model_len "$MATH_MAXLEN" --dtype bfloat16 \
      --inflate_x32 --enforce_eager $flag \
      >"$logdir/${b}.log" 2>&1
  else
    b="$what"; logdir="$OUT_ROOT/$name/eval_general/logs"
    mkdir -p "$OUT_ROOT/$name/eval_general/$b" "$logdir"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$RQ/scripts/eval_general_vllm.py" \
      --model "$model" --tokenizer "$model" \
      --benchmark "$b" --output_dir "$OUT_ROOT/$name/eval_general/$b" \
      --max_tokens "$GEN_MAXTOK" --max_model_len "$GEN_MAXLEN" \
      --temperature 0.0 --top_p 1.0 \
      --tensor_parallel_size 1 --gpu_memory_utilization "$GPU_UTIL" \
      --dtype bfloat16 --enforce_eager --no_tqdm \
      --max_samples "$MAX_SAMPLES" \
      >"$logdir/${b}.log" 2>&1
  fi
}

# ---- 3. worker pool ---------------------------------------------------------
declare -A PID_INFO
FREE=("${GPUS[@]}")
ok=0; fail=0
reap() {
  wait -n 2>/dev/null
  for pid in "${!PID_INFO[@]}"; do
    kill -0 "$pid" 2>/dev/null && continue
    IFS='|' read -r g nm <<< "${PID_INFO[$pid]}"
    if wait "$pid"; then ok=$((ok+1)); echo "[$(ts)] [done] $nm (gpu $g) OK"
    else fail=$((fail+1)); echo "[$(ts)] [done] $nm (gpu $g) FAIL"; fi
    unset "PID_INFO[$pid]"
    # A finished vLLM does not hand its memory back instantly. Launching the
    # next job on the same card immediately cost four jobs to
    # "Free memory on device (24.53/79.15 GiB) ... less than desired GPU memory
    # utilization (0.45, 35.62 GiB)". Wait for the card to actually drain.
    for _ in $(seq 1 "${DRAIN_TRIES:-30}"); do
      used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
      total=$(nvidia-smi -i "$g" --query-gpu=memory.total --format=csv,noheader,nounits)
      need=$(awk -v t="$total" -v u="$GPU_UTIL" 'BEGIN{printf "%d", t*u}')
      [[ $((total - used)) -ge $need ]] && break
      sleep 5
    done
    FREE+=("$g")
    return
  done
}
for job in "${JOBS[@]}"; do
  IFS='|' read -r model suite what <<< "$job"
  while [[ ${#FREE[@]} -eq 0 ]]; do reap; done
  gpu="${FREE[0]}"; FREE=("${FREE[@]:1}")
  label="$(basename "$model")/$suite/${what%%=*}"
  echo "[$(ts)] [launch] $label -> gpu $gpu"
  run_job "$gpu" "$model" "$suite" "$what" & PID_INFO[$!]="$gpu|$label"
done
while [[ ${#PID_INFO[@]} -gt 0 ]]; do reap; done
echo "[$(ts)] [pool] done  ok=$ok fail=$fail"

OUT_ROOT="$OUT_ROOT" "$PY" "$RQ/scripts/collect_model_scores.py" || true
echo "[$(ts)] ALL DONE"
