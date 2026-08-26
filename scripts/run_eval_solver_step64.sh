#!/usr/bin/env bash
# ==============================================================================
# Evaluate /data1/yhoon113/RQ-Evolve-4B-Solver_step64 on 7 Math Benchmarks
# using GPUs 1, 3, 4, 6 via parallel worker pool (vLLM, TP=1 per GPU).
# ==============================================================================
set -euo pipefail

REPO="/data1/yhoon113/R-Q-Evolve"
MODEL_PATH="${MODEL_PATH:-/data1/yhoon113/RQ-Evolve-4B-Solver_step64}"
OUTPUT_DIR="${OUTPUT_DIR:-$MODEL_PATH/eval}"
LOGS_DIR="$OUTPUT_DIR/logs"

# GPU allocation: 1, 3, 4, 6
GPU_LIST="${GPU_LIST:-1,3,4,6}"
IFS=',' read -ra GPUS <<< "$GPU_LIST"

CONDA_ENV="${CONDA_ENV:-vllm}"
source /data1/yhoon113/miniforge3/etc/profile.d/conda.sh
set +u
conda activate "$CONDA_ENV"
set -u

export CUDA_HOME=/data1/yhoon113/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/data1/yhoon113/miniforge3/envs/vllm/lib/libgomp.so.1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

PY="$CONDA_PREFIX/bin/python"
EVAL_SCRIPT="$REPO/scripts/eval_vllm_math.py"

# Hyperparameters (R-Zero parity)
MAXTOK="${MAXTOK:-4096}"
MAXMODELLEN="${MAXMODELLEN:-8192}"
GPT_RECHECK="${GPT_RECHECK:-1}"
GPU_MEM="${GPU_MEM:-0.85}"

# Benchmarks to evaluate
declare -a SPECS=(
  "math500=test-time-compute/test_MATH:test"
  "gsm8k=openai/gsm8k:test"
  "amc23=test-time-compute/test_amc23:test"
  "aime24=test-time-compute/test_aime24:test"
  "aime25=test-time-compute/aime_2025:test"
  "minerva_math=test-time-compute/test_minerva_math:test"
  "olympiadbench=test-time-compute/test_olympiadbench:test"
)

ts() { date +'%Y-%m-%d %H:%M:%S'; }

echo "=============================================================================="
echo "[$(ts)] Target Model : $MODEL_PATH"
echo "[$(ts)] Output Dir   : $OUTPUT_DIR"
echo "[$(ts)] Assigned GPUs: ${GPUS[*]}"
echo "=============================================================================="

# 1. Check if model files are ready; wait if requested or not yet ready
if [[ "${WAIT_FOR_DOWNLOAD:-0}" == "1" ]] || [[ ! -f "$MODEL_PATH/config.json" ]] || [[ ! -f "$MODEL_PATH/model.safetensors" ]]; then
  echo "[$(ts)] Checking/waiting for model download completion in $MODEL_PATH..."
  while true; do
    if [[ -f "$MODEL_PATH/config.json" ]] && [[ -f "$MODEL_PATH/model.safetensors" ]]; then
      # Check if model.safetensors file size is stable (not actively being written)
      size1=$(stat -c%s "$MODEL_PATH/model.safetensors" 2>/dev/null || echo 0)
      sleep 5
      size2=$(stat -c%s "$MODEL_PATH/model.safetensors" 2>/dev/null || echo 0)
      if [[ "$size1" -eq "$size2" ]] && [[ "$size1" -gt 1000000 ]]; then
        echo "[$(ts)] Model download confirmed ready (size: $size2 bytes)."
        break
      fi
    fi
    echo "[$(ts)] Waiting for model download to complete... (sleep 15s)"
    sleep 15
  done
fi

mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"
cd "$REPO"

# 2. Worker pool execution
declare -A PID_INFO # pid -> "gpu|name"
FREE=("${GPUS[@]}")
ok=0
fail=0

reap() {
  wait -n 2>/dev/null
  for pid in "${!PID_INFO[@]}"; do
    kill -0 "$pid" 2>/dev/null && continue
    IFS='|' read -r g nm <<< "${PID_INFO[$pid]}"
    if wait "$pid"; then
      ok=$((ok + 1))
      echo "[$(ts)] [DONE] $nm (GPU $g) - SUCCESS"
    else
      fail=$((fail + 1))
      echo "[$(ts)] [DONE] $nm (GPU $g) - FAILED -> check $LOGS_DIR/${nm}.log"
    fi
    unset "PID_INFO[$pid]"
    FREE+=("$g")
    return
  done
}

run_job() {
  local gpu="$1" spec="$2"
  local name="${spec%%=*}"
  local out="$OUTPUT_DIR/$name"
  local log="$LOGS_DIR/${name}.log"
  mkdir -p "$out"

  local recheck_flag="--gpt_recheck"
  [[ "$GPT_RECHECK" == "0" ]] && recheck_flag="--no_gpt_recheck"

  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$EVAL_SCRIPT" \
    --model "$MODEL_PATH" \
    --tokenizer "$MODEL_PATH" \
    --output_dir "$out" \
    --benchmark "$spec" \
    --max_tokens "$MAXTOK" \
    --temperature 0.0 \
    --top_p 1.0 \
    --n 1 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization "$GPU_MEM" \
    --max_model_len "$MAXMODELLEN" \
    --dtype bfloat16 \
    --inflate_x32 \
    --enforce_eager \
    $recheck_flag \
    >"$log" 2>&1
}

echo "[$(ts)] Launching ${#SPECS[@]} benchmark jobs across GPUs: ${GPUS[*]}"

for spec in "${SPECS[@]}"; do
  name="${spec%%=*}"
  while [[ ${#FREE[@]} -eq 0 ]]; do
    reap
  done
  gpu="${FREE[0]}"
  FREE=("${FREE[@]:1}")
  echo "[$(ts)] [LAUNCH] $name -> GPU $gpu"
  run_job "$gpu" "$spec" &
  PID_INFO[$!]="$gpu|$name"
done

while [[ ${#PID_INFO[@]} -gt 0 ]]; do
  reap
done

echo "=============================================================================="
echo "[$(ts)] Evaluation finished! (Success: $ok, Failed: $fail)"
echo "=============================================================================="

# 3. Aggregate results into summary table
OUT="$OUTPUT_DIR" "$PY" - <<'PY'
import json
import os

out = os.environ["OUT"]
benchmarks = ["math500", "gsm8k", "amc23", "aime24", "aime25", "minerva_math", "olympiadbench"]

print("\n" + "=" * 65)
print(f"{'Benchmark':<18} | {'Pass@1 (%)':<12} | {'Pre-GPT (%)':<12} | {'Flips':<6} | {'N':<6}")
print("-" * 65)

accs = []
for name in benchmarks:
    summary_file = os.path.join(out, name, "summary.json")
    if not os.path.isfile(summary_file):
        print(f"{name:<18} | {'(no summary)':<12} | {'-':<12} | {'-':<6} | {'-':<6}")
        continue
    try:
        with open(summary_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        b = data.get("benchmarks", {}).get(name, {})
        p1 = b.get("pass_at_1", 0.0) * 100
        pre = b.get("pass_at_1_pre_gpt", 0.0) * 100
        flips = b.get("gpt_flips", 0)
        n = b.get("num_examples", 0)
        accs.append(p1)
        print(f"{name:<18} | {p1:>10.2f}% | {pre:>10.2f}% | {flips:>6} | {n:>6}")
    except Exception as e:
        print(f"{name:<18} | Error loading summary: {e}")

print("-" * 65)
if accs:
    avg = sum(accs) / len(accs)
    print(f"{f'AVG ({len(accs)} benches)':<18} | {avg:>10.2f}% |")
print("=" * 65 + "\n")
PY
