#!/usr/bin/env bash
# End-to-end Evolved Performance Score evaluation.
#
#  1. Build/reuse a fixed benchmark, or validate a prebuilt benchmark.
#  2. Reuse or merge every saved VERL checkpoint to Hugging Face format.
#  3. Evaluate the base model (step 0) and every saved checkpoint in a GPU pool.
#  4. Plot EPS against global step with inner/outer evolution overlaid.
#
# Common overrides:
#   BASE=/path/to/run BASE_MODEL=/path/to/base-model \
#   GPU_LIST=0,1,2,3 bash scripts/run_evolved_performance.sh
#
#   STEPS_LIST=32,64,128 FORCE=1 GPU_LIST=0,1 \
#   bash scripts/run_evolved_performance.sh
set -uo pipefail

RQ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data1/yhoon113/miniforge3/envs/vllm/bin/python}"
BASE="${BASE:-$RQ/rq_output/rq_evolve_base_4b}"
CONFIG="${CONFIG:-$RQ/configs/rq_evolve_4b_base.yaml}"
BENCH_DIR="${BENCH_DIR:-$RQ/benchmarks/evolved_performance_seed_id_v1}"
RESULTS_DIR="${RESULTS_DIR:-$BASE/evolved_performance}"
SEED_DIR="${SEED_DIR:-$RQ/seed_programs}"
BENCHMARK_NAME="${BENCHMARK_NAME:-evolved_performance_seed_id_v1}"
EXAMPLES_PER_PROGRAM="${EXAMPLES_PER_PROGRAM:-40}"
SEED_START="${SEED_START:-1000000}"
MAXTOK="${MAXTOK:-4096}"
MAXMODELLEN="${MAXMODELLEN:-8192}"
FORCE="${FORCE:-0}"
INCLUDE_BASE="${INCLUDE_BASE:-1}"
IFS=',' read -ra GPUS <<< "${GPU_LIST:-0,1,2,3,4,5,6,7}"

if [[ ! -x "$PY" ]]; then
  echo "Python executable not found: $PY" >&2
  exit 1
fi
if [[ ! -d "$BASE" ]]; then
  echo "Run directory not found: $BASE" >&2
  exit 1
fi
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "GPU_LIST selected no GPUs" >&2
  exit 1
fi

if [[ -z "${BASE_MODEL:-}" ]] && [[ "$INCLUDE_BASE" == "1" ]]; then
  BASE_MODEL="$($PY - "$CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
value = OmegaConf.select(cfg, "verl_config.actor_rollout_ref.model.path")
print(value or "")
PY
)"
fi
if [[ "$INCLUDE_BASE" == "1" ]] && [[ -z "${BASE_MODEL:-}" ]]; then
  echo "Could not infer BASE_MODEL from $CONFIG; set BASE_MODEL=/path/to/model" >&2
  exit 1
fi

if [[ -n "${STEPS_LIST:-}" ]]; then
  IFS=',' read -ra STEPS <<< "$STEPS_LIST"
else
  mapfile -t STEPS < <(
    find "$BASE" -maxdepth 1 -name 'global_step_*' -type d -printf '%f\n' 2>/dev/null \
      | sed 's/^global_step_//' | sort -n
  )
fi
if [[ ${#STEPS[@]} -eq 0 ]]; then
  echo "No global_step_* checkpoints under $BASE" >&2
  exit 1
fi

export PYTHONPATH="$RQ/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
if [[ -d /data1/yhoon113/cuda-12.8 ]]; then
  export CUDA_HOME="${CUDA_HOME:-/data1/yhoon113/cuda-12.8}"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

ts() { date +%H:%M:%S; }

mkdir -p "$RESULTS_DIR/logs"

BENCH="$BENCH_DIR/benchmark.jsonl"
MANIFEST="$BENCH_DIR/manifest.json"
if [[ "${PREBUILT_BENCHMARK:-0}" == "1" ]]; then
  echo "[$(ts)] [benchmark] validating prebuilt benchmark at $BENCH_DIR"
  "$PY" - "$RQ" "$BENCH" "$MANIFEST" "$BENCHMARK_NAME" <<'PY'
import sys
from pathlib import Path

root, benchmark, manifest, expected_name = sys.argv[1:]
sys.path.insert(0, str(Path(root) / "src"))
from rq_evolve.evolved_performance import load_benchmark

rows, payload = load_benchmark(benchmark, manifest)
if payload.get("benchmark") != expected_name:
    raise ValueError(
        f"prebuilt benchmark is {payload.get('benchmark')!r}, "
        f"not {expected_name!r}"
    )
print(
    f"[EPB] validated {len(rows)} examples; "
    f"sha256={payload['benchmark_sha256']}"
)
PY
else
  echo "[$(ts)] [benchmark] preparing fixed generator benchmark"
  BENCH_ARGS=(
    "$RQ/scripts/build_evolved_performance_bench.py"
    --seed-dir "$SEED_DIR"
    --output-dir "$BENCH_DIR"
    --benchmark-name "$BENCHMARK_NAME"
    --examples-per-program "$EXAMPLES_PER_PROGRAM"
    --seed-start "$SEED_START"
  )
  if [[ "${OVERLAP_AUDIT:-1}" == "1" ]] && \
     [[ -f "$BASE/rq_archive/rq_used_seeds.json" ]]; then
    BENCH_ARGS+=(
      --used-seeds-json "$BASE/rq_archive/rq_used_seeds.json"
      --overlap-audit-output "$RESULTS_DIR/overlap_audit.json"
    )
  fi
  if [[ "${FORCE_BENCH:-0}" == "1" ]]; then
    BENCH_ARGS+=(--force)
  fi
  "$PY" "${BENCH_ARGS[@]}"
fi

BENCHMARK_HASH="$($PY - "$MANIFEST" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["benchmark_sha256"])
PY
)"

echo "[$(ts)] [merge] checking ${#STEPS[@]} checkpoints"
merge_pids=()
for step in "${STEPS[@]}"; do
  checkpoint="$BASE/global_step_$step"
  hf="$checkpoint/hf_merged"
  if [[ -f "$hf/config.json" ]] && compgen -G "$hf/*.safetensors" >/dev/null; then
    echo "[$(ts)] [merge] step $step: reusing $hf"
    continue
  fi
  actor="$checkpoint/actor"
  if [[ ! -d "$actor" ]]; then
    echo "[$(ts)] [merge] step $step: neither complete $hf nor $actor exists" >&2
    exit 1
  fi
  log="$RESULTS_DIR/logs/merge_step_${step}.log"
  echo "[$(ts)] [merge] step $step: $actor -> $hf"
  (
    "$PY" "$RQ/scripts/merge_fsdp_to_hf.py" \
      --ckpt_dir "$actor" --out_dir "$hf" >"$log" 2>&1
  ) &
  merge_pids+=("$!")
done
merge_fail=0
for pid in "${merge_pids[@]}"; do
  wait "$pid" || merge_fail=$((merge_fail + 1))
done
if [[ $merge_fail -gt 0 ]]; then
  echo "[$(ts)] [merge] $merge_fail merge(s) failed; see $RESULTS_DIR/logs" >&2
  exit 1
fi

JOBS=()
if [[ "$INCLUDE_BASE" == "1" ]]; then
  JOBS+=("0|$BASE_MODEL")
fi
for step in "${STEPS[@]}"; do
  JOBS+=("$step|$BASE/global_step_$step/hf_merged")
done

run_job() {
  local gpu="$1" step="$2" model="$3"
  local out="$RESULTS_DIR/global_step_$step"
  local log="$RESULTS_DIR/logs/global_step_${step}.log"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" \
    "$RQ/scripts/eval_evolved_performance_vllm.py" \
    --model "$model" --tokenizer "$model" \
    --benchmark "$BENCH" --manifest "$MANIFEST" \
    --output-dir "$out" --global-step "$step" \
    --max-tokens "$MAXTOK" --max-model-len "$MAXMODELLEN" \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
    --dtype bfloat16 --trust-remote-code --enforce-eager \
    --vllm-sampler-backend pytorch \
    >"$log" 2>&1
}

declare -A PID_INFO
FREE=("${GPUS[@]}")
ok=0
fail=0

reap_one() {
  wait -n 2>/dev/null || true
  local pid gpu step
  for pid in "${!PID_INFO[@]}"; do
    kill -0 "$pid" 2>/dev/null && continue
    IFS='|' read -r gpu step <<< "${PID_INFO[$pid]}"
    if wait "$pid"; then
      ok=$((ok + 1))
      echo "[$(ts)] [done] global_step_$step on GPU $gpu"
    else
      fail=$((fail + 1))
      echo "[$(ts)] [fail] global_step_$step; see $RESULTS_DIR/logs/global_step_${step}.log" >&2
    fi
    unset "PID_INFO[$pid]"
    FREE+=("$gpu")
    return
  done
}

echo "[$(ts)] [eval] ${#JOBS[@]} model(s) across GPUs: ${GPUS[*]}"
for job in "${JOBS[@]}"; do
  IFS='|' read -r step model <<< "$job"
  summary="$RESULTS_DIR/global_step_$step/summary.json"
  if [[ "$FORCE" != "1" ]] && [[ -f "$summary" ]]; then
    summary_hash="$($PY - "$summary" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8")).get("benchmark_sha256", ""))
except Exception:
    print("")
PY
)"
    if [[ "$summary_hash" == "$BENCHMARK_HASH" ]]; then
      echo "[$(ts)] [skip] global_step_$step summary matches benchmark (FORCE=1 to redo)"
      continue
    fi
    echo "[$(ts)] [redo] global_step_$step summary has a stale benchmark hash"
  fi
  while [[ ${#FREE[@]} -eq 0 ]]; do reap_one; done
  gpu="${FREE[0]}"
  FREE=("${FREE[@]:1}")
  echo "[$(ts)] [launch] global_step_$step -> GPU $gpu"
  run_job "$gpu" "$step" "$model" &
  PID_INFO[$!]="$gpu|$step"
done
while [[ ${#PID_INFO[@]} -gt 0 ]]; do reap_one; done

if [[ $fail -gt 0 ]]; then
  echo "[$(ts)] [eval] completed with $fail failure(s), $ok success(es)" >&2
  exit 1
fi

echo "[$(ts)] [plot] combining checkpoint scores with inner/outer evolution"
PLOT_ARGS=(
  "$RQ/scripts/plot_evolved_performance.py"
  --run-dir "$BASE"
  --results-dir "$RESULTS_DIR"
)
if [[ -n "${PLOT_TITLE:-}" ]]; then
  PLOT_ARGS+=(--title "$PLOT_TITLE")
fi
"$PY" "${PLOT_ARGS[@]}"
echo "[$(ts)] [done] $RESULTS_DIR/evolved_performance.png"
