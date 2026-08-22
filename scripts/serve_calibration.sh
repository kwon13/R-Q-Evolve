#!/usr/bin/env bash
# Stand up 4B and 8B as long-lived vLLM servers for seed-difficulty calibration.
#
#   bash scripts/serve_calibration.sh          # start both, wait until ready
#   bash scripts/serve_calibration.sh stop     # shut both down
#
# The point is the EDIT LOOP: probe_seed_solvability.py reloads the model on
# every invocation (2-3 minutes), which makes tuning a seed's magnitudes by
# measurement impractical. With the servers resident, scripts/calibrate_seeds.py
# re-measures in seconds, so a seed can be edited and re-scored as fast as it
# can be reasoned about.
#
# One GPU each, tensor-parallel 1. Both models fit with room to spare (4B ~8 GiB
# of weights, 8B ~16 GiB, on 97 GiB cards) and TP=1 avoids the illegal-memory
# access this box hits on sm_120 with TP=2. GPUs 2-7 stay free.
set -uo pipefail

ROOT=/data1/yhoon113/R-Q-Evolve
LOGDIR=$ROOT/rq_output/calibration_logs
cd "$ROOT"; mkdir -p "$LOGDIR"

PORT_4B=${PORT_4B:-8401}
PORT_8B=${PORT_8B:-8801}

if [[ "${1:-start}" == "stop" ]]; then
  for f in "$LOGDIR"/*.pid; do
    [ -f "$f" ] || continue
    pid=$(cat "$f"); echo "[calib] stopping $(basename "$f" .pid) pid=$pid"
    kill "$pid" 2>/dev/null; rm -f "$f"
  done
  sleep 5; pkill -f 'vllm serve' 2>/dev/null
  sleep 3; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  exit 0
fi

set +u
source /data1/yhoon113/miniforge3/etc/profile.d/conda.sh
conda activate azr-bw-blackwell || { echo "[calib] conda activate failed" >&2; exit 1; }
set -u

serve() {  # name, model, gpu, port
  local name=$1 model=$2 gpu=$3 port=$4
  echo "[calib] $name on GPU $gpu, port $port"
  # --enforce-eager: torch.compile fails on this sm_120 build.
  CUDA_VISIBLE_DEVICES=$gpu nohup vllm serve "$model" \
    --served-model-name "$name" \
    --port "$port" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.60 \
    --max-model-len 12000 \
    --dtype bfloat16 \
    --enforce-eager \
    --trust-remote-code \
    > "$LOGDIR/$name.log" 2>&1 &
  echo "$!" > "$LOGDIR/$name.pid"
}

serve qwen3-4b-base /data1/yhoon113/qwen3-4b-base 0 "$PORT_4B"
serve qwen3-8b-base /data1/yhoon113/qwen3-8b-base 1 "$PORT_8B"

# Wait for both to answer, and fail loudly rather than leaving a half-up pair.
for spec in "qwen3-4b-base:$PORT_4B" "qwen3-8b-base:$PORT_8B"; do
  name=${spec%%:*}; port=${spec##*:}
  printf "[calib] waiting for %s " "$name"
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      echo "ready"; break
    fi
    if ! kill -0 "$(cat "$LOGDIR/$name.pid")" 2>/dev/null; then
      echo "DIED -- last log lines:"; tail -20 "$LOGDIR/$name.log"; exit 1
    fi
    printf "."; sleep 5
  done
  curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 || {
    echo " TIMED OUT"; tail -20 "$LOGDIR/$name.log"; exit 1; }
done

echo "[calib] both up.  4B :$PORT_4B   8B :$PORT_8B   logs in $LOGDIR"
echo "[calib] measure with: python scripts/calibrate_seeds.py"
