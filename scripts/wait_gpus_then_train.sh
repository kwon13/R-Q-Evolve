#!/bin/bash
# Wait until every GPU is idle, then start the training run in tmux.
#
# Waits only. It never signals another process: the GPUs are shared, and a
# stray kill here would take down someone else's job. If the machine never
# frees up, this simply keeps waiting and reports that.
set -u

ROOT=/data1/yhoon113/R-Q-Evolve
cd "$ROOT" || exit 1
mkdir -p logs
WAITLOG="$ROOT/logs/gpu_wait_$(date +%Y%m%d_%H%M%S).log"

POLL_S=60          # how often to look
SETTLE_CHECKS=3    # consecutive idle readings required
SETTLE_GAP_S=45    # spacing between them

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WAITLOG"; }

busy_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null; }

log "waiting for all GPUs to go idle (poll ${POLL_S}s, need ${SETTLE_CHECKS} clean reads ${SETTLE_GAP_S}s apart)"
log "current holders:"
for p in $(busy_pids); do
  ps -o pid,user,etime,cmd --no-headers -p "$p" 2>/dev/null | cut -c1-120 | tee -a "$WAITLOG"
done

while true; do
  if [ -z "$(busy_pids)" ]; then
    # A single empty reading can be the gap between two of someone else's
    # jobs, so require several in a row before taking the machine.
    clean=1
    for i in $(seq 2 "$SETTLE_CHECKS"); do
      sleep "$SETTLE_GAP_S"
      if [ -n "$(busy_pids)" ]; then
        log "GPUs re-occupied during settle; back to waiting"
        clean=0
        break
      fi
    done
    [ "$clean" -eq 1 ] && break
  fi
  sleep "$POLL_S"
done

log "all GPUs idle -- starting training"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$WAITLOG"

tmux kill-session -t rqtrain 2>/dev/null
tmux new-session -d -s rqtrain -c "$ROOT" \
  "export PATH=/data1/yhoon113/miniforge3/envs/vllm/bin:\$PATH; \
   bash scripts/run_train_rq_evolve_base.sh; echo '[tmux] EXIT CODE '\$?; exec bash"

sleep 25
if tmux has-session -t rqtrain 2>/dev/null; then
  log "tmux session 'rqtrain' started"
  ls -t "$ROOT"/logs/rq_evolve_base_*.log 2>/dev/null | head -1 | tee -a "$WAITLOG"
else
  log "ERROR: tmux session did not start"
  exit 1
fi
