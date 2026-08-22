#!/usr/bin/env bash
# Emit a line only when something worth reading happens in the U ablation:
# a cell opens, a candidate is actually inserted, or a run stops.
# Silence means "still grinding at the same coverage".
set -uo pipefail
ROOT=/data1/yhoon113/R-Q-Evolve
LOGDIR=$ROOT/rq_output/u_ablation_logs
declare -A last_cov last_ins
for a in withU noU; do last_cov[$a]=""; last_ins[$a]=""; done

while true; do
  alive=0
  for a in withU noU; do
    pid=$(cat "$LOGDIR/$a.pid" 2>/dev/null || echo 0)
    if kill -0 "$pid" 2>/dev/null; then alive=1; else
      if [ "${last_cov[$a]}" != "DEAD" ]; then
        echo "[$a] STOPPED (pid $pid gone) -- $(grep -ac Traceback "$LOGDIR/$a.log" 2>/dev/null) traceback(s) in log"
        last_cov[$a]=DEAD
      fi
      continue
    fi
    line=$(grep -a "outer iteration" "$LOGDIR/$a.log" 2>/dev/null | tail -1)
    [ -z "$line" ] && continue
    it=$(sed -n "s/.*'outer_iteration': \([0-9]*\).*/\1/p" <<<"$line")
    cov=$(sed -n "s/.*'coverage': \([0-9.]*\).*/\1/p" <<<"$line")
    ins=$(sed -n "s/.*'status_inserted': \([0-9]*\).*/\1/p" <<<"$line")
    ins=${ins:-0}
    if [ -n "$cov" ] && [ "$cov" != "${last_cov[$a]}" ] && [ "${last_cov[$a]}" != "" ]; then
      echo "[$a] iter $it: coverage ${last_cov[$a]} -> $cov  (NEW CELL)"
    fi
    # Same-cell replacements are the common case and say little on their own;
    # only a cell that OPENS changes what the coverage comparison reads.
    last_cov[$a]=$cov; last_ins[$a]=$ins
  done
  [ "$alive" -eq 0 ] && { echo "both runs stopped; watcher exiting"; break; }
  sleep 120
done
