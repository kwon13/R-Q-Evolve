#!/bin/bash
# Drives the signed-covariance experiment end to end on one GPU:
#   phase 1 (already running) -> entropy -> analysis, for the CURRENT archive (27),
#   then the same three phases for the UNION of every champion ever archived (182).
set -u
ROOT=/data1/yhoon113/R-Q-Evolve
PY=/data1/yhoon113/miniforge3/envs/azr-bw-blackwell/bin/python
CUR=$ROOT/analysis/rq_evolve_base_8b/cov_sign
UNI=$ROOT/analysis/rq_evolve_base_8b/cov_sign_union
cd "$ROOT"

echo "[pipe] waiting for phase-1 generation to finish"
until [ -f "$CUR/meta_generate.json" ]; do
  pgrep -f "cov_sign_generate.py --g 32$" >/dev/null || { sleep 5; [ -f "$CUR/meta_generate.json" ] || { echo "[pipe] FATAL: generator gone without meta"; exit 1; }; }
  sleep 20
done

echo "[pipe] === current archive: entropy ==="
$PY scripts/cov_sign_entropy.py --out-dir "$CUR" || exit 1
echo "[pipe] === current archive: analysis ==="
$PY scripts/cov_sign_analyze.py --out-dir "$CUR" --label "current archive, 27 champions" || exit 1

echo "[pipe] === union population: generation ==="
$PY scripts/cov_sign_generate.py --g 32 --out-dir "$UNI" \
  --archive-glob "$ROOT/rq_output/rq_evolve_base_8b/rq_archive/archive_iter*.json" || exit 1
echo "[pipe] === union population: entropy ==="
$PY scripts/cov_sign_entropy.py --out-dir "$UNI" || exit 1
echo "[pipe] === union population: analysis ==="
$PY scripts/cov_sign_analyze.py --out-dir "$UNI" --label "all champions ever archived" || exit 1
echo "[pipe] ALL DONE"
