#!/bin/bash
# QUICKSTART — minimal recipe to extend or audit the r93 archive.
# Source this in a clean shell to get all the environment + path helpers.

# --- Paths -------------------------------------------------------------------
export ACT_ROOT=/data1/Kane/ACT
export ACT_PY=/data1/Kane/miniconda3/envs/act-py312/bin/python
export VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
export ARCHIVE_BASE=$ACT_ROOT/audit_results/r93_rerun_20260525T083118Z
export CONSOLIDATED=$ARCHIVE_BASE/CONSOLIDATED_RESULTS
export OFFICIAL_ZERO_TOL=/data1/Kane/HyZor/arXiv-2512.19007v1/generated/2025/zero_tol/longtable.tex
export OFFICIAL_SMALL_TOL=/data1/Kane/HyZor/arXiv-2512.19007v1/generated/2025/small_tol/longtable.tex

# --- Strict-watchdog env (every paper-grade run) ------------------------------
export PYTHONPATH=$ACT_ROOT
export ACT_VNNLIB_ROOT=$VNNLIB_ROOT
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# --- Helper: run one benchmark with strict watchdog ---------------------------
# Usage: act_run <bench> "<iids>" <wall_s> <rss_gb> [cpu|cuda] [out_label]
act_run() {
  local bench=$1 ids=$2 wall=$3 rss=$4 device=${5:-cpu} label=${6:-run}
  local out="$ARCHIVE_BASE/${label}_${bench}_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$out"
  $ACT_PY -m act.pipeline.watchdog_runner \
      --benchmark "$bench" --instance-ids "$ids" \
      --wall-s "$wall" --startup-grace-s 8 --poll-interval-s 0.5 \
      --rss-cap-gb "$rss" --grace-kill-s 3 \
      --device "$device" --dtype float64 --strict-bounded-failure \
      --out-dir "$out" --canonical-root "$VNNLIB_ROOT" \
      > "$out/driver.log" 2>&1
  local rc=$?
  echo "$bench rc=$rc -> $out"
  [ -f "$out/watchdog_summary.json" ] && $ACT_PY -c "
import json
d = json.load(open('$out/watchdog_summary.json'))
print('  counts:', d.get('counts'))"
  echo "Next: ln -sfn $out $CONSOLIDATED/$bench/_source_<label> && cd $CONSOLIDATED && python3 build_csvs.py"
}

# --- Helper: rebuild + audit --------------------------------------------------
act_audit() {
  cd $CONSOLIDATED
  python3 build_csvs.py
  python3 soundness_check.py
}

# --- Helper: pre-flight -------------------------------------------------------
act_preflight() {
  echo "=== Competing ACT processes ==="
  ps -ef | grep -E "watchdog_runner|act\.pipeline" | grep -v grep | head -5
  echo "  (should be empty)"
  echo "=== System ==="
  uptime
  free -h | grep -E "Mem:|Swap:"
  echo "=== GPU ==="
  nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null
}

echo "ACT r93 QUICKSTART loaded."
echo "Helpers: act_run <bench> <ids> <wall> <rss> [cpu|cuda] [label]"
echo "         act_audit              # rebuild CSVs + cross-check vs official"
echo "         act_preflight          # check no competing processes, RAM/GPU OK"
echo "Read $CONSOLIDATED/PLAYBOOK.md for full context."
