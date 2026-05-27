#!/bin/bash
# Round 5 CUDA-only parallel: runs alongside the CPU-only Round 5 sweep so
# GPU starts producing data immediately rather than waiting for CPU section.
set -u

PY=/data1/Kane/miniconda3/envs/act-py312/bin/python
ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
BASE=/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/round5_cuda_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$BASE"
echo "BASE=$BASE STREAM=cuda_only" | tee "$BASE/README.txt"
date -u | tee -a "$BASE/README.txt"

export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=$ROOT
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

run_one() {
  local bench=$1 ids=$2 wall=$3 rss=$4
  local out="$BASE/$bench"
  mkdir -p "$out"
  echo "=== $(date -u) :: cuda :: $bench wall=${wall}s ===" | tee -a "$BASE/README.txt"
  $PY -m act.pipeline.watchdog_runner \
      --benchmark "$bench" --instance-ids "$ids" \
      --wall-s "$wall" --startup-grace-s 15 --poll-interval-s 0.5 \
      --rss-cap-gb "$rss" --grace-kill-s 3 \
      --device cuda --dtype float64 --strict-bounded-failure \
      --out-dir "$out" --canonical-root "$ROOT" \
      > "$out/driver.log" 2>&1
  echo "  rc=$?" | tee -a "$BASE/README.txt"
  [ -f "$out/watchdog_summary.json" ] && $PY -c "
import json
d = json.load(open('$out/watchdog_summary.json'))
print('  counts:', d.get('counts'))" | tee -a "$BASE/README.txt"
}

ids_range() { seq -s, "$1" "$2"; }

# Order: smallest first so we can confirm GPU path quickly
run_one collins_aerospace_benchmark "$(ids_range 0 5)" 120 8
run_one ml4acopf_2024 "$(ids_range 0 68)" 60 8
run_one lsnc_relu "$(ids_range 0 79)" 60 8
run_one yolo_2023 "$(ids_range 0 71)" 90 24

date -u | tee -a "$BASE/README.txt"
echo "=== round5_cuda_only done ===" | tee -a "$BASE/README.txt"
