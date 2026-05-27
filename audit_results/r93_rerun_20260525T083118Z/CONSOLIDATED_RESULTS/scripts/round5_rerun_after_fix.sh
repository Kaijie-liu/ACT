#!/bin/bash
# Round 5: re-run the 4 benchmarks that the Round-4 raw-first fix regressed,
# now under the simplify-first/raw-fallback ordering (act_fixes_diff #5).
# Run BOTH CPU and GPU per benchmark to refresh CPU baseline AND get GPU.
set -u

PY=/data1/Kane/miniconda3/envs/act-py312/bin/python
ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
BASE=/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/round5_aftersimplify_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$BASE"
echo "BASE=$BASE" | tee "$BASE/README.txt"
date -u | tee -a "$BASE/README.txt"

export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=$ROOT
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

run_one() {
  local bench=$1 ids=$2 wall=$3 rss=$4 device=$5
  local out="$BASE/${bench}_${device}"
  mkdir -p "$out"
  echo "=== $(date -u) :: $bench/$device wall=${wall}s ===" | tee -a "$BASE/README.txt"
  $PY -m act.pipeline.watchdog_runner \
      --benchmark "$bench" --instance-ids "$ids" \
      --wall-s "$wall" --startup-grace-s 15 --poll-interval-s 0.5 \
      --rss-cap-gb "$rss" --grace-kill-s 3 \
      --device "$device" --dtype float64 --strict-bounded-failure \
      --out-dir "$out" --canonical-root "$ROOT" \
      > "$out/driver.log" 2>&1
  echo "  rc=$?" | tee -a "$BASE/README.txt"
  [ -f "$out/watchdog_summary.json" ] && $PY -c "
import json
d = json.load(open('$out/watchdog_summary.json'))
print('  counts:', d.get('counts'))" | tee -a "$BASE/README.txt"
}

ids_range() { seq -s, "$1" "$2"; }

# ml4acopf 0..68 (69 inst) — was 5 CERT in Round 1 pre-regression
run_one ml4acopf_2024 "$(ids_range 0 68)" 60 8 cpu
run_one ml4acopf_2024 "$(ids_range 0 68)" 60 8 cuda

# lsnc_relu 0..79 (80 inst) — was 78 UNK in Round 1
run_one lsnc_relu "$(ids_range 0 79)" 60 8 cpu
run_one lsnc_relu "$(ids_range 0 79)" 60 8 cuda

# yolo_2023 0..71 (72 inst) — was 0V/72RSS on CPU, all ERROR on GPU
run_one yolo_2023 "$(ids_range 0 71)" 90 24 cpu
run_one yolo_2023 "$(ids_range 0 71)" 90 24 cuda

# collins_aerospace_benchmark 0..5 (6 inst) — slow spec parse, give 120s wall
run_one collins_aerospace_benchmark "$(ids_range 0 5)" 120 8 cpu
run_one collins_aerospace_benchmark "$(ids_range 0 5)" 120 8 cuda

date -u | tee -a "$BASE/README.txt"
echo "=== round5_aftersimplify done ===" | tee -a "$BASE/README.txt"
