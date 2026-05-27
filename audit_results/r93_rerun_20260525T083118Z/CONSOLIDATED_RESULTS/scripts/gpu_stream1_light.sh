#!/bin/bash
# GPU Stream 1: light/medium benchmarks (small models, fast per-inst).
# Sequential within stream; runs in parallel with stream2 + stream3.
# Each ACT subprocess uses device=cuda dtype=float64 to match the CPU
# bit-identical convention.
set -u

PY=/data1/Kane/miniconda3/envs/act-py312/bin/python
ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
BASE=/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/gpu_stream1_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$BASE"
echo "BASE=$BASE STREAM=gpu1" | tee "$BASE/README.txt"
date -u | tee -a "$BASE/README.txt"

export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=$ROOT
# Don't oversubscribe CPU side of GPU run; let CUDA threads breathe
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

run_one() {
  local bench=$1 instance_ids=$2 wall=$3
  local out="$BASE/$bench"
  mkdir -p "$out"
  echo "=== $(date -u) :: gpu1 :: $bench ===" | tee -a "$BASE/README.txt"
  echo "  wall=${wall}s ids=$(echo $instance_ids | cut -c1-60)..." | tee -a "$BASE/README.txt"
  $PY -m act.pipeline.watchdog_runner \
      --benchmark "$bench" --instance-ids "$instance_ids" \
      --wall-s "$wall" --startup-grace-s 15 --poll-interval-s 0.5 \
      --rss-cap-gb 32 --grace-kill-s 3 \
      --device cuda --dtype float64 --strict-bounded-failure \
      --out-dir "$out" --canonical-root "$ROOT" \
      > "$out/driver.log" 2>&1
  echo "  -> rc=$?" | tee -a "$BASE/README.txt"
  [ -f "$out/watchdog_summary.json" ] && $PY -c "
import json
d = json.load(open('$out/watchdog_summary.json'))
print('   counts:', d.get('counts'))" | tee -a "$BASE/README.txt"
  $PY -c "
import glob, json
from collections import Counter
fs = sorted(glob.glob('$out/per_instance_${bench}_*.json'))
agg = Counter()
for f in fs:
    if 'watchdog' in f: continue
    try: d = json.load(open(f))
    except: continue
    for p in d.get('per_instance', []):
        agg[p.get('cli_normalized','?')] += 1
print('   verdicts:', dict(agg))" | tee -a "$BASE/README.txt"
}

ids_range() { seq -s, "$1" "$2"; }

# Order: smallest first to validate GPU path before committing to longer runs
run_one collins_aerospace_benchmark "$(ids_range 0 5)" 60
run_one cersyve "$(ids_range 0 11)" 60
run_one tllverifybench_2023 "$(ids_range 0 31)" 60
run_one traffic_signs_recognition_2023 "$(ids_range 0 44)" 60
run_one soundnessbench "$(ids_range 0 49)" 60
run_one ml4acopf_2024 "$(ids_range 0 68)" 60
run_one dist_shift_2023 "$(ids_range 0 71)" 60
run_one lsnc_relu "$(ids_range 0 79)" 60
run_one metaroom_2023 "$(ids_range 0 99)" 90
run_one cora_2024 "$(ids_range 0 179)" 60
run_one cgan_2023 "$(ids_range 0 20)" 120

date -u | tee -a "$BASE/README.txt"
echo "=== gpu_stream1 done ===" | tee -a "$BASE/README.txt"
