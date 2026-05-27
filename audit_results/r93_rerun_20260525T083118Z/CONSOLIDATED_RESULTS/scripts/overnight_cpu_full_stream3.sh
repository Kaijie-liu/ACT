#!/bin/bash
# Stream 3 (heavy CNN, 24 GiB peak, ~11 hr): cifar100 + tinyimagenet.
# This is the wall-time bottleneck for the 3-stream parallel design.
set -u

PY=/data1/Kane/miniconda3/envs/act-py312/bin/python
ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
BASE=/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/overnight_cpu_full_stream3_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$BASE"
echo "BASE=$BASE STREAM=3" | tee "$BASE/README.txt"
date -u | tee -a "$BASE/README.txt"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=$ROOT

run_one() {
  local bench=$1 instance_ids=$2 wall=$3 rss_gb=$4
  local out="$BASE/$bench"
  mkdir -p "$out"
  echo "=== $(date -u) :: stream3 :: $bench ===" | tee -a "$BASE/README.txt"
  echo "  wall=${wall}s rss=${rss_gb}GiB ids=$(echo $instance_ids | cut -c1-60)..." | tee -a "$BASE/README.txt"
  $PY -m act.pipeline.watchdog_runner \
      --benchmark "$bench" --instance-ids "$instance_ids" \
      --wall-s "$wall" --startup-grace-s 8 --poll-interval-s 0.5 \
      --rss-cap-gb "$rss_gb" --grace-kill-s 3 \
      --device cpu --dtype float64 --strict-bounded-failure \
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

run_one cifar100_2024 "$(ids_range 5 199)" 120 24
run_one tinyimagenet_2024 "$(ids_range 5 199)" 120 24

date -u | tee -a "$BASE/README.txt"
echo "=== stream3 done ===" | tee -a "$BASE/README.txt"
