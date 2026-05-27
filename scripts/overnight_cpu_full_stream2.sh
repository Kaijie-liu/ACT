#!/bin/bash
# Stream 2 (medium, 24 GiB peak): metaroom + vggnet16 + yolo.
set -u

PY=/data1/Kane/miniconda3/envs/act-py312/bin/python
ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
BASE=/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/overnight_cpu_full_stream2_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$BASE"
echo "BASE=$BASE STREAM=2" | tee "$BASE/README.txt"
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
  echo "=== $(date -u) :: stream2 :: $bench ===" | tee -a "$BASE/README.txt"
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

run_one metaroom_2023 "$(ids_range 5 99)" 90 24
run_one vggnet16_2022 "$(ids_range 5 17)" 120 24
run_one yolo_2023 "$(ids_range 10 71)" 120 24

date -u | tee -a "$BASE/README.txt"
echo "=== stream2 done ===" | tee -a "$BASE/README.txt"
