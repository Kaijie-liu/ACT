#!/bin/bash
# GPU Stream 3: heavy CNN — vggnet16 (138M params), yolo, cifar100, tinyimagenet.
# These hit RSS-cap on CPU; GPU has 96 GB so they should run. Sequential
# within stream to avoid GPU OOM (single inst can take 10-30 GB GPU mem).
set -u

PY=/data1/Kane/miniconda3/envs/act-py312/bin/python
ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
BASE=/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/gpu_stream3_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$BASE"
echo "BASE=$BASE STREAM=gpu3" | tee "$BASE/README.txt"
date -u | tee -a "$BASE/README.txt"

export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=$ROOT
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

run_one() {
  local bench=$1 instance_ids=$2 wall=$3
  local out="$BASE/$bench"
  mkdir -p "$out"
  echo "=== $(date -u) :: gpu3 :: $bench ===" | tee -a "$BASE/README.txt"
  echo "  wall=${wall}s ids=$(echo $instance_ids | cut -c1-60)..." | tee -a "$BASE/README.txt"
  $PY -m act.pipeline.watchdog_runner \
      --benchmark "$bench" --instance-ids "$instance_ids" \
      --wall-s "$wall" --startup-grace-s 30 --poll-interval-s 0.5 \
      --rss-cap-gb 64 --grace-kill-s 3 \
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

# Order: smallest first (vggnet16 has only 18 inst)
run_one vggnet16_2022 "$(ids_range 0 17)" 240
run_one yolo_2023 "$(ids_range 0 71)" 180
run_one cifar100_2024 "$(ids_range 0 199)" 120
run_one tinyimagenet_2024 "$(ids_range 0 199)" 120

date -u | tee -a "$BASE/README.txt"
echo "=== gpu_stream3 done ===" | tee -a "$BASE/README.txt"
