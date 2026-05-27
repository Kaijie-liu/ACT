#!/bin/bash
# Overnight CPU FULL-COVERAGE run for benchmarks that previously only had
# bounded smokes. Skips iids already covered by Round 2 smoke. Strict
# watchdog throughout; per-instance JSON + driver.log + watchdog_summary.json
# saved per benchmark. CONSOLIDATED_RESULTS ingestion happens after-the-fact
# via build_csvs.py (manual `ln -s` + rebuild).
#
# Total wall budget target: ~17 hours sequential.
#
# Memory discipline: RSS cap <= 24 GiB per inst. System has 64 GiB free at
# launch, swap already saturated; do NOT raise the cap without flushing swap.
set -u  # do NOT set -e: continue past single-benchmark failures

PY=/data1/Kane/miniconda3/envs/act-py312/bin/python
ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
BASE=/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/overnight_cpu_full_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$BASE"
echo "BASE=$BASE" | tee "$BASE/README.txt"
date -u | tee -a "$BASE/README.txt"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=$ROOT

run_one() {
  local bench=$1
  local instance_ids=$2
  local wall=$3
  local rss_gb=$4
  local out="$BASE/$bench"
  mkdir -p "$out"
  echo "=== $(date -u) :: $bench ===" | tee -a "$BASE/README.txt"
  local ids_preview=$(echo "$instance_ids" | cut -c1-80)
  echo "  out=$out wall=${wall}s rss=${rss_gb}GiB ids=${ids_preview}..." | tee -a "$BASE/README.txt"
  $PY -m act.pipeline.watchdog_runner \
      --benchmark "$bench" --instance-ids "$instance_ids" \
      --wall-s "$wall" --startup-grace-s 8 --poll-interval-s 0.5 \
      --rss-cap-gb "$rss_gb" --grace-kill-s 3 \
      --device cpu --dtype float64 --strict-bounded-failure \
      --out-dir "$out" --canonical-root "$ROOT" \
      > "$out/driver.log" 2>&1
  local rc=$?
  echo "  -> rc=$rc" | tee -a "$BASE/README.txt"
  if [ -f "$out/watchdog_summary.json" ]; then
    $PY -c "
import json
d = json.load(open('$out/watchdog_summary.json'))
print('   counts:', d.get('counts'))
" | tee -a "$BASE/README.txt"
  fi
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
print('   verdicts:', dict(agg))
" | tee -a "$BASE/README.txt"
}

ids_range() { seq -s, "$1" "$2"; }

# ===== Tier A — light/medium (smoke had OK or mixed) =====
run_one cora_2024 "$(ids_range 10 179)" 60 6
run_one soundnessbench "$(ids_range 10 49)" 60 8
run_one traffic_signs_recognition_2023 "$(ids_range 5 44)" 60 8
run_one metaroom_2023 "$(ids_range 5 99)" 90 24

# ===== Tier B — heavy CNN (smoke had RSS/TO issues; bigger budget) =====
run_one vggnet16_2022 "$(ids_range 5 17)" 120 24
run_one yolo_2023 "$(ids_range 10 71)" 120 24
run_one cifar100_2024 "$(ids_range 5 199)" 120 24
run_one tinyimagenet_2024 "$(ids_range 5 199)" 120 24

# ===== Tier C — nn4sys query-explosion family (likely all TO) =====
# Per memory: lindex_200+ expands to ~400 queries per instance; expected
# behavior is bounded UNKNOWN_TIMEOUT, but it's our last unrun ACT-supported
# subset and must be archived for completeness.
run_one nn4sys "$(ids_range 107 193)" 60 6

date -u | tee -a "$BASE/README.txt"
echo "=== overnight CPU FULL-COVERAGE done ===" | tee -a "$BASE/README.txt"
