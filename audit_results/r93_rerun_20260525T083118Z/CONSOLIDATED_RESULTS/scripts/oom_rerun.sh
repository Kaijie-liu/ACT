#!/bin/bash
# Rerun all 87 GPU OOM iids individually with bigger wall + RSS budget.
# GPU is currently idle (95 GB free) so resource contention shouldn't recur.
# Writes to a new BASE so it ingests as separate `_source_oom_rerun` per bench.
set -u

PY=/data1/Kane/miniconda3/envs/act-py312/bin/python
ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
BASE=/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/oom_rerun_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$BASE"
echo "BASE=$BASE" | tee "$BASE/README.txt"
date -u | tee -a "$BASE/README.txt"

export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=$ROOT
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

run_one() {
  local bench=$1 ids=$2 wall=$3
  local out="$BASE/$bench"
  mkdir -p "$out"
  echo "=== $(date -u) :: cuda OOM rerun :: $bench wall=${wall}s ids=$ids ===" | tee -a "$BASE/README.txt"
  $PY -m act.pipeline.watchdog_runner \
      --benchmark "$bench" --instance-ids "$ids" \
      --wall-s "$wall" --startup-grace-s 30 --poll-interval-s 0.5 \
      --rss-cap-gb 64 --grace-kill-s 3 \
      --device cuda --dtype float64 --strict-bounded-failure \
      --out-dir "$out" --canonical-root "$ROOT" \
      > "$out/driver.log" 2>&1
  echo "  rc=$?" | tee -a "$BASE/README.txt"
  [ -f "$out/watchdog_summary.json" ] && $PY -c "
import json
d = json.load(open('$out/watchdog_summary.json'))
print('  counts:', d.get('counts'))" | tee -a "$BASE/README.txt"
}

# Small benchmarks first (quick win)
run_one ml4acopf_2024       "58,59,60"           120
run_one metaroom_2023       "30,33"              180
run_one tinyimagenet_2024   "66,67"              180
run_one relusplitter        "14"                 120

# cifar100 79 OOM iids — biggest chunk
run_one cifar100_2024 \
  "100,101,102,103,104,105,106,107,109,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,137,138,140,141,142,143,146,147,148,149,150,151,152,155,156,157,159,161,162,163,164,165,167,169,170,171,172,174,175,176,177,178,179,181,184,185,186,187,188,189,190,191,192,193,195,197,198" \
  180

date -u | tee -a "$BASE/README.txt"
echo "=== oom_rerun done ===" | tee -a "$BASE/README.txt"
