#!/bin/bash
# Overnight CPU serial run of benchmarks that have not had full-bench CPU
# runs yet, but whose conversion path has been validated by smokes.
#
# All runs use the process-level watchdog with --strict-bounded-failure
# so bounded UNKNOWN counts as run_status=FAILED (paper-grade discipline)
# and the driver exit code is non-zero whenever the watchdog had to
# terminate. The CLI itself stays in non-strict mode (formal receipts
# only when --formal is set inline).
#
# Sequential (not parallel) — these are CPU-only and would contend.
# Each benchmark gets its own output dir; per-instance JSONs come from
# the CLI proper.
#
# Total wall budget target: ~3-4 hours.
set -u  # do NOT set -e: we want to continue past a single benchmark's failure

PY=/data1/Kane/miniconda3/envs/act-py312/bin/python
ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
BASE=/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/overnight_cpu_$(date -u +%Y%m%dT%H%M%SZ)
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
  local n_max=$2
  local wall=$3
  local rss_gb=$4
  local instance_ids=${5:-}
  local out="$BASE/$bench"
  mkdir -p "$out"
  echo "=== $(date -u) :: $bench ===" | tee -a "$BASE/README.txt"
  echo "  out=$out wall=${wall}s rss=${rss_gb}GiB max=$n_max ids=${instance_ids:-all}" | tee -a "$BASE/README.txt"
  local args=()
  args+=(--benchmark "$bench")
  if [ -n "$instance_ids" ]; then
    args+=(--instance-ids "$instance_ids")
  else
    # build comma list 0..n_max-1
    local n=$((n_max - 1))
    local ids
    ids=$(seq -s, 0 "$n")
    args+=(--instance-ids "$ids")
  fi
  args+=(--wall-s "$wall" --startup-grace-s 8 --poll-interval-s 0.5)
  args+=(--rss-cap-gb "$rss_gb" --grace-kill-s 3)
  args+=(--device cpu --dtype float64)
  args+=(--strict-bounded-failure)
  args+=(--out-dir "$out" --canonical-root "$ROOT")
  $PY -m act.pipeline.watchdog_runner "${args[@]}" \
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
}

# ----- Lineup -----
# 1. lsnc_relu full (80)   — fast UNK on smoke, ~0.5s/inst expected
run_one lsnc_relu 80 30 4

# 2. ml4acopf_2024 full (69) — R15 just landed; 9-family gate passed
run_one ml4acopf_2024 69 45 6

# 3. nn4sys 50 first      — exclude lindex_200+ which is query-explosion
#    Picks: pensieve sub-families (0..104) plus lindex_1 (iids 105,106)
run_one nn4sys 50 40 6 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,105,106

# 4. relusplitter first 30 — only 1-instance smoke before
run_one relusplitter 30 30 4

# 5. yolo_2023 first 10   — known ~90s/inst, strict cap will terminate slow ones
run_one yolo_2023 10 60 16

date -u | tee -a "$BASE/README.txt"
echo "=== overnight CPU run done ===" | tee -a "$BASE/README.txt"
