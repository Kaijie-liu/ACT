#!/bin/bash
# Round 2: more benchmarks we either haven't touched on CPU at all, or
# only ran a fragment of. Serial under the strict watchdog; everything
# gets a wall + RSS cap so a single bad instance can't eat the night.
#
# Tier 1 — high confidence runnable (R11–R17 op coverage proves conversion):
#   * nn4sys mscn + remaining pensieve (iids 50..104)
#   * relusplitter remaining (iids 30..219)
#   * cersyve native (tries the R16 ready-check path on the OnnxSlice
#     model — failure was 5/5 ERROR pre-R16; this gates whether R16
#     also unblocked cersyve incidentally)
#
# Tier 2 — bounded-smoke blocked pre-R16, may now run for at least a
# subset; smoke each with 10 instances:
#   * cora_2024
#   * soundnessbench
#   * collins_aerospace_benchmark (6 total)
#
# Tier 3 — large CNN, scale risk; 5 instances each with bigger RSS cap:
#   * traffic_signs_recognition_2023
#   * vggnet16_2022
#   * metaroom_2023
#   * cifar100_2024
#   * tinyimagenet_2024
#
# Soundness discipline: --strict-bounded-failure throughout, so bounded
# UNKNOWN counts as run_status=FAILED at the watchdog summary level.
# The CLI per-instance JSONs still carry the real verdict for audit.
set -u

PY=/data1/Kane/miniconda3/envs/act-py312/bin/python
ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
BASE=/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/overnight_cpu_round2_$(date -u +%Y%m%dT%H%M%SZ)
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
  echo "  out=$out wall=${wall}s rss=${rss_gb}GiB ids=$instance_ids" | tee -a "$BASE/README.txt"
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
  # Also aggregate real CLI verdicts (CERT/FAL/UNK/ERR) where available.
  $PY -c "
import glob, json
from collections import Counter
fs = sorted(glob.glob('$out/per_instance_${bench}_*.json'))
agg = Counter()
for f in fs:
    if 'watchdog' in f: continue
    d = json.load(open(f))
    for p in d.get('per_instance', []):
        agg[p['cli_normalized']] += 1
print('   verdicts:', dict(agg))
" | tee -a "$BASE/README.txt"
}

# Helper: build ids "lo,lo+1,...,hi"
ids_range() {
  seq -s, "$1" "$2"
}

# ===== Tier 1 — high confidence =====
# nn4sys: remaining mscn+pensieve subset (50..104). Excludes lindex_200+
# (107..194) which is the query-explosion family — bounded UNKNOWN_TIMEOUT
# is the expected outcome but at 30s/inst they would still chew up the
# whole night.
run_one nn4sys "$(ids_range 50 104)" 30 6

# relusplitter remaining (30..219)
run_one relusplitter "$(ids_range 30 219)" 30 6

# cersyve: 12 small instances; R16 ready-check may have unblocked native
run_one cersyve "$(ids_range 0 11)" 30 4

# ===== Tier 2 — bounded-smoke blocked pre-R16 =====
# cora_2024: 10-instance smoke
run_one cora_2024 "$(ids_range 0 9)" 30 4

# soundnessbench: 10-instance smoke
run_one soundnessbench "$(ids_range 0 9)" 30 4

# collins_aerospace_benchmark: 6 total
run_one collins_aerospace_benchmark "$(ids_range 0 5)" 30 6

# ===== Tier 3 — large CNN scale risk =====
# 5 instances each with 60s + 16GiB cap. 0 ERROR is the win condition
# even if all RSS-cap out.
run_one traffic_signs_recognition_2023 "$(ids_range 0 4)" 60 16
run_one vggnet16_2022 "$(ids_range 0 4)" 60 16
run_one metaroom_2023 "$(ids_range 0 4)" 60 16
run_one cifar100_2024 "$(ids_range 0 4)" 60 16
run_one tinyimagenet_2024 "$(ids_range 0 4)" 60 16

date -u | tee -a "$BASE/README.txt"
echo "=== overnight CPU round 2 done ===" | tee -a "$BASE/README.txt"
