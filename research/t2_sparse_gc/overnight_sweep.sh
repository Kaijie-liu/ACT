#!/bin/bash
# Overnight sweep: compare 5 modes on a slice of cifar100 resnet_large
# and tinyimagenet. Records verdict, wall, RSS per iid per mode.
set -u
export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

RSS_CAP=24
WALL=300

declare -A MODES
MODES[t2b_only]="ACT_HZ_DENSE_TO_SPARSE=1 ACT_HZ_PRECONV_SPARSE=1 ACT_HZ_PRECONV_BUDGET_MIB=512"
MODES[b3_full_kmax2000]="ACT_HZ_DENSE_TO_SPARSE=1 ACT_HZ_PRECONV_SPARSE=1 ACT_HZ_PRECONV_BUDGET_MIB=512 ACT_HZ_SPARSE_EQ_LAGR=1 ACT_HZ_SPARSE_EQ_LAGR_K_MAX=2000"
MODES[b3_full_kmax500]="ACT_HZ_DENSE_TO_SPARSE=1 ACT_HZ_PRECONV_SPARSE=1 ACT_HZ_PRECONV_BUDGET_MIB=512 ACT_HZ_SPARSE_EQ_LAGR=1 ACT_HZ_SPARSE_EQ_LAGR_K_MAX=500"
MODES[b3_compact_kmax2000]="ACT_HZ_DENSE_TO_SPARSE=1 ACT_HZ_PRECONV_SPARSE=1 ACT_HZ_PRECONV_BUDGET_MIB=512 ACT_HZ_SPARSE_EQ_LAGR=1 ACT_HZ_SPARSE_EQ_LAGR_K_MAX=2000 ACT_HZ_SPARSE_EQ_LAGR_COMPACT=1"
MODES[b3_compact_kmax5000]="ACT_HZ_DENSE_TO_SPARSE=1 ACT_HZ_PRECONV_SPARSE=1 ACT_HZ_PRECONV_BUDGET_MIB=512 ACT_HZ_SPARSE_EQ_LAGR=1 ACT_HZ_SPARSE_EQ_LAGR_K_MAX=5000 ACT_HZ_SPARSE_EQ_LAGR_COMPACT=1"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/data1/Kane/ACT/audit_results/overnight_b3_${STAMP}"
mkdir -p "$ROOT"
echo "ROOT=$ROOT"

# Smaller iid set for sweep
declare -A BENCHES
BENCHES[cifar100_2024]="100,120,140,160,180"
BENCHES[tinyimagenet_2024]="0,40,80,120,160"

for BENCH in cifar100_2024 tinyimagenet_2024; do
  IIDS="${BENCHES[$BENCH]}"
  for MODE in t2b_only b3_full_kmax2000 b3_full_kmax500 b3_compact_kmax2000 b3_compact_kmax5000; do
    OUT="$ROOT/${BENCH}_${MODE}"
    mkdir -p "$OUT"
    echo "===> bench=$BENCH mode=$MODE iids=$IIDS @ $(date)" | tee -a "$ROOT/log.txt"
    # Unset all knobs first
    unset ACT_HZ_DENSE_TO_SPARSE ACT_HZ_PRECONV_SPARSE ACT_HZ_PRECONV_BUDGET_MIB ACT_HZ_SPARSE_EQ_LAGR ACT_HZ_SPARSE_EQ_LAGR_K_MAX ACT_HZ_SPARSE_EQ_LAGR_COMPACT
    # Apply mode knobs
    eval "export ${MODES[$MODE]}"
    /data1/Kane/miniconda3/envs/act-py312/bin/python \
      -m act.pipeline.watchdog_runner \
      --benchmark "$BENCH" --instance-ids "$IIDS" \
      --wall-s "$WALL" --startup-grace-s 8 --poll-interval-s 0.5 \
      --rss-cap-gb "$RSS_CAP" --grace-kill-s 3 \
      --device cpu --dtype float64 \
      --out-dir "$OUT" \
      --canonical-root /data1/Kane/data/vnncomp2025_benchmarks/benchmarks \
      > "$OUT/driver.log" 2>&1
    /data1/Kane/miniconda3/envs/act-py312/bin/python <<EOF | tee -a "$ROOT/log.txt"
import json, glob
from collections import Counter
c = Counter()
rss_list = []
for r in json.load(open("$OUT/watchdog_summary.json")).get('results', []):
    c[r.get('cli_normalized', '?')] += 1
    if r.get('peak_rss_mb'):
        rss_list.append(float(r['peak_rss_mb']))
mean_rss = sum(rss_list)/max(len(rss_list), 1)
print(f"  bench=$BENCH mode=$MODE verdicts={dict(c)} mean_RSS_MB={mean_rss:.0f} n={len(rss_list)}")
EOF
  done
done
echo "===> SWEEP DONE @ $(date)" | tee -a "$ROOT/log.txt"
