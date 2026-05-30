#!/bin/bash
# T2 pilot: cifar100_2024 iid 0 — measure RSS reduction from
# ACT_HZ_DENSE_TO_SPARSE=1 (+ optional prune) vs baseline.
#
# RSS cap deliberately set to 24 GiB to match the historical
# cifar100 RSS-bound cohort. If sparse path keeps peak RSS below
# the cap on instances that baseline OOMs at, that's the T2 win.
#
# Usage:
#   bash pilot_cifar100_iid0.sh baseline    # knobs off
#   bash pilot_cifar100_iid0.sh sparse_t2   # ACT_HZ_DENSE_TO_SPARSE=1
#   bash pilot_cifar100_iid0.sh both        # prune + sparse
set -u

MODE="${1:-sparse_t2}"
BENCH=cifar100_2024
IID=0
WALL=120
RSS_CAP_GB=24

case "$MODE" in
    baseline)
        unset ACT_HZ_PRUNE_GC ACT_HZ_DENSE_TO_SPARSE
        ;;
    sparse_t2)
        export ACT_HZ_PRUNE_GC=0
        export ACT_HZ_DENSE_TO_SPARSE=1
        export ACT_HZ_SPARSE_GC_DENSITY=0.10
        ;;
    both)
        export ACT_HZ_PRUNE_GC=1
        export ACT_HZ_PRUNE_GC_THRESH=1e-9
        export ACT_HZ_DENSE_TO_SPARSE=1
        export ACT_HZ_SPARSE_GC_DENSITY=0.10
        ;;
    *)
        echo "usage: $0 {baseline,sparse_t2,both}" >&2
        exit 2
        ;;
esac

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="/data1/Kane/ACT/audit_results/t2_pilot_${MODE}_${STAMP}"
mkdir -p "$OUT"

export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "=== T2 pilot $MODE bench=$BENCH iid=$IID wall=${WALL}s rss_cap=${RSS_CAP_GB}GB ==="
echo "ACT_HZ_PRUNE_GC=${ACT_HZ_PRUNE_GC:-unset}"
echo "ACT_HZ_DENSE_TO_SPARSE=${ACT_HZ_DENSE_TO_SPARSE:-unset}"
echo "OUT=$OUT"
echo ""

START_T=$(date +%s)
/data1/Kane/miniconda3/envs/act-py312/bin/python \
    -m act.pipeline.watchdog_runner \
    --benchmark "$BENCH" --instance-ids "$IID" \
    --wall-s "$WALL" --startup-grace-s 8 --poll-interval-s 0.5 \
    --rss-cap-gb "$RSS_CAP_GB" --grace-kill-s 3 \
    --device cpu --dtype float64 \
    --out-dir "$OUT" \
    --canonical-root /data1/Kane/data/vnncomp2025_benchmarks/benchmarks \
    > "$OUT/driver.log" 2>&1
RC=$?
END_T=$(date +%s)

WALL_S=$((END_T - START_T))
echo "rc=$RC  wall=${WALL_S}s"

# Parse verdict + peak RSS.
/data1/Kane/miniconda3/envs/act-py312/bin/python <<EOF
import glob, json
results = sorted(glob.glob("$OUT/per_instance_*.json"))
for f in results:
    if 'watchdog' in f: continue
    d = json.load(open(f))
    for p in d.get('per_instance', []):
        print(f"verdict={p.get('cli_normalized')} wall_s={p.get('wall_s', 0):.1f}")
        break
    break
ws = json.load(open("$OUT/watchdog_summary.json"))
r = ws.get('results', [{}])[0]
print(f"watchdog_status={r.get('watchdog_status')} peak_rss_mb={r.get('peak_rss_mb')}")
EOF
