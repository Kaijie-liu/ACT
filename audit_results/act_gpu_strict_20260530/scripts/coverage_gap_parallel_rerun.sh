#!/bin/bash
# Coverage gap rerun — parallel to ongoing cora 180
# Targets benchmarks where new code might trigger unseen V/A
set -u
export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/data1/Kane/ACT/audit_results/coverage_gap_${STAMP}"
mkdir -p "$ROOT"
LOG="$ROOT/MAIN.log"
echo "Coverage gap rerun started: $(date)" | tee "$LOG"

spawn() {
    local bench=$1; local iids=$2; local wall=$3; local rss=$4
    local OUT="$ROOT/${bench}"
    mkdir -p "$OUT"
    (PYTHONPATH=/data1/Kane/ACT \
     ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks \
     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /data1/Kane/miniconda3/envs/act-py312/bin/python \
        -m act.pipeline.watchdog_runner \
        --benchmark "$bench" --instance-ids "$iids" \
        --wall-s "$wall" --startup-grace-s 8 --poll-interval-s 0.5 \
        --rss-cap-gb "$rss" --grace-kill-s 3 \
        --device cuda --dtype float64 \
        --out-dir "$OUT" \
        --canonical-root /data1/Kane/data/vnncomp2025_benchmarks/benchmarks \
        > "$OUT/d.log" 2>&1) &
    echo "  spawned $bench PID=$! n_iids=$(echo $iids | tr ',' '\n' | wc -l) wall=${wall}s" | tee -a "$LOG"
}

# HIGH-priority: metaroom 13 r93-non-CERT iids (singleton fastpath candidate)
spawn "metaroom_2023" "14,22,26,27,28,30,33,35,45,49,78,95,97" 120 6

# MEDIUM-priority: extended samples on 0-verdict GPU benches
spawn "cifar100_2024" "$(seq -s, 0 29)" 180 8
spawn "tinyimagenet_2024" "$(seq -s, 0 29)" 180 8
spawn "dist_shift_2023" "$(seq -s, 0 29)" 120 6
spawn "yolo_2023" "$(seq -s, 0 19)" 180 8

echo "Waiting for all 5..." | tee -a "$LOG"
wait

# Synthesis
echo "" | tee -a "$LOG"
echo "==== SYNTHESIS ====" | tee -a "$LOG"
/data1/Kane/miniconda3/envs/act-py312/bin/python <<EOF | tee -a "$LOG"
import json, glob, os
from collections import Counter
root = "$ROOT"
for bench in ['metaroom_2023','cifar100_2024','tinyimagenet_2024','dist_shift_2023','yolo_2023']:
    c = Counter(); walls = []
    for f in sorted(glob.glob(os.path.join(root, bench, 'per_instance_*.json'))):
        try:
            d = json.load(open(f))
            for p in d.get('per_instance', []):
                c[p.get('cli_normalized','?')] += 1
                if p.get('wall_s'): walls.append(float(p['wall_s']))
                break
        except: pass
    n = sum(c.values()); mw = sum(walls)/max(len(walls),1) if walls else 0
    V = c.get('CERTIFIED',0); A = c.get('FALSIFIED',0)
    marker = " ⭐" if (V + A) > 0 else ""
    print(f"  {bench:30s} n={n:>3} V={V:>3} A={A:>2} {dict(c)} mean={mw:.0f}s{marker}")
EOF
echo "DONE: $(date)" | tee -a "$LOG"
echo "ROOT=$ROOT"
