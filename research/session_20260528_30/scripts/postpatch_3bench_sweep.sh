#!/bin/bash
# Sequential GPU sweep — cgan full + safenlp sample + cora sample
# Tests if overnight changes (zero-width prune, singleton fastpath, eq-layer bridge fix, fail-closed)
# give signal on ERROR-heavy benchmarks.
set -u
export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/data1/Kane/ACT/audit_results/postpatch_3bench_${STAMP}"
mkdir -p "$ROOT"
LOG="$ROOT/MAIN.log"
echo "Started: $(date)" | tee "$LOG"

run_bench() {
    local bench=$1; local iids=$2; local wall=$3
    local OUT="$ROOT/${bench}"
    mkdir -p "$OUT"
    echo "===> $bench iids=$iids wall=${wall}s @ $(date)" | tee -a "$LOG"
    /data1/Kane/miniconda3/envs/act-py312/bin/python \
        -m act.pipeline.watchdog_runner \
        --benchmark "$bench" --instance-ids "$iids" \
        --wall-s "$wall" --startup-grace-s 8 --poll-interval-s 0.5 \
        --rss-cap-gb 24 --grace-kill-s 3 \
        --device cuda --dtype float64 \
        --out-dir "$OUT" \
        --canonical-root /data1/Kane/data/vnncomp2025_benchmarks/benchmarks \
        > "$OUT/d.log" 2>&1
    /data1/Kane/miniconda3/envs/act-py312/bin/python <<EOF | tee -a "$LOG"
import json, glob
from collections import Counter
c = Counter(); walls = []
for f in sorted(glob.glob("$OUT/per_instance_*.json")):
    try:
        d = json.load(open(f))
        for p in d.get('per_instance', []):
            c[p.get('cli_normalized','?')] += 1
            if p.get('wall_s'): walls.append(float(p['wall_s']))
            break
    except: pass
mw = sum(walls)/max(len(walls),1) if walls else 0
print(f"  RESULT $bench: {dict(c)}  mean_wall={mw:.1f}s")
EOF
}

# Order: smallest first
run_bench "cgan_2023" "$(seq -s, 0 20)" 300
run_bench "safenlp_2024" "$(seq -s, 0 29)" 60
run_bench "cora_2024" "$(seq -s, 0 29)" 300

echo "ALL DONE: $(date)" | tee -a "$LOG"
echo "ROOT=$ROOT"
