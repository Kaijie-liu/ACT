#!/bin/bash
# Resume cora 180 — only the 129 iids NOT covered in first batch
set -u
export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/data1/Kane/ACT/audit_results/cora_resume_${STAMP}"
mkdir -p "$ROOT"
LOG="$ROOT/MAIN.log"
echo "cora resume started: $(date)" | tee "$LOG"

spawn_batch() {
    local batch_name=$1; local iids=$2; local wall=$3
    local OUT="$ROOT/${batch_name}"
    mkdir -p "$OUT"
    (PYTHONPATH=/data1/Kane/ACT \
     ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks \
     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /data1/Kane/miniconda3/envs/act-py312/bin/python \
        -m act.pipeline.watchdog_runner \
        --benchmark cora_2024 --instance-ids "$iids" \
        --wall-s "$wall" --startup-grace-s 8 --poll-interval-s 0.5 \
        --rss-cap-gb 12 --grace-kill-s 3 \
        --device cuda --dtype float64 \
        --out-dir "$OUT" \
        --canonical-root /data1/Kane/data/vnncomp2025_benchmarks/benchmarks \
        > "$OUT/d.log" 2>&1) &
    echo "  spawned $batch_name PID=$! n_iids=$(echo $iids | tr ',' '\n' | wc -l)" | tee -a "$LOG"
}

# Missing: 19-44 (26), 61-89 (29), 98-134 (37), 143-179 (37) = 129 total
# Split into 4 batches for parallelism
spawn_batch "missing_a" "$(seq -s, 19 44)" 300
spawn_batch "missing_b" "$(seq -s, 61 89)" 300
spawn_batch "missing_c" "$(seq -s, 98 134)" 300
spawn_batch "missing_d" "$(seq -s, 143 179)" 300

wait

/data1/Kane/miniconda3/envs/act-py312/bin/python <<EOF | tee -a "$LOG"
import json, glob
from collections import Counter
total = Counter()
for f in sorted(glob.glob("$ROOT/*/per_instance_*.json")):
    try:
        d = json.load(open(f))
        for p in d.get('per_instance', []):
            total[p.get('cli_normalized','?')] += 1
            break
    except: pass
n = sum(total.values())
V = total.get('CERTIFIED',0); A = total.get('FALSIFIED',0)
print(f"cora RESUME: n={n}  V={V} A={A} decided={V+A}")
print(f"  verdicts: {dict(total)}")
EOF
echo "DONE: $(date)" | tee -a "$LOG"
echo "ROOT=$ROOT"
