#!/bin/bash
# cora_2024 full 180 instances — 4-way parallel batches
# Validate the 30-inst sample signal (10/30 decided vs baseline 20/180)
set -u
export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/data1/Kane/ACT/audit_results/cora_full180_${STAMP}"
mkdir -p "$ROOT"
LOG="$ROOT/MAIN.log"
echo "cora_2024 full 180 sweep started: $(date)" | tee "$LOG"

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

# 4-way split: 0-44, 45-89, 90-134, 135-179
spawn_batch "b0_44" "$(seq -s, 0 44)" 300
spawn_batch "b45_89" "$(seq -s, 45 89)" 300
spawn_batch "b90_134" "$(seq -s, 90 134)" 300
spawn_batch "b135_179" "$(seq -s, 135 179)" 300

echo "Waiting for 4 batches..." | tee -a "$LOG"
wait

# Synthesis
echo "" | tee -a "$LOG"
echo "==== FINAL ====" | tee -a "$LOG"
/data1/Kane/miniconda3/envs/act-py312/bin/python <<EOF | tee -a "$LOG"
import json, glob, os
from collections import Counter
total = Counter(); walls = []
for f in sorted(glob.glob("$ROOT/*/per_instance_*.json")):
    try:
        d = json.load(open(f))
        for p in d.get('per_instance', []):
            total[p.get('cli_normalized','?')] += 1
            if p.get('wall_s'): walls.append(float(p['wall_s']))
            break
    except: pass
n = sum(total.values())
V = total.get('CERTIFIED',0); A = total.get('FALSIFIED',0)
ERR = sum(v for k, v in total.items() if k.startswith('ERROR'))
mw = sum(walls)/max(len(walls),1) if walls else 0
print(f"cora_2024 FULL 180: n={n} V={V} A={A} ERR={ERR} mean_wall={mw:.0f}s")
print(f"  Verdicts: {dict(total)}")
print()
print(f"  r93 baseline (gpu_full):  16 CERT + 4 FAL = 20 decided / 180 (11.1%)")
print(f"  post-patch (this run):    {V} CERT + {A} FAL = {V+A} decided / {n} ({(V+A)/max(n,1)*100:.1f}%)")
delta = (V + A) - 20
marker = "⭐ LIFT" if delta > 0 else ("✗ DROP" if delta < 0 else "  flat")
print(f"  {marker}: net delta = {delta:+d}")
EOF
echo "DONE: $(date)" | tee -a "$LOG"
echo "ROOT=$ROOT"
