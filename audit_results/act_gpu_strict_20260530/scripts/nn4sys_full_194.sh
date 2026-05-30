#!/bin/bash
set -u
export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/data1/Kane/ACT/audit_results/nn4sys_gather_full_${STAMP}"
mkdir -p "$ROOT"
LOG="$ROOT/MAIN.log"
echo "nn4sys full 194 with exact gather/slice: $(date)" | tee "$LOG"

spawn_batch() {
    local name=$1; local iids=$2; local wall=$3
    local OUT="$ROOT/$name"
    mkdir -p "$OUT"
    (PYTHONPATH=/data1/Kane/ACT \
     ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks \
     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /data1/Kane/miniconda3/envs/act-py312/bin/python \
        -m act.pipeline.watchdog_runner \
        --benchmark nn4sys --instance-ids "$iids" \
        --wall-s "$wall" --startup-grace-s 8 --poll-interval-s 0.5 \
        --rss-cap-gb 20 --grace-kill-s 3 \
        --device cuda --dtype float64 \
        --out-dir "$OUT" \
        --canonical-root /data1/Kane/data/vnncomp2025_benchmarks/benchmarks \
        > "$OUT/d.log" 2>&1) &
    echo "  spawned $name PID=$!" | tee -a "$LOG"
}

# Split 194 into 4 batches of ~48-49
spawn_batch b0_48 "$(seq -s, 0 48)" 180
spawn_batch b49_96 "$(seq -s, 49 96)" 180
spawn_batch b97_144 "$(seq -s, 97 144)" 180
spawn_batch b145_193 "$(seq -s, 145 193)" 180

wait
/data1/Kane/miniconda3/envs/act-py312/bin/python <<EOF | tee -a "$LOG"
import json, glob, csv
from collections import Counter
# Cross-reference with r93
r93 = {}
with open("/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/CONSOLIDATED_RESULTS/nn4sys/per_instance.csv") as f:
    for row in csv.DictReader(f):
        if row['source'] == 'gpu_full':
            r93[int(row['iid'])] = row['verdict']
my = {}
for f in sorted(glob.glob("$ROOT/*/per_instance_*.json")):
    try:
        d = json.load(open(f))
        for p in d.get('per_instance', []):
            iid = int(p.get('official_instance_id', p.get('instance_index')))
            my[iid] = p.get('cli_normalized', '?')
            break
    except: pass

new_cert = []; new_fal = []; lost = []
for iid in range(194):
    r = r93.get(iid, 'NOT_IN_R93')
    m = my.get(iid, 'NOT_RUN')
    if r not in ('CERTIFIED', 'FALSIFIED') and m == 'CERTIFIED':
        new_cert.append(iid)
    elif r not in ('CERTIFIED', 'FALSIFIED') and m == 'FALSIFIED':
        new_fal.append(iid)
    elif r in ('CERTIFIED', 'FALSIFIED') and m not in ('CERTIFIED', 'FALSIFIED'):
        lost.append((iid, r, m))

c = Counter(my.values())
print(f"nn4sys FULL: n={sum(c.values())}/194  {dict(c)}")
print(f"  r93 baseline: 4 CERT")
print(f"  NEW CERT: {len(new_cert)} iids: {new_cert[:20]}{'...' if len(new_cert) > 20 else ''}")
print(f"  NEW FAL:  {len(new_fal)} iids: {new_fal[:20]}{'...' if len(new_fal) > 20 else ''}")
print(f"  LOST:     {len(lost)} iids: {lost[:5]}")
print(f"  NET DELTA: +{len(new_cert) + len(new_fal) - len(lost)}")
EOF
echo "DONE: $(date)" | tee -a "$LOG"
echo "ROOT=$ROOT"
