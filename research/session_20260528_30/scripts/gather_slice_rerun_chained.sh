#!/bin/bash
# After nn4sys 194 finishes, chain rerun on ml4acopf / collins_aero / lsnc_relu
# to harvest gather/slice exact gains.
set -u
export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/data1/Kane/ACT/audit_results/gather_slice_chain_${STAMP}"
mkdir -p "$ROOT"
LOG="$ROOT/MAIN.log"
echo "Gather/slice chain rerun: $(date)" | tee "$LOG"

spawn() {
    local bench=$1; local iids=$2; local wall=$3; local rss=$4
    local OUT="$ROOT/$bench"
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
    echo "  spawned $bench PID=$!" | tee -a "$LOG"
}

# ml4acopf rerun full 69 (was +20 CERT, might be more now)
spawn ml4acopf_2024 "$(seq -s, 0 68)" 180 12

# lsnc_relu full 80 (was 80 ERR/UNKNOWN)
spawn lsnc_relu "$(seq -s, 0 79)" 120 8

# collins_aerospace_benchmark 6 (was all ERR)
spawn collins_aerospace_benchmark "$(seq -s, 0 5)" 300 16

# Also retry safenlp_2024 sample 100 (no gather/slice but env bridge may help on different iids)
spawn safenlp_2024 "$(seq -s, 30 129)" 60 8

# Also retry tllverifybench full 32 (small, dense)
spawn tllverifybench_2023 "$(seq -s, 0 31)" 120 8

wait
echo "==== Synthesis ====" | tee -a "$LOG"
/data1/Kane/miniconda3/envs/act-py312/bin/python <<EOF | tee -a "$LOG"
import json, glob, csv, os
from collections import Counter

r93_root = "/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/CONSOLIDATED_RESULTS"

def load_r93(bench):
    r93 = {}
    p = os.path.join(r93_root, bench, 'per_instance.csv')
    if not os.path.exists(p): return r93
    with open(p) as f:
        for row in csv.DictReader(f):
            if row['source'] == 'gpu_full':
                r93[int(row['iid'])] = row['verdict']
    return r93

root = "$ROOT"
for bench in ['ml4acopf_2024', 'lsnc_relu', 'collins_aerospace_benchmark', 'safenlp_2024', 'tllverifybench_2023']:
    r93 = load_r93(bench)
    my = {}
    for f in sorted(glob.glob(os.path.join(root, bench, 'per_instance_*.json'))):
        try:
            d = json.load(open(f))
            for p in d.get('per_instance', []):
                iid = int(p.get('official_instance_id', p.get('instance_index')))
                my[iid] = p.get('cli_normalized', '?')
                break
        except: pass
    new_cert = []; new_fal = []; lost = []
    for iid, m in my.items():
        r = r93.get(iid, 'NOT_IN_R93')
        if r not in ('CERTIFIED', 'FALSIFIED') and m == 'CERTIFIED':
            new_cert.append(iid)
        elif r not in ('CERTIFIED', 'FALSIFIED') and m == 'FALSIFIED':
            new_fal.append(iid)
        elif r in ('CERTIFIED', 'FALSIFIED') and m not in ('CERTIFIED', 'FALSIFIED'):
            lost.append((iid, r, m))
    c = Counter(my.values())
    marker = "⭐" if len(new_cert) + len(new_fal) > 0 else ""
    print(f"  {bench:35s} n={sum(c.values()):>3}  {dict(c)}")
    print(f"  {'':35s}    NEW CERT={len(new_cert)} NEW FAL={len(new_fal)} LOST={len(lost)} {marker}")
    if new_cert: print(f"      new cert iids: {sorted(new_cert)[:15]}")
    if new_fal:  print(f"      new fal  iids: {sorted(new_fal)[:15]}")
EOF
echo "DONE: $(date)" | tee -a "$LOG"
echo "ROOT=$ROOT"
