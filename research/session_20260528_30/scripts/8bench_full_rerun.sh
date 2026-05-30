#!/bin/bash
# Full rerun of 8 untested-this-session benchmarks with NEW code
# (gather/slice exact, sigmoid cap, upsample/convtranspose, env bridge, etc)
set -u
export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/data1/Kane/ACT/audit_results/eight_bench_rerun_${STAMP}"
mkdir -p "$ROOT"
LOG="$ROOT/MAIN.log"
echo "8-bench rerun: $(date)" | tee "$LOG"

spawn() {
    local bench=$1; local iids=$2; local wall=$3; local rss=$4; local label="${5:-default}"
    local OUT="$ROOT/${bench}_${label}"
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
    echo "  spawned $bench/$label PID=$!" | tee -a "$LOG"
}

# ── Wave 1: small benchmarks (in parallel, all 4 batches each where applicable) ──
echo "=== Wave 1: small benchmarks ===" | tee -a "$LOG"

# vggnet16: 18 inst — split into 2 batches of ~9 (each VGG inst is heavy)
spawn vggnet16_2022 "0,1,2,3,4,5,6,7,8" 300 16 "b0_8"
spawn vggnet16_2022 "9,10,11,12,13,14,15,16,17" 300 16 "b9_17"

# collins_rul_cnn: 62 inst — 4 batches
spawn collins_rul_cnn_2022 "$(seq -s, 0 15)" 180 8 "b0_15"
spawn collins_rul_cnn_2022 "$(seq -s, 16 30)" 180 8 "b16_30"
spawn collins_rul_cnn_2022 "$(seq -s, 31 46)" 180 8 "b31_46"
spawn collins_rul_cnn_2022 "$(seq -s, 47 61)" 180 8 "b47_61"

# linearizenn: 60 — 4 batches
spawn linearizenn_2024 "$(seq -s, 0 14)" 180 8 "b0_14"
spawn linearizenn_2024 "$(seq -s, 15 29)" 180 8 "b15_29"
spawn linearizenn_2024 "$(seq -s, 30 44)" 180 8 "b30_44"
spawn linearizenn_2024 "$(seq -s, 45 59)" 180 8 "b45_59"

# acasxu: 186 — 4 batches of ~47 (fast)
spawn acasxu_2023 "$(seq -s, 0 46)" 120 6 "b0_46"
spawn acasxu_2023 "$(seq -s, 47 92)" 120 6 "b47_92"
spawn acasxu_2023 "$(seq -s, 93 139)" 120 6 "b93_139"
spawn acasxu_2023 "$(seq -s, 140 185)" 120 6 "b140_185"

# malbeware: 150 — 4 batches  
spawn malbeware "$(seq -s, 0 37)" 120 6 "b0_37"
spawn malbeware "$(seq -s, 38 74)" 120 6 "b38_74"
spawn malbeware "$(seq -s, 75 112)" 120 6 "b75_112"
spawn malbeware "$(seq -s, 113 149)" 120 6 "b113_149"

# sat_relu smoke: 20 inst
spawn sat_relu "$(seq -s, 0 19)" 120 8 "smoke"

# cctsdb smoke: 10 inst
spawn cctsdb_yolo_2023 "$(seq -s, 0 9)" 120 8 "smoke"

# relusplitter: 220 inst — 4 batches of 55
spawn relusplitter "$(seq -s, 0 54)" 180 8 "b0_54"
spawn relusplitter "$(seq -s, 55 109)" 180 8 "b55_109"
spawn relusplitter "$(seq -s, 110 164)" 180 8 "b110_164"
spawn relusplitter "$(seq -s, 165 219)" 180 8 "b165_219"

echo "Total: $(echo "spawned" | wc -l) parallel spawns... actually launched: 24" | tee -a "$LOG"
echo "Waiting..." | tee -a "$LOG"
wait

# Final synthesis
echo "" | tee -a "$LOG"
echo "=== SYNTHESIS ===" | tee -a "$LOG"
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
for bench in ['vggnet16_2022','collins_rul_cnn_2022','linearizenn_2024','acasxu_2023','malbeware','sat_relu','cctsdb_yolo_2023','relusplitter']:
    r93 = load_r93(bench)
    rv = sum(1 for v in r93.values() if v == 'CERTIFIED')
    ra = sum(1 for v in r93.values() if v == 'FALSIFIED')
    my = {}
    for d in glob.glob(os.path.join(root, bench + '_*')):
        for f in sorted(glob.glob(os.path.join(d, 'per_instance_*.json'))):
            try:
                data = json.load(open(f))
                for p in data.get('per_instance', []):
                    iid = int(p.get('official_instance_id', p.get('instance_index')))
                    my[iid] = p.get('cli_normalized', '?')
                    break
            except: pass
    c = Counter(my.values())
    n = sum(c.values())
    mv = c.get('CERTIFIED', 0); ma = c.get('FALSIFIED', 0)
    new_cert = sum(1 for iid,v in my.items() if r93.get(iid,'?') not in ('CERTIFIED','FALSIFIED') and v == 'CERTIFIED')
    new_fal = sum(1 for iid,v in my.items() if r93.get(iid,'?') not in ('CERTIFIED','FALSIFIED') and v == 'FALSIFIED')
    lost = sum(1 for iid,v in my.items() if r93.get(iid,'?') in ('CERTIFIED','FALSIFIED') and v not in ('CERTIFIED','FALSIFIED'))
    marker = " ⭐" if (new_cert + new_fal) > 0 else ""
    print(f"  {bench:35s} n={n:>3}  V={mv:>3} A={ma:>3}  NEW: {new_cert}C+{new_fal}F  LOST: {lost}  delta=+{new_cert+new_fal-lost}{marker}")
EOF

echo "DONE: $(date)" | tee -a "$LOG"
echo "ROOT=$ROOT"
