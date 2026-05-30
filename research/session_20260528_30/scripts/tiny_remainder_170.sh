#!/bin/bash
# tinyimagenet 30-199 (170 remaining) in 4 parallel batches
# Goal: validate the 1/30 FAL rate from sample (iid 6)
set -u
export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/data1/Kane/ACT/audit_results/tiny_remainder_${STAMP}"
mkdir -p "$ROOT"
LOG="$ROOT/MAIN.log"
echo "tinyimagenet 30-199 sweep started: $(date)" | tee "$LOG"

spawn_batch() {
    local batch_name=$1; local iids=$2; local wall=$3
    local OUT="$ROOT/${batch_name}"
    mkdir -p "$OUT"
    (PYTHONPATH=/data1/Kane/ACT \
     ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks \
     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /data1/Kane/miniconda3/envs/act-py312/bin/python \
        -m act.pipeline.watchdog_runner \
        --benchmark tinyimagenet_2024 --instance-ids "$iids" \
        --wall-s "$wall" --startup-grace-s 8 --poll-interval-s 0.5 \
        --rss-cap-gb 8 --grace-kill-s 3 \
        --device cuda --dtype float64 \
        --out-dir "$OUT" \
        --canonical-root /data1/Kane/data/vnncomp2025_benchmarks/benchmarks \
        > "$OUT/d.log" 2>&1) &
    echo "  spawned $batch_name PID=$! n_iids=$(echo $iids | tr ',' '\n' | wc -l)" | tee -a "$LOG"
}

# Split 30-199 into 4 batches
spawn_batch "ba_30_72" "$(seq -s, 30 72)" 180
spawn_batch "bb_73_115" "$(seq -s, 73 115)" 180
spawn_batch "bc_116_157" "$(seq -s, 116 157)" 180
spawn_batch "bd_158_199" "$(seq -s, 158 199)" 180

wait

/data1/Kane/miniconda3/envs/act-py312/bin/python <<EOF | tee -a "$LOG"
import json, glob
from collections import Counter
total = Counter()
fal_iids = []; cert_iids = []
for f in sorted(glob.glob("$ROOT/*/per_instance_*.json")):
    try:
        d = json.load(open(f))
        for p in d.get('per_instance', []):
            v = p.get('cli_normalized','?')
            total[v] += 1
            iid = p.get('official_instance_id', p.get('instance_index'))
            if v == 'CERTIFIED': cert_iids.append(iid)
            elif v == 'FALSIFIED': fal_iids.append(iid)
            break
    except: pass
n = sum(total.values())
V = total.get('CERTIFIED',0); A = total.get('FALSIFIED',0)
print(f"tinyimagenet 30-199: n={n}/170  V={V} A={A} decided={V+A}")
print(f"  verdicts: {dict(total)}")
if cert_iids: print(f"  CERT iids: {cert_iids}")
if fal_iids:  print(f"  FAL  iids: {fal_iids}")
EOF
echo "DONE: $(date)" | tee -a "$LOG"
echo "ROOT=$ROOT"
