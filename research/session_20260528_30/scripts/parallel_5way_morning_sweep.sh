#!/bin/bash
# Parallel GPU sweep — 5 benches in parallel
# 1) cgan_2023 priority order (Fix#8/9 candidates first)
# 2) safenlp_2024 30-inst smoke
# 3) cora_2024 30-inst smoke  
# 4) metaroom_2023 singleton 44 iids (re-confirm overnight gain)
# 5) ml4acopf_2024 30-inst sample (re-confirm overnight +20 CERT)
set -u
export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/data1/Kane/ACT/audit_results/parallel_5way_${STAMP}"
mkdir -p "$ROOT"
LOG="$ROOT/MAIN.log"
echo "Started: $(date)" | tee "$LOG"

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
    echo "  spawned $bench PID=$! wall=${wall}s rss=${rss}GB" | tee -a "$LOG"
}

# Priority cgan order: ERR-candidates (18,19,20) first, then quick UNKNOWNs (12-15), then TIMEOUTs
spawn "cgan_2023" "18,19,20,12,13,14,15,0,1,2,3,4,5,6,7,8,9,10,11,16,17" 180 12

# safenlp 30 smoke (small networks, fast)
spawn "safenlp_2024" "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29" 60 8

# cora 30 smoke (GNN 784-dim)
spawn "cora_2024" "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29" 300 12

# metaroom singleton 44 (re-confirm overnight gain) — iids from overnight memory
spawn "metaroom_2023" "0,1,2,3,4,5,6,7,8,9,10,11,12,13" 60 8

# ml4acopf 30-instance sample (re-confirm overnight +20 CERT)
spawn "ml4acopf_2024" "0,3,6,9,12,15,17,18,20,21,24,27,30,33,36,37,39,42,45,48,50,51,54,57,60,63,66,68,1,2" 180 12

echo "Waiting for all 5 to finish..." | tee -a "$LOG"
wait

# Synthesis
echo "" | tee -a "$LOG"
echo "==== SYNTHESIS ====" | tee -a "$LOG"
/data1/Kane/miniconda3/envs/act-py312/bin/python <<EOF | tee -a "$LOG"
import json, glob, os
from collections import Counter
root = "$ROOT"
for bench in ['cgan_2023','safenlp_2024','cora_2024','metaroom_2023','ml4acopf_2024']:
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
    ERR = c.get('ERROR',0) + sum(v for k, v in c.items() if k.startswith('ERROR'))
    marker = " ⭐" if (V + A) > 0 else ""
    print(f"  {bench:30s} n={n:>3} V={V:>3} A={A:>2} ERR={ERR:>2} {dict(c)} mean={mw:.0f}s{marker}")
EOF
echo "ALL DONE: $(date)" | tee -a "$LOG"
