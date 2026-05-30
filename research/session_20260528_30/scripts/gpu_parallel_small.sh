#!/bin/bash
# Parallel GPU sweep for SMALL benchmarks (excludes vggnet16 which eats too much memory)
# Each benchmark spawn its own process. GPU has 96 GB, each fits 5-15 GB.
set -u
export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/data1/Kane/ACT/audit_results/d_gpu_parallel_${STAMP}"
mkdir -p "$ROOT"
echo "GPU parallel small-bench sweep started: $(date)" > "$ROOT/MAIN.log"

# Spawn function: (bench, iids, wall, mode_d_filter)
spawn_run() {
    local bench=$1
    local iids=$2
    local wall=$3
    local mode=$4   # baseline or d_filter
    local OUT="$ROOT/${bench}_${mode}"
    mkdir -p "$OUT"
    if [ "$mode" = "d_filter" ]; then
        ENV="ACT_HZ_DENSE_PEE_REDUNDANCY=1"
    else
        ENV=""
    fi
    (env $ENV \
        PYTHONPATH=/data1/Kane/ACT \
        ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        /data1/Kane/miniconda3/envs/act-py312/bin/python \
        -m act.pipeline.watchdog_runner \
        --benchmark "$bench" --instance-ids "$iids" \
        --wall-s "$wall" --startup-grace-s 8 --poll-interval-s 0.5 \
        --rss-cap-gb 80 --grace-kill-s 3 \
        --device cuda --dtype float64 \
        --out-dir "$OUT" \
        --canonical-root /data1/Kane/data/vnncomp2025_benchmarks/benchmarks \
        > "$OUT/d.log" 2>&1) &
    echo "  spawned $bench $mode PID=$!" | tee -a "$ROOT/MAIN.log"
}

# Wave 1: small benchmarks in parallel, baseline first
echo "=== Wave 1: baselines in parallel ===" | tee -a "$ROOT/MAIN.log"
spawn_run "dist_shift_2023" "0,7,14,21,28,35,42,49,56,63" 180 "baseline"
spawn_run "traffic_signs_recognition_2023" "0,5,10,15,20,25,30,35,40" 180 "baseline"
spawn_run "soundnessbench" "0,5,10,15,20,25,30,35,40,45" 120 "baseline"
spawn_run "yolo_2023" "0,7,14,21,28,35,42,49,56,63" 180 "baseline"
spawn_run "tinyimagenet_2024" "0,40,80,120,160" 240 "baseline"
spawn_run "cifar100_2024" "0,40,80,120,160" 240 "baseline"

wait
echo "=== Wave 1 complete: $(date) ===" | tee -a "$ROOT/MAIN.log"

# Wave 2: d_filter in parallel
echo "=== Wave 2: d_filter in parallel ===" | tee -a "$ROOT/MAIN.log"
spawn_run "dist_shift_2023" "0,7,14,21,28,35,42,49,56,63" 180 "d_filter"
spawn_run "traffic_signs_recognition_2023" "0,5,10,15,20,25,30,35,40" 180 "d_filter"
spawn_run "soundnessbench" "0,5,10,15,20,25,30,35,40,45" 120 "d_filter"
spawn_run "yolo_2023" "0,7,14,21,28,35,42,49,56,63" 180 "d_filter"
spawn_run "tinyimagenet_2024" "0,40,80,120,160" 240 "d_filter"
spawn_run "cifar100_2024" "0,40,80,120,160" 240 "d_filter"

wait
echo "=== Wave 2 complete: $(date) ===" | tee -a "$ROOT/MAIN.log"

# Synthesis
echo "" | tee -a "$ROOT/MAIN.log"
echo "==== SYNTHESIS ====" | tee -a "$ROOT/MAIN.log"
/data1/Kane/miniconda3/envs/act-py312/bin/python <<EOF | tee -a "$ROOT/MAIN.log"
import json, glob, os
from collections import defaultdict
root = "$ROOT"
results = defaultdict(dict)
for sub in sorted(os.listdir(root)):
    full = os.path.join(root, sub)
    if not os.path.isdir(full): continue
    if '_baseline' in sub:
        bench = sub.replace('_baseline','')
        mode = 'baseline'
    elif '_d_filter' in sub:
        bench = sub.replace('_d_filter','')
        mode = 'd_filter'
    else: continue
    fs = sorted([f for f in glob.glob(full + '/per_instance_*.json') if 'watchdog' not in f])
    cert=fal=unk=to=rss=err=0; walls=[]
    for fname in fs:
        try:
            d = json.load(open(fname))
            for p in d.get('per_instance', []):
                v = p.get('cli_normalized','?')
                if v=='CERTIFIED': cert+=1
                elif v=='FALSIFIED': fal+=1
                elif v=='UNKNOWN': unk+=1
                elif v=='UNKNOWN_TIMEOUT': to+=1
                elif v=='UNKNOWN_RESOURCE_LIMIT': rss+=1
                else: err+=1
                if p.get('wall_s'): walls.append(float(p['wall_s']))
                break
        except: pass
    mw = sum(walls)/max(len(walls),1)
    results[bench][mode] = (cert, fal, unk, to, rss, err, mw, len(fs))

print(f"{'Benchmark':35s} {'mode':10s} {'V':>3s} {'A':>3s} {'U':>3s} {'TO':>3s} {'RSS':>3s} {'E':>2s} {'n':>3s} {'mean_wall':>10s}")
for bench in sorted(results.keys()):
    for mode in ('baseline', 'd_filter'):
        if mode not in results[bench]: continue
        r = results[bench][mode]
        print(f"{bench:35s} {mode:10s} {r[0]:>3} {r[1]:>3} {r[2]:>3} {r[3]:>3} {r[4]:>3} {r[5]:>2} {r[7]:>3} {r[6]:>9.1f}s")
print()
print("D vs baseline winners (V+A):")
for bench in sorted(results.keys()):
    if 'baseline' in results[bench] and 'd_filter' in results[bench]:
        b_va = results[bench]['baseline'][0] + results[bench]['baseline'][1]
        d_va = results[bench]['d_filter'][0] + results[bench]['d_filter'][1]
        if d_va > b_va:
            print(f"  ⭐ {bench}: baseline V+A={b_va} → D V+A={d_va} (+{d_va-b_va})")
        elif d_va < b_va:
            print(f"  ✗ {bench}: REGRESSION baseline V+A={b_va} → D V+A={d_va}")
EOF
echo "" | tee -a "$ROOT/MAIN.log"
echo "ALL DONE: $(date)" | tee -a "$ROOT/MAIN.log"
