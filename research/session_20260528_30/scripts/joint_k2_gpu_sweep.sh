#!/bin/bash
# Joint K=2 envelope GPU sweep — parallel across benchmarks (GPU has 96 GB, ~6 parallel).
# Compares baseline (knob OFF) vs joint_k2 (knob ON) on 7 GPU 0-verdict benchmarks.
set -u
export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/data1/Kane/ACT/audit_results/joint_k2_gpu_${STAMP}"
mkdir -p "$ROOT"
LOG="$ROOT/MAIN.log"
echo "Joint K=2 envelope GPU sweep started: $(date)" | tee "$LOG"

spawn_run() {
    local bench=$1; local iids=$2; local wall=$3; local mode=$4
    local OUT="$ROOT/${bench}_${mode}"
    mkdir -p "$OUT"
    local env_args=""
    if [ "$mode" = "joint_k2" ]; then
        env_args="ACT_HZ_JOINT_K2=1 ACT_HZ_JOINT_K2_MAX_PAIRS=32 ACT_HZ_JOINT_K2_MIN_COSSIM=0.3 ACT_HZ_JOINT_K2_LP_TIMEOUT_S=2.0"
    fi
    (env $env_args \
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
    echo "  spawned $bench $mode PID=$!" | tee -a "$LOG"
}

# Wave 1: SMALL benchmarks in parallel (6 in parallel, each ~5-15 GB GPU)
echo "=== Wave 1: small benchmarks parallel (baseline) ===" | tee -a "$LOG"
spawn_run "dist_shift_2023" "0,7,14,21,28,35,42,49,56,63" 180 "baseline"
spawn_run "traffic_signs_recognition_2023" "0,5,10,15,20,25,30,35,40" 180 "baseline"
spawn_run "soundnessbench" "0,5,10,15,20,25,30,35,40,45" 120 "baseline"
spawn_run "yolo_2023" "0,7,14,21,28,35,42,49,56,63" 180 "baseline"
spawn_run "tinyimagenet_2024" "0,40,80,120,160" 240 "baseline"
spawn_run "cifar100_2024" "0,40,80,120,160" 240 "baseline"
wait
echo "=== Wave 1 (baseline) DONE: $(date) ===" | tee -a "$LOG"

echo "=== Wave 2: small benchmarks parallel (joint_k2) ===" | tee -a "$LOG"
spawn_run "dist_shift_2023" "0,7,14,21,28,35,42,49,56,63" 180 "joint_k2"
spawn_run "traffic_signs_recognition_2023" "0,5,10,15,20,25,30,35,40" 180 "joint_k2"
spawn_run "soundnessbench" "0,5,10,15,20,25,30,35,40,45" 120 "joint_k2"
spawn_run "yolo_2023" "0,7,14,21,28,35,42,49,56,63" 180 "joint_k2"
spawn_run "tinyimagenet_2024" "0,40,80,120,160" 240 "joint_k2"
spawn_run "cifar100_2024" "0,40,80,120,160" 240 "joint_k2"
wait
echo "=== Wave 2 (joint_k2) DONE: $(date) ===" | tee -a "$LOG"

# Wave 3: vggnet16 sequentially (41 GB each — too heavy for parallel)
echo "=== Wave 3: vggnet16 sequential ===" | tee -a "$LOG"
for MODE in baseline joint_k2; do
    spawn_run "vggnet16_2022" "0,1,2,3,4" 600 "$MODE"
    wait
done
echo "=== Wave 3 DONE: $(date) ===" | tee -a "$LOG"

# Synthesis
echo "" | tee -a "$LOG"
echo "==== SYNTHESIS ====" | tee -a "$LOG"
/data1/Kane/miniconda3/envs/act-py312/bin/python <<EOF | tee -a "$LOG"
import json, glob, os
from collections import defaultdict, Counter
root = "$ROOT"
results = defaultdict(dict)
for sub in sorted(os.listdir(root)):
    full = os.path.join(root, sub)
    if not os.path.isdir(full): continue
    if '_baseline' in sub: bench, mode = sub.rsplit('_baseline', 1)[0], 'baseline'
    elif '_joint_k2' in sub: bench, mode = sub.rsplit('_joint_k2', 1)[0], 'joint_k2'
    else: continue
    c = Counter(); walls = []
    for f in sorted(glob.glob(os.path.join(full, 'per_instance_*.json'))):
        try:
            d = json.load(open(f))
            for p in d.get('per_instance', []):
                v = p.get('cli_normalized','?')
                c[v] += 1
                if p.get('wall_s'): walls.append(float(p['wall_s']))
                break
        except: pass
    mw = sum(walls)/max(len(walls),1)
    n = sum(c.values())
    results[bench][mode] = (c.get('CERTIFIED',0), c.get('FALSIFIED',0),
                              c.get('UNKNOWN',0),
                              c.get('UNKNOWN_TIMEOUT',0) + c.get('UNKNOWN_RESOURCE_LIMIT',0),
                              n, mw)

print(f"{'Benchmark':35s} {'mode':10s} {'V':>3s} {'A':>3s} {'U':>3s} {'T+R':>4s} {'n':>3s} {'mean_wall':>10s}")
for bench in sorted(results.keys()):
    for mode in ('baseline', 'joint_k2'):
        if mode not in results[bench]: continue
        v, a, u, tr, n, mw = results[bench][mode]
        print(f"{bench:35s} {mode:10s} {v:>3} {a:>3} {u:>3} {tr:>4} {n:>3} {mw:>9.1f}s")
print()
print("V+A delta (joint_k2 - baseline):")
for bench in sorted(results.keys()):
    if 'baseline' in results[bench] and 'joint_k2' in results[bench]:
        bV, bA = results[bench]['baseline'][:2]
        kV, kA = results[bench]['joint_k2'][:2]
        delta = (kV + kA) - (bV + bA)
        marker = "⭐" if delta > 0 else ("✗" if delta < 0 else " ")
        print(f"  {marker} {bench}: baseline V+A={bV+bA} → joint_k2 V+A={kV+kA} ({delta:+d})")
EOF
echo "ALL DONE: $(date)" | tee -a "$LOG"
echo "ROOT=$ROOT"
