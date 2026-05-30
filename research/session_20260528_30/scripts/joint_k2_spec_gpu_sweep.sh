#!/bin/bash
# Spec-aware joint K=2 GPU sweep — parallel 6 benches.
set -u
export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/data1/Kane/ACT/audit_results/joint_k2_spec_gpu_${STAMP}"
mkdir -p "$ROOT"
LOG="$ROOT/MAIN.log"
echo "Spec-aware joint K=2 GPU sweep started: $(date)" | tee "$LOG"

spawn_run() {
    local bench=$1; local iids=$2; local wall=$3
    local OUT="$ROOT/${bench}"
    mkdir -p "$OUT"
    (env ACT_HZ_JOINT_K2=1 ACT_HZ_JOINT_K2_SPEC=1 \
         ACT_HZ_JOINT_K2_MAX_PAIRS=32 ACT_HZ_JOINT_K2_MIN_COSSIM=0.3 \
         ACT_HZ_JOINT_K2_SPEC_TOPK=8 ACT_HZ_JOINT_K2_LP_TIMEOUT_S=2.0 \
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
    echo "  spawned $bench PID=$!" | tee -a "$LOG"
}

echo "=== Wave 1: 6 benches parallel (joint_k2 + spec-aware) ===" | tee -a "$LOG"
spawn_run "dist_shift_2023" "0,7,14,21,28,35,42,49,56,63" 180
spawn_run "traffic_signs_recognition_2023" "0,5,10,15,20,25,30,35,40" 180
spawn_run "soundnessbench" "0,5,10,15,20,25,30,35,40,45" 120
spawn_run "yolo_2023" "0,7,14,21,28,35,42,49,56,63" 180
spawn_run "tinyimagenet_2024" "0,40,80,120,160" 240
spawn_run "cifar100_2024" "0,40,80,120,160" 240
wait
echo "=== Wave 1 DONE: $(date) ===" | tee -a "$LOG"

# Synthesis
echo "" | tee -a "$LOG"
echo "==== SYNTHESIS ====" | tee -a "$LOG"
/data1/Kane/miniconda3/envs/act-py312/bin/python <<EOF | tee -a "$LOG"
import json, glob, os
from collections import defaultdict, Counter
root = "$ROOT"
# Compare against baseline from previous sweep
prev_root = "/data1/Kane/ACT/audit_results/joint_k2_gpu_20260528T094001Z"

def gather(bench, mode_root):
    c = Counter(); walls = []
    for f in sorted(glob.glob(os.path.join(mode_root, 'per_instance_*.json'))):
        try:
            d = json.load(open(f))
            for p in d.get('per_instance', []):
                v = p.get('cli_normalized','?')
                c[v] += 1
                if p.get('wall_s'): walls.append(float(p['wall_s']))
                break
        except: pass
    mw = sum(walls)/max(len(walls),1) if walls else 0
    return c, mw

benches = ['cifar100_2024','dist_shift_2023','soundnessbench','tinyimagenet_2024','traffic_signs_recognition_2023','yolo_2023']
print(f"{'Benchmark':35s} {'mode':12s} {'V':>3s} {'A':>3s} {'U':>3s} {'T+R':>4s} {'n':>3s} {'mean_wall':>10s}")
print('-'*90)
total_baseline = 0; total_spec = 0
for bench in benches:
    base_dir = os.path.join(prev_root, f"{bench}_baseline")
    spec_dir = os.path.join(root, bench)
    cb, mb = gather(bench, base_dir)
    cs, ms = gather(bench, spec_dir)
    Vb,Ab = cb.get('CERTIFIED',0), cb.get('FALSIFIED',0)
    Vs,As = cs.get('CERTIFIED',0), cs.get('FALSIFIED',0)
    Ub = cb.get('UNKNOWN',0); Us = cs.get('UNKNOWN',0)
    Tb = cb.get('UNKNOWN_TIMEOUT',0)+cb.get('UNKNOWN_RESOURCE_LIMIT',0); Ts = cs.get('UNKNOWN_TIMEOUT',0)+cs.get('UNKNOWN_RESOURCE_LIMIT',0)
    nb, ns = sum(cb.values()), sum(cs.values())
    print(f"{bench:35s} {'baseline':12s} {Vb:>3} {Ab:>3} {Ub:>3} {Tb:>4} {nb:>3} {mb:>9.1f}s")
    print(f"{bench:35s} {'spec_k2':12s} {Vs:>3} {As:>3} {Us:>3} {Ts:>4} {ns:>3} {ms:>9.1f}s")
    delta = (Vs+As) - (Vb+Ab)
    marker = "⭐ LIFT" if delta > 0 else ("✗ DROP" if delta < 0 else "  flat")
    print(f"  {marker} V+A delta: {Vb+Ab} → {Vs+As} ({delta:+d})")
    total_baseline += Vb+Ab; total_spec += Vs+As
print()
print(f"  TOTAL: baseline V+A={total_baseline} → spec_k2 V+A={total_spec} ({total_spec-total_baseline:+d})")
EOF
echo "ALL DONE: $(date)" | tee -a "$LOG"
echo "ROOT=$ROOT"
