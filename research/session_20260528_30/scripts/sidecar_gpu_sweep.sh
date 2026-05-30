#!/bin/bash
# Multi-corner sidecar sweep on GPU 0-verdict benchmarks
set -u
export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ACT_HZ_MULTI_CORNER_SIDECAR=1
export ACT_HZ_MULTI_CORNER_MAX=64

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/data1/Kane/ACT/audit_results/sidecar_multicorner_gpu_${STAMP}"
mkdir -p "$ROOT"
LOG="$ROOT/MAIN.log"
echo "Multi-corner sidecar GPU sweep started: $(date)" | tee "$LOG"

# Same configs as prior D-filter sweep for direct comparability
declare -a CONFIGS=(
    "dist_shift_2023|0,7,14,21,28,35,42,49,56,63|180"
    "soundnessbench|0,5,10,15,20,25,30,35,40,45|120"
    "yolo_2023|0,7,14,21,28,35,42,49,56,63|180"
    "traffic_signs_recognition_2023|0,5,10,15,20,25,30,35,40|180"
    "cifar100_2024|0,40,80,120,160|240"
    "tinyimagenet_2024|0,40,80,120,160|240"
    "vggnet16_2022|0,1,2,3,4|600"
)

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r bench iids wall <<< "$cfg"
    OUT="$ROOT/${bench}"
    mkdir -p "$OUT"
    echo "===> $bench iids=$iids wall=${wall}s @ $(date)" | tee -a "$LOG"
    /data1/Kane/miniconda3/envs/act-py312/bin/python \
        -m act.pipeline.watchdog_runner \
        --benchmark "$bench" --instance-ids "$iids" \
        --wall-s "$wall" --startup-grace-s 8 --poll-interval-s 0.5 \
        --rss-cap-gb 80 --grace-kill-s 3 \
        --device cuda --dtype float64 \
        --out-dir "$OUT" \
        --canonical-root /data1/Kane/data/vnncomp2025_benchmarks/benchmarks \
        > "$OUT/d.log" 2>&1
    /data1/Kane/miniconda3/envs/act-py312/bin/python <<EOF | tee -a "$LOG"
import json, glob
from collections import Counter
c = Counter(); promoted = 0; tried = 0
for f in sorted(glob.glob("$OUT/per_instance_*.json")):
    try:
        d = json.load(open(f))
        for p in d.get('per_instance', []):
            v = p.get('cli_normalized','?')
            c[v] += 1
            break
    except: pass
print(f"     {bench='$bench'} verdicts={dict(c)}")
EOF
done
echo "===> ALL DONE $(date)" | tee -a "$LOG"
