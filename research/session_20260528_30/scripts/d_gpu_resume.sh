#!/bin/bash
# Resume GPU autosweep with longer walls for heavy benchmarks
set -u
export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/data1/Kane/ACT/audit_results/d_gpu_resume_${STAMP}"
mkdir -p "$ROOT"
LOG="$ROOT/MAIN.log"
echo "D GPU resume started $(date)" | tee "$LOG"

# Updated configs: longer walls for heavy benchmarks
declare -a CONFIGS=(
    "vggnet16_2022|0,1,2,3,4|600"
    "traffic_signs_recognition_2023|0,5,10,15,20,25,30,35,40|180"
    "soundnessbench|0,5,10,15,20,25,30,35,40,45|120"
    "dist_shift_2023|0,7,14,21,28,35,42,49,56,63|180"
    "yolo_2023|0,7,14,21,28,35,42,49,56,63|180"
    "tinyimagenet_2024|0,40,80,120,160|240"
    "cifar100_2024|0,40,80,120,160|240"
)

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r bench iids wall <<< "$cfg"
    for MODE in baseline d_filter; do
        case "$MODE" in
            baseline) unset ACT_HZ_DENSE_PEE_REDUNDANCY ;;
            d_filter) export ACT_HZ_DENSE_PEE_REDUNDANCY=1 ;;
        esac
        OUT="$ROOT/${bench}_${MODE}"
        mkdir -p "$OUT"
        echo "===> $bench $MODE iids=$iids wall=${wall}s @ $(date)" | tee -a "$LOG"
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
c = Counter(); cert_iids = []; fal_iids = []; walls = []
for f in sorted(glob.glob("$OUT/per_instance_*.json")):
    if 'watchdog' in f: continue
    d = json.load(open(f))
    for p in d.get('per_instance', []):
        v = p.get('cli_normalized', '?')
        c[v] += 1
        iid = p.get('official_instance_id', p.get('instance_index'))
        if v == 'CERTIFIED': cert_iids.append(iid)
        elif v == 'FALSIFIED': fal_iids.append(iid)
        if p.get('wall_s'): walls.append(float(p['wall_s']))
        break
mean_wall = sum(walls)/max(len(walls),1)
print(f"     bench=$bench mode=$MODE verdicts={dict(c)} mean_wall={mean_wall:.1f}s CERT={cert_iids} FAL={fal_iids}")
EOF
    done
done

echo "===> ALL DONE $(date)" | tee -a "$LOG"
