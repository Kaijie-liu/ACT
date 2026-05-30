#!/bin/bash
set -u
export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/data1/Kane/ACT/audit_results/nn4sys_gather_smoke_${STAMP}"
mkdir -p "$ROOT"
echo "ROOT=$ROOT"

/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline.watchdog_runner \
    --benchmark nn4sys --instance-ids 0,1,2,137,150 \
    --wall-s 180 --startup-grace-s 8 --poll-interval-s 0.5 \
    --rss-cap-gb 16 --grace-kill-s 3 \
    --device cuda --dtype float64 \
    --out-dir "$ROOT" \
    --canonical-root /data1/Kane/data/vnncomp2025_benchmarks/benchmarks \
    > "$ROOT/d.log" 2>&1

/data1/Kane/miniconda3/envs/act-py312/bin/python <<EOF
import json, glob
from collections import Counter
c = Counter(); walls = []
for f in sorted(glob.glob("$ROOT/per_instance_*.json")):
    try:
        d = json.load(open(f))
        for p in d.get('per_instance', []):
            v = p.get('cli_normalized','?')
            c[v] += 1
            if p.get('wall_s'): walls.append(float(p['wall_s']))
            iid = p.get('official_instance_id', p.get('instance_index'))
            print(f"  iid={iid} -> {v}  wall={p.get('wall_s'):.1f}s")
            break
    except: pass
mw = sum(walls)/max(len(walls),1)
print(f"nn4sys smoke: {dict(c)} mean_wall={mw:.0f}s")
EOF
