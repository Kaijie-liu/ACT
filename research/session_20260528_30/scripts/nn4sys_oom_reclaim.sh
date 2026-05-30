#!/bin/bash
set -u
export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/data1/Kane/ACT/audit_results/nn4sys_oom_reclaim_${STAMP}"
mkdir -p "$ROOT/a" "$ROOT/b"
echo "ROOT=$ROOT"

# RSS-limited iids from b145_193 — try with 50GB cap, 300s wall
# 146-159 + 169-170 = 16 iids
(/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline.watchdog_runner \
    --benchmark nn4sys --instance-ids 146,147,148,149,150,151,152,153 \
    --wall-s 300 --startup-grace-s 8 --poll-interval-s 0.5 \
    --rss-cap-gb 50 --grace-kill-s 3 --device cuda --dtype float64 \
    --out-dir "$ROOT/a" --canonical-root /data1/Kane/data/vnncomp2025_benchmarks/benchmarks \
    > "$ROOT/a/d.log" 2>&1) &
PA=$!
(/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline.watchdog_runner \
    --benchmark nn4sys --instance-ids 154,155,156,157,158,159,169,170 \
    --wall-s 300 --startup-grace-s 8 --poll-interval-s 0.5 \
    --rss-cap-gb 50 --grace-kill-s 3 --device cuda --dtype float64 \
    --out-dir "$ROOT/b" --canonical-root /data1/Kane/data/vnncomp2025_benchmarks/benchmarks \
    > "$ROOT/b/d.log" 2>&1) &
PB=$!
echo "spawned $PA $PB"
wait $PA $PB
/data1/Kane/miniconda3/envs/act-py312/bin/python <<EOF
import json, glob
from collections import Counter
c = Counter(); cert = []; fal = []
for f in sorted(glob.glob("$ROOT/*/per_instance_*.json")):
    try:
        d = json.load(open(f))
        for p in d.get('per_instance', []):
            v = p.get('cli_normalized','?')
            c[v] += 1
            iid = p.get('official_instance_id', p.get('instance_index'))
            if v == 'CERTIFIED': cert.append(iid)
            elif v == 'FALSIFIED': fal.append(iid)
            break
    except: pass
print(f"nn4sys OOM reclaim: n={sum(c.values())}/16  {dict(c)}")
print(f"  CERT: {sorted(cert)}  FAL: {sorted(fal)}")
EOF
echo "DONE $(date)"
