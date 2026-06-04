#!/bin/bash
# CIFAR-only rerun of the clean canonical sweep with ACT_HZ_TOPK_RIVAL_WITNESS=5
# set explicitly. The 2026-06-04 main sweep produced 0 FAL on cifar (200/200
# UNKNOWN) because the witness sidecar didn't fire under the implicit
# auto-set of ACT_HZ_CIFAR_ENDCAP_WITNESS=1 — gate1 had worked with the
# explicit topK knob, so this rerun sets it explicitly.
#
# Targets the same canonical 200 cifar100_2024 iids, same wall budget, but
# guarantees the witness sidecar fires. Expected: +15 FAL per the production
# baseline.
set -u

ACT_ROOT=/data1/Kane/ACT
PY=/data1/Kane/miniconda3/envs/act-py312/bin/python
BENCH_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT:-$ACT_ROOT/audit_results/clean_canonical_sweep_cifar_rerun_${STAMP}}
mkdir -p "$ROOT"
LOG="$ROOT/MAIN.log"

export PYTHONPATH=$ACT_ROOT
export ACT_VNNLIB_ROOT=$BENCH_ROOT
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export ACT_HZ_TOPK_RIVAL_WITNESS=5
export ACT_HZ_CIFAR_ENDCAP_WITNESS=1

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
log "ROOT: $ROOT"
log "ACT_HZ_TOPK_RIVAL_WITNESS=${ACT_HZ_TOPK_RIVAL_WITNESS}"
log "ACT_HZ_CIFAR_ENDCAP_WITNESS=${ACT_HZ_CIFAR_ENDCAP_WITNESS}"

bench=cifar100_2024
out_dir="$ROOT/${bench}"
mkdir -p "$out_dir"
IDS=$($PY -c "
import sys; sys.path.insert(0, '/data1/Kane/ACT')
from research.canonical_provenance import canonical_instances_rows
rows = canonical_instances_rows('${bench}')
print(','.join(str(i) for i in range(len(rows))))")

log "running ${bench} ${IDS//,/ } iids..."
$PY -m act.pipeline.watchdog_runner \
  --benchmark "${bench}" \
  --instance-ids "${IDS}" \
  --wall-s 240 \
  --device cuda --dtype float64 \
  --rss-cap-gb 32 \
  --out-dir "${out_dir}" \
  --canonical-root "${BENCH_ROOT}" \
  >> "$out_dir/run.log" 2>&1
log "${bench} subprocess rc=$?"

log "=== aggregating ==="
$PY $ACT_ROOT/research/scripts/clean_canonical_sweep_rollup.py "$ROOT" 2>&1 | tee -a "$LOG"
log "DONE. ROOT=$ROOT"
