#!/bin/bash
# Clean canonical full sweep — §9 stabilization per advisor 2026-06-04.
#
# Runs the production HZ pipeline on the 5 target benchmarks under
# canonical-root. Post-attaches provenance hashes (canonical root +
# instances.csv sha256 + onnx sha256 + vnnlib sha256) to every
# per-instance receipt. Aggregates a consolidated table at the end.
#
# Principles preserved: no CROWN/backward/autograd/Gurobi/B&B/fallback/
# random/PGD. All env knobs match production defaults; the script does
# NOT override A++ closed-form or topK witness sidecars (they are
# closed-negative per atlas v3).
#
# Wall estimate: 5 benchmarks × 783 total iids × ~15-30s/iid ≈ 3-7 hours.
set -u

ACT_ROOT=/data1/Kane/ACT
PY=/data1/Kane/miniconda3/envs/act-py312/bin/python
BENCH_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT:-$ACT_ROOT/audit_results/clean_canonical_sweep_${STAMP}}
mkdir -p "$ROOT"
LOG="$ROOT/MAIN.log"

export PYTHONPATH=$ACT_ROOT
export ACT_VNNLIB_ROOT=$BENCH_ROOT
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Wall budget per iid (seconds). Per-benchmark overrides below.
WALL_DEFAULT=300
RSS_CAP_GB_DEFAULT=32

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "ROOT: $ROOT"
log "canonical_root: $BENCH_ROOT"
log "python: $PY"

run_benchmark() {
  local bench=$1
  local n_iids=$2
  local wall=$3
  local rss_gb=$4
  log "=== ${bench} (n=${n_iids}, wall=${wall}s, rss=${rss_gb}GB) ==="
  local out_dir="$ROOT/${bench}"
  mkdir -p "$out_dir"
  local IDS
  IDS=$(/data1/Kane/miniconda3/envs/act-py312/bin/python -c "
import sys
sys.path.insert(0, '/data1/Kane/ACT')
from research.canonical_provenance import canonical_instances_rows
rows = canonical_instances_rows('${bench}')
print(','.join(str(i) for i in range(len(rows))))")
  if [ -z "$IDS" ]; then
    log "ERROR: no iids resolved for ${bench}"
    return 1
  fi
  log "running ${bench} all ${n_iids} canonical iids..."
  set +e
  $PY -m act.pipeline.watchdog_runner \
    --benchmark "${bench}" \
    --instance-ids "${IDS}" \
    --wall-s "${wall}" \
    --device cuda --dtype float64 \
    --rss-cap-gb "${rss_gb}" \
    --out-dir "${out_dir}" \
    --canonical-root "${BENCH_ROOT}" \
    >> "$out_dir/run.log" 2>&1
  local rc=$?
  set -e
  log "${bench} subprocess rc=${rc}"
}

# Per-benchmark walls / rss caps. Conservative defaults; overrides only
# where a benchmark is known to need different budgets.
run_benchmark cifar100_2024       200 240 32
run_benchmark tinyimagenet_2024   200 300 40
run_benchmark cctsdb_yolo_2023     39 240 32
run_benchmark nn4sys              194 120 16
run_benchmark malbeware           150 120 16

log "=== aggregating provenance + verdict counts ==="
$PY /data1/Kane/ACT/research/scripts/clean_canonical_sweep_rollup.py "$ROOT" 2>&1 | tee -a "$LOG"

log "DONE. ROOT=$ROOT"
