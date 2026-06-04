#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/Kane/ACT/audit_results/soundness_affine_tail_$(date -u +%Y%m%dT%H%M%SZ)"
BENCH_ROOT="/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
PY="/data1/Kane/miniconda3/envs/act-py312/bin/python"
mkdir -p "$ROOT"
echo "$ROOT"

export PYTHONPATH=/data1/Kane/ACT
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

run_one() {
  local iid="$1"
  local od="$ROOT/iid_${iid}"
  local sd="$od/snap"
  mkdir -p "$sd"
  ACT_HZ_R05_AFFINE_INHERIT=1 \
  ACT_HZ_CONV_FALLBACK_SAFE=1 \
  ACT_HZ_GIRARD_PRESERVE_ROOT=1 \
  ACT_HZ_ENDCAP_SNAPSHOT_DIR="$sd" \
  ACT_HZ_ENDCAP_SNAPSHOT_KIND=FLATTEN \
  ACT_HZ_ENDCAP_ROOT_ONLY=1 \
  "$PY" -m act.pipeline.watchdog_runner \
    --benchmark soundnessbench --instance-ids "$iid" \
    --wall-s 120 --device cuda --dtype float64 \
    --out-dir "$od/hz" --canonical-root "$BENCH_ROOT" \
    > "$od/hz.log" 2>&1 || true
  local snap
  snap="$(ls "$sd"/L*_FLATTEN.pkl 2>/dev/null | head -1 || true)"
  if [[ -n "$snap" ]]; then
    "$PY" research/generic_affine_tail_feas.py \
      --snapshot "$snap" \
      --onnx "$BENCH_ROOT/soundnessbench/onnx/model.onnx" \
      --vnnlib "$BENCH_ROOT/soundnessbench/vnnlib/model_${iid}.vnnlib" \
      --out "$od/affine_tail.json" \
      --time-limit-s 10 \
      > "$od/affine_tail.log" 2>&1 || true
  fi
}

export -f run_one
export ROOT BENCH_ROOT PY

printf "%s\n" $(seq 0 49) | xargs -n 1 -P "${SOUND_AFFINE_JOBS:-6}" bash -lc 'run_one "$0"'

"$PY" - <<'PY' "$ROOT"
import json, glob, sys
from collections import Counter
root = sys.argv[1]
c = Counter()
fals = []
missing = []
for iid in range(50):
    p = f"{root}/iid_{iid}/affine_tail.json"
    if not glob.glob(p):
        c["NO_RESULT"] += 1
        missing.append(iid)
        continue
    d = json.load(open(p))
    v = d.get("verdict", "?")
    c[v] += 1
    if v == "FAL":
        fals.append(iid)
print("SOUNDNESS_AFFINE_TAIL", dict(c), "FAL_iids", fals, "missing", missing)
PY
