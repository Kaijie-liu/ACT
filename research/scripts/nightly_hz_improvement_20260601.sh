#!/bin/bash
# Seven-hour HZ improvement queue for 2026-06-01.
#
# Principles: no CROWN/backward/autograd/Gurobi/B&B/fallback/random sampling.
# Every experiment below uses ACT/HZ forward propagation, SciPy HiGHS LP
# subproblems, and strict ORT replay for any FALSIFIED receipt.
set -u

ACT_ROOT=/data1/Kane/ACT
PY=/data1/Kane/miniconda3/envs/act-py312/bin/python
BENCH_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT:-$ACT_ROOT/audit_results/nightly_hz_20260601_${STAMP}}
mkdir -p "$ROOT"
LOG="$ROOT/MAIN.log"

export PYTHONPATH=$ACT_ROOT
export ACT_VNNLIB_ROOT=$BENCH_ROOT
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

summarize_dir() {
  local label=$1
  local dir=$2
  "$PY" - "$label" "$dir" <<'PY' | tee -a "$LOG"
import json, glob, os, sys
from collections import Counter
label, root = sys.argv[1], sys.argv[2]
c = Counter()
cert = []
fal = []
for f in sorted(glob.glob(os.path.join(root, "**", "per_instance*.json"), recursive=True)):
    try:
        d = json.load(open(f))
        rows = d.get("per_instance", [])
        for p in rows:
            v = p.get("cli_normalized") or p.get("verdict") or p.get("status") or "?"
            iid = p.get("official_instance_id", p.get("instance_index"))
            c[v] += 1
            if v == "CERTIFIED":
                cert.append(iid)
            elif v == "FALSIFIED":
                fal.append(iid)
    except Exception:
        pass
print(f"SUMMARY {label}: {dict(c)}")
if cert:
    print(f"  CERT iids: {cert[:60]}")
if fal:
    print(f"  FAL  iids: {fal[:60]}")
PY
}

bad_iids_from_full() {
  local bench=$1
  "$PY" - "$bench" <<'PY'
import json, glob, sys
bench = sys.argv[1]
root = "/data1/Kane/ACT/audit_results/full_vnncomp_20260530T092003Z"
best = {}
rank = {"CERTIFIED": 0, "FALSIFIED": 0, "UNKNOWN": 1, "UNKNOWN_TIMEOUT": 2}
for f in glob.glob(f"{root}/{bench}_*/per_instance*.json") + glob.glob(f"{root}/{bench}*/per_instance*.json"):
    try:
        d = json.load(open(f))
        for p in d.get("per_instance", []):
            iid = p.get("official_instance_id", p.get("instance_index"))
            if iid is None:
                continue
            iid = int(iid)
            v = p.get("cli_normalized") or p.get("verdict") or p.get("status") or "?"
            if iid not in best or rank.get(v, 3) < rank.get(best[iid], 3):
                best[iid] = v
    except Exception:
        pass
bad = [iid for iid, v in sorted(best.items()) if v not in ("CERTIFIED", "FALSIFIED")]
print(",".join(map(str, bad)))
PY
}

run_watchdog() {
  local bench=$1
  local iids=$2
  local wall=$3
  local rss=$4
  local out=$5
  mkdir -p "$out"
  "$PY" -m act.pipeline.watchdog_runner \
    --benchmark "$bench" --instance-ids "$iids" \
    --wall-s "$wall" --startup-grace-s 8 --poll-interval-s 0.5 \
    --rss-cap-gb "$rss" --grace-kill-s 3 \
    --device cuda --dtype float64 \
    --out-dir "$out" \
    --canonical-root "$BENCH_ROOT" \
    > "$out/d.log" 2>&1
}

run_acasxu_dag_probe() {
  log "P1 ACASXu DAG probe: first 40 undecided/error instances."
  local bad
  bad=$(bad_iids_from_full acasxu_2023)
  "$PY" - "$bad" "$ROOT/acasxu_stage1_batches.txt" <<'PY'
import sys
iids = [int(x) for x in sys.argv[1].split(",") if x.strip()][:40]
out = sys.argv[2]
batches = [iids[i::4] for i in range(4)]
with open(out, "w") as f:
    for b in batches:
        f.write(",".join(map(str, b)) + "\n")
PY
  for b in 0 1 2 3; do
    local iids
    iids=$(sed -n "$((b+1))p" "$ROOT/acasxu_stage1_batches.txt")
    [ -z "$iids" ] && continue
    (
      export ACT_HZ_SMALL_DENSE_LP=smalldense_dag
      export ACT_HZ_SMALL_DENSE_LP_TIME_LIMIT_S=20
      export ACT_HZ_SMALL_DENSE_LP_REFINEMENT_PASSES=20
      run_watchdog acasxu_2023 "$iids" 220 8 "$ROOT/acasxu_dag_b$b"
    ) &
  done
  wait
  summarize_dir "acasxu_dag_stage1" "$ROOT/acasxu_dag_b0/.."
}

run_cifar_pass1() {
  local iid=$1
  local out="$ROOT/cifar_iid${iid}_p1"
  if [ -f "$out/diag.jsonl" ] && [ -f "$out/trace.json" ] && [ -f "$out/out_hz.npz" ]; then
    log "P2 pass1 reuse cifar iid=$iid"
    return 0
  fi
  mkdir -p "$out"
  log "P2 pass1 mine cifar iid=$iid"
  HYZOR_LARGE_CLS_EQ_LAYERS=10 \
  ACT_HZ_RELU_TRACE=1 \
  ACT_HZ_RELU_TRACE_DUMP_FILE="$out/trace.json" \
  ACT_HZ_OUTPUT_HZ_DUMP_FILE="$out/out_hz.npz" \
  ACT_HZ_PHANTOM_MARGIN_DIAG=1 \
  ACT_HZ_PHANTOM_MARGIN_OUT="$out/diag.jsonl" \
  ACT_HZ_PHANTOM_MARGIN_TIMEOUT_S=240 \
  ACT_HZ_PHANTOM_MARGIN_MAX_RIVALS=30 \
  run_watchdog cifar100_2024 "$iid" 720 24 "$out/run"
}

diag_is_promising() {
  local diag=$1
  "$PY" - "$diag" <<'PY'
import json, sys
try:
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    d = rows[-1]
    m = float(d.get("lp_phantom_margin_max", 1e9))
    loose = int(d.get("n_lp_loose", 999))
    # Pair cuts have only moved margins by O(0.1) so far; don't spend
    # overnight time on max margins above 1.8 or many loose rivals.
    print("1" if (m <= 1.8 and loose <= 6) else "0")
except Exception:
    print("0")
PY
}

target_rivals_from_diag() {
  local diag=$1
  "$PY" - "$diag" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
d = rows[-1]
ids = d.get("worst_rival_ids", [])
margins = d.get("worst_rival_margins", [])
pos = [int(i) for i, m in zip(ids, margins) if float(m) > 0][:3]
print(",".join(map(str, pos)))
PY
}

run_cifar_p2_grid_for_iid() {
  local iid=$1
  local p1="$ROOT/cifar_iid${iid}_p1"
  local diag="$p1/diag.jsonl"
  [ ! -f "$diag" ] && return 0
  [ "$(diag_is_promising "$diag")" != "1" ] && {
    log "P2 skip iid=$iid: not promising"
    return 0
  }
  local rivals
  rivals=$(target_rivals_from_diag "$diag")
  [ -z "$rivals" ] && {
    log "P2 skip iid=$iid: no positive rivals"
    return 0
  }
  log "P2 grid iid=$iid rivals=$rivals"
  for enc in eq_lagr_v8 triangle; do
    for budget in 6 10; do
      local cfg="$ROOT/cifar_iid${iid}_p2_${enc}_b${budget}_whitelist"
      if compgen -G "$cfg/run/per_instance*.json" > /dev/null; then
        log "P2 reuse existing result iid=$iid enc=$enc budget=$budget whitelist"
        summarize_dir "cifar_iid${iid}_${enc}_b${budget}_whitelist" "$cfg/run"
        continue
      fi
      mkdir -p "$cfg"
      "$PY" "$ACT_ROOT/research/two_pass_corr_cuts_selector.py" \
        --trace "$p1/trace.json" \
        --diag "$diag" \
        --out-hz "$p1/out_hz.npz" \
        --target-rivals "$rivals" \
        --top-binaries-per-layer 6 \
        --global-pair-budget "$budget" \
        --encoding-filter "$enc" \
        --output "$cfg/targets.json" \
        > "$cfg/selector.log" 2>&1 || continue
      (
        HYZOR_LARGE_CLS_EQ_LAYERS=10 \
        ACT_HZ_CORR_PAIR_CUTS=1 \
        ACT_HZ_CORR_PAIR_CUT_MAX_PAIRS="$budget" \
        ACT_HZ_CORR_PAIR_CUT_DIRS=8 \
        ACT_HZ_CORR_PAIR_CUT_TARGET_FILE="$cfg/targets.json" \
        run_watchdog cifar100_2024 "$iid" 420 28 "$cfg/run"
      )
      summarize_dir "cifar_iid${iid}_${enc}_b${budget}_whitelist" "$cfg/run"
    done
  done
}

run_cifar_dense_conv_queue() {
  log "P2 Dense-conv CIFAR queue: mine selected iids, then run targeted eq-vs-triangle grid only for near-misses."
  # Keep this focused. iids 0/8 are known near-ish; the rest fill coverage
  # from the still-0 dense-conv set without blindly sweeping all 200.
  local iids=(0 8 14 16 18 20 22 24 26 28)
  for iid in "${iids[@]}"; do
    run_cifar_pass1 "$iid"
    run_cifar_p2_grid_for_iid "$iid"
  done
}

main() {
  log "ROOT=$ROOT"
  log "GPU pre-flight: $(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader)"
  log "P0 soundness gate"
  bash "$ACT_ROOT/tests/run_soundness_tests.sh" > "$ROOT/soundness.log" 2>&1 || {
    log "ABORT: soundness runner failed; see $ROOT/soundness.log"
    exit 2
  }
  tail -5 "$ROOT/soundness.log" | tee -a "$LOG"

  if [ "${SKIP_ACASXU:-0}" = "1" ]; then
    log "P1 ACASXu DAG probe skipped by SKIP_ACASXU=1"
  else
    run_acasxu_dag_probe
  fi
  run_cifar_dense_conv_queue

  summarize_dir "nightly_all" "$ROOT"
  log "DONE ROOT=$ROOT"
}

main "$@"
