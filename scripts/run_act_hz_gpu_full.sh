#!/usr/bin/env bash
# Run the current ACT/HyZor verifier on GPU over the VNN-COMP benchmark set.
#
# This is intentionally a run-layer wrapper only:
#   - verifier: act.pipeline.watchdog_runner
#   - solver:   --solvers hybridz
#   - device:   cuda
#   - dtype:    float64
#   - result:   one directory per benchmark plus SUMMARY.tsv / SUMMARY.md
#
# Default mode is AUDIT_MODE=raw: collect the verifier's own GPU verdicts
# without enabling formal FAL receipt replay. Use AUDIT_MODE=formal later
# when you want paper-grade FALSIFIED receipts.
#
# It replaces the older ad-hoc gpu_stream*.sh scripts with one entry point
# whose purpose is simple: "run our current method on GPU and summarize it".

set -u

ACT_ROOT=${ACT_ROOT:-/data1/Kane/ACT}
PY=${PY:-/data1/Kane/miniconda3/envs/act-py312/bin/python}
BENCH_ROOT=${BENCH_ROOT:-/data1/Kane/data/vnncomp2025_benchmarks/benchmarks}
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
BASE=${BASE:-$ACT_ROOT/audit_results/gpu_full_ready_cam5_${STAMP}}

mkdir -p "$BASE"
LOG="$BASE/RUN.log"
exec > >(tee -a "$LOG") 2>&1

export PYTHONPATH="$ACT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ACT_VNNLIB_ROOT="$BENCH_ROOT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

# Keep old improvement-profile defaults active if the CLI still reads them.
# These are selection/config knobs, not alternate verifiers.
export HYZOR_USE_ACT=${HYZOR_USE_ACT:-1}
export HYZOR_TF_MODE=${HYZOR_TF_MODE:-interval}
export HYZOR_PURE_HZ_MODE=${HYZOR_PURE_HZ_MODE:-1}
export HYZOR_SAT_SIDECAR=${HYZOR_SAT_SIDECAR:-1}
export HYZOR_LARGE_CLS_EQ_LAYERS=${HYZOR_LARGE_CLS_EQ_LAYERS:-1}

# Override lists:
#   ONLY="cifar100_2024 tinyimagenet_2024" bash scripts/run_act_hz_gpu_full.sh
#   SKIP="vggnet16_2022 yolo_2023" bash scripts/run_act_hz_gpu_full.sh
ONLY=${ONLY:-}
SKIP=${SKIP:-}
DRY_RUN=${DRY_RUN:-0}
AUDIT_MODE=${AUDIT_MODE:-raw}  # raw | formal
RESUME=${RESUME:-1}            # 1 skips completed iids in an existing BASE

msg() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

bench_count() {
  local bench=$1
  "$PY" - "$BENCH_ROOT" "$bench" <<'PY'
import csv
import sys
from pathlib import Path
root = Path(sys.argv[1])
bench = sys.argv[2]
p = root / bench / "instances.csv"
if not p.exists():
    raise SystemExit(f"missing {p}")
with p.open(newline="") as f:
    rows = list(csv.reader(f))
print(len(rows))
PY
}

ids_for_bench() {
  local bench=$1
  local n
  n=$(bench_count "$bench")
  if [ "$n" -le 0 ]; then
    echo ""
  else
    seq -s, 0 "$((n - 1))"
  fi
}

missing_ids_for_bench() {
  local bench=$1
  local out=$2
  local n=$3
  "$PY" - "$bench" "$out" "$n" <<'PY'
import glob
import json
import sys
from pathlib import Path

bench, out, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
done = set()
for f in sorted(out.glob(f"per_instance_{bench}_*.json")):
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    for p in d.get("per_instance", []):
        iid = p.get("official_instance_id", p.get("instance_id"))
        if iid is not None:
            done.add(int(iid))
missing = [str(i) for i in range(n) if i not in done]
print(",".join(missing))
PY
}

contains_word() {
  local needle=$1
  local haystack=$2
  for x in $haystack; do
    if [ "$x" = "$needle" ]; then return 0; fi
  done
  return 1
}

should_run() {
  local bench=$1
  if [ -n "$ONLY" ] && ! contains_word "$bench" "$ONLY"; then return 1; fi
  if [ -n "$SKIP" ] && contains_word "$bench" "$SKIP"; then return 1; fi
  return 0
}

wall_for_bench() {
  case "$1" in
    safenlp_2024|nn4sys|relusplitter) echo 60 ;;
    acasxu_2023|linearizenn_2024|malbeware|collins_rul_cnn_2022) echo 60 ;;
    cersyve|tllverifybench_2023|dist_shift_2023|lsnc_relu|ml4acopf_2024) echo 60 ;;
    metaroom_2023|cora_2024|soundnessbench|traffic_signs_recognition_2023) echo 90 ;;
    cgan_2023) echo 180 ;;
    cifar100_2024|tinyimagenet_2024|yolo_2023|cctsdb_yolo_2023) echo 180 ;;
    vggnet16_2022) echo 300 ;;
    collins_aerospace_benchmark) echo 240 ;;
    *) echo 120 ;;
  esac
}

rss_for_bench() {
  case "$1" in
    vggnet16_2022|collins_aerospace_benchmark) echo 72 ;;
    cifar100_2024|tinyimagenet_2024|yolo_2023|cctsdb_yolo_2023|cgan_2023) echo 48 ;;
    *) echo 24 ;;
  esac
}

min_free_mb_for_bench() {
  case "$1" in
    vggnet16_2022|collins_aerospace_benchmark) echo 70000 ;;
    cifar100_2024|tinyimagenet_2024|yolo_2023|cctsdb_yolo_2023|cgan_2023) echo 45000 ;;
    safenlp_2024|relusplitter|nn4sys) echo 22000 ;;
    *) echo 12000 ;;
  esac
}

gpu_free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '
}

wait_for_gpu_free() {
  local need=$1
  local free
  while true; do
    free=$(gpu_free_mb 2>/dev/null || echo 0)
    if [ "${free:-0}" -ge "$need" ]; then
      return 0
    fi
    msg "GPU free ${free}MiB < ${need}MiB; waiting 60s"
    sleep 60
  done
}

summarize_bench() {
  local bench=$1
  local out=$2
  "$PY" - "$bench" "$out" <<'PY'
import glob
import json
import sys
from collections import Counter
bench, out = sys.argv[1], sys.argv[2]
chosen = {}
for f in sorted(glob.glob(f"{out}/per_instance_{bench}_*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for p in d.get("per_instance", []):
        iid = p.get("official_instance_id", p.get("instance_id"))
        if iid is None:
            iid = len(chosen)
        # If a child wrote a partial JSON and the watchdog later synthesized
        # a bounded UNKNOWN for the same iid, the watchdog row is authoritative.
        pri = 1 if "watchdog" in f else 0
        old = chosen.get(int(iid))
        if old is None or pri >= old[0]:
            chosen[int(iid)] = (pri, p)
c = Counter()
walls = []
for _iid, (_pri, p) in chosen.items():
    v = p.get("cli_normalized") or p.get("verdict") or "?"
    c[v] += 1
    if p.get("wall_s") is not None:
        try:
            walls.append(float(p["wall_s"]))
        except Exception:
            pass
print(f"verdicts={dict(c)} mean_wall={sum(walls)/max(len(walls),1):.1f}s")
PY
}

run_one() {
  local lane=$1
  local bench=$2
  if ! should_run "$bench"; then
    msg "[$lane] skip $bench"
    return 0
  fi
  if [ ! -f "$BENCH_ROOT/$bench/instances.csv" ]; then
    msg "[$lane] missing benchmark dir: $bench"
    return 0
  fi
  local ids wall rss min_free out n
  n=$(bench_count "$bench")
  out="$BASE/$bench"
  if [ "$RESUME" = "1" ] && [ -d "$out" ]; then
    ids=$(missing_ids_for_bench "$bench" "$out" "$n")
  else
    ids=$(ids_for_bench "$bench")
  fi
  wall=$(wall_for_bench "$bench")
  rss=$(rss_for_bench "$bench")
  min_free=$(min_free_mb_for_bench "$bench")
  mkdir -p "$out"

  if [ -z "$ids" ]; then
    msg "[$lane] $bench n=$n already complete under BASE; skip"
    return 0
  fi

  local n_run
  n_run=$(echo "$ids" | tr ',' '\n' | wc -l)
  msg "[$lane] $bench total=$n run=$n_run wall=${wall}s rss=${rss}GB need_free=${min_free}MiB"
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  wait_for_gpu_free "$min_free"
  "$PY" -m act.pipeline.watchdog_runner \
    --benchmark "$bench" --instance-ids "$ids" \
    --wall-s "$wall" --startup-grace-s 20 --poll-interval-s 0.5 \
    --rss-cap-gb "$rss" --grace-kill-s 3 \
    --device cuda --dtype float64 --strict-bounded-failure \
    $( [ "$AUDIT_MODE" = "raw" ] && printf '%s' '--raw-verdicts' ) \
    --out-dir "$out" --canonical-root "$BENCH_ROOT" \
    > "$out/driver.log" 2>&1
  local rc=$?
  msg "[$lane] done $bench rc=$rc $(summarize_bench "$bench" "$out")"
}

lane_light() {
  run_one light cersyve
  run_one light tllverifybench_2023
  run_one light traffic_signs_recognition_2023
  run_one light soundnessbench
  run_one light ml4acopf_2024
  run_one light dist_shift_2023
  run_one light lsnc_relu
  run_one light metaroom_2023
  run_one light cora_2024
  run_one light cgan_2023
}

lane_medium() {
  run_one medium acasxu_2023
  run_one medium linearizenn_2024
  run_one medium collins_rul_cnn_2022
  run_one medium malbeware
  run_one medium nn4sys
  run_one medium relusplitter
  run_one medium safenlp_2024
}

lane_heavy() {
  # Heavy lane is intentionally sequential. Do not split this lane unless
  # you are willing to rerun OOM instances afterwards.
  run_one heavy vggnet16_2022
  run_one heavy collins_aerospace_benchmark
  run_one heavy cctsdb_yolo_2023
  run_one heavy yolo_2023
  run_one heavy cifar100_2024
  run_one heavy tinyimagenet_2024
}

write_summary() {
  "$PY" - "$BASE" <<'PY'
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for bench_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    bench = bench_dir.name
    chosen = {}
    for f in sorted(bench_dir.glob(f"per_instance_{bench}_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for p in d.get("per_instance", []):
            iid = p.get("official_instance_id", p.get("instance_id"))
            if iid is None:
                iid = len(chosen)
            pri = 1 if "watchdog" in f.name else 0
            old = chosen.get(int(iid))
            if old is None or pri >= old[0]:
                chosen[int(iid)] = (pri, p)
    c = Counter()
    for _iid, (_pri, p) in chosen.items():
        v = p.get("cli_normalized") or p.get("verdict") or "?"
        c[v] += 1
    total = sum(c.values())
    V = c.get("CERTIFIED", 0)
    A = c.get("FALSIFIED", 0)
    U = c.get("UNKNOWN", 0)
    T = c.get("UNKNOWN_TIMEOUT", 0)
    O = c.get("UNKNOWN_RESOURCE_LIMIT", 0) + c.get("ERROR_OutOfMemoryError", 0)
    E = sum(v for k, v in c.items() if k.startswith("ERROR")) - c.get("ERROR_OutOfMemoryError", 0)
    rows.append((bench, V, A, U, T, O, E, total, dict(c)))

tsv = root / "SUMMARY.tsv"
with tsv.open("w") as f:
    f.write("benchmark\tV\tA\tU\tT\tOOM\tERR\tTOTAL\tcounts\n")
    for r in rows:
        f.write("\t".join(map(str, r)) + "\n")

md = root / "SUMMARY.md"
with md.open("w") as f:
    f.write(f"# ACT/HyZor GPU Full Sweep\n\n")
    f.write(f"Root: `{root}`\n\n")
    f.write("| Benchmark | V | A | U | T | OOM | ERR | TOTAL |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    tot = Counter()
    for bench, V, A, U, T, O, E, total, counts in rows:
        f.write(f"| `{bench}` | {V} | {A} | {U} | {T} | {O} | {E} | {total} |\n")
        tot["V"] += V; tot["A"] += A; tot["U"] += U
        tot["T"] += T; tot["OOM"] += O; tot["ERR"] += E; tot["TOTAL"] += total
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    f.write(f"| **TOTAL** | **{tot['V']}** | **{tot['A']}** | **{tot['U']}** | **{tot['T']}** | **{tot['OOM']}** | **{tot['ERR']}** | **{tot['TOTAL']}** |\n")
print(f"summary: {md}")
print(f"summary_tsv: {tsv}")
PY
}

msg "ACT_ROOT=$ACT_ROOT"
msg "BENCH_ROOT=$BENCH_ROOT"
msg "BASE=$BASE"
msg "branch=$(git -C "$ACT_ROOT" branch --show-current 2>/dev/null || true)"
msg "commit=$(git -C "$ACT_ROOT" rev-parse --short HEAD 2>/dev/null || true)"
msg "ONLY=${ONLY:-<all>} SKIP=${SKIP:-<none>} DRY_RUN=$DRY_RUN AUDIT_MODE=$AUDIT_MODE RESUME=$RESUME"

if [ "$DRY_RUN" = "1" ]; then
  lane_light
  lane_medium
  lane_heavy
  msg "dry-run complete"
  exit 0
fi

lane_light &
p1=$!
lane_medium &
p2=$!
lane_heavy &
p3=$!

msg "launched lanes: light=$p1 medium=$p2 heavy=$p3"
wait "$p1"; r1=$?
wait "$p2"; r2=$?
wait "$p3"; r3=$?
msg "lanes finished rc: light=$r1 medium=$r2 heavy=$r3"

write_summary
msg "DONE. See $BASE/SUMMARY.md"

exit 0
