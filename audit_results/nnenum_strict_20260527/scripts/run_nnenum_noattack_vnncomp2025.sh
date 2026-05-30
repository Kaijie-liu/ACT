#!/usr/bin/env bash
# Run nnenum on all VNN-COMP 2025 benchmarks.
#
# nnenum has NO attack/PGD/falsification mechanism in its codebase
# (verified by grep of /data1/Kane/nnenum/src/nnenum/). It is a pure
# star-set + zonotope verifier. So there is no "--disable_attack" flag
# — being attack-free is inherent. This script is the no-attack runner
# by construction.
#
# CLI: python -m nnenum.nnenum -o ONNX -v VNNLIB -t TIMEOUT -f RESULTS -s CATEGORY
# Output verdicts in RESULTS file: holds | violated | unknown | timeout | error
# (we normalize to sat/unsat/timeout/unknown/error for downstream consistency)
#
# Same layout as abcrown / NeuralSAT runners:
#   $RESULTS_ROOT/$BENCH/<idx>__<onnx>__<vnnlib>.result        normalized verdict
#   $RESULTS_ROOT/$BENCH/<idx>__<onnx>__<vnnlib>.log           full stdout+stderr
#   $RESULTS_ROOT/$BENCH/<idx>__<onnx>__<vnnlib>.raw           raw nnenum result text
#   $RESULTS_ROOT/$BENCH/<idx>__<onnx>__<vnnlib>.json          per-instance metadata
#   $RESULTS_ROOT/$BENCH/_summary.csv                          per-benchmark CSV
#   $RESULTS_ROOT/_run.log                                     driver log
#   $RESULTS_ROOT/_run.meta.json                               provenance
#
set -uo pipefail

NNENUM_DIR=${NNENUM_DIR:-/data1/Kane/nnenum}
BENCH_ROOT=${BENCH_ROOT:-/data1/Kane/data/vnncomp2025_benchmarks/benchmarks}
RESULTS_ROOT=${RESULTS_ROOT:-/data1/Kane/ACT/audit_results/nnenum_noattack_20260525}
PY_BIN=${PY_BIN:-/data1/Kane/miniconda3/envs/nnenumenv/bin/python}
TIMEOUT_CAP=${NNENUM_TIMEOUT_CAP:-0}
KILL_GRACE=${NNENUM_KILL_GRACE:-90}
ONLY_BENCH=${1:-}

# nnenum is CPU-only — explicitly hide GPU so it doesn't accidentally
# touch CUDA (and so it doesn't conflict with concurrent GPU users).
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export PYTHONPATH="$NNENUM_DIR/src:${PYTHONPATH:-}"
# nnenum uses GLPK by default; Gurobi only when explicitly selected in
# settings (e.g. tllverifybench). Inherit pre-set GRB_LICENSE_FILE.
export GRB_LICENSE_FILE=${GRB_LICENSE_FILE:-/data1/Kane/ACT/modules/gurobi/gurobi.lic}

mkdir -p "$RESULTS_ROOT"
DRIVER_LOG="$RESULTS_ROOT/_run.log"
META_JSON="$RESULTS_ROOT/_run.meta.json"

log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$DRIVER_LOG"; }

# Map 2025 benchmark folder -> the -s "settings" string nnenum recognizes.
# nnenum matches on substring (see src/nnenum/nnenum.py:349-388). For
# benchmarks with no specific branch (vit, yolo, ...), passing the folder
# name lets nnenum fall through to its input-size-based default
# (control vs image), which is the same behavior as VNN-COMP setup.
declare -A SETTINGS_MAP=(
  [acasxu_2023]=acasxu
  [cctsdb_yolo_2023]=cctsdb_yolo
  [cersyve]=cersyve
  [cgan_2023]=cgan
  [cifar100_2024]=cifar100
  [collins_aerospace_benchmark]=collins_aerospace
  [collins_rul_cnn_2022]=collins_rul_cnn
  [cora_2024]=cora
  [dist_shift_2023]=dist_shift
  [linearizenn_2024]=linearizenn
  [lsnc_relu]=lsnc
  [malbeware]=malbeware
  [metaroom_2023]=metaroom
  [ml4acopf_2024]=ml4acopf
  [nn4sys]=nn4sys
  [relusplitter]=relusplitter
  [safenlp_2024]=safenlp
  [sat_relu]=sat_relu
  [soundnessbench]=soundnessbench
  [test]=test
  [tinyimagenet_2024]=tinyimagenet
  [tllverifybench_2023]=tllverifybench
  [traffic_signs_recognition_2023]=traffic_signs
  [vggnet16_2022]=vggnet16
  [vit_2023]=vit
  [yolo_2023]=yolo
)

# Order: small/fast first to validate, large at end. vggnet16 and
# collins are at the end because they're known difficult — but we keep
# them so we get an honest "tried and failed" record.
BENCH_ORDER=(
  test
  cersyve cgan_2023
  tllverifybench_2023 cctsdb_yolo_2023 traffic_signs_recognition_2023
  collins_rul_cnn_2022 linearizenn_2024 ml4acopf_2024 dist_shift_2023
  yolo_2023 lsnc_relu soundnessbench
  metaroom_2023 cora_2024 relusplitter sat_relu malbeware
  acasxu_2023 nn4sys
  vit_2023 tinyimagenet_2024 cifar100_2024
  safenlp_2024
  vggnet16_2022 collins_aerospace_benchmark
)

cat >"$META_JSON" <<EOF
{
  "tool": "nnenum",
  "tool_dir": "$NNENUM_DIR",
  "tool_commit": "$(git -C "$NNENUM_DIR" rev-parse HEAD 2>/dev/null || echo unknown)",
  "python": "$PY_BIN",
  "python_version": "$($PY_BIN --version 2>&1 | head -1)",
  "started_at": "$(date -Iseconds)",
  "host": "$(hostname)",
  "bench_root": "$BENCH_ROOT",
  "results_root": "$RESULTS_ROOT",
  "flags": {
    "attack_path_exists": false,
    "note": "nnenum has no attack mechanism in its codebase; this run is attack-free by construction.",
    "TIMEOUT_CAP_SEC": $TIMEOUT_CAP,
    "KILL_GRACE_SEC": $KILL_GRACE,
    "lp_solver_default": "GLPK (swiglpk); Gurobi only when explicitly selected per benchmark setting"
  }
}
EOF

log "=== nnenum VNN-COMP 2025 sweep (attack-free by construction)"
log "results -> $RESULTS_ROOT"

# Normalize nnenum's verdict text to the same labels we use elsewhere.
# nnenum actually writes: 'unsat' | 'sat\n((X_0 ...))' | 'timeout' | 'unknown' | 'error'.
# (Older versions used 'holds'/'violated'; map both.)
normalize_verdict() {
  local raw="$1"
  case "$raw" in
    unsat|holds)        echo unsat ;;
    sat|violated)       echo sat ;;
    timeout)            echo timeout ;;
    unknown)            echo unknown ;;
    error)              echo error ;;
    "")                 echo missing_result ;;
    *)                  echo "raw_$raw" ;;
  esac
}

run_benchmark() {
  local BENCH=$1
  local SETTING=${SETTINGS_MAP[$BENCH]:-$BENCH}
  local BENCH_DIR=$BENCH_ROOT/$BENCH
  local CSV=$BENCH_DIR/instances.csv
  local OUT_DIR=$RESULTS_ROOT/$BENCH
  local SUMMARY=$OUT_DIR/_summary.csv

  if [[ ! -f "$CSV" ]]; then
    log "[$BENCH] SKIP — instances.csv not found"
    return 0
  fi

  mkdir -p "$OUT_DIR"
  if [[ ! -f "$SUMMARY" ]]; then
    echo "idx,onnx,vnnlib,csv_timeout,used_timeout,wall_sec,verdict_raw,verdict,exit_code,result_file,log_file" >"$SUMMARY"
  fi
  local NTOTAL=$(wc -l <"$CSV")
  log "[$BENCH] setting=$SETTING instances=$NTOTAL"

  local idx=0
  local n_done=0 n_skip=0 n_sat=0 n_unsat=0 n_to=0 n_err=0
  while IFS=, read -r ONNX_REL VNNLIB_REL CSV_TIMEOUT REST; do
    idx=$((idx+1))
    [[ -z "${ONNX_REL// }" ]] && continue
    local ONNX="$BENCH_DIR/$ONNX_REL"
    local VNNLIB="$BENCH_DIR/$VNNLIB_REL"
    local ONNX_TAG=$(basename "$ONNX_REL" .onnx)
    local VNN_TAG=$(basename "$VNNLIB_REL" .vnnlib)
    local IDX_PAD=$(printf "%04d" "$idx")
    local STEM="${IDX_PAD}__${ONNX_TAG}__${VNN_TAG}"
    local RES="$OUT_DIR/$STEM.result"      # normalized verdict
    local RAW="$OUT_DIR/$STEM.raw"         # raw nnenum output
    local LOG="$OUT_DIR/$STEM.log"
    local JSON="$OUT_DIR/$STEM.json"

    if [[ -f "$RES" && -s "$RES" ]]; then
      local prev=$(head -n1 "$RES" | tr -d '[:space:]')
      if [[ -n "$prev" ]]; then n_skip=$((n_skip+1)); continue; fi
    fi

    local USE_TO=${CSV_TIMEOUT//[[:space:]]/}
    USE_TO=${USE_TO%.*}   # strip decimal so bash arithmetic works (e.g. "480.0" -> "480")
    if [[ "$TIMEOUT_CAP" -gt 0 && "$USE_TO" -gt "$TIMEOUT_CAP" ]]; then
      USE_TO=$TIMEOUT_CAP
    fi
    local KILL_AT=$((USE_TO + KILL_GRACE))

    rm -f "$RES" "$RAW"
    local T0=$(date +%s.%N)
    set +e
    # nnenum: pass -p to cap per-instance worker processes, nice +10 to yield CPU
    # if other heavier verifiers (NeuralSAT/ACT/PyRAT) compete.
    timeout --kill-after=10 "${KILL_AT}s" \
      nice -n 10 \
      "$PY_BIN" -m nnenum.nnenum \
        -o "$ONNX" \
        -v "$VNNLIB" \
        -t "$USE_TO" \
        -f "$RAW" \
        -p "${NNENUM_PROCS:-8}" \
        -s "$SETTING" >"$LOG" 2>&1
    local RC=$?
    set -e
    local T1=$(date +%s.%N)
    local WALL=$(awk "BEGIN{printf \"%.2f\", $T1-$T0}")

    local VERDICT_RAW=""
    if [[ -f "$RAW" && -s "$RAW" ]]; then
      VERDICT_RAW=$(head -n1 "$RAW" | tr -d '[:space:]')
    fi
    local VERDICT
    VERDICT=$(normalize_verdict "$VERDICT_RAW")
    if [[ "$RC" == "124" || "$RC" == "137" ]]; then
      VERDICT="timeout_killed"
    fi
    echo "$VERDICT" >"$RES"

    case "$VERDICT" in
      sat) n_sat=$((n_sat+1)) ;;
      unsat) n_unsat=$((n_unsat+1)) ;;
      timeout*) n_to=$((n_to+1)) ;;
      unknown) n_to=$((n_to+1)) ;;
      *) n_err=$((n_err+1)) ;;
    esac
    n_done=$((n_done+1))

    printf '{"idx":%d,"benchmark":"%s","setting":"%s","onnx":"%s","vnnlib":"%s","csv_timeout":%s,"used_timeout":%s,"wall_sec":%s,"verdict_raw":"%s","verdict":"%s","exit_code":%d,"attack_path_exists":false}\n' \
      "$idx" "$BENCH" "$SETTING" "$ONNX_REL" "$VNNLIB_REL" "$CSV_TIMEOUT" "$USE_TO" "$WALL" "$VERDICT_RAW" "$VERDICT" "$RC" >"$JSON"

    printf '%d,"%s","%s",%s,%s,%s,"%s","%s",%d,"%s","%s"\n' \
      "$idx" "$ONNX_REL" "$VNNLIB_REL" "$CSV_TIMEOUT" "$USE_TO" "$WALL" "$VERDICT_RAW" "$VERDICT" "$RC" "$RES" "$LOG" >>"$SUMMARY"

    if (( idx % 5 == 0 )) || (( idx == NTOTAL )); then
      log "[$BENCH] $idx/$NTOTAL sat=$n_sat unsat=$n_unsat timeout=$n_to err=$n_err resumed=$n_skip"
    fi
  done <"$CSV"
  log "[$BENCH] DONE — total=$idx new=$n_done resumed=$n_skip sat=$n_sat unsat=$n_unsat timeout=$n_to err=$n_err"
}

if [[ -n "$ONLY_BENCH" ]]; then
  for B in $ONLY_BENCH; do run_benchmark "$B"; done
else
  for B in "${BENCH_ORDER[@]}"; do run_benchmark "$B"; done
fi

log "=== sweep complete"
