#!/usr/bin/env bash
# Serial rerun of the 22 NeuralSAT instances suspected of resource-contention
# failure during the parallel sweep on 2026-05-27. One instance at a time, no
# competing GPU load.
#
# - 3 collins_rul_cnn_2022 instances: SIGKILL with wall-clock < 50% of timeout
#   (high-confidence OOM-killer during dual-lane parallel run).
# - 19 ml4acopf_2024 instances: SIGSEGV (exit 139) during dual-lane run; could
#   be OOM-induced memory corruption OR a known NeuralSAT segfault on ACOPF
#   graphs. Serial rerun disambiguates.
#
# Original artifacts (.raw, .log, .json, .result) are moved to
# <bench>/_oom_rerun_backup/ before the rerun overwrites in place.
# Per-instance new outcome is appended to _oom_rerun_results.csv with diagnosis.

set -uo pipefail

NEURALSAT_DIR=${NEURALSAT_DIR:-/data1/Kane/neuralsat}
BENCH_ROOT=${BENCH_ROOT:-/data1/Kane/data/vnncomp2025_benchmarks/benchmarks}
RESULTS_ROOT=${RESULTS_ROOT:-/data1/Kane/ACT/audit_results/neuralsat_strict_20260527}
PY_BIN=${PY_BIN:-/data1/Kane/miniconda3/envs/neuralsat/bin/python}
KILL_GRACE=${NS_KILL_GRACE:-90}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONPATH="$NEURALSAT_DIR/src:${PYTHONPATH:-}"
export GRB_LICENSE_FILE=${GRB_LICENSE_FILE:-/data1/Kane/ACT/modules/gurobi/gurobi.lic}

OUT_CSV="$RESULTS_ROOT/_oom_rerun_results.csv"
LOG_FILE="$RESULTS_ROOT/_oom_rerun.log"

log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$LOG_FILE"; }

# Format: BENCH|IDX|ONNX_REL|VNNLIB_REL|TIMEOUT|ORIGINAL_FAILURE
INSTANCES=(
  "collins_rul_cnn_2022|21|onnx/NN_rul_small_window_20.onnx|vnnlib/if_then_7levels_w20.vnnlib|1800|SIGKILL_early(597s/1800s)"
  "collins_rul_cnn_2022|22|onnx/NN_rul_small_window_20.onnx|vnnlib/if_then_9levels_w20.vnnlib|1800|SIGKILL_early(597s/1800s)"
  "collins_rul_cnn_2022|43|onnx/NN_rul_full_window_20.onnx|vnnlib/if_then_9levels_w20.vnnlib|1800|SIGKILL_early(882s/1800s)"
  "ml4acopf_2024|15|./onnx/118_ieee_ml4acopf.onnx|./vnnlib/118_ieee_prop2.vnnlib|600|SIGSEGV"
  "ml4acopf_2024|16|./onnx/118_ieee_ml4acopf.onnx|./vnnlib/118_ieee_prop4.vnnlib|600|SIGSEGV"
  "ml4acopf_2024|17|./onnx/118_ieee_ml4acopf.onnx|./vnnlib/118_ieee_prop3.vnnlib|600|SIGSEGV"
  "ml4acopf_2024|18|./onnx/118_ieee_ml4acopf.onnx|./vnnlib/118_ieee_prop5.vnnlib|600|SIGSEGV"
  "ml4acopf_2024|19|./onnx/118_ieee_ml4acopf.onnx|./vnnlib/118_ieee_prop6.vnnlib|600|SIGSEGV"
  "ml4acopf_2024|43|./onnx/14_ieee_ml4acopf.onnx|./vnnlib/14_ieee_prop9.vnnlib|600|SIGSEGV"
  "ml4acopf_2024|44|./onnx/14_ieee_ml4acopf.onnx|./vnnlib/14_ieee_prop7.vnnlib|600|SIGSEGV"
  "ml4acopf_2024|45|./onnx/14_ieee_ml4acopf.onnx|./vnnlib/14_ieee_prop2.vnnlib|600|SIGSEGV"
  "ml4acopf_2024|46|./onnx/14_ieee_ml4acopf.onnx|./vnnlib/14_ieee_prop8.vnnlib|600|SIGSEGV"
  "ml4acopf_2024|47|./onnx/14_ieee_ml4acopf.onnx|./vnnlib/14_ieee_prop4.vnnlib|600|SIGSEGV"
  "ml4acopf_2024|48|./onnx/14_ieee_ml4acopf.onnx|./vnnlib/14_ieee_prop6.vnnlib|600|SIGSEGV"
  "ml4acopf_2024|49|./onnx/14_ieee_ml4acopf.onnx|./vnnlib/14_ieee_prop12.vnnlib|600|SIGSEGV"
  "ml4acopf_2024|50|./onnx/14_ieee_ml4acopf.onnx|./vnnlib/14_ieee_prop13.vnnlib|600|SIGSEGV"
  "ml4acopf_2024|51|./onnx/14_ieee_ml4acopf.onnx|./vnnlib/14_ieee_prop11.vnnlib|600|SIGSEGV"
  "ml4acopf_2024|52|./onnx/14_ieee_ml4acopf.onnx|./vnnlib/14_ieee_prop1.vnnlib|600|SIGSEGV"
  "ml4acopf_2024|53|./onnx/14_ieee_ml4acopf.onnx|./vnnlib/14_ieee_prop3.vnnlib|600|SIGSEGV"
  "ml4acopf_2024|54|./onnx/14_ieee_ml4acopf.onnx|./vnnlib/14_ieee_prop14.vnnlib|600|SIGSEGV"
  "ml4acopf_2024|55|./onnx/14_ieee_ml4acopf.onnx|./vnnlib/14_ieee_prop10.vnnlib|600|SIGSEGV"
  "ml4acopf_2024|56|./onnx/14_ieee_ml4acopf.onnx|./vnnlib/14_ieee_prop5.vnnlib|600|SIGSEGV"
)

# CSV header (write once if missing)
if [[ ! -f "$OUT_CSV" ]]; then
  echo "rerun_at,bench,idx,onnx,vnnlib,timeout,original_failure,new_wall_sec,new_verdict_raw,new_verdict,new_exit_code,diagnosis" > "$OUT_CSV"
fi

log "=== serial rerun start — ${#INSTANCES[@]} instances ==="
log "policy: one instance at a time, no parallel lanes, single GPU process"
log "originals backed up to <bench>/_oom_rerun_backup/"

n=0
for entry in "${INSTANCES[@]}"; do
  n=$((n+1))
  IFS='|' read -r BENCH IDX ONNX_REL VNNLIB_REL TO ORIG_FAIL <<<"$entry"

  BENCH_DIR="$BENCH_ROOT/$BENCH"
  ONNX="$BENCH_DIR/$ONNX_REL"
  VNNLIB="$BENCH_DIR/$VNNLIB_REL"
  ONNX_TAG=$(basename "$ONNX_REL" .onnx)
  VNN_TAG=$(basename "$VNNLIB_REL" .vnnlib)
  IDX_PAD=$(printf "%04d" "$IDX")
  STEM="${IDX_PAD}__${ONNX_TAG}__${VNN_TAG}"

  OUT_DIR="$RESULTS_ROOT/$BENCH"
  BACKUP_DIR="$OUT_DIR/_oom_rerun_backup"
  mkdir -p "$BACKUP_DIR"

  RES="$OUT_DIR/$STEM.result"
  RAW="$OUT_DIR/$STEM.raw"
  LOG="$OUT_DIR/$STEM.log"
  JSON="$OUT_DIR/$STEM.json"

  log "[$n/${#INSTANCES[@]}] $BENCH idx=$IDX  orig=$ORIG_FAIL"

  # Move originals to backup (preserve forensics)
  for ext in result raw log json; do
    f="$OUT_DIR/$STEM.$ext"
    [[ -f "$f" ]] && mv -f "$f" "$BACKUP_DIR/" 2>/dev/null
  done

  KILL_AT=$((TO + KILL_GRACE))

  T0=$(date +%s.%N)
  timeout --kill-after=10 "${KILL_AT}s" \
    "$PY_BIN" "$NEURALSAT_DIR/src/main.py" \
      --net "$ONNX" --spec "$VNNLIB" \
      --timeout "$TO" \
      --result_file "$RAW" \
      --export_cex \
      --disable_attack >"$LOG" 2>&1
  RC=$?
  T1=$(date +%s.%N)
  WALL=$(awk "BEGIN{printf \"%.2f\", $T1-$T0}")

  VERDICT_RAW=""
  if [[ -f "$RAW" && -s "$RAW" ]]; then
    VERDICT_RAW=$(head -n1 "$RAW" | tr -d '[:space:]')
  fi
  case "$VERDICT_RAW" in
    unsat|holds)        VERDICT=unsat ;;
    sat|violated)       VERDICT=sat ;;
    timeout)            VERDICT=timeout ;;
    unknown|early_stop) VERDICT=unknown ;;
    error)              VERDICT=error ;;
    "")                 VERDICT=missing_result ;;
    *)                  VERDICT="raw_$VERDICT_RAW" ;;
  esac
  if [[ "$RC" == "124" || "$RC" == "137" ]]; then VERDICT=timeout_killed; fi

  # Diagnosis
  if [[ "$VERDICT" == "missing_result" ]]; then
    if [[ "$RC" == "139" ]]; then
      DIAG="reproduces_SIGSEGV_serial: NeuralSAT internal segfault (NOT OOM)"
    elif [[ "$RC" == "134" ]]; then
      DIAG="reproduces_SIGABRT_serial"
    else
      DIAG="missing_result_serial_rc=$RC"
    fi
  elif [[ "$VERDICT" == "timeout_killed" && "$RC" == "137" ]]; then
    DIAG="reproduces_SIGKILL_serial: likely real timeout escalation OR persistent OOM"
  elif [[ "$VERDICT" =~ ^(unsat|sat|timeout|unknown)$ ]]; then
    DIAG="RECOVERED_serial: original failure WAS resource-contention (OOM-induced)"
  else
    DIAG="serial_outcome=$VERDICT  rc=$RC"
  fi

  log "  -> verdict=$VERDICT rc=$RC wall=${WALL}s  ::  $DIAG"

  echo "$RES" > /dev/null # placeholder for re-writing .result
  if [[ -f "$RAW" && -n "$VERDICT" && "$VERDICT" != "missing_result" ]]; then
    echo "$VERDICT" > "$RES"
  fi

  # Append to tracking CSV
  printf '"%s","%s",%d,"%s","%s",%d,"%s",%s,"%s","%s",%d,"%s"\n' \
    "$(date -Iseconds)" "$BENCH" "$IDX" "$ONNX_REL" "$VNNLIB_REL" "$TO" "$ORIG_FAIL" "$WALL" "$VERDICT_RAW" "$VERDICT" "$RC" "$DIAG" >> "$OUT_CSV"

  # Refresh per-instance JSON sidecar
  printf '{"idx":%d,"benchmark":"%s","lane":"serial_rerun","onnx":"%s","vnnlib":"%s","csv_timeout":%d,"used_timeout":%d,"wall_sec":%s,"verdict_raw":"%s","verdict":"%s","exit_code":%d,"attack_disabled":true,"rerun_reason":"%s"}\n' \
    "$IDX" "$BENCH" "$ONNX_REL" "$VNNLIB_REL" "$TO" "$TO" "$WALL" "$VERDICT_RAW" "$VERDICT" "$RC" "$ORIG_FAIL" > "$JSON"

  sleep 2   # let GPU memory drain between instances
done

log "=== serial rerun complete ==="
log "summary written to: $OUT_CSV"
log "originals preserved in: <bench>/_oom_rerun_backup/"
