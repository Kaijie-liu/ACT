#!/usr/bin/env bash
# NNV STRICT no-helper sweep over VNN-COMP 2025.
#
# Invariants:
#  - Calls nnv_strict_run_one.m per instance with NNV_STRICT_NO_HELPER=1.
#  - run_vnncomp_instance.m MUST have been patched by
#    scripts/nnv_patches/apply_strict_patch.sh.
#  - External `timeout` enforces per-instance wall-clock limit (NNV has no
#    internal timeout; it expects a Python parent to cancel it).
#  - Idempotent: pre-existing non-empty .result files are skipped on resume.
#
# Usage:
#   run_nnv_strict_vnncomp2025.sh                 # all benchmarks, light->heavy
#   run_nnv_strict_vnncomp2025.sh acasxu_2023     # just one
#   BENCH_OVERRIDE="acasxu_2023 sat_relu" run_nnv_strict_vnncomp2025.sh
#
# Env knobs:
#   NNV_TIMEOUT_CAP=N    cap per-instance wall clock at N seconds (0 = use csv)
#   NNV_KILL_GRACE=N     extra seconds before SIGKILL after timeout (default 30)
#   NNV_GPU_FREE_GB_MIN  wait until this GB of GPU free before each bench (default 20)
#   NNV_GPU_POLL_S       GPU poll interval (default 30)
#   NNV_MATLAB_BIN       MATLAB binary (default /data1/Kane/MATLAB/bin/matlab)
#
# NOTE: script-level "set -e" is NOT used — see neuralsat fix history; a single
# non-zero exit in any helper would silently kill the sweep mid-flight.
set -uo pipefail

BENCH_ROOT=${BENCH_ROOT:-/data1/Kane/data/vnncomp2025_benchmarks/benchmarks}
RESULTS_ROOT=${RESULTS_ROOT:-/data1/Kane/ACT/audit_results/nnv_strict_20260527}
SCRIPTS_DIR=${SCRIPTS_DIR:-/data1/Kane/ACT/scripts}
MATLAB_BIN=${NNV_MATLAB_BIN:-/data1/Kane/MATLAB/bin/matlab}
TIMEOUT_CAP=${NNV_TIMEOUT_CAP:-0}
KILL_GRACE=${NNV_KILL_GRACE:-30}
GPU_FREE_GB_MIN=${NNV_GPU_FREE_GB_MIN:-20}
GPU_POLL_S=${NNV_GPU_POLL_S:-30}

NNV_ENTRY=/data1/Kane/nnv/code/nnv/examples/Submission/VNN_COMP2025/run_vnncomp_instance.m

# --- pre-flight checks ---
if [[ ! -x "$MATLAB_BIN" ]]; then
  echo "ERROR: MATLAB not found at $MATLAB_BIN (set NNV_MATLAB_BIN)" >&2; exit 1
fi
if ! grep -q "STRICT-MODE PATCH (ACT paper" "$NNV_ENTRY" 2>/dev/null; then
  echo "ERROR: NNV STRICT patch not applied to $NNV_ENTRY" >&2
  echo "       Run: $SCRIPTS_DIR/nnv_patches/apply_strict_patch.sh" >&2
  exit 1
fi
if [[ ! -f "$SCRIPTS_DIR/nnv_strict_run_one.m" ]]; then
  echo "ERROR: $SCRIPTS_DIR/nnv_strict_run_one.m missing" >&2; exit 1
fi

mkdir -p "$RESULTS_ROOT"
META_PATH="$RESULTS_ROOT/_run.meta.json"
DRIVER_LOG="$RESULTS_ROOT/_run.log"
PID_PATH="$RESULTS_ROOT/_run.pid"
echo $$ > "$PID_PATH"

log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$DRIVER_LOG"; }

# --- benchmark order: light -> heavy. Mirrors CORA TRUESTRICT ordering.
#     STRICT-impossible benchmarks (cp-star-only) are STILL listed so the
#     runner writes `unsupported_strict` markers and the audit is complete.
FULL_ORDER=(
  test                              #   5  smoke
  cersyve                           #  12  cp-star → unsupported_strict
  lsnc_relu                         #  80  NNV error (IR unsupported)
  soundnessbench                    #  50  cp-star → unsupported_strict
  sat_relu                          # 100  small ReLU MLP, exact-star
  cgan_2023                         #  21  mixed (transformer subset → cp-star)
  tllverifybench_2023               #  32  small MLP, relax-star
  malbeware                         # 150  exact-star
  traffic_signs_recognition_2023    #  45  NNV error (IR unsupported)
  collins_rul_cnn_2022              #  62  approx-star
  linearizenn_2024                  #  60  approx-star (fallback rejected)
  dist_shift_2023                   #  72  exact-star
  ml4acopf_2024                     #  69  cp-star → unsupported_strict
  metaroom_2023                     # 100  approx-star
  cctsdb_yolo_2023                  #  39  NNV error
  yolo_2023                         #  72  cp-star → unsupported_strict
  relusplitter                      # 220  relax-star
  cora_2024                         # 180  mixed (-set → cp-star, else relax-star)
  acasxu_2023                       # 186  exact-star / approx-star
  nn4sys                            # 194  mixed (lindex approx-star, else cp-star)
  safenlp_2024                      # 1080 approx-star / exact-star
  vit_2023                          # 200  cp-star → unsupported_strict
  tinyimagenet_2024                 # 200  cp-star → unsupported_strict
  cifar100_2024                     # 200  cp-star → unsupported_strict
  vggnet16_2022                     #  18  cp-star → unsupported_strict
  collins_aerospace_benchmark       #   6  cp-star → unsupported_strict
)

if [[ $# -gt 0 ]]; then
  BENCHES=("$@")
elif [[ -n "${BENCH_OVERRIDE:-}" ]]; then
  read -r -a BENCHES <<< "$BENCH_OVERRIDE"
else
  BENCHES=("${FULL_ORDER[@]}")
fi

# --- meta.json on first launch ---
if [[ ! -f "$META_PATH" ]]; then
  NNV_COMMIT=$(git -C /data1/Kane/nnv rev-parse HEAD 2>/dev/null || echo unknown)
  MATLAB_VER=$("$MATLAB_BIN" -batch "disp(version)" 2>/dev/null | tr -d '\n' | tr -s ' ')
  cat >"$META_PATH" <<EOF
{
  "tool": "NNV (vnncomp2025, STRICT no-helper)",
  "tool_dir": "/data1/Kane/nnv",
  "tool_commit": "$NNV_COMMIT",
  "matlab_bin": "$MATLAB_BIN",
  "matlab_version": "$MATLAB_VER",
  "started_at": "$(date -Iseconds)",
  "host": "$(hostname)",
  "bench_root": "$BENCH_ROOT",
  "results_root": "$RESULTS_ROOT",
  "patches": [
    "/data1/Kane/ACT/scripts/nnv_patches/run_vnncomp_instance.m.patch"
  ],
  "flags": {
    "NNV_STRICT_NO_HELPER": "1",
    "falsify_single": "SKIPPED (random sampling + lb/ub corner eval would be helpers)",
    "cp_star_reachability": "REJECTED (conformal-prediction, not sound; written as unsupported_strict)",
    "sound_methods_allowed": ["approx-star", "exact-star", "relax-star-area"],
    "TIMEOUT_CAP_SEC": $TIMEOUT_CAP,
    "KILL_GRACE_SEC": $KILL_GRACE,
    "GPU_FREE_GB_MIN": $GPU_FREE_GB_MIN,
    "GPU_POLL_S": $GPU_POLL_S
  }
}
EOF
fi

# --- gpu gate ---
gpu_wait() {
  local need_mib=$((GPU_FREE_GB_MIN * 1024))
  local first=1
  while :; do
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [[ -z "$free" ]]; then free=0; fi
    if (( free >= need_mib )); then
      (( first == 0 )) && log "[GPU gate] free=${free}MiB OK, starting $1"
      return
    fi
    if (( first == 1 )); then
      log "[GPU gate] $1 waiting (free=${free}MiB < ${need_mib}MiB)"
      first=0
    fi
    sleep "$GPU_POLL_S"
  done
}

# --- per-instance MATLAB invocation ---
run_one() {
  local CAT=$1 ONNX=$2 VNNLIB=$3 OUT=$4 LOG=$5 USE_TO=$6
  local KILL_AT=$((USE_TO + KILL_GRACE))
  # nnv_strict_run_one.m sets NNV_STRICT_NO_HELPER=1 itself; we also set in env
  # so any cold MATLAB workers (parpool etc.) inherit it.
  NNV_STRICT_NO_HELPER=1 \
  timeout --kill-after=10 "${KILL_AT}s" \
    "$MATLAB_BIN" -nodisplay -nosplash -nojvm -batch \
      "addpath('$SCRIPTS_DIR'); nnv_strict_run_one('$CAT','$ONNX','$VNNLIB','$OUT');" \
      >"$LOG" 2>&1
  return $?
}

# --- per-benchmark loop ---
run_benchmark() {
  local BENCH=$1
  local BENCH_DIR=$BENCH_ROOT/$BENCH
  local CSV=$BENCH_DIR/instances.csv
  local OUT_DIR=$RESULTS_ROOT/$BENCH
  local SUMMARY=$OUT_DIR/_summary.csv

  if [[ ! -f "$CSV" ]]; then
    log "[$BENCH] SKIP — instances.csv not found"; return 0
  fi

  gpu_wait "[$BENCH]"
  mkdir -p "$OUT_DIR"
  if [[ ! -f "$SUMMARY" ]]; then
    echo "idx,onnx,vnnlib,csv_timeout,used_timeout,wall_sec,verdict_raw,verdict,exit_code,result_file,log_file" >"$SUMMARY"
  fi
  local NTOTAL
  NTOTAL=$(wc -l <"$CSV")
  log "[$BENCH] instances=$NTOTAL"

  local idx=0 n_done=0 n_skip=0 n_sat=0 n_unsat=0 n_to=0 n_uns=0 n_un=0 n_err=0
  while IFS=, read -r ONNX_REL VNNLIB_REL CSV_TIMEOUT REST; do
    idx=$((idx+1))
    ONNX_REL="${ONNX_REL//$'\r'/}"; VNNLIB_REL="${VNNLIB_REL//$'\r'/}"
    CSV_TIMEOUT="${CSV_TIMEOUT//[[:space:]]/}"
    [[ -z "${ONNX_REL// }" ]] && continue
    local ONNX="$BENCH_DIR/$ONNX_REL"
    local VNNLIB="$BENCH_DIR/$VNNLIB_REL"
    local ONNX_TAG VNN_TAG
    ONNX_TAG=$(basename "$ONNX_REL" .onnx)
    VNN_TAG=$(basename "$VNNLIB_REL" .vnnlib)
    local IDX_PAD
    IDX_PAD=$(printf "%04d" "$idx")
    local STEM="${IDX_PAD}__${ONNX_TAG}__${VNN_TAG}"
    local RES="$OUT_DIR/$STEM.result"
    local LOG="$OUT_DIR/$STEM.log"
    local JSON="$OUT_DIR/$STEM.json"

    if [[ -f "$RES" && -s "$RES" ]]; then
      local prev
      prev=$(head -n1 "$RES" | tr -d '[:space:]')
      if [[ -n "$prev" ]]; then n_skip=$((n_skip+1)); continue; fi
    fi

    local USE_TO=${CSV_TIMEOUT%.*}
    if [[ "$TIMEOUT_CAP" -gt 0 && "$USE_TO" -gt "$TIMEOUT_CAP" ]]; then USE_TO=$TIMEOUT_CAP; fi
    if [[ -z "$USE_TO" || "$USE_TO" -le 0 ]]; then USE_TO=60; fi

    rm -f "$RES"
    local T0 T1 WALL RC
    T0=$(date +%s.%N)
    run_one "$BENCH" "$ONNX" "$VNNLIB" "$RES" "$LOG" "$USE_TO"
    RC=$?
    T1=$(date +%s.%N)
    WALL=$(awk "BEGIN{printf \"%.2f\", $T1-$T0}")

    local VERDICT_RAW=""
    if [[ -f "$RES" && -s "$RES" ]]; then
      VERDICT_RAW=$(head -n1 "$RES" | tr -d '[:space:]')
    fi
    local VERDICT
    case "$VERDICT_RAW" in
      unsat|verified|holds|safe)            VERDICT=unsat ;;
      sat|violated|falsified|unsafe)        VERDICT=sat ;;
      timeout|timed_out)                    VERDICT=timeout ;;
      unknown)                              VERDICT=unknown ;;
      unsupported_strict)                   VERDICT=unsupported_strict ;;
      error)                                VERDICT=error ;;
      "")                                   VERDICT=missing_result ;;
      *)                                    VERDICT="raw_$VERDICT_RAW" ;;
    esac
    # 124 = timeout SIGTERM, 137 = SIGKILL after grace
    if [[ "$RC" == "124" || "$RC" == "137" ]]; then
      VERDICT=timeout_killed
      echo "timeout" >"$RES"
    fi

    case "$VERDICT" in
      sat) n_sat=$((n_sat+1)) ;;
      unsat) n_unsat=$((n_unsat+1)) ;;
      timeout*) n_to=$((n_to+1)) ;;
      unknown) n_un=$((n_un+1)) ;;
      unsupported_strict) n_uns=$((n_uns+1)) ;;
      *) n_err=$((n_err+1)) ;;
    esac
    n_done=$((n_done+1))

    printf '{"idx":%d,"benchmark":"%s","onnx":"%s","vnnlib":"%s","csv_timeout":%s,"used_timeout":%s,"wall_sec":%s,"verdict_raw":"%s","verdict":"%s","exit_code":%d,"strict":true}\n' \
      "$idx" "$BENCH" "$ONNX_REL" "$VNNLIB_REL" "$CSV_TIMEOUT" "$USE_TO" "$WALL" "$VERDICT_RAW" "$VERDICT" "$RC" >"$JSON"
    printf '%d,"%s","%s",%s,%s,%s,"%s","%s",%d,"%s","%s"\n' \
      "$idx" "$ONNX_REL" "$VNNLIB_REL" "$CSV_TIMEOUT" "$USE_TO" "$WALL" "$VERDICT_RAW" "$VERDICT" "$RC" "$RES" "$LOG" >>"$SUMMARY"

    if (( idx % 5 == 0 )) || (( idx == NTOTAL )); then
      log "[$BENCH] $idx/$NTOTAL sat=$n_sat unsat=$n_unsat timeout=$n_to unknown=$n_un unsupported=$n_uns err=$n_err resumed=$n_skip"
    fi
  done <"$CSV"
  log "[$BENCH] DONE — total=$idx new=$n_done resumed=$n_skip sat=$n_sat unsat=$n_unsat timeout=$n_to unknown=$n_un unsupported=$n_uns err=$n_err"
}

log "=== NNV STRICT sweep start — patch verified, NNV_STRICT_NO_HELPER=1"
log "=== benches: ${BENCHES[*]}"
for B in "${BENCHES[@]}"; do
  run_benchmark "$B"
done
log "=== NNV STRICT sweep complete"
