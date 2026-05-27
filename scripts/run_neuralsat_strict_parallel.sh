#!/usr/bin/env bash
# NeuralSAT STRICT (no-attack) parallel sweep over VNN-COMP 2025.
# - `--disable_attack` is hardcoded -> Settings.use_attack=False, gating
#   _pre_attack / _mip_attack / _attack (audited 2026-05-27).
# - Two lanes by default, processing disjoint benchmark sets.
# - Before each benchmark, polls nvidia-smi and waits if free VRAM < NS_GPU_FREE_GB_MIN.
#
# Usage:
#   run_neuralsat_strict_parallel.sh                 # 2 lanes (default split)
#   run_neuralsat_strict_parallel.sh laneA           # only run Lane A
#   run_neuralsat_strict_parallel.sh laneB           # only run Lane B
#
set -uo pipefail

NEURALSAT_DIR=${NEURALSAT_DIR:-/data1/Kane/neuralsat}
BENCH_ROOT=${BENCH_ROOT:-/data1/Kane/data/vnncomp2025_benchmarks/benchmarks}
RESULTS_ROOT=${RESULTS_ROOT:-/data1/Kane/ACT/audit_results/neuralsat_strict_20260527}
PY_BIN=${PY_BIN:-/data1/Kane/miniconda3/envs/neuralsat/bin/python}
TIMEOUT_CAP=${NS_TIMEOUT_CAP:-0}
KILL_GRACE=${NS_KILL_GRACE:-90}
GPU_FREE_GB_MIN=${NS_GPU_FREE_GB_MIN:-30}
GPU_CHECK_INTERVAL_S=${NS_GPU_CHECK_INTERVAL_S:-30}
WHICH_LANE=${1:-}   # empty -> both; "laneA" or "laneB" -> only that lane

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONPATH="$NEURALSAT_DIR/src:${PYTHONPATH:-}"
export GRB_LICENSE_FILE=${GRB_LICENSE_FILE:-/data1/Kane/ACT/modules/gurobi/gurobi.lic}

mkdir -p "$RESULTS_ROOT"
META_PATH="$RESULTS_ROOT/_run.meta.json"
DRIVER_LOG="$RESULTS_ROOT/_run.log"

log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$DRIVER_LOG"; }

# --- BENCH ORDER: split into 2 disjoint lists, interleaved to balance load.
# Lane A: "evens" (indices 0,2,4,...). Lane B: "odds".
# Order is roughly light->heavy so a slow lane gets light first.
FULL_ORDER=(
  test
  cersyve
  lsnc_relu
  soundnessbench
  sat_relu
  cgan_2023
  tllverifybench_2023
  malbeware
  traffic_signs_recognition_2023
  collins_rul_cnn_2022
  linearizenn_2024
  dist_shift_2023
  ml4acopf_2024
  metaroom_2023
  cctsdb_yolo_2023
  yolo_2023
  relusplitter
  cora_2024
  acasxu_2023
  nn4sys
  safenlp_2024
  vit_2023
  tinyimagenet_2024
  cifar100_2024
  vggnet16_2022
  collins_aerospace_benchmark
)
LANE_A=()
LANE_B=()
for i in "${!FULL_ORDER[@]}"; do
  if (( i % 2 == 0 )); then LANE_A+=("${FULL_ORDER[i]}")
  else                       LANE_B+=("${FULL_ORDER[i]}")
  fi
done

# Write meta on first launch only
if [[ ! -f "$META_PATH" ]]; then
  cat >"$META_PATH" <<EOF
{
  "tool": "NeuralSAT (STRICT no-attack)",
  "tool_dir": "$NEURALSAT_DIR",
  "tool_commit": "$(git -C "$NEURALSAT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)",
  "python": "$PY_BIN",
  "python_version": "$($PY_BIN --version 2>&1 | head -1)",
  "torch_version": "$($PY_BIN -c 'import torch; print(torch.__version__)' 2>/dev/null)",
  "started_at": "$(date -Iseconds)",
  "host": "$(hostname)",
  "bench_root": "$BENCH_ROOT",
  "results_root": "$RESULTS_ROOT",
  "flags": {
    "disable_attack": true,
    "use_attack_resolved_to": false,
    "audited": "Settings.use_attack gates _pre_attack/_mip_attack/_attack in verifier; no other random sampling fallback for SAT",
    "TIMEOUT_CAP_SEC": $TIMEOUT_CAP,
    "KILL_GRACE_SEC": $KILL_GRACE,
    "GPU_FREE_GB_MIN": $GPU_FREE_GB_MIN,
    "GPU_CHECK_INTERVAL_S": $GPU_CHECK_INTERVAL_S
  }
}
EOF
fi

gpu_wait() {
  local need_mib=$((GPU_FREE_GB_MIN * 1024))
  local first=1
  while :; do
    local free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    [[ -z "$free" ]] && free=0
    if (( free >= need_mib )); then
      (( first == 0 )) && log "[GPU gate] free=${free}MiB OK, starting $1"
      return
    fi
    (( first == 1 )) && { log "[GPU gate] $1 waiting (free=${free}MiB < ${need_mib}MiB)"; first=0; }
    sleep "$GPU_CHECK_INTERVAL_S"
  done
}

run_benchmark() {
  local BENCH=$1 LANE_TAG=$2
  local BENCH_DIR=$BENCH_ROOT/$BENCH
  local CSV=$BENCH_DIR/instances.csv
  local OUT_DIR=$RESULTS_ROOT/$BENCH
  local SUMMARY=$OUT_DIR/_summary.csv

  if [[ ! -f "$CSV" ]]; then
    log "[$LANE_TAG][$BENCH] SKIP — instances.csv not found"; return 0
  fi

  gpu_wait "[$LANE_TAG][$BENCH]"

  mkdir -p "$OUT_DIR"
  if [[ ! -f "$SUMMARY" ]]; then
    echo "idx,onnx,vnnlib,csv_timeout,used_timeout,wall_sec,verdict_raw,verdict,exit_code,result_file,log_file" >"$SUMMARY"
  fi
  local NTOTAL=$(wc -l <"$CSV")
  log "[$LANE_TAG][$BENCH] instances=$NTOTAL"

  local idx=0
  local n_done=0 n_skip=0 n_sat=0 n_unsat=0 n_to=0 n_err=0
  while IFS=, read -r ONNX_REL VNNLIB_REL CSV_TIMEOUT REST; do
    idx=$((idx+1))
    ONNX_REL="${ONNX_REL//$'\r'/}"; VNNLIB_REL="${VNNLIB_REL//$'\r'/}"
    CSV_TIMEOUT="${CSV_TIMEOUT//[[:space:]]/}"
    [[ -z "${ONNX_REL// }" ]] && continue
    local ONNX="$BENCH_DIR/$ONNX_REL"
    local VNNLIB="$BENCH_DIR/$VNNLIB_REL"
    local ONNX_TAG=$(basename "$ONNX_REL" .onnx)
    local VNN_TAG=$(basename "$VNNLIB_REL" .vnnlib)
    local IDX_PAD=$(printf "%04d" "$idx")
    local STEM="${IDX_PAD}__${ONNX_TAG}__${VNN_TAG}"
    local RES="$OUT_DIR/$STEM.result"
    local RAW="$OUT_DIR/$STEM.raw"
    local LOG="$OUT_DIR/$STEM.log"
    local JSON="$OUT_DIR/$STEM.json"

    if [[ -f "$RES" && -s "$RES" ]]; then
      local prev=$(head -n1 "$RES" | tr -d '[:space:]')
      if [[ -n "$prev" ]]; then n_skip=$((n_skip+1)); continue; fi
    fi

    local USE_TO=${CSV_TIMEOUT%.*}
    if [[ "$TIMEOUT_CAP" -gt 0 && "$USE_TO" -gt "$TIMEOUT_CAP" ]]; then USE_TO=$TIMEOUT_CAP; fi
    local KILL_AT=$((USE_TO + KILL_GRACE))

    rm -f "$RES" "$RAW"
    local T0=$(date +%s.%N)
    # NOTE: script-level "set -e" is intentionally NOT used. Re-enabling -e here
    # historically caused gpu_wait's "[[ -z "$free" ]] && free=0" line to kill
    # the lane subshell silently when $free was non-empty. Just let RC propagate.
    timeout --kill-after=10 "${KILL_AT}s" \
      "$PY_BIN" "$NEURALSAT_DIR/src/main.py" \
        --net "$ONNX" --spec "$VNNLIB" \
        --timeout "$USE_TO" \
        --result_file "$RAW" \
        --export_cex \
        --disable_attack >"$LOG" 2>&1
    local RC=$?
    local T1=$(date +%s.%N)
    local WALL=$(awk "BEGIN{printf \"%.2f\", $T1-$T0}")

    local VERDICT_RAW=""
    if [[ -f "$RAW" && -s "$RAW" ]]; then
      VERDICT_RAW=$(head -n1 "$RAW" | tr -d '[:space:]')
    fi
    local VERDICT
    # NeuralSAT writes one of: unsat | sat | timeout | unknown to --result_file.
    # (Some versions use holds/violated; map both for safety.)
    case "$VERDICT_RAW" in
      unsat|holds)     VERDICT=unsat ;;
      sat|violated)    VERDICT=sat ;;
      timeout)         VERDICT=timeout ;;
      unknown)         VERDICT=unknown ;;
      error)           VERDICT=error ;;
      "")              VERDICT=missing_result ;;
      *)               VERDICT="raw_$VERDICT_RAW" ;;
    esac
    if [[ "$RC" == "124" || "$RC" == "137" ]]; then VERDICT=timeout_killed; fi
    echo "$VERDICT" >"$RES"

    case "$VERDICT" in
      sat) n_sat=$((n_sat+1)) ;;
      unsat) n_unsat=$((n_unsat+1)) ;;
      timeout*) n_to=$((n_to+1)) ;;
      unknown) n_to=$((n_to+1)) ;;
      *) n_err=$((n_err+1)) ;;
    esac
    n_done=$((n_done+1))

    printf '{"idx":%d,"benchmark":"%s","lane":"%s","onnx":"%s","vnnlib":"%s","csv_timeout":%s,"used_timeout":%s,"wall_sec":%s,"verdict_raw":"%s","verdict":"%s","exit_code":%d,"attack_disabled":true}\n' \
      "$idx" "$BENCH" "$LANE_TAG" "$ONNX_REL" "$VNNLIB_REL" "$CSV_TIMEOUT" "$USE_TO" "$WALL" "$VERDICT_RAW" "$VERDICT" "$RC" >"$JSON"
    printf '%d,"%s","%s",%s,%s,%s,"%s","%s",%d,"%s","%s"\n' \
      "$idx" "$ONNX_REL" "$VNNLIB_REL" "$CSV_TIMEOUT" "$USE_TO" "$WALL" "$VERDICT_RAW" "$VERDICT" "$RC" "$RES" "$LOG" >>"$SUMMARY"

    if (( idx % 10 == 0 )) || (( idx == NTOTAL )); then
      log "[$LANE_TAG][$BENCH] $idx/$NTOTAL sat=$n_sat unsat=$n_unsat timeout=$n_to err=$n_err resumed=$n_skip"
    fi
  done <"$CSV"
  log "[$LANE_TAG][$BENCH] DONE — total=$idx new=$n_done resumed=$n_skip sat=$n_sat unsat=$n_unsat timeout=$n_to err=$n_err"
}

run_lane() {
  local LANE_TAG=$1
  shift
  log "===[$LANE_TAG] start — $# benchmarks"
  for B in "$@"; do run_benchmark "$B" "$LANE_TAG"; done
  log "===[$LANE_TAG] sweep complete"
}

case "$WHICH_LANE" in
  laneA) run_lane laneA "${LANE_A[@]}" ;;
  laneB) run_lane laneB "${LANE_B[@]}" ;;
  *)
    # Run both lanes in parallel
    run_lane laneA "${LANE_A[@]}" &
    PA=$!
    run_lane laneB "${LANE_B[@]}" &
    PB=$!
    log "===dual-lane: A pid=$PA   B pid=$PB"
    wait $PA $PB
    log "===dual-lane: both done"
    ;;
esac
