#!/usr/bin/env bash
# Run alpha-beta-CROWN on all VNN-COMP 2025 benchmarks with PGD disabled.
#
# Per-instance: invokes complete_verifier/vnncomp_main.py with --NOPGD,
# which sets attack.pgd_order=skip (this auto-disables MIP adv_warmup and
# input-BaB check_adv per upstream).
#
# Resumable: skips instances whose .result file already exists with a
# non-empty first line. Delete the instance result file to force re-run.
#
# Outputs:
#   $RESULTS_ROOT/$BENCH/<idx>__<onnx>__<vnnlib>.result        verdict line
#   $RESULTS_ROOT/$BENCH/<idx>__<onnx>__<vnnlib>.log           full stdout+stderr
#   $RESULTS_ROOT/$BENCH/<idx>__<onnx>__<vnnlib>.json          per-instance metadata
#   $RESULTS_ROOT/$BENCH/_summary.csv                          benchmark-level CSV
#   $RESULTS_ROOT/_run.log                                     top-level driver log
#   $RESULTS_ROOT/_run.meta.json                               run-level provenance
#
# Usage:
#   ./run_abcrown_nopgd_vnncomp2025.sh                 # full sweep
#   ./run_abcrown_nopgd_vnncomp2025.sh test            # only `test` benchmark
#   ABCROWN_TIMEOUT_CAP=60 ./run_abcrown_nopgd_vnncomp2025.sh   # cap each
#                                                                instance to 60s
#
set -uo pipefail

ABCROWN_DIR=${ABCROWN_DIR:-/data1/Kane/GenBaB/alpha-beta-CROWN}
BENCH_ROOT=${BENCH_ROOT:-/data1/Kane/data/vnncomp2025_benchmarks/benchmarks}
RESULTS_ROOT=${RESULTS_ROOT:-/data1/Kane/ACT/audit_results/abcrown_nopgd_20260525}
PY_BIN=${PY_BIN:-/data1/Kane/miniconda3/envs/GenBaB/bin/python}
# Cap per-instance timeout (seconds). 0 = use whatever instances.csv says.
TIMEOUT_CAP=${ABCROWN_TIMEOUT_CAP:-0}
# Hard wall-clock kill buffer beyond the abcrown timeout (seconds).
KILL_GRACE=${ABCROWN_KILL_GRACE:-90}
# Only run benchmarks whose name is in this whitespace-separated list.
ONLY_BENCH=${1:-}

# Pin GPU; abcrown will pick up CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONPATH="${ABCROWN_DIR}:${PYTHONPATH:-}"

mkdir -p "$RESULTS_ROOT"
DRIVER_LOG="$RESULTS_ROOT/_run.log"
META_JSON="$RESULTS_ROOT/_run.meta.json"

log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$DRIVER_LOG"; }

# benchmark folder -> abcrown CATEGORY name (defined by vnncomp_main.py)
# unsupported entries map to NONE and are skipped with a logged note.
declare -A CAT_MAP=(
  [acasxu_2023]=acasxu_2023
  [cctsdb_yolo_2023]=cctsdb_yolo_2023
  [cersyve]=NONE
  [cgan_2023]=cgan_2023
  [cifar100_2024]=cifar100
  [collins_aerospace_benchmark]=collins_aerospace_benchmark
  [collins_rul_cnn_2022]=collins_rul_cnn
  [cora_2024]=cora
  [dist_shift_2023]=dist_shift_2023
  [linearizenn_2024]=linearizenn
  [lsnc_relu]=lsnc
  [malbeware]=NONE
  [metaroom_2023]=metaroom_2023
  [ml4acopf_2024]=ml4acopf_2024
  [nn4sys]=nn4sys
  [relusplitter]=NONE
  [safenlp_2024]=safenlp_2024
  [sat_relu]=NONE
  [soundnessbench]=NONE
  [test]=test
  [tinyimagenet_2024]=tinyimagenet
  [tllverifybench_2023]=tllverifybench_2023
  [traffic_signs_recognition_2023]=traffic_signs_recognition_2023
  [vggnet16_2022]=vggnet16_2022
  [vit_2023]=vit_2023
  [yolo_2023]=yolo_2023
)

# Order: small/fast benchmarks first to validate pipeline, then medium, then large.
# vggnet16_2022 and collins_aerospace_benchmark are pushed to the END because:
#   - vggnet16_2022: vgg16-7.onnx not in local benchmark distribution (would need
#     to download ~528MB from ONNX zoo or rerun setup.sh's sciebo download).
#   - collins_aerospace_benchmark: YOLO bound prop requires >5GiB single
#     allocation on top of base usage — OOMs on this GPU at batch_size=1.
# Both will be recorded as errored in metadata; remove from this list if you
# don't want them in the output at all.
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

# Write top-level provenance once.
cat >"$META_JSON" <<EOF
{
  "tool": "alpha-beta-CROWN",
  "tool_dir": "$ABCROWN_DIR",
  "tool_commit": "$(git -C "$ABCROWN_DIR" rev-parse HEAD 2>/dev/null || echo unknown)",
  "python": "$PY_BIN",
  "python_version": "$($PY_BIN --version 2>&1 | head -1)",
  "torch_version": "$($PY_BIN -c 'import torch; print(torch.__version__)' 2>/dev/null)",
  "cuda_available": $($PY_BIN -c 'import torch; print("true" if torch.cuda.is_available() else "false")' 2>/dev/null),
  "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)",
  "started_at": "$(date -Iseconds)",
  "host": "$(hostname)",
  "bench_root": "$BENCH_ROOT",
  "results_root": "$RESULTS_ROOT",
  "flags": {
    "NOPGD": true,
    "pgd_order_passed_through_vnncomp_main": "skip",
    "TIMEOUT_CAP_SEC": $TIMEOUT_CAP,
    "KILL_GRACE_SEC": $KILL_GRACE
  }
}
EOF

log "=== abcrown VNN-COMP 2025 sweep, --NOPGD, results -> $RESULTS_ROOT"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
log "torch: $($PY_BIN -c 'import torch; print(torch.__version__, "cuda=" + str(torch.cuda.is_available()))' 2>&1 | tail -1)"

run_benchmark() {
  local BENCH=$1
  local CAT=${CAT_MAP[$BENCH]:-UNKNOWN}
  local BENCH_DIR=$BENCH_ROOT/$BENCH
  local CSV=$BENCH_DIR/instances.csv
  local OUT_DIR=$RESULTS_ROOT/$BENCH
  local SUMMARY=$OUT_DIR/_summary.csv

  if [[ "$CAT" == "NONE" ]]; then
    log "[$BENCH] SKIP — no abcrown CATEGORY mapping (benchmark not supported by GenBaB clone of αβ-CROWN; would need vnncomp2025 fork)"
    return 0
  fi
  if [[ "$CAT" == "UNKNOWN" ]]; then
    log "[$BENCH] SKIP — unknown benchmark folder"
    return 0
  fi
  if [[ ! -f "$CSV" ]]; then
    log "[$BENCH] SKIP — instances.csv not found"
    return 0
  fi

  mkdir -p "$OUT_DIR"
  if [[ ! -f "$SUMMARY" ]]; then
    echo "idx,onnx,vnnlib,csv_timeout,used_timeout,wall_sec,verdict,exit_code,result_file,log_file" >"$SUMMARY"
  fi
  local NTOTAL=$(wc -l <"$CSV")
  log "[$BENCH] CATEGORY=$CAT instances=$NTOTAL"

  local idx=0
  local n_done=0 n_skip=0 n_sat=0 n_unsat=0 n_to=0 n_err=0
  while IFS=, read -r ONNX_REL VNNLIB_REL CSV_TIMEOUT REST; do
    idx=$((idx+1))
    [[ -z "${ONNX_REL// }" ]] && continue
    local ONNX="$BENCH_DIR/$ONNX_REL"
    local VNNLIB="$BENCH_DIR/$VNNLIB_REL"
    local ONNX_TAG=$(basename "$ONNX_REL" .onnx)
    local VNN_TAG=$(basename "$VNNLIB_REL" .vnnlib)
    # zero-pad idx for stable sort
    local IDX_PAD=$(printf "%04d" "$idx")
    local STEM="${IDX_PAD}__${ONNX_TAG}__${VNN_TAG}"
    local RES="$OUT_DIR/$STEM.result"
    local LOG="$OUT_DIR/$STEM.log"
    local JSON="$OUT_DIR/$STEM.json"

    # Resume: skip if result file exists AND has a non-empty first line.
    if [[ -f "$RES" && -s "$RES" ]]; then
      local prev_verdict=$(head -n1 "$RES" | tr -d '[:space:]')
      if [[ -n "$prev_verdict" ]]; then
        n_skip=$((n_skip+1))
        continue
      fi
    fi

    # Some benchmark CSVs have decimal timeouts (e.g. "480.0").
    # Strip the fractional part so bash arithmetic works; abcrown also
    # accepts integer seconds via --timeout.
    local USE_TO=${CSV_TIMEOUT//[[:space:]]/}
    USE_TO=${USE_TO%.*}
    if [[ "$TIMEOUT_CAP" -gt 0 && "$USE_TO" -gt "$TIMEOUT_CAP" ]]; then
      USE_TO=$TIMEOUT_CAP
    fi
    local KILL_AT=$((USE_TO + KILL_GRACE))

    rm -f "$RES"
    local T0=$(date +%s.%N)
    set +e
    timeout --kill-after=10 "${KILL_AT}s" \
      "$PY_BIN" "$ABCROWN_DIR/complete_verifier/vnncomp_main.py" \
        "$CAT" "$ONNX" "$VNNLIB" "$RES" "$USE_TO" \
        --NOPGD >"$LOG" 2>&1
    local RC=$?
    set -e
    local T1=$(date +%s.%N)
    local WALL=$(awk "BEGIN{printf \"%.2f\", $T1-$T0}")

    local VERDICT
    if [[ -f "$RES" && -s "$RES" ]]; then
      VERDICT=$(head -n1 "$RES" | tr -d '[:space:]')
    else
      VERDICT="missing_result"
    fi
    # RC=124 from `timeout` => hard kill (treat as timeout)
    if [[ "$RC" == "124" || "$RC" == "137" ]]; then
      VERDICT="timeout_killed"
    fi

    case "$VERDICT" in
      sat) n_sat=$((n_sat+1)) ;;
      unsat) n_unsat=$((n_unsat+1)) ;;
      timeout*) n_to=$((n_to+1)) ;;
      unknown) n_to=$((n_to+1)) ;;
      *) n_err=$((n_err+1)) ;;
    esac
    n_done=$((n_done+1))

    printf '{"idx":%d,"benchmark":"%s","category":"%s","onnx":"%s","vnnlib":"%s","csv_timeout":%s,"used_timeout":%s,"wall_sec":%s,"verdict":"%s","exit_code":%d,"pgd_disabled":true}\n' \
      "$idx" "$BENCH" "$CAT" "$ONNX_REL" "$VNNLIB_REL" "$CSV_TIMEOUT" "$USE_TO" "$WALL" "$VERDICT" "$RC" >"$JSON"

    printf '%d,"%s","%s",%s,%s,%s,%s,%d,"%s","%s"\n' \
      "$idx" "$ONNX_REL" "$VNNLIB_REL" "$CSV_TIMEOUT" "$USE_TO" "$WALL" "$VERDICT" "$RC" "$RES" "$LOG" >>"$SUMMARY"

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
