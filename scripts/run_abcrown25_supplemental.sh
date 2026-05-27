#!/usr/bin/env bash
# Run abcrown vnncomp2025 fork on the 5 vnncomp25-new benchmarks that the
# GenBaB clone couldn't handle (no exp_configs/vnncomp25/ dir there).
#
# This runner is intended to be SUPPLEMENTAL to the main abcrown sweep
# in audit_results/abcrown_nopgd_20260525. Results go to a separate dir
# (audit_results/abcrown25_supplemental_<date>) so the two abcrown versions
# stay traceable.
#
# Tool: /data1/Kane/alpha-beta-CROWN_vnncomp2025  (commit 61b5ff8)
# Conda env: abcrown25 (Python 3.11 + torch 2.9.1+cu128 + fork's onnx2pytorch)
# Patch applied: auto_LiRPA/parse_graph.py wraps torch.onnx._globals import
#                in try/except (removed in torch 2.7+).
#
# All attack paths disabled via --NOPGD (translates to --pgd_order=skip).
#
set -uo pipefail

FORK_DIR=${FORK_DIR:-/data1/Kane/alpha-beta-CROWN_vnncomp2025}
BENCH_ROOT=${BENCH_ROOT:-/data1/Kane/data/vnncomp2025_benchmarks/benchmarks}
RESULTS_ROOT=${RESULTS_ROOT:-/data1/Kane/ACT/audit_results/abcrown25_supplemental_20260526}
PY_BIN=${PY_BIN:-/data1/Kane/miniconda3/envs/abcrown25/bin/python}
TIMEOUT_CAP=${ABCROWN25_TIMEOUT_CAP:-0}
KILL_GRACE=${ABCROWN25_KILL_GRACE:-90}
ONLY_BENCH=${1:-}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONPATH="${FORK_DIR}:${PYTHONPATH:-}"
export GRB_LICENSE_FILE=${GRB_LICENSE_FILE:-/data1/Kane/ACT/modules/gurobi/gurobi.lic}

mkdir -p "$RESULTS_ROOT"
DRIVER_LOG="$RESULTS_ROOT/_run.log"
META_JSON="$RESULTS_ROOT/_run.meta.json"

log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$DRIVER_LOG"; }

# Default benchmark set: only the 5 vnncomp25-new benchmarks the GenBaB
# clone couldn't handle. Override with arg 1 to run a different subset.
# Skipping lsnc_relu because its ONNX uses aten::ATen sum which
# auto_LiRPA can't bound, and its yaml relies on pgd_order=before to
# work around that — incompatible with --NOPGD.
BENCH_ORDER=(
  cersyve
  malbeware
  relusplitter
  sat_relu
  soundnessbench
)

cat >"$META_JSON" <<EOF
{
  "tool": "alpha-beta-CROWN (vnncomp2025 fork)",
  "tool_dir": "$FORK_DIR",
  "tool_commit": "$(git -C "$FORK_DIR" rev-parse HEAD 2>/dev/null || echo unknown)",
  "auto_lirpa_patch": "parse_graph.py: try/except around torch.onnx._globals import",
  "python": "$PY_BIN",
  "python_version": "$($PY_BIN --version 2>&1 | head -1)",
  "torch_version": "$($PY_BIN -c 'import torch; print(torch.__version__)' 2>/dev/null)",
  "cuda_available": $($PY_BIN -c 'import torch; print("true" if torch.cuda.is_available() else "false")' 2>/dev/null),
  "started_at": "$(date -Iseconds)",
  "host": "$(hostname)",
  "bench_root": "$BENCH_ROOT",
  "results_root": "$RESULTS_ROOT",
  "flags": {
    "NOPGD": true,
    "pgd_order_passed_through_vnncomp_main": "skip",
    "purpose": "supplemental vnncomp25-new benchmarks that GenBaB clone could not handle",
    "lsnc_relu_skipped": "fork's auto_LiRPA also can't bound aten::ATen sum; the official yaml uses pgd_order=before to bypass via PGD-found witness, incompatible with --NOPGD"
  }
}
EOF

log "=== abcrown vnncomp2025-FORK supplemental sweep, --NOPGD"
log "results -> $RESULTS_ROOT"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

run_benchmark() {
  local BENCH=$1
  local CAT=$BENCH       # fork's vnncomp_main accepts the folder name directly
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
    local IDX_PAD=$(printf "%04d" "$idx")
    local STEM="${IDX_PAD}__${ONNX_TAG}__${VNN_TAG}"
    local RES="$OUT_DIR/$STEM.result"
    local LOG="$OUT_DIR/$STEM.log"
    local JSON="$OUT_DIR/$STEM.json"

    if [[ -f "$RES" && -s "$RES" ]]; then
      local prev=$(head -n1 "$RES" | tr -d '[:space:]')
      if [[ -n "$prev" ]]; then n_skip=$((n_skip+1)); continue; fi
    fi

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
      "$PY_BIN" "$FORK_DIR/complete_verifier/vnncomp_main.py" \
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

    printf '{"idx":%d,"benchmark":"%s","category":"%s","onnx":"%s","vnnlib":"%s","csv_timeout":%s,"used_timeout":%s,"wall_sec":%s,"verdict":"%s","exit_code":%d,"pgd_disabled":true,"tool":"abcrown_vnncomp2025_fork"}\n' \
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
