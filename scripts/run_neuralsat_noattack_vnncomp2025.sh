#!/usr/bin/env bash
# Run NeuralSAT on all VNN-COMP 2025 benchmarks with attack disabled.
#
# Per-instance: invokes src/main.py with --disable_attack, which sets
# Settings.use_attack=False (gates PGD + random + MIP attackers in
# verifier/utils.py and main_dec.py). Analog of abcrown's --NOPGD.
#
# Resumable: skips instances whose .result file exists with a non-empty
# first line. Delete the instance result file to force re-run.
#
# Outputs (same layout as the abcrown runner):
#   $RESULTS_ROOT/$BENCH/<idx>__<onnx>__<vnnlib>.result        verdict
#   $RESULTS_ROOT/$BENCH/<idx>__<onnx>__<vnnlib>.log           stdout+stderr
#   $RESULTS_ROOT/$BENCH/<idx>__<onnx>__<vnnlib>.json          metadata
#   $RESULTS_ROOT/$BENCH/_summary.csv                          per-benchmark
#   $RESULTS_ROOT/_run.log                                     driver log
#   $RESULTS_ROOT/_run.meta.json                               provenance
#
set -uo pipefail

NEURALSAT_DIR=${NEURALSAT_DIR:-/data1/Kane/neuralsat}
BENCH_ROOT=${BENCH_ROOT:-/data1/Kane/data/vnncomp2025_benchmarks/benchmarks}
RESULTS_ROOT=${RESULTS_ROOT:-/data1/Kane/ACT/audit_results/neuralsat_noattack_20260525}
PY_BIN=${PY_BIN:-/data1/Kane/miniconda3/envs/neuralsat/bin/python}
TIMEOUT_CAP=${NEURALSAT_TIMEOUT_CAP:-0}
KILL_GRACE=${NEURALSAT_KILL_GRACE:-90}
# Block startup until VRAM headroom available (MiB). 0 = don't block.
WAIT_VRAM_MIB=${NEURALSAT_WAIT_VRAM_MIB:-0}
ONLY_BENCH=${1:-}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# NeuralSAT expects gurobi lic at ~/gurobi.lic per its run_instance.sh.
# Inherit any pre-set GRB_LICENSE_FILE; otherwise fall back to ~/gurobi.lic.
export GRB_LICENSE_FILE=${GRB_LICENSE_FILE:-$HOME/gurobi.lic}

mkdir -p "$RESULTS_ROOT"
DRIVER_LOG="$RESULTS_ROOT/_run.log"
META_JSON="$RESULTS_ROOT/_run.meta.json"

log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$DRIVER_LOG"; }

# Optional: wait until VRAM is free (avoid stepping on a concurrent run).
if [[ "$WAIT_VRAM_MIB" -gt 0 ]]; then
  while :; do
    FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [[ "$FREE_MIB" -ge "$WAIT_VRAM_MIB" ]]; then break; fi
    log "Waiting for GPU: free=${FREE_MIB} MiB, need=${WAIT_VRAM_MIB} MiB"
    sleep 60
  done
fi

# NeuralSAT doesn't need per-benchmark configs — it auto-handles the spec.
# All 2025 benchmarks are runnable in principle (will be marked errored if
# NeuralSAT genuinely can't handle the architecture).
BENCH_ORDER=(
  test
  vggnet16_2022 collins_aerospace_benchmark cersyve cgan_2023
  tllverifybench_2023 cctsdb_yolo_2023 traffic_signs_recognition_2023
  collins_rul_cnn_2022 linearizenn_2024 ml4acopf_2024 dist_shift_2023
  yolo_2023 lsnc_relu soundnessbench
  metaroom_2023 cora_2024 relusplitter sat_relu malbeware
  acasxu_2023 nn4sys
  vit_2023 tinyimagenet_2024 cifar100_2024
  safenlp_2024
)

cat >"$META_JSON" <<EOF
{
  "tool": "NeuralSAT",
  "tool_dir": "$NEURALSAT_DIR",
  "tool_commit": "$(git -C "$NEURALSAT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)",
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
    "disable_attack": true,
    "use_attack_resolved_to": false,
    "TIMEOUT_CAP_SEC": $TIMEOUT_CAP,
    "KILL_GRACE_SEC": $KILL_GRACE
  }
}
EOF

log "=== NeuralSAT VNN-COMP 2025 sweep, --disable_attack, results -> $RESULTS_ROOT"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
log "torch: $($PY_BIN -c 'import torch; print(torch.__version__, "cuda=" + str(torch.cuda.is_available()))' 2>&1 | tail -1)"

run_benchmark() {
  local BENCH=$1
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
  log "[$BENCH] instances=$NTOTAL"

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
    local CEX="$OUT_DIR/$STEM.cex"

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

    rm -f "$RES"
    local T0=$(date +%s.%N)
    set +e
    timeout --kill-after=10 "${KILL_AT}s" \
      "$PY_BIN" "$NEURALSAT_DIR/src/main.py" \
        --net "$ONNX" \
        --spec "$VNNLIB" \
        --timeout "$USE_TO" \
        --result_file "$RES" \
        --export_cex \
        --disable_attack >"$LOG" 2>&1
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

    printf '{"idx":%d,"benchmark":"%s","onnx":"%s","vnnlib":"%s","csv_timeout":%s,"used_timeout":%s,"wall_sec":%s,"verdict":"%s","exit_code":%d,"attack_disabled":true}\n' \
      "$idx" "$BENCH" "$ONNX_REL" "$VNNLIB_REL" "$CSV_TIMEOUT" "$USE_TO" "$WALL" "$VERDICT" "$RC" >"$JSON"

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
