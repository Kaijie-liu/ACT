#!/bin/bash
# Single-instance PyRAT runner for HYB_Z STRICT sweep.
# Uses vnn_config_2025_hybz/ ini and writes to results_pure_hybz/ tree.

BENCH="$1"
ONNX="$2"
VNNLIB="$3"
TIMEOUT_S="$4"
OUT_CSV="$5"

PYRAT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INI="${PYRAT_DIR}/benchmarks/vnn_files/vnn_config_2025_hybz/${BENCH}.ini"

if [[ ! -f "$INI" ]]; then
  echo "missing ini: $INI" >&2
  echo "${BENCH},${ONNX},${VNNLIB},error_no_ini,0,0,1" >> "$OUT_CSV"
  exit 1
fi

# shellcheck disable=SC1091
source /data1/Kane/miniconda3/etc/profile.d/conda.sh
conda activate pyrat >/dev/null 2>&1

if [[ "$ONNX" == *.gz ]]; then
  UNCOMP="${ONNX%.gz}"
  [[ ! -f "$UNCOMP" ]] && gzip -dk "$ONNX"
  ONNX="$UNCOMP"
fi

LOG_DIR="${PYRAT_DIR}/results_pure_hybz/${BENCH}/logs"
mkdir -p "$LOG_DIR"
TAG="$(basename "$ONNX" .onnx)__$(basename "$VNNLIB" .vnnlib)__$$"
LOG_OUT="${LOG_DIR}/${TAG}.out"
LOG_ERR="${LOG_DIR}/${TAG}.err"

START_NS=$(date +%s%N)
KILL_AFTER=$(( TIMEOUT_S + 30 ))

timeout --kill-after=15 "$KILL_AFTER" python "${PYRAT_DIR}/run_pure.py" --strict \
    --config "$INI" \
    --model_path "$ONNX" \
    --property_path "$VNNLIB" \
    --timeout "$TIMEOUT_S" \
    > "$LOG_OUT" 2> "$LOG_ERR"
RC=$?

END_NS=$(date +%s%N)
WALL=$(awk -v a="$START_NS" -v b="$END_NS" 'BEGIN{printf "%.3f",(b-a)/1e9}')

LINE=$(tr '\r' '\n' < "$LOG_OUT" | grep -E "^Result = " | tail -1)
VERDICT="error"; REPORTED="-1"
if [[ -n "$LINE" ]]; then
  RESULT_RAW=$(echo "$LINE" | sed -nE 's/^Result = ([^,]+),.*$/\1/p' | tr -d ' ')
  REPORTED=$(echo "$LINE" | sed -nE 's/.*Time = ([0-9.]+) s.*$/\1/p')
  case "$RESULT_RAW" in
    True)    VERDICT="verified" ;;
    False)   VERDICT="falsified" ;;
    Unknown) VERDICT="unknown" ;;
    Timeout) VERDICT="timeout" ;;
    *)       VERDICT="error" ;;
  esac
elif [[ "$RC" -eq 124 || "$RC" -eq 137 ]]; then
  VERDICT="timeout"
fi

echo "${BENCH},${ONNX},${VNNLIB},${VERDICT},${WALL},${REPORTED:-0},${RC}" >> "$OUT_CSV"
