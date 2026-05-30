#!/bin/bash
# Single-instance PyRAT runner (strict no-adv-search mode).
#
# Usage:
#   scripts/run_instance.sh <benchmark_name> <onnx_path> <vnnlib_path> <timeout_s> <out_csv>
#
# Writes one CSV row appended to <out_csv>:
#   benchmark,onnx,vnnlib,verdict,wall_s,reported_s,returncode
#
# Verdict mapping (PyRAT stdout "Result = X"):
#   True     -> verified  (analyzer proved the property)
#   False    -> falsified (would only occur if abstract analysis itself
#                          concludes UNSAFE; no adv search is used)
#   Unknown  -> unknown
#   Timeout  -> timeout
#   *        -> error

BENCH="$1"
ONNX="$2"
VNNLIB="$3"
TIMEOUT_S="$4"
OUT_CSV="$5"

PYRAT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# Use the EXACT competition-commit (4a9a4f065a6...) .ini files, with the two
# trivial parameter renames applied for HEAD-pyrat parser compatibility
# (nb_repeat -> nb_restart, step_ratio -> lr_attack). Semantically identical.
INI="${PYRAT_DIR}/benchmarks/vnn_files/vnn_config_2025_competition/${BENCH}.ini"

if [[ ! -f "$INI" ]]; then
  echo "missing ini: $INI" >&2
  echo "${BENCH},${ONNX},${VNNLIB},error_no_ini,0,0,1" >> "$OUT_CSV"
  exit 1
fi

# Conda env (pyrat needs Python 3.10). Activation script references unset
# variables, so do NOT use `set -u` here.
# shellcheck disable=SC1091
source /data1/Kane/miniconda3/etc/profile.d/conda.sh
conda activate pyrat >/dev/null 2>&1

# Decompress onnx if .gz (vnncomp convention)
if [[ "$ONNX" == *.gz ]]; then
  UNCOMP="${ONNX%.gz}"
  [[ ! -f "$UNCOMP" ]] && gzip -dk "$ONNX"
  ONNX="$UNCOMP"
fi

LOG_DIR="${PYRAT_DIR}/results_pure/${BENCH}/logs"
mkdir -p "$LOG_DIR"
# Tag includes shell PID so duplicate (onnx, vnnlib) rows in instances.csv
# (sat_relu has 2 such dups) don't race-overwrite each other's log files
# under parallel workers.
TAG="$(basename "$ONNX" .onnx)__$(basename "$VNNLIB" .vnnlib)__$$"
LOG_OUT="${LOG_DIR}/${TAG}.out"
LOG_ERR="${LOG_DIR}/${TAG}.err"

START_NS=$(date +%s%N)

# +30 s outer kill bound (PyRAT honours --timeout but give margin)
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

# PyRAT writes a tqdm-style progress bar with carriage returns and then
# appends "Result = X, Time = Y s, ..." on the SAME physical line. So
# `grep -E "^Result"` misses it. Strip \r -> \n first, then anchor the regex.
LINE=$(tr '\r' '\n' < "$LOG_OUT" | grep -E "^Result = " | tail -1)
VERDICT="error"
REPORTED="-1"
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
