#!/usr/bin/env bash
# Parallel retry of missing vit_2023 instances on fork.
# Splits missing-result instances into N chunks and runs them concurrently.
# Each chunk is a single abcrown sub-process that loops over its instances.
#
# Args: $1 = N (chunks; default 2)
# Env:  CHUNK_INDEX, CHUNK_TOTAL are passed into the python loop.
set -uo pipefail

FORK_DIR=${FORK_DIR:-/data1/Kane/alpha-beta-CROWN_vnncomp2025}
BENCH_ROOT=${BENCH_ROOT:-/data1/Kane/data/vnncomp2025_benchmarks/benchmarks}
RESULTS_ROOT=${RESULTS_ROOT:-/data1/Kane/ACT/audit_results/abcrown_nopgd_20260525}
PY_BIN=${PY_BIN:-/data1/Kane/miniconda3/envs/abcrown25/bin/python}
N_CHUNKS=${1:-2}
KILL_GRACE=${ABCROWN25_KILL_GRACE:-90}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONPATH="${FORK_DIR}:${PYTHONPATH:-}"
export GRB_LICENSE_FILE=${GRB_LICENSE_FILE:-/data1/Kane/ACT/modules/gurobi/gurobi.lic}

bench=vit_2023
bdir="$BENCH_ROOT/$bench"
csv="$bdir/instances.csv"
out_dir="$RESULTS_ROOT/$bench"
mkdir -p "$out_dir"

# Collect missing-result instances (those whose .result file doesn't exist or is empty)
missing_file=$(mktemp /tmp/vit_missing.XXXXXX.csv)
trap "rm -f '$missing_file'" EXIT
idx=0
while IFS=, read -r onnx_rel vnnlib_rel csv_timeout REST; do
  idx=$((idx+1))
  onnx_rel="${onnx_rel//$'\r'/}"
  vnnlib_rel="${vnnlib_rel//$'\r'/}"
  csv_timeout="${csv_timeout//[[:space:]]/}"
  onnx_tag=$(basename "$onnx_rel" .onnx)
  vnn_tag=$(basename "$vnnlib_rel" .vnnlib)
  stem=$(printf "%04d__%s__%s" "$idx" "$onnx_tag" "$vnn_tag")
  res="$out_dir/$stem.result"
  if [[ -f "$res" && -s "$res" ]]; then continue; fi
  printf '%d,%s,%s,%s\n' "$idx" "$onnx_rel" "$vnnlib_rel" "${csv_timeout%.*}" >> "$missing_file"
done <"$csv"

n_missing=$(wc -l <"$missing_file")
echo "[vit_2023 parallel retry] missing=$n_missing  chunks=$N_CHUNKS"
if (( n_missing == 0 )); then
  echo "nothing to retry"; exit 0
fi

# Split into N chunks (round-robin by line index for balanced load)
mkdir -p "$RESULTS_ROOT/_vit_parallel"
for i in $(seq 0 $((N_CHUNKS-1))); do
  chunk_file="$RESULTS_ROOT/_vit_parallel/chunk_${i}_of_${N_CHUNKS}.csv"
  awk -v c=$i -v n=$N_CHUNKS 'NR % n == c' "$missing_file" > "$chunk_file"
  echo "  chunk $i: $(wc -l <"$chunk_file") instances -> $chunk_file"
done

run_chunk() {
  local chunk_idx=$1
  local chunk_file=$2
  local chunk_log="$RESULTS_ROOT/_vit_parallel/chunk_${chunk_idx}_driver.log"

  while IFS=, read -r idx onnx_rel vnnlib_rel csv_timeout; do
    [[ -z "$idx" ]] && continue
    # Strip any whitespace/CR (instances.csv may have CRLF line endings)
    idx="${idx//[[:space:]]/}"
    onnx_rel="${onnx_rel//$'\r'/}"
    vnnlib_rel="${vnnlib_rel//$'\r'/}"
    csv_timeout="${csv_timeout//[[:space:]]/}"
    local onnx="$bdir/$onnx_rel"
    local vnnlib="$bdir/$vnnlib_rel"
    local onnx_tag=$(basename "$onnx_rel" .onnx)
    local vnn_tag=$(basename "$vnnlib_rel" .vnnlib)
    local IDX_PAD=$(printf "%04d" "$idx")
    local stem="${IDX_PAD}__${onnx_tag}__${vnn_tag}"
    local res="$out_dir/$stem.result"
    local log="$out_dir/$stem.log"
    local json="$out_dir/$stem.json"
    local USE_TO=$csv_timeout
    local KILL_AT=$((USE_TO + KILL_GRACE))

    rm -f "$res"
    local T0=$(date +%s.%N)
    set +e
    timeout --kill-after=10 "${KILL_AT}s" \
      "$PY_BIN" "$FORK_DIR/complete_verifier/vnncomp_main.py" \
        "vit_2023" "$onnx" "$vnnlib" "$res" "$USE_TO" --NOPGD >"$log" 2>&1
    local RC=$?
    set -e
    local T1=$(date +%s.%N)
    local WALL=$(awk "BEGIN{printf \"%.2f\", $T1-$T0}")

    local VERDICT="missing_result"
    if [[ -f "$res" && -s "$res" ]]; then
      VERDICT=$(head -n1 "$res" | tr -d '[:space:]')
    fi
    if [[ "$RC" == "124" || "$RC" == "137" ]]; then VERDICT="timeout_killed"; fi

    printf '{"idx":%s,"benchmark":"vit_2023","category":"vit_2023","onnx":"%s","vnnlib":"%s","csv_timeout":%s,"used_timeout":%s,"wall_sec":%s,"verdict":"%s","exit_code":%d,"pgd_disabled":true,"tool":"abcrown_vnncomp2025_fork","chunk":%d}\n' \
      "$idx" "$onnx_rel" "$vnnlib_rel" "$csv_timeout" "$USE_TO" "$WALL" "$VERDICT" "$RC" "$chunk_idx" > "$json"

    echo "[chunk $chunk_idx idx=$idx] wall=${WALL}s verdict=$VERDICT" >> "$chunk_log"
  done <"$chunk_file"
  echo "chunk $chunk_idx DONE" >> "$chunk_log"
}

# Launch N chunks in parallel
chunk_pids=()
for i in $(seq 0 $((N_CHUNKS-1))); do
  run_chunk $i "$RESULTS_ROOT/_vit_parallel/chunk_${i}_of_${N_CHUNKS}.csv" &
  chunk_pids+=($!)
done
echo "chunk PIDs: ${chunk_pids[*]}"
wait
echo "ALL CHUNKS DONE"
