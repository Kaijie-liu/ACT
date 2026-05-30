#!/bin/bash
# Run every benchmark PyRAT participated in (per arXiv-2512.19007v1) sequentially.
# Each benchmark uses its own internal parallelism (nb_process / CUDA).
#
# Categorisation:
#   TINY_CPU   – fully CPU, instances are tiny: workers=6
#   MID_GPU    – uses GPU but moderate memory: workers=2 + GPU gate (>= MID_MIN_GB)
#   BIG_GPU    – large memory (cifar100/vit/yolo/vggnet): workers=1 + GPU gate (>= BIG_MIN_GB)
#
# GPU gate: before each MID/BIG bench, poll `nvidia-smi` until enough VRAM
#   is free. Compatible with other GPU jobs (neuralsat/ACT) sharing the card.
#
# Resume: a benchmark is skipped if results_pure/<bench>/results.csv already
#   has exactly N data rows where N = lines in instances.csv.
#
# Usage:
#   scripts/run_all_2025.sh [gpu_id]      # default gpu_id=0
# Env knobs:
#   PYRAT_MID_MIN_GB        default 20
#   PYRAT_BIG_MIN_GB        default 40
#   PYRAT_GATE_INTERVAL_S   default 30
#   PYRAT_GATE_TIMEOUT_S    default 7200  (give up gating after 2h, run anyway)
#
# Output:
#   results_pure/<bench>/results.csv
#   results_pure/_run_all.log

set -u

GPU="${1:-0}"
PYRAT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PYRAT_DIR"
LOG="$PYRAT_DIR/results_pure/_run_all.log"
mkdir -p "$(dirname "$LOG")"

TINY_MIN_GB="${PYRAT_TINY_MIN_GB:-8}"
MID_MIN_GB="${PYRAT_MID_MIN_GB:-20}"
BIG_MIN_GB="${PYRAT_BIG_MIN_GB:-40}"
GATE_INTERVAL="${PYRAT_GATE_INTERVAL_S:-30}"
GATE_TIMEOUT="${PYRAT_GATE_TIMEOUT_S:-7200}"

date | tee -a "$LOG"
echo "=== gate thresholds: TINY>=${TINY_MIN_GB}GB  MID>=${MID_MIN_GB}GB  BIG>=${BIG_MIN_GB}GB  poll=${GATE_INTERVAL}s  budget=${GATE_TIMEOUT}s ===" | tee -a "$LOG"

TINY_CPU=(
  acasxu_2023
  sat_relu
  dist_shift_2023
  tllverifybench_2023
  collins_rul_cnn_2022
  lsnc_relu
  nn4sys
  collins_aerospace_benchmark
  test
)
MID_GPU=(
  cersyve
  soundnessbench
  malbeware
  linearizenn_2024
  ml4acopf_2024
  metaroom_2023
  safenlp_2024
  cora_2024
  relusplitter
  cgan_2023
)
BIG_GPU=(
  traffic_signs_recognition_2023
  cifar100_2024
  tinyimagenet_2024
  vit_2023
  vggnet16_2022
  yolo_2023
  cctsdb_yolo_2023
)

# --- helpers --------------------------------------------------------------

is_complete () {
  local b="$1"
  local csv="$PYRAT_DIR/results_pure/$b/results.csv"
  local inst="/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/$b/instances.csv"
  [ -f "$csv" ] || return 1
  [ -f "$inst" ] || return 1
  local done_rows total
  done_rows=$(( $(wc -l < "$csv") - 1 ))
  total=$(wc -l < "$inst")
  [ "$done_rows" -ge "$total" ]
}

free_gb () {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'
}

gpu_users () {
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader,nounits \
    | awk -F', ' '{printf "    pid=%-7s %4d MiB  %s\n", $1, $2, $3}'
}

gpu_gate () {
  local need_gb="$1" label="$2"
  local waited=0 fg
  fg=$(free_gb)
  if [ "$fg" -ge "$need_gb" ]; then
    echo "    [gate] $label: ${fg} GB free >= ${need_gb} GB, proceeding" | tee -a "$LOG"
    return 0
  fi
  echo "    [gate] $label: only ${fg} GB free, need ${need_gb} GB, waiting..." | tee -a "$LOG"
  gpu_users | tee -a "$LOG"
  while [ "$waited" -lt "$GATE_TIMEOUT" ]; do
    sleep "$GATE_INTERVAL"
    waited=$(( waited + GATE_INTERVAL ))
    fg=$(free_gb)
    if [ "$fg" -ge "$need_gb" ]; then
      echo "    [gate] $label: now ${fg} GB free (waited ${waited} s), proceeding" | tee -a "$LOG"
      return 0
    fi
  done
  echo "    [gate] $label: budget ${GATE_TIMEOUT}s exhausted, free=${fg} GB, proceeding anyway" | tee -a "$LOG"
  return 0
}

run_bench () {
  local b="$1" workers="$2" gate="$3"
  if is_complete "$b"; then
    echo "=== $(date '+%H:%M:%S')  bench=$b  SKIP (already complete) ===" | tee -a "$LOG"
    return 0
  fi
  if [ "$gate" -gt 0 ]; then
    gpu_gate "$gate" "$b"
  fi
  echo "=== $(date '+%H:%M:%S')  bench=$b  workers=$workers  gpu=$GPU ===" | tee -a "$LOG"
  python "$PYRAT_DIR/scripts/run_benchmark.py" "$b" \
      --workers "$workers" --gpu "$GPU" 2>&1 | tee -a "$LOG"
}

# --- run ------------------------------------------------------------------

for b in "${TINY_CPU[@]}"; do run_bench "$b" 6 "$TINY_MIN_GB"; done
for b in "${MID_GPU[@]}";  do run_bench "$b" 2 "$MID_MIN_GB"; done
for b in "${BIG_GPU[@]}";  do run_bench "$b" 1 "$BIG_MIN_GB"; done

echo "=== $(date)  ALL DONE ===" | tee -a "$LOG"
python "$PYRAT_DIR/scripts/aggregate_results.py" 2>&1 | tee -a "$LOG"
