#!/bin/bash
# HYB_Z STRICT sweep — main supervisor.
# Per-instance timeouts come from each benchmark's instances.csv (VNN-COMP 2025 official).
#
# Categorisation (same as the strict run):
#   TINY_CPU – workers=6, GPU gate >= TINY_MIN_GB
#   MID_GPU  – workers=2, GPU gate >= MID_MIN_GB
#   BIG_GPU  – workers=1, GPU gate >= BIG_MIN_GB
#
# Resume: a benchmark is skipped if results_pure_hybz/<bench>/results.csv has all rows.
#
# Usage:
#   scripts/run_hybz_all_2025.sh [gpu_id]    # default 0
# Env knobs:
#   PYRAT_HYBZ_TINY_MIN_GB    default 8
#   PYRAT_HYBZ_MID_MIN_GB     default 20
#   PYRAT_HYBZ_BIG_MIN_GB     default 40
#   PYRAT_HYBZ_GATE_INTERVAL_S  default 30
#   PYRAT_HYBZ_GATE_TIMEOUT_S   default 7200

set -u
GPU="${1:-0}"
PYRAT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PYRAT_DIR"
LOG="$PYRAT_DIR/results_pure_hybz/_run_all.log"
mkdir -p "$(dirname "$LOG")"

TINY_MIN_GB="${PYRAT_HYBZ_TINY_MIN_GB:-8}"
MID_MIN_GB="${PYRAT_HYBZ_MID_MIN_GB:-20}"
BIG_MIN_GB="${PYRAT_HYBZ_BIG_MIN_GB:-40}"
GATE_INTERVAL="${PYRAT_HYBZ_GATE_INTERVAL_S:-30}"
GATE_TIMEOUT="${PYRAT_HYBZ_GATE_TIMEOUT_S:-7200}"

date | tee -a "$LOG"
echo "=== HYB_Z STRICT sweep  TINY>=${TINY_MIN_GB}GB  MID>=${MID_MIN_GB}GB  BIG>=${BIG_MIN_GB}GB ===" | tee -a "$LOG"

TINY_CPU=(
  acasxu_2023 sat_relu dist_shift_2023 tllverifybench_2023
  collins_rul_cnn_2022 lsnc_relu nn4sys collins_aerospace_benchmark test
)
MID_GPU=(
  cersyve soundnessbench malbeware linearizenn_2024 ml4acopf_2024
  metaroom_2023 safenlp_2024 cora_2024 relusplitter cgan_2023
)
BIG_GPU=(
  traffic_signs_recognition_2023 cifar100_2024 tinyimagenet_2024
  vit_2023 vggnet16_2022 yolo_2023 cctsdb_yolo_2023
)

is_complete () {
  local b="$1"
  local csv="$PYRAT_DIR/results_pure_hybz/$b/results.csv"
  local inst="/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/$b/instances.csv"
  [ -f "$csv" ] && [ -f "$inst" ] || return 1
  [ "$(($(wc -l < "$csv") - 1))" -ge "$(wc -l < "$inst")" ]
}

free_gb () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'; }

gpu_gate () {
  local need_gb="$1" label="$2" waited=0 fg
  fg=$(free_gb)
  if [ "$fg" -ge "$need_gb" ]; then
    echo "    [gate] $label: ${fg} GB free >= ${need_gb} GB, proceeding" | tee -a "$LOG"; return 0
  fi
  echo "    [gate] $label: only ${fg} GB free, need ${need_gb} GB, waiting..." | tee -a "$LOG"
  while [ "$waited" -lt "$GATE_TIMEOUT" ]; do
    sleep "$GATE_INTERVAL"; waited=$(( waited + GATE_INTERVAL )); fg=$(free_gb)
    if [ "$fg" -ge "$need_gb" ]; then
      echo "    [gate] $label: now ${fg} GB free (waited ${waited} s)" | tee -a "$LOG"; return 0
    fi
  done
  echo "    [gate] $label: budget exhausted, proceeding" | tee -a "$LOG"
}

run_bench () {
  local b="$1" workers="$2" gate="$3"
  if is_complete "$b"; then
    echo "=== $(date '+%H:%M:%S')  bench=$b  SKIP (already complete) ===" | tee -a "$LOG"; return 0
  fi
  [ "$gate" -gt 0 ] && gpu_gate "$gate" "$b"
  echo "=== $(date '+%H:%M:%S')  bench=$b  workers=$workers  gpu=$GPU ===" | tee -a "$LOG"
  python "$PYRAT_DIR/scripts/run_hybz_benchmark.py" "$b" \
      --workers "$workers" --gpu "$GPU" 2>&1 | tee -a "$LOG"
}

for b in "${TINY_CPU[@]}"; do run_bench "$b" 6 "$TINY_MIN_GB"; done
for b in "${MID_GPU[@]}";  do run_bench "$b" 2 "$MID_MIN_GB"; done
for b in "${BIG_GPU[@]}";  do run_bench "$b" 1 "$BIG_MIN_GB"; done

echo "=== $(date)  HYB_Z ALL DONE ===" | tee -a "$LOG"
