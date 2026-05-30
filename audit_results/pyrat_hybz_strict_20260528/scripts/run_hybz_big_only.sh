#!/bin/bash
# HYB_Z BIG-only parallel supervisor.
# Runs only the heavy GPU benches alongside the main supervisor's TINY+MID work.

set -u
GPU="${1:-0}"
PYRAT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PYRAT_DIR"
LOG="$PYRAT_DIR/results_pure_hybz/_run_big.log"
mkdir -p "$(dirname "$LOG")"

BIG_MIN_GB="${PYRAT_HYBZ_BIG_MIN_GB:-40}"
GATE_INTERVAL="${PYRAT_HYBZ_GATE_INTERVAL_S:-30}"
GATE_TIMEOUT="${PYRAT_HYBZ_GATE_TIMEOUT_S:-7200}"

date | tee -a "$LOG"
echo "=== HYB_Z BIG-only supervisor   BIG>=${BIG_MIN_GB}GB ===" | tee -a "$LOG"

BIG_GPU=(cifar100_2024 tinyimagenet_2024 vit_2023 yolo_2023
         vggnet16_2022 traffic_signs_recognition_2023 cctsdb_yolo_2023)

is_complete () {
  local b="$1"
  local csv="$PYRAT_DIR/results_pure_hybz/$b/results.csv"
  local inst="/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/$b/instances.csv"
  [ -f "$csv" ] && [ -f "$inst" ] || return 1
  [ "$(($(wc -l < "$csv") - 1))" -ge "$(wc -l < "$inst")" ]
}
is_running () { pgrep -af "scripts/run_hybz_benchmark\.py $1" >/dev/null 2>&1; }
free_gb () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'; }
gpu_gate () {
  local need_gb="$1" label="$2" waited=0 fg
  fg=$(free_gb)
  if [ "$fg" -ge "$need_gb" ]; then
    echo "    [gate-big] $label: ${fg} GB free, proceeding" | tee -a "$LOG"; return 0
  fi
  echo "    [gate-big] $label: only ${fg} GB free, waiting..." | tee -a "$LOG"
  while [ "$waited" -lt "$GATE_TIMEOUT" ]; do
    sleep "$GATE_INTERVAL"; waited=$(( waited + GATE_INTERVAL )); fg=$(free_gb)
    if [ "$fg" -ge "$need_gb" ]; then
      echo "    [gate-big] $label: now ${fg} GB free (waited ${waited} s)" | tee -a "$LOG"; return 0
    fi
  done
  echo "    [gate-big] $label: budget exhausted, proceeding" | tee -a "$LOG"
}
run_one () {
  local b="$1"
  if is_complete "$b"; then
    echo "=== $(date '+%H:%M:%S')  [big] bench=$b  SKIP (complete) ===" | tee -a "$LOG"; return 0
  fi
  if is_running "$b"; then
    echo "=== $(date '+%H:%M:%S')  [big] bench=$b  SKIP (running elsewhere) ===" | tee -a "$LOG"; return 0
  fi
  gpu_gate "$BIG_MIN_GB" "$b"
  echo "=== $(date '+%H:%M:%S')  [big] bench=$b  workers=1  gpu=$GPU ===" | tee -a "$LOG"
  python "$PYRAT_DIR/scripts/run_hybz_benchmark.py" "$b" --workers 1 --gpu "$GPU" 2>&1 | tee -a "$LOG"
}

for b in "${BIG_GPU[@]}"; do run_one "$b"; done
echo "=== $(date)  HYB_Z BIG-only DONE ===" | tee -a "$LOG"
