#!/bin/bash
# Run ONLY the BIG_GPU benchmarks in parallel with the main supervisor.
# Shares gate + skip-if-complete logic from run_all_2025.sh's vocabulary.
#
# Use case: while the main supervisor (`run_all_2025.sh`) is grinding through
# CPU-only TINY benches (acasxu/sat_relu/dist_shift), the GPU sits idle.
# This script picks up the heaviest GPU benches so we don't waste time.
# When the main supervisor later reaches BIG_GPU, skip-if-complete makes those
# benches no-ops.
#
# Order: heaviest GPU-memory first so failures (OOM) happen early.
#
# Usage:
#   scripts/run_big_only.sh [gpu_id]      # default gpu_id=0
# Env knobs (same defaults as run_all_2025.sh):
#   PYRAT_BIG_MIN_GB        default 40
#   PYRAT_GATE_INTERVAL_S   default 30
#   PYRAT_GATE_TIMEOUT_S    default 7200
#
# Output: appends to results_pure/_run_big.log

set -u

GPU="${1:-0}"
PYRAT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PYRAT_DIR"
LOG="$PYRAT_DIR/results_pure/_run_big.log"
mkdir -p "$(dirname "$LOG")"

BIG_MIN_GB="${PYRAT_BIG_MIN_GB:-40}"
GATE_INTERVAL="${PYRAT_GATE_INTERVAL_S:-30}"
GATE_TIMEOUT="${PYRAT_GATE_TIMEOUT_S:-7200}"

date | tee -a "$LOG"
echo "=== BIG-only supervisor   BIG>=${BIG_MIN_GB}GB  poll=${GATE_INTERVAL}s  budget=${GATE_TIMEOUT}s ===" | tee -a "$LOG"

# Same set as run_all_2025.sh's BIG_GPU, ordered heavy -> light so OOM bites first.
BIG_GPU=(
  cifar100_2024
  tinyimagenet_2024
  vit_2023
  yolo_2023
  vggnet16_2022
  traffic_signs_recognition_2023
  cctsdb_yolo_2023
)

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

is_running () {
  local b="$1"
  pgrep -af "scripts/run_benchmark\.py $b" >/dev/null 2>&1
}

free_gb () {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'
}

gpu_gate () {
  local need_gb="$1" label="$2"
  local waited=0 fg
  fg=$(free_gb)
  if [ "$fg" -ge "$need_gb" ]; then
    echo "    [gate-big] $label: ${fg} GB free >= ${need_gb} GB, proceeding" | tee -a "$LOG"
    return 0
  fi
  echo "    [gate-big] $label: only ${fg} GB free, need ${need_gb} GB, waiting..." | tee -a "$LOG"
  while [ "$waited" -lt "$GATE_TIMEOUT" ]; do
    sleep "$GATE_INTERVAL"
    waited=$(( waited + GATE_INTERVAL ))
    fg=$(free_gb)
    if [ "$fg" -ge "$need_gb" ]; then
      echo "    [gate-big] $label: now ${fg} GB free (waited ${waited} s), proceeding" | tee -a "$LOG"
      return 0
    fi
  done
  echo "    [gate-big] $label: budget ${GATE_TIMEOUT}s exhausted, proceeding anyway" | tee -a "$LOG"
}

run_one () {
  local b="$1"
  if is_complete "$b"; then
    echo "=== $(date '+%H:%M:%S')  [big] bench=$b  SKIP (already complete) ===" | tee -a "$LOG"
    return 0
  fi
  if is_running "$b"; then
    echo "=== $(date '+%H:%M:%S')  [big] bench=$b  SKIP (already running elsewhere) ===" | tee -a "$LOG"
    return 0
  fi
  gpu_gate "$BIG_MIN_GB" "$b"
  echo "=== $(date '+%H:%M:%S')  [big] bench=$b  workers=1  gpu=$GPU ===" | tee -a "$LOG"
  python "$PYRAT_DIR/scripts/run_benchmark.py" "$b" --workers 1 --gpu "$GPU" 2>&1 | tee -a "$LOG"
}

for b in "${BIG_GPU[@]}"; do run_one "$b"; done

echo "=== $(date)  BIG-only DONE ===" | tee -a "$LOG"
