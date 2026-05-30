#!/bin/bash
# HYB_Z STRICT — parallel small-GPU stream.
#
# Runs the 6 medium-sized GPU benches (N <= 200 each, T moderate) in
# REVERSE chain order, so it converges toward the chain supervisor from the
# tail. Both share is_complete / is_running gates and the GPU; the chain
# SKIPs anything this stream finishes first.
#
# Deliberately EXCLUDES: cifar100_2024, tinyimagenet_2024, vit_2023,
# relusplitter — these 4 are heavy ResNet/ViT/oval21 models where parallel
# hyb_z could OOM (con_z had 104 OOM under similar dual supervisor setup).
#
# Usage:
#   scripts/run_hybz_gpu_mid_parallel.sh [gpu_id]   # default 0

set -u
GPU="${1:-0}"
PYRAT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PYRAT_DIR"
LOG="$PYRAT_DIR/results_pure_hybz/_run_gpu_mid.log"
mkdir -p "$(dirname "$LOG")"

GATE_MIN_GB="${PYRAT_HYBZ_MID_MIN_GB:-25}"
GATE_INTERVAL="${PYRAT_HYBZ_GATE_INTERVAL_S:-30}"
GATE_TIMEOUT="${PYRAT_HYBZ_GATE_TIMEOUT_S:-7200}"

date | tee -a "$LOG"
echo "=== HYB_Z parallel GPU-mid stream  gate>=${GATE_MIN_GB}GB ===" | tee -a "$LOG"

# Reverse order from chain — meet chain in the middle of this list
GPU_MID_BENCHES=(
  cora_2024                           # 180  (cuda+torch)
  malbeware                           # 150  (cuda+torch)
  metaroom_2023                       # 100  (cuda+torch)
  yolo_2023                           # 72   (cuda+torch)
  ml4acopf_2024                       # 69   (cuda+flexible)
  collins_rul_cnn_2022                # 62   (library=torch)
)

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
    echo "    [gate-mid] $label: ${fg} GB free >= ${need_gb} GB, proceeding" | tee -a "$LOG"; return 0
  fi
  echo "    [gate-mid] $label: only ${fg} GB free, waiting..." | tee -a "$LOG"
  while [ "$waited" -lt "$GATE_TIMEOUT" ]; do
    sleep "$GATE_INTERVAL"; waited=$(( waited + GATE_INTERVAL )); fg=$(free_gb)
    if [ "$fg" -ge "$need_gb" ]; then
      echo "    [gate-mid] $label: now ${fg} GB free (waited ${waited} s)" | tee -a "$LOG"; return 0
    fi
  done
  echo "    [gate-mid] $label: budget exhausted, proceeding" | tee -a "$LOG"
}

run_one () {
  local b="$1"
  if is_complete "$b"; then
    echo "=== $(date '+%H:%M:%S')  [mid] bench=$b  SKIP (complete) ===" | tee -a "$LOG"; return 0
  fi
  if is_running "$b"; then
    echo "=== $(date '+%H:%M:%S')  [mid] bench=$b  SKIP (chain is on it) ===" | tee -a "$LOG"; return 0
  fi
  gpu_gate "$GATE_MIN_GB" "$b"
  echo "=== $(date '+%H:%M:%S')  [mid] bench=$b  workers=1  gpu=$GPU ===" | tee -a "$LOG"
  python "$PYRAT_DIR/scripts/run_hybz_benchmark.py" "$b" --workers 1 --gpu "$GPU" 2>&1 | tee -a "$LOG"
}

for b in "${GPU_MID_BENCHES[@]}"; do run_one "$b"; done
echo "=== $(date)  HYB_Z GPU-mid parallel DONE ===" | tee -a "$LOG"
