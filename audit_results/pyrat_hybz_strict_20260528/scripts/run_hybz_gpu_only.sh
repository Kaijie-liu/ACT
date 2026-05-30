#!/bin/bash
# HYB_Z STRICT — GPU-ONLY phase supervisor.
#
# Runs only the 16 hyb_z benches whose .ini specifies device="cuda" or
# library="torch". CPU-only benches (acasxu, sat_relu, dist_shift,
# tllverifybench, collins_rul_cnn, lsnc_relu, nn4sys, collins_aerospace,
# cersyve, soundnessbench... wait some are torch.
#
# CPU-only benches DEFERRED to scripts/run_hybz_cpu_only.sh (run after NNV finishes).
#
# This supervisor runs ONE bench at a time, workers=1 per bench, so the GPU
# is fully dedicated and no concurrent supervisor races for VRAM.
#
# Usage:
#   scripts/run_hybz_gpu_only.sh [gpu_id]    # default 0

set -u
GPU="${1:-0}"
PYRAT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PYRAT_DIR"
LOG="$PYRAT_DIR/results_pure_hybz/_run_gpu_only.log"
mkdir -p "$(dirname "$LOG")"

GATE_MIN_GB="${PYRAT_HYBZ_GATE_MIN_GB:-30}"
GATE_INTERVAL="${PYRAT_HYBZ_GATE_INTERVAL_S:-30}"
GATE_TIMEOUT="${PYRAT_HYBZ_GATE_TIMEOUT_S:-7200}"

date | tee -a "$LOG"
echo "=== HYB_Z GPU-only sweep   gate>=${GATE_MIN_GB}GB ===" | tee -a "$LOG"

# 16 GPU benches (device=cuda OR library=torch), ordered light->heavy by N.
# cctsdb_yolo_2023 is CPU-only in hyb_z config and deferred to run_hybz_cpu_only.sh.
GPU_BENCHES=(
  test                                # 5    (library=torch)
  vggnet16_2022                       # 18   (cuda+torch)
  cgan_2023                           # 21   (cuda+torch)
  tllverifybench_2023                 # 32   (cuda+flexible)
  traffic_signs_recognition_2023      # 45   (cuda+torch)
  soundnessbench                      # 50   (library=torch)
  collins_rul_cnn_2022                # 62   (library=torch)
  ml4acopf_2024                       # 69   (cuda+flexible)
  yolo_2023                           # 72   (cuda+torch)
  metaroom_2023                       # 100  (cuda+torch)
  malbeware                           # 150  (cuda+torch)
  cora_2024                           # 180  (cuda+torch)
  cifar100_2024                       # 200  (cuda+torch)
  tinyimagenet_2024                   # 200  (cuda+torch)
  vit_2023                            # 200  (cuda+torch)
  relusplitter                        # 220  (cuda+torch)
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
  echo "    [gate] $label: budget exhausted, proceeding anyway" | tee -a "$LOG"
}

run_bench () {
  local b="$1"
  if is_complete "$b"; then
    echo "=== $(date '+%H:%M:%S')  [gpu] bench=$b  SKIP (complete) ===" | tee -a "$LOG"; return 0
  fi
  gpu_gate "$GATE_MIN_GB" "$b"
  echo "=== $(date '+%H:%M:%S')  [gpu] bench=$b  workers=1  gpu=$GPU ===" | tee -a "$LOG"
  python "$PYRAT_DIR/scripts/run_hybz_benchmark.py" "$b" \
      --workers 1 --gpu "$GPU" 2>&1 | tee -a "$LOG"
}

for b in "${GPU_BENCHES[@]}"; do run_bench "$b"; done

echo "=== $(date)  HYB_Z GPU-only DONE ===" | tee -a "$LOG"
