#!/bin/bash
# HYB_Z STRICT — CPU-only phase.
#
# Runs the 10 hyb_z benches whose .ini does NOT specify device="cuda" or
# library="torch" — i.e. they execute on CPU with numpy/flexible defaults.
# Sequential, workers=4 (modest CPU concurrency since each pyrat invocation
# already forks ~20 nb_process children internally).
#
# Should be launched only AFTER any concurrent CPU-heavy work (e.g. NNV
# MATLAB sweep) has freed the cores. The chain wrapper run_hybz_chain.sh
# enforces this ordering.
#
# Usage:
#   scripts/run_hybz_cpu_only.sh [gpu_id]   # gpu unused here, kept for symmetry

set -u
GPU="${1:-0}"
PYRAT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PYRAT_DIR"
LOG="$PYRAT_DIR/results_pure_hybz/_run_cpu_only.log"
mkdir -p "$(dirname "$LOG")"

date | tee -a "$LOG"
echo "=== HYB_Z CPU-only sweep   workers=4 per bench ===" | tee -a "$LOG"

CPU_BENCHES=(
  collins_aerospace_benchmark         # 6     fast (likely all unknown/error)
  cersyve                             # 12    zono
  cctsdb_yolo_2023                    # 39    exhaustive=True
  lsnc_relu                           # 80    T=25 short
  sat_relu                            # 100   zono
  linearizenn_2024                    # 60    zono
  dist_shift_2023                     # 72
  acasxu_2023                         # 186   T=116
  nn4sys                              # 194   T=800
  safenlp_2024                        # 1080  T=20 short
)

is_complete () {
  local b="$1"
  local csv="$PYRAT_DIR/results_pure_hybz/$b/results.csv"
  local inst="/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/$b/instances.csv"
  [ -f "$csv" ] && [ -f "$inst" ] || return 1
  [ "$(($(wc -l < "$csv") - 1))" -ge "$(wc -l < "$inst")" ]
}

run_bench () {
  local b="$1" workers="$2"
  if is_complete "$b"; then
    echo "=== $(date '+%H:%M:%S')  [cpu] bench=$b  SKIP (complete) ===" | tee -a "$LOG"; return 0
  fi
  echo "=== $(date '+%H:%M:%S')  [cpu] bench=$b  workers=$workers ===" | tee -a "$LOG"
  python "$PYRAT_DIR/scripts/run_hybz_benchmark.py" "$b" \
      --workers "$workers" --gpu "$GPU" 2>&1 | tee -a "$LOG"
}

for b in "${CPU_BENCHES[@]}"; do run_bench "$b" 4; done

echo "=== $(date)  HYB_Z CPU-only DONE ===" | tee -a "$LOG"
