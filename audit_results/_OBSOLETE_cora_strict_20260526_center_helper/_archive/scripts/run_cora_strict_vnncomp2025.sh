#!/usr/bin/env bash
# Launch CORA STRICT (pure verifier, no helpers) sweep with a GPU memory gate.
# Safe to run in parallel with other GPU workloads — will wait for free VRAM.
set -uo pipefail

MATLAB_BIN=${MATLAB_BIN:-/data1/Kane/MATLAB/bin/matlab}
RESULTS_ROOT=${RESULTS_ROOT:-/data1/Kane/ACT/audit_results/cora_strict_20260526}
ONLY_BENCH=${1:-}
TIMEOUT_CAP=${CORA_TIMEOUT_CAP:-0}
GPU_FREE_GB_MIN=${CORA_GPU_FREE_GB_MIN:-40}
GPU_CHECK_INTERVAL_S=${CORA_GPU_CHECK_INTERVAL_S:-30}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
mkdir -p "$RESULTS_ROOT"

"$MATLAB_BIN" -nodisplay -nosplash -batch \
  "addpath('/data1/Kane/ACT/scripts'); cora_strict_sweep_runner('', '$ONLY_BENCH', '$RESULTS_ROOT', $TIMEOUT_CAP, $GPU_FREE_GB_MIN, $GPU_CHECK_INTERVAL_S);"
