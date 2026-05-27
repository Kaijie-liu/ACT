#!/usr/bin/env bash
# Wrapper that launches the in-MATLAB CORA sweep (all instances in one
# MATLAB session). CORA has no attack/PGD path in its codebase, so this
# is attack-free by construction.
set -uo pipefail

MATLAB_BIN=${MATLAB_BIN:-/data1/Kane/MATLAB/bin/matlab}
SWEEP_M=/data1/Kane/ACT/scripts/cora_sweep_runner.m
RESULTS_ROOT=${RESULTS_ROOT:-/data1/Kane/ACT/audit_results/cora_noattack_20260526}
ONLY_BENCH=${1:-}
TIMEOUT_CAP=${CORA_TIMEOUT_CAP:-0}
KILL_GRACE=${CORA_KILL_GRACE:-90}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
mkdir -p "$RESULTS_ROOT"

# Call the in-MATLAB driver. Pass args as MATLAB strings.
"$MATLAB_BIN" -nodisplay -nosplash -batch \
  "addpath('/data1/Kane/ACT/scripts'); cora_sweep_runner('', '$ONLY_BENCH', '$RESULTS_ROOT', $TIMEOUT_CAP, $KILL_GRACE);"
