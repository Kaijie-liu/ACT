#!/bin/bash
# ACT soundness test runner.
# Runs all per-file soundness suites and reports aggregate pass/fail.
#
# Conventions:
# - Test files use `_run_all()` (SmallDenseDAG and correlated pair cuts
#   suites) or expose top-level `test_*` functions
#   (test_hz_reduction_soundness, pytest-compatible).
# - This script invokes each style appropriately and returns non-zero on any
#   suite failure.
#
# Usage:
#   bash tests/run_soundness_tests.sh
set -u

PY=/data1/Kane/miniconda3/envs/act-py312/bin/python
cd "$(dirname "$0")/.."

PASS=0
FAIL=0

echo "=== tests/test_smalldense_dag_soundness.py ==="
if $PY tests/test_smalldense_dag_soundness.py; then
    PASS=$((PASS+1))
else
    FAIL=$((FAIL+1))
fi

echo
echo "=== tests/test_correlated_pair_cuts_soundness.py ==="
if $PY tests/test_correlated_pair_cuts_soundness.py; then
    PASS=$((PASS+1))
else
    FAIL=$((FAIL+1))
fi

echo
echo "=== tests/test_specaware_bound_cache.py ==="
PYTHONPATH=/data1/Kane/HyZor:/data1/Kane/ACT $PY tests/test_specaware_bound_cache.py
CACHE_RC=$?
if [ "$CACHE_RC" -eq 0 ]; then
    PASS=$((PASS+1))
else
    FAIL=$((FAIL+1))
fi

echo
echo "=== tests/test_specaware_direct_query.py ==="
PYTHONPATH=/data1/Kane/HyZor:/data1/Kane/ACT $PY tests/test_specaware_direct_query.py
DQ_RC=$?
if [ "$DQ_RC" -eq 0 ]; then
    PASS=$((PASS+1))
else
    FAIL=$((FAIL+1))
fi

echo
echo "=== tests/test_cli_env_no_leak.py ==="
PYTHONPATH=/data1/Kane/ACT $PY tests/test_cli_env_no_leak.py
EL_RC=$?
if [ "$EL_RC" -eq 0 ]; then
    PASS=$((PASS+1))
else
    FAIL=$((FAIL+1))
fi

echo
echo "=== tests/test_hz_dense_batched_matmul.py ==="
PYTHONPATH=/data1/Kane/ACT $PY tests/test_hz_dense_batched_matmul.py
BM_RC=$?
if [ "$BM_RC" -eq 0 ]; then
    PASS=$((PASS+1))
else
    FAIL=$((FAIL+1))
fi

echo
echo "=== tests/test_hz_factor_aware_add.py ==="
PYTHONPATH=/data1/Kane/ACT $PY tests/test_hz_factor_aware_add.py
FA_RC=$?
if [ "$FA_RC" -eq 0 ]; then
    PASS=$((PASS+1))
else
    FAIL=$((FAIL+1))
fi

# Note (advisor 2026-06-02): tests/test_cifar_endcap_15_receipts_audit.py
# is NOT included in the portable regression suite because it depends on
# machine-local audit artifacts at /data1/Kane/ACT/audit_results/. Run it
# explicitly via:
#   CIFAR_15_AUDIT_DIR=<sweep_dir> python tests/test_cifar_endcap_15_receipts_audit.py
# The test self-skips when no audit dir is found, so it never falsely
# breaks portable CI.

echo
echo "=== tests/test_generic_mlp_endcap_gate.py ==="
# Structural gate for the generic MLP end-cap profile. Pure-Python
# (no GPU / no ONNX). Covers Tiny/CIFAR/YOLO/relusplitter/vgg/etc.
PYTHONPATH=/data1/Kane/ACT $PY tests/test_generic_mlp_endcap_gate.py
MLPGATE_RC=$?
if [ "$MLPGATE_RC" -eq 0 ]; then
    PASS=$((PASS+1))
else
    FAIL=$((FAIL+1))
fi

echo
echo "=== tests/test_concat_shared_prefix.py ==="
# Shared-prefix concat fast path: sound vs block-diag, ng reduction,
# fallback on mismatched shared rows. Env knob default OFF so test
# explicitly toggles ACT_HZ_CONCAT_SHARED_PREFIX inside.
PYTHONPATH=/data1/Kane/ACT $PY tests/test_concat_shared_prefix.py
CONCAT_RC=$?
if [ "$CONCAT_RC" -eq 0 ]; then
    PASS=$((PASS+1))
else
    FAIL=$((FAIL+1))
fi

echo
echo "=== tests/test_lp_backend_guard.py ==="
# Single invocation: capture stdout once, judge both exit code AND output.
# (The test file's __main__ harness catches AssertionError, so RC alone is
# insufficient. We grep the captured output for FAIL/ERR lines instead of
# re-running the suite.)
GUARD_OUT=$(PYTHONPATH=/data1/Kane/HyZor:/data1/Kane/ACT $PY tests/test_lp_backend_guard.py 2>&1)
GUARD_RC=$?
printf '%s\n' "$GUARD_OUT"
if [ "$GUARD_RC" -eq 0 ] && ! printf '%s\n' "$GUARD_OUT" | grep -qE '^(FAIL|ERR )'; then
    PASS=$((PASS+1))
else
    FAIL=$((FAIL+1))
fi

echo
echo "=== tests/test_hz_reduction_soundness.py (CPU subset) ==="
$PY - <<'PYINLINE'
import sys
sys.path.insert(0, '/data1/Kane/ACT')
import tests.test_hz_reduction_soundness as m
fails = 0
total = 0
skipped = 0
for attr in sorted(dir(m)):
    if not attr.startswith('test_'):
        continue
    total += 1
    try:
        getattr(m, attr)()
    except RuntimeError as e:
        if 'cuda' in str(e).lower():
            skipped += 1
            continue
        fails += 1
    except Exception:
        fails += 1
print(f'test_hz_reduction_soundness: {total-fails-skipped}/{total} PASS, '
      f'{skipped} cuda-skipped, {fails} FAIL')
sys.exit(1 if fails else 0)
PYINLINE
RC=$?
if [ "$RC" -eq 0 ]; then
    PASS=$((PASS+1))
else
    FAIL=$((FAIL+1))
fi

echo
echo "=== Aggregate: $PASS suites PASS, $FAIL FAIL ==="
exit $FAIL
