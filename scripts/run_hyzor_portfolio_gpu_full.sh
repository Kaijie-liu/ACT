#!/usr/bin/env bash
# HyZor Portfolio Runner — strict P1-P5 forward HZ portfolio.
#
# Per advisor 2026-06-07: single command that runs multiple in-principle
# profiles per iid, records provenance, aggregates to paper-grade V/A.
#
# Profiles (all forward, all P1-P5 clean):
#   - regular K=None        small dense, MLP
#   - sparse-slack K=128    cifar/tiny ResNet
#   - sparse-slack K=64     vggnet, heavy conv
#   - hz_only K=None        memory-tight ReLU
#   - hz_only K=128         deep + memory-tight
#
# Acceptance gates:
#   - reproduce strict_555_FINAL (555 records)
#   - reproduce SESSION_CANONICAL_648 (648 records)
#   - 0 overlap with BASELINE_KEYS_COMPONENTS (1538 keys)
#   - 2013 floor first, then 2107 candidate
#
# Usage:
#   bash scripts/run_hyzor_portfolio_gpu_full.sh [n_workers]
#
# No PGD / no BaB / no MILP / no Gurobi / no random certify / no ORT in cert path.

set -e
cd "$(dirname "$0")/.."

N_WORKERS="${1:-4}"
BASE="/data1/Kane/ACT/audit_results/hyzor_portfolio_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$BASE"

echo "BASE: $BASE"
echo "Workers: $N_WORKERS"
echo "Method: HyZor portfolio (FCHZ_walker × 5 profiles, first CERT wins)"
echo ""

# Verify environment
python_bin="/data1/Kane/miniconda3/envs/act-py312/bin/python"
[ -x "$python_bin" ] || { echo "Python env missing"; exit 1; }

# Launch
nohup $python_bin scripts/portfolio_runner.py $N_WORKERS > "$BASE/RUN.log" 2>&1 &
PID=$!
echo "Launched PID=$PID"
echo "Output: /tmp/hyzor_portfolio_*/  (or follow $BASE/RUN.log)"
echo ""
echo "To monitor:"
echo "  tail -f $BASE/RUN.log"
echo ""
echo "To stop:"
echo "  kill -9 $PID"
