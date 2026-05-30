#!/bin/bash
# E4: regression test pack — 8 iids covering distinct op patterns + recent
# fix areas. Run before every ACT-side fix commit to catch silent breakage
# like the Round-4 raw-first regression.
#
# Time: ~5 min on idle CPU.
# Expected: 0 ERROR, verdicts match canonical (printed below).
set -u

PY=/data1/Kane/miniconda3/envs/act-py312/bin/python
ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
BASE=/tmp/act_regression_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$BASE"

export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=$ROOT
export OMP_NUM_THREADS=1
export ACT_HZ_JOINT_K2=1
export ACT_HZ_JOINT_K2_SPEC=1
export ACT_HZ_JOINT_K2_MAX_PAIRS=32
export ACT_HZ_JOINT_K2_MIN_COSSIM=0.3
export ACT_HZ_JOINT_K2_SPEC_TOPK=8
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# (bench, iid, expected_verdict, wall_s, rss_gb, fix_relevance)
PROBES=(
    "acasxu_2023:0:UNSAT_or_CERT:30:4:baseline_dense"           # small dense
    "collins_rul_cnn_2022:0:FALSIFIED:30:8:CNN_with_FAL"        # CNN with FAL (catches Conv/LP-witness regressions)
    "malbeware:0:CERTIFIED:30:8:CNN_with_CERT"                  # CNN with CERT
    "ml4acopf_2024:0:any:30:8:fix5_simplify_first"              # Fix #5 sentinel (transpose-heavy)
    "lsnc_relu:0:any:30:8:fix5_simplify_first"                  # Fix #5 sentinel (similar pattern)
    "nn4sys:137:CERTIFIED:60:8:fix1-4_mscn"                     # Round 4 fix territory (mscn_128d_dual)
    "collins_aerospace_benchmark:1:any:60:8:fix6_lrelu_alpha"   # Fix #6 sentinel
    "safenlp_2024:0:UNKNOWN:30:4:large_lp"                      # large LP path (r93 canonical: UNKNOWN @ 1.0s)
)

PASS=0
FAIL=0
echo "=== ACT regression pack ==="
for probe in "${PROBES[@]}"; do
    bench=$(echo $probe | cut -d: -f1)
    iid=$(echo $probe | cut -d: -f2)
    expect=$(echo $probe | cut -d: -f3)
    wall=$(echo $probe | cut -d: -f4)
    rss=$(echo $probe | cut -d: -f5)
    tag=$(echo $probe | cut -d: -f6)
    out="$BASE/${bench}_${iid}"
    mkdir -p "$out"
    $PY -m act.pipeline.watchdog_runner \
        --benchmark "$bench" --instance-ids "$iid" \
        --wall-s "$wall" --startup-grace-s 8 --poll-interval-s 0.5 \
        --rss-cap-gb "$rss" --grace-kill-s 3 \
        --device cpu --dtype float64 \
        --out-dir "$out" --canonical-root "$ROOT" \
        > "$out/driver.log" 2>&1
    actual=$($PY -c "
import json, glob
for f in glob.glob('$out/per_instance_*.json'):
    if 'watchdog' in f: continue
    d = json.load(open(f))
    for p in d.get('per_instance', []):
        print(p['cli_normalized'])
        break
    break
")
    if [ -z "$actual" ]; then
        actual=$($PY -c "
import json
d = json.load(open('$out/watchdog_summary.json'))
print(d['results'][0].get('cli_normalized', '?'))")
    fi
    # PASS criteria
    ok="✗"
    if [ "$expect" = "any" ]; then
        # Any non-ERROR verdict passes (we care that conversion + analyzer ran)
        case "$actual" in
            ERROR*) ok="✗" ;;
            *) ok="✓" ;;
        esac
    elif [ "$expect" = "UNSAT_or_CERT" ]; then
        case "$actual" in
            CERTIFIED|UNKNOWN_TIMEOUT|UNKNOWN) ok="✓" ;;
            *) ok="✗" ;;
        esac
    elif [ "$actual" = "$expect" ]; then
        ok="✓"
    fi
    if [ "$ok" = "✓" ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
    printf "  [%s] %-35s iid=%-3s tag=%-22s expect=%-15s got=%-30s\n" \
        "$ok" "$bench" "$iid" "$tag" "$expect" "$actual"
done
echo ""
echo "=== Result: $PASS PASS, $FAIL FAIL ==="
echo "BASE=$BASE"
[ $FAIL -eq 0 ] && exit 0 || exit 1
