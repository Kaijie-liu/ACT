#!/bin/bash
# ACT VNN-COMP 2025 STRICT CPU sweep — driver
# Same as GPU driver but with --device cpu and adjusted RSS/wall budgets.
set -u
export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT:-"/data1/Kane/ACT/audit_results/act_cpu_full_${STAMP}"}
mkdir -p "$ROOT"
LOG="$ROOT/_run.log"
echo "ACT CPU strict sweep started: $(date)" | tee "$LOG"

PY=/data1/Kane/miniconda3/envs/act-py312/bin/python

# CPU per-bench budgets are GENERALLY LARGER walls (CPU slower than GPU)
declare -A WALL_S RSS_GB
WALL_S[acasxu_2023]=120;                RSS_GB[acasxu_2023]=8
WALL_S[malbeware]=120;                  RSS_GB[malbeware]=8
WALL_S[linearizenn_2024]=300;           RSS_GB[linearizenn_2024]=12
WALL_S[sat_relu]=300;                   RSS_GB[sat_relu]=12
WALL_S[collins_rul_cnn_2022]=300;       RSS_GB[collins_rul_cnn_2022]=12
WALL_S[cgan_2023]=600;                  RSS_GB[cgan_2023]=24
WALL_S[dist_shift_2023]=180;            RSS_GB[dist_shift_2023]=16
WALL_S[nn4sys]=300;                     RSS_GB[nn4sys]=24
WALL_S[ml4acopf_2024]=300;              RSS_GB[ml4acopf_2024]=16
WALL_S[metaroom_2023]=600;              RSS_GB[metaroom_2023]=24
WALL_S[cora_2024]=600;                  RSS_GB[cora_2024]=16
WALL_S[safenlp_2024]=120;               RSS_GB[safenlp_2024]=12
WALL_S[tllverifybench_2023]=180;        RSS_GB[tllverifybench_2023]=12
WALL_S[tinyimagenet_2024]=600;          RSS_GB[tinyimagenet_2024]=24
WALL_S[vggnet16_2022]=900;              RSS_GB[vggnet16_2022]=24
WALL_S[cifar100_2024]=600;              RSS_GB[cifar100_2024]=24
WALL_S[yolo_2023]=300;                  RSS_GB[yolo_2023]=16
WALL_S[traffic_signs_recognition_2023]=300; RSS_GB[traffic_signs_recognition_2023]=16
WALL_S[soundnessbench]=180;             RSS_GB[soundnessbench]=12
WALL_S[cersyve]=120;                    RSS_GB[cersyve]=8
WALL_S[lsnc_relu]=180;                  RSS_GB[lsnc_relu]=12
WALL_S[collins_aerospace_benchmark]=600; RSS_GB[collins_aerospace_benchmark]=24
WALL_S[cctsdb_yolo_2023]=120;           RSS_GB[cctsdb_yolo_2023]=12
WALL_S[relusplitter]=300;               RSS_GB[relusplitter]=16

spawn_bench() {
    local bench=$1
    local instances_csv=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/$bench/instances.csv
    if [ ! -f "$instances_csv" ]; then return; fi
    local n=$(wc -l < "$instances_csv")
    local iids=$(seq -s, 0 $((n-1)))
    local wall=${WALL_S[$bench]:-300}
    local rss=${RSS_GB[$bench]:-16}
    local OUT="$ROOT/$bench"
    mkdir -p "$OUT"
    echo "  $bench: $n iids, wall=${wall}s, rss=${rss}GB" | tee -a "$LOG"
    $PY -m act.pipeline.watchdog_runner \
        --benchmark "$bench" --instance-ids "$iids" \
        --wall-s "$wall" --startup-grace-s 8 --poll-interval-s 0.5 \
        --rss-cap-gb "$rss" --grace-kill-s 3 \
        --device cpu --dtype float64 \
        --out-dir "$OUT" \
        --canonical-root /data1/Kane/data/vnncomp2025_benchmarks/benchmarks \
        > "$OUT/d.log" 2>&1
}

for bench in "${!WALL_S[@]}"; do
    spawn_bench "$bench"
done

echo "ALL DONE: $(date)" | tee -a "$LOG"
echo "Results at: $ROOT"
