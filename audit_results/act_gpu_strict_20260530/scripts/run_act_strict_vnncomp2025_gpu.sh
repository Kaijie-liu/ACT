#!/bin/bash
# ACT VNN-COMP 2025 STRICT GPU sweep — unified driver
# Author: BUPTlkj (with assistance)
# Date: 2026-05-30
#
# This script reproduces the GPU sweep results in this archive.
# Calls watchdog_runner per-benchmark with appropriate wall/RSS budgets.
#
# Pre-requisites:
#   - Python env at /data1/Kane/miniconda3/envs/act-py312
#   - ACT repo at /data1/Kane/ACT with session patches applied
#     (see ../patches/session_dirty.patch)
#   - VNN-COMP 2025 benchmarks at /data1/Kane/data/vnncomp2025_benchmarks/benchmarks
#   - NVIDIA GPU with ≥ 24 GB VRAM (sweep runs 4-way parallel)
#
# Helper-free enforcement:
#   - small_dense_lp default is `specaware` (no random WitnessExtract)
#   - P1-P6 principles enforced via code defaults (see patches/README.md)

set -u
export PYTHONPATH=/data1/Kane/ACT
export ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT:-"/data1/Kane/ACT/audit_results/act_gpu_full_${STAMP}"}
mkdir -p "$ROOT"
LOG="$ROOT/_run.log"
echo "ACT GPU strict sweep started: $(date)" | tee "$LOG"

PY=/data1/Kane/miniconda3/envs/act-py312/bin/python

# Per-benchmark wall and rss budgets (informed by overnight + session experience)
# Format: bench:iids_pattern:wall_s:rss_gb
declare -A WALL_S RSS_GB
WALL_S[acasxu_2023]=120;                RSS_GB[acasxu_2023]=6
WALL_S[malbeware]=120;                  RSS_GB[malbeware]=6
WALL_S[linearizenn_2024]=180;           RSS_GB[linearizenn_2024]=8
WALL_S[sat_relu]=120;                   RSS_GB[sat_relu]=8
WALL_S[collins_rul_cnn_2022]=180;       RSS_GB[collins_rul_cnn_2022]=8
WALL_S[cgan_2023]=300;                  RSS_GB[cgan_2023]=20
WALL_S[dist_shift_2023]=120;            RSS_GB[dist_shift_2023]=14
WALL_S[nn4sys]=180;                     RSS_GB[nn4sys]=20
WALL_S[ml4acopf_2024]=180;              RSS_GB[ml4acopf_2024]=12
WALL_S[metaroom_2023]=120;              RSS_GB[metaroom_2023]=8
WALL_S[cora_2024]=300;                  RSS_GB[cora_2024]=12
WALL_S[safenlp_2024]=60;                RSS_GB[safenlp_2024]=8
WALL_S[tllverifybench_2023]=120;        RSS_GB[tllverifybench_2023]=8
WALL_S[tinyimagenet_2024]=180;          RSS_GB[tinyimagenet_2024]=8
WALL_S[vggnet16_2022]=600;              RSS_GB[vggnet16_2022]=16
WALL_S[cifar100_2024]=240;              RSS_GB[cifar100_2024]=8
WALL_S[yolo_2023]=180;                  RSS_GB[yolo_2023]=8
WALL_S[traffic_signs_recognition_2023]=180; RSS_GB[traffic_signs_recognition_2023]=8
WALL_S[soundnessbench]=120;             RSS_GB[soundnessbench]=8
WALL_S[cersyve]=120;                    RSS_GB[cersyve]=6
WALL_S[lsnc_relu]=120;                  RSS_GB[lsnc_relu]=8
WALL_S[collins_aerospace_benchmark]=300; RSS_GB[collins_aerospace_benchmark]=16
WALL_S[cctsdb_yolo_2023]=120;           RSS_GB[cctsdb_yolo_2023]=8
WALL_S[relusplitter]=180;               RSS_GB[relusplitter]=12
# vit_2023 not supported

# Helper to spawn a watchdog runner for one benchmark with all its iids
spawn_bench() {
    local bench=$1
    local instances_csv=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/$bench/instances.csv
    if [ ! -f "$instances_csv" ]; then
        echo "  SKIP $bench: no instances.csv" | tee -a "$LOG"
        return
    fi
    local n=$(wc -l < "$instances_csv")
    local iids=$(seq -s, 0 $((n-1)))
    local wall=${WALL_S[$bench]:-180}
    local rss=${RSS_GB[$bench]:-12}
    local OUT="$ROOT/$bench"
    mkdir -p "$OUT"
    echo "  $bench: $n iids, wall=${wall}s, rss=${rss}GB" | tee -a "$LOG"
    $PY -m act.pipeline.watchdog_runner \
        --benchmark "$bench" --instance-ids "$iids" \
        --wall-s "$wall" --startup-grace-s 8 --poll-interval-s 0.5 \
        --rss-cap-gb "$rss" --grace-kill-s 3 \
        --device cuda --dtype float64 \
        --out-dir "$OUT" \
        --canonical-root /data1/Kane/data/vnncomp2025_benchmarks/benchmarks \
        > "$OUT/d.log" 2>&1
}

# Sequential, one benchmark at a time. For 4-way parallel within a benchmark,
# split iids into N batches and use `&` + `wait`. See nn4sys_full_194.sh for example.
for bench in "${!WALL_S[@]}"; do
    spawn_bench "$bench"
done

echo "ALL DONE: $(date)" | tee -a "$LOG"
echo "Results at: $ROOT"
