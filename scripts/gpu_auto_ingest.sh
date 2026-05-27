#!/bin/bash
# Automated ingest: waits for all 3 GPU streams to write their final
# watchdog_summary.json, then symlinks each benchmark's source dir into
# CONSOLIDATED_RESULTS, rebuilds CSVs, runs the official cross-check,
# and writes a final summary report.
#
# Invoke as nohup BG; this script polls every 60s.
set -u

LOG=/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/gpu_auto_ingest.log
exec > >(tee -a "$LOG") 2>&1

PY=/data1/Kane/miniconda3/envs/act-py312/bin/python
ARCHIVE=/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z
CONSOL=$ARCHIVE/CONSOLIDATED_RESULTS

# Stream BASE dirs (set by relaunch command; these are the new timestamp)
S1=$ARCHIVE/gpu_stream1_20260526T132419Z
S2=$ARCHIVE/gpu_stream2_20260526T132419Z
S3=$ARCHIVE/gpu_stream3_20260526T132419Z

echo "=== auto-ingest start $(date -u) ==="
echo "Watching:"
echo "  S1 = $S1"
echo "  S2 = $S2"
echo "  S3 = $S3"

# Wait until each stream's README.txt has the "=== gpu_streamN done ===" marker.
done_marker() {
  local base=$1 stream_name=$2
  grep -q "=== ${stream_name} done ===" "$base/README.txt" 2>/dev/null
}

while true; do
  d1=$(done_marker "$S1" gpu_stream1 && echo "✓" || echo "·")
  d2=$(done_marker "$S2" gpu_stream2 && echo "✓" || echo "·")
  d3=$(done_marker "$S3" gpu_stream3 && echo "✓" || echo "·")
  echo "[$(date -u)] stream1=$d1  stream2=$d2  stream3=$d3"
  if [ "$d1" = "✓" ] && [ "$d2" = "✓" ] && [ "$d3" = "✓" ]; then
    echo "All 3 streams done. Proceeding to ingest."
    break
  fi
  sleep 300  # 5 min poll
done

echo ""
echo "=== Symlink each benchmark's source dir into CONSOLIDATED_RESULTS ==="
ingest() {
  local bench=$1 src=$2 label=$3
  if [ -d "$src" ]; then
    mkdir -p "$CONSOL/$bench"
    ln -sfn "$src" "$CONSOL/$bench/_source_$label"
    echo "  $bench/_source_$label -> $src"
  fi
}

# Stream 1 (light)
ingest collins_aerospace_benchmark        "$S1/collins_aerospace_benchmark"        gpu_full
ingest cersyve                            "$S1/cersyve"                            gpu_full
ingest tllverifybench_2023                "$S1/tllverifybench_2023"                gpu_full
ingest traffic_signs_recognition_2023     "$S1/traffic_signs_recognition_2023"     gpu_full
ingest soundnessbench                     "$S1/soundnessbench"                     gpu_full
ingest ml4acopf_2024                      "$S1/ml4acopf_2024"                      gpu_full
ingest dist_shift_2023                    "$S1/dist_shift_2023"                    gpu_full
ingest lsnc_relu                          "$S1/lsnc_relu"                          gpu_full
ingest metaroom_2023                      "$S1/metaroom_2023"                      gpu_full
ingest cora_2024                          "$S1/cora_2024"                          gpu_full
ingest cgan_2023                          "$S1/cgan_2023"                          gpu_full
# Stream 2 (medium)
ingest nn4sys                             "$S2/nn4sys"                             gpu_full
ingest relusplitter                       "$S2/relusplitter"                       gpu_full
ingest safenlp_2024                       "$S2/safenlp_2024"                       gpu_full
# Stream 3 (heavy)
ingest vggnet16_2022                      "$S3/vggnet16_2022"                      gpu_full
ingest yolo_2023                          "$S3/yolo_2023"                          gpu_full
ingest cifar100_2024                      "$S3/cifar100_2024"                      gpu_full
ingest tinyimagenet_2024                  "$S3/tinyimagenet_2024"                  gpu_full

echo ""
echo "=== Rebuild per_instance.csv from all _source_* (watchdog synthetic wins) ==="
cd "$CONSOL"
$PY build_csvs.py 2>&1 | tail -60

echo ""
echo "=== Run official-label cross-check (zero_tol + small_tol) ==="
$PY soundness_check.py 2>&1 | tail -60

echo ""
echo "=== Per-benchmark CPU↔GPU bit-identity check ==="
$PY << 'PY'
"""For each benchmark with both CPU and GPU canonical sources, compare
per-iid verdicts and report (a) total instances, (b) bit-identical count,
(c) any divergent iids."""
import csv, glob
from collections import defaultdict
CONSOL = "/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/CONSOLIDATED_RESULTS"

# Map: bench -> (cpu_src, gpu_src) — pick the AUTHORITATIVE source per side
PAIRS = {
    "acasxu_2023":               ("cpu_auto", "gpu"),
    "collins_rul_cnn_2022":      ("cpu",      "gpu"),
    "linearizenn_2024":          ("cpu_R9",   "gpu"),
    "malbeware":                 ("cpu",      "gpu"),
    "sat_relu":                  ("cpu",      "gpu"),
    # New GPU sources just ingested:
    "safenlp_2024":              ("cpu_auto", "gpu_full"),
    "tllverifybench_2023":       ("cpu_witness", "gpu_full"),
    "dist_shift_2023":           ("cpu",      "gpu_full"),
    "metaroom_2023":             ("full_r3",  "gpu_full"),
    "cora_2024":                 ("full_r3_s1", "gpu_full"),
    "ml4acopf_2024":             ("cpu",      "gpu_full"),
    "relusplitter":              ("cpu2",     "gpu_full"),
    "nn4sys":                    ("lindex200_fixed", "gpu_full"),
    "lsnc_relu":                 ("cpu",      "gpu_full"),
    "collins_aerospace_benchmark": ("cpu_smoke", "gpu_full"),
    "traffic_signs_recognition_2023": ("full_r3", "gpu_full"),
    "vggnet16_2022":             ("smoke_longwall", "gpu_full"),
    "yolo_2023":                 ("full_r3",  "gpu_full"),
    "soundnessbench":            ("full_r3",  "gpu_full"),
    "cifar100_2024":             ("full_r3",  "gpu_full"),
    "tinyimagenet_2024":         ("full_r3",  "gpu_full"),
    "cersyve":                   ("cpu_native_r2", "gpu_full"),
    "cgan_2023":                 ("smoke_realfile", "gpu_full"),
}

print(f"{'benchmark':<35} {'cpu_n':>5} {'gpu_n':>5} {'agree':>6} {'diff':>5} divergent_iids")
for bench, (cpu_s, gpu_s) in PAIRS.items():
    csv_path = f"{CONSOL}/{bench}/per_instance.csv"
    try:
        rows = list(csv.DictReader(open(csv_path)))
    except FileNotFoundError:
        continue
    cpu = {r["iid"]: r["verdict"] for r in rows if r["source"] == cpu_s}
    gpu = {r["iid"]: r["verdict"] for r in rows if r["source"] == gpu_s}
    if not gpu:
        print(f"{bench:<35} (no gpu_full source yet, skipping)")
        continue
    common = set(cpu) & set(gpu)
    agree = sum(1 for i in common if cpu[i] == gpu[i])
    diff = [(i, cpu[i], gpu[i]) for i in common if cpu[i] != gpu[i]]
    print(f"{bench:<35} {len(cpu):>5} {len(gpu):>5} {agree:>6} {len(diff):>5} {[i for i,_,_ in diff[:5]]}")
    if diff and len(diff) <= 8:
        for i, cv, gv in diff[:8]:
            print(f"    iid={i}: cpu={cv}  gpu={gv}")
PY

echo ""
echo "=== auto-ingest done $(date -u) ==="
