#!/bin/bash
# Chain wrapper: HYB_Z GPU-only -> CPU-only -> aggregator.
#
# Runs the full hyb_z sweep unattended:
#   1. Phase A: scripts/run_hybz_gpu_only.sh  (16 GPU benches, single supervisor)
#   2. Phase B: scripts/run_hybz_cpu_only.sh  (10 CPU benches, workers=4)
#   3. Phase C: scripts/aggregate_results.py with results_pure_hybz/ as target
#
# Designed to run overnight while another CPU-heavy job (e.g. NNV MATLAB
# sweep) finishes — the CPU phase only starts after GPU phase, which gives
# the NNV run ~time-of-GPU-phase headroom to finish its own work.

set -u
PYRAT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PYRAT_DIR"

GPU="${1:-0}"
LOG="$PYRAT_DIR/results_pure_hybz/_chain.log"
mkdir -p "$(dirname "$LOG")"

date | tee -a "$LOG"
echo "=== HYB_Z chain wrapper: GPU-only -> CPU-only -> aggregate ===" | tee -a "$LOG"

# ----- Phase A: GPU -------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Phase A: GPU-only sweep ===" | tee -a "$LOG"
bash "$PYRAT_DIR/scripts/run_hybz_gpu_only.sh" "$GPU"

# ----- Phase B: CPU -------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Phase B: CPU-only sweep ===" | tee -a "$LOG"
bash "$PYRAT_DIR/scripts/run_hybz_cpu_only.sh" "$GPU"

# ----- Phase C: aggregate -------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Phase C: aggregate ===" | tee -a "$LOG"
# Patch aggregator for hybz dir
python3 - << 'PYEOF' 2>&1 | tee -a "$LOG"
import csv, os
from collections import Counter
from pathlib import Path
DST = Path("/data1/Kane/pyrat/results_pure_hybz")
rows = []
for d in sorted(DST.iterdir()):
    if not d.is_dir() or d.name.startswith("_"): continue
    csv_p = d / "results.csv"
    if not csv_p.exists(): continue
    verd = Counter(); wall = 0.0
    with open(csv_p) as fh:
        rdr = csv.reader(fh); next(rdr, None)
        for r in rdr:
            if len(r) < 7: continue
            verd[r[3]] += 1
            try: wall += float(r[4])
            except: pass
    v = verd.get("verified",0); f = verd.get("falsified",0)
    u = verd.get("unknown",0);  t = verd.get("timeout",0); e = verd.get("error",0)
    rows.append((d.name, v+f+u+t+e, v, f, u, t, e, wall))

summary_csv = DST / "_summary.csv"
with open(summary_csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["benchmark","N","V","F","U","T","E","wall_sec"])
    tot = [0,0,0,0,0,0]; total_wall = 0
    for r in rows:
        w.writerow(list(r[:7]) + [f"{r[7]:.1f}"])
        for i in range(1,7): tot[i-1] += r[i]
        total_wall += r[7]
    w.writerow(["TOTAL"] + tot + [f"{total_wall:.1f}"])
print(f"wrote {summary_csv}")
print(f"TOTAL: N={tot[0]}  V={tot[1]}  F={tot[2]}  U={tot[3]}  T={tot[4]}  E={tot[5]}  wall={total_wall/3600:.1f}h")
PYEOF

echo "" | tee -a "$LOG"
echo "=== $(date)  HYB_Z CHAIN ALL DONE ===" | tee -a "$LOG"
