#!/usr/bin/env python3
"""
Re-run any (onnx, vnnlib) rows from <bench>/instances.csv that are missing
from <bench>/results.csv. Use after the main run to fill in rows that were
dropped due to the timeout race in v8 (e.g. lsnc_relu).

Usage:
   scripts/fill_missing.py <bench> [<bench> ...]
"""
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

PYRAT_DIR = Path(__file__).resolve().parent.parent
BENCH_ROOT = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks")
RUN_INSTANCE = PYRAT_DIR / "scripts" / "run_instance.sh"


def missing_rows(bench):
    inst = BENCH_ROOT / bench / "instances.csv"
    csv_p = PYRAT_DIR / "results_pure" / bench / "results.csv"
    want = []
    with open(inst) as fh:
        for r in csv.reader(fh):
            if len(r) < 3:
                continue
            want.append((r[0], r[1], int(float(r[2]))))
    done_keys = set()
    if csv_p.exists():
        with open(csv_p) as fh:
            for r in csv.reader(fh):
                if len(r) < 3 or r[0] == "benchmark":
                    continue
                onnx_short = r[1].split("/onnx/")[-1] if "/onnx/" in r[1] else r[1].split("/")[-1]
                vnn_short  = r[2].split("/vnnlib/")[-1] if "/vnnlib/" in r[2] else r[2].split("/")[-1]
                done_keys.add((f"onnx/{onnx_short}", f"vnnlib/{vnn_short}"))
    return [w for w in want if (w[0], w[1]) not in done_keys]


def fill(bench):
    miss = missing_rows(bench)
    if not miss:
        print(f"[{bench}] nothing missing")
        return
    csv_p = PYRAT_DIR / "results_pure" / bench / "results.csv"
    print(f"[{bench}] filling {len(miss)} missing row(s) into {csv_p}")
    for i, (onnx_rel, vnn_rel, t_s) in enumerate(miss, 1):
        onnx = str(BENCH_ROOT / bench / onnx_rel)
        vnnlib = str(BENCH_ROOT / bench / vnn_rel)
        cmd = ["bash", str(RUN_INSTANCE), bench, onnx, vnnlib, str(t_s), str(csv_p)]
        t0 = time.time()
        try:
            subprocess.run(cmd, check=False, timeout=int(t_s) + 120)
        except subprocess.TimeoutExpired:
            pass
        print(f"  [{i}/{len(miss)}] {time.time()-t0:.1f}s  {onnx_rel}  {vnn_rel}",
              flush=True)


if __name__ == "__main__":
    for b in sys.argv[1:]:
        fill(b)
