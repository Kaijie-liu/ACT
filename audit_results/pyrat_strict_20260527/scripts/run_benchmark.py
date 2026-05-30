#!/usr/bin/env python3
"""
Run one VNN-COMP 2025 benchmark through PyRAT in strict no-adv-search mode.

Workers run in parallel. Default workers=1 because PyRAT internally uses
nb_process and CUDA already — going above 1 mostly helps tiny benchmarks
(acasxu, sat_relu, dist_shift) and may OOM the GPU on cifar100 / vit / yolo.

Per-instance verdict is appended to results_pure/<bench>/results.csv.
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PYRAT_DIR = Path(__file__).resolve().parent.parent
BENCH_ROOT = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks")
RUN_INSTANCE = PYRAT_DIR / "scripts" / "run_instance.sh"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", help="benchmark dir name (e.g. acasxu_2023)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel workers; keep at 1 for GPU-heavy configs")
    ap.add_argument("--gpu", default=None,
                    help="CUDA_VISIBLE_DEVICES to pin (e.g. '0' or '0,1')")
    ap.add_argument("--timeout-cap", type=int, default=None,
                    help="override per-instance timeout (seconds)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N instances (smoke test)")
    ap.add_argument("--from-index", type=int, default=0,
                    help="resume from row index N (0-based)")
    return ap.parse_args()


def run_one(row, bench, out_csv, gpu_env):
    onnx_rel, vnnlib_rel, timeout_s = row[0], row[1], row[2]
    onnx = str(BENCH_ROOT / bench / onnx_rel)
    vnnlib = str(BENCH_ROOT / bench / vnnlib_rel)
    env = os.environ.copy()
    if gpu_env is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_env

    cmd = ["bash", str(RUN_INSTANCE), bench, onnx, vnnlib, str(timeout_s), str(out_csv)]
    t0 = time.time()
    # Outer subprocess timeout MUST exceed the inner watchdog
    # (run_instance.sh: timeout_s + 30 SIGTERM, +15 SIGKILL = inner deadline + 45 s)
    # plus a generous CSV-flush margin to avoid clobbering the row write.
    # Without this margin Python could SIGKILL the bash wrapper BEFORE it
    # appends the CSV line, silently dropping the row (cf. lsnc_relu loss
    # of 12/80 in v8 first pass).
    try:
        subprocess.run(cmd, env=env, check=False,
                       timeout=int(timeout_s) + 120)
    except subprocess.TimeoutExpired:
        pass
    return time.time() - t0, onnx_rel, vnnlib_rel


def main():
    args = parse_args()
    bench = args.benchmark
    bench_dir = BENCH_ROOT / bench
    inst_csv = bench_dir / "instances.csv"
    if not inst_csv.exists():
        sys.exit(f"missing {inst_csv}")

    rows = []
    with open(inst_csv) as fh:
        for r in csv.reader(fh):
            if not r or len(r) < 3:
                continue
            timeout_s = int(float(r[2]))
            if args.timeout_cap is not None:
                timeout_s = min(timeout_s, args.timeout_cap)
            rows.append((r[0], r[1], timeout_s))

    rows = rows[args.from_index:]
    if args.limit:
        rows = rows[:args.limit]

    out_dir = PYRAT_DIR / "results_pure" / bench
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "results.csv"

    if args.from_index == 0 and not out_csv.exists():
        with open(out_csv, "w") as fh:
            fh.write("benchmark,onnx,vnnlib,verdict,wall_s,reported_s,returncode\n")

    print(f"[{bench}] {len(rows)} instances, workers={args.workers}, "
          f"gpu={args.gpu or 'inherit'}, out={out_csv}")

    started = time.time()
    if args.workers <= 1:
        for i, row in enumerate(rows, 1):
            dt, onnx_rel, vnnlib_rel = run_one(row, bench, out_csv, args.gpu)
            print(f"  [{i}/{len(rows)}] {dt:6.1f}s  {onnx_rel}  {vnnlib_rel}",
                  flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, row, bench, out_csv, args.gpu): row
                    for row in rows}
            done = 0
            for fut in as_completed(futs):
                done += 1
                dt, onnx_rel, vnnlib_rel = fut.result()
                print(f"  [{done}/{len(rows)}] {dt:6.1f}s  {onnx_rel}  {vnnlib_rel}",
                      flush=True)

    print(f"[{bench}] done in {time.time()-started:.0f}s -> {out_csv}")


if __name__ == "__main__":
    main()
