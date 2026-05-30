#!/usr/bin/env python3
"""
Per-benchmark runner for the HYB_Z STRICT sweep.
Reads vnncomp2025 instances.csv, drives scripts/run_hybz_instance.sh.
Writes to /data1/Kane/pyrat/results_pure_hybz/<bench>/results.csv.
"""
import argparse, csv, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PYRAT_DIR    = Path(__file__).resolve().parent.parent
BENCH_ROOT   = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks")
RUN_INSTANCE = PYRAT_DIR / "scripts" / "run_hybz_instance.sh"
RES_ROOT     = PYRAT_DIR / "results_pure_hybz"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--timeout-cap", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--from-index", type=int, default=0)
    return ap.parse_args()


def run_one(row, bench, out_csv, gpu_env):
    onnx_rel, vnn_rel, timeout_s = row
    onnx = str(BENCH_ROOT / bench / onnx_rel.lstrip("./"))
    vnnlib = str(BENCH_ROOT / bench / vnn_rel.lstrip("./"))
    env = os.environ.copy()
    if gpu_env is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_env
    cmd = ["bash", str(RUN_INSTANCE), bench, onnx, vnnlib, str(timeout_s), str(out_csv)]
    t0 = time.time()
    try:
        subprocess.run(cmd, env=env, check=False,
                       timeout=int(timeout_s) + 120)
    except subprocess.TimeoutExpired:
        pass
    return time.time()-t0, onnx_rel, vnn_rel


def main():
    args = parse_args()
    bench = args.benchmark
    inst_csv = BENCH_ROOT / bench / "instances.csv"
    if not inst_csv.exists():
        sys.exit(f"missing {inst_csv}")
    rows = []
    with open(inst_csv) as fh:
        for r in csv.reader(fh):
            if len(r) < 3:
                continue
            t = int(float(r[2]))
            if args.timeout_cap:
                t = min(t, args.timeout_cap)
            rows.append((r[0], r[1], t))
    rows = rows[args.from_index:]
    if args.limit:
        rows = rows[:args.limit]

    out_dir = RES_ROOT / bench
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
            dt, on, vn = run_one(row, bench, out_csv, args.gpu)
            print(f"  [{i}/{len(rows)}] {dt:6.1f}s  {on}  {vn}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, r, bench, out_csv, args.gpu): r for r in rows}
            done = 0
            for fut in as_completed(futs):
                done += 1
                dt, on, vn = fut.result()
                print(f"  [{done}/{len(rows)}] {dt:6.1f}s  {on}  {vn}", flush=True)
    print(f"[{bench}] done in {time.time()-started:.0f}s -> {out_csv}")


if __name__ == "__main__":
    main()
