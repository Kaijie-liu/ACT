#!/usr/bin/env python3
"""
Find every 'error' row whose log shows a CUDA-OOM / CUBLAS / allocation
failure, re-run those instances SERIALLY (workers=1, single bench at a
time) so the GPU is exclusive, and replace the row in results.csv with
the new outcome.

Use after the parallel main run is done. Safe to run idempotently —
non-OOM error rows are left alone.
"""
import csv
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

PYRAT_DIR = Path("/data1/Kane/pyrat")
BENCH_ROOT = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks")
RUN_INSTANCE = PYRAT_DIR / "scripts" / "run_instance.sh"
RES = PYRAT_DIR / "results_pure"

OOM_PAT = re.compile(
    r"OutOfMemoryError"
    r"|CUDA error: out of memory"
    r"|CUBLAS_STATUS_ALLOC_FAILED"
    r"|MemoryError"
    r"|cuDNN error: CUDNN_STATUS_INTERNAL_ERROR"
    r"|Cannot allocate memory"
    r"|cublasCreate"
)
RESULT_RE = re.compile(r"Result\s*=\s*(\w+),\s*Time\s*=\s*([0-9.]+)\s*s")
VERDICT_MAP = {"True": "verified", "False": "falsified",
               "Unknown": "unknown", "Timeout": "timeout"}


def is_oom_err(logs_dir: Path, onnx: str, vnn: str) -> bool:
    """Check if any matching .err file contains an OOM marker."""
    onnx_stem = Path(onnx).stem
    vnn_stem = Path(vnn).stem
    matches = list(logs_dir.glob(f"{onnx_stem}__{vnn_stem}__*.err")) or \
              list(logs_dir.glob(f"{onnx_stem}__{vnn_stem}.err"))
    for f in matches:
        try:
            if OOM_PAT.search(f.read_text(errors="replace")):
                return True
        except Exception:
            pass
    return False


def get_timeout_for(bench: str, onnx_rel: str, vnn_rel: str) -> int:
    inst = BENCH_ROOT / bench / "instances.csv"
    # Some benchmarks use './onnx/...' prefix, others use 'onnx/...'. Match by basename.
    onnx_bn = Path(onnx_rel).name
    vnn_bn  = Path(vnn_rel).name
    for r in csv.reader(open(inst)):
        if len(r) < 3:
            continue
        if Path(r[0]).name == onnx_bn and Path(r[1]).name == vnn_bn:
            return int(float(r[2]))
    return 0


def rerun_one(bench: str, onnx_abs: str, vnn_abs: str, t_s: int) -> tuple:
    """Run the instance via run_instance.sh, return (verdict, reported_s, wall_s, rc).
       Writes to a tmp csv and parses, then returns the row contents."""
    tmp_csv = RES / f"_retry_{bench}_tmp.csv"
    if tmp_csv.exists():
        tmp_csv.unlink()
    cmd = ["bash", str(RUN_INSTANCE), bench, onnx_abs, vnn_abs,
           str(t_s), str(tmp_csv)]
    t0 = time.time()
    try:
        subprocess.run(cmd, check=False, timeout=int(t_s) + 120)
    except subprocess.TimeoutExpired:
        pass
    wall = time.time() - t0
    new_row = None
    if tmp_csv.exists():
        rows = list(csv.reader(open(tmp_csv)))
        if rows:
            new_row = rows[-1]
        tmp_csv.unlink()
    return new_row, wall


def retry_bench(bench: str) -> tuple:
    """Returns (n_retried, n_recovered)."""
    csv_p = RES / bench / "results.csv"
    logs = RES / bench / "logs"
    if not csv_p.exists() or not logs.exists():
        return 0, 0
    rows = list(csv.reader(open(csv_p)))
    header, body = rows[0], rows[1:]
    oom_idx = []
    for i, r in enumerate(body):
        if len(r) < 7 or r[3] != "error":
            continue
        if is_oom_err(logs, r[1], r[2]):
            oom_idx.append(i)
    if not oom_idx:
        return 0, 0
    print(f"[{bench}] {len(oom_idx)} OOM-flagged rows to retry")
    recovered = 0
    for k, i in enumerate(oom_idx, 1):
        r = body[i]
        onnx = r[1]
        vnn  = r[2]
        # Resolve relative paths for timeout lookup
        onnx_rel = f"onnx/{Path(onnx).name}"
        vnn_rel  = f"vnnlib/{Path(vnn).name}"
        t_s = get_timeout_for(bench, onnx_rel, vnn_rel)
        if t_s == 0:
            print(f"  [{k}/{len(oom_idx)}] WARN: no timeout for {onnx_rel} {vnn_rel}; skipping")
            continue
        new_row, wall = rerun_one(bench, onnx, vnn, t_s)
        if new_row is None or len(new_row) < 4:
            print(f"  [{k}/{len(oom_idx)}] {wall:.1f}s  retry produced no row :: {Path(onnx).name} {Path(vnn).name}")
            continue
        old_verdict = r[3]
        new_verdict = new_row[3]
        if new_verdict != "error":
            body[i] = new_row
            recovered += 1
            print(f"  [{k}/{len(oom_idx)}] {wall:.1f}s  {old_verdict} -> {new_verdict}  :: {Path(onnx).name} {Path(vnn).name}")
        else:
            body[i] = new_row
            print(f"  [{k}/{len(oom_idx)}] {wall:.1f}s  still error  :: {Path(onnx).name} {Path(vnn).name}")
    # Atomic rewrite
    tmp = csv_p.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(body)
    tmp.replace(csv_p)
    return len(oom_idx), recovered


def main():
    benches = sorted(d.name for d in RES.iterdir()
                     if d.is_dir() and not d.name.startswith("_"))
    grand_retry = 0
    grand_recovered = 0
    t0 = time.time()
    for b in benches:
        n_retry, n_recovered = retry_bench(b)
        grand_retry += n_retry
        grand_recovered += n_recovered
    print(f"\n=== retry_oom summary ({(time.time()-t0)/60:.1f} min) ===")
    print(f"  total retried:  {grand_retry}")
    print(f"  recovered (no longer error): {grand_recovered}")
    print(f"  still error:    {grand_retry - grand_recovered}")


if __name__ == "__main__":
    main()
