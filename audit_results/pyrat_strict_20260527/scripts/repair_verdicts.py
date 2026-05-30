#!/usr/bin/env python3
"""
Re-parse already-completed PyRAT logs and rewrite verdicts in results.csv.

Use this after fixing the carriage-return parsing bug in run_instance.sh:
some instances that printed "Result = True" buried inside a tqdm progress
line were mis-classified as 'error' or 'timeout'. This script reads each
.out log directly, finds the *real* Result line (carriage-return aware),
and rewrites the CSV row in-place.

Safe to run while the supervisor is still active: it only touches CSV rows
whose .out file already exists. New rows appended after the script will be
parsed correctly by the patched run_instance.sh.
"""
import csv
import re
import sys
from pathlib import Path

PYRAT_DIR = Path("/data1/Kane/pyrat")
RES_DIR = PYRAT_DIR / "results_pure"

RESULT_RE = re.compile(r"Result\s*=\s*(\w+),\s*Time\s*=\s*([0-9.]+)\s*s")
VERDICT_MAP = {"True": "verified", "False": "falsified",
               "Unknown": "unknown", "Timeout": "timeout"}


def parse_log(log_path: Path):
    """Read the .out file, strip CRs, find the final Result line."""
    try:
        data = log_path.read_bytes().replace(b"\r", b"\n").decode("utf-8", "replace")
    except FileNotFoundError:
        return None, None
    last = None
    for m in RESULT_RE.finditer(data):
        last = m
    if last is None:
        return None, None
    return VERDICT_MAP.get(last.group(1), "error"), last.group(2)


def repair_csv(bench_dir: Path):
    csv_p = bench_dir / "results.csv"
    if not csv_p.exists():
        return None
    logs_dir = bench_dir / "logs"
    if not logs_dir.exists():
        return None
    rows = list(csv.reader(open(csv_p)))
    header, body = rows[0], rows[1:]
    changed = 0
    for row in body:
        if len(row) < 7:
            continue
        bench, onnx, vnnlib, verdict, wall, reported, rc = row[:7]
        tag = f"{Path(onnx).stem}__{Path(vnnlib).stem}"
        log_path = logs_dir / f"{tag}.out"
        new_v, new_t = parse_log(log_path)
        if new_v is None:
            continue
        if new_v != verdict:
            row[3] = new_v
            row[5] = new_t
            changed += 1
    # Atomic rewrite
    tmp = csv_p.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(body)
    tmp.replace(csv_p)
    return changed


def main():
    targets = sys.argv[1:]
    if not targets:
        targets = [p.name for p in RES_DIR.iterdir()
                   if p.is_dir() and not p.name.startswith("_")]
    grand_changed = 0
    for b in sorted(targets):
        d = RES_DIR / b
        if not d.is_dir():
            continue
        n = repair_csv(d)
        if n is None:
            continue
        print(f"  {b}: rewrote {n} verdicts")
        grand_changed += n
    print(f"total changed: {grand_changed}")


if __name__ == "__main__":
    main()
