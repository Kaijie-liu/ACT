#!/usr/bin/env python3
"""Build one final-outcome row per ``(source, official instance id)``.

The source directories contain two forms of authoritative records:

* normal ``per_instance_*.json`` records for completed solver calls;
* watchdog synthetic records for instances terminated by wall/RSS bounds.

Both must be retained.  When a watchdog record supersedes an in-flight child
record for the same iid, the watchdog outcome wins; otherwise a timed-out
instance could be incorrectly persisted as an ordinary solver UNKNOWN.

Usage:
    cd /data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/CONSOLIDATED_RESULTS
    python3 build_csvs.py
"""

from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DST = Path(__file__).resolve().parent
CANONICAL_ROOT = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks")
CSV_KEYS = [
    "source",
    "iid",
    "verdict",
    "internal_status",
    "reportable_status",
    "count_bucket",
    "wall_s",
    "watchdog_status",
    "watchdog_synthetic",
    "strict_bounded_failure",
    "run_status",
    "peak_rss_mb",
    "onnx_model",
    "vnnlib_spec",
    "q_statuses",
    "q_reportables",
    "q_receipts",
    "error",
    "json_path",
]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as fp:
            return json.load(fp)
    except (OSError, json.JSONDecodeError):
        return None


def _iid(record: dict[str, Any]) -> Any:
    return record.get(
        "official_instance_id",
        record.get("instance_index", record.get("iid", "?")),
    )


def _manifest(benchmark: str) -> dict[int, tuple[str, str]]:
    path = CANONICAL_ROOT / benchmark / "instances.csv"
    if not path.exists():
        return {}
    rows: dict[int, tuple[str, str]] = {}
    with path.open(newline="") as fp:
        for iid, row in enumerate(csv.reader(fp)):
            if len(row) >= 2:
                rows[iid] = (row[0], row[1])
    return rows


def _json_cell(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def _normal_row(
    source_label: str,
    path: Path,
    document: dict[str, Any],
    record: dict[str, Any],
    manifest: dict[int, tuple[str, str]],
) -> dict[str, Any]:
    iid = _iid(record)
    model = record.get("onnx_model", "")
    spec = record.get("vnnlib_spec", "")
    if isinstance(iid, int) and iid in manifest:
        model = model or manifest[iid][0]
        spec = spec or manifest[iid][1]
    return {
        "source": source_label,
        "iid": iid,
        "verdict": record.get("cli_normalized", "?"),
        "internal_status": record.get("internal_status", ""),
        "reportable_status": record.get("reportable_status", ""),
        "count_bucket": record.get("count_bucket", ""),
        "wall_s": round(float(record.get("wall_s", 0) or 0), 3),
        "watchdog_status": record.get("watchdog_status", ""),
        "watchdog_synthetic": bool(document.get("watchdog_synthetic", False)),
        "strict_bounded_failure": document.get("strict_bounded_failure", ""),
        "run_status": document.get("run_status", ""),
        "peak_rss_mb": record.get("peak_rss_mb", ""),
        "onnx_model": model,
        "vnnlib_spec": spec,
        "q_statuses": _json_cell(record.get("q_statuses")),
        "q_reportables": _json_cell(record.get("q_reportables")),
        "q_receipts": _json_cell(record.get("q_receipts")),
        "error": record.get("error", ""),
        "json_path": str(path),
    }


def _receipt_rows(
    src_dir: Path, source_label: str, manifest: dict[int, tuple[str, str]]
) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(glob.glob(str(src_dir / "*_q*_*.json"))):
        path = Path(name)
        if "per_instance" in path.name or "watchdog" in path.name:
            continue
        document = _read_json(path)
        if document is None:
            continue
        witness_id = document.get("witness_id", {})
        iid = document.get(
            "official_instance_id",
            document.get("instance_index", document.get("iid", witness_id.get("instance_id", ""))),
        )
        model = document.get("onnx_model", "")
        spec = document.get("vnnlib_spec", "")
        if isinstance(iid, int) and iid in manifest:
            model = model or manifest[iid][0]
            spec = spec or manifest[iid][1]
        rows.append(
            {
                "source": source_label,
                "iid": iid,
                "verdict": document.get("verdict", document.get("status", "RECEIPT_ONLY")),
                "internal_status": "",
                "reportable_status": "",
                "count_bucket": "",
                "wall_s": round(float(document.get("wall_s", 0) or 0), 3),
                "watchdog_status": "",
                "watchdog_synthetic": False,
                "strict_bounded_failure": "",
                "run_status": "",
                "peak_rss_mb": "",
                "onnx_model": model,
                "vnnlib_spec": spec,
                "q_statuses": "",
                "q_reportables": "",
                "q_receipts": "",
                "error": "",
                "json_path": str(path),
            }
        )
    return rows


def harvest(src_dir: Path, source_label: str, benchmark: str) -> list[dict[str, Any]]:
    manifest = _manifest(benchmark)
    candidates: list[dict[str, Any]] = []
    for path in sorted(src_dir.glob("per_instance_*.json")):
        document = _read_json(path)
        if document is None:
            continue
        for record in document.get("per_instance", []):
            candidates.append(_normal_row(source_label, path, document, record, manifest))
    if not candidates:
        return _receipt_rows(src_dir, source_label, manifest)

    selected: dict[Any, dict[str, Any]] = {}
    for row in candidates:
        iid = row["iid"]
        current = selected.get(iid)
        if current is None:
            selected[iid] = row
            continue
        if row["watchdog_synthetic"] and not current["watchdog_synthetic"]:
            selected[iid] = row
            continue
        if current["watchdog_synthetic"] and not row["watchdog_synthetic"]:
            continue
        if row["verdict"] != current["verdict"]:
            raise ValueError(
                f"{benchmark}/{source_label}: conflicting final records for iid={iid}: "
                f"{current['verdict']} vs {row['verdict']}"
            )
        if row["json_path"] > current["json_path"]:
            selected[iid] = row
    return [selected[iid] for iid in sorted(selected, key=lambda value: (str(type(value)), str(value)))]


def main() -> None:
    bench_dirs = sorted(path for path in DST.iterdir() if path.is_dir())
    for bench_dir in bench_dirs:
        sources = sorted(bench_dir.glob("_source_*"))
        if not sources:
            continue
        all_rows = []
        for source in sources:
            if source.is_symlink():
                label = source.name.removeprefix("_source_")
                all_rows.extend(harvest(source.resolve(), label, bench_dir.name))
        if not all_rows:
            print(f"  {bench_dir.name}: (no rows, {len(sources)} sources)")
            continue
        out_csv = bench_dir / "per_instance.csv"
        with out_csv.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=CSV_KEYS)
            writer.writeheader()
            writer.writerows(all_rows)
        by_src: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in all_rows:
            by_src[row["source"]][row["verdict"]] += 1
        print(f"  {bench_dir.name}: {len(all_rows)} rows -> {out_csv.name}")
        for source, counts in by_src.items():
            print(f"      {source:20s} ({sum(counts.values()):4d}): {dict(counts)}")


if __name__ == "__main__":
    main()
