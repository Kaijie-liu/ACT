"""Roll up clean canonical sweep — attach provenance hashes + tally verdicts.

Per advisor 2026-06-04 §9. For each per-instance JSON written by
watchdog_runner under ``<ROOT>/<benchmark>/``:
  1. resolve the canonical iid from the vnnlib basename.
  2. compute the provenance hash bundle via canonical_provenance.
  3. write a sidecar `<iid>_provenance.json`.
  4. tally V / A / U / TIMEOUT / OOM / ERROR per benchmark.

Output:
  <ROOT>/clean_canonical_summary.json
  <ROOT>/clean_canonical_summary.csv
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ACT_ROOT = Path("/data1/Kane/ACT")
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))

from research.canonical_provenance import (  # noqa: E402
    CANONICAL_ROOT, canonical_instances_rows,
    canonical_instances_csv_sha256, sha256_file,
)


VERDICT_BUCKETS = (
    "FALSIFIED", "CERTIFIED", "VERIFIED",
    "UNKNOWN", "UNKNOWN_TIMEOUT", "UNKNOWN_RESOURCE_LIMIT",
    "ERROR", "ERROR_WATCHDOG_EXIT_NONZERO",
)


def resolve_iid(bench: str, vnn_path: str) -> int:
    """Reverse-lookup iid from vnnlib basename."""
    name = os.path.basename(vnn_path)
    rows = canonical_instances_rows(bench)
    for i, (_onnx, vnn, _to) in enumerate(rows):
        if os.path.basename(vnn) == name:
            return i
    return -1


def main(root: Path) -> int:
    bench_dirs = [
        d for d in sorted(root.iterdir())
        if d.is_dir() and (
            (d / "run.log").exists()
            or list(d.glob("per_instance*.json"))
        )
    ]
    print(f"[rollup] root={root}")
    print(f"[rollup] benchmark dirs: {len(bench_dirs)}")

    summary: Dict[str, Dict[str, Any]] = {}
    for bench_dir in bench_dirs:
        bench = bench_dir.name
        print(f"[rollup] {bench}")
        try:
            csv_sha = canonical_instances_csv_sha256(bench)
            n_rows = len(canonical_instances_rows(bench))
        except Exception as e:
            print(f"[rollup]   skip (no canonical csv): {e}")
            continue

        bucket_counts: Counter = Counter()
        per_instance: List[Dict[str, Any]] = []
        fal_receipts: List[str] = []
        for pjson in sorted(bench_dir.glob("per_instance*.json")):
            try:
                d = json.load(open(pjson))
            except Exception:
                continue
            rows = d.get("per_instance", [])
            for r in rows:
                iid = int(r.get("official_instance_id", -1))
                rep = r.get("reportable_status", "")
                vnn = r.get("vnnlib_spec", "") or r.get("spec_path", "")
                onnx = r.get("onnx_model", "") or r.get("model_path", "")
                # Build provenance for this iid.
                onnx_p = (CANONICAL_ROOT / bench / onnx) if onnx else None
                vnn_p = (CANONICAL_ROOT / bench / vnn) if vnn else None
                onnx_sha = (sha256_file(onnx_p) if (onnx_p and onnx_p.exists())
                            else "")
                vnn_sha = (sha256_file(vnn_p) if (vnn_p and vnn_p.exists())
                           else "")
                prov = {
                    "canonical_root": str(CANONICAL_ROOT),
                    "benchmark": bench,
                    "iid": iid,
                    "instances_csv_sha256": csv_sha,
                    "onnx_path": str(onnx_p) if onnx_p else "",
                    "onnx_sha256": onnx_sha,
                    "vnnlib_path": str(vnn_p) if vnn_p else "",
                    "vnnlib_sha256": vnn_sha,
                }
                # Sidecar provenance.
                if iid >= 0:
                    (bench_dir / f"iid{iid:03d}_provenance.json").write_text(
                        json.dumps(prov, indent=2)
                    )
                per_instance.append({
                    "iid": iid,
                    "reportable": rep,
                    "wall_s": r.get("wall_s"),
                    "vnnlib_sha256": vnn_sha,
                    "onnx_sha256": onnx_sha,
                    "fal_receipt_path": r.get("q_solver_stats", [{}])[0].get(
                        "fal_receipt_path", "") if r.get("q_solver_stats")
                        else "",
                })
                bucket_counts[rep] += 1
                if rep == "FALSIFIED":
                    fal_receipts.append(str(pjson))
        print(
            f"[rollup]   covered={sum(bucket_counts.values())}/{n_rows}  "
            f"FAL={bucket_counts.get('FALSIFIED', 0)}  "
            f"CERT={bucket_counts.get('CERTIFIED', 0) + bucket_counts.get('VERIFIED', 0)}  "
            f"UNK={sum(v for k, v in bucket_counts.items() if k.startswith('UNKNOWN'))}  "
            f"ERR={sum(v for k, v in bucket_counts.items() if k.startswith('ERROR'))}"
        )
        summary[bench] = {
            "n_canonical": n_rows,
            "n_covered": sum(bucket_counts.values()),
            "csv_sha256": csv_sha,
            "verdict_counts": dict(bucket_counts),
            "n_fal_receipts": len(fal_receipts),
            "per_instance": per_instance,
        }

    out_json = root / "clean_canonical_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[rollup] summary: {out_json}")

    # CSV view
    out_csv = root / "clean_canonical_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "benchmark", "n_canonical", "n_covered",
            "FALSIFIED", "CERTIFIED+VERIFIED", "UNKNOWN_any", "ERROR_any",
            "csv_sha256_short",
        ])
        for bench, s in summary.items():
            bc = s["verdict_counts"]
            cert = bc.get("CERTIFIED", 0) + bc.get("VERIFIED", 0)
            unk = sum(v for k, v in bc.items() if k.startswith("UNKNOWN"))
            err = sum(v for k, v in bc.items() if k.startswith("ERROR"))
            w.writerow([
                bench, s["n_canonical"], s["n_covered"],
                bc.get("FALSIFIED", 0), cert, unk, err,
                s["csv_sha256"][:16],
            ])
    print(f"[rollup] csv: {out_csv}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: clean_canonical_sweep_rollup.py <ROOT>")
        sys.exit(1)
    sys.exit(main(Path(sys.argv[1])))
