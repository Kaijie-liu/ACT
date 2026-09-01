"""Independent semantic audit for the true monolithic F0 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from act.pipeline.moe.experiment1 import _sha256, _write_json
from act.pipeline.moe.monolithic_f0_baseline import summarize


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(output_dir: Path) -> dict:
    runtime = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    config = runtime["config"]
    rows = _rows(output_dir / "results.jsonl")
    reference_path = Path(config["route_a_reference_results"])
    reference_all = _rows(reference_path)
    ranks = {int(row["sample_rank"]) for row in rows}
    reference = [row for row in reference_all if int(row["sample_rank"]) in ranks]
    issues: list[str] = []
    expected = 1 if runtime["smoke"] else int(config["expected_sample_ranks"])
    if len(rows) != expected:
        issues.append(f"row count {len(rows)} != {expected}")
    if len(ranks) != len(rows):
        issues.append("duplicate sample rank")
    for row in rows:
        if row["status"] == "UNSAFE" and not row.get("full_model_witness_valid"):
            issues.append(f"rank {row['sample_rank']} has unvalidated UNSAFE")
        if row["status"] == "SAFE" and not row.get("properties"):
            issues.append(f"rank {row['sample_rank']} SAFE has no property rows")
        if row["status"] == "SAFE" and not all(
            item["status"] == "SAFE" for item in row["properties"]
        ):
            issues.append(f"rank {row['sample_rank']} SAFE has an unresolved property")
        if row.get("formulation") and "single disjunctive MILP" not in row["formulation"]:
            issues.append(f"rank {row['sample_rank']} formulation identity changed")
        witness = row.get("witness_path")
        if witness and _sha256(output_dir / witness) != row.get("witness_sha256"):
            issues.append(f"rank {row['sample_rank']} witness hash differs")
    recomputed = summarize(rows, reference)
    stored = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    if recomputed != stored:
        issues.append("summary differs from independent recomputation")
    report = {
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "raw_hashes": {
            "results.jsonl": _sha256(output_dir / "results.jsonl"),
            "config.json": _sha256(output_dir / "config.json"),
        },
        "recomputed_summary": recomputed,
    }
    _write_json(output_dir / "audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    args = parser.parse_args()
    report = audit(Path(args.output_dir))
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
