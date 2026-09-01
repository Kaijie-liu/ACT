# ===- act/pipeline/moe/audit_exact_big_m_engineering.py ------------====#
"""Independent audit for the paired exact-support big-M experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from act.pipeline.moe.exact_big_m_engineering import (
    paired_audit_issues,
    summarize_conditions,
)
from act.pipeline.moe.experiment1 import _sha256, _write_json


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(output_dir: Path) -> dict:
    runtime = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    conditions = _jsonl(output_dir / "conditions.jsonl")
    experts = _jsonl(output_dir / "experts.jsonl")
    expected = int(runtime["config"]["expected_sample_ranks"])
    issues = paired_audit_issues(conditions, experts, expected_ranks=expected)
    recomputed = summarize_conditions(conditions)
    stored = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    if recomputed != stored:
        issues.append("stored summary differs from independent recomputation")
    report = {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "issue_count": len(issues),
        "raw_hashes": {
            "conditions.jsonl": _sha256(output_dir / "conditions.jsonl"),
            "experts.jsonl": _sha256(output_dir / "experts.jsonl"),
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
