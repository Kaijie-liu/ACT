"""Validate the partial six-dimension certification-gap evidence matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from act.pipeline.moe.experiment1 import PROJECT_ROOT, _inside, _sha256, _write_json


EXPECTED_STATUS = "PARTIAL_RETRIEVAL_NO_PREVALENCE"


def validate_matrix(
    protocol: dict[str, Any],
    screening: dict[str, Any],
    matrix: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    coding = protocol["coding"]
    dimensions = tuple(coding)
    records = matrix.get("records", [])
    if matrix.get("status") != EXPECTED_STATUS:
        issues.append("matrix does not retain the partial/no-prevalence status")
    if matrix.get("author_contact") != "NOT_PERFORMED":
        issues.append("matrix claims an unauthorized author contact")
    expected_indices = {
        int(record["record_index"])
        for record in screening.get("included_records", [])
    }
    actual_indices = [int(record["record_index"]) for record in records]
    if set(actual_indices) != expected_indices:
        issues.append("matrix records differ from adjudicated included records")
    if len(set(actual_indices)) != len(actual_indices):
        issues.append("matrix contains duplicate record indices")
    required_semantics = set(protocol["required_semantic_fields"])
    for record in records:
        identity = f"record {record.get('record_index')}"
        observed = record.get("dimensions", {})
        if set(observed) != set(dimensions):
            issues.append(f"{identity} does not contain exactly six frozen dimensions")
        for dimension in dimensions:
            value = observed.get(dimension, {})
            if value.get("code") not in coding[dimension]:
                issues.append(f"{identity} has invalid enum for {dimension}")
            evidence = value.get("evidence", [])
            if not evidence:
                issues.append(f"{identity}/{dimension} has no primary-source evidence")
            for item in evidence:
                if not all(item.get(key) for key in ("url", "locator", "finding")):
                    issues.append(f"{identity}/{dimension} has incomplete evidence")
                if not str(item.get("url", "")).startswith("https://"):
                    issues.append(f"{identity}/{dimension} uses a non-HTTPS source")
        semantics = record.get("semantics", {})
        if set(semantics) != required_semantics:
            issues.append(f"{identity} lacks a frozen semantic field")
        if not record.get("primary_sources"):
            issues.append(f"{identity} has no primary source")
    claim_limit = str(matrix.get("claim_limit", "")).lower()
    if "does not estimate" not in claim_limit or "prevalence" not in claim_limit:
        issues.append("claim limit does not explicitly forbid prevalence inference")
    return issues


def audit(protocol_path: Path, screening_path: Path, matrix_path: Path) -> dict[str, Any]:
    protocol_path = _inside(protocol_path, PROJECT_ROOT)
    screening_path = _inside(screening_path, PROJECT_ROOT)
    matrix_path = _inside(matrix_path, PROJECT_ROOT)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    screening = json.loads(screening_path.read_text(encoding="utf-8"))
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    issues = validate_matrix(protocol, screening, matrix)
    records = matrix.get("records", [])
    dimensions = protocol["coding"]
    return {
        "schema_version": 1,
        "status": EXPECTED_STATUS,
        "scope": "independent_schema_and_evidence_completeness_audit",
        "issues": issues,
        "issue_count": len(issues),
        "records": len(records),
        "dimensions_per_record": len(dimensions),
        "coded_cells": len(records) * len(dimensions),
        "all_cells_have_primary_source_locator": not issues,
        "author_contact": matrix.get("author_contact"),
        "prevalence_claim_allowed": False,
        "artifact_sha256": {
            "protocol": _sha256(protocol_path),
            "screening": _sha256(screening_path),
            "matrix": _sha256(matrix_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--screening", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.protocol, args.screening, args.matrix)
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["issue_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
