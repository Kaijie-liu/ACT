"""Audit the partial source-native and one-hop snowball survey artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from act.pipeline.moe.experiment1 import PROJECT_ROOT, _inside, _sha256, _write_json


ALLOWED_EXCLUSIONS = {
    "E_ATTACK_ONLY",
    "E_ROUTER_ONLY",
    "E_EXPERT_ONLY",
    "E_NO_DYNAMIC_ROUTING",
    "E_NO_BOUNDED_INPUT_PROPERTY",
    "E_POSITION_OR_SURVEY",
    "E_DUPLICATE_VERSION",
    "E_NO_FULL_TEXT",
    "E_OUTSIDE_WINDOW",
}


def audit_document(document: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if document.get("status") != "PARTIAL_RETRIEVAL_NO_PREVALENCE":
        issues.append("artifact lacks the frozen partial/no-prevalence status")
    if document.get("prevalence_claim_allowed") is not False:
        issues.append("prevalence claims are not explicitly disabled")
    if document.get("zero_citation_counts_are_not_absence_evidence") is not True:
        issues.append("zero citation counts are not labeled as coverage-limited")
    if document.get("author_contact") != "NOT_PERFORMED":
        issues.append("artifact records unauthorized author contact")
    sources = document.get("source_native_export_audit", [])
    if len(sources) != 11 or len({row.get("source") for row in sources}) != len(sources):
        issues.append("source-native audit does not contain eleven unique sources")
    for row in sources:
        if not str(row.get("endpoint", "")).startswith("https://"):
            issues.append(f"source {row.get('source')} lacks an HTTPS endpoint")
        if not row.get("status") or not row.get("limitation"):
            issues.append(f"source {row.get('source')} lacks status or limitation")
    candidates = document.get("snowball_candidates", [])
    if len(candidates) != 13:
        issues.append("snowball candidate count is not the frozen thirteen")
    keys = [row.get("dedup_key") for row in candidates]
    urls = [row.get("primary_url") for row in candidates]
    if len(set(keys)) != len(keys):
        issues.append("snowball candidates contain duplicate dedup keys")
    if len(set(urls)) != len(urls):
        issues.append("snowball candidates contain duplicate primary URLs")
    for row in candidates:
        if row.get("preliminary_decision") != "EXCLUDE":
            issues.append(f"candidate {row.get('dedup_key')} is not conservatively excluded")
        if row.get("exclusion_code") not in ALLOWED_EXCLUSIONS:
            issues.append(f"candidate {row.get('dedup_key')} has an invalid exclusion code")
        if not str(row.get("primary_url", "")).startswith("https://"):
            issues.append(f"candidate {row.get('dedup_key')} lacks a primary HTTPS URL")
        if not row.get("discovery_edges") or not row.get("rationale"):
            issues.append(f"candidate {row.get('dedup_key')} lacks discovery evidence or rationale")
    coverage = document.get("snowball_coverage", {})
    if coverage.get("non_seed_candidates") != len(candidates):
        issues.append("snowball coverage count differs from candidate rows")
    if coverage.get("new_included_families") != 0:
        issues.append("partial snowballing unexpectedly claims a new included family")
    if len(coverage.get("seed_families_reencountered_and_deduplicated", [])) != 4:
        issues.append("seed-family deduplication count is not four")
    return issues


def audit(path: Path) -> dict[str, Any]:
    path = _inside(path, PROJECT_ROOT)
    document = json.loads(path.read_text(encoding="utf-8"))
    issues = audit_document(document)
    return {
        "schema_version": 1,
        "scope": "partial_survey_source_and_snowball_audit",
        "status": document.get("status"),
        "issue_count": len(issues),
        "issues": issues,
        "source_rows": len(document.get("source_native_export_audit", [])),
        "snowball_candidates": len(document.get("snowball_candidates", [])),
        "new_included_families": document.get("snowball_coverage", {}).get(
            "new_included_families"
        ),
        "artifact_sha256": _sha256(path),
        "prevalence_claim_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = _inside(args.output, PROJECT_ROOT)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite survey audit {output}")
    result = audit(args.input)
    _write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["issue_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
