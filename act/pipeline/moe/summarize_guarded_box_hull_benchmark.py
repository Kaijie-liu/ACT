"""Publish an audited guarded-box-hull engineering benchmark summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any, Sequence

from act.pipeline.moe.experiment1 import (
    PROJECT_ROOT,
    WRITE_ROOT,
    _inside,
    _sha256,
    _write_json,
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def summarize(branches: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ratios = [
        float(branch["scipy"]["wall_seconds"])
        / float(branch["highspy"]["wall_seconds"])
        for branch in branches
        if float(branch["highspy"]["wall_seconds"]) > 0.0
    ]
    highspy_seconds = sum(float(branch["highspy"]["wall_seconds"]) for branch in branches)
    scipy_seconds = sum(float(branch["scipy"]["wall_seconds"]) for branch in branches)
    max_difference = max(
        (float(branch["bound_max_abs_diff"]) for branch in branches),
        default=None,
    )
    return {
        "scope": "engineering_performance_rerun_not_confirmatory_overwrite",
        "rows": len(rows),
        "route_branches": len(branches),
        "all_rows_complete": all(bool(row.get("complete")) for row in rows),
        "all_paired_complete": all(bool(branch.get("paired_complete")) for branch in branches),
        "all_bounds_within_tolerance": all(
            bool(branch.get("within_tolerance")) for branch in branches
        ),
        "bound_max_abs_difference": max_difference,
        "backend_order_counts": {
            "highspy->scipy": sum(
                branch.get("backend_order") == ["highspy", "scipy"]
                for branch in branches
            ),
            "scipy->highspy": sum(
                branch.get("backend_order") == ["scipy", "highspy"]
                for branch in branches
            ),
        },
        "highspy": {
            "wall_seconds": highspy_seconds,
            "model_builds": sum(
                int(branch["highspy"]["telemetry"]["model_builds"])
                for branch in branches
            ),
            "solves": sum(
                int(branch["highspy"]["telemetry"]["solves"])
                for branch in branches
            ),
            "cold_start_solves": sum(
                int(branch["highspy"]["telemetry"]["cold_start_solves"])
                for branch in branches
            ),
            "fallback_sides": sum(
                int(branch["highspy"]["fallback_sides"]) for branch in branches
            ),
        },
        "scipy": {
            "wall_seconds": scipy_seconds,
            "model_builds": sum(
                int(branch["scipy"]["telemetry"]["model_builds"])
                for branch in branches
            ),
            "solves": sum(
                int(branch["scipy"]["telemetry"]["solves"])
                for branch in branches
            ),
            "cold_start_solves": sum(
                int(branch["scipy"]["telemetry"]["cold_start_solves"])
                for branch in branches
            ),
            "fallback_sides": sum(
                int(branch["scipy"]["fallback_sides"]) for branch in branches
            ),
        },
        "speed": {
            "eligible": bool(branches)
            and all(bool(branch.get("paired_complete")) for branch in branches)
            and all(bool(branch.get("within_tolerance")) for branch in branches),
            "scipy_over_highspy_wall_ratio_median": statistics.median(ratios)
            if ratios
            else None,
            "scipy_over_highspy_total_wall_ratio": scipy_seconds / highspy_seconds
            if highspy_seconds > 0.0
            else None,
            "interpretation": (
                "descriptive paired engineering speed ratio; accepted basis "
                "submission is recorded without a solver-internal warm-start claim"
            ),
        },
        "original_confirmatory_overall_solved_rate_immutable": 0.56,
    }


def publish(result_dir: Path, audit_path: Path, output_path: Path) -> dict[str, Any]:
    result_dir = _inside(result_dir, WRITE_ROOT)
    audit_path = _inside(audit_path, WRITE_ROOT)
    output_path = _inside(output_path, PROJECT_ROOT)
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite published result {output_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("issue_count") != 0:
        raise RuntimeError("guarded-box benchmark cannot be published with audit issues")
    if not audit.get("speed_conclusion_eligible"):
        raise RuntimeError("guarded-box benchmark is not eligible for a speed conclusion")
    branches = _jsonl(result_dir / "branches.jsonl")
    rows = _jsonl(result_dir / "rows.jsonl")
    result = {
        "schema_version": 1,
        "result": summarize(branches, rows),
        "independent_audit": {
            "issue_count": audit["issue_count"],
            "bound_artifacts_replayed": audit["bound_artifacts_replayed"],
            "bound_disagreements": audit["bound_disagreements"],
            "sha256": _sha256(audit_path),
        },
        "artifact_sha256": {
            "runtime_config": _sha256(result_dir / "config.json"),
            "branches_jsonl": _sha256(result_dir / "branches.jsonl"),
            "rows_jsonl": _sha256(result_dir / "rows.jsonl"),
            "runner_summary": _sha256(result_dir / "summary.json"),
        },
    }
    _write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.result_dir, args.audit, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
