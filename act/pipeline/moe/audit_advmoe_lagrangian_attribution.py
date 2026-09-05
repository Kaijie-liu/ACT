"""Independently audit the AdvMoE Lagrangian attribution grid extension."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from act.pipeline.moe.advmoe_lagrangian_attribution import (
    ENDPOINT_POSITIVE,
    POSITIVE,
    select_closest_residual_rows,
)
from act.pipeline.moe.published_moe_router_gradient_audit import _sha256


WORKSPACE = Path("/data1/Kane/MOE")


def _inside(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(WORKSPACE)
    return resolved


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _independent_aggregate(
    calls: list[dict[str, Any]], multipliers: list[float], tolerance: float
) -> dict[str, Any]:
    if len(calls) != len(multipliers) or not calls:
        return {"complete": False}
    if any(not bool(call.get("complete")) or call.get("status") == "ERROR" for call in calls):
        return {"complete": False}
    lower = np.asarray([call.get("lower_bounds", []) for call in calls], dtype=np.float64)
    if lower.shape != (len(calls), 9) or not np.isfinite(lower).all():
        return {"complete": False}
    selected = np.argmax(lower, axis=0)
    best = lower[selected, np.arange(9)]
    return {
        "complete": True,
        "lower_bounds": best,
        "selected_multipliers": np.asarray(multipliers)[selected],
        "minimum_lower_bound": float(best.min()),
        "status": POSITIVE if bool(np.all(best >= tolerance)) else "UNKNOWN_RELAXATION",
    }


def audit(config_path: Path, summary_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path)
    summary_path = _inside(summary_path)
    config = _json(config_path)
    summary = _json(summary_path)
    issues: list[str] = []
    if summary.get("config", {}).get("sha256") != _sha256(config_path):
        issues.append("summary/config hash mismatch")
    parent_rows_entry = config["parent_development"]["rows"]
    parent_rows_path = _inside(Path(parent_rows_entry["path"]))
    if _sha256(parent_rows_path) != parent_rows_entry["sha256"]:
        issues.append("parent row hash mismatch")
    parent_rows = _jsonl(parent_rows_path)
    selected = select_closest_residual_rows(parent_rows, int(config["selection"]["rows"]))
    expected_ids = [row["row_id"] for row in selected]
    if expected_ids != config["selection"]["expected_row_ids"]:
        issues.append("frozen selection does not match independently derived selection")
    if summary.get("parent", {}).get("selected_row_ids") != expected_ids:
        issues.append("summary selected row IDs mismatch")

    rows_path = _inside(Path(summary["rows"]["path"]))
    if _sha256(rows_path) != summary["rows"]["sha256"]:
        issues.append("attribution row hash mismatch")
    rows = _jsonl(rows_path)
    if [row.get("row_id") for row in rows] != expected_ids:
        issues.append("attribution row order mismatch")
    parent_by_id = {row["row_id"]: row for row in selected}
    parent_multipliers = [
        float(value) for value in summary["multiplier_grid"]["parent"]
    ]
    extension_multipliers = [
        float(value) for value in config["multiplier_extension"]["multipliers"]
    ]
    if summary["multiplier_grid"].get("extension") != extension_multipliers:
        issues.append("extension multiplier grid mismatch")

    tolerance = float(config["numerical"]["safe_positive_margin"])
    complete_gains = 0
    property_improvements = 0
    for row in rows:
        parent = parent_by_id.get(row.get("row_id"))
        if parent is None:
            continue
        if row.get("parent_prediction_flip_witness"):
            issues.append(f"selected row has a parent prediction witness: {row['row_id']}")
        combined_positive = True
        for route, branch in enumerate(row.get("branches", [])):
            parent_branch = parent["lagrangian_guard_crown"][route]
            extension = branch.get("extension")
            calls = list(parent_branch["calls"])
            multipliers = list(parent_multipliers)
            if parent_branch["status"] == POSITIVE:
                if extension is not None:
                    issues.append(f"unnecessary positive-branch extension: {row['row_id']}:{route}")
            else:
                if extension is None:
                    issues.append(f"missing residual-branch extension: {row['row_id']}:{route}")
                    combined_positive = False
                    continue
                new_calls = extension.get("calls", [])
                observed = [float(call.get("lagrangian_multiplier", math.nan)) for call in new_calls]
                if observed != extension_multipliers:
                    issues.append(f"extension calls/grid mismatch: {row['row_id']}:{route}")
                calls.extend(new_calls)
                multipliers.extend(extension_multipliers)
            aggregate = _independent_aggregate(calls, multipliers, tolerance)
            if not aggregate.get("complete"):
                issues.append(f"incomplete independent aggregate: {row['row_id']}:{route}")
                combined_positive = False
                continue
            recorded = branch["combined"]
            if not np.allclose(
                aggregate["lower_bounds"], recorded.get("lower_bounds", []),
                rtol=0.0, atol=1e-12,
            ):
                issues.append(f"combined lower-bound mismatch: {row['row_id']}:{route}")
            if not np.allclose(
                aggregate["selected_multipliers"], recorded.get("selected_multipliers", []),
                rtol=0.0, atol=1e-15,
            ):
                issues.append(f"selected multiplier mismatch: {row['row_id']}:{route}")
            if aggregate["status"] != recorded.get("status"):
                issues.append(f"combined status mismatch: {row['row_id']}:{route}")
            old = np.asarray(parent_branch["lower_bounds"], dtype=np.float64)
            delta = aggregate["lower_bounds"] - old
            expected_improved = int(np.sum(delta > tolerance))
            if branch.get("strict_property_improvements") != expected_improved:
                issues.append(f"property improvement mismatch: {row['row_id']}:{route}")
            if int(np.sum(delta < -tolerance)) != 0:
                issues.append(f"combined grid regressed a property: {row['row_id']}:{route}")
            property_improvements += expected_improved
            combined_positive &= aggregate["status"] == POSITIVE
        if len(row.get("branches", [])) != 2:
            issues.append(f"branch count mismatch: {row['row_id']}")
            combined_positive = False
        if bool(row.get("combined_endpoint_positive")) != combined_positive:
            issues.append(f"endpoint status mismatch: {row['row_id']}")
        parent_positive = parent["statuses"]["lagrangian_guard_ablation"] == ENDPOINT_POSITIVE
        complete_gains += int(combined_positive and not parent_positive)

    expected_outcome = (
        "FINITE_GRID_TRUNCATION_CONFIRMED_AT_ENDPOINT"
        if complete_gains
        else (
            "FINITE_GRID_CONTRIBUTES_WITHOUT_ENDPOINT_GAIN"
            if property_improvements
            else "FINITE_GRID_NOT_PRIMARY_ON_CLOSEST_RESIDUALS"
        )
    )
    recorded_outcome = summary.get("outcome", {})
    if recorded_outcome.get("complete_endpoint_gains") != complete_gains:
        issues.append("summary complete-gain count mismatch")
    if recorded_outcome.get("strict_property_improvements") != property_improvements:
        issues.append("summary property-improvement count mismatch")
    if recorded_outcome.get("classification") != expected_outcome:
        issues.append("summary attribution classification mismatch")
    if summary.get("status") != "COMPLETED_NOT_INDEPENDENTLY_AUDITED":
        issues.append("unexpected runner status")
    return {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "scope": "ADV_MOE_LAGRANGIAN_ATTRIBUTION_GRID_EXTENSION_AUDIT_R1",
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "result": {"path": str(summary_path), "sha256": _sha256(summary_path)},
        "rows": {"path": str(rows_path), "sha256": _sha256(rows_path)},
        "independent_counts": {
            "selected_rows": len(rows),
            "complete_endpoint_gains": complete_gains,
            "strict_property_improvements": property_improvements,
        },
        "issues": issues,
    }


def _write(path: Path, value: dict[str, Any]) -> None:
    path = _inside(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(arguments.config, arguments.summary)
    _write(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
