"""Independently audit the official-scale RT-ER B3 artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256
from act.pipeline.moe.icml2025_b3 import (
    route_applicability_census,
    route_status_at_epsilon,
    select_boundary_cohort,
)


def crown_backend_failure_counts(
    rows: list[dict[str, Any]],
) -> tuple[int, int]:
    """Count backend exceptions and all branches lacking finite complete bounds."""

    backend_errors = sum(
        row.get("crown", {}).get("status") == "ERROR" for row in rows
    )
    incomplete_bounds = sum(
        not bool(row.get("crown", {}).get("complete", False)) for row in rows
    )
    return backend_errors, incomplete_bounds


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(prepare_path: Path, crown_path: Path, output_path: Path) -> dict[str, Any]:
    prepare_path = _inside(prepare_path, WRITE_ROOT)
    crown_path = _inside(crown_path, WRITE_ROOT)
    output_path = _inside(output_path, WRITE_ROOT)
    if output_path.exists():
        raise RuntimeError(f"B3 audit refuses to overwrite {output_path}")
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    crown = json.loads(crown_path.read_text(encoding="utf-8"))
    config_path = _inside(Path(prepare["config"]["path"]), PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if prepare.get("status") != "PREPARED_CROWN_NOT_RUN":
        issues.append("prepare status changed")
    if crown.get("status") != "COMPLETED_NUMERICAL_CONFORMANCE_ONLY":
        issues.append("CROWN result status changed")
    if prepare["config"]["sha256"] != _sha256(config_path):
        issues.append("config hash changed")
    if crown.get("prepare", {}).get("sha256") != _sha256(prepare_path):
        issues.append("CROWN prepare hash changed")
    if prepare.get("b2_gate", {}).get("status") != "COMPLETED_AUDITED":
        issues.append("B2 completion gate missing")
    if prepare.get("b2_gate", {}).get("audit_status") != "PASS":
        issues.append("B2 audit gate missing")
    if int(prepare.get("b2_gate", {}).get("issues", -1)) != 0:
        issues.append("B2 audit issues are nonzero")
    checkpoint = _inside(Path(prepare["checkpoint"]["path"]), WRITE_ROOT)
    if _sha256(checkpoint) != prepare["checkpoint"]["sha256"]:
        issues.append("checkpoint hash changed")
    telemetry_dir = _inside(Path(prepare["telemetry"]["directory"]), WRITE_ROOT)
    telemetry_path = telemetry_dir / "per_input.npz"
    telemetry_summary = telemetry_dir / "summary.json"
    if _sha256(telemetry_path) != prepare["telemetry"]["per_input_sha256"]:
        issues.append("telemetry arrays changed")
    if _sha256(telemetry_summary) != prepare["telemetry"]["summary_sha256"]:
        issues.append("telemetry summary changed")
    with np.load(telemetry_path, allow_pickle=False) as arrays:
        selection = config["selection"]
        expected_indices = select_boundary_cohort(
            arrays["clean_correct"],
            arrays["radius_uppers"],
            samples=int(selection["samples"]),
            multiplier=float(selection["route_radius_multiplier"]),
            cap=float(selection["route_radius_cap_over_255"]) / 255.0,
        )
        if prepare["selection"]["dataset_indices"] != expected_indices.tolist():
            issues.append("boundary cohort selection changed")
        expected_applicability = route_applicability_census(
            arrays["radius_lowers"],
            arrays["radius_uppers"],
            arrays["clean_correct"],
            config["applicability_epsilon_over_255"],
        )
        if prepare["route_applicability_census"] != expected_applicability:
            issues.append("route applicability census changed")
        selected_lowers = arrays["radius_lowers"][expected_indices]
        selected_uppers = arrays["radius_uppers"][expected_indices]
    if len(prepare.get("rows", [])) != 20:
        issues.append("adaptive row count is not 20")
    fixed_rows = prepare.get("fixed_radius_rows", [])
    if len(fixed_rows) != 100:
        issues.append("fixed-radius row count is not 100")
    branches = prepare.get("branches", [])
    if len(branches) != 480:
        issues.append("prepared branch-attempt count is not 480")
    branch_artifact = _inside(
        Path(prepare["branches_artifact"]["path"]), WRITE_ROOT
    )
    if _sha256(branch_artifact) != prepare["branches_artifact"]["sha256"]:
        issues.append("branch JSONL hash changed")
    for branch in branches:
        if branch["feasibility"] != "infeasible":
            path = _inside(Path(branch["hull_artifact"]), WRITE_ROOT)
            if _sha256(path) != branch["hull_artifact_sha256"]:
                issues.append(f"hull hash changed for {branch['row_id']} expert {branch['expert']}")
    by_row: dict[str, list[dict[str, Any]]] = {}
    for branch in branches:
        by_row.setdefault(str(branch["row_id"]), []).append(branch)
    for row in fixed_rows:
        epsilon = float(row["epsilon"])
        slot = int(row["sample_slot"])
        expected_status = route_status_at_epsilon(
            float(selected_lowers[slot]), float(selected_uppers[slot]), epsilon
        )
        if row["route_status"] != expected_status:
            issues.append(f"route status changed for {row['row_id']}")
        candidates = sorted(
            int(branch["expert"])
            for branch in by_row.get(str(row["row_id"]), [])
            if branch["feasibility"] != "infeasible"
        )
        if candidates != row["candidate_experts"]:
            issues.append(f"candidate set changed for {row['row_id']}")
    rows_artifact = _inside(Path(crown["rows_artifact"]["path"]), WRITE_ROOT)
    if _sha256(rows_artifact) != crown["rows_artifact"]["sha256"]:
        issues.append("CROWN rows artifact hash changed")
    with rows_artifact.open(encoding="utf-8") as handle:
        crown_branches = [json.loads(line) for line in handle if line.strip()]
    expected_feasible = sum(
        branch["feasibility"] != "infeasible" for branch in branches
    )
    if len(crown_branches) != expected_feasible or crown["branches"] != expected_feasible:
        issues.append("CROWN feasible-branch count changed")
    backend_errors, incomplete_bounds = crown_backend_failure_counts(crown_branches)
    if backend_errors:
        issues.append(f"CROWN backend errors are nonzero: {backend_errors}")
    if incomplete_bounds:
        issues.append(f"CROWN incomplete branch bounds are nonzero: {incomplete_bounds}")
    if int(crown.get("backend_error_count", -1)) != backend_errors:
        issues.append("CROWN backend-error summary failed recomputation")
    if int(crown.get("incomplete_bound_count", -1)) != incomplete_bounds:
        issues.append("CROWN incomplete-bound summary failed recomputation")
    expected_gradient_tracking = bool(
        config.get("crown", {}).get("gradient_tracking", True)
    )
    if bool(crown.get("gradient_tracking_enabled")) != expected_gradient_tracking:
        issues.append("CROWN gradient-tracking summary changed")
    if any(
        bool(branch.get("crown", {}).get("gradient_tracking_enabled"))
        != expected_gradient_tracking
        for branch in crown_branches
    ):
        issues.append("CROWN branch gradient-tracking policy changed")
    if int(crown.get("formal_safe_count", -1)) != 0:
        issues.append("non-outward CROWN result was promoted to formal SAFE")
    prohibited = {"SAFE", "UNSAFE"}
    for branch in crown_branches:
        if branch.get("formal_status") in prohibited:
            issues.append("branch contains prohibited formal verdict")
    crown_fixed = crown.get("fixed_radius_samples", [])
    if len(crown_fixed) != 100:
        issues.append("CROWN fixed-radius result count is not 100")
    recomputed_table: dict[str, Any] = {}
    for numerator in config["primary_table_epsilon_over_255"]:
        rows = [
            row
            for row in crown_fixed
            if float(row["epsilon_over_255"]) == float(numerator)
        ]
        recomputed_table[str(numerator)] = {
            "samples": len(rows),
            "route_status_counts": dict(Counter(row["route_status"] for row in rows)),
            "route_invariance_status_counts": dict(
                Counter(row["route_invariance_crown"] for row in rows)
            ),
            "route_a_status_counts": dict(
                Counter(row["route_a_crown"] for row in rows)
            ),
            "formal_safe_count": 0,
        }
    if crown.get("fixed_radius_table") != recomputed_table:
        issues.append("fixed-radius table failed recomputation")
    result = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "scope": "INDEPENDENT_B3_IDENTITY_ROUTE_AND_TABLE_AUDIT",
        "prepare": {"path": str(prepare_path), "sha256": _sha256(prepare_path)},
        "crown": {"path": str(crown_path), "sha256": _sha256(crown_path)},
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "adaptive_rows": len(prepare.get("rows", [])),
        "fixed_radius_rows": len(fixed_rows),
        "branch_attempts": len(branches),
        "feasible_branches": expected_feasible,
        "backend_errors": backend_errors,
        "incomplete_bounds": incomplete_bounds,
        "fixed_radius_table": recomputed_table,
        "formal_safe_count": int(crown.get("formal_safe_count", -1)),
        "claim_scope": (
            "Exact route analysis plus numerical CROWN conformance filters only; "
            "no non-outward CROWN margin is promoted to formal SAFE or UNSAFE."
        ),
    }
    _write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path, required=True)
    parser.add_argument("--crown", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.prepare, args.crown, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
