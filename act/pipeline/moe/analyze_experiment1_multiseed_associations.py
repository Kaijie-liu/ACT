"""Derive non-causal module associations from the frozen multi-seed R1 rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Sequence

from act.pipeline.moe.published_moe_router_gradient_audit import _sha256


WORKSPACE = Path("/data1/Kane/MOE")
PROJECT_ROOT = WORKSPACE / "ACT"
RESULT_ROOT = PROJECT_ROOT / "data/moe/results"
TRACKED_ROOT = PROJECT_ROOT / "act/pipeline/moe/results"
DEFAULT_JSON = TRACKED_ROOT / "experiment1_multiseed_associations_20260906_r1.json"
DEFAULT_CSV = TRACKED_ROOT / "experiment1_multiseed_associations_20260906_r1.csv"
DEFAULT_RUNS = (
    RESULT_ROOT / "experiment1_multiseed_replication_seed1_r1",
    RESULT_ROOT / "experiment1_multiseed_replication_seed2_r1",
)


def _inside(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(WORKSPACE)
    return resolved


def _json_lines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in _inside(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


def _tier(row: dict[str, Any]) -> str:
    if row.get("reason") == "SAFE_GATE_ELIMINATION":
        return "TIER1_GATE_ELIMINATION"
    if row.get("f0_invoked"):
        return "TIER2_F0"
    if row.get("reason") == "NO_ROUTE_BOUNDARY_WITHIN_SEARCH":
        return "BOUNDARY_NOT_APPLICABLE_WITHIN_CAP"
    return "TIER1_OR_PRE_F0"


def _guard_eliminated(row: dict[str, Any]) -> int | None:
    gate = row.get("gate")
    if not gate:
        return None
    return sum(
        int(branch["guard_accounting"]["binary_eliminated"])
        for branch in gate.get("branches", [])
    )


def _f0_timing(row: dict[str, Any]) -> dict[str, Any]:
    if not row.get("f0_invoked"):
        return {"kind": "NOT_INVOKED", "seconds": None, "lower_bound_seconds": None}
    if row.get("f0_seconds") is not None:
        return {
            "kind": "OBSERVED",
            "seconds": float(row["f0_seconds"]),
            "lower_bound_seconds": None,
        }
    return {
        "kind": "RIGHT_CENSORED_AT_INSTANCE_DEADLINE",
        "seconds": None,
        "lower_bound_seconds": 0.0,
    }


def _model_rows(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = _inside(run_dir)
    census_path = run_dir / "census/results.jsonl"
    boundary_path = run_dir / "boundary/results.jsonl"
    audit_path = run_dir / "independent_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("issue_count") != 0 or audit.get("issues") != []:
        raise RuntimeError(f"R1 independent audit did not pass: {audit_path}")
    artifact_hashes = audit["artifact_hashes_before_audit"]
    if artifact_hashes.get("census/results.jsonl") != _sha256(census_path):
        raise RuntimeError(f"census row identity mismatch: {run_dir}")
    if artifact_hashes.get("boundary/results.jsonl") != _sha256(boundary_path):
        raise RuntimeError(f"boundary row identity mismatch: {run_dir}")
    census = _json_lines(census_path)
    boundary = _json_lines(boundary_path)
    if len(census) != 160 or len(boundary) != 40:
        raise RuntimeError(f"R1 row count differs: {run_dir}")
    by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in census:
        by_rank[int(row["sample_rank"])].append(row)
    if any(len(rows) != 4 for rows in by_rank.values()) or len(by_rank) != 40:
        raise RuntimeError(f"fixed-radius cluster structure differs: {run_dir}")

    model_id = run_dir.name.replace("experiment1_multiseed_replication_", "").replace(
        "_r1", ""
    )
    derived: list[dict[str, Any]] = []
    for boundary_row in boundary:
        rank = int(boundary_row["sample_rank"])
        fixed = sorted(by_rank[rank], key=lambda row: float(row["epsilon"]))
        unstable = [row for row in fixed if row["route_set_unstable"]]
        zono_reduced = [
            row
            for row in unstable
            if int(row["exact_candidate_count"]) < int(row["zonotope_candidate_count"])
        ]
        ibp_reduced = [
            row
            for row in unstable
            if int(row["exact_candidate_count"]) < int(row["ibp_candidate_count"])
        ]
        timing = _f0_timing(boundary_row)
        derived.append(
            {
                "model_id": model_id,
                "sample_rank": rank,
                "dataset_index": int(boundary_row["dataset_index"]),
                "boundary_status": boundary_row["status"],
                "boundary_reason": boundary_row["reason"],
                "route_upper": boundary_row.get("route_upper"),
                "verification_epsilon": boundary_row.get("epsilon"),
                "exact_feasible_pair_count": boundary_row.get(
                    "exact_feasible_pair_count"
                ),
                "fixed_route_unstable_rows": len(unstable),
                "fixed_zonotope_reduction_rows": len(zono_reduced),
                "fixed_zonotope_reduction_labels": [
                    row["epsilon_label"] for row in zono_reduced
                ],
                "fixed_ibp_reduction_rows": len(ibp_reduced),
                "fixed_ibp_reduction_labels": [row["epsilon_label"] for row in ibp_reduced],
                "boundary_guard_binaries_eliminated": _guard_eliminated(boundary_row),
                "f0_invoked": bool(boundary_row.get("f0_invoked")),
                "decision_tier": _tier(boundary_row),
                "unique_safe": bool(boundary_row.get("unique_safe")),
                "full_model_witness_valid": bool(
                    boundary_row.get("full_model_witness_valid")
                ),
                "candidate_seconds": boundary_row.get("candidate_seconds"),
                "gate_seconds": boundary_row.get("gate_seconds"),
                "f0_time_kind": timing["kind"],
                "f0_seconds": timing["seconds"],
                "f0_seconds_lower_bound": timing["lower_bound_seconds"],
                "total_seconds": float(boundary_row["total_seconds"]),
                "fixed_radius_rows": [
                    {
                        "epsilon_label": row["epsilon_label"],
                        "route_set_unstable": bool(row["route_set_unstable"]),
                        "ibp_candidates": int(row["ibp_candidate_count"]),
                        "zonotope_candidates": int(row["zonotope_candidate_count"]),
                        "exact_candidates": int(row["exact_candidate_count"]),
                        "feasible_route_sets": int(row["exact_feasible_pair_count"]),
                        "guard_binaries_eliminated": sum(
                            int(branch["guard_accounting"]["binary_eliminated"])
                            for branch in row["branches"]
                        ),
                        "total_seconds": float(row["total_seconds"]),
                    }
                    for row in fixed
                ],
            }
        )

    safe = [row for row in derived if row["boundary_status"] == "SAFE"]
    summary = {
        "model_id": model_id,
        "rows": len(derived),
        "status_counts": dict(Counter(row["boundary_status"] for row in derived)),
        "safe_by_tier": dict(Counter(row["decision_tier"] for row in safe)),
        "safe_without_observed_zonotope_reduction_at_any_fixed_radius": sum(
            row["fixed_zonotope_reduction_rows"] == 0 for row in safe
        ),
        "safe_with_observed_zonotope_reduction_at_a_fixed_radius": sum(
            row["fixed_zonotope_reduction_rows"] > 0 for row in safe
        ),
        "f0_invoked": sum(row["f0_invoked"] for row in derived),
        "f0_observed_times": sum(row["f0_time_kind"] == "OBSERVED" for row in derived),
        "f0_right_censored_times": sum(
            row["f0_time_kind"] == "RIGHT_CENSORED_AT_INSTANCE_DEADLINE"
            for row in derived
        ),
    }
    return derived, summary


def analyze(run_dirs: Sequence[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    raw: list[dict[str, str]] = []
    for run_dir in run_dirs:
        model_rows, summary = _model_rows(run_dir)
        rows.extend(model_rows)
        summaries.append(summary)
        raw.append(
            {
                "run_dir": str(_inside(run_dir)),
                "census_rows_sha256": _sha256(_inside(run_dir) / "census/results.jsonl"),
                "boundary_rows_sha256": _sha256(
                    _inside(run_dir) / "boundary/results.jsonl"
                ),
                "independent_audit_sha256": _sha256(
                    _inside(run_dir) / "independent_audit.json"
                ),
            }
        )
    return {
        "experiment": "experiment1_multiseed_associations_r1",
        "status": "DERIVED_FROM_FROZEN_ROWS_NO_NEW_SOLVER_QUERY",
        "model_summaries": summaries,
        "rows": rows,
        "raw_artifacts": raw,
        "cost_accounting_correction": {
            "historical_summary_immutable": True,
            "f0_missing_seconds_are_zero": False,
            "right_censored_f0_rows": sum(
                row["f0_time_kind"] == "RIGHT_CENSORED_AT_INSTANCE_DEADLINE"
                for row in rows
            ),
            "interpretation": (
                "A hard-deadline row that entered F0 but did not persist f0_seconds "
                "has unknown right-censored F0 cost; it is never imputed as zero."
            ),
        },
        "claim_boundary": (
            "This is a within-model descriptive association table over previously "
            "executed rows. Candidate reduction is observed only at the four fixed "
            "census radii, while the end-to-end verdict uses a model-specific boundary "
            "radius. The table is not a causal ablation and issues no new certificate."
        ),
    }


def _csv_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "model_id",
        "sample_rank",
        "dataset_index",
        "boundary_status",
        "boundary_reason",
        "route_upper",
        "verification_epsilon",
        "exact_feasible_pair_count",
        "fixed_route_unstable_rows",
        "fixed_zonotope_reduction_rows",
        "fixed_ibp_reduction_rows",
        "boundary_guard_binaries_eliminated",
        "f0_invoked",
        "decision_tier",
        "unique_safe",
        "full_model_witness_valid",
        "candidate_seconds",
        "gate_seconds",
        "f0_time_kind",
        "f0_seconds",
        "f0_seconds_lower_bound",
        "total_seconds",
    )
    return [{field: row.get(field) for field in fields} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    run_dirs = tuple(args.run_dir) if args.run_dir else DEFAULT_RUNS
    result = analyze(run_dirs)
    json_path = _inside(args.json)
    csv_path = _inside(args.csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    csv_rows = _csv_rows(result["rows"])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({key: result[key] for key in ("status", "model_summaries", "cost_accounting_correction")}, indent=2))


if __name__ == "__main__":
    main()
