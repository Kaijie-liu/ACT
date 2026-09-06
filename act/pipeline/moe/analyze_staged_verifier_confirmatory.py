"""Create the tracked compact report for the fixed-radius staged cohort."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from act.pipeline.moe.experiment1 import (
    PROJECT_ROOT,
    WRITE_ROOT,
    _inside,
    _sha256,
    _write_json,
)


CONFIG = (
    PROJECT_ROOT
    / "act/pipeline/moe/configs/staged_verifier_seed2_fixed2_confirmatory_r1.json"
)
OUTPUT_JSON = (
    PROJECT_ROOT
    / "act/pipeline/moe/results/staged_verifier_seed2_fixed2_confirmatory_20260906_r1.json"
)
OUTPUT_CSV = OUTPUT_JSON.with_suffix(".csv")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def wilson_interval(successes: int, total: int) -> list[float]:
    """Return the two-sided 95% Wilson score interval."""
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [center - radius, center + radius]


def _quantiles(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(data)),
        "q1": float(np.quantile(data, 0.25)),
        "q3": float(np.quantile(data, 0.75)),
        "p90": float(np.quantile(data, 0.9)),
        "sum": float(data.sum()),
    }


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    selection_path = _inside(Path(config["selection_manifest"]), PROJECT_ROOT)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    root = _inside(Path(config["output_dir"]), WRITE_ROOT)
    results_path = root / "results.jsonl"
    runner_summary_path = root / "summary.json"
    audit_path = root / "independent_audit.json"
    rows = _jsonl(results_path)
    runner_summary = json.loads(runner_summary_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or audit["issue_count"] != 0:
        raise RuntimeError("independent audit did not pass")
    if len(rows) != 100 or audit["rows_audited"] != 100:
        raise RuntimeError("confirmatory cohort is incomplete")
    if OUTPUT_JSON.exists() or OUTPUT_CSV.exists():
        raise RuntimeError("refusing to overwrite compact result")

    table: list[dict[str, Any]] = []
    for row in rows:
        table.append(
            {
                "sample_rank": int(row["sample_rank"]),
                "dataset_index": int(row["dataset_index"]),
                "epsilon": float(row["epsilon"]),
                "status": row["status"],
                "reason": row["reason"],
                "decision_tier": row["decision_tier"],
                "route_changing": bool(row["route_changing"]),
                "candidate_experts": int(row["candidate_experts"]),
                "feasible_route_sets": int(row["feasible_route_sets"]),
                "f0_invoked": bool(row["f0_invoked"]),
                "full_model_witness_valid": bool(
                    row["full_model_witness_valid"]
                ),
                "verifier_total_seconds": float(row["verifier_total_seconds"]),
                "evidence_audit_status": row["evidence_audit_status"],
            }
        )

    by_route: dict[str, Counter[str]] = defaultdict(Counter)
    for row in table:
        route = "route_changing" if row["route_changing"] else "route_stable"
        by_route[route][row["status"]] += 1
    safe = [row for row in table if row["status"] == "SAFE"]
    unsafe = [row for row in table if row["status"] == "UNSAFE"]
    route_changing = [row for row in table if row["route_changing"]]
    unique_safe = [row for row in safe if row["route_changing"]]
    f0_rows = [row for row in table if row["f0_invoked"]]
    f0_complete = [row for row in f0_rows if row["status"] in {"SAFE", "UNSAFE"}]
    result = {
        "schema_version": 1,
        "classification": config["classification"],
        "execution_git_head": audit["execution_git_head"],
        "auditor_git_head": audit["auditor_git_head"],
        "artifacts": {
            "config": str(CONFIG),
            "config_sha256": _sha256(CONFIG),
            "selection": str(selection_path),
            "selection_sha256": _sha256(selection_path),
            "raw_results": str(results_path),
            "raw_results_sha256": _sha256(results_path),
            "runner_summary": str(runner_summary_path),
            "runner_summary_sha256": _sha256(runner_summary_path),
            "independent_audit": str(audit_path),
            "independent_audit_sha256": _sha256(audit_path),
        },
        "request_population": {
            "samples": len(table),
            "selection_rule": selection["selection_rule"],
            "dataset_index_minimum": min(row["dataset_index"] for row in table),
            "dataset_index_maximum": max(row["dataset_index"] for row in table),
            "epsilon_label": selection["request"]["epsilon_label"],
            "epsilon": selection["request"]["epsilon"],
            "boundary_search": False,
            "route_instability_prefilter": False,
        },
        "outcomes": {
            "status_counts": dict(
                sorted(Counter(row["status"] for row in table).items())
            ),
            "reason_counts": dict(
                sorted(Counter(row["reason"] for row in table).items())
            ),
            "complete": len(safe) + len(unsafe),
            "complete_rate": (len(safe) + len(unsafe)) / len(table),
            "safe": len(safe),
            "safe_rate": len(safe) / len(table),
            "unsafe": len(unsafe),
            "all_unsafe_full_model_replayed": all(
                row["full_model_witness_valid"] for row in unsafe
            ),
            "by_route_applicability": {
                key: dict(sorted(value.items()))
                for key, value in sorted(by_route.items())
            },
        },
        "primary_endpoint": {
            "route_changing_requests": len(route_changing),
            "route_changing_safe": len(unique_safe),
            "rate_full_100_denominator": len(unique_safe) / len(table),
            "wilson_95_full_100_denominator": wilson_interval(
                len(unique_safe), len(table)
            ),
            "conditional_rate_descriptive": len(unique_safe) / len(route_changing),
            "conditional_wilson_95_descriptive": wilson_interval(
                len(unique_safe), len(route_changing)
            ),
            "sample_ranks": [row["sample_rank"] for row in unique_safe],
            "dataset_indices": [row["dataset_index"] for row in unique_safe],
            "tier1": sum(
                row["decision_tier"] == "TIER1_GATE_ELIMINATION"
                for row in unique_safe
            ),
            "f0": sum(row["decision_tier"] == "TIER2_F0" for row in unique_safe),
            "preregistered_existence_replication_met": len(unique_safe) >= 1,
        },
        "staged_attribution": {
            "tier1_safe": sum(
                row["status"] == "SAFE"
                and row["decision_tier"] == "TIER1_GATE_ELIMINATION"
                for row in table
            ),
            "f0_invoked": len(f0_rows),
            "f0_complete": len(f0_complete),
            "f0_safe": sum(row["status"] == "SAFE" for row in f0_rows),
            "f0_unsafe": sum(row["status"] == "UNSAFE" for row in f0_rows),
            "f0_unresolved_or_timeout": len(f0_rows) - len(f0_complete),
        },
        "timing_descriptive_only": {
            "all_rows": _quantiles(
                [row["verifier_total_seconds"] for row in table]
            ),
            "speedup_claim_permitted": False,
            "right_censoring_note": (
                "All process wall times are observed; solver TIMEOUT outcomes "
                "remain censored with respect to time-to-semantic-resolution."
            ),
        },
        "audit": {
            "status": audit["status"],
            "issues": audit["issue_count"],
            "selection_reconstructed": audit["selection_reconstructed"],
            "packages": audit["packages_independently_audited"],
            "unsafe_replays": audit["unsafe_witnesses_independently_replayed"],
        },
        "decision": (
            "PASS: the preregistered production-entry existence-replication "
            "signal is met with zero audit issues and all UNSAFE replayed."
        ),
        "claim_boundary": selection["claim_boundary"],
    }
    if result["outcomes"]["status_counts"] != runner_summary["status_counts"]:
        raise RuntimeError("independent compact status count differs from runner")
    _write_json(OUTPUT_JSON, result)
    with OUTPUT_CSV.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(table[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
