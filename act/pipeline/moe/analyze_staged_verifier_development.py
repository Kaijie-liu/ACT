"""Create the tracked compact report for the staged-verifier development run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256, _write_json


CONFIG = (
    PROJECT_ROOT
    / "act/pipeline/moe/configs/staged_verifier_seed2_unresolved_dev_r1.json"
)
OUTPUT_JSON = (
    PROJECT_ROOT
    / "act/pipeline/moe/results/staged_verifier_seed2_unresolved_20260906_r1.json"
)
OUTPUT_CSV = OUTPUT_JSON.with_suffix(".csv")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
    root = _inside(Path(config["output_dir"]), WRITE_ROOT)
    results_path = root / "results.jsonl"
    runner_summary_path = root / "summary.json"
    audit_path = root / "independent_audit.json"
    rows = _jsonl(results_path)
    source_rows = {
        int(row["sample_rank"]): row
        for row in _jsonl(_inside(Path(config["source_results"]), WRITE_ROOT))
    }
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or audit["issue_count"] != 0:
        raise RuntimeError("independent audit did not pass")
    if OUTPUT_JSON.exists() or OUTPUT_CSV.exists():
        raise RuntimeError("refusing to overwrite compact result")

    table: list[dict[str, Any]] = []
    for row in rows:
        rank = int(row["sample_rank"])
        source = source_rows[rank]
        evidence = json.loads(
            (Path(row["package"]) / "evidence.json").read_text(encoding="utf-8")
        )
        table.append(
            {
                "sample_rank": rank,
                "dataset_index": int(row["dataset_index"]),
                "epsilon": float(row["epsilon"]),
                "source_status": row["source_status"],
                "source_reason": row["source_reason"],
                "source_total_seconds": float(source["total_seconds"]),
                "production_status": row["production_status"],
                "production_reason": row["production_reason"],
                "decision_tier": evidence["verdict"]["decision_tier"],
                "production_wall_seconds": float(row["production_wall_seconds"]),
                "tier1_seconds": float(evidence["timing"]["tier1_seconds"]),
                "tier2_seconds": float(evidence["timing"]["tier2_seconds"]),
                "new_complete_outcome": row["production_status"]
                in {"SAFE", "UNSAFE"},
                "full_model_witness_valid": bool(
                    evidence["verdict"]["full_model_witness_valid"]
                ),
                "candidate_experts": len(
                    evidence["route_coverage"]["candidate_experts"]
                ),
                "feasible_route_sets": len(
                    evidence["route_coverage"]["feasible_route_sets"]
                ),
                "evidence_audit_status": row["evidence_audit_status"],
            }
        )

    new_rows = [row for row in table if row["new_complete_outcome"]]
    production_times = [row["production_wall_seconds"] for row in table]
    source_times = [row["source_total_seconds"] for row in table]
    deltas = [new - old for new, old in zip(production_times, source_times)]
    result = {
        "schema_version": 1,
        "classification": config["classification"],
        "execution_git_head": audit["runtime_git_head"],
        "auditor_git_head": audit["auditor_git_head"],
        "artifacts": {
            "config": str(CONFIG),
            "config_sha256": _sha256(CONFIG),
            "raw_results": str(results_path),
            "raw_results_sha256": _sha256(results_path),
            "initial_runner_summary": str(runner_summary_path),
            "initial_runner_summary_sha256": _sha256(runner_summary_path),
            "independent_audit": str(audit_path),
            "independent_audit_sha256": _sha256(audit_path),
        },
        "selection": {
            "rows": len(table),
            "source_reason_counts": {
                reason: sum(row["source_reason"] == reason for row in table)
                for reason in sorted({row["source_reason"] for row in table})
            },
            "outcome_selected": True,
        },
        "outcome": {
            "status_counts": {
                status: sum(row["production_status"] == status for row in table)
                for status in sorted({row["production_status"] for row in table})
            },
            "new_complete_outcomes": len(new_rows),
            "new_safe": sum(row["production_status"] == "SAFE" for row in new_rows),
            "new_unsafe": sum(
                row["production_status"] == "UNSAFE" for row in new_rows
            ),
            "new_complete_ranks": [row["sample_rank"] for row in new_rows],
            "all_new_unsafe_full_model_replayed": all(
                row["full_model_witness_valid"] for row in new_rows
            ),
            "primary_signal_met": len(new_rows) >= 1,
        },
        "timeout_accounting": {
            "outer_hard_timeouts": audit["outer_hard_timeouts"],
            "solver_reported_timeouts": audit["solver_reported_timeouts"],
            "initial_summary_correction": audit["accounting_corrections"],
        },
        "timing_descriptive_only": {
            "production": _quantiles(production_times),
            "historical_source": _quantiles(source_times),
            "paired_delta_production_minus_source": _quantiles(deltas),
            "production_lower_on_rows": sum(delta < 0 for delta in deltas),
            "warning": (
                "Runs were not interleaved and historical time includes "
                "experiment-only work; these values do not establish speedup."
            ),
        },
        "audit": {
            "status": audit["status"],
            "issues": audit["issue_count"],
            "packages": audit["packages_independently_audited"],
            "unsafe_replays": audit["unsafe_witnesses_independently_replayed"],
        },
        "decision": (
            "The preregistered development signal is met. A separate new HZ "
            "cohort may now be frozen; this run is not confirmatory evidence."
        ),
        "claim_boundary": (
            "Outcome-selected development engineering only. Do not revise R1, "
            "report prevalence, claim speedup, or access the locked Lagrangian "
            "holdout."
        ),
    }
    _write_json(OUTPUT_JSON, result)
    with OUTPUT_CSV.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(table[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
