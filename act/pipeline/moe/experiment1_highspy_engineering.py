# ===- experiment1_highspy_engineering.py - Incremental rerun ---------====#
"""Engineering-only incremental HiGHS rerun on all 20 frozen D0 rows.

The runner delegates frozen selection/radius/replay semantics to Experiment 1D,
then writes a separate paired telemetry summary.  It never changes the original
56/100 confirmatory endpoint.  F0 sessions are reusable only within one fixed
augmented property HZ (low budget to escalation); different augmented HZs are
intentionally rebuilt and never counted as cross-property model reuse.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256, _write_json
from act.pipeline.moe.experiment1d import run as run_experiment1d


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "act/pipeline/moe/configs/experiment1_highspy_engineering_r2.json"
)

_COUNT_KEYS = (
    "model_builds",
    "model_build_failures",
    "objective_update_calls",
    "objective_coefficients_changed",
    "row_pool_additions",
    "row_update_calls",
    "row_coefficients_changed",
    "row_bound_updates",
    "integrality_update_calls",
    "budget_extension_calls",
    "solves",
    "cold_start_solves",
    "basis_submission_attempts",
    "basis_submissions_accepted",
    "basis_valid_after_solve",
    "simplex_iterations",
    "ipm_iterations",
    "mip_nodes",
)
_TIME_KEYS = (
    "model_build_seconds",
    "objective_update_seconds",
    "row_update_seconds",
    "budget_extension_seconds",
    "solve_seconds",
    "total_seconds",
)


def _support_sessions(record: dict[str, Any] | None):
    for layer in (record or {}).get("layers", []):
        for kind, key in (
            ("guarded_support_lp", "lp_incremental_telemetry"),
            ("guarded_support_milp", "milp_incremental_telemetry"),
        ):
            telemetry = layer.get(key)
            if telemetry is not None:
                yield kind, telemetry


def _terminal_sessions(row: dict[str, Any]):
    gate = row.get("gate") or {}
    for branch in gate.get("branches", []):
        yield from _support_sessions(branch.get("support"))
        attempts = branch.get("attempts", [])
        if attempts:
            telemetry = attempts[-1].get("metadata", {}).get("incremental_hz")
            if telemetry is not None:
                yield "expert_property", telemetry

    f0 = row.get("f0") or {}
    for pair in f0.get("pairs", []):
        yield from _support_sessions(pair.get("expert_a_support"))
        yield from _support_sessions(pair.get("expert_b_support"))
        for prop in pair.get("property_rows", []):
            attempts = prop.get("attempts", [])
            if attempts and attempts[-1].get("incremental_hz") is not None:
                yield "f0_augmented_property", attempts[-1]["incremental_hz"]


def summarize_incremental_telemetry(
    rows: Iterable[dict[str, Any]],
    baseline_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(rows)
    baseline = {int(row["sample_rank"]): row for row in baseline_rows}
    totals = {key: 0 for key in _COUNT_KEYS}
    totals.update({key: 0.0 for key in _TIME_KEYS})
    status_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    sessions = []
    support_identity_counts: Counter[str] = Counter()
    for row in rows:
        gate = row.get("gate") or {}
        for branch in gate.get("branches", []):
            identity = branch.get("support_identity")
            if identity is not None:
                support_identity_counts[
                    "identical" if identity.get("structural_identity") else "drift"
                ] += 1
        f0 = row.get("f0") or {}
        for pair in f0.get("pairs", []):
            for key in ("expert_a_support_identity", "expert_b_support_identity"):
                identity = pair.get(key)
                if identity is not None:
                    support_identity_counts[
                        "identical"
                        if identity.get("structural_identity")
                        else "drift"
                    ] += 1
        for category, telemetry in _terminal_sessions(row):
            category_counts[category] += 1
            sessions.append((category, telemetry))
            for key in _COUNT_KEYS:
                totals[key] += int(telemetry.get(key, 0))
            for key in _TIME_KEYS:
                totals[key] += float(telemetry.get(key, 0.0))
            status_counts.update(telemetry.get("status_counts", {}))

    transitions = Counter()
    paired_seconds = []
    for row in rows:
        old = baseline.get(int(row["sample_rank"]))
        if old is None:
            transitions["MISSING_BASELINE"] += 1
            continue
        transitions[f"{old['status']}->{row['status']}"] += 1
        paired_seconds.append(
            float(row.get("total_seconds", 0.0))
            - float(old.get("total_seconds", 0.0))
        )
    return {
        "scope": "engineering_performance_rerun_not_confirmatory_overwrite",
        "rows": len(rows),
        "parent_confirmatory_endpoint_immutable": "56/100",
        "parent_applicable_coverage_immutable": "56/76",
        "incremental_sessions": len(sessions),
        "session_category_counts": dict(category_counts),
        "telemetry_totals": {
            **totals,
            "status_counts": dict(status_counts),
        },
        "sessions_with_build_error": sum(
            telemetry.get("build_error") is not None
            for _, telemetry in sessions
        ),
        "support_signature_comparisons": dict(support_identity_counts),
        "support_signature_drift_semantics": (
            "expected engineering effect of changing the support solver; "
            "candidate, route-set, radius, and parent artifact identities remain frozen"
        ),
        "paired_status_transitions": dict(transitions),
        "paired_total_seconds_difference": paired_seconds,
        "f0_reuse_scope": "same_augmented_property_low_to_escalation_only",
        "f0_cross_augmented_hz_reuse": False,
        "f0_cross_augmented_hz_reason": (
            "each property has a distinct McCormick-augmented SparseHZono"
        ),
        "scientific_interpretation": (
            "paired engineering performance only; does not replace or backfill "
            "the frozen confirmatory result"
        ),
    }


def _validate_engineering_config(config: dict[str, Any]) -> None:
    if config.get("scope") != "engineering_performance_rerun_not_confirmatory_overwrite":
        raise ValueError("incremental rerun must be explicitly engineering-only")
    if int(config.get("expected_rows", -1)) != 20:
        raise ValueError("incremental rerun must retain all 20 frozen rows")
    if float(config.get("instance_timeout_seconds", -1)) != 900.0:
        raise ValueError("incremental rerun must retain the 900-second deadline")
    if config.get("support", {}).get("solver_backend") != "highspy_incremental":
        raise ValueError("guarded support must explicitly opt into highspy_incremental")
    solver = config.get("solver", {})
    if solver.get("backend") != "highspy_incremental":
        raise ValueError("expert properties must explicitly opt into highspy_incremental")
    if solver.get("f0_backend") != "highspy_incremental":
        raise ValueError("F0 must explicitly opt into highspy_incremental")
    if config.get("engineering_allow_support_solver_drift") is not True:
        raise ValueError(
            "engineering rerun must explicitly record support-solver signature drift"
        )


def run(config_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_engineering_config(config)
    baseline_path = _inside(Path(config["baseline_results_jsonl"]), WRITE_ROOT)
    if _sha256(baseline_path) != config["baseline_results_sha256"]:
        raise RuntimeError("paired Experiment 1D baseline artifact changed")
    result = run_experiment1d(config_path)
    output_dir = _inside(Path(result["output_dir"]), WRITE_ROOT)
    rows = [
        json.loads(line)
        for line in (output_dir / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    baseline_rows = [
        json.loads(line)
        for line in baseline_path.read_text(encoding="utf-8").splitlines()
    ]
    summary = summarize_incremental_telemetry(rows, baseline_rows)
    _write_json(output_dir / "incremental_telemetry_summary.json", summary)
    return {**result, "incremental_telemetry_summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    arguments = parser.parse_args()
    print(json.dumps(run(Path(arguments.config)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
