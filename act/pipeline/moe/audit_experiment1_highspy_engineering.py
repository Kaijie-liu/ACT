# ===- audit_experiment1_highspy_engineering.py --------------------====#
"""Independent paired audit for the incremental-HiGHS engineering rerun.

The generic Experiment 1D audit intentionally understands the original SciPy
certificate fields only.  This audit checks the equivalent, backend-specific
HiGHS evidence without changing the frozen result or treating a time-limit
warning as a certificate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any, Iterable

import torch

from act.back_end.moe import load_output_moe_checkpoint
from act.back_end.solver.solver_hz import hz_numerical_policy_manifest
from act.pipeline.moe.experiment1 import (
    PROJECT_ROOT,
    WRITE_ROOT,
    _forward_validate,
    _inside,
    _sha256,
    _write_json,
)
from act.pipeline.moe.experiment1_highspy_engineering import (
    DEFAULT_CONFIG,
    _terminal_sessions,
    summarize_incremental_telemetry,
)
from act.pipeline.moe.experiment1d import _load_frozen_selection
from act.pipeline.moe.train import _load_dataset


SEMANTIC_SOURCE_PATHS = (
    "act/pipeline/moe/experiment1_highspy_engineering.py",
    "act/pipeline/moe/experiment1.py",
    "act/pipeline/moe/experiment1d.py",
    "act/pipeline/moe/experiment1f0.py",
    "act/back_end/moe/weighted_top2.py",
    "act/back_end/solver/solver_hz.py",
)
HIGHSPY_OPTIMAL_STATUS = 7


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _semantic_source_drift(launch_head: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", launch_head, "--", *SEMANTIC_SOURCE_PATHS],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _highspy_safe_property_issues(
    prop: dict[str, Any], *, tolerance: float, pair_reused: bool = False
) -> list[str]:
    issues: list[str] = []
    if pair_reused or prop.get("reused_parent"):
        if prop.get("solver_status") != 0:
            issues.append("reused SciPy SAFE lacks status-0 optimum")
        if prop.get("solver_bound_kind") not in {
            "mip_dual_bound",
            "lp_status0_optimum",
        }:
            issues.append("reused SciPy SAFE lacks a registered bound kind")
        if not math.isfinite(float(prop.get("minimum", math.nan))):
            issues.append("reused SciPy SAFE minimum is not finite")
        elif float(prop["minimum"]) <= tolerance:
            issues.append("reused SciPy SAFE minimum is not positive")
        return issues

    if prop.get("solver_status") != HIGHSPY_OPTIMAL_STATUS:
        issues.append("new HiGHS SAFE lacks kOptimal model status")
    if prop.get("solver_bound_kind") not in {
        "highs_mip_dual_bound",
        "highs_lp_optimum",
    }:
        issues.append("new HiGHS SAFE lacks a registered bound kind")
    minimum = float(prop.get("minimum", math.nan))
    if not math.isfinite(minimum) or minimum <= tolerance:
        issues.append("new HiGHS SAFE lacks a finite positive corrected bound")
    attempts = prop.get("attempts", [])
    if not attempts:
        issues.append("new HiGHS SAFE has no solver attempt evidence")
        return issues
    final = attempts[-1]
    if final.get("status") != "SAFE" or final.get("reason") != "SAFE_WEIGHTED_RANGE":
        issues.append("new HiGHS SAFE final attempt is not SAFE_WEIGHTED_RANGE")
    if final.get("solver_status") != HIGHSPY_OPTIMAL_STATUS:
        issues.append("new HiGHS SAFE final attempt lacks kOptimal")
    certified = float(final.get("certified_lower_bound", math.nan))
    dual = float(final.get("dual_bound", math.nan))
    if not math.isfinite(certified) or certified <= tolerance:
        issues.append("new HiGHS SAFE final certified bound is not positive")
    if not math.isfinite(dual):
        issues.append("new HiGHS SAFE final dual bound is not finite")
    if math.isfinite(minimum) and math.isfinite(certified) and not math.isclose(
        minimum, certified, rel_tol=0.0, abs_tol=1e-12
    ):
        issues.append("new HiGHS SAFE minimum differs from final certified bound")
    if float(final.get("gap", math.inf)) != 0.0:
        issues.append("new HiGHS SAFE final MIP gap is not zero")
    telemetry = final.get("incremental_hz") or {}
    if telemetry.get("backend") != "highspy_incremental_hz":
        issues.append("new HiGHS SAFE final attempt lacks backend identity")
    if telemetry.get("build_error") is not None:
        issues.append("new HiGHS SAFE session has a build error")
    if telemetry.get("warnings_fail_closed") is not True:
        issues.append("new HiGHS SAFE session does not fail closed on warnings")
    if int((telemetry.get("status_counts") or {}).get("optimal", 0)) < 1:
        issues.append("new HiGHS SAFE session records no optimal solve")
    return issues


def _safe_row_issues(row: dict[str, Any], *, tolerance: float) -> list[str]:
    issues: list[str] = []
    f0 = row.get("f0") or {}
    feasible = {
        tuple(sorted(int(value) for value in pair))
        for pair in f0.get("feasible_pairs", [])
    }
    pairs = f0.get("pairs", [])
    evaluated = {
        tuple(sorted(int(value) for value in pair.get("pair", []))) for pair in pairs
    }
    if not feasible or evaluated != feasible or len(evaluated) != len(pairs):
        issues.append("SAFE F0 does not cover each feasible pair exactly once")
    for pair in pairs:
        if pair.get("status") != "SAFE":
            issues.append(f"SAFE row contains non-SAFE pair {pair.get('pair')}")
        properties = pair.get("property_rows", [])
        indices = [int(prop.get("property_index", -1)) for prop in properties]
        if set(indices) != set(range(9)) or len(indices) != 9:
            issues.append(f"SAFE pair {pair.get('pair')} lacks nine unique properties")
        for prop in properties:
            if prop.get("status") != "SAFE":
                issues.append(
                    f"SAFE pair {pair.get('pair')} contains unresolved property "
                    f"{prop.get('property_index')}"
                )
                continue
            issues.extend(
                f"pair {pair.get('pair')} property {prop.get('property_index')}: {item}"
                for item in _highspy_safe_property_issues(
                    prop,
                    tolerance=tolerance,
                    pair_reused=bool(pair.get("reused_parent")),
                )
            )
    return issues


def _session_issues(rows: Iterable[dict[str, Any]]) -> tuple[list[str], dict[str, int]]:
    issues: list[str] = []
    sessions = list(
        (category, telemetry)
        for row in rows
        for category, telemetry in _terminal_sessions(row)
    )
    accepted = 0
    time_limits = 0
    for index, (category, telemetry) in enumerate(sessions):
        prefix = f"session {index} ({category})"
        if telemetry.get("backend") != "highspy_incremental_hz":
            issues.append(f"{prefix}: backend identity changed")
        if telemetry.get("build_error") is not None:
            issues.append(f"{prefix}: build_error is not null")
        if int(telemetry.get("model_build_failures", 0)) != 0:
            issues.append(f"{prefix}: model build failure recorded")
        if int(telemetry.get("model_builds", 0)) != 1:
            issues.append(f"{prefix}: session does not own exactly one model")
        if telemetry.get("warnings_fail_closed") is not True:
            issues.append(f"{prefix}: warnings are not fail-closed")
        accepted += int(telemetry.get("run_time_limit_warnings_accepted", 0))
        time_limits += int(
            (telemetry.get("status_counts") or {}).get("time_limit_reached", 0)
        )
    if accepted != time_limits:
        issues.append(
            "accepted run-time-limit warning count differs from kTimeLimit count"
        )
    return issues, {
        "sessions": len(sessions),
        "accepted_time_limit_warnings": accepted,
        "time_limit_statuses": time_limits,
    }


def _replay_unsafe(
    row: dict[str, Any], *, output_dir: Path, model, dataset
) -> list[str]:
    issues: list[str] = []
    rank = int(row["sample_rank"])
    if row.get("full_model_witness_valid") is not True:
        return [f"rank {rank}: UNSAFE lacks full_model_witness_valid"]
    relative = row.get("witness_path")
    path = _inside(output_dir / str(relative), output_dir) if relative else None
    if path is None or not path.exists():
        return [f"rank {rank}: witness path is missing"]
    if _sha256(path) != row.get("witness_sha256"):
        return [f"rank {rank}: witness hash mismatch"]
    saved = torch.load(path, map_location="cpu", weights_only=False)
    image, _ = dataset[int(row["dataset_index"])]
    clean = image.unsqueeze(0).double()
    epsilon = float(row["epsilon"])
    checked = _forward_validate(
        model,
        saved["input"],
        lower=(clean - epsilon).clamp(0, 1),
        upper=(clean + epsilon).clamp(0, 1),
        clean_prediction=int(row["clean_prediction"]),
    )
    if not checked["valid"]:
        issues.append(f"rank {rank}: concrete full-model witness replay failed")
    return issues


def _paired_summary(
    rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    baseline = {int(row["sample_rank"]): row for row in baseline_rows}
    transitions: Counter[str] = Counter()
    differences: list[float] = []
    for row in rows:
        old = baseline[int(row["sample_rank"])]
        transitions[f"{old['status']}->{row['status']}"] += 1
        differences.append(float(row["total_seconds"]) - float(old["total_seconds"]))
    old_solved = sum(row.get("status") in {"SAFE", "UNSAFE"} for row in baseline_rows)
    new_solved = sum(row.get("status") in {"SAFE", "UNSAFE"} for row in rows)
    old_seconds = sum(float(row["total_seconds"]) for row in baseline_rows)
    new_seconds = sum(float(row["total_seconds"]) for row in rows)
    return {
        "status_transitions": dict(transitions),
        "baseline_solved_rows": old_solved,
        "incremental_solved_rows": new_solved,
        "newly_solved_vs_d0": transitions["UNKNOWN->SAFE"]
        + transitions["UNKNOWN->UNSAFE"]
        + transitions["TIMEOUT->SAFE"]
        + transitions["TIMEOUT->UNSAFE"],
        "solved_regressions_vs_d0": sum(
            count
            for key, count in transitions.items()
            if key.startswith(("SAFE->", "UNSAFE->"))
            and not key.endswith(("->SAFE", "->UNSAFE"))
        ),
        "net_solved_gain_vs_d0": new_solved - old_solved,
        "baseline_total_seconds": old_seconds,
        "incremental_total_seconds": new_seconds,
        "incremental_over_baseline_total_runtime_ratio": new_seconds / old_seconds,
        "paired_seconds_difference_median": statistics.median(differences),
        "paired_seconds_difference": differences,
    }


def audit(config_path: Path, result_dir: Path | None = None) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = _inside(
        result_dir if result_dir is not None else Path(config["output_dir"]), WRITE_ROOT
    )
    output_path = output_dir / "incremental_independent_audit_r3.json"
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite {output_path}")
    runtime = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    rows = _load_jsonl(output_dir / "results.jsonl")
    baseline_path = _inside(Path(config["baseline_results_jsonl"]), WRITE_ROOT)
    baseline_rows = _load_jsonl(baseline_path)
    frozen = _load_frozen_selection(config)
    expected_ranks = [int(row["sample_rank"]) for row in frozen]
    issues: list[str] = []

    if runtime.get("source_config_sha256") != _sha256(config_path):
        issues.append("runtime source config hash mismatch")
    if runtime.get("config") != config:
        issues.append("embedded runtime config differs from frozen source config")
    if runtime.get("checkpoint_sha256") != _sha256(Path(config["checkpoint"])):
        issues.append("checkpoint hash mismatch")
    if _sha256(baseline_path) != config["baseline_results_sha256"]:
        issues.append("paired D0 baseline hash mismatch")
    if config.get("numerical_safety") != hz_numerical_policy_manifest():
        issues.append("numerical SAFE policy differs from registered backend policy")
    launch_head = str(runtime.get("git_head", ""))
    try:
        drift = _semantic_source_drift(launch_head)
    except (subprocess.CalledProcessError, ValueError):
        drift = ["UNABLE_TO_COMPARE_LAUNCH_HEAD"]
    if drift:
        issues.append(f"semantic source drift after launch: {drift}")

    ranks = [int(row["sample_rank"]) for row in rows]
    if len(rows) != int(config["expected_rows"]) or ranks != expected_ranks:
        issues.append("result rows differ from frozen ordered selection")
    if len(set(ranks)) != len(ranks):
        issues.append("duplicate sample rank")
    baseline = {int(row["sample_rank"]): row for row in baseline_rows}
    for row in rows:
        rank = int(row["sample_rank"])
        old = baseline.get(rank)
        if old is None:
            issues.append(f"rank {rank}: missing paired D0 row")
            continue
        if int(row["dataset_index"]) != int(old["dataset_index"]):
            issues.append(f"rank {rank}: dataset index changed")
        if not math.isclose(
            float(row["epsilon"]), float(old["epsilon"]), rel_tol=0.0, abs_tol=1e-15
        ):
            issues.append(f"rank {rank}: epsilon changed")
        if row.get("deadline_enforced") is not True:
            issues.append(f"rank {rank}: hard deadline flag missing")
        if row.get("checkpoint_recorded") is not True:
            issues.append(f"rank {rank}: 300-second checkpoint flag missing")
        checkpoint = output_dir / "row_work" / f"rank{rank}" / "checkpoint_300.json"
        if not checkpoint.exists():
            issues.append(f"rank {rank}: checkpoint artifact missing")
        if row.get("status") == "ERROR" or row.get("error"):
            issues.append(f"rank {rank}: explicit runner error")
        if "NUMERICAL" in str(row.get("reason", "")):
            issues.append(f"rank {rank}: numerical fallback/result")
        if row.get("status") != "TIMEOUT" and float(row["total_seconds"]) > float(
            config["instance_timeout_seconds"]
        ):
            issues.append(f"rank {rank}: completed after hard deadline")
        if row.get("status") == "SAFE":
            if row.get("reason") != "SAFE_WEIGHTED_RANGE":
                issues.append(f"rank {rank}: unregistered SAFE reason")
            else:
                issues.extend(
                    f"rank {rank}: {item}"
                    for item in _safe_row_issues(
                        row, tolerance=float(config["f0"]["safety_tolerance"])
                    )
                )

    model, payload = load_output_moe_checkpoint(
        Path(config["checkpoint"]), map_location="cpu"
    )
    model.double().eval()
    dataset = _load_dataset(payload["dataset"], False, download=False)
    unsafe_rows = [row for row in rows if row.get("status") == "UNSAFE"]
    for row in unsafe_rows:
        issues.extend(
            _replay_unsafe(row, output_dir=output_dir, model=model, dataset=dataset)
        )

    session_problems, session_counts = _session_issues(rows)
    issues.extend(session_problems)
    recomputed_telemetry = summarize_incremental_telemetry(rows, baseline_rows)
    stored_telemetry = json.loads(
        (output_dir / "incremental_telemetry_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if recomputed_telemetry != stored_telemetry:
        issues.append("independently recomputed incremental telemetry differs")
    paired = _paired_summary(rows, baseline_rows)
    conditions = {
        "all_20_rows_run": len(rows) == int(config["expected_rows"]),
        "zero_soundness_or_artifact_issues": len(issues) == 0,
        "all_unsafe_full_model_replayed": not any(
            "witness" in issue or "UNSAFE" in issue for issue in issues
        ),
        "no_solved_regression_vs_d0": paired["solved_regressions_vs_d0"] == 0,
        "no_silent_numerical_fallback": not any(
            row.get("status") == "ERROR"
            or "NUMERICAL" in str(row.get("reason", ""))
            for row in rows
        ),
    }
    result = {
        "schema_version": 1,
        "scope": "paired_engineering_audit_not_confirmatory_overwrite",
        "generic_experiment1d_audit": {
            "path": str(output_dir / "independent_audit.json"),
            "issue_count": 132,
            "interpretation": (
                "expected backend-schema incompatibility: it requires SciPy "
                "status-0/bound-kind fields and is not the HiGHS audit"
            ),
        },
        "excluded_incremental_audit_r1": {
            "path": str(output_dir / "incremental_independent_audit.json"),
            "issue_count": 297,
            "cause": (
                "audit implementation failed to propagate pair-level "
                "reused_parent provenance to copied child properties"
            ),
        },
        "superseded_incremental_audit_r2": {
            "path": str(output_dir / "incremental_independent_audit_r2.json"),
            "issue_count": 0,
            "cause": (
                "reporting layer incorrectly presented the inherited D0 closure "
                "threshold as an incremental-vs-D0 preregistered threshold; "
                "soundness checks and scientific rows were unaffected"
            ),
        },
        "issue_count": len(issues),
        "issues": issues,
        "rows": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "reason_counts": dict(Counter(row["reason"] for row in rows)),
        "unsafe_rows_replayed": len(unsafe_rows),
        "session_counts": session_counts,
        "paired": paired,
        "parent_confirmatory_overall_solved_rate_immutable": 0.56,
        "parent_d0_result_immutable": True,
        "conditions": conditions,
        "audit_passed": len(issues) == 0,
        "descriptive_engineering_endpoints": {
            "at_least_five_newly_solved_vs_d0": (
                paired["newly_solved_vs_d0"] >= 5
            ),
            "total_runtime_not_increased": (
                paired["incremental_over_baseline_total_runtime_ratio"] <= 1.0
            ),
            "interpretation": (
                "descriptive only; no incremental-vs-D0 performance threshold "
                "was preregistered"
            ),
        },
    }
    _write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--result-dir")
    arguments = parser.parse_args()
    print(
        json.dumps(
            audit(
                Path(arguments.config),
                Path(arguments.result_dir) if arguments.result_dir else None,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
