"""Independently audit and aggregate the common three-model performance task."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from typing import Any, Mapping

from act.pipeline.moe.accounting import guard_binary_accounting
from act.pipeline.moe.audit_staged_evidence import audit_evidence_package
from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256, _write_json


MODEL_IDS = ("seed0", "seed1", "seed2")
SELECTION = (
    PROJECT_ROOT
    / "act/pipeline/moe/configs/staged_verifier_multimodel_fixed2_selection_r1.json"
)
SELECTION_AUDIT = (
    PROJECT_ROOT
    / "act/pipeline/moe/results/staged_verifier_multimodel_selection_20260906_r1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "act/pipeline/moe/results/staged_verifier_multimodel_bundle_20260906_r1.json"
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "q1": None, "q3": None, "p90": None}
    if len(values) == 1:
        return {"median": values[0], "q1": values[0], "q3": values[0], "p90": values[0]}
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    deciles = statistics.quantiles(values, n=10, method="inclusive")
    return {
        "median": statistics.median(values),
        "q1": quartiles[0],
        "q3": quartiles[2],
        "p90": deciles[8],
    }


def _audit_census(
    model_id: str,
    config: Mapping[str, Any],
    samples: list[dict[str, Any]],
    issues: list[str],
) -> dict[str, Any]:
    output = _inside(Path(config["output_dir"]), WRITE_ROOT)
    runtime_path = output / "config.json"
    rows_path = output / "census/results.jsonl"
    summary_path = output / "census/summary.json"
    for path in (runtime_path, rows_path, summary_path):
        if not path.is_file():
            issues.append(f"{model_id} census missing {path}")
    if not all(path.is_file() for path in (runtime_path, rows_path, summary_path)):
        return {}
    runtime = _load(runtime_path)
    rows = _load_jsonl(rows_path)
    if runtime.get("source_config_sha256") != _sha256(
        PROJECT_ROOT / f"act/pipeline/moe/configs/staged_verifier_multimodel_{model_id}_census_r1.json"
    ):
        issues.append(f"{model_id} census config hash differs")
    if runtime.get("selection_manifest_sha256") != _sha256(SELECTION):
        issues.append(f"{model_id} census selection hash differs")
    if runtime.get("checkpoint_sha256") != _sha256(Path(config["checkpoint"])):
        issues.append(f"{model_id} census checkpoint hash differs")
    if len(rows) != len(samples):
        issues.append(f"{model_id} census row count differs")

    unstable: list[dict[str, Any]] = []
    width_ratios: list[float] = []
    exact_subset = True
    accounts = Counter()
    branches = 0
    branches_with_elimination = 0
    for expected, row in zip(samples, rows):
        rank = int(expected["sample_rank"])
        if int(row.get("sample_rank", -1)) != rank or int(
            row.get("dataset_index", -1)
        ) != int(expected["dataset_index"]):
            issues.append(f"{model_id} census identity differs at rank {rank}")
        if row.get("status") != "COMPLETE":
            issues.append(f"{model_id} census incomplete at rank {rank}")
            continue
        if row.get("exact_candidate_minimal") is not True or row.get(
            "exact_pairs_complete"
        ) is not True:
            issues.append(f"{model_id} exact router result incomplete at rank {rank}")
        exact = set(int(value) for value in row.get("exact_candidates", []))
        ibp = set(int(value) for value in row.get("ibp_candidates", []))
        zonotope = set(int(value) for value in row.get("zonotope_candidates", []))
        if not exact.issubset(ibp) or not exact.issubset(zonotope):
            exact_subset = False
            issues.append(f"{model_id} exact candidate subset violation at rank {rank}")
        if row.get("route_set_unstable") is True:
            unstable.append(row)
            denominator = int(row.get("candidate_pruned_monolithic_width", 0))
            if denominator <= 0:
                issues.append(f"{model_id} invalid width denominator at rank {rank}")
            else:
                width_ratios.append(
                    float(row["route_conditioned_max_width"]) / denominator
                )
        for branch in row.get("branches", []):
            branches += 1
            account = branch.get("guard_accounting", {})
            try:
                rebuilt = guard_binary_accounting(
                    int(account["binaries_before"]),
                    int(account["binaries_after"]),
                    {
                        "lp_eliminated": int(account["lp_support_eliminated"]),
                        "milp_eliminated": int(account["milp_support_eliminated"]),
                    },
                ).as_dict()
            except Exception as error:
                issues.append(f"{model_id} guard accounting failed: {error}")
                continue
            if rebuilt != account:
                issues.append(f"{model_id} guard accounting record differs")
            branches_with_elimination += int(rebuilt["binary_eliminated"] > 0)
            accounts.update({key: int(value) for key, value in rebuilt.items()})

    reduction_zonotope = sum(
        int(row["exact_candidate_count"]) < int(row["zonotope_candidate_count"])
        for row in unstable
    )
    reduction_ibp = sum(
        int(row["exact_candidate_count"]) < int(row["ibp_candidate_count"])
        for row in unstable
    )
    return {
        "rows": len(rows),
        "route_unstable_rows": len(unstable),
        "exact_candidate_subset_consistent": exact_subset,
        "exact_reduces_zonotope_rows": reduction_zonotope,
        "exact_reduces_zonotope_rate": (
            reduction_zonotope / len(unstable) if unstable else 0.0
        ),
        "exact_reduces_ibp_rows": reduction_ibp,
        "exact_reduces_ibp_rate": reduction_ibp / len(unstable) if unstable else 0.0,
        "route_unstable_width_ratio": _quantiles(width_ratios),
        "guard": {
            "branches": branches,
            "branches_with_elimination": branches_with_elimination,
            "accounting": dict(sorted(accounts.items())),
        },
        "artifacts": {
            "runtime": {"path": str(runtime_path), "sha256": _sha256(runtime_path)},
            "rows": {"path": str(rows_path), "sha256": _sha256(rows_path)},
            "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
        },
    }


def _audit_verdict(
    model_id: str,
    config: Mapping[str, Any],
    samples: list[dict[str, Any]],
    issues: list[str],
) -> dict[str, Any]:
    output = _inside(Path(config["output_dir"]), WRITE_ROOT)
    runtime_path = output / "config.json"
    rows_path = output / "results.jsonl"
    summary_path = output / "summary.json"
    for path in (runtime_path, rows_path, summary_path):
        if not path.is_file():
            issues.append(f"{model_id} verdict missing {path}")
    if not all(path.is_file() for path in (runtime_path, rows_path, summary_path)):
        return {}
    runtime = _load(runtime_path)
    rows = _load_jsonl(rows_path)
    source_config = (
        PROJECT_ROOT
        / f"act/pipeline/moe/configs/staged_verifier_multimodel_{model_id}_fixed2_r1.json"
    )
    if runtime.get("source_config_sha256") != _sha256(source_config):
        issues.append(f"{model_id} verdict config hash differs")
    if runtime.get("selection_manifest_sha256") != _sha256(SELECTION):
        issues.append(f"{model_id} verdict selection hash differs")
    if len(rows) != len(samples):
        issues.append(f"{model_id} verdict row count differs")

    statuses = Counter()
    reasons = Counter()
    route_changing_safe = 0
    tier1_safe = 0
    f0_invoked = 0
    f0_resolved = 0
    f0_safe = 0
    packages_audited = 0
    unsafe_replayed = 0
    for expected, row in zip(samples, rows):
        rank = int(expected["sample_rank"])
        if int(row.get("sample_rank", -1)) != rank or int(
            row.get("dataset_index", -1)
        ) != int(expected["dataset_index"]):
            issues.append(f"{model_id} verdict identity differs at rank {rank}")
        statuses[str(row.get("status"))] += 1
        reasons[str(row.get("reason"))] += 1
        route_changing_safe += int(
            row.get("status") == "SAFE" and row.get("route_changing") is True
        )
        tier1_safe += int(
            row.get("status") == "SAFE"
            and row.get("decision_tier") == "TIER1_GATE_ELIMINATION"
        )
        invoked = row.get("f0_invoked") is True
        f0_invoked += int(invoked)
        f0_resolved += int(invoked and row.get("status") in {"SAFE", "UNSAFE"})
        f0_safe += int(invoked and row.get("status") == "SAFE")
        if row.get("status") == "ERROR":
            issues.append(f"{model_id} process error at rank {rank}")
        package_value = row.get("package")
        if package_value is None:
            if not (
                row.get("status") == "TIMEOUT"
                and row.get("outer_hard_timeout") is True
            ):
                issues.append(f"{model_id} missing evidence package at rank {rank}")
            continue
        package = _inside(Path(package_value), WRITE_ROOT)
        package_audit = audit_evidence_package(
            package, replay_unsafe=row.get("status") == "UNSAFE"
        )
        packages_audited += 1
        if package_audit.get("status") != "PASS":
            issues.append(f"{model_id} evidence audit failed at rank {rank}")
        manifest = _load(package / "manifest.json")
        if manifest.get("status") != row.get("status") or manifest.get(
            "reason"
        ) != row.get("reason"):
            issues.append(f"{model_id} package verdict differs at rank {rank}")
        if row.get("status") == "UNSAFE":
            valid = row.get("full_model_witness_valid") is True and package_audit.get(
                "replay_unsafe"
            ) is True
            unsafe_replayed += int(valid)
            if not valid:
                issues.append(f"{model_id} UNSAFE replay failed at rank {rank}")

    complete = statuses["SAFE"] + statuses["UNSAFE"]
    return {
        "rows": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "complete_outcomes": complete,
        "complete_outcome_rate": complete / len(rows) if rows else 0.0,
        "route_changing_safe": route_changing_safe,
        "tier1_safe": tier1_safe,
        "f0_invoked": f0_invoked,
        "f0_resolved": f0_resolved,
        "f0_resolution_rate": f0_resolved / f0_invoked if f0_invoked else 0.0,
        "f0_safe": f0_safe,
        "packages_audited": packages_audited,
        "unsafe_replayed": unsafe_replayed,
        "all_unsafe_replayed": unsafe_replayed == statuses["UNSAFE"],
        "artifacts": {
            "runtime": {"path": str(runtime_path), "sha256": _sha256(runtime_path)},
            "rows": {"path": str(rows_path), "sha256": _sha256(rows_path)},
            "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
        },
    }


def analyze() -> dict[str, Any]:
    selection = _load(SELECTION)
    selection_audit = _load(SELECTION_AUDIT)
    issues: list[str] = []
    if selection_audit.get("status") != "PASS" or selection_audit.get(
        "issue_count"
    ) != 0:
        issues.append("pre-endpoint common selection audit did not pass")
    if selection_audit.get("selection", {}).get("sha256") != _sha256(SELECTION):
        issues.append("selection changed after pre-endpoint audit")
    samples = selection["samples"]
    thresholds = selection["per_model_performance_bundle"]
    models: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        census_config_path = (
            PROJECT_ROOT
            / f"act/pipeline/moe/configs/staged_verifier_multimodel_{model_id}_census_r1.json"
        )
        verdict_config_path = (
            PROJECT_ROOT
            / f"act/pipeline/moe/configs/staged_verifier_multimodel_{model_id}_fixed2_r1.json"
        )
        census_config = _load(census_config_path)
        verdict_config = _load(verdict_config_path)
        census = _audit_census(model_id, census_config, samples, issues)
        verdict = _audit_verdict(model_id, verdict_config, samples, issues)
        if not census or not verdict:
            models[model_id] = {"census": census, "verdict": verdict, "gates": {}}
            continue
        account = census["guard"]["accounting"]
        guard_identity = account.get("binary_eliminated") == (
            account.get("lp_support_eliminated", 0)
            + account.get("milp_support_eliminated", 0)
            + account.get("structural_or_propagation_eliminated", 0)
        )
        width = census["route_unstable_width_ratio"]
        gates = {
            "integrity": (
                census["exact_candidate_subset_consistent"]
                and guard_identity
                and verdict["all_unsafe_replayed"]
            ),
            "route_changing_safe": verdict["route_changing_safe"]
            >= int(thresholds["minimum_route_changing_safe_requests"]),
            "complete_outcome_rate": verdict["complete_outcome_rate"]
            >= float(thresholds["minimum_complete_outcome_rate"]),
            "candidate_reduction_zonotope": census["exact_reduces_zonotope_rate"]
            >= float(
                thresholds[
                    "minimum_exact_vs_zonotope_strict_reduction_rate_on_route_unstable_rows"
                ]
            ),
            "route_unstable_width_median": width["median"] is not None
            and float(width["median"])
            <= float(thresholds["maximum_route_unstable_width_ratio_median"]),
            "route_unstable_width_p90": width["p90"] is not None
            and float(width["p90"])
            < float(thresholds["maximum_route_unstable_width_ratio_p90_strict"]),
            "f0_resolution": verdict["f0_resolution_rate"]
            >= float(thresholds["minimum_f0_complete_resolution_rate_when_invoked"]),
            "guard_binary_elimination": int(account.get("binary_eliminated", 0))
            >= int(thresholds["minimum_guard_binary_eliminations"]),
        }
        models[model_id] = {
            "checkpoint_sha256": selection["models"][model_id]["checkpoint_sha256"],
            "clean_accuracy": selection["models"][model_id]["clean_accuracy"],
            "census": census,
            "verdict": verdict,
            "gates": gates,
            "full_bundle_pass": all(gates.values()),
        }

    all_model_pass = bool(models) and all(
        row.get("full_bundle_pass") is True for row in models.values()
    )
    return {
        "schema_version": 1,
        "experiment": "staged_verifier_multimodel_fixed2_r1",
        "classification": "PREREGISTERED_COMMON_FIXED_TASK_CROSS_MODEL_BUNDLE",
        "selection": {"path": str(SELECTION), "sha256": _sha256(SELECTION)},
        "thresholds": thresholds,
        "models": models,
        "cross_model_decision": {
            "models_passing_complete_bundle": sum(
                row.get("full_bundle_pass") is True for row in models.values()
            ),
            "registered_models": len(MODEL_IDS),
            "stable_complete_bundle_supported": all_model_pass and not issues,
            "rule": selection["cross_model_success_rule"],
        },
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "claim_boundary": selection["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze()
    output = _inside(args.output, PROJECT_ROOT)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    _write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
