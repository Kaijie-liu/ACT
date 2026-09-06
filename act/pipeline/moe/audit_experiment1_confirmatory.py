# ===- act/pipeline/moe/audit_experiment1_confirmatory.py - Audit ----====#

"""Independent confirmatory census, certificate, and witness audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from act.back_end.moe import load_output_moe_checkpoint
from act.back_end.solver.solver_hz import hz_numerical_policy_manifest
from act.pipeline.moe.accounting import guard_binary_accounting
from act.pipeline.moe.experiment1 import (
    PROJECT_ROOT,
    WRITE_ROOT,
    _forward_validate,
    _inside,
    _sha256,
    _write_json,
)
from act.pipeline.moe.experiment1_confirmatory import (
    DEFAULT_CONFIG,
    SEMANTIC_REASONS,
)
from act.pipeline.moe.train import _load_dataset


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _wilson(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [center - half, center + half]


def _quantiles(values: Iterable[float]) -> dict[str, float | None]:
    data = [float(value) for value in values]
    if not data:
        return {"median": None, "q1": None, "q3": None, "p90": None}
    if len(data) == 1:
        return {
            "median": data[0],
            "q1": data[0],
            "q3": data[0],
            "p90": data[0],
        }
    q = statistics.quantiles(data, n=4, method="inclusive")
    d = statistics.quantiles(data, n=10, method="inclusive")
    return {
        "median": statistics.median(data),
        "q1": q[0],
        "q3": q[2],
        "p90": d[8],
    }


def _cluster_bootstrap(
    rows: Sequence[dict[str, Any]],
    numerator,
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    clusters: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[int(row["sample_rank"])].append(row)
    ranks = sorted(clusters)
    if not ranks:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(replicates), dtype=np.float64)
    for slot in range(int(replicates)):
        sampled = rng.choice(ranks, size=len(ranks), replace=True)
        values = [row for rank in sampled for row in clusters[int(rank)]]
        estimates[slot] = sum(bool(numerator(row)) for row in values) / len(values)
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def _audit_safe_f0(
    row: dict[str, Any], issues: list[str], *, safe_tolerance: float
) -> None:
    rank = row["sample_rank"]
    f0 = row.get("f0") or {}
    feasible = {tuple(pair) for pair in f0.get("feasible_pairs", [])}
    evaluated = {tuple(pair["pair"]) for pair in f0.get("pairs", [])}
    if not feasible or evaluated != feasible:
        issues.append(f"rank {rank}: F0 SAFE lacks complete feasible-pair coverage")
    for pair in f0.get("pairs", []):
        properties = pair.get("property_rows", [])
        if pair.get("status") != "SAFE" or len(properties) != 9:
            issues.append(f"rank {rank}: F0 SAFE pair is incomplete")
            continue
        if any(prop.get("status") != "SAFE" for prop in properties):
            issues.append(f"rank {rank}: F0 SAFE pair contains non-safe property")
        for prop in properties:
            if (
                prop.get("minimum") is None
                or prop["minimum"] <= safe_tolerance
            ):
                issues.append(f"rank {rank}: F0 SAFE margin is not strictly positive")
            if prop.get("solver_status") != 0:
                issues.append(f"rank {rank}: F0 SAFE solver status is not optimal")
            if prop.get("solver_certified_lower_bound") is None:
                issues.append(f"rank {rank}: F0 SAFE lacks certified lower bound")
            if prop.get("solver_bound_kind") not in {
                "mip_dual_bound",
                "lp_status0_optimum",
            }:
                issues.append(f"rank {rank}: F0 SAFE lacks certified bound kind")


def audit(config_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.load(config_path.open(encoding="utf-8"))
    output_dir = _inside(Path(config["output_dir"]), WRITE_ROOT)
    audit_path = output_dir / "independent_audit.json"
    if audit_path.exists():
        raise RuntimeError(f"refusing to overwrite {audit_path}")
    runtime = json.load((output_dir / "config.json").open(encoding="utf-8"))
    selection = json.load(
        (output_dir / "sample_indices.json").open(encoding="utf-8")
    )["samples"]
    census = _load_jsonl(output_dir / "census/results.jsonl")
    boundary = _load_jsonl(output_dir / "boundary/results.jsonl")
    census_csv = _load_csv(output_dir / "census/results.csv")
    boundary_csv = _load_csv(output_dir / "boundary/results.csv")
    issues: list[str] = []

    if config.get("selection_manifest"):
        expected_ranks = [int(row["sample_rank"]) for row in selection]
        manifest_path = _inside(Path(config["selection_manifest"]), PROJECT_ROOT)
        manifest = json.load(manifest_path.open(encoding="utf-8"))
        if selection != manifest.get("samples"):
            issues.append("runtime selection differs from tracked manifest")
        if runtime.get("selection_manifest_sha256") != _sha256(manifest_path):
            issues.append("runtime selection manifest hash mismatch")
    else:
        expected_ranks = list(
            range(
                int(config["rank_start"]),
                int(config["rank_start"]) + int(config["sample_count"]),
            )
        )
    selected_ranks = [int(row["sample_rank"]) for row in selection]
    if selected_ranks != expected_ranks:
        issues.append("confirmatory selection ranks differ from registration")
    if runtime["source_config_sha256"] != _sha256(config_path):
        issues.append("runtime source config hash mismatch")
    if runtime["checkpoint_sha256"] != _sha256(Path(config["checkpoint"])):
        issues.append("runtime checkpoint hash mismatch")
    expected_census_rows = int(config["sample_count"]) * len(
        config["fixed_epsilons"]
    )
    expected_boundary_rows = int(config["sample_count"])
    if len(census) != expected_census_rows or len(census_csv) != len(census):
        issues.append(
            f"census does not contain {expected_census_rows} matching JSONL/CSV rows"
        )
    if len(boundary) != expected_boundary_rows or len(boundary_csv) != len(boundary):
        issues.append(
            f"boundary does not contain {expected_boundary_rows} matching JSONL/CSV rows"
        )
    actual_policy = hz_numerical_policy_manifest()
    if config.get("numerical_safety") != actual_policy:
        issues.append("tracked numerical SAFE policy differs from implementation")
    if runtime.get("numerical_safety") != actual_policy:
        issues.append("runtime numerical SAFE policy differs from implementation")

    expected_eps = {item["label"] for item in config["fixed_epsilons"]}
    census_keys = [
        (int(row["sample_rank"]), row.get("epsilon_label")) for row in census
    ]
    if len(census_keys) != len(set(census_keys)):
        issues.append("census contains duplicate rank/epsilon rows")
    by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in census:
        by_rank[int(row["sample_rank"])].append(row)
        if row.get("status") != "COMPLETE":
            issues.append(
                f"census rank {row['sample_rank']} {row.get('epsilon_label')} "
                f"is {row.get('status')}"
            )
        if row.get("exact_candidate_minimal") is not True:
            issues.append(f"census rank {row['sample_rank']} candidate set incomplete")
        if row.get("exact_pairs_complete") is not True:
            issues.append(f"census rank {row['sample_rank']} route sets incomplete")
        if not set(row.get("exact_candidates", [])).issubset(
            row.get("ibp_candidates", [])
        ):
            issues.append(f"census rank {row['sample_rank']} exact not subset of IBP")
        if not set(row.get("exact_candidates", [])).issubset(
            row.get("zonotope_candidates", [])
        ):
            issues.append(
                f"census rank {row['sample_rank']} exact not subset of zonotope"
            )
        for branch in row.get("branches", []):
            account = branch["guard_accounting"]
            try:
                recomputed = guard_binary_accounting(
                    account["binaries_before"],
                    account["binaries_after"],
                    {
                        "lp_eliminated": account["lp_support_eliminated"],
                        "milp_eliminated": account["milp_support_eliminated"],
                    },
                ).as_dict()
            except Exception as exc:
                issues.append(f"census guard accounting error: {exc}")
            else:
                if recomputed != account:
                    issues.append("census guard accounting record changed")
    for rank in expected_ranks:
        labels = {row["epsilon_label"] for row in by_rank.get(rank, [])}
        if labels != expected_eps:
            issues.append(f"census rank {rank} has epsilon labels {sorted(labels)}")

    boundary_ranks = [int(row["sample_rank"]) for row in boundary]
    if boundary_ranks != expected_ranks:
        issues.append("boundary result ranks differ from frozen selection")
    checkpoint = _inside(Path(config["checkpoint"]), WRITE_ROOT)
    model, payload = load_output_moe_checkpoint(checkpoint, map_location="cpu")
    model.double().eval()
    dataset = _load_dataset(payload["dataset"], False, download=False)
    replayed = 0
    unique_safe = 0
    for row in boundary:
        rank = int(row["sample_rank"])
        if row.get("deadline_enforced") is not True:
            issues.append(f"rank {rank}: hard instance deadline was not enforced")
        if (
            row.get("status") != "TIMEOUT"
            and float(row.get("total_seconds", 0.0))
            > float(config["instance_timeout_seconds"])
        ):
            issues.append(f"rank {rank}: non-timeout row exceeded hard deadline")
        if row.get("error") or row.get("status") == "ERROR":
            issues.append(f"rank {rank}: explicit runner error")
        if "NUMERICAL" in str(row.get("reason", "")):
            issues.append(f"rank {rank}: numerical fallback/result")
        bracket = row.get("bracket")
        if bracket is not None:
            if not (
                bracket["lower_status"] == "stable"
                and bracket["upper_status"] == "unstable"
                and float(bracket["lower"]) < float(bracket["upper"])
            ):
                issues.append(f"rank {rank}: route bracket is not strict")
            expected_epsilon = (
                float(config["route_radius"]["primary_multiplier"])
                * float(bracket["upper"])
            )
            if not math.isclose(
                float(row["epsilon"]), expected_epsilon, rel_tol=0.0, abs_tol=1e-15
            ):
                issues.append(f"rank {rank}: primary epsilon changed")
        if row.get("f0_invoked") and row.get("gate_reason") not in SEMANTIC_REASONS:
            issues.append(f"rank {rank}: F0 invoked outside semantic incompleteness")
        if row.get("reason") == "SAFE_GATE_ELIMINATION":
            gate = row.get("gate") or {}
            if gate.get("status") != "SAFE" or gate.get("reason") != "SAFE_PROVED":
                issues.append(f"rank {rank}: gate SAFE aggregate mismatch")
            if not gate.get("branches") or any(
                branch.get("unknown_reason") != "SAFE_PROVED"
                for branch in gate.get("branches", [])
            ):
                issues.append(f"rank {rank}: gate SAFE has unresolved branch")
        if row.get("reason") == "SAFE_WEIGHTED_RANGE":
            _audit_safe_f0(
                row,
                issues,
                safe_tolerance=float(
                    config["numerical_safety"]["safe_positive_margin"]
                ),
            )
        is_unique = (
            int(row.get("exact_feasible_pair_count", 0)) > 1
            and row.get("route_invariance_status") == "UNKNOWN"
            and row.get("status") == "SAFE"
        )
        if bool(row.get("unique_safe")) != is_unique:
            issues.append(f"rank {rank}: unique SAFE flag mismatch")
        unique_safe += int(is_unique)
        if row.get("status") == "UNSAFE":
            if not row.get("full_model_witness_valid"):
                issues.append(f"rank {rank}: UNSAFE lacks full-forward flag")
                continue
            witness = row.get("witness_path")
            path = (
                _inside(output_dir / "boundary" / witness, output_dir / "boundary")
                if isinstance(witness, str) and witness
                else None
            )
            if path is None or not path.exists() or _sha256(path) != row.get(
                "witness_sha256"
            ):
                issues.append(f"rank {rank}: witness path/hash mismatch")
                continue
            saved = torch.load(path, map_location="cpu", weights_only=False)
            image, _ = dataset[int(row["dataset_index"])]
            x = image.unsqueeze(0).double()
            epsilon = float(row["epsilon"])
            checked = _forward_validate(
                model,
                saved["input"],
                lower=(x - epsilon).clamp(0, 1),
                upper=(x + epsilon).clamp(0, 1),
                clean_prediction=int(row["clean_prediction"]),
            )
            if not checked["valid"]:
                issues.append(f"rank {rank}: replayed witness is invalid")
            replayed += int(checked["valid"])

    unstable = [
        row
        for row in census
        if row.get("status") == "COMPLETE" and row.get("route_set_unstable")
    ]
    reduction_ibp = sum(
        row["exact_candidate_count"] < row["ibp_candidate_count"]
        for row in unstable
    )
    reduction_zono = sum(
        row["exact_candidate_count"] < row["zonotope_candidate_count"]
        for row in unstable
    )
    candidate = {
        "route_unstable_rows": len(unstable),
        "exact_reduces_ibp_rows": reduction_ibp,
        "exact_reduces_ibp_rate": reduction_ibp / len(unstable) if unstable else 0.0,
        "exact_reduces_ibp_cluster_bootstrap_95": _cluster_bootstrap(
            unstable,
            lambda row: row["exact_candidate_count"] < row["ibp_candidate_count"],
            replicates=int(config["bootstrap"]["replicates"]),
            seed=int(config["bootstrap"]["seed"]),
        ),
        "exact_reduces_zonotope_rows": reduction_zono,
        "exact_reduces_zonotope_rate": (
            reduction_zono / len(unstable) if unstable else 0.0
        ),
        "exact_reduces_zonotope_cluster_bootstrap_95": _cluster_bootstrap(
            unstable,
            lambda row: (
                row["exact_candidate_count"] < row["zonotope_candidate_count"]
            ),
            replicates=int(config["bootstrap"]["replicates"]),
            seed=int(config["bootstrap"]["seed"]) + 1,
        ),
    }
    widths = [
        row["route_conditioned_max_width"]
        / row["candidate_pruned_monolithic_width"]
        for row in census
        if row.get("status") == "COMPLETE"
        and row.get("candidate_pruned_monolithic_width", 0) > 0
    ]
    unstable_widths = [
        row["route_conditioned_max_width"]
        / row["candidate_pruned_monolithic_width"]
        for row in unstable
        if row.get("candidate_pruned_monolithic_width", 0) > 0
    ]
    branches = [
        branch
        for row in census
        if row.get("status") == "COMPLETE"
        for branch in row.get("branches", [])
    ]
    account_keys = (
        "binaries_before",
        "binaries_after",
        "binary_eliminated",
        "lp_support_eliminated",
        "milp_support_eliminated",
        "structural_or_propagation_eliminated",
    )
    account = {
        key: sum(branch["guard_accounting"][key] for branch in branches)
        for key in account_keys
    }
    if account["binary_eliminated"] != (
        account["lp_support_eliminated"]
        + account["milp_support_eliminated"]
        + account["structural_or_propagation_eliminated"]
    ):
        issues.append("aggregate guard accounting identity does not close")
    support_seconds = sum(branch["support"]["seconds"] for branch in branches)
    gate_branches = [
        branch
        for row in boundary
        for branch in (row.get("gate") or {}).get("branches", [])
    ]
    for branch in gate_branches:
        account_row = branch.get("guard_accounting")
        if account_row is None:
            issues.append("boundary gate branch lacks guard accounting")
            continue
        try:
            recomputed = guard_binary_accounting(
                account_row["binaries_before"],
                account_row["binaries_after"],
                {
                    "lp_eliminated": account_row["lp_support_eliminated"],
                    "milp_eliminated": account_row["milp_support_eliminated"],
                },
            ).as_dict()
        except Exception as exc:
            issues.append(f"boundary guard accounting error: {exc}")
        else:
            if recomputed != account_row:
                issues.append("boundary guard accounting record changed")
    matched_solved = sum(
        branch.get("matched_no_support_status") in {"certified", "falsified"}
        for branch in gate_branches
    )
    support_solved = sum(
        branch.get("branch_status") in {"certified", "falsified"}
        for branch in gate_branches
    )
    statuses = Counter(row.get("status") for row in boundary)
    solved = statuses["SAFE"] + statuses["UNSAFE"]
    invoked = [row for row in boundary if row.get("f0_invoked")]
    semantic = [row for row in boundary if row.get("gate_reason") in SEMANTIC_REASONS]
    f0_resolved = sum(row.get("status") in {"SAFE", "UNSAFE"} for row in invoked)
    observed_f0_seconds = [
        float(row["f0_seconds"])
        for row in invoked
        if row.get("f0_seconds") is not None
    ]
    right_censored_f0 = [
        row
        for row in invoked
        if row.get("f0_seconds") is None
        or (row.get("f0_time_observation") or {}).get("kind")
        == "RIGHT_CENSORED_AT_INSTANCE_DEADLINE"
    ]
    known_censored_lower_bounds = [
        float(row["f0_time_observation"]["lower_bound_seconds"])
        for row in right_censored_f0
        if (row.get("f0_time_observation") or {}).get("lower_bound_seconds")
        is not None
    ]
    endpoints = {
        "status_counts": dict(statuses),
        "reason_counts": dict(Counter(row.get("reason") for row in boundary)),
        "unique_safe_samples": unique_safe,
        "unique_safe_rate": unique_safe / len(boundary),
        "unique_safe_wilson_95": _wilson(unique_safe, len(boundary)),
        "solved_rows": solved,
        "solved_rate": solved / len(boundary),
        "unsafe_rows": statuses["UNSAFE"],
        "unsafe_witnesses_replayed": replayed,
        "base_semantic_incompleteness": len(semantic),
        "f0_invoked": len(invoked),
        "f0_resolved": f0_resolved,
        "f0_resolution_rate": f0_resolved / len(semantic) if semantic else 0.0,
        "f0_added_safe": sum(row.get("status") == "SAFE" for row in invoked),
        "f0_added_unsafe": sum(row.get("status") == "UNSAFE" for row in invoked),
        "f0_remaining_unknown_timeout": sum(
            row.get("status") not in {"SAFE", "UNSAFE"} for row in invoked
        ),
        "f0_seconds": sum(observed_f0_seconds),
        "f0_seconds_semantics": "observed_completed_F0_rows_only",
        "f0_observed_time_rows": len(observed_f0_seconds),
        "f0_right_censored_time_rows": len(right_censored_f0),
        "f0_right_censored_known_lower_bound_seconds": sum(
            known_censored_lower_bounds
        ),
        "f0_paired_runtime_overhead": _quantiles(observed_f0_seconds),
    }
    guard = {
        "branches": len(branches),
        "branches_with_elimination": sum(
            branch["guard_accounting"]["binary_eliminated"] > 0
            for branch in branches
        ),
        "accounting": account,
        "support_seconds": support_seconds,
        "seconds_per_eliminated_binary": (
            support_seconds / account["binary_eliminated"]
            if account["binary_eliminated"]
            else None
        ),
        "matched_no_support_solved_branches": matched_solved,
        "support_solved_branches": support_solved,
        "matched_no_support_solved_rate": (
            matched_solved / len(gate_branches) if gate_branches else 0.0
        ),
        "support_solved_rate": (
            support_solved / len(gate_branches) if gate_branches else 0.0
        ),
    }
    width = {
        "unconditional": _quantiles(widths),
        "route_unstable": _quantiles(unstable_widths),
    }
    thresholds = config["go_thresholds"]
    conditions = {
        "independent_audit_zero_issues": len(issues) == 0,
        "all_unsafe_replayed": replayed == statuses["UNSAFE"],
        "unique_safe_count": unique_safe >= int(thresholds["unique_safe_samples"]),
        "unique_safe_rate": (
            unique_safe / len(boundary) >= float(thresholds["unique_safe_rate"])
        ),
        "end_to_end_solved_rate": (
            solved / len(boundary) >= float(thresholds["end_to_end_solved_rate"])
        ),
        "candidate_reduction_ibp": candidate["exact_reduces_ibp_rate"]
        >= float(thresholds["candidate_reduction_rate"]),
        "candidate_reduction_zonotope": candidate["exact_reduces_zonotope_rate"]
        >= float(thresholds["candidate_reduction_rate"]),
        "route_unstable_width_median": (
            width["route_unstable"]["median"] is not None
            and width["route_unstable"]["median"]
            < float(thresholds["conditional_width_median"])
        ),
        "route_unstable_width_p90": (
            width["route_unstable"]["p90"] is not None
            and width["route_unstable"]["p90"]
            < float(thresholds["conditional_width_p90_strict_upper"])
        ),
        "f0_semantic_resolution": endpoints["f0_resolution_rate"]
        >= float(thresholds["f0_semantic_resolution_rate"]),
        "guard_identity_closed": account["binary_eliminated"]
        == account["lp_support_eliminated"]
        + account["milp_support_eliminated"]
        + account["structural_or_propagation_eliminated"],
        "no_silent_numerical_fallback": not any(
            row.get("status") == "ERROR"
            or "NUMERICAL" in str(row.get("reason", ""))
            for row in (*census, *boundary)
        ),
    }
    report = {
        "audit": "experiment1_confirmatory_independent",
        "issues": issues,
        "issue_count": len(issues),
        "soundness_audit_passed": not issues,
        "candidate_reduction": candidate,
        "width": width,
        "guard_support": guard,
        "end_to_end": endpoints,
        "go_conditions": conditions,
        "public_baseline_unlocked": all(conditions.values()),
        "artifact_hashes_before_audit": {
            str(path.relative_to(output_dir)): _sha256(path)
            for path in sorted(output_dir.rglob("*"))
            if path.is_file() and path != audit_path
        },
    }
    _write_json(audit_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    report = audit(Path(args.config).resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
