"""Independent artifact and full-forward audit for AdvMoE two-path runs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from act.pipeline.moe.advmoe_adapter import construct_official_init
from act.pipeline.moe.advmoe_router_bracket import load_cifar10_test_archive


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True
    ).strip()


def _write(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def expected_crown_status(record: dict[str, Any], tolerance: float) -> str:
    lower = np.asarray(record.get("lower_bounds", []), dtype=np.float64)
    upper = np.asarray(record.get("upper_bounds", []), dtype=np.float64)
    complete = bool(record.get("complete"))
    if (
        not complete
        or lower.size == 0
        or lower.shape != upper.shape
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
    ):
        return "UNKNOWN_INCOMPLETE"
    if bool(np.all(lower >= float(tolerance))):
        return "CERTIFIED_MARGIN_FILTER"
    return "UNKNOWN_RELAXATION"


def lagrangian_branch_issues(
    branch: dict[str, Any],
    *,
    expected_multipliers: list[float],
    property_rows: int,
    tolerance: float,
) -> list[str]:
    """Independently validate one frozen multiplier-grid aggregation."""

    issues: list[str] = []
    calls = branch.get("calls", [])
    if len(calls) != len(expected_multipliers) or not calls:
        return ["Lagrangian grid call count mismatch"]
    for index, (call, expected_multiplier) in enumerate(
        zip(calls, expected_multipliers)
    ):
        if call.get("lagrangian_multiplier") != float(expected_multiplier):
            issues.append(f"Lagrangian call {index} multiplier mismatch")
        expected_call_status = expected_crown_status(call, tolerance)
        if call.get("status") != expected_call_status:
            issues.append(f"Lagrangian call {index} CROWN status mismatch")

    if any(call.get("status") == "ERROR" for call in calls):
        expected_status = "ERROR"
        expected_complete = False
        best: np.ndarray | None = None
        selected_multipliers: list[float] = []
    elif any(not bool(call.get("complete")) for call in calls):
        expected_status = "UNKNOWN_INCOMPLETE"
        expected_complete = False
        best = None
        selected_multipliers = []
    else:
        values = np.asarray(
            [call.get("lower_bounds", []) for call in calls], dtype=np.float64
        )
        expected_complete = bool(
            values.shape == (len(calls), int(property_rows))
            and np.isfinite(values).all()
        )
        if expected_complete:
            best = values.max(axis=0)
            expected_status = (
                "CERTIFIED_MARGIN_FILTER"
                if bool(np.all(best >= float(tolerance)))
                else "UNKNOWN_RELAXATION"
            )
            selected = values.argmax(axis=0)
            selected_multipliers = [
                float(expected_multipliers[index]) for index in selected
            ]
        else:
            expected_status = "UNKNOWN_INCOMPLETE"
            best = None
            selected_multipliers = []

    if branch.get("status") != expected_status:
        issues.append("Lagrangian aggregate status mismatch")
    if bool(branch.get("complete")) != expected_complete:
        issues.append("Lagrangian completeness mismatch")
    if best is None:
        if branch.get("lower_bounds") not in (None, []):
            issues.append("incomplete Lagrangian aggregate retains lower bounds")
        if branch.get("selected_multipliers") not in (None, []):
            issues.append("incomplete Lagrangian aggregate retains multiplier selection")
        if branch.get("minimum_lower_bound") is not None:
            issues.append("incomplete Lagrangian aggregate retains minimum bound")
    else:
        recorded = np.asarray(branch.get("lower_bounds", []), dtype=np.float64)
        if recorded.shape != best.shape or not np.array_equal(recorded, best):
            issues.append("Lagrangian best bounds mismatch")
        if branch.get("selected_multipliers") != selected_multipliers:
            issues.append("Lagrangian selected multipliers mismatch")
        if branch.get("minimum_lower_bound") != float(best.min()):
            issues.append("Lagrangian minimum lower bound mismatch")
    return issues


def _aggregate(row: dict[str, Any], schema_version: int) -> dict[str, str]:
    filtered = "CERTIFIED_MARGIN_FILTER"
    clean_route = int(row["clean_route"])
    path = [value["status"] for value in row["path_crown"]]
    eta = [value["status"] for value in row["eta_crown"]]
    route_invariance = (
        "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
        if row["router_crown"]["status"] == filtered and path[clean_route] == filtered
        else "UNKNOWN"
    )
    route_a = (
        "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
        if len(path) == 2 and all(value == filtered for value in path)
        else "UNKNOWN"
    )
    eta_result = (
        "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
        if len(eta) == 2 and all(value == filtered for value in eta)
        else "UNKNOWN"
    )
    lagrangian = [
        value["status"] for value in row.get("lagrangian_guard_crown", [])
    ]
    lagrangian_result = (
        "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
        if len(lagrangian) == 2
        and all(value == filtered for value in lagrangian)
        else "UNKNOWN"
    )
    portfolio = (
        "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
        if any(
            value == "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
            for value in (
                route_invariance,
                route_a,
                eta_result,
                lagrangian_result,
            )
        )
        else "UNKNOWN"
    )
    selected_filter = route_a if schema_version == 1 else portfolio
    endpoint = (
        "UNSAFE_FULL_FORWARD_REPLAY"
        if row["attack"]["prediction_flip"]
        else selected_filter if selected_filter != "UNKNOWN" else "UNKNOWN"
    )
    statuses = {
        "route_invariance": route_invariance,
        "route_a_two_path": route_a,
        "eta_guard_ablation": eta_result,
        "endpoint": endpoint,
    }
    if schema_version >= 2:
        statuses["portfolio"] = portfolio
        statuses["lagrangian_guard_ablation"] = lagrangian_result
    return statuses


def expected_filter_witness_conflicts(row: dict[str, Any]) -> dict[str, bool]:
    """Independently rebuild router/output contradictions from raw evidence."""

    positive = "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
    statuses = row["statuses"]
    prediction_flip = bool(row["attack"]["prediction_flip"])
    route_flip = int(row["attack"]["attacked_route"]) != int(row["clean_route"])
    router_conflict = route_flip and (
        row["router_crown"]["status"] == "CERTIFIED_MARGIN_FILTER"
    )
    output_conflict = prediction_flip and any(
        statuses.get(name) == positive
        for name in (
            "route_invariance",
            "route_a_two_path",
            "eta_guard_ablation",
            "lagrangian_guard_ablation",
            "portfolio",
        )
    )
    return {
        "router_filter_route_conflict": router_conflict,
        "output_filter_prediction_conflict": output_conflict,
        "any": router_conflict or output_conflict,
    }


def summary_table_issues(
    recorded: dict[str, Any],
    expected: dict[str, Any],
    schema_version: int,
) -> list[str]:
    """Validate versioned table fields without mixing frozen schema v1 and v2."""

    issues: list[str] = []
    common = (
        "samples",
        "route_invariance",
        "route_a_two_path",
        "eta_guard_ablation",
        "endpoint",
    )
    for key in common:
        if recorded.get(key) != expected[key]:
            issues.append(f"summary table {key} mismatch")
    if schema_version == 1:
        if (
            recorded.get("route_attack_or_prediction_witnesses")
            != expected["prediction_flip_witnesses"]
        ):
            issues.append("legacy witness count mismatch")
    else:
        if (
            recorded.get("lagrangian_guard_ablation")
            != expected["lagrangian_guard_ablation"]
        ):
            issues.append("summary table lagrangian_guard_ablation mismatch")
        for key in (
            "portfolio",
            "prediction_flip_witnesses",
            "route_flip_witnesses",
            "both_flip_witnesses",
        ):
            if recorded.get(key) != expected[key]:
                issues.append(f"summary table {key} mismatch")
        if "route_attack_or_prediction_witnesses" in recorded:
            issues.append("schema v2 retains ambiguous legacy witness field")
    return issues


def independent_tables(
    rows: list[dict[str, Any]], radii_over_255: list[float]
) -> dict[str, dict[str, Any]]:
    """Rebuild per-radius status and concrete-witness tables from raw rows."""

    tables: dict[str, dict[str, Any]] = {}
    for numerator in radii_over_255:
        selected = [
            row for row in rows
            if float(row["epsilon_over_255"]) == float(numerator)
        ]
        prediction_flips = [bool(row["attack"]["prediction_flip"]) for row in selected]
        route_flips = [
            int(row["attack"]["attacked_route"]) != int(row["clean_route"])
            for row in selected
        ]
        tables[str(float(numerator))] = {
            "samples": len(selected),
            "route_invariance": dict(
                Counter(row["statuses"]["route_invariance"] for row in selected)
            ),
            "route_a_two_path": dict(
                Counter(row["statuses"]["route_a_two_path"] for row in selected)
            ),
            "eta_guard_ablation": dict(
                Counter(row["statuses"]["eta_guard_ablation"] for row in selected)
            ),
            "lagrangian_guard_ablation": dict(
                Counter(
                    row["statuses"].get("lagrangian_guard_ablation", "UNKNOWN")
                    for row in selected
                )
            ),
            "portfolio": dict(
                Counter(row["statuses"].get("portfolio", "UNKNOWN") for row in selected)
            ),
            "endpoint": dict(Counter(row["statuses"]["endpoint"] for row in selected)),
            "prediction_flip_witnesses": sum(prediction_flips),
            "route_flip_witnesses": sum(route_flips),
            "both_flip_witnesses": sum(
                prediction and route
                for prediction, route in zip(prediction_flips, route_flips)
            ),
        }
    return tables


def audit(config_path: Path, result_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    workspace = Path(config["workspace_boundary"])
    config_path = _inside(config_path, workspace)
    result_path = _inside(result_path, workspace)
    output_dir = _inside(Path(config["output_dir"]), workspace)
    source = _inside(Path(config["official_source"]["repository"]), workspace)
    checkpoint = _inside(Path(config["checkpoint"]["path"]), workspace)
    archive = _inside(Path(config["dataset_archive"]), workspace)
    issues: list[str] = []
    schema_version = int(result.get("schema_version", 1))
    if schema_version not in (1, 2):
        issues.append(f"unsupported result schema version: {schema_version}")
    if result_path != output_dir / "summary.json":
        issues.append("summary path differs from configured output identity")
    if result.get("status") != "COMPLETED_NOT_INDEPENDENTLY_AUDITED":
        issues.append("runner status is not completed")
    if result.get("scope") != config.get("scope"):
        issues.append("scope mismatch")
    if result.get("config", {}).get("sha256") != _sha256(config_path):
        issues.append("config hash mismatch")
    if _sha256(checkpoint) != config["checkpoint"]["sha256"]:
        issues.append("checkpoint hash mismatch")
    if _sha256(archive) != config["dataset_archive_sha256"]:
        issues.append("dataset archive hash mismatch")
    if _git(source, "rev-parse", "HEAD") != config["official_source"]["commit"]:
        issues.append("official source commit changed")
    if _git(source, "status", "--porcelain=v1"):
        issues.append("official source clone is dirty")

    rows_path = _inside(Path(result["rows"]["path"]), workspace)
    attack_path = _inside(Path(result["attack_endpoints"]["path"]), workspace)
    if _sha256(rows_path) != result["rows"]["sha256"]:
        issues.append("rows artifact hash mismatch")
    if _sha256(attack_path) != result["attack_endpoints"]["sha256"]:
        issues.append("attack artifact hash mismatch")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    expected_rows = int(config["selection"]["samples"]) * len(config["radii_over_255"])
    if len(rows) != expected_rows or result["rows"].get("count") != expected_rows:
        issues.append("row count mismatch")
    if len({row["row_id"] for row in rows}) != len(rows):
        issues.append("duplicate row identifier")

    tolerance = float(config["numerical"]["safe_positive_margin"])
    lagrangian_config = config.get("lagrangian_guard_ablation", {"enabled": False})
    lagrangian_enabled = bool(lagrangian_config.get("enabled", False))
    lagrangian_multipliers = [
        float(value) for value in lagrangian_config.get("multipliers", [])
    ]
    if lagrangian_enabled and not lagrangian_multipliers:
        issues.append("enabled Lagrangian audit has no frozen multiplier grid")
    if any(
        not math.isfinite(value) or value < 0.0
        for value in lagrangian_multipliers
    ):
        issues.append("Lagrangian multiplier grid is not finite and nonnegative")
    recomputed_conflict_count = 0
    for row in rows:
        bounds = [row["router_crown"], *row["path_crown"], *row["eta_crown"]]
        for bound in bounds:
            if bound.get("status") == "ERROR":
                issues.append(f"{row['row_id']}: backend error")
            expected = expected_crown_status(bound, tolerance)
            if bound.get("status") != expected:
                issues.append(f"{row['row_id']}: CROWN status does not recompute")
        lagrangian_branches = row.get("lagrangian_guard_crown", [])
        expected_branch_count = 2 if lagrangian_enabled else 0
        if len(lagrangian_branches) != expected_branch_count:
            issues.append(f"{row['row_id']}: Lagrangian branch count mismatch")
        for branch in lagrangian_branches:
            issues.extend(
                f"{row['row_id']}: {issue}"
                for issue in lagrangian_branch_issues(
                    branch,
                    expected_multipliers=lagrangian_multipliers,
                    property_rows=9,
                    tolerance=tolerance,
                )
            )
        if row.get("statuses") != _aggregate(row, schema_version):
            issues.append(f"{row['row_id']}: aggregate statuses do not recompute")
        expected_conflicts = expected_filter_witness_conflicts(row)
        if schema_version == 1:
            legacy_expected = bool(row["attack"]["prediction_flip"]) and any(
                row["statuses"].get(name)
                == "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
                for name in ("route_a_two_path", "eta_guard_ablation")
            )
            if bool(row.get("positive_filter_witness_conflict")) != legacy_expected:
                issues.append(f"{row['row_id']}: legacy conflict flag does not recompute")
        elif row.get("filter_witness_conflicts") != expected_conflicts:
            issues.append(f"{row['row_id']}: witness conflicts do not recompute")
        if expected_conflicts["any"]:
            recomputed_conflict_count += 1
            issues.append(f"{row['row_id']}: positive filter conflicts with witness")

    recorded_conflict_count = (
        result.get("positive_filter_witness_conflicts")
        if schema_version == 1
        else result.get("filter_witness_conflicts")
    )
    if recorded_conflict_count != recomputed_conflict_count:
        issues.append("top-level filter/witness conflict count mismatch")

    rebuilt_tables = independent_tables(rows, config["radii_over_255"])
    for radius, expected_table in rebuilt_tables.items():
        recorded = result.get("tables", {}).get(radius, {})
        issues.extend(
            f"radius {radius}: {issue}"
            for issue in summary_table_issues(
                recorded, expected_table, schema_version
            )
        )

    inputs, labels = load_cifar10_test_archive(archive)
    model, router, _moe_type = construct_official_init(int(config["checkpoint"]["seed"]))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    router.load_state_dict(payload["router"])
    model.router = router
    model.eval()
    with np.load(attack_path, allow_pickle=False) as raw:
        centers = torch.from_numpy(raw["centers"].copy())
        endpoints = torch.from_numpy(raw["endpoints"].copy())
        indices = raw["dataset_indices"].astype(np.int64)
    configured_indices = np.asarray(result["selection"]["dataset_indices"], dtype=np.int64)
    if not np.array_equal(indices, configured_indices):
        issues.append("selected indices differ across artifacts")
    with torch.no_grad():
        all_predictions = []
        for start in range(0, len(inputs), int(config["batch_size"])):
            all_predictions.append(
                model(torch.from_numpy(inputs[start : start + int(config["batch_size"])]))
                .argmax(dim=1)
                .numpy()
            )
        all_predictions = np.concatenate(all_predictions)
    expected_indices = np.flatnonzero(all_predictions == labels)[: int(config["selection"]["samples"])]
    if not np.array_equal(indices, expected_indices):
        issues.append("selection is not the first deterministic clean-correct cohort")

    replay_rows = []
    for row_index, row in enumerate(rows):
        center = centers[int(row["sample_slot"])]
        endpoint = endpoints[row_index]
        epsilon = float(row["epsilon"])
        lower = torch.clamp(center - epsilon, 0.0, 1.0)
        upper = torch.clamp(center + epsilon, 0.0, 1.0)
        if bool(torch.any(endpoint < lower - 1e-7)) or bool(torch.any(endpoint > upper + 1e-7)):
            issues.append(f"{row['row_id']}: attack endpoint leaves box")
        with torch.no_grad():
            prediction = int(model(endpoint.unsqueeze(0)).argmax(dim=1).item())
            route = int(router(endpoint.unsqueeze(0)).argmax(dim=1).item())
        prediction_flip = prediction != int(row["clean_prediction"])
        if prediction != int(row["attack"]["attacked_prediction"]):
            issues.append(f"{row['row_id']}: attacked prediction does not replay")
        if route != int(row["attack"]["attacked_route"]):
            issues.append(f"{row['row_id']}: attacked route does not replay")
        if prediction_flip != bool(row["attack"]["prediction_flip"]):
            issues.append(f"{row['row_id']}: prediction-flip flag does not replay")
        maximum_linf = float((endpoint - center).abs().max().item())
        if not math.isclose(maximum_linf, float(row["attack"]["maximum_linf"]), abs_tol=1e-12):
            issues.append(f"{row['row_id']}: L-infinity norm does not recompute")
        replay_rows.append(
            {
                "row_id": row["row_id"],
                "prediction": prediction,
                "route": route,
                "prediction_flip": prediction_flip,
                "maximum_linf": maximum_linf,
            }
        )

    equivalence = result.get("equivalence", {})
    if not equivalence.get("router", {}).get("outputs_equal"):
        issues.append("router lowering equivalence gate is absent")
    if not all(row.get("outputs_close") and row.get("predictions_equal") for row in equivalence.get("paths", [])):
        issues.append("path lowering equivalence gate is absent")
    if float(equivalence.get("dynamic_selected_max_abs_error", math.inf)) > float(config["numerical"]["equivalence_atol"]):
        issues.append("dynamic/static selected path exceeds equivalence tolerance")
    if result.get("formal_safe_count") != 0:
        issues.append("non-outward-rounded run reports formal SAFE")
    return {
        "schema_version": 2,
        "status": "PASS" if not issues else "FAIL",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "result": {"path": str(result_path), "sha256": _sha256(result_path)},
        "rows": {"count": len(rows), "sha256": _sha256(rows_path)},
        "independent_tables": rebuilt_tables,
        "audited_result_schema_version": schema_version,
        "legacy_field_note": (
            "schema v1 field route_attack_or_prediction_witnesses counts prediction "
            "flips only; schema v2 separates route, prediction, and both"
        ),
        "attack_endpoints": {"sha256": _sha256(attack_path), "replays": replay_rows},
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(arguments.config, arguments.result)
    _write(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
