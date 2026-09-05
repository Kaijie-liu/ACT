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


def _aggregate(row: dict[str, Any]) -> dict[str, str]:
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
    endpoint = (
        "UNSAFE_FULL_FORWARD_REPLAY"
        if row["attack"]["prediction_flip"]
        else route_a if route_a != "UNKNOWN" else "UNKNOWN"
    )
    return {
        "route_invariance": route_invariance,
        "route_a_two_path": route_a,
        "eta_guard_ablation": eta_result,
        "endpoint": endpoint,
    }


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
    for row in rows:
        bounds = [row["router_crown"], *row["path_crown"], *row["eta_crown"]]
        for bound in bounds:
            if bound.get("status") == "ERROR":
                issues.append(f"{row['row_id']}: backend error")
            expected = expected_crown_status(bound, tolerance)
            if bound.get("status") != expected:
                issues.append(f"{row['row_id']}: CROWN status does not recompute")
        if row.get("statuses") != _aggregate(row):
            issues.append(f"{row['row_id']}: aggregate statuses do not recompute")
        if row.get("positive_filter_witness_conflict"):
            issues.append(f"{row['row_id']}: positive filter conflicts with witness")

    rebuilt_tables = independent_tables(rows, config["radii_over_255"])
    for radius, expected_table in rebuilt_tables.items():
        recorded = result.get("tables", {}).get(radius, {})
        for key in (
            "samples",
            "route_invariance",
            "route_a_two_path",
            "eta_guard_ablation",
            "endpoint",
        ):
            if recorded.get(key) != expected_table[key]:
                issues.append(f"radius {radius}: summary table {key} mismatch")
        if recorded.get("route_attack_or_prediction_witnesses") != expected_table["prediction_flip_witnesses"]:
            issues.append(f"radius {radius}: legacy witness count mismatch")

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
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "result": {"path": str(result_path), "sha256": _sha256(result_path)},
        "rows": {"count": len(rows), "sha256": _sha256(rows_path)},
        "independent_tables": rebuilt_tables,
        "legacy_field_note": (
            "runner field route_attack_or_prediction_witnesses counts prediction "
            "flips only; independent tables separate route, prediction, and both"
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
