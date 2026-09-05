"""Replay and independently audit the fixed-multiplier-family diagnosis."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from act.pipeline.moe.advmoe_adapter import specialize_advmoe_path
from act.pipeline.moe.advmoe_lagrangian_family_diagnosis import (
    _scalar_values,
    select_mu0_blocking_obligation,
)
from act.pipeline.moe.advmoe_two_path import _load_model, top1_property_rows
from act.pipeline.moe.published_moe_router_gradient_audit import _sha256


WORKSPACE = Path("/data1/Kane/MOE")


def _inside(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(WORKSPACE)
    return resolved


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _dual_upper_independent(safety: np.ndarray, margin: np.ndarray) -> dict[str, Any]:
    if not np.any(margin > 0.0):
        return {"bounded": False, "upper_bound": None, "selected_multiplier": None}
    candidates = {0.0}
    for left in range(len(safety)):
        for right in range(left + 1, len(safety)):
            denominator = float(margin[left] - margin[right])
            if denominator == 0.0:
                continue
            value = float((safety[left] - safety[right]) / denominator)
            if math.isfinite(value) and value >= 0.0:
                candidates.add(value)
    evaluated = [
        (float(np.min(safety - value * margin)), value) for value in candidates
    ]
    upper, multiplier = max(evaluated, key=lambda item: item[0])
    return {"bounded": True, "upper_bound": upper, "selected_multiplier": multiplier}


def _classification_independent(
    *,
    full_forward_unsafe: bool,
    dual: dict[str, Any],
    alpha_lower: float | None,
    plain_lower: float,
    tolerance: float,
) -> str:
    if full_forward_unsafe:
        return "TRUE_UNSAFE_FULL_FORWARD_WITNESS"
    if dual["bounded"] and float(dual["upper_bound"]) < -tolerance:
        return "FIXED_MULTIPLIER_FAMILY_BLOCKED_BY_CONCRETE_POINTS"
    if alpha_lower is not None and alpha_lower >= tolerance:
        return "PLAIN_CROWN_RELAXATION_CONFIRMED_CLOSED_BY_ALPHA"
    if alpha_lower is not None and alpha_lower > plain_lower + tolerance:
        return "BACKEND_TIGHTENING_CONTRIBUTES_WITHOUT_CLOSURE"
    return "UNRESOLVED_AFTER_REGISTERED_DIAGNOSTIC"


def audit(config_path: Path, summary_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path)
    summary_path = _inside(summary_path)
    config = _json(config_path)
    summary = _json(summary_path)
    issues: list[str] = []
    if summary.get("config", {}).get("sha256") != _sha256(config_path):
        issues.append("summary/config hash mismatch")
    for name, entry in config["inputs"].items():
        path = _inside(Path(entry["path"]))
        if _sha256(path) != entry["sha256"]:
            issues.append(f"input identity mismatch: {name}")
    stage_a_rows = _jsonl(_inside(Path(config["inputs"]["stage_a_rows"]["path"])))
    obligations = [
        {"row_id": row["row_id"], **select_mu0_blocking_obligation(
            row, float(config["numerical"]["safe_positive_margin"])
        )}
        for row in stage_a_rows
    ]
    if obligations != config["obligations"] or obligations != summary.get("obligations"):
        issues.append("blocking obligation identity mismatch")

    rows_path = _inside(Path(summary["rows"]["path"]))
    points_path = _inside(Path(summary["diagnostic_points"]["path"]))
    if _sha256(rows_path) != summary["rows"]["sha256"]:
        issues.append("row artifact hash mismatch")
    if _sha256(points_path) != summary["diagnostic_points"]["sha256"]:
        issues.append("point artifact hash mismatch")
    rows = _jsonl(rows_path)
    points_payload = np.load(points_path)
    points = points_payload["points"]
    if list(points.shape) != summary["diagnostic_points"]["shape"]:
        issues.append("point shape mismatch")
    if points.shape[0] != len(rows):
        issues.append("point/row count mismatch")

    parent_config = _json(_inside(Path(config["inputs"]["parent_config"]["path"])))
    parent_rows = _jsonl(_inside(Path(config["inputs"]["parent_rows"]["path"])))
    parent_by_id = {row["row_id"]: row for row in parent_rows}
    model, router, moe_type, _checkpoint = _load_model(parent_config, WORKSPACE)
    specialized = [specialize_advmoe_path(model, route, moe_type)[0].eval() for route in (0, 1)]
    replay_device = str(config["targeted_search"]["device"])
    model = model.to(replay_device).eval()
    router = router.to(replay_device).eval()
    specialized = [path.to(replay_device).eval() for path in specialized]
    tolerance = float(config["numerical"]["safe_positive_margin"])
    counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        parent = parent_by_id[row["row_id"]]
        route = int(row["route"])
        properties = top1_property_rows(int(parent["clean_prediction"]))
        property_row = properties[int(row["property_index"])][0]
        tensor = torch.from_numpy(points[index]).to(replay_device)
        lower = torch.clamp(tensor[0:1] - float(parent["epsilon"]), 0.0, 1.0)
        upper = torch.clamp(tensor[0:1] + float(parent["epsilon"]), 0.0, 1.0)
        if bool(torch.any(tensor < lower - 1e-7) or torch.any(tensor > upper + 1e-7)):
            issues.append(f"point outside registered box: {row['row_id']}")
        with torch.no_grad():
            safety_tensor, margin_tensor = _scalar_values(
                router, specialized[route], tensor, route=route, property_row=property_row
            )
            predictions = model(tensor).argmax(dim=1).cpu().numpy()
            routes = router(tensor).argmax(dim=1).cpu().numpy()
        safety = safety_tensor.cpu().numpy().astype(np.float64)
        margin = margin_tensor.cpu().numpy().astype(np.float64)
        recorded_values = row["point_values"]
        if not np.allclose(safety, recorded_values["safety"], rtol=0.0, atol=1e-6):
            issues.append(f"safety replay mismatch: {row['row_id']}")
        if not np.allclose(margin, recorded_values["route_margin"], rtol=0.0, atol=1e-6):
            issues.append(f"margin replay mismatch: {row['row_id']}")
        if predictions.tolist() != recorded_values["dynamic_predictions"]:
            issues.append(f"prediction replay mismatch: {row['row_id']}")
        if routes.tolist() != recorded_values["dynamic_routes"]:
            issues.append(f"route replay mismatch: {row['row_id']}")
        dual = _dual_upper_independent(safety, margin)
        recorded_dual = row["finite_point_dual_upper"]
        if dual["bounded"] != recorded_dual["bounded"]:
            issues.append(f"dual boundedness mismatch: {row['row_id']}")
        if dual["bounded"]:
            if not math.isclose(
                float(dual["upper_bound"]), float(recorded_dual["upper_bound"]),
                rel_tol=0.0, abs_tol=1e-9,
            ):
                issues.append(f"dual upper mismatch: {row['row_id']}")
        route_consistent = (margin >= 0.0) & (safety < 0.0)
        full_unsafe = bool(
            np.any(
                route_consistent
                & (predictions != int(parent["clean_prediction"]))
                & (routes == route)
            )
        )
        alpha = row["optimized_crown_mu0"]
        alpha_lower = (
            float(alpha["lower_bounds"][0])
            if bool(alpha.get("complete")) and len(alpha.get("lower_bounds", [])) == 1
            else None
        )
        expected = _classification_independent(
            full_forward_unsafe=full_unsafe,
            dual=recorded_dual,
            alpha_lower=alpha_lower,
            plain_lower=float(row["plain_crown_lower_bound"]),
            tolerance=tolerance,
        )
        if expected != row["classification"]:
            issues.append(f"classification mismatch: {row['row_id']}")
        counts[expected] = counts.get(expected, 0) + 1
    if counts != summary.get("classifications"):
        issues.append("summary classification counts mismatch")
    if summary.get("status") != "COMPLETED_NOT_INDEPENDENTLY_AUDITED":
        issues.append("unexpected runner status")
    return {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "scope": "ADV_MOE_LAGRANGIAN_FAMILY_DIAGNOSIS_AUDIT_R1",
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "result": {"path": str(summary_path), "sha256": _sha256(summary_path)},
        "rows": {"path": str(rows_path), "sha256": _sha256(rows_path)},
        "diagnostic_points": {"path": str(points_path), "sha256": _sha256(points_path)},
        "replay_device": replay_device,
        "classifications": counts,
        "issues": issues,
    }


def _write(path: Path, value: dict[str, Any]) -> None:
    path = _inside(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(arguments.config, arguments.summary)
    _write(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
