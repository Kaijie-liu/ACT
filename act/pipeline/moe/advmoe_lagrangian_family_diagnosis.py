"""Diagnose backend, witness, and fixed-multiplier-family attribution layers."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch

from act.back_end.moe.tie_safe_implication import LagrangianTop1GuardedProperty
from act.pipeline.moe.advmoe_adapter import (
    CrownCompatibleAdvMoePath,
    CrownCompatibleAdvMoeRouter,
    specialize_advmoe_path,
)
from act.pipeline.moe.advmoe_lagrangian_attribution import _append_json, _inside
from act.pipeline.moe.advmoe_router_bracket import load_cifar10_test_archive
from act.pipeline.moe.advmoe_two_path import _cleanup_cuda, _load_model, top1_property_rows
from act.pipeline.moe.crown_adapter_cohort import (
    _crown_bounds,
    validate_crown_configuration,
)
from act.pipeline.moe.published_moe_router_gradient_audit import _sha256


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], text=True
    ).strip()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def select_mu0_blocking_obligation(
    attribution_row: dict[str, Any], tolerance: float
) -> dict[str, Any]:
    """Choose the most negative necessary property still selecting mu=0."""

    candidates = []
    for branch in attribution_row["branches"]:
        lower = branch["combined"]["lower_bounds"]
        multipliers = branch["combined"]["selected_multipliers"]
        for property_index, (value, multiplier) in enumerate(zip(lower, multipliers)):
            if float(value) < float(tolerance) and float(multiplier) == 0.0:
                candidates.append(
                    {
                        "route": int(branch["route"]),
                        "property_index": property_index,
                        "plain_crown_lower_bound": float(value),
                    }
                )
    if not candidates:
        raise RuntimeError(f"no mu=0 blocker for {attribution_row['row_id']}")
    return min(
        candidates,
        key=lambda value: (
            value["plain_crown_lower_bound"],
            value["route"],
            value["property_index"],
        ),
    )


def finite_point_dual_upper(
    safety: np.ndarray, margin: np.ndarray
) -> dict[str, Any]:
    """Maximize the lower envelope over nonnegative scalar multipliers.

    For concrete points x_t, this computes
    ``sup_mu>=0 min_t (s_t - mu*m_t)``.  It is an upper bound on the exact
    fixed-multiplier certificate value over the full input box.
    """

    safety = np.asarray(safety, dtype=np.float64)
    margin = np.asarray(margin, dtype=np.float64)
    if safety.ndim != 1 or safety.shape != margin.shape or not safety.size:
        raise ValueError("finite-point safety and margin arrays must align")
    if not np.isfinite(safety).all() or not np.isfinite(margin).all():
        raise ValueError("finite-point values must be finite")
    if not np.any(margin > 0.0):
        return {
            "bounded": False,
            "upper_bound": None,
            "selected_multiplier": None,
            "active_point_indices": [],
        }
    candidates = [0.0]
    for left in range(len(safety)):
        for right in range(left + 1, len(safety)):
            denominator = margin[left] - margin[right]
            if denominator == 0.0:
                continue
            value = (safety[left] - safety[right]) / denominator
            if math.isfinite(float(value)) and value >= 0.0:
                candidates.append(float(value))
    values = np.asarray(
        [np.min(safety - multiplier * margin) for multiplier in candidates],
        dtype=np.float64,
    )
    best = int(np.argmax(values))
    multiplier = float(candidates[best])
    point_values = safety - multiplier * margin
    upper = float(values[best])
    active = np.flatnonzero(np.isclose(point_values, upper, rtol=0.0, atol=1e-9))
    return {
        "bounded": True,
        "upper_bound": upper,
        "selected_multiplier": multiplier,
        "active_point_indices": active.astype(int).tolist(),
        "candidate_breakpoints": len(candidates),
        "semantics": (
            "executed-float finite-point upper bound on the exact scalar "
            "fixed-multiplier certificate family"
        ),
    }


def _scalar_values(
    router: torch.nn.Module,
    expert: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    route: int,
    property_row: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = router(inputs)
    logits = expert(inputs)
    row = torch.as_tensor(property_row, device=logits.device, dtype=logits.dtype)
    safety = logits @ row
    margin = scores[:, int(route)] - scores[:, 1 - int(route)]
    return safety, margin


def targeted_phi_points(
    router: torch.nn.Module,
    expert: torch.nn.Module,
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    route: int,
    property_row: np.ndarray,
    multipliers: list[float],
    steps: int,
    restarts: int,
    step_size: float,
    seed: int,
    device: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Find one low executed-float phi point for each frozen multiplier."""

    router = router.to(device).eval()
    expert = expert.to(device).eval()
    center = center.to(device)
    lower = lower.to(device)
    upper = upper.to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    endpoints = []
    records = []
    for multiplier in multipliers:
        best_value = math.inf
        best_point = center.detach().clone()
        for restart in range(int(restarts)):
            if restart == 0:
                candidate = center.detach().clone()
            else:
                random = torch.rand(
                    center.shape,
                    generator=generator,
                    device=device,
                    dtype=center.dtype,
                )
                candidate = lower + random * (upper - lower)
            for _step in range(int(steps)):
                candidate = candidate.detach().requires_grad_(True)
                safety, margin = _scalar_values(
                    router,
                    expert,
                    candidate,
                    route=route,
                    property_row=property_row,
                )
                value = safety - float(multiplier) * margin
                gradient = torch.autograd.grad(value.sum(), candidate)[0]
                candidate = torch.maximum(
                    lower,
                    torch.minimum(upper, candidate.detach() - step_size * gradient.sign()),
                )
                with torch.no_grad():
                    check_safety, check_margin = _scalar_values(
                        router,
                        expert,
                        candidate,
                        route=route,
                        property_row=property_row,
                    )
                    check = float(
                        (check_safety - float(multiplier) * check_margin).item()
                    )
                    if check < best_value:
                        best_value = check
                        best_point = candidate.detach().clone()
        with torch.no_grad():
            safety, margin = _scalar_values(
                router,
                expert,
                best_point,
                route=route,
                property_row=property_row,
            )
        endpoints.append(best_point.cpu().numpy())
        records.append(
            {
                "target_multiplier": float(multiplier),
                "phi": best_value,
                "safety": float(safety.item()),
                "route_margin": float(margin.item()),
            }
        )
    return np.concatenate(endpoints, axis=0), records


def _classify(
    *,
    full_forward_unsafe: bool,
    dual_upper: dict[str, Any],
    alpha_lower: float | None,
    plain_lower: float,
    tolerance: float,
) -> str:
    if full_forward_unsafe:
        return "TRUE_UNSAFE_FULL_FORWARD_WITNESS"
    if dual_upper["bounded"] and float(dual_upper["upper_bound"]) < -tolerance:
        return "FIXED_MULTIPLIER_FAMILY_BLOCKED_BY_CONCRETE_POINTS"
    if alpha_lower is not None and alpha_lower >= tolerance:
        return "PLAIN_CROWN_RELAXATION_CONFIRMED_CLOSED_BY_ALPHA"
    if alpha_lower is not None and alpha_lower > plain_lower + tolerance:
        return "BACKEND_TIGHTENING_CONTRIBUTES_WITHOUT_CLOSURE"
    return "UNRESOLVED_AFTER_REGISTERED_DIAGNOSTIC"


def run(config_path: Path) -> dict[str, Any]:
    config = _json(config_path)
    workspace = Path(config["workspace_boundary"])
    config_path = _inside(config_path, workspace)
    repository = _inside(Path(config["act_repository"]), workspace)
    output_dir = _inside(Path(config["output_dir"]), workspace)
    if config.get("status") != "PREREGISTERED_NOT_RUN":
        raise RuntimeError("family diagnostic configuration is not frozen")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if _git(repository, "branch", "--show-current") != config["required_branch"]:
        raise RuntimeError("ACT branch gate failed")
    if _git(repository, "status", "--porcelain=v1"):
        raise RuntimeError("ACT worktree is dirty")

    identities = {}
    for name, entry in config["inputs"].items():
        path = _inside(Path(entry["path"]), workspace)
        if _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"input identity mismatch: {name}")
        identities[name] = path
    stage_a_summary = _json(identities["stage_a_summary"])
    stage_a_audit = _json(identities["stage_a_audit"])
    if stage_a_audit.get("status") != "PASS" or stage_a_audit.get("issues") != []:
        raise RuntimeError("Stage-A audit gate failed")
    stage_a_rows = _jsonl(identities["stage_a_rows"])
    obligations = [
        {"row_id": row["row_id"], **select_mu0_blocking_obligation(
            row, float(config["numerical"]["safe_positive_margin"])
        )}
        for row in stage_a_rows
    ]
    if obligations != config["obligations"]:
        raise RuntimeError("frozen blocking obligations changed")

    parent_config = _json(identities["parent_config"])
    parent_rows = _jsonl(identities["parent_rows"])
    parent_by_id = {row["row_id"]: row for row in parent_rows}
    attack_payload = np.load(identities["parent_attack_endpoints"])
    if attack_payload["endpoints"].shape[0] != len(parent_rows):
        raise RuntimeError("parent attack endpoint count mismatch")
    parent_position = {row["row_id"]: index for index, row in enumerate(parent_rows)}

    crown = config["optimized_crown"]
    crown_configuration = validate_crown_configuration(
        method=crown["method"],
        track_gradients=bool(crown["gradient_tracking"]),
        bound_options=crown["bound_options"],
    )
    free, total = torch.cuda.mem_get_info(crown["device"])
    if free / 1024**3 < float(crown["minimum_free_gpu_memory_gib"]):
        raise RuntimeError("GPU memory gate failed")

    archive = _inside(Path(parent_config["dataset_archive"]), workspace)
    inputs, _labels = load_cifar10_test_archive(archive)
    model, router, moe_type, checkpoint = _load_model(parent_config, workspace)
    specialized = [
        specialize_advmoe_path(model, route, moe_type)[0].eval()
        for route in (0, 1)
    ]
    router_adapter = CrownCompatibleAdvMoeRouter(router).eval()
    path_adapters = [CrownCompatibleAdvMoePath(path).eval() for path in specialized]
    tolerance = float(config["numerical"]["safe_positive_margin"])
    multipliers = [float(value) for value in stage_a_summary["multiplier_grid"]["combined"]]

    output_dir.mkdir(parents=True)
    rows_path = output_dir / "rows.jsonl"
    all_points = []
    records = []
    started = time.monotonic()
    with rows_path.open("x", encoding="utf-8") as handle:
        for obligation_index, obligation in enumerate(obligations):
            parent = parent_by_id[obligation["row_id"]]
            route = int(obligation["route"])
            property_index = int(obligation["property_index"])
            properties = top1_property_rows(int(parent["clean_prediction"]))
            property_row, offset = properties[property_index]
            if float(offset) != 0.0:
                raise RuntimeError("registered AdvMoE properties require zero offset")
            dataset_index = int(parent["dataset_index"])
            center = torch.from_numpy(inputs[dataset_index : dataset_index + 1])
            epsilon = float(parent["epsilon"])
            lower = torch.clamp(center - epsilon, 0.0, 1.0)
            upper = torch.clamp(center + epsilon, 0.0, 1.0)

            matrix = np.asarray(property_row, dtype=np.float64)[None, :]
            compiled_mu0 = LagrangianTop1GuardedProperty(
                copy.deepcopy(router_adapter),
                copy.deepcopy(path_adapters[route]),
                route,
                matrix,
                np.asarray([0.0]),
                np.zeros((1, 1), dtype=np.float64),
            )
            alpha = _crown_bounds(
                compiled_mu0,
                center,
                lower,
                upper,
                property_rows=None,
                device=crown["device"],
                tolerance=tolerance,
                method=crown["method"],
                track_gradients=bool(crown["gradient_tracking"]),
                bound_options=crown["bound_options"],
            )
            _cleanup_cuda()

            targeted_points, targeted_records = targeted_phi_points(
                router,
                specialized[route],
                center,
                lower,
                upper,
                route=route,
                property_row=property_row,
                multipliers=multipliers,
                steps=int(config["targeted_search"]["steps"]),
                restarts=int(config["targeted_search"]["restarts"]),
                step_size=epsilon / float(config["targeted_search"]["step_divisor"]),
                seed=int(config["targeted_search"]["seed"]) + obligation_index,
                device=config["targeted_search"]["device"],
            )
            parent_endpoint = attack_payload["endpoints"][
                parent_position[obligation["row_id"]]
            ][None, ...]
            points = np.concatenate([center.numpy(), parent_endpoint, targeted_points], axis=0)
            points_tensor = torch.from_numpy(points).to(crown["device"])
            literal_router = router.to(crown["device"]).eval()
            literal_path = specialized[route].to(crown["device"]).eval()
            literal_model = model.to(crown["device"]).eval()
            with torch.no_grad():
                safety_tensor, margin_tensor = _scalar_values(
                    literal_router,
                    literal_path,
                    points_tensor,
                    route=route,
                    property_row=property_row,
                )
                dynamic_logits = literal_model(points_tensor)
                dynamic_predictions = dynamic_logits.argmax(dim=1)
                dynamic_routes = literal_router(points_tensor).argmax(dim=1)
            safety = safety_tensor.cpu().numpy().astype(np.float64)
            margin = margin_tensor.cpu().numpy().astype(np.float64)
            dual_upper = finite_point_dual_upper(safety, margin)
            route_consistent_violation = (margin >= 0.0) & (safety < 0.0)
            full_forward_unsafe = bool(
                np.any(
                    route_consistent_violation
                    & (dynamic_predictions.cpu().numpy() != int(parent["clean_prediction"]))
                    & (dynamic_routes.cpu().numpy() == route)
                )
            )
            alpha_lower = (
                float(alpha["lower_bounds"][0])
                if bool(alpha.get("complete")) and len(alpha.get("lower_bounds", [])) == 1
                else None
            )
            classification = _classify(
                full_forward_unsafe=full_forward_unsafe,
                dual_upper=dual_upper,
                alpha_lower=alpha_lower,
                plain_lower=float(obligation["plain_crown_lower_bound"]),
                tolerance=tolerance,
            )
            record = {
                **obligation,
                "dataset_index": dataset_index,
                "epsilon": epsilon,
                "clean_prediction": int(parent["clean_prediction"]),
                "clean_route": int(parent["clean_route"]),
                "optimized_crown_mu0": alpha,
                "targeted_search": targeted_records,
                "point_values": {
                    "safety": safety.tolist(),
                    "route_margin": margin.tolist(),
                    "dynamic_predictions": dynamic_predictions.cpu().numpy().astype(int).tolist(),
                    "dynamic_routes": dynamic_routes.cpu().numpy().astype(int).tolist(),
                    "route_consistent_static_violations": int(np.sum(route_consistent_violation)),
                    "full_forward_unsafe_witnesses": int(
                        np.sum(
                            route_consistent_violation
                            & (dynamic_predictions.cpu().numpy() != int(parent["clean_prediction"]))
                            & (dynamic_routes.cpu().numpy() == route)
                        )
                    ),
                },
                "finite_point_dual_upper": dual_upper,
                "classification": classification,
            }
            records.append(record)
            all_points.append(points)
            _append_json(handle, record)
            model.cpu()
            router.cpu()
            specialized[route].cpu()
            _cleanup_cuda()

    points_path = output_dir / "diagnostic_points.npz"
    with points_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            points=np.stack(all_points, axis=0),
            row_ids=np.asarray([row["row_id"] for row in records]),
            routes=np.asarray([row["route"] for row in records], dtype=np.int64),
            property_indices=np.asarray(
                [row["property_index"] for row in records], dtype=np.int64
            ),
        )
        handle.flush()
        os.fsync(handle.fileno())
    counts: dict[str, int] = {}
    for row in records:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    result = {
        "schema_version": 1,
        "status": "COMPLETED_NOT_INDEPENDENTLY_AUDITED",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "rows": {"path": str(rows_path), "sha256": _sha256(rows_path), "count": len(records)},
        "diagnostic_points": {
            "path": str(points_path),
            "sha256": _sha256(points_path),
            "shape": list(np.stack(all_points, axis=0).shape),
        },
        "obligations": obligations,
        "classifications": counts,
        "optimized_crown_configuration": crown_configuration,
        "runtime_seconds": time.monotonic() - started,
        "gpu": {"free_gib_before": free / 1024**3, "total_gib": total / 1024**3},
        "claim_scope": (
            "five deterministic development blockers; executed-float attribution "
            "diagnostic, not prevalence and not formal SAFE"
        ),
    }
    _write_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.config)
    print(json.dumps(result["classifications"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
