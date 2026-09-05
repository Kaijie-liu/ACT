"""Five-radius staged verification for the accepted AdvMoE compatibility model."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from act.back_end.moe.tie_safe_implication import TieSafeTop1Implication
from act.pipeline.moe.advmoe_adapter import (
    CrownCompatibleAdvMoePath,
    CrownCompatibleAdvMoeRouter,
    adapter_equivalence,
    construct_official_init,
    path_adapter_equivalence,
    specialize_advmoe_path,
)
from act.pipeline.moe.advmoe_router_bracket import load_cifar10_test_archive
from act.pipeline.moe.crown_adapter_cohort import (
    _crown_bounds,
    validate_crown_configuration,
)
from act.pipeline.moe.published_moe_router_gradient_audit import _sha256


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True
    ).strip()


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _append_json(handle, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def top1_property_rows(prediction: int, classes: int = 10):
    rows = []
    for competitor in range(classes):
        if competitor == int(prediction):
            continue
        row = np.zeros(classes, dtype=np.float64)
        row[int(prediction)] = 1.0
        row[competitor] = -1.0
        rows.append((row, 0.0))
    return tuple(rows)


def aggregate_filters(
    *,
    clean_route: int,
    router_status: str,
    path_statuses: list[str],
    eta_statuses: list[str],
    attack_prediction_flip: bool,
) -> dict[str, str]:
    filtered = "CERTIFIED_MARGIN_FILTER"
    route_invariance = (
        "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
        if router_status == filtered and path_statuses[int(clean_route)] == filtered
        else "UNKNOWN"
    )
    route_a = (
        "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
        if len(path_statuses) == 2 and all(value == filtered for value in path_statuses)
        else "UNKNOWN"
    )
    eta = (
        "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
        if len(eta_statuses) == 2 and all(value == filtered for value in eta_statuses)
        else "UNKNOWN"
    )
    portfolio = (
        "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
        if any(
            value == "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
            for value in (route_invariance, route_a, eta)
        )
        else "UNKNOWN"
    )
    if attack_prediction_flip:
        endpoint = "UNSAFE_FULL_FORWARD_REPLAY"
    elif portfolio == "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE":
        endpoint = portfolio
    else:
        endpoint = "UNKNOWN"
    return {
        "route_invariance": route_invariance,
        "route_a_two_path": route_a,
        "eta_guard_ablation": eta,
        "portfolio": portfolio,
        "endpoint": endpoint,
    }


def filter_witness_conflicts(
    *,
    router_status: str,
    statuses: dict[str, str],
    prediction_flip: bool,
    route_flip: bool,
) -> dict[str, bool]:
    """Return every concrete-witness contradiction checked by schema v2.

    CROWN margins in this runner are numerical filters rather than formal SAFE
    results.  They must nevertheless agree with replayed concrete witnesses.
    Keeping router and output contradictions separate prevents either kind of
    failure from being hidden behind a single runner-supplied boolean.
    """

    positive = "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
    output_filter_positive = any(
        statuses.get(name) == positive
        for name in (
            "route_invariance",
            "route_a_two_path",
            "eta_guard_ablation",
            "portfolio",
        )
    )
    router_filter_route_conflict = bool(route_flip) and (
        router_status == "CERTIFIED_MARGIN_FILTER"
    )
    output_filter_prediction_conflict = bool(prediction_flip) and output_filter_positive
    return {
        "router_filter_route_conflict": router_filter_route_conflict,
        "output_filter_prediction_conflict": output_filter_prediction_conflict,
        "any": router_filter_route_conflict or output_filter_prediction_conflict,
    }


def evaluate_path_lowering_equivalence(
    specialized_paths: list[torch.nn.Module],
    inputs: torch.Tensor,
    *,
    absolute_tolerance: float,
) -> list[dict[str, Any]]:
    """Apply the registered absolute tolerance to every static-path lowering."""

    return [
        path_adapter_equivalence(
            path,
            inputs,
            atol=float(absolute_tolerance),
            rtol=0.0,
        )
        for path in specialized_paths
    ]


def select_batched_static_path_logits(
    specialized_paths: list[torch.nn.Module],
    inputs: torch.Tensor,
    routes: torch.Tensor,
) -> torch.Tensor:
    """Evaluate both paths with the dynamic-forward batch shape, then select."""

    if len(specialized_paths) != 2:
        raise ValueError("AdvMoE requires exactly two static paths")
    if routes.ndim != 1 or routes.shape[0] != inputs.shape[0]:
        raise ValueError("one route is required for every input")
    all_logits = torch.stack([path(inputs) for path in specialized_paths], dim=1)
    slots = torch.arange(inputs.shape[0], device=all_logits.device)
    return all_logits[slots, routes.to(device=all_logits.device, dtype=torch.long)]


def _cleanup_cuda() -> None:
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_model(config: dict[str, Any], workspace: Path):
    checkpoint = _inside(Path(config["checkpoint"]["path"]), workspace)
    if _sha256(checkpoint) != config["checkpoint"]["sha256"]:
        raise RuntimeError("checkpoint hash mismatch")
    model, router, moe_type = construct_official_init(
        int(config["checkpoint"]["seed"])
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload["epoch"]) != int(config["checkpoint"]["epoch"]):
        raise RuntimeError("checkpoint epoch mismatch")
    model.load_state_dict(payload["state_dict"])
    router.load_state_dict(payload["router"])
    model.router = router
    return model.eval(), router.eval(), moe_type, checkpoint


def _predict(model, inputs: np.ndarray, batch_size: int) -> np.ndarray:
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(inputs[start : start + batch_size])
            rows.append(model(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(rows).astype(np.int64)


def _prediction_attack(
    model,
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    prediction: int,
    *,
    steps: int,
    restarts: int,
    step_size: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    model = model.to(device).eval()
    center = center.to(device)
    lower = lower.to(device)
    upper = upper.to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    target = torch.tensor([int(prediction)], device=device)
    best_loss = -math.inf
    best = center.detach().clone()
    for _restart in range(int(restarts)):
        random = torch.rand(
            center.shape, generator=generator, device=device, dtype=center.dtype
        )
        candidate = (lower + random * (upper - lower)).detach()
        for _step in range(int(steps)):
            candidate.requires_grad_(True)
            loss = F.cross_entropy(model(candidate), target)
            gradient = torch.autograd.grad(loss, candidate)[0]
            candidate = torch.maximum(
                lower,
                torch.minimum(upper, candidate.detach() + step_size * gradient.sign()),
            )
            with torch.no_grad():
                value = float(F.cross_entropy(model(candidate), target).item())
                if value > best_loss:
                    best_loss = value
                    best = candidate.detach().clone()
    with torch.no_grad():
        logits = model(best)
        attacked_prediction = int(logits.argmax(dim=1).item())
        attacked_route = int(model.router(best).argmax(dim=1).item())
    endpoint = best.detach().cpu()
    return {
        "prediction_flip": attacked_prediction != int(prediction),
        "attacked_prediction": attacked_prediction,
        "attacked_route": attacked_route,
        "cross_entropy": best_loss,
        "maximum_linf": float((endpoint - center.detach().cpu()).abs().max().item()),
        "endpoint": endpoint.numpy(),
    }


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    workspace = Path(config["workspace_boundary"])
    act_repository = _inside(Path(config["act_repository"]), workspace)
    source = _inside(Path(config["official_source"]["repository"]), workspace)
    output_dir = _inside(Path(config["output_dir"]), workspace)
    archive = _inside(Path(config["dataset_archive"]), workspace)
    config_path = _inside(config_path, workspace)
    if config.get("status") != "PREREGISTERED_NOT_RUN":
        raise RuntimeError("config is not frozen")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if _git(act_repository, "branch", "--show-current") != config["required_branch"]:
        raise RuntimeError("ACT branch gate failed")
    if _git(act_repository, "status", "--porcelain=v1"):
        raise RuntimeError("ACT worktree is dirty")
    if _git(source, "rev-parse", "HEAD") != config["official_source"]["commit"]:
        raise RuntimeError("official source commit mismatch")
    if _git(source, "rev-parse", "HEAD^{tree}") != config["official_source"]["tree"]:
        raise RuntimeError("official source tree mismatch")
    if _git(source, "status", "--porcelain=v1"):
        raise RuntimeError("official source clone is dirty")
    if _sha256(archive) != config["dataset_archive_sha256"]:
        raise RuntimeError("dataset archive hash mismatch")
    crown_gradient_tracking = bool(config["crown"].get("gradient_tracking", True))
    crown_bound_options = config["crown"].get("bound_options")
    crown_configuration = validate_crown_configuration(
        method=config["crown"]["method"],
        track_gradients=crown_gradient_tracking,
        bound_options=crown_bound_options,
    )
    for gate in config["accepted_audits"]:
        path = _inside(Path(gate["path"]), workspace)
        if _sha256(path) != gate["sha256"]:
            raise RuntimeError(f"accepted audit hash mismatch: {gate['label']}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "PASS" or value.get("issues") != []:
            raise RuntimeError(f"accepted audit gate failed: {gate['label']}")
    free, total = torch.cuda.mem_get_info(config["crown"]["device"])
    if free / 1024**3 < float(config["crown"]["minimum_free_gpu_memory_gib"]):
        raise RuntimeError("GPU memory gate failed")

    torch.set_num_threads(int(config["torch_threads"]))
    torch.set_num_interop_threads(1)
    inputs, labels = load_cifar10_test_archive(archive)
    model, router, moe_type, checkpoint = _load_model(config, workspace)
    predictions = _predict(model, inputs, int(config["batch_size"]))
    eligible = np.flatnonzero(predictions == labels)
    samples = int(config["selection"]["samples"])
    if len(eligible) < samples:
        raise RuntimeError("insufficient clean-correct samples")
    indices = eligible[:samples]
    centers = torch.from_numpy(inputs[indices])
    with torch.no_grad():
        clean_routes = router(centers).argmax(dim=1).numpy().astype(np.int64)

    specialized = [
        specialize_advmoe_path(model, route, moe_type)[0].eval()
        for route in (0, 1)
    ]
    path_adapters = [CrownCompatibleAdvMoePath(path).eval() for path in specialized]
    router_adapter = CrownCompatibleAdvMoeRouter(router).eval()
    router_equivalence = adapter_equivalence(router, centers)
    path_equivalence = evaluate_path_lowering_equivalence(
        specialized,
        centers,
        absolute_tolerance=float(config["numerical"]["equivalence_atol"]),
    )
    if not router_equivalence["outputs_equal"] or not all(
        row["outputs_close"] for row in path_equivalence
    ):
        raise RuntimeError("CROWN lowering equivalence gate failed")
    with torch.no_grad():
        dynamic_logits = model(centers)
        selected_logits = select_batched_static_path_logits(
            specialized,
            centers,
            torch.from_numpy(clean_routes),
        )
    dynamic_selected_error = float((dynamic_logits - selected_logits).abs().max().item())
    if dynamic_selected_error > float(config["numerical"]["equivalence_atol"]):
        raise RuntimeError("dynamic and selected static paths differ")

    output_dir.mkdir(parents=True)
    rows_path = output_dir / "rows.jsonl"
    attack_endpoints: list[np.ndarray] = []
    row_records: list[dict[str, Any]] = []
    started = time.monotonic()
    tolerance = float(config["numerical"]["safe_positive_margin"])
    with rows_path.open("x", encoding="utf-8") as rows_handle:
        for slot, dataset_index in enumerate(indices):
            center = centers[slot : slot + 1]
            prediction = int(predictions[dataset_index])
            clean_route = int(clean_routes[slot])
            properties = top1_property_rows(prediction)
            for numerator in config["radii_over_255"]:
                epsilon = float(numerator) / 255.0
                lower = torch.clamp(center - epsilon, 0.0, 1.0)
                upper = torch.clamp(center + epsilon, 0.0, 1.0)
                route_row = np.asarray(
                    [1.0, -1.0] if clean_route == 0 else [-1.0, 1.0],
                    dtype=np.float64,
                )
                route_bound = _crown_bounds(
                    copy.deepcopy(router_adapter), center, lower, upper,
                    property_rows=((route_row, 0.0),),
                    device=config["crown"]["device"], tolerance=tolerance,
                    method=config["crown"]["method"],
                    track_gradients=crown_gradient_tracking,
                    bound_options=crown_bound_options,
                )
                _cleanup_cuda()
                path_bounds = []
                eta_bounds = []
                for route in (0, 1):
                    bound = _crown_bounds(
                        copy.deepcopy(path_adapters[route]), center, lower, upper,
                        property_rows=properties,
                        device=config["crown"]["device"], tolerance=tolerance,
                        method=config["crown"]["method"],
                        track_gradients=crown_gradient_tracking,
                        bound_options=crown_bound_options,
                    )
                    path_bounds.append(bound)
                    _cleanup_cuda()
                    if config["guard_ablation"]["enabled"]:
                        matrix = np.stack([row for row, _offset in properties])
                        offset = np.asarray([offset for _row, offset in properties])
                        implication = TieSafeTop1Implication(
                            copy.deepcopy(router_adapter),
                            copy.deepcopy(path_adapters[route]),
                            route,
                            matrix,
                            offset,
                            eta=tolerance,
                        )
                        eta_bound = _crown_bounds(
                            implication, center, lower, upper,
                            property_rows=None,
                            device=config["crown"]["device"], tolerance=tolerance,
                            method=config["crown"]["method"],
                            track_gradients=crown_gradient_tracking,
                            bound_options=crown_bound_options,
                        )
                        eta_bounds.append(eta_bound)
                        _cleanup_cuda()
                attack = _prediction_attack(
                    model, center, lower, upper, prediction,
                    steps=int(config["attack"]["steps"]),
                    restarts=int(config["attack"]["restarts"]),
                    step_size=epsilon / float(config["attack"]["step_divisor"]),
                    seed=int(config["attack"]["seed"]) + len(row_records),
                    device=config["attack"]["device"],
                )
                attack_endpoints.append(attack.pop("endpoint"))
                statuses = aggregate_filters(
                    clean_route=clean_route,
                    router_status=route_bound["status"],
                    path_statuses=[row["status"] for row in path_bounds],
                    eta_statuses=[row["status"] for row in eta_bounds],
                    attack_prediction_flip=bool(attack["prediction_flip"]),
                )
                route_flip = int(attack["attacked_route"]) != clean_route
                witness_conflicts = filter_witness_conflicts(
                    router_status=route_bound["status"],
                    statuses=statuses,
                    prediction_flip=bool(attack["prediction_flip"]),
                    route_flip=route_flip,
                )
                record = {
                    "row_id": f"sample{slot}:eps{numerator}",
                    "sample_slot": slot,
                    "dataset_index": int(dataset_index),
                    "epsilon_over_255": float(numerator),
                    "epsilon": epsilon,
                    "label": int(labels[dataset_index]),
                    "clean_prediction": prediction,
                    "clean_route": clean_route,
                    "router_crown": route_bound,
                    "path_crown": path_bounds,
                    "eta_crown": eta_bounds,
                    "attack": attack,
                    "statuses": statuses,
                    "filter_witness_conflicts": witness_conflicts,
                }
                row_records.append(record)
                _append_json(rows_handle, record)

    attack_path = output_dir / "attack_endpoints.npz"
    with attack_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            dataset_indices=indices.astype(np.int64),
            centers=centers.numpy(),
            endpoints=np.concatenate(attack_endpoints, axis=0),
            labels=labels[indices].astype(np.int64),
        )
        handle.flush()
        os.fsync(handle.fileno())
    errors = sum(
        row["router_crown"]["status"] == "ERROR"
        or any(value["status"] == "ERROR" for value in row["path_crown"])
        or any(value["status"] == "ERROR" for value in row["eta_crown"])
        for row in row_records
    )
    conflicts = sum(row["filter_witness_conflicts"]["any"] for row in row_records)
    tables = {}
    for numerator in config["radii_over_255"]:
        selected = [
            row for row in row_records
            if row["epsilon_over_255"] == float(numerator)
        ]
        prediction_flips = [bool(row["attack"]["prediction_flip"]) for row in selected]
        route_flips = [
            int(row["attack"]["attacked_route"]) != int(row["clean_route"])
            for row in selected
        ]
        tables[str(numerator)] = {
            "samples": len(selected),
            "route_invariance": dict(Counter(row["statuses"]["route_invariance"] for row in selected)),
            "route_a_two_path": dict(Counter(row["statuses"]["route_a_two_path"] for row in selected)),
            "eta_guard_ablation": dict(Counter(row["statuses"]["eta_guard_ablation"] for row in selected)),
            "portfolio": dict(Counter(row["statuses"]["portfolio"] for row in selected)),
            "endpoint": dict(Counter(row["statuses"]["endpoint"] for row in selected)),
            "prediction_flip_witnesses": sum(prediction_flips),
            "route_flip_witnesses": sum(route_flips),
            "both_flip_witnesses": sum(
                prediction and route
                for prediction, route in zip(prediction_flips, route_flips)
            ),
        }
    result = {
        "schema_version": 2,
        "status": "COMPLETED_NOT_INDEPENDENTLY_AUDITED" if errors == 0 and conflicts == 0 else "FAILED",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "dataset": {"archive": str(archive), "sha256": _sha256(archive)},
        "selection": {"dataset_indices": indices.tolist(), "clean_correct": True},
        "equivalence": {
            "router": router_equivalence,
            "paths": path_equivalence,
            "dynamic_selected_max_abs_error": dynamic_selected_error,
        },
        "rows": {"path": str(rows_path), "sha256": _sha256(rows_path), "count": len(row_records)},
        "attack_endpoints": {"path": str(attack_path), "sha256": _sha256(attack_path)},
        "tables": tables,
        "backend_configuration": crown_configuration,
        "backend_error_rows": errors,
        "filter_witness_conflicts": conflicts,
        "formal_safe_count": 0,
        "numerical_scope": "positive CROWN margins are filters, never formal SAFE, because the backend is not outward rounded",
        "runtime_seconds": time.monotonic() - started,
        "gpu": {"free_gib_before": free / 1024**3, "total_gib": total / 1024**3},
        "official_source_clean_after": not bool(_git(source, "status", "--porcelain=v1")),
    }
    _write_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.config)
    print(json.dumps({"status": result["status"], "tables": result["tables"]}, indent=2, sort_keys=True))
    if result["status"] == "FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
