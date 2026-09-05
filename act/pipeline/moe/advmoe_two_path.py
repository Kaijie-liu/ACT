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

from act.back_end.moe.tie_safe_implication import (
    LagrangianTop1GuardedProperty,
    TieSafeTop1Implication,
)
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


def resolve_selection_manifest(
    config: dict[str, Any],
    workspace: Path,
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    checkpoint_sha256: str,
    dataset_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve an explicit clean-correct cohort and bind its exclusions.

    Schema-v1 frozen configurations retain their historical first-N fallback.
    Every new schema-v2 experiment must provide a hashed selection manifest.
    """

    manifest_value = config.get("selection_manifest")
    if manifest_value is None:
        if int(config.get("schema_version", 1)) >= 2:
            raise ValueError("schema-v2 execution requires a selection manifest")
        eligible = np.flatnonzero(predictions == labels)
        samples = int(config["selection"]["samples"])
        if len(eligible) < samples:
            raise RuntimeError("insufficient clean-correct samples")
        return eligible[:samples], {
            "mode": "LEGACY_FIRST_N_CLEAN_CORRECT",
            "clean_correct_ranks": list(range(samples)),
            "development_exclusion_dataset_indices": [],
        }

    manifest_path = _inside(Path(manifest_value), workspace)
    expected_manifest_hash = str(config.get("selection_manifest_sha256", ""))
    observed_manifest_hash = _sha256(manifest_path)
    if observed_manifest_hash != expected_manifest_hash:
        raise RuntimeError("selection manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_NOT_RUN":
        raise RuntimeError("selection manifest is not frozen")
    if manifest.get("dataset_archive_sha256") != dataset_sha256:
        raise RuntimeError("selection manifest dataset hash mismatch")
    if manifest.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("selection manifest checkpoint hash mismatch")

    ranks = [int(value) for value in manifest.get("clean_correct_ranks", [])]
    indices = np.asarray(manifest.get("ordered_dataset_indices", []), dtype=np.int64)
    if not ranks or len(ranks) != len(indices):
        raise ValueError("selection manifest ranks/indices are empty or mismatched")
    if len(set(ranks)) != len(ranks) or any(value < 0 for value in ranks):
        raise ValueError("selection manifest clean-correct ranks must be unique")
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("selection manifest dataset indices must be unique")
    eligible = np.flatnonzero(predictions == labels)
    if max(ranks) >= len(eligible):
        raise ValueError("selection manifest rank exceeds clean-correct population")
    expected_indices = eligible[np.asarray(ranks, dtype=np.int64)]
    if not np.array_equal(indices, expected_indices):
        raise RuntimeError("selection manifest indices do not match frozen ranks")
    if int(config["selection"]["samples"]) != len(indices):
        raise RuntimeError("selection manifest count differs from configuration")

    exclusion = manifest.get("development_exclusion", {})
    excluded_indices = {
        int(value) for value in exclusion.get("ordered_dataset_indices", [])
    }
    exclusion_sources = exclusion.get("sources", [])
    for source in exclusion_sources:
        source_path = _inside(Path(source["path"]), workspace)
        if _sha256(source_path) != source["sha256"]:
            raise RuntimeError("development-exclusion source hash mismatch")
    overlap = sorted(excluded_indices.intersection(indices.tolist()))
    if overlap:
        raise RuntimeError("selection manifest overlaps development exclusion")
    return indices, {
        "mode": "FROZEN_SELECTION_MANIFEST",
        "manifest_path": str(manifest_path),
        "manifest_sha256": observed_manifest_hash,
        "clean_correct_ranks": ranks,
        "development_exclusion_dataset_indices": sorted(excluded_indices),
        "development_exclusion_sources": exclusion_sources,
    }


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
    lagrangian_statuses: list[str] | None = None,
    graph_matched_mu0_statuses: list[str] | None = None,
    separate_interval_statuses: list[str] | None = None,
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
    lagrangian_statuses = lagrangian_statuses or []
    lagrangian = (
        "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
        if len(lagrangian_statuses) == 2
        and all(value == filtered for value in lagrangian_statuses)
        else "UNKNOWN"
    )
    graph_matched_mu0_statuses = graph_matched_mu0_statuses or []
    graph_matched_mu0 = (
        "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
        if len(graph_matched_mu0_statuses) == 2
        and all(value == filtered for value in graph_matched_mu0_statuses)
        else "UNKNOWN"
    )
    separate_interval_statuses = separate_interval_statuses or []
    separate_interval = (
        "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
        if len(separate_interval_statuses) == 2
        and all(value == filtered for value in separate_interval_statuses)
        else "UNKNOWN"
    )
    portfolio = (
        "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
        if any(
            value == "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
            for value in (route_invariance, route_a, eta, lagrangian)
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
        "lagrangian_guard_ablation": lagrangian,
        "lagrangian_mu0_graph_matched": graph_matched_mu0,
        "lagrangian_separate_interval": separate_interval,
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
            "lagrangian_guard_ablation",
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


def aggregate_lagrangian_grid_calls(
    calls: list[dict[str, Any]],
    multipliers: list[float],
    *,
    property_rows: int,
    tolerance: float,
) -> dict[str, Any]:
    """Select the strongest row-wise bound from a frozen multiplier grid."""

    if len(calls) != len(multipliers) or not calls:
        raise ValueError("one non-empty CROWN call is required per multiplier")
    if property_rows < 1:
        raise ValueError("property_rows must be positive")
    if any(not math.isfinite(value) or value < 0.0 for value in multipliers):
        raise ValueError("lagrangian multipliers must be finite and nonnegative")
    if any(call.get("status") == "ERROR" for call in calls):
        return {
            "status": "ERROR",
            "complete": False,
            "lower_bounds": [],
            "selected_multipliers": [],
            "minimum_lower_bound": None,
            "calls": calls,
        }
    if any(not bool(call.get("complete")) for call in calls):
        return {
            "status": "UNKNOWN_INCOMPLETE",
            "complete": False,
            "lower_bounds": [],
            "selected_multipliers": [],
            "minimum_lower_bound": None,
            "calls": calls,
        }
    lower = np.asarray([call.get("lower_bounds", []) for call in calls], dtype=np.float64)
    if lower.shape != (len(calls), int(property_rows)) or not np.isfinite(lower).all():
        return {
            "status": "UNKNOWN_INCOMPLETE",
            "complete": False,
            "lower_bounds": [],
            "selected_multipliers": [],
            "minimum_lower_bound": None,
            "calls": calls,
        }
    selected = np.argmax(lower, axis=0)
    best = lower[selected, np.arange(property_rows)]
    status = (
        "CERTIFIED_MARGIN_FILTER"
        if bool(np.all(best >= float(tolerance)))
        else "UNKNOWN_RELAXATION"
    )
    return {
        "status": status,
        "complete": True,
        "lower_bounds": best.tolist(),
        "selected_multipliers": [float(multipliers[index]) for index in selected],
        "minimum_lower_bound": float(best.min()),
        "calls": calls,
        "selection_semantics": "row-wise maximum over frozen nonnegative multiplier grid",
        "negative_bound_semantics": "UNKNOWN_NEVER_UNSAFE",
        "numerical_soundness_status": "POSITIVE_MARGIN_FILTER_NOT_OUTWARD_ROUNDED",
    }


def lagrangian_mu0_call(branch: dict[str, Any]) -> dict[str, Any]:
    """Return the unique graph-matched mu=0 call from one frozen grid."""

    matches = [
        call
        for call in branch.get("calls", [])
        if float(call.get("lagrangian_multiplier", math.nan)) == 0.0
    ]
    if len(matches) != 1:
        return {
            "status": "UNKNOWN_INCOMPLETE",
            "complete": False,
            "lower_bounds": [],
            "upper_bounds": [],
            "minimum_lower_bound": None,
            "error": "frozen multiplier grid does not contain exactly one mu=0",
        }
    return matches[0]


def validate_lagrangian_multiplier_protocol(
    lagrangian_config: dict[str, Any],
) -> dict[str, Any]:
    """Validate a frozen raw or router-scale-normalized multiplier grid."""

    multipliers = [
        float(value) for value in lagrangian_config.get("multipliers", [])
    ]
    if not multipliers:
        raise ValueError("enabled Lagrangian guard ablation needs multipliers")
    if any(not math.isfinite(value) or value < 0.0 for value in multipliers):
        raise ValueError("Lagrangian guard multipliers must be finite and nonnegative")
    if multipliers.count(0.0) != 1:
        raise ValueError("Lagrangian grid must contain exactly one graph-matched mu=0")

    normalization = lagrangian_config.get(
        "scale_normalization", {"rule": "NONE_RAW_GRID"}
    )
    rule = normalization.get("rule")
    if rule == "NONE_RAW_GRID":
        return {
            "rule": rule,
            "resolved_multipliers": multipliers,
        }
    if rule != "DEVELOPMENT_MEDIAN_CLEAN_ABS_ROUTER_MARGIN":
        raise ValueError("unsupported Lagrangian multiplier normalization rule")
    scale = float(normalization.get("scale", math.nan))
    coefficients = [
        float(value) for value in normalization.get("normalized_coefficients", [])
    ]
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Lagrangian router-margin scale must be finite and positive")
    if len(coefficients) != len(multipliers) or any(
        not math.isfinite(value) or value < 0.0 for value in coefficients
    ):
        raise ValueError("normalized multiplier coefficients are malformed")
    expected = [value / scale for value in coefficients]
    if any(
        not math.isclose(observed, target, rel_tol=1e-12, abs_tol=1e-15)
        for observed, target in zip(multipliers, expected)
    ):
        raise ValueError("resolved multipliers do not match the frozen scale")
    if not normalization.get("development_source_sha256"):
        raise ValueError("normalized multiplier grid lacks development provenance")
    return {
        "rule": rule,
        "scale": scale,
        "normalized_coefficients": coefficients,
        "resolved_multipliers": multipliers,
        "development_source": normalization.get("development_source"),
        "development_source_sha256": normalization["development_source_sha256"],
    }


def separate_interval_lagrangian_grid(
    path_bound: dict[str, Any],
    *,
    margin_lower: float,
    margin_upper: float,
    multipliers: list[float],
    tolerance: float,
) -> dict[str, Any]:
    """Combine separately bounded safety and router-margin intervals."""

    if any(not math.isfinite(value) or value < 0.0 for value in multipliers):
        raise ValueError("lagrangian multipliers must be finite and nonnegative")
    if not multipliers:
        raise ValueError("separate-interval control needs a multiplier grid")
    if (
        path_bound.get("status") == "ERROR"
        or not bool(path_bound.get("complete"))
        or not math.isfinite(float(margin_lower))
        or not math.isfinite(float(margin_upper))
        or float(margin_lower) > float(margin_upper)
    ):
        return {
            "status": "UNKNOWN_INCOMPLETE",
            "complete": False,
            "lower_bounds": [],
            "selected_multipliers": [],
            "minimum_lower_bound": None,
            "grid_lower_bounds": [],
        }
    safety_lower = np.asarray(path_bound.get("lower_bounds", []), dtype=np.float64)
    if safety_lower.ndim != 1 or not safety_lower.size or not np.isfinite(
        safety_lower
    ).all():
        return {
            "status": "UNKNOWN_INCOMPLETE",
            "complete": False,
            "lower_bounds": [],
            "selected_multipliers": [],
            "minimum_lower_bound": None,
            "grid_lower_bounds": [],
        }
    grid = np.stack(
        [safety_lower - float(multiplier) * float(margin_upper)
         for multiplier in multipliers],
        axis=0,
    )
    selected = np.argmax(grid, axis=0)
    best = grid[selected, np.arange(safety_lower.size)]
    status = (
        "CERTIFIED_MARGIN_FILTER"
        if bool(np.all(best >= float(tolerance)))
        else "UNKNOWN_RELAXATION"
    )
    return {
        "status": status,
        "complete": True,
        "lower_bounds": best.tolist(),
        "selected_multipliers": [float(multipliers[index]) for index in selected],
        "minimum_lower_bound": float(best.min()),
        "grid_lower_bounds": grid.tolist(),
        "margin_interval": [float(margin_lower), float(margin_upper)],
        "formula": "lower(safety)-mu*upper(selected_margin)",
        "numerical_soundness_status": "COMPOSED_NUMERICAL_FILTER_NOT_OUTWARD_ROUNDED",
    }


def comparison_budget_ledger(
    *,
    clean_route: int,
    router_bound: dict[str, Any],
    path_bounds: list[dict[str, Any]],
    eta_bounds: list[dict[str, Any]],
    lagrangian_bounds: list[dict[str, Any]],
    statuses: dict[str, str],
    total_wall_budget_seconds: float,
) -> dict[str, Any]:
    """Apply a common evidence cutoff to fully completed method executions."""

    budget = float(total_wall_budget_seconds)
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError("comparison wall budget must be finite and positive")

    def call_seconds(record: dict[str, Any]) -> float:
        value = float(record.get("accounted_wall_seconds", math.nan))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("bound call lacks finite accounted wall time")
        return value

    mu0_calls = [lagrangian_mu0_call(branch) for branch in lagrangian_bounds]
    method_costs = {
        "route_invariance": (
            call_seconds(router_bound) + call_seconds(path_bounds[int(clean_route)])
        ),
        "unguarded_two_path": sum(call_seconds(value) for value in path_bounds),
        "eta_guard": sum(call_seconds(value) for value in eta_bounds),
        "lagrangian_mu0_graph_matched": sum(
            call_seconds(value) for value in mu0_calls
        ),
        "lagrangian_grid": sum(
            call_seconds(call)
            for branch in lagrangian_bounds
            for call in branch.get("calls", [])
        ),
        "lagrangian_separate_interval": (
            call_seconds(router_bound)
            + sum(call_seconds(value) for value in path_bounds)
        ),
    }
    status_names = {
        "route_invariance": "route_invariance",
        "unguarded_two_path": "route_a_two_path",
        "eta_guard": "eta_guard_ablation",
        "lagrangian_mu0_graph_matched": "lagrangian_mu0_graph_matched",
        "lagrangian_grid": "lagrangian_guard_ablation",
        "lagrangian_separate_interval": "lagrangian_separate_interval",
    }
    methods: dict[str, Any] = {}
    for method, cost in method_costs.items():
        within = bool(cost <= budget)
        mechanism_status = statuses[status_names[method]]
        methods[method] = {
            "accounted_wall_seconds": float(cost),
            "within_budget": within,
            "mechanism_status": mechanism_status,
            "budget_status": (
                mechanism_status if within else "UNKNOWN_BUDGET_EXHAUSTED"
            ),
        }
    return {
        "total_wall_budget_seconds": budget,
        "acceptance_semantics": (
            "A method contributes only if every required call completed and its "
            "sum of call-level graph/build/solve orchestration wall time is within "
            "the common cutoff. Overshooting results are retained but excluded."
        ),
        "attack_time_included": False,
        "methods": methods,
    }


def evaluate_lagrangian_guard_grid(
    router: torch.nn.Module,
    expert: torch.nn.Module,
    expert_index: int,
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    property_rows: tuple[tuple[np.ndarray, float], ...],
    multipliers: list[float],
    device: str,
    tolerance: float,
    method: str,
    track_gradients: bool,
    bound_options: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run a shared-input Lagrangian guard compiler over a frozen grid."""

    branch_started = time.monotonic()
    matrix = np.stack([row for row, _offset in property_rows])
    offset = np.asarray([value for _row, value in property_rows])
    # This runner is intentionally scoped to the audited AdvMoE E=2 model.
    # The compiler itself supports arbitrary E; the official-scale experiment
    # has exactly one outside expert for each hard-top1 branch.
    outside_experts = 1
    calls: list[dict[str, Any]] = []
    for multiplier in multipliers:
        call_started = time.monotonic()
        compiled = LagrangianTop1GuardedProperty(
            copy.deepcopy(router),
            copy.deepcopy(expert),
            int(expert_index),
            matrix,
            offset,
            np.full((len(property_rows), outside_experts), float(multiplier)),
        )
        call = _crown_bounds(
            compiled,
            center,
            lower,
            upper,
            property_rows=None,
            device=device,
            tolerance=tolerance,
            method=method,
            track_gradients=track_gradients,
            bound_options=bound_options,
        )
        call["lagrangian_multiplier"] = float(multiplier)
        _cleanup_cuda()
        call["accounted_wall_seconds"] = time.monotonic() - call_started
        calls.append(call)
    result = aggregate_lagrangian_grid_calls(
        calls,
        multipliers,
        property_rows=len(property_rows),
        tolerance=tolerance,
    )
    result["accounted_wall_seconds"] = time.monotonic() - branch_started
    return result


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
    lagrangian_config = config.get("lagrangian_guard_ablation", {"enabled": False})
    lagrangian_enabled = bool(lagrangian_config.get("enabled", False))
    lagrangian_multipliers = []
    multiplier_protocol = {"rule": "DISABLED", "resolved_multipliers": []}
    if lagrangian_enabled:
        multiplier_protocol = validate_lagrangian_multiplier_protocol(
            lagrangian_config
        )
        lagrangian_multipliers = multiplier_protocol["resolved_multipliers"]
        if multiplier_protocol["rule"] != "NONE_RAW_GRID":
            scale_source = _inside(
                Path(multiplier_protocol["development_source"]), workspace
            )
            if _sha256(scale_source) != multiplier_protocol["development_source_sha256"]:
                raise RuntimeError("Lagrangian development-scale source hash mismatch")
    comparison_config = config.get("comparison", {"enabled": False})
    comparison_enabled = bool(comparison_config.get("enabled", False))
    if comparison_enabled:
        if not lagrangian_enabled or not bool(config["guard_ablation"]["enabled"]):
            raise ValueError("comparison requires eta and Lagrangian methods")
        if comparison_config.get("execution_order") != (
            "SAMPLE_RADIUS_BRANCH_BLOCKED_METHOD_INTERLEAVED"
        ):
            raise ValueError("unsupported comparison execution order")
        comparison_budget = float(
            comparison_config["total_wall_budget_seconds_per_sample_radius_method"]
        )
        if not math.isfinite(comparison_budget) or comparison_budget <= 0.0:
            raise ValueError("comparison wall budget must be finite and positive")
    else:
        comparison_budget = math.inf
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
    indices, selection_identity = resolve_selection_manifest(
        config,
        workspace,
        predictions,
        labels,
        checkpoint_sha256=_sha256(checkpoint),
        dataset_sha256=_sha256(archive),
    )
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
                call_started = time.monotonic()
                route_bound = _crown_bounds(
                    copy.deepcopy(router_adapter), center, lower, upper,
                    property_rows=((route_row, 0.0),),
                    device=config["crown"]["device"], tolerance=tolerance,
                    method=config["crown"]["method"],
                    track_gradients=crown_gradient_tracking,
                    bound_options=crown_bound_options,
                )
                _cleanup_cuda()
                route_bound["accounted_wall_seconds"] = (
                    time.monotonic() - call_started
                )
                path_bounds = []
                eta_bounds = []
                lagrangian_bounds = []
                for route in (0, 1):
                    call_started = time.monotonic()
                    bound = _crown_bounds(
                        copy.deepcopy(path_adapters[route]), center, lower, upper,
                        property_rows=properties,
                        device=config["crown"]["device"], tolerance=tolerance,
                        method=config["crown"]["method"],
                        track_gradients=crown_gradient_tracking,
                        bound_options=crown_bound_options,
                    )
                    _cleanup_cuda()
                    bound["accounted_wall_seconds"] = (
                        time.monotonic() - call_started
                    )
                    path_bounds.append(bound)
                    if config["guard_ablation"]["enabled"]:
                        call_started = time.monotonic()
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
                        _cleanup_cuda()
                        eta_bound["accounted_wall_seconds"] = (
                            time.monotonic() - call_started
                        )
                        eta_bounds.append(eta_bound)
                    if lagrangian_enabled:
                        lagrangian_bounds.append(
                            evaluate_lagrangian_guard_grid(
                                router_adapter,
                                path_adapters[route],
                                route,
                                center,
                                lower,
                                upper,
                                property_rows=properties,
                                multipliers=lagrangian_multipliers,
                                device=config["crown"]["device"],
                                tolerance=tolerance,
                                method=config["crown"]["method"],
                                track_gradients=crown_gradient_tracking,
                                bound_options=crown_bound_options,
                            )
                        )
                graph_matched_mu0_bounds = [
                    lagrangian_mu0_call(branch) for branch in lagrangian_bounds
                ]
                separate_interval_bounds = []
                if lagrangian_enabled:
                    route_lower = np.asarray(
                        route_bound.get("lower_bounds", []), dtype=np.float64
                    )
                    route_upper = np.asarray(
                        route_bound.get("upper_bounds", []), dtype=np.float64
                    )
                    route_interval_complete = bool(
                        route_bound.get("complete")
                        and route_lower.shape == (1,)
                        and route_upper.shape == (1,)
                        and np.isfinite(route_lower).all()
                        and np.isfinite(route_upper).all()
                    )
                    for route in (0, 1):
                        if route_interval_complete:
                            if route == clean_route:
                                margin_lower = float(route_lower[0])
                                margin_upper = float(route_upper[0])
                            else:
                                margin_lower = -float(route_upper[0])
                                margin_upper = -float(route_lower[0])
                        else:
                            margin_lower = math.nan
                            margin_upper = math.nan
                        separate_interval_bounds.append(
                            separate_interval_lagrangian_grid(
                                path_bounds[route],
                                margin_lower=margin_lower,
                                margin_upper=margin_upper,
                                multipliers=lagrangian_multipliers,
                                tolerance=tolerance,
                            )
                        )
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
                    lagrangian_statuses=[row["status"] for row in lagrangian_bounds],
                    graph_matched_mu0_statuses=[
                        row["status"] for row in graph_matched_mu0_bounds
                    ],
                    separate_interval_statuses=[
                        row["status"] for row in separate_interval_bounds
                    ],
                    attack_prediction_flip=bool(attack["prediction_flip"]),
                )
                comparison = (
                    comparison_budget_ledger(
                        clean_route=clean_route,
                        router_bound=route_bound,
                        path_bounds=path_bounds,
                        eta_bounds=eta_bounds,
                        lagrangian_bounds=lagrangian_bounds,
                        statuses=statuses,
                        total_wall_budget_seconds=comparison_budget,
                    )
                    if comparison_enabled
                    else None
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
                    "lagrangian_guard_crown": lagrangian_bounds,
                    "lagrangian_mu0_graph_matched_crown": (
                        graph_matched_mu0_bounds
                    ),
                    "lagrangian_separate_interval": separate_interval_bounds,
                    "comparison": comparison,
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
        or any(value["status"] == "ERROR" for value in row["lagrangian_guard_crown"])
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
            "lagrangian_guard_ablation": dict(Counter(row["statuses"]["lagrangian_guard_ablation"] for row in selected)),
            "lagrangian_mu0_graph_matched": dict(Counter(row["statuses"]["lagrangian_mu0_graph_matched"] for row in selected)),
            "lagrangian_separate_interval": dict(Counter(row["statuses"]["lagrangian_separate_interval"] for row in selected)),
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
        "selection": {
            "dataset_indices": indices.tolist(),
            "clean_correct": True,
            **selection_identity,
        },
        "equivalence": {
            "router": router_equivalence,
            "paths": path_equivalence,
            "dynamic_selected_max_abs_error": dynamic_selected_error,
        },
        "rows": {"path": str(rows_path), "sha256": _sha256(rows_path), "count": len(row_records)},
        "attack_endpoints": {"path": str(attack_path), "sha256": _sha256(attack_path)},
        "tables": tables,
        "backend_configuration": crown_configuration,
        "lagrangian_multiplier_protocol": multiplier_protocol,
        "comparison_configuration": comparison_config,
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
