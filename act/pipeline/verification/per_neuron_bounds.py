#!/usr/bin/env python3
#===- act/pipeline/verification/per_neuron_bounds.py - Per-Neuron Bounds --====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

from act.back_end.analyze import analyze
from act.back_end.verifier import find_entry_layer_id
from act.back_end.transfer_functions import set_transfer_function_mode


@dataclass(frozen=True)
class LayerBounds:
    layer_id: int
    kind: str
    lb: torch.Tensor
    ub: torch.Tensor
    shape: Tuple[int, ...]


def compute_abstract_bounds(
    act_net,
    entry_fact,
    *,
    tf_mode: str = "interval",
) -> Tuple[Dict[int, LayerBounds], List[str]]:
    """
    Compute abstract bounds for all layers in the ACT net.
    """
    errors: List[str] = []
    bounds_by_layer: Dict[int, LayerBounds] = {}

    set_transfer_function_mode(tf_mode)
    entry_id = find_entry_layer_id(act_net)
    _before, after, _globalC = analyze(act_net, entry_id, entry_fact)

    for layer in getattr(act_net, "layers", []):
        lid = layer.id
        if lid not in after:
            errors.append(f"Missing bounds for layer_id={lid} (kind={layer.kind})")
            continue
        fact = after[lid]
        lb = fact.bounds.lb
        ub = fact.bounds.ub
        if lb.shape != ub.shape:
            errors.append(
                f"Bounds shape mismatch at layer_id={lid}: lb={tuple(lb.shape)} ub={tuple(ub.shape)}"
            )
            continue
        if not torch.isfinite(lb).all() or not torch.isfinite(ub).all():
            errors.append(f"Non-finite bounds at layer_id={lid}")
            continue
        bounds_by_layer[lid] = LayerBounds(
            layer_id=lid,
            kind=layer.kind,
            lb=lb,
            ub=ub,
            shape=tuple(lb.shape),
        )

    return bounds_by_layer, errors


@dataclass(frozen=True)
class ActivationEvent:
    name: str
    module_type: str
    call_index: int
    shape: Tuple[int, ...]
    tensor: torch.Tensor


def collect_concrete_activations(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    *,
    strict_single_call_per_module: bool = False,
) -> Tuple[List[ActivationEvent], List[str], List[str]]:
    """
    Collect concrete activations with forward hooks.
    """
    errors: List[str] = []
    warnings: List[str] = []
    events: List[ActivationEvent] = []
    call_counts: Dict[int, int] = {}
    hooks = []

    def _hook(module, inputs, output):
        module_id = id(module)
        call_counts[module_id] = call_counts.get(module_id, 0) + 1
        if strict_single_call_per_module and call_counts[module_id] > 1:
            errors.append(f"Module called multiple times: {module.__class__.__name__}")
        if not torch.is_tensor(output):
            warnings.append(f"Non-tensor output from {module.__class__.__name__}")
            return
        tensor = output.detach()
        shape = tuple(int(x) for x in tensor.shape)
        events.append(
            ActivationEvent(
                name=module.__class__.__name__,
                module_type=module.__class__.__name__,
                call_index=call_counts[module_id] - 1,
                shape=shape,
                tensor=tensor,
            )
        )

    for module in model.modules():
        if module is model:
            continue
        hooks.append(module.register_forward_hook(_hook))

    try:
        with torch.no_grad():
            model(input_tensor)
    finally:
        for h in hooks:
            h.remove()

    return events, errors, warnings


@dataclass(frozen=True)
class AlignmentResult:
    ok: bool
    mapping: Dict[int, ActivationEvent]
    errors: List[str]
    warnings: List[str]
    meta: Dict[str, Any]


_ACT_KIND_TO_MODULE = {
    "DENSE": "Linear",
    "CONV1D": "Conv1d",
    "CONV2D": "Conv2d",
    "CONV3D": "Conv3d",
    "RELU": "ReLU",
    "SIGMOID": "Sigmoid",
    "TANH": "Tanh",
    "SILU": "SiLU",
    "LRELU": "LeakyReLU",
    "FLATTEN": "Flatten",
    "MAXPOOL1D": "MaxPool1d",
    "MAXPOOL2D": "MaxPool2d",
    "MAXPOOL3D": "MaxPool3d",
    "AVGPOOL1D": "AvgPool1d",
    "AVGPOOL2D": "AvgPool2d",
    "AVGPOOL3D": "AvgPool3d",
    "ADAPTIVEAVGPOOL1D": "AdaptiveAvgPool1d",
    "ADAPTIVEAVGPOOL2D": "AdaptiveAvgPool2d",
    "ADAPTIVEAVGPOOL3D": "AdaptiveAvgPool3d",
}


def align_activations_to_act_layers(
    act_net,
    events: List[ActivationEvent],
    *,
    mode: str = "hookable_order_strict",
    hookable_kinds: Tuple[str, ...] = (
        "Linear",
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "ReLU",
        "Sigmoid",
        "Tanh",
        "SiLU",
        "LeakyReLU",
        "Flatten",
        "MaxPool1d",
        "MaxPool2d",
        "MaxPool3d",
        "AvgPool1d",
        "AvgPool2d",
        "AvgPool3d",
        "AdaptiveAvgPool1d",
        "AdaptiveAvgPool2d",
        "AdaptiveAvgPool3d",
    ),
) -> AlignmentResult:
    """
    Align activation events to ACT layer IDs.
    """
    errors: List[str] = []
    warnings: List[str] = []
    mapping: Dict[int, ActivationEvent] = {}

    if mode != "hookable_order_strict":
        return AlignmentResult(
            ok=False,
            mapping={},
            errors=[f"Unsupported alignment mode: {mode}"],
            warnings=[],
            meta={},
        )

    hookable_events = [e for e in events if e.module_type in hookable_kinds]
    hookable_layers = [
        L for L in getattr(act_net, "layers", [])
        if _ACT_KIND_TO_MODULE.get(L.kind) in hookable_kinds
    ]

    if len(hookable_events) != len(hookable_layers):
        errors.append(
            f"Hookable count mismatch: events={len(hookable_events)} layers={len(hookable_layers)}"
        )

    def _numel(shape: Tuple[int, ...]) -> int:
        prod = 1
        for s in shape:
            prod *= int(s)
        return int(prod)

    def _drop_batch_if_and_only_if_batch1(
        raw_shape: Tuple[int, ...],
        expected_shape: Tuple[int, ...] | None,
    ) -> Tuple[Tuple[int, ...], bool, str]:
        """
        Strictly tolerate ONLY batch=1 differences.
        """
        if expected_shape is None:
            return raw_shape, False, "expected_shape_missing"
        if not raw_shape:
            return raw_shape, False, "raw_shape_empty"
        if raw_shape[0] != 1:
            return raw_shape, False, "raw_first_dim_not_1"
        if len(raw_shape) != len(expected_shape) + 1:
            return raw_shape, False, "rank_not_expected_plus_one"
        candidate = tuple(raw_shape[1:])
        if candidate != expected_shape:
            return raw_shape, False, "drop_would_not_match_expected"
        return candidate, True, "dropped_batch1"

    for idx, layer in enumerate(hookable_layers):
        if idx >= len(hookable_events):
            break
        ev = hookable_events[idx]
        expected = _ACT_KIND_TO_MODULE.get(layer.kind)
        if expected != ev.module_type:
            errors.append(
                f"Kind/type mismatch at position {idx}: act_kind={layer.kind} event_type={ev.module_type}"
            )
        expected_shape = None
        meta = getattr(layer, "meta", {}) or {}
        if "output_shape" in meta:
            expected_shape = tuple(int(x) for x in meta["output_shape"])
        elif "shape" in meta:
            expected_shape = tuple(int(x) for x in meta["shape"])
        if expected_shape is not None:
            raw_shape = tuple(ev.shape)
            no_batch_shape, dropped, drop_reason = _drop_batch_if_and_only_if_batch1(
                raw_shape,
                expected_shape,
            )
            ev_numel = _numel(no_batch_shape)
            exp_numel = _numel(expected_shape)
            if ev_numel != exp_numel:
                errors.append(
                    f"Shape mismatch at layer_id={layer.id}: "
                    f"event_raw={raw_shape} event_no_batch={no_batch_shape} "
                    f"expected={expected_shape} "
                    f"dropped_batch={dropped} drop_reason={drop_reason} "
                    f"event_numel={ev_numel} expected_numel={exp_numel}"
                )
        mapping[layer.id] = ev

    ok = len(errors) == 0
    meta = {
        "mode": mode,
        "hookable_events": len(hookable_events),
        "hookable_layers": len(hookable_layers),
    }
    return AlignmentResult(ok=ok, mapping=mapping, errors=errors, warnings=warnings, meta=meta)


def _is_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())


def compare_bounds_per_neuron(
    *,
    bounds_by_layer: Dict[int, LayerBounds],
    concrete_by_layer: Dict[int, torch.Tensor],
    atol: float = 1e-6,
    rtol: float = 0.0,
    topk: int = 10,
    nan_policy: str = "error",
) -> Dict[str, Any]:
    """
    Compare per-neuron concrete activations against abstract bounds.
    """
    errors: List[str] = []
    warnings: List[str] = []
    violations_topk: List[Dict[str, Any]] = []
    layerwise_stats: List[Dict[str, Any]] = []
    violations_total = 0

    if set(bounds_by_layer.keys()) != set(concrete_by_layer.keys()):
        missing = set(bounds_by_layer.keys()) - set(concrete_by_layer.keys())
        extra = set(concrete_by_layer.keys()) - set(bounds_by_layer.keys())
        errors.append(f"Layer key mismatch: missing={sorted(missing)} extra={sorted(extra)}")

    if errors:
        return {
            "status": "ERROR",
            "violations_total": 0,
            "violations_topk": [],
            "layerwise_stats": [],
            "errors": errors,
            "warnings": warnings,
        }

    candidates: List[Dict[str, Any]] = []

    for layer_id, bounds in bounds_by_layer.items():
        concrete = concrete_by_layer[layer_id]
        lb = bounds.lb
        ub = bounds.ub

        if nan_policy == "error":
            if not _is_finite(concrete) or not _is_finite(lb) or not _is_finite(ub):
                errors.append(f"Non-finite value at layer_id={layer_id}")
                continue

        concrete_flat = concrete.reshape(-1)
        lb_flat = lb.reshape(-1)
        ub_flat = ub.reshape(-1)
        if concrete_flat.numel() != lb_flat.numel():
            errors.append(
                f"Shape mismatch at layer_id={layer_id}: "
                f"concrete_numel={concrete_flat.numel()} bounds_numel={lb_flat.numel()}"
            )
            continue
        tol = atol + rtol * concrete_flat.abs()

        diff_low = (lb_flat - tol) - concrete_flat
        diff_high = concrete_flat - (ub_flat + tol)
        gap = torch.maximum(diff_low, diff_high)
        gap = torch.clamp(gap, min=0.0)

        violations_mask = gap > 0
        num_violations = int(violations_mask.sum().item())
        violations_total += num_violations

        if num_violations > 0:
            gap_vals = gap[violations_mask]
            max_gap = float(gap_vals.max().item())
            mean_gap = float(gap_vals.mean().item())
        else:
            max_gap = 0.0
            mean_gap = 0.0

        layerwise_stats.append(
            {
                "layer_id": int(layer_id),
                "kind": bounds.kind,
                "shape": list(bounds.shape),
                "num_neurons": int(concrete_flat.numel()),
                "num_violations": int(num_violations),
                "max_gap": float(max_gap),
                "mean_gap": float(mean_gap),
                "lb_min": float(lb_flat.min().item()) if lb_flat.numel() > 0 else 0.0,
                "lb_max": float(lb_flat.max().item()) if lb_flat.numel() > 0 else 0.0,
                "ub_min": float(ub_flat.min().item()) if ub_flat.numel() > 0 else 0.0,
                "ub_max": float(ub_flat.max().item()) if ub_flat.numel() > 0 else 0.0,
                "concrete_min": float(concrete_flat.min().item()) if concrete_flat.numel() > 0 else 0.0,
                "concrete_max": float(concrete_flat.max().item()) if concrete_flat.numel() > 0 else 0.0,
                "layer_status": "FAIL" if num_violations > 0 else "PASS",
            }
        )

        if topk > 0:
            k = min(int(topk), int(concrete_flat.numel()))
            if k > 0:
                vals, idxs = torch.topk(gap, k=k)
                for v, i in zip(vals.tolist(), idxs.tolist()):
                    if v <= 0:
                        continue
                    i = int(i)
                    candidates.append(
                        {
                            "layer_id": int(layer_id),
                            "kind": bounds.kind,
                            "neuron_index": i,
                            "gap": float(v),
                            "concrete": float(concrete_flat[i].item()),
                            "lb": float(lb_flat[i].item()),
                            "ub": float(ub_flat[i].item()),
                        }
                    )

    if errors:
        return {
            "status": "ERROR",
            "violations_total": 0,
            "violations_topk": [],
            "layerwise_stats": [],
            "errors": errors,
            "warnings": warnings,
        }

    candidates.sort(key=lambda x: x["gap"], reverse=True)
    violations_topk = candidates[: int(topk)]

    status = "FAIL" if violations_total > 0 else "PASS"
    return {
        "status": status,
        "violations_total": int(violations_total),
        "violations_topk": violations_topk,
        "layerwise_stats": layerwise_stats,
        "errors": errors,
        "warnings": warnings,
    }


@dataclass(frozen=True)
class PerNeuronCheckConfig:
    atol: float = 1e-6
    rtol: float = 0.0
    topk: int = 10
    nan_policy: str = "error"


def run_per_neuron_bounds_check(
    *,
    act_net,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    entry_fact,
    tf_mode: str,
    config: PerNeuronCheckConfig,
) -> Dict[str, Any]:
    """
    Full per-neuron bounds validation pipeline for a single input sample.
    """
    errors: List[str] = []
    warnings: List[str] = []

    bounds_by_layer, bounds_errors = compute_abstract_bounds(
        act_net,
        entry_fact,
        tf_mode=tf_mode,
    )
    if bounds_errors:
        errors.extend(bounds_errors)

    events, event_errors, event_warnings = collect_concrete_activations(
        model,
        input_tensor,
    )
    if event_errors:
        errors.extend(event_errors)
    if event_warnings:
        warnings.extend(event_warnings)

    if errors:
        return {
            "status": "ERROR",
            "errors": errors,
            "warnings": warnings,
            "violations_total": 0,
            "violations_topk": [],
            "layerwise_stats": [],
            "alignment": {},
            "total_checks": 0,
            "worst_gap": 0.0,
        }

    alignment = align_activations_to_act_layers(
        act_net,
        events,
        mode="hookable_order_strict",
    )
    if alignment.errors:
        return {
            "status": "ERROR",
            "errors": alignment.errors,
            "warnings": warnings + alignment.warnings,
            "violations_total": 0,
            "violations_topk": [],
            "layerwise_stats": [],
            "alignment": alignment.meta,
            "total_checks": 0,
            "worst_gap": 0.0,
        }

    concrete_by_layer = {lid: ev.tensor for lid, ev in alignment.mapping.items()}
    missing_bounds = [lid for lid in concrete_by_layer.keys() if lid not in bounds_by_layer]
    if missing_bounds:
        return {
            "status": "ERROR",
            "errors": [f"Missing bounds for layer_ids={sorted(missing_bounds)}"],
            "warnings": warnings,
            "violations_total": 0,
            "violations_topk": [],
            "layerwise_stats": [],
            "alignment": alignment.meta,
            "total_checks": 0,
            "worst_gap": 0.0,
        }

    bounds_for_compare = {lid: bounds_by_layer[lid] for lid in concrete_by_layer.keys()}
    compare = compare_bounds_per_neuron(
        bounds_by_layer=bounds_for_compare,
        concrete_by_layer=concrete_by_layer,
        atol=config.atol,
        rtol=config.rtol,
        topk=config.topk,
        nan_policy=config.nan_policy,
    )

    if compare.get("status") == "ERROR":
        return {
            "status": "ERROR",
            "errors": compare.get("errors", []),
            "warnings": warnings + compare.get("warnings", []),
            "violations_total": 0,
            "violations_topk": [],
            "layerwise_stats": [],
            "alignment": alignment.meta,
            "total_checks": 0,
            "worst_gap": 0.0,
        }

    layerwise_stats = compare.get("layerwise_stats", [])
    total_checks = sum(int(s.get("num_neurons", 0)) for s in layerwise_stats)
    worst_gap = 0.0
    for s in layerwise_stats:
        worst_gap = max(worst_gap, float(s.get("max_gap", 0.0)))

    compare["alignment"] = alignment.meta
    compare["warnings"] = warnings + compare.get("warnings", [])
    compare["total_checks"] = int(total_checks)
    compare["worst_gap"] = float(worst_gap)
    return compare
