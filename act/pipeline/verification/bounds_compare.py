#!/usr/bin/env python3
#===- act/pipeline/verification/bounds_compare.py - Bounds Compare -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

from typing import Any, Dict, List

import torch

from .bounds_core import LayerBounds


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
