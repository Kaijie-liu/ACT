#!/usr/bin/env python3
#===- act/pipeline/verification/layer_alignment.py - Alignment ---------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .activations import ActivationEvent


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
        Strictly tolerate ONLY batch=1 differences:
          raw=(1, *expected) -> drop leading 1

        Returns:
          (shape_after, dropped, reason)
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
