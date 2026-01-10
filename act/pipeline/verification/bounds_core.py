#!/usr/bin/env python3
#===- act/pipeline/verification/bounds_core.py - Abstract Bounds Core --====#
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
    before, after, _globalC = analyze(act_net, entry_id, entry_fact)

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
