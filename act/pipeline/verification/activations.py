#!/usr/bin/env python3
#===- act/pipeline/verification/activations.py - Activation Capture ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple, Type

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ActivationEvent:
    module_path: str
    module_type: str
    call_idx: int
    shape: Tuple[int, ...]
    tensor: torch.Tensor


def collect_concrete_activations(
    model: nn.Module,
    x: torch.Tensor,
    *,
    include_types: Tuple[Type[nn.Module], ...] = (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ReLU,
    ),
    strict_single_call_per_module: bool = True,
    detach: bool = True,
    drop_batch_dim: bool = True,
) -> Tuple[List[ActivationEvent], List[str], List[str]]:
    """
    Collect activation events for a forward pass.

    Returns:
        events, errors, warnings
    """
    events: List[ActivationEvent] = []
    errors: List[str] = []
    warnings: List[str] = []

    call_counts: dict[int, int] = {}
    hooks: List[Any] = []

    def make_hook(module_path: str, module: nn.Module):
        def hook(_mod, _inp, out):
            mid = id(module)
            count = call_counts.get(mid, 0)
            call_counts[mid] = count + 1
            if strict_single_call_per_module and count > 0:
                errors.append(
                    f"Module called multiple times in forward: {module_path} "
                    f"(type={type(module).__name__})"
                )

            if not torch.is_tensor(out):
                errors.append(
                    f"Activation output is not a tensor for module {module_path} "
                    f"(type={type(module).__name__})"
                )
                return

            t = out.detach() if detach else out
            if drop_batch_dim and t.dim() > 0 and t.shape[0] == 1:
                t = t.squeeze(0)
            events.append(
                ActivationEvent(
                    module_path=module_path,
                    module_type=type(module).__name__,
                    call_idx=count,
                    shape=tuple(t.shape),
                    tensor=t,
                )
            )
        return hook

    for name, module in model.named_modules():
        if name == "":
            continue
        if isinstance(module, include_types):
            hooks.append(module.register_forward_hook(make_hook(name, module)))

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    if not events:
        warnings.append("No activation events collected (check include_types).")

    return events, errors, warnings
