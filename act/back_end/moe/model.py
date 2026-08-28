# ===- act/back_end/moe/model.py - Concrete Output-Level MoE -----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from act.back_end.moe.schema import GateKind, OutputLevelMoESpec


@dataclass(frozen=True)
class RoutingDecision:
    scores: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor


class OutputLevelMoE(nn.Module):
    """Trainable concrete semantics for a final MoE classifier/regressor.

    Verification decomposes this module into its router and expert components;
    the full dynamic forward is retained as the ground-truth implementation for
    training, attacks, and counterexample validation.
    """

    def __init__(
        self,
        router: nn.Module,
        experts: Sequence[nn.Module],
        spec: OutputLevelMoESpec,
        *,
        shared_expert: nn.Module | None = None,
        straight_through_hard: bool = True,
    ) -> None:
        super().__init__()
        if len(experts) != spec.num_experts:
            raise ValueError("expert count does not match MoE spec")
        self.router = router
        self.experts = nn.ModuleList(experts)
        self.shared_expert = shared_expert
        self.spec = spec
        self.straight_through_hard = bool(straight_through_hard)

    def route(self, x: torch.Tensor) -> RoutingDecision:
        scores = self.router(x)
        if scores.ndim != 2 or scores.shape[1] != self.spec.num_experts:
            raise ValueError(
                "router must return [batch, num_experts], got "
                f"{tuple(scores.shape)}"
            )
        selected_scores, indices = torch.topk(
            scores, self.spec.top_k, dim=-1, largest=True, sorted=True
        )
        if self.spec.gate == GateKind.HARD_TOP1:
            weights = torch.ones_like(selected_scores)
        elif self.spec.gate == GateKind.SELECTED_SOFTMAX:
            weights = torch.softmax(selected_scores, dim=-1)
        elif self.spec.gate == GateKind.NORMALIZED_SIGMOID:
            raw = torch.sigmoid(selected_scores)
            weights = raw / raw.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(raw.dtype).tiny
            )
        elif self.spec.gate == GateKind.SWITCH_PROB:
            full = torch.softmax(scores, dim=-1)
            weights = torch.gather(full, 1, indices)
        else:
            raise NotImplementedError(f"unsupported gate {self.spec.gate.value}")
        return RoutingDecision(scores=scores, indices=indices, weights=weights)

    def forward_with_routing(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, RoutingDecision]:
        decision = self.route(x)
        expert_values = torch.stack([expert(x) for expert in self.experts], dim=1)
        if (
            self.spec.gate == GateKind.HARD_TOP1
            and self.training
            and self.straight_through_hard
        ):
            soft = torch.softmax(decision.scores, dim=-1)
            hard = torch.zeros_like(soft).scatter_(1, decision.indices, 1.0)
            surrogate = hard + soft - soft.detach()
            output = (surrogate.unsqueeze(-1) * expert_values).sum(dim=1)
        else:
            gather_index = decision.indices.unsqueeze(-1).expand(
                -1, -1, expert_values.shape[-1]
            )
            selected = torch.gather(expert_values, 1, gather_index)
            output = (decision.weights.unsqueeze(-1) * selected).sum(dim=1)
        if self.shared_expert is not None:
            output = output + self.shared_expert(x)
        return output, decision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_routing(x)[0]
