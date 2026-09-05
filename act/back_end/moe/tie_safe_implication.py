# ===- act/back_end/moe/tie_safe_implication.py -----------------------====#
"""Tie-safe compilation of a hard top-1 route implication.

For branch ``i`` define ``g_i = max_{j != i}(r_j-r_i)`` and let ``s_i`` be
the minimum safety-row value of expert ``i``.  Under tie-inclusive routing the
branch is legal when ``g_i <= 0``.  The tempting property
``max(g_i, s_i) >= 0`` is unsound at a tie.  This module instead compiles

``max(g_i - eta, s_i) >= 0``

for a strictly positive, disclosed ``eta``.  It is sound and is conservative
only for non-member branch obligations with ``0 < g_i < eta``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn as nn

from act.back_end.solver.solver_hz import HZ_NUMERICAL_POLICY


def relu_pairwise_max(values: torch.Tensor) -> torch.Tensor:
    """Compute a last-axis maximum using only affine operations and ReLU."""

    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("values must have a non-empty final dimension")
    result = values[..., 0]
    for index in range(1, values.shape[-1]):
        candidate = values[..., index]
        result = result + torch.relu(candidate - result)
    return result


def relu_pairwise_min(values: torch.Tensor) -> torch.Tensor:
    return -relu_pairwise_max(-values)


def top1_branch_guard_values(
    router_scores: torch.Tensor,
    expert_indices: int | torch.Tensor,
) -> torch.Tensor:
    """Return ``max_{j != i}(r_j-r_i)`` for one branch per sample."""

    if router_scores.ndim != 2 or router_scores.shape[1] < 2:
        raise ValueError("router_scores must be [batch, experts] with E >= 2")
    batch, experts = router_scores.shape
    if isinstance(expert_indices, int):
        indices = torch.full(
            (batch,), int(expert_indices), dtype=torch.long, device=router_scores.device
        )
    else:
        indices = torch.as_tensor(
            expert_indices, dtype=torch.long, device=router_scores.device
        ).reshape(-1)
        if indices.numel() != batch:
            raise ValueError("expert_indices must have one entry per sample")
    if bool(torch.any(indices < 0)) or bool(torch.any(indices >= experts)):
        raise IndexError("expert index is outside router scores")
    selected = router_scores.gather(1, indices[:, None]).squeeze(1)
    mask = torch.nn.functional.one_hot(indices, num_classes=experts).bool()
    outside = router_scores.masked_fill(mask, -torch.inf)
    competitor = outside.max(dim=1).values
    return competitor - selected


@dataclass(frozen=True)
class EtaBandAudit:
    samples: int
    branch_obligations: int
    legal_branches: int
    overcheck_branches: int
    overcheck_samples: int
    numerical_boundary_branches: int
    numerical_boundary_samples: int
    eta: float
    boundary_tolerance: float

    def as_dict(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "branch_obligations": self.branch_obligations,
            "legal_branches": self.legal_branches,
            "overcheck_branches": self.overcheck_branches,
            "overcheck_samples": self.overcheck_samples,
            "numerical_boundary_branches": self.numerical_boundary_branches,
            "numerical_boundary_samples": self.numerical_boundary_samples,
            "eta": self.eta,
            "boundary_tolerance": self.boundary_tolerance,
            "overcheck_definition": "0 < g_i < eta",
        }


def audit_eta_overcheck_band(
    router_scores: torch.Tensor,
    *,
    eta: float = HZ_NUMERICAL_POLICY.safe_positive_margin,
    boundary_tolerance: float = HZ_NUMERICAL_POLICY.feasibility_tolerance,
) -> EtaBandAudit:
    """Count the eta-only obligations over every sample/expert branch pair."""

    eta = float(eta)
    boundary_tolerance = float(boundary_tolerance)
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta must be finite and strictly positive")
    if not math.isfinite(boundary_tolerance) or boundary_tolerance < 0.0:
        raise ValueError("boundary_tolerance must be finite and nonnegative")
    if router_scores.ndim != 2 or router_scores.shape[1] < 2:
        raise ValueError("router_scores must be [batch, experts] with E >= 2")
    batch, experts = router_scores.shape
    guards = torch.stack(
        [top1_branch_guard_values(router_scores, index) for index in range(experts)],
        dim=1,
    )
    overcheck = (guards > 0.0) & (guards < eta)
    boundary = guards.abs() <= boundary_tolerance
    legal = guards <= 0.0
    return EtaBandAudit(
        samples=int(batch),
        branch_obligations=int(batch * experts),
        legal_branches=int(legal.sum().item()),
        overcheck_branches=int(overcheck.sum().item()),
        overcheck_samples=int(overcheck.any(dim=1).sum().item()),
        numerical_boundary_branches=int(boundary.sum().item()),
        numerical_boundary_samples=int(boundary.any(dim=1).sum().item()),
        eta=eta,
        boundary_tolerance=boundary_tolerance,
    )


class TieSafeTop1Implication(nn.Module):
    """Compile one hard top-1 branch implication to a scalar ReLU DAG.

    ``router_logits`` must return all affine/continuous route scores, not the
    integer output of an argmax module.  The result has shape ``[batch, 1]`` and
    is intended to be verified nonnegative over the original input box.
    """

    def __init__(
        self,
        router_logits: nn.Module,
        expert: nn.Module,
        expert_index: int,
        property_matrix: torch.Tensor | Sequence[Sequence[float]],
        property_offset: torch.Tensor | Sequence[float],
        *,
        eta: float = HZ_NUMERICAL_POLICY.safe_positive_margin,
    ) -> None:
        super().__init__()
        eta = float(eta)
        if not math.isfinite(eta) or eta <= 0.0:
            raise ValueError("eta must be finite and strictly positive")
        matrix = torch.as_tensor(property_matrix)
        offset = torch.as_tensor(property_offset)
        if matrix.ndim != 2 or matrix.shape[0] < 1:
            raise ValueError("property_matrix must be non-empty and two-dimensional")
        if offset.ndim != 1 or offset.shape[0] != matrix.shape[0]:
            raise ValueError("property_offset must match property rows")
        if not matrix.is_floating_point():
            matrix = matrix.double()
        if not offset.is_floating_point():
            offset = offset.double()
        self.router_logits = router_logits
        self.expert = expert
        self.expert_index = int(expert_index)
        self.eta = eta
        self.register_buffer("property_matrix", matrix)
        self.register_buffer("property_offset", offset)

    def forward_components(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self.router_logits(x)
        output = self.expert(x)
        if scores.ndim != 2 or scores.shape[1] < 2:
            raise ValueError("router_logits must return [batch, experts], E >= 2")
        if not 0 <= self.expert_index < scores.shape[1]:
            raise IndexError("expert_index is outside router output")
        if output.ndim != 2 or output.shape[1] != self.property_matrix.shape[1]:
            raise ValueError("expert output width differs from property matrix")
        selected = scores[:, self.expert_index : self.expert_index + 1]
        outside = torch.cat(
            [scores[:, : self.expert_index], scores[:, self.expert_index + 1 :]],
            dim=1,
        )
        guard = relu_pairwise_max(outside - selected)
        matrix = self.property_matrix.to(dtype=output.dtype, device=output.device)
        offset = self.property_offset.to(dtype=output.dtype, device=output.device)
        rows = output @ matrix.transpose(0, 1) + offset
        safety = relu_pairwise_min(rows)
        compiled = guard - self.eta + torch.relu(safety - (guard - self.eta))
        return guard, safety, compiled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_components(x)[2].unsqueeze(1)


class LagrangianTop1GuardedProperty(nn.Module):
    """Compile a hard-top1 guarded property into ordinary network outputs.

    For branch ``i``, define each selected margin
    ``m_j = r_i - r_j``. The branch is legal, including ties, exactly when all
    ``m_j >= 0``. For safety row ``s_l >= 0`` and registered nonnegative
    multipliers ``mu_lj``, this module emits

    ``phi_l = s_l - sum_j mu_lj * m_j``.

    On the legal branch ``phi_l <= s_l``. Therefore proving every compiled row
    nonnegative over the original input box is a sound sufficient proof of the
    expert property on the guarded cell. A tied competitor contributes exactly
    zero, so its obligation cannot be discharged by the reduction. Other
    strictly lower competitors may still make the sufficient condition
    conservative. It is intentionally not an exact encoding of the guarded
    optimization problem.
    """

    def __init__(
        self,
        router_logits: nn.Module,
        expert: nn.Module,
        expert_index: int,
        property_matrix: torch.Tensor | Sequence[Sequence[float]],
        property_offset: torch.Tensor | Sequence[float],
        multipliers: torch.Tensor | Sequence[Sequence[float]],
    ) -> None:
        super().__init__()
        matrix = torch.as_tensor(property_matrix)
        offset = torch.as_tensor(property_offset)
        multiplier_tensor = torch.as_tensor(multipliers)
        if matrix.ndim != 2 or matrix.shape[0] < 1:
            raise ValueError("property_matrix must be non-empty and two-dimensional")
        if offset.ndim != 1 or offset.shape[0] != matrix.shape[0]:
            raise ValueError("property_offset must match property rows")
        if multiplier_tensor.ndim != 2:
            raise ValueError("multipliers must be [property rows, outside experts]")
        if multiplier_tensor.shape[0] != matrix.shape[0]:
            raise ValueError("multipliers must have one row per property")
        if multiplier_tensor.shape[1] < 1:
            raise ValueError("multipliers require at least one outside expert")
        if not matrix.is_floating_point():
            matrix = matrix.double()
        if not offset.is_floating_point():
            offset = offset.double()
        multiplier_tensor = multiplier_tensor.to(dtype=matrix.dtype)
        if not bool(torch.isfinite(multiplier_tensor).all()):
            raise ValueError("multipliers must be finite")
        if bool(torch.any(multiplier_tensor < 0)):
            raise ValueError("multipliers must be nonnegative")
        self.router_logits = router_logits
        self.expert = expert
        self.expert_index = int(expert_index)
        self.register_buffer("property_matrix", matrix)
        self.register_buffer("property_offset", offset)
        self.register_buffer("multipliers", multiplier_tensor)

    def forward_components(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self.router_logits(x)
        output = self.expert(x)
        if scores.ndim != 2 or scores.shape[1] < 2:
            raise ValueError("router_logits must return [batch, experts], E >= 2")
        if not 0 <= self.expert_index < scores.shape[1]:
            raise IndexError("expert_index is outside router output")
        outside_width = scores.shape[1] - 1
        if self.multipliers.shape[1] != outside_width:
            raise ValueError("multiplier width differs from outside expert count")
        if output.ndim != 2 or output.shape[1] != self.property_matrix.shape[1]:
            raise ValueError("expert output width differs from property matrix")
        selected = scores[:, self.expert_index : self.expert_index + 1]
        outside = torch.cat(
            [scores[:, : self.expert_index], scores[:, self.expert_index + 1 :]],
            dim=1,
        )
        selected_margins = selected - outside
        matrix = self.property_matrix.to(dtype=output.dtype, device=output.device)
        offset = self.property_offset.to(dtype=output.dtype, device=output.device)
        multipliers = self.multipliers.to(dtype=output.dtype, device=output.device)
        safety_rows = output @ matrix.transpose(0, 1) + offset
        compiled_rows = safety_rows - selected_margins @ multipliers.transpose(0, 1)
        return selected_margins, safety_rows, compiled_rows

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_components(x)[2]
