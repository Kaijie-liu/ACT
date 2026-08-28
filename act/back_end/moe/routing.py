# ===- act/back_end/moe/routing.py - Router Bound Utilities ------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

from __future__ import annotations

import torch

from act.back_end.core import Bounds


def _score_tensors(bounds: Bounds) -> tuple[torch.Tensor, torch.Tensor]:
    lb, ub = bounds.lb, bounds.ub
    if lb.shape != ub.shape:
        raise ValueError("score lower/upper bound shapes differ")
    if lb.ndim == 1:
        lb, ub = lb.unsqueeze(0), ub.unsqueeze(0)
    if lb.ndim != 2:
        raise ValueError("router score bounds must be [batch, num_experts]")
    if not bool(torch.isfinite(lb).all() and torch.isfinite(ub).all()):
        raise ValueError("router score bounds must be finite")
    if bool((lb > ub).any()):
        raise ValueError("router score lower bound exceeds upper bound")
    return lb, ub


def pairwise_margin_bounds(bounds: Bounds) -> Bounds:
    """Return interval bounds for ``r_j - r_i`` as ``[B, i, j]`` tensors."""
    lb, ub = _score_tensors(bounds)
    margin_lb = lb[:, None, :] - ub[:, :, None]
    margin_ub = ub[:, None, :] - lb[:, :, None]
    return Bounds(margin_lb, margin_ub)


def interval_candidate_mask(bounds: Bounds, top_k: int) -> torch.Tensor:
    """Sound top-k candidate mask under interval router score bounds.

    An expert is impossible only when at least ``top_k`` other experts are
    definitely strictly above it.  Ties remain candidates, matching the
    ``ANY_LEGAL_TOPK`` verification semantics.
    """
    lb, ub = _score_tensors(bounds)
    experts = lb.shape[1]
    if not 1 <= int(top_k) <= experts:
        raise ValueError("top_k must lie in [1, num_experts]")
    definitely_higher = lb[:, None, :] > ub[:, :, None]
    eye = torch.eye(experts, dtype=torch.bool, device=lb.device).unsqueeze(0)
    count = (definitely_higher & ~eye).sum(dim=-1)
    return count <= int(top_k) - 1


def interval_big_m(bounds: Bounds) -> torch.Tensor:
    """Safe ``M[j,i]`` bounds for all differences ``r_j-r_i``."""
    return pairwise_margin_bounds(bounds).ub


def top2_sigmoid_gate_range(margin_bounds: Bounds) -> Bounds:
    """Map bounds on ``r_a-r_b`` to the selected-softmax top-2 weight."""
    if margin_bounds.lb.shape != margin_bounds.ub.shape:
        raise ValueError("margin bound shapes differ")
    return Bounds(torch.sigmoid(margin_bounds.lb), torch.sigmoid(margin_bounds.ub))


def selected_softmax_weight_range(pairwise_bounds: Bounds, expert: int) -> Bounds:
    """Bound one selected-softmax weight from pairwise score margins.

    ``pairwise_bounds`` contains ``r_j-r_i`` in its last two dimensions.  The
    returned interval assumes every listed expert belongs to the selected set;
    callers may slice those dimensions to a route set before invoking it.
    """
    lb, ub = pairwise_bounds.lb, pairwise_bounds.ub
    if lb.shape != ub.shape or lb.ndim < 2 or lb.shape[-1] != lb.shape[-2]:
        raise ValueError("pairwise bounds must end in a square [i, j] matrix")
    count = lb.shape[-1]
    if not 0 <= int(expert) < count:
        raise IndexError("expert index out of range")
    others = torch.arange(count, device=lb.device) != int(expert)
    lower = 1.0 / (1.0 + torch.exp(ub[..., expert, others]).sum(dim=-1))
    upper = 1.0 / (1.0 + torch.exp(lb[..., expert, others]).sum(dim=-1))
    return Bounds(lower, upper)
