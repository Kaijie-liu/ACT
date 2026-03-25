#===- act/back_end/hybridz_tf/tf_transformer.py - HybridZ Transformer TF -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ Transformer Transfer Functions. Implements HybridZ-based transfer
#   functions for Transformer layers including layer normalization, attention
#   mechanisms, and position encoding.
#
#===---------------------------------------------------------------------===#

import torch
import math
from act.back_end.core import Bounds, Fact, Layer, ConSet
from act.back_end.hybridz_tf.tf_mlp import _hz_from_bounds_fresh


@torch.no_grad()
def hybridz_tf_layernorm(L: Layer, Bin: Bounds, tf=None):
    """Layer normalization (conservative)."""
    normalized_shape = L.params.get("normalized_shape")
    eps = float(L.params.get("eps", 1e-5))
    weight = L.params.get("weight")
    bias = L.params.get("bias")

    input_range = Bin.ub - Bin.lb
    if torch.all(input_range < eps):
        lb_norm = torch.zeros_like(Bin.lb)
        ub_norm = torch.zeros_like(Bin.ub)
    else:
        scale_factor = 3.0
        lb_norm = torch.full_like(Bin.lb, -scale_factor)
        ub_norm = torch.full_like(Bin.ub, scale_factor)

    if weight is not None:
        weight_pos = torch.clamp(weight, min=0)
        weight_neg = torch.clamp(weight, max=0)
        lb_out = weight_pos * lb_norm + weight_neg * ub_norm
        ub_out = weight_pos * ub_norm + weight_neg * lb_norm
    else:
        lb_out = lb_norm
        ub_out = ub_norm

    if bias is not None:
        lb_out += bias
        ub_out += bias

    Bout = Bounds(lb=lb_out, ub=ub_out)
    if tf is not None:
        tf._hz_cache[L.id] = _hz_from_bounds_fresh(Bout, Bout.lb.dtype, Bout.lb.device)

    cons = ConSet()
    cons.add_op(f"layernorm:{L.id}", list(L.out_vars + L.in_vars),
                normalized_shape=normalized_shape, eps=eps)
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_gelu(L: Layer, Bin: Bounds, tf=None):
    """GELU activation (piecewise linear approximation)."""
    breakpoints = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], device=Bin.lb.device, dtype=Bin.lb.dtype)

    lb = torch.zeros_like(Bin.lb)
    ub = torch.zeros_like(Bin.ub)

    for i in range(len(Bin.lb)):
        x_min, x_max = Bin.lb[i].item(), Bin.ub[i].item()
        y_candidates = [gelu_approx(x_min), gelu_approx(x_max)]
        for bp in breakpoints:
            if x_min <= bp <= x_max:
                y_candidates.append(gelu_approx(bp.item()))
        if x_min <= -0.7 <= x_max:
            y_candidates.append(gelu_approx(-0.7))
        y_candidates = torch.tensor(y_candidates, device=Bin.lb.device, dtype=Bin.lb.dtype)
        lb[i] = torch.min(y_candidates)
        ub[i] = torch.max(y_candidates)

    Bout = Bounds(lb=lb, ub=ub)
    if tf is not None:
        tf._hz_cache[L.id] = _hz_from_bounds_fresh(Bout, Bout.lb.dtype, Bout.lb.device)

    cons = ConSet()
    cons.add_op(f"gelu:{L.id}", list(L.out_vars + L.in_vars))
    return Fact(bounds=Bout, cons=cons)


def gelu_approx(x: float) -> float:
    """Approximate GELU function value."""
    sqrt_2_pi = math.sqrt(2.0 / math.pi)
    inner = sqrt_2_pi * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + math.tanh(inner))


@torch.no_grad()
def hybridz_tf_softmax(L: Layer, Bin: Bounds, tf=None):
    """Softmax with simplex constraints."""
    n = len(Bin.lb)
    lb = torch.zeros_like(Bin.lb)
    ub = torch.ones_like(Bin.ub)

    max_input = torch.max(Bin.ub)
    min_input = torch.min(Bin.lb)

    for i in range(n):
        if Bin.lb[i] > max_input - 1e-6:
            others_max = torch.max(torch.cat([Bin.ub[:i], Bin.ub[i+1:]]))
            if Bin.lb[i] > others_max + 1.0:
                lb[i] = 0.7
        if Bin.ub[i] < min_input + 1e-6:
            others_min = torch.min(torch.cat([Bin.lb[:i], Bin.lb[i+1:]]))
            if Bin.ub[i] < others_min - 1.0:
                ub[i] = 0.3

    Bout = Bounds(lb=lb, ub=ub)
    if tf is not None:
        tf._hz_cache[L.id] = _hz_from_bounds_fresh(Bout, Bout.lb.dtype, Bout.lb.device)

    cons = ConSet()
    rowsize = len(L.out_vars)
    cons.add_op(f"softmax:{L.id}", list(L.out_vars), rowsize=rowsize)
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_posenc(L: Layer, Bin: Bounds, tf=None):
    """Positional encoding."""
    max_len = L.params.get("max_len", 1000)
    d_model = L.params.get("d_model", Bin.lb.shape[-1])

    pe_range = 1.0
    lb = Bin.lb - pe_range
    ub = Bin.ub + pe_range
    Bout = Bounds(lb=lb, ub=ub)
    if tf is not None:
        tf._hz_cache[L.id] = _hz_from_bounds_fresh(Bout, Bout.lb.dtype, Bout.lb.device)

    cons = ConSet()
    cons.add_op(f"posenc:{L.id}", list(L.out_vars + L.in_vars),
                max_len=max_len, d_model=d_model)
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_attention_scores(L: Layer, Q_bounds: Bounds, K_bounds: Bounds, tf=None):
    """Attention score computation: Q @ K^T / sqrt(d_k)."""
    d_k = L.params.get("d_k", Q_bounds.lb.shape[-1])
    scale = 1.0 / math.sqrt(d_k)

    q_lb, q_ub = Q_bounds.lb, Q_bounds.ub
    k_lb, k_ub = K_bounds.lb, K_bounds.ub

    products = []
    for i in range(len(q_lb)):
        for j in range(len(k_lb)):
            corners = torch.tensor([
                q_lb[i] * k_lb[j],
                q_lb[i] * k_ub[j],
                q_ub[i] * k_lb[j],
                q_ub[i] * k_ub[j]
            ])
            products.append((torch.min(corners), torch.max(corners)))

    lb = torch.tensor([p[0] for p in products]) * scale
    ub = torch.tensor([p[1] for p in products]) * scale

    Bout = Bounds(lb=lb, ub=ub)
    if tf is not None:
        tf._hz_cache[L.id] = _hz_from_bounds_fresh(Bout, Bout.lb.dtype, Bout.lb.device)

    cons = ConSet()
    cons.add_op(f"att_scores:{L.id}", list(L.out_vars + L.in_vars), d_k=d_k)
    return Fact(bounds=Bout, cons=cons)
