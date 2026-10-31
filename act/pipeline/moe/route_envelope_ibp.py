"""Opt-in S2 route-envelope IBP; no existing verifier entry point is changed.

The real-arithmetic inclusion rule covers every tie-legal selected-softmax
top-2 execution. PyTorch arithmetic is the declared numerical policy, not an
outward-rounded proof. A candidate pair is box-compatible, not proven reachable.
"""
from dataclasses import dataclass
from itertools import combinations

import torch
from torch import nn
from torch.nn import functional as F


def check_box(lower, upper):
    if (lower.shape != upper.shape or lower.dtype != upper.dtype or
            lower.device != upper.device or lower.ndim < 2 or
            not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or
            not (lower <= upper).all()):
        raise ValueError("invalid finite input interval")


def interval(module, lower, upper):
    """IBP for the explicitly supported affine/ReLU MLP language."""
    if isinstance(module, nn.Sequential):
        for child in module:
            lower, upper = interval(child, lower, upper)
        return lower, upper
    if isinstance(module, nn.Flatten):
        return module(lower), module(upper)
    if isinstance(module, nn.Linear):
        center, radius = (lower + upper) * .5, (upper - lower) * .5
        center = F.linear(center, module.weight, module.bias)
        radius = F.linear(radius, module.weight.abs())
        return center - radius, center + radius
    if isinstance(module, nn.ReLU):
        return lower.clamp_min(0), upper.clamp_min(0)
    raise ValueError(f"unsupported IBP operation: {type(module).__name__}")


def projected_interval(network, lower, upper, rows=None, constant=0.):
    """Fold property rows into the last affine layer BEFORE interval evaluation.

    rows is [properties, classes] or [batch, properties, classes]. The latter
    supports different true labels in one training batch without class cropping.
    """
    layers = list(network) if isinstance(network, nn.Sequential) else [network]
    if not layers or not isinstance(layers[-1], nn.Linear):
        raise ValueError("property projection requires a final affine layer")
    for layer in layers[:-1]:
        lower, upper = interval(layer, lower, upper)
    last = layers[-1]
    if rows is None:
        lo, hi = interval(last, lower, upper)
        return lo + constant, hi + constant
    if (rows.ndim not in (2, 3) or rows.shape[-1] != last.out_features or
            (rows.ndim == 3 and rows.shape[0] != lower.shape[0]) or
            rows.device != lower.device or rows.dtype != lower.dtype or
            not torch.isfinite(rows).all()):
        raise ValueError("property dimensions/dtype/device or coefficients")
    weight = rows @ last.weight
    bias = rows @ last.bias if last.bias is not None else 0.
    center, radius = (lower + upper) * .5, (upper - lower) * .5
    if rows.ndim == 2:
        mid = F.linear(center, weight) + bias + constant
        rad = F.linear(radius, weight.abs())
    else:
        mid = torch.einsum('bph,bh->bp', weight, center) + bias + constant
        rad = torch.einsum('bph,bh->bp', weight.abs(), radius)
    return mid - rad, mid + rad


def margin_rows(labels, classes, *, dtype):
    """All classes except the true one; one common convention for all arms."""
    if classes < 2 or labels.ndim != 1 or labels.dtype != torch.long:
        raise ValueError("classification margin dimensions")
    if not ((labels >= 0) & (labels < classes)).all():
        raise ValueError("invalid labels")
    ids = torch.arange(classes, device=labels.device).expand(len(labels), -1)
    others = ids[ids != labels[:, None]].reshape(len(labels), classes - 1)
    eye = torch.eye(classes, dtype=dtype, device=labels.device)
    return eye[labels, None, :] - eye[others]


@dataclass
class Envelope:
    lower: torch.Tensor
    upper: torch.Tensor
    router_lower: torch.Tensor
    router_upper: torch.Tensor
    pairs: tuple
    candidates: torch.Tensor
    gate_lower: torch.Tensor
    gate_upper: torch.Tensor

    @property
    def single_route(self):
        # Completeness of candidate coverage makes exactly one sufficient.
        # Zero regularizer at tau=0 is deliberately NOT used here.
        return self.candidates.sum(1) == 1


def route_envelope(layer, lower, upper, rows=None, constant=0.):
    """Whole-box union enclosure for ONE output routed layer.

    E=1 is an explicit weight-one degenerate case, requiring k=1. It is not
    claimed to be a valid top-2 selection from a singleton expert set.
    """
    check_box(lower, upper)
    experts = len(layer.experts)
    if layer.mode != 'selected_softmax' or layer.k != min(2, experts) or experts < 1:
        raise ValueError("requires selected-softmax top-2 (singleton: top-1)")
    rl, ru = interval(layer.router, lower, upper)
    if rl.shape != (lower.shape[0], experts):
        raise ValueError("router/expert cardinality mismatch")
    check_box(rl, ru)
    values = [projected_interval(e, lower, upper, rows, constant) for e in layer.experts]
    el = torch.stack([v[0] for v in values], dim=1)
    eu = torch.stack([v[1] for v in values], dim=1)
    check_box(el, eu)
    if experts == 1:
        one = torch.ones_like(rl)
        return Envelope(el[:, 0], eu[:, 0], rl, ru, ((0,),), one.bool(), one, one)
    pairs = tuple(combinations(range(experts), 2))
    mask, gl, gu, lows, highs = [], [], [], [], []
    for a, b in pairs:
        others = [c for c in range(experts) if c not in (a, b)]
        if others:
            threshold = rl[:, others].amax(1)
            possible = (ru[:, a] >= threshold) & (ru[:, b] >= threshold)
        else:
            possible = torch.ones(len(rl), dtype=torch.bool, device=rl.device)
        # Weight of a, the first expert of the CANONICAL unordered pair.
        lamlo = torch.sigmoid(rl[:, a] - ru[:, b])
        lamhi = torch.sigmoid(ru[:, a] - rl[:, b])
        lowends = [t[:, None]*el[:, a] + (1-t[:, None])*el[:, b]
                   for t in (lamlo, lamhi)]
        highends = [t[:, None]*eu[:, a] + (1-t[:, None])*eu[:, b]
                    for t in (lamlo, lamhi)]
        lows.append(torch.minimum(*lowends).masked_fill(~possible[:, None], torch.inf))
        highs.append(torch.maximum(*highends).masked_fill(~possible[:, None], -torch.inf))
        mask.append(possible); gl.append(lamlo); gu.append(lamhi)
    candidates = torch.stack(mask, dim=1)
    if not candidates.any(1).all():
        raise ValueError("empty candidate cover")
    return Envelope(torch.stack(lows, 1).amin(1), torch.stack(highs, 1).amax(1),
                    rl, ru, pairs, candidates, torch.stack(gl, 1), torch.stack(gu, 1))


def route_exclusion_penalty(bounds, clean_indices):
    """tau=0, sum over excluded experts, mean over batch; no stability claim."""
    rl, ru = bounds.router_lower, bounds.router_upper
    if (clean_indices.shape != (len(rl), min(2, rl.shape[1])) or
            clean_indices.dtype != torch.long or
            not ((clean_indices >= 0) & (clean_indices < rl.shape[1])).all()):
        raise ValueError("clean route shape or index")
    if clean_indices.shape[1] == 2 and (clean_indices[:, 0] == clean_indices[:, 1]).any():
        raise ValueError("duplicate expert in clean route")
    chosen_lower = rl.gather(1, clean_indices).amin(1, keepdim=True)
    excluded = torch.ones_like(rl, dtype=torch.bool).scatter(1, clean_indices, False)
    return (ru - chosen_lower).clamp_min(0).masked_fill(~excluded, 0).sum(1).mean()


def certified_loss(lower_margins):
    """Upper bound on CE in real arithmetic; the true-class logit is zero."""
    zero = lower_margins.new_zeros((len(lower_margins), 1))
    return torch.logsumexp(torch.cat((zero, -lower_margins), dim=1), dim=1).mean()


def model_bounds(model, lower, upper, rows=None, constant=0.):
    """The SAME bound function is called in training and evaluation."""
    check_box(lower, upper)
    if isinstance(model, nn.Sequential):
        lo, hi = projected_interval(model, lower, upper, rows, constant)
        check_box(lo, hi)
        return lo, hi, None
    if (getattr(model, 'residual', False) or not hasattr(model, 'layers') or
            len(model.layers) != 1):
        raise ValueError("S2a supports dense MLP or one output MoE layer only")
    result = route_envelope(model.layers[0], lower, upper, rows, constant)
    return result.lower, result.upper, result
