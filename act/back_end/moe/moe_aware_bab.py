"""Opt-in MoE guard/margin frontend; no production dispatch is changed.

Guard optimization reuses author INVPROP, not a new general duality rule.
All floating bounds retain the numerical-policy grade. The caller owns the
hard request deadline, original-model replay and complete history aggregation.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import time

import torch
from torch import nn

from .multilayer_external import StaticPrefix


def tick(deadline):
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError('MoE conditional-bound shared deadline')


@dataclass(frozen=True)
class MarginCell:
    layer: int
    lower: float
    upper: float

    def __post_init__(self):
        if (type(self.layer) is not int or self.layer < 0
                or not math.isfinite(self.lower) or not math.isfinite(self.upper)
                or self.lower > self.upper):
            raise ValueError('finite ordered margin cell')


def closed_cells(layer, lower, upper, pieces):
    """Outward conversion of exact dyadic-input cuts; adjacent cells overlap."""
    MarginCell(layer, lower, upper)
    if type(pieces) is not int or pieces not in (2, 4):
        raise ValueError('registered two/four-way diagnostic')
    if lower == upper:
        return (MarginCell(layer, lower, upper),)
    a, b = Fraction(lower), Fraction(upper)
    cuts = [a + (b-a)*i/pieces for i in range(pieces+1)]
    def outward(value, down):
        result = float(value)
        if (down and Fraction(result) > value) or (not down and Fraction(result) < value):
            result = math.nextafter(result, -math.inf if down else math.inf)
        return result
    return tuple(MarginCell(layer, outward(x, True), outward(y, False))
                 for x, y in zip(cuts, cuts[1:]))


def propagated_gate_scores(widths, difference_norms, layer_lipschitz, property_norm=1.):
    """Pointwise fixed-history error contributions, not an optimum-gap bound.

Whether supplied D/K values are certified is a separate caller contract; this
function alone cannot turn heuristic estimates into proof evidence.
"""
    n = len(widths)
    if (not n or len(difference_norms) != n or len(layer_lipschitz) != n
            or any(not math.isfinite(v) or v < 0
                   for v in (*widths, *difference_norms, *layer_lipschitz, property_norm))
            or any(v > 1 for v in widths)):
        raise ValueError('complete finite nonnegative pointwise bounds')
    scores = [0.]*n
    suffix = property_norm
    for i in reversed(range(n)):
        scores[i] = widths[i]*difference_norms[i]*suffix
        suffix *= layer_lipschitz[i]
    if any(not math.isfinite(v) for v in scores):
        raise ValueError('nonfinite propagated error')
    return tuple(scores)


class GuardObjectives(nn.Module):
    """Joint graph [linear objectives, selected-minus-outsider guards, cells].

body._run must return outputs and the SAME execution's per-layer routers.
Prefix objectives optionally apply the next router, but guards never include
that undecided layer. Cell constraints are shifted in float64 with zero RHS.
"""
    def __init__(self, body, history, experts, rows, constants, *, source_id,
                 next_router=None, cells=()):
        super().__init__()
        if (not isinstance(source_id, str) or len(source_id) != 64
                or any(c not in '0123456789abcdef' for c in source_id)):
            raise ValueError('caller-bound source identity')
        if (rows.ndim != 2 or not len(rows) or constants.shape != (len(rows),)
                or rows.dtype != torch.float64 or constants.dtype != torch.float64
                or rows.device.type != 'cpu' or constants.device.type != 'cpu'
                or not torch.isfinite(rows).all() or not torch.isfinite(constants).all()):
            raise ValueError('all finite CPU/float64 objective rows/constants')
        history = tuple(tuple(p) for p in history)
        if len(history) > len(experts):
            raise ValueError('prefix dimensions')
        for pair, count in zip(history, experts):
            if (len(pair) != 2 or any(type(i) is not int for i in pair)
                    or not 0 <= pair[0] < pair[1] < count):
                raise ValueError('canonical tie-legal top2 pair')
        if any(m.training for m in body.modules()):
            raise ValueError('eval static body')
        if any(p.dtype != torch.float64 or p.device.type != 'cpu' for p in body.parameters()):
            raise ValueError('CPU/float64 static body')
        if any(not isinstance(cell, MarginCell) or cell.layer >= len(history) for cell in cells):
            raise ValueError('cell must bind an already selected layer')
        self.body, self.next_router = body, next_router
        self.history, self.source_id, self.cells = history, source_id, tuple(cells)
        self.guard_rows = tuple((i, selected, outside)
            for i, (pair, count) in enumerate(zip(history, experts))
            for selected in pair for outside in range(count) if outside not in pair)
        self.objective_count = len(rows)
        self.guard_count = len(self.guard_rows)+2*len(cells)
        self.register_buffer('rows', rows.detach().clone())
        self.register_buffer('constants', constants.detach().clone())

    def forward(self, x):
        out, scores = self.body._run(x)
        if self.next_router is not None:
            out = self.next_router(out)
        targets = torch.nn.functional.linear(out, self.rows, self.constants)
        guards = [scores[i][:, a:a+1]-scores[i][:, c:c+1]
                  for i, a, c in self.guard_rows]
        for cell in self.cells:
            a, b = self.history[cell.layer]
            margin = scores[cell.layer][:, a:a+1]-scores[cell.layer][:, b:b+1]
            guards.extend((margin-cell.lower, cell.upper-margin))
        # The first input to final Concat is deliberately a distinct target node.
        return torch.cat([targets, *guards], dim=1) if guards else targets


def history_objectives(model, history, rows, thresholds, *, source_id, cells=()):
    if len(history) != len(model.layers):
        raise ValueError('complete history for output properties')
    body = StaticPrefix(model, history).eval()
    return GuardObjectives(body, history, body.experts, rows, -thresholds,
                           source_id=source_id, cells=cells).eval()


def prefix_objectives(model, prefix, *, source_id, cells=()):
    """LB(-next margins) gives conditional UB for strictly-safe pruning."""
    body = StaticPrefix(model, prefix, next_router=True).eval()
    count = len(model.layers[len(prefix)].experts)
    ordered = tuple((i, j) for i in range(count) for j in range(count) if i != j)
    rows = torch.zeros((len(ordered), count), dtype=torch.float64)
    for k, (i, j) in enumerate(ordered):
        rows[k, i], rows[k, j] = -1., 1.
    graph = GuardObjectives(body, prefix, body.experts, rows,
        torch.zeros(len(rows), dtype=torch.float64), source_id=source_id,
        next_router=body.next_router, cells=cells).eval()
    return graph, ordered


class GuardMaximum(nn.Module):
    """Ordinary GenBaB alternative: per-property max(p, -all guards)."""
    def __init__(self, joint):
        super().__init__()
        self.joint = joint

    def forward(self, x):
        out = self.joint(x)
        values = out[:, :self.joint.objective_count]
        for j in range(self.joint.guard_count):
            atom = -out[:, self.joint.objective_count+j:self.joint.objective_count+j+1]
            values = values+torch.relu(atom-values)
        return values


def sliced_endpoint(model, domain, rows, thresholds, cell, *, variant,
                    input_features, expected_source_id):
    """M-C first-layer slice + M-A endpoint; returns joint graph and scoped domain.

The domain's first range is conditional on the returned graph's cell guards.
Never publish it as an unconditional original-prefix CROWN range.
"""
    if not isinstance(cell, MarginCell) or cell.layer != 0:
        raise ValueError('first-layer V2 endpoint diagnostic only')
    return margin_endpoint(model, domain, rows, thresholds, cell, variant=variant,
        input_features=input_features, expected_source_id=expected_source_id)


def margin_endpoint(model, domain, rows, thresholds, cell, *, variant,
                    input_features, expected_source_id):
    """Any-layer closed cell, on the SAME relaxed execution's router state.

    An original execution embeds with its original inner gate values. Only the
    last gate is endpointized: every router/cell guard precedes that gate.
    Earlier independent gates remain variables even when branching later.
    """
    from .gate_decoupling import (GateDomain, validate_domain, sigmoid_interval,
                                 build_graph, domain_digest)
    import hashlib
    import json
    validate_domain(domain, domain.history, expected_source_id=expected_source_id)
    if (variant not in ('V2_LO', 'V2_HI') or not isinstance(cell, MarginCell)
            or cell.layer >= len(domain.history)):
        raise ValueError('existing-layer V2 endpoint cell')
    layer = cell.layer
    lo, hi = max(cell.lower, domain.margins[layer][0]), min(cell.upper, domain.margins[layer][1])
    if lo > hi:
        raise ValueError('cell outside parent range')
    cell = MarginCell(layer, lo, hi)
    identity = dict(parent=domain_digest(domain), layer=layer, lower=lo, upper=hi,
                    mode='MARGIN_CONDITIONAL_V2')
    sid = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    margins = list(domain.margins); margins[layer] = (lo, hi)
    conditional = GateDomain(sid, domain.history, tuple(margins),
                             tuple(sigmoid_interval(*m) for m in margins))
    body = build_graph(model, domain.history, rows, thresholds, conditional,
                       variant=variant, input_features=input_features, expected_source_id=sid)
    experts = tuple(len(layer.experts) for layer in model.layers)
    graph = GuardObjectives(body, domain.history, experts, rows, -thresholds,
                            source_id=sid, cells=(cell,)).eval()
    graph.parent_domain_sha256 = domain_digest(domain)
    graph.variant = variant
    return graph, conditional


def margin_priority(model, domain, lower, upper, rows, *, deadline):
    """Cheap numerical error contributions for ordering only, never acceptance.

    IBP covers the independent-gate state boxes in real arithmetic; float64
    operations here are NOT outward certified. D uses absolute difference
    amplitude, K includes residual connections, and suffix includes the head.
    A nonfinite/unsupported score must cause a caller fallback, not a proof.
    """
    from .gate_decoupling import validate_domain, domain_digest
    validate_domain(domain, domain.history, expected_source_id=domain.source_id)
    tick(deadline)
    if (len(domain.history) != len(model.layers) or any(m.training for m in model.modules())
            or lower.shape != upper.shape or lower.shape[0] != 1
            or any(t.dtype != torch.float64 or t.device.type != 'cpu' or not torch.isfinite(t).all()
                   for t in (lower, upper, rows)) or (lower > upper).any()
            or rows.ndim != 2 or not len(rows)):
        raise ValueError('same finite eval CPU/float64 source')
    def interval(module, lo, hi):
        tick(deadline)
        if module is None or type(module) is nn.Identity:
            return lo, hi, 1.
        if type(module) is nn.Sequential:
            lip = 1.
            for child in module:
                lo, hi, value = interval(child, lo, hi); lip *= value
            return lo, hi, lip
        if type(module) is nn.Flatten:
            return module(lo), module(hi), 1.
        if type(module) is nn.ReLU:
            return lo.clamp_min(0), hi.clamp_min(0), 1.
        if type(module) is nn.Linear:
            w = module.weight.detach(); pos, neg = w.clamp_min(0), w.clamp_max(0)
            low = torch.nn.functional.linear(lo, pos, module.bias)+torch.nn.functional.linear(hi, neg)
            high = torch.nn.functional.linear(hi, pos, module.bias)+torch.nn.functional.linear(lo, neg)
            return low, high, float(w.abs().sum(1).max())
        raise ValueError('unsupported priority operator; do not invent a score')
    widths = [hi-lo for lo, hi in domain.intervals]
    differences, lipschitz = [], []
    with torch.no_grad():
        lo, hi, _ = interval(getattr(model, 'stem', None), lower.flatten(1), upper.flatten(1))
        for layer, (a, b), (gate_lo, gate_hi) in zip(model.layers, domain.history, domain.intervals):
            tick(deadline)
            al, ah, ka = interval(layer.experts[a], lo, hi)
            bl, bh, kb = interval(layer.experts[b], lo, hi)
            differences.append(float(torch.maximum((al-bh).abs(), (ah-bl).abs()).max()))
            lipschitz.append(max(ka, kb)+(1. if getattr(model, 'residual', False) else 0.))
            ml = torch.minimum(gate_lo*al+(1-gate_lo)*bl, gate_hi*al+(1-gate_hi)*bl)
            mh = torch.maximum(gate_lo*ah+(1-gate_lo)*bh, gate_hi*ah+(1-gate_hi)*bh)
            lo, hi = (lo+ml, hi+mh) if getattr(model, 'residual', False) else (ml, mh)
        _, _, head_lip = interval(getattr(model, 'head', None), lo, hi)
        norm = float(rows.abs().sum(1).max())*head_lip
    scores = propagated_gate_scores(widths, differences, lipschitz, norm)
    selected = max(range(len(scores)), key=lambda i: (scores[i], -i))
    tick(deadline)
    return dict(source_id=domain.source_id, parent_domain_sha256=domain_digest(domain),
        widths=widths, difference_absolute_amplitudes=differences,
        layer_lipschitz=lipschitz, property_head_norm=norm, scores=list(scores), selected_layer=selected,
        tie_rule='lowest layer index', heuristic_only=True, certified_error_bound=False)


def conditional_bounds(graph, center, lower, upper, *, deadline,
                       use_guards=False, iterations=20):
    """CROWN support of [I, -beta], jointly optimized with author alpha tensors.

    This uses the author's differentiable CROWN/alpha implementation. It does
    not patch INVPROP (whose pinned intermediate-only API failed the direct
    target control) or introduce a new abstract domain. An outer owned request
    supervisor enforces the hard deadline; every evaluation also checks it.
    """
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    tick(deadline)
    start = time.monotonic()
    if type(use_guards) is not bool or type(iterations) is not int or iterations < 1:
        raise ValueError('explicit mode and finite iteration count')
    if (center.shape != lower.shape or center.shape != upper.shape or center.shape[0] != 1
            or any(x.dtype != torch.float64 or x.device.type != 'cpu'
                   or not torch.isfinite(x).all() for x in (center, lower, upper))
            or not torch.all(lower <= center) or not torch.all(center <= upper)):
        raise ValueError('one finite CPU/float64 materialized input box')
    k, g = graph.objective_count, graph.guard_count
    bounded = BoundedModule(graph, center, device='cpu')
    def inputs():
        return (BoundedTensor(center.clone(), PerturbationLpNorm(norm=float('inf'),
            x_L=lower.clone(), x_U=upper.clone())),)
    eye = torch.eye(k, dtype=torch.float64)
    # Initialize every guard-only activation as well as the objectives. Nonzero
    # initialization coefficients are for allocating alpha tensors, not an
    # accepted candidate. Off/on use exactly this same initialization.
    initial_C = torch.cat((eye, -torch.ones((k, g), dtype=torch.float64)), 1).unsqueeze(0)
    bounded.init_alpha(inputs(), c=initial_C)
    tick(deadline)
    nodes = bounded.get_enabled_opt_act()
    alphas = []
    for node in nodes:
        node.opt_start()
        for a in node.alpha.values():
            a.requires_grad_()
            alphas.append(a)
    beta = torch.zeros((k, g), dtype=torch.float64, requires_grad=use_guards and g > 0)
    groups = []
    if alphas:
        groups.append(dict(params=alphas, lr=.5))
    if beta.requires_grad:
        groups.append(dict(params=[beta], lr=.1))
    optimizer = torch.optim.Adam(groups) if groups else None
    params = alphas + ([beta] if beta.requires_grad else [])
    best = zero = None
    selected_beta = torch.zeros_like(beta)
    evaluated = 0
    try:
        for iteration in range(iterations):
            tick(deadline)
            if not torch.isfinite(beta).all() or (beta < 0).any():
                raise ValueError('finite nonnegative guard multipliers required')
            C = torch.cat((eye, -beta), 1).unsqueeze(0)
            lb, _ = bounded.compute_bounds(x=inputs(), C=C, method='CROWN',
                bound_upper=False, reuse_alpha=True)
            tick(deadline)
            evaluated += 1
            candidate = lb[0].detach()
            if not torch.isfinite(candidate).all():
                raise ValueError('nonfinite conditional CROWN candidate')
            if best is None:
                zero = candidate.clone()
                best = candidate.clone()
            better = candidate > best
            selected_beta[better] = beta.detach()[better]
            best = torch.maximum(best, candidate)
            if optimizer is None or iteration+1 == iterations:
                break
            if iteration == 0 and beta.requires_grad:
                # Author CROWN skips exactly-zero C columns (including their
                # gradient paths). Keep the evaluated zero candidate, then use
                # a fixed positive start so every guard can receive a gradient.
                # This is optimizer initialization, not a soundness assumption.
                with torch.no_grad():
                    beta.fill_(.1)
                continue
            loss = -lb.sum()
            if not loss.requires_grad:
                break
            # Do not accumulate gradients into the caller's model parameters.
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            optimizer.zero_grad(set_to_none=True)
            for param, grad in zip(params, grads):
                if grad is not None and not torch.isfinite(grad).all():
                    raise ValueError('nonfinite candidate gradient')
                param.grad = grad
            optimizer.step()
            with torch.no_grad():
                beta.clamp_(min=0.)
            for node in nodes:
                node.clip_alpha()
    finally:
        for node in nodes:
            node.opt_end()
    tick(deadline)
    return dict(source_id=graph.source_id, history=[list(p) for p in graph.history],
        cells=[dict(layer=c.layer, lower=c.lower, upper=c.upper) for c in graph.cells],
        lower=best.tolist(), zero_lower=zero.tolist(), objective_count=k, guard_count=g,
        initialization_calls=1, candidate_evaluations=evaluated, use_guards=use_guards,
        guard_start=.1 if use_guards and g else None,
        beta_nonzero=int((selected_beta > 0).sum()), beta_count=selected_beta.numel(),
        selected_beta=selected_beta.tolist(), alpha_parameters=sum(a.numel() for a in alphas),
        engine_adapter='differentiable_CROWN_joint_alpha_guard_beta',
        seconds=time.monotonic()-start, complete_MoE=False, formal_SAFE=False,
        evidence_grade='GUARD_CONDITIONAL_CROWN_NUMERICAL_LOWER' if use_guards and g
                       else 'WHOLE_BOX_CROWN_NUMERICAL_LOWER')
