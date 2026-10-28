"""Opt-in, baseline-preserving guard refinement on the existing CROWN engine.

The extra optimization is charged. Pointwise retention is an arithmetic
property, not finite-budget dominance over another verifier. Bounds have the
same numerical-policy grade as CROWN; no upstream proof gap is repaired here.
"""
from __future__ import annotations

import hashlib
import math
import time

import torch

from .moe_aware_bab import tick


class CandidateEnvelope:
    """Keep a separately evaluated witness multiplier for every objective."""
    def __init__(self, properties, guards):
        self.lower = None
        self.beta = torch.zeros((properties, guards), dtype=torch.float64)
        self.origin = [None] * properties

    def add(self, lower, beta, origin):
        lower, beta = lower.detach(), beta.detach()
        if (lower.shape != (len(self.origin),) or beta.shape != self.beta.shape
                or not torch.isfinite(lower).all() or not torch.isfinite(beta).all()
                or (beta < 0).any()):
            raise ValueError('finite complete bound and nonnegative multiplier rows')
        better = torch.ones_like(lower, dtype=torch.bool) if self.lower is None else lower > self.lower
        self.beta[better] = beta[better]
        for i in better.nonzero().flatten().tolist():
            self.origin[i] = origin
        self.lower = lower.clone() if self.lower is None else torch.maximum(self.lower, lower)


def _alpha_identity(named):
    value = hashlib.sha256()
    for name, alpha in named:
        value.update(repr((name, tuple(alpha.shape), str(alpha.dtype))).encode())
        value.update(alpha.detach().contiguous().numpy().tobytes())
    return value.hexdigest()


def refined_conditional_bounds(graph, center, lower, upper, *, deadline,
                               use_guards=True, baseline_evaluations=20,
                               refinement_updates=40, beta_probe=1e-4,
                               report=None):
    """Exact old alpha-only prefix, then a reset-Adam refinement from alpha*.

    alpha* is the earliest evaluated prefix state maximizing the sum of all
    property bounds (the old optimizer's objective). This is one actual state,
    not a fabricated mix of per-property alpha layouts. The entire prefix's
    per-property envelope is retained, including rows not best at alpha*.

    Guards start at zero. CROWN may omit zero-C gradient paths: one fixed small
    positive probe evaluates that derivative, then beta is reset to zero BEFORE
    applying the gradient update. The probe is itself compared as a valid
    candidate. Forty updates plus their final evaluation are explicit costs.
    use_guards=False is the equal-extra-alpha-work attribution control.
    """
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

    tick(deadline)
    began = time.monotonic()
    if (type(use_guards) is not bool or type(baseline_evaluations) is not int
            or baseline_evaluations < 1 or type(refinement_updates) is not int
            or refinement_updates < 0 or not math.isfinite(beta_probe) or beta_probe <= 0):
        raise ValueError('explicit finite optimization design')
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
    initial_c = torch.cat((eye, -torch.ones((k, g), dtype=torch.float64)), 1).unsqueeze(0)
    bounded.init_alpha(inputs(), c=initial_c)  # Same allocation as the old off.
    tick(deadline)
    nodes, named = bounded.get_enabled_opt_act(), []
    for node in nodes:
        node.opt_start()
        for key, alpha in node.alpha.items():
            alpha.requires_grad_()
            named.append(((node.name, key), alpha))
    alphas = [a for _, a in named]
    beta = torch.zeros((k, g), dtype=torch.float64)
    envelope = CandidateEnvelope(k, g)
    trace = []
    calls = dict(baseline=0, refinement=0, probe=0)

    def evaluate(phase, iteration):
        tick(deadline)
        if not torch.isfinite(beta).all() or (beta < 0).any():
            raise ValueError('nonnegative finite beta before CROWN')
        c = torch.cat((eye, -beta), 1).unsqueeze(0)
        lb, _ = bounded.compute_bounds(x=inputs(), C=c, method='CROWN',
            bound_upper=False, reuse_alpha=True)
        tick(deadline)
        envelope.add(lb[0], beta, f'{phase}:{iteration}')
        calls[phase] += 1
        trace.append(dict(phase=phase, iteration=iteration, lower=lb[0].detach().tolist(),
            beta_nonzero=int((beta.detach() > 0).sum())))
        return lb

    def update(lb, parameters, optimizer):
        if optimizer is None or not lb.requires_grad:
            return False
        gradients = torch.autograd.grad(-lb.sum(), parameters, allow_unused=True)
        optimizer.zero_grad(set_to_none=True)
        for param, grad in zip(parameters, gradients):
            if grad is not None and not torch.isfinite(grad).all():
                raise ValueError('finite candidate gradients')
            param.grad = grad
        optimizer.step()
        with torch.no_grad():
            beta.clamp_(min=0.)
        for node in nodes:
            node.clip_alpha()
        return True

    try:
        optimizer = torch.optim.Adam([dict(params=alphas, lr=.5)]) if alphas else None
        best_sum, alpha_star, star_iteration = -math.inf, None, None
        # Preserve the old off prefix, including its constant/no-gradient exit.
        for iteration in range(baseline_evaluations):
            lb = evaluate('baseline', iteration)
            score = sum(trace[-1]['lower'])
            if not math.isfinite(score):
                raise ValueError('finite alpha-star selection objective')
            if score > best_sum:
                best_sum, star_iteration = score, iteration
                alpha_star = [a.detach().clone() for a in alphas]
            if optimizer is None or iteration + 1 == baseline_evaluations:
                break
            if not update(lb, alphas, optimizer):
                break
        baseline = envelope.lower.clone()
        prefix_seconds = time.monotonic() - began
        with torch.no_grad():
            for alpha, saved in zip(alphas, alpha_star):
                alpha.copy_(saved)
        star_hash = _alpha_identity(named)
        if report is not None:
            report('baseline', dict(lower=baseline.tolist(), alpha_star_iteration=star_iteration,
                alpha_star_sha256=star_hash, evaluations=calls['baseline'], seconds=prefix_seconds))
        tick(deadline)
        beta.requires_grad_(use_guards and g > 0)
        groups = ([dict(params=alphas, lr=.5)] if alphas else [])
        if beta.requires_grad:
            groups.append(dict(params=[beta], lr=.1))
        optimizer = torch.optim.Adam(groups) if groups else None
        parameters = alphas + ([beta] if beta.requires_grad else [])
        refinement_only = CandidateEnvelope(k, g)
        completed_updates = 0
        for iteration in range(refinement_updates + 1):
            lb = evaluate('refinement', iteration)
            refinement_only.add(lb[0], beta, f'refinement:{iteration}')
            if iteration == refinement_updates:
                break
            if iteration == 0 and beta.requires_grad:
                with torch.no_grad():
                    beta.fill_(beta_probe)
                lb = evaluate('probe', 0)
                refinement_only.add(lb[0], beta, 'probe:0')
                # Compute before reset: no in-place mutation of an autograd input.
                gradients = (torch.autograd.grad(-lb.sum(), parameters, allow_unused=True)
                             if lb.requires_grad else [None] * len(parameters))
                optimizer.zero_grad(set_to_none=True)
                for param, grad in zip(parameters, gradients):
                    if grad is not None and not torch.isfinite(grad).all():
                        raise ValueError('finite probe gradients')
                    param.grad = grad
                with torch.no_grad():
                    beta.zero_()
                optimizer.step()
                with torch.no_grad():
                    beta.clamp_(min=0.)
                for node in nodes:
                    node.clip_alpha()
                completed_updates += 1
            else:
                completed_updates += int(update(lb, parameters, optimizer))
        tick(deadline)
        if not torch.all(envelope.lower >= baseline):
            raise ValueError('complete baseline envelope must survive refinement')
        return dict(source_id=graph.source_id, history=[list(p) for p in graph.history],
            cells=[dict(layer=c.layer, lower=c.lower, upper=c.upper) for c in graph.cells],
            lower=envelope.lower.tolist(), zero_lower=baseline.tolist(),
            baseline_lower=baseline.tolist(), refinement_lower=refinement_only.lower.tolist(),
            improvement=(envelope.lower-baseline).tolist(), selected_origin=envelope.origin,
            selected_beta=envelope.beta.tolist(), beta_nonzero=int((envelope.beta > 0).sum()),
            beta_count=envelope.beta.numel(), objective_count=k, guard_count=g,
            use_guards=use_guards, initialization_calls=1, evaluations=calls,
            candidate_evaluations=sum(calls.values()), refinement_updates=completed_updates,
            baseline_limit=baseline_evaluations, refinement_update_limit=refinement_updates,
            beta_start=0., beta_probe=beta_probe if beta.requires_grad else None,
            alpha_star_iteration=star_iteration, alpha_star_sha256=star_hash,
            alpha_star_rule='earliest evaluated prefix state maximizing sum of property bounds',
            baseline_seconds=prefix_seconds, seconds=time.monotonic()-began, trace=trace,
            engine_adapter='baseline_preserving_CROWN_guard_refinement_v1',
            complete_MoE=False, formal_SAFE=False,
            evidence_grade='GUARD_CONDITIONAL_CROWN_NUMERICAL_LOWER' if use_guards and g
                           else 'WHOLE_BOX_CROWN_NUMERICAL_LOWER')
    finally:
        for node in nodes:
            node.opt_end()
