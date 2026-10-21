"""Sequential top-2 static histories and independent generic prefix pruning.

The external frontends do not import ACT history coverage or HZ. Prefix CROWN
bounds are on the original box, ignoring preceding guards conservatively.
This module makes numerical claims only; exact reachable-history counts are
not inferred from an unpruned prefix tree.
"""
from __future__ import annotations
import itertools
import math
import time
import torch
from torch import nn


class SequentialTop2(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.layers:
            x = layer(x)
        return x


class StaticPrefix(nn.Module):
    """Independent static compiler, with the next router as the prefix output."""
    def __init__(self, model, history, *, next_router=False):
        super().__init__()
        if len(history) > len(model.layers) or (next_router and len(history) >= len(model.layers)):
            raise ValueError('prefix depth')
        self.routers = nn.ModuleList()
        self.expert_a, self.expert_b = nn.ModuleList(), nn.ModuleList()
        self.history = tuple(tuple(pair) for pair in history)
        self.experts = tuple(len(layer.experts) for layer in model.layers)
        for i, pair in enumerate(self.history):
            layer = model.layers[i]
            if (layer.mode != 'selected_softmax' or layer.k != 2 or len(pair) != 2
                    or any(type(v) is not int for v in pair) or not 0 <= pair[0] < pair[1] < len(layer.experts)):
                raise ValueError('canonical selected-softmax top2 history')
            self.routers.append(layer.router)
            self.expert_a.append(layer.experts[pair[0]])
            self.expert_b.append(layer.experts[pair[1]])
        self.next_router = model.layers[len(history)].router if next_router else None

    def _run(self, x):
        x = x.flatten(1)
        scores = []
        for i, pair in enumerate(self.history):
            r = self.routers[i](x)
            a, b = pair
            weight = torch.sigmoid(r[:, a:a+1] - r[:, b:b+1])
            ea, eb = self.expert_a[i](x), self.expert_b[i](x)
            x = eb + weight * (ea - eb)
            scores.append(r)
        return x, scores

    def forward(self, x):
        x, _ = self._run(x)
        return self.next_router(x) if self.next_router is not None else x


class GuardedHistory(StaticPrefix):
    def __init__(self, model, history, rows, thresholds):
        if len(history) != len(model.layers):
            raise ValueError('complete static history required')
        super().__init__(model, history)
        if rows.ndim != 2 or thresholds.shape != (len(rows),):
            raise ValueError('linear property shape')
        self.register_buffer('rows', rows.detach().clone())
        self.register_buffer('constants', -thresholds.detach().clone())

    def forward(self, x):
        out, scores = self._run(x)
        values = torch.nn.functional.linear(out, self.rows, self.constants)
        for pair, r, experts in zip(self.history, scores, self.experts):
            for selected in pair:
                for outside in range(experts):
                    if outside not in pair:
                        atom = r[:, outside:outside+1] - r[:, selected:selected+1]
                        values = values + torch.relu(atom-values)
        return values


def router_upper_bounds(model, prefix, center, lower, upper, deadline):
    """All ordered margin upper bounds, independently computed by plain CROWN."""
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    if time.monotonic() >= deadline:
        raise TimeoutError('prefix deadline')
    graph = StaticPrefix(model, prefix, next_router=True).eval()
    experts = len(model.layers[len(prefix)].experts)
    pairs = tuple((i, j) for i in range(experts) for j in range(experts) if i != j)
    rows = torch.zeros((1, len(pairs), experts), dtype=torch.float64)
    for k, (i, j) in enumerate(pairs):
        rows[0, k, i], rows[0, k, j] = 1., -1.
    bounded = BoundedModule(graph, center, device='cpu')
    x = BoundedTensor(center, PerturbationLpNorm(norm=float('inf'), x_L=lower, x_U=upper))
    _, hi = bounded.compute_bounds(x=(x,), C=rows, method='CROWN')
    if time.monotonic() >= deadline:
        raise TimeoutError('prefix bound reception deadline')
    return {pair: float(hi[0, k]) for k, pair in enumerate(pairs)}


def excluded_pair(pair, experts, bounds, tolerance=1e-9):
    """Strict dominance only: zero/tie, missing and nonfinite bounds stay live."""
    for i in pair:
        for j in range(experts):
            value = bounds.get((i, j))
            if j not in pair and value is not None and math.isfinite(value) and value < -tolerance:
                return dict(selected=i, outside=j, upper=value, tolerance=tolerance)
    return None


def verify_histories(model, *, center, lower, upper, rows, thresholds, pruning, deadline,
                     verify_graph, bound_router=router_upper_bounds, publish=lambda _: None):
    """One shared request budget, complete prefix-tree coverage or explicit missing."""
    from act.back_end.moe.class_separated_top1 import validate_replay
    if type(pruning) is not bool:
        raise ValueError('explicit pruning flag')
    experts = tuple(len(layer.experts) for layer in model.layers)
    counts = tuple(math.comb(e, 2) for e in experts)
    expected = math.prod(counts)
    result = dict(status='UNKNOWN', expected_histories=expected, verified_histories=0,
                  pruned_histories=0, attempted_histories=0, prefix_queries=0,
                  histories=[], exclusions=[], prefix_bounds=[], complete=False,
                  evidence_grade='GENBAB_NUMERICAL_FILTER', formal_SAFE=False)
    def tick():
        if time.monotonic() >= deadline:
            raise TimeoutError('shared request deadline')
    def walk(prefix):
        tick()
        depth = len(prefix)
        if depth == len(experts):
            graph = GuardedHistory(model, prefix, rows, thresholds).eval()
            result['attempted_histories'] += 1
            publish(result)
            outcome, attacks = verify_graph(graph, lower, upper, len(rows), deadline)
            record = dict(history=[list(p) for p in prefix], outcome=outcome)
            result['histories'].append(record)
            if outcome['status'] == 'verified':
                result['verified_histories'] += 1
            if attacks is not None:
                for point in attacks.detach().cpu().double().reshape(-1, *center.shape[1:]):
                    tick()
                    if validate_replay(model, point.reshape_as(center), lower, upper, rows, thresholds):
                        result.update(status='UNSAFE_REPLAYED', witness=point.reshape_as(center).tolist())
                        return
            publish(result)
            return
        bounds = {}
        if pruning:
            before = time.monotonic()
            bounds = bound_router(model, prefix, center, lower, upper, deadline)
            result['prefix_queries'] += 1
            result['prefix_bounds'].append(dict(prefix=[list(p) for p in prefix],
                upper=[[i,j,value if math.isfinite(value) else None] for (i,j),value in bounds.items()],
                seconds=time.monotonic()-before, guarantee='WHOLE_BOX_CROWN_NUMERICAL_UPPER'))
        for pair in itertools.combinations(range(experts[depth]), 2):
            tick()
            reason = excluded_pair(pair, experts[depth], bounds)
            if reason is not None:
                covered = math.prod(counts[depth+1:])
                result['pruned_histories'] += covered
                result['exclusions'].append(dict(prefix=[list(p) for p in (*prefix, pair)], covered=covered, reason=reason))
            else:
                walk((*prefix, pair))
            publish(result)
            if result['status'] == 'UNSAFE_REPLAYED':
                return
    tick()
    if validate_replay(model, center, lower, upper, rows, thresholds):
        result.update(status='UNSAFE_REPLAYED', witness=center.tolist())
    else:
        try:
            walk(())
        except TimeoutError:
            result['status'] = 'TIMEOUT'
        result['complete'] = result['verified_histories'] + result['pruned_histories'] == expected
        if result['complete'] and result['status'] != 'UNSAFE_REPLAYED':
            result['status'] = 'POSITIVE'
    result['unvisited_histories'] = expected-result['pruned_histories']-result['attempted_histories']
    result['unclosed_histories'] = expected-result['pruned_histories']-result['verified_histories']
    if time.monotonic() >= deadline:
        result.update(status='TIMEOUT', complete=False)
    if result['status'] == 'UNSAFE_REPLAYED':
        result['evidence_grade'] = 'FULL_MODEL_REPLAY'
    elif result['status'] != 'POSITIVE':
        result['evidence_grade'] = 'NONE'
    return result
