"""Opt-in hard-core diagnostic graphs; never an automatic production fallback.

V2 removes gate/input equalities while retaining the same experts and guards.
Only the last gate can be endpointized through the registered affine suffix.
V3 fixes all gates and makes no containment claim about the original network.
Prefix bounds and sigmoid endpoints retain the numerical-policy guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time


VARIANTS = ('V2_LO', 'V2_HI', 'V3_MIDPOINT')


def canonical_history(history, experts):
    if not history or len(history) != len(experts):
        raise ValueError('complete nonempty history required')
    result = []
    for pair, count in zip(history, experts):
        if (type(count) is not int or count < 2 or len(pair) != 2
                or any(type(i) is not int for i in pair)
                or not 0 <= pair[0] < pair[1] < count):
            raise ValueError('canonical selected-softmax top2 pair')
        result.append(tuple(pair))
    return tuple(result)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def sigmoid_interval(lo, hi):
    """Policy-level padding, NOT a certified transcendental enclosure."""
    if not _finite(lo) or not _finite(hi) or lo > hi:
        raise ValueError('finite ordered margin bounds required')
    def sigmoid(x):
        if x >= 0:
            return 1. / (1. + math.exp(-x))
        value = math.exp(x)
        return value / (1. + value)
    return (max(0., math.nextafter(sigmoid(lo), -math.inf)),
            min(1., math.nextafter(sigmoid(hi), math.inf)))


@dataclass(frozen=True)
class GateDomain:
    source_id: str
    history: tuple
    margins: tuple
    intervals: tuple


def domain_from_prefix_bounds(history, experts, records, *, source_id, expected_source_id,
                              allow_conditional=False):
    """Bind same-request prefix CROWN bounds to canonical first-expert weights.

    ``source_id`` binds checkpoint/input/property/source-version at the caller.
    Equality here is an identity check, not independent proof of CROWN bounds.
    Missing bounds are explicit failures, never invented constant gate values.
    """
    if (not isinstance(source_id, str) or len(source_id) != 64
            or any(c not in '0123456789abcdef' for c in source_id)
            or source_id != expected_source_id):
        raise ValueError('same-object prefix source binding')
    if type(allow_conditional) is not bool:
        raise ValueError('explicit conditional-prefix policy')
    history = canonical_history(history, experts)
    prefixes = {}
    for record in records:
        prefix = tuple(tuple(p) for p in record['prefix'])
        if prefix in prefixes:
            raise ValueError('duplicate prefix record')
        prefixes[prefix] = record
    margins = []
    for layer, (a, b) in enumerate(history):
        record = prefixes.get(history[:layer])
        if record is None:
            raise ValueError('missing or incompatible original-prefix range')
        if record.get('guarantee') == 'GUARD_CONDITIONAL_CROWN_NUMERICAL_UPPER' and allow_conditional:
            # The complete original history satisfies every prior guard. The
            # caller additionally checks model/input/backend support binding;
            # do not relabel a conditional bound as a whole-box bound.
            support = record.get('guard_support', {})
            if (not layer or tuple(map(tuple, support.get('history', ()))) != history[:layer]
                    or support.get('cells') != [] or support.get('use_guards') is not True):
                raise ValueError('conditional range must use exactly prior history guards')
        elif record.get('guarantee') != 'WHOLE_BOX_CROWN_NUMERICAL_UPPER':
            raise ValueError('missing or incompatible original-prefix range')
        values = {}
        for i, j, value in record['upper']:
            if (type(i) is not int or type(j) is not int or i == j
                    or not 0 <= i < experts[layer] or not 0 <= j < experts[layer]
                    or (i, j) in values):
                raise ValueError('invalid/duplicate ordered margin')
            values[i, j] = value
        upper, reverse = values.get((a, b)), values.get((b, a))
        if not _finite(upper) or not _finite(reverse):
            raise ValueError('both finite margin directions required')
        lo, hi = -reverse, upper
        sigmoid_interval(lo, hi)  # Reject inverted ranges before graph creation.
        margins.append((lo, hi))
    return GateDomain(source_id, history, tuple(margins),
                      tuple(sigmoid_interval(*m) for m in margins))


def validate_domain(domain, history, *, expected_source_id):
    if (not isinstance(domain, GateDomain) or domain.source_id != expected_source_id
            or not isinstance(domain.source_id, str) or len(domain.source_id) != 64
            or any(c not in '0123456789abcdef' for c in domain.source_id)
            or domain.history != tuple(map(tuple, history))
            or len(domain.margins) != len(history)
            or len(domain.intervals) != len(history)
            or domain.intervals != tuple(sigmoid_interval(*m) for m in domain.margins)):
        raise ValueError('gate domain/source/history mismatch')


def domain_digest(domain):
    return hashlib.sha256(json.dumps(dict(source_id=domain.source_id, history=domain.history,
        margins=domain.margins, intervals=domain.intervals), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def fixed_weights(domain, variant):
    if variant not in VARIANTS:
        raise ValueError('explicit diagnostic variant required')
    if variant == 'V3_MIDPOINT':
        return tuple(lo + (hi-lo)/2 for lo, hi in domain.intervals)
    endpoint = 0 if variant == 'V2_LO' else 1
    return (None,) * (len(domain.history)-1) + (domain.intervals[-1][endpoint],)


def build_graph(model, history, rows, thresholds, domain, *, variant,
                input_features, expected_source_id):
    """Flat input [original x, inner independent gates] for V2; just x for V3.

    This constructs no candidate and invokes no solver. Runtime receipt/source
    checks and both-endpoint aggregation remain the caller's responsibility.
    Every output is max(property, all outsider-minus-selected guard atoms).
    """
    import torch
    from torch import nn
    if any(m.training for m in model.modules()):
        raise ValueError('eval object required')
    if any(p.device.type != 'cpu' or p.dtype != torch.float64 for p in model.parameters()):
        raise ValueError('registered CPU/float64 object required')
    if type(input_features) is not int or input_features <= 0:
        raise ValueError('positive original input dimension')
    if hasattr(model, 'layers'):
        layers = tuple(model.layers)
        if any(l.k != 2 or l.mode != 'selected_softmax' for l in layers):
            raise ValueError('only selected-softmax top2 layers')
        stem, head = getattr(model, 'stem', None), getattr(model, 'head', None)
        residual = getattr(model, 'residual', False)
    else:
        from act.back_end.moe.schema import GateKind
        if (model.spec.top_k != 2 or model.spec.gate != GateKind.SELECTED_SOFTMAX
                or not model.spec.normalized or model.shared_expert is not None):
            raise ValueError('registered output-level top2 object required')
        layers, stem, head, residual = (model,), None, None, False
    if type(residual) is not bool or (head is not None and type(head) not in (nn.Linear, nn.Identity)):
        raise ValueError('endpoint rule requires an affine final suffix')
    experts = tuple(len(layer.experts) for layer in layers)
    history = canonical_history(history, experts)
    validate_domain(domain, history, expected_source_id=expected_source_id)
    weights = fixed_weights(domain, variant)
    if (rows.ndim != 2 or not len(rows) or thresholds.shape != (len(rows),)
            or rows.dtype != torch.float64 or thresholds.dtype != torch.float64
            or rows.device.type != 'cpu' or thresholds.device.type != 'cpu'
            or not torch.isfinite(rows).all() or not torch.isfinite(thresholds).all()):
        raise ValueError('complete finite CPU/float64 property matrix/constants')
    components = ([stem] if stem is not None else []) + ([head] if head is not None else [])
    components += [l.router for l in layers]
    components += [e for l in layers for e in l.experts]
    allowed = (nn.Sequential, nn.Linear, nn.ReLU, nn.Flatten, nn.Identity)
    if any(type(m) not in allowed for component in components for m in component.modules()):
        raise ValueError('registered affine/ReLU MLP family only')

    class GateHistory(nn.Module):
        def __init__(self):
            super().__init__()
            self.routers = nn.ModuleList([l.router for l in layers])
            self.a = nn.ModuleList([l.experts[p[0]] for l, p in zip(layers, history)])
            self.b = nn.ModuleList([l.experts[p[1]] for l, p in zip(layers, history)])
            self.stem, self.head = stem, head
            self.register_buffer('rows', rows.detach().clone())
            self.register_buffer('constants', -thresholds.detach().clone())
            self.variant, self.source_id = variant, domain.source_id
            self.input_features = input_features
            self.independent_gate_count = sum(w is None for w in weights)
            self.history, self.weights = history, weights
            self.formal_SAFE = False

        def _run(self, flat):
            x = flat[:, :input_features]
            if self.stem is not None:
                x = self.stem(x)
            scores = []
            variable = 0
            for i, fixed in enumerate(weights):
                scores.append(self.routers[i](x))
                if fixed is None:
                    gate = flat[:, input_features+variable:input_features+variable+1]
                    variable += 1
                else:
                    gate = fixed
                ea, eb = self.a[i](x), self.b[i](x)
                mixed = eb + gate * (ea-eb)
                x = x+mixed if residual else mixed
            if self.head is not None:
                x = self.head(x)
            return x, scores

        def forward(self, flat):
            x, scores = self._run(flat)
            values = torch.nn.functional.linear(x, self.rows, self.constants)
            for pair, r, count in zip(history, scores, experts):
                for selected in pair:
                    for outside in range(count):
                        if outside not in pair:
                            atom = r[:, outside:outside+1]-r[:, selected:selected+1]
                            values = values+torch.relu(atom-values)
            return values

    return GateHistory().eval()


def extend_box(center, lower, upper, domain, *, variant, expected_source_id):
    import torch
    validate_domain(domain, domain.history, expected_source_id=expected_source_id)
    fixed_weights(domain, variant)
    if (center.shape != lower.shape or center.shape != upper.shape or center.ndim < 2
            or center.shape[0] != 1 or any(t.device.type != 'cpu' or t.dtype != torch.float64
                or not torch.isfinite(t).all() for t in (center, lower, upper))
            or not ((lower <= center) & (center <= upper)).all()):
        raise ValueError('one finite registered box')
    flat = [t.flatten(1).detach().clone() for t in (center, lower, upper)]
    if variant.startswith('V2') and len(domain.intervals) > 1:
        ranges = domain.intervals[:-1]
        more = ([a+(b-a)/2 for a, b in ranges], [a for a, _ in ranges], [b for _, b in ranges])
        flat = [torch.cat((t, torch.tensor([v], dtype=torch.float64)), dim=1) for t, v in zip(flat, more)]
    return dict(center=flat[0], lower=flat[1], upper=flat[2])


def receive_diagnosis(domain, group, records, *, property_count, expected_source_id, deadline):
    """Local-history aggregation, never complete-MoE SAFE or original UNSAFE.

    Each solver outcome must cover every property on its endpoint graph.
    ``falsified`` on a relaxed/counterfactual graph is still not original UNSAFE.
    Source loading and raw-result binding are additionally required at the worker.
    """
    if not _finite(deadline):
        raise ValueError('finite diagnostic deadline')
    if time.monotonic() >= deadline:
        raise TimeoutError('same diagnostic reception deadline')
    validate_domain(domain, domain.history, expected_source_id=expected_source_id)
    if group not in ('V2', 'V3') or type(property_count) is not int or property_count <= 0:
        raise ValueError('diagnostic group and all properties required')
    expected = ('V2_LO', 'V2_HI') if group == 'V2' else ('V3_MIDPOINT',)
    raw = [([[float(j == i) for j in range(property_count)]], [0.]) for i in range(property_count)]
    # Accept the ordinary JSON array form, not a differently grouped AND spec.
    raw = json.loads(json.dumps(raw))
    seen = {}
    for record in records:
        variant = record['variant']
        if variant not in expected or variant in seen:
            raise ValueError('wrong/duplicate endpoint')
        if (record['domain_sha256'] != domain_digest(domain)
                or record['source_id'] != expected_source_id):
            raise ValueError('diagnostic result bound to another source/domain')
        outcome = record['outcome']
        if outcome['status'] not in ('verified', 'unknown', 'falsified'):
            raise ValueError('unexpected solver outcome')
        if json.loads(json.dumps(outcome['actual_raw_spec'])) != raw:
            raise ValueError('missing/reordered/combined property alternatives')
        seen[variant] = outcome['status']
    closed = set(seen) == set(expected) and all(s == 'verified' for s in seen.values())
    if time.monotonic() >= deadline:
        raise TimeoutError('same diagnostic reception deadline')
    return dict(status='POSITIVE' if closed and group == 'V2' else
                'DIAGNOSTIC_CLOSED' if closed else 'UNKNOWN',
                group=group, closed=closed, original_history_closed=closed and group == 'V2',
                expected=list(expected), received=list(seen), missing=[v for v in expected if v not in seen],
                evidence_grade='GENBAB_NUMERICAL_HISTORY_FILTER' if closed and group == 'V2' else
                'COUNTERFACTUAL_MIDPOINT_ONLY' if closed else 'NONE',
                formal_SAFE=False, complete_MoE=False, original_UNSAFE=False)
