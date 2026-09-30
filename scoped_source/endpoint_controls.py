"""Frozen rational three-expert control, not a trained/native model adapter.

Manual duals and an exact feasible McCormick point replace numerical solving.
All three unordered pairs are retained, including the empty {0,2} guard.
"""
from fractions import Fraction as F
from itertools import combinations
import json
from pathlib import Path
import time

from scoped_source.endpoint_build import build
from scoped_source.endpoint_check import check, THRESHOLD
from scoped_source.sparse_ir import row, lp_record
from source_enclosure.format import identity
from upstream_source.checker import csr

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'configs/h2_endpoint_algebra_20260930.json'
REGISTERED_PROTOCOL_SHA256 = '11345d6ad4dcc92b9bc014391f55877cc51b91461d0f7245d04ab2247e958b97'


def fixture(protocol=None):
    protocol = json.loads(PROTOCOL.read_text()) if protocol is None else protocol
    # This is a fixed analytic fixture, not a configurable source adapter.
    # Bind even fields that are descriptive rather than used as loop controls.
    if identity(protocol) != REGISTERED_PROTOCOL_SHA256:
        raise ValueError('not the registered analytic protocol')
    if F(protocol['acceptance_threshold']) != THRESHOLD:
        raise ValueError('registered acceptance gate differs from checker')
    source = protocol['source']
    names = ['x'] + [v for i in range(3) for v in (f'p{i}', f'n{i}')]
    bounds = {'x': tuple(map(F, source['input']))}
    bounds.update({v: (F(0), F(1)) for v in names[1:]})
    scores = [tuple(map(F, v)) for v in source['router']]
    c = F(source['expert_margin_constant']); forms = []
    for i, value in enumerate(source['expert_input_coefficients']):
        coefficients = {'x': F(value), f'p{i}': F(source['expert_relu_coefficients'][0]),
                        f'n{i}': F(source['expert_relu_coefficients'][1])}
        forms.append({'c': [str(coefficients.get(v, F(0))) for v in names], 'offset': str(c)})
    rows = []; tags = {}

    def add(tag, coefficients, rhs):
        tags[tag] = len(rows); rows.append(row('le', coefficients, rhs))

    for i in range(3):
        p, n = f'p{i}', f'n{i}'
        add(p, {p: F(-1)}, F(0))
        add(p+'-x', {'x': F(1), p: F(-1)}, F(0))
        add(p+'-upper', {p: F(1), 'x': F(-1, 2)}, F(1, 2))
        add(n, {n: F(-1)}, F(0))
        add(n+'+x', {'x': F(-1), n: F(-1)}, F(0))
        add(n+'-upper', {n: F(1), 'x': F(1, 2)}, F(1, 2))
    duties = []
    for pair, gate in zip(combinations(range(3), 2), protocol['pair_gate_intervals']):
        guards = []
        for selected in pair:
            for outside in range(3):
                if outside in pair:
                    continue
                coefficient = scores[outside][0]-scores[selected][0]
                offset = scores[outside][1]-scores[selected][1]
                guards.append(row('le', {'x': coefficient}, -offset))
        duties.append({'pair': list(pair), 'competitor': 1, 'variables': names[:],
                       'base': lp_record(names, bounds, rows+guards, {}, F(0)),
                       'a': forms[pair[0]], 'b': forms[pair[1]], 'gate': gate[:]})
    request = {'schema': 'GIVEN_LP_TOP2_ENDPOINT_REQUEST_V1',
               'experts': 3, 'classes': 2, 'label': 0, 'duties': duties}
    return protocol, request, tags


def certificate(lp, multipliers):
    """Hand-selected inequality multipliers; arithmetic only, no optimizer."""
    y = [F(0)]*len(lp['b'])
    for index, value in multipliers.items():
        y[index] = F(value)
    coefficients = list(map(F, lp['c'])); value = F(lp['offset'])
    for row_, rhs, dual in zip(csr(lp['A']), lp['b'], y):
        value += F(rhs)*dual
        for j, a in row_.items():
            coefficients[j] -= a*dual
    value += sum((min(r*F(a), r*F(b)) for r, a, b in
                  zip(coefficients, lp['lower'], lp['upper'])), F(0))
    return {'lp_sha256': identity(lp), 'inequality_dual': list(map(str, y)),
            'equality_dual': ['0']*len(lp['h']), 'claimed_lower_bound': str(value)}


def manual_proposer(tags):
    def propose(duty, weight, lp):
        pair = duty['pair']; terms = {}
        def use(tag, value):
            terms[tags[tag]] = -F(value)
        if pair == [0, 1] and weight == 0:
            use('p1-x', F(1, 4)); use('n1', F(1, 4))
        elif pair == [0, 1] and weight == F(1, 2):
            for tag in ('p0', 'n0+x', 'p1', 'n1+x'):
                use(tag, F(1, 8))
        elif pair == [1, 2]:
            use('p1-x', weight/4); use('n1', weight/4)
            use('p2-x', (1-weight)/4); use('n2', (1-weight)/4)
        elif pair == [0, 2]:
            # Last guard is r1-r2 = x+2 <= 0, inconsistent with x >= -1.
            # Retain and prove the duty rather than omit this route.
            terms[len(lp['b'])-1] = F(-2)
        return certificate(lp, terms)
    return propose


def exact_point(lp, point):
    values = list(map(F, point))
    if len(values) != len(lp['c']):
        raise ValueError('point dimension')
    if any(not F(a) <= v <= F(b) for v, a, b in zip(values, lp['lower'], lp['upper'])):
        raise ValueError('point outside variable box')
    for matrix, rhs in [('A', 'b'), ('E', 'h')]:
        for r, b in zip(csr(lp[matrix]), lp[rhs]):
            value = sum((a*values[j] for j, a in r.items()), F(0))
            if (value > F(b)) if matrix == 'A' else (value != F(b)):
                raise ValueError('infeasible point')
    return F(lp['offset']) + sum((F(c)*v for c, v in zip(lp['c'], values)), F(0))


def mccormick(duty, difference_lower=F(-3, 2), difference_upper=F(3, 2)):
    """Same guarded P/gate with four planes; no claimed native optimum."""
    names = duty['variables']; base = duty['base']
    rows = []
    for matrix, rhs, sense in [('A', 'b', 'le'), ('E', 'h', 'eq')]:
        rows += [row(sense, {names[j]: a for j, a in r.items()}, F(b))
                 for r, b in zip(csr(base[matrix]), base[rhs])]
    delta = {v: F(a)-F(b) for v, a, b in zip(names, duty['a']['c'], duty['b']['c'])}
    offset = F(duty['a']['offset'])-F(duty['b']['offset'])
    low, high = map(F, duty['gate']); dl, du = difference_lower, difference_upper
    for gate, difference, sign in [(low, dl, 1), (high, du, 1), (high, dl, -1), (low, du, -1)]:
        terms = {v: sign*gate*a for v, a in delta.items()}
        terms.update(gate=sign*difference, product=F(-sign))
        rows.append(row('le', terms, sign*gate*(difference-offset)))
    products = [g*d for g in (low, high) for d in (dl, du)]
    bounds = {v: (F(a), F(b)) for v, a, b in zip(names, base['lower'], base['upper'])}
    bounds.update(gate=(low, high), product=(min(products), max(products)))
    objective = {v: F(a) for v, a in zip(names, duty['b']['c'])}
    objective['product'] = F(1)
    return lp_record(names+['gate', 'product'], bounds, rows, objective, F(duty['b']['offset']))


def routes(x, protocol):
    scores = [F(a)*x+F(b) for a, b in protocol['source']['router']]
    return [list(p) for p in combinations(range(3), 2)
            if all(scores[i] >= scores[j] for i in p for j in range(3) if j not in p)]


def report():
    protocol, request, tags = fixture()
    deadline = time.monotonic()+300
    proof = build(request, expected_request_sha256=identity(request), deadline=deadline,
                  proposer=manual_proposer(tags))
    checked = check(request, proof, expected_request_sha256=identity(request), deadline=deadline)
    if checked['status'] != 'CHECKED_POSITIVE_GIVEN_BASE_AND_GATE':
        raise ValueError('registered complete endpoint control failed')
    signs = []
    lower, upper = map(F, protocol['source']['input'])
    scores = [tuple(map(F, values)) for values in protocol['source']['router']]
    for duty in request['duties']:
        a, b = duty['pair']
        slope, bias = scores[a][0]-scores[b][0], scores[a][1]-scores[b][1]
        lo, hi = min(slope*lower, slope*upper)+bias, max(slope*lower, slope*upper)+bias
        justified = ['0', '1/2'] if hi <= 0 else ['1/2', '1'] if lo >= 0 else ['0', '1']
        if duty['gate'] != justified:
            raise ValueError('synthetic affine sign does not justify gate range')
        signs.append({'pair': duty['pair'], 'router_margin': [str(lo), str(hi)],
                      'gate': justified, 'rule': 'sigmoid_monotonicity_and_sigmoid_zero_equals_half'})
    pair01 = request['duties'][0]
    lp = mccormick(pair01)
    point = ['0']*7 + ['1/4', '-1/8']
    objective = exact_point(lp, point)
    if objective != F(-1, 40):
        raise ValueError('registered same-base separation failed')
    route_witnesses = [{'x': str(x), 'pairs': routes(x, protocol)}
                       for x in (F(-1), F(-1, 2), F(1))]
    if [r['pairs'] for r in route_witnesses] != [[[1, 2]], [[0, 1], [1, 2]], [[0, 1]]]:
        raise ValueError('route change/tie coverage')
    # Witnesses force any valid difference interval to contain [-1/2,1].
    differences = []
    for x in (F(-1, 2), F(1)):
        actual = [x] + [v for _ in range(3) for v in (max(x, F(0)), max(-x, F(0)))]
        exact_point(pair01['base'], actual)
        differences.append(sum(((F(a)-F(b))*v for a, b, v in
                                zip(pair01['a']['c'], pair01['b']['c'], actual)), F(0)))
    # Even the hypothetical narrowest interval compatible with those witnesses
    # admits the negative point. This interval need not contain all relaxed P.
    tight_witness_objective = exact_point(mccormick(pair01, *differences), point)
    # On actual ReLUs, A(-1/2) is an expert-only violation on a legal tie pair.
    expert_only = (F(protocol['source']['expert_relu_coefficients'][1])*F(1, 2) +
                   F(protocol['source']['expert_input_coefficients'][0])*F(-1, 2) +
                   F(protocol['source']['expert_margin_constant']))
    return {'schema': 'H2_ENDPOINT_ALGEBRA_RESULT_V1', 'protocol_sha256': identity(protocol),
            'request_sha256': identity(request), 'proof_sha256': identity(proof),
            'endpoint_check': checked, 'mccormick': {
                'same_guarded_base': True, 'same_gate': True, 'difference_range': ['-3/2', '3/2'],
                'lp_sha256': identity(lp), 'feasible_point': point, 'checked_objective': str(objective),
                'exact_optimum_claimed': False, 'network_counterexample': False,
                'true_difference_witness_values': list(map(str, differences)),
                'narrowest_witness_compatible_range_objective': str(tight_witness_objective)},
            'route_witnesses': route_witnesses,
            'synthetic_gate_sign_checks': signs,
            'expert_only_margin_at_legal_tie': str(expert_only),
            'native_solver_calls': 0, 'real_model_requests': 0,
            'source_adapter_integrated': False, 'runtime_or_external_gain_claimed': False}
