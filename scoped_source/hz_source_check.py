"""Independent declared-source / actual-HZ / endpoint connection checker.

Acceptance calls no numerical proposal, propagation kernel or solver. ACT's
package initializers have transitive numerical imports: this is not a portable
standard-library entry. Cooperative, finite intake; no native-float proof.
"""
from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations
import math

from scoped_source.graph import validate, operator, clock
from source_enclosure.format import unpack, pack, identity, clean
from source_enclosure.check import check_box, check_affine, check_relu
from upstream_source.checker import hz
from act.back_end.moe.check_hz_endpoints import check_request

SCHEMA = 'CHECKED_HZ_SOURCE_ENDPOINT_V1'
FIELDS = ('c', 'Gc', 'Gb', 'Ac', 'Ab', 'b', 'Auc', 'Aub', 'ub')


def exact_float(value):
    q = F(value)
    f = float(q)
    if not math.isfinite(f) or F(f) != q:
        raise ValueError('coefficient not exactly representable in binary64')
    return f


def checked_state(state):
    h, c, b = unpack(state)
    if not 1 <= len(c)+len(b) <= 128 or len(h['c']) > 128 or len(h['b'])+len(h['ub']) > 256:
        raise ValueError('finite source connection capacity')
    for key in FIELDS:
        for row in h[key]:
            for v in row.values() if isinstance(row, dict) else [row]:
                exact_float(v)
    return h, c, b


def state_snapshot(state, snapshot):
    s, _, _ = checked_state(state)
    if snapshot['exact'] is not False or s != hz(snapshot):
        raise ValueError('checked state and actual HZ snapshot differ')


def relu_step(source, target, certificate, tag):
    """Translate only new blocked inequalities; verify every actual coefficient."""
    s, _, _ = checked_state(source)
    t, c, b = checked_state(target)
    k = sum(v == 'unstable' for v in certificate['branches'])
    n = len(s['ub'])
    if len(t['ub']) != n+2*k:
        raise ValueError('ReLU actual blocked row inventory')
    canonical = deepcopy(t)
    order = list(range(n))+[j for i in range(k) for j in (n+i, n+k+i)]
    for key in ('Auc', 'Aub', 'ub'):
        canonical[key] = [t[key][j] for j in order]
    return check_relu(source, pack(canonical, c, b), certificate, tag)


def network(doc, name, initial, trace, prefix, deadline):
    tick = clock(deadline)
    graph = next(g for g in doc['networks'] if g['name'] == name)
    if len(trace) != len(graph['layers']):
        raise ValueError('complete ordered layer trace required')
    state = initial
    shape = doc['center']['shape']
    rows = []
    for i, (layer, step) in enumerate(zip(graph['layers'], trace)):
        tick()
        if set(step) != {'index', 'kind', 'target', 'nominal', 'certificate'}:
            raise ValueError('layer trace schema')
        if (step['index'], step['kind']) != (i, layer['kind']):
            raise ValueError('layer trace order/operator')
        target = step['target']
        checked_state(target)
        shape, op, bias = operator(shape, layer)
        tag = f'{prefix}/layer{i}'
        if layer['kind'] == 'Linear':
            result = check_affine(state, target, op, bias, step['nominal'], step['certificate'], tag)
        elif layer['kind'] == 'ReLU':
            if step['nominal'] is not None:
                raise ValueError('unexpected ReLU nominal')
            result = relu_step(state, target, step['certificate'], tag)
        else:
            if step['nominal'] is not None or step['certificate'] is not None or target != state:
                raise ValueError('flatten changed exact factor expression')
            result = {'status': 'CHECKED_IDENTITY_FLATTEN'}
        state = target
        rows.append(result)
    tick()
    return state, rows


def route_entry(input_state, router, target, pair, experts):
    """Check conditional route rows, NOT redundant/global guard attachment."""
    x, xc, xb = checked_state(input_state)
    r, rc, rb = checked_state(router)
    t, tc, tb = checked_state(target)
    if rc[:len(xc)] != xc or rb[:len(xb)] != xb or (rc, rb) != (tc, tb):
        raise ValueError('route entry lost router/input factor identity')
    if r['frame_id'] != x['frame_id'] or t['frame_id'] != r['frame_id']:
        raise ValueError('route entry frame')
    expected = deepcopy(r)
    for key in ('c', 'Gc', 'Gb'):
        expected[key] = deepcopy(x[key])
    for a in pair:
        for o in range(experts):
            if o in pair:
                continue
            for output, constraint in [('Gc', 'Auc'), ('Gb', 'Aub')]:
                cols = r[output][o].keys() | r[output][a].keys()
                expected[constraint].append(clean({j: r[output][o].get(j, F(0))-r[output][a].get(j, F(0)) for j in cols}))
            expected['ub'].append(r['c'][a]-r['c'][o])
    if any(expected[k] != t[k] for k in FIELDS):
        raise ValueError('conditional guard or restored input expression mismatch')


def gate_range(router, pair):
    r, _, _ = checked_state(router)
    a, b = pair
    center = r['c'][a]-r['c'][b]
    radius = F(0)
    for key in ('Gc', 'Gb'):
        for j in r[key][a].keys() | r[key][b].keys():
            radius += abs(r[key][a].get(j, F(0))-r[key][b].get(j, F(0)))
    lo, hi = center-radius, center+radius
    bounds = (F(1, 2), F(1, 2)) if lo == hi == 0 else (F(0), F(1, 2)) if hi <= 0 else (F(1, 2), F(1)) if lo >= 0 else (F(0), F(1))
    return {'pair': list(pair), 'weight_expert': a, 'router_sha256': identity(router),
            'margin_box': [str(lo), str(hi)], 'bounds': list(map(str, bounds))}


def properties(request):
    e, c, y = request['experts'], request['classes'], request['label']
    if not 2 <= e <= 4 or not 2 <= c <= 5:
        raise ValueError('finite endpoint source dimensions')
    return [{'id': f'class{k}', 'q': [str(int(i == y)-int(i == k)) for i in range(c)],
             'offset': str(-F(request['margin']))} for k in range(c) if k != y]


def check(doc, package, *, expected_source_sha256, deadline):
    tick = clock(deadline)
    before = identity(package)
    r, lower, upper = validate(doc, expected_source_sha256, tick)
    props = properties(r)
    if (set(package) != {'schema', 'source_sha256', 'input', 'router', 'pairs', 'endpoint_request', 'proof'}
            or package['schema'] != SCHEMA or package['source_sha256'] != expected_source_sha256):
        raise ValueError('source package identity/schema')
    initial = package['input']
    checked_state(initial)
    check_box(lower, upper, initial)
    router, steps = network(doc, 'router', initial, package['router'], 'router', deadline)
    roster = [list(p) for p in combinations(range(r['experts']), 2)]
    if [p['pair'] for p in package['pairs']] != roster:
        raise ValueError('all tie-legal pairs required, no exclusions')
    req = package['endpoint_request']
    context = {'request': expected_source_sha256, 'domain': identity(r), 'guard': 'ALL_TIE_LEGAL_PAIRS'}
    if (req['context'] != context or req['properties'] != props or req['experts'] != r['experts']
            or req['classes'] != r['classes'] or [p['pair'] for p in req['pairs']] != roster):
        raise ValueError('endpoint source/domain/property binding')
    for p, endpoint in zip(package['pairs'], req['pairs']):
        tick()
        if set(p) != {'pair', 'entry', 'a', 'b', 'gate_evidence'}:
            raise ValueError('source pair schema')
        a, b = p['pair']
        entry = p['entry']
        route_entry(initial, router, entry, (a, b), r['experts'])
        left, ls = network(doc, f'expert{a}', entry, p['a'], f'pair{a}-{b}/expert{a}', deadline)
        right, rs = network(doc, f'expert{b}', entry, p['b'], f'pair{a}-{b}/expert{b}', deadline)
        steps += ls+rs
        _, ec, eb = unpack(entry)
        private = []
        for terminal in (left, right):
            _, ci, bi = unpack(terminal)
            if ci[:len(ec)] != ec or bi[:len(eb)] != eb:
                raise ValueError('expert does not extend same checked route entry')
            private += ci[len(ec):]+bi[len(eb):]
        if len(set(private)) != len(private) or set(private).intersection(ec+eb):
            raise ValueError('private expert factor alias')
        for name, state in [('entry', entry), ('a', left), ('b', right)]:
            state_snapshot(state, endpoint['sources'][name])
        expected_gate = gate_range(router, (a, b))
        if (p['gate_evidence'] != expected_gate or endpoint['gate']['bounds'] != expected_gate['bounds']
                or endpoint['relation_mode'] != 'shared_input'):
            raise ValueError('gate orientation/premise or shared relation changed')
    accepted = check_request(req, package['proof'], expected_request_sha256=identity(req), deadline=deadline)
    tick()
    if identity(package) != before or identity(doc) != expected_source_sha256:
        raise ValueError('source/package changed while checking')
    positive = accepted['positive'] == accepted['required']
    return {'status': 'CHECKED_POSITIVE_DECLARED_REAL_SOURCE' if positive else accepted['status'],
            'source_sha256': expected_source_sha256, 'package_sha256': before,
            'required': accepted['required'], 'positive': accepted['positive'],
            'checked_endpoints': accepted['checked_endpoints'], 'missing_endpoints': accepted['missing_endpoints'],
            'results': accepted['results'], 'checked_source_steps': len(steps),
            'affine_error_factors': sum(s.get('error_factors', 0) for s in steps),
            'source_lowering_checked': True, 'deployed_float_SAFE': False,
            'hard_budget_supervision': False, 'portable_distribution': False,
            'remaining_trust': ['declaration_corresponds_to_intended_program', 'exact_checker_implementation'],
            'scope': 'declared real graph; actual stored binary64 parameters, not floating execution'}
