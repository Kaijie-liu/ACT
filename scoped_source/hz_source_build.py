"""Untrusted additive source-to-HybridZ endpoint control producer, CPU only."""
from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations
import time

from scoped_source.graph import validate, operator, clock
from source_enclosure.format import unpack, pack, identity, clean
from source_enclosure import produce
from upstream_source.checker import hz as parsed_hz
from act.back_end.solver.hz_lp_export import snapshot
from act.back_end.solver.solver_hz import SparseHZono, sparse_hz_linear
from act.back_end.hybridz_tf.tf_mlp import sparse_hz_apply_relu_exact
from act.back_end.moe.hz_endpoints import prepare_request, propose_request
from scoped_source.hz_source_check import SCHEMA, exact_float, state_snapshot, relu_step, properties


def live(state):
    import numpy as np
    import scipy.sparse as sp
    h, c, b = unpack(state)
    args = {k: np.array([exact_float(v) for v in h[k]], dtype=np.float64) for k in ('c', 'b', 'ub')}
    for key in ('Gc', 'Gb', 'Ac', 'Ab', 'Auc', 'Aub'):
        width = len(c) if key.endswith('c') else len(b)
        data, cols, ptr = [], [], [0]
        for row in h[key]:
            for j, v in sorted(row.items()):
                if v:
                    cols.append(j)
                    data.append(exact_float(v))
            ptr.append(len(data))
        args[key] = sp.csr_matrix((data, cols, ptr), shape=(len(h[key]), width), dtype=np.float64)
    result = SparseHZono(**args, frame_id=h['frame_id'], exact=False)
    state_snapshot(state, snapshot(result))
    return result


def apply_relu(state, tag):
    import numpy as np
    s, ci, bi = unpack(state)
    ci, bi = list(ci), list(bi)
    ranges, branches, slots = [], [], []
    for i, center in enumerate(s['c']):
        radius = sum((abs(v) for key in ('Gc', 'Gb') for v in s[key][i].values()), F(0))
        lo, hi = center-radius, center+radius
        ranges.append([str(lo), str(hi)])
        branch = 'active' if lo >= 0 else 'inactive' if hi <= 0 else 'unstable'
        branches.append(branch)
        if branch == 'unstable':
            slots.append((len(ci), len(ci)+1, len(bi)))
            ci += [f'{tag}/negative/{i}', f'{tag}/positive/{i}']
            bi.append(f'{tag}/sign/{i}')
    lower = np.array([exact_float(F(r[0])) for r in ranges])
    upper = np.array([exact_float(F(r[1])) for r in ranges])
    result = sparse_hz_apply_relu_exact(live(state), lower, upper, slots, len(ci), len(bi))
    target = pack(parsed_hz(snapshot(result)), ci, bi)
    certificate = {'source': identity(state), 'tag': tag, 'ranges': ranges, 'branches': branches}
    relu_step(state, target, certificate, tag)
    return target, certificate


def propagate(doc, name, initial, prefix, deadline):
    import numpy as np
    import scipy.sparse as sp
    tick = clock(deadline)
    graph = next(g for g in doc['networks'] if g['name'] == name)
    state, shape, trace = initial, doc['center']['shape'], []
    for i, layer in enumerate(graph['layers']):
        tick()
        width = len(unpack(state)[0]['c'])
        shape, op, bias = operator(shape, layer)
        tag = f'{prefix}/layer{i}'
        nominal = certificate = None
        if layer['kind'] == 'Linear':
            rr, cc, vv = [], [], []
            for row, coefficients in enumerate(op):
                for col, v in coefficients.items():
                    rr.append(row); cc.append(col); vv.append(exact_float(v))
            matrix = sp.csr_matrix((vv, (rr, cc)), shape=(len(op), width))
            transformed = sparse_hz_linear(live(state), matrix, np.array([exact_float(v) for v in bias]))
            nominal = snapshot(transformed)
            target, certificate = produce.affine(state, op, bias, nominal, tag)
        elif layer['kind'] == 'ReLU':
            target, certificate = apply_relu(state, tag)
        else:
            target = deepcopy(state)
        live(target)  # exact round-trip, including all inherited constraints
        trace.append({'index': i, 'kind': layer['kind'], 'target': target,
                      'nominal': nominal, 'certificate': certificate})
        state = target
    tick()
    return state, trace


def entry_for(initial, router, pair, experts):
    x, _, _ = unpack(initial)
    r, ci, bi = unpack(router)
    result = deepcopy(r)
    for key in ('c', 'Gc', 'Gb'):
        result[key] = deepcopy(x[key])
    for a in pair:
        for o in range(experts):
            if o in pair:
                continue
            for key, dest in [('Gc', 'Auc'), ('Gb', 'Aub')]:
                row = dict(r[key][o])
                for j, v in r[key][a].items():
                    row[j] = row.get(j, F(0))-v
                result[dest].append(clean(row))
            result['ub'].append(r['c'][a]-r['c'][o])
    return pack(result, ci, bi)


def gate(router, pair):
    r, _, _ = unpack(router)
    a, b = pair
    d = r['c'][a]-r['c'][b]
    radius = F(0)
    for key in ('Gc', 'Gb'):
        row = dict(r[key][a])
        for j, v in r[key][b].items():
            row[j] = row.get(j, F(0))-v
        radius += sum(map(abs, row.values()), F(0))
    lo, hi = d-radius, d+radius
    bounds = ['0', '1']
    if lo >= 0: bounds = ['1/2', '1']
    if hi <= 0: bounds = ['0', '1/2']
    if lo == hi == 0: bounds = ['1/2', '1/2']
    return {'pair': list(pair), 'weight_expert': a, 'router_sha256': identity(router),
            'margin_box': [str(lo), str(hi)], 'bounds': bounds}


def build(doc, *, expected_source_sha256, deadline, observe=None):
    tick = clock(deadline)
    started = time.monotonic()
    r, lower, upper = validate(doc, expected_source_sha256, tick)
    props = properties(r)
    initial = produce.box(lower, upper)
    router, rt = propagate(doc, 'router', initial, 'router', deadline)
    records, live_pairs = [], []
    for pair in combinations(range(r['experts']), 2):
        tick()
        a, b = pair
        entry = entry_for(initial, router, pair, r['experts'])
        left, lt = propagate(doc, f'expert{a}', entry, f'pair{a}-{b}/expert{a}', deadline)
        right, bt = propagate(doc, f'expert{b}', entry, f'pair{a}-{b}/expert{b}', deadline)
        evidence = gate(router, pair)
        records.append({'pair': list(pair), 'entry': entry, 'a': lt, 'b': bt, 'gate_evidence': evidence})
        live_pairs.append({'pair': list(pair), 'entry': live(entry), 'a': live(left), 'b': live(right),
                           'gate': evidence['bounds']})
    context = {'request': expected_source_sha256, 'domain': identity(r), 'guard': 'ALL_TIE_LEGAL_PAIRS'}
    req = prepare_request(live_pairs, props, experts=r['experts'], classes=r['classes'],
                          context=context, deadline=deadline)
    prepared = time.monotonic()
    if observe is not None:
        observe('construction', prepared-started)
    proof = propose_request(req, expected_request_sha256=identity(req), deadline=deadline)
    if observe is not None:
        observe('proposals', time.monotonic()-prepared)
    tick()
    return {'schema': SCHEMA, 'source_sha256': expected_source_sha256, 'input': initial,
            'router': rt, 'pairs': records, 'endpoint_request': req, 'proof': proof}
