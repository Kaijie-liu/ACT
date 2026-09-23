"""Solver-free checking of the declared source, every guard and fresh output LP.

Does not import capture/build or any producer. Construction acceptance is NOT
lower-bound acceptance, reachability, a SAFE decision, or native float proof.
"""
from fractions import Fraction as F
from itertools import combinations
from source_enclosure.format import unpack, identity
from source_enclosure.check import check_box, check_relu
from full_source.check_lift import check as check_affine
from full_source.check_obligations import check as check_outputs
from scoped_source.graph import validate, operator, clock


def check_join(base, sources, target):
    root, rc, rb = unpack(base)
    parts = [unpack(s) for s in sources]
    out, oc, ob = unpack(target)
    for h, ci, bi in parts:
        if h['frame_id'] != root['frame_id'] or ci[:len(rc)] != rc or bi[:len(rb)] != rb:
            raise ValueError('shared input frame/factor prefix')
        for a, b, rhs in [('Ac', 'Ab', 'b'), ('Auc', 'Aub', 'ub')]:
            for key in (a, b, rhs):
                if h[key][:len(root[rhs])] != root[key]:
                    raise ValueError('shared input constraint prefix')
    required_c = rc + [v for _, c, _ in parts for v in c[len(rc):]]
    required_b = rb + [v for _, _, b in parts for v in b[len(rb):]]
    if (oc != required_c or ob != required_b or out['frame_id'] != root['frame_id'] or
            len(set(oc + ob)) != len(oc + ob)):
        raise ValueError('private factor collision/joint frame')
    positions = ({v: i for i, v in enumerate(oc)}, {v: i for i, v in enumerate(ob)})
    def remap(rows, ids, kind):
        mapping = positions[kind]
        return [{mapping[ids[j]]: x for j, x in row.items()} for row in rows]
    if out['c'] != [v for h, _, _ in parts for v in h['c']]:
        raise ValueError('router/expert output order')
    for key, kind in [('Gc', 0), ('Gb', 1)]:
        required = []
        for h, c, b in parts:
            required.extend(remap(h[key], (c, b)[kind], kind))
        if out[key] != required:
            raise ValueError('shared output coefficient mapping')
    for a, b, rhs in [('Ac', 'Ab', 'b'), ('Auc', 'Aub', 'ub')]:
        n = len(root[rhs])
        for key, kind in [(a, 0), (b, 1)]:
            required = list(root[key])
            for h, ci, bi in parts:
                required.extend(remap(h[key][n:], (ci, bi)[kind], kind))
            if out[key] != required:
                raise ValueError('joined inherited constraint mapping')
        if out[rhs] != root[rhs] + [v for h, _, _ in parts for v in h[rhs][n:]]:
            raise ValueError('joined constraint RHS')


def check_guards(source, target, pair, experts):
    s, c, b = unpack(source); t, ct, bt = unpack(target)
    if (ct != c or bt != b or t['frame_id'] != s['frame_id'] or
            any(t[k] != s[k] for k in ('c', 'Gc', 'Gb', 'Ac', 'Ab', 'b'))):
        raise ValueError('guard attachment changed source')
    comparisons = [(i, j) for i in pair for j in range(experts) if j not in pair]
    for dst, src in [('Auc', 'Gc'), ('Aub', 'Gb')]:
        extra = []
        for i, j in comparisons:
            coeff = {}
            for col in s[src][i].keys() | s[src][j].keys():
                value = s[src][j].get(col, F(0)) - s[src][i].get(col, F(0))
                if value: coeff[col] = value
            extra.append(coeff)
        if t[dst] != s[dst] + extra:
            raise ValueError('non-strict tie-legal top2 guard coefficients')
    if t['ub'] != s['ub'] + [s['c'][i] - s['c'][j] for i, j in comparisons]:
        raise ValueError('top2 guard RHS/sign')


def check_projection(source, target, experts, classes, label, margin):
    s, c, b = unpack(source); t, ct, bt = unpack(target)
    if (ct != c or bt != b or t['frame_id'] != s['frame_id'] or len(s['c']) != experts + 2*classes or
            any(t[k] != s[k] for k in ('Ac', 'Ab', 'b', 'Auc', 'Aub', 'ub'))):
        raise ValueError('projection source/constraints/factors')
    required = s['c'][experts:]
    for index in (label, classes + label): required[index] -= F(margin)
    if t['c'] != required or any(t[k] != s[k][experts:] for k in ('Gc', 'Gb')):
        raise ValueError('classification margin projection/order')


def check(doc, bundle, *, expected_source_sha256, deadline):
    tick = clock(deadline)
    r, lo, hi = validate(doc, expected_source_sha256, tick)
    if (set(bundle) != {'schema', 'source_sha256', 'input', 'networks', 'pairs', 'lower_bound_certificates'} or
            bundle['schema'] != 'SCOPED_SOURCE_CONSTRUCTION_V1' or
            bundle['source_sha256'] != expected_source_sha256 or bundle['lower_bound_certificates'] != []):
        raise ValueError('construction identity; no historical/candidate certificates admitted')
    root = bundle['input']; check_box(lo, hi, root); tick()
    if [v['name'] for v in bundle['networks']] != [v['name'] for v in doc['networks']]:
        raise ValueError('complete network inventory')
    ends = {}; steps_checked = 0
    for graph, trace in zip(doc['networks'], bundle['networks']):
        state = root; shape = r['center']['shape']
        if len(trace['steps']) != len(graph['layers']): raise ValueError('missing network layer')
        for layer, step in zip(graph['layers'], trace['steps']):
            tick(); out_shape, op, bias = operator(shape, layer)
            if (step['index'] != layer['index'] or step['kind'] != layer['kind'] or
                    step['input_shape'] != shape or step['output_shape'] != out_shape or
                    step['source_sha256'] != identity(state)):
                raise ValueError('source step identity/shape/operator')
            tag = graph['name'] + '/layer/' + str(layer['index'])
            target = step['state']
            if layer['kind'] == 'Flatten':
                if target != state or step['proof'] is not None: raise ValueError('flatten changed source')
            elif layer['kind'] == 'ReLU': check_relu(state, target, step['proof'], tag)
            else: check_affine(state, target, op, bias, step['proof'], tag)
            state = target; shape = out_shape; steps_checked += 1; tick()
        ends[graph['name']] = state
    pairs = [list(p) for p in combinations(range(r['experts']), 2)]
    if [p['pair'] for p in bundle['pairs']] != pairs: raise ValueError('all unordered pairs, including ties, required')
    outputs = 0
    for entry in bundle['pairs']:
        tick(); pair = entry['pair']
        check_join(root, [ends['router']] + [ends[f'expert{i}'] for i in pair], entry['joint'])
        check_guards(entry['joint'], entry['guarded'], pair, r['experts'])
        check_projection(entry['guarded'], entry['projected'], r['experts'], r['classes'], r['label'], r['margin'])
        checked = check_outputs(entry['projected'], entry['base'], entry['obligations'], pair, r['classes'], r['label'])
        outputs += checked['obligations']; tick()
    return {'status': 'CHECKED_DECLARED_SOURCE_AND_ALL_OUTPUT_LP_CONSTRUCTIONS',
        'source_sha256': expected_source_sha256, 'networks_checked': len(ends), 'steps_checked': steps_checked,
        'guarded_pairs': len(pairs), 'output_obligations': outputs, 'lower_bounds_checked': 0,
        'complete_output_positive_proof': False, 'reachability_exclusions': 0, 'route_changing_established': False,
        'trusted': ['declared graph corresponds to intended program', 'standard-library checker implementation'],
        'unproved': ['positive output lower bounds', 'native floating execution', 'raw-data preprocessing'],
        'deadline_contract': 'cooperative checks; outer process watchdog required before real execution'}
