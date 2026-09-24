"""Independent, solver/producer-free router exclusions and retained source checks.

An exclusion is valid only for a strictly positive whole-domain lower bound
on an outsider minus an insider. A zero bound NEVER excludes tie-legal top2.
"""
from fractions import Fraction as F
from itertools import combinations

from full_source.check_lift import check as check_affine
from full_source.check_obligations import check as check_outputs
from scoped_proof.evidence import lower_bound
from scoped_source.check import check_guards, check_join, check_projection
from scoped_source.graph import clock, operator, validate
from source_enclosure.check import check_box, check_relu
from source_enclosure.format import identity, sparse, unpack


def check_network(graph, trace, root, shape, tick):
    if (set(trace) != {'name', 'steps'} or trace['name'] != graph['name'] or
            len(trace['steps']) != len(graph['layers'])):
        raise ValueError('network trace inventory')
    state = root
    for layer, step in zip(graph['layers'], trace['steps']):
        tick()
        next_shape, op, bias = operator(shape, layer)
        if (set(step) != {'index', 'kind', 'input_shape', 'output_shape', 'source_sha256', 'state', 'proof'} or
                step['index'] != layer['index'] or step['kind'] != layer['kind'] or
                step['input_shape'] != shape or step['output_shape'] != next_shape or
                step['source_sha256'] != identity(state)):
            raise ValueError('source layer identity')
        target = step['state']
        tag = graph['name'] + '/layer/' + str(layer['index'])
        if layer['kind'] == 'Flatten':
            if target != state or step['proof'] is not None:
                raise ValueError('flatten changed state')
        elif layer['kind'] == 'ReLU':
            check_relu(state, target, step['proof'], tag)
        else:
            check_affine(state, target, op, bias, step['proof'], tag)
        state, shape = target, next_shape
        tick()
    return state


def reconstruct_margin(state, higher, lower):
    """Rebuild the LP from checked router factors, not a supplied matrix."""
    hz, ci, bi = unpack(state)
    n = len(ci) + len(bi)
    rows = []
    for i in (higher, lower):
        row = dict(hz['Gc'][i])
        row.update({j + len(ci): v for j, v in hz['Gb'][i].items()})
        rows.append(row)
    coefficients = [rows[0].get(j, F(0)) - rows[1].get(j, F(0)) for j in range(n)]
    inequalities = [dict(a) for a in hz['Auc']]
    equalities = [dict(a) for a in hz['Ac']]
    for target, binaries in ((inequalities, hz['Aub']), (equalities, hz['Ab'])):
        for row, binary in zip(target, binaries):
            for col, value in binary.items():
                row[len(ci) + col] = value
    return {'matrix_format': 'csr_v1', 'c': list(map(str, coefficients)),
            'offset': str(hz['c'][higher] - hz['c'][lower]),
            'A': sparse(inequalities, n), 'b': list(map(str, hz['ub'])),
            'E': sparse(equalities, n), 'h': list(map(str, hz['b'])),
            'lower': [-1] * n, 'upper': [1] * n}


def check_frontier(doc, prefix, candidates, *, expected_source_sha256, deadline):
    tick = clock(deadline)
    request, lo, hi = validate(doc, expected_source_sha256, tick)
    if (set(prefix) != {'schema', 'source_sha256', 'input', 'router'} or
            prefix['schema'] != 'CHECKED_ROUTE_PREFIX_V1' or
            prefix['source_sha256'] != expected_source_sha256):
        raise ValueError('router prefix binding')
    root = prefix['input']
    check_box(lo, hi, root)
    state = check_network(doc['networks'][0], prefix['router'], root,
                          request['center']['shape'], tick)
    state_hash = identity(state)
    if type(candidates) is not list:
        raise ValueError('route candidate list required')
    seen = set()
    bounds = []
    e = request['experts']
    for candidate in candidates:
        tick()
        if (set(candidate) != {'schema', 'source_sha256', 'router_sha256', 'higher', 'lower', 'certificate'} or
                candidate['schema'] != 'ROUTER_MARGIN_CANDIDATE_V1' or
                candidate['source_sha256'] != expected_source_sha256 or
                candidate['router_sha256'] != state_hash):
            raise ValueError('route candidate source binding')
        j, i = candidate['higher'], candidate['lower']
        if type(i) is not int or type(j) is not int or not 0 <= i < e or not 0 <= j < e or i == j:
            raise ValueError('route margin direction')
        if (j, i) in seen:
            raise ValueError('duplicate route certificate')
        seen.add((j, i))
        certificate = candidate['certificate']
        if certificate is None:
            bounds.append({'higher': j, 'lower': i, 'status': 'NO_CANDIDATE'})
            continue
        if set(certificate) != {'lp_sha256', 'inequality_dual', 'equality_dual'}:
            raise ValueError('unexpected route certificate fields')
        checked = lower_bound(reconstruct_margin(state, j, i), certificate)
        bounds.append({'higher': j, 'lower': i,
                       'status': 'STRICT_DOMINANCE' if F(checked['checked_lower_bound']) > 0 else 'NOT_PROVED',
                       **checked})
    bounds.sort(key=lambda row: (row['higher'], row['lower']))
    decisions = []
    for pair in combinations(range(e), 2):
        tick()
        witnesses = [b for b in bounds if b['status'] == 'STRICT_DOMINANCE'
                     and b['lower'] in pair and b['higher'] not in pair]
        decision = {'pair': list(pair), 'status': 'RETAINED'}
        if witnesses:
            w = witnesses[0]
            decision.update(status='EXCLUDED_BY_CHECKED_STRICT_MARGIN',
                            higher=w['higher'], lower=w['lower'],
                            checked_lower_bound=w['checked_lower_bound'], lp_sha256=w['lp_sha256'])
        decisions.append(decision)
    retained = [p['pair'] for p in decisions if p['status'] == 'RETAINED']
    if not retained:
        raise ValueError('no route remains on a nonempty declared domain')
    tick()
    return {'schema': 'CHECKED_ROUTE_FRONTIER_V1', 'source_sha256': expected_source_sha256,
            'router_sha256': state_hash, 'bounds': bounds, 'pairs': decisions,
            'total_pairs': len(decisions), 'retained_pairs': len(retained),
            'excluded_pairs': len(decisions) - len(retained),
            'needed_experts': sorted({i for pair in retained for i in pair}),
            'route_changing_established': False, 'complete_output_positive_proof': False}


def check(doc, bundle, *, expected_source_sha256, deadline):
    tick = clock(deadline)
    if (set(bundle) != {'schema', 'source_sha256', 'prefix', 'route_candidates', 'frontier', 'experts', 'pairs', 'lower_bound_certificates'} or
            bundle['schema'] != 'CHECKED_ROUTE_CONSTRUCTION_V1' or
            bundle['source_sha256'] != expected_source_sha256 or bundle['lower_bound_certificates'] != []):
        raise ValueError('construction binding; no borrowed output proofs')
    frontier = check_frontier(doc, bundle['prefix'], bundle['route_candidates'],
                             expected_source_sha256=expected_source_sha256, deadline=deadline)
    if bundle['frontier'] != frontier:
        raise ValueError('changed/missing route coverage')
    needed = frontier['needed_experts']
    names = [f'expert{i}' for i in needed]
    if [t['name'] for t in bundle['experts']] != names:
        raise ValueError('required expert trace missing/duplicated')
    request = doc['request']
    root = bundle['prefix']['input']
    ends = {'router': bundle['prefix']['router']['steps'][-1]['state']}
    for i, trace in zip(needed, bundle['experts']):
        ends[trace['name']] = check_network(doc['networks'][i + 1], trace, root,
            request['center']['shape'], tick)
    retained = [p['pair'] for p in frontier['pairs'] if p['status'] == 'RETAINED']
    if [p['pair'] for p in bundle['pairs']] != retained:
        raise ValueError('retained pair inventory')
    outputs = 0
    for entry in bundle['pairs']:
        tick()
        if set(entry) != {'pair', 'joint', 'guarded', 'projected', 'base', 'obligations'}:
            raise ValueError('pair construction fields')
        pair = entry['pair']
        check_join(root, [ends['router']] + [ends[f'expert{i}'] for i in pair], entry['joint'])
        check_guards(entry['joint'], entry['guarded'], pair, request['experts'])
        check_projection(entry['guarded'], entry['projected'], request['experts'],
                         request['classes'], request['label'], request['margin'])
        checked = check_outputs(entry['projected'], entry['base'], entry['obligations'],
                                pair, request['classes'], request['label'])
        outputs += checked['obligations']
    tick()
    return {'status': 'CHECKED_ROUTE_COVER_AND_RETAINED_OUTPUT_CONSTRUCTIONS',
            'source_sha256': expected_source_sha256, 'frontier': frontier,
            'expert_traces_checked': len(needed), 'output_obligations': outputs,
            'original_output_obligations': frontier['total_pairs'] * (request['classes'] - 1),
            'excluded_output_obligations': frontier['excluded_pairs'] * (request['classes'] - 1),
            'lower_bounds_checked': 0, 'complete_output_positive_proof': False,
            'route_changing_established': False, 'native_float_proof': False,
            'trusted': ['declared graph/program correspondence', 'stored-center preprocessing',
                        'standard-library checker implementation/runtime']}
