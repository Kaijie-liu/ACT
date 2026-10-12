"""Opt-in sparse enclosure with once-per-expert templates and block support.

Only the enclosure implementation changes; original finite source and endpoint
limits, exact source equations and CPU candidate algorithm remain unchanged.
"""
from copy import deepcopy
from itertools import combinations
import time

from source_enclosure import produce
from source_enclosure.format import identity, unpack
from scoped_source.graph import validate, operator, clock
from scoped_source.hz_lifted_source import owned, relu
from scoped_source.check_hz_row_enclosure import parse
from scoped_source import hz_sparse_rows as rows
from scoped_source.hz_source_build import live, entry_for, gate
from scoped_source.hz_source_check import exact_float, properties
from scoped_source.hz_templates import make_common
from scoped_source.check_hz_sparse_source import SCHEMA, CONTEXT_GUARD
from act.back_end.solver.solver_hz import sparse_hz_linear
from act.back_end.solver.hz_lp_export import snapshot
from act.back_end.moe.block_endpoints import prepare_request, propose_request


def enclosure(reference, tag, deadline):
    tick = clock(deadline)
    parse(reference)  # Retain old global source admission before the row API.
    proof = rows.produce(reference, expected_reference_sha256=identity(reference),
                         owner=tag, deadline=deadline)
    target = proof['target']; parse(target, target=True)
    actual, snap = rows.instantiate(target, expected_target_sha256=proof['target_sha256'], deadline=deadline)
    tick()
    return target, {'reference': reference, 'enclosure': proof, 'snapshot': snap}


def propagate(doc, name, initial, prefix, deadline):
    import numpy as np
    import scipy.sparse as sp
    tick = clock(deadline); state = initial; shape = doc['center']['shape']; trace = []
    graph = next(g for g in doc['networks'] if g['name'] == name)
    for i, layer in enumerate(graph['layers']):
        tick(); tag = f'{prefix}/layer{i}'; width = len(unpack(state['state'])[0]['c'])
        shape, op, bias = operator(shape, layer); nominal = certificate = None
        if layer['kind'] == 'Linear':
            rr, cc, data = [], [], []
            for row, coefficients in enumerate(op):
                for col, value in coefficients.items():
                    rr.append(row); cc.append(col); data.append(exact_float(value))
            matrix = sp.csr_matrix((data, (rr, cc)), shape=(len(op), width))
            nominal = snapshot(sparse_hz_linear(live(state['state']), matrix,
                                                np.array([exact_float(v) for v in bias])))
            ref, certificate = produce.affine(state['state'], op, bias, nominal, tag)
        elif layer['kind'] == 'ReLU':
            ref, certificate = relu(state['state'], tag)
        else:
            ref = deepcopy(state['state'])
        state, record = enclosure(owned(ref, state, tag), tag, deadline)
        trace.append({'index': i, 'kind': layer['kind'], 'nominal': nominal,
                      'certificate': certificate, 'lift': record})
    tick()
    return state, trace


def build(doc, *, expected_source_sha256, deadline, observe=None):
    tick = clock(deadline); start = time.monotonic()
    r, lower, upper = validate(doc, expected_source_sha256, tick); props = properties(r)
    initial, inp = enclosure(owned(produce.box(lower, upper)), 'input', deadline)
    router, rt = propagate(doc, 'router', initial, 'router', deadline)
    base = make_common(initial, router); states, templates = [], []
    for i in range(r['experts']):
        state, trace = propagate(doc, f'expert{i}', base, f'template/expert{i}', deadline)
        states.append(state); templates.append({'expert': i, 'trace': trace})
    pairs, live_pairs = [], []
    for a, b in combinations(range(r['experts']), 2):
        tick(); tag = f'pair{a}-{b}'
        ref = owned(entry_for(initial['state'], router['state'], (a,b), r['experts']), router, tag+'/guard')
        entry, record = enclosure(ref, tag+'/guard', deadline)
        if (entry['state']['continuous_ids'] != base['state']['continuous_ids']
                or entry['state']['binary_ids'] != base['state']['binary_ids']
                or entry['ownership'] != base['ownership']):
            raise ValueError('guard factor ownership/identity')
        evidence = gate(router['state'], (a,b))
        pairs.append({'pair': [a,b], 'entry': record, 'gate_evidence': evidence})
        live_pairs.append({'pair': [a,b], 'common': live(base['state']), 'entry': live(entry['state']),
                           'a': live(states[a]['state']), 'b': live(states[b]['state']), 'gate': evidence['bounds']})
    context = {'request': expected_source_sha256, 'domain': identity(r), 'guard': CONTEXT_GUARD}
    req = prepare_request(live_pairs, props, experts=r['experts'], classes=r['classes'],
                          context=context, deadline=deadline)
    made = time.monotonic()
    if observe: observe('construction', made-start)
    if identity(doc) != expected_source_sha256: raise ValueError('source changed before proposals')
    tick()
    proof = propose_request(req, expected_request_sha256=identity(req), deadline=deadline)
    if observe: observe('proposals', time.monotonic()-made)
    if identity(doc) != expected_source_sha256: raise ValueError('source changed during generation')
    result = {'schema': SCHEMA, 'source_sha256': expected_source_sha256, 'input': inp, 'router': rt,
              'common': base, 'templates': templates, 'pairs': pairs, 'endpoint_request': req, 'proof': proof}
    tick()
    return result
