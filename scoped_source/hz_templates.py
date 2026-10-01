"""Propagate each expert once; specialize checked shared-input pair views.

Finite opt-in source algorithm. No cross-request cache, changed support method,
native solve, guard tightening or CUDA. Every endpoint proof is fresh.
"""
from copy import deepcopy
from itertools import combinations
import time

from source_enclosure.format import unpack, pack, identity
from source_enclosure import produce
from scoped_source.graph import validate, clock
from scoped_source.hz_lifted_source import owned, enclosure, propagate
from scoped_source.hz_source_build import live, entry_for, gate
from scoped_source.hz_source_check import properties
from scoped_source.check_hz_binary64 import REFERENCE
from scoped_source.check_hz_templates import SCHEMA, VIEW, CONTEXT_GUARD
from act.back_end.moe.hz_endpoints import prepare_request, propose_request


def make_common(initial, router):
    x, _, _ = unpack(initial['state']); h, c, b = unpack(router['state'])
    h = deepcopy(h)
    for key in ('c', 'Gc', 'Gb'): h[key] = deepcopy(x[key])
    return {'schema': REFERENCE, 'state': pack(h,c,b), 'ownership': deepcopy(router['ownership'])}


def specialize(common, template, entry, *, pair, expert, deadline):
    tick = clock(deadline)
    h0, _, _ = unpack(common['state']); h, c, b = unpack(template['state'])
    g, _, _ = unpack(entry['state']); target = deepcopy(h)
    for ck, bk, rhs in (('Ac','Ab','b'), ('Auc','Aub','ub')):
        start = len(h0[rhs])
        for key in (ck,bk,rhs): target[key] = deepcopy(g[key])+deepcopy(h[key][start:])
    result = {'schema': VIEW, 'pair': list(pair), 'expert': expert,
              'common_sha256': identity(common), 'template_sha256': identity(template),
              'entry_sha256': identity(entry),
              'target': {'schema': REFERENCE, 'state': pack(target,c,b),
                         'ownership': deepcopy(template['ownership'])}}
    tick(); return result


def build(doc, *, expected_source_sha256, deadline, observe=None):
    tick = clock(deadline); start = time.monotonic()
    r, lower, upper = validate(doc, expected_source_sha256, tick); props = properties(r)
    initial, inp = enclosure(owned(produce.box(lower,upper)), 'input', deadline)
    router, rt = propagate(doc, 'router', initial, 'router', deadline)
    base = make_common(initial, router); templates = []; states = []
    for i in range(r['experts']):
        tick()
        state, trace = propagate(doc, f'expert{i}', base, f'template/expert{i}', deadline)
        states.append(state); templates.append({'expert': i, 'trace': trace})
    pairs = []; live_pairs = []
    for a,b in combinations(range(r['experts']),2):
        tick(); tag = f'pair{a}-{b}'
        reference = owned(entry_for(initial['state'],router['state'],(a,b),r['experts']),router,tag+'/guard')
        entry, record = enclosure(reference, tag+'/guard', deadline)
        views = [specialize(base,states[i],entry,pair=[a,b],expert=i,deadline=deadline) for i in (a,b)]
        evidence = gate(router['state'],(a,b))
        pairs.append({'pair':[a,b], 'entry':record, 'views':views, 'gate_evidence':evidence})
        live_pairs.append({'pair':[a,b], 'entry':live(entry['state']),
                           'a':live(views[0]['target']['state']), 'b':live(views[1]['target']['state']),
                           'gate':evidence['bounds']})
    context = {'request':expected_source_sha256,'domain':identity(r),'guard':CONTEXT_GUARD}
    req = prepare_request(live_pairs,props,experts=r['experts'],classes=r['classes'],context=context,deadline=deadline)
    made = time.monotonic()
    if observe: observe('construction',made-start)
    proof = propose_request(req,expected_request_sha256=identity(req),deadline=deadline)
    if observe: observe('proposals',time.monotonic()-made)
    if identity(doc) != expected_source_sha256: raise ValueError('source changed during generation')
    result = {'schema':SCHEMA,'source_sha256':expected_source_sha256,'input':inp,'router':rt,
              'common':base,'templates':templates,'pairs':pairs,'endpoint_request':req,'proof':proof}
    tick(); return result
