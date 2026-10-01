"""Checked-template source producer feeding original blocks, not joint HZs."""
from itertools import combinations
import time

from source_enclosure.format import identity
from source_enclosure import produce
from scoped_source.graph import validate, clock
from scoped_source.hz_lifted_source import owned, enclosure, propagate
from scoped_source.hz_source_build import live, entry_for, gate
from scoped_source.hz_source_check import properties
from scoped_source.hz_templates import make_common
from scoped_source.check_hz_block_source import SCHEMA, CONTEXT_GUARD
from act.back_end.moe.block_endpoints import prepare_request, propose_request


def build(doc, *, expected_source_sha256, deadline, observe=None):
    tick = clock(deadline); start = time.monotonic()
    r, lower, upper = validate(doc,expected_source_sha256,tick); props = properties(r)
    initial, inp = enclosure(owned(produce.box(lower,upper)),'input',deadline)
    router, rt = propagate(doc,'router',initial,'router',deadline)
    base = make_common(initial,router); states=[]; templates=[]
    for i in range(r['experts']):
        state, trace = propagate(doc,f'expert{i}',base,f'template/expert{i}',deadline)
        states.append(state); templates.append({'expert':i,'trace':trace})
    pairs=[]; live_pairs=[]
    for a,b in combinations(range(r['experts']),2):
        tick(); tag=f'pair{a}-{b}'
        ref=owned(entry_for(initial['state'],router['state'],(a,b),r['experts']),router,tag+'/guard')
        entry,record=enclosure(ref,tag+'/guard',deadline); evidence=gate(router['state'],(a,b))
        pairs.append({'pair':[a,b],'entry':record,'gate_evidence':evidence})
        live_pairs.append({'pair':[a,b],'common':live(base['state']),'entry':live(entry['state']),
                           'a':live(states[a]['state']),'b':live(states[b]['state']),'gate':evidence['bounds']})
    ctx={'request':expected_source_sha256,'domain':identity(r),'guard':CONTEXT_GUARD}
    req=prepare_request(live_pairs,props,experts=r['experts'],classes=r['classes'],context=ctx,deadline=deadline)
    made=time.monotonic()
    if observe:observe('construction',made-start)
    proof=propose_request(req,expected_request_sha256=identity(req),deadline=deadline)
    if observe:observe('proposals',time.monotonic()-made)
    if identity(doc)!=expected_source_sha256:raise ValueError('source changed during generation')
    result={'schema':SCHEMA,'source_sha256':expected_source_sha256,'input':inp,'router':rt,'common':base,
            'templates':templates,'pairs':pairs,'endpoint_request':req,'proof':proof}
    tick();return result
