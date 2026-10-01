"""Opt-in H2 endpoint objectives on actual shared/private SparseHZono pairs.

No production default change, native solve or CUDA. Given-HZ/gate only.
"""
from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations
import time

from act.back_end.moe.batched_support import prepare_batch, propose_batch
from act.back_end.moe.check_hz_endpoints import SCHEMA, PROOF, validate_request, check_request
from act.back_end.moe.weighted_top2 import shared_input_pair_hz, independent_input_pair_hz
from scoped_source.rowwise_bound import clock, identity, rational


def prepare_request(pairs, properties, *, experts, classes, context, deadline, relation_mode='shared_input'):
    """Pairs supply entry/a/b live HZs and gate; no unchecked pair is dropped."""
    tick=clock(deadline)
    if type(experts) is not int or not 2<=experts<=4: raise ValueError('finite expert capacity')
    if [p['pair'] for p in pairs]!=[list(p) for p in combinations(range(experts),2)]:
        raise ValueError('complete unordered pair input roster')
    props=[{'id':p['id'],'q':[str(rational(v)) for v in p['q']],
            'offset':str(rational(p['offset']))} for p in properties]
    builder={'shared_input':shared_input_pair_hz,'independent_inputs':independent_input_pair_hz}.get(relation_mode)
    if builder is None: raise ValueError('registered relation mode required')
    request={'schema':SCHEMA,'experts':experts,'classes':classes,'context':deepcopy(context),'properties':props,'pairs':[]}
    for p in pairs:
        tick(); sources={}
        for name in ('entry','a','b'):
            # The support preparer rejects noncanonical CSR before snapshotting.
            zero=[{'id':'snapshot','q':[0]*p[name].n_out,'offset':0,'side':'min'}]
            sources[name]=prepare_batch(p[name],zero,context=context,deadline=deadline)['source']
        gate=[rational(v) for v in p['gate']]
        if len(gate)!=2 or not 0<=gate[0]<=gate[1]<=1: raise ValueError('gate interval')
        joint=builder(p['entry'],p['a'],p['b']); queries=[]
        for pi,prop in enumerate(props):
            for wi,t in enumerate(sorted(set(gate))):
                # Producer uses B+t(A-B); checker independently uses convex weights.
                q=[F(0)]*joint.output_hz.n_out
                for ai,bi,w in zip(joint.a_rows,joint.b_rows,map(rational,prop['q'])):
                    q[bi]+=w; q[ai]+=t*w; q[bi]-=t*w
                queries.append({'id':f'p{pi}:e{wi}','q':list(map(str,q)),'offset':prop['offset'],'side':'min'})
        ctx={'request':context['request'],'domain':context['domain'],'guard':identity(sources['entry']),
             'caller_context':identity(context),'pair':list(p['pair']),'relation_mode':relation_mode}
        batch=prepare_batch(joint.output_hz,queries,context=ctx,deadline=deadline)
        request['pairs'].append({'pair':list(p['pair']),'sources':sources,'relation_mode':relation_mode,'batch':batch,
                                'gate':{'pair':list(p['pair']),'domain':context['domain'],
                                        'guard_sha256':identity(sources['entry']),'bounds':list(map(str,gate)),
                                        'premise':'SUPPLIED_GATE_RANGE_NOT_INDEPENDENTLY_PROVED'}})
    validate_request(request,expected_request_sha256=identity(request),deadline=deadline); tick()
    return request


def propose_request(request, *, expected_request_sha256, deadline, device='cpu'):
    tick=clock(deadline)
    if device!='cpu': raise ValueError('only finite CPU endpoint controls admitted')
    pairs=validate_request(request,expected_request_sha256=expected_request_sha256,deadline=deadline)
    proof={'schema':PROOF,'request_sha256':expected_request_sha256,'pairs':[]}
    for p in pairs:
        tick(); batch=p['batch']
        candidates=propose_batch(batch,expected_batch_sha256=identity(batch),deadline=deadline)
        proof['pairs'].append({'pair':p['pair'][:],'candidates':candidates})
    tick()
    if identity(request)!=expected_request_sha256: raise ValueError('proposal request pollution')
    return proof


def support_endpoints(pairs, properties, *, experts, classes, context, deadline, relation_mode='shared_input'):
    start=time.monotonic()
    request=prepare_request(pairs,properties,experts=experts,classes=classes,context=context,
                            deadline=deadline,relation_mode=relation_mode)
    prepared=time.monotonic(); anchor=identity(request)
    proof=propose_request(request,expected_request_sha256=anchor,deadline=deadline)
    proposed=time.monotonic()
    checked=check_request(request,proof,expected_request_sha256=anchor,deadline=deadline)
    end=time.monotonic(); clock(deadline)()
    return {'request':request,'proof':proof,'checked':checked,
            'cost_seconds':{'prepare':prepared-start,'propose':proposed-prepared,'check':end-proposed,'total':end-start}}
