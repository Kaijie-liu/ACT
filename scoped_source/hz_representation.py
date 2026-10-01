"""Finite opt-in representation control over fresh actual shared HybridZ.

No change to the production verifier or old propagation/optimization kernels.
"""
from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations
import time

from scoped_source.graph import validate
from scoped_source.hz_source_build import propagate,entry_for,gate,live
from scoped_source.hz_source_check import SCHEMA as SOURCE_SCHEMA,properties
from source_enclosure.produce import box
from source_enclosure.format import unpack
from act.back_end.moe.hz_endpoints import prepare_request,propose_request
from act.back_end.moe.batched_support import _candidate_columns
from act.back_end.solver.rational_mccormick import build as build_mc
from scoped_source.rowwise_native import _evaluate
from scoped_source.rowwise_bound import identity,rational,clock,rows
from scoped_source.check_hz_representation import SCHEMA


def prepare_source(doc,deadline):
    """Create no endpoint answers in the common source preparation."""
    tick=clock(deadline); anchor=identity(doc)
    r,lo,hi=validate(doc,anchor,tick); initial=box(lo,hi)
    router,rt=propagate(doc,'router',initial,'router',deadline)
    records=[]; pairs=[]
    for a,b in combinations(range(r['experts']),2):
        tick(); entry=entry_for(initial,router,(a,b),r['experts'])
        left,lt=propagate(doc,f'expert{a}',entry,f'pair{a}-{b}/expert{a}',deadline)
        right,bt=propagate(doc,f'expert{b}',entry,f'pair{a}-{b}/expert{b}',deadline)
        evidence=gate(router,(a,b))
        records.append({'pair':[a,b],'entry':entry,'a':lt,'b':bt,'gate_evidence':evidence})
        pairs.append({'pair':[a,b],'entry':live(entry),'a':live(left),'b':live(right),'gate':evidence['bounds']})
    req=prepare_request(pairs,properties(r),experts=r['experts'],classes=r['classes'],
                        context={'request':anchor,'domain':identity(r),'guard':'ALL_TIE_LEGAL_PAIRS'},deadline=deadline)
    proof={'schema':'GUARDED_HZ_ENDPOINT_PROOF_V1','request_sha256':identity(req),
           'pairs':[{'pair':p['pair'],'candidates':None} for p in req['pairs']]}
    tick()
    return {'schema':SOURCE_SCHEMA,'source_sha256':anchor,'input':initial,'router':rt,
            'pairs':records,'endpoint_request':req,'proof':proof}


def difference(source,q,deadline):
    """Producer computes q(A-B) directly; independent bridge does two projections."""
    tick=clock(deadline); nc=source['Gc']['shape'][1]; nb=source['Gb']['shape'][1]
    weights=list(map(rational,q)); weights+= [-v for v in weights]
    vector=[F(0)]*(nc+nb)
    for key,shift,width in [('Gc',0,nc),('Gb',nc,nb)]:
        for w,row in zip(weights,rows(source[key],(len(weights),width),tick)):
            for j,v in row: vector[j+shift]+=w*v
    constant=sum((w*rational(v) for w,v in zip(weights,source['c'])),F(0))
    radius=sum(map(abs,vector),F(0))
    return list(map(str,(constant-radius,constant+radius)))


def propose_lp(lp,deadline):
    """Same registered algorithm; only proposal-side RHS/box copies are floats."""
    tick=clock(deadline)
    base={k:v for k,v in lp.items() if k not in ('c','offset')}
    for k in ('b','h','lower','upper'): base[k]=[float(rational(v)) for v in base[k]]
    ys,ts=_candidate_columns(base,[[float(rational(v)) for v in lp['c']]],
                             [float(rational(lp['offset']))],deadline=deadline)
    if len(ys)!=1 or len(ts)!=1: raise ValueError('MC candidate count')
    candidate={'lp_sha256':identity(lp),'inequality_dual':ys[0],'equality_dual':ts[0]}
    zero={'lp_sha256':identity(lp),'inequality_dual':[0]*len(lp['b']),'equality_dual':[0]*len(lp['h'])}
    value=_evaluate(lp,candidate,tick); zero_value=_evaluate(lp,zero,tick)
    if zero_value>value: candidate,value=zero,zero_value
    candidate['claimed_lower_bound']=str(value)
    return candidate


def build(doc,mode,deadline,observe):
    if mode not in ('endpoints','mccormick'): raise ValueError('representation mode')
    t=time.monotonic(); lower=prepare_source(doc,deadline); observe('propagation_and_joint',time.monotonic()-t)
    req=lower['endpoint_request']
    result={'schema':SCHEMA,'source_sha256':identity(doc),'mode':mode,'lowering':lower,'proof':None,'duties':[]}
    t=time.monotonic()
    if mode=='mccormick':
        for p in req['pairs']:
            for prop in req['properties']:
                clock(deadline)(); src=p['batch']['source']
                record=build_mc(src,prop['q'],prop['offset'],p['gate']['bounds'],difference(src,prop['q'],deadline))
                result['duties'].append({'pair':p['pair'],'property':prop['id'],'construction':record,'certificate':None})
    observe('representation',time.monotonic()-t); t=time.monotonic()
    if mode=='endpoints': result['proof']=propose_request(req,expected_request_sha256=identity(req),deadline=deadline)
    else:
        for item in result['duties']: item['certificate']=propose_lp(item['construction']['lp'],deadline)
    observe('proposals',time.monotonic()-t); clock(deadline)()
    return result


def registered_point(package):
    """Frozen weighted-sign proposal only; independent exact check decides validity."""
    record=package['lowering']['pairs'][0]
    states=[unpack(record[k][-1]['target']) for k in ('a','b')]
    _,shared_c,shared_b=unpack(record['entry'])
    ci=list(states[0][1])+list(states[1][1][len(shared_c):])
    bi=list(states[0][2])+list(states[1][2][len(shared_b):])
    def value(name):
        if name.startswith('input/c/') or '/error/' in name: return '0'
        if '/negative/' in name: return '-1'
        if '/positive/' in name or '/sign/' in name: return '1'
        raise ValueError('registered point has no assignment for factor')
    return {'factor_ids':ci+bi,'point':[value(n) for n in ci+bi]+['1/4','-1/8']}
