"""Fresh exact source references, checked row enclosure and endpoint proposals."""
from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations
import time

from source_enclosure import produce
from source_enclosure.format import pack,unpack,identity
from scoped_source.graph import validate,operator,clock
from scoped_source.hz_binary64 import REFERENCE
from scoped_source.hz_row_enclosure import produce as row_lift
from scoped_source.hz_source_build import live,entry_for,gate
from scoped_source.hz_source_check import exact_float,properties
from scoped_source.check_hz_lifted_source import SCHEMA
from act.back_end.solver.solver_hz import sparse_hz_linear
from act.back_end.solver.hz_lp_export import snapshot
from act.back_end.moe.hz_endpoints import prepare_request,propose_request


def owned(state,previous=None,tag='input'):
    _,c,b=unpack(state)
    if previous is None:co,bo=[],[]
    else:co,bo=(list(previous['ownership'][k]) for k in ('continuous','binary'))
    return {'schema':REFERENCE,'state':state,
            'ownership':{'continuous':co+[tag]*(len(c)-len(co)),'binary':bo+[tag]*(len(b)-len(bo))}}


def enclosure(reference,tag,deadline):
    proof=row_lift(reference,expected_reference_sha256=identity(reference),owner=tag,deadline=deadline)
    target=proof['target'];actual=live(target['state'])
    record={'reference':reference,'enclosure':proof,'snapshot':snapshot(actual)}
    clock(deadline)();return target,record


def relu(state,tag):
    target,certificate=produce.relu(state,tag)
    h,c,b=unpack(target);n=len(unpack(state)[0]['ub']);k=certificate['branches'].count('unstable')
    # Match the prior HZ connection's actual-kernel blocked constraint order.
    order=list(range(n))+list(range(n,n+2*k,2))+list(range(n+1,n+2*k,2))
    for key in ('Auc','Aub','ub'):h[key]=[h[key][j] for j in order]
    return pack(h,c,b),certificate


def propagate(doc,name,initial,prefix,deadline):
    import numpy as np
    import scipy.sparse as sp
    tick=clock(deadline);state=initial;shape=doc['center']['shape'];trace=[]
    graph=next(g for g in doc['networks'] if g['name']==name)
    for i,layer in enumerate(graph['layers']):
        tick();tag=f'{prefix}/layer{i}';width=len(unpack(state['state'])[0]['c'])
        shape,op,bias=operator(shape,layer);nominal=certificate=None
        if layer['kind']=='Linear':
            rr=[];cc=[];data=[]
            for row,coefficients in enumerate(op):
                for col,v in coefficients.items():rr.append(row);cc.append(col);data.append(exact_float(v))
            matrix=sp.csr_matrix((data,(rr,cc)),shape=(len(op),width))
            nominal=snapshot(sparse_hz_linear(live(state['state']),matrix,np.array([exact_float(v) for v in bias])))
            ref,certificate=produce.affine(state['state'],op,bias,nominal,tag)
        elif layer['kind']=='ReLU':ref,certificate=relu(state['state'],tag)
        else:ref=deepcopy(state['state'])
        reference=owned(ref,state,tag);state,record=enclosure(reference,tag,deadline)
        trace.append({'index':i,'kind':layer['kind'],'nominal':nominal,'certificate':certificate,'lift':record})
    tick();return state,trace


def build(doc,*,expected_source_sha256,deadline,observe=None):
    tick=clock(deadline);start=time.monotonic();r,lower,upper=validate(doc,expected_source_sha256,tick)
    props=properties(r);initial,inp=enclosure(owned(produce.box(lower,upper)),'input',deadline)
    router,rt=propagate(doc,'router',initial,'router',deadline);pairs=[];live_pairs=[]
    for a,b in combinations(range(r['experts']),2):
        tick();tag=f'pair{a}-{b}'
        ref=owned(entry_for(initial['state'],router['state'],(a,b),r['experts']),router,tag+'/guard')
        entry,record=enclosure(ref,tag+'/guard',deadline)
        left,lt=propagate(doc,f'expert{a}',entry,tag+f'/expert{a}',deadline)
        right,bt=propagate(doc,f'expert{b}',entry,tag+f'/expert{b}',deadline)
        evidence=gate(router['state'],(a,b))
        pairs.append({'pair':[a,b],'entry':record,'a':lt,'b':bt,'gate_evidence':evidence})
        live_pairs.append({'pair':[a,b],'entry':live(entry['state']),'a':live(left['state']),
                           'b':live(right['state']),'gate':evidence['bounds']})
    context={'request':expected_source_sha256,'domain':identity(r),'guard':'ALL_TIE_LEGAL_PAIRS_LIFTED_V1'}
    req=prepare_request(live_pairs,props,experts=r['experts'],classes=r['classes'],context=context,deadline=deadline)
    made=time.monotonic()
    if observe:observe('construction',made-start)
    proof=propose_request(req,expected_request_sha256=identity(req),deadline=deadline)
    if observe:observe('proposals',time.monotonic()-made)
    if identity(doc)!=expected_source_sha256:raise ValueError('source changed during generation')
    tick()
    return {'schema':SCHEMA,'source_sha256':expected_source_sha256,'input':inp,'router':rt,
            'pairs':pairs,'endpoint_request':req,'proof':proof}
