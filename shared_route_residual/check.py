"""Independent simultaneous exact residual proof, no proposer/model/solver.

For score i=c_i+G_i xi and E xi=h, any equality potential z_i gives
i=(c_i+z_i h)+(G_i-z_i E)xi on the source domain. Difference the residual
VECTORS before box bounding. A difference of arbitrary lower bounds is invalid;
inequality multipliers are deliberately not supported in these potentials.
"""
from fractions import Fraction as F
from itertools import combinations

from checked_route_frontier.check import check_network
from scoped_source.graph import clock, validate
from source_enclosure.check import check_box
from source_enclosure.format import identity, sparse, unpack
from upstream_source.checker import csr
from shared_route_residual.format import SCHEMA, binding


def evaluate(state, potentials, *, deadline):
    """Exact algebra on a GIVEN stored HZ. Public check() supplies source checking.

No module-global cache, mutation, producer residual or certificate sign is
trusted. All source matrices, including unused inequalities, are parsed and
validated. Only nonzero equality potentials require exact row contractions.
"""
    tick=clock(deadline);h,ci,bi=unpack(state);tick()
    n=len(ci)+len(bi);e=len(h['c']);z=csr(potentials,[e,len(h['b'])])
    offsets=[];residuals=[];products=0
    for i,dual in enumerate(z):
        tick();offset=h['c'][i];r=dict(h['Gc'][i])
        r.update({j+len(ci):v for j,v in h['Gb'][i].items()})
        for row,weight in dual.items():
            tick();offset+=weight*h['b'][row]
            for col,value in h['Ac'][row].items():
                r[col]=r.get(col,F(0))-weight*value;products+=1
            for col,value in h['Ab'][row].items():
                col+=len(ci);r[col]=r.get(col,F(0))-weight*value;products+=1
        offsets.append(offset);residuals.append({j:v for j,v in r.items() if v})
    bounds=[];coordinates=0
    for higher in range(e):
        for lower in range(e):
            if higher==lower:continue
            tick();a,b=residuals[higher],residuals[lower]
            difference={j:a.get(j,F(0))-b.get(j,F(0)) for j in a.keys()|b.keys()}
            correction=-sum(map(abs,difference.values()),F(0))
            coordinates+=len(difference)
            value=offsets[higher]-offsets[lower]+correction
            bounds.append({'higher':higher,'lower':lower,'checked_lower_bound':str(value),
                'residual_box_term':str(correction),'nonzero_residual_coordinates':sum(v!=0 for v in difference.values()),
                'status':'STRICT_DOMINANCE' if value>0 else 'NOT_PROVED'})
    residual_identity=identity({'offsets':list(map(str,offsets)), 'rows':sparse(residuals,n)})
    tick()
    return {'bounds':bounds,'residual_identity':residual_identity,
        'counts':{'algebra_final_state_parses':1,'score_potentials':e,'derived_margins':len(bounds),
                  'nonzero_equality_products':products,'difference_coordinates_visited':coordinates},
        'scope':'arithmetic for a given HZ; source enclosure requires outer check'}


def check(doc, prefix, certificate, *, expected_source_sha256, invocation, deadline):
    tick=clock(deadline);request,lo,hi=validate(doc,expected_source_sha256,tick)
    if (set(prefix)!={'schema','source_sha256','input','router'} or
            prefix['schema']!='CHECKED_ROUTE_PREFIX_V1' or prefix['source_sha256']!=expected_source_sha256):
        raise ValueError('source-bound router prefix')
    root=prefix['input'];check_box(lo,hi,root)
    state=check_network(doc['networks'][0],prefix['router'],root,request['center']['shape'],tick)
    if (type(certificate)is not dict or set(certificate)!={'schema','binding','score_equalities'} or
            certificate['schema']!=SCHEMA or certificate['binding']!=binding(doc,prefix,invocation)):
        raise ValueError('request/source/factor/run certificate binding')
    result=evaluate(state,certificate['score_equalities'],deadline=deadline)
    proof_hash=identity(certificate);bounds=result['bounds'];pairs=[]
    for pair in combinations(range(request['experts']),2):
        tick();decision={'pair':list(pair),'status':'RETAINED'}
        witnesses=[b for b in bounds if b['status']=='STRICT_DOMINANCE' and b['lower'] in pair and b['higher'] not in pair]
        if witnesses:
            w=witnesses[0]
            decision.update(status='EXCLUDED_BY_CHECKED_STRICT_MARGIN',higher=w['higher'],lower=w['lower'],
                checked_lower_bound=w['checked_lower_bound'],evidence_sha256=identity({
                    'certificate_sha256':proof_hash,'residual_sha256':result['residual_identity'],
                    'higher':w['higher'],'lower':w['lower']}))
        pairs.append(decision)
    kept=[d['pair'] for d in pairs if d['status']=='RETAINED']
    if not kept:raise ValueError('all routes excluded on nonempty box')
    tick()
    return {'schema':'CHECKED_SHARED_ROUTER_RESIDUAL_V1','source_sha256':expected_source_sha256,
        'request_sha256':identity(request),'invocation':invocation,'certificate_sha256':proof_hash,
        'bounds':bounds,'pairs':pairs,'total_pairs':len(pairs),'retained_pairs':len(kept),
        'excluded_pairs':len(pairs)-len(kept),'needed_experts':sorted({i for p in kept for i in p}),
        'original_output_obligations':len(pairs)*(request['classes']-1),
        'retained_output_obligations':len(kept)*(request['classes']-1),
        'residual_identity':result['residual_identity'],'algebra_counts':result['counts'],
        'complete_output_positive_proof':False,'native_float_proof':False,'route_changing_established':False,
        'trusted':['declared graph/program correspondence','stored-center preprocessing','checker implementation/runtime']}
