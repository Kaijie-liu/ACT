"""Propose affine routing certificates using disjoint pooled input intervals."""
from fractions import Fraction as F
import itertools
from router_source.checker import compact,digest,inputs


def propose(doc):
    r=doc['request'];shape,p,params,image=inputs(doc,r)
    _,c,h,w=shape;ph,pw=h//p,w//p;n=c*ph*pw
    lows=[];highs=[]
    for channel in range(c):
        for y in range(ph):
            for x in range(pw):
                ids=[(channel*h+y*p+dy)*w+x*p+dx for dy in range(p) for dx in range(p)]
                lows.append(sum((image['lower'][i] for i in ids),F(0))/(p*p))
                highs.append(sum((image['upper'][i] for i in ids),F(0))/(p*p))
    margins={};records=[]
    for a,b in itertools.combinations(range(r['experts']),2):
        coeff=[params['weight'][a*n+j]-params['weight'][b*n+j] for j in range(n)]
        base=params['bias'][a]-params['bias'][b]
        lo=base+sum((v*(l if v>=0 else u) for v,l,u in zip(coeff,lows,highs)),F(0))
        hi=base+sum((v*(u if v>=0 else l) for v,l,u in zip(coeff,lows,highs)),F(0))
        margins[(a,b)]=(lo,hi);records.append({'pair':[a,b],'lower':str(lo),'upper':str(hi)})
    routes=[]
    for pair in itertools.combinations(range(r['experts']),2):
        row={'pair':list(pair),'kind':'covered'}
        for outside in range(r['experts']):
            if outside in pair:continue
            for inside in pair:
                bounds=margins[tuple(sorted((outside,inside)))]
                lower=bounds[0] if outside<inside else -bounds[1]
                if lower>0:
                    row={'pair':list(pair),'kind':'excluded','witness':[outside,inside],
                         'strict_lower_bound':str(lower)};break
            if row['kind']=='excluded':break
        routes.append(row)
    return {'schema':'AFFINE_ROUTER_BOX_COVER_V1','source_sha256':digest(compact(doc)),
            'margins':records,'routes':routes}
