"""Independent same-source representation bridge, no producer or optimizer.

Finite declared real graphs only. Existing source, map, MC and exact bound
checkers are reused without changing their mathematics or acceptance gate.
"""
from fractions import Fraction as F

from scoped_source.hz_source_check import check as check_source
from act.back_end.moe.check_hz_endpoints import _source, THRESHOLD
from act.back_end.solver.check_rational_mccormick import check_construction
from scoped_source.rowwise_bound import check_bound, clock, identity, rational, rows

SCHEMA='HZ_SOURCE_REPRESENTATION_V1'


def difference_bounds(source,q,deadline):
    """Independent expert projections on the original shared +/-1 factor box."""
    h,nc,nb=_source(source,clock(deadline)); m=len(q)
    if len(h['c'])!=2*m: raise ValueError('joint property dimensions')
    centers=[F(0),F(0)]; vectors=[[F(0)]*(nc+nb) for _ in range(2)]
    for side in range(2):
        for i,w in enumerate(map(rational,q)):
            r=side*m+i; centers[side]+=w*h['c'][r]
            for key,shift in [('Gc',0),('Gb',nc)]:
                for j,v in h[key][r].items(): vectors[side][j+shift]+=w*v
    center=centers[0]-centers[1]
    radius=sum((abs(a-b) for a,b in zip(*vectors)),F(0))
    return list(map(str,(center-radius,center+radius)))


def construction(pair,prop,record,deadline):
    clock(deadline)()
    src=pair['batch']['source']
    difference=difference_bounds(src,prop['q'],deadline)
    checked=check_construction(record,source_hash=identity(src),q=prop['q'],offset=prop['offset'],
                               gate=pair['gate']['bounds'],difference=difference)
    clock(deadline)()
    return checked


def check(doc,package,*,expected_source_sha256,expected_mode,deadline):
    tick=clock(deadline); before=identity(package)
    if (set(package)!={'schema','source_sha256','mode','lowering','proof','duties'}
            or package['schema']!=SCHEMA or package['source_sha256']!=expected_source_sha256
            or package['mode']!=expected_mode or expected_mode not in ('endpoints','mccormick')):
        raise ValueError('representation source/mode/schema')
    lower=package['lowering']; req=lower['endpoint_request']
    if any(p['candidates'] is not None for p in lower['proof']['pairs']):
        raise ValueError('source preparation contains foreign endpoint answers')
    if expected_mode=='endpoints':
        if package['duties']!=[] or package['proof'] is None: raise ValueError('endpoint proof inventory')
        accepted=check_source(doc,dict(lower,proof=package['proof']),
                              expected_source_sha256=expected_source_sha256,deadline=deadline)
        result={'status':accepted['status'],'required':accepted['required'],'positive':accepted['positive'],
                'checked_targets':accepted['checked_endpoints'],'missing_targets':accepted['missing_endpoints'],
                'results':accepted['results'],'source_lowering_checked':accepted['source_lowering_checked']}
    else:
        if package['proof'] is not None: raise ValueError('MC cannot borrow endpoint answers')
        # A missing endpoint proof checks source/maps/gate but establishes no
        # output result. MC obligations below must independently close all rows.
        lowered=check_source(doc,lower,expected_source_sha256=expected_source_sha256,deadline=deadline)
        expected=[(p,prop) for p in req['pairs'] for prop in req['properties']]
        if len(package['duties'])!=len(expected): raise ValueError('complete MC duty roster required')
        results=[]; missing=0
        for item,(pair,prop) in zip(package['duties'],expected):
            tick()
            if (set(item)!={'pair','property','construction','certificate'}
                    or item['pair']!=pair['pair'] or item['property']!=prop['id']):
                raise ValueError('MC pair/property coverage')
            construction(pair,prop,item['construction'],deadline)
            lp=item['construction']['lp']; candidate=item['certificate']
            bound=None; detail=None
            if candidate is None: missing+=1
            else:
                detail=check_bound(lp,candidate,deadline=deadline)
                bound=rational(detail['checked_lower_bound'])
            results.append({'pair':pair['pair'],'property':prop['id'],'lp_sha256':identity(lp),
                            'difference':item['construction']['difference'],'gate':item['construction']['gate'],
                            'lower_bound':None if bound is None else str(bound),
                            'positive':bound is not None and bound>THRESHOLD,'bound_check':detail})
        positive=sum(r['positive'] for r in results)
        result={'status':'CHECKED_POSITIVE_DECLARED_REAL_SOURCE' if positive==len(results)
                else 'UNKNOWN_MISSING_EVIDENCE' if missing else 'UNKNOWN_NONPOSITIVE',
                'required':len(results),'positive':positive,'checked_targets':len(results)-missing,
                'missing_targets':missing,'results':results,'source_lowering_checked':lowered['source_lowering_checked']}
    tick()
    if identity(package)!=before or identity(doc)!=expected_source_sha256:
        raise ValueError('source or representation changed during checking')
    return dict(result,source_sha256=expected_source_sha256,package_sha256=before,mode=expected_mode,
                deployed_float_SAFE=False,hard_budget_supervision=False,
                remaining_trust=['declaration_corresponds_to_intended_program','exact_checker_implementation'])


def check_point(lp,point,deadline):
    """Use only after construction validation; exact feasibility, not optimality."""
    tick=clock(deadline); x=list(map(rational,point)); n=len(lp['c'])
    if len(x)!=n: raise ValueError('exact point dimension')
    if any(not rational(a)<=v<=rational(b) for a,v,b in zip(lp['lower'],x,lp['upper'])):
        raise ValueError('exact point outside box')
    for name,rhs in [('A','b'),('E','h')]:
        for row,b in zip(rows(lp[name],(len(lp[rhs]),n),tick),lp[rhs]):
            value=sum((v*x[j] for j,v in row),F(0)); b=rational(b)
            if (value>b if name=='A' else value!=b): raise ValueError('exact point violates constraint')
    tick(); return rational(lp['offset'])+sum((rational(c)*v for c,v in zip(lp['c'],x)),F(0))
