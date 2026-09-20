"""Bounded untrusted range candidates, exact precheck, durable per-side records."""
import math
import time
from fractions import Fraction as F
from source_enclosure.format import unpack,identity,rational,sparse,clean
from source_ranges.produce import range_fact
from source_ranges.check import check_range
from act.back_end.solver.sparse_lp_certificate import evaluate
from full_bounds.worker import publish


def program(source,row,bias):
    s,c,b=unpack(source);n=len(c)+len(b);center=rational(bias);coeff={}
    for j,w in row.items():
        w=rational(w);center+=w*s['c'][j]
        for key,shift in [('Gc',0),('Gb',len(c))]:
            for k,v in s[key][j].items():coeff[k+shift]=coeff.get(k+shift,F(0))+w*v
    coeff=clean(coeff);radius=sum(map(abs,coeff.values()),F(0))
    lp={'matrix_format':'csr_v1','c':[str(coeff.get(i,F(0))) for i in range(n)],'offset':str(center),'lower':[-1]*n,'upper':[1]*n}
    for key,ck,bk,rhs in [('A','Auc','Aub','ub'),('E','Ac','Ab','b')]:
        lp[key]=sparse([{**x,**{j+len(c):v for j,v in y.items()}} for x,y in zip(s[ck],s[bk])],n)
        lp['b' if key=='A' else 'h']=list(map(str,s[rhs]))
    return lp,(center-radius,center+radius)


def solve(lp,limit):
    import importlib.metadata
    from scipy.optimize import linprog
    from scipy.optimize._highspy import _core
    from scipy.sparse import csr_matrix
    def mat(m):return csr_matrix(([float(F(v)) for v in m['data']],m['indices'],m['indptr']),shape=m['shape'])
    t=time.monotonic();c=list(map(lambda v:float(F(v)),lp['c']));a,e=mat(lp['A']),mat(lp['E'])
    b,h=[float(F(v)) for v in lp['b']],[float(F(v)) for v in lp['h']]
    conversion=time.monotonic()-t;t=time.monotonic()
    r=linprog(c,A_ub=a if b else None,b_ub=b or None,A_eq=e if h else None,b_eq=h or None,
        bounds=list(zip(lp['lower'],lp['upper'])),method='highs',options={'time_limit':limit,'threads':1})
    seconds=time.monotonic()-t;cert=None
    if r.success:
        yd=[min(0.,float(v)) for v in r.ineqlin.marginals];zd=list(map(float,r.eqlin.marginals))
        if any(not math.isfinite(v) for v in yd+zd):raise ValueError('nonfinite multipliers')
        cert={'lp_sha256':identity(lp),'inequality_dual':yd,'equality_dual':zd}
    return {'candidate':cert,'native_seconds':seconds,'conversion_seconds':conversion,
        'status':int(r.status),'success':bool(r.success),'message':str(r.message),'native_limit_seconds':limit,
        'environment':{'scipy':importlib.metadata.version('scipy'),'numpy':importlib.metadata.version('numpy'),
            'bundled_highs':'.'.join(str(getattr(_core,'HIGHS_VERSION_'+k)) for k in ('MAJOR','MINOR','PATCH')),
            'method':'highs','threads':1}}


def propose(source,row,bias,context,index,directory,key,deadline,limit=3.):
    start=time.monotonic();lp,box=program(source,row,bias);assembly=time.monotonic()-start
    neg={**lp,'c':[str(-F(v)) for v in lp['c']],'offset':str(-F(lp['offset']))};records=[];checks=[]
    for side,p in [('lower',lp),('negative_upper',neg)]:
        # If the build window is gone, do not start another native call.
        left=deadline-time.monotonic()
        if left<=0:break
        cap=min(limit,left)
        publish(directory/f'{key}_{side}_entered.json',{'context':context,'row':index,'source_sha256':identity(source),
            'lp_sha256':identity(p),'side':side,'native_limit_seconds':cap})
        record=solve(p,cap);records.append(record)
        # Returned candidates, including unsuccessful outcomes, survive a later failure.
        publish(directory/f'{key}_{side}.json',record)
    fact=None;check_seconds=0.
    if len(records)==2 and all(r['candidate'] is not None for r in records):
        t=time.monotonic();lo=evaluate(lp,records[0]['candidate'])[0];hi=-evaluate(neg,records[1]['candidate'])[0]
        lo,hi=max(lo,box[0]),min(hi,box[1])
        if lo>hi:raise ValueError('inconsistent range: no infeasibility/SAFE promotion')
        fact=range_fact(source,row,bias,context,index,[lo,hi],
            (records[0]['candidate']['inequality_dual'],records[0]['candidate']['equality_dual']),
            (records[1]['candidate']['inequality_dual'],records[1]['candidate']['equality_dual']))
        _,checked=check_range(source,row,bias,fact,context,index);checks.append(checked)
        check_seconds=time.monotonic()-t
    result={'context':context,'row':index,'source_sha256':identity(source),'calls':len(records),
        'outcome':'CHECKED_RANGE' if fact is not None else 'FALLBACK_NO_TWO_SIDED_CANDIDATE',
        'fact_sha256':identity(fact) if fact is not None else None,'assembly_seconds':assembly,
        'native_seconds':sum(r['native_seconds'] for r in records),
        'conversion_seconds':sum(r['conversion_seconds'] for r in records),'check_seconds':check_seconds,
        'seconds':time.monotonic()-start,'checks':checks,'generator_box':list(map(str,box)),
        'checked_range':fact['range'] if fact is not None else None,
        'strictly_tighter':fact is not None and (F(fact['range'][0])>box[0] or F(fact['range'][1])<box[1])}
    publish(directory/f'{key}_complete.json',result)
    return fact,result
