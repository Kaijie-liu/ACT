"""Standalone stdlib-only exact primal/dual check. Runs relocated with python -I -S.

Only the supplied rational LP is checked. A feasible relaxed point is NOT a
network counterexample. No feasibility tolerance, rounding repair or optimizer.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

MAX_BYTES = 128*1024*1024


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def rational(v):
    if type(v) is int:return F(v)
    if type(v) is float and math.isfinite(v):return F.from_float(v)
    if type(v) is str and len(v)<=8192:return F(v)
    raise ValueError('finite exact-binary float, integer or rational string required')


def strict_json(raw):
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('duplicate JSON key')
            d[k]=v
        return d
    def invalid(s):raise ValueError('nonfinite JSON number: '+s)
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)


def deadline_tick(deadline):
    def tick():
        if time.monotonic()>=deadline:raise TimeoutError('diagnostic deadline')
    return tick


def matrix_rows(m,n,tick):
    if set(m)!={'shape','indptr','indices','data'}:raise ValueError('CSR fields')
    shape,ptr,indices,data=(m[k] for k in ('shape','indptr','indices','data'))
    if (len(shape)!=2 or any(type(v)is not int or v<0 for v in shape) or shape[1]!=n or
        len(ptr)!=shape[0]+1 or ptr[0]!=0 or ptr[-1]!=len(data) or len(indices)!=len(data) or
        any(type(v)is not int for v in ptr+indices)):raise ValueError('CSR dimensions')
    for i in range(shape[0]):
        tick()
        if not 0<=ptr[i]<=ptr[i+1]<=len(data):raise ValueError('CSR pointer')
        row=[];previous=-1
        for k in range(ptr[i],ptr[i+1]):
            if k%256==0:tick()
            j=indices[k]
            if not previous<j<n:raise ValueError('CSR canonical index')
            row.append((j,rational(data[k])));previous=j
        yield row


def validate_statement(s,lp,expected,tick):
    tick()
    fields={'schema','request_id','source_sha256','export_sha256','pair','property_index',
            'property','lp_sha256','acceptance_threshold'}
    if set(s)!=fields or s['schema']!='LP_OBLIGATION_IDENTITY_V1' or identity(s)!=expected:
        raise ValueError('obligation binding')
    for key in ('request_id','source_sha256','export_sha256','lp_sha256'):
        v=s[key]
        if type(v)is not str or len(v)!=64 or any(c not in '0123456789abcdef' for c in v):
            raise ValueError('identity digest format')
    pair=s['pair']
    if (type(pair)is not list or len(pair)!=2 or any(type(i)is not int or i<0 for i in pair)
        or pair!=sorted(set(pair)) or type(s['property_index'])is not int or s['property_index']<0):
        raise ValueError('pair/property identity')
    prop=s['property']
    if set(prop)!={'q','constant'} or type(prop['q'])is not list or not prop['q']:
        raise ValueError('property fields')
    for v in prop['q']+[prop['constant']]:rational(v)
    threshold=rational(s['acceptance_threshold'])
    if threshold<0:raise ValueError('negative acceptance threshold')
    if identity(lp)!=s['lp_sha256']:raise ValueError('LP identity')
    tick();return threshold


def inspect_bounds(lp,primal,dual,statement_id,tick=lambda:None):
    """Every supplied constraint is checked even when a lower bound is already positive."""
    tick()
    if set(lp)!={'matrix_format','c','offset','lower','upper','A','b','E','h'} or lp['matrix_format']!='csr_v1':
        raise ValueError('sparse finite-box LP required')
    c,lo,hi=([rational(v) for v in lp[k]] for k in ('c','lower','upper'))
    offset=rational(lp['offset']);n=len(c);sha=identity(lp)
    if not n or len(lo)!=n or len(hi)!=n or any(a>b for a,b in zip(lo,hi)):
        raise ValueError('finite variable box')
    x=None
    if primal is not None:
        if (set(primal)!={'lp_sha256','statement_sha256','x','claimed_objective'} or
            primal['lp_sha256']!=sha or primal['statement_sha256']!=statement_id):
            raise ValueError('primal identity')
        x=[rational(v) for v in primal['x']]
        if len(x)!=n:raise ValueError('point dimensions')
    residual=c[:];base=offset
    if dual is not None:
        if (set(dual)!={'lp_sha256','inequality_dual','equality_dual','claimed_lower_bound'} or
                dual['lp_sha256']!=sha):raise ValueError('dual identity/fields')
        rational(dual['claimed_lower_bound'])
    counts={'box':0,'inequality':0,'equality':0};maxima={k:F(0) for k in counts};examples=[]
    def violation(kind,index,amount):
        if amount>0:
            counts[kind]+=1;maxima[kind]=max(maxima[kind],amount)
            if len(examples)<8:examples.append({'kind':kind,'index':index,'exact_violation':str(amount)})
    if x is not None:
        for j,(v,a,b) in enumerate(zip(x,lo,hi)):
            tick();violation('box',j,max(a-v,v-b,F(0)))
    row_counts={}
    for m,rhs,key,ineq in (('A','b','inequality_dual',True),('E','h','equality_dual',False)):
        right=[rational(v) for v in lp[rhs]]
        multipliers=None if dual is None else [rational(v) for v in dual[key]]
        if lp[m]['shape'][0]!=len(right) or multipliers is not None and len(multipliers)!=len(right):
            raise ValueError('row/rhs/dual length')
        row_counts[m]=len(right)
        for i,row in enumerate(matrix_rows(lp[m],n,tick)):
            if x is not None:
                lhs=sum((v*x[j] for j,v in row),F(0));delta=lhs-right[i]
                violation('inequality' if ineq else 'equality',i,max(delta,F(0)) if ineq else abs(delta))
            if multipliers is not None:
                d=multipliers[i]
                if ineq and d>0:raise ValueError('inequality dual must be nonpositive')
                base+=d*right[i]
                for j,v in row:residual[j]-=d*v
    value=None if x is None else offset+sum((a*b for a,b in zip(c,x)),F(0))
    if x is not None and rational(primal['claimed_objective'])!=value:
        raise ValueError('claimed objective differs from exact recomputation')
    feasible=x is not None and not any(counts.values())
    upper=value if feasible else None
    lower=None;correction=None
    if dual is not None:
        correction=sum((r*(a if r>=0 else b) for r,a,b in zip(residual,lo,hi)),F(0))
        lower=base+correction
        if rational(dual['claimed_lower_bound'])>lower:raise ValueError('dual overclaim')
    if lower is not None and upper is not None and lower>upper:raise ValueError('invalid bound ordering')
    tick()
    return {'primal_status':'MISSING' if x is None else 'EXACT_FEASIBLE' if feasible else 'NOT_EXACTLY_FEASIBLE',
            'violation_counts':counts,'maximum_exact_violations':{k:str(v) for k,v in maxima.items()},
            'first_violations':examples,'candidate_objective_NOT_UPPER_UNLESS_FEASIBLE':None if value is None else str(value),
            'lower_bound':None if lower is None else str(lower),
            'upper_bound':None if upper is None else str(upper),
            'residual_box_term':None if correction is None else str(correction),
            'checked_rows':row_counts,'checked_variables':n,
            'exact_gap':None if lower is None or upper is None else str(upper-lower),
            'exact_optimality':lower is not None and upper is not None and lower==upper}


def check(bundle,expected_statement,tick=lambda:None):
    tick()
    if set(bundle)!={'schema','statement','lp','primal','dual'} or bundle['schema']!='LP_SANDWICH_V1':
        raise ValueError('bundle fields/schema')
    threshold=validate_statement(bundle['statement'],bundle['lp'],expected_statement,tick)
    result=inspect_bounds(bundle['lp'],bundle['primal'],bundle['dual'],expected_statement,tick)
    lower=None if result['lower_bound'] is None else F(result['lower_bound'])
    upper=None if result['upper_bound'] is None else F(result['upper_bound'])
    if lower is not None and lower>threshold:state='LP_POSITIVE_LOWER_BOUND'
    elif upper is not None and upper<=0:state='LP_NONPOSITIVE_FEASIBLE_POINT'
    elif upper is not None and upper<=threshold:state='LP_UPPER_AT_OR_BELOW_ACCEPTANCE_THRESHOLD'
    else:state='UNRESOLVED_CANDIDATE_VS_LP_RELAXATION'
    tick()
    return {'status':'CHECKED_LP_DIAGNOSTIC','classification':state,**result,
            'statement_sha256':expected_statement,'lp_sha256':bundle['statement']['lp_sha256'],
            'acceptance_threshold':str(threshold),'network_SAFE':False,'network_UNSAFE':False,
            'scope':'Supplied exact rational continuous LP only; network-to-HZ, guards, route exclusions and property lowering NOT checked.'}


def main():
    started=time.monotonic();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle',type=Path);p.add_argument('--bundle-sha256',required=True)
    p.add_argument('--statement-sha256',required=True);p.add_argument('--timeout-seconds',type=float,required=True)
    a=p.parse_args()
    if not math.isfinite(a.timeout_seconds) or not 0<a.timeout_seconds<=300:
        p.error('timeout must be finite and in (0,300]')
    deadline=started+a.timeout_seconds;tick=deadline_tick(deadline)
    def alarm(signum,frame):raise TimeoutError('hard diagnostic deadline')
    signal.signal(signal.SIGALRM,alarm)
    try:
        signal.setitimer(signal.ITIMER_REAL,max(.000001,deadline-time.monotonic()))
        tick()
        if a.bundle.stat().st_size>MAX_BYTES:raise ValueError('bundle byte cap')
        raw=a.bundle.read_bytes();tick()
        if len(raw)>MAX_BYTES or hashlib.sha256(raw).hexdigest()!=a.bundle_sha256:
            raise ValueError('bundle bytes/identity')
        result=check(strict_json(raw),a.statement_sha256,tick)
        forbidden=('numpy','scipy','torch','highspy','gurobipy','act')
        imported=any(k.split('.')[0] in forbidden for k in sys.modules)
        if imported:raise ValueError('forbidden checker dependency')
        result.update(solver_or_model_imported=False,site_disabled=bool(sys.flags.no_site),
                      isolated=bool(sys.flags.isolated),seconds=time.monotonic()-started)
        tick();print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True);tick()
        return 0
    except TimeoutError:
        print(json.dumps({'status':'TIMEOUT','network_SAFE':False,'network_UNSAFE':False}),flush=True);return 3
    except Exception as exc:
        print(json.dumps({'status':'REJECTED','error':str(exc),'network_SAFE':False,'network_UNSAFE':False}),flush=True);return 2
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':sys.exit(main())
