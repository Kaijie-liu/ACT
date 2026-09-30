"""Opt-in native proposal adapter without the legacy list(rows(...)) copy.

Independent native-side parser. Only propose() imports SciPy. Proposed floats
are not accepted without original-LP exact evaluation and independent checking.
Not a whole-request supervisor; use one caller-provided absolute deadline.
"""
from fractions import Fraction as F
import hashlib
import json
import math
import time


def _identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _rational(value):
    if type(value) is int or type(value) is str:
        return F(value)
    if type(value) is float and math.isfinite(value):
        return F.from_float(value)
    raise ValueError('native finite rational')


def _clock(deadline):
    if type(deadline) not in (int,float) or not math.isfinite(deadline) or deadline-time.monotonic()>300:
        raise ValueError('bounded absolute native deadline')
    def tick():
        if time.monotonic()>=deadline: raise TimeoutError('rowwise native deadline')
    tick(); return tick


def native_rows(matrix, n, tick):
    if type(matrix) is not dict or not {'shape','indptr','indices','data'} <= set(matrix):
        raise ValueError('native CSR fields')
    shape=matrix['shape']; ptr=matrix['indptr']; columns=matrix['indices']; data=matrix['data']
    if (any(type(v) is not list for v in (shape,ptr,columns,data)) or len(shape)!=2
            or any(type(v) is not int or v<0 for v in shape) or shape[1]!=n
            or len(ptr)!=shape[0]+1 or ptr[0]!=0 or ptr[-1]!=len(data) or len(columns)!=len(data)):
        raise ValueError('native CSR dimensions')
    for i,v in enumerate(ptr):
        if i%256==0: tick()
        if type(v) is not int or not 0<=v<=len(data): raise ValueError('native CSR pointer')
    for i in range(shape[0]):
        tick(); start,stop=ptr[i:i+2]
        if start>stop: raise ValueError('native CSR pointer order')
        result=[]; previous=-1
        for k in range(start,stop):
            if k%256==0: tick()
            j=columns[k]
            if type(j) is not int or not previous<j<n: raise ValueError('native CSR columns')
            previous=j; result.append((j,_rational(data[k])))
        tick(); yield result
    tick()


def validate_native(lp, *, deadline):
    tick=_clock(deadline); before=_identity(lp); tick()
    if type(lp) is not dict or not {'matrix_format','c','lower','upper','offset','A','b','E','h'}<=set(lp):
        raise ValueError('native LP fields')
    if any(type(lp[k]) is not list for k in ('c','lower','upper','b','h')):
        raise ValueError('native JSON vectors')
    if lp['matrix_format']!='csr_v1' or not len(lp['c']): raise ValueError('native LP format')
    n=len(lp['c']); total_rows=entries=maximum=0
    if len(lp['lower'])!=n or len(lp['upper'])!=n: raise ValueError('native box shape')
    _rational(lp['offset'])  # declared-source LP contract: offset is mandatory
    for j in range(n):
        if j%256==0: tick()
        _rational(lp['c'][j])
        if _rational(lp['lower'][j])>_rational(lp['upper'][j]): raise ValueError('native box order')
    for matrix,rhs in (('A','b'),('E','h')):
        if lp[matrix]['shape'][0]!=len(lp[rhs]): raise ValueError('native row/RHS shape')
        for i,row in enumerate(native_rows(lp[matrix],n,tick)):
            _rational(lp[rhs][i]); total_rows+=1; entries+=len(row); maximum=max(maximum,len(row)); del row
    tick()
    if _identity(lp)!=before: raise ValueError('LP changed during native validation')
    tick(); return {'lp_sha256':before,'rows_checked':total_rows,'entries_checked':entries,'maximum_row_entries':maximum}


def _evaluate(lp, candidate, tick):
    """Untrusted claim construction; separate from the independent checker."""
    residual=[_rational(v) for v in lp['c']]; value=_rational(lp['offset'])
    for matrix,rhs,key,signed in (('A','b','inequality_dual',True),('E','h','equality_dual',False)):
        dual=candidate[key]
        if lp[matrix]['shape'][0]!=len(lp[rhs]) or len(dual)!=len(lp[rhs]): raise ValueError('native dual coverage')
        for i,row in enumerate(native_rows(lp[matrix],len(residual),tick)):
            d=_rational(dual[i])
            if signed and d>0: raise ValueError('native dual sign')
            value+=d*_rational(lp[rhs][i])
            for p,(j,v) in enumerate(row):
                if p%256==0: tick()
                residual[j]-=d*v
            del row
    for i,r in enumerate(residual):
        if i%256==0: tick()
        value+=min(r*_rational(lp['lower'][i]),r*_rational(lp['upper'][i]))
    tick(); return value


def propose(lp, *, deadline):
    tick=_clock(deadline); validated=validate_native(lp,deadline=deadline); tick()
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix
    tick()
    def vector(values):
        result=[]
        for i,x in enumerate(values):
            if i%256==0: tick()
            value=float(_rational(x))
            if not math.isfinite(value): raise ValueError('nonfinite native conversion')
            result.append(value)
        tick(); return result
    def matrix(name):
        m=lp[name]
        if not m['shape'][0]: return None
        # validate_native exhausted all exact rows first; no all-row list here.
        result=csr_matrix((vector(m['data']),m['indices'],m['indptr']),shape=m['shape']); tick(); return result
    c=vector(lp['c']); a=matrix('A'); b=vector(lp['b']) or None
    e=matrix('E'); h=vector(lp['h']) or None
    bounds=list(zip(vector(lp['lower']),vector(lp['upper'])))
    if _identity(lp)!=validated['lp_sha256']: raise ValueError('LP changed before native call')
    tick(); remaining=deadline-time.monotonic()
    if remaining<=0: raise TimeoutError('no native budget remains')
    result=linprog(c,A_ub=a,b_ub=b,A_eq=e,b_eq=h,bounds=bounds,method='highs',options={'time_limit':remaining})
    tick()
    if not result.success: raise ValueError('proposal solver did not complete')
    def multipliers(values,signed):
        out=[]
        for i,v in enumerate(values):
            if i%256==0: tick()
            f=float(v)
            if not math.isfinite(f): raise ValueError('nonfinite native dual')
            out.append(min(0.,f) if signed else f)
        return out
    cert={'lp_sha256':validated['lp_sha256'],
          'inequality_dual':multipliers(result.ineqlin.marginals,True),
          'equality_dual':multipliers(result.eqlin.marginals,False)}
    cert['claimed_lower_bound']=str(_evaluate(lp,cert,tick))
    from scoped_source.rowwise_bound import check_bound
    check_bound(lp,cert,deadline=deadline)
    if _identity(lp)!=validated['lp_sha256']: raise ValueError('LP changed after native call')
    tick(); return cert
