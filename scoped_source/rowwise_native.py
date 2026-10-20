"""Opt-in native proposal adapter without the legacy list(rows(...)) copy.

Independent native-side parser. Only proposal calls import SciPy. Proposed floats
are not accepted without original-LP exact evaluation and independent checking.
Not a whole-request supervisor; use one caller-provided absolute deadline.
"""
from fractions import Fraction as F
import hashlib
import json
import math
import time


def _identity(value, work=None):
    if work is not None: work.tick()
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if work is not None: work.visit('coefficient_visits',(len(raw)+4095)//4096)
    result=hashlib.sha256(raw).hexdigest()
    if work is not None: work.tick()
    return result


def _rational(value, work=None):
    if work is not None:
        # The opt-in raw-proposal path shares the caller's stored-bit/deadline
        # contract. It must not enter the legacy unmetered exact-bound path.
        from scoped_source.check_margin_hz_source import parameter
        return parameter(value,work)
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


def native_rows(matrix, n, tick, work=None):
    if type(matrix) is not dict or not {'shape','indptr','indices','data'} <= set(matrix):
        raise ValueError('native CSR fields')
    shape=matrix['shape']; ptr=matrix['indptr']; columns=matrix['indices']; data=matrix['data']
    if (any(type(v) is not list for v in (shape,ptr,columns,data)) or len(shape)!=2
            or any(type(v) is not int or v<0 for v in shape) or shape[1]!=n
            or len(ptr)!=shape[0]+1 or ptr[0]!=0 or ptr[-1]!=len(data) or len(columns)!=len(data)):
        raise ValueError('native CSR dimensions')
    for i,v in enumerate(ptr):
        if i%256==0: tick()
        if work is not None: work.visit('coefficient_visits')
        if type(v) is not int or not 0<=v<=len(data): raise ValueError('native CSR pointer')
    for i in range(shape[0]):
        tick(); start,stop=ptr[i:i+2]
        if work is not None: work.visit('row_visits')
        if start>stop: raise ValueError('native CSR pointer order')
        result=[]; previous=-1
        for k in range(start,stop):
            if k%256==0: tick()
            j=columns[k]
            if type(j) is not int or not previous<j<n: raise ValueError('native CSR columns')
            previous=j; result.append((j,_rational(data[k],work)))
        tick(); yield result
    tick()


def validate_native(lp, *, deadline, work=None):
    if work is not None and work.deadline!=deadline:
        raise ValueError('native validation must continue caller deadline')
    tick=_clock(deadline); before=_identity(lp,work); tick()
    if type(lp) is not dict or not {'matrix_format','c','lower','upper','offset','A','b','E','h'}<=set(lp):
        raise ValueError('native LP fields')
    if any(type(lp[k]) is not list for k in ('c','lower','upper','b','h')):
        raise ValueError('native JSON vectors')
    if lp['matrix_format']!='csr_v1' or not len(lp['c']): raise ValueError('native LP format')
    n=len(lp['c']); total_rows=entries=maximum=0
    if len(lp['lower'])!=n or len(lp['upper'])!=n: raise ValueError('native box shape')
    _rational(lp['offset'],work)  # declared-source LP contract: offset is mandatory
    for j in range(n):
        if j%256==0: tick()
        _rational(lp['c'][j],work)
        if _rational(lp['lower'][j],work)>_rational(lp['upper'][j],work): raise ValueError('native box order')
    for matrix,rhs in (('A','b'),('E','h')):
        if lp[matrix]['shape'][0]!=len(lp[rhs]): raise ValueError('native row/RHS shape')
        for i,row in enumerate(native_rows(lp[matrix],n,tick,work)):
            _rational(lp[rhs][i],work); total_rows+=1; entries+=len(row); maximum=max(maximum,len(row)); del row
    tick()
    if _identity(lp,work)!=before: raise ValueError('LP changed during native validation')
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


class NativeProposalFailure(ValueError):
    """A numerical proposal outcome; never a route exclusion or proof."""
    def __init__(self, metadata):
        super().__init__('proposal solver did not complete')
        self.metadata=metadata


def _dual_proposal(lp, *, deadline, work=None, query_seconds=None, observe=None):
    """Shared HiGHS core; no exact objective claim or acceptance in this core."""
    if work is not None and work.deadline!=deadline:
        raise ValueError('native proposal must continue caller deadline')
    if query_seconds is not None and (type(query_seconds) not in (int,float)
            or not math.isfinite(query_seconds) or not 0<query_seconds<=300):
        raise ValueError('finite native query allowance')
    if observe is not None and not callable(observe): raise ValueError('native event callback')
    tick=_clock(deadline); validated=validate_native(lp,deadline=deadline,work=work); tick()
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix
    tick()
    def vector(values):
        result=[]
        for i,x in enumerate(values):
            if i%256==0: tick()
            value=float(_rational(x,work))
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
    if _identity(lp,work)!=validated['lp_sha256']: raise ValueError('LP changed before native call')
    if observe is not None:
        observe(dict(event='INTENT',lp_sha256=validated['lp_sha256'],
                     execution='NOT_YET_CONFIRMED_ENTERED',deadline=deadline))
        tick()
        if _identity(lp,work)!=validated['lp_sha256']: raise ValueError('LP changed in native intent')
    tick(); remaining=deadline-time.monotonic()
    if remaining<=0: raise TimeoutError('no native budget remains')
    allowance=min(remaining,query_seconds) if query_seconds is not None else remaining
    solve_started_at=time.monotonic()
    try:
        result=linprog(c,A_ub=a,b_ub=b,A_eq=e,b_eq=h,bounds=bounds,method='highs',options={'time_limit':allowance})
    except Exception as exc:
        if observe is not None:
            observe(dict(event='EXCEPTION',lp_sha256=validated['lp_sha256'],
                         exception_type=type(exc).__name__,seconds=time.monotonic()-solve_started_at))
        raise
    solve_finished_at=time.monotonic()
    # Legacy adapters/stubs promised only success and marginal arrays. Require
    # status/message only on the explicitly opted-in strong-reference path.
    if work is not None and type(result.status) is not int:
        raise ValueError('integer native status required')
    metadata=dict(lp_sha256=validated['lp_sha256'],
                  status=result.status if work is not None else None,
                  success=bool(result.success),message=str(result.message)[:512] if work is not None else '',
                  seconds=solve_finished_at-solve_started_at,started_at=solve_started_at,
                  finished_at=solve_finished_at,time_limit=allowance,
                  candidate_only=True,exact_optimality_checked=False)
    if observe is not None: observe(dict(event='RETURN',**metadata))
    tick()
    if _identity(lp,work)!=validated['lp_sha256']: raise ValueError('LP changed during native solve')
    if not result.success or (work is not None and metadata['status']!=0):
        raise NativeProposalFailure(metadata)
    def multipliers(values,signed):
        out=[]
        for i,v in enumerate(values):
            if i%256==0: tick()
            f=float(v)
            if not math.isfinite(f): raise ValueError('nonfinite native dual')
            if work is not None: _rational(f,work)
            out.append(min(0.,f) if signed else f)
        return out
    cert={'lp_sha256':validated['lp_sha256'],
          'inequality_dual':multipliers(result.ineqlin.marginals,True),
          'equality_dual':multipliers(result.eqlin.marginals,False)}
    if len(cert['inequality_dual'])!=len(lp['b']) or len(cert['equality_dual'])!=len(lp['h']):
        raise ValueError('native complete marginal rows')
    tick(); return cert,metadata


def propose_duals(lp, *, work, query_seconds, observe):
    """Opt-in raw marginals for existing metered source-aware receivers.

    Required callback publishes intent/return before exact receipt. It is not
    a supervisor. The caller must keep its same Work and hard deadline through
    claim construction, independent reception, serialization and cleanup.
    The caller must bound/freeze the LP before this low-level routine allocates
    its canonical identity or native arrays; margin_lp_candidates does so.
    There is deliberately no raw success→SAFE or lower-bound claim transition.
    """
    if work is None or not callable(observe): raise ValueError('continued Work and observer required')
    if type(query_seconds) not in (int,float) or not math.isfinite(query_seconds) or not 0<query_seconds<=300:
        raise ValueError('explicit finite native query allowance')
    work.tick()
    candidate,metadata=_dual_proposal(lp,deadline=work.deadline,work=work,
                                     query_seconds=query_seconds,observe=observe)
    work.tick(); return dict(candidate=candidate,metadata=metadata)


def propose(lp, *, deadline):
    """Legacy given-LP claim/check API; no opt-in development contract change."""
    tick=_clock(deadline)
    cert,_=_dual_proposal(lp,deadline=deadline)
    cert['claimed_lower_bound']=str(_evaluate(lp,cert,tick))
    from scoped_source.rowwise_bound import check_bound
    check_bound(lp,cert,deadline=deadline)
    if _identity(lp)!=cert['lp_sha256']: raise ValueError('LP changed after native call')
    tick(); return cert
