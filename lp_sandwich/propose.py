"""Untrusted, one-shot candidate capture. Not wired into production.

Native LP status/objective/marginals and point are preserved BEFORE checking.
No retries, rounding repairs, basis reconstruction or replacement objective.
A future full request integration must provide its own outer process watchdog.
"""
from pathlib import Path
import json
import math
import time

from lp_sandwich.check import identity, rational, validate_statement, inspect_bounds, check, deadline_tick


def save_new(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def native_record(result):
    def scalar(v):
        if v is None:return None
        v=float(v)
        return v if math.isfinite(v) else str(v)
    def vector(v):return None if v is None else [scalar(x) for x in v]
    out={'success':bool(result.success),'status':int(result.status),'message':str(result.message),
         'objective_without_offset':scalar(result.fun),'x':vector(result.x),
         'iterations':None if getattr(result,'nit',None) is None else int(result.nit),
         'trusted':False,'meaning':'solver proposal metadata; not an optimality/feasibility certificate'}
    for key in ('ineqlin','eqlin','lower','upper'):
        part=getattr(result,key,None)
        out[key]={'residual':vector(getattr(part,'residual',None)),
                  'marginals':vector(getattr(part,'marginals',None))}
    return out


def bundle_from_native(lp,statement,native,tick=lambda:None):
    """May reject nonfinite/malformed native values; exact infeasibility is retained."""
    sha=identity(statement);validate_statement(statement,lp,sha,tick)
    x=native['x'];primal=None;dual=None
    if x is not None:
        if len(x)!=len(lp['c']):raise ValueError('native point dimensions')
        obj=rational(lp['offset'])+sum((rational(a)*rational(b) for a,b in zip(lp['c'],x)),rational(0))
        primal={'lp_sha256':identity(lp),'statement_sha256':sha,'x':x,'claimed_objective':str(obj)}
    y=native['ineqlin']['marginals'];z=native['eqlin']['marginals']
    if y is not None and z is not None:
        from act.back_end.solver.sparse_lp_certificate import evaluate
        dual={'lp_sha256':identity(lp),'inequality_dual':[str(min(rational(v),0)) for v in y],
              'equality_dual':[str(rational(v)) for v in z]}
        tick();dual['claimed_lower_bound']=str(evaluate(lp,dual)[0]);tick()
    return {'schema':'LP_SANDWICH_V1','statement':statement,'lp':lp,'primal':primal,'dual':dual}


def propose_to(lp,statement,destination,*,deadline):
    started=time.monotonic()
    if not math.isfinite(deadline) or deadline>started+300:raise ValueError('bounded original deadline required')
    tick=deadline_tick(deadline);tick();sha=identity(statement)
    validate_statement(statement,lp,sha,tick)
    # Validate every CSR/box field before passing it to the untrusted solver.
    inspect_bounds(lp,None,None,sha,tick)
    root=Path(destination).resolve()
    if not root.is_relative_to(Path('/data1/Kane/MOE')):raise ValueError('outside workspace')
    root.mkdir(exist_ok=False)
    save_new(root/'input.json',{'lp':lp,'statement':statement})
    calls=0;result=None;error=None;status='ERROR';native_seconds=None
    try:
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix
        def vector(values):return [float(rational(v)) for v in values]
        def matrix(name):
            m=lp[name]
            return None if not m['shape'][0] else csr_matrix((vector(m['data']),m['indices'],m['indptr']),shape=m['shape'])
        A,E=matrix('A'),matrix('E');tick()
        grant=min(60,deadline-time.monotonic())
        if grant<=0:raise TimeoutError('no solver time remaining')
        entered=time.monotonic();calls=1
        result=linprog(vector(lp['c']),A_ub=A,b_ub=vector(lp['b']) or None,
            A_eq=E,b_eq=vector(lp['h']) or None,bounds=list(zip(vector(lp['lower']),vector(lp['upper']))),
            method='highs',options={'time_limit':grant})
        native_seconds=time.monotonic()-entered
        native=native_record(result);native.update(lp_sha256=identity(lp),statement_sha256=sha,
            granted_seconds=grant,native_seconds=native_seconds)
        save_new(root/'native.json',native);tick()
        bundle=bundle_from_native(lp,statement,native,tick)
        save_new(root/'bundle.json',bundle);tick()
        diagnostic=check(bundle,sha,tick)
        save_new(root/'diagnostic.json',diagnostic);tick();status='COMPLETED_DIAGNOSTIC'
    except TimeoutError as exc:status='TIMEOUT';error=str(exc)
    except Exception as exc:error=repr(exc)
    terminal={'status':status,'error':error,'native_calls':calls,'native_seconds':native_seconds,
        'elapsed_seconds':time.monotonic()-started,'deadline_monotonic':deadline,
        'statement_sha256':sha,'lp_sha256':identity(lp),
        'outer_watchdog_provided':False,'network_SAFE':False,'network_UNSAFE':False,
        'scope':'candidate-capture component only, NOT an end-to-end production timing experiment'}
    if time.monotonic()>=deadline:terminal['status']='TIMEOUT'
    save_new(root/'terminal.json',terminal)
    return terminal
