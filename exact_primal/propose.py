"""Untrusted active-face candidate construction; never certifies feasibility.

No optimizer, no modification of the LP, no native-point repair in old runs.
A separate unchanged lp_sandwich checker must inspect ALL original constraints.
This small control implementation is intentionally capped, not a large LP solver.
"""
from fractions import Fraction as F
import math
import time
from lp_sandwich.check import identity,rational,validate_statement,inspect_bounds,matrix_rows

POLICY={'max_variables':64,'max_rows':512,'max_nnz':8192,'max_bits':4096,
        'max_operations':200000,'active_hint_radius':'1/100000000',
        'attempts':1,'free_variables':'original exact-binary candidate values',
        'acceptance_tolerance':0,'solver_calls':0}

class Limit(Exception):pass
class Inconsistent(Exception):pass

class Budget:
    def __init__(self,deadline):self.deadline=deadline;self.operations=0
    def tick(self):
        if time.monotonic()>=self.deadline:raise TimeoutError('original deadline')
    def value(self,v):
        self.tick();self.operations+=1
        if self.operations>POLICY['max_operations']:raise Limit('operation cap')
        v=rational(v) if not isinstance(v,F) else v
        if max(abs(v.numerator).bit_length(),v.denominator.bit_length())>POLICY['max_bits']:
            raise Limit('rational bit cap')
        return v

def system_solution(rows,point,budget):
    """Sparse rational forward elimination/back substitution, stable row order.

    Dependent inconsistent hinted rows terminate this attempt, NOT prove the
    original LP infeasible. No row is relaxed, deleted or searched over.
    """
    pivots={}
    for coeff,rhs in rows:
        row={j:budget.value(v) for j,v in coeff.items() if v};rhs=budget.value(rhs)
        for j,(known,right) in sorted(pivots.items()):
            if j not in row:continue
            scale=row.pop(j)
            rhs=budget.value(rhs-scale*right)
            for k,v in known.items():
                if k==j:continue
                value=budget.value(row.get(k,F(0))-scale*v)
                if value:row[k]=value
                else:row.pop(k,None)
        if not row:
            if rhs:raise Inconsistent('selected active system inconsistent')
            continue
        j=min(row);scale=row[j]
        pivots[j]=({k:budget.value(v/scale) for k,v in row.items()},budget.value(rhs/scale))
    x=point[:]
    for j,(row,rhs) in sorted(pivots.items(),reverse=True):
        value=rhs
        for k,v in row.items():
            if k!=j:value=budget.value(value-v*x[k])
        x[j]=budget.value(value)
    return x,sorted(pivots)

def propose(lp,statement,candidate,expected_statement,*,deadline):
    started=time.monotonic()
    if not math.isfinite(deadline) or deadline>started+300:raise ValueError('bounded original deadline required')
    b=Budget(deadline);selected=[];out=None;error=None;status='ERROR';pivots=[]
    # Input statements and the unchanged LP remain independently hash-bound.
    result={'schema':'EXACT_PRIMAL_CANDIDATE_V1','policy':dict(POLICY),'statement_sha256':expected_statement,
        'network_SAFE':False,'network_UNSAFE':False,'feasibility_certified':False,'solver_calls':0,
        'deadline_monotonic':deadline,'attempts':1}
    try:
        b.tick()
        if len(lp['c'])>POLICY['max_variables'] or len(lp['b'])+len(lp['h'])>POLICY['max_rows'] or\
           len(lp['A']['data'])+len(lp['E']['data'])>POLICY['max_nnz']:raise Limit('input size cap')
        for name in ('c','lower','upper','b','h'):
            for v in lp[name]:b.value(v)
        b.value(lp['offset'])
        for name in ('A','E'):
            for v in lp[name]['data']:b.value(v)
        validate_statement(statement,lp,expected_statement,b.tick)
        inspect_bounds(lp,None,None,expected_statement,b.tick)
        if set(candidate)!={'statement_sha256','lp_sha256','x'} or\
           candidate['statement_sha256']!=expected_statement or candidate['lp_sha256']!=identity(lp):
            raise ValueError('candidate provenance')
        point=[b.value(v) for v in candidate['x']]
        n=len(point)
        if n!=len(lp['c']):raise ValueError('point dimensions')
        result.update(lp_sha256=identity(lp),input_candidate_sha256=identity(candidate))
        lo=[b.value(v) for v in lp['lower']];hi=[b.value(v) for v in lp['upper']]
        rows=[]
        for i,row in enumerate(matrix_rows(lp['E'],n,b.tick)):
            rows.append((dict(row),b.value(lp['h'][i])));selected.append({'kind':'equality','index':i})
        radius=F(POLICY['active_hint_radius'])
        for j,(x,l,u) in enumerate(zip(point,lo,hi)):
            if l==u:
                rows.append(({j:F(1)},l));selected.append({'kind':'fixed_box','index':j})
            else:
                for name,v in (('lower',l),('upper',u)):
                    if abs(x-v)<=radius:
                        rows.append(({j:F(1)},v));selected.append({'kind':name,'index':j})
        for i,row in enumerate(matrix_rows(lp['A'],n,b.tick)):
            rhs=b.value(lp['b'][i]);lhs=F(0)
            for j,v in row:lhs=b.value(lhs+v*point[j])
            if abs(lhs-rhs)<=radius:
                rows.append((dict(row),rhs));selected.append({'kind':'inequality','index':i})
        x,pivots=system_solution(rows,point,b)
        value=b.value(lp['offset'])
        for a,v in zip(lp['c'],x):value=b.value(value+b.value(a)*v)
        out={'schema':'LP_SANDWICH_V1','lp':lp,'statement':statement,'dual':None,
             'primal':{'lp_sha256':identity(lp),'statement_sha256':expected_statement,
                      'x':[str(v) for v in x],'claimed_objective':str(value)}}
        b.tick();status='CANDIDATE_ONLY'
    except Inconsistent as exc:status='UNRESOLVED_ACTIVE_SYSTEM';error=str(exc)
    except Limit as exc:status='LIMIT';error=str(exc)
    except TimeoutError as exc:status='TIMEOUT';error=str(exc)
    except Exception as exc:status='ERROR';error=repr(exc)
    result.update(status=status,error=error,selected_rows=selected,pivot_columns=pivots,
        operations=b.operations,seconds=time.monotonic()-started,bundle=out,
        scope='untrusted candidate construction only; requires isolated full original-LP check')
    return result
