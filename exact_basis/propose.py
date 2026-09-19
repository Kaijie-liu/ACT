"""Untrusted original-coordinate basis proposal, no optimizer or acceptance.

M z = rhs with z=(x,s), E x=h and A x+s=b. Nonbasic anchors are
explicit; sparse exact elimination constructs basic values. The original LP
checker, not this generator, decides feasibility and objective validity.
"""
from fractions import Fraction as F
import math
import time
from lp_sandwich.check import identity,validate_statement,inspect_bounds,matrix_rows
from exact_primal.propose import Budget,Limit

COORDINATES='ORIGINAL_X_PLUS_POSITIVE_AX_LE_B_SLACK'
LIMITS={'variables':64,'equations':64,'input_nnz':8192,'live_elimination_nnz':4096}

class Singular(Exception):pass

def col(kind,index):return {'kind':kind,'index':index}

def rows_for(lp,tick=lambda:None):
    n=len(lp['c']);out=[]
    for kind,rhs in (('E','h'),('A','b')):
        for i,row in enumerate(matrix_rows(lp[kind],n,tick)):
            out.append({'kind':kind,'index':i,'row_sha256':identity({
                'entries':[[j,str(v)] for j,v in row],'rhs':str(F(lp[rhs][i]))})})
    return out

def manifest(lp,statement,candidate,basic_columns,anchors):
    """Make a declarative hint; does not validate or prove the chosen basis."""
    return {'schema':'ORIGINAL_LP_BASIS_V1','coordinates':COORDINATES,
        'lp_sha256':identity(lp),'statement_sha256':identity(statement),
        'candidate_sha256':identity(candidate),'rows':rows_for(lp),
        'basic_columns':basic_columns,'anchors':anchors}

def key(c,n,na):
    if set(c)!={'kind','index'} or type(c['index']) is not int:raise ValueError('column format')
    k,i=c['kind'],c['index']
    if k not in ('x','slack') or not 0<=i<(n if k=='x' else na):raise ValueError('original column index')
    return (k,i)

def validate(lp,s,c,h,expected_s,expected_h,budget):
    budget.tick();n=len(lp['c']);m=len(lp['b'])+len(lp['h']);na=len(lp['b'])
    if n>LIMITS['variables'] or m>LIMITS['equations'] or\
       len(lp['A']['data'])+len(lp['E']['data'])>LIMITS['input_nnz']:raise Limit('control size cap')
    for name in ('c','lower','upper','b','h'):
        for v in lp[name]:budget.value(v)
    budget.value(lp['offset'])
    for name in ('A','E'):
        for v in lp[name]['data']:budget.value(v)
    validate_statement(s,lp,expected_s,budget.tick);inspect_bounds(lp,None,None,expected_s,budget.tick)
    if set(c)!={'lp_sha256','statement_sha256','x'} or c['lp_sha256']!=identity(lp) or\
       c['statement_sha256']!=expected_s or len(c['x'])!=n:raise ValueError('candidate binding')
    point=[budget.value(v) for v in c['x']]
    if set(h)!={'schema','coordinates','lp_sha256','statement_sha256','candidate_sha256','rows','basic_columns','anchors'} or\
       h['schema']!='ORIGINAL_LP_BASIS_V1' or h['coordinates']!=COORDINATES or identity(h)!=expected_h or\
       (h['lp_sha256'],h['statement_sha256'],h['candidate_sha256'])!=(identity(lp),expected_s,identity(c)):
        raise ValueError('mapping binding / unsupported transformed coordinates')
    expected_rows=rows_for(lp,budget.tick)
    if len(h['rows'])!=m or sorted(h['rows'],key=lambda r:(r['kind'],r['index']))!=\
       sorted(expected_rows,key=lambda r:(r['kind'],r['index'])):raise ValueError('original row coverage/hash')
    basic=[key(x,n,na) for x in h['basic_columns']]
    if len(basic)!=m or len(set(basic))!=m:raise ValueError('basis column count/uniqueness')
    anchors={}
    for item in h['anchors']:
        if set(item)!={'column','at'}:raise ValueError('anchor fields')
        k=key(item['column'],n,na);at=item['at']
        if k in anchors or k in basic:raise ValueError('duplicate/overlapping anchor')
        if k[0]=='slack':
            if at!='zero':raise ValueError('slack anchor must be original slack zero')
            value=F(0)
        else:
            if at not in ('lower','upper','candidate'):raise ValueError('unbound anchor value')
            value=point[k[1]] if at=='candidate' else budget.value(lp[at][k[1]])
        anchors[k]=value
    allcols={('x',i) for i in range(n)}|{('slack',i) for i in range(na)}
    if set(basic)|set(anchors)!=allcols:raise ValueError('missing original coordinate')
    return basic,anchors

def solve(rows,budget,stats):
    """Fixed-order sparse elimination; exact row pivoting, no alternative basis."""
    m=len(rows);pivots={};live=0
    def observe(size):
        stats['peak_elimination_nnz']=max(stats['peak_elimination_nnz'],size)
        if size>LIMITS['live_elimination_nnz']:raise Limit('sparse fill-in cap')
    for coeff,right in rows:
        row=dict(coeff);rhs=right;observe(live+len(row))
        for j,(known,b) in sorted(pivots.items()):
            if j not in row:continue
            scale=row.pop(j);rhs=budget.value(rhs-scale*b)
            for k,v in known.items():
                if k==j:continue
                old=k in row;value=budget.value(row.get(k,F(0))-scale*v)
                if value:
                    row[k]=value
                    if not old:stats['fill_in_insertions']+=1
                else:row.pop(k,None)
                observe(live+len(row))
        if not row:raise Singular('dependent or inconsistent selected square basis')
        j=min(row);scale=row[j]
        normalized={k:budget.value(v/scale) for k,v in row.items()}
        pivots[j]=(normalized,budget.value(rhs/scale));live+=len(normalized)
    if set(pivots)!=set(range(m)):raise Singular('basis rank incomplete')
    values=[F(0)]*m
    for j,(row,rhs) in sorted(pivots.items(),reverse=True):
        for k,v in row.items():
            if k!=j:rhs=budget.value(rhs-v*values[k])
        values[j]=rhs
    return values

def propose(lp,statement,candidate,hint,expected_statement,expected_hint,*,deadline):
    start=time.monotonic()
    if not math.isfinite(deadline) or deadline>start+300:raise ValueError('bounded original deadline required')
    b=Budget(deadline);status='ERROR';error=None;bundle=None;slacks=None
    stats={'peak_elimination_nnz':0,'fill_in_insertions':0};assembled=None
    try:
        basic,anchors=validate(lp,statement,candidate,hint,expected_statement,expected_hint,b)
        lookup={k:i for i,k in enumerate(basic)};original={};n=len(lp['c'])
        for kind,rhs in (('E','h'),('A','b')):
            for i,entries in enumerate(matrix_rows(lp[kind],n,b.tick)):
                row={('x',j):v for j,v in entries if v}
                if kind=='A':row[('slack',i)]=F(1)
                original[(kind,i)]=(row,b.value(lp[rhs][i]))
        system=[]
        for ref in hint['rows']:
            row,rhs=original[(ref['kind'],ref['index'])];reduced={}
            for k,v in row.items():
                if k in lookup:reduced[lookup[k]]=v
                else:rhs=b.value(rhs-v*anchors[k])
            system.append((reduced,rhs))
        assembled=identity([{'row':[[j,str(v)] for j,v in sorted(r.items())],'rhs':str(h)} for r,h in system])
        basic_values=solve(system,b,stats);values={**anchors,**dict(zip(basic,basic_values))}
        x=[values[('x',i)] for i in range(n)];slacks=[str(values[('slack',i)]) for i in range(len(lp['b']))]
        objective=b.value(lp['offset'])
        for c,v in zip(lp['c'],x):objective=b.value(objective+b.value(c)*v)
        bundle={'schema':'LP_SANDWICH_V1','lp':lp,'statement':statement,'dual':None,
            'primal':{'lp_sha256':identity(lp),'statement_sha256':expected_statement,'x':[str(v) for v in x],
                      'claimed_objective':str(objective)}}
        b.tick();status='CANDIDATE_ONLY'
    except Singular as exc:status='UNRESOLVED_SINGULAR_BASIS';error=str(exc)
    except Limit as exc:status='LIMIT';error=str(exc)
    except TimeoutError as exc:status='TIMEOUT';error=str(exc)
    except Exception as exc:error=repr(exc)
    if status!='CANDIDATE_ONLY':bundle=None;slacks=None
    return {'schema':'ORIGINAL_BASIS_PROPOSAL_V1','status':status,'error':error,'bundle':bundle,
        'statement_sha256':expected_statement,'hint_sha256':expected_hint,'assembled_system_sha256':assembled,
        'slacks':slacks,'stats':stats,'operations':b.operations,'seconds':time.monotonic()-start,
        'deadline_monotonic':deadline,'solver_calls':0,'attempts':1,'limits':dict(LIMITS),
        'feasibility_certified':False,'network_SAFE':False,'network_UNSAFE':False,
        'scope':'original-coordinate analytic candidate only; full original LP check still required'}
