"""Bound, checked symbolic schedules; numerical factors are never cached.

A pattern mismatch invalidates the plan and pays for one dynamic factorization
of that prime in the same budget. Identity errors fail closed, not fallback.
Every computed field vector is checked against the original modular equations.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import heapq
from lp_sandwich.check import identity
from .engine import POLICY, product, field_value, Singular, BadPrime


class PlanMismatch(Exception):
    pass


@dataclass(frozen=True)
class Plan:
    binding: str
    initial: tuple
    steps: tuple
    checksum: str

    @property
    def units(self):
        return sum(len(x)+1 for x in self.initial)+sum(2+len(s[2])+len(s[3]) for s in self.steps)


def payload(binding,initial,steps):
    return {'version':'CROSS_PRIME_SYMBOLIC_PLAN_V1','binding':binding,
            'initial':initial,'steps':steps}


def bind(system,scope,b):
    b.where('plan','binding')
    if not isinstance(scope,str) or not scope:raise ValueError('explicit nonempty scope')
    rows=[];m=len(system);count=0
    if m>POLICY['equations']:b.fail('equations',m,POLICY['equations'])
    for row,rhs in system:
        entries=[]
        for j,v in sorted(row.items()):
            b.visit();count+=1
            if count>POLICY['input_nnz']:b.fail('input_nnz',count,POLICY['input_nnz'])
            if type(j) is not int or not 0<=j<m:raise ValueError('column binding')
            entries.append((j,str(b.value(v))))
        rows.append((entries,str(b.value(rhs))))
    result=identity({'scope':scope,'rows':rows})
    b.tick()
    return result


def shape(rows,b):
    result=[]
    for i in range(len(rows)):
        for _ in rows[i]:b.visit()
        result.append(tuple(sorted(rows[i])))
    return tuple(result)


def add_units(b,stats,units):
    for _ in range(units):b.visit()
    b.plan_units+=units
    stats['peak_plan_units']=max(stats.get('peak_plan_units',0),b.plan_units)
    if b.plan_units>POLICY['plan_entries']:
        b.fail('plan_entries',b.plan_units,POLICY['plan_entries'])


def validate(plan,binding,m,b,stats):
    b.where('plan','validate')
    if not isinstance(plan,Plan) or plan.binding!=binding:raise ValueError('plan binding')
    if type(plan.initial) is not tuple or type(plan.steps) is not tuple:
        raise ValueError('immutable plan containers required')
    if len(plan.initial)!=m or len(plan.steps)!=m:raise ValueError('plan dimension')
    for keys in plan.initial:
        if (type(keys) is not tuple or any(type(j) is not int or not 0<=j<m for j in keys)
                or tuple(sorted(set(keys)))!=keys):raise ValueError('initial plan index schema')
    for step in plan.steps:
        if (type(step) is not tuple or len(step)!=4 or type(step[2]) is not tuple
                or type(step[3]) is not tuple):raise ValueError('immutable plan step schema')
    if plan.units>POLICY['plan_entries']:b.fail('plan_entries',plan.units,POLICY['plan_entries'])
    b.plan_units=0;add_units(b,stats,plan.units)
    if identity(payload(plan.binding,plan.initial,plan.steps))!=plan.checksum:
        raise ValueError('plan integrity')
    rowids=[];cols=[]
    for rid,col,keys,affected in plan.steps:
        if (type(rid) is not int or type(col) is not int or not 0<=rid<m or not 0<=col<m
            or col not in keys or rid in affected
            or tuple(sorted(set(keys)))!=keys or tuple(sorted(set(affected)))!=affected
            or any(type(j) is not int or not 0<=j<m for j in keys+affected)):
            raise ValueError('plan index schema')
        rowids.append(rid);cols.append(col)
    if len(set(rowids))!=m or len(set(cols))!=m:raise ValueError('plan permutation')
    b.tick()


def dynamic(system, p, b, stats, *, capture=False):
    """Independent sparse factorization; no exact cross-products or row LCM."""
    m = len(system)
    source_nnz = sum(len(row) for row, _ in system)
    rows, right = {}, {}
    b.where('finite_field','map',prime=p)
    for i, (row, rhs) in enumerate(system):
        rows[i] = {}
        for j, v in row.items():
            value = field_value(v, p, b, stats)
            if value:
                rows[i][j] = value
        right[i] = field_value(rhs, p, b, stats)
    initial = shape(rows, b) if capture else ()
    steps=[]
    if capture: add_units(b,stats,sum(len(x) for x in initial)+len(initial))
    incidence = {j:set() for j in range(m)}
    for i,row in rows.items():
        for j in row:
            b.visit(); incidence[j].add(i)
    active = sum(map(len, rows.values()))
    retained = 0
    heap = [(len(ids),j) for j,ids in incidence.items()]
    heapq.heapify(heap)
    pivots = []

    def observe():
        b.tick()
        stats['peak_active_nnz'] = max(stats['peak_active_nnz'], active)
        stats['peak_live_nnz'] = max(stats['peak_live_nnz'], source_nnz+active+retained+b.plan_units+b.owned_source_units)
        stats['peak_heap_entries'] = max(stats['peak_heap_entries'], len(heap))
        for name,value in (('live_nnz',source_nnz+active+retained+b.plan_units+b.owned_source_units),
                           ('fill_insertions',stats['fill_insertions']),
                           ('heap_entries',len(heap))):
            if value > POLICY[name]:
                b.fail(name,value,POLICY[name])
    observe()
    b.where('finite_field','elimination',prime=p)
    while rows:
        b.visit()
        while heap:
            degree,col = heapq.heappop(heap)
            stats['heap_pops'] += 1; b.visit()
            if col in incidence and degree == len(incidence[col]):
                break
        else:
            raise Singular('no active pivot')
        if not degree:
            raise Singular('singular over this finite field only')
        stats['pivot_row_candidates'] += degree
        pivot_id = min(incidence[col], key=lambda i:(len(rows[i]),i))
        pivot = rows.pop(pivot_id)
        rhs = right.pop(pivot_id)
        active -= len(pivot)
        inv = pow(pivot[col],-1,p)
        normalized = {j:product(v,inv,p,b,stats) for j,v in pivot.items()}
        target = product(rhs,inv,p,b,stats)
        retained += len(normalized)
        pivots.append((col,normalized,target))
        stats['pivots'] += 1
        if degree == 1:
            stats['singleton_column_pivots'] += 1
        affected = sorted(incidence[col]-{pivot_id})
        if capture:
            steps.append((pivot_id,col,tuple(sorted(pivot)),tuple(affected)))
            add_units(b,stats,2+len(pivot)+len(affected))
        changed = set(pivot)
        for j in pivot:
            incidence[j].remove(pivot_id)
        del incidence[col]
        for i in affected:
            b.visit(); stats['row_updates'] += 1
            row = rows[i]; factor = row.pop(col); active -= 1
            right[i] = (right[i]-product(factor,target,p,b,stats)) % p
            for j,v in normalized.items():
                if j == col:
                    continue
                existed = j in row
                value = (row.get(j,0)-product(factor,v,p,b,stats)) % p
                if value:
                    row[j] = value
                    if not existed:
                        active += 1
                        stats['fill_insertions'] += 1
                        incidence[j].add(i)
                elif existed:
                    del row[j]; active -= 1; incidence[j].remove(i)
                changed.add(j)
                observe()
            if not row:
                raise Singular('dependent selected basis modulo prime; no LP verdict')
        for j in changed:
            if j in incidence:
                heapq.heappush(heap,(len(incidence[j]),j))
        if len(heap) > max(1024,4*len(incidence)):
            heap = [(len(ids),j) for j,ids in incidence.items()]
            heapq.heapify(heap); stats['heap_rebuilds'] += 1
        observe()
    x = [0]*m
    b.where('finite_field','back_substitution',prime=p)
    for col,row,rhs in reversed(pivots):
        for j,v in row.items():
            if j != col:
                rhs = (rhs-product(v,x[j],p,b,stats)) % p
        x[col] = rhs
    return x, initial, tuple(steps)


def replay(system,p,b,stats,plan):
    m=len(system);source_nnz=sum(len(r) for r,_ in system)
    b.where('finite_field','map',prime=p,mode='replay')
    rows={};right={}
    for i,(row,rhs) in enumerate(system):
        rows[i]={}
        for j,v in row.items():
            value=field_value(v,p,b,stats)
            if value:rows[i][j]=value
        right[i]=field_value(rhs,p,b,stats)
    if shape(rows,b)!=plan.initial:raise PlanMismatch('initial_modular_cancellation')
    active=sum(map(len,rows.values()));retained=0;pivots=[]
    def observe():
        b.tick()
        live=source_nnz+active+retained+b.plan_units+b.owned_source_units
        stats['peak_live_nnz']=max(stats['peak_live_nnz'],live)
        stats['peak_active_nnz']=max(stats['peak_active_nnz'],active)
        if live>POLICY['live_nnz']:b.fail('live_nnz',live,POLICY['live_nnz'])
        if stats['fill_insertions']>POLICY['fill_insertions']:
            b.fail('fill_insertions',stats['fill_insertions'],POLICY['fill_insertions'])
    observe()
    b.where('finite_field','elimination',prime=p,mode='replay')
    for rid,col,keys,affected in plan.steps:
        b.visit()
        if rid not in rows or not rows[rid].get(col):raise PlanMismatch('zero_planned_pivot')
        row=rows[rid]
        if tuple(sorted(row))!=keys:raise PlanMismatch('pivot_pattern_changed')
        for i in affected:
            b.visit()
            if i not in rows or not rows[i].get(col):raise PlanMismatch('affected_pattern_changed')
        pivot=rows.pop(rid);rhs=right.pop(rid);active-=len(pivot)
        inv=pow(pivot[col],-1,p)
        normalized={j:product(v,inv,p,b,stats) for j,v in pivot.items()}
        target=product(rhs,inv,p,b,stats)
        retained+=len(normalized);pivots.append((col,normalized,target));stats['pivots']+=1
        if not affected:stats['singleton_column_pivots']+=1
        for i in affected:
            b.visit();stats['row_updates']+=1
            row=rows[i];factor=row.pop(col);active-=1
            right[i]=(right[i]-product(factor,target,p,b,stats))%p
            for j,v in normalized.items():
                if j==col:continue
                existed=j in row
                value=(row.get(j,0)-product(factor,v,p,b,stats))%p
                if value:
                    row[j]=value
                    if not existed:active+=1;stats['fill_insertions']+=1
                elif existed:
                    del row[j];active-=1
                observe()
            if not row:raise PlanMismatch('dependent_row_after_update')
        observe()
    if rows:raise PlanMismatch('uncovered_rows')
    x=[0]*m;b.where('finite_field','back_substitution',prime=p,mode='replay')
    for col,row,rhs in reversed(pivots):
        for j,v in row.items():
            if j!=col:rhs=(rhs-product(v,x[j],p,b,stats))%p
        x[col]=rhs
    return x


def field_residual(system,x,p,b,stats):
    b.where('finite_field','residual_check',prime=p)
    if len(x)!=len(system) or any(type(v) is not int or not 0<=v<p for v in x):
        raise ValueError('field result dimensions')
    for row,rhs in system:
        total=0
        for j,v in row.items():
            total=(total+product(field_value(v,p,b,stats),x[j],p,b,stats))%p
        if total!=field_value(rhs,p,b,stats):return False
    return True
