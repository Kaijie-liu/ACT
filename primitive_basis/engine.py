"""Sparse primitive integer rows, exact late division, fixed arithmetic caps.

This is NOT Bareiss: row content is explicitly removed after cross elimination.
Every resulting integer (including products before cancellation) is guarded.
Candidate generation is not a feasibility or optimality certificate.
"""
from fractions import Fraction as F
from hashlib import sha256
from math import gcd
import heapq
import math
import time
from lp_sandwich.check import rational, identity
from sparse_basis.engine import POLICY as OLD_POLICY, Limit, Singular

POLICY={**OLD_POLICY,'arithmetic':'primitive_integer_rows_cross_gcd_then_content_V1',
        'raw_integer_products':'checked_before_subtraction_and_content_reduction',
        'trace_prefix':64}


class Budget:
    def __init__(self,deadline):
        now=time.monotonic()
        if not math.isfinite(deadline) or deadline>now+300:raise ValueError('bounded original deadline required')
        self.deadline,self.operations=deadline,0
        self.context={'phase':'admission'}
        self.arithmetic={'max_integer_bits':0,'max_fraction_numerator_bits':0,
                         'max_fraction_denominator_bits':0,'first_limit':None}

    def where(self,phase,operation,**fields):
        self.context={'phase':phase,'operation':operation,**fields}

    def tick(self):
        if time.monotonic()>=self.deadline:raise TimeoutError('original deadline')

    def visit(self):
        self.tick();self.operations+=1
        if self.operations>POLICY['operations']:
            self.fail('operations',self.operations,POLICY['operations'])

    def fail(self,kind,observed,cap,**extra):
        if self.arithmetic['first_limit'] is None:
            self.arithmetic['first_limit']={**self.context,'kind':kind,'observed':observed,'cap':cap,**extra}
        raise Limit(kind+' budget')

    def integer(self,v):
        self.visit()
        if type(v) is not int:raise ValueError('integer arithmetic required')
        bits=abs(v).bit_length()
        self.arithmetic['max_integer_bits']=max(self.arithmetic['max_integer_bits'],bits)
        if bits>POLICY['max_bits']:
            self.fail('integer bits',bits,POLICY['max_bits'],value_sha256=sha256(format(v,'x').encode()).hexdigest())
        return v

    def value(self,v):
        self.visit();v=v if isinstance(v,F) else rational(v)
        n,d=abs(v.numerator).bit_length(),v.denominator.bit_length()
        self.arithmetic['max_fraction_numerator_bits']=max(self.arithmetic['max_fraction_numerator_bits'],n)
        self.arithmetic['max_fraction_denominator_bits']=max(self.arithmetic['max_fraction_denominator_bits'],d)
        if max(n,d)>POLICY['max_bits']:
            self.fail('rational bits',max(n,d),POLICY['max_bits'],numerator_bits=n,denominator_bits=d,
                      value_sha256=sha256((format(v.numerator,'x')+'/'+format(v.denominator,'x')).encode()).hexdigest())
        return v


def statistics():
    v=dict.fromkeys(('peak_active_nnz','peak_live_nnz','peak_heap_entries','fill_insertions',
        'heap_pops','heap_rebuilds','pivots','pivot_row_candidates','singleton_column_pivots',
        'row_updates','content_divisions','denominator_rows','cross_gcd_cancellations','trace_count'),0)
    v.update(pivot_prefix=[],pivot_sha256=sha256().hexdigest())
    return v


def primitive(row,rhs,b,stats):
    """Divide a whole equality by nonzero content, RHS included, deterministic sign."""
    content=0
    for v in (*row.values(),rhs):
        b.visit();content=gcd(content,abs(v))
    if not row:raise Singular('zero row in selected square basis; no LP verdict')
    sign=1 if row[min(row)]>0 else -1
    if content>1:stats['content_divisions']+=1
    divisor=content*sign
    # Nonempty row implies strictly positive content; divisions are exact.
    return {j:v//divisor for j,v in row.items()},rhs//divisor


def clear_row(row,rhs,b,stats,index):
    b.where('row_clear','input',row=index)
    vals={j:b.value(v) for j,v in row.items() if v};rhs=b.value(rhs)
    denominator=1
    for v in (*vals.values(),rhs):
        b.where('row_clear','lcm',row=index)
        denominator=b.integer((denominator//gcd(denominator,v.denominator))*v.denominator)
    if denominator!=1:stats['denominator_rows']+=1
    out={}
    for j,v in vals.items():
        b.where('row_clear','coefficient',row=index,column=j)
        out[j]=b.integer(v.numerator*(denominator//v.denominator))
    b.where('row_clear','rhs',row=index)
    rhs=b.integer(rhs.numerator*(denominator//rhs.denominator))
    return primitive(out,rhs,b,stats)


def eliminate(system,b,stats):
    m=len(system)
    if m>POLICY['equations']:b.fail('equations',m,POLICY['equations'])
    if sum(len(r) for r,_ in system)>POLICY['input_nnz']:
        b.fail('input_nnz',sum(len(r) for r,_ in system),POLICY['input_nnz'])
    rows,right={},{}
    for i,(row,rhs) in enumerate(system):
        if any(type(j) is not int or not 0<=j<m for j in row):raise ValueError('basis column range')
        rows[i],right[i]=clear_row(row,rhs,b,stats,i)
    system.clear()
    incidence={j:set() for j in range(m)}
    for i,row in rows.items():
        for j in row:b.visit();incidence[j].add(i)
    active_nnz=sum(map(len,rows.values()));pivot_nnz=0
    heap=[(len(ids),j) for j,ids in incidence.items()];heapq.heapify(heap)
    pivots=[];trace=sha256()

    def observe():
        b.tick()
        stats['peak_active_nnz']=max(stats['peak_active_nnz'],active_nnz)
        stats['peak_live_nnz']=max(stats['peak_live_nnz'],active_nnz+pivot_nnz)
        stats['peak_heap_entries']=max(stats['peak_heap_entries'],len(heap))
        for name,value,cap in (('live_nnz',active_nnz+pivot_nnz,POLICY['live_nnz']),
                               ('fill_insertions',stats['fill_insertions'],POLICY['fill_insertions']),
                               ('heap_entries',len(heap),POLICY['heap_entries'])):
            if value>cap:b.fail(name,value,cap)
    observe()
    while rows:
        b.where('elimination','pivot_selection');b.visit()
        while heap:
            degree,col=heapq.heappop(heap);stats['heap_pops']+=1;b.visit()
            if col in incidence and degree==len(incidence[col]):break
        else:raise Singular('no active pivot column')
        if not degree:raise Singular('rank deficient selected basis; no LP verdict')
        stats['pivot_row_candidates']+=degree
        pivot_id=min(incidence[col],key=lambda i:(len(rows[i]),i))
        pivot=rows.pop(pivot_id);rhs=right.pop(pivot_id)
        active_nnz-=len(pivot);pivot_nnz+=len(pivot)
        pivots.append((col,pivot,rhs));stats['pivots']+=1
        item={'column':col,'row':pivot_id,'degree':degree}
        trace.update((identity(item)+'\n').encode());stats['pivot_sha256']=trace.hexdigest();stats['trace_count']+=1
        if len(stats['pivot_prefix'])<POLICY['trace_prefix']:stats['pivot_prefix'].append(item)
        if degree==1:stats['singleton_column_pivots']+=1
        affected=sorted(incidence[col]-{pivot_id});changed=set(pivot)
        for j in pivot:incidence[j].remove(pivot_id)
        del incidence[col]
        for i in affected:
            b.visit();stats['row_updates']+=1
            row=rows[i];a,factor=pivot[col],row[col];common=gcd(abs(a),abs(factor))
            if common>1:stats['cross_gcd_cancellations']+=1
            left,right_scale=a//common,factor//common
            updated={}
            for j in sorted(set(row)|set(pivot)):
                if j==col:continue  # identically zero by the selected multipliers
                b.where('elimination','row_product',row=i,column=j,pivot_column=col)
                lv=b.integer(left*row.get(j,0));rv=b.integer(right_scale*pivot.get(j,0))
                b.where('elimination','row_subtract',row=i,column=j,pivot_column=col)
                value=b.integer(lv-rv)
                if value:updated[j]=value
            b.where('elimination','rhs_product',row=i,pivot_column=col)
            lv=b.integer(left*right[i]);rv=b.integer(right_scale*rhs)
            b.where('elimination','rhs_subtract',row=i,pivot_column=col)
            new_rhs=b.integer(lv-rv)
            updated,new_rhs=primitive(updated,new_rhs,b,stats)
            for j in row:
                if j!=col and j not in updated:incidence[j].remove(i)
            for j in updated:
                if j not in row:incidence[j].add(i);stats['fill_insertions']+=1
            active_nnz+=len(updated)-len(row)
            changed.update(row);changed.update(updated)
            rows[i],right[i]=updated,new_rhs;observe()
        for j in changed:
            if j in incidence:heapq.heappush(heap,(len(incidence[j]),j))
        if len(heap)>max(1024,4*len(incidence)):
            heap=[(len(ids),j) for j,ids in incidence.items()];heapq.heapify(heap);stats['heap_rebuilds']+=1
        observe()
    values=[F(0)]*m
    for col,row,rhs in reversed(pivots):
        v=b.value(rhs)
        for j,coeff in row.items():
            if j==col:continue
            b.where('back_substitution','multiply',column=j,pivot_column=col)
            product=b.value(coeff*values[j])
            b.where('back_substitution','subtract',column=j,pivot_column=col)
            v=b.value(v-product)
        b.where('back_substitution','divide',pivot_column=col)
        values[col]=b.value(v/row[col])
    return values
