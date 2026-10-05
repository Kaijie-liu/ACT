"""Complete exact consumer diagnostic from externally authenticated C9/C58 inputs.

Not an HZ writer. Original source/numeric qualification is inherited through
the worker's frozen artifact binding; source-to-removed-row binding is fresh.
"""
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
import numpy as np
from experiments.neural_hz_20260831.c54_exact_dyadic_v1 import unpack
from experiments.neural_hz_20260831.c58_reconstruction_equations_v1 import audit as inverse_audit
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import word, native_word, fraction, product, plus, in_window


def interval(value):
    m,e=value
    if not m:return -10**9,10**9
    floor=abs(m).bit_length()-1+e
    return -20-floor,40-floor-(abs(m)!=1)


def gauge(words):
    low,high=-10**9,10**9
    for value in words:
        a,b=interval(value);low=max(low,a);high=min(high,b)
    return low,high,None if low>high else min(max(0,low),high)


class Products:
    def __init__(self,pool):
        self.pool=pool;self.values={};self.lookups=0;self.hits=0
    def __call__(self,a,b):
        self.pool.charge('c59_canonical_product_lookup',32);self.lookups+=1
        key=(a,b)
        if key in self.values:self.hits+=1;return self.values[key]
        result=product(a,b,self.pool)
        self.pool.charge('c59_distinct_Fraction_product_proof',96)
        if fraction(result)!=fraction(a)*fraction(b):raise ValueError('exact consumer product differs')
        self.values[key]=result;return result


def decode_frame(packet,pool):
    count=packet['n_cont'];pool.charge('c59_complete_inverse_unpack',16*count+1024)
    table=unpack(packet['scalars'])
    ratios=[word(table[int(n)][0],table[int(n)][1]-table[int(d)][1]) for n,d in packet['pairs']]
    roots=(packet['frame'] & np.uint64(2**32-1)).astype(np.int64)
    ids=(packet['frame'] >> np.uint64(32)).astype(np.int64)
    weights=[ratios[int(i)] for i in ids]
    proof=inverse_audit(packet,roots,weights,n_bin=packet['n_bin'],
                        source_binding=packet['source_binding'],pool=pool)
    return roots,weights,proof


class Census:
    def __init__(self,roots,weights,removed,pool):
        self.roots=roots;self.weights=weights;self.removed=removed;self.pool=pool
        self.chosen=roots!=np.arange(len(roots));self.products=Products(pool)
        self.counts=Counter();self.shifts=Counter();self.spans=Counter();self.bits=Counter()
        self.digest=hashlib.sha256()

    def translate(self,cols,values):
        selected=self.chosen[cols];positions=np.flatnonzero(selected)
        self.pool.charge('c59_affected_consumer_row',32+3*len(cols)+16*len(positions))
        untouched=~selected;updates={}
        for i in positions:
            col=int(cols[i]);root=int(self.roots[col])
            value=self.products(native_word(values[i]),self.weights[col])
            updates[root]=plus(updates.get(root,(0,0)),value,self.pool)
        self.pool.charge('c59_changed_root_ordering',len(updates)*max(1,len(updates).bit_length()))
        collisions=0
        for root in sorted(updates):
            self.pool.charge('c59_original_root_binary_search',max(1,len(cols).bit_length())+8)
            i=int(np.searchsorted(cols,root))
            if i<len(cols) and int(cols[i])==root:
                if self.chosen[root]:raise ValueError('inverse root was not retained')
                updates[root]=plus(updates[root],native_word(values[i]),self.pool)
                untouched[i]=False;collisions+=1
        nonzero={c:v for c,v in updates.items() if v[0]}
        self.counts['changed_terms']+=len(positions)
        self.counts['coalesced_terms']+=len(positions)-len(updates)+collisions
        self.counts['cancelled_roots']+=len(updates)-len(nonzero)
        return nonzero,untouched,-len(positions)-collisions+len(nonzero)

    def inspect_row(self,updates,untouched_values,binary_values,rhs,*,kind,index):
        self.pool.charge('c59_complete_row_range_and_digest',
                         64+40*len(updates)+8*(len(untouched_values)+len(binary_values)))
        low,high,unused=gauge(updates.values())
        for a in (untouched_values,binary_values):
            nonzero=np.abs(a[a!=0])
            if len(nonzero):
                x,y,_=gauge([native_word(float(nonzero.min())),native_word(float(nonzero.max()))])
                low=max(low,x);high=min(high,y)
        if rhs:
            a,b=interval(native_word(rhs));low=max(low,a);high=min(high,b)
        shift=None if low>high else min(max(0,low),high)
        self.counts['affected_surviving_rows']+=1
        self.counts['row_gauge_incompatible']+=shift is None
        self.counts['rows_requiring_nonzero_gauge']+=shift not in (None,0)
        self.spans[str(max(0,60+low-high))]+=1
        if shift is not None:self.shifts[str(shift)]+=1
        for m,e in updates.values():
            self.bits[str(abs(m).bit_length())]+=1
            self.counts['derived_nonzero_coefficients']+=1
            self.counts['derived_outside_original_window']+=not in_window((m,e))
            self.counts['derived_mantissa_over53']+=abs(m).bit_length()>53
        self.digest.update(json.dumps([kind,index,low,high,shift,sorted(updates.items())]).encode())
        return low,high,shift

    def finish(self,hz,observe=None):
        matrices={}
        for name,bname,rhs,predicate in [('Ac','Ab',hz.b,True),('Auc','Aub',hz.bu,True),('Gc','Gb',hz.c,False)]:
            matrix=getattr(hz,name);binary=getattr(hz,bname);nr=matrix.shape[0]
            if rhs.shape!=(nr,) or binary.shape!=(nr,hz.n_bin):raise ValueError('complete predicate RHS/binary layout')
            self.pool.charge('c59_full_consumer_incidence_scan',3*matrix.nnz+8*nr+1024)
            selected=self.chosen[matrix.indices]
            prefix=np.empty(matrix.nnz+1,np.int64);prefix[0]=0;np.cumsum(selected,out=prefix[1:])
            hits=prefix[matrix.indptr[1:]]-prefix[matrix.indptr[:-1]];affected=hits>0
            removed_nnz=0;removed_rows=0
            if name=='Ac':
                removed_nnz=int(np.diff(matrix.indptr)[self.removed].sum())
                removed_rows=int(self.removed.sum());affected[self.removed]=False
            if not predicate and affected.any():raise ValueError('output-dead inverse does not preserve actual output map')
            rows=np.flatnonzero(affected);after=matrix.nnz-removed_nnz
            del selected,prefix,hits,affected
            previous=dict(self.counts)
            for index in rows:
                a,b=map(int,matrix.indptr[index:index+2]);ba,bb=map(int,binary.indptr[index:index+2])
                updates,untouched,delta=self.translate(matrix.indices[a:b],matrix.data[a:b]);after+=delta
                self.inspect_row(updates,matrix.data[a:b][untouched],binary.data[ba:bb],float(rhs[index]),kind=name,index=int(index))
            local={k:v-previous.get(k,0) for k,v in self.counts.items()}
            matrices[name]=dict(original_rows=nr,original_nnz=int(matrix.nnz),removed_rows=removed_rows,
                removed_defining_nnz=removed_nnz,surviving_rows=nr-removed_rows,
                affected_surviving_rows=len(rows),symbolic_after_nnz=int(after),counts=local,
                untouched_binary_nnz=int(binary.nnz),RHS_retained=True)
            if observe:observe(dict(event='complete_consumer_matrix_profile',matrix=name,**matrices[name]))
        old=hz.Ac.nnz+hz.Ab.nnz+hz.Auc.nnz+hz.Aub.nnz
        after=matrices['Ac']['symbolic_after_nnz']+hz.Ab.nnz+matrices['Auc']['symbolic_after_nnz']+hz.Aub.nnz
        all_rows=sum(m['surviving_rows'] for m in matrices.values())
        lower=64*(after+hz.Gc.nnz+hz.Gb.nnz+all_rows)
        return dict(schema='c59_complete_surviving_consumer_profile_v1',matrices=matrices,counts=dict(self.counts),
            required_shift_histogram=dict(self.shifts),row_exponent_span_histogram=dict(self.spans),
            derived_mantissa_histogram=dict(self.bits),complete_consumer_sha256=self.digest.hexdigest(),
            exact_product_unique_pairs=len(self.products.values),exact_product_lookups=self.products.lookups,
            exact_product_reuses=self.products.hits,original_predicate_nnz=int(old),symbolic_predicate_nnz=int(after),
            symbolic_predicate_nnz_strictly_reduced=bool(after<old),symbolic_removed_continuous=int(self.chosen.sum()),
            original_continuous=hz.n_cont,symbolic_retained_continuous=int(hz.n_cont-self.chosen.sum()),
            original_binary_retained=hz.n_bin,all_rows_have_bounded_positive_gauge=self.counts['row_gauge_incompatible']==0,
            unchanged_C56_full_scan_lower_bound=int(lower),unchanged_C56_16M_scan_fits=bool(lower<=16_000_000),
            complete_consumer_profile=True,physical_HZ_or_native_realization_proved=False,
            full_C54_coalescing_closure_proved=False,source_first_writer_executed=False,formal_gain=0)


def assess(saved,packet,*,pool,enabled=False,observe=None):
    if not enabled:return None
    hz=saved['hz'];old=int(saved['old_n_cont']);logical=int(saved['logical_n_cont']);old_eq=int(saved['old_n_eq'])
    if packet['n_cont']!=hz.n_cont or packet['n_bin']!=hz.n_bin or not hz.exact:raise ValueError('original HZ frame differs')
    roots,weights,proof=decode_frame(packet,pool)
    chosen=np.flatnonzero(roots!=np.arange(hz.n_cont));removed=np.zeros(hz.n_eq,bool)
    if np.any(chosen<old) or np.any(chosen>=logical):raise ValueError('old/radix coordinate cannot be removed')
    pool.charge('c59_complete_defining_row_binding',160*len(chosen)+8*hz.n_cont+hz.n_eq+1024)
    proofs=set()
    for slot in chosen:
        index=int(saved['eq_roots'][old_eq+int(slot)-old]);a,b=map(int,hz.Ac.indptr[index:index+2])
        cols=hz.Ac.indices[a:b];values=hz.Ac.data[a:b]
        if (len(cols)!=2 or int(cols[-1])!=slot or not 0<=int(cols[0])<slot
                or hz.Ab.indptr[index]!=hz.Ab.indptr[index+1] or hz.b[index]!=0 or removed[index]
                or values[-1]<=0):raise ValueError('unproved original homogeneous defining EQ')
        parent=int(cols[0]);pivot=native_word(values[-1])
        if pivot[0]!=1 or roots[slot]!=roots[parent]:raise ValueError('positive dyadic defining pivot/shared root differs')
        key=(float(values[0]),float(values[-1]),weights[parent],weights[int(slot)])
        if key not in proofs:
            if F(key[0])*fraction(key[2])+F(key[1])*fraction(key[3])!=0:raise ValueError('inverse fails original defining equation')
            proofs.add(key)
        removed[index]=True
    if observe:observe(dict(event='complete_original_defining_equations_rebound',coordinates=len(chosen),
        original_frame_coordinates=hz.n_cont,independent_equation_keys=len(proofs),inverse_packet_proof=proof,
        complete_C58_original_source_qualification_inherited=True))
    census=Census(roots,weights,removed,pool)
    result=census.finish(hz,observe)
    result['fresh_original_defining_rows_proved']=len(chosen)
    result['inverse_packet_equations_checked']=proof
    return result
