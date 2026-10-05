"""Source-associated half HZ; no native admission or benchmark promotion.

Only build() accepts a genuinely issued C32 state. It runs the unchanged
complete C40 transaction, then issues a NEW, nonportable, object-bound receipt.
The retained state has neither the old post HZ nor its receipt. Revalidation
derives C46's anchors and degrees from the original independent proof chain;
callers cannot provide a replacement hash, degree vector or success flag.

This intentionally pays repeated whole-source checks. It is a correctness
bridge, NOT a paid fresh C34 runtime, a full LIVE boundary, or native recovery.
"""
from dataclasses import dataclass,fields as dc_fields
import hashlib
import json
import weakref
import numpy as np
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from act.back_end.hybridz_tf.tf_cnn import SparseHZAffineExpr
from experiments.neural_hz_20260831.c32_splice_binding_v1 import (
    SplicedState,load_transfer,semantic_fingerprint,source_verify)
from experiments.neural_hz_20260831.c32_fresh_lineage_v1 import ReversibleLineage,decode
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import Overlay
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import RADIX,UID_LIMIT
from experiments.neural_hz_20260831.c40_half_gauge_transaction_v1 import functional,unpack_descriptors
from experiments.neural_hz_20260831.c40_half_gauge_inverse_v1 import inverse_hashes
from experiments.neural_hz_20260831.c46_half_lineage_extension_v1 import extend,_seal_hz
from experiments.neural_hz_20260831.c47_source_payment_floor_v1 import metadata_work


def _array(value,dtype,*,pool):
    if (type(value) is not np.ndarray or value.dtype!=np.dtype(dtype)
            or value.ndim!=1 or not value.flags.c_contiguous or not value.flags.aligned):
        raise ValueError('exact native contiguous numeric vector required')
    if value.size>64_000_000:raise MemoryError('unchanged numeric entry cap')
    pool.charge('c47_numeric_owner_and_domain',8*value.size+64)
    owner=value;seen=set()
    while owner.base is not None:
        pool.charge('c47_numeric_owner_chain',64)
        if id(owner) in seen or type(owner.base) is not np.ndarray:
            raise ValueError('unknown external or cyclic numeric owner')
        seen.add(id(owner));owner=owner.base
    if not owner.flags.owndata or owner.dtype!=value.dtype or not owner.flags.c_contiguous:
        raise ValueError('unregistered numeric owner layout')
    if not np.isfinite(value).all():raise ValueError('nonfinite bound source numeric data')
    return int(value.size)


def _matrix(m,shape,*,pool):
    if type(m) is not sp.csr_matrix or m.shape!=shape:
        raise ValueError('complete actual CSR geometry differs from global frame')
    total=sum(_array(getattr(m,n),dtype,pool=pool) for n,dtype in
        (('data',np.float64),('indices',np.int32),('indptr',np.int32)))
    pool.charge('c47_actual_CSR_domain_and_order',8*m.data.size+8*(shape[0]+1)+128)
    if (m.indices.size!=m.data.size or m.indptr.shape!=(shape[0]+1,) or m.indptr[0]!=0
            or m.indptr[-1]!=m.data.size or np.any(m.indptr[1:]<m.indptr[:-1])
            or np.any(m.indices<0) or np.any(m.indices>=shape[1]) or np.any(m.data==0.)):
        raise ValueError('incomplete, zero or out-of-frame CSR payload')
    # Do not trust cached has_canonical_format/has_sorted_indices flags.
    for row in range(shape[0]):
        a,b=map(int,m.indptr[row:row+2])
        if b-a>1 and np.any(m.indices[a+1:b]<=m.indices[a:b-1]):
            raise ValueError('actual CSR rows are not sorted and unique')
    return total


def checked_hz(hz,*,pool):
    pool.charge('c47_actual_HZ_geometry_header',512)
    if (type(hz) is not SparseHZono or hz.exact is not True
            or type(hz.frame_id) is not int
            or set(vars(hz))!={f.name for f in dc_fields(SparseHZono)}):
        raise ValueError('complete exact HZ schema and literal shared frame required')
    sizes=[_array(getattr(hz,n),np.float64,pool=pool) for n in ('c','b','ub')]
    if type(hz.Gc) is not sp.csr_matrix or type(hz.Gb) is not sp.csr_matrix:
        raise ValueError('actual HZ output maps required')
    no,ne,ni=sizes;nc,nb=hz.n_cont,hz.n_bin
    if min(no,ne,ni,nc,nb)<0 or max(no,nc,nb)>64_000_000 or ne+ni>UID_LIMIT:
        raise ValueError('actual global HZ frame outside frozen domain')
    total=sum(sizes)
    for name,shape in [('Gc',(no,nc)),('Gb',(no,nb)),('Ac',(ne,nc)),('Ab',(ne,nb)),
                       ('Auc',(ni,nc)),('Aub',(ni,nb))]:
        total+=_matrix(getattr(hz,name),shape,pool=pool)
    if total>64_000_000:raise MemoryError('unchanged complete numeric entry cap')
    return (hz.frame_id,no,nc,nb,ne,ni),total


def _json(value,pool):
    # Source reports are already registered by C32. This is explicit work for
    # their new serialization/hash; it is not an allocator or LIVE metric.
    raw=json.dumps(value,sort_keys=True,allow_nan=False).encode()
    pool.charge('c47_report_serialization_and_hash',4*len(raw)+256)
    return hashlib.sha256(raw).hexdigest()


def _source_checks(state,*,pool):
    f=state.original_fields;lin=state.lineage
    if type(f) is not dict or type(lin) is not ReversibleLineage:
        raise ValueError('complete original source and exact old lineage required')
    expr=f['expression']
    if type(expr) is not SparseHZAffineExpr:raise ValueError('original source expression required')
    entries=0
    for hz in (f['hz'],*(t.source for t in expr.terms)):
        _,n=checked_hz(hz,pool=pool);entries+=n
    entries+=_array(expr.bias,np.float64,pool=pool)
    for term in expr.terms:
        for op in term.operators:
            if type(op) is sp.csr_matrix:entries+=_matrix(op,op.shape,pool=pool)
            elif type(op) is ImplicitConv2DOp:
                # Full original source proof also binds shape/kernel geometry.
                if type(op._kernel) is not np.ndarray or op._kernel.dtype!=np.float64:
                    raise ValueError('registered exact source kernel required')
                entries+=_array(op._kernel.reshape(-1),np.float64,pool=pool)
                if op._row_mask is not None:entries+=_array(op._row_mask.reshape(-1),bool,pool=pool)
            else:raise ValueError('unregistered source operator in C47 bridge')
    for name in ('eq_roots','eq_scales','ineq_roots','ineq_scales','def_rows','owners'):
        entries+=_array(f[name],np.int64,pool=pool)
    entries+=_array(f['uid_slabs'],np.uint64,pool=pool)
    entries+=_array(f['keep'],bool,pool=pool)
    for name,dtype in (('columns',np.int32),('retired',np.uint64),('tails',np.uint64)):
        entries+=_array(getattr(lin,name),dtype,pool=pool)
    entries+=_array(state.events,np.uint64,pool=pool)
    pool.charge('c47_complete_inherited_source_hash_and_overlay_allowance',64*entries+4096)
    for raw in (state.source_proof_bytes,state.transfer_proof_bytes):
        if type(raw) is not bytes:raise ValueError('original independent proof bytes required')
        pool.charge('c47_full_proof_parse_and_authentication',16*len(raw)+256)
    _json(f['report'],pool)
    proof=load_transfer(state.transfer_proof_bytes,state.transfer_proof_sha256)
    source=source_verify(f,state.source_proof_bytes,expected_proof_sha256=state.source_proof_sha256)
    semantic=semantic_fingerprint(f,lin,pool=pool)
    overlay=Overlay(f['owners'],state.events,state.old_uid_ceiling);overlay.validate()
    event_sha=hashlib.sha256(memoryview(state.events).cast('B')).hexdigest()
    pre_sha=_seal_hz(f['hz'],pool)
    if (state.source_proof_sha256!=proof['new_source_proof_sha256']
            or source['complete_original_source_image_sha256']!=proof['new_closed_identity']
            or pre_sha!=proof['pre_HZ_sha256'] or semantic!=proof['semantic_lineage_sha256']
            or event_sha!=proof['actual_phase_events_sha256']
            or state.old_uid_ceiling!=f['report']['radix_uid_base']+16384
            or lin.eq_roots is not f['eq_roots'] or lin.eq_scales is not f['eq_scales']
            or lin.old_n_cont!=f['old_n_cont'] or lin.old_n_eq!=f['old_n_eq']):
        raise ValueError('original source, lineage, events or complete independent proof association changed')
    return proof,source,semantic,overlay,event_sha


@dataclass
class BoundHalfState:
    original_fields:dict
    hz:object
    lineage:object
    events:object
    old_uid_ceiling:int
    source_proof_bytes:bytes
    source_proof_sha256:str
    transfer_proof_bytes:bytes
    transfer_proof_sha256:str
    construction_report:dict
    original_post_geometry:tuple
    half_words:object
    gauge_runs:object
    transaction_report:dict
    receipt:object=None

    def _check(self,*,pool):
        pool.charge('c47_complete_bound_state_header',1024)
        if set(vars(self))!={f.name for f in dc_fields(BoundHalfState)}:
            raise ValueError('unregistered half-state retained payload')
        proof,source,semantic,overlay,event_sha=_source_checks(self,pool=pool)
        geometry,_=checked_hz(self.hz,pool=pool)
        _array(self.half_words,np.uint64,pool=pool);_array(self.gauge_runs,np.uint64,pool=pool)
        desc=unpack_descriptors(self.half_words,pool=pool)
        before=self.original_post_geometry
        if (type(before) is not tuple or len(before)!=6 or any(type(v) is not int for v in before)
                or geometry[:4]!=before[:4] or geometry[4]!=before[4]-len(desc)
                or geometry[5]!=before[5] or geometry[0]!=self.original_fields['hz'].frame_id):
            raise ValueError('new HZ does not retain the independently bound old global geometry')
        f,lin=self.original_fields,self.lineage
        pool.charge('c47_derived_degree_and_complete_removed_sets',metadata_work(len(lin.eq_roots),len(desc)))
        removed=set(map(int,lin.columns))
        for at in np.flatnonzero(lin.eq_roots<0):removed.add(lin.old_n_cont+int(at)-lin.old_n_eq)
        degrees=np.empty(len(desc),np.int32)
        for i,t in enumerate(desc):
            col,parent=int(t['column']),int(t['parent'])
            if not f['old_n_cont']<=col<f['logical_n_cont'] or col in removed or parent in removed:
                raise ValueError('half child/parent incompatible with bound old witness lineage')
            index=col-f['old_n_cont']
            degree=lin.owner_query(index,overlay,pool=pool)//RADIX
            if type(degree) is not int or not 3<=degree<=before[4]+before[5]:
                raise ValueError('complete bound integer incidence degree outside structural domain')
            degrees[i]=degree
        inverse=inverse_hashes(self.hz,self.hz,desc,self.gauge_runs,degrees,pool=pool,enabled=True)
        if inverse['complete_original_post_sha256']!=proof['actual_spliced_HZ_sha256']:
            raise ValueError('complete new-state inverse differs from original independent post-HZ proof')
        # No selected factor can remain in output or any EQ/INEQ row. This is
        # checked independently of the count proof and cached CSR flags.
        children=set(map(int,desc['column']))
        for name in ('Gc','Ac','Auc'):
            cols=getattr(self.hz,name).indices
            pool.charge('c47_complete_erased_column_absence',8*len(cols))
            if any(int(c) in children for c in cols):raise ValueError('erased half column remains in new HZ')
        post_sha=_seal_hz(self.hz,pool)
        stamp=(source['complete_original_source_image_sha256'],semantic,lin.seal,event_sha,
            self.source_proof_sha256,self.transfer_proof_sha256,post_sha,before,
            hashlib.sha256(self.half_words.tobytes()+self.gauge_runs.tobytes()).hexdigest(),
            _json(self.construction_report,pool),_json(self.transaction_report,pool))
        report=dict(source_association_proved_relative_to_issued_C32_chain=True,
            complete_original_post_sha256=proof['actual_spliced_HZ_sha256'],new_post_sha256=post_sha,
            paired_lineage_sha256=lin.seal,complete_source_sha256=source['complete_original_source_image_sha256'],
            inherited_prefix_continuous_factors=f['old_n_cont'],half_count=len(desc),
            derived_original_incidence_degrees=degrees.tolist(),no_caller_supplied_degree_or_anchor=True,
            complete_actual_CSR_geometry_order_and_owner_checked=True,
            old_post_HZ_or_old_receipt_retained_by_bound_state=False,
            source_proof_is_test_only=proof.get('test_only_dense_oracle') is True,
            original_network_input_or_property_binding_proved=False,native_admission_proved=False,
            complete_live_or_runtime_payment_proved=False,solver_executed=False,formal_gain=0)
        return stamp,report,degrees

    def validate(self,*,pool):
        if type(self.receipt) is not _Receipt or self.receipt not in _ISSUED:
            raise ValueError('half state lacks its own issued source binding')
        owner,expected=_ISSUED[self.receipt]
        if owner() is not self:raise ValueError('receipt belongs to a different half-state object')
        stamp,report,_=self._check(pool=pool)
        if stamp!=expected:raise ValueError('bound half state changed after issue')
        return report

    def reconstruct_fraction(self,continuous,*,pool):
        before=self.validate(pool=pool)
        # Degrees and original image/lineage anchors are derived above from
        # the issued source proof, never accepted from this method's caller.
        pool.charge('c47_derived_witness_arguments',32*before['half_count']+256)
        result,report=extend(self.hz,self.lineage,self.half_words,self.gauge_runs,
            np.asarray(before['derived_original_incidence_degrees'],np.int32),
            before['complete_original_post_sha256'],before['paired_lineage_sha256'],continuous,
            input_n_cont=self.original_fields['old_n_cont'],pool=pool,enabled=True)
        after=self.validate(pool=pool)
        if before!=after:raise ValueError('source changed during bound reconstruction')
        report.update(source_association_proved_relative_to_issued_C32_chain=True,
            inherited_prefix_unchanged_not_original_network_input_binding=True)
        return result,report

    def numeric_roots(self,*,pool):
        self.validate(pool=pool)
        result={k:v for k,v in vars(self).items() if k not in ('receipt','lineage')}
        result['lineage']=dict(vars(self.lineage))
        result['issued_source_record']=_ISSUED[self.receipt][1]
        return result


_KEY=object()
_ISSUED=weakref.WeakKeyDictionary()


class _Receipt:
    __slots__=('__weakref__',)
    def __new__(cls,key):
        if key is not _KEY:raise ValueError('only complete source construction issues half receipts')
        return super().__new__(cls)
    def __reduce__(self):raise TypeError('half source bindings are not portable certificates')


def build(state,*,pool,branch=None,enabled=False):
    if not enabled:return None
    if type(state) is not SplicedState:raise ValueError('genuinely issued original C32 state required')
    # Admission of an unissued object is forbidden, even if fields hash alike.
    _source_checks(state,pool=pool)
    geometry,_=checked_hz(state.hz,pool=pool)
    initial=state.validate()
    post,end,words,runs,transaction=functional(state,state.hz,pool=pool,branch=branch,enabled=True)
    # C40's final argument is deliberately the same post HZ: this bridge does
    # not invent a new Dense/ASSERT source binding or claim a native suffix.
    if _seal_hz(end,pool)!=_seal_hz(post,pool):raise ValueError('unexpected additional final-HZ semantics')
    fields={k:getattr(state,k) for k in ('original_fields','lineage','events','old_uid_ceiling',
        'source_proof_bytes','source_proof_sha256','transfer_proof_bytes','transfer_proof_sha256','construction_report')}
    result=BoundHalfState(**fields,hz=post,original_post_geometry=geometry,
        half_words=words,gauge_runs=runs,transaction_report=transaction)
    stamp,report,_=result._check(pool=pool)
    if state.validate()['complete_new_HZ_sha256']!=initial['complete_new_HZ_sha256']:
        raise ValueError('source changed during complete half construction')
    result.receipt=_Receipt(_KEY);_ISSUED[result.receipt]=(weakref.ref(result),stamp)
    return result,report
