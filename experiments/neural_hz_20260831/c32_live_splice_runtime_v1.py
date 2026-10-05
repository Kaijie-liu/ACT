"""Default-off single real native phase -> first-write -> owned lineage.

No complete old post matrix is built. Full source/splice proof bytes are bound
independently; failed selected work raises the existing no-fallback exception.
The pre-source image is reversible metadata, never a reused old Closed token.
"""

from contextlib import contextmanager
import gc
import hashlib
import sys
import weakref
import numpy as np

from act.back_end.hybridz_tf import tf_mlp as mlp
from experiments.neural_hz_20260831 import c9_live_runtime_v1 as base
from experiments.neural_hz_20260831.c9_integrated_suffix_v1 import lift as original_lift
from experiments.neural_hz_20260831.c25_live_binding_v1 import bind
from experiments.neural_hz_20260831.c24_closed_state_v1 import export as export_closed
from experiments.neural_hz_20260831.c24_checked_overlay_v1 import build as build_overlay
from experiments.neural_hz_20260831.c32_fresh_lineage_v1 import generate,compile_owned
from experiments.neural_hz_20260831.c32_native_blocks_v1 import phase_blocks
from experiments.neural_hz_20260831.c30_append_discovery_v1 import discover_append
from experiments.neural_hz_20260831.c30_first_write_v1 import splice_append
from experiments.neural_hz_20260831.c32_boundary_budget_v1 import WriterPool,remaining_pool,finish_local
from experiments.neural_hz_20260831.c32_splice_binding_v1 import SplicedState,admit,load_transfer
from experiments.neural_hz_20260831.c10_fused_rows_v1 import WorkPool as CoupledPool

SelectedRejected=base.SelectedRejected
EXTRA={'closed_binding','ownership_permit','budget','phase_binding','native_block_calls'}


def numeric_roots(state):
    if not EXTRA<=set(state):raise ValueError('missing explicit native splice state fields')
    roots=base.numeric_roots({k:v for k,v in state.items() if k not in EXTRA})
    pool=state['budget'];permit=state['ownership_permit']
    roots.update(closed_binding=state['closed_binding'],phase_binding=state['phase_binding'],
        native_block_calls=state['native_block_calls'],
        work_accounting=None if pool is None else dict(whole_base=pool.whole_base,branch_base=pool.branch_base,
            actual_incremental_work=pool.used,capacity=pool.capacity,parts=dict(pool.parts)),
        # Permit contains only weak references to already-exposed map arrays;
        # this separate Python footprint has no hidden numeric allocation.
        fresh_permit_python_bytes=0 if permit is None else sys.getsizeof(permit)+sys.getsizeof(permit.roots)+sys.getsizeof(permit.scales))
    return roots


def check_native_view(closed,hz,views,lb,ub,slots,nc,nb,*,pool):
    pool.charge('native_splice_view_and_scope_metadata',256)
    closed.validate()
    pre=closed.hz
    if (not any(hz is view for view in views) or hz.frame_id!=pre.frame_id or not hz.exact
            or hz.n_out!=pre.n_out or hz.n_cont!=pre.n_cont or hz.n_bin!=pre.n_bin
            or any(getattr(hz,k) is not getattr(pre,k) for k in ('Ac','Ab','Auc','Aub'))):
        raise ValueError('native input is not a bound predicate-preserving materialized view')
    for name in ('b','ub'):
        a,b=getattr(hz,name),getattr(pre,name)
        if (a.shape!=b.shape or a.dtype!=b.dtype or a.strides!=b.strides
                or a.__array_interface__['data'][0]!=b.__array_interface__['data'][0]):
            raise ValueError('native view changed bound source RHS storage')
    lower,upper=np.asarray(lb),np.asarray(ub)
    pool.charge('native_splice_actual_bounds_and_slots',32*hz.n_out)
    if (lower.shape!=(hz.n_out,) or upper.shape!=lower.shape or lower.dtype!=np.float64
            or upper.dtype!=np.float64 or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower>upper) or type(nc) is not int or type(nb) is not int
            or nc<pre.n_cont or nb<pre.n_bin):
        raise ValueError('native bounds/shared frame changed')
    k=int(np.count_nonzero((lower<0.)&(upper>0.)))
    array=np.asarray(slots) if k else np.empty((0,3),np.int64)
    pool.charge('native_splice_slot_disjointness',16*k*max(1,(2*k).bit_length()))
    if (len(slots)!=k or array.shape!=(k,3) or array.dtype.kind not in 'iu'
            or np.any(array[:,:2]<pre.n_cont) or np.any(array[:,:2]>=nc)
            or np.any(array[:,2]<pre.n_bin) or np.any(array[:,2]>=nb)
            or np.unique(array[:,:2]).size!=2*k or np.unique(array[:,2]).size!=k):
        raise ValueError('native phase slots overlap source or each other')
    return k


@contextmanager
def installed(*,enabled=False,proof_bytes=None,expected_proof_sha256=None,
              transfer_bytes=None,expected_transfer_sha256=None,
              before=None,ready=None,consumed=None,emit=None):
    if not enabled:
        yield
        return
    if type(proof_bytes) is not bytes or hashlib.sha256(proof_bytes).hexdigest()!=expected_proof_sha256:
        raise ValueError('missing/changed independently anchored C31 source proof')
    transfer=load_transfer(transfer_bytes,expected_transfer_sha256)
    if transfer['new_source_proof_sha256']!=expected_proof_sha256:
        raise ValueError('source/splice proof chain mismatched')
    if base.lift is not original_lift:raise ValueError('native splice requires unmodified original lift hook')
    original_helper=mlp.sparse_hz_apply_relu_exact;original_view=base.value_view
    selected=[]

    def report(name,**values):
        if emit is not None:emit(dict(event=name,**values))

    def entering(state):
        if selected:raise ValueError('nested selected native splice scope')
        state.update(closed_binding=None,ownership_permit=None,budget=None,phase_binding=None,native_block_calls=0)
        selected.append(state)
        if before is not None:before(state)

    def fresh(*args,**kwargs):
        if len(selected)!=1:raise ValueError('fresh C31 generator outside one native scope')
        draft,permit=generate(*args,**kwargs)
        selected[0]['ownership_permit']=permit
        return draft

    def built(state):
        draft=state['lifted'];draft_ref=weakref.ref(draft)
        retired=[weakref.ref(n[k]) for n in draft.nodes for k in ('support','needed','slots','exponents')]
        closed,binding=bind(draft,proof_bytes,expected_proof_sha256=expected_proof_sha256,enabled=True)
        state['lifted']=closed
        del draft;gc.collect()
        if draft_ref() is not None or any(ref() is not None for ref in retired):
            raise ValueError('fresh proof binding retains unpublished graph arrays')
        state['budget']=CoupledPool(closed.report['total_work_upper'],closed.report['largest_branch_work_upper'])
        state['budget'].charge('fresh_ownership_permit_issuance',32)
        state['budget'].charge('native_splice_scope_setup_and_publication',512)
        binding.update(all_new_graph_fields_physically_retired=True,retired_field_arrays=len(retired))
        state['closed_binding']=binding
        report('c32_fresh_C31_native_bound',**binding)
        if ready is not None:ready(state)

    def guarded_view(lifted,mask):
        if isinstance(lifted,SplicedState):raise ValueError('materialization after selected phase publication')
        return original_view(lifted,mask)

    def helper(hz,lb,ub,slots,n_cont,n_bin):
        if not selected:return original_helper(hz,lb,ub,slots,n_cont,n_bin)
        state=selected[-1]
        try:
            if state['native_block_calls'] or state['phase_binding'] is not None:
                raise ValueError('second selected native phase helper call')
            state['native_block_calls']=1
            closed=state['lifted'];pool=state['budget']
            k=check_native_view(closed,hz,state['views'],lb,ub,slots,n_cont,n_bin,pool=pool)
            view=phase_blocks(hz,lb,ub,slots,n_cont,n_bin,source_pre=closed.hz,enabled=True)
            # Keep the previous conservative event exposure bound, although no
            # old-post slicing is performed now. Do not claim a fake slice copy.
            pool.charge('conservative_original_phase_exposure_bound',8*(view.eq_c.nnz+view.le_c.nnz)+4*(view.eq_c.shape[0]+view.le_c.shape[0]))
            first=closed.report['radix_uid_base']+16384
            overlay,event_report=build_overlay(closed,[(view.eq_c,first),(view.le_c,first+view.eq_c.shape[0])],
                pool=pool,enabled=True)
            local=remaining_pool(pool)
            plans,discovery=discover_append(closed,view,overlay,pool=local,enabled=True)
            discovery_work,discovery_parts=local.used,dict(local.parts)
            finish_local(pool,local,'complete_append_discovery')
            if not plans:raise ValueError('selected native source has no exact reducing unit population')
            payload_cap=2*sum(m.nnz for name in ('Ac','Ab','Auc','Aub') for m in view.blocks(name))+view.n_eq+view.n_ineq
            writer_pool=WriterPool(pool,payload_cap=int(payload_cap))
            new,writer=splice_append(view,plans,pool=writer_pool,enabled=True)
            fields,raw=export_closed(closed)
            if raw!=proof_bytes:raise ValueError('fresh source proof record unexpectedly changed')
            closed_ref=weakref.ref(closed)
            state['lifted']=None
            del closed;gc.collect()
            if closed_ref() is not None:raise ValueError('published/aliased old Closed prevents owned lineage edit')
            local=remaining_pool(pool)
            lineage=compile_owned(fields['eq_roots'],fields['eq_scales'],plans,old_n_cont=fields['old_n_cont'],
                old_n_eq=fields['old_n_eq'],pool=local,ownership=state['ownership_permit'],enabled=True)
            lineage_work,lineage_parts=local.used,dict(local.parts)
            finish_local(pool,local,'complete_owned_lineage')
            state['ownership_permit']=None
            if lineage is None:raise ValueError('new actual predicates lack owned reversible reconstruction')
            construction=dict(schema='c32_actual_native_first_write_construction_v1',
                source_closed_identity=state['closed_binding']['closed_identity'],
                actual_native_helper_calls=1,new_phase_binaries=k,new_phase_continuous=2*k,
                new_EQ_rows=view.eq_c.shape[0],new_INEQ_rows=view.le_c.shape[0],
                native_old_predicate_padding_executed=False,complete_old_post_HZ_built=False,
                discovery=discovery,discovery_work=discovery_work,discovery_work_parts=discovery_parts,
                writer=writer,event_report=event_report,lineage_work=lineage_work,lineage_work_parts=lineage_parts,
                writer_dispatch_work=writer_pool.dispatch_work,
                source_maps_reused_by_identity=True,old_Closed_physically_retired=True,
                generation_and_incremental_whole_work=pool.whole_base+pool.used,
                generation_and_incremental_branch_work=pool.branch_base+pool.used,
                incremental_work_parts=dict(pool.parts),
                native_payload_work=writer_pool.native.used,native_payload_cap=writer_pool.native.cap,
                source_and_splice_hash_authentication_is_separate=True,
                root_domain_base_feasibility_proved=False,formal_gain=0)
            result,authentication=admit(enabled=True,original_fields=fields,hz=new,lineage=lineage,
                events=overlay.events,old_uid_ceiling=first,source_proof_bytes=raw,
                source_proof_sha256=expected_proof_sha256,transfer_proof_bytes=transfer_bytes,
                transfer_proof_sha256=expected_transfer_sha256,construction_report=construction)
            state['lifted']=result
            state['phase_binding']=authentication
            report('c32_actual_native_first_write_bound',construction=construction,authentication=authentication)
            return new
        except Exception as exc:raise SelectedRejected(str(exc)) from exc

    def applied(state,fact):
        if state is not selected[-1] or state['consumer_construction'] is None:
            raise ValueError('selected native phase scope/construction missing')
        result=state['lifted']
        if not isinstance(result,SplicedState) or state['native_block_calls']!=1:
            raise ValueError('native path bypassed the exact first-write helper')
        actual=state['tf']._sparse_hz_cache.get(state['layer'].id)
        if actual is not result.hz:raise ValueError('native cache did not publish the actual bound splice')
        result.validate()
        selected.pop()
        if consumed is not None:consumed(state,fact)

    base.lift=fresh;base.value_view=guarded_view;mlp.sparse_hz_apply_relu_exact=helper
    try:
        with base.installed(enabled=True,before=entering,ready=built,consumed=applied,emit=emit):yield
    finally:
        mlp.sparse_hz_apply_relu_exact=original_helper;base.value_view=original_view;base.lift=original_lift
