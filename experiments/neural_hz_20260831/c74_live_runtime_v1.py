"""One default-off fresh local-source/native splice; source maps never edited."""
from contextlib import contextmanager
import gc
import weakref
import numpy as np
from act.back_end.hybridz_tf import tf_mlp as mlp
from experiments.neural_hz_20260831 import c9_live_runtime_v1 as base
from experiments.neural_hz_20260831.c9_integrated_suffix_v1 import lift as original_lift
from experiments.neural_hz_20260831.c69_birth_emission_v1 import lift
from experiments.neural_hz_20260831.c74_native_binding_v1 import (
    SourceState, NativeState, admit_source, admit_native, phase_image, load_transfer)
from experiments.neural_hz_20260831.c32_live_splice_runtime_v1 import check_native_view
from experiments.neural_hz_20260831.c32_native_blocks_v1 import phase_blocks
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import build as build_overlay
from experiments.neural_hz_20260831.c30_append_discovery_v1 import discover_append
from experiments.neural_hz_20260831.c30_first_write_v1 import splice_append
from experiments.neural_hz_20260831.c32_boundary_budget_v1 import WriterPool, remaining_pool, finish_local
from experiments.neural_hz_20260831.c73_outer_query_v1 import compile_journal
from experiments.neural_hz_20260831.c62_local_equations_v1 import SCHEMA
from experiments.neural_hz_20260831.c10_fused_rows_v1 import WorkPool as CoupledPool

SelectedRejected = base.SelectedRejected
EXTRA = {'source_binding', 'budget', 'phase_binding', 'native_block_calls'}


def numeric_roots(state):
    if not EXTRA <= set(state): raise ValueError('missing complete new runtime fields')
    roots = base.numeric_roots({k:v for k,v in state.items() if k not in EXTRA})
    p = state['budget']
    roots.update(source_binding=state['source_binding'], phase_binding=state['phase_binding'],
        native_block_calls=state['native_block_calls'], construction=state['construction'],
        consumer_construction=state['consumer_construction'],
        work_accounting=None if p is None else dict(whole_base=p.whole_base,
            branch_base=p.branch_base, used=p.used, capacity=p.capacity, parts=dict(p.parts)))
    return roots


def extra_bound(n_out, value_nnz, unstable, phase_nnz, phase_rows):
    """At most two materializations in the existing selected ReLU control flow.

    Both are charged even when only one occurs. A third request fails closed.
    Part count64 dominates the explicit fixed operation names in this module
    and the reused complete C73 component dispatch; runtime checks that bound.
    """
    if any(type(x) is not int or x < 0 for x in
           (n_out, value_nnz, unstable, phase_nnz, phase_rows)):
        raise ValueError('nonnegative actual structural populations required')
    return dict(source_binding_setup=256, scope_publication=512,
        value_views=2*(64+32*n_out+8*value_nnz),
        view_scope=256, actual_bounds=32*n_out,
        slot_disjointness=16*unstable*max(1,(2*unstable).bit_length()),
        phase_exposure=8*phase_nnz+4*phase_rows,
        native_binding_metadata=512, report_bookkeeping=512+64*64)


@contextmanager
def installed(*, enabled=False, source_bytes=None, source_sha=None,
              transfer_bytes=None, transfer_sha=None,
              before=None, ready=None, consumed=None, emit=None):
    if not enabled:
        yield
        return
    transfer = load_transfer(transfer_bytes, transfer_sha)
    if transfer['source_proof_sha256'] != source_sha:
        raise ValueError('source/native independent proof chain differs')
    if base.lift is not original_lift: raise ValueError('fresh original lift hook required')
    original_helper, original_view = mlp.sparse_hz_apply_relu_exact, base.value_view
    selected = []

    def report(name, **values):
        if emit is not None: emit(dict(event=name, **values))

    def entering(state):
        if selected: raise ValueError('nested selected scope')
        state.update(source_binding=None, budget=None, phase_binding=None, native_block_calls=0)
        selected.append(state)
        if before is not None: before(state)

    def built(state):
        candidate = state['lifted']; fields = candidate['fields']; g = fields['report']
        pool = CoupledPool(g['total_work_upper'], g['largest_branch_work_upper'])
        state['budget'] = pool
        pool.charge('c74_source_binding_setup',256)
        pool.charge('c74_scope_publication',512)
        refs = [weakref.ref(n[k]) for n in candidate['construction']['nodes']
                for k in ('support','needed','slots','exponents')]
        source, binding = admit_source(fields, source_bytes, expected_sha256=source_sha, enabled=True)
        state['lifted'] = source
        del candidate; gc.collect()
        if any(r() is not None for r in refs): raise ValueError('new construction graph still retained')
        state['source_binding'] = dict(binding, retired_graph_arrays=len(refs),
            fresh_original_generator=True, archived_HZ_loaded=False)
        report('c74_fresh_source_bound', **state['source_binding'])
        if ready is not None: ready(state)

    def value_view(source, keep):
        if type(source) is not SourceState or len(selected) != 1:
            raise ValueError('view requires current immutable original source')
        state = selected[0]
        if len(state['views']) >= 2: raise ValueError('native materialization call bound exceeded')
        state['budget'].charge('c74_materialized_value_view',
            64+32*source.hz.n_out+8*(source.hz.Gc.nnz+source.hz.Gb.nnz))
        return original_view(source, keep)

    def helper(hz, lb, ub, slots, n_cont, n_bin):
        if not selected: return original_helper(hz, lb, ub, slots, n_cont, n_bin)
        state = selected[0]
        try:
            if state['native_block_calls']: raise ValueError('second selected native helper')
            state['native_block_calls'] = 1
            source, pool = state['lifted'], state['budget']
            k = check_native_view(source,hz,state['views'],lb,ub,slots,n_cont,n_bin,pool=pool)
            view = phase_blocks(hz,lb,ub,slots,n_cont,n_bin,source_pre=source.hz,enabled=True)
            pool.charge('conservative_original_phase_exposure_bound',
                8*(view.eq_c.nnz+view.le_c.nnz)+4*(len(view.eq_rhs)+len(view.le_rhs)))
            first = source.report['radix_uid_base']+16384
            local = remaining_pool(pool)
            overlay, event = build_overlay(source.owners,
                [(view.eq_c,first),(view.le_c,first+len(view.eq_rhs))],
                old_n_cont=source.old_n_cont,old_uid_ceiling=first,pool=local,enabled=True)
            finish_local(pool,local,'complete_actual_append_overlay')
            local = remaining_pool(pool)
            plans, discovery = discover_append(source,view,overlay,pool=local,enabled=True)
            finish_local(pool,local,'all_actual_consumer_discovery')
            if not plans: raise ValueError('selected structure has no strict unit reduction')
            payload_cap = 2*sum(m.nnz for name in ('Ac','Ab','Auc','Aub')
                                for m in view.blocks(name))+view.n_eq+view.n_ineq
            writer_pool = WriterPool(pool,payload_cap=int(payload_cap))
            new, writer = splice_append(view,plans,pool=writer_pool,enabled=True)
            local = remaining_pool(pool)
            journal = compile_journal(source.eq_roots,source.eq_scales,plans,
                old_n_cont=source.old_n_cont,old_n_eq=source.old_n_eq,
                source_n_cont=source.hz.n_cont,source_schema=SCHEMA,pool=local,enabled=True)
            finish_local(pool,local,'complete_actual_local_splice_journal')
            pool.charge('c74_native_binding_metadata',512)
            pool.charge('c74_report_bookkeeping',512+64*(len(pool.parts)+1))
            if len(pool.parts)>64: raise ValueError('runtime bookkeeping population exceeds bound')
            construction = dict(schema='c74_fresh_local_native_construction_v1',
                new_phase_binaries=k,new_phase_continuous=2*k,actual_native_helper_calls=1,
                view_calls=len(state['views']),discovery=discovery,event_report=event,writer=writer,
                source_maps_shared_unchanged=True,native_old_predicate_padding_executed=False,
                complete_old_post_HZ_built=False,
                generation_and_incremental_whole_work=pool.whole_base+pool.used,
                generation_and_incremental_branch_work=pool.branch_base+pool.used,
                incremental_work_parts=dict(pool.parts),native_payload_work=writer_pool.native.used,
                native_payload_cap=writer_pool.native.cap,formal_gain=0)
            result, binding = admit_native(enabled=True,source=source,hz=new,lineage=journal,
                events=overlay.events,actual_phase_image=phase_image(source,view,transfer),
                transfer_proof_bytes=transfer_bytes,expected_transfer_sha256=transfer_sha,
                construction_report=construction)
            state['lifted'], state['phase_binding'] = result, binding
            report('c74_fresh_native_bound',construction=construction,binding=binding)
            return new
        except Exception as exc: raise SelectedRejected(str(exc)) from exc

    def applied(state, fact):
        if (state is not selected[-1] or state['consumer_construction'] is None
                or type(state['lifted']) is not NativeState or state['native_block_calls'] != 1
                or state['tf']._sparse_hz_cache.get(state['layer'].id) is not state['lifted'].hz):
            raise ValueError('fresh native source/cache publication missing')
        state['lifted'].validate()
        selected.pop()
        if consumed is not None: consumed(state,fact)

    base.lift, base.value_view, mlp.sparse_hz_apply_relu_exact = lift, value_view, helper
    try:
        with base.installed(enabled=True,before=entering,ready=built,consumed=applied,emit=emit): yield
    finally:
        base.lift, base.value_view, mlp.sparse_hz_apply_relu_exact = original_lift, original_view, original_helper
