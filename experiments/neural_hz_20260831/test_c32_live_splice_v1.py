"""Independent dense toy oracle plus actual shared native call boundary."""
import gc
import hashlib
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds
from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.hybridz_tf import tf_cnn as cnn, tf_mlp as mlp
from act.back_end.hybridz_tf.hybridz_tf import HybridzTF
from experiments.neural_hz_20260831 import c32_live_splice_runtime_v1 as runtime
from experiments.neural_hz_20260831 import c32_fresh_lineage_v1 as owned
from experiments.neural_hz_20260831.c32_native_blocks_v1 import phase_blocks
from experiments.neural_hz_20260831.c32_splice_binding_v1 import semantic_fingerprint, export, admit
from experiments.neural_hz_20260831.c32_boundary_budget_v1 import WriterPool,remaining_pool,finish_local
from experiments.neural_hz_20260831.c31_prepared_emission_v1 import lift
from experiments.neural_hz_20260831.c9_integrated_suffix_v1 import lift as original_lift
from experiments.neural_hz_20260831.c24_closed_state_v1 import close, export as export_closed
from experiments.neural_hz_20260831.c24_checked_overlay_v1 import build as build_overlay
from experiments.neural_hz_20260831.c30_append_discovery_v1 import discover_append
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import compile_lineage
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c10_fused_rows_v1 import WorkPool as CoupledPool
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.test_c6_support_affine_plan_v1 import fixture
from experiments.neural_hz_20260831.test_c9_live_runtime_v1 import frame
from experiments.neural_hz_20260831.test_c30_first_write_v1 import full_reference


def toy():
    expr,_=fixture()
    keep=np.ones(expr.n_out,bool)
    draft=lift(expr,keep,enabled=True,frame_widths=(12,6))
    original=original_lift(expr,keep,enabled=True,frame_widths=(12,6))
    maps={k:getattr(original,k) for k in ('eq_roots','eq_scales','ineq_roots','ineq_scales','def_rows')}
    closed,_=close(draft,original.hz,maps,enabled=True)
    _,raw=export_closed(closed)
    bounds=Bounds(torch.tensor([-2.,.2,-2.,-2.,.1,-2.],dtype=torch.float64),
                  torch.tensor([2.,2.,-1.,2.,2.,2.],dtype=torch.float64))
    forced=bounds.ub.numpy()<=0.
    lower,upper=mlp._sparse_relu_bounds(closed.hz,bounds,forced_stable_negative=forced)
    count=int(np.count_nonzero((lower<0)&(upper>0)))
    nc,nb=closed.hz.n_cont,closed.hz.n_bin
    slots=[(nc+2*i,nc+2*i+1,nb+i) for i in range(count)]
    post=mlp.sparse_hz_apply_relu_exact(closed.hz,lower,upper,slots,nc+2*count,nb+count)
    view=phase_blocks(closed.hz,lower,upper,slots,nc+2*count,nb+count,source_pre=closed.hz,enabled=True)
    first=closed.report['radix_uid_base']+16384
    overlay,_=build_overlay(closed,[(view.eq_c,first),(view.le_c,first+count)],pool=WorkPool(256_000_000),enabled=True)
    plans,_=discover_append(closed,view,overlay,pool=WorkPool(256_000_000),enabled=True)
    assert plans, 'toy must exercise an actual reducing population'
    dense=full_reference(post,plans)
    want=SparseHZono(post.c,post.Gc,post.Gb,*(sp.csr_matrix(dense[n]) for n in ('Ac','Ab')),dense['b'],
        *(sp.csr_matrix(dense[n]) for n in ('Auc','Aub')),dense['ub'],frame_id=post.frame_id,exact=True)
    functional=compile_lineage(closed.eq_roots,closed.eq_scales,plans,old_n_cont=closed.old_n_cont,
        old_n_eq=closed.old_n_eq,pool=WorkPool(256_000_000),enabled=True)
    # Test-only independent dense row addition/deletion; never a real-input
    # proof-builder or an authorized replacement for C15/C30's full proof.
    proof=dict(schema='c32_independent_C31_C30_transfer_v1',completed=True,
        full_C31_source_math_and_report_checked=True,full_C30_HZ_UID_box_reconstruction_checked=True,
        all_pre_HZ_map_owner_UID_bits_equal=True,new_source_proof_sha256=hashlib.sha256(raw).hexdigest(),
        new_closed_identity=closed.fingerprint(),pre_HZ_sha256=source_digest(closed.hz),
        actual_spliced_HZ_sha256=source_digest(want),semantic_lineage_sha256=functional.fingerprint(),
        actual_phase_events_sha256=hashlib.sha256(overlay.events.tobytes()).hexdigest(),formal_gain=0,
        test_only_dense_oracle=True)
    transfer=json.dumps(proof,sort_keys=True).encode()
    return expr,bounds,raw,transfer,want


def execute(monkeypatch,*,layer_id=78,ready=None,corrupt=None):
    expr,bounds,raw,transfer,want=toy()
    tf=frame();retained=[]
    tf._neural_hz_sparse_phase_selective_materialization=True
    tf._neural_hz_sparse_phase_separated_relu=True
    def apply(self,layer,*args):
        result=cnn.sparse_hz_apply_affine_expr_layer(layer,expr,bounds,None,self)
        if corrupt=='rhs':result[1].b[0]+=.125
        if corrupt!='missing_cache':self._sparse_hz_cache[layer.id]=result[1]
        return result
    monkeypatch.setattr(HybridzTF,'apply',apply)
    def on_ready(state):
        retained.append(state)
        if ready:ready(state)
    with runtime.installed(enabled=True,proof_bytes=raw,expected_proof_sha256=hashlib.sha256(raw).hexdigest(),
            transfer_bytes=transfer,expected_transfer_sha256=hashlib.sha256(transfer).hexdigest(),ready=on_ready):
        result=HybridzTF.apply(tf,SimpleNamespace(id=layer_id,kind='RELU'))
    return tf,retained[0],result[1],want


def test_default_off_and_missing_proof_restore():
    old=(runtime.base.lift,mlp.sparse_hz_apply_relu_exact,HybridzTF.apply)
    assert phase_blocks(object(),object(),object(),object(),object(),object(),source_pre=object()) is None
    assert admit(original_fields=object()) is None
    with runtime.installed(proof_bytes=object(),transfer_bytes=object()):
        assert old==(runtime.base.lift,mlp.sparse_hz_apply_relu_exact,HybridzTF.apply)
    with pytest.raises(ValueError):
        with runtime.installed(enabled=True):pass
    assert old==(runtime.base.lift,mlp.sparse_hz_apply_relu_exact,HybridzTF.apply)


@pytest.mark.parametrize('layer_id',[1,78,9123])
def test_actual_native_first_write_and_new_bound_state(monkeypatch,layer_id):
    tf,state,actual,want=execute(monkeypatch,layer_id=layer_id)
    assert source_digest(actual)==source_digest(want)
    new=state['lifted'];new.validate()
    assert tf._sparse_hz_cache[layer_id] is new.hz is actual
    assert state['native_block_calls']==1 and len(state['views'])==2
    assert state['closed_binding']['all_new_graph_fields_physically_retired']
    assert new.construction_report['old_Closed_physically_retired']
    assert not new.construction_report['complete_old_post_HZ_built']
    assert new.lineage.eq_roots is new.original_fields['eq_roots']
    assert runtime.numeric_roots(state)['hz'] is actual
    payload=pickle.loads(pickle.dumps(export(new),protocol=5))
    restored,_=admit(enabled=True,**payload)
    assert source_digest(restored.hz)==source_digest(actual)


@pytest.mark.parametrize('bad',['rhs','missing_cache'])
def test_native_publication_failure_no_fallback(monkeypatch,bad):
    old=(runtime.base.lift,mlp.sparse_hz_apply_relu_exact)
    with pytest.raises(runtime.SelectedRejected):execute(monkeypatch,corrupt=bad)
    assert old==(runtime.base.lift,mlp.sparse_hz_apply_relu_exact)


@pytest.mark.parametrize('field',['source','map','events','post','report','hidden','receipt'])
def test_whole_binding_rejects_mutation_and_outer_reseal(monkeypatch,field):
    _,state,actual,_=execute(monkeypatch)
    new=state['lifted']
    if field=='source':new.original_fields['hz'].b[0]+=.125
    elif field=='map':new.lineage.eq_scales[0]+=1
    elif field=='events':
        new.events=new.events.copy();new.events[0]+=np.uint64(1)
    elif field=='post':actual.b[0]+=.125
    elif field=='report':new.construction_report['native_payload_work']=0
    elif field=='hidden':new.hidden=np.zeros(4)
    else:new.receipt=object()
    with pytest.raises(ValueError):new.validate()


def test_retained_old_closed_rejects_in_place_map_edit(monkeypatch):
    retained=[]
    with pytest.raises(runtime.SelectedRejected,match='old Closed'):
        execute(monkeypatch,ready=lambda state:retained.append(state['lifted']))
    retained[0].validate()


@pytest.mark.parametrize('mixed,subtract,redirect',[(False,False,False),(True,True,False),(False,True,True)])
def test_sparse_whole_semantic_digest_matches_functional_reference(mixed,subtract,redirect):
    from experiments.neural_hz_20260831.test_c26_tagged_transplant_v1 import setup
    from experiments.neural_hz_20260831.c22_uid_runs_v1 import pack
    hz,kw,info,eq,le,plans,overlay=setup(mixed=mixed,subtract=subtract,redirect=redirect)
    ref=compile_lineage(kw['eq_roots'],kw['eq_scales'],plans,old_n_cont=kw['old_n_cont'],
        old_n_eq=kw['old_n_eq'],pool=WorkPool(256_000_000),enabled=True)
    permit=owned._FreshPermit(owned._FRESH_KEY,kw['eq_roots'],kw['eq_scales'])
    actual=owned.compile_owned(kw['eq_roots'],kw['eq_scales'],plans,old_n_cont=kw['old_n_cont'],
        old_n_eq=kw['old_n_eq'],pool=WorkPool(256_000_000),ownership=permit,enabled=True)
    fields={'uid_slabs':np.asarray([pack(1,0,len(kw['eq_roots'])-1)],np.uint64)}
    assert semantic_fingerprint(fields,actual,pool=WorkPool(256_000_000))==ref.fingerprint()


def test_paid_native_and_incremental_ledgers_enforce_unchanged_caps():
    coupled=CoupledPool(255_000_000,190_000_000)
    writer=WriterPool(coupled,payload_cap=9)
    writer.charge('native_copy',9);writer.charge('metadata',11)
    assert writer.native.used==9 and coupled.used==19 and writer.used==20
    local=remaining_pool(coupled);local.charge('discovery',23)
    finish_local(coupled,local,'complete_discovery')
    assert coupled.used==42
    with pytest.raises(MemoryError):writer.charge('native_copy',1)
    with pytest.raises(MemoryError):coupled.charge('metadata',1_000_000)


def test_actual_phase_does_not_pad_or_build_full_post(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('old complete assembly executed')
    def ready(state):
        monkeypatch.setattr(mlp,'sparse_hz_pad_frame',forbidden)
        monkeypatch.setattr(sp,'vstack',forbidden)
        monkeypatch.setattr(sp,'hstack',forbidden)
    _,state,actual,want=execute(monkeypatch,ready=ready)
    assert source_digest(actual)==source_digest(want)


@pytest.mark.parametrize('bad',['view','rhs_alias','bounds','width','slots'])
def test_native_boundary_rejects_invalid_source_or_slots(monkeypatch,bad):
    original=runtime.check_native_view
    def changed(closed,hz,views,lb,ub,slots,nc,nb,*,pool):
        if bad=='view':views=[]
        elif bad=='rhs_alias':hz.b=hz.b.copy()
        elif bad=='bounds':lb=lb.astype(np.float32)
        elif bad=='width':nc-=1
        else:slots=[slots[0]]*len(slots)
        return original(closed,hz,views,lb,ub,slots,nc,nb,pool=pool)
    monkeypatch.setattr(runtime,'check_native_view',changed)
    with pytest.raises(runtime.SelectedRejected):execute(monkeypatch)


def test_selected_budget_failure_does_not_fallback(monkeypatch):
    original=runtime.remaining_pool
    monkeypatch.setattr(runtime,'remaining_pool',lambda p:WorkPool(0))
    with pytest.raises(runtime.SelectedRejected):execute(monkeypatch)


def test_source_transfer_hash_mismatch_before_install():
    expr,bounds,raw,transfer,want=toy()
    old=(runtime.base.lift,mlp.sparse_hz_apply_relu_exact,HybridzTF.apply)
    with pytest.raises(ValueError):
        with runtime.installed(enabled=True,proof_bytes=raw,expected_proof_sha256=hashlib.sha256(raw).hexdigest(),
            transfer_bytes=transfer+b' ',expected_transfer_sha256=hashlib.sha256(transfer).hexdigest()):pass
    assert old==(runtime.base.lift,mlp.sparse_hz_apply_relu_exact,HybridzTF.apply)


@pytest.mark.parametrize('corrupt',[False,True])
def test_independent_slot_audit_never_assembles_second_native_HZ(monkeypatch,corrupt):
    from experiments.neural_hz_20260831.c32_native_slot_audit_v1 import verify
    tf,state,actual,want=execute(monkeypatch)
    bounds=Bounds(torch.tensor([-2.,.2,-2.,-2.,.1,-2.],dtype=torch.float64),
                  torch.tensor([2.,2.,-1.,2.,2.,2.],dtype=torch.float64))
    monkeypatch.setattr(mlp,'sparse_hz_apply_relu_exact',lambda *a,**k:pytest.fail('second assembly'))
    if corrupt:
        tf._sparse_frame_widths[7]=(1,1)
        with pytest.raises(ValueError):verify(state,bounds)
    else:
        report=verify(state,bounds)
        assert report['global_widths_exact'] and not report['second_native_HZ_assembly_executed']
