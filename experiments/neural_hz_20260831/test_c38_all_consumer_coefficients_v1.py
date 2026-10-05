from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace
import copy
import numpy as np
import pytest
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import incidence_oracle,BranchPool
from experiments.neural_hz_20260831.c38_all_consumer_coefficients_v1 import coefficient,discover,audit_and_screen,census
from experiments.neural_hz_20260831.test_c32_live_splice_v1 import execute


def fixture(*,failure=None,shared=False,internal=False,live=False,binary=False,box=False):
    ac=np.zeros((4,5));ac[0,[0,3]]=[-.5,1.];ac[1,[1,4]]=[-.25,1.]
    ac[2,[0,3,4]]=[.5,1.,1.];ac[3,[2,3,4]]=[.25,.5,.5]
    au=np.array([[-.5,-.25,0.,1.,1.]])
    if failure:
        ac[1,1]=-(1.+2.**-26)/2
        if failure=='late':ac[3,4]=1.+2.**-27
        elif failure=='first':ac[2,4]=1.+2.**-27
        else:au[0,4]=1.+2.**-27;au[0,1]=0.
    if shared:ac[1,1]=0.;ac[1,0]=-.25
    if internal:ac[1,1]=0.;ac[1,3]=-.25
    if box:ac[1,1]=-.95
    ab=np.zeros((4,1));ab[3,0]=.125
    if binary:ab[1,0]=.125
    gc=np.zeros((1,5));gc[0,3 if live else 2]=1.
    hz=SparseHZono(np.zeros(1),sp.csr_matrix(gc),sp.csr_matrix((1,1)),
        sp.csr_matrix(ac),sp.csr_matrix(ab),np.array([0.,.125,0.,.25]),
        sp.csr_matrix(au),sp.csr_matrix([[.25]]),np.array([.5]),frame_id=123,exact=True)
    state=SimpleNamespace(hz=hz,original_fields=dict(old_n_cont=3,logical_n_cont=5,old_n_eq=0,eq_roots=np.array([0,1],np.int64)))
    tables=dict(definitions=np.array([0,1],np.int64),eq=np.array([10,11,12,13]),le=np.array([21]))
    words=incidence_oracle(hz,tables['eq'],tables['le'],3,5,pool=WorkPool(256_000_000))
    return state,tables,words


def screen(state,tables,words,*,pool=None,branch=None):
    pool=WorkPool(256_000_000) if pool is None else pool
    table,discovery=discover(state,state.hz,tables,words,pool=pool)
    report,journal=audit_and_screen(state.hz,tables,words,table,old_nc=3,logical_nc=5,
        pool=pool,branch=pool if branch is None else branch)
    return report,table,journal,discovery


def test_default_off_never_reads_inputs():
    assert census(object(),object(),pool=object()) is None


@pytest.mark.parametrize('q,w,r,reason,value',[
    (1.,.5,.5,0,1.),(1.,-.5,.5,0,0.),(-1.,0.,.5,0,-.5),
    (1.+2.**-26,0.,(1.+2.**-27)/2,2,None),
    (2.**-20,0.,.5,3,2.**-21),(2.**40,2.**40,1.,3,2.**41),
    (0.,0.,.5,1,None),(2.**-21,0.,.5,1,None),(1.,2.**-21,.5,1,None),
    (float('inf'),0.,.5,1,None),(1.,float('nan'),.5,1,None),(1.,0.,float('inf'),1,None),
])
def test_exact_coefficient_fail_closed_reason(q,w,r,reason,value):
    got,combined,zero=coefficient(q,w,r,pool=WorkPool(256_000_000))
    assert got==reason
    if value is not None:assert combined==value
    if reason==0:
        assert F(combined)==F(w)+F(q)*F(r)
        assert zero==(combined==0.)


def test_joint_exactness_can_pass_when_the_standalone_product_is_inexact():
    q=1.+2.**-26;r=(1.+2.**-27)/2;w=(1.-2.**-53)/2
    assert F(float(F(q)*F(r)))!=F(q)*F(r)
    reason,value,zero=coefficient(q,w,r,pool=WorkPool(256_000_000))
    assert reason==0 and not zero and F(value)==F(w)+F(q)*F(r)


def test_all_actual_EQ_INEQ_consumers_and_complete_UID_journal():
    state,tables,words=fixture();before=state.hz.Ac.copy()
    report,table,journal,discovery=screen(state,tables,words)
    assert report['complete_actual_incidence_equal'] and report['all_current_physical_rows_scanned']==5
    assert report['body_coefficient_survivors']==2 and report['all_consumer_occurrences']==6
    assert np.array_equal(table['seen'],[3,3]) and np.array_equal(table['definitions_seen'],[1,1])
    assert report['observed_overlaps']==3 and report['observed_exact_cancellations']==2
    assert report['homogeneous_body_survivors']==1 and report['nonhomogeneous_body_survivors']==1
    assert set(map(int,journal))=={12,13,21,(1<<20)|12,(1<<20)|13,(1<<20)|21}
    assert report['numerical_occurrences_checked']==6 and report['unexamined_occurrences_after_necessary_failure']==0
    assert discovery['all_MAIN_classified']==2 and sum(discovery['discovery_counts'].values())==2
    assert np.array_equal(before.data,state.hz.Ac.data) and state.hz.n_bin==1
    assert not any(report[k] for k in ('joint_RHS_arithmetic_executed','simultaneous_RHS_proved',
        'new_lineage_or_reconstruction_proved','actual_nnz_reduction_proved','new_HZ_constructed','solver_executed','formal_gain'))


@pytest.mark.parametrize('failure,checked,uid,inequality',[('first',1,12,False),('late',2,13,False),('ineq',3,21,True)])
def test_failure_still_counts_all_later_rows_without_marking_them_passed(failure,checked,uid,inequality):
    state,tables,words=fixture(failure=failure)
    report,table,journal,_=screen(state,tables,words)
    assert report['body_coefficient_survivors']==1 and report['all_consumer_occurrences']==6
    assert table['reason'][1]==2 and table['checked'][1]==checked and table['seen'][1]==3
    assert table['failed_uid'][1]==uid and bool(table['failed_inequality'][1])==inequality
    exact=F(float(table['failed_w'][1]))+F(float(table['failed_q'][1]))*F(float(table['ratio'][1]))
    assert F(float(table['failed_float_candidate'][1]))!=exact
    assert set(map(int,journal))=={12,13,21}
    assert report['unexamined_occurrences_after_necessary_failure']==3-checked


@pytest.mark.parametrize('options,selected,key',[
    ({'live':True},1,'output_live'),({'binary':True},1,'shape_rejected'),
    ({'box':True},1,'local_scalar_or_box_rejected'),
])
def test_uniform_definition_guards_discard_only_the_registered_structural_cohort(options,selected,key):
    state,tables,words=fixture(**options)
    table,report=discover(state,state.hz,tables,words,pool=WorkPool(256_000_000))
    assert len(table)==selected and report['discovery_counts'][key]==1


@pytest.mark.parametrize('options',[{'shared':True},{'internal':True}])
def test_no_greedy_subset_when_whole_parent_independence_fails(options):
    state,tables,words=fixture(**options)
    with pytest.raises(ValueError,match='entire proposed cohort'):
        discover(state,state.hz,tables,words,pool=WorkPool(256_000_000))


@pytest.mark.parametrize('bad',['claim','uid','uid_range','uid_length','degree','definition','nan','zero','noncanonical','budget'])
def test_incomplete_or_corrupt_actual_incidence_cannot_publish_a_subset(bad):
    state,tables,words=fixture();pool=WorkPool(256_000_000)
    table,_=discover(state,state.hz,tables,words,pool=pool)
    if bad=='claim':words[0]+=1
    elif bad=='uid':tables['le'][0]=tables['eq'][0]
    elif bad=='uid_range':tables['le'][0]=1<<20
    elif bad=='uid_length':tables['le']=np.array([],np.int64)
    elif bad=='degree':table['degree'][0]+=1
    elif bad=='definition':table['definition'][0]=100
    elif bad=='nan':state.hz.Ac.data[0]=np.nan
    elif bad=='zero':state.hz.Ac.data[0]=0.
    elif bad=='noncanonical':state.hz.Ac.has_canonical_format=False
    else:pool=WorkPool(0)
    with pytest.raises((ValueError,MemoryError)):
        audit_and_screen(state.hz,tables,words,table,old_nc=3,logical_nc=5,pool=pool,branch=pool)


def test_discovery_rejects_incomplete_definition_map_and_active_removed_factor():
    state,tables,words=fixture();state.original_fields['eq_roots'][0]=-1
    with pytest.raises(ValueError,match='removed factor'):discover(state,state.hz,tables,words,pool=WorkPool(256_000_000))
    state.original_fields['eq_roots']=np.array([0])
    with pytest.raises(ValueError,match='incomplete'):discover(state,state.hz,tables,words,pool=WorkPool(256_000_000))


def test_branch_is_nested_in_the_whole_work_and_guard_is_precharged():
    state,tables,words=fixture();whole=WorkPool(256_000_000);branch=BranchPool(whole)
    screen(state,tables,words,pool=whole,branch=branch)
    assert whole.used>branch.used>0
    assert set(branch.parts)=={'independent_complete_incidence','compare_every_transferred_MAIN_word'}
    with pytest.raises(MemoryError):coefficient(1.,0.,.5,pool=WorkPool(127))


@pytest.mark.parametrize('layer',[2,78,1001])
def test_public_census_on_actual_bound_toy_uses_no_layer_identity(monkeypatch,layer):
    from experiments.neural_hz_20260831 import test_c32_live_splice_v1 as native_toy
    from act.back_end.hybridz_tf import tf_cnn as cnn
    original_fixture=native_toy.fixture
    def single_term():
        expr,op=original_fixture()
        return cnn.SparseHZAffineExpr((expr.terms[0],),expr.bias,expr.n_out,expr.frame_id),op
    # A different SYNTHETIC input is generated and independently proved before
    # construction. No bound rows, receipt or discovery guard is monkeypatched.
    monkeypatch.setattr(native_toy,'fixture',single_term)
    _,runtime,hz,_=execute(monkeypatch,layer_id=layer)
    report,table,journal=census(runtime['lifted'],hz,pool=WorkPool(256_000_000),enabled=True)
    assert report['complete_actual_incidence_equal'] and report['all_source_and_final_bytes_unchanged']
    assert len(table)==report['proposed_factors'] and len(journal)==report['survivor_UID_journal_entries']
    assert report['body_coefficient_survivors']<=len(table) and not report['new_HZ_constructed']


@pytest.mark.parametrize('layer',[2,78,1001])
def test_original_bound_multibranch_toy_correctly_rejects_shared_parents(monkeypatch,layer):
    _,runtime,hz,_=execute(monkeypatch,layer_id=layer)
    with pytest.raises(ValueError,match='entire proposed cohort'):
        census(runtime['lifted'],hz,pool=WorkPool(256_000_000),enabled=True)


@pytest.mark.parametrize('bad',['post','source','lineage','receipt','copy','budget','branch','type'])
def test_complete_source_or_proof_corruption_fails_closed(monkeypatch,bad):
    _,runtime,hz,_=execute(monkeypatch);state=runtime['lifted'];pool=WorkPool(256_000_000);branch=None
    if bad=='post':hz.b[0]+=.125
    elif bad=='source':state.original_fields['hz'].b[0]+=.125
    elif bad=='lineage':state.lineage.eq_roots[0]+=1
    elif bad=='receipt':state.receipt=object()
    elif bad=='copy':hz=copy.copy(hz);hz.Ac=hz.Ac.copy()
    elif bad=='budget':pool=WorkPool(0)
    elif bad=='branch':branch=BranchPool(WorkPool(256_000_000))
    else:state=SimpleNamespace()
    with pytest.raises((ValueError,MemoryError)):
        census(state,hz,pool=pool,branch=branch,enabled=True)


def test_worker_strict_restore_ceremony_is_identical_to_completed_C37():
    root=Path(__file__).resolve().parent
    expected=(root/'c37_definition_inventory_worker_v1.py').read_text().replace(
        'c37_definition_inventory_v1','c38_all_consumer_coefficients_v1').replace(
        'c37_definition_inventory_20260911_v1','c38_all_consumer_coefficients_20260911_v1').replace(
        '(result,table),measurement','(result,table,journal),measurement').replace(
        'np.savez(f,table=table)','np.savez(f,table=table,survivor_consumer_uids=journal)').replace(
        "event='complete_inventory_saved'","event='complete_coefficient_screen_saved'").replace(
        "local_affine_boxed_scalars=result['local_affine_boxed_scalars']","body_coefficient_survivors=result['body_coefficient_survivors']")
    assert (root/'c38_all_consumer_coefficients_worker_v1.py').read_text()==expected
