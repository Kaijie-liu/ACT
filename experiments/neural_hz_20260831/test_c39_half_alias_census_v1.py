from pathlib import Path
import copy
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.c38_all_consumer_coefficients_v1 import discover
from experiments.neural_hz_20260831.c39_half_alias_census_v1 import select_halves,audit_rows,census
from experiments.neural_hz_20260831.test_c38_all_consumer_coefficients_v1 import fixture
from experiments.neural_hz_20260831.test_c32_live_splice_v1 import execute


def setup(two=False):
    state,tables,words=fixture()
    if two:
        state.hz.Ac.data[state.hz.Ac.indptr[1]]=-.5;state.hz.b[1]=0.
    pool=WorkPool(256_000_000)
    cohort,discovery=discover(state,state.hz,tables,words,pool=pool)
    half,algebra=select_halves(cohort,pool=pool)
    return state,tables,words,half,algebra,pool


def test_default_off_never_inspects_inputs():
    assert census(object(),object(),pool=object()) is None


@pytest.mark.parametrize('two',[False,True])
def test_complete_joint_rows_include_EQ_INEQ_and_unselected_factors(two):
    state,tables,words,half,algebra,pool=setup(two)
    report,rows=audit_rows(state.hz,tables,words,half,old_nc=3,logical_nc=5,pool=pool,branch=pool)
    assert algebra['all_one_parent_factors_classified']==2 and len(half)==1+int(two)
    assert sum(algebra['complete_one_parent_algebraic_partition'].values())==2
    assert report['complete_actual_incidence_equal'] and report['all_current_physical_rows_scanned']==5
    assert report['all_joint_row_arithmetic_proved'] and not report['whole_cohort_rejected']
    assert report['complete_touched_rows']==3 and report['all_half_consumer_occurrences']==3*len(half)
    assert np.array_equal(rows['uid'],[12,13,21]) and np.all(rows['decision']==1)
    assert report['complete_touched_binary_nnz']==2 and report['hypothetical_full_predicate_nnz_delta']<0
    assert not report['complete_HZ_transformation_proved'] and not report['whole_LIVE_decrease_proved']
    assert state.hz.n_bin==1 and not report['solver_executed'] and report['formal_gain']==0


def test_late_failure_rejects_entire_transaction_and_preserves_complete_row_journal():
    state,tables,words,half,_,pool=setup(True)
    a,b=state.hz.Ac.indptr[3:5];indices=state.hz.Ac.indices[a:b]
    state.hz.Ac.data[a+int(np.searchsorted(indices,2))]=2.**-23
    report,rows=audit_rows(state.hz,tables,words,half,old_nc=3,logical_nc=5,pool=pool,branch=pool)
    assert report['whole_cohort_rejected'] and not report['all_joint_row_arithmetic_proved']
    assert report['hypothetical_full_predicate_nnz_delta'] is None and not report['successful_subset_published']
    assert report['all_half_consumer_occurrences']==6 and report['complete_touched_rows']==3
    assert report['first_failure']['uid']==13 and report['first_failure']['column']==2
    assert report['first_failure']['diagnostic_value']==2.**-23
    assert np.array_equal(rows['decision'],[1,2,0]) and np.all(half['seen']==3)
    assert report['numeric_rows_unknown_after_first_failure']==1


@pytest.mark.parametrize('bad',['claim','uid','degree','definition','canonical','nan','budget'])
def test_partial_or_corrupt_actual_scan_cannot_publish_complete_evidence(bad):
    state,tables,words,half,_,pool=setup(True)
    if bad=='claim':words[0]+=1
    elif bad=='uid':tables['le'][0]=tables['eq'][0]
    elif bad=='degree':half['degree'][0]+=1
    elif bad=='definition':half['definition'][0]=100
    elif bad=='canonical':state.hz.Ac.has_canonical_format=False
    elif bad=='nan':state.hz.Ac.data[0]=np.nan
    else:pool=WorkPool(0)
    with pytest.raises((ValueError,MemoryError)):
        audit_rows(state.hz,tables,words,half,old_nc=3,logical_nc=5,pool=pool,branch=pool)


def test_same_whole_pool_pays_current_branch_and_row_arithmetic():
    state,tables,words,half,_,pool=setup(True);branch=BranchPool(pool)
    audit_rows(state.hz,tables,words,half,old_nc=3,logical_nc=5,pool=pool,branch=branch)
    assert pool.used>branch.used>0 and pool.parts['gauge_exact_joint_parent']>0


@pytest.mark.parametrize('layer',[2,78,1001])
def test_actual_independently_proved_synthetic_source_and_final_unchanged(monkeypatch,layer):
    from experiments.neural_hz_20260831 import test_c32_live_splice_v1 as native_toy
    from act.back_end.hybridz_tf import tf_cnn as cnn
    original=native_toy.fixture
    def single():
        expr,op=original()
        return cnn.SparseHZAffineExpr((expr.terms[0],),expr.bias,expr.n_out,expr.frame_id),op
    monkeypatch.setattr(native_toy,'fixture',single)
    _,runtime,hz,_=execute(monkeypatch,layer_id=layer)
    report,table,rows=census(runtime['lifted'],hz,pool=WorkPool(256_000_000),enabled=True)
    assert report['complete_actual_incidence_equal'] and report['all_source_and_final_bytes_unchanged']
    assert len(table)==report['proposed_half_aliases'] and len(rows)==report['complete_touched_rows']
    assert not report['complete_HZ_transformation_proved'] and not report['solver_executed']


@pytest.mark.parametrize('bad',['post','source','lineage','receipt','copy','budget','branch'])
def test_public_bound_state_corruption_fails_closed(monkeypatch,bad):
    _,runtime,hz,_=execute(monkeypatch);state=runtime['lifted'];pool=WorkPool(256_000_000);branch=None
    if bad=='post':hz.b[0]+=.125
    elif bad=='source':state.original_fields['hz'].b[0]+=.125
    elif bad=='lineage':state.lineage.eq_roots[0]+=1
    elif bad=='receipt':state.receipt=object()
    elif bad=='copy':hz=copy.copy(hz);hz.Ac=hz.Ac.copy()
    elif bad=='budget':pool=WorkPool(0)
    else:branch=BranchPool(WorkPool(256_000_000))
    with pytest.raises((ValueError,MemoryError)):census(state,hz,pool=pool,branch=branch,enabled=True)


def test_worker_complete_strict_restore_matches_C38():
    root=Path(__file__).resolve().parent
    expected=(root/'c38_all_consumer_coefficients_worker_v1.py').read_text().replace(
        'c38_all_consumer_coefficients_v1','c39_half_alias_census_v1').replace(
        'c38_all_consumer_coefficients_20260911_v1','c39_half_alias_row_gauge_20260911_v1').replace(
        'survivor_consumer_uids=journal','consumer_rows=journal').replace(
        "event='complete_coefficient_screen_saved'","event='complete_joint_gauge_screen_saved'").replace(
        "body_coefficient_survivors=result['body_coefficient_survivors']","all_joint_row_arithmetic_proved=result['all_joint_row_arithmetic_proved']")
    assert (root/'c39_half_alias_row_gauge_worker_v1.py').read_text()==expected
