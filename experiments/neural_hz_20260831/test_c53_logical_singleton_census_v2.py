import numpy as np
import pytest
from experiments.neural_hz_20260831.test_c53_source_normalization_v1 import fixture
from experiments.neural_hz_20260831.c53_source_normalization_v1 import assess as source_assess
from experiments.neural_hz_20260831.c53_logical_singleton_census_v2 import assess
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool


def complete(kind='chain'):
    candidate,saved=fixture(kind)
    saved.update({n:getattr(candidate,n) for n in ('old_n_eq','eq_roots','eq_scales','ineq_roots','ineq_scales','def_rows')})
    return candidate,saved


@pytest.mark.parametrize('kind',['chain','shared','conv'])
def test_logical_row_census_equals_full_original_source_census(kind):
    _,saved=complete(kind)
    a=source_assess(saved,pool=WorkPool(256_000_000),enabled=True)
    b=assess(saved,pool=WorkPool(256_000_000),enabled=True)
    for n in ('current_C52_signed_unit_population','potential_scale_aware_singleton_only_lower_bound',
            'normalized_power_histogram','potential_chain_depth_histogram','potential_chain_power_histogram','potential_identity_sha256'):
        assert a[n]==b[n]
    assert b['totals']['MAIN_rows']==a['totals']['MAIN_rows']
    assert b['no_first_merge_proved'] and not b['actual_forwarding_or_new_HZ_constructed']


@pytest.mark.parametrize('bad',['roots','pivot','output'])
def test_complete_logical_frame_binding_rejects_mismatch(bad):
    _,saved=complete()
    if bad=='roots':saved['eq_roots'][-1]=saved['eq_roots'][-2]
    elif bad=='pivot':saved['eq_scales'][-1]+=1
    else:saved['hz'].Gc.data[0]*=2
    with pytest.raises(ValueError):assess(saved,pool=WorkPool(256_000_000),enabled=True)


def test_default_off_has_no_input_access():
    assert assess(object(),pool=object()) is None


def test_unchanged_work_cap_rejects():
    _,saved=complete()
    with pytest.raises(MemoryError):assess(saved,pool=WorkPool(0),enabled=True)


def test_existing_packed_logical_row_is_not_skipped():
    from act.back_end.hybridz_tf import tf_cnn as cnn
    from experiments.neural_hz_20260831.test_c5_live_value_contraction_v1 import source
    from experiments.neural_hz_20260831.c9_integrated_suffix_v1 import lift
    s=source(8);s.Gc.data[0]=2.**-70
    expr=cnn.SparseHZAffineExpr((cnn.SparseHZAffineTerm(s,()),),np.zeros(8),8,s.frame_id)
    candidate=lift(expr,np.ones(8,bool),enabled=True)
    saved={n:getattr(candidate,n) for n in ('old_n_cont','logical_n_cont','old_n_bin','root','keep','old_n_eq','eq_roots','eq_scales','ineq_roots','ineq_scales','def_rows')}
    saved.update(definition_graph=candidate.nodes,expression=expr,hz=candidate.hz)
    result=assess(saved,pool=WorkPool(256_000_000),enabled=True)
    assert len(candidate.def_rows)>0 and result['totals']['packed_MAIN_rows_recovered']>0
    assert result['radix_definitions_recovered_for_MAIN']==len(candidate.def_rows)
    assert result['no_first_merge_proved']
