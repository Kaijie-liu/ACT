# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full original nonconvex source fixtures and whole-transaction guards."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.test_c112_factored_key_v1 import CASES
from experiments.neural_hz_20260831.c115_atomic_fixture_v1 import complete
from experiments.neural_hz_20260831.c115_atomic_word_plan_v1 import construct,prepare_tile,emit_tile,_plan_row,reduce_rows
from experiments.neural_hz_20260831.c95_word_filter_v1 import prepare_words


@pytest.mark.parametrize('c,k,mode,grid',CASES)
def test_complete_atomic_native_source_and_inverse(c,k,mode,grid):
    report,before,after,old,new,expected=complete(c,k,mode,grid,pool=WorkPool(64_000_000))
    assert report['exact_all_output_polynomials'] and report['all_new_boxes_redundant']
    assert report['actual_inverse_both_binary_phases'] and report['all_source_EQ_INEQ_retained']
    assert report['once_prepared_all_tiles_before_emission']
    assert before['old_inverse'] is after['old_inverse']
    assert report['nnz_delta']<=0 and report['new_factors']<=0
    if report['omitted_factors']:assert report['strict_complete_numeric_win']


def test_single_consumption_default_off_and_global_offset():
    pool=WorkPool(1_000_000)
    assert construct(None,None,None,None,None,0,pool=pool) is None and pool.used==0
    assert prepare_tile(None,None,None,None,None,0,pool=pool) is None
    assert emit_tile(None,0,pool=pool) is None
    _,kernel=prepare_words(np.arange(1,19,dtype=np.float32).reshape(2,1,3,3)/32,pool=pool,enabled=True)
    ready=prepare_tile(kernel,np.arange(16,dtype=np.int64).reshape(1,4,4),
        np.zeros((1,4,4),np.int32),np.arange(16,24,dtype=np.int64).reshape(2,2,2),
        np.full((2,2,2),20,np.int32),24,pool=pool,enabled=True)
    count=ready.inventory['new_factors'];report,packet=emit_tile(ready,100,pool=pool,enabled=True)
    assert report['new_factors']==count and ready.state is None
    assert np.array_equal(packet['pivots'][:count],np.arange(100,100+count))
    assert np.all((packet['columns']<24)|(packet['columns']>=100))
    with pytest.raises(ValueError):emit_tile(ready,100,pool=pool,enabled=True)


def test_whole_neutral_M_proposal_rolls_back_without_root_gain():
    pool=WorkPool(1_000_000)
    def planned(terms,pivot,power=0):
        pool.charge('test_complete_symbolic_row',64+8*(len(terms)+1))
        return _plan_row(terms,pivot,power,pool=pool)
    # Seven retained roots; two four-parent sinks share one root with unequal
    # coefficients, so neither signed output cancels it. Total nnz is neutral.
    rows=[planned([(i,-1,0)],10+i) for i in range(7)]
    rows += [planned([(c,-1,-2) for c in (10,11,12,13)],17),
             planned([(13,-1,-3),*((c,-1,-2) for c in (14,15,16))],18)]
    rows += [planned([(17,-1,-1),(18,-1,-1)],7),planned([(17,-1,-1),(18,1,-1)],8)]
    before=deepcopy(rows)
    plan=reduce_rows(rows,9,{17,18},10,0,pool=pool)
    assert plan['decisions'][0]['nnz_delta']==0
    assert plan['all_M_changes_rolled_back'] and not plan['removed_m']
    assert plan['rows']==before and rows==before


def test_zero_resource_budget_fails_closed():
    with pytest.raises(MemoryError):
        prepare_tile({'numerator':np.ones((1,1,4,4),np.int64),'exponent':np.zeros((1,1),np.int32)},
            np.zeros((1,4,4),np.int64),np.zeros((1,4,4),np.int32),
            np.arange(16,20,dtype=np.int64).reshape(1,2,2),np.full((1,2,2),20,np.int32),
            20,pool=WorkPool(0),enabled=True)
