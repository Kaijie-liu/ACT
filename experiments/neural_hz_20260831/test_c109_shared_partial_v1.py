# SPDX-License-Identifier: AGPL-3.0-or-later
"""Complete exact ordinary fixtures and the dense shared-partial cost theorem."""
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c109_shared_partial_v1 import fixture,prove_and_measure,MODES,GRIDS


@pytest.mark.parametrize('channels,outputs',[(1,2),(3,4)])
@pytest.mark.parametrize('mode',MODES)
def test_complete_native_polynomial_inverse_and_nonconvex_source(channels,outputs,mode):
    pool=WorkPool(256_000_000)
    old,new,expected,initial=fixture(channels,outputs,mode,pool=pool,enabled=True)
    report,before,after=prove_and_measure(old,new,expected,initial,pool=pool)
    assert old['source'] is new['source']
    assert report['exact_all_output_polynomials'] and report['all_new_boxes_redundant']
    assert report['actual_inverse_both_binary_phases'] and report['all_source_EQ_INEQ_retained']
    assert int(before['integrality'].sum()) == int(after['integrality'].sum()) == 1
    assert report['new_factors'] == report['selected_partial_groups']
    assert report['new_factors'] <= 16384 and report['work'] <= 256_000_000
    assert not report['source_runtime_LIVE_admitted'] and report['formal_gain']==0


@pytest.mark.parametrize('grid',GRIDS)
def test_dense_all_groups_complete_cost_formula(grid):
    pool=WorkPool(256_000_000)
    old,new,expected,initial=fixture(1,2,'dense',grid=grid,pool=pool,enabled=True)
    report,_,_=prove_and_measure(old,new,expected,initial,pool=pool)
    gy,gx=grid
    count=8*gy*(gx-1)
    assert report['maximum_occurrences'] <= 4
    assert report['selected_partial_groups']==count
    assert report['selected_occurrences']==4*count
    assert report['nnz_delta']==-count
    assert report['byte_delta']==61*count
    assert report['entry_delta']==9*count
    assert not report['strict_complete_numeric_win']


def test_default_off_and_zero_work_rejection():
    pool=WorkPool(0)
    assert fixture(1,2,'dense',pool=pool) is None and pool.used==0
    with pytest.raises(MemoryError):fixture(1,2,'dense',pool=pool,enabled=True)


def test_original_shared_names_and_bounds_not_independent_copies():
    pool=WorkPool(256_000_000)
    old,new,expected,report=fixture(3,4,'shared',pool=pool,enabled=True)
    report,before,after=prove_and_measure(old,new,expected,report,pool=pool)
    assert np.array_equal(old['source']['parent_ids'][0],old['source']['parent_ids'][1])
    assert old['source']['old_inverse'] is new['source']['old_inverse']
    assert old['source']['binary_ids'] == new['source']['binary_ids'] == (0,)
    assert np.all(after['var_lb']==-1) and np.all(after['var_ub']==1)
