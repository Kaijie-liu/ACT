# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full same quotient and actual global source-plan qualification."""
import inspect
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.test_c112_factored_key_v1 import CASES
from experiments.neural_hz_20260831.c113_prepared_fixture_v1 import complete
from experiments.neural_hz_20260831.c113_complete_source_fixture_v1 import complete_source
from experiments.neural_hz_20260831.c113_prepared_tile_v1 import construct,prepare_tile,emit_tile
from experiments.neural_hz_20260831.c95_word_filter_v1 import prepare_words


@pytest.mark.parametrize('c,k,mode,grid',CASES)
def test_full_once_prepared_native_inverse_and_complete_bill(c,k,mode,grid):
    report,before,after,old,new,expected=complete(c,k,mode,grid,pool=WorkPool(64_000_000))
    assert report['complete_C112_quotient_unchanged']
    assert report['complete_C112_key_work_not_repeated']
    assert report['once_prepared_all_tiles_before_emission']
    assert report['exact_all_output_polynomials'] and report['all_new_boxes_redundant']
    assert report['actual_inverse_both_binary_phases'] and report['all_source_EQ_INEQ_retained']
    assert report['constructor_vs_C112_delta']==report['preparation_inventory_work']
    assert before['old_inverse'] is after['old_inverse']


@pytest.mark.parametrize('mode',['dense','masked'])
def test_actual_complete_multi_tile_source_and_independent_owners(mode):
    report,held=complete_source(mode,pool=WorkPool(64_000_000))
    assert report['all_MAIN_owners_independently_equal']
    assert report['actual_complete_inverse_equal'] and report['original_expression_unchanged']
    assert report['global_generation']['once_prepared_tiles']>=2
    assert report['new_nnz']<report['old_nnz']
    assert all(p['original_source_equivalence'] and p['universal_unique_box_extension']
               for p in report['block_proofs'])


def test_default_off_single_consumption_and_original_exact_emitter():
    pool=WorkPool(1_000_000)
    assert construct(None,None,None,None,None,0,pool=pool) is None and pool.used==0
    assert prepare_tile(None,None,None,None,None,0,pool=pool) is None
    assert emit_tile(None,0,pool=pool) is None
    _,ready=prepare_words(np.arange(1,19,dtype=np.float32).reshape(2,1,3,3)/32,pool=pool,enabled=True)
    prepared=prepare_tile(ready,np.arange(16,dtype=np.int64).reshape(1,4,4),
        np.zeros((1,4,4),np.int32),np.arange(16,24,dtype=np.int64).reshape(2,2,2),
        np.full((2,2,2),20,np.int32),24,pool=pool,enabled=True)
    count=prepared.inventory['new_factors']
    report,packet=emit_tile(prepared,24,pool=pool,enabled=True)
    assert count==report['new_factors'] and prepared.state is None
    with pytest.raises(ValueError):emit_tile(prepared,24,pool=pool,enabled=True)
    from experiments.neural_hz_20260831 import c96_word_row_v1 as old
    from experiments.neural_hz_20260831 import c113_prepared_tile_v1 as new
    assert inspect.getsource(old._row)==inspect.getsource(new._row)
    assert inspect.getsource(old._emit_row)==inspect.getsource(new._emit_row)
