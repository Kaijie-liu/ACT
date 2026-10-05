# SPDX-License-Identifier: AGPL-3.0-or-later
"""Whole ordinary source-first identity and raw signed/dyadic routing tests."""
from fractions import Fraction as F
import inspect
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c109_shared_partial_v1 import MODES,GRIDS
from experiments.neural_hz_20260831.c111_form_reuse_v1 import routes
from experiments.neural_hz_20260831.c111_source_first_fixture_v1 import complete
from experiments.neural_hz_20260831.c111_source_first_row_v1 import construct

CASES=[(c,k,mode,(2,2)) for c,k in ((1,2),(3,4)) for mode in MODES]
CASES += [(1,2,'dense',grid) for grid in GRIDS]


@pytest.mark.parametrize('channels,outputs,mode,grid',CASES)
def test_complete_old_and_new_nonconvex_sources_and_literal_polynomials(channels,outputs,mode,grid):
    pool=WorkPool(256_000_000)
    report,before,after,old,new,_=complete(channels,outputs,mode,grid,pool=pool)
    assert report['exact_all_output_polynomials'] and report['actual_inverse_both_binary_phases']
    assert report['all_new_boxes_redundant'] and report['all_source_EQ_INEQ_retained']
    assert old['source'] is new['source'] and before['old_inverse'] is after['old_inverse']
    assert int(before['integrality'].sum())==int(after['integrality'].sum())==1
    expected=4 if channels==3 and mode=='masked' else 128 if channels==3 and mode=='shared' else 0
    assert report['omitted_factors']==expected
    assert report['strict_complete_numeric_win']==bool(expected)
    assert report['original_native_comparator_reproduced']
    assert not report['source_runtime_LIVE_admitted'] and report['formal_gain']==0
    assert pool.parts['c111_complete_existing_form_keys']>0


@pytest.mark.parametrize('sign,power',[(1,0),(-1,0),(1,2),(-1,2)])
def test_complete_actual_columns_sign_and_power_routes(sign,power):
    forms={(0,0):[(4,1,0),(7,-1,1)],(0,1):[(7,-sign,1+power),(4,sign,power)]}
    pool=WorkPool(256_000_000);representatives,mapping=routes(forms,set(forms),pool=pool)
    assert len(representatives)==1
    for key,(rep,s,e) in mapping.items():
        lhs={c:F(n)*F(2)**p for c,n,p in forms[key]}
        rhs={c:F(n)*F(2)**p*s*F(2)**e for c,n,p in forms[rep]}
        assert lhs==rhs
    assert pool.used==2*(32+8*2)


def test_distinct_original_ids_are_not_equal_and_nonunique_forms_remain_independent():
    forms={(0,0):[(0,1,0),(1,-1,0)],(0,1):[(2,1,0),(3,-1,0)],
           (1,0):[(4,1,0),(4,1,0)],(1,1):[(4,1,0),(4,1,0)]}
    representatives,mapping=routes(forms,set(forms),pool=WorkPool(256_000_000))
    assert len(representatives)==4 and all(key==rep for key,(rep,_,_) in mapping.items())


def test_default_off_and_complete_prepaid_key_cap():
    pool=WorkPool(0)
    assert construct(None,None,None,None,None,0,pool=pool) is None and pool.used==0
    with pytest.raises(MemoryError):routes({(0,0):[(0,1,0),(1,1,0)]},{(0,0)},pool=pool)


def test_original_exact_emitter_and_tariffs_are_unchanged():
    from experiments.neural_hz_20260831 import c96_word_row_v1 as old
    from experiments.neural_hz_20260831 import c111_source_first_row_v1 as new
    assert inspect.getsource(old._emit_row)==inspect.getsource(new._emit_row)
    assert inspect.getsource(old._row)==inspect.getsource(new._row)
