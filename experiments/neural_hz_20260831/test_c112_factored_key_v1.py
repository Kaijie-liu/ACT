# SPDX-License-Identifier: AGPL-3.0-or-later
"""Same full literal quotient, once-bound frames and exact stencil templates."""
import inspect
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c85_exact_tile_algebra_v1 import T
from experiments.neural_hz_20260831.c109_shared_partial_v1 import MODES,GRIDS
from experiments.neural_hz_20260831.c111_form_reuse_v1 import routes as old_routes
from experiments.neural_hz_20260831.c112_factored_key_v1 import routes,template
from experiments.neural_hz_20260831.c112_factored_key_fixture_v1 import complete
from experiments.neural_hz_20260831.c112_source_first_row_v1 import construct

CASES=[(c,k,mode,(2,2)) for c,k in ((1,2),(3,4)) for mode in MODES]
CASES += [(1,2,'dense',grid) for grid in GRIDS]


@pytest.mark.parametrize('channels,outputs,mode,grid',CASES)
def test_complete_same_quotient_source_inverse_and_substantive_work_reduction(channels,outputs,mode,grid):
    pool=WorkPool(256_000_000)
    report,before,after,old,new,_=complete(channels,outputs,mode,grid,pool=pool)
    assert report['complete_C111_quotient_unchanged']
    assert report['normalization_reduction_at_least_20_percent'] and report['whole_constructor_work_reduced']
    assert report['exact_all_output_polynomials'] and report['actual_inverse_both_binary_phases']
    assert report['all_new_boxes_redundant'] and report['all_source_EQ_INEQ_retained']
    assert old['source'] is new['source'] and before['old_inverse'] is after['old_inverse']
    assert int(before['integrality'].sum())==int(after['integrality'].sum())==1
    omitted=4 if channels==3 and mode=='masked' else 128 if channels==3 and mode=='shared' else 0
    assert report['omitted_factors']==omitted
    assert report['strict_complete_numeric_win']==bool(omitted)
    assert report['formal_gain']==0 and not report['source_runtime_LIVE_admitted']


@pytest.mark.parametrize('layout',['disjoint','shared','dyadic_shared','partial_overlap'])
def test_exact_same_full_routes_for_distinct_shared_shifted_and_partially_shared_frames(layout):
    ids=np.arange(48,dtype=np.int64).reshape(3,16)
    powers=np.tile(np.arange(16,dtype=np.int32)%3,(3,1))
    if layout in ('shared','dyadic_shared'):ids[1:]=ids[0]
    if layout=='dyadic_shared':powers[1]+=2;powers[2]+=1
    if layout=='partial_overlap':ids[1,:12]=ids[0,:12]
    stencil=np.kron(T,T)
    forms={(c,t):[(int(ids[c,p]),int(stencil[t,p]),int(powers[c,p]))
                 for p in range(16) if stencil[t,p]] for c in range(3) for t in range(16)}
    keep=set(forms);old_pool=WorkPool(256_000_000);new_pool=WorkPool(256_000_000)
    before=old_routes(forms,keep,pool=old_pool)
    after=routes(forms,keep,ids,powers,pool=new_pool)
    assert after==before
    if layout=='partial_overlap':
        assert 0<new_pool.parts['c111_complete_existing_form_keys']<old_pool.used
        assert new_pool.parts['c112_exact_mask_templates']>0
    else:
        assert new_pool.parts.get('c111_complete_existing_form_keys',0)==0
        assert new_pool.used<old_pool.used


def test_every_signed_template_coefficient_matches_the_original_stencil():
    pool=WorkPool(256_000_000);stencil=np.kron(T,T)
    for mask in (0,65535,0xAAAA,0x5555,0x0F0F,0x3333,0x7777,0xEFFF):
        for t,(positive,negative,sign) in enumerate(template(mask,pool=pool)):
            assert positive&negative==0
            for p in range(16):
                actual=sign*((positive>>p&1)-(negative>>p&1))
                assert actual==(int(stencil[t,p]) if mask>>p&1 else 0)
    assert pool.used==8*(32+8*16)


def test_default_off_and_frame_binding_prepaid_before_work():
    pool=WorkPool(0)
    assert construct(None,None,None,None,None,0,pool=pool) is None and pool.used==0
    with pytest.raises(MemoryError):
        routes({(0,0):[(0,1,0),(1,1,0)]},{(0,0)},np.arange(16).reshape(1,16),
               np.zeros((1,16),np.int32),pool=pool)


def test_original_emitter_and_old_native_tariffs_are_unchanged():
    from experiments.neural_hz_20260831 import c96_word_row_v1 as old
    from experiments.neural_hz_20260831 import c112_source_first_row_v1 as new
    assert inspect.getsource(old._emit_row)==inspect.getsource(new._emit_row)
    assert inspect.getsource(old._row)==inspect.getsource(new._row)
