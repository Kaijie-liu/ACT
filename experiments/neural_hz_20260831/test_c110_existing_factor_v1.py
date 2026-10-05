# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ordinary whole-circuit quotient, all-factor inverse and complete costs."""
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c86_complete_tile_hz_v1 import project_outputs
from experiments.neural_hz_20260831.c109_shared_partial_v1 import fixture,MODES,GRIDS
from experiments.neural_hz_20260831.c110_existing_factor_v1 import (
    quotient,prove_and_measure,definition,literal)

CASES = [(c,k,mode,(2,2)) for c,k in ((1,2),(3,4)) for mode in MODES]
CASES += [(1,2,'dense',g) for g in GRIDS]


@pytest.mark.parametrize('channels,outputs,mode,grid',CASES)
def test_complete_spatial_polynomials_all_factor_inverse_and_numeric_cost(channels,outputs,mode,grid):
    pool=WorkPool(256_000_000)
    old,unused,expected,_=fixture(channels,outputs,mode,grid=grid,pool=pool,enabled=True)
    del unused
    report,new=quotient(old,pool=pool,enabled=True)
    report,before,after=prove_and_measure(old,new,expected,report,pool=pool)
    assert report['exact_all_output_polynomials'] and report['exact_all_old_factor_inverse_polynomials']
    assert report['actual_inverse_both_binary_phases'] and report['all_normalized_boxes_proved']
    assert int(before['integrality'].sum()) == int(after['integrality'].sum()) == 1
    assert old['source'] is new['source']
    assert before['old_inverse'] is after['old_inverse']
    removed=4 if channels==3 and mode=='masked' else 128 if channels==3 and mode=='shared' else 0
    assert report['removed_factors']==removed
    assert report['mathematical_numeric_accepted'] == report['strict_complete_numeric_win'] == bool(removed)
    if removed:
        assert after['quotient_inverse'].shape==(removed,3)
        assert report['nnz_delta']<0 and report['byte_delta']<0 and report['entry_delta']<0
    else:
        assert new is old
        assert report['nnz_delta']==report['byte_delta']==report['entry_delta']==0
    assert report['formal_gain']==0 and not report['full_source_LIVE_and_runtime_payment_proved']


def ordinary_affine_problem():
    def row(slot,terms,rhs=0.):
        return dict(slot=slot,coefficients=tuple(sorted([*terms,(slot,1.)])),rhs=rhs,gauge=0)
    aux=[row(5,[(0,-.125),(1,-.125)],.125),
         row(6,[(5,-.5),(4,-.25)]),
         row(7,[(0,.25),(1,.25)],-.25),
         row(8,[(0,-.25),(4,-.25)],.25),
         row(9,[(5,-.5),(7,-.25),(4,-.25)])]
    source=dict(n_cont=5,binary_ids=(0,),centers=np.zeros(2),powers=np.zeros(2,np.int32),
        parent_ids=np.array([0,1],np.int64),weights=np.empty(0,np.float32),bias=np.zeros(1),
        output_ids=np.array([2],np.int64),old_inverse=np.array([[3,0,1,2]],np.int64),
        source_rows=((True,((3,1.),(0,-.5)),(),0.),
                     (True,((0,1.),),((0,-.5),),0.),
                     (False,((1,1.),),((0,.25),),1.5)))
    outputs=[dict(slot=None,coefficients=((2,1.),(6,-.25),(7,-.0625),(9,-.125)),rhs=0.,gauge=0)]
    return dict(source=source,base=5,n_cont=10,aux=aux,outputs=outputs)


def test_proportional_later_representative_nonzero_rhs_and_consumer_collision():
    old=ordinary_affine_problem();pool=WorkPool(256_000_000)
    expected=project_outputs(old['aux'],old['outputs'],old['base'])
    report,new=quotient(old,pool=pool,enabled=True)
    report,_,after=prove_and_measure(old,new,expected,report,pool=pool)
    assert report['removed_factors']==1 and report['strict_complete_numeric_win']
    assert tuple(after['quotient_inverse'][0,:2])==(5,5)
    assert literal(after['quotient_inverse'][0,2])==-.5
    assert new['kept_original_aux']==(7,8,6,9)
    assert new['aux'][-1]['coefficients']==((4,-.25),(8,1.))


@pytest.mark.parametrize('difference',['id','rhs'])
def test_equal_numeric_coefficients_do_not_merge_distinct_ids_or_rhs(difference):
    problem=ordinary_affine_problem()
    first=problem['aux'][0]
    terms=((0,-.125),(4 if difference=='id' else 1,-.125),(8,1.))
    other=dict(slot=8,coefficients=terms,rhs=.25 if difference=='rhs' else .125,gauge=0)
    assert definition(first,5)[1][0] != definition(other,5)[1][0]


def test_source_namespaces_are_not_cached_or_merged_by_value():
    first,second=ordinary_affine_problem(),ordinary_affine_problem()
    assert first['source'] is not second['source']
    pool=WorkPool(256_000_000)
    for old in (first,second):
        report,new=quotient(old,pool=pool,enabled=True)
        assert report['removed_factors']==1 and new['source'] is old['source']


def test_default_off_and_exhausted_work_fail_before_rewrite():
    old=ordinary_affine_problem();pool=WorkPool(0)
    assert quotient(old,pool=pool) is None and pool.used==0
    with pytest.raises(MemoryError):quotient(old,pool=pool,enabled=True)
