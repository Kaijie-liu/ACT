from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import RADIX
from experiments.neural_hz_20260831.c37_definition_inventory_v1 import local_scalar,dependencies,classify,census,TAGS
from experiments.neural_hz_20260831.test_c32_live_splice_v1 import execute


def fixture(constant=False,binary=False,live=False):
    ac=np.zeros((4,7))
    ac[0,[0,2]]=[-.5,1.]
    ac[1,[2,3]]=[-.25,1.]
    ac[2,[3,4]]=[-.5,1.]
    ac[3,[0,1,6]]=[-.25,-.25,1.]
    rhs=np.array([0.,.125,0.,0.])
    if constant:ac[0,0]=0.;rhs[0]=.125
    ab=np.zeros((4,1))
    if binary:ab[1,0]=.25
    gc=np.zeros((1,7));gc[0,4 if live else 0]=1.
    hz=SparseHZono(np.zeros(1),sp.csr_matrix(gc),sp.csr_matrix((1,1)),sp.csr_matrix(ac),sp.csr_matrix(ab),rhs,
        sp.csr_matrix((0,7)),sp.csr_matrix((0,1)),np.zeros(0),frame_id=123,exact=True)
    roots=np.array([0,1,2,-4,3],np.int64)
    state=SimpleNamespace(hz=hz,original_fields=dict(old_n_cont=2,logical_n_cont=7,old_n_eq=0,eq_roots=roots))
    tables=dict(definitions=np.array([0,1,2,-1,3],np.int64))
    words=np.array([sum(RADIX+10+r for r in range(4) if ac[r,col]!=0.) for col in range(2,7)],np.int64)
    return state,tables,words


def test_default_off_never_inspects_input():
    assert census(object(),object(),pool=object()) is None


@pytest.mark.parametrize('a,p,d',[(.5,1.,0.),(-.25,1.,.125),(0.,2.,1.),(.75,1.,.25),(-.5,2.,-.5)])
def test_local_scalar_exact_identity_and_box(a,p,d):
    got=local_scalar(a,p,d,pool=WorkPool(256_000_000))
    assert got['scalar_exact']==1 and got['local_box']==1
    for u in (F(-1),F(-1,3),F(0),F(1,3),F(1)):
        z=F(got['ratio'])*u+F(got['offset'])
        assert F(p)*z+F(a)*u==F(d) and abs(z)<=1


@pytest.mark.parametrize('a,p,d,exact,box',[
    (.75,1.,.5,1,0),(.25,1.5,0.,0,-1),(2.**-21,1.,0.,0,-1),
    (.25,2.**-21,0.,0,-1),(0.,2.**40,2.**-1074,0,1),
    (0.,2.**-20,float.fromhex('0x1.fffffffffffffp+1023'),0,0),
])
def test_scalar_failures_preserve_unknowns_and_no_rounded_offset(a,p,d,exact,box):
    result=local_scalar(a,p,d,pool=WorkPool(256_000_000))
    assert result['scalar_exact']==exact and result['local_box']==box


def test_complete_inventory_separates_zero_consumer_wide_and_local_boxes():
    state,tables,words=fixture()
    report,table=classify(state,state.hz,tables,words,pool=WorkPool(256_000_000))
    assert report['all_MAIN_classified']==5 and report['surviving_definitions']==4
    assert report['full_surviving_degree_histogram']=={'1':2,'2':2}
    assert report['non_degree_two_count']==2 and report['direct_no_consumer_rows']==2
    assert report['no_consumer_local_box_proved']==1 and report['no_consumer_box_unknown']==1
    assert report['local_affine_boxed_scalars']==3
    assert report['local_affine_chain_depth_histogram']=={'1':1,'2':1,'3':1}
    assert report['legacy_aliases_depending_on_local_affine']==1
    assert report['local_affine_with_legacy_alias_children']==1
    assert table['local_box'][-1]==-1 and table['disjoint_nnz_delta'][-1]==-3
    assert not report['actual_nnz_reduction_proved'] and not report['composed_chain_arithmetic_proved']


def test_constant_chain_metadata_does_not_compose_offsets():
    state,tables,words=fixture(constant=True)
    report,table=classify(state,state.hz,tables,words,pool=WorkPool(256_000_000))
    assert report['local_affine_constant_definitions']==1
    assert np.array_equal(table['chain_depth'][:3],[1,2,3])
    assert np.array_equal(table['chain_anchor'][:3],[-1,-1,-1])
    assert table['offset'][1]==.125  # Not the composed value .15625.


def test_binary_definition_is_retained_and_not_called_an_affine_scalar():
    state,tables,words=fixture(binary=True)
    report,table=classify(state,state.hz,tables,words,pool=WorkPool(256_000_000))
    assert report['direct_binary_definitions']==1 and table['local_box'][1]==-1
    assert not table['local_affine'][1] and table['chain_depth'][2]==1
    assert state.hz.n_bin==1 and state.hz.Ab.nnz==1


def test_output_liveness_keeps_box_fact_but_excludes_local_elimination_flag():
    state,tables,words=fixture(live=True)
    report,table=classify(state,state.hz,tables,words,pool=WorkPool(256_000_000))
    assert report['surviving_output_live']==1 and table['local_box'][2]==1
    assert not table['local_affine'][2]


@pytest.mark.parametrize('bad',['degree','definition','legacy_forward','source_size','workcap'])
def test_invalid_inventory_transaction_cannot_publish_partial_table(bad):
    state,tables,words=fixture();pool=WorkPool(256_000_000)
    if bad=='degree':words[3]=RADIX+20
    elif bad=='definition':tables['definitions'][0]=-1
    elif bad=='legacy_forward':state.original_fields['eq_roots'][3]=-7
    elif bad=='source_size':state.original_fields['eq_roots']=np.r_[state.original_fields['eq_roots'],1]
    else:pool=WorkPool(0)
    with pytest.raises((ValueError,MemoryError)):classify(state,state.hz,tables,words,pool=pool)


def test_surviving_parent_reference_to_removed_factor_is_rejected():
    state,tables,words=fixture()
    _,table=classify(state,state.hz,tables,words,pool=WorkPool(256_000_000))
    table['parent'][-1]=5  # Current MAIN6 cannot use already-erased MAIN5.
    with pytest.raises(ValueError,match='already removed'):
        dependencies(table,state.original_fields['eq_roots'],first=2,old_n_eq=0,pool=WorkPool(256_000_000))


@pytest.mark.parametrize('ids',[2,78,1001])
def test_complete_actual_toy_inventory_uses_checked_maps_not_layer_identity(monkeypatch,ids):
    _,runtime,hz,_=execute(monkeypatch,layer_id=ids)
    report,table=census(runtime['lifted'],hz,pool=WorkPool(256_000_000),enabled=True)
    assert report['complete_actual_incidence_equal'] and report['all_source_and_final_bytes_unchanged']
    assert len(table)==report['all_MAIN_classified']
    assert report['surviving_definitions']+report['already_legacy_aliases']+report['already_unit_splices']==len(table)


@pytest.mark.parametrize('bad',['post','source','lineage','receipt','copy','budget'])
def test_corrupt_actual_source_or_proof_does_not_become_inventory_evidence(monkeypatch,bad):
    _,runtime,hz,_=execute(monkeypatch);state=runtime['lifted'];pool=WorkPool(256_000_000)
    if bad=='post':hz.b[0]+=.125
    elif bad=='source':state.original_fields['hz'].b[0]+=.125
    elif bad=='lineage':state.lineage.eq_roots[0]+=1
    elif bad=='receipt':state.receipt=object()
    elif bad=='copy':
        import copy
        hz=copy.copy(hz);hz.Ac=hz.Ac.copy()
    else:pool=WorkPool(0)
    with pytest.raises((ValueError,MemoryError)):census(state,hz,pool=pool,enabled=True)


def test_scalar_precharge_before_exact_arithmetic():
    with pytest.raises(MemoryError):local_scalar(.25,1.,.125,pool=WorkPool(159))


def test_worker_retains_the_complete_strict_restore_ceremony():
    root=Path(__file__).resolve().parent
    expected=(root/'c36_exact_joint_sum_worker_v1.py').read_text().replace(
        'c36_joint_sum_census_v1','c37_definition_inventory_v1').replace(
        'c36_exact_joint_sum_census_20260911_v1','c37_definition_inventory_20260911_v1').replace(
        "event='complete_census_saved'","event='complete_inventory_saved'").replace(
        "independent_pairs=result['simultaneous_independent_pairs']","local_affine_boxed_scalars=result['local_affine_boxed_scalars']")
    assert (root/'c37_definition_inventory_worker_v1.py').read_text()==expected
