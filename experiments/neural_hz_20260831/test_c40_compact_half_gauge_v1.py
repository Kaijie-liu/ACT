from fractions import Fraction as F
from types import SimpleNamespace
import copy
import numpy as np
import pytest
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import incidence_oracle
from experiments.neural_hz_20260831.c38_all_consumer_coefficients_v1 import discover
from experiments.neural_hz_20260831.c39_half_alias_census_v1 import select_halves,audit_rows
from experiments.neural_hz_20260831.c40_compact_half_gauge_v1 import materialize,pack_run,unpack_run,DESC,schedules
from experiments.neural_hz_20260831.c40_half_gauge_inverse_v1 import inverse_hashes
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


def fixture(binary=True,small=False):
    q=2.**-20 if small else .75
    ac=np.array([[-.5,0.,0.,1.,0.],[0.,.5,0.,0.,1.],[0.,0.,.25,q,.5],[0.,0.,.5,-.5,q]])
    ab=np.zeros((4,1));ab[3,0]=.125 if binary else 0.
    auc=np.array([[0.,0.,.25,1.,-1.]]);aub=np.array([[.25 if binary else 0.]])
    gc=sp.csr_matrix([[0.,0.,1.,0.,0.]])
    hz=SparseHZono(np.array([0.]),gc,sp.csr_matrix((1,1)),sp.csr_matrix(ac),sp.csr_matrix(ab),
        np.array([0.,-0.,.25,-.5]),sp.csr_matrix(auc),sp.csr_matrix(aub),np.array([.5]),frame_id=123,exact=True)
    final=SparseHZono(np.array([.125]),gc*2,sp.csr_matrix((1,1)),hz.Ac,hz.Ab,hz.b,hz.Auc,hz.Aub,hz.ub,frame_id=123,exact=True)
    state=SimpleNamespace(hz=hz,original_fields=dict(old_n_cont=3,logical_n_cont=5,old_n_eq=0,eq_roots=np.array([0,1],np.int64)))
    tables=dict(definitions=np.array([0,1],np.int64),eq=np.array([10,11,12,13]),le=np.array([21]))
    pool=WorkPool(256_000_000);words=incidence_oracle(hz,tables['eq'],tables['le'],3,5,pool=pool)
    cohort,_=discover(state,final,tables,words,pool=pool);half,_=select_halves(cohort,pool=pool)
    report,rows=audit_rows(hz,tables,words,half,old_nc=3,logical_nc=5,pool=pool,branch=pool)
    assert report['all_joint_row_arithmetic_proved']
    return hz,final,half,rows


def actual(binary=True,small=False):
    hz,final,half,rows=fixture(binary,small);pool=WorkPool(256_000_000)
    out,end,desc,runs,report=materialize(hz,final,half,rows,pool=pool,enabled=True)
    bycol={int(t['column']):int(t['degree']) for t in half}
    degree=np.array([bycol[int(t['column'])] for t in desc])
    inv=inverse_hashes(out,end,desc,runs,degree,pool=pool,enabled=True)
    return hz,final,half,rows,out,end,desc,runs,report,inv,pool,degree


def test_default_off_does_not_inspect_any_input():
    assert materialize(*([object()]*4),pool=object()) is None
    assert inverse_hashes(*([object()]*5),pool=object()) is None


@pytest.mark.parametrize('binary',[False,True])
@pytest.mark.parametrize('small',[False,True])
def test_actual_new_HZ_complete_inverse_and_independent_dense_Fraction_oracle(binary,small):
    hz,final,half,rows,out,end,desc,runs,report,inv,pool,degree=actual(binary,small)
    assert inv['complete_original_post_sha256']==source_digest(hz)
    assert inv['complete_original_final_sha256']==source_digest(final)
    assert out.n_eq==hz.n_eq-2 and out.n_cont==hz.n_cont and out.n_bin==hz.n_bin==1
    assert report['actual_predicate_nnz_delta']==-4 and desc.dtype==DESC
    assert desc.nbytes==30 and runs.nbytes==16 and len(runs)==2
    for name in ('Ac','Ab','Auc','Aub'):assert getattr(out,name) is getattr(end,name)
    assert out.Gc is hz.Gc and end.Gc is final.Gc
    assert inv['all_original_child_occurrences_reconstructed'] and not inv['complete_old_HZ_materialized']
    for name in ('Ac','Ab','Auc','Aub'):
        src=getattr(hz,name).toarray();want=[];is_c=name in ('Ac','Auc');is_le=name.startswith('Au')
        for row in range(src.shape[0]):
            if not is_le and row in (0,1):continue
            values=[2*F(float(v)) for v in src[row]]
            if is_c:
                for t in desc:
                    col,parent,sign=map(int,(t['column'],t['parent'],t['sign']))
                    values[parent]+=sign*F(float(src[row,col]));values[col]=F(0)
            want.append([float(v) for v in values])
        assert np.array_equal(getattr(out,name).toarray(),np.asarray(want))
    assert np.array_equal(out.b,hz.b[2:]*2) and np.array_equal(out.ub,hz.ub*2)
    assert bool(desc['negative_zero'][1]) and pool.used>0


def test_zero_binary_payload_reuses_owners_and_inverse_pointer_segments():
    *_,report,inv,pool,degree=actual(False)
    assert report['matrices']['Ab']['binary_data_indices_shared']
    assert pool.parts['half_inverse_binary_pointer_segments']>0
    assert pool.parts['half_inverse_binary_zero_interval_proof']>0


@pytest.mark.parametrize('kind,start,length',[(False,0,1),(True,0,1<<20),(False,(1<<20)-1,1),(True,19,300)])
def test_lossless_compact_row_intervals(kind,start,length):
    assert unpack_run(pack_run(kind,start,length))==(kind,start,length)


@pytest.mark.parametrize('args',[(False,0,0),(False,-1,1),(False,(1<<20)-1,2),(1,0,1),(False,1.5,1)])
def test_interval_guard(args):
    with pytest.raises(ValueError):pack_run(*args)


@pytest.mark.parametrize('runs',[
    [pack_run(False,1,3),pack_run(False,4,1)],
    [pack_run(False,1,3),pack_run(False,2,1)],
    [pack_run(True,0,1),pack_run(False,1,1)],
    [np.uint64(1<<41)],
])
def test_overlapping_adjacent_unsorted_or_reserved_intervals_fail(runs):
    with pytest.raises(ValueError):schedules(np.array(runs,np.uint64),pool=WorkPool(256_000_000))


@pytest.mark.parametrize('bad',['Ac_value','Ac_column','b','Aub','Gc','sign','negative_zero'])
def test_any_changed_reconstructed_original_byte_cannot_match_bound_source(bad):
    hz,final,half,rows,out,end,desc,runs,report,inv,pool,degree=actual()
    expected=source_digest(hz)
    if bad=='Ac_value':out.Ac.data[0]+=2.**-20
    elif bad=='Ac_column':out.Ac.indices[0]=2
    elif bad=='b':out.b[0]+=.125
    elif bad=='Aub':out.Aub.data[0]+=.125
    elif bad=='Gc':out.Gc.data[0]+=.125
    elif bad=='sign':desc['sign'][0]*=-1
    else:desc['negative_zero'][1]=False
    try:got=inverse_hashes(out,end,desc,runs,degree,pool=WorkPool(256_000_000),enabled=True)
    except (ValueError,IndexError):return
    assert got['complete_original_post_sha256']!=expected


@pytest.mark.parametrize('bad',['degree','definition','power','duplicate_parent','remaining_child'])
def test_inverse_completeness_and_descriptor_guards(bad):
    hz,final,half,rows,out,end,desc,runs,report,inv,pool,degree=actual()
    if bad=='degree':degree[0]+=1
    elif bad=='definition':desc['definition'][1]=desc['definition'][0]
    elif bad=='power':desc['power'][0]=61
    elif bad=='duplicate_parent':desc['parent'][1]=desc['parent'][0]
    else:out.Ac.indices[0]=desc['column'][0]
    with pytest.raises(ValueError):inverse_hashes(out,end,desc,runs,degree,pool=WorkPool(256_000_000),enabled=True)


def test_global_parent_isolation_cannot_be_inferred_from_own_parent_checks():
    hz,final,half,rows=fixture()
    # Each row must exclude ALL selected parents, not only the parent of a
    # child present in that row. The simple inverse otherwise is ambiguous.
    a,b=hz.Ac.indptr[2:4];hz.Ac.indices[a]=1
    with pytest.raises(ValueError,match='GLOBAL'):
        materialize(hz,final,half,rows,pool=WorkPool(256_000_000),enabled=True)


def test_budget_failure_before_construction_leaves_source_bytes_unchanged():
    hz,final,half,rows=fixture();before=source_digest(hz)
    with pytest.raises(MemoryError):materialize(hz,final,half,rows,pool=WorkPool(0),enabled=True)
    assert source_digest(hz)==before
