from fractions import Fraction as F
from itertools import product
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c38_all_consumer_coefficients_v1 import coefficient
from experiments.neural_hz_20260831.c39_half_alias_row_gauge_v1 import row_patch,reconstruct_half,RowRejected,LOW,HIGH


def row():
    return (np.array([0,1,2,3,5,7]),np.array([.25,-.125,.25,.125,.5,.75]),
        np.array([0,2]),np.array([.25,-.5]),.375,[(5,1,1),(7,0,-1)])


def patch(args,**kw):return row_patch(*args,pool=WorkPool(256_000_000),enabled=True,**kw)


def materialize(cc,cv,patch):
    # Independent test-only dense dictionary interpreter, not a real writer.
    result={int(c):2*F(float(v)) for c,v in zip(cc,cv)}
    for c in patch['removed_columns']:del result[int(c)]
    for c,v in zip(patch['parent_columns'],patch['parent_values']):
        if v:result[int(c)]=F(float(v))
        else:result.pop(int(c),None)
    return result


def test_default_off_inspects_nothing():
    assert row_patch(*([object()]*6),pool=object()) is None
    assert reconstruct_half(object(),object(),pool=object()) is None


def test_complete_shared_row_fraction_EQ_INEQ_and_binary_equivalence():
    cc,cv,bc,bv,rhs,aliases=args=row();got=patch(args);after=materialize(cc,cv,got)
    assert got['row_gauge_exponent']==1 and got['rhs']==2*rhs
    assert got['parent_overlaps']==2 and got['binary_nnz']==len(bc)
    assert len(after)==got['new_continuous_nnz'] and got['consumer_nnz_delta']==-2
    grid=[F(-1),F(-1,3),F(0),F(1,3),F(1)]
    for point in product(grid,repeat=4):
        retained=dict(zip((0,1,2,3),point))
        full=reconstruct_half(retained,aliases,pool=WorkPool(256_000_000),enabled=True)
        assert full[5]==full[1]/2 and full[7]==-full[0]/2
        for binary in product((-1,1),repeat=2):
            before=sum(F(float(v))*full[int(c)] for c,v in zip(cc,cv))+sum(F(float(v))*b for v,b in zip(bv,binary))-F(rhs)
            updated=sum(v*retained[c] for c,v in after.items())+sum(2*F(float(v))*b for v,b in zip(bv,binary))-F(got['rhs'])
            assert updated==2*before and (updated==0)==(before==0) and (updated<=0)==(before<=0)
    # Full original system also has TWO two-coefficient defining rows.
    assert len(after)+len(bc)-(len(cc)+len(bc)+2*len(aliases))==-6
    assert not got['complete_HZ_or_lineage_proved'] and got['formal_gain']==0


@pytest.mark.parametrize('sign',[-1,1])
def test_exact_half_window_failure_is_avoided_by_changed_row_not_relaxed_window(sign):
    assert coefficient(LOW,0.,sign/2,pool=WorkPool(256_000_000))[0]==3
    args=(np.array([2,3]),np.array([.25,LOW]),np.array([0]),np.array([.25]),.5,[(3,0,sign)])
    result=patch(args)
    assert result['parent_values'][0]==sign*LOW and result['rhs']==1.
    assert abs(result['parent_values'][0])>=LOW


def test_shared_row_scaled_once_and_source_never_modified():
    args=row();copies=[x.copy() for x in args[:4]];got=patch(args)
    assert got['rhs']==.75 and got['selected_occurrences']==2
    assert np.array_equal(got['parent_columns'],[0,1]) and np.array_equal(got['parent_values'],[-.25,.25])
    for before,after in zip(copies,args[:4]):assert np.array_equal(before,after)


@pytest.mark.parametrize('q,w,want',[(1.,-.5,0.),(-1.,.5,0.),(-HIGH,HIGH,HIGH),(HIGH,-HIGH,-HIGH)])
def test_exact_joint_cancellation_or_large_intermediate(q,w,want):
    got=patch((np.array([0,3]),np.array([w,q]),np.array([],int),np.array([],float),0.,[(3,0,1)]))
    assert got['parent_values'][0]==want and got['exact_cancellations']==int(want==0.)
    assert got['new_continuous_nnz']==int(want!=0.)


@pytest.mark.parametrize('bad,reason',[
    ('q_small','unchanged_projection_operand_window'),('w_small','unchanged_projection_operand_window'),
    ('joint_inexact','inexact_joint_parent'),('joint_high','joint_parent_window'),
    ('other_small','scaled_other_continuous_window'),('other_high','scaled_other_continuous_window'),
    ('binary_small','scaled_binary_window'),('binary_high','scaled_binary_window'),
    ('rhs','scaled_RHS_overflow'),('missing','missing_actual_alias_occurrence'),
    ('nan','noncanonical_or_nonfinite_source_row'),('zero','noncanonical_or_nonfinite_source_row'),
    ('unsorted','noncanonical_or_nonfinite_source_row'),('float32','noncanonical_or_nonfinite_source_row'),
])
def test_fail_closed_including_other_coefficients_binaries_and_RHS(bad,reason):
    cc,cv,bc,bv,rhs,aliases=row()
    if bad=='q_small':cv[4]=LOW/2
    elif bad=='w_small':cv[1]=LOW/2
    elif bad=='joint_inexact':cv[1]=1.;cv[4]=LOW*(1.+2.**-32)
    elif bad=='joint_high':cv[1]=HIGH;cv[4]=HIGH
    elif bad=='other_small':cv[2]=LOW/4
    elif bad=='other_high':cv[2]=HIGH
    elif bad=='binary_small':bv[0]=LOW/4
    elif bad=='binary_high':bv[0]=HIGH
    elif bad=='rhs':rhs=np.finfo(np.float64).max
    elif bad=='missing':aliases=[(6,1,1)]
    elif bad=='nan':cv[0]=np.nan
    elif bad=='zero':bv[0]=0.
    elif bad=='unsorted':cc=cc[::-1]
    elif bad=='float32':cv=cv.astype(np.float32)
    with pytest.raises(RowRejected,match=reason):patch((cc,cv,bc,bv,rhs,aliases))


@pytest.mark.parametrize('aliases',[
    [(5,1,1),(5,0,-1)],[(5,1,1),(7,1,-1)],[(5,1,1),(7,5,1)],
    [(5,6,1)],[(5,1,0)],[(5,1,True)],[(5.5,1,1)],
])
def test_alias_identity_and_parent_independence(aliases):
    cc,cv,bc,bv,rhs,_=row()
    with pytest.raises(RowRejected):patch((cc,cv,bc,bv,rhs,aliases))


def test_no_alias_row_must_not_be_rescaled():
    args=list(row());args[-1]=[]
    with pytest.raises(RowRejected,match='untouched_row'):patch(args)


@pytest.mark.parametrize('bad',['rounded','box','missing','duplicate','internal','existing','bad_sign'])
def test_inverse_rejects_missing_or_inexact_global_latents(bad):
    values={0:F(1,3),1:F(-1,3)};aliases=[(5,1,1),(7,0,-1)]
    if bad=='rounded':values[0]=1./3
    elif bad=='box':values[0]=F(2)
    elif bad=='missing':del values[0]
    elif bad=='duplicate':aliases=[(5,0,1),(7,0,1)]
    elif bad=='internal':aliases=[(5,0,1),(7,5,1)]
    elif bad=='existing':values[5]=F(0)
    else:aliases=[(5,0,0)]
    with pytest.raises(ValueError):reconstruct_half(values,aliases,pool=WorkPool(256_000_000),enabled=True)


def test_all_work_is_precharged_and_no_row_or_witness_fallback():
    with pytest.raises(MemoryError):row_patch(*row(),pool=WorkPool(0),enabled=True)
    with pytest.raises(MemoryError):reconstruct_half({0:F(0)},[(3,0,1)],pool=WorkPool(0),enabled=True)
