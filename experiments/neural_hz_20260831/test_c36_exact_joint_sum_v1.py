from fractions import Fraction as F
from pathlib import Path
import numpy as np
import pytest
import scipy.sparse as sp
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c36_exact_joint_sum_v1 import prove,exact_float,CODE,GUARDS
from experiments.neural_hz_20260831.c36_joint_sum_census_v1 import classify,census,DTYPE
from experiments.neural_hz_20260831.c35_nonunit_exact_census_v1 import independent_mask
from experiments.neural_hz_20260831.test_c35_current_nonunit_census_v1 import simple
from experiments.neural_hz_20260831.test_c32_live_splice_v1 import execute


def pair(values=(.25,),tail=(.5,),p=1.,r=.5,d=0.,c=0.):
    pv=np.array(values,np.float64);pc=np.arange(len(pv),dtype=np.int32)
    tc=np.r_[pc,len(pc)].astype(np.int32);tv=np.r_[tail,-p*r].astype(np.float64)
    return pc,pv,tc,tv,len(pc),p,-p*r,d,c


def test_default_off_no_access():
    assert census(object(),object(),pool=object()) is None


def test_actual_joint_can_be_exact_when_intermediate_product_is_inexact():
    a=1.+2.**-27;r=1.+2.**-26;t=1.-2.**-53
    args=pair((a,),(t,),p=2.,r=r)
    result=prove(*args,pool=WorkPool(256_000_000))
    assert result['reason']==CODE['individual_exact'] and result['old_products_exact']==0
    assert result['overlap_present']==1 and result['nnz_delta']==-3
    assert all(result[g]==1 for g in GUARDS)
    assert not exact_float(F(a)*F(r))[0]
    joint=F(a)*F(r)+F(t)
    assert exact_float(joint)==(True,float.fromhex('0x1.0000003000000p+1'))
    for x in (F(-1),F(-1,3),F(0),F(1,3),F(1)):
        z=-F(a)*x/F(2.)
        for binary in (F(-1),F(1)):
            old=F(-2*r)*z+F(t)*x+F(.25)*binary
            new=joint*x+F(.25)*binary
            assert abs(z)<=1 and old==new and (old<=0)==(new<=0)


@pytest.mark.parametrize('r',[-2.,-.75,-.5,.5,.75,1.,2.])
def test_general_joint_same_identity_for_equalities_and_inequalities(r):
    args=pair((.25,-.125),(.5,.25),p=1.,r=r,d=.125,c=.25)
    got=prove(*args,pool=WorkPool(256_000_000));assert got['reason']==0
    for x in (F(-1),F(0),F(1,3),F(1)):
        for y in (F(-1),F(0),F(1,2),F(1)):
            z=F(.125)-F(.25)*x+F(.125)*y
            lhs=F(-r)*z+F(.5)*x+F(.25)*y-F(.25)
            rhs=(F(r)*F(.25)+F(.5))*x+(-F(r)*F(.125)+F(.25))*y-(F(.25)+F(r)*F(.125))
            assert abs(z)<=1 and lhs==rhs and (lhs<=0)==(rhs<=0)


def test_exact_joint_RHS_does_not_require_representable_intermediate_offset():
    r=1.+2.**-26;d=1.+2.**-27;c=1.-2.**-53
    got=prove(*pair((.25,),(.5,),p=4.,r=r,d=d,c=c),pool=WorkPool(256_000_000))
    assert got['reason']==0 and got['rhs_exact']==1
    assert not exact_float(F(r)*F(d))[0] and exact_float(F(c)+F(r)*F(d))[0]


def test_exact_zero_is_removed_not_a_tolerance_or_underflow():
    got=prove(*pair((.5,),(-.5,),r=1.),pool=WorkPool(256_000_000))
    assert got['reason']==0 and got['cancellations']==1 and got['nnz_delta']==-4
    got=prove(*pair((.5,),(-.5+2.**-40,),r=1.),pool=WorkPool(256_000_000))
    assert got['reason']==CODE['joint_coefficient_window'] and got['cancellations']==0


@pytest.mark.parametrize('kind',['pivot','pivot_low','q','parent','consumer','rhs','joint','window','box'])
def test_all_necessary_failures_and_unknown_guard_preservation(kind):
    args=list(pair())
    if kind=='pivot':args[5]=1.5
    elif kind=='pivot_low':args[5]=2.**-21
    elif kind=='q':args[6]=2.**-21
    elif kind=='parent':args[1][0]=2.**-21
    elif kind=='consumer':args[3][0]=2.**-21
    elif kind=='rhs':args[7]=2.**-53;args[8]=1.;args[6]=-1.
    elif kind=='joint':args=list(pair((1.+2.**-52,),(1.,),p=2.,r=1.+2.**-51))
    elif kind=='window':args=list(pair((.5,),(-.5+2.**-40,),r=1.))
    else:args=list(pair((.75,.5),(.5,.25),p=1.))
    got=prove(*args,pool=WorkPool(256_000_000))
    name=dict(pivot='pivot_window',pivot_low='pivot_window',q='operand_window',parent='operand_window',
        consumer='operand_window',rhs='rhs_inexact',joint='joint_coefficient_inexact',
        window='joint_coefficient_window',box='nonredundant_box')[kind]
    assert got['reason']==CODE[name]
    if kind!='box':assert got['redundant_box']==-1
    assert got['old_products_exact']==-1


def test_failed_first_parent_does_not_claim_no_overlap_in_unvisited_suffix():
    pc=np.array([0,1]);pv=np.array([1.+2.**-52,.25])
    tc=np.array([1,2]);tv=np.array([.5,-(1.+2.**-51)])
    got=prove(pc,pv,tc,tv,1,1.,float(tv[-1]),0.,0.,pool=WorkPool(256_000_000))
    assert got['reason']==CODE['joint_coefficient_inexact'] and got['checked']==1
    assert got['overlap_present']==-1 and got['overlaps']==0
    assert got['joint_exact']==0 and got['operand_window']==-1


def test_nonhead_without_actual_overlap_is_proved_only_after_complete_scan():
    pc=np.array([0]);pv=np.array([.25]);tc=np.array([1,2]);tv=np.array([.5,-.5])
    got=prove(pc,pv,tc,tv,1,1.,-.5,0.,0.,pool=WorkPool(256_000_000))
    assert got['reason']==0 and got['overlap_present']==0


def test_scalar_joint_guard_precharged_before_any_product():
    with pytest.raises(MemoryError):prove(*pair(),pool=WorkPool(160+20+127))


def test_scalar_joint_matches_independent_exact_fraction_population():
    rng=np.random.default_rng(3609)
    for _ in range(160):
        bits=int(rng.choice([12,24,48]))
        vals=rng.integers(1,2**bits,size=2).astype(np.float64)*2.**(-bits-4)
        tail=rng.integers(-2**bits,2**bits,size=2).astype(np.float64)*2.**(-bits-2)
        ratio=float(rng.integers(1,2**bits))*2.**-bits
        args=pair(vals,tail,p=1.,r=ratio)
        result=prove(*args,pool=WorkPool(256_000_000))
        exact=[F(ratio)*F(float(a))+F(float(t)) for a,t in zip(vals,tail)]
        good=all(v==F(float(v)) and (not v or 2.**-20<=abs(float(v))<=2.**40) for v in exact)
        operands=all(2.**-20<=abs(float(v))<=2.**40 for v in (*vals,*tail,ratio))
        assert (result['reason']==0)==(good and operands)


@pytest.mark.parametrize('kind',[False,True])
def test_complete_shape_rule_finds_head_and_nonhead_EQ_LE_pairs(kind):
    state,tables,words=simple(kind)
    # Same original input/global identities; this is a synthetic DIAGNOSTIC fixture.
    if kind:state.hz.Auc=sp.csr_matrix([[.5,0.,-.5,.25]])
    else:state.hz.Ac=sp.csr_matrix([[.25,-.125,1.,0.],[.5,0.,-.5,.25]])
    report,table=classify(state,state.hz,tables,words,pool=WorkPool(256_000_000))
    assert report['simultaneous_independent_pairs']==1 and report['numeric_nonhead_pairs']==1
    assert report['nonhead_overlap_present_proved']==1 and report['potential_predicate_nnz_delta']==-3
    assert not report['runtime_payment_proved'] and not report['new_joint_lineage_proved']


@pytest.mark.parametrize('ids',[5,78,207])
def test_complete_actual_toy_source_UID_incidence_binding_without_id_rule(monkeypatch,ids):
    _,runtime,hz,_=execute(monkeypatch,layer_id=ids)
    result,table=census(runtime['lifted'],hz,pool=WorkPool(256_000_000),enabled=True)
    assert result['complete_actual_incidence_equal'] and result['all_source_and_final_bytes_unchanged']
    assert len(table)==result['all_MAIN_classified'] and np.all(table['reason']>=0)


@pytest.mark.parametrize('bad',['post','source','lineage','receipt','final_copy','workcap'])
def test_invalid_source_cannot_become_a_joint_sum_proof(monkeypatch,bad):
    _,runtime,hz,_=execute(monkeypatch);state=runtime['lifted'];pool=WorkPool(256_000_000)
    if bad=='post':hz.b[0]+=.125
    elif bad=='source':state.original_fields['hz'].b[0]+=.125
    elif bad=='lineage':state.lineage.eq_roots[0]+=1
    elif bad=='receipt':state.receipt=object()
    elif bad=='final_copy':
        import copy
        hz=copy.copy(hz);hz.Ac=hz.Ac.copy()
    else:pool=WorkPool(0)
    with pytest.raises((ValueError,MemoryError)):census(state,hz,pool=pool,enabled=True)


@pytest.mark.parametrize('conflict',['chain','same_consumer','same_definition'])
def test_simultaneous_joint_population_rejects_both_dependency_endpoints(conflict):
    table=np.zeros(2,DTYPE);table['reason']=0;table['column']=[4,7]
    table['definition']=[0,1];table['consumer']=[2,3]
    if conflict=='chain':table['consumer'][0]=1
    elif conflict=='same_consumer':table['consumer'][1]=2
    else:table['definition'][1]=0
    assert independent_mask(table,pool=WorkPool(256_000_000))==2
    assert not table['independent'].any()


def test_worker_is_only_census_module_and_target_change_from_strict_C35_ceremony():
    root=Path(__file__).resolve().parent
    old=(root/'c35_current_nonunit_worker_v1.py').read_text()
    expected=old.replace('c35_nonunit_exact_census_v1','c36_joint_sum_census_v1').replace(
        'c35_current_nonunit_census_20260911_v1','c36_exact_joint_sum_census_20260911_v1')
    assert (root/'c36_exact_joint_sum_worker_v1.py').read_text()==expected
