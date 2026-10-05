from fractions import Fraction as F
from types import SimpleNamespace
import numpy as np
import pytest
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import RADIX
from experiments.neural_hz_20260831.c35_current_uid_census_v1 import current_tables,checked_incidence
from experiments.neural_hz_20260831.c35_nonunit_exact_census_v1 import prove,census,classify,independent_mask,DTYPE,CODE,GUARDS
from experiments.neural_hz_20260831.test_c32_live_splice_v1 import execute


def test_default_off():
    assert census(object(),object(),pool=object()) is None


@pytest.mark.parametrize('q',[-.5,.5,-2.,2.,-1.,1.,-.75,.75])
def test_exact_nonunit_affine_identity_and_box(q):
    values=np.array([.25,-.125]);p=1.;d=.125;c=.25
    proof=prove(values,p,q,d,c,pool=WorkPool(256_000_000))
    assert proof['reason']==CODE['individual_exact']
    assert all(proof[g]==1 for g in GUARDS)
    r=F(proof['ratio'])
    for x in (F(-1),F(0),F(1,3),F(1)):
        for y in (F(-1),F(1,2),F(1)):
            z=(F(d)-F(.25)*x+F(.125)*y)/F(p)
            assert abs(z)<=1
            for tail in (F(-1),F(0),F(1)):
                for binary in (F(-1),F(1)):
                    old=F(q)*z+F(.5)*tail+F(.75)*binary-F(c)
                    new=r*F(.25)*x-r*F(.125)*y+F(.5)*tail+F(.75)*binary-(F(c)+r*F(d))
                    assert old==new
                    assert (old<=0)==(new<=0)


@pytest.mark.parametrize('name,values,p,q,d,c,reason',[
    ('pivot',[.25],1.5,-.5,0.,0.,'pivot_window'),
    ('pivot_low',[.25],2.**-21,-.5,0.,0.,'pivot_window'),
    ('q_window',[.25],1.,2.**-21,0.,0.,'operand_window'),
    ('parent_window',[2.**-21],1.,-.5,0.,0.,'operand_window'),
    ('inexact',[1.+2.**-52],1.,-(1.+2.**-51),0.,0.,'coefficient_product_inexact'),
    ('result_window',[2.**-20],1.,-.5,0.,0.,'coefficient_result_window'),
    ('offset',[.25],1.,-(1.+2.**-51),1.+2.**-52,0.,'offset_product_inexact'),
    ('rhs',[.25],1.,-1.,2.**-53,1.,'rhs_inexact'),
    ('box',[.75,.5],1.,-.5,0.,0.,'nonredundant_box'),
])
def test_each_numeric_boundary_is_fail_closed(name,values,p,q,d,c,reason):
    got=prove(np.array(values),p,q,d,c,pool=WorkPool(256_000_000))
    assert got['reason']==CODE[reason]
    if reason!='nonredundant_box':assert got['redundant_box']==-1


def test_short_circuit_does_not_read_failed_tail_or_mark_it_proved():
    values=np.r_[np.full(8,1.+2.**-52),np.nan]
    got=prove(values,1.,-(1.+2.**-51),0.,0.,pool=WorkPool(256_000_000))
    assert got['reason']==CODE['coefficient_product_inexact'] and got['checked']==8
    assert got['operand_window']==-1 and got['redundant_box']==-1


def test_precharge_failure_no_result():
    with pytest.raises(MemoryError):prove(np.array([.25]),1.,-.5,0.,0.,pool=WorkPool(0))


def test_exact_nonunit_guard_against_independent_fraction_all_products():
    rng=np.random.default_rng(3509)
    for _ in range(100):
        # Ordinary finite dyadics only, no actual target data or target ids.
        bits=int(rng.choice([12,24,48]))
        values=rng.integers(1,2**bits,size=3).astype(np.float64)*2.**(-bits-4)
        q=-float(rng.integers(1,2**min(bits,48)))*2.**(-min(bits,48))
        got=prove(values,1.,q,0.,0.,pool=WorkPool(256_000_000))
        exact=all(F(float(v*(-q)))==F(float(v))*F(-q) for v in values)
        window=all(2.**-20<=abs(v*(-q))<=2.**40 for v in values)
        operand=all(2.**-20<=abs(v)<=2.**40 for v in values) and abs(q)>=2.**-20
        assert (got['reason']==CODE['individual_exact'])==(exact and window and operand)


@pytest.mark.parametrize('ids',[7,78,199])
def test_complete_current_mapping_and_incidence_after_real_toy_splice(monkeypatch,ids):
    _,runtime,hz,_=execute(monkeypatch,layer_id=ids)
    state=runtime['lifted'];before=state.validate()
    pool=WorkPool(256_000_000)
    tables,report=current_tables(state,pool=pool)
    words=checked_incidence(state,tables,pool=pool)
    assert report['all_current_physical_UIDs']==hz.n_eq+hz.n_ineq
    assert len(words)==state.original_fields['logical_n_cont']-state.original_fields['old_n_cont']
    output,table=census(state,hz,pool=WorkPool(256_000_000),enabled=True)
    assert output['complete_actual_incidence_equal'] and len(table)==len(words)
    assert table['reason'].min()>=0
    assert state.validate()['complete_new_HZ_sha256']==before['complete_new_HZ_sha256']


@pytest.mark.parametrize('bad',['post','source','lineage','receipt','final_copy','workcap'])
def test_census_rejects_corrupted_or_incomplete_bound_transaction(monkeypatch,bad):
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


def test_stale_consumer_UIDs_cannot_pass_complete_incidence(monkeypatch):
    _,runtime,_,_=execute(monkeypatch);state=runtime['lifted'];pool=WorkPool(256_000_000)
    assert len(state.lineage.retired)>0
    monkeypatch.setattr(type(state.lineage),'retired_to',lambda self,uid,*,pool:None)
    tables,_=current_tables(state,pool=pool)
    with pytest.raises(ValueError,match='incidence'):
        checked_incidence(state,tables,pool=pool)


def test_branch_cannot_hide_work_in_an_independent_allowance(monkeypatch):
    from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
    _,runtime,hz,_=execute(monkeypatch)
    with pytest.raises(ValueError,match='same whole'):
        census(runtime['lifted'],hz,pool=WorkPool(256_000_000),
            branch=BranchPool(WorkPool(256_000_000)),enabled=True)


def simple(kind=False):
    d=np.array([[.25,-.125,1.,0.]])
    c=np.array([[0.,0.,-.5,.25]])
    ac=d if kind else np.r_[d,c]
    auc=c if kind else np.zeros((0,4))
    ab=np.array([[0.]]) if kind else np.array([[0.],[.5]])
    aub=np.array([[.5]]) if kind else np.zeros((0,1))
    hz=SparseHZono(np.zeros(1),sp.csr_matrix([[0.,0.,0.,1.]]),sp.csr_matrix((1,1)),
        sp.csr_matrix(ac),sp.csr_matrix(ab),np.array([.125]) if kind else np.array([.125,.25]),
        sp.csr_matrix(auc),sp.csr_matrix(aub),np.array([.25]) if kind else np.zeros(0),frame_id=123,exact=True)
    state=SimpleNamespace(hz=hz,original_fields=dict(old_n_cont=2,logical_n_cont=3,old_n_eq=0,
        eq_roots=np.array([0],np.int64)))
    tables=dict(definitions=np.array([0]),eq=np.array([2]) if kind else np.array([2,3]),
        lookup={2:(False,0),3:(kind,0 if kind else 1)})
    return state,tables,np.array([2*RADIX+5],np.int64)


@pytest.mark.parametrize('kind',[False,True])
def test_complete_diagnostic_covers_EQ_and_LE_same_rule(kind):
    state,tables,words=simple(kind)
    report,table=classify(state,state.hz,tables,words,pool=WorkPool(256_000_000))
    assert report['nonunit_independent_pairs']==1 and table['ratio'][0]==.5
    assert report['potential_predicate_nnz_delta']==-2
    assert not report['new_multiplier_lineage_proved'] and not report['runtime_payment_proved']


@pytest.mark.parametrize('bad',['live','degree','missing','binary','head','direct'])
def test_structural_rejections_separate_from_numeric_proofs(bad):
    state,tables,words=simple()
    if bad=='live':state.hz.Gc=sp.csr_matrix([[0.,0.,1.,0.]])
    elif bad=='degree':words[0]+=RADIX+7
    elif bad=='missing':tables['definitions'][0]=-1
    elif bad=='binary':state.hz.Ab=sp.csr_matrix([[.25],[.5]])
    elif bad=='head':state.hz.Ac=sp.csr_matrix([[.25,-.125,1.,0.],[.25,0.,-.5,.25]])
    else:state.hz.Ac=sp.csr_matrix([[.25,-.125,1.,.25],[0.,0.,-.5,.25]])
    names=dict(live='output_live',degree='not_degree_two',missing='missing_definition',binary='binary_definition',
        head='consumer_not_head',direct='not_direct_definition')
    _,table=classify(state,state.hz,tables,words,pool=WorkPool(256_000_000))
    assert table['reason'][0]==CODE[names[bad]]
    assert all(table[g][0]==-1 for g in GUARDS)


@pytest.mark.parametrize('conflict',['none','chain','shared_consumer','shared_definition','shared_column'])
def test_dependency_filter_rejects_both_endpoints_not_greedy_subset(conflict):
    table=np.zeros(2,DTYPE);table['reason']=CODE['individual_exact']
    table['column']=[10,20];table['definition']=[0,1];table['consumer']=[2,3]
    if conflict=='chain':table['consumer'][0]=1
    elif conflict=='shared_consumer':table['consumer'][1]=2
    elif conflict=='shared_definition':table['definition'][1]=0
    elif conflict=='shared_column':table['column'][1]=10
    rejected=independent_mask(table,pool=WorkPool(256_000_000))
    assert rejected==(0 if conflict=='none' else 2)
    assert table['independent'].sum()==(2 if conflict=='none' else 0)
