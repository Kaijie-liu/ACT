from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import pytest
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c28_consumer_discovery_v1 import discover
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c17_ownership_audit_v1 import actual_words
from experiments.neural_hz_20260831.c22_uid_runs_v1 import pack
from experiments.neural_hz_20260831.c24_uid_slabs_v1 import closed_uid_tables
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import build
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import verify_all_and_discover,BranchPool
from experiments.neural_hz_20260831.c26_transplant_audit_v1 import plans_from_checked_incidence
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.test_c15_unit_row_splice_v1 import fixture


def source(*,mixed=False,subtract=False,phase=False,redirect=False,change=None,old_consumers=0):
    post,kw,info=fixture(count=12,mixed=mixed,subtract=subtract)
    if change=='no_pivot':
        for col,out,d,kind,row,*_ in info:
            m=post.Auc if kind else post.Ac;m.data[m.indptr[row]]*=.5
    elif change=='output_live':
        gc=post.Gc.tolil();gc[0,info[0][0]]=1.;post.Gc=gc.tocsr()
    elif change=='binary_definition':
        ab=post.Ab.tolil();ab[info[0][2],0]=.125;post.Ab=ab.tocsr()
    elif change=='lower_consumer_head':
        ac=post.Ac.tolil();ac[info[0][4],0]=.125;post.Ac=ac.tocsr()
    elif change=='extra_old_consumer':
        extra=sp.csr_matrix(([.125],([0],[info[0][0]])),shape=(1,post.n_cont))
        post.Ac=sp.vstack((post.Ac,extra),format='csr');post.Ab=sp.vstack((post.Ab,sp.csr_matrix((1,1))),format='csr')
        post.b=np.r_[post.b,0.]
    elif change=='empty_consumer_row':
        post.Ac=sp.vstack((post.Ac,sp.csr_matrix((1,post.n_cont))),format='csr')
        post.Ab=sp.vstack((post.Ab,sp.csr_matrix((1,1))),format='csr');post.b=np.r_[post.b,0.]
    elif change=='inexact_RHS':
        post.b[info[0][2]]=2.**-20;post.b[info[0][4]]=2.**40
    elif change=='longer_tail':
        ac=post.Ac.tolil();ac[info[0][4],post.n_cont-1]=.125;post.Ac=ac.tocsr()
    if redirect:
        assert not mixed and not phase and change is None
        kw.update(logical_n_cont=post.n_cont,eq_roots=np.arange(post.n_eq,dtype=np.int64),eq_scales=np.zeros(post.n_eq,np.int64))
    if phase:
        count=len(info)
        assert not old_consumers or not mixed
        stop=count+1+old_consumers
        pre=SparseHZono(post.c,post.Gc,post.Gb,post.Ac[:stop],post.Ab[:stop],post.b[:stop],
            post.Auc[:0],post.Aub[:0],post.ub[:0],frame_id=post.frame_id,exact=True)
    else:pre=post
    main=kw['logical_n_cont']-kw['old_n_cont']
    first=kw['old_n_eq']+pre.n_ineq;radix=first+main
    closed=SimpleNamespace(**kw,hz=pre,ineq_roots=np.arange(pre.n_ineq,dtype=np.int64),
        ineq_scales=np.zeros(pre.n_ineq,np.int64),uid_slabs=np.array([pack(first,0,main)],np.uint64),
        report={'radix_uid_base':radix},validate=lambda:None)
    closed.def_rows=np.arange(kw['old_n_eq']+main,pre.n_eq,dtype=np.int64)
    eq,le=closed_uid_tables(closed)
    closed.owners=actual_words(pre,kw['old_n_cont'],kw['logical_n_cont'],eq,le)
    boundary=radix+16384;ne=post.n_eq-pre.n_eq
    overlay,_=build(closed.owners,[(post.Ac[pre.n_eq:],boundary),(post.Auc[pre.n_ineq:],boundary+ne)],
        old_n_cont=kw['old_n_cont'],old_uid_ceiling=boundary,pool=WorkPool(256_000_000),enabled=True)
    eq=np.r_[eq,np.arange(boundary,boundary+ne,dtype=np.int64)]
    le=np.r_[le,np.arange(boundary+ne,boundary+ne+post.n_ineq-pre.n_ineq,dtype=np.int64)]
    words=actual_words(post,kw['old_n_cont'],kw['logical_n_cont'],eq,le)
    return closed,post,overlay,eq,le,words,info


def reference(closed,post,overlay,eq,le,words):
    p=WorkPool(256_000_000)
    columns,_=verify_all_and_discover(closed,post,overlay,words,eq,le,whole=p,branch=BranchPool(p))
    return plans_from_checked_incidence(closed,post,columns,words,eq,le,pool=p)


def test_default_off_no_source_or_state_read():
    assert discover(object(),object(),object(),pool=object()) is None


@pytest.mark.parametrize('mixed',[False,True])
@pytest.mark.parametrize('subtract',[False,True])
@pytest.mark.parametrize('phase',[False,True])
def test_complete_consumer_direction_matches_independent_factor_direction(mixed,subtract,phase):
    c,h,o,eq,le,w,info=source(mixed=mixed,subtract=subtract,phase=phase)
    before=source_digest(h);p=WorkPool(256_000_000)
    plans,stats=discover(c,h,o,pool=p,enabled=True)
    assert [asdict(v) for v in plans]==[asdict(v) for v in reference(c,h,o,eq,le,w)]
    assert len(plans)==len(info) and stats['all_physical_consumer_rows']==h.n_eq+h.n_ineq
    assert stats['selected_new_consumers' if phase else 'selected_old_consumers']==len(info)
    assert source_digest(h)==before and stats['no_dense_UID_or_MAIN_oracle_built']
    assert 'consumer_row_header' in p.parts and 'independent_complete_incidence' not in p.parts


@pytest.mark.parametrize('subtract',[False,True])
def test_existing_MAIN_consumer_redirects_are_not_lost(subtract):
    c,h,o,eq,le,w,info=source(redirect=True,subtract=subtract)
    plans,_=discover(c,h,o,pool=WorkPool(256_000_000),enabled=True)
    assert [asdict(v) for v in plans]==[asdict(v) for v in reference(c,h,o,eq,le,w)]
    assert all(p.consumer_main is not None for p in plans)


@pytest.mark.parametrize('change',['no_pivot','output_live','binary_definition','lower_consumer_head',
    'extra_old_consumer','empty_consumer_row','longer_tail'])
def test_zero_hit_and_all_necessary_guards_preserve_complete_selection(change):
    c,h,o,eq,le,w,info=source(change=change)
    plans,_=discover(c,h,o,pool=WorkPool(256_000_000),enabled=True)
    assert [asdict(v) for v in plans]==[asdict(v) for v in reference(c,h,o,eq,le,w)]
    if change=='no_pivot':assert not plans
    elif change in {'empty_consumer_row','longer_tail'}:assert len(plans)==12
    else:assert len(plans)==11


def test_inexact_RHS_fails_closed():
    c,h,o,eq,le,w,info=source(change='inexact_RHS')
    with pytest.raises(ValueError,match='RHS'):discover(c,h,o,pool=WorkPool(256_000_000),enabled=True)


@pytest.mark.parametrize('third',[False,True])
def test_old_and_new_consumers_coexist_and_append_can_disqualify_old_pair(third):
    c,h,o,eq,le,w,info=source(phase=True,old_consumers=6)
    if third:
        extra=sp.csr_matrix(([.125],([0],[info[0][0]])),shape=(1,h.n_cont))
        h.Ac=sp.vstack((h.Ac,extra),format='csr');h.Ab=sp.vstack((h.Ab,sp.csr_matrix((1,1))),format='csr');h.b=np.r_[h.b,0.]
        boundary=o.old_uid_ceiling;ne=h.n_eq-c.hz.n_eq
        o,_=build(c.owners,[(h.Ac[c.hz.n_eq:],boundary),(h.Auc[c.hz.n_ineq:],boundary+ne)],
            old_n_cont=c.old_n_cont,old_uid_ceiling=boundary,pool=WorkPool(256_000_000),enabled=True)
        oldeq,oldle=closed_uid_tables(c)
        eq=np.r_[oldeq,np.arange(boundary,boundary+ne,dtype=np.int64)]
        le=oldle;w=actual_words(h,c.old_n_cont,c.logical_n_cont,eq,le)
    plans,stats=discover(c,h,o,pool=WorkPool(256_000_000),enabled=True)
    assert [asdict(v) for v in plans]==[asdict(v) for v in reference(c,h,o,eq,le,w)]
    assert stats['selected_new_consumers']==6 and stats['selected_old_consumers']==6-int(third)


def test_mutually_dependent_selected_definitions_abort_whole_transaction():
    h=SparseHZono(np.zeros(1),sp.csr_matrix(([1.],([0],[3])),shape=(1,4)),sp.csr_matrix((1,1)),
        sp.csr_matrix([[1.,0.,0.,0.],[-.25,1.,0.,0.],[0.,-1.,1.,0.],[0.,0.,-1.,1.]]),
        sp.csr_matrix([[-1.],[0.],[0.],[0.]]),np.zeros(4),sp.csr_matrix((0,4)),sp.csr_matrix((0,1)),np.zeros(0),frame_id=51)
    c=SimpleNamespace(hz=h,old_n_cont=1,logical_n_cont=3,old_n_eq=1,eq_roots=np.arange(3,dtype=np.int64),
        ineq_roots=np.zeros(0,np.int64),def_rows=np.array([3],np.int64),uid_slabs=np.array([pack(1,0,2)],np.uint64),
        report={'radix_uid_base':3})
    c.owners=actual_words(h,1,3,np.arange(4,dtype=np.int64),np.zeros(0,np.int64))
    o,_=build(c.owners,[],old_n_cont=1,old_uid_ceiling=16387,pool=WorkPool(256_000_000),enabled=True)
    with pytest.raises(ValueError,match='simultaneous independent'):
        discover(c,h,o,pool=WorkPool(256_000_000),enabled=True)


@pytest.mark.parametrize('bad',['frame','base_identity','phase_boundary','missing_uid','work'])
def test_source_and_budget_guards_no_partial_return(bad):
    c,h,o,eq,le,w,info=source()
    pool=WorkPool(256_000_000)
    if bad=='frame':h.frame_id+=1;c.hz=SimpleNamespace(**vars(h));c.hz.frame_id-=1
    elif bad=='base_identity':c.owners=c.owners.copy()
    elif bad=='phase_boundary':c.report['radix_uid_base']+=1
    elif bad=='missing_uid':c.uid_slabs=np.empty(0,np.uint64)
    else:pool=WorkPool(0)
    with pytest.raises((ValueError,MemoryError)):discover(c,h,o,pool=pool,enabled=True)
