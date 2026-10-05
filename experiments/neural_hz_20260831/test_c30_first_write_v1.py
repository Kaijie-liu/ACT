from dataclasses import asdict, replace
from fractions import Fraction as F
import pickle
import numpy as np
import pytest
import scipy.sparse as sp

from experiments.neural_hz_20260831.c30_first_write_v1 import AppendView, splice_append
from experiments.neural_hz_20260831.c30_append_discovery_v1 import discover_append
from experiments.neural_hz_20260831.c28_consumer_discovery_v1 import discover
from experiments.neural_hz_20260831.test_c28_consumer_discovery_v1 import source
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import compile_lineage
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.test_c10_alias_quotient_v1 import feasible


def view_from(c,h):
    p = c.hz
    return AppendView(p,h.Ac[p.n_eq:],h.Ab[p.n_eq:],h.b[p.n_eq:],
        h.Auc[p.n_ineq:],h.Aub[p.n_ineq:],h.ub[p.n_ineq:],h.c,h.Gc,h.Gb)


def full_reference(h, plans):
    # Independent dense row-add/delete fixture oracle, not the segment writer.
    ac, ab, auc, aub = (getattr(h,k).toarray().copy() for k in ('Ac','Ab','Auc','Aub'))
    b, ub = h.b.copy(), h.ub.copy()
    for p in plans:
        m, rhs = (auc,ub) if p.inequality else (ac,b)
        m[p.consumer] += p.sign*ac[p.definition]
        rhs[p.consumer] += p.sign*b[p.definition]
    keep = np.ones(h.n_eq,bool); keep[[p.definition for p in plans]] = False
    return dict(Ac=ac[keep],Ab=ab[keep],Auc=auc,Aub=aub,b=b[keep],ub=ub)


def test_default_off_reads_nothing():
    assert splice_append(object(),object(),pool=object()) is None
    assert discover_append(object(),object(),object(),pool=object()) is None


@pytest.mark.parametrize('mixed',[False,True])
@pytest.mark.parametrize('subtract',[False,True])
@pytest.mark.parametrize('phase',[False,True])
def test_whole_append_discovery_writer_and_fraction_reconstruction(mixed,subtract,phase,monkeypatch):
    c,h,o,eq,le,w,info = source(mixed=mixed,subtract=subtract,phase=phase)
    before = source_digest(h); pre_before = source_digest(c.hz)
    view = view_from(c,h); dp = WorkPool(256_000_000)
    plans,stats = discover_append(c,view,o,pool=dp,enabled=True)
    expected,oldstats = discover(c,h,o,pool=WorkPool(256_000_000),enabled=True)
    assert [asdict(p) for p in plans] == [asdict(p) for p in expected]
    assert all(stats[k] == v for k,v in oldstats.items())
    assert dp.parts['append_consumer_global_row_routing'] == 4*(h.n_eq+h.n_ineq)
    ref = full_reference(h,plans)
    def forbidden(*a,**kw): raise AssertionError('aggregate assembly or densification')
    with monkeypatch.context() as patch:
        patch.setattr(sp,'vstack',forbidden); patch.setattr(sp,'hstack',forbidden)
        patch.setattr(sp.csr_matrix,'toarray',forbidden)
        new,report = splice_append(view,plans,pool=WorkPool(256_000_000),enabled=True)
    for name,want in ref.items():
        got = getattr(new,name)
        if sp.issparse(got):
            assert got.has_canonical_format and not np.any(got.data == 0.)
            got = got.toarray()
        assert np.array_equal(got.view(np.uint64),want.view(np.uint64))
    assert new.Gc is h.Gc and new.Gb is h.Gb and np.shares_memory(new.c,h.c)
    assert new.n_cont == h.n_cont and new.n_bin == h.n_bin and new.frame_id == h.frame_id
    assert report['candidate_native_payload_transfers'] == report['baseline_native_payload_transfers']-5*len(plans)
    expected_parents = sum(int(h.Ac.indptr[p.definition+1]-h.Ac.indptr[p.definition]-1) for p in plans)
    assert sum(v.get('parent_terms_written_once',0) for v in report['matrices'].values()) == expected_parents
    assert sum(v.get('actual_parent_negations',0) for v in report['matrices'].values()) == sum(
        int(h.Ac.indptr[p.definition+1]-h.Ac.indptr[p.definition]-1) for p in plans if p.sign == -1)
    lineage = compile_lineage(c.eq_roots,c.eq_scales,plans,old_n_cont=c.old_n_cont,
        old_n_eq=c.old_n_eq,pool=WorkPool(256_000_000),enabled=True)
    for z in (-1,1):
        for x in (F(-1),F(0),F(1)):
            point = [F(0)]*h.n_cont; point[0],point[1] = F(z),x
            for i,(col,out,d,kind,target,pivot,sign,offset,beta) in enumerate(info):
                prefix = -F(.25)*point[0]-(F(.125)*point[1] if i%2 else 0)
                point[out] = F(.125)+sign*(F(offset)-prefix)-F(beta)*z
            extended = lineage.reconstruct_fraction(new,point,pool=WorkPool(256_000_000))
            assert feasible(new,point,(z,)) and feasible(h,extended,(z,))
    assert source_digest(h) == before and source_digest(c.hz) == pre_before
    assert source_digest(pickle.loads(pickle.dumps(new,protocol=5))) == source_digest(new)


@pytest.mark.parametrize('change',['no_pivot','output_live','binary_definition','lower_consumer_head',
    'extra_old_consumer','empty_consumer_row','longer_tail'])
def test_complete_zero_hit_and_filter_behavior(change):
    c,h,o,*_ = source(change=change)
    v = view_from(c,h)
    plans,_ = discover_append(c,v,o,pool=WorkPool(256_000_000),enabled=True)
    expected,_ = discover(c,h,o,pool=WorkPool(256_000_000),enabled=True)
    assert plans == expected
    result = splice_append(v,plans,pool=WorkPool(256_000_000),enabled=True)
    if not plans: assert result is None
    else:
        for k,w in full_reference(h,plans).items():
            value = getattr(result[0],k)
            assert np.array_equal(value.toarray() if sp.issparse(value) else value,w)


@pytest.mark.parametrize('redirect',[False,True])
def test_mixed_old_new_and_MAIN_uid_transfer(redirect):
    c,h,o,*_ = source(redirect=True) if redirect else source(phase=True,old_consumers=6)
    plans,stats = discover_append(c,view_from(c,h),o,pool=WorkPool(256_000_000),enabled=True)
    expected,_ = discover(c,h,o,pool=WorkPool(256_000_000),enabled=True)
    assert plans == expected
    if redirect: assert all(p.consumer_main is not None for p in plans)
    else: assert stats['selected_new_consumers'] == stats['selected_old_consumers'] == 6


@pytest.mark.parametrize('bad',['duplicate','column','definition','consumer','kind','pivot','sign','offset','tail','dependency','order','budget'])
def test_plan_mutation_and_budget_reject_without_source_write(bad):
    c,h,o,*_ = source(phase=True)
    v = view_from(c,h); plans,_ = discover_append(c,v,o,pool=WorkPool(256_000_000),enabled=True)
    before = source_digest(h); pool = WorkPool(256_000_000)
    if bad == 'duplicate': plans.append(plans[0])
    elif bad == 'dependency': plans[0] = replace(plans[0],consumer=plans[1].definition)
    elif bad == 'order': plans.reverse()
    elif bad == 'budget': pool = WorkPool(400)
    else:
        values = dict(column=0,definition=-1,consumer=h.n_eq,kind=1,pivot=.75,sign=0,
                      offset=.123,tail=(999,))
        plans[0] = replace(plans[0],**{('inequality' if bad == 'kind' else bad):values[bad]})
    with pytest.raises((ValueError,MemoryError)):
        splice_append(v,plans,pool=pool,enabled=True)
    assert source_digest(h) == before


@pytest.mark.parametrize('bad',['dtype','shape','RHS','width','frame','source','base','boundary','budget'])
def test_append_guards(bad):
    c,h,o,*_ = source(phase=True)
    v = view_from(c,h); pool = WorkPool(256_000_000)
    if bad == 'dtype': v = replace(v,eq_c=v.eq_c.astype(np.float32))
    elif bad == 'shape': v = replace(v,eq_b=v.eq_b[:0])
    elif bad == 'RHS': v = replace(v,eq_rhs=v.eq_rhs.reshape(-1,1))
    elif bad == 'width': v = replace(v,Gc=v.Gc[:,:-1])
    elif bad == 'frame': c.hz.exact = False
    elif bad == 'source': v = replace(v,pre=h)
    elif bad == 'base': c.owners = c.owners.copy()
    elif bad == 'boundary': c.report['radix_uid_base'] += 1
    else: pool = WorkPool(0)
    with pytest.raises((ValueError,MemoryError)): discover_append(c,v,o,pool=pool,enabled=True)


def test_inexact_rhs_stops_before_writer():
    c,h,o,*_ = source(change='inexact_RHS')
    with pytest.raises(ValueError,match='RHS'):
        discover_append(c,view_from(c,h),o,pool=WorkPool(256_000_000),enabled=True)


@pytest.mark.parametrize('negative',[False,True])
def test_empty_parent_and_empty_final_continuous_row(negative):
    c,h,o,*_ = source(phase=True)
    # Rebind a toy source with a constant-only definition and no continuous
    # consumer tail. This checks physical edge behavior, not an admission proof.
    view = view_from(c,h)
    plans,_ = discover_append(c,view,o,pool=WorkPool(256_000_000),enabled=True)
    p=plans[0];ac=c.hz.Ac.tolil();ac[p.definition,:]=0.;ac[p.definition,p.column]=p.pivot
    c.hz.Ac=ac.tocsr()
    tail=view.eq_c.tolil();at=p.consumer-c.hz.n_eq
    tail[at,:]=0.;tail[at,p.column]=p.pivot if negative else -p.pivot
    view=replace(view,eq_c=tail.tocsr())
    plans[0]=replace(p,sign=-1 if negative else 1,tail=())
    new,report=splice_append(view,plans,pool=WorkPool(256_000_000),enabled=True)
    target=p.consumer-len(plans)
    assert new.Ac.indptr[target] == new.Ac.indptr[target+1]
    assert new.b[target] == view.eq_rhs[at]+plans[0].sign*p.offset
    assert not np.shares_memory(new.Ac.data,c.hz.Ac.data)


def test_actual_writer_exhaustion_after_plan_checks_still_keeps_source():
    c,h,o,*_=source(phase=True);view=view_from(c,h)
    plans,_=discover_append(c,view,o,pool=WorkPool(256_000_000),enabled=True)
    before=source_digest(h)
    with pytest.raises(MemoryError):splice_append(view,plans,pool=WorkPool(3000),enabled=True)
    assert source_digest(h)==before
