"""Ordinary source, exact recurrence, complete rows and inverse qualification."""
from fractions import Fraction as F
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import native_word
from experiments.neural_hz_20260831.c62_precision_plan_v1 import choose as reference_choose,plan
from experiments.neural_hz_20260831.c64_compact_boundary_dp_v1 import choose
from experiments.neural_hz_20260831.c64_gauged_products_v1 import products,gauge_row,gauge_definition
from experiments.neural_hz_20260831.c66_birth_emission_v1 import lift
from experiments.neural_hz_20260831.c62_physical_quotient_v1 import prepare,emit,same_matrix,equal
from experiments.neural_hz_20260831.c62_local_equations_v1 import reconstruct
from experiments.neural_hz_20260831.c17_ownership_audit_v1 import actual_words
from experiments.neural_hz_20260831.test_c61_retained_boundary_v1 import data
from experiments.neural_hz_20260831.test_c62_physical_boundary_v2 import complete,old_fields,residual
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


@pytest.mark.parametrize('kind',['chain','branch','mixed','halves'])
def test_compact_recurrence_matches_independent_dictionary_boundary(kind):
    parents,ratios,external,_,_,_=data(kind);nc=max(parents)+1
    local={v:native_word(float(r)) for v,r in ratios.items()}
    maxima={v:max((abs(native_word(float(c))[0]) for c in cs),default=0) for v,cs in external.items()}
    expected=reference_choose(parents,local,maxima,WorkPool(256_000_000))
    dense=np.zeros(nc,np.uint64)
    for v,b in maxima.items():dense[v]=b
    selected,roots,weights,stats=choose(parents,local,dense,nc,pool=WorkPool(256_000_000))
    assert {v:bool(selected[v]) for v in parents}==expected[0]
    assert {v:int(roots[v]) for v in parents}==expected[1]
    assert {v:weights[v] for v in parents}==expected[2]
    for name in ('optimum','ancestor_states','leaf_states','nonleaf_states'):assert stats[name]==expected[3][name]


def test_keep_on_tie_and_zero_prepaid_budget():
    parents={1:0,2:1};local={1:(3,-2),2:(3,-2)};maxima=np.array([0,0,(1<<50)+1],np.uint64)
    selected,_,_,stats=choose(parents,local,maxima,3,pool=WorkPool(256_000_000))
    assert stats['optimum']==1 and selected.tolist()==[False,False,True]
    with pytest.raises(MemoryError):choose(parents,local,maxima,3,pool=WorkPool(0))


@pytest.mark.parametrize('ratio',[.5,-.75,2.**-10])
@pytest.mark.parametrize('small',[2.**-20,2.**-10])
def test_dominant_source_gauge_matches_full_binary_rhs_oracle(ratio,small):
    original=np.array([small,-.125]);ratios=np.full(2,ratio)
    changed,proof=products(original,ratios,pool=WorkPool(256_000_000))
    assert all(F(float(p))==F(float(a))*F(float(b)) for p,a,b in zip(changed,original,ratios))
    cv=np.r_[changed,1.];bv=np.array([.0625]);rhs=.03125
    oracle=gauge_row(cv,bv,rhs,pool=WorkPool(256_000_000))
    got=gauge_definition(cv,bv,rhs,proof['minimum_absolute_product'],pool=WorkPool(256_000_000))
    assert equal(oracle[0],got[0]) and equal(oracle[1],got[1]) and oracle[2:]==got[2:]


def source_observer(saved,expected):
    """Independent all-physical-row/source-map and generic incidence oracle."""
    def check(encoder,nodes,root,er,es,lr,ls,tracker):
        for name,got in [('eq_roots',er),('eq_scales',es),('ineq_roots',lr),('ineq_scales',ls),('def_rows',encoder.def_rows)]:
            assert np.array_equal(np.asarray(got,np.int64),saved[name])
        for rows,cm,bm,rhs in [(encoder.eq,'Ac','Ab','b'),(encoder.ineq,'Auc','Aub','ub')]:
            matrix=getattr(saved['hz'],cm);binary=getattr(saved['hz'],bm);rvalues=getattr(saved['hz'],rhs)
            assert len(rows)==matrix.shape[0]
            for r,(cc,cv,bc,bv,value) in enumerate(rows):
                a,b=matrix.indptr[r:r+2];c,d=binary.indptr[r:r+2]
                assert np.array_equal(cc,matrix.indices[a:b]) and equal(cv,matrix.data[a:b])
                assert np.array_equal(bc,binary.indices[c:d]) and equal(bv,binary.data[c:d]) and value==rvalues[r]
        assert tracker.parents==expected['parents'] and tracker.local==expected['local']
        assert tracker.tags==expected['tags'] and tracker.numerators==expected['numerators']
        maxima={v:0 for v in tracker.parents};hits=set();own=set(tracker.defining.values())
        for row,(cc,cv,bc,bv,rhs) in enumerate(encoder.eq):
            if row in own:continue
            for position,(c,value) in enumerate(zip(cc,cv)):
                if int(c) in maxima:
                    maxima[int(c)]=max(maxima[int(c)],abs(native_word(value)[0]));hits.add((row,position))
        assert {v:int(tracker.maxima[v]) for v in maxima}==maxima
        assert {(int(r),int(p)) for r,ps in tracker.hits for p in ps}==hits
        for cc,cv,bc,bv,rhs in encoder.ineq:assert not any(int(c) in maxima for c in cc)
    return check


@pytest.mark.parametrize('kind',['chain','shared','conv_disjoint'])
def test_fresh_source_complete_predicates_local_inverse_and_actual_owners(kind):
    _,saved=complete(kind);legacy,_=old_fields(saved);original_sha=source_digest(saved['hz'])
    pool=WorkPool(256_000_000);boundary=plan(saved,pool=pool,enabled=True)
    prepared=prepare(saved,legacy,boundary,pool=pool,enabled=True)
    reference,_=emit(legacy,prepared,pool=pool,enabled=True)
    candidate=lift(saved['expression'],saved['keep'],enabled=True,
        frame_widths=(saved['old_n_cont'],saved['old_n_bin']),before_fold=source_observer(saved,boundary))
    fields=candidate['fields'];hz=fields['hz'];ref=reference['fields'];construction=candidate['construction']
    assert candidate['independent_qualification_pending'] and source_digest(saved['hz'])==original_sha
    for name in ('Gc','Gb','Ac','Ab','Auc','Aub'):assert same_matrix(getattr(hz,name),getattr(ref['hz'],name))
    for name in ('c','b','ub'):assert equal(getattr(hz,name),getattr(ref['hz'],name))
    for name in ('keep','eq_roots','eq_scales','ineq_roots','ineq_scales','def_rows','radix_gauges','owners','uid_slabs'):
        assert equal(fields[name],ref[name])
    expected=actual_words(hz,fields['old_n_cont'],fields['logical_n_cont'],construction['eq_uids'],construction['ineq_uids'])
    assert np.array_equal(expected,fields['owners']) and hz.n_bin==saved['hz'].n_bin
    values=[F((i%5)-2,5) for i in range(hz.n_cont)];binary=[F(-1 if i%2 else 1) for i in range(hz.n_bin)]
    full=reconstruct(values,fields['eq_roots'],fields['eq_scales'],old_n_cont=fields['old_n_cont'],
        old_n_eq=fields['old_n_eq'],n_cont=hz.n_cont,schema=candidate['lineage_schema'])
    for cm,bm,rhs in [('Ac','Ab','b'),('Auc','Aub','ub')]:
        part=prepared['parts'][cm]
        for r,target in enumerate(part['mapping']):
            original=residual(saved['hz'],cm,bm,rhs,r,full,binary)
            if target<0:assert original==0
            else:assert residual(hz,cm,bm,rhs,int(target),values,binary)==original*F(2)**int(part['qrows'][r])
    assert fields['report']['alias_quotient']['identity_sha256']==boundary['report']['identity_sha256']


def test_merging_conv_remains_fail_closed_and_original_is_unchanged():
    _,saved=complete('conv');before=source_digest(saved['hz'])
    with pytest.raises(ValueError,match='coalescence outside complete precision-boundary class'):
        lift(saved['expression'],saved['keep'],enabled=True)
    assert source_digest(saved['hz'])==before


@pytest.mark.parametrize('kwargs',[{'max_work':0},{'max_branch_work':0},{'max_entries':0},
    {'max_work':256_000_001},{'max_branch_work':200_000_001},{'max_entries':64_000_001}])
def test_default_off_and_unchanged_generator_caps(kwargs):
    assert lift(object(),object()) is None
    _,saved=complete('chain')
    with pytest.raises((ValueError,MemoryError)):lift(saved['expression'],saved['keep'],enabled=True,**kwargs)
