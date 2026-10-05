import copy
import numpy as np
import pytest
from experiments.neural_hz_20260831.c48_empty_alias_span_v1 import index
from experiments.neural_hz_20260831.c48_alias_span_census_v1 import assess
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.test_c24_dense_closed_v1 import fixture
from experiments.neural_hz_20260831.c20_product_census_v1 import census as original_census


def test_default_off_no_source_access():
    assert index(object(),pool=object()) is None
    assert assess(object(),old_nc=object(),logical_nc=object(),old_eq=object(),eq_roots=object(),
        def_rows=object(),output_slots=object(),pool=object()) is None


@pytest.mark.parametrize('dtype',[np.int32,np.int64])
@pytest.mark.parametrize('hits',[[],[3],[40],[100],[3,40,100],list(range(128))])
def test_every_canonical_interval_and_removed_own_pivot_matches_lookup(dtype,hits):
    lookup=np.full(128,-1,np.int64);lookup[hits]=np.arange(len(hits))
    pool=WorkPool(256_000_000);r=index(lookup,pool=pool,enabled=True)
    for start in (0,2,20,41,64):
        for stop in (start+1,min(start+16,128),min(start+17,128),128):
            cols=np.arange(start,stop,dtype=dtype)
            for own in (False,True):
                considered=cols[:-1] if own else cols
                possible=r.may_hit(considered,original_width=len(cols))
                if not possible:assert np.all(lookup[considered]<0)
                if len(cols)<=16:assert possible
    report=r.finish()
    assert report['temporary_successor_physically_retired'] and not report['complete_source_incidence_or_new_quotient_proved']


def test_nonalias_far_pivot_does_not_prevent_a_valid_parent_span_miss():
    lookup=np.full(200,-1,np.int64);lookup[80]=0
    pool=WorkPool(256_000_000);r=index(lookup,pool=pool,enabled=True)
    cols=np.r_[np.arange(10,30,dtype=np.int64),180]
    assert not r.may_hit(cols,original_width=len(cols))
    assert r.may_hit(np.r_[cols[:-1],80,180],original_width=len(cols)+1)
    assert r.finish()['full_row_coefficient_scans_removed']==21


@pytest.mark.parametrize('bad',['source','index','retained','budget','closed','dtype','bounds','length'])
def test_no_changed_table_unpaid_or_stale_proof_can_finish(bad):
    lookup=np.full(128,-1,np.int64);lookup[70]=1;pool=WorkPool(256_000_000)
    if bad=='budget':
        with pytest.raises(MemoryError):index(lookup,pool=WorkPool(0),enabled=True)
        return
    r=index(lookup,pool=pool,enabled=True)
    if bad=='source':lookup[70]=-1
    elif bad=='index':r.following.flags.writeable=True;r.following[0]=0
    elif bad=='retained':retained=r.following
    elif bad=='closed':r.finish()
    else:
        cols=np.arange(20,dtype=np.int64)
        if bad=='dtype':cols=cols.astype(np.float64)
        elif bad=='bounds':cols[-1]=150
        with pytest.raises(ValueError):r.may_hit(cols,original_width=30 if bad=='length' else 20)
        return
    with pytest.raises(ValueError):r.finish()


@pytest.mark.parametrize('kind',['ordinary','old_radix','main_radix','main_binary','shared_conv'])
def test_complete_source_rows_and_all_old_product_inputs_match_independent_census(kind):
    _,original,_=fixture(kind);hz=original.hz;root=original.nodes[original.root]
    kw=dict(old_nc=original.old_n_cont,logical_nc=original.logical_n_cont,old_eq=original.old_n_eq,
        eq_roots=original.eq_roots,def_rows=original.def_rows,output_slots=root['slots'][root['needed']])
    old=original_census(hz,**kw)
    whole=WorkPool(256_000_000);branch=BranchPool(whole)
    new=assess(hz,**kw,pool=branch,enabled=True)
    assert new['local_aliases']==old['local_aliases']
    assert new['all_original_hits']==old['counts']['hits']
    assert new['all_original_hit_rows']==sum(old['row_classes'].values())
    assert new['source_HZ_unchanged'] and new['all_original_product_input_images_equal']
    assert new['all_rows_scanned']==hz.n_eq+hz.n_ineq and whole.used==branch.used
    assert not new['predicates_changed'] and not new['actual_updated_C31_generation_report_proved']


@pytest.mark.parametrize('bad',['map','output','canonical','index','zero','nan','budget'])
def test_full_census_cannot_publish_on_corrupt_or_partial_source(bad):
    _,original,_=fixture();hz=original.hz;root=original.nodes[original.root]
    kw=dict(old_nc=original.old_n_cont,logical_nc=original.logical_n_cont,old_eq=original.old_n_eq,
        eq_roots=original.eq_roots.copy(),def_rows=original.def_rows,output_slots=root['slots'][root['needed']])
    pool=WorkPool(256_000_000)
    if bad=='map':kw['eq_roots'][0]=kw['eq_roots'][-1]
    elif bad=='output':kw['output_slots']=np.empty(0,np.int64)
    elif bad=='canonical':
        r=int(np.flatnonzero(np.diff(hz.Ac.indptr)>1)[0]);a=int(hz.Ac.indptr[r]);hz.Ac.indices[a+1]=hz.Ac.indices[a]
        hz.Ac.has_canonical_format=True
    elif bad=='index':hz.Ac.indices[0]=hz.n_cont
    elif bad=='zero':hz.Ac.data[0]=0.
    elif bad=='nan':hz.Ac.data[0]=np.nan
    else:pool=WorkPool(0)
    with pytest.raises((ValueError,MemoryError)):assess(hz,**kw,pool=pool,enabled=True)
