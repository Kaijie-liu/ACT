import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import UID_LIMIT
from experiments.neural_hz_20260831.c32_fresh_lineage_v1 import ReversibleLineage,encode_splice
from experiments.neural_hz_20260831.c35_current_uid_census_v1 import current_tables,checked_incidence
from experiments.neural_hz_20260831.c45_streamed_uid_queries_v1 import _Queries,indexed_current_tables
from experiments.neural_hz_20260831.test_c32_live_splice_v1 import execute


def lineage(deleted,retired):
    roots=np.array([encode_splice(int(row),0,True,1.,1,0) for row in deleted],np.int64)
    value=ReversibleLineage(roots,np.zeros(len(roots),np.int64),np.arange(len(roots),dtype=np.int32),
        np.array([(a<<20)|b for a,b in retired],np.uint64),np.zeros(0,np.uint64),0,0)
    value.seal=value.fingerprint();value.validate();return value


def test_default_off_does_not_read_state():
    assert indexed_current_tables(object(),pool=object()) is None


@pytest.mark.parametrize('removed',[[],[0],[15],[0,3,5,15]])
def test_every_rank_and_UID_matches_existing_lineage(removed):
    retired=[(i*17,UID_LIMIT-1-i) for i in range(len(removed))]
    source=lineage(removed,retired);a=WorkPool(256_000_000);b=WorkPool(256_000_000)
    index=_Queries(source,16,b)
    for row in range(16):assert index.eq_row(row,pool=b)==source.eq_row(row,pool=a)
    for uid in [*range(80),UID_LIMIT-1]:assert index.retired_to(uid,pool=b)==source.retired_to(uid,pool=a)
    report=index.finish();assert report['owned_query_tables_physically_retired']
    with pytest.raises(ValueError):index.retired_to(0,pool=b)


@pytest.mark.parametrize('layer_id',[1,78,9123])
def test_unchanged_full_C35_maps_and_independent_incidence_on_actual_toy(monkeypatch,layer_id):
    _,runtime,_,_=execute(monkeypatch,layer_id=layer_id);state=runtime['lifted'];before=state.validate()
    old,a=current_tables(state,pool=WorkPool(256_000_000));pool=WorkPool(256_000_000)
    new,b=indexed_current_tables(state,pool=pool,enabled=True)
    for name in ('eq','le','definitions'):assert np.array_equal(old[name],new[name])
    assert old['lookup']==new['lookup'] and all(b[k]==v for k,v in a.items())
    assert np.array_equal(checked_incidence(state,old,pool=WorkPool(256_000_000)),
        checked_incidence(state,new,pool=pool))
    after=state.validate()
    elapsed='complete_source_splice_authentication_elapsed_s'
    assert {k:v for k,v in after.items() if k!=elapsed}=={k:v for k,v in before.items() if k!=elapsed}
    assert b['query_payment']['owned_query_tables_physically_retired']
    assert not b['complete_source_admission_proved'] and not b['actual_native_payment_proved']


@pytest.mark.parametrize('bad',['skip','repeat','range','pool','early','mutate','reseal','index','owner','rows','budget'])
def test_incomplete_or_changed_query_transaction_is_rejected(bad):
    source=lineage([2],[(4,1)]);pool=WorkPool(0 if bad=='budget' else 256_000_000)
    with pytest.raises((ValueError,MemoryError)):
        index=_Queries(source,UID_LIMIT+1 if bad=='rows' else 4,pool)
        if bad=='skip':index.eq_row(1,pool=pool)
        elif bad=='repeat':index.eq_row(0,pool=pool);index.eq_row(0,pool=pool)
        elif bad=='range':index.retired_to(UID_LIMIT,pool=pool)
        elif bad=='pool':index.retired_to(0,pool=WorkPool(256_000_000))
        elif bad=='early':index.finish()
        elif bad in ('mutate','reseal'):
            for row in range(4):index.eq_row(row,pool=pool)
            source.eq_scales[0]+=1
            if bad=='reseal':source.seal=source.fingerprint()
            index.finish()
        elif bad in ('index','owner'):
            for row in range(4):index.eq_row(row,pool=pool)
            index.replacements.flags.writeable=True
            if bad=='index':
                index.replacements[0]=5;index.replacements.flags.writeable=False
            index.finish()


def test_bounded_ordered_query_payment_is_smaller_on_repeated_structure():
    # Synthetic repeated structure, no archived source maps or target id.
    rows=65536;removed=list(range(128,rows,256));retired=[(i*101,i) for i in range(len(removed))]
    source=lineage(removed,retired);old=WorkPool(256_000_000);new=WorkPool(256_000_000)
    index=_Queries(source,rows,new)
    for row in range(rows):
        assert index.eq_row(row,pool=new)==source.eq_row(row,pool=old)
        assert index.retired_to(row,pool=new)==source.retired_to(row,pool=old)
    report=index.finish()
    assert new.used<old.used and report['query_owned_bytes']<=4*UID_LIMIT+4*len(removed)
