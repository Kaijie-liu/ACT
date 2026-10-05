"""Every original query and whole proof retained with exact interval guards."""
import pickle
import pytest
from experiments.neural_hz_20260831.c73_outer_query_v1 import compile_journal,GuardedLocalSpliceJournal
from experiments.neural_hz_20260831.c68_local_splice_v1 import LocalSpliceJournal
from experiments.neural_hz_20260831.c62_local_equations_v1 import SCHEMA
from experiments.neural_hz_20260831.c70_native_proof_v1 import verify,verify_inverse,digest
from experiments.neural_hz_20260831.c30_append_discovery_v1 import discover_append
from experiments.neural_hz_20260831.c30_first_write_v1 import splice_append
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.test_c28_consumer_discovery_v1 import source
from experiments.neural_hz_20260831.test_c30_first_write_v1 import view_from


def pool():return WorkPool(256_000_000)


def journal(c,plans):
    return compile_journal(c.eq_roots,c.eq_scales,plans,old_n_cont=c.old_n_cont,
        old_n_eq=c.old_n_eq,source_n_cont=c.hz.n_cont,source_schema=SCHEMA,pool=pool(),enabled=True)


@pytest.mark.parametrize('mixed',[False,True])
@pytest.mark.parametrize('subtract',[False,True])
@pytest.mark.parametrize('phase',[False,True])
def test_every_original_query_and_independent_full_proof(mixed,subtract,phase):
    c,h,o,eq,le,*_=source(mixed=mixed,subtract=subtract,phase=phase)
    view=view_from(c,h);plans,_=discover_append(c,view,o,pool=pool(),enabled=True)
    new,_=splice_append(view,plans,pool=pool(),enabled=True)
    j=journal(c,plans);old=LocalSpliceJournal(**vars(j))
    for r in range(h.n_eq+3):assert j.eq_row(r,pool=pool())==old.eq_row(r,pool=pool())
    for uid in list(map(int,eq))+list(map(int,le))+[-1,0,2**20-1]:
        assert j.retired_to(uid,pool=pool())==old.retired_to(uid,pool=pool())
    proof=verify(c,view,o,plans,new,j,pool=pool())
    assert proof['all_original_UIDs']==len(eq)+len(le)
    assert proof['all_original_EQ_rank_queries']==h.n_eq
    assert verify_inverse(c,new,j,plans,pool=pool())==verify_inverse(c,new,old,plans,pool=pool())


def test_empty_indexes_keep_identity_without_search():
    c,*_=source();j=journal(c,[]);p=pool()
    for r in range(c.hz.n_eq+1):assert j.eq_row(r,pool=p)==r
    for uid in (-1,0,1,2**20-1):assert j.retired_to(uid,pool=p) is None
    assert set(p.parts)=={'c73_empty_EQ_index','c73_empty_retired_index'}


def test_default_off_and_budget_rejection():
    assert compile_journal(None,None,None,old_n_cont=None,old_n_eq=None,
        source_n_cont=None,source_schema=None,pool=None) is None
    c,*_=source()
    with pytest.raises(MemoryError):
        compile_journal(c.eq_roots,c.eq_scales,[],old_n_cont=c.old_n_cont,old_n_eq=c.old_n_eq,
            source_n_cont=c.hz.n_cont,source_schema=SCHEMA,pool=WorkPool(0),enabled=True)


def test_wire_schema_numeric_bits_and_original_reader_restore():
    c,h,o,*_=source(phase=True);view=view_from(c,h)
    plans,_=discover_append(c,view,o,pool=pool(),enabled=True)
    j=journal(c,plans);before=digest(vars(j))
    fields=pickle.loads(pickle.dumps(vars(j),protocol=5))
    assert digest(fields)==before and type(j) is GuardedLocalSpliceJournal
    original=LocalSpliceJournal(**fields)
    for r in range(h.n_eq):assert original.eq_row(r,pool=pool())==j.eq_row(r,pool=pool())
    assert j.eq_roots is c.eq_roots and j.eq_scales is c.eq_scales


def test_outer_shortcuts_remove_actual_binary_search_not_its_tariff():
    c,h,o,*_=source(phase=True);plans,_=discover_append(c,view_from(c,h),o,pool=pool(),enabled=True)
    j=journal(c,plans);old=LocalSpliceJournal(**vars(j));p=pool();q=pool()
    assert j.eq_row(0,pool=p)==old.eq_row(0,pool=q)==0
    assert j.retired_to(0,pool=p)==old.retired_to(0,pool=q) is None
    assert p.used==32 and p.used<q.used
    inside=pool();original=pool();row=plans[0].definition
    assert j.eq_row(row,pool=inside)==old.eq_row(row,pool=original) is None
    assert inside.used==original.used+16
