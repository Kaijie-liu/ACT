from collections import Counter
import random
import pytest
from experiments.neural_hz_20260831 import c43_paged_integer_summary_v1 as paged
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool


@pytest.mark.parametrize('seed',[0,1,27])
def test_every_exported_statistic_matches_independent_exact_counter(seed):
    rng=random.Random(seed);values=[rng.randrange(-500,2000) for _ in range(20000)]
    values+=[2**80]*40+[-2**80]*19+[7]*21
    table=paged.Summary(WorkPool(256_000_000))
    for v in values:table.add(v)
    r=table.report();oracle=Counter(values);outside={v:n for v,n in oracle.items() if not -5<=v<=256}
    assert r['integer_literal_occurrences']==len(values) and r['distinct_integer_literal_values']==len(oracle)
    assert r['outside_small_range_repeated_occurrences']==sum(outside.values())-len(outside)
    assert r['outside_small_range_value_multiplicity_histogram']==dict(Counter(paged.category(n) for n in outside.values()))
    assert r['paged_counter_numeric_bytes']==sum(a.nbytes for a in table.pages.values())


@pytest.mark.parametrize('count',[1,2,3,4,7,8,15,16,17,256])
def test_all_saturation_boundaries_keep_exact_category_and_total(count):
    table=paged.Summary(WorkPool(10000))
    for _ in range(count):table.add(700)
    r=table.report();assert r['outside_small_range_value_multiplicity_histogram']=={paged.category(count):1}
    assert r['outside_small_range_repeated_occurrences']==count-1


def test_dense_range_uses_pages_not_python_entry_per_integer():
    table=paged.Summary(WorkPool(256_000_000))
    for v in range(10000):table.add(v)
    assert table.distinct==10000 and len(table.pages)==40 and sum(a.nbytes for a in table.pages.values())==10240


def test_sparse_entry_cap_and_zero_budget_fail_closed(monkeypatch):
    monkeypatch.setattr(paged,'MAX_SLOTS',512);table=paged.Summary(WorkPool(10000))
    table.add(0);table.add(256)
    with pytest.raises(MemoryError):table.add(512)
    with pytest.raises(MemoryError):paged.Summary(WorkPool(0)).add(7)
    with pytest.raises(ValueError):table.add(True)
