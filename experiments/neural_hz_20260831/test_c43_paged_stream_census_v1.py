import hashlib
import io
import pickle
import pytest
from experiments.neural_hz_20260831.c42_pickle_stream_census_v1 import census as old
from experiments.neural_hz_20260831.c43_paged_stream_census_v1 import census
from experiments.neural_hz_20260831.c43_paged_integer_summary_v1 import Summary
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool


@pytest.mark.parametrize('protocol',[0,1,2,3,4,5])
def test_every_previously_exported_count_matches_complete_dictionary_oracle(protocol):
    raw=pickle.dumps(dict(numbers=[-256,-6,-5,0,256,257,700,2**80]*30,payload=bytes(8000 if protocol==0 else 90000)),protocol=protocol)
    kw=dict(expected_sha256=hashlib.sha256(raw).hexdigest(),expected_bytes=len(raw),enabled=True)
    want=old(io.BytesIO(raw),pool=WorkPool(256_000_000),**kw)
    actual=census(io.BytesIO(raw),pool=WorkPool(256_000_000),**kw)
    for name,value in want.items():
        if name not in ('schema','diagnostic_work','work_parts'):assert actual[name]==value


def test_more_than_one_million_dense_values_fit_real_bounded_numeric_pages():
    table=Summary(WorkPool(256_000_000))
    for value in range(1_000_001):table.add(value)
    r=table.report()
    assert r['distinct_integer_literal_values']==1_000_001
    assert r['paged_counter_numeric_slots']==1_000_192 and r['paged_counter_pages']==3907
    assert r['outside_small_range_repeated_occurrences']==0


def test_default_off_never_reads_a_stream():
    assert census(object(),expected_sha256=object(),expected_bytes=object(),pool=object()) is None


def test_old_protocol0_oversized_line_is_rejected_by_both_versions():
    raw=pickle.dumps(bytes(90000),protocol=0)
    for fn in (old,census):
        with pytest.raises(ValueError,match='unbounded or truncated pickle line'):
            fn(io.BytesIO(raw),expected_sha256=hashlib.sha256(raw).hexdigest(),expected_bytes=len(raw),
                pool=WorkPool(256_000_000),enabled=True)
