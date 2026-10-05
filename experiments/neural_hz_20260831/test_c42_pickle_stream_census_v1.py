from collections import Counter
import hashlib
import io
import pickle
import pickletools
import numpy as np
import pytest
from experiments.neural_hz_20260831.c42_pickle_stream_census_v1 import census,Reader
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool


def scan(raw,pool=None):
    return census(io.BytesIO(raw),expected_sha256=hashlib.sha256(raw).hexdigest(),expected_bytes=len(raw),
        pool=pool if pool is not None else WorkPool(256_000_000),enabled=True)


@pytest.mark.parametrize('protocol',[0,1,2,3,4,5])
def test_complete_opcode_and_integer_counts_equal_stdlib_oracle(protocol):
    value={'ints':[-9,300,300,301,2**63,2**63,True,False,None],
        'float':[-0.,1.5],'str':'hello','bytes':b'abc','containers':({1,2},(1,2))}
    raw=pickle.dumps(value,protocol=protocol);r=scan(raw);ops=list(pickletools.genops(raw))
    assert r['opcode_counts']==dict(Counter(op.name for op,arg,pos in ops))
    ints=[arg for op,arg,pos in ops if op.name in ('INT','BININT','BININT1','BININT2','LONG','LONG1','LONG4') and type(arg) is int]
    assert r['integer_literal_occurrences']==len(ints)
    assert r['distinct_integer_literal_values']==len(set(ints))
    outside=[v for v in ints if not -5<=v<=256]
    assert r['outside_small_range_repeated_occurrences']==len(outside)-len(set(outside))
    assert not r['unpickler_or_reducer_executed']


def test_large_numeric_payload_uses_bounded_reads_and_no_numpy_decode():
    a=np.arange(30000,dtype=np.float64);raw=pickle.dumps({'a':a,'again':a,'bytes':bytes(200000)},protocol=5)
    r=scan(raw)
    assert r['largest_stream_read_bytes']<=65536 and r['largest_opaque_argument_bytes']>=a.nbytes
    assert r['opaque_payload_bytes']>=a.nbytes+200000
    assert r['opcode_count']==sum(1 for _ in pickletools.genops(raw))


def forbidden_reducer():raise AssertionError('census executed a reducer')
class Deferred:
    def __reduce__(self):return forbidden_reducer,()


def test_object_constructor_is_never_executed():
    raw=pickle.dumps(Deferred(),protocol=5)
    assert scan(raw)['opcode_counts']['REDUCE']==1


@pytest.mark.parametrize('bad',['truncated','trailing','opcode','hash','budget','nested_frame','huge_integer'])
def test_incomplete_unknown_or_unpaid_image_never_reports_complete(bad):
    raw=pickle.dumps([300,301],protocol=5)
    if bad=='truncated':raw=raw[:-1]
    elif bad=='trailing':raw+=b'x'
    elif bad=='opcode':raw=b'\xff.'
    elif bad=='nested_frame':raw=b'\x80\x05\x95'+(10).to_bytes(8,'little')+b'\x95'+(1).to_bytes(8,'little')+b'.'
    elif bad=='huge_integer':raw=b'\x80\x05\x8a\xff'+bytes(255)+b'.'
    with pytest.raises((ValueError,MemoryError)):
        if bad=='hash':census(io.BytesIO(raw),expected_sha256='0'*64,expected_bytes=len(raw),pool=WorkPool(10000),enabled=True)
        else:scan(raw,WorkPool(0) if bad=='budget' else None)


def test_default_off_and_bounded_reader():
    assert census(object(),expected_sha256=object(),expected_bytes=object(),pool=object()) is None
    with pytest.raises(ValueError):Reader(io.BytesIO()).read(65537)
    with pytest.raises(ValueError):Reader(io.BytesIO(b'x'*65537)).readline()
