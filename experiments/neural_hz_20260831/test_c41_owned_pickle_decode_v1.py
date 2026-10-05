import hashlib
import io
import pickle
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load,_OwnedUnpickler
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_partial_csr_owner_ledger_v3 import snapshot_partial_csr_owners
from experiments.neural_hz_20260831.s0_c2_whole_state_ledger_prototype import WholeStateRoots
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect


def decoded(value,pool=None):
    raw=pickle.dumps(value,protocol=5);budget=pool if pool is not None else WorkPool(256_000_000)
    return load(io.BytesIO(raw),expected_sha256=hashlib.sha256(raw).hexdigest(),pool=budget,enabled=True)


def test_default_off_never_reads_source():
    assert load(object(),expected_sha256=object(),pool=object()) is None


@pytest.mark.parametrize('dtype',['bool','i4','u8','f8'])
@pytest.mark.parametrize('order',['C','F'])
def test_bits_readonly_memo_aliases_and_existing_ledger(dtype,order):
    a=np.array([[0,1],[1,0]],dtype=dtype,order=order);a.flags.writeable=False
    value,report=decoded({'a':a,'same':[a,a]});out=value['a']
    assert out is value['same'][0] is value['same'][1]
    assert np.array_equal(out,a) and out.strides==a.strides and not out.flags.writeable
    m=collect(SimpleNamespace(),{'payload':value}).measure()
    assert (m.resident_bytes,m.resident_entries)==(a.nbytes,a.size)
    assert report['copied_numeric_bytes']==a.nbytes and report['readonly_backing_groups']==1
    assert report['decoder_temporary_roots_released'] and report['source_binding_still_required']


def test_signed_zero_and_nonfinite_bits_not_silently_normalized():
    a=np.array([0.,-0.,np.inf,-np.inf,np.nan]);a.flags.writeable=False
    value,_=decoded(a)
    assert value.tobytes()==a.tobytes()


def test_writable_existing_adapter_has_no_new_payload_copy():
    a=np.arange(17,dtype=np.float64);out,report=decoded(a)
    assert out.flags.writeable and np.array_equal(out,a) and report['copied_numeric_bytes']==0
    assert snapshot_partial_csr_owners(WholeStateRoots(active={'a':out},consumer_gc_enabled=False)).resident_bytes==a.nbytes


def test_full_hash_is_checked_before_any_reducer_and_trailing_data_rejected():
    raw=pickle.dumps(np.zeros(2),protocol=5)
    with pytest.raises(ValueError,match='before decoding'):
        load(io.BytesIO(raw),expected_sha256='0'*64,pool=WorkPool(0),enabled=True)
    raw+=b'x'
    with pytest.raises(ValueError,match='trailing bytes'):
        load(io.BytesIO(raw),expected_sha256=hashlib.sha256(raw).hexdigest(),pool=WorkPool(1000),enabled=True)


@pytest.mark.parametrize('bad',['budget','dtype','shape','length','endian','order'])
def test_unknown_or_unpaid_numeric_construction_fails_closed(bad):
    pool=WorkPool(0 if bad=='budget' else 1000);reader=_OwnedUnpickler(io.BytesIO(),pool)
    data=b'\0'*16;dtype=np.dtype('i8');shape=(2,);order='C'
    if bad=='dtype':dtype=np.dtype('c16');shape=(1,)
    elif bad=='shape':shape=[2]
    elif bad=='length':shape=(3,)
    elif bad=='endian':dtype=np.dtype('>i8')
    elif bad=='order':order='A'
    with pytest.raises((MemoryError,ValueError)):reader.frombuffer(data,dtype,shape,order)


def test_shared_readonly_buffer_is_one_owned_group_and_dtype_alias_rejected():
    reader=_OwnedUnpickler(io.BytesIO(),WorkPool(10000));data=bytes(range(16))
    a=reader.frombuffer(data,np.dtype('i4'),(4,),'C')
    b=reader.frombuffer(data,np.dtype('i4'),(2,2),'F')
    assert np.shares_memory(a,b) and reader.copied_bytes==16 and reader.restored_arrays==2
    with pytest.raises(ValueError,match='dtype alias'):reader.frombuffer(data,np.dtype('i8'),(2,),'C')


def test_complete_actual_type_toy_source_is_freshly_bound_after_decode(monkeypatch):
    from experiments.neural_hz_20260831.test_c32_live_splice_v1 import execute
    from experiments.neural_hz_20260831.c32_splice_binding_v1 import export,admit
    from experiments.neural_hz_20260831.c32_live_splice_runtime_v1 import numeric_roots
    from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
    _,runtime,hz,_=execute(monkeypatch);original=runtime['lifted']
    value,report=decoded(dict(fields=export(original),runtime=numeric_roots(runtime),same_hz=hz))
    new,binding=admit(enabled=True,**value['fields'])
    assert new.receipt is not original.receipt and source_digest(new.hz)==source_digest(hz)
    assert new.hz is value['same_hz'] is value['runtime']['hz']
    assert new.events is value['runtime']['phase_events'] and not new.events.flags.writeable
    assert new.lineage.eq_roots is new.original_fields['eq_roots']
    assert new.lineage.eq_scales is new.original_fields['eq_scales']
    assert binding['complete_new_HZ_sha256']==source_digest(hz)
    assert report['copied_numeric_bytes']>=new.events.nbytes
    corrupted=dict(value['fields']);corrupted['events']=new.events.copy();corrupted['events'][0]+=np.uint64(1)
    with pytest.raises(ValueError):admit(enabled=True,**corrupted)
