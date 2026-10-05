"""Original physical storage identity, complete values and independent proofs."""
import hashlib
import io
import pickle
import numpy as np
import pytest
import torch
from experiments.neural_hz_20260831.c79_storage_identity_decode_v1 import load
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.test_c77_integer_memo_v1 import convert
from experiments.neural_hz_20260831.test_c74_native_binding_v1 import native
from experiments.neural_hz_20260831.c74_native_binding_v1 import admit_source,admit_native
from experiments.neural_hz_20260831.c73_outer_query_v1 import GuardedLocalSpliceJournal
from experiments.neural_hz_20260831.c70_native_proof_v1 import verify_inverse
from types import SimpleNamespace


def decode(raw):
    return load(io.BytesIO(raw),expected_sha256=hashlib.sha256(raw).hexdigest(),
        pool=WorkPool(256_000_000),enabled=True)


@pytest.mark.parametrize('protocol',[4,5])
@pytest.mark.parametrize('dtype',[torch.float64,torch.float32,torch.int64,torch.int32,
    torch.int16,torch.int8,torch.uint8,torch.bool])
def test_shared_strided_offset_views_and_distinct_equal_storage(protocol,dtype):
    a=torch.arange(240).to(dtype).reshape(20,12);b=a[1:18,2:10];c=a.clone();d=a.t()
    values=[a,b,c,d,a]
    restored,report=decode(pickle.dumps(values,protocol=protocol))
    assert restored[0] is restored[4] and restored[0] is not restored[1]
    for x,y in zip(values,restored):
        assert torch.equal(x,y) and x.dtype==y.dtype and x.shape==y.shape
        assert x.stride()==y.stride() and x.storage_offset()==y.storage_offset()
    ids=[v.untyped_storage()._cdata for v in restored]
    assert ids[0]==ids[1]==ids[3] and ids[0]!=ids[2]
    assert report['original_tensor_storage_identities']==2 and report['tensor_storage_reuses']==2
    original=collect(SimpleNamespace(),{'complete':values}).measure()
    changed=collect(SimpleNamespace(),{'complete':restored}).measure()
    assert original.resident_bytes==changed.resident_bytes and original.resident_entries==changed.resident_entries


def test_default_off_and_complete_external_hash_gate():
    assert load(None,expected_sha256=None,pool=None) is None
    with pytest.raises(ValueError,match='before decoding'):
        load(io.BytesIO(pickle.dumps(torch.arange(30))),expected_sha256='0'*64,
            pool=WorkPool(256_000_000),enabled=True)


def test_integer_wire_plus_complete_source_native_proofs_and_inverse():
    state,plans,_=native();a=torch.arange(30,dtype=torch.float64)
    payload=dict(fields=state.source.original_fields,source_proof=state.source.proof_bytes,
        hz=state.hz,journal=vars(state.lineage),events=state.events,phase=state.actual_phase_image,
        transfer=state.transfer_proof_bytes,report=state.construction_report,
        tensor_views=[a,a[2:20]],metadata=[list(range(300,5000)),tuple(range(300,5000))])
    _,_,raw=convert(payload);got,report=decode(raw)
    source,_=admit_source(got['fields'],got['source_proof'],expected_sha256=state.source.expected_proof_sha256,enabled=True)
    restored,_=admit_native(enabled=True,source=source,hz=got['hz'],
        lineage=GuardedLocalSpliceJournal(**got['journal']),events=got['events'],actual_phase_image=got['phase'],
        transfer_proof_bytes=got['transfer'],expected_transfer_sha256=state.expected_transfer_sha256,
        construction_report=got['report'])
    assert restored.lineage.eq_roots is source.eq_roots
    assert verify_inverse(source,restored.hz,restored.lineage,plans,pool=WorkPool(256_000_000))['all_equations_exact']
    assert report['tensor_storage_reuses']==1


def test_readonly_numeric_owner_and_original_array_sharing_unchanged():
    a=np.arange(100,dtype=np.float64);a.flags.writeable=False
    t=torch.arange(200,dtype=torch.float64)
    got,report=decode(pickle.dumps({'a':a,'same':a,'other':a.copy(),'views':[t,t[10:]]},protocol=5))
    assert got['a'] is got['same'] and got['a'] is not got['other']
    assert not got['a'].flags.writeable and np.array_equal(got['a'],a)
    assert report['copied_numeric_entries']==100


def test_storage_map_does_not_escape_its_one_checkpoint():
    a=torch.arange(40);raw=pickle.dumps([a,a[2:]],protocol=5)
    first,_=decode(raw);second,_=decode(raw)
    assert first[0].untyped_storage()._cdata==first[1].untyped_storage()._cdata
    assert second[0].untyped_storage()._cdata==second[1].untyped_storage()._cdata
    assert first[0].untyped_storage()._cdata!=second[0].untyped_storage()._cdata
