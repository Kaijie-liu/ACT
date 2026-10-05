"""Ordinary variable-index sequences, complete payloads and original sharing."""
import io
import pickle
import pickletools
import numpy as np
import pytest
import torch
from experiments.neural_hz_20260831.c77_integer_memo_v1 import rewrite
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.test_c74_native_binding_v1 import native
from experiments.neural_hz_20260831.c74_native_binding_v1 import admit_source,admit_native
from experiments.neural_hz_20260831.c73_outer_query_v1 import GuardedLocalSpliceJournal
from experiments.neural_hz_20260831.c70_native_proof_v1 import verify_inverse
import hashlib


def convert(value, protocol=5):
    raw=pickle.dumps(value,protocol=protocol)
    count=sum(op.name=='MEMOIZE' for op,_,_ in pickletools.genops(raw))
    output=io.BytesIO()
    with io.BufferedReader(io.BytesIO(raw),buffer_size=65536) as stream:
        report=rewrite(stream,output,original_memo_count=count,pool=WorkPool(256_000_000),enabled=True)
    result=output.getvalue()
    assert hashlib.sha256(result).hexdigest()==report['output_sha256']
    assert report['all_noninteger_payloads_preserved']
    return pickle.loads(result),report,result


@pytest.mark.parametrize('protocol',[4,5])
@pytest.mark.parametrize('order',['ascending','descending','repeated','mixed'])
def test_exact_integer_values_and_types_across_prefixes(protocol,order):
    values=list(range(257,9000))
    if order=='descending':values.reverse()
    if order=='repeated':values=[1000]*40+values
    if order=='mixed':values=values[:40]+[False,True,-1,0,256,65537,2**48]+values[40:]
    source=[list(values),tuple(values),list(values)]
    restored,report,_=convert(source,protocol)
    assert restored==source
    assert [type(v) for v in restored[0]]==[type(v) for v in source[0]]
    assert restored[0] is not restored[2]
    assert restored[0][1] is restored[2][1]
    assert report['reused_integer_memos']>0


def test_large_frame_payloads_and_original_mutable_memo_indices():
    array=np.arange(100000,dtype=np.float64);readonly=np.arange(800,dtype=np.int64)
    readonly.flags.writeable=False
    first=list(range(1000,10000));other=list(range(1000,10000))
    source={'ids':first,'same_ids':first,'different_ids':other,
            'array':array,'same_array':array,'different_array':array.copy(),
            'readonly':readonly,'payload':b'z'*100000,'unicode':'NN变量'*30000}
    restored,report,raw=convert(source)
    assert restored['ids'] is restored['same_ids'] and restored['ids'] is not restored['different_ids']
    assert restored['array'] is restored['same_array'] and restored['array'] is not restored['different_array']
    assert np.array_equal(restored['array'],array) and np.array_equal(restored['readonly'],readonly)
    assert not restored['readonly'].flags.writeable
    assert restored['payload']==source['payload'] and restored['unicode']==source['unicode']
    with io.BytesIO(raw) as stream:
        checked,decoder=load(stream,expected_sha256=report['output_sha256'],pool=WorkPool(256_000_000),enabled=True)
    assert checked['array'] is checked['same_array'] and not checked['readonly'].flags.writeable
    assert decoder['encoded_object_aliases_preserved_by_pickle_memo']


def test_distinct_equal_tensors_stay_distinct():
    a=torch.arange(400,dtype=torch.float64);b=a.clone()
    got,_,_=convert({'a':a,'same':a,'b':b,'ids':list(range(300,700))})
    assert got['a'] is got['same'] and got['a'] is not got['b']
    assert got['a'].data_ptr()!=got['b'].data_ptr() and torch.equal(got['a'],a)


def test_complete_source_native_binding_and_exact_inverse_survive():
    state,plans,_=native()
    payload=dict(fields=state.source.original_fields,source_proof=state.source.proof_bytes,
        hz=state.hz,journal=vars(state.lineage),events=state.events,phase=state.actual_phase_image,
        transfer=state.transfer_proof_bytes,report=state.construction_report,
        metadata=[list(range(300,2300)),tuple(range(300,2300))])
    got,_,_=convert(payload)
    source,_=admit_source(got['fields'],got['source_proof'],
        expected_sha256=state.source.expected_proof_sha256,enabled=True)
    restored,_=admit_native(enabled=True,source=source,hz=got['hz'],
        lineage=GuardedLocalSpliceJournal(**got['journal']),events=got['events'],
        actual_phase_image=got['phase'],transfer_proof_bytes=got['transfer'],
        expected_transfer_sha256=state.expected_transfer_sha256,construction_report=got['report'])
    assert restored.lineage.eq_roots is source.eq_roots
    assert verify_inverse(source,restored.hz,restored.lineage,plans,pool=WorkPool(256_000_000))['all_equations_exact']


def test_default_off_wrong_original_memo_population_and_budget():
    assert rewrite(None,None,original_memo_count=None,pool=None) is None
    raw=pickle.dumps({'ids':list(range(300,700))},protocol=5)
    with io.BufferedReader(io.BytesIO(raw)) as stream:
        with pytest.raises(MemoryError):rewrite(stream,None,original_memo_count=3,pool=WorkPool(0),enabled=True)
    with io.BufferedReader(io.BytesIO(raw)) as stream:
        with pytest.raises(ValueError):rewrite(stream,None,original_memo_count=0,pool=WorkPool(256_000_000),enabled=True)


def test_preflight_and_written_stream_have_identical_bills_and_hashes():
    value={'ids':list(range(300,5000)),'payload':b'v'*200000,'again':tuple(range(300,5000))}
    raw=pickle.dumps(value,protocol=5);count=sum(op.name=='MEMOIZE' for op,_,_ in pickletools.genops(raw))
    results=[]
    for target in (None,io.BytesIO()):
        p=WorkPool(256_000_000)
        with io.BufferedReader(io.BytesIO(raw),buffer_size=65536) as stream:
            r=rewrite(stream,target,original_memo_count=count,pool=p,enabled=True)
        results.append((r,p.used,p.parts))
    assert results[0]==results[1]
