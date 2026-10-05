from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c40_half_gauge_transaction_v1 import (
    pack_descriptors,unpack_descriptors,functional,checkpoint_payload,checkpoint_closure)
from experiments.neural_hz_20260831.test_c40_compact_half_gauge_v1 import actual
from experiments.neural_hz_20260831.test_c32_live_splice_v1 import execute
from experiments.neural_hz_20260831.c32_splice_binding_v1 import export
from experiments.neural_hz_20260831.c32_live_splice_runtime_v1 import numeric_roots


def test_default_off_and_unbound_source_never_construct():
    assert functional(object(),object(),pool=object()) is None
    with pytest.raises(ValueError):functional(object(),object(),pool=WorkPool(256_000_000),enabled=True)


def test_numeric_word_codec_preserves_every_bit_and_requires_no_structured_owner():
    *_,desc,runs,report,inv,pool,degrees=actual()
    packed=pack_descriptors(desc,pool=pool);decoded=unpack_descriptors(packed,pool=pool)
    assert packed.flags.owndata and packed.dtype==np.uint64 and packed.nbytes==16*len(desc)
    assert np.array_equal(desc.view(np.uint8),decoded.view(np.uint8))


@pytest.mark.parametrize('bad',['odd','reserved','parent','power'])
def test_corrupt_descriptor_words_fail_closed(bad):
    *_,desc,runs,report,inv,pool,degrees=actual();packed=pack_descriptors(desc,pool=pool)
    if bad=='odd':packed=packed[:-1]
    elif bad=='reserved':packed[1]|=np.uint64(1<<50)
    elif bad=='parent':packed[0]=np.uint64((999<<32)|3)
    else:packed[1]|=np.uint64(63<<32)
    with pytest.raises(ValueError):unpack_descriptors(packed,pool=pool)


def test_complete_supplied_checkpoint_roots_are_preserved_and_old_receipt_schema_is_not_reused(monkeypatch):
    tf,runtime,hz,_=execute(monkeypatch)
    untouched=np.arange(7,dtype=np.float64);roots=numeric_roots(runtime)
    # C32's SYNTHETIC fixture uses SimpleNamespace for its layer, unlike the
    # real checkpoint's registered Layer. Expose all these toy scalar fields;
    # do not relax the production closed-schema collector.
    roots['layer']=dict(vars(roots['layer']))
    saved=dict(schema='c34_reconstructable_final_native_checkpoint_v1',spliced_state_fields=export(runtime['lifted']),
        runtime_numeric_roots=roots,final_hz=hz,hz_cache={78:hz},untouched_payload=untouched)
    pool=WorkPool(256_000_000);candidate=checkpoint_payload(saved,hz,hz,hz,np.array([1,2],np.uint64),
        np.array([],np.uint64),pool=pool)
    assert candidate['untouched_payload'] is untouched and saved['spliced_state_fields']['hz'] is hz
    assert 'spliced_state_fields' not in candidate and 'runtime_numeric_roots' not in candidate
    assert 'hz' not in candidate['original_splice_fields_without_current_HZ']
    assert candidate['schema']!='c34_reconstructable_final_native_checkpoint_v1'
    report=checkpoint_closure(saved,candidate,pool=pool)
    assert report['numeric_resident_byte_delta']==16 and not report['strict_checkpoint_numeric_decrease']
    assert report['original_caller_roots_missing_from_archive'] and not report['whole_C34_LIVE_gate_proved']


def test_no_budget_for_a_codec_is_not_free_metadata():
    *_,desc,runs,report,inv,pool,degrees=actual()
    with pytest.raises(MemoryError):pack_descriptors(desc,pool=WorkPool(0))


def test_worker_preserves_the_complete_strict_C39_restore_ceremony():
    root=Path(__file__).resolve().parent
    start="            freeze=json.loads((DIRECTORY/'preregistered.json').read_text())"
    stop="            emit(dict(event='whole_source_final_input_property_restored',charged_work=whole.used))"
    def section(name):
        text=(root/name).read_text();return text[text.index(start):text.index(stop)+len(stop)]
    assert section('c40_compact_half_gauge_worker_v1.py')==section('c39_half_alias_row_gauge_worker_v1.py')
