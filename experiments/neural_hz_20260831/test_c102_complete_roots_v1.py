"""Full original C5 oracle: every field, token, identity and numeric owner."""
from collections import OrderedDict
from dataclasses import asdict
from types import SimpleNamespace
import tracemalloc
import weakref
import numpy as np
import pytest
import torch
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect as original
from experiments.neural_hz_20260831.c102_complete_roots_v1 import collect
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.test_c74_native_binding_v1 import native


def compare(tf,extra=None):
    old=original(tf,extra);events=[];pool=WorkPool(256_000_000)
    new=collect(tf,extra,pool=pool,enabled=True,observe=events.append)
    assert new.fingerprint==old.fingerprint
    assert new.schema_counts==old.schema_counts
    assert new.python_shallow_bytes==old.python_shallow_bytes
    assert new.unique_objects==old.unique_objects
    assert new.numeric.keys()==old.numeric.keys()
    assert all(new.numeric[k] is v for k,v in old.numeric.items())
    assert asdict(new.measure())==asdict(old.measure())
    assert events[-1]['event']=='complete_root_traversal_finished'
    return new,events,pool


@pytest.mark.parametrize('kind',['list','tuple','set','frozenset'])
@pytest.mark.parametrize('order',['ascending','descending','repeated'])
def test_integer_container_values_identity_and_full_paths(kind,order):
    values=list(range(257,9500))
    if order=='descending':values.reverse()
    if order=='repeated':values=values[:30]*400
    ctor={'list':list,'tuple':tuple,'set':set,'frozenset':frozenset}[kind]
    v=ctor(values)
    compare(SimpleNamespace(metadata={'variables':v,'same':v,'nested':[v,(-1,False,True,3.5)]}))


def test_distinct_equal_int_atoms_and_mutable_aliases_are_not_merged():
    a=int('1000000');b=int('1000000');assert a==b and a is not b
    first=[a,b,a];second=[a,b,a]
    compare(SimpleNamespace(state={'first':first,'same':first,'different':second,
        'scalar_a':a,'scalar_b':b,'bool':True,'numpy':np.int64(a),
        'large_positive':2**80,'large_negative':-(2**80)}))


def test_mixed_unicode_nested_metadata_and_scalar_fields():
    values=list(range(350,1800))+[False,None,b'blob','变量',2.5,np.float64(2.5)]+list(range(300,5000))
    compare(SimpleNamespace(**{"quote'and\\\"":{('tuple',3):values,'scalar':1024},'single':7}),
        {'outside':OrderedDict(mixed=values,tuple=(values,))})


def test_full_numeric_and_real_state_dict_fields_and_weak_roots():
    model=torch.nn.Linear(4,2).double();state=model.state_dict(keep_vars=True)
    a=np.arange(80,dtype=np.float64);a.flags.writeable=False
    weak=weakref.WeakValueDictionary(held=model)
    tf=SimpleNamespace(state=state,arrays=[a,a],ids=list(range(300,900)),weak=weak)
    one,_,_=compare(tf)
    state._metadata['']['version']+=1
    two,_,_=compare(tf)
    assert one.fingerprint!=two.fingerprint


def test_complete_fresh_source_native_fixture_and_all_actual_roots():
    state,_,_=native()
    roots=state.numeric_roots()
    compare(SimpleNamespace(),{'full_native':roots,'cache':[state.hz,state.source.hz],
        'variable_lists':[list(range(300,5000)),tuple(range(300,5000))]})


@pytest.mark.parametrize('kind',['cycle','unknown','ordered_attribute'])
def test_original_unknown_schema_and_cycle_rejections_remain(kind):
    if kind=='cycle':value=[];value.append(value)
    elif kind=='unknown':value=object()
    else:value=OrderedDict();value.hidden=np.ones(4)
    tf=SimpleNamespace(value=value)
    with pytest.raises(ValueError):original(tf)
    with pytest.raises(ValueError):collect(tf,pool=WorkPool(256_000_000),enabled=True)


def test_default_off_and_prepaid_budget_failure():
    assert collect(None) is None
    with pytest.raises(MemoryError):collect(SimpleNamespace(ids=[500]),pool=WorkPool(0),enabled=True)


def test_integer_workspaces_are_in_active_tracer_and_input_is_unchanged():
    values=list(range(300,20000));snapshot=tuple(values)
    assert not tracemalloc.is_tracing()
    tracemalloc.start(1)
    try:
        events=[];collect(SimpleNamespace(ids=values),pool=WorkPool(256_000_000),enabled=True,observe=events.append)
        current,peak=tracemalloc.get_traced_memory()
        assert peak>=262144+events[-1]['identity_page_bytes']
    finally:tracemalloc.stop()
    assert tuple(values)==snapshot and all(a is b for a,b in zip(values,snapshot))
