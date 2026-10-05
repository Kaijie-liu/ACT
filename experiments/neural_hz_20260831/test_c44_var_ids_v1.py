import pickle
import random
import numpy as np
import pytest
from experiments.neural_hz_20260831.c44_var_ids_v1 import VarIds,paid,span,snapshot
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool


@pytest.fixture
def scope():
    with paid(WorkPool(256_000_000)):yield


def test_default_off_and_unpaid_construction_are_rejected():
    assert span(object(),object()) is None and snapshot(object()) is None
    with pytest.raises(ValueError):span(0,3,enabled=True)
    with pytest.raises(TypeError):VarIds()


@pytest.mark.parametrize('seed',[0,1,11,53])
def test_all_indices_and_signed_slices_match_literal_values(scope,seed):
    rng=random.Random(seed);values=[10,11,12,30,31,100,101,102,103,104]
    compact=snapshot(values,enabled=True)
    for i in range(-len(values),len(values)):assert compact[i]==values[i] and type(compact[i]) is int
    for _ in range(100):
        a=rng.choice([None,*range(-15,15)]);b=rng.choice([None,*range(-15,15)]);d=rng.choice([-5,-2,-1,1,2,3])
        index=slice(a,b,d);assert list(compact[index])==values[index]
        assert list(compact[index][::-1])==values[index][::-1]
    for i in (-len(values)-1,len(values)):
        with pytest.raises(IndexError):_ = compact[i]


def test_copy_concat_snapshot_repeat_and_container_identities(scope):
    a=span(5,4,enabled=True);b=span(9,3,enabled=True);old=[40,41]
    assert list(a+b)==list(range(5,12)) and len((a+b).runs)==1
    assert list(old+a)==[40,41,5,6,7,8] and list(a+old)==[5,6,7,8,40,41]
    snapshot_old=snapshot(old,enabled=True);old[0]=99;assert list(snapshot_old)==[40,41]
    copied=a.copy();assert copied==a and copied is not a and a[:] is not a
    assert list(3*a)==list(a)*3 and list(a*-1)==[]
    assert a==[5,6,7,8] and [5,6,7,8]==a and a!=(5,6,7,8)
    with pytest.raises(TypeError):a.size=0
    with pytest.raises(TypeError):a[0]=7
    with pytest.raises(AttributeError):a.append(9)


def test_queries_and_numpy_materialization(scope):
    a=span(100,10,enabled=True)[::-2]+span(105,3,enabled=True)
    values=list(a)
    for v in range(98,112):
        assert (v in a)==(v in values)
        if v in values:assert a.index(v)==values.index(v)
        else:
            with pytest.raises(ValueError):a.index(v)
    assert a.maximum()==max(values) and np.array_equal(np.asarray(a,dtype=np.int64),np.array(values))
    assert a.index(105,3)==values.index(105,3)
    with pytest.raises(ValueError):a.index(105,0,1)


def test_different_run_partitions_compare_by_values(scope):
    a=span(0,7,enabled=True)[::2];b=snapshot([0,2,4,6],enabled=True)
    assert a==b and not (a==span(0,4,enabled=True))


def test_pickle_preserves_outer_aliases_and_validates_geometry(scope):
    a=span(10,1000000,enabled=True);raw=pickle.dumps([a,a,a.copy()],protocol=5)
    restored=pickle.loads(raw)
    assert len(raw)<200 and restored[0] is restored[1] and restored[0] is not restored[2]
    assert restored[0]==a
    object.__setattr__(restored[0],'size',2)
    with pytest.raises(ValueError):restored[0].validate()


@pytest.mark.parametrize('kind',['negative','bool','too_many','fields','size','budget'])
def test_unknown_unpaid_or_mutated_descriptor_fails_closed(scope,kind):
    if kind=='budget':
        with paid(WorkPool(0)),pytest.raises(MemoryError):span(0,2,enabled=True)
        return
    if kind in ('negative','bool','too_many'):
        args=(-1,2) if kind=='negative' else (True,2) if kind=='bool' else (0,64_000_001)
        with pytest.raises((ValueError,MemoryError)):span(*args,enabled=True)
        return
    a=span(0,3,enabled=True);object.__setattr__(a,'hidden' if kind=='fields' else 'size',np.zeros(3) if kind=='fields' else 10)
    with pytest.raises(ValueError):a.validate()
