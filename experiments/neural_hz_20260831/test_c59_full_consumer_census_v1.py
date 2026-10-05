"""Whole ordinary source fixtures and independent exact forward equations."""
from fractions import Fraction as F
from collections import defaultdict
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c58_source_inverse_census_v2 import assess as inverse
from experiments.neural_hz_20260831.c57_real_scalar_census_v2 import assess as legacy
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import fraction
from experiments.neural_hz_20260831.c59_full_consumer_census_v1 import assess,decode_frame,Census,gauge,Products
from experiments.neural_hz_20260831.test_c53_logical_singleton_census_v2 import complete


@pytest.mark.parametrize('values,expected',[
    ([(1,-38),(1,0)],18), ([(3,-39),(1,22)],18), ([(3,-39),(3,21)],None)])
def test_exact_positive_row_gauge_interval(values,expected):
    low,high,shift=gauge(values)
    assert shift==expected
    if shift is not None:
        for value in values:
            a=abs(fraction(value)*F(2)**shift)
            assert F(1,2**20)<=a<=2**40
    else:assert low>high


@pytest.mark.parametrize('kind',['chain','shared','conv'])
def test_complete_original_consumer_profile_and_fraction_rows(kind):
    _,saved=complete(kind);pool=WorkPool(256_000_000)
    packet=inverse(saved,pool=pool,enabled=True)['general_scalar_diagnostic']['packet']
    result=assess(saved,packet,pool=pool,enabled=True)
    old=legacy(saved,pool=pool,enabled=True)['general_scalar_diagnostic']
    assert result['symbolic_predicate_nnz']==old['symbolic_exact_predicate_nnz']
    assert result['symbolic_removed_continuous']==old['counts'].get('singletons',0)
    assert result['original_binary_retained']==saved['hz'].n_bin
    roots,weights,_=decode_frame(packet,pool)
    census=Census(roots,weights,np.zeros(saved['hz'].n_eq,bool),pool)
    for name in ('Ac','Auc','Gc'):
        matrix=getattr(saved['hz'],name)
        for r in range(matrix.shape[0]):
            a,b=map(int,matrix.indptr[r:r+2]);cols=matrix.indices[a:b];data=matrix.data[a:b]
            got,mask,delta=census.translate(cols,data)
            actual={c:fraction(v) for c,v in got.items()}
            actual.update({int(c):F(float(v)) for c,v in zip(cols[mask],data[mask])})
            expected=defaultdict(F)
            for c,v in zip(cols,data):expected[int(roots[c])]+=F(float(v))*fraction(weights[c])
            expected={c:v for c,v in expected.items() if v}
            assert actual==expected and len(actual)==len(cols)+delta
    assert not result['physical_HZ_or_native_realization_proved']


@pytest.mark.parametrize('which',['binary','rhs'])
def test_binary_and_rhs_cannot_be_omitted_from_gauge(which):
    census=Census(np.array([0]),[(1,0)],np.array([],bool),WorkPool(256_000_000))
    binary=np.array([2.**23]) if which=='binary' else np.array([])
    rhs=2.**23 if which=='rhs' else 0.
    assert gauge([(3,-39)])[2]==18
    assert census.inspect_row({0:(3,-39)},np.array([]),binary,rhs,kind='Ac',index=0)[2] is None


def test_shared_root_cancellation_retains_original_unaffected_term():
    census=Census(np.array([0,0,0,3]),[(1,0),(3,-2),(3,-2),(1,0)],np.array([],bool),WorkPool(256_000_000))
    got,mask,delta=census.translate(np.array([1,2,3]),np.array([.7,-.7,.9]))
    assert got=={} and mask.tolist()==[False,False,True] and delta==-2


def test_canonical_product_reuse_is_exact():
    products=Products(WorkPool(256_000_000));a=(3,-2);b=(7,-3)
    assert products(a,b)==products(a,b)==(21,-5)
    assert products.lookups==2 and products.hits==1 and len(products.values)==1


def test_default_off_and_budget_reject():
    assert assess(None,None,pool=None) is None
    with pytest.raises(MemoryError):Products(WorkPool(0))((3,-2),(7,-3))


def test_changed_defining_equation_rejects_before_consumers():
    _,saved=complete();pool=WorkPool(256_000_000)
    packet=inverse(saved,pool=pool,enabled=True)['general_scalar_diagnostic']['packet']
    roots,weights,_=decode_frame(packet,pool)
    slot=int(np.flatnonzero(roots!=np.arange(len(roots)))[0])
    index=int(saved['eq_roots'][saved['old_n_eq']+slot-saved['old_n_cont']])
    saved['hz'].b[index]=1.
    with pytest.raises(ValueError):assess(saved,packet,pool=pool,enabled=True)


def test_product_precision_remains_bounded():
    with pytest.raises(MemoryError):Products(WorkPool(256_000_000))(((1<<300)+1,-301),((1<<300)+1,-301))
