"""Ordinary normalized-expression proofs and complete existing source fixtures."""
from fractions import Fraction as F
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c60_normalized_carrier_cost_v1 import recipe,CarrierCensus
from experiments.neural_hz_20260831.c60_source_native_census_v1 import assess
from experiments.neural_hz_20260831.c59_full_consumer_census_v2 import assess as old_assess
from experiments.neural_hz_20260831.c58_source_inverse_census_v2 import assess as inverse
from experiments.neural_hz_20260831.test_c53_logical_singleton_census_v2 import complete


@pytest.mark.parametrize('m,k',[(2**53+3,0),(2**76+43,0),(2**75+2**37+5,1),(2**76+11,4)])
def test_actual_digit_recurrence_and_all_signed_root_boxes(m,k):
    digits=recipe(m,k,WorkPool(256_000_000));coefficient=F(0)
    for digit,width in digits:
        coefficient=(coefficient+F(digit,2**k))/2**width
        assert abs(coefficient)*2**k<=1
    assert coefficient==F(m,2**(m.bit_length()+k))
    assert sum(d << sum(w for _,w in digits[:i]) for i,(d,w) in enumerate(digits))==m


def fresh():
    return CarrierCensus(np.array([0,1]),[(1,0),(1,0)],np.array([],bool),WorkPool(256_000_000))


def test_exponent_and_global_sign_share_only_the_normalized_expression():
    c=fresh();m=2**76+43
    c.inspect_row({0:(m,-77),1:(-m,-77)},np.array([]),np.array([]),0.,kind='Ac',index=0)
    c.inspect_row({0:(-m,-76),1:(m,-76)},np.array([]),np.array([]),0.,kind='Ac',index=1)
    assert len(c.plans)==1 and len(c.legacy_keys)==2
    assert c.native['group_occurrences']==2 and c.native['long_coefficient_terms']==4
    assert c.native['auxiliary_continuous']==2 and c.native['auxiliary_nnz']==7
    assert c.native['native_gauge_incompatible_rows']==0


def test_different_original_root_vectors_are_not_shared():
    c=fresh();m=2**76+43
    c.inspect_row({0:(m,-77)},np.array([]),np.array([]),0.,kind='Ac',index=0)
    c.inspect_row({1:(m,-77)},np.array([]),np.array([]),0.,kind='Ac',index=1)
    assert len(c.plans)==2 and len(c.recipes)==1
    assert c.native['auxiliary_continuous']==4


def test_complete_auxiliary_nnz_includes_every_root_and_previous_digit():
    c=fresh();m=2**76+43
    c.inspect_row({0:(m,-77),1:(m,-77)},np.array([]),np.array([]),0.,kind='Ac',index=0)
    explicit_rows=[{2:1,0:1,1:1},{3:1,2:1,0:1,1:1}]
    assert c.native['auxiliary_nnz']==sum(map(len,explicit_rows))
    assert 2*c.native['auxiliary_nnz']+3*c.native['auxiliary_continuous']==20


@pytest.mark.parametrize('kind',['chain','shared','conv'])
def test_complete_unchanged_source_profile_and_total_long_population(kind):
    _,saved=complete(kind);pool=WorkPool(256_000_000)
    packet=inverse(saved,pool=pool,enabled=True)['general_scalar_diagnostic']['packet']
    original=old_assess(saved,packet,pool=pool,enabled=True)
    actual=assess(saved,packet,pool=pool,enabled=True);native=actual.pop('native_carrier_profile')
    assert original==actual
    assert native['counts']['long_coefficient_terms']==actual['counts'].get('derived_mantissa_over53',0)
    assert native['added_native_numeric_entries']==2*native['counts']['auxiliary_nnz']+3*native['counts']['auxiliary_continuous']
    assert not native['full_native_HZ_physical_or_source_first_proved']


def test_default_off_and_unchanged_work_rejection():
    assert assess(None,None,pool=None) is None
    with pytest.raises(MemoryError):recipe(2**76+43,0,WorkPool(0))
