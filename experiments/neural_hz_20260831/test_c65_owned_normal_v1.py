"""Owned source-domain binding and independent general-reader equality."""
import numpy as np
import pytest
from experiments.neural_hz_20260831.c31_prepared_owned_rows_v1 import PreparedOwnedEncoder
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import Ledger
from experiments.neural_hz_20260831.c24_dense_rows_v1 import WorkPool
from experiments.neural_hz_20260831.c62_precision_plan_v1 import odd_significands
from experiments.neural_hz_20260831.c65_owned_normal_tracker_v1 import BirthTracker


def make():
    pool=WorkPool(0,0);ledger=Ledger(np.empty(0,np.int64),8,pool=pool)
    encoder=PreparedOwnedEncoder(8,1,200,pool=pool,ledger=ledger,radix_uid_base=100)
    tracker=BirthTracker(encoder,8,0,np.empty(0,np.int64))
    return encoder,tracker,pool


@pytest.mark.parametrize('packed',[False,True])
def test_actual_owned_main_and_radix_coefficients_match_general_reader(packed):
    encoder,tracker,pool=make()
    values=np.array([.75,-.375,2.**-80 if packed else 2.**-20,1.])
    encoder.encode_uid(7,np.array([0,1,2,3]),values,np.array([0]),np.array([.125]),.03125)
    for r,row in enumerate(encoder.eq):
        positions=np.arange(len(row[0]));got=tracker._owned_odd(r,positions)
        assert np.array_equal(got,odd_significands(row[1]))
    assert pool.parts['c65_owned_normal_domain_read']==8*len(encoder.eq)
    encoder.discard_unpublished_heads()
    with pytest.raises(ValueError,match='outside original owned emission'):tracker._owned_odd(0,np.array([0]))


def test_existing_emitted_rows_cannot_bind_a_fresh_producer():
    encoder,_,_=make()
    encoder.encode_uid(7,np.array([0]),np.array([.5]),np.empty(0,np.int64),np.empty(0),0.)
    with pytest.raises(ValueError,match='fresh empty'):BirthTracker(encoder,8,0,np.empty(0,np.int64))


@pytest.mark.parametrize('value',[0.,np.inf,np.nan])
def test_upstream_normal_domain_rejection_is_preserved(value):
    encoder,tracker,_=make()
    with pytest.raises(ValueError):encoder.encode_uid(7,np.array([0]),np.array([value]),np.empty(0,np.int64),np.empty(0),0.)
    assert not encoder.eq
    with pytest.raises(ValueError):tracker._owned_odd(0,np.array([0]))
