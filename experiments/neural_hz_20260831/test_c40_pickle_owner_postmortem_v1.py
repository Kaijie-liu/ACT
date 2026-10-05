"""Synthetic post-exit diagnosis only; not part of C40's frozen target run.

Do not weaken the owner ledger or reload either actual archived HZ here.
An explicit tiny copy illustrates a future paid restoration obligation;
it is not an implementation of source-bound checkpoint restoration.
"""
import hashlib
import pickle
import numpy as np
import pytest
from experiments.neural_hz_20260831.c5_partial_csr_owner_ledger_v3 import snapshot_partial_csr_owners
from experiments.neural_hz_20260831.s0_c2_whole_state_ledger_prototype import WholeStateRoots,WholeStateReject


def ledger(a):
    return snapshot_partial_csr_owners(WholeStateRoots(active={'phase_events':a},consumer_gc_enabled=False))


def readonly_source():
    a=np.arange(200,dtype=np.uint64);a.flags.writeable=False
    return a


def test_protocol5_readonly_numeric_roundtrip_reproduces_exact_owner_rejection():
    a=readonly_source();decoded=pickle.loads(pickle.dumps(a,protocol=5))
    assert ledger(a).resident_bytes==1600
    assert np.array_equal(a,decoded) and not decoded.flags.writeable
    owner=decoded
    while isinstance(owner,np.ndarray):owner=owner.base
    assert type(owner) is bytes
    with pytest.raises(WholeStateReject,match='unknown_numpy_external_buffer'):
        ledger(decoded)


def test_explicit_owned_copy_preserves_readonly_bits_and_pays_numeric_storage():
    a=readonly_source();decoded=pickle.loads(pickle.dumps(a,protocol=5))
    owned=decoded.copy();owned.flags.writeable=False
    assert owned.flags.owndata and owned.base is None and not owned.flags.writeable
    assert hashlib.sha256(a.tobytes()).digest()==hashlib.sha256(owned.tobytes()).digest()
    assert not decoded.flags.owndata and not decoded.flags.writeable
    result=ledger(owned)
    assert (result.resident_bytes,result.resident_entries)==(1600,200)


def test_writable_protocol5_control_uses_existing_registered_buffer_adapter():
    a=np.arange(200,dtype=np.uint64);decoded=pickle.loads(pickle.dumps(a,protocol=5))
    assert np.array_equal(a,decoded) and decoded.flags.writeable
    result=ledger(decoded)
    assert (result.resident_bytes,result.resident_entries)==(1600,200)
