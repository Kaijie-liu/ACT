from dataclasses import asdict

import numpy as np
import pytest
import scipy.sparse as sp

from experiments.neural_hz_20260831.c5_explicit_schema_ledger_v2 import snapshot_known_buffers
from experiments.neural_hz_20260831.c5_partial_csr_owner_ledger_v3 import snapshot_partial_csr_owners
from experiments.neural_hz_20260831.s0_c2_whole_state_ledger_prototype import WholeStateRoots, WholeStateReject


def matrix(values):
    out = sp.csr_matrix((values, np.arange(values.size, dtype=np.int32), np.array([0, values.size], dtype=np.int32)),
                        shape=(1, max(1, values.size)), copy=False)
    # This SciPy constructor may copy short inputs despite copy=False. Install
    # and assert the public view to test the intended retained-owner state.
    out.data = values
    assert out.data is values
    return out


@pytest.mark.parametrize("serialized", [False, True])
def test_short_csr_keeps_full_owner_bytes_and_active_entries(serialized):
    owner = np.frombuffer(bytearray(128), dtype=np.float64) if serialized else np.zeros(16)
    value = matrix(owner[2:5])
    roots = WholeStateRoots(active={"value": value})
    with pytest.raises(WholeStateReject, match="full_.*owner"):
        snapshot_known_buffers(roots)
    measured = snapshot_partial_csr_owners(roots)
    assert measured.resident_bytes == 128 + value.indices.nbytes + value.indptr.nbytes
    assert measured.resident_entries == 3
    assert value.data.base is owner or np.shares_memory(value.data, owner)


@pytest.mark.parametrize("serialized", [False, True])
def test_identical_and_disjoint_csr_views_union_entries(serialized):
    owner = np.frombuffer(bytearray(128), dtype=np.float64) if serialized else np.zeros(16)
    a, b, c = matrix(owner[:3]), matrix(owner[5:7]), matrix(owner[:3])
    roots = WholeStateRoots(active={"a": a, "b": b, "c": c})
    result = snapshot_partial_csr_owners(roots)
    assert result.resident_bytes == 128 + sum(x.indices.nbytes + x.indptr.nbytes for x in (a, b, c))
    assert result.resident_entries == 5


@pytest.mark.parametrize("mode", ["overlap", "dtype", "dense", "gapped"])
def test_old_rejection_conditions_not_relaxed(mode):
    owner = np.zeros(16)
    a = matrix(owner[:3])
    if mode == "overlap":
        roots = {"a": a, "b": matrix(owner[1:5])}
    elif mode == "dtype":
        roots = {"a": a, "b": matrix(owner.view(np.int64)[:3])}
    elif mode == "dense":
        roots = {"a": a, "b": owner[:3]}
    else:
        # Deliberately retain a gapped public CSR payload after constructor.
        a.data = owner[::2]
        a.indices = np.arange(8, dtype=np.int32)
        a.indptr = np.array([0, 8], dtype=np.int32)
        roots = {"a": a}
    with pytest.raises(WholeStateReject):
        snapshot_partial_csr_owners(WholeStateRoots(active=roots))


def test_full_owner_ledger_unchanged():
    value = matrix(np.arange(5, dtype=np.float64))
    roots = WholeStateRoots(active={"value": value})
    assert asdict(snapshot_partial_csr_owners(roots)) == asdict(snapshot_known_buffers(roots))
