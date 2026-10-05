"""Eight ordinary custody tests; unchanged strict C62/C5 ledger semantics."""
import numpy as np
import pytest

from experiments.neural_hz_20260831 import c129_owned_evidence_v1 as owned
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.s0_c2_whole_state_ledger_prototype import WholeStateReject
from experiments.neural_hz_20260831.test_c5_live_value_contraction_v1 import source


class Pool:
    def __init__(self, cap=256000000):
        self.cap, self.used, self.calls = cap, 0, []

    def charge(self, label, amount):
        assert type(label) is str and type(amount) is int and amount >= 0
        self.calls.append((label, amount))
        if self.used + amount > self.cap:
            raise MemoryError("test pool exhausted before allocation")
        self.used += amount


def _all_source_arrays(value):
    arrays = {name: getattr(value, name) for name in ("c", "b", "ub")}
    for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
        for part in ("data", "indices", "indptr"):
            arrays[name+"_"+part] = getattr(getattr(value, name), part)
    return arrays


def _assert_copy(actual, original):
    assert type(actual) is np.ndarray
    assert actual.dtype == original.dtype and actual.shape == original.shape
    assert actual.tobytes(order="C") == original.tobytes(order="C")
    assert actual.flags.owndata and actual.flags.c_contiguous and actual.base is None
    assert not actual.flags.writeable and not np.shares_memory(actual, original)


def test_full_nonconvex_live_source_and_detached_csr_evidence_pass_unchanged_ledger():
    live = source(8)
    before = source_digest(live)
    arrays = _all_source_arrays(live)
    baseline = numeric_layout({"live_source": live}, Pool())
    payment = Pool()
    snapshots = owned.snapshot(arrays, pool=payment)
    complete = numeric_layout({"live_source": live, "owned_evidence": snapshots}, Pool())
    assert set(snapshots) == set(arrays) and len(snapshots) == 21
    for name in arrays:
        _assert_copy(snapshots[name], arrays[name])
    assert complete.resident_bytes == baseline.resident_bytes + sum(a.nbytes for a in snapshots.values())
    assert complete.resident_entries == baseline.resident_entries + sum(a.size for a in snapshots.values())
    assert payment.used == 1024 + 8 * sum(a.size for a in arrays.values())
    assert source_digest(live) == before
    assert live.n_bin == 2 and live.n_eq == 1 and live.n_ineq == 1


def test_actual_bare_csr_alias_still_fails_the_unmodified_strict_ledger():
    live = source(8)
    before = source_digest(live)
    with pytest.raises(WholeStateReject, match="incompatible_storage_alias"):
        numeric_layout({"live_source": live, "raw_evidence": {"Gc_data": live.Gc.data}}, Pool())
    assert source_digest(live) == before


def test_all_ordinary_primitive_dtypes_shapes_and_signed_zero_bytes_are_preserved():
    arrays = {"bool": np.array([[False, True], [True, False]], dtype=bool)}
    for dtype in (np.int8, np.int16, np.int32, np.int64):
        arrays[np.dtype(dtype).str] = np.array([[-2, 0], [1, 3]], dtype=dtype)
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        arrays[np.dtype(dtype).str] = np.array([[0, 3], [1, 7]], dtype=dtype)
    for dtype in (np.float16, np.float32, np.float64):
        arrays[np.dtype(dtype).str] = np.array([[-0., 0.], [-1.5, .125]], dtype=dtype)
    payment = Pool()
    snapshots = owned.snapshot(arrays, pool=payment)
    assert set(snapshots) == set(arrays)
    for name, original in arrays.items():
        _assert_copy(snapshots[name], original)
        if original.dtype.kind == "f":
            assert np.signbit(snapshots[name][0, 0])
            assert not np.signbit(snapshots[name][0, 1])
    assert payment.used == 1024 + 8 * sum(a.size for a in arrays.values())


def test_shared_noncontiguous_inputs_get_distinct_owned_snapshots_without_source_mutation():
    storage = np.arange(12, dtype=np.float64).reshape(3, 4)
    arrays = {"first": storage, "again": storage, "strided": storage[:, ::2]}
    before = storage.copy()
    snapshots = owned.snapshot(arrays, pool=Pool())
    assert np.array_equal(storage, before) and storage.flags.writeable
    for name, original in arrays.items():
        _assert_copy(snapshots[name], original)
    names = list(snapshots)
    for index, left in enumerate(names):
        for right in names[index+1:]:
            assert not np.shares_memory(snapshots[left], snapshots[right])
    storage[:] = -9
    assert np.array_equal(snapshots["first"], before)
    assert np.array_equal(snapshots["again"], before)
    assert np.array_equal(snapshots["strided"], before[:, ::2])
    with pytest.raises(ValueError):
        snapshots["first"][0, 0] = 7


def test_empty_arrays_and_scalar_shape_remain_complete_owned_fields():
    arrays = {"empty_float": np.empty((0,), dtype=np.float64),
              "empty_indices": np.empty((2, 0, 3), dtype=np.int32),
              "scalar": np.array(4, dtype=np.int64)}
    payment = Pool()
    snapshots = owned.snapshot(arrays, pool=payment)
    assert set(snapshots) == set(arrays)
    for name, original in arrays.items():
        _assert_copy(snapshots[name], original)
    assert snapshots["scalar"].shape == () and payment.used == 1032


def test_both_header_and_full_copy_fee_fail_before_any_snapshot_allocation(monkeypatch):
    arrays = {"data": np.arange(4, dtype=np.float64)}
    original = arrays["data"].copy()
    def forbidden(*args, **kwargs):
        raise AssertionError("unpaid evidence allocation")
    monkeypatch.setattr(owned.np, "array", forbidden)
    for cap, expected_used in ((1023, 0), (1024+8*4-1, 1024)):
        payment = Pool(cap)
        with pytest.raises(MemoryError):
            owned.snapshot(arrays, pool=payment)
        assert payment.used == expected_used
    assert np.array_equal(arrays["data"], original)


def test_unsupported_containers_keys_and_array_payloads_reject_without_copying(monkeypatch):
    class ArraySubclass(np.ndarray):
        pass
    class StringSubclass(str):
        pass
    ordinary = np.array([1., 2.])
    invalid = [[], {}, {1: ordinary}, {StringSubclass("not_exact_str"): ordinary},
               {"list": [1., 2.]}, {"subclass": ordinary.view(ArraySubclass)},
               {"object": np.array([object()], dtype=object)},
               {"complex": np.array([1.+2.j])}]
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported payload reached numeric copying")
    monkeypatch.setattr(owned.np, "array", forbidden)
    for arrays in invalid:
        with pytest.raises(ValueError):
            owned.snapshot(arrays, pool=Pool())


def test_original_header_caps_reject_tiny_inputs_before_copy_allocation(monkeypatch):
    arrays = {"data": np.arange(4, dtype=np.float64)}
    def forbidden(*args, **kwargs):
        raise AssertionError("over-limit header reached numeric copying")
    for name, limit in (("MAX_ARRAYS", 0), ("MAX_ENTRIES", 3), ("MAX_BYTES", 31)):
        with monkeypatch.context() as patch:
            patch.setattr(owned, name, limit)
            patch.setattr(owned.np, "array", forbidden)
            with pytest.raises((MemoryError, ValueError)):
                owned.snapshot(arrays, pool=Pool())
