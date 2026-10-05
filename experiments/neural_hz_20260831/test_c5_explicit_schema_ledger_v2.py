from dataclasses import asdict
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.specs import InputSpec, OutputSpec
from experiments.neural_hz_20260831.c5_explicit_schema_ledger_v2 import parameter_roots, snapshot_known_buffers
from experiments.neural_hz_20260831.s0_c2_whole_state_ledger_prototype import WholeStateRoots, WholeStateReject, snapshot_whole_state
from experiments.neural_hz_20260831.test_c5_live_value_contraction_v1 import source


def net(params):
    return SimpleNamespace(layers=[SimpleNamespace(id=0, params=params)])


def test_complete_known_schemas_keep_tensor_label_and_spec_aliases():
    x = torch.arange(8, dtype=torch.float64)
    label = torch.tensor([1])
    inp = InputSpec(kind="BOX", lb=x, ub=x)
    out = OutputSpec(kind="RANGE", lb=x, ub=x)
    roots = parameter_roots(net({"input": LabeledInputTensor(x, label), "spec": inp, "output": out}))
    assert sum(value is x for value in roots.values()) == 5
    assert sum(value is label for value in roots.values()) == 1
    ledger = snapshot_known_buffers(WholeStateRoots(active=roots))
    assert ledger.resident_bytes == x.untyped_storage().nbytes() + label.untyped_storage().nbytes() + inp.p_norm.untyped_storage().nbytes()


@pytest.mark.parametrize("mode", ["unknown", "extra", "missing", "cycle", "subclass"])
def test_unregistered_schema_and_cycles_reject(mode):
    value = LabeledInputTensor(torch.zeros(1))
    if mode == "unknown":
        value = SimpleNamespace(tensor=torch.zeros(1))
    elif mode == "extra":
        value.hidden = torch.zeros(1)
    elif mode == "missing":
        del value.label
    elif mode == "cycle":
        value = []
        value.append(value)
    else:
        class Child(LabeledInputTensor):
            pass
        value = Child(torch.zeros(1))
    with pytest.raises(ValueError):
        parameter_roots(net({"value": value}))


def test_protocol5_buffers_charge_actual_backing_once_without_copy():
    x = pickle.loads(pickle.dumps(np.arange(12, dtype=np.float64), protocol=5))
    roots = WholeStateRoots(active={"first": x, "same": x.reshape(3, 4)})
    with pytest.raises(WholeStateReject, match="external_buffer"):
        snapshot_whole_state(roots)
    result = snapshot_known_buffers(roots)
    assert result.resident_bytes == x.nbytes and result.resident_entries == x.size
    assert result.numeric_storage_count == 1


def test_disjoint_small_views_charge_complete_bytearray_owner():
    backing = bytearray(128)
    full = np.frombuffer(backing, dtype=np.float64)
    result = snapshot_known_buffers(WholeStateRoots(active={"a": full[:2], "b": full[4:6]}))
    assert result.resident_bytes == 128 and result.resident_entries == 16


@pytest.mark.parametrize("mode", ["overlap", "dtype", "gapped", "bytes"])
def test_buffer_rejections_remain_fail_closed(mode):
    full = np.frombuffer(bytearray(128), dtype=np.float64)
    if mode == "overlap":
        roots = {"a": full[:10], "b": full[5:]}
    elif mode == "dtype":
        roots = {"a": full, "b": full.view(np.int64)}
    elif mode == "gapped":
        roots = {"a": full[::2]}
    else:
        roots = {"a": np.frombuffer(bytes(128), dtype=np.float64)}
    with pytest.raises(WholeStateReject):
        snapshot_known_buffers(WholeStateRoots(active=roots))


def test_original_native_owner_ledger_is_unchanged():
    x = np.arange(12, dtype=np.float64)
    roots = WholeStateRoots(active={"x": x, "y": x.reshape(3, 4)})
    assert asdict(snapshot_known_buffers(roots)) == asdict(snapshot_whole_state(roots))


def test_full_serialized_hz_including_empty_csr_arrays():
    hz = source(8)
    hz.Gb.data[:] = 0
    hz.Gb.eliminate_zeros()
    serialized = pickle.loads(pickle.dumps(hz, protocol=5))
    old = snapshot_whole_state(WholeStateRoots(sparse_hz={0: hz}))
    new = snapshot_known_buffers(WholeStateRoots(sparse_hz={0: serialized}))
    assert old.resident_bytes == new.resident_bytes
    assert old.resident_entries == new.resident_entries
