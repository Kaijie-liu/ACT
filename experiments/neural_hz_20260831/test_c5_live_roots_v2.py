from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.neural_hz_20260831.c5_live_roots_v1 import collect as old_collect
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect


def test_real_state_dict_tensor_aliases_and_module_metadata_are_preserved():
    model = torch.nn.Linear(4, 2).double()
    state = model.state_dict(keep_vars=True)
    roots = {"state": state}
    tf = SimpleNamespace()
    with pytest.raises(ValueError, match="OrderedDict"):
        old_collect(tf, roots)
    first = collect(tf, roots)
    assert any(value is model.weight for value in first.numeric.values())
    assert any(value is model.bias for value in first.numeric.values())
    state._metadata[""]["version"] += 1
    assert collect(tf, roots).fingerprint != first.fingerprint


def test_nested_ordered_dicts_deduplicate_actual_buffer_owners():
    a = np.arange(5, dtype=np.float64)
    tf = SimpleNamespace(state=OrderedDict(a=a, nested=OrderedDict(b=a)))
    result = collect(tf)
    assert result.measure().resident_bytes == a.nbytes


def test_unregistered_ordered_attributes_are_not_ignored():
    state = OrderedDict()
    state.hidden = np.ones(8)
    with pytest.raises(ValueError, match="OrderedDict attributes"):
        collect(SimpleNamespace(state=state))
