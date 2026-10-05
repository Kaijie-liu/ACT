import pytest
import torch
from torch import nn

from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _model_float_dtype


def test_parameter_model_keeps_existing_dtype():
    assert _model_float_dtype(nn.Linear(2, 1).double()) == torch.float64


def test_buffer_only_converted_model_has_a_dtype():
    model = nn.Module()
    model.register_buffer("weights", torch.ones(3, dtype=torch.float64))
    model.register_buffer("indices", torch.arange(3))
    assert list(model.parameters()) == []
    assert _model_float_dtype(model) == torch.float64


def test_payload_free_model_has_explicit_converter_default():
    assert _model_float_dtype(nn.Identity()) == torch.float64


def test_mixed_floating_payloads_fail_closed():
    model = nn.Linear(2, 1).double()
    model.register_buffer("other_weights", torch.ones(2, dtype=torch.float32))
    with pytest.raises(ValueError, match="mixed floating"):
        _model_float_dtype(model)
