#!/usr/bin/env python3
#===- tests/test_torch2act_meta_filtering.py - Torch2ACT Meta ---------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

import torch

from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.specs import InKind, OutKind, InputSpec, OutputSpec
from act.front_end.verifiable_model import InputLayer, InputSpecLayer, OutputSpecLayer
from act.pipeline.verification.torch2act import TorchToACT, _filter_layer_meta


def _make_wrapped_conv() -> torch.nn.Sequential:
    x = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
    labeled = LabeledInputTensor(x, label=0)
    input_layer = InputLayer(labeled, shape=(1, 1, 4, 4), dtype=torch.float64)
    input_spec = InputSpec(kind=InKind.BOX, lb=x.clone(), ub=(x + 1.0))
    input_spec_layer = InputSpecLayer(spec=input_spec)

    conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=True)
    relu = torch.nn.ReLU()
    flatten = torch.nn.Flatten()
    linear = torch.nn.Linear(4, 2)

    output_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=0)
    output_spec_layer = OutputSpecLayer(spec=output_spec)

    return torch.nn.Sequential(
        input_layer,
        input_spec_layer,
        conv,
        relu,
        flatten,
        linear,
        output_spec_layer,
    )


def test_torch2act_filters_unknown_meta_keys() -> None:
    wrapped = _make_wrapped_conv()
    TorchToACT(wrapped).run()


def test_filter_layer_meta_keeps_required_keys() -> None:
    meta = {
        "input_shape": (1, 1, 4, 4),
        "output_shape": (1, 1, 2, 2),
        "torch_path": "0",
        "torch_type": "Conv2d",
        "flat_in_dim": 16,
        "flat_out_dim": 4,
        "logical_shape": (1, 1, 4, 4),
        "converter": "torch2act",
    }
    filtered = _filter_layer_meta("CONV2D", meta)
    assert filtered["input_shape"] == meta["input_shape"]
    assert filtered["output_shape"] == meta["output_shape"]
    assert "torch_path" not in filtered
    assert "converter" not in filtered


def test_filter_layer_meta_unknown_kind_raises() -> None:
    try:
        _filter_layer_meta("UNKNOWN_KIND", {"input_shape": (1,), "output_shape": (1,)})
    except KeyError:
        return
    raise AssertionError("Expected KeyError for unknown kind")


def test_conv2d_padding_same_rejected() -> None:
    x = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
    labeled = LabeledInputTensor(x, label=0)
    input_layer = InputLayer(labeled, shape=(1, 1, 4, 4), dtype=torch.float64)
    input_spec = InputSpec(kind=InKind.BOX, lb=x.clone(), ub=(x + 1.0))
    input_spec_layer = InputSpecLayer(spec=input_spec)

    conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding="same", bias=True)
    relu = torch.nn.ReLU()
    flatten = torch.nn.Flatten()
    linear = torch.nn.Linear(16, 2)

    output_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=0)
    output_spec_layer = OutputSpecLayer(spec=output_spec)

    wrapped = torch.nn.Sequential(
        input_layer,
        input_spec_layer,
        conv,
        relu,
        flatten,
        linear,
        output_spec_layer,
    )

    try:
        TorchToACT(wrapped).run()
    except ValueError as e:
        assert "padding string not supported" in str(e)
        return
    raise AssertionError("Expected ValueError for padding='same'")


def test_maxpool2d_meta_has_shapes() -> None:
    x = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
    labeled = LabeledInputTensor(x, label=0)
    input_layer = InputLayer(labeled, shape=(1, 1, 4, 4), dtype=torch.float64)
    input_spec = InputSpec(kind=InKind.BOX, lb=x.clone(), ub=(x + 1.0))
    input_spec_layer = InputSpecLayer(spec=input_spec)

    pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    flatten = torch.nn.Flatten()
    linear = torch.nn.Linear(4, 2)

    output_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=0)
    output_spec_layer = OutputSpecLayer(spec=output_spec)

    wrapped = torch.nn.Sequential(
        input_layer,
        input_spec_layer,
        pool,
        flatten,
        linear,
        output_spec_layer,
    )

    net = TorchToACT(wrapped).run()
    pools = [layer for layer in net.layers if layer.kind == "MAXPOOL2D"]
    assert len(pools) == 1
    assert "input_shape" in pools[0].meta
    assert "output_shape" in pools[0].meta


def test_avgpool2d_meta_has_shapes() -> None:
    x = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
    labeled = LabeledInputTensor(x, label=0)
    input_layer = InputLayer(labeled, shape=(1, 1, 4, 4), dtype=torch.float64)
    input_spec = InputSpec(kind=InKind.BOX, lb=x.clone(), ub=(x + 1.0))
    input_spec_layer = InputSpecLayer(spec=input_spec)

    pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
    flatten = torch.nn.Flatten()
    linear = torch.nn.Linear(4, 2)

    output_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=0)
    output_spec_layer = OutputSpecLayer(spec=output_spec)

    wrapped = torch.nn.Sequential(
        input_layer,
        input_spec_layer,
        pool,
        flatten,
        linear,
        output_spec_layer,
    )

    net = TorchToACT(wrapped).run()
    pools = [layer for layer in net.layers if layer.kind == "AVGPOOL2D"]
    assert len(pools) == 1
    assert "input_shape" in pools[0].meta
    assert "output_shape" in pools[0].meta


def test_adaptiveavgpool2d_meta_has_shapes() -> None:
    x = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
    labeled = LabeledInputTensor(x, label=0)
    input_layer = InputLayer(labeled, shape=(1, 1, 4, 4), dtype=torch.float64)
    input_spec = InputSpec(kind=InKind.BOX, lb=x.clone(), ub=(x + 1.0))
    input_spec_layer = InputSpecLayer(spec=input_spec)

    pool = torch.nn.AdaptiveAvgPool2d((2, 2))
    flatten = torch.nn.Flatten()
    linear = torch.nn.Linear(4, 2)

    output_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=0)
    output_spec_layer = OutputSpecLayer(spec=output_spec)

    wrapped = torch.nn.Sequential(
        input_layer,
        input_spec_layer,
        pool,
        flatten,
        linear,
        output_spec_layer,
    )

    net = TorchToACT(wrapped).run()
    pools = [layer for layer in net.layers if layer.kind == "AVGPOOL2D"]
    assert len(pools) == 1
    assert "input_shape" in pools[0].meta
    assert "output_shape" in pools[0].meta
    assert pools[0].meta.get("output_size") == (2, 2)
