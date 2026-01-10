#!/usr/bin/env python3
#===- tests/test_layer_alignment_hookable_order_strict.py - Alignment --====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

import torch

from act.pipeline.verification.activations import ActivationEvent
from act.pipeline.verification.layer_alignment import align_activations_to_act_layers


class _Layer:
    def __init__(self, layer_id: int, kind: str, output_shape):
        self.id = layer_id
        self.kind = kind
        self.meta = {"output_shape": output_shape}


class _Net:
    def __init__(self, layers):
        self.layers = layers


def test_alignment_ok() -> None:
    net = _Net([
        _Layer(0, "DENSE", (1, 2)),
        _Layer(1, "RELU", (1, 2)),
    ])
    events = [
        ActivationEvent("l0", "Linear", 0, (1, 2), torch.zeros(1, 2)),
        ActivationEvent("l1", "ReLU", 0, (1, 2), torch.zeros(1, 2)),
    ]
    result = align_activations_to_act_layers(net, events)
    assert result.ok is True
    assert result.mapping[0].module_type == "Linear"


def test_alignment_count_mismatch() -> None:
    net = _Net([
        _Layer(0, "DENSE", (1, 2)),
        _Layer(1, "RELU", (1, 2)),
    ])
    events = [
        ActivationEvent("l0", "Linear", 0, (1, 2), torch.zeros(1, 2)),
    ]
    result = align_activations_to_act_layers(net, events)
    assert result.ok is False
    assert result.errors


def test_alignment_kind_mismatch() -> None:
    net = _Net([_Layer(0, "DENSE", (1, 2))])
    events = [
        ActivationEvent("l0", "Conv2d", 0, (1, 2), torch.zeros(1, 2)),
    ]
    result = align_activations_to_act_layers(net, events)
    assert result.ok is False
    assert result.errors


def test_alignment_shape_mismatch() -> None:
    net = _Net([_Layer(0, "DENSE", (1, 2))])
    events = [
        ActivationEvent("l0", "Linear", 0, (1, 3), torch.zeros(1, 3)),
    ]
    result = align_activations_to_act_layers(net, events)
    assert result.ok is False
    assert result.errors
