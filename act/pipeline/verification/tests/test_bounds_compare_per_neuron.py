#!/usr/bin/env python3
#===- tests/test_bounds_compare_per_neuron.py - Bounds Compare Tests ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

import torch

from act.pipeline.verification.bounds_compare import compare_bounds_per_neuron
from act.pipeline.verification.bounds_core import LayerBounds


def test_compare_bounds_per_neuron_pass() -> None:
    lb = torch.tensor([0.0, 0.0])
    ub = torch.tensor([1.0, 1.0])
    bounds = {
        0: LayerBounds(layer_id=0, kind="DENSE", lb=lb, ub=ub, shape=(2,)),
    }
    concrete = {0: torch.tensor([0.25, 0.75])}
    result = compare_bounds_per_neuron(
        bounds_by_layer=bounds,
        concrete_by_layer=concrete,
        atol=1e-6,
        rtol=0.0,
        topk=5,
    )
    assert result["status"] == "PASS"
    assert result["violations_total"] == 0


def test_compare_bounds_per_neuron_fail() -> None:
    lb = torch.tensor([0.0, 0.0])
    ub = torch.tensor([1.0, 1.0])
    bounds = {
        0: LayerBounds(layer_id=0, kind="DENSE", lb=lb, ub=ub, shape=(2,)),
    }
    concrete = {0: torch.tensor([1.5, -0.5])}
    result = compare_bounds_per_neuron(
        bounds_by_layer=bounds,
        concrete_by_layer=concrete,
        atol=1e-6,
        rtol=0.0,
        topk=5,
    )
    assert result["status"] == "FAIL"
    assert result["violations_total"] == 2
    assert result["violations_topk"]
    assert result["violations_topk"][0]["gap"] > 0.0


def test_compare_bounds_per_neuron_error_on_shape() -> None:
    lb = torch.tensor([0.0, 0.0])
    ub = torch.tensor([1.0, 1.0])
    bounds = {
        0: LayerBounds(layer_id=0, kind="DENSE", lb=lb, ub=ub, shape=(2,)),
    }
    concrete = {0: torch.tensor([0.5])}
    result = compare_bounds_per_neuron(
        bounds_by_layer=bounds,
        concrete_by_layer=concrete,
        atol=1e-6,
        rtol=0.0,
        topk=5,
    )
    assert result["status"] == "ERROR"
    assert result["violations_total"] == 0
