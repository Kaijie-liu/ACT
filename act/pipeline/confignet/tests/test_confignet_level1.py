#!/usr/bin/env python3
#===- tests/test_confignet_level1.py - Unit Tests for ConfigNet Level1 --====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Minimal pytest coverage:
#     - Level1 must find a concrete counterexample for an untrained random net
#       under TOP1_ROBUST spec with enough samples (almost surely).
#
# Run:
#   pytest -q
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import torch

from act.front_end.specs import InKind, OutKind
from act.pipeline.confignet.schema import (
    InstanceSpec,
    ModelFamily,
    MLPConfig,
    InputSpecConfig,
    OutputSpecConfig,
)
from act.pipeline.confignet.level1_check import run_level1_check


def test_level1_finds_counterexample_top1() -> None:
    # A simple random MLP
    inst = InstanceSpec(
        instance_id="unit_mlp_top1",
        seed=12345,
        family=ModelFamily.MLP,
        model_cfg=MLPConfig(
            input_shape=(1, 16),
            hidden_sizes=(32, 32),
            activation="relu",
            dropout_p=0.0,
            num_classes=10,
        ),
        input_spec=InputSpecConfig(
            kind=InKind.BOX,
            value_range=(0.0, 1.0),
            lb_val=0.0,
            ub_val=1.0,
        ),
        output_spec=OutputSpecConfig(
            kind=OutKind.TOP1_ROBUST,
            y_true=0,   # fixed label
            margin=0.0,
        ),
        meta={},
    )

    # With random weights, probability pred==0 is ~0.1; so in 50 samples,
    # probability of never seeing a counterexample is ~0.1^50 (negligible).
    results = run_level1_check(
        [inst],
        num_samples=50,
        device="cpu",
        dtype=torch.float64,
        stop_on_first_cex=True,
    )
    r = results[0]
    assert r.found_cex is True
    assert r.first_cex is not None
    assert isinstance(r.first_cex.output_explanation, str)
