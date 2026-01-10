#!/usr/bin/env python3
#===- tests/test_confignet_input_sampling_status.py - Sampling Status ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
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
from act.pipeline.confignet.input_sampling import sample_feasible_inputs


def test_input_sampling_status_ok() -> None:
    inst = InstanceSpec(
        instance_id="linpoly_ok",
        seed=1,
        family=ModelFamily.MLP,
        model_cfg=MLPConfig(
            input_shape=(1, 1),
            hidden_sizes=(2,),
            activation="relu",
            dropout_p=0.0,
            num_classes=2,
        ),
        input_spec=InputSpecConfig(
            kind=InKind.LIN_POLY,
            value_range=(0.0, 1.0),
            lb_val=0.0,
            ub_val=1.0,
            derive_poly_from_box=True,
        ),
        output_spec=OutputSpecConfig(
            kind=OutKind.TOP1_ROBUST,
            y_true=0,
        ),
        meta={},
    )

    xs, status = sample_feasible_inputs(
        inst,
        num_samples=3,
        seed=0,
        device="cpu",
        dtype=torch.float64,
        strict_input=True,
        return_status=True,
    )
    assert len(xs) == 3
    assert status["input_sampling_status"] == "ok"
    assert "rejection_rate" in status


def test_input_sampling_status_fallback() -> None:
    # Unsatisfiable: x <= -1 with value_range [0,1]
    A = torch.tensor([[1.0]], dtype=torch.float64)
    b = torch.tensor([-1.0], dtype=torch.float64)

    inst = InstanceSpec(
        instance_id="linpoly_bad",
        seed=2,
        family=ModelFamily.MLP,
        model_cfg=MLPConfig(
            input_shape=(1, 1),
            hidden_sizes=(2,),
            activation="relu",
            dropout_p=0.0,
            num_classes=2,
        ),
        input_spec=InputSpecConfig(
            kind=InKind.LIN_POLY,
            value_range=(0.0, 1.0),
            A=A,
            b=b,
            derive_poly_from_box=False,
        ),
        output_spec=OutputSpecConfig(
            kind=OutKind.TOP1_ROBUST,
            y_true=0,
        ),
        meta={},
    )

    xs, status = sample_feasible_inputs(
        inst,
        num_samples=1,
        seed=0,
        device="cpu",
        dtype=torch.float64,
        strict_input=False,
        max_tries=5,
        return_status=True,
    )
    assert len(xs) == 1
    assert status["input_sampling_status"] == "fallback"
