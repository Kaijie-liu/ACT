#!/usr/bin/env python3
#===- tests/test_confignet_lin_poly_sampling.py - LIN_POLY Sampling ----====#
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


def test_lin_poly_rejection_sampling_satisfies_constraints() -> None:
    # Constraints: 0 <= x <= 1
    A = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    b = torch.tensor([1.0, 0.0], dtype=torch.float64)

    inst = InstanceSpec(
        instance_id="linpoly_unit",
        seed=123,
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

    xs = sample_feasible_inputs(
        inst,
        num_samples=5,
        seed=0,
        device="cpu",
        dtype=torch.float64,
        strict_input=True,
    )
    for x in xs:
        x_flat = x.reshape(-1)
        assert torch.all(A.matmul(x_flat) <= b + 1e-8)
