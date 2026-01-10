#!/usr/bin/env python3
#===- tests/test_confignet_sampling_new_specs.py - New Spec Sampling ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

import torch

from act.front_end.specs import InKind, OutKind
from act.pipeline.confignet import ConfigNetConfig, ModelFamily
from act.pipeline.confignet.sampler import sample_instances
from act.pipeline.confignet.builders import build_wrapped_model


def _build_one(cfg: ConfigNetConfig) -> None:
    inst = sample_instances(cfg)[0]
    _ = build_wrapped_model(inst, device="cpu", dtype=torch.float64)


def test_sample_lin_poly_builds() -> None:
    cfg = ConfigNetConfig(
        num_instances=1,
        base_seed=7,
        families=(ModelFamily.MLP,),
        input_kind_choices=(InKind.LIN_POLY,),
        output_kind_choices=(OutKind.TOP1_ROBUST,),
    )
    _build_one(cfg)


def test_sample_output_linear_le_builds() -> None:
    cfg = ConfigNetConfig(
        num_instances=1,
        base_seed=8,
        families=(ModelFamily.MLP,),
        output_kind_choices=(OutKind.LINEAR_LE,),
    )
    _build_one(cfg)


def test_sample_output_range_builds() -> None:
    cfg = ConfigNetConfig(
        num_instances=1,
        base_seed=9,
        families=(ModelFamily.MLP,),
        output_kind_choices=(OutKind.RANGE,),
    )
    _build_one(cfg)
