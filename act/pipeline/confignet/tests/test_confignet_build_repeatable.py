#!/usr/bin/env python3
#===- tests/test_confignet_build_repeatable.py - Build Repeatability ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

import torch

from act.pipeline.confignet import ConfigNetConfig, ModelFamily
from act.pipeline.confignet.sampler import sample_instances
from act.pipeline.confignet.builders import build_wrapped_model


def test_build_wrapped_model_repeatable() -> None:
    cfg = ConfigNetConfig(
        num_instances=1,
        base_seed=123,
        families=(ModelFamily.MLP,),
    )
    inst = sample_instances(cfg)[0]

    m1 = build_wrapped_model(inst, device="cpu", dtype=torch.float64)
    m2 = build_wrapped_model(inst, device="cpu", dtype=torch.float64)

    s1 = m1.state_dict()
    s2 = m2.state_dict()
    assert s1.keys() == s2.keys()
    for k in s1.keys():
        assert torch.equal(s1[k], s2[k])
