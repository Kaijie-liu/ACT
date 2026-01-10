#!/usr/bin/env python3
#===- tests/test_confignet_rng_no_side_effect.py - RNG Tests ----------====#
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


def test_build_wrapped_model_no_rng_side_effect() -> None:
    cfg = ConfigNetConfig(
        num_instances=1,
        base_seed=123,
        families=(ModelFamily.MLP,),
    )
    inst = sample_instances(cfg)[0]

    torch.manual_seed(999)
    state_before = torch.get_rng_state()
    _ = build_wrapped_model(inst, device="cpu", dtype=torch.float64)
    state_after = torch.get_rng_state()

    assert torch.equal(state_before, state_after)
