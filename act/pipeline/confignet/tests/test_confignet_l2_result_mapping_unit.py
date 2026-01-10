#!/usr/bin/env python3
#===- tests/test_confignet_l2_result_mapping_unit.py - L2 Mapping ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

import torch

from act.front_end.specs import InKind, OutKind
from act.pipeline.confignet.act_driver_l1l2 import _run_l2
from act.pipeline.confignet.factory_adapter import ConfignetFactoryAdapter
from act.pipeline.confignet.schema import (
    InstanceSpec,
    ModelFamily,
    MLPConfig,
    InputSpecConfig,
    OutputSpecConfig,
)
from act.pipeline.verification.validate_verifier import VerificationValidator


def _make_instance(instance_id: str, seed: int) -> InstanceSpec:
    return InstanceSpec(
        instance_id=instance_id,
        seed=seed,
        family=ModelFamily.MLP,
        model_cfg=MLPConfig(
            input_shape=(1, 2),
            hidden_sizes=(2,),
            activation="relu",
            dropout_p=0.0,
            num_classes=2,
        ),
        input_spec=InputSpecConfig(
            kind=InKind.LINF_BALL,
            center_val=0.5,
            eps=0.1,
        ),
        output_spec=OutputSpecConfig(
            kind=OutKind.TOP1_ROBUST,
            y_true=0,
        ),
        meta={},
    )


def test_l2_mapping_selects_worst_layer() -> None:
    inst = _make_instance("cfg_0", 11)
    validator = VerificationValidator(device="cpu", dtype=torch.float64)
    adapter = ConfignetFactoryAdapter(device="cpu", dtype=torch.float64)

    def _fake_configure_for_validation(*_args, **_kwargs):
        return None

    def _fake_validate_bounds_per_neuron(*_args, **_kwargs):
        validator.validation_results.append(
            {
                "validation_type": "bounds_per_neuron",
                "network": inst.instance_id,
                "validation_status": "FAILED",
                "total_checks": 5,
                "violations_total": 2,
                "violations_topk": [],
                "layerwise_stats": [
                    {"layer_id": 1, "num_violations": 0, "max_gap": 1.0, "kind": "ReLU"},
                    {"layer_id": 2, "num_violations": 2, "max_gap": 0.5, "kind": "Linear"},
                ],
                "errors": [],
                "warnings": [],
            }
        )
        return {}

    adapter.configure_for_validation = _fake_configure_for_validation
    validator.validate_bounds_per_neuron = _fake_validate_bounds_per_neuron

    l2 = _run_l2(
        inst,
        object(),
        object(),
        device="cpu",
        dtype=torch.float64,
        tf_mode="interval",
        samples=1,
        strict=True,
        atol=1e-6,
        rtol=0.0,
        topk=10,
        validator=validator,
        adapter=adapter,
        seed_inputs=0,
    )
    assert l2["status"] == "FAILED"
    assert l2["worst_gap"] == 0.5
    assert l2["worst_layer"]["layer_id"] == 2
