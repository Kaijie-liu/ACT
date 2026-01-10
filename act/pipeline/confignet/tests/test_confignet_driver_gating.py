#!/usr/bin/env python3
#===- tests/test_confignet_driver_gating.py - Driver Gating Tests ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

import torch

from act.pipeline.confignet.policy import apply_policy
from act.pipeline.confignet.verdicts import Verdict
from act.pipeline.confignet.driver_levels import run_driver_levels, SCHEMA_VERSION_V1
from act.pipeline.confignet.schema import ConfigNetConfig


def test_policy_falsified_when_level1_has_cex() -> None:
    final_v, gating = apply_policy(
        base_verdict=Verdict.CERTIFIED,
        level1_found_cex=True,
        level1_summary={"found_cex": True, "num_cex": 1},
    )
    assert final_v == Verdict.FALSIFIED
    assert gating.prohibited_certified_due_to_cex is True


def test_driver_gating_enforces_falsified_on_cex(monkeypatch) -> None:
    import act.pipeline.confignet.driver_levels as driver_levels

    def _fake_level1_suite(*args, **kwargs):
        return [
            {
                "instance_spec": {"instance_id": "fake_0"},
                "level1_result": {"found_cex": True, "num_cex": 1, "first_cex": None},
                "input_sampling": {"input_sampling_status": "ok"},
            }
        ]

    monkeypatch.setattr(driver_levels, "run_level1_suite", _fake_level1_suite)
    cfg = ConfigNetConfig(num_instances=1, base_seed=0)
    records = run_driver_levels(
        cfg,
        device="cpu",
        dtype=torch.float64,
        base_seed=0,
        n_inputs_l1=0,
        strict_input=True,
        command="pytest",
    )

    assert len(records) == 1
    r0 = records[0]
    assert r0["schema_version"] == SCHEMA_VERSION_V1
    assert r0["final_verdict"] == Verdict.FALSIFIED.value
    assert r0["gating"]["prohibited_certified_due_to_cex"] is True
    assert r0["gating"]["reason"] == "level1_concrete_counterexample"
