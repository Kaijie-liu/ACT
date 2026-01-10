#!/usr/bin/env python3
#===- tests/test_confignet_jsonl_schema_v1.py - JSONL Schema Tests -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

import torch

from act.pipeline.confignet.schema import ConfigNetConfig
from act.pipeline.confignet.verdicts import Verdict
from act.pipeline.confignet.driver_levels import run_driver_levels, SCHEMA_VERSION_V1


def test_jsonl_schema_v1_fields_present() -> None:
    cfg = ConfigNetConfig(num_instances=2, base_seed=0)
    records = run_driver_levels(
        cfg,
        device="cpu",
        dtype=torch.float64,
        base_seed=0,
        n_inputs_l1=0,
        strict_input=True,
        command="pytest",
    )
    assert len(records) == 2
    r0 = records[0]

    assert r0["schema_version"] == SCHEMA_VERSION_V1
    assert "instance_spec" in r0
    assert "level1" in r0
    assert "gating" in r0
    assert "final_verdict" in r0
    assert "run_meta" in r0

    assert "hash" in r0
    assert "git_sha" in r0
    assert "timestamp" in r0

    assert r0["final_verdict"] in {v.value for v in Verdict}
    assert "summary" in r0["level1"]
    assert "found_cex" in r0["level1"]["summary"]
    assert "num_cex" in r0["level1"]["summary"]
    assert isinstance(r0["gating"].get("details", {}), dict)
    assert isinstance(r0["schema_version"], str)
    assert isinstance(r0["hash"], str)
