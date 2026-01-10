#!/usr/bin/env python3
#===- tests/test_confignet_l1l2_smoke_real.py - L1L2 Smoke ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

from types import SimpleNamespace

from act.pipeline.confignet.act_driver_l1l2 import run_confignet_l1l2
from act.pipeline.confignet.jsonl import read_jsonl
from act.pipeline.confignet.schema_v2 import validate_record_v2


def test_confignet_l1l2_smoke_real(tmp_path) -> None:
    args = SimpleNamespace(
        instances=1,
        seed=0,
        samples=1,
        tf_mode="interval",
        tf_modes=["interval"],
        strict=True,
        atol=1e-6,
        rtol=0.0,
        topk=3,
        device="cpu",
        dtype="float64",
        jsonl=str(tmp_path / "out.jsonl"),
    )
    summary = run_confignet_l1l2(args)
    records = read_jsonl(args.jsonl)
    assert len(records) == 1
    validate_record_v2(records[0])
    assert records[0]["schema_version"] == "confignet_l1l2_v2"
    assert summary["records"] == 1
