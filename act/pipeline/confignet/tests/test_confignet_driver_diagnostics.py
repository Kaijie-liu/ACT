#!/usr/bin/env python3
#===- tests/test_confignet_driver_diagnostics.py - Driver Diagnostics -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import List

import torch

from act.front_end.specs import InKind, OutKind
from act.pipeline.confignet.act_driver_l1l2 import run_confignet_l1l2
from act.pipeline.confignet.jsonl import read_jsonl
from act.pipeline.confignet.schema import (
    InstanceSpec,
    ModelFamily,
    MLPConfig,
    InputSpecConfig,
    OutputSpecConfig,
)
from act.pipeline.confignet.schema_v2 import validate_record_v2


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


def _fake_build_case(self, inst, *, seed_inputs, num_samples=1, deterministic_algos=False, strict_input=True):
    return {
        "generated": object(),
        "torch_model": torch.nn.Linear(2, 2),
        "act_model": object(),
        "inputs": [torch.zeros(1, 2)],
        "spec": {},
        "instance_meta": {
            "net_family": "MLP",
            "arch": {"depth": 1},
            "spec": {"input_spec": {}, "output_spec": {}},
            "input_shape": [1, 2],
            "eps": 0.1,
        },
    }


def _make_args(tmp_path) -> SimpleNamespace:
    return SimpleNamespace(
        instances=1,
        seed=0,
        samples=1,
        tf_mode="interval",
        tf_modes=["interval"],
        strict=True,
        atol=1e-6,
        rtol=0.0,
        topk=10,
        device="cpu",
        dtype="float64",
        jsonl=str(tmp_path / "out.jsonl"),
    )


def test_failed_l2_prints_worst_layer_and_topk(caplog, monkeypatch, tmp_path) -> None:
    insts = [_make_instance("cfg_0", 11)]

    def _fake_sample_instances(_cfg) -> List[InstanceSpec]:
        return insts

    def _fake_run_l1(*_args, **_kwargs):
        return {
            "status": "PASSED",
            "counterexample_found": False,
            "verifier_status": "NOT_RUN",
            "policy_applied": True,
            "final_verdict_after_policy": "PASS",
        }

    def _fake_run_l2(*_args, **_kwargs):
        return {
            "status": "FAILED",
            "tf_mode": "interval",
            "samples": 1,
            "checks_total": 10,
            "violations_total": 7,
            "worst_gap": 0.123,
            "worst_layer": {"layer_id": 3},
            "topk": [
                {"neuron_index": 1, "gap": 0.1},
                {"neuron_index": 2, "gap": 0.05},
                {"neuron_index": 3, "gap": 0.02},
            ],
            "layerwise_stats": [],
            "errors": [],
            "warnings": [],
        }

    monkeypatch.setattr("act.pipeline.confignet.act_driver_l1l2.sample_instances", _fake_sample_instances)
    monkeypatch.setattr("act.pipeline.confignet.factory_adapter.ConfignetFactoryAdapter.build_case", _fake_build_case)
    monkeypatch.setattr("act.pipeline.confignet.act_driver_l1l2._run_l1", _fake_run_l1)
    monkeypatch.setattr("act.pipeline.confignet.act_driver_l1l2._run_l2", _fake_run_l2)

    caplog.set_level(logging.INFO, logger="act.pipeline.confignet.act_driver_l1l2")
    run_confignet_l1l2(_make_args(tmp_path))
    text = caplog.text
    assert "🚨 [CN][l1l2]" in text
    assert "violations_total=7" in text
    assert "worst_gap=0.123" in text
    assert "worst_layer=" in text
    assert "topk:" in text
    assert "neuron_index" in text


def test_error_prints_first_three_errors(caplog, monkeypatch, tmp_path) -> None:
    insts = [_make_instance("cfg_0", 11)]

    def _fake_sample_instances(_cfg) -> List[InstanceSpec]:
        return insts

    def _fake_build_case_error(self, inst, *, seed_inputs, num_samples=1, deterministic_algos=False, strict_input=True):
        raise RuntimeError("boom-0")

    monkeypatch.setattr("act.pipeline.confignet.act_driver_l1l2.sample_instances", _fake_sample_instances)
    monkeypatch.setattr("act.pipeline.confignet.factory_adapter.ConfignetFactoryAdapter.build_case", _fake_build_case_error)

    caplog.set_level(logging.INFO, logger="act.pipeline.confignet.act_driver_l1l2")
    run_confignet_l1l2(_make_args(tmp_path))
    text = caplog.text
    assert "❌ [CN][l1l2]" in text
    assert "errors:" in text
    assert "boom-0" in text


def test_exception_still_writes_jsonl_record(tmp_path, monkeypatch) -> None:
    insts = [_make_instance("cfg_0", 11)]

    def _fake_sample_instances(_cfg) -> List[InstanceSpec]:
        return insts

    def _fake_build_case_error(self, inst, *, seed_inputs, num_samples=1, deterministic_algos=False, strict_input=True):
        raise RuntimeError("boom-1")

    monkeypatch.setattr("act.pipeline.confignet.act_driver_l1l2.sample_instances", _fake_sample_instances)
    monkeypatch.setattr("act.pipeline.confignet.factory_adapter.ConfignetFactoryAdapter.build_case", _fake_build_case_error)

    args = _make_args(tmp_path)
    run_confignet_l1l2(args)
    records = read_jsonl(args.jsonl)
    assert len(records) == 1
    validate_record_v2(records[0])
    assert records[0]["final"]["final_verdict"] == "ERROR"
    assert "boom-1" in " ".join(records[0].get("errors", []))
