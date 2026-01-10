#!/usr/bin/env python3
#===- tests/test_level1_counterexample_policy.py - L1 Policy ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

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


def test_l1_counterexample_forces_failed(monkeypatch, tmp_path) -> None:
    insts = [_make_instance("cfg_0", 11)]

    def _fake_sample_instances(_cfg) -> List[InstanceSpec]:
        return insts

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

    def _fake_run_l1(*_args, **_kwargs):
        return {
            "status": "FAILED",
            "counterexample_found": True,
            "verifier_status": "NOT_RUN",
            "policy_applied": True,
            "final_verdict_after_policy": "FAILED",
        }

    def _fake_run_l2(*_args, **_kwargs):
        return {
            "status": "PASSED",
            "tf_mode": "interval",
            "samples": 1,
            "checks_total": 4,
            "violations_total": 0,
            "worst_gap": 0.0,
            "worst_layer": None,
            "topk": [],
            "layerwise_stats": [],
            "errors": [],
            "warnings": [],
        }

    monkeypatch.setattr("act.pipeline.confignet.act_driver_l1l2.sample_instances", _fake_sample_instances)
    monkeypatch.setattr("act.pipeline.confignet.factory_adapter.ConfignetFactoryAdapter.build_case", _fake_build_case)
    monkeypatch.setattr("act.pipeline.confignet.act_driver_l1l2._run_l1", _fake_run_l1)
    monkeypatch.setattr("act.pipeline.confignet.act_driver_l1l2._run_l2", _fake_run_l2)

    args = _make_args(tmp_path)
    summary = run_confignet_l1l2(args)
    records = read_jsonl(args.jsonl)
    assert records[0]["l1"]["counterexample_found"] is True
    assert records[0]["final"]["final_verdict"] == "FAILED"
    assert records[0]["final"]["final_verdict"] != "PASS"
    assert summary["exit_code"] == 1


def test_l1_no_cex_allows_pass(monkeypatch, tmp_path) -> None:
    insts = [_make_instance("cfg_0", 11)]

    def _fake_sample_instances(_cfg) -> List[InstanceSpec]:
        return insts

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
            "status": "PASSED",
            "tf_mode": "interval",
            "samples": 1,
            "checks_total": 4,
            "violations_total": 0,
            "worst_gap": 0.0,
            "worst_layer": None,
            "topk": [],
            "layerwise_stats": [],
            "errors": [],
            "warnings": [],
        }

    monkeypatch.setattr("act.pipeline.confignet.act_driver_l1l2.sample_instances", _fake_sample_instances)
    monkeypatch.setattr("act.pipeline.confignet.factory_adapter.ConfignetFactoryAdapter.build_case", _fake_build_case)
    monkeypatch.setattr("act.pipeline.confignet.act_driver_l1l2._run_l1", _fake_run_l1)
    monkeypatch.setattr("act.pipeline.confignet.act_driver_l1l2._run_l2", _fake_run_l2)

    args = _make_args(tmp_path)
    summary = run_confignet_l1l2(args)
    records = read_jsonl(args.jsonl)
    assert records[0]["l1"]["counterexample_found"] is False
    assert records[0]["final"]["final_verdict"] == "PASS"
    assert summary["exit_code"] == 0
