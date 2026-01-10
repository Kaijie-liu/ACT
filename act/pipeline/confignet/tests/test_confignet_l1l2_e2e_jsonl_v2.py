#!/usr/bin/env python3
#===- tests/test_confignet_l1l2_e2e_jsonl_v2.py - E2E JSONL v2 -------====#
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


def _strip_nondet(record):
    record = dict(record)
    record.pop("timing", None)
    record.pop("created_at_utc", None)
    return record


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


def _make_args(tmp_path, *, out_jsonl: str):
    return SimpleNamespace(
        instances=2,
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
        jsonl=str(tmp_path / out_jsonl),
    )


def test_confignet_l1l2_e2e_minimal_writes_jsonl_v2(tmp_path, monkeypatch) -> None:
    insts = [_make_instance("cfg_0", 11), _make_instance("cfg_1", 22)]

    def _fake_sample_instances(_cfg) -> List[InstanceSpec]:
        return insts

    def _fake_run_l1(inst, *_args, **_kwargs):
        if inst.instance_id == "cfg_0":
            return {
                "status": "FAILED",
                "counterexample_found": True,
                "verifier_status": "NOT_RUN",
                "policy_applied": True,
                "final_verdict_after_policy": "FAILED",
            }
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
            "checks_total": 10,
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

    args = _make_args(tmp_path, out_jsonl="out.jsonl")
    summary = run_confignet_l1l2(args)

    records = read_jsonl(args.jsonl)
    assert len(records) == 2
    for rec in records:
        validate_record_v2(rec)
        assert rec["schema_version"] == "confignet_l1l2_v2"
        assert "run_id" in rec
        assert "canonical_hash" in rec
        assert "l1" in rec
        assert "l2" in rec
        assert "final" in rec
        assert "run_meta" in rec
        assert "seeds" in rec
        assert "instance" in rec

    run_ids = {rec["run_id"] for rec in records}
    hashes = {rec["canonical_hash"] for rec in records}
    assert len(run_ids) == 1
    assert len(hashes) == 2

    counts = {"PASS": 0, "FAILED": 0, "ERROR": 0}
    for rec in records:
        counts[rec["final"]["final_verdict"]] += 1
    assert summary["passed"] == counts["PASS"]
    assert summary["failed"] == counts["FAILED"]
    assert summary["errors"] == counts["ERROR"]
    assert summary["exit_code"] == 1


def test_confignet_l1l2_determinism_same_seed_same_hash(tmp_path, monkeypatch) -> None:
    insts = [_make_instance("cfg_0", 11), _make_instance("cfg_1", 22)]

    def _fake_sample_instances(_cfg) -> List[InstanceSpec]:
        return insts

    def _fake_run_l1(inst, *_args, **_kwargs):
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
            "checks_total": 10,
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

    args_a = _make_args(tmp_path, out_jsonl="a.jsonl")
    args_b = _make_args(tmp_path, out_jsonl="b.jsonl")
    run_confignet_l1l2(args_a)
    run_confignet_l1l2(args_b)

    rec_a = read_jsonl(args_a.jsonl)
    rec_b = read_jsonl(args_b.jsonl)
    assert len(rec_a) == len(rec_b)
    for a, b in zip(rec_a, rec_b):
        assert a["run_id"] == b["run_id"]
        assert a["canonical_hash"] == b["canonical_hash"]


def test_confignet_top_level_routing_produces_same_records(tmp_path, monkeypatch) -> None:
    insts = [_make_instance("cfg_0", 11), _make_instance("cfg_1", 22)]

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
            "status": "PASSED",
            "tf_mode": "interval",
            "samples": 1,
            "checks_total": 10,
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

    top_path = tmp_path / "top.jsonl"
    sub_path = tmp_path / "sub.jsonl"

    from act.pipeline.cli import main as pipeline_main
    from act.pipeline.confignet.cli import main as confignet_main

    top_code = pipeline_main(
        [
            "confignet",
            "l1l2",
            "--num",
            "2",
            "--seed",
            "0",
            "--n-inputs",
            "1",
            "--tf-modes",
            "interval",
            "--out_jsonl",
            str(top_path),
        ]
    )
    sub_code = confignet_main(
        [
            "l1l2",
            "--num",
            "2",
            "--seed",
            "0",
            "--n-inputs",
            "1",
            "--tf-modes",
            "interval",
            "--out_jsonl",
            str(sub_path),
        ]
    )

    assert top_code == sub_code

    top_records = [_strip_nondet(r) for r in read_jsonl(top_path)]
    sub_records = [_strip_nondet(r) for r in read_jsonl(sub_path)]
    assert top_records == sub_records
