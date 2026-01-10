#!/usr/bin/env python3
#===- tests/test_confignet_driver_l1l2_unit.py - Driver Unit Tests ---====#
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

from dataclasses import replace

from act.pipeline.confignet.act_driver_l1l2 import (
    _derive_seeds,
    _finalize_verdict,
    run_confignet_l1l2,
)
from act.pipeline.confignet.schema import (
    InstanceSpec,
    ModelFamily,
    MLPConfig,
    InputSpecConfig,
    OutputSpecConfig,
)
from act.front_end.specs import InKind, OutKind


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


def test_derive_seeds_deterministic() -> None:
    a = _derive_seeds(0, 3)
    b = _derive_seeds(0, 3)
    c = _derive_seeds(0, 4)
    assert a == b
    assert a["seed_instance"] != c["seed_instance"]


def test_finalize_verdict_priority() -> None:
    l1_ok = {"final_verdict_after_policy": "PASS", "status": "PASSED"}
    l1_fail = {"final_verdict_after_policy": "FAILED", "status": "FAILED"}
    l2_ok = {"status": "PASSED"}
    l2_fail = {"status": "FAILED"}
    l2_err = {"status": "ERROR"}
    solver_ok = {"status": "PASSED", "verdict": "UNKNOWN"}
    solver_err = {"status": "ERROR", "verdict": "ERROR"}

    out = _finalize_verdict(l1_ok, l2_ok, solver_ok, ["x"])
    assert out["final_verdict"] == "ERROR"
    assert out["exit_code"] == 2

    out = _finalize_verdict(l1_fail, l2_ok, solver_ok, [])
    assert out["final_verdict"] == "FAILED"
    assert out["exit_code"] == 1
    assert out["reason"] == "l1_failed"

    out = _finalize_verdict(l1_ok, l2_fail, solver_ok, [])
    assert out["final_verdict"] == "FAILED"
    assert out["exit_code"] == 1
    assert out["reason"] == "l2_failed"

    out = _finalize_verdict(l1_ok, l2_ok, solver_ok, [])
    assert out["final_verdict"] == "PASS"
    assert out["exit_code"] == 0
    assert out["reason"] == "ok"

    out = _finalize_verdict(l1_ok, l2_err, solver_ok, [])
    assert out["final_verdict"] == "ERROR"
    assert out["exit_code"] == 2

    out = _finalize_verdict(l1_ok, l2_ok, solver_err, [])
    assert out["final_verdict"] == "ERROR"
    assert out["exit_code"] == 2


def test_driver_builds_valid_record_v2(monkeypatch, tmp_path) -> None:
    insts = [_make_instance("cfg_0", 11), _make_instance("cfg_1", 22)]

    def _fake_sample_instances(_cfg) -> List[InstanceSpec]:
        return insts

    def _fake_build_case(self, inst, *, seed_inputs, num_samples=1, deterministic_algos=False, strict_input=True):
        return {
            "generated": None,
            "torch_model": torch.nn.Linear(2, 2),
            "act_model": None,
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

    args = SimpleNamespace(
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
        jsonl=str(tmp_path / "out.jsonl"),
    )
    summary = run_confignet_l1l2(args)
    assert summary["records"] == 2

    lines = (tmp_path / "out.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    import json
    from act.pipeline.confignet.schema_v2 import validate_record_v2

    records = [json.loads(line) for line in lines]
    for rec in records:
        validate_record_v2(rec)
    run_ids = {rec["run_id"] for rec in records}
    hashes = {rec["canonical_hash"] for rec in records}
    assert len(run_ids) == 1
    assert len(hashes) == 2


def test_driver_summary_counts_match_records(monkeypatch, tmp_path) -> None:
    insts = [
        _make_instance("cfg_0", 11),
        _make_instance("cfg_1", 22),
        _make_instance("cfg_2", 33),
    ]

    def _fake_sample_instances(_cfg) -> List[InstanceSpec]:
        return insts

    def _fake_build_case(self, inst, *, seed_inputs, num_samples=1, deterministic_algos=False, strict_input=True):
        return {
            "generated": None,
            "torch_model": torch.nn.Linear(2, 2),
            "act_model": None,
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

    def _fake_run_l1(inst, *_args, **_kwargs):
        if inst.instance_id == "cfg_0":
            return {
                "status": "PASSED",
                "counterexample_found": False,
                "verifier_status": "NOT_RUN",
                "policy_applied": True,
                "final_verdict_after_policy": "PASS",
            }
        if inst.instance_id == "cfg_1":
            return {
                "status": "FAILED",
                "counterexample_found": True,
                "verifier_status": "NOT_RUN",
                "policy_applied": True,
                "final_verdict_after_policy": "FAILED",
            }
        return {
            "status": "ERROR",
            "counterexample_found": False,
            "verifier_status": "NOT_RUN",
            "policy_applied": True,
            "final_verdict_after_policy": "ERROR",
        }

    def _fake_run_l2(inst, *_args, **_kwargs):
        if inst.instance_id == "cfg_2":
            return {
                "status": "ERROR",
                "tf_mode": "interval",
                "samples": 1,
                "checks_total": 0,
                "violations_total": 0,
                "worst_gap": None,
                "worst_layer": None,
                "topk": [],
                "layerwise_stats": [],
                "errors": ["l2 error"],
                "warnings": [],
            }
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

    args = SimpleNamespace(
        instances=3,
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
    summary = run_confignet_l1l2(args)

    import json
    lines = (tmp_path / "out.jsonl").read_text().strip().splitlines()
    records = [json.loads(line) for line in lines]
    counts = {"PASS": 0, "FAILED": 0, "ERROR": 0}
    for rec in records:
        counts[rec["final"]["final_verdict"]] += 1

    assert summary["passed"] == counts["PASS"]
    assert summary["failed"] == counts["FAILED"]
    assert summary["errors"] == counts["ERROR"]
    assert summary["exit_code"] == 2


def test_run_l2_maps_validation_status_to_schema(monkeypatch) -> None:
    from act.pipeline.confignet.act_driver_l1l2 import _run_l2
    from act.pipeline.confignet.factory_adapter import ConfignetFactoryAdapter
    from act.pipeline.confignet.schema_v2 import build_record_v2, validate_record_v2
    from act.pipeline.confignet.act_driver_l1l2 import _derive_seeds
    from act.pipeline.confignet.act_driver_l1l2 import _finalize_verdict
    from act.pipeline.verification.validate_verifier import VerificationValidator

    inst = _make_instance("cfg_0", 11)
    seeds = _derive_seeds(0, 0, inst.instance_id)
    inst_seeded = replace(inst, seed=seeds["seed_instance"])
    gen = object()

    validator = VerificationValidator(device="cpu", dtype=torch.float64)
    adapter = ConfignetFactoryAdapter(device="cpu", dtype=torch.float64)

    def _fake_configure_for_validation(
        self,
        instances,
        generated,
        *,
        seed_inputs_by_name=None,
        act_nets_by_name=None,
        strict_input=True,
    ):
        self._instances = {inst.instance_id: inst for inst in instances}
        self._generated = {}
        self._act_nets = dict(act_nets_by_name or {})
        self._call_counts = {inst.instance_id: 0 for inst in instances}
        self._seed_inputs_by_name = seed_inputs_by_name or {}
        self._strict_input = bool(strict_input)

    def _fake_validate_bounds_per_neuron(*_args, **_kwargs):
        validator.validation_results.append(
            {
                "validation_type": "bounds_per_neuron",
                "network": inst_seeded.instance_id,
                "validation_status": "PASS",
                "total_checks": 1,
                "violations_total": 0,
                "violations_topk": [],
                "layerwise_stats": [],
                "errors": [],
                "warnings": [],
            }
        )
        return {}

    monkeypatch.setattr(
        "act.pipeline.confignet.factory_adapter.ConfignetFactoryAdapter.configure_for_validation",
        _fake_configure_for_validation,
    )
    monkeypatch.setattr(validator, "validate_bounds_per_neuron", _fake_validate_bounds_per_neuron)

    l2 = _run_l2(
        inst_seeded,
        gen,
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
        seed_inputs=seeds["seed_inputs"],
    )
    assert l2["status"] == "PASSED"

    l1 = {
        "status": "PASSED",
        "counterexample_found": False,
        "verifier_status": "NOT_RUN",
        "policy_applied": True,
        "final_verdict_after_policy": "PASS",
    }
    solver = {"status": "PASSED", "verdict": "UNKNOWN"}
    final = _finalize_verdict(l1, l2, solver, [])
    run_meta = {
        "seed_root": 0,
        "instances": 1,
        "samples": 1,
        "tf_mode": "interval",
        "strict": True,
        "device": "cpu",
        "dtype": "float64",
        "atol": 1e-6,
        "rtol": 0.0,
        "topk": 10,
    }
    record = build_record_v2(
        run_meta=run_meta,
        seeds=seeds,
        instance={
            "net_family": "MLP",
            "arch": inst_seeded.model_cfg.to_dict(),
            "spec": {
                "input_spec": inst_seeded.input_spec.to_dict(),
                "output_spec": inst_seeded.output_spec.to_dict(),
            },
            "input_shape": list(inst_seeded.model_cfg.input_shape),
            "eps": float(inst_seeded.input_spec.eps or 0.0),
        },
        l1=l1,
        l2=l2,
        final=final,
    )
    validate_record_v2(record)


def test_adapter_generate_test_input_uses_seed_inputs_base(monkeypatch) -> None:
    from act.pipeline.confignet.factory_adapter import ConfignetFactoryAdapter
    from act.pipeline.confignet.seeds import derive_seed

    inst = _make_instance("cfg_0", 11)
    adapter = ConfignetFactoryAdapter(device="cpu", dtype=torch.float64)
    adapter._instances = {inst.instance_id: inst}
    adapter._generated = {}
    adapter._act_nets = {}
    adapter._call_counts = {inst.instance_id: 0}
    adapter._seed_inputs_by_name = {inst.instance_id: 123}

    seen: List[int] = []

    def _fake_sample_feasible_inputs(*_args, **_kwargs):
        seen.append(_kwargs["seed"])
        return [torch.zeros(1, 2)]

    monkeypatch.setattr(
        "act.pipeline.confignet.factory_adapter.sample_feasible_inputs",
        _fake_sample_feasible_inputs,
    )

    adapter.generate_test_input(inst.instance_id)
    adapter.generate_test_input(inst.instance_id)
    assert seen[0] == derive_seed(123, 0, "inputs|level2")
    assert seen[1] == derive_seed(123, 1, "inputs|level2")


def test_driver_act_build_error_produces_l2_error(monkeypatch, tmp_path) -> None:
    insts = [_make_instance("cfg_0", 11)]

    def _fake_sample_instances(_cfg) -> List[InstanceSpec]:
        return insts

    def _fake_build_case(self, inst, *, seed_inputs, num_samples=1, deterministic_algos=False, strict_input=True):
        return {
            "generated": object(),
            "torch_model": torch.nn.Linear(2, 2),
            "act_model": None,
            "act_build_error": "boom",
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

    monkeypatch.setattr("act.pipeline.confignet.act_driver_l1l2.sample_instances", _fake_sample_instances)
    monkeypatch.setattr("act.pipeline.confignet.factory_adapter.ConfignetFactoryAdapter.build_case", _fake_build_case)
    monkeypatch.setattr("act.pipeline.confignet.act_driver_l1l2._run_l1", _fake_run_l1)

    args = SimpleNamespace(
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
    summary = run_confignet_l1l2(args)
    assert summary["errors"] == 1
    assert summary["exit_code"] == 2

    import json
    lines = (tmp_path / "out.jsonl").read_text().strip().splitlines()
    record = json.loads(lines[0])
    assert record["l2"]["status"] == "ERROR"
    assert record["final"]["final_verdict"] == "ERROR"
    assert "boom" in " ".join(record.get("errors", []))


def test_adapter_generate_test_input_respects_strict_input(monkeypatch) -> None:
    from act.pipeline.confignet.factory_adapter import ConfignetFactoryAdapter

    inst = _make_instance("cfg_0", 11)
    adapter = ConfignetFactoryAdapter(device="cpu", dtype=torch.float64)
    adapter._instances = {inst.instance_id: inst}
    adapter._generated = {}
    adapter._act_nets = {}
    adapter._call_counts = {inst.instance_id: 0}
    adapter._seed_inputs_by_name = {inst.instance_id: 123}
    adapter._strict_input = False

    seen: List[bool] = []

    def _fake_sample_feasible_inputs(*_args, **_kwargs):
        seen.append(bool(_kwargs["strict_input"]))
        return [torch.zeros(1, 2)]

    monkeypatch.setattr(
        "act.pipeline.confignet.factory_adapter.sample_feasible_inputs",
        _fake_sample_feasible_inputs,
    )

    adapter.generate_test_input(inst.instance_id)
    assert seen == [False]
