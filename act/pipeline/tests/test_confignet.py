#!/usr/bin/env python3
#===- tests/test_confignet.py - Confignet Consolidated Tests ---------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import pytest
import torch

from act.front_end.specs import InKind, OutKind
from act.pipeline.cli import main as pipeline_main
from act.pipeline.confignet import (
    SCHEMA_VERSION_V1,
    ConfignetFactoryAdapter,
    ConfigNetConfig,
    InputSpecConfig,
    InstanceSpec,
    MLPConfig,
    ModelFamily,
    OutputSpecConfig,
    Verdict,
    _derive_seeds,
    _finalize_verdict,
    _run_l2,
    apply_policy,
    build_record_v2,
    build_wrapped_model,
    canonical_hash,
    canonical_hash_obj,
    compute_run_id,
    derive_seed,
    make_record,
    read_jsonl,
    run_confignet_l1l2,
    run_driver_levels,
    sample_feasible_inputs,
    sample_instances,
    tensor_digest,
    validate_record_v2,
    main as confignet_main,
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


def _make_args(tmp_path, **overrides) -> SimpleNamespace:
    base = dict(
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
        run_solver=False,
        solver="torchlp",
        solver_timeout=None,
        solver_on_failure=False,
        no_strict_input=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _base_record_payload() -> Tuple[Dict[str, object], Dict[str, int], Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object]]:
    run_meta = {
        "seed_root": 0,
        "instances": 2,
        "samples": 1,
        "tf_mode": "interval",
        "strict": True,
        "device": "cpu",
        "dtype": "float64",
        "atol": 1e-6,
        "rtol": 0.0,
        "topk": 10,
    }
    seeds = {"seed_root": 0, "seed_instance": 1, "seed_inputs": 2}
    instance = {
        "net_family": "MLP",
        "arch": {"depth": 2, "width": 8},
        "spec": {"eps": 0.1, "norm": "Linf"},
        "input_shape": [1, 2],
        "eps": 0.1,
    }
    l1 = {
        "status": "PASSED",
        "counterexample_found": False,
        "verifier_status": "NOT_RUN",
        "policy_applied": True,
        "final_verdict_after_policy": "PASS",
    }
    l2 = {
        "status": "PASSED",
        "tf_mode": "interval",
        "samples": 1,
        "checks_total": 8,
        "violations_total": 0,
        "worst_gap": 0.0,
        "worst_layer": None,
        "topk": [],
        "layerwise_stats": [],
    }
    final = {"final_verdict": "PASS", "exit_code": 0, "reason": "ok"}
    return run_meta, seeds, instance, l1, l2, final


class TestSamplerSeeds:
    def test_sample_instances_deterministic(self) -> None:
        cfg = ConfigNetConfig(num_instances=3, base_seed=0)
        a = [inst.to_dict() for inst in sample_instances(cfg)]
        b = [inst.to_dict() for inst in sample_instances(cfg)]
        assert a == b

    def test_derive_seed_stable_and_distinct(self) -> None:
        a = derive_seed(123, 5, "abc")
        b = derive_seed(123, 5, "abc")
        c = derive_seed(123, 6, "abc")
        d = derive_seed(123, 5, "def")
        assert a == b
        assert a != c
        assert a != d

    def test_seed_chain_uniqueness_small_range(self) -> None:
        seed_root = 7
        instance_seeds = []
        input_seeds = []
        for idx in range(10):
            instance_id = f"cfg_{idx}"
            seed_instance = derive_seed(seed_root, idx, instance_id)
            seed_inputs = derive_seed(seed_instance, 0, "inputs|l1l2")
            instance_seeds.append(seed_instance)
            input_seeds.append(seed_inputs)
        assert len(set(instance_seeds)) == len(instance_seeds)
        assert len(set(input_seeds)) == len(input_seeds)

    def test_driver_seed_inputs_from_instance(self) -> None:
        inst = _make_instance("cfg_0", 11)
        seeds = _derive_seeds(0, inst.seed)
        assert seeds["seed_instance"] == inst.seed
        assert seeds["seed_inputs"] == derive_seed(inst.seed, 0, "inputs|l1l2")


class TestBuild:
    def test_build_wrapped_model_repeatable(self) -> None:
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

    def test_build_wrapped_model_no_rng_side_effect(self) -> None:
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


class TestSamplingSpecs:
    def _build_one(self, cfg: ConfigNetConfig) -> None:
        inst = sample_instances(cfg)[0]
        _ = build_wrapped_model(inst, device="cpu", dtype=torch.float64)

    def test_sample_lin_poly_builds(self) -> None:
        cfg = ConfigNetConfig(
            num_instances=1,
            base_seed=7,
            families=(ModelFamily.MLP,),
            input_kind_choices=(InKind.LIN_POLY,),
            output_kind_choices=(OutKind.TOP1_ROBUST,),
        )
        self._build_one(cfg)

    def test_sample_output_linear_le_builds(self) -> None:
        cfg = ConfigNetConfig(
            num_instances=1,
            base_seed=8,
            families=(ModelFamily.MLP,),
            output_kind_choices=(OutKind.LINEAR_LE,),
        )
        self._build_one(cfg)

    def test_sample_output_range_builds(self) -> None:
        cfg = ConfigNetConfig(
            num_instances=1,
            base_seed=9,
            families=(ModelFamily.MLP,),
            output_kind_choices=(OutKind.RANGE,),
        )
        self._build_one(cfg)

    def test_input_sampling_status_ok(self) -> None:
        inst = InstanceSpec(
            instance_id="linpoly_ok",
            seed=1,
            family=ModelFamily.MLP,
            model_cfg=MLPConfig(
                input_shape=(1, 1),
                hidden_sizes=(2,),
                activation="relu",
                dropout_p=0.0,
                num_classes=2,
            ),
            input_spec=InputSpecConfig(
                kind=InKind.LIN_POLY,
                value_range=(0.0, 1.0),
                lb_val=0.0,
                ub_val=1.0,
                derive_poly_from_box=True,
            ),
            output_spec=OutputSpecConfig(
                kind=OutKind.TOP1_ROBUST,
                y_true=0,
            ),
            meta={},
        )

        xs, status = sample_feasible_inputs(
            inst,
            num_samples=3,
            seed=0,
            device="cpu",
            dtype=torch.float64,
            strict_input=True,
            return_status=True,
        )
        assert len(xs) == 3
        assert status["input_sampling_status"] == "ok"
        assert "rejection_rate" in status

    def test_input_sampling_status_fallback(self) -> None:
        A = torch.tensor([[1.0]], dtype=torch.float64)
        b = torch.tensor([-1.0], dtype=torch.float64)

        inst = InstanceSpec(
            instance_id="linpoly_bad",
            seed=2,
            family=ModelFamily.MLP,
            model_cfg=MLPConfig(
                input_shape=(1, 1),
                hidden_sizes=(2,),
                activation="relu",
                dropout_p=0.0,
                num_classes=2,
            ),
            input_spec=InputSpecConfig(
                kind=InKind.LIN_POLY,
                value_range=(0.0, 1.0),
                A=A,
                b=b,
                derive_poly_from_box=False,
            ),
            output_spec=OutputSpecConfig(
                kind=OutKind.TOP1_ROBUST,
                y_true=0,
            ),
            meta={},
        )

        xs, status = sample_feasible_inputs(
            inst,
            num_samples=1,
            seed=0,
            device="cpu",
            dtype=torch.float64,
            strict_input=False,
            max_tries=5,
            return_status=True,
        )
        assert len(xs) == 1
        assert status["input_sampling_status"] == "fallback"

    def test_lin_poly_rejection_sampling_satisfies_constraints(self) -> None:
        A = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
        b = torch.tensor([1.0, 0.0], dtype=torch.float64)

        inst = InstanceSpec(
            instance_id="linpoly_unit",
            seed=123,
            family=ModelFamily.MLP,
            model_cfg=MLPConfig(
                input_shape=(1, 1),
                hidden_sizes=(2,),
                activation="relu",
                dropout_p=0.0,
                num_classes=2,
            ),
            input_spec=InputSpecConfig(
                kind=InKind.LIN_POLY,
                value_range=(0.0, 1.0),
                A=A,
                b=b,
                derive_poly_from_box=False,
            ),
            output_spec=OutputSpecConfig(
                kind=OutKind.TOP1_ROBUST,
                y_true=0,
            ),
            meta={},
        )

        xs = sample_feasible_inputs(
            inst,
            num_samples=5,
            seed=0,
            device="cpu",
            dtype=torch.float64,
            strict_input=True,
        )
        for x in xs:
            x_flat = x.reshape(-1)
            assert torch.all(A.matmul(x_flat) <= b + 1e-8)


class TestPolicy:
    def test_policy_falsified_when_level1_has_cex(self) -> None:
        final_v, gating = apply_policy(
            base_verdict=Verdict.CERTIFIED,
            level1_found_cex=True,
            level1_summary={"found_cex": True, "num_cex": 1},
        )
        assert final_v == Verdict.FALSIFIED
        assert gating.prohibited_certified_due_to_cex is True

    def test_driver_gating_enforces_falsified_on_cex(self, monkeypatch) -> None:
        import act.pipeline.confignet as driver_levels

        def _fake_level1_suite(*_args, **_kwargs):
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


class TestDriverL1Policy:
    def test_l1_counterexample_forces_failed(self, monkeypatch, tmp_path) -> None:
        insts = [_make_instance("cfg_0", 11)]

        def _fake_sample_instances(_cfg) -> List[InstanceSpec]:
            return insts

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

        monkeypatch.setattr("act.pipeline.confignet.sample_instances", _fake_sample_instances)
        monkeypatch.setattr("act.pipeline.confignet.ConfignetFactoryAdapter.build_case", _fake_build_case)
        monkeypatch.setattr("act.pipeline.confignet._run_l1", _fake_run_l1)
        monkeypatch.setattr("act.pipeline.confignet._run_l2", _fake_run_l2)

        args = _make_args(tmp_path)
        summary = run_confignet_l1l2(args)
        records = read_jsonl(args.jsonl)
        assert records[0]["l1"]["counterexample_found"] is True
        assert records[0]["final"]["final_verdict"] == "FAILED"
        assert records[0]["final"]["final_verdict"] != "PASS"
        assert summary["exit_code"] == 1

    def test_l1_no_cex_allows_pass(self, monkeypatch, tmp_path) -> None:
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

        monkeypatch.setattr("act.pipeline.confignet.sample_instances", _fake_sample_instances)
        monkeypatch.setattr("act.pipeline.confignet.ConfignetFactoryAdapter.build_case", _fake_build_case)
        monkeypatch.setattr("act.pipeline.confignet._run_l1", _fake_run_l1)
        monkeypatch.setattr("act.pipeline.confignet._run_l2", _fake_run_l2)

        args = _make_args(tmp_path)
        summary = run_confignet_l1l2(args)
        records = read_jsonl(args.jsonl)
        assert records[0]["l1"]["counterexample_found"] is False
        assert records[0]["final"]["final_verdict"] == "PASS"
        assert summary["exit_code"] == 0


class TestDriverL1L2Unit:
    def test_derive_seeds_deterministic(self) -> None:
        a = _derive_seeds(0, 3)
        b = _derive_seeds(0, 3)
        c = _derive_seeds(0, 4)
        assert a == b
        assert a["seed_instance"] != c["seed_instance"]

    def test_finalize_verdict_priority(self) -> None:
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

    def test_driver_builds_valid_record_v2(self, monkeypatch, tmp_path) -> None:
        insts = [_make_instance("cfg_0", 11), _make_instance("cfg_1", 22)]

        def _fake_sample_instances(_cfg) -> List[InstanceSpec]:
            return insts

        def _fake_build_case_local(self, inst, *, seed_inputs, num_samples=1, deterministic_algos=False, strict_input=True):
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

        monkeypatch.setattr("act.pipeline.confignet.sample_instances", _fake_sample_instances)
        monkeypatch.setattr("act.pipeline.confignet.ConfignetFactoryAdapter.build_case", _fake_build_case_local)
        monkeypatch.setattr("act.pipeline.confignet._run_l1", _fake_run_l1)
        monkeypatch.setattr("act.pipeline.confignet._run_l2", _fake_run_l2)

        args = _make_args(tmp_path, instances=2)
        summary = run_confignet_l1l2(args)
        assert summary["records"] == 2

        lines = (tmp_path / "out.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        records = [json.loads(line) for line in lines]
        for rec in records:
            validate_record_v2(rec)
        run_ids = {rec["run_id"] for rec in records}
        hashes = {rec["canonical_hash"] for rec in records}
        assert len(run_ids) == 1
        assert len(hashes) == 2

    def test_driver_summary_counts_match_records(self, monkeypatch, tmp_path) -> None:
        insts = [
            _make_instance("cfg_0", 11),
            _make_instance("cfg_1", 22),
            _make_instance("cfg_2", 33),
        ]

        def _fake_sample_instances(_cfg) -> List[InstanceSpec]:
            return insts

        def _fake_build_case_local(self, inst, *, seed_inputs, num_samples=1, deterministic_algos=False, strict_input=True):
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

        monkeypatch.setattr("act.pipeline.confignet.sample_instances", _fake_sample_instances)
        monkeypatch.setattr("act.pipeline.confignet.ConfignetFactoryAdapter.build_case", _fake_build_case_local)
        monkeypatch.setattr("act.pipeline.confignet._run_l1", _fake_run_l1)
        monkeypatch.setattr("act.pipeline.confignet._run_l2", _fake_run_l2)

        args = _make_args(tmp_path, instances=3)
        summary = run_confignet_l1l2(args)

        lines = (tmp_path / "out.jsonl").read_text().strip().splitlines()
        records = [json.loads(line) for line in lines]
        counts = {"PASS": 0, "FAILED": 0, "ERROR": 0}
        for rec in records:
            counts[rec["final"]["final_verdict"]] += 1

        assert summary["passed"] == counts["PASS"]
        assert summary["failed"] == counts["FAILED"]
        assert summary["errors"] == counts["ERROR"]
        assert summary["exit_code"] == 2

    def test_run_l2_maps_validation_status_to_schema(self, monkeypatch) -> None:
        inst = _make_instance("cfg_0", 11)
        seeds = _derive_seeds(0, inst.seed)
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
                    "network": inst.instance_id,
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
            "act.pipeline.confignet.ConfignetFactoryAdapter.configure_for_validation",
            _fake_configure_for_validation,
        )
        monkeypatch.setattr(validator, "validate_bounds_per_neuron", _fake_validate_bounds_per_neuron)

        l2 = _run_l2(
            inst,
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
                "arch": inst.model_cfg.to_dict(),
                "spec": {
                    "input_spec": inst.input_spec.to_dict(),
                    "output_spec": inst.output_spec.to_dict(),
                },
                "input_shape": list(inst.model_cfg.input_shape),
                "eps": float(inst.input_spec.eps or 0.0),
            },
            l1=l1,
            l2=l2,
            final=final,
        )
        validate_record_v2(record)

    def test_adapter_generate_test_input_uses_seed_inputs_base(self, monkeypatch) -> None:
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
            "act.pipeline.confignet_adapter.sample_feasible_inputs",
            _fake_sample_feasible_inputs,
        )

        adapter.generate_test_input(inst.instance_id)
        adapter.generate_test_input(inst.instance_id)
        assert seen[0] == derive_seed(123, 0, "inputs|level2")
        assert seen[1] == derive_seed(123, 1, "inputs|level2")

    def test_driver_act_build_error_produces_l2_error(self, monkeypatch, tmp_path) -> None:
        insts = [_make_instance("cfg_0", 11)]

        def _fake_sample_instances(_cfg) -> List[InstanceSpec]:
            return insts

        def _fake_build_case_error(self, inst, *, seed_inputs, num_samples=1, deterministic_algos=False, strict_input=True):
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

        monkeypatch.setattr("act.pipeline.confignet.sample_instances", _fake_sample_instances)
        monkeypatch.setattr("act.pipeline.confignet.ConfignetFactoryAdapter.build_case", _fake_build_case_error)
        monkeypatch.setattr("act.pipeline.confignet._run_l1", _fake_run_l1)

        args = _make_args(tmp_path)
        summary = run_confignet_l1l2(args)
        assert summary["errors"] == 1
        assert summary["exit_code"] == 2

        lines = (tmp_path / "out.jsonl").read_text().strip().splitlines()
        record = json.loads(lines[0])
        assert record["l2"]["status"] == "ERROR"
        assert record["final"]["final_verdict"] == "ERROR"
        assert "boom" in " ".join(record.get("errors", []))

    def test_adapter_generate_test_input_respects_strict_input(self, monkeypatch) -> None:
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
            "act.pipeline.confignet_adapter.sample_feasible_inputs",
            _fake_sample_feasible_inputs,
        )

        adapter.generate_test_input(inst.instance_id)
        assert seen == [False]


class TestL2Mapping:
    def _fake_validate(self, validator: VerificationValidator, inst: InstanceSpec, max_gap: float) -> None:
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
                    {"layer_id": 2, "num_violations": 2, "max_gap": max_gap, "kind": "Linear"},
                ],
                "errors": [],
                "warnings": [],
            }
        )

    def test_l2_mapping_selects_worst_layer(self) -> None:
        inst = _make_instance("cfg_0", 11)
        validator = VerificationValidator(device="cpu", dtype=torch.float64)
        adapter = ConfignetFactoryAdapter(device="cpu", dtype=torch.float64)

        def _fake_configure_for_validation(*_args, **_kwargs):
            return None

        def _fake_validate_bounds_per_neuron(*_args, **_kwargs):
            self._fake_validate(validator, inst, 0.5)
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

    def test_l2_hybridz_mapping_selects_worst_layer(self) -> None:
        inst = _make_instance("cfg_0", 11)
        validator = VerificationValidator(device="cpu", dtype=torch.float64)
        adapter = ConfignetFactoryAdapter(device="cpu", dtype=torch.float64)

        def _fake_configure_for_validation(*_args, **_kwargs):
            return None

        def _fake_validate_bounds_per_neuron(*_args, **_kwargs):
            self._fake_validate(validator, inst, 0.5)
            return {}

        adapter.configure_for_validation = _fake_configure_for_validation
        validator.validate_bounds_per_neuron = _fake_validate_bounds_per_neuron

        l2 = _run_l2(
            inst,
            object(),
            object(),
            device="cpu",
            dtype=torch.float64,
            tf_mode="hybridz",
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


class TestJsonl:
    def test_tensor_digest_stable_sha(self) -> None:
        t = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        d1 = tensor_digest(t)
        d2 = tensor_digest(t)
        assert d1["sha256"] == d2["sha256"]

    def test_tensor_digest_inline_values_size(self) -> None:
        small = torch.arange(4, dtype=torch.float32)
        small_digest = tensor_digest(small, max_inline_numel=8)
        assert "values" in small_digest

        large = torch.arange(100, dtype=torch.float32)
        large_digest = tensor_digest(large, max_inline_numel=8)
        assert "values" not in large_digest

    def test_confignet_stable_hash_same_payload(self) -> None:
        cfg = ConfigNetConfig(num_instances=1, base_seed=123)
        a = sample_instances(cfg)[0]
        b = sample_instances(cfg)[0]

        ha = canonical_hash(a.to_dict())
        hb = canonical_hash(b.to_dict())
        assert ha == hb

    def test_confignet_hash_changes_on_field_change(self) -> None:
        cfg = ConfigNetConfig(num_instances=1, base_seed=123)
        inst = sample_instances(cfg)[0]

        altered = replace(inst, output_spec=replace(inst.output_spec, y_true=(inst.output_spec.y_true or 0) + 1))

        h1 = canonical_hash(inst.to_dict())
        h2 = canonical_hash(altered.to_dict())
        assert h1 != h2

    def test_make_record_includes_git_sha(self) -> None:
        payload = {"k": "v"}
        rec = make_record(payload, include_timestamp=False)
        assert "git_sha" in rec

    def test_jsonl_schema_v1_fields_present(self, monkeypatch) -> None:
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
            self._generated = {gi.instance_spec.instance_id: gi for gi in generated}
            self._act_nets = {inst.instance_id: object() for inst in instances}
            self._call_counts = {inst.instance_id: 0 for inst in instances}
            self._seed_inputs_by_name = seed_inputs_by_name or {}
            self._strict_input = bool(strict_input)

        def _fake_validate_counterexamples(self, networks=None, solvers=None):
            nets = networks or []
            sols = solvers or ["torchlp"]
            for net in nets:
                for solver in sols:
                    self.validation_results.append(
                        {
                            "network": net,
                            "solver": solver,
                            "validation_type": "counterexample",
                            "concrete_counterexample": False,
                            "verifier_result": "UNKNOWN",
                            "validation_status": "INCONCLUSIVE",
                        }
                    )
            return {"total": len(nets) * len(sols), "passed": 0, "failed": 0, "errors": 0}

        monkeypatch.setattr(
            "act.pipeline.confignet.ConfignetFactoryAdapter.configure_for_validation",
            _fake_configure_for_validation,
        )
        monkeypatch.setattr(
            "act.pipeline.verification.validate_verifier.VerificationValidator.validate_counterexamples",
            _fake_validate_counterexamples,
        )

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

    def test_validate_record_v2_happy_path(self) -> None:
        run_meta, seeds, instance, l1, l2, final = _base_record_payload()
        record = build_record_v2(
            run_meta=run_meta,
            seeds=seeds,
            instance=instance,
            l1=l1,
            l2=l2,
            final=final,
            timing={"wall_time": 0.1},
        )
        validate_record_v2(record)

    def test_validate_record_v2_missing_field_fails(self) -> None:
        run_meta, seeds, instance, l1, l2, final = _base_record_payload()
        record = build_record_v2(
            run_meta=run_meta,
            seeds=seeds,
            instance=instance,
            l1=l1,
            l2=l2,
            final=final,
        )
        del record["l2"]["violations_total"]
        with pytest.raises(AssertionError):
            validate_record_v2(record)

    def test_run_id_deterministic(self) -> None:
        run_meta, *_rest = _base_record_payload()
        r1 = compute_run_id(run_meta)
        r2 = compute_run_id(run_meta)
        assert r1 == r2

    def test_canonical_hash_ignores_timing(self) -> None:
        run_meta, seeds, instance, l1, l2, final = _base_record_payload()
        record_a = build_record_v2(
            run_meta=run_meta,
            seeds=seeds,
            instance=instance,
            l1=l1,
            l2=l2,
            final=final,
            timing={"wall_time": 0.1},
        )
        record_b = build_record_v2(
            run_meta=run_meta,
            seeds=seeds,
            instance=instance,
            l1=l1,
            l2=l2,
            final=final,
            timing={"wall_time": 9.9},
        )
        assert canonical_hash_obj(record_a) == canonical_hash_obj(record_b)


class TestDiagnostics:
    def test_failed_l2_prints_worst_layer_and_topk(self, caplog, monkeypatch, tmp_path) -> None:
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

        monkeypatch.setattr("act.pipeline.confignet.sample_instances", _fake_sample_instances)
        monkeypatch.setattr("act.pipeline.confignet.ConfignetFactoryAdapter.build_case", _fake_build_case)
        monkeypatch.setattr("act.pipeline.confignet._run_l1", _fake_run_l1)
        monkeypatch.setattr("act.pipeline.confignet._run_l2", _fake_run_l2)

        caplog.set_level(logging.INFO, logger="act.pipeline.confignet")
        run_confignet_l1l2(_make_args(tmp_path))
        text = caplog.text
        assert "🚨 [CN][l1l2]" in text
        assert "violations_total=7" in text
        assert "worst_gap=0.123" in text
        assert "worst_layer=" in text
        assert "topk:" in text
        assert "neuron_index" in text

    def test_error_prints_first_three_errors(self, caplog, monkeypatch, tmp_path) -> None:
        insts = [_make_instance("cfg_0", 11)]

        def _fake_sample_instances(_cfg) -> List[InstanceSpec]:
            return insts

        def _fake_build_case_error(self, inst, *, seed_inputs, num_samples=1, deterministic_algos=False, strict_input=True):
            raise RuntimeError("boom-0")

        monkeypatch.setattr("act.pipeline.confignet.sample_instances", _fake_sample_instances)
        monkeypatch.setattr("act.pipeline.confignet.ConfignetFactoryAdapter.build_case", _fake_build_case_error)

        caplog.set_level(logging.INFO, logger="act.pipeline.confignet")
        run_confignet_l1l2(_make_args(tmp_path))
        text = caplog.text
        assert "❌ [CN][l1l2]" in text
        assert "errors:" in text
        assert "boom-0" in text

    def test_exception_still_writes_jsonl_record(self, tmp_path, monkeypatch) -> None:
        insts = [_make_instance("cfg_0", 11)]

        def _fake_sample_instances(_cfg) -> List[InstanceSpec]:
            return insts

        def _fake_build_case_error(self, inst, *, seed_inputs, num_samples=1, deterministic_algos=False, strict_input=True):
            raise RuntimeError("boom-1")

        monkeypatch.setattr("act.pipeline.confignet.sample_instances", _fake_sample_instances)
        monkeypatch.setattr("act.pipeline.confignet.ConfignetFactoryAdapter.build_case", _fake_build_case_error)

        args = _make_args(tmp_path)
        run_confignet_l1l2(args)
        records = read_jsonl(args.jsonl)
        assert len(records) == 1
        validate_record_v2(records[0])
        assert records[0]["final"]["final_verdict"] == "ERROR"
        assert "boom-1" in " ".join(records[0].get("errors", []))


class TestSolverIntegration:
    def test_solver_field_present_not_run(self) -> None:
        run_meta, seeds, instance, l1, l2, final = _base_record_payload()
        record = build_record_v2(
            run_meta=run_meta,
            seeds=seeds,
            instance=instance,
            l1=l1,
            l2=l2,
            final=final,
        )
        validate_record_v2(record)
        assert record["solver"]["status"] == "NOT_RUN"
        assert record["solver"]["verdict"] == "NOT_RUN"

    def test_solver_gating_l1_overrides_certified(self, monkeypatch, tmp_path) -> None:
        insts = [_make_instance("cfg_0", 11)]

        def _fake_sample_instances(_cfg) -> List[InstanceSpec]:
            return insts

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

        def _fake_run_solver(*_args, **_kwargs):
            return {
                "status": "PASSED",
                "solver_name": "torchlp",
                "verdict": "CERTIFIED",
                "time_sec": 0.01,
                "errors": [],
                "warnings": [],
                "details": {},
            }

        monkeypatch.setattr("act.pipeline.confignet.sample_instances", _fake_sample_instances)
        monkeypatch.setattr("act.pipeline.confignet.ConfignetFactoryAdapter.build_case", _fake_build_case)
        monkeypatch.setattr("act.pipeline.confignet._run_l1", _fake_run_l1)
        monkeypatch.setattr("act.pipeline.confignet._run_l2", _fake_run_l2)
        monkeypatch.setattr("act.pipeline.confignet._run_solver", _fake_run_solver)

        summary = run_confignet_l1l2(_make_args(tmp_path, run_solver=True, solver_on_failure=True))
        records = read_jsonl(_make_args(tmp_path, run_solver=True, solver_on_failure=True).jsonl)
        assert summary["failed"] == 1
        assert summary["exit_code"] == 1
        assert records[0]["final"]["final_verdict"] == "FAILED"


class TestCli:
    def test_cli_level1_runs(self, tmp_path: Path) -> None:
        out_jsonl = tmp_path / "level1.jsonl"
        code = confignet_main([
            "level1",
            "--num",
            "1",
            "--seed",
            "0",
            "--n-inputs",
            "2",
            "--device",
            "cpu",
            "--dtype",
            "float64",
            "--out_jsonl",
            str(out_jsonl),
        ])
        assert code == 0
        assert out_jsonl.exists()


class TestE2E:
    def _strip_nondet(self, record: Dict[str, object]) -> Dict[str, object]:
        record = dict(record)
        record.pop("timing", None)
        record.pop("created_at_utc", None)
        return record

    def test_confignet_l1l2_e2e_minimal_writes_jsonl_v2(self, tmp_path, monkeypatch) -> None:
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

        monkeypatch.setattr("act.pipeline.confignet.sample_instances", _fake_sample_instances)
        monkeypatch.setattr("act.pipeline.confignet.ConfignetFactoryAdapter.build_case", _fake_build_case)
        monkeypatch.setattr("act.pipeline.confignet._run_l1", _fake_run_l1)
        monkeypatch.setattr("act.pipeline.confignet._run_l2", _fake_run_l2)

        args = _make_args(tmp_path, instances=2, jsonl=str(tmp_path / "out.jsonl"))
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

    def test_confignet_l1l2_determinism_same_seed_same_hash(self, tmp_path, monkeypatch) -> None:
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

        monkeypatch.setattr("act.pipeline.confignet.sample_instances", _fake_sample_instances)
        monkeypatch.setattr("act.pipeline.confignet.ConfignetFactoryAdapter.build_case", _fake_build_case)
        monkeypatch.setattr("act.pipeline.confignet._run_l1", _fake_run_l1)
        monkeypatch.setattr("act.pipeline.confignet._run_l2", _fake_run_l2)

        args_a = _make_args(tmp_path, instances=2, jsonl=str(tmp_path / "a.jsonl"))
        args_b = _make_args(tmp_path, instances=2, jsonl=str(tmp_path / "b.jsonl"))
        run_confignet_l1l2(args_a)
        run_confignet_l1l2(args_b)

        rec_a = read_jsonl(args_a.jsonl)
        rec_b = read_jsonl(args_b.jsonl)
        assert len(rec_a) == len(rec_b)
        for a, b in zip(rec_a, rec_b):
            assert a["run_id"] == b["run_id"]
            assert a["canonical_hash"] == b["canonical_hash"]

    def test_confignet_top_level_routing_produces_same_records(self, tmp_path, monkeypatch) -> None:
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

        monkeypatch.setattr("act.pipeline.confignet.sample_instances", _fake_sample_instances)
        monkeypatch.setattr("act.pipeline.confignet.ConfignetFactoryAdapter.build_case", _fake_build_case)
        monkeypatch.setattr("act.pipeline.confignet._run_l1", _fake_run_l1)
        monkeypatch.setattr("act.pipeline.confignet._run_l2", _fake_run_l2)

        top_path = tmp_path / "top.jsonl"
        sub_path = tmp_path / "sub.jsonl"

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

        top_records = [self._strip_nondet(r) for r in read_jsonl(top_path)]
        sub_records = [self._strip_nondet(r) for r in read_jsonl(sub_path)]
        assert top_records == sub_records


class TestSmoke:
    def test_confignet_l1l2_smoke_real(self, tmp_path) -> None:
        args = _make_args(
            tmp_path,
            samples=1,
            topk=3,
            jsonl=str(tmp_path / "smoke.jsonl"),
        )
        summary = run_confignet_l1l2(args)
        records = read_jsonl(args.jsonl)
        assert summary["records"] == 1
        assert len(records) == 1
        validate_record_v2(records[0])
        assert records[0]["schema_version"] == "confignet_l1l2_v2"
