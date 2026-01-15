#!/usr/bin/env python3
#===- tests/test_confignet.py - Confignet Tests (examples_config) ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch
import yaml

from act.front_end.specs import InKind, OutKind
from act.pipeline.confignet import (
    ConfigNetConfig,
    InstanceSpec,
    MLPConfig,
    ModelFamily,
    InputSpecConfig,
    OutputSpecConfig,
    build_generated_instance,
    build_wrapped_model,
    canonical_hash,
    canonical_hash_obj,
    compute_run_id,
    derive_seed,
    materialize_to_examples_config,
    make_record,
    sample_instances,
    tensor_digest,
)
from act.pipeline.confignet_io import (
    DEFAULT_EXAMPLES_CONFIG,
    build_record_v2,
    validate_record_v2,
)
from act.pipeline.verification.model_factory import ModelFactory


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

class TestMaterializeExamplesConfig:
    def test_materialize_to_examples_config_creates_entries_and_nets(self, tmp_path: Path) -> None:
        cfg = ConfigNetConfig(
            num_instances=1,
            base_seed=1,
            families=(ModelFamily.MLP,),
        )
        inst = sample_instances(cfg)[0]
        gen = build_generated_instance(inst, device="cpu", dtype=torch.float64)

        config_copy = tmp_path / "examples_config.yaml"
        base_text = Path(DEFAULT_EXAMPLES_CONFIG).read_text(encoding="utf-8")
        config_copy.write_text(base_text, encoding="utf-8")
        nets_dir = tmp_path / "nets"

        name_map = materialize_to_examples_config(
            instances=[inst],
            generated=[gen],
            config_path=str(config_copy),
            nets_dir=str(nets_dir),
        )
        assert len(name_map) == 1
        name = list(name_map.values())[0]
        assert name.startswith("cfg_seed")

        data = yaml.safe_load(config_copy.read_text(encoding="utf-8"))
        assert name in data["networks"]
        net_path = nets_dir / f"{name}.json"
        assert net_path.exists()

        factory = ModelFactory(config_path=str(config_copy), nets_dir=str(nets_dir))
        model = factory.create_model(name, load_weights=True)
        assert model is not None


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
