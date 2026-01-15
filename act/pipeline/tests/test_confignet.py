#!/usr/bin/env python3
#===- tests/test_confignet.py - Confignet Tests (examples_config) ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

from pathlib import Path
import torch
import yaml

from act.front_end.specs import InKind, OutKind
from act.back_end.confignet import (
    ConfigNetConfig,
    InstanceSpec,
    MLPConfig,
    ModelFamily,
    InputSpecConfig,
    OutputSpecConfig,
    build_generated_instance,
    build_wrapped_model,
    derive_seed,
    materialize_to_examples_config,
    sample_instances,
)
from act.back_end.confignet_io import DEFAULT_EXAMPLES_CONFIG
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


 
