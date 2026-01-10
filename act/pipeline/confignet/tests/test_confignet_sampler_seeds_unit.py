#!/usr/bin/env python3
#===- tests/test_confignet_sampler_seeds_unit.py - Sampler Seeds ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

from act.pipeline.confignet.sampler import sample_instances
from act.pipeline.confignet.schema import ConfigNetConfig
from act.pipeline.confignet.seeds import derive_seed


def test_sample_instances_deterministic() -> None:
    cfg = ConfigNetConfig(num_instances=3, base_seed=0)
    a = [inst.to_dict() for inst in sample_instances(cfg)]
    b = [inst.to_dict() for inst in sample_instances(cfg)]
    assert a == b


def test_derive_seed_stable_and_distinct() -> None:
    a = derive_seed(123, 5, "abc")
    b = derive_seed(123, 5, "abc")
    c = derive_seed(123, 6, "abc")
    d = derive_seed(123, 5, "def")
    assert a == b
    assert a != c
    assert a != d


def test_seed_chain_uniqueness_small_range() -> None:
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
