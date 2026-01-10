#!/usr/bin/env python3
#===- tests/test_confignet_jsonl_hash_stable.py - JSONL Hash Tests -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

from dataclasses import replace

from act.pipeline.confignet.schema import ConfigNetConfig
from act.pipeline.confignet.sampler import sample_instances
from act.pipeline.confignet.jsonl import canonical_hash, make_record


def test_confignet_stable_hash_same_payload() -> None:
    cfg = ConfigNetConfig(num_instances=1, base_seed=123)
    a = sample_instances(cfg)[0]
    b = sample_instances(cfg)[0]

    ha = canonical_hash(a.to_dict())
    hb = canonical_hash(b.to_dict())
    assert ha == hb


def test_confignet_hash_changes_on_field_change() -> None:
    cfg = ConfigNetConfig(num_instances=1, base_seed=123)
    inst = sample_instances(cfg)[0]

    altered = replace(inst, output_spec=replace(inst.output_spec, y_true=(inst.output_spec.y_true or 0) + 1))

    h1 = canonical_hash(inst.to_dict())
    h2 = canonical_hash(altered.to_dict())
    assert h1 != h2


def test_make_record_includes_git_sha() -> None:
    payload = {"k": "v"}
    rec = make_record(payload, include_timestamp=False)
    assert "git_sha" in rec
