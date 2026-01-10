#!/usr/bin/env python3
#===- tests/test_confignet_jsonl_schema_v2.py - JSONL Schema v2 -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

import pytest

from act.pipeline.confignet.schema_v2 import build_record_v2, validate_record_v2
from act.pipeline.confignet.jsonl import canonical_hash_obj, compute_run_id


def _base_payload():
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


def test_validate_record_v2_happy_path() -> None:
    run_meta, seeds, instance, l1, l2, final = _base_payload()
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


def test_validate_record_v2_missing_field_fails() -> None:
    run_meta, seeds, instance, l1, l2, final = _base_payload()
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


def test_run_id_deterministic() -> None:
    run_meta, _, _, _, _, _ = _base_payload()
    r1 = compute_run_id(run_meta)
    r2 = compute_run_id(run_meta)
    assert r1 == r2


def test_canonical_hash_ignores_timing() -> None:
    run_meta, seeds, instance, l1, l2, final = _base_payload()
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
