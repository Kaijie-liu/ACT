#!/usr/bin/env python3
#===- tests/test_jsonl_v2_helpers.py - JSONL v2 Helpers -------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

from pathlib import Path

from act.pipeline.confignet.jsonl import (
    canonicalize_record_v2,
    summarize_jsonl_v2,
    write_record_jsonl,
)
from act.pipeline.confignet.schema_v2 import build_record_v2


def _base_record() -> dict:
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
    record = build_record_v2(
        run_meta=run_meta,
        seeds=seeds,
        instance=instance,
        l1=l1,
        l2=l2,
        final=final,
        timing={"wall_time": 0.1},
        created_at_utc="2025-01-01T00:00:00Z",
    )
    record["timestamp"] = 123.0
    return record


def test_canonicalize_record_v2_drops_nondet_fields() -> None:
    record_a = _base_record()
    record_b = _base_record()
    record_b["timing"] = {"wall_time": 9.9}
    record_b["created_at_utc"] = "2099-01-01T00:00:00Z"
    record_b["timestamp"] = 999.0
    assert canonicalize_record_v2(record_a) == canonicalize_record_v2(record_b)


def test_summarize_jsonl_v2_counts_and_aggregates(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    rec_pass = _base_record()
    rec_fail = _base_record()
    rec_err = _base_record()
    rec_fail["final"]["final_verdict"] = "FAILED"
    rec_fail["final"]["reason"] = "l2_failed"
    rec_fail["l2"]["status"] = "FAILED"
    rec_fail["l2"]["violations_total"] = 2
    rec_fail["l2"]["checks_total"] = 4
    rec_fail["l2"]["worst_gap"] = 0.5
    rec_err["final"]["final_verdict"] = "ERROR"
    rec_err["final"]["reason"] = "error"
    rec_err["l2"]["status"] = "ERROR"
    rec_err["l2"]["checks_total"] = 0
    rec_err["l2"]["violations_total"] = 0
    rec_err["l2"]["worst_gap"] = None

    write_record_jsonl(str(path), rec_pass)
    write_record_jsonl(str(path), rec_fail)
    write_record_jsonl(str(path), rec_err)

    summary = summarize_jsonl_v2(str(path))
    assert summary["records"] == 3
    assert summary["pass"] == 1
    assert summary["failed"] == 1
    assert summary["error"] == 1
    assert summary["by_reason"]["ok"] == 1
    assert summary["by_reason"]["l2_failed"] == 1
    assert summary["by_reason"]["error"] == 1
    assert summary["checks_total"] == 12
    assert summary["violations_total"] == 2
    assert summary["worst_gap_max"] == 0.5
