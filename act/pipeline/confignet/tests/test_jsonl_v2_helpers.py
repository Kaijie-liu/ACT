#!/usr/bin/env python3
#===- tests/test_jsonl_v2_helpers.py - JSONL v2 Helpers -------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

from act.pipeline.confignet.jsonl import (
    canonicalize_record_v2,
    summarize_jsonl_v2,
    write_record_jsonl,
)
from act.pipeline.confignet.schema_v2 import build_record_v2


def _base_record(final_verdict: str, *, reason: str, worst_gap: float) -> dict:
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
        "topk": 5,
    }
    seeds = {"seed_root": 0, "seed_instance": 1, "seed_inputs": 2}
    instance = {
        "net_family": "MLP",
        "arch": {"depth": 1},
        "spec": {"input_spec": {}, "output_spec": {}},
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
        "checks_total": 3,
        "violations_total": 0,
        "worst_gap": worst_gap,
        "worst_layer": None,
        "topk": [],
        "layerwise_stats": [],
    }
    final = {"final_verdict": final_verdict, "exit_code": 0, "reason": reason}
    return build_record_v2(
        run_meta=run_meta,
        seeds=seeds,
        instance=instance,
        l1=l1,
        l2=l2,
        final=final,
        timing={"wall_time": 0.1},
    )


def test_canonicalize_record_v2_drops_nondet_fields() -> None:
    rec_a = _base_record("PASS", reason="ok", worst_gap=0.0)
    rec_b = _base_record("PASS", reason="ok", worst_gap=0.0)
    rec_a["timing"]["wall_time"] = 0.1
    rec_b["timing"]["wall_time"] = 9.9
    rec_a["created_at_utc"] = "2025-01-01T00:00:00Z"
    rec_b["created_at_utc"] = "2025-01-02T00:00:00Z"
    rec_a["solver"]["time_sec"] = 0.1
    rec_b["solver"]["time_sec"] = 9.9
    canon_a = canonicalize_record_v2(rec_a)
    canon_b = canonicalize_record_v2(rec_b)
    assert canon_a == canon_b


def test_summarize_jsonl_v2_counts_and_aggregates(tmp_path) -> None:
    p = tmp_path / "out.jsonl"
    write_record_jsonl(str(p), _base_record("PASS", reason="ok", worst_gap=0.0))
    write_record_jsonl(str(p), _base_record("FAILED", reason="l1_failed", worst_gap=0.5))
    write_record_jsonl(str(p), _base_record("ERROR", reason="error", worst_gap=0.2))

    summary = summarize_jsonl_v2(str(p))
    assert summary["records"] == 3
    assert summary["pass"] == 1
    assert summary["failed"] == 1
    assert summary["error"] == 1
    assert summary["by_reason"]["ok"] == 1
    assert summary["by_reason"]["l1_failed"] == 1
    assert summary["by_reason"]["error"] == 1
    assert summary["checks_total"] == 9
    assert summary["violations_total"] == 0
    assert summary["worst_gap_max"] == 0.5
