#!/usr/bin/env python3
#===- act/pipeline/confignet/jsonl_schema.py - JSONL Schema v2 --------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Schema v2 for unified ConfigNet L1/L2 JSONL records.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .jsonl import canonical_hash_obj, compute_run_id

SCHEMA_VERSION_V2 = "confignet_l1l2_v2"


def build_record_v2(
    *,
    run_meta: Dict[str, Any],
    seeds: Dict[str, Any],
    instance: Dict[str, Any],
    l1: Dict[str, Any],
    l2: Dict[str, Any],
    final: Dict[str, Any],
    solver: Optional[Dict[str, Any]] = None,
    timing: Optional[Dict[str, Any]] = None,
    errors: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
    created_at_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a v2 JSONL record from component dicts.
    """
    record: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V2,
        "run_id": compute_run_id(run_meta),
        "run_meta": dict(run_meta),
        "seeds": dict(seeds),
        "instance": dict(instance),
        "l1": dict(l1),
        "l2": dict(l2),
        "solver": dict(solver)
        if solver is not None
        else {
            "status": "NOT_RUN",
            "solver_name": "NOT_RUN",
            "verdict": "NOT_RUN",
            "time_sec": None,
            "errors": [],
            "warnings": [],
            "details": {},
        },
        "final": dict(final),
    }
    if timing is not None:
        record["timing"] = dict(timing)
    if errors:
        record["errors"] = list(errors)
    if warnings:
        record["warnings"] = list(warnings)
    if created_at_utc is not None:
        record["created_at_utc"] = str(created_at_utc)

    record["canonical_hash"] = canonical_hash_obj(record)
    return record


def _require_key(d: Dict[str, Any], key: str) -> Any:
    assert key in d, f"missing key: {key}"
    return d[key]


def _require_type(val: Any, typ, name: str) -> None:
    assert isinstance(val, typ), f"{name} must be {typ}, got {type(val)}"


def validate_record_v2(record: Dict[str, Any]) -> None:
    """
    Validate required fields and types for schema v2.
    """
    _require_type(record, dict, "record")
    assert record.get("schema_version") == SCHEMA_VERSION_V2, "invalid schema_version"
    _require_type(_require_key(record, "run_id"), str, "run_id")
    _require_type(_require_key(record, "canonical_hash"), str, "canonical_hash")

    run_meta = _require_key(record, "run_meta")
    _require_type(run_meta, dict, "run_meta")
    for key, typ in (
        ("seed_root", int),
        ("instances", int),
        ("samples", int),
        ("tf_mode", str),
        ("strict", bool),
        ("device", str),
        ("dtype", str),
        ("atol", float),
        ("rtol", float),
        ("topk", int),
    ):
        _require_type(_require_key(run_meta, key), typ, f"run_meta.{key}")

    seeds = _require_key(record, "seeds")
    _require_type(seeds, dict, "seeds")
    for key in ("seed_root", "seed_instance", "seed_inputs"):
        _require_type(_require_key(seeds, key), int, f"seeds.{key}")

    instance = _require_key(record, "instance")
    _require_type(instance, dict, "instance")
    for key, typ in (
        ("net_family", str),
        ("arch", dict),
        ("spec", dict),
        ("input_shape", list),
        ("eps", (int, float)),
    ):
        _require_type(_require_key(instance, key), typ, f"instance.{key}")

    l1 = _require_key(record, "l1")
    _require_type(l1, dict, "l1")
    for key, typ in (
        ("status", str),
        ("counterexample_found", bool),
        ("verifier_status", str),
        ("policy_applied", bool),
        ("final_verdict_after_policy", str),
    ):
        _require_type(_require_key(l1, key), typ, f"l1.{key}")

    l1_status = l1["status"]
    assert l1_status in ("PASSED", "FAILED", "ERROR", "INCONCLUSIVE", "ACCEPTABLE"), "invalid l1.status"

    l2 = _require_key(record, "l2")
    _require_type(l2, dict, "l2")
    for key, typ in (
        ("status", str),
        ("tf_mode", str),
        ("samples", int),
        ("checks_total", int),
        ("violations_total", int),
        ("worst_gap", (int, float, type(None))),
        ("topk", list),
        ("layerwise_stats", list),
    ):
        _require_type(_require_key(l2, key), typ, f"l2.{key}")
    l2_status = l2["status"]
    assert l2_status in ("PASSED", "FAILED", "ERROR"), "invalid l2.status"

    final = _require_key(record, "final")
    _require_type(final, dict, "final")
    for key, typ in (
        ("final_verdict", str),
        ("exit_code", int),
        ("reason", str),
    ):
        _require_type(_require_key(final, key), typ, f"final.{key}")

    solver = _require_key(record, "solver")
    _require_type(solver, dict, "solver")
    for key, typ in (
        ("status", str),
        ("solver_name", str),
        ("verdict", str),
        ("time_sec", (int, float, type(None))),
        ("errors", list),
        ("warnings", list),
        ("details", dict),
    ):
        _require_type(_require_key(solver, key), typ, f"solver.{key}")
    assert final["final_verdict"] in ("PASS", "FAILED", "ERROR"), "invalid final.final_verdict"
