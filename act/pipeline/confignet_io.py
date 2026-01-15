#!/usr/bin/env python3
#===- act/pipeline/confignet_io.py - ConfigNet JSONL IO ---------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   JSONL audit utilities and schema validation for ConfigNet runs.
#   - canonical hashing + run_id
#   - JSONL read/write helpers
#   - v2 record build/validate
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import torch

SCHEMA_VERSION_V2 = "confignet_l1l2_v2"


def _canonical_dumps(obj: Any) -> str:
    """Stable JSON serialization with sorted keys and compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_hash(obj: Any) -> str:
    """SHA256 hash of canonical JSON representation."""
    data = _canonical_dumps(obj).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _strip_keys(obj: Any, ignore_keys: Set[str]) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_keys(v, ignore_keys) for k, v in obj.items() if k not in ignore_keys}
    if isinstance(obj, list):
        return [_strip_keys(v, ignore_keys) for v in obj]
    return obj


def canonical_hash_obj(obj: Dict[str, Any], *, ignore_keys: Optional[Set[str]] = None) -> str:
    """
    Canonical hash with optional key filtering for non-deterministic fields.
    """
    ignore = ignore_keys or {"timing", "created_at_utc", "timestamp"}
    stripped = _strip_keys(obj, ignore)
    return canonical_hash(stripped)


def compute_run_id(run_meta: Dict[str, Any]) -> str:
    """Compute deterministic run_id from run_meta."""
    return canonical_hash_obj(run_meta, ignore_keys=set())


def current_git_sha() -> str:
    """Return current git SHA (or 'unknown' if unavailable)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        sha = out.stdout.strip()
        return sha if sha else "unknown"
    except Exception:
        return "unknown"


def write_jsonl_records(path: str, records: Iterable[Dict[str, Any]], sort_keys: bool = False) -> None:
    """Write JSONL records to path, creating parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=sort_keys, ensure_ascii=True))
            f.write("\n")


def write_record_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append a single JSONL record to path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True))
        f.write("\n")


def read_jsonl(path: str) -> list[Dict[str, Any]]:
    """Read JSONL records from path."""
    out: list[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def make_record(payload: Dict[str, Any], include_timestamp: bool = True) -> Dict[str, Any]:
    """
    Attach stable hash + git SHA (+ optional timestamp) to a payload.

    Note: hash is computed ONLY from payload (not from git_sha/timestamp).
    """
    rec: Dict[str, Any] = dict(payload)
    rec["hash"] = canonical_hash(payload)
    rec["git_sha"] = current_git_sha()
    if include_timestamp:
        rec["timestamp"] = time.time()
    return rec


def tensor_digest(t: torch.Tensor, *, max_inline_numel: int = 64) -> Dict[str, Any]:
    """
    Serialize a tensor with stable metadata + SHA256 digest.

    If numel <= max_inline_numel, inline values to aid debugging.
    """
    if not torch.is_tensor(t):
        raise TypeError(f"tensor_digest expects torch.Tensor, got {type(t)}")
    t_cpu = t.detach().cpu().contiguous()
    raw = t_cpu.numpy().tobytes()
    digest = hashlib.sha256(raw).hexdigest()
    out: Dict[str, Any] = {
        "shape": list(t_cpu.shape),
        "dtype": str(t_cpu.dtype),
        "device": str(t_cpu.device),
        "sha256": digest,
    }
    if t_cpu.numel() <= int(max_inline_numel):
        out["values"] = t_cpu.tolist()
    return out


def canonicalize_record_v2(
    record: Dict[str, Any],
    *,
    ignore_keys: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Return a canonicalized copy of a v2 record, dropping non-deterministic keys.
    """
    ignore = ignore_keys or {"timing", "created_at_utc", "timestamp"}
    stripped = _strip_keys(record, ignore)
    if not isinstance(stripped, dict):
        raise TypeError("canonicalize_record_v2 expects a dict record")
    solver = stripped.get("solver")
    if isinstance(solver, dict):
        solver.pop("time_sec", None)
        solver.pop("details", None)
    return stripped


def summarize_jsonl_v2(path: str) -> Dict[str, Any]:
    """
    Summarize v2 JSONL records for quick aggregation in tests or debugging.
    """
    records = read_jsonl(path)
    counts = {"PASS": 0, "FAILED": 0, "ERROR": 0}
    by_reason: Dict[str, int] = {}
    checks_total = 0
    violations_total = 0
    worst_gap_max = 0.0

    for rec in records:
        final = rec.get("final", {})
        verdict = final.get("final_verdict")
        if verdict in counts:
            counts[verdict] += 1
        reason = final.get("reason", "unknown")
        by_reason[reason] = by_reason.get(reason, 0) + 1

        l2 = rec.get("l2", {})
        checks_total += int(l2.get("checks_total", 0) or 0)
        violations_total += int(l2.get("violations_total", 0) or 0)
        gap = l2.get("worst_gap", 0.0)
        if gap is None:
            gap = 0.0
        worst_gap_max = max(worst_gap_max, float(gap))

    return {
        "records": len(records),
        "pass": counts["PASS"],
        "failed": counts["FAILED"],
        "error": counts["ERROR"],
        "by_reason": by_reason,
        "checks_total": int(checks_total),
        "violations_total": int(violations_total),
        "worst_gap_max": float(worst_gap_max),
    }


# --------------------------
# JSONL v2 schema
# --------------------------

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
