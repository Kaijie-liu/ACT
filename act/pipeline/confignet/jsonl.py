#!/usr/bin/env python3
#===- act/pipeline/confignet/jsonl.py - JSONL Audit Utilities ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   JSONL audit utilities with stable serialization and hashing.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

import torch


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
