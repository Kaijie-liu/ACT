from __future__ import annotations

import json
import hashlib
import time
import subprocess
from pathlib import Path
from typing import Iterable, Dict, Any


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def current_git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def write_jsonl_records(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def make_record(
    run_seed: int,
    case_id: int,
    model_config: Dict[str, Any],
    spec_config: Dict[str, Any],
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = {
        "run_seed": run_seed,
        "case_id": case_id,
        "model_config": model_config,
        "spec_config": spec_config,
        "overrides": overrides or {},
    }
    return {
        **payload,
        "hash": canonical_hash(payload),
        "timestamp": time.time(),
        "git_sha": current_git_sha(),
    }
