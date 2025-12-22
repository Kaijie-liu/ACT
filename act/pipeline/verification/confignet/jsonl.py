from __future__ import annotations

import json
import time
import subprocess
from pathlib import Path
from typing import Iterable, Dict, Any

from act.pipeline.verification.confignet.configs import canonical_hash


def current_git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def write_jsonl_records(path: str | Path, records: Iterable[Dict[str, Any]], sort_keys: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=sort_keys) + "\n")


def make_record(
    run_seed: int,
    case_id: int,
    model_config: Dict[str, Any],
    spec_config: Dict[str, Any],
    overrides: Dict[str, Any] | None = None,
    include_timestamp: bool = True,
) -> Dict[str, Any]:
    payload = {
        "run_seed": run_seed,
        "case_id": case_id,
        "model_config": model_config,
        "spec_config": spec_config,
        "overrides": overrides or {},
    }
    record = {
        **payload,
        "hash": canonical_hash(payload),
        "git_sha": current_git_sha(),
    }
    if include_timestamp:
        record["timestamp"] = time.time()
    return record
