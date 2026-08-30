"""Wait for epoch 50 rehearsal and final B1 landing without touching training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

from act.pipeline.moe.baseline_icml2025_b1_landing import run_final, run_rehearsal
from act.pipeline.moe.baseline_icml2025_b1_smoke import PROJECT_ROOT, _inside, _sha256


def _atomic_state(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def watch(protocol_path: Path) -> None:
    protocol_path = _inside(protocol_path, PROJECT_ROOT)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    run_root = _inside(Path(protocol["run_root"]))
    progress_path = run_root / "progress.json"
    state_path = run_root / "landing/hook_state.json"
    failure_path = run_root / "landing/landing_hook_failure.json"
    rehearsal_path = run_root / "landing/rehearsal_epoch050/B1_LANDING_REHEARSAL.json"
    poll_seconds = float(protocol["poll_seconds"])
    try:
        while True:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            completed = [int(row["epoch"]) for row in progress.get("completed", [])]
            _atomic_state(
                state_path,
                {
                    "schema_version": 1,
                    "status": "WAITING",
                    "protocol_sha256": _sha256(protocol_path),
                    "completed_epochs": completed,
                    "rehearsal_complete": rehearsal_path.is_file(),
                    "progress_status": progress.get("status"),
                    "checked_unix_seconds": time.time(),
                },
            )
            if int(protocol["rehearsal_epoch"]) in completed and not rehearsal_path.exists():
                run_rehearsal(protocol_path)
            if progress.get("status") == "FAILED":
                raise RuntimeError("B1 supervisor reported FAILED")
            if progress.get("status") == "PASSED":
                run_final(protocol_path)
                _atomic_state(
                    state_path,
                    {
                        "schema_version": 1,
                        "status": "LANDED",
                        "protocol_sha256": _sha256(protocol_path),
                        "completed_epochs": completed,
                        "checked_unix_seconds": time.time(),
                    },
                )
                return
            time.sleep(poll_seconds)
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "protocol": str(protocol_path),
            "protocol_sha256": _sha256(protocol_path),
            "failed_unix_seconds": time.time(),
        }
        if not failure_path.exists():
            _atomic_state(failure_path, failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    watch(args.protocol)


if __name__ == "__main__":
    main()
