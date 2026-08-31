"""Wait for epoch 50 rehearsal and final B1 landing without touching training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time
import traceback

from act.pipeline.moe.baseline_icml2025_b1_landing import (
    RetryableGpuLandingError,
    run_final,
    run_rehearsal,
)
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


def _read_json_with_retries(
    path: Path, *, attempts: int, delay_seconds: float
) -> dict:
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise TypeError("JSON root is not an object")
            return value
        except (OSError, json.JSONDecodeError, TypeError) as error:
            errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            if attempt < attempts:
                time.sleep(delay_seconds)
    raise RuntimeError(f"failed to read {path} after {attempts} attempts: {errors}")


def _staleness_record(
    progress_path: Path,
    progress: dict,
    protocol: dict,
    *,
    now: float,
) -> dict:
    gate = protocol["staleness_detection"]
    durations: list[float] = []
    for row in progress.get("completed", []):
        metrics_path = Path(str(row.get("metrics", "")))
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            duration = float(metrics["training"]["epoch_time"])
            if duration > 0:
                durations.append(duration)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    median_epoch = statistics.median(durations) if durations else None
    threshold = max(
        float(gate["minimum_staleness_seconds"]),
        float(gate["epoch_duration_multiplier"])
        * (median_epoch if median_epoch is not None else float(gate["fallback_epoch_seconds"])),
    )
    progress_age = max(0.0, now - progress_path.stat().st_mtime)
    heartbeat_rows: list[dict] = []
    for raw_path in gate["heartbeat_paths"]:
        path = Path(str(raw_path))
        if path.is_file():
            heartbeat_rows.append(
                {"path": str(path), "age_seconds": max(0.0, now - path.stat().st_mtime)}
            )
    freshest_age = min(
        (float(row["age_seconds"]) for row in heartbeat_rows),
        default=progress_age,
    )
    return {
        "progress_mtime_age_seconds": progress_age,
        "freshest_heartbeat_age_seconds": freshest_age,
        "threshold_seconds": threshold,
        "median_completed_epoch_seconds": median_epoch,
        "heartbeat_rows": heartbeat_rows,
        "suspected": bool(progress_age > threshold and freshest_age > threshold),
    }


def _attempt_rehearsal(
    protocol_path: Path,
    failure_path: Path,
    *,
    attempt_count: int,
) -> bool:
    """Run a rehearsal without allowing its failure to stop final watching."""
    try:
        run_rehearsal(protocol_path)
    except Exception as error:
        _atomic_state(
            failure_path,
            {
                "schema_version": 1,
                "status": "REHEARSAL_FAILED",
                "nonfatal_to_final_landing": True,
                "attempt_count": attempt_count,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "protocol_sha256": _sha256(protocol_path),
                "last_failed_unix_seconds": time.time(),
            },
        )
        return False
    return True


def _record_hook_failure(
    legacy_path: Path,
    state_path: Path,
    failure: dict,
) -> Path:
    """Retain every watcher failure while preserving the legacy first record."""
    history_root = legacy_path.parent / "landing_hook_failures"
    history_root.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time_ns())
    history_path = history_root / f"failure_{stamp}.json"
    _atomic_state(history_path, failure)
    if not legacy_path.exists():
        _atomic_state(legacy_path, failure)
    _atomic_state(
        state_path,
        {
            "schema_version": 1,
            "status": "FAILED",
            "protocol_sha256": failure["protocol_sha256"],
            "error_type": failure["error_type"],
            "error": failure["error"],
            "failure_record": str(history_path),
            "checked_unix_seconds": time.time(),
        },
    )
    return history_path


def watch(protocol_path: Path) -> None:
    protocol_path = _inside(protocol_path, PROJECT_ROOT)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    run_root = _inside(Path(protocol["run_root"]))
    progress_path = run_root / "progress.json"
    state_path = run_root / "landing/hook_state.json"
    failure_path = run_root / "landing/landing_hook_failure.json"
    rehearsal_path = run_root / "landing/rehearsal_epoch050/B1_LANDING_REHEARSAL.json"
    rehearsal_failure_path = run_root / "landing/rehearsal_epoch050/REHEARSAL_FAILED.json"
    stall_path = run_root / "landing/STALLED_SUSPECTED.json"
    poll_seconds = float(protocol["poll_seconds"])
    read_policy = protocol["progress_read_retry"]
    rehearsal_attempts = 0
    next_rehearsal_attempt = 0.0
    gpu_wait_started: float | None = None
    try:
        while True:
            progress = _read_json_with_retries(
                progress_path,
                attempts=int(read_policy["attempts"]),
                delay_seconds=float(read_policy["delay_seconds"]),
            )
            completed = [int(row["epoch"]) for row in progress.get("completed", [])]
            now = time.time()
            staleness = _staleness_record(progress_path, progress, protocol, now=now)
            state_status = "STALLED_SUSPECTED" if staleness["suspected"] else "WAITING"
            if staleness["suspected"] and not stall_path.exists():
                _atomic_state(
                    stall_path,
                    {
                        "schema_version": 1,
                        "status": "STALLED_SUSPECTED",
                        "not_a_failure_verdict": True,
                        "protocol_sha256": _sha256(protocol_path),
                        "completed_epochs": completed,
                        "staleness": staleness,
                        "observed_unix_seconds": now,
                    },
                )
            _atomic_state(
                state_path,
                {
                    "schema_version": 1,
                    "status": state_status,
                    "protocol_sha256": _sha256(protocol_path),
                    "completed_epochs": completed,
                    "rehearsal_complete": rehearsal_path.is_file(),
                    "rehearsal_attempts": rehearsal_attempts,
                    "progress_status": progress.get("status"),
                    "staleness": staleness,
                    "checked_unix_seconds": now,
                },
            )
            if (
                int(protocol["rehearsal_epoch"]) in completed
                and not rehearsal_path.exists()
                and time.monotonic() >= next_rehearsal_attempt
            ):
                rehearsal_attempts += 1
                rehearsal_passed = _attempt_rehearsal(
                    protocol_path,
                    rehearsal_failure_path,
                    attempt_count=rehearsal_attempts,
                )
                if not rehearsal_passed:
                    next_rehearsal_attempt = time.monotonic() + float(
                        protocol["rehearsal_retry_seconds"]
                    )
            if progress.get("status") == "FAILED":
                raise RuntimeError("B1 supervisor reported FAILED")
            if progress.get("status") == "PASSED":
                try:
                    run_final(protocol_path)
                except RetryableGpuLandingError as error:
                    if gpu_wait_started is None:
                        gpu_wait_started = time.monotonic()
                    elapsed = time.monotonic() - gpu_wait_started
                    gate = protocol["gpu_resource_gate"]
                    if elapsed >= float(gate["maximum_wait_seconds"]):
                        raise RuntimeError(
                            f"GPU resource retry budget exhausted after {elapsed:.1f}s"
                        ) from error
                    _atomic_state(
                        state_path,
                        {
                            "schema_version": 1,
                            "status": "WAITING_FOR_GPU",
                            "protocol_sha256": _sha256(protocol_path),
                            "completed_epochs": completed,
                            "reason": str(error),
                            "wait_elapsed_seconds": elapsed,
                            "maximum_wait_seconds": float(gate["maximum_wait_seconds"]),
                            "next_retry_seconds": float(gate["retry_seconds"]),
                            "checked_unix_seconds": time.time(),
                        },
                    )
                    time.sleep(float(gate["retry_seconds"]))
                    continue
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
        _record_hook_failure(failure_path, state_path, failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    watch(args.protocol)


if __name__ == "__main__":
    main()
