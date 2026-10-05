#!/usr/bin/env python3
"""Seal Trial 9 artifacts only after the bound process has exited."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path


FORMAT_VERSION = "neural_hz_trial9_exit_seal_v1"
EXPECTED_RESULT = {
    "schema": "neural_hz_shadow_v1",
    "bench": "tinyimagenet_2024",
    "iid": 143,
    "arm": "sparse_phase_implicit_relu_census",
    "stop_after_layer": 36,
    "solver_timeout_s": 0.001,
    "memory_gb": 16.0,
    "sparse_entry_limit": 64_000_000,
    "representation": "sparse",
}
EXPECTED_PROVENANCE = {
    "branch": "redu-hz",
    "commit": "f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac",
    "candidate_sha256": (
        "16f35a7cd7cb6d8f592d345be736af3a0348571f3ff00bc8ea3d8235a9cd2d89"
    ),
}
ALLOWED_VERDICTS = {"CERT", "ADV", "UNKNOWN", "ERROR"}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    )


def _contained_path(
    path: Path, root: Path, *, must_exist: bool, regular_if_present: bool
) -> Path:
    resolved_root = root.resolve(strict=True)
    if path.is_symlink():
        raise ValueError(f"symlink paths are not allowed: {path}")
    if must_exist:
        resolved = path.resolve(strict=True)
    else:
        parent = path.parent.resolve(strict=True)
        resolved = parent / path.name
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"path escapes experiment root: {path}") from exc
    if regular_if_present and resolved.exists() and not resolved.is_file():
        raise ValueError(f"artifact is not a regular file: {path}")
    return resolved


def _read_start_ticks(proc_root: Path, pid: int) -> int | None:
    stat_path = proc_root / str(pid) / "stat"
    try:
        raw = stat_path.read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    right_parenthesis = raw.rfind(")")
    if right_parenthesis < 0:
        raise ValueError(f"malformed process stat: {stat_path}")
    fields_after_comm = raw[right_parenthesis + 1 :].split()
    # fields_after_comm[0] is field 3 (state); starttime is field 22.
    if len(fields_after_comm) <= 19:
        raise ValueError(f"truncated process stat: {stat_path}")
    return int(fields_after_comm[19])


def wait_for_original_exit(
    proc_root: Path, pid: int, start_ticks: int, poll_seconds: float
) -> dict[str, object]:
    while True:
        observed = _read_start_ticks(proc_root, pid)
        if observed is None:
            return {
                "observation": "ORIGINAL_PID_EXITED",
                "observed_start_ticks": None,
            }
        if observed != start_ticks:
            return {
                "observation": "PID_REUSED",
                "observed_start_ticks": observed,
            }
        time.sleep(poll_seconds)


def _read_stable_artifact(path: Path) -> tuple[bytes, dict[str, int]]:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(payload) != after.st_size:
        raise RuntimeError(f"artifact changed while sealing: {path}")
    return payload, {
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "size_bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
    }


def _artifact_record(path: Path) -> tuple[dict[str, object], bytes | None]:
    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": None}, None
    payload, stat_record = _read_stable_artifact(path)
    return {
        "path": str(path),
        "exists": True,
        "sha256": _sha256_bytes(payload),
        **stat_record,
    }, payload


def _validate_complete_result(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {
            "valid": False,
            "errors": ["top-level JSON is not an object"],
            "checks": {},
            "observed_verdict": None,
        }
    checks: dict[str, bool] = {
        f"result.{field}": value.get(field) == expected
        for field, expected in EXPECTED_RESULT.items()
    }
    provenance = value.get("provenance")
    for field, expected in EXPECTED_PROVENANCE.items():
        checks[f"provenance.{field}"] = (
            isinstance(provenance, dict) and provenance.get(field) == expected
        )
    checks["census_stop_reached"] = value.get("census_stop_reached") is True
    verdict = value.get("verdict")
    checks["verdict_allowed"] = verdict in ALLOWED_VERDICTS
    errors = sorted(name for name, passed in checks.items() if not passed)
    return {
        "valid": not errors,
        "errors": errors,
        "checks": checks,
        "observed_verdict": verdict if isinstance(verdict, str) else None,
    }


def build_seal_record(
    *,
    experiment_root: Path,
    result_path: Path,
    log_path: Path,
    pid: int,
    start_ticks: int,
    process_observation: dict[str, object],
) -> dict[str, object]:
    result_record, result_bytes = _artifact_record(result_path)
    log_record, _ = _artifact_record(log_path)
    validation: dict[str, object] | None = None
    observed_verdict = None
    if result_bytes is None:
        classification = "NO_RESULT_JSON"
    else:
        try:
            decoded = result_bytes.decode("utf-8")
            value = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            classification = "TRUNCATED_OR_CORRUPT"
            validation = {
                "valid": False,
                "errors": [f"{type(exc).__name__}: {exc}"],
                "checks": {},
                "observed_verdict": None,
            }
        else:
            validation = _validate_complete_result(value)
            observed_verdict = validation["observed_verdict"]
            classification = (
                "COMPLETE_JSON_VALIDATED"
                if validation["valid"]
                else "COMPLETE_JSON_METADATA_MISMATCH"
            )
    record: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_root": str(experiment_root),
        "process_binding": {
            "pid": pid,
            "start_ticks": start_ticks,
            **process_observation,
        },
        "classification": classification,
        "exit_status": "UNRECOVERABLE",
        "verdict": observed_verdict,
        "formal_gain": 0,
        "expected_result": EXPECTED_RESULT,
        "expected_provenance": EXPECTED_PROVENANCE,
        "result_artifact": result_record,
        "log_artifact": log_record,
        "result_validation": validation,
    }
    record["seal_payload_sha256"] = _sha256_json(record)
    return record


def exclusive_atomic_publish(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite seal sidecar: {path}")
    parent = path.parent.resolve(strict=True)
    temporary = parent / (
        f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    )
    descriptor = None
    directory_descriptor = None
    linked = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o664,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing seal")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(temporary, path)
        linked = True
        directory_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(directory_descriptor)
        temporary.unlink()
        os.fsync(directory_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        if temporary.exists():
            temporary.unlink()
        if not linked and path.exists():
            # A competing publisher owns the final path; never remove it.
            pass


def seal_after_exit(
    *,
    experiment_root: Path,
    result_path: Path,
    log_path: Path,
    sidecar_path: Path,
    proc_root: Path,
    pid: int,
    start_ticks: int,
    poll_seconds: float,
) -> dict[str, object]:
    root = experiment_root.resolve(strict=True)
    result = _contained_path(
        result_path, root, must_exist=False, regular_if_present=True
    )
    log = _contained_path(log_path, root, must_exist=False, regular_if_present=True)
    sidecar = _contained_path(
        sidecar_path, root, must_exist=False, regular_if_present=False
    )
    if sidecar in (result, log):
        raise ValueError("seal sidecar must differ from result and log")
    if sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError(f"refusing to overwrite seal sidecar: {sidecar}")
    process_observation = wait_for_original_exit(
        proc_root.resolve(strict=True), pid, start_ticks, poll_seconds
    )
    record = build_seal_record(
        experiment_root=root,
        result_path=result,
        log_path=log,
        pid=pid,
        start_ticks=start_ticks,
        process_observation=process_observation,
    )
    payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("ascii")
    exclusive_atomic_publish(sidecar, payload)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--start-ticks", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument(
        "--experiment-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.pid <= 0 or args.start_ticks <= 0:
        raise ValueError("pid and start ticks must be positive")
    if args.poll_seconds <= 0:
        raise ValueError("poll seconds must be positive")
    record = seal_after_exit(
        experiment_root=args.experiment_root,
        result_path=args.result,
        log_path=args.log,
        sidecar_path=args.sidecar,
        proc_root=args.proc_root,
        pid=args.pid,
        start_ticks=args.start_ticks,
        poll_seconds=args.poll_seconds,
    )
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
