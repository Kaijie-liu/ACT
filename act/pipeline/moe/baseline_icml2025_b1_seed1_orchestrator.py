"""Resource-gated unattended supervisor and landing orchestration for B1 seed 1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any

from act.pipeline.moe.baseline_icml2025_b1_landing import gpu_memory_bytes
from act.pipeline.moe.baseline_icml2025_b1_smoke import PROJECT_ROOT, _inside, _sha256


ACT_PYTHON = Path("/data1/Kane/miniconda3/envs/act-py312/bin/python")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def resource_snapshot(protocol: dict[str, Any]) -> dict[str, Any]:
    gate = protocol["launch_resource_gate"]
    free_gpu, total_gpu = gpu_memory_bytes(int(gate["device_index"]))
    disk = os.statvfs("/data1/Kane/MOE")
    free_disk = int(disk.f_bavail * disk.f_frsize)
    branch = str(protocol["branch"])
    remote = str(protocol["remote"])
    current_branch = _git("branch", "--show-current")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=all"))
    synchronized = _git("rev-parse", "HEAD") == _git("rev-parse", f"{remote}/{branch}")
    return {
        "free_gpu_memory_bytes": free_gpu,
        "total_gpu_memory_bytes": total_gpu,
        "minimum_free_gpu_memory_bytes": int(gate["minimum_free_memory_bytes"]),
        "free_disk_bytes": free_disk,
        "minimum_free_disk_bytes": int(gate["minimum_free_disk_bytes"]),
        "current_branch": current_branch,
        "required_branch": branch,
        "worktree_clean": not dirty,
        "local_remote_synchronized": synchronized,
        "ready": bool(
            free_gpu >= int(gate["minimum_free_memory_bytes"])
            and free_disk >= int(gate["minimum_free_disk_bytes"])
            and current_branch == branch
            and not dirty
            and synchronized
        ),
    }


def _validate_protocol(protocol_path: Path) -> tuple[dict[str, Any], Path]:
    protocol_path = _inside(protocol_path, PROJECT_ROOT)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if int(protocol.get("seed", -1)) != 1:
        raise RuntimeError("seed1 orchestrator requires a protocol frozen to seed 1")
    run_root = _inside(Path(protocol["run_root"]))
    if run_root.exists():
        raise RuntimeError(f"seed1 orchestrator refuses an existing run root: {run_root}")
    smoke = protocol["supervisor"]
    smoke_path = _inside(Path(smoke["smoke_summary"]))
    if _sha256(smoke_path) != smoke["smoke_summary_sha256"]:
        raise RuntimeError("frozen readiness-smoke identity changed")
    telemetry = _inside(Path(smoke["telemetry_config"]), PROJECT_ROOT)
    telemetry_value = json.loads(telemetry.read_text(encoding="utf-8"))
    if telemetry_value.get("training", {}).get("seeds") != [1]:
        raise RuntimeError("seed1 telemetry config identity is not seed 1")
    return protocol, run_root


def run(protocol_path: Path, state_path: Path) -> None:
    protocol, run_root = _validate_protocol(protocol_path)
    state_path = _inside(state_path)
    if state_path.exists():
        raise RuntimeError(f"orchestrator refuses to overwrite {state_path}")
    gate = protocol["launch_resource_gate"]
    wait_started = time.monotonic()
    while True:
        snapshot = resource_snapshot(protocol)
        elapsed = time.monotonic() - wait_started
        _atomic_json(
            state_path,
            {
                "schema_version": 1,
                "status": "READY_TO_LAUNCH" if snapshot["ready"] else "WAITING_FOR_RESOURCES",
                "protocol": str(protocol_path),
                "protocol_sha256": _sha256(protocol_path),
                "wait_elapsed_seconds": elapsed,
                "resource_snapshot": snapshot,
                "checked_unix_seconds": time.time(),
            },
        )
        if snapshot["ready"]:
            break
        if elapsed >= float(gate["maximum_wait_seconds"]):
            raise RuntimeError("seed1 launch resource wait budget exhausted")
        time.sleep(float(gate["retry_seconds"]))

    # Derive log identities from the immutable run-root name.  A failed attempt
    # is retained, so a repaired attempt must not collide with its sibling logs.
    supervisor_log = run_root.parent / f"{run_root.name}_supervisor.log"
    watcher_log = run_root.parent / f"{run_root.name}_landing_watch.log"
    for path in (supervisor_log, watcher_log):
        if path.exists():
            raise RuntimeError(f"orchestrator refuses to overwrite {path}")
    supervisor_command = [
        str(ACT_PYTHON),
        "-m",
        "act.pipeline.moe.baseline_icml2025_b1_supervisor",
        "--run-root",
        str(run_root),
        "--smoke-summary",
        str(protocol["supervisor"]["smoke_summary"]),
        "--telemetry-config",
        str(protocol["supervisor"]["telemetry_config"]),
        "--seed",
        "1",
        "--epochs",
        str(int(protocol["supervisor"]["epochs"])),
    ]
    watcher_command = [
        str(ACT_PYTHON),
        "-m",
        "act.pipeline.moe.baseline_icml2025_b1_landing_watch",
        "--protocol",
        str(protocol_path),
    ]
    supervisor: subprocess.Popen | None = None
    watcher: subprocess.Popen | None = None
    try:
        with supervisor_log.open("xb") as supervisor_handle:
            supervisor = subprocess.Popen(
                supervisor_command,
                cwd=PROJECT_ROOT,
                stdout=supervisor_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            progress_path = run_root / "progress.json"
            deadline = time.monotonic() + 300.0
            while not progress_path.is_file():
                if supervisor.poll() is not None:
                    raise RuntimeError(
                        f"seed1 supervisor exited {supervisor.returncode} before progress creation"
                    )
                if time.monotonic() >= deadline:
                    raise RuntimeError("seed1 supervisor did not create progress.json in 300 seconds")
                time.sleep(1.0)
            with watcher_log.open("xb") as watcher_handle:
                watcher = subprocess.Popen(
                    watcher_command,
                    cwd=PROJECT_ROOT,
                    stdout=watcher_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                _atomic_json(
                    state_path,
                    {
                        "schema_version": 1,
                        "status": "RUNNING",
                        "protocol": str(protocol_path),
                        "protocol_sha256": _sha256(protocol_path),
                        "supervisor_pid": supervisor.pid,
                        "watcher_pid": watcher.pid,
                        "supervisor_log": str(supervisor_log),
                        "watcher_log": str(watcher_log),
                        "launch_resource_snapshot": snapshot,
                        "launched_unix_seconds": time.time(),
                    },
                )
                supervisor_return = supervisor.wait()
                if supervisor_return != 0:
                    watcher.terminate()
                    watcher.wait(timeout=30)
                    raise RuntimeError(f"seed1 supervisor exited {supervisor_return}")
                watcher_return = watcher.wait()
                if watcher_return != 0:
                    raise RuntimeError(f"seed1 landing watcher exited {watcher_return}")
        _atomic_json(
            state_path,
            {
                "schema_version": 1,
                "status": "LANDED",
                "protocol": str(protocol_path),
                "protocol_sha256": _sha256(protocol_path),
                "run_root": str(run_root),
                "supervisor_return_code": 0,
                "watcher_return_code": 0,
                "completed_unix_seconds": time.time(),
            },
        )
    except BaseException as error:
        if supervisor is not None and supervisor.poll() is None:
            supervisor.terminate()
        if watcher is not None and watcher.poll() is None:
            watcher.terminate()
        _atomic_json(
            state_path,
            {
                "schema_version": 1,
                "status": "FAILED",
                "protocol": str(protocol_path),
                "protocol_sha256": _sha256(protocol_path),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "failed_unix_seconds": time.time(),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.protocol, arguments.state)


if __name__ == "__main__":
    main()
