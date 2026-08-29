"""Supervise the immutable-checkpoint B1 RT-ER seed-0 reproduction.

The author script remains byte-for-byte unchanged.  This process launches it in
an isolated run directory, freezes its process group after every scheduled
checkpoint, preserves an epoch-qualified copy, runs exact router telemetry, and
only then resumes training.  A telemetry failure stops the run rather than
silently leaving a gap in the preregistered trajectory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Any

import torch

from act.pipeline.moe.baseline_icml2025_b1_smoke import (
    MOE_ROOT,
    OFFICIAL_COMMIT,
    OFFICIAL_REPO,
    PROJECT_ROOT,
    _inside,
    _repo_value,
    _sha256,
    prepare_dataset,
)


BLACKWELL_PYTHON = MOE_ROOT / "envs/rt-er-blackwell/bin/python"
ACT_PYTHON = Path("/data1/Kane/miniconda3/envs/act-py312/bin/python")
JPEG_LIBRARY = MOE_ROOT / "envs/rt-er-blackwell/lib/libjpeg.so.8"
LAUNCHER = PROJECT_ROOT / "act/pipeline/moe/baseline_icml2025_official_launcher.py"
DEFAULT_TELEMETRY_CONFIG = (
    PROJECT_ROOT
    / "act/pipeline/moe/configs/icml2025_route_telemetry_blackwell_seed0.json"
)
CHECKPOINT_EPOCHS = tuple(range(10, 131, 10))
OFFICIAL_CHECKPOINT = Path("checkpoint/res18_moe-RT_ER-6.t7")
REPRODUCTION_LABEL = "official-code, Blackwell-compatible deps + FFCV"


def epoch_checkpoint_name(epoch: int) -> str:
    if int(epoch) not in CHECKPOINT_EPOCHS:
        raise ValueError(f"epoch {epoch} is outside the frozen checkpoint schedule")
    return f"epoch_{int(epoch):03d}.t7"


def epoch_directory_name(epoch: int) -> str:
    if int(epoch) not in CHECKPOINT_EPOCHS:
        raise ValueError(f"epoch {epoch} is outside the frozen checkpoint schedule")
    return f"epoch_{int(epoch):03d}"


def official_launcher_command(
    python: Path,
    *,
    seed: int,
    epochs: int,
    wandb_log: Path,
    launcher_manifest: Path,
) -> list[str]:
    return [
        str(python),
        "-u",
        str(LAUNCHER),
        "--seed",
        str(int(seed)),
        "--epochs",
        str(int(epochs)),
        "--wandb-log",
        str(wandb_log),
        "--launcher-manifest",
        str(launcher_manifest),
    ]


def telemetry_command(
    python: Path,
    *,
    config: Path,
    checkpoint: Path,
    output_dir: Path,
    metrics: Path,
    seed: int,
    epoch: int,
    device: str,
) -> list[str]:
    return [
        str(python),
        "-m",
        "act.pipeline.moe.icml2025_route_telemetry",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(int(seed)),
        "--epoch",
        str(int(epoch)),
        "--device",
        device,
        "--metrics",
        str(metrics),
    ]


def _atomic_json(path: Path, value: Any, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    mode = "w" if overwrite else "x"
    with temporary.open(mode, encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if not overwrite and path.exists():
        temporary.unlink()
        raise FileExistsError(path)
    temporary.replace(path)


def _checkpoint_epoch(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        epoch = payload.get("epoch")
        if not isinstance(epoch, int):
            return None
        return int(epoch) + 1
    except (EOFError, OSError, RuntimeError, ValueError):
        return None


def _checkpoint_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _allowed_readonly_python(path: Path) -> Path:
    """Allow the two disclosed environment roots without widening write scope."""

    resolved = path.resolve()
    allowed_roots = (
        (MOE_ROOT / "envs").resolve(),
        Path("/data1/Kane/miniconda3").resolve(),
    )
    if not any(resolved.is_relative_to(root) for root in allowed_roots):
        raise RuntimeError(f"Python executable is outside the allowed environment roots: {path}")
    return resolved


def _read_wandb_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def metrics_for_epoch(records: list[dict[str, Any]], epoch: int) -> dict[str, Any] | None:
    """Recover the val log followed by the matching zero-based train log."""

    target = int(epoch) - 1
    latest_validation: dict[str, Any] | None = None
    for record in records:
        if record.get("event") != "log" or not isinstance(record.get("payload"), dict):
            continue
        payload = record["payload"]
        if {"val_loss", "val_sa", "val_ra"}.issubset(payload):
            latest_validation = dict(payload)
        if payload.get("epoch") == target:
            if latest_validation is None:
                return None
            return {
                "epoch": int(epoch),
                "official_zero_based_epoch": target,
                "validation": latest_validation,
                "training": dict(payload),
            }
    return None


def _process_environment(seed: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": str(int(seed)),
            "PYTHONDONTWRITEBYTECODE": "1",
            "LD_PRELOAD": str(JPEG_LIBRARY),
        }
    )
    return environment


def _stop_group(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGSTOP)


def _continue_group(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGCONT)


def _terminate_group(process: subprocess.Popen, timeout: float = 30.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGCONT)
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=timeout)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=timeout)


def _validate_smoke(path: Path) -> dict[str, Any]:
    path = _inside(path)
    smoke = json.loads(path.read_text(encoding="utf-8"))
    if smoke.get("status") != "PASSED":
        raise RuntimeError("B1 real-data smoke has not passed")
    if smoke.get("label") != REPRODUCTION_LABEL:
        raise RuntimeError("B1 smoke label differs from the frozen reproduction label")
    if smoke.get("official_source", {}).get("commit") != OFFICIAL_COMMIT:
        raise RuntimeError("B1 smoke used a different official commit")
    return smoke


def run(
    run_root: Path,
    smoke_summary: Path,
    telemetry_config: Path,
    *,
    seed: int,
    epochs: int,
    blackwell_python: Path,
    act_python: Path,
    poll_seconds: float,
) -> dict[str, Any]:
    run_root = _inside(run_root)
    smoke_summary = _inside(smoke_summary)
    telemetry_config = _inside(telemetry_config, PROJECT_ROOT)
    blackwell_python = _allowed_readonly_python(blackwell_python)
    act_python = _allowed_readonly_python(act_python)
    if run_root.exists():
        raise RuntimeError(f"B1 supervisor refuses to overwrite {run_root}")
    if int(seed) != 0 or int(epochs) != 130:
        raise RuntimeError("current B1 gate is frozen to seed 0 and 130 epochs")
    if _repo_value("rev-parse", "HEAD") != OFFICIAL_COMMIT:
        raise RuntimeError("official repository commit changed")
    if _repo_value("status", "--porcelain"):
        raise RuntimeError("official repository is not fully clean")
    for executable in (blackwell_python, act_python):
        if not executable.is_file():
            raise RuntimeError(f"required Python executable is missing: {executable}")
    if not JPEG_LIBRARY.is_file():
        raise RuntimeError("required Blackwell libjpeg compatibility library is missing")
    config = json.loads(telemetry_config.read_text(encoding="utf-8"))
    if config.get("label") != REPRODUCTION_LABEL:
        raise RuntimeError("telemetry config does not carry the Blackwell reproduction label")
    if config.get("training", {}).get("seeds") != [0]:
        raise RuntimeError("telemetry config is not frozen to seed 0")
    smoke = _validate_smoke(smoke_summary)

    run_root.mkdir(parents=True)
    prepare_dataset(run_root)
    checkpoints_dir = run_root / "checkpoints"
    metrics_dir = run_root / "metrics"
    telemetry_dir = run_root / "telemetry"
    logs_dir = run_root / "logs"
    for directory in (checkpoints_dir, metrics_dir, telemetry_dir, logs_dir):
        directory.mkdir()
    official_checkpoint = run_root / OFFICIAL_CHECKPOINT
    wandb_log = logs_dir / "wandb.jsonl"
    launcher_manifest = run_root / "launcher_manifest.json"
    official_log = logs_dir / "official_training.log"
    progress_path = run_root / "progress.json"
    started_wall = time.time()
    started = time.monotonic()
    manifest = {
        "schema_version": 1,
        "status": "RUNNING",
        "label": REPRODUCTION_LABEL,
        "seed": int(seed),
        "epochs": int(epochs),
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "run_root": str(run_root),
        "official_repository": str(OFFICIAL_REPO),
        "official_commit": OFFICIAL_COMMIT,
        "official_source_modified": False,
        "scientific_patch": False,
        "smoke_summary": str(smoke_summary),
        "smoke_summary_sha256": _sha256(smoke_summary),
        "telemetry_config": str(telemetry_config),
        "telemetry_config_sha256": _sha256(telemetry_config),
        "started_unix_seconds": started_wall,
        "process_environment": {
            "PYTHONHASHSEED": str(seed),
            "PYTHONDONTWRITEBYTECODE": "1",
            "LD_PRELOAD": str(JPEG_LIBRARY),
        },
        "execution_policy": (
            "pause author process group after checkpoint metrics flush; preserve immutable "
            "checkpoint; run exact telemetry synchronously; resume only after success"
        ),
    }
    _atomic_json(run_root / "supervisor_manifest.json", manifest, overwrite=False)
    command = official_launcher_command(
        blackwell_python,
        seed=seed,
        epochs=epochs,
        wandb_log=wandb_log,
        launcher_manifest=launcher_manifest,
    )
    completed: list[dict[str, Any]] = []
    process: subprocess.Popen | None = None
    log_handle = official_log.open("xb")
    try:
        process = subprocess.Popen(
            command,
            cwd=run_root,
            env=_process_environment(seed),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        preserved_signature: tuple[int, int] | None = None
        for expected_epoch in CHECKPOINT_EPOCHS:
            candidate_signature: tuple[int, int] | None = None
            stable_polls = 0
            observed_epoch: int | None = None
            while True:
                return_code = process.poll()
                signature = _checkpoint_signature(official_checkpoint)
                if signature is None or signature == preserved_signature:
                    candidate_signature = signature
                    stable_polls = 0
                elif signature == candidate_signature:
                    stable_polls += 1
                else:
                    candidate_signature = signature
                    stable_polls = 1
                if stable_polls >= 2:
                    observed_epoch = _checkpoint_epoch(official_checkpoint)
                metrics = metrics_for_epoch(_read_wandb_records(wandb_log), expected_epoch)
                if observed_epoch is not None and observed_epoch > expected_epoch:
                    raise RuntimeError(
                        f"lost epoch {expected_epoch} before preservation; observed {observed_epoch}"
                    )
                if observed_epoch == expected_epoch and metrics is not None:
                    break
                if return_code is not None:
                    raise RuntimeError(
                        f"official training exited {return_code} before epoch {expected_epoch}"
                    )
                time.sleep(poll_seconds)

            _stop_group(process)
            try:
                if _checkpoint_epoch(official_checkpoint) != expected_epoch:
                    raise RuntimeError("checkpoint changed while the training group was frozen")
                preserved_signature = _checkpoint_signature(official_checkpoint)
                preserved = checkpoints_dir / epoch_checkpoint_name(expected_epoch)
                if preserved.exists():
                    raise RuntimeError(f"immutable checkpoint already exists: {preserved}")
                shutil.copy2(official_checkpoint, preserved)
                if _checkpoint_epoch(preserved) != expected_epoch:
                    raise RuntimeError("preserved checkpoint epoch mismatch")
                checkpoint_hash = _sha256(preserved)
                metrics_path = metrics_dir / f"epoch_{expected_epoch:03d}.json"
                _atomic_json(metrics_path, metrics, overwrite=False)
                telemetry_output = telemetry_dir / epoch_directory_name(expected_epoch)
                telemetry_log = logs_dir / f"telemetry_epoch_{expected_epoch:03d}.log"
                telemetry_invocation = telemetry_command(
                    act_python,
                    config=telemetry_config,
                    checkpoint=preserved,
                    output_dir=telemetry_output,
                    metrics=metrics_path,
                    seed=seed,
                    epoch=expected_epoch,
                    device="cuda",
                )
                with telemetry_log.open("xb") as telemetry_handle:
                    telemetry_run = subprocess.run(
                        telemetry_invocation,
                        cwd=PROJECT_ROOT,
                        env={
                            **os.environ,
                            "PYTHONDONTWRITEBYTECODE": "1",
                            "PYTHONHASHSEED": str(seed),
                        },
                        stdout=telemetry_handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if telemetry_run.returncode != 0:
                    raise RuntimeError(
                        f"epoch {expected_epoch} telemetry exited {telemetry_run.returncode}"
                    )
                telemetry_summary = telemetry_output / "summary.json"
                if not telemetry_summary.is_file():
                    raise RuntimeError("telemetry completed without summary.json")
                telemetry_value = json.loads(telemetry_summary.read_text(encoding="utf-8"))
                if (
                    telemetry_value.get("epoch") != expected_epoch
                    or telemetry_value.get("seed") != seed
                    or telemetry_value.get("checkpoint", {}).get("sha256")
                    != checkpoint_hash
                ):
                    raise RuntimeError("telemetry summary identity mismatch")
                completed.append(
                    {
                        "epoch": expected_epoch,
                        "checkpoint": str(preserved),
                        "checkpoint_sha256": checkpoint_hash,
                        "metrics": str(metrics_path),
                        "metrics_sha256": _sha256(metrics_path),
                        "telemetry": str(telemetry_output),
                        "telemetry_summary_sha256": _sha256(telemetry_summary),
                    }
                )
                _atomic_json(
                    progress_path,
                    {
                        "schema_version": 1,
                        "status": "RUNNING",
                        "completed": completed,
                        "next_epoch": (
                            expected_epoch + 10 if expected_epoch < CHECKPOINT_EPOCHS[-1] else None
                        ),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    overwrite=True,
                )
            finally:
                _continue_group(process)

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"official training exited with status {return_code}")
        if [item["epoch"] for item in completed] != list(CHECKPOINT_EPOCHS):
            raise RuntimeError("B1 checkpoint/telemetry schedule is incomplete")
        if _repo_value("status", "--porcelain"):
            raise RuntimeError("official repository became dirty during B1")
        summary = {
            **manifest,
            "status": "PASSED",
            "official_return_code": return_code,
            "completed": completed,
            "runtime_seconds": time.monotonic() - started,
            "official_log": str(official_log),
            "official_log_sha256": _sha256(official_log),
            "launcher_manifest": str(launcher_manifest),
            "launcher_manifest_sha256": _sha256(launcher_manifest),
            "official_repository_clean_after": True,
        }
        _atomic_json(run_root / "summary.json", summary, overwrite=False)
        _atomic_json(
            progress_path,
            {"schema_version": 1, "status": "PASSED", "completed": completed},
            overwrite=True,
        )
        return summary
    except BaseException as error:
        if process is not None:
            _terminate_group(process)
        failure = {
            **manifest,
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "completed": completed,
            "runtime_seconds": time.monotonic() - started,
            "official_return_code": process.poll() if process is not None else None,
        }
        _atomic_json(run_root / "failure.json", failure, overwrite=False)
        raise
    finally:
        log_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--smoke-summary", type=Path, required=True)
    parser.add_argument("--telemetry-config", type=Path, default=DEFAULT_TELEMETRY_CONFIG)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=130)
    parser.add_argument("--blackwell-python", type=Path, default=BLACKWELL_PYTHON)
    parser.add_argument("--act-python", type=Path, default=ACT_PYTHON)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    arguments = parser.parse_args()
    result = run(
        arguments.run_root,
        arguments.smoke_summary,
        arguments.telemetry_config,
        seed=arguments.seed,
        epochs=arguments.epochs,
        blackwell_python=arguments.blackwell_python,
        act_python=arguments.act_python,
        poll_seconds=arguments.poll_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
