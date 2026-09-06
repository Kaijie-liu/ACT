# ===- act/pipeline/moe/experiment1_multiseed_training.py ------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""Run the frozen seed-1/2 verification-scale MoE training replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from act.util.path_config import get_torchvision_data_root


PROJECT_ROOT = Path("/data1/Kane/MOE/ACT").resolve()
WRITE_ROOT = Path("/data1/Kane/MOE").resolve()
EXPECTED_BRANCH = "feat/moe-route-verification"
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "act/pipeline/moe/configs/experiment1_multiseed_training_r1.json"
)


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise RuntimeError(f"path escapes {root}: {resolved}")
    return resolved


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _training_command(
    python: Path,
    training: dict[str, Any],
    seed: int,
    checkpoint: Path,
) -> list[str]:
    command = [
        str(python),
        "-m",
        "act.pipeline.moe",
        "--dataset",
        str(training["dataset"]),
        "--num-experts",
        str(training["num_experts"]),
        "--top-k",
        str(training["top_k"]),
        "--gate",
        str(training["gate"]),
        "--router-hidden",
        *[str(value) for value in training["router_hidden"]],
        "--expert-hidden",
        *[str(value) for value in training["expert_hidden"]],
        "--epochs",
        str(training["epochs"]),
        "--batch-size",
        str(training["batch_size"]),
        "--learning-rate",
        str(training["learning_rate"]),
        "--weight-decay",
        str(training["weight_decay"]),
        "--balance-loss",
        str(training["balance_loss"]),
        "--balance-coefficient",
        str(training["balance_coefficient"]),
        "--validation-fraction",
        str(training["validation_fraction"]),
        "--workers",
        str(training["workers"]),
        "--seed",
        str(seed),
        "--device",
        str(training["device"]),
        "--output",
        str(checkpoint),
    ]
    command.append("--download" if bool(training["download"]) else "--no-download")
    return command


def _stream(command: Iterable[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return int(process.wait())


def _validate_config(config: dict[str, Any]) -> None:
    seeds = [int(row["seed"]) for row in config["seeds"]]
    if seeds != [1, 2] or len(set(seeds)) != len(seeds):
        raise RuntimeError("registered seeds must be exactly [1, 2]")
    training = config["training"]
    expected = {
        "dataset": "CIFAR10",
        "num_experts": 8,
        "top_k": 2,
        "gate": "selected_softmax",
        "router_hidden": [128],
        "expert_hidden": [256, 128],
        "epochs": 50,
        "batch_size": 256,
        "balance_loss": "switch",
        "balance_coefficient": 0.1,
        "validation_fraction": 0.1,
    }
    for key, value in expected.items():
        if training.get(key) != value:
            raise RuntimeError(f"frozen training value changed: {key}")


def run(config_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    if Path.cwd().resolve() != PROJECT_ROOT:
        raise RuntimeError(f"run from {PROJECT_ROOT}")
    if _git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError("multiseed training requires the feature branch")
    if _git("status", "--porcelain"):
        raise RuntimeError("multiseed training requires a clean worktree")
    expected_python = Path(config["python"]).resolve()
    allowed_python = Path(
        "/data1/Kane/miniconda3/envs/act-py312/bin/python"
    ).resolve()
    if expected_python != allowed_python:
        raise RuntimeError(f"unregistered Python interpreter: {expected_python}")
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"requires {expected_python}, got {sys.executable}")
    configured_dataset_root = _inside(Path(config["dataset_root"]), WRITE_ROOT)
    actual_dataset_root = Path(get_torchvision_data_root()).resolve()
    if configured_dataset_root != actual_dataset_root:
        raise RuntimeError(
            f"dataset root mismatch: {actual_dataset_root} != {configured_dataset_root}"
        )
    raw_dataset = configured_dataset_root / "CIFAR10/raw/cifar-10-batches-py"
    if not raw_dataset.is_dir():
        raise RuntimeError(f"CIFAR-10 raw data missing: {raw_dataset}")

    output_dir = _inside(Path(config["output_dir"]), WRITE_ROOT)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    head = _git("rev-parse", "HEAD")
    summary: dict[str, Any] = {
        "experiment": config["experiment"],
        "status": "RUNNING",
        "git_branch": EXPECTED_BRANCH,
        "git_head": head,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "python": str(expected_python),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dataset_root": str(configured_dataset_root),
        "dataset_manifest": _file_manifest(raw_dataset),
        "selection_policy": config["frozen_interpretation"]["selection"],
        "started_unix": time.time(),
        "seeds": [],
    }
    _write_json(summary_path, summary)

    for registered in config["seeds"]:
        seed = int(registered["seed"])
        checkpoint = _inside(Path(registered["checkpoint"]), WRITE_ROOT)
        log_path = _inside(Path(registered["log"]), WRITE_ROOT)
        if checkpoint.exists() or log_path.exists():
            raise RuntimeError(
                f"registered seed {seed} would overwrite an existing artifact"
            )
        command = _training_command(expected_python, config["training"], seed, checkpoint)
        record: dict[str, Any] = {
            "seed": seed,
            "status": "RUNNING",
            "checkpoint": str(checkpoint),
            "log": str(log_path),
            "command": command,
            "started_unix": time.time(),
        }
        summary["seeds"].append(record)
        _write_json(summary_path, summary)
        returncode = _stream(command, log_path)
        record["returncode"] = returncode
        record["finished_unix"] = time.time()
        record["elapsed_seconds"] = record["finished_unix"] - record["started_unix"]
        if returncode != 0 or not checkpoint.is_file():
            record["status"] = "FAILED_PRESERVED"
            record["log_sha256"] = _sha256(log_path)
            summary["status"] = "FAILED_PRESERVED"
            _write_json(summary_path, summary)
            continue
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        record.update(
            status="COMPLETED",
            checkpoint_bytes=checkpoint.stat().st_size,
            checkpoint_sha256=_sha256(checkpoint),
            log_sha256=_sha256(log_path),
            factory_config=payload.get("factory_config"),
            training_config=payload.get("training_config"),
            validation_metrics=payload.get("validation_metrics"),
            test_metrics=payload.get("test_metrics"),
        )
        _write_json(summary_path, summary)

    completed = sum(row["status"] == "COMPLETED" for row in summary["seeds"])
    summary["finished_unix"] = time.time()
    summary["elapsed_seconds"] = summary["finished_unix"] - summary["started_unix"]
    summary["completed_seeds"] = completed
    summary["status"] = "COMPLETED" if completed == len(config["seeds"]) else "FAILED_PRESERVED"
    _write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main() -> None:
    result = run(build_parser().parse_args().config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
