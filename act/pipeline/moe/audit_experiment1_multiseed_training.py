# ===- act/pipeline/moe/audit_experiment1_multiseed_training.py ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""Independently replay the frozen multi-seed training artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split

from act.back_end.moe import load_output_moe_checkpoint
from act.pipeline.moe.experiment1_multiseed_training import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    WRITE_ROOT,
    _file_manifest,
    _inside,
    _sha256,
    _validate_config,
    _write_json,
)
from act.pipeline.moe.train import _load_dataset, evaluate


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def _check_metrics(
    issues: list[str],
    prefix: str,
    expected: dict[str, Any],
    observed: dict[str, Any],
) -> None:
    for key in ("accuracy", "load_entropy", "effective_experts", "max_expert_load", "min_expert_load"):
        if not _close(expected[key], observed[key]):
            issues.append(f"{prefix} {key} differs")
    for key in ("route_counts", "samples"):
        if expected[key] != observed[key]:
            issues.append(f"{prefix} {key} differs")
    if len(expected["route_frequencies"]) != len(observed["route_frequencies"]):
        issues.append(f"{prefix} route frequency width differs")
    elif any(
        not _close(left, right)
        for left, right in zip(expected["route_frequencies"], observed["route_frequencies"])
    ):
        issues.append(f"{prefix} route frequencies differ")


def audit(config_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    summary_path = _inside(Path(config["output_dir"]) / "summary.json", WRITE_ROOT)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if summary.get("status") != "COMPLETED":
        issues.append("training summary is not complete")
    if summary.get("config_sha256") != _sha256(config_path):
        issues.append("config hash mismatch")
    raw_dataset = (
        _inside(Path(config["dataset_root"]), WRITE_ROOT)
        / "CIFAR10/raw/cifar-10-batches-py"
    )
    if summary.get("dataset_manifest") != _file_manifest(raw_dataset):
        issues.append("dataset manifest mismatch")

    records = summary.get("seeds", [])
    expected_seeds = [int(row["seed"]) for row in config["seeds"]]
    if [row.get("seed") for row in records] != expected_seeds:
        issues.append("executed seed sequence differs from registration")
    full_train = _load_dataset("CIFAR10", True, download=False)
    test_data = _load_dataset("CIFAR10", False, download=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    replay: list[dict[str, Any]] = []
    registered_by_seed = {int(row["seed"]): row for row in config["seeds"]}
    for record in records:
        seed = int(record["seed"])
        registered = registered_by_seed.get(seed)
        if registered is None:
            issues.append(f"unregistered seed {seed}")
            continue
        if record.get("status") != "COMPLETED":
            issues.append(f"seed {seed} did not complete")
            continue
        checkpoint = _inside(Path(registered["checkpoint"]), WRITE_ROOT)
        log_path = _inside(Path(registered["log"]), WRITE_ROOT)
        if not checkpoint.is_file() or not log_path.is_file():
            issues.append(f"seed {seed} artifact missing")
            continue
        if record.get("checkpoint_sha256") != _sha256(checkpoint):
            issues.append(f"seed {seed} checkpoint hash mismatch")
        if record.get("log_sha256") != _sha256(log_path):
            issues.append(f"seed {seed} log hash mismatch")
        epoch_lines = sum(
            line.startswith("epoch=")
            for line in log_path.read_text(encoding="utf-8").splitlines()
        )
        if epoch_lines != int(config["training"]["epochs"]):
            issues.append(f"seed {seed} log has {epoch_lines} epoch rows")

        model, payload = load_output_moe_checkpoint(checkpoint, map_location=device)
        model.to(device).eval()
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            issues.append(f"seed {seed} checkpoint contains non-finite state")
        embedded = payload.get("training_config", {})
        expected_training = config["training"]
        for key in (
            "balance_loss",
            "balance_coefficient",
            "validation_fraction",
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "seed",
        ):
            expected = seed if key == "seed" else expected_training[key]
            if embedded.get(key) != expected:
                issues.append(f"seed {seed} embedded {key} differs")

        validation_size = max(
            1,
            int(round(len(full_train) * float(expected_training["validation_fraction"]))),
        )
        _, validation_data = random_split(
            full_train,
            (len(full_train) - validation_size, validation_size),
            generator=torch.Generator().manual_seed(seed),
        )
        loader_options = {
            "batch_size": int(expected_training["batch_size"]),
            "num_workers": 0,
            "pin_memory": device.type == "cuda",
        }
        observed_validation = evaluate(
            model, DataLoader(validation_data, shuffle=False, **loader_options), device
        )
        observed_test = evaluate(
            model, DataLoader(test_data, shuffle=False, **loader_options), device
        )
        _check_metrics(
            issues,
            f"seed {seed} validation",
            payload["validation_metrics"],
            observed_validation,
        )
        _check_metrics(
            issues,
            f"seed {seed} test",
            payload["test_metrics"],
            observed_test,
        )
        replay.append(
            {
                "seed": seed,
                "checkpoint_sha256": _sha256(checkpoint),
                "validation_metrics": observed_validation,
                "test_metrics": observed_test,
                "state_finite": True,
                "epoch_log_rows": epoch_lines,
            }
        )

    result = {
        "experiment": config["experiment"],
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "replay_device": str(device),
        "replayed_seeds": replay,
        "claim_boundary": (
            "Training identity, finiteness, and saved validation/test metrics are "
            "independently replayed. No formal verification endpoint is queried."
        ),
    }
    output = _inside(Path(config["audit_output"]), WRITE_ROOT)
    _write_json(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main() -> None:
    result = audit(build_parser().parse_args().config)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
