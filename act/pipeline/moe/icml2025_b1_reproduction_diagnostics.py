"""Reconstruct B1 seed-0 training diagnostics and paper/code configuration deltas."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np

from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside


OFFICIAL_REPO = Path("/data1/Kane/MOE/baselines/Robust-MoE-Dual-Model")
OFFICIAL_SOURCE = OFFICIAL_REPO / "cifar10_RT_ER.py"
PAPER = PROJECT_ROOT / "data/moe/papers/icml2025/zhang25cj.pdf"
LANDED = (
    PROJECT_ROOT
    / "act/pipeline/moe/results/baseline/icml2025_rt_er_b1_landed_seed0.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path = _inside(path, WRITE_ROOT)
    if path.exists():
        raise RuntimeError(f"refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _source_anchor(lines: list[str], needle: str) -> dict[str, Any]:
    matches = [index + 1 for index, line in enumerate(lines) if needle in line]
    if len(matches) != 1:
        raise RuntimeError(f"expected one source anchor for {needle!r}, found {matches}")
    line = matches[0]
    return {"path": str(OFFICIAL_SOURCE), "line": line, "text": lines[line - 1].strip()}


def _paper_text() -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(PAPER), "-"],
        check=True,
        text=True,
        capture_output=True,
    )
    return re.sub(r"\s+", " ", result.stdout)


def _wandb_training_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            payload = record.get("payload")
            if (
                record.get("event") == "log"
                and isinstance(payload, dict)
                and isinstance(payload.get("epoch"), int)
            ):
                rows.append(
                    {
                        "epoch": int(payload["epoch"]) + 1,
                        "train_loss": float(payload["train_loss"]),
                        "standard_accuracy_percent": float(payload["sa"]),
                        "robust_accuracy_percent": float(payload["ra"]),
                        "learning_rate": float(payload["lr"]),
                        "epoch_time_seconds": float(payload["epoch_time"]),
                    }
                )
    if [row["epoch"] for row in rows] != list(range(1, 131)):
        raise RuntimeError("wandb compatibility log does not contain exactly epochs 1..130")
    return rows


def _validation_rows(landing: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in landing["validated_checkpoint_schedule"]:
        metrics_path = _inside(Path(artifact["metrics"]), WRITE_ROOT)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        validation = metrics["validation"]
        training = metrics["training"]
        rows.append(
            {
                "epoch": int(artifact["epoch"]),
                "validation_accumulated_loss": float(validation["val_loss"]),
                "validation_standard_accuracy_percent": float(validation["val_sa"]),
                "validation_robust_accuracy_percent": float(validation["val_ra"]),
                "validation_ra_over_sa": float(validation["val_ra"])
                / float(validation["val_sa"]),
                "training_loss": float(training["train_loss"]),
                "training_standard_accuracy_percent": float(training["sa"]),
                "training_robust_accuracy_percent": float(training["ra"]),
                "learning_rate": float(training["lr"]),
                "epoch_time_seconds": float(training["epoch_time"]),
                "checkpoint_sha256": artifact["checkpoint_sha256"],
            }
        )
    return rows


def _trajectory_summary(validation: list[dict[str, Any]], training: list[dict[str, Any]]) -> dict[str, Any]:
    endpoint = validation[-1]
    post20 = [row for row in validation if row["epoch"] >= 20]
    epochs = np.asarray([row["epoch"] for row in post20], dtype=np.float64)
    standard = np.asarray(
        [row["validation_standard_accuracy_percent"] for row in post20], dtype=np.float64
    )
    slope = float(np.polyfit(epochs, standard, 1)[0])
    best = max(validation, key=lambda row: row["validation_standard_accuracy_percent"])
    return {
        "checkpoint_rows": len(validation),
        "training_epoch_rows": len(training),
        "endpoint": {
            "epoch": endpoint["epoch"],
            "standard_accuracy_percent": endpoint["validation_standard_accuracy_percent"],
            "robust_accuracy_percent": endpoint["validation_robust_accuracy_percent"],
            "robust_to_standard_ratio": endpoint["validation_ra_over_sa"],
            "paper_standard_gap_percentage_points": 77.81
            - endpoint["validation_standard_accuracy_percent"],
            "paper_robust_gap_percentage_points": 69.09
            - endpoint["validation_robust_accuracy_percent"],
        },
        "best_checkpoint_standard_accuracy": {
            "epoch": best["epoch"],
            "percent": best["validation_standard_accuracy_percent"],
        },
        "epoch20_through_130_standard_accuracy": {
            "minimum_percent": float(standard.min()),
            "maximum_percent": float(standard.max()),
            "linear_slope_percentage_points_per_epoch": slope,
        },
        "interpretation": (
            "The low, nearly flat checkpoint trajectory and endpoint RA/SA ratio are "
            "diagnostic of a model that learned little under this execution. They do "
            "not by themselves identify a causal hyperparameter or implementation fault."
        ),
    }


def _plot(path: Path, validation: list[dict[str, Any]], training: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in training]
    check_epochs = [row["epoch"] for row in validation]
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 6.2), constrained_layout=True)
    axes[0, 0].plot(
        epochs,
        [row["standard_accuracy_percent"] for row in training],
        label="train clean",
        linewidth=1.0,
    )
    axes[0, 0].plot(
        epochs,
        [row["robust_accuracy_percent"] for row in training],
        label="train PGD-10",
        linewidth=1.0,
    )
    axes[0, 0].scatter(
        check_epochs,
        [row["validation_standard_accuracy_percent"] for row in validation],
        label="test clean (every 10 epochs)",
        s=16,
    )
    axes[0, 0].scatter(
        check_epochs,
        [row["validation_robust_accuracy_percent"] for row in validation],
        label="test PGD-50 (every 10 epochs)",
        s=16,
    )
    axes[0, 0].axhline(77.81, linestyle="--", color="C0", alpha=0.5, label="paper SA")
    axes[0, 0].axhline(69.09, linestyle="--", color="C1", alpha=0.5, label="paper RA")
    axes[0, 0].set(ylabel="accuracy (%)", xlabel="epoch", title="Accuracy trajectory")
    axes[0, 0].legend(fontsize=7, ncol=2)

    axes[0, 1].plot(
        epochs,
        [row["train_loss"] for row in training],
        color="C2",
        linewidth=1.0,
    )
    axes[0, 1].set(ylabel="mean training loss", xlabel="epoch", title="Training loss")

    axes[1, 0].plot(
        check_epochs,
        [row["validation_accumulated_loss"] for row in validation],
        marker="o",
        markersize=3,
        color="C3",
    )
    axes[1, 0].set(
        ylabel="accumulated test loss",
        xlabel="epoch",
        title="Official PGD-50 test loss (batch-sum)",
    )

    axes[1, 1].plot(
        epochs,
        [row["learning_rate"] for row in training],
        color="C4",
        linewidth=1.0,
    )
    axes[1, 1].set(ylabel="learning rate", xlabel="epoch", title="CyclicLR trajectory")
    path = _inside(path, WRITE_ROOT)
    if path.exists():
        raise RuntimeError(f"refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg")
    plt.close(figure)


def _hyperparameter_audit(source_lines: list[str], paper_text: str, training: list[dict[str, Any]]) -> dict[str, Any]:
    required_paper_fragments = [
        "ResNet18-based MoE for 130 epochs",
        "130 epochs on CIFAR-10",
        "Cyclic Learning Rate strategy",
        "starting at 0.0001",
        "data augmentation",
        "50-step PGD",
        "10-step PGD",
    ]
    missing = [fragment for fragment in required_paper_fragments if fragment not in paper_text]
    if missing:
        raise RuntimeError(f"paper text anchors missing: {missing}")
    items = [
        {
            "field": "epochs",
            "paper": "130 epochs for CIFAR-10",
            "released_script": "default 200 epochs",
            "executed": "launcher explicitly passes 130 epochs",
            "classification": "RECONCILED_BY_DISCLOSED_PAPER_CONFIG_ARGUMENT",
            "anchors": [_source_anchor(source_lines, "--n_epochs")],
        },
        {
            "field": "optimizer",
            "paper": "optimizer family not specified in the located experiment setup",
            "released_script": "Adam",
            "executed": "Adam with lr=1e-4",
            "classification": "PAPER_UNDERSPECIFIED",
            "anchors": [_source_anchor(source_lines, "optimizer = optim.Adam")],
        },
        {
            "field": "learning_rate_schedule",
            "paper": "cyclic learning rate described as starting at 1e-4",
            "released_script": "CyclicLR(base_lr=5e-5, max_lr=1e-4, step_size_up=500)",
            "executed": "PyTorch initializes the optimizer group at 5e-5; all 130 logged values are retained",
            "classification": "TEXT_CODE_SEMANTIC_AMBIGUITY",
            "anchors": [_source_anchor(source_lines, "scheduler = torch.optim.lr_scheduler.CyclicLR")],
        },
        {
            "field": "weight_decay",
            "paper": "not specified in the located experiment setup",
            "released_script": "Adam default weight_decay=0",
            "executed": "0",
            "classification": "PAPER_UNDERSPECIFIED",
            "anchors": [_source_anchor(source_lines, "optimizer = optim.Adam")],
        },
        {
            "field": "rt_er_beta",
            "paper": "Table 9 includes beta=6 for the reported 77.81/69.09 result",
            "released_script": "default beta=6",
            "executed": "launcher passes beta=6",
            "classification": "CONSISTENT",
            "anchors": [_source_anchor(source_lines, "--beta")],
        },
        {
            "field": "augmentation",
            "paper": "states data augmentation and cites Rebuffi et al.; exact transforms not specified",
            "released_script": "horizontal flip, translate padding 2, cutout 8",
            "executed": "unchanged source path; launcher does not disable augmentation",
            "classification": "PAPER_UNDERSPECIFIED_SOURCE_PATH_CONFIRMED",
            "anchors": [
                _source_anchor(source_lines, "RandomHorizontalFlip(),"),
                _source_anchor(source_lines, "RandomTranslate(padding=2)"),
                _source_anchor(source_lines, "Cutout(8,"),
            ],
        },
        {
            "field": "attack",
            "paper": "epsilon=8/255, PGD-10 training, PGD-50 evaluation",
            "released_script": "epsilon=8/255, step=2/255, 10/50 steps",
            "executed": "same released path",
            "classification": "CONSISTENT_WITH_ADDITIONAL_CODE_ONLY_STEP_SIZE_DETAIL",
            "anchors": [
                _source_anchor(source_lines, "epsilon = 8 / 255"),
                _source_anchor(source_lines, "step_size = 2 / 255"),
                _source_anchor(source_lines, "num_step_train = 10"),
                _source_anchor(source_lines, "num_step_val = 50"),
            ],
        },
        {
            "field": "mixed_precision",
            "paper": "not specified in the located experiment setup",
            "released_script": "AMP enabled unless --noamp is passed",
            "executed": "AMP enabled",
            "classification": "PAPER_UNDERSPECIFIED",
            "anchors": [_source_anchor(source_lines, "use_amp = not args.noamp")],
        },
        {
            "field": "router_parameter_scope",
            "paper": "Equation (2) writes minimization over robust-MoE parameters Theta_R without a router exception",
            "released_script": "hard argmax prevents the released loss from updating router tensors",
            "executed": "router tensors remain bitwise fixed and have no Adam state",
            "classification": "MODEL_PARAMETER_SCOPE_GAP",
            "anchors": [_source_anchor(source_lines, "optimizer = optim.Adam")],
        },
    ]
    return {
        "schema_version": 1,
        "status": "COMPLETED",
        "scope": "PAPER_VS_RELEASED_CIFAR10_RT_ER_CONFIGURATION_AUDIT",
        "paper": {"path": str(PAPER), "sha256": _sha256(PAPER)},
        "official_source": {
            "path": str(OFFICIAL_SOURCE),
            "sha256": _sha256(OFFICIAL_SOURCE),
            "commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=OFFICIAL_REPO,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip(),
        },
        "items": items,
        "learning_rate_observation": {
            "first_logged_epoch": training[0],
            "minimum_logged": min(row["learning_rate"] for row in training),
            "maximum_logged": max(row["learning_rate"] for row in training),
            "important_scope": "checkpoint logs alone undersample the within-cycle schedule; all 130 epoch-end values are retained",
        },
        "conclusion": (
            "The located paper text does not specify optimizer family, weight decay, "
            "mixed precision, or exact augmentation transforms. These are artifact "
            "underspecification, not proven contradictions. The cyclic-LR wording and "
            "released initial optimizer rate differ semantically and remain labeled an "
            "ambiguity. The router parameter-scope gap is independently supported by "
            "frozen tensor and optimizer-state evidence."
        ),
    }


def run(output: Path, hyperparameters: Path, figure: Path) -> None:
    landing = json.loads(LANDED.read_text(encoding="utf-8"))
    supervisor_path = Path(landing["supervisor_summary"]["path"])
    supervisor = json.loads(supervisor_path.read_text(encoding="utf-8"))
    wandb_path = Path(supervisor["run_root"]) / "logs/wandb.jsonl"
    training = _wandb_training_rows(wandb_path)
    validation = _validation_rows(landing)
    source_lines = OFFICIAL_SOURCE.read_text(encoding="utf-8").splitlines()
    launcher = json.loads(Path(supervisor["launcher_manifest"]).read_text(encoding="utf-8"))
    source_augmentation = {
        "configured": True,
        "source_unchanged": bool(not supervisor["official_source_modified"]),
        "launcher_arguments": launcher["official_arguments"],
        "disable_flag_present": "--noaug" in launcher["official_arguments"],
        "anchors": [
            _source_anchor(source_lines, "RandomHorizontalFlip(),"),
            _source_anchor(source_lines, "RandomTranslate(padding=2)"),
            _source_anchor(source_lines, "Cutout(8,"),
        ],
        "runtime_evidence_scope": (
            "The unchanged pinned source constructed the training loader and the launcher "
            "did not request --noaug. No tensor-level augmentation trace was recorded, so "
            "this is source-to-execution identity evidence rather than sampled-transform replay."
        ),
    }
    diagnostics = {
        "schema_version": 1,
        "status": "COMPLETED",
        "scope": "ICML2025_RT_ER_B1_SEED0_TIER0_DIAGNOSTICS",
        "landing": {"path": str(LANDED), "sha256": _sha256(LANDED)},
        "supervisor": {"path": str(supervisor_path), "sha256": _sha256(supervisor_path)},
        "wandb_log": {"path": str(wandb_path), "sha256": _sha256(wandb_path)},
        "trajectory_summary": _trajectory_summary(validation, training),
        "validation_checkpoints": validation,
        "training_epochs": training,
        "augmentation_path": source_augmentation,
        "figure": str(figure),
    }
    _plot(figure, validation, training)
    _write_json(output, diagnostics)
    audit = _hyperparameter_audit(source_lines, _paper_text(), training)
    _write_json(hyperparameters, audit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hyperparameters", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.output, arguments.hyperparameters, arguments.figure)


if __name__ == "__main__":
    main()
