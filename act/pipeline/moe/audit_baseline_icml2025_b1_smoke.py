"""Independently audit a completed real-data ICML-2025 RT-ER B1 smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import torch


MOE_ROOT = Path("/data1/Kane/MOE")
OFFICIAL_REPO = MOE_ROOT / "baselines/Robust-MoE-Dual-Model"
OFFICIAL_COMMIT = "30ef94d77b5451595b82e739aa8938e1f4c4521f"
ARCHIVE_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
LABEL = "official-code, Blackwell-compatible deps + FFCV"


def _inside(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(MOE_ROOT):
        raise RuntimeError(f"audit path escapes {MOE_ROOT}: {path}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=OFFICIAL_REPO,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def audit(smoke_dir: Path, log_path: Path) -> dict[str, Any]:
    smoke_dir = _inside(smoke_dir)
    log_path = _inside(log_path)
    issues: list[str] = []
    summary_path = smoke_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError("smoke summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    def require(condition: bool, message: str) -> None:
        if not condition:
            issues.append(message)

    require(summary.get("status") == "PASSED", "runner status is not PASSED")
    require(summary.get("label") == LABEL, "reproduction label mismatch")
    require(summary.get("seed") == 0, "smoke seed is not zero")
    require(
        summary.get("official_source", {}).get("commit") == OFFICIAL_COMMIT,
        "official commit mismatch",
    )
    require(
        summary.get("official_source", {}).get("full_status") == "",
        "runner observed a dirty official repository",
    )
    require(summary.get("training_batch", {}).get("batch_size") == 4, "smoke batch changed")
    require(
        summary.get("training_batch", {}).get("expert_loss_denominator") == 512,
        "author expert-loss denominator changed",
    )
    require(
        summary.get("training_batch", {}).get("optimizer_step_executed") is True,
        "optimizer step was not recorded",
    )
    require(
        summary.get("training_batch", {}).get("gradients_finite") is True,
        "non-finite gradients were recorded",
    )
    require(
        summary.get("evaluation_batch", {}).get("outputs_finite") is True,
        "non-finite evaluation outputs were recorded",
    )
    require(
        summary.get("checkpoint", {}).get("restored_logit_max_abs_error") == 0.0,
        "checkpoint roundtrip was not exact",
    )
    require(
        summary.get("dataset", {}).get("source_archive_sha256") == ARCHIVE_SHA256,
        "CIFAR archive identity mismatch",
    )
    require(summary.get("environment", {}).get("torch") == "2.11.0+cu130", "Torch changed")
    require(summary.get("environment", {}).get("ffcv") == "1.0.2", "FFCV changed")

    artifacts = [
        (Path(summary["checkpoint"]["path"]), summary["checkpoint"]["sha256"]),
        (
            Path(summary["dataset"]["source_archive"]),
            summary["dataset"]["source_archive_sha256"],
        ),
        (
            smoke_dir / "beton/cifar_train_smoke.beton",
            summary["dataset"]["ffcv_train_beton_sha256"],
        ),
        (
            smoke_dir / "beton/cifar_test_smoke.beton",
            summary["dataset"]["ffcv_test_beton_sha256"],
        ),
    ]
    artifact_audit = []
    for path, expected in artifacts:
        path = _inside(path)
        actual = _sha256(path) if path.is_file() else None
        require(actual == expected, f"artifact hash mismatch: {path}")
        artifact_audit.append(
            {"path": str(path), "expected_sha256": expected, "actual_sha256": actual}
        )

    checkpoint_path = Path(summary["checkpoint"]["path"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("net", {})
    require(payload.get("epoch") == 0, "smoke checkpoint epoch changed")
    require(payload.get("acc") is None, "smoke checkpoint must not claim test accuracy")
    require(isinstance(payload.get("optimizer"), dict), "optimizer state is absent")
    require(isinstance(payload.get("scaler"), dict), "GradScaler state is absent")
    require("router.gate.weight" in state, "router weight is absent")
    require("router.gate.bias" in state, "router bias is absent")
    for expert in range(4):
        require(
            any(str(key).startswith(f"experts.{expert}.net.") for key in state),
            f"expert {expert} state is absent",
        )

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    require("Traceback (most recent call last)" not in log_text, "smoke log contains traceback")
    require(_git("rev-parse", "HEAD") == OFFICIAL_COMMIT, "official HEAD changed")
    official_status = _git("status", "--porcelain")
    require(not official_status, "official repository is not fully clean after smoke")
    return {
        "schema_version": 1,
        "status": "PASSED" if not issues else "FAILED",
        "issues": issues,
        "issue_count": len(issues),
        "smoke_summary": str(summary_path),
        "smoke_summary_sha256": _sha256(summary_path),
        "smoke_log": str(log_path),
        "smoke_log_sha256": _sha256(log_path),
        "artifact_audit": artifact_audit,
        "checkpoint_state_keys": len(state),
        "all_four_experts_present": all(
            any(str(key).startswith(f"experts.{expert}.net.") for key in state)
            for expert in range(4)
        ),
        "official_repository_status_after": official_status,
        "excluded_predecessors": [
            "seed0_paper130_smoke_r1: pre-experiment Python entrypoint failure",
            "seed0_paper130_smoke_r2: pre-objective dtype instrumentation failure",
        ],
        "scientific_scope": summary.get("scientific_scope"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = _inside(arguments.output)
    if output.exists():
        raise RuntimeError(f"audit refuses to overwrite {output}")
    result = audit(arguments.smoke_dir, arguments.log)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
