"""Independently replay the full-test AdvMoE init route-share diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from act.pipeline.moe.advmoe_adapter import construct_official_init, state_dict_sha256
from act.pipeline.moe.advmoe_init_route_share import _forward_scores
from act.pipeline.moe.advmoe_router_bracket import load_cifar10_test_archive
from act.pipeline.moe.published_moe_router_gradient_audit import (
    MOE_ROOT,
    PROJECT_ROOT,
    _inside,
    _sha256,
)


def run(
    config_path: Path,
    result_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    result_dir = _inside(result_dir, MOE_ROOT)
    output_path = _inside(output_path, PROJECT_ROOT)
    if output_path.exists():
        raise RuntimeError(f"refuses to overwrite {output_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifact_path = result_dir / "full_test_scores_routes.npz"
    issues: list[str] = []
    if summary.get("status") != "COMPLETED_NOT_INDEPENDENTLY_AUDITED":
        issues.append("raw status changed")
    if summary.get("config", {}).get("sha256") != _sha256(config_path):
        issues.append("config hash changed")
    if summary.get("artifacts", {}).get("scores_routes", {}).get("sha256") != _sha256(
        artifact_path
    ):
        issues.append("score/route artifact hash changed")

    archive = _inside(Path(config["dataset_archive"]), MOE_ROOT)
    inputs, labels = load_cifar10_test_archive(archive)
    _model, router, _moe_type = construct_official_init(int(config["model_seed"]))
    del _model
    if state_dict_sha256(router) != summary.get("router_sha256"):
        issues.append("router identity changed")
    device = torch.device(str(config["device"]))
    router = router.to(device).eval()
    replay_scores = _forward_scores(
        router,
        inputs,
        device=device,
        batch_size=int(config["batch_size"]),
    )
    replay_routes = replay_scores.argmax(axis=1).astype(np.int64)
    replay_signed = replay_scores[:, 0].astype(np.float64) - replay_scores[:, 1].astype(
        np.float64
    )
    with np.load(artifact_path, allow_pickle=False) as arrays:
        stored_scores = arrays["scores"].copy()
        stored_routes = arrays["routes"].astype(np.int64)
        stored_labels = arrays["labels"].astype(np.int64)
        stored_signed = arrays["signed_difference"].astype(np.float64)
    if not np.array_equal(stored_labels, labels):
        issues.append("ordered CIFAR-10 labels changed")
    if not np.allclose(replay_scores, stored_scores, atol=2e-6, rtol=2e-6):
        issues.append("full-test scores failed replay")
    if not np.array_equal(replay_routes, stored_routes):
        issues.append("full-test routes failed replay")
    if not np.allclose(replay_signed, stored_signed, atol=2e-6, rtol=2e-6):
        issues.append("signed score differences failed replay")
    counts = np.bincount(replay_routes, minlength=2)
    if counts.tolist() != summary.get("route_counts"):
        issues.append("route counts changed")
    if counts.tolist() != [10000, 0]:
        issues.append("registered all-route-0 result changed")
    if not np.all(replay_signed > 0):
        issues.append("replayed signed margins are not uniformly positive")
    if any(row.get("status") != "NO_OPPOSITE_ROUTE_IN_OFFICIAL_TEST_SET" for row in summary.get("line_rows", [])):
        issues.append("line search did not stop on the missing opposite route")
    mean = float(replay_signed.mean())
    std = float(replay_signed.std(ddof=0))
    ratio = abs(mean) / std
    reported = summary.get("signed_score_difference", {})
    if not np.isclose(ratio, reported.get("absolute_mean_over_standard_deviation"), rtol=1e-12):
        issues.append("bias-dominance ratio changed")

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "scope": "INDEPENDENT_AUDIT_ADV_MOE_INIT_FULL_TEST_ROUTE_SHARE",
        "config_sha256": _sha256(config_path),
        "summary_sha256": _sha256(summary_path),
        "artifact_sha256": _sha256(artifact_path),
        "replayed_route_counts": counts.astype(int).tolist(),
        "replayed_signed_mean": mean,
        "replayed_signed_standard_deviation": std,
        "replayed_absolute_mean_over_standard_deviation": ratio,
        "conclusion": (
            "The official ordered CIFAR-10 test set routes 10000/10000 inputs "
            "to expert 0 at initialization. No opposite-route test image exists, "
            "so the preregistered cross-route line construction is inapplicable."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.result_dir, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
