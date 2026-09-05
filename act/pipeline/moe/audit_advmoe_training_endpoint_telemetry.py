"""Independently audit raw AdvMoE endpoint telemetry artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from act.pipeline.moe.audit_advmoe_training import floating_tensor_summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def json_nonfinite_paths(value: Any, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        paths.append(prefix)
    elif isinstance(value, dict):
        for key, child in value.items():
            paths.extend(json_nonfinite_paths(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(json_nonfinite_paths(child, f"{prefix}[{index}]"))
    return paths


def audit(config_path: Path, result_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    workspace = Path(config["workspace_boundary"])
    config_path = _inside(config_path, workspace)
    result_path = _inside(result_path, workspace)
    output_dir = _inside(Path(config["output_dir"]), workspace)

    issues: list[str] = []
    if result.get("status") != "COMPLETED_NOT_INDEPENDENTLY_AUDITED":
        issues.append("result has an unexpected status")
    if result.get("scope") != config.get("scope"):
        issues.append("result scope differs from config")
    if result.get("config", {}).get("sha256") != _sha256(config_path):
        issues.append("result config hash mismatch")
    if result_path.resolve() != (output_dir / "summary.json").resolve():
        issues.append("result is outside the configured output identity")

    nonfinite_json = json_nonfinite_paths(result)
    if nonfinite_json:
        issues.append(
            f"summary JSON contains {len(nonfinite_json)} non-finite values"
        )

    specs = {spec["label"]: spec for spec in config["model_states"]}
    rows = {row.get("identity", {}).get("label"): row for row in result.get("rows", [])}
    if set(rows) != set(specs):
        issues.append("result model-state labels differ from config")

    row_audits: list[dict[str, Any]] = []
    for label, spec in specs.items():
        row = rows.get(label)
        if row is None:
            continue
        artifact = _inside(Path(row["artifact"]["path"]), workspace)
        if not artifact.is_file():
            issues.append(f"{label}: raw artifact is missing")
            continue
        artifact_hash = _sha256(artifact)
        if artifact_hash != row["artifact"].get("sha256"):
            issues.append(f"{label}: raw artifact hash mismatch")
        with np.load(artifact, allow_pickle=False) as raw:
            array_finiteness = {
                name: bool(np.isfinite(raw[name]).all())
                for name in raw.files
                if np.issubdtype(raw[name].dtype, np.number)
            }
            eval_scores = raw["eval_scores"]
            train_scores = raw["train_scores"]
            eval_counts = np.bincount(
                eval_scores.argmax(axis=1), minlength=2
            ).astype(int).tolist()
            train_counts = np.bincount(
                train_scores.argmax(axis=1), minlength=2
            ).astype(int).tolist()
            attack_success = int(raw["attack_success"].astype(bool).sum())
            maximum_linf = float(raw["attack_linf"].max())
        bad_arrays = sorted(
            name for name, finite in array_finiteness.items() if not finite
        )
        if bad_arrays:
            issues.append(
                f"{label}: {len(bad_arrays)} raw numerical arrays contain non-finite values"
            )
        if eval_counts != row["EVAL_CURRENT_RUNNING_STATS"].get("route_counts"):
            issues.append(f"{label}: eval route counts do not recompute")
        if train_counts != row["TRAIN_ORDERED_TEST_BATCH_STATS"].get("route_counts"):
            issues.append(f"{label}: train route counts do not recompute")
        if attack_success != row["diagnostic_subset"]["strong_pgd"].get(
            "route_flip_count"
        ):
            issues.append(f"{label}: attack success count does not recompute")
        if not math.isclose(
            maximum_linf,
            float(row["diagnostic_subset"]["strong_pgd"].get("maximum_linf")),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            issues.append(f"{label}: attack L-infinity maximum does not recompute")

        checkpoint_finiteness = None
        if spec["kind"] == "CHECKPOINT":
            checkpoint = _inside(Path(spec["path"]), workspace)
            if _sha256(checkpoint) != spec["sha256"]:
                issues.append(f"{label}: checkpoint hash mismatch")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            checkpoint_finiteness = floating_tensor_summary(payload.get("router", {}))
            if checkpoint_finiteness["all_finite"] is not True:
                issues.append(f"{label}: checkpoint router contains non-finite tensors")

        row_audits.append(
            {
                "label": label,
                "artifact": {"path": str(artifact), "sha256": artifact_hash},
                "nonfinite_raw_arrays": bad_arrays,
                "eval_route_counts_recomputed": eval_counts,
                "train_route_counts_recomputed": train_counts,
                "attack_success_count_recomputed": attack_success,
                "maximum_linf_recomputed": maximum_linf,
                "checkpoint_router_finiteness": checkpoint_finiteness,
            }
        )

    return {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "result": {"path": str(result_path), "sha256": _sha256(result_path)},
        "summary_nonfinite_value_count": len(nonfinite_json),
        "summary_nonfinite_paths": nonfinite_json,
        "rows": row_audits,
        "issues": issues,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(arguments.config, arguments.result)
    _write(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
