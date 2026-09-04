"""Independently audit the frozen RT-ER B2 conformance artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256


def _maximum_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(prepare_path: Path, crown_path: Path, output_path: Path) -> dict[str, Any]:
    prepare_path = _inside(prepare_path, WRITE_ROOT)
    crown_path = _inside(crown_path, WRITE_ROOT)
    output_path = _inside(output_path, WRITE_ROOT)
    if output_path.exists():
        raise RuntimeError(f"B2 audit refuses to overwrite {output_path}")
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    crown = json.loads(crown_path.read_text(encoding="utf-8"))
    config_path = _inside(Path(prepare["config"]["path"]), PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if prepare.get("status") != "PREPARED_CROWN_NOT_RUN":
        issues.append("prepare status changed")
    if crown.get("status") != "PASS":
        issues.append("CROWN worker did not pass")
    if prepare["config"]["sha256"] != _sha256(config_path):
        issues.append("config hash changed")
    if crown.get("prepare", {}).get("sha256") != _sha256(prepare_path):
        issues.append("CROWN prepare hash changed")
    reference_path = _inside(Path(prepare["artifact"]["path"]), WRITE_ROOT)
    crown_arrays_path = _inside(Path(crown["arrays"]["path"]), WRITE_ROOT)
    if prepare["artifact"]["sha256"] != _sha256(reference_path):
        issues.append("reference artifact hash changed")
    if crown["arrays"]["sha256"] != _sha256(crown_arrays_path):
        issues.append("CROWN array artifact hash changed")
    metrics: dict[str, Any] = {}
    with np.load(reference_path, allow_pickle=False) as reference, np.load(
        crown_arrays_path, allow_pickle=False
    ) as observed:
        expected_indices = np.arange(
            int(config["selection"]["start"]),
            int(config["selection"]["start"])
            + int(config["selection"]["samples"]),
            dtype=np.int64,
        )
        if not np.array_equal(reference["dataset_indices"], expected_indices):
            issues.append("selection is no longer the first 1000 ordered inputs")
        for family in config["selection"]["probe_families"]:
            direct = observed[f"{family}__direct_expert_logits"]
            bounded = observed[f"{family}__bounded_expert_logits"]
            expected_expert = reference[f"{family}__expert_logits"]
            expected_selected = reference[f"{family}__selected_logits"]
            expected_predictions = reference[f"{family}__predictions"]
            routes = reference[f"{family}__routes"].astype(np.int64)
            selected = direct[np.arange(len(routes)), routes]
            predictions = selected.argmax(axis=1)
            direct_error = _maximum_abs(direct, expected_expert)
            bounded_error = _maximum_abs(bounded, direct)
            selected_error = _maximum_abs(selected, expected_selected)
            prediction_agreement = float(np.mean(predictions == expected_predictions))
            metrics[family] = {
                "samples": int(len(routes)),
                "direct_cross_runtime_maximum_abs_error": direct_error,
                "auto_lirpa_concrete_maximum_abs_error": bounded_error,
                "selected_cross_runtime_maximum_abs_error": selected_error,
                "prediction_agreement": prediction_agreement,
            }
            if direct_error > float(
                config["tolerances"]["cross_runtime_direct_logit_atol"]
            ):
                issues.append(f"{family} direct cross-runtime logits exceed tolerance")
            if bounded_error > float(config["tolerances"]["auto_lirpa_logit_atol"]):
                issues.append(f"{family} auto_LiRPA logits exceed tolerance")
            if selected_error > float(
                config["tolerances"]["cross_runtime_direct_logit_atol"]
            ):
                issues.append(f"{family} selected logits exceed tolerance")
            if prediction_agreement != float(
                config["tolerances"]["required_prediction_agreement"]
            ):
                issues.append(f"{family} predictions disagree")
    if prepare.get("model_identity", {}).get("batchnorm_all_eval") is not True:
        issues.append("BatchNorm eval identity missing")
    result = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "scope": "INDEPENDENT_B2_ARTIFACT_AND_LOGIT_RECOMPUTATION",
        "prepare": {"path": str(prepare_path), "sha256": _sha256(prepare_path)},
        "crown": {"path": str(crown_path), "sha256": _sha256(crown_path)},
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "metrics": metrics,
        "claim_scope": config["claim_scope"],
    }
    _write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path, required=True)
    parser.add_argument("--crown", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.prepare, args.crown, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
