"""Independent audit for the B1 Tier-0 trajectory and Tier-1 configuration tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from act.pipeline.moe.experiment1 import WRITE_ROOT, _inside


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path = _inside(path, WRITE_ROOT)
    if path.exists():
        raise RuntimeError(f"refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(diagnostics_path: Path, hyperparameters_path: Path, output: Path) -> dict[str, Any]:
    diagnostics_path = _inside(diagnostics_path, WRITE_ROOT)
    hyperparameters_path = _inside(hyperparameters_path, WRITE_ROOT)
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    hyperparameters = json.loads(hyperparameters_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    validation = diagnostics.get("validation_checkpoints", [])
    training = diagnostics.get("training_epochs", [])
    if [row.get("epoch") for row in validation] != list(range(10, 131, 10)):
        issues.append("validation checkpoint epochs are not exactly 10..130 by tens")
    if [row.get("epoch") for row in training] != list(range(1, 131)):
        issues.append("training epoch records are not exactly 1..130")
    endpoint = diagnostics.get("trajectory_summary", {}).get("endpoint", {})
    if not math.isclose(float(endpoint.get("standard_accuracy_percent", -1)), 34.22):
        issues.append("endpoint SA is not 34.22")
    if not math.isclose(float(endpoint.get("robust_accuracy_percent", -1)), 32.70):
        issues.append("endpoint RA is not 32.70")
    if not math.isclose(
        float(endpoint.get("robust_to_standard_ratio", -1)), 32.70 / 34.22,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        issues.append("endpoint RA/SA ratio does not recompute")
    if not math.isclose(
        float(endpoint.get("paper_standard_gap_percentage_points", -1)), 77.81 - 34.22,
        abs_tol=1e-12,
    ):
        issues.append("paper SA gap does not recompute")
    if not diagnostics.get("augmentation_path", {}).get("configured"):
        issues.append("augmentation source path was not confirmed")
    if diagnostics.get("augmentation_path", {}).get("disable_flag_present"):
        issues.append("launcher unexpectedly passed --noaug")
    figure = Path(diagnostics.get("figure", ""))
    if not figure.is_file() or figure.stat().st_size == 0:
        issues.append("trajectory figure is missing or empty")
    source = Path(hyperparameters["official_source"]["path"])
    paper = Path(hyperparameters["paper"]["path"])
    if _sha256(source) != hyperparameters["official_source"]["sha256"]:
        issues.append("official source hash changed")
    if _sha256(paper) != hyperparameters["paper"]["sha256"]:
        issues.append("paper hash changed")
    items = {row["field"]: row for row in hyperparameters.get("items", [])}
    expected = {
        "epochs",
        "optimizer",
        "learning_rate_schedule",
        "weight_decay",
        "rt_er_beta",
        "augmentation",
        "attack",
        "mixed_precision",
        "router_parameter_scope",
    }
    if set(items) != expected:
        issues.append("hyperparameter audit fields differ from the frozen nine-field set")
    if items.get("optimizer", {}).get("classification") != "PAPER_UNDERSPECIFIED":
        issues.append("optimizer was overstated as a paper/code contradiction")
    if items.get("learning_rate_schedule", {}).get("classification") != "TEXT_CODE_SEMANTIC_AMBIGUITY":
        issues.append("learning-rate wording was not retained as ambiguity")
    for item in items.values():
        for anchor in item.get("anchors", []):
            line = int(anchor["line"])
            lines = Path(anchor["path"]).read_text(encoding="utf-8").splitlines()
            if line < 1 or line > len(lines) or lines[line - 1].strip() != anchor["text"]:
                issues.append(f"source anchor changed for {item['field']} at line {line}")
    result = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "scope": "INDEPENDENT_B1_TIER0_TIER1_DIAGNOSTIC_AUDIT",
        "issue_count": len(issues),
        "issues": issues,
        "diagnostics": {"path": str(diagnostics_path), "sha256": _sha256(diagnostics_path)},
        "hyperparameters": {
            "path": str(hyperparameters_path),
            "sha256": _sha256(hyperparameters_path),
        },
        "figure": {"path": str(figure), "sha256": _sha256(figure) if figure.is_file() else None},
        "recomputed": {
            "training_epoch_rows": len(training),
            "validation_checkpoint_rows": len(validation),
            "endpoint_ra_over_sa": 32.70 / 34.22,
            "paper_sa_gap_percentage_points": 77.81 - 34.22,
        },
    }
    _write(output, result)
    if issues:
        raise RuntimeError(f"B1 diagnostic audit failed: {issues}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--hyperparameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.diagnostics, arguments.hyperparameters, arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
