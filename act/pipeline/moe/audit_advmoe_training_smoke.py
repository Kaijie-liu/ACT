from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(config_path: Path, result_path: Path) -> dict[str, Any]:
    config = _load(config_path)
    result = _load(result_path)
    issues: list[str] = []
    if result.get("status") != "PASS":
        issues.append("smoke result is not PASS")
    if result.get("config", {}).get("sha256") != _sha256(config_path):
        issues.append("config hash mismatch")
    source = result.get("official_source", {})
    if source.get("commit") != config["official_source"]["commit"]:
        issues.append("official source commit mismatch")
    if source.get("tree") != config["official_source"]["tree"]:
        issues.append("official source tree mismatch")
    if source.get("clean") is not True:
        issues.append("official clone was not clean")
    checks = result.get("checks", {})
    expected_checks = set(config["preflight_gates"]["required_smoke_checks"])
    if not checks or not all(value is True for value in checks.values()):
        issues.append("one or more executable smoke checks failed")
    if result.get("batch", {}).get("size") != config["run"]["batch_size"]:
        issues.append("smoke did not use one full real training batch")
    device = result.get("device", {})
    if device.get("capability") != [12, 0]:
        issues.append("smoke did not execute on the registered sm_120 device")
    checkpoint = Path(result.get("resume", {}).get("checkpoint", ""))
    if not checkpoint.is_file():
        issues.append("smoke checkpoint is missing")
    elif result.get("resume", {}).get("checkpoint_sha256") != _sha256(checkpoint):
        issues.append("smoke checkpoint hash mismatch")
    if result.get("resume", {}).get("maximum_logit_error") != 0.0:
        issues.append("resumed model logits are not exact")
    if result.get("resume", {}).get("maximum_router_score_error") != 0.0:
        issues.append("resumed router scores are not exact")
    return {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "scope": "INDEPENDENT_ADV_MOE_TRAINING_SMOKE_AUDIT",
        "issue_count": len(issues),
        "issues": issues,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "result": {"path": str(result_path), "sha256": _sha256(result_path)},
        "required_check_labels": sorted(expected_checks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    report = audit(arguments.config, arguments.result)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
