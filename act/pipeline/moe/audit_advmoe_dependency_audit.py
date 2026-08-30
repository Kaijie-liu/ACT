"""Independently validate the read-only AdvMoE dependency audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from act.pipeline.moe.advmoe_architecture_audit import REPO, REPOSITORY
from act.pipeline.moe.published_moe_router_gradient_audit import (
    BASELINE_ROOT,
    MOE_ROOT,
    PROJECT_ROOT,
    _git,
    _inside,
    _sha256,
)


def validate(raw: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if raw.get("status") != "COMPLETED_NOT_INDEPENDENTLY_AUDITED":
        issues.append("raw dependency result is not audit-ready")
    repo = _inside(REPO, BASELINE_ROOT)
    if _git(repo, "rev-parse", "HEAD") != REPOSITORY["commit"]:
        issues.append("AdvMoE repository commit changed")
    if _git(repo, "status", "--porcelain"):
        issues.append("AdvMoE repository worktree is dirty")
    spec = raw.get("official_dependency_specification", {})
    requirements = repo / "requirements.txt"
    if spec.get("requirements_sha256") != _sha256(requirements):
        issues.append("requirements identity changed")
    if spec.get("versioned_entries") != ["scipy==1.6.0"]:
        issues.append("versioned dependency set changed")
    for key in ("torch_version_pinned", "cuda_version_pinned", "python_version_pinned"):
        if spec.get(key) is not False:
            issues.append(f"unexpected exact-environment pin: {key}")
    if spec.get("readme_anchor_present") is not True:
        issues.append("README dependency command anchor changed")
    if spec.get("readme_requested_file_exists") is not False:
        issues.append("README dependency filename mismatch changed")

    training = raw.get("training_entrypoint_probe", {})
    if training.get("exit_code") == 0 or training.get("reached_argument_parser") is not False:
        issues.append("training entrypoint unexpectedly became runnable")
    if training.get("first_missing_import") != "h5py":
        issues.append("first missing training import changed")
    blackwell = raw.get("blackwell_model_only_probe", {})
    if blackwell.get("status") != "PASS" or blackwell.get("capability") != [12, 0]:
        issues.append("Blackwell model-only probe did not pass on sm_120")
    if blackwell.get("finite") is not True or blackwell.get("shape") != [2, 2]:
        issues.append("Blackwell model-only output changed")

    classification = raw.get("classification", {})
    expected = {
        "exact_author_environment_defined": False,
        "model_only_blackwell_compatible_in_act_py312": True,
        "training_entrypoint_runnable_in_act_py312": False,
        "existing_crown_environment_is_training_environment": False,
        "next_environment_label": "OFFICIAL_CODE_BLACKWELL_COMPATIBLE_DEPENDENCY_REPRODUCTION",
        "installation_performed": False,
        "environment_created": False,
    }
    if classification != expected:
        issues.append("dependency classification changed")
    return issues


def run(raw_path: Path, output_path: Path) -> dict[str, Any]:
    raw_path = _inside(raw_path, MOE_ROOT)
    output_path = _inside(output_path, PROJECT_ROOT)
    if output_path.exists():
        raise RuntimeError(f"output already exists: {output_path}")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    issues = validate(raw)
    result = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "raw_result": {"path": str(raw_path), "sha256": _sha256(raw_path)},
        "conclusion": raw["classification"],
        "claim_boundary": raw["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if issues:
        raise RuntimeError(f"dependency audit found {len(issues)} issue(s)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.raw, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
