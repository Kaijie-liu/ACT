"""Independently audit the pinned three-pipeline router-gradient result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from act.pipeline.moe.published_moe_router_gradient_audit import (
    BASELINE_ROOT,
    MOE_ROOT,
    PROJECT_ROOT,
    REPOSITORIES,
    _git,
    _inside,
    _sha256,
)


REQUIRED_ANCHORS = {
    "robust_moe_cnn": {
        "router_parameter_layer",
        "hard_argmax_dispatch",
        "straight_through_backward",
        "main_optimizer_created_before_router_attachment",
        "router_attached_after_main_optimizer",
        "separate_router_optimizer",
        "clean_router_scores",
        "adversarial_router_scores",
        "explicit_supervised_router_loss",
        "router_gradient_zeroed",
        "router_optimizer_step",
        "router_optimizer_checkpointed",
    },
    "vmoe": {
        "router_dense_parameter",
        "differentiable_gate_softmax",
        "importance_auxiliary_loss",
        "load_auxiliary_loss",
        "selected_gate_combine_weights",
        "router_invoked_in_moe_block",
        "auxiliary_losses_collected",
        "gradient_over_full_parameter_tree",
        "main_plus_auxiliary_objective",
        "full_state_parameters_supplied",
        "gradients_applied",
        "published_e8_k2_configuration",
        "positive_importance_weight",
        "positive_load_weight",
    },
}


def validate(raw: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if raw.get("status") != "COMPLETED_NOT_INDEPENDENTLY_AUDITED":
        issues.append("raw result is not audit-ready")
    pipelines = raw.get("pipelines", {})
    if set(pipelines) != set(REPOSITORIES):
        issues.append("pipeline identity set changed")
        return issues

    expected_classes = {
        "rt_er": "RELEASED_TRAINING_PATH_DOES_NOT_UPDATE_ROUTER",
        "robust_moe_cnn": "TRAINED_BY_EXPLICIT_ROUTER_OBJECTIVE",
        "vmoe": "TRAINED_END_TO_END_BY_COMBINE_WEIGHTS_AND_AUXILIARY_LOSSES",
    }
    for key, spec in REPOSITORIES.items():
        record = pipelines[key]
        repo = _inside(Path(spec["path"]), BASELINE_ROOT)
        if _git(repo, "rev-parse", "HEAD") != spec["commit"]:
            issues.append(f"{key}: repository commit changed")
        if _git(repo, "status", "--porcelain"):
            issues.append(f"{key}: repository worktree is dirty")
        if record.get("commit") != spec["commit"]:
            issues.append(f"{key}: recorded commit mismatch")
        if record.get("router_update_class") != expected_classes[key]:
            issues.append(f"{key}: router update classification changed")
        for item in record.get("files", []):
            path = _inside(repo / item["path"], repo)
            if _sha256(path) != item["sha256"]:
                issues.append(f"{key}: source hash changed for {item['path']}")

        anchors = record.get("anchors", [])
        if key in REQUIRED_ANCHORS:
            anchor_keys = {item.get("key") for item in anchors}
            if anchor_keys != REQUIRED_ANCHORS[key]:
                issues.append(f"{key}: evidence anchor set changed")
        for item in anchors:
            path = _inside(repo / item["path"], repo)
            lines = path.read_text(encoding="utf-8").splitlines()
            line = item.get("line")
            if not isinstance(line, int) or line < 1 or line > len(lines):
                issues.append(f"{key}: invalid line for {item.get('key')}")
                continue
            line_hash = hashlib.sha256(lines[line - 1].encode("utf-8")).hexdigest()
            if line_hash != item.get("matched_line_sha256"):
                issues.append(f"{key}: anchor line changed for {item.get('key')}")
            if _sha256(path) != item.get("file_sha256"):
                issues.append(f"{key}: anchor file changed for {item.get('key')}")

    rt_dynamic = pipelines["rt_er"].get("dynamic_evidence", {})
    if rt_dynamic.get("router_parameter_tensors_changed") != 0:
        issues.append("RT-ER dynamic router tensor count is not zero")
    if rt_dynamic.get("router_parameters_with_adam_state") != 0:
        issues.append("RT-ER router unexpectedly has optimizer state")
    if int(rt_dynamic.get("expert_parameter_tensors_changed", 0)) <= 0:
        issues.append("RT-ER expert-change control is absent")
    prior = pipelines["rt_er"].get("prior_audit", {})
    prior_path = _inside(Path(prior.get("path", "")), PROJECT_ROOT)
    if not prior_path.is_file() or _sha256(prior_path) != prior.get("sha256"):
        issues.append("RT-ER prior audit identity changed")

    if pipelines["robust_moe_cnn"].get("license") != "NOT_FOUND":
        issues.append("robust-moe-cnn license classification changed")
    robust_anchor = {
        item["key"]: item["line"]
        for item in pipelines["robust_moe_cnn"].get("anchors", [])
    }
    order_keys = [
        "main_optimizer_created_before_router_attachment",
        "router_attached_after_main_optimizer",
        "separate_router_optimizer",
    ]
    if all(key in robust_anchor for key in order_keys):
        values = [robust_anchor[key] for key in order_keys]
        if values != sorted(values) or len(set(values)) != 3:
            issues.append("robust-moe-cnn optimizer construction order changed")

    if pipelines["vmoe"].get("paper_config_has_router_freeze_rule") is not False:
        issues.append("V-MoE paper config freeze classification changed")
    vmoe_config = _inside(
        Path(REPOSITORIES["vmoe"]["path"])
        / "vmoe/configs/vmoe_paper/pretrain_imagenet21k.py",
        Path(REPOSITORIES["vmoe"]["path"]),
    ).read_text(encoding="utf-8")
    if "frozen_pattern" in vmoe_config or "trainable_pattern" in vmoe_config:
        issues.append("V-MoE paper config currently contains a freeze rule")

    finding = raw.get("cross_pipeline_finding", {})
    if finding.get("static_released_router_training_paths") != ["rt_er"]:
        issues.append("cross-pipeline static classification changed")
    if finding.get("learned_router_training_paths") != [
        "robust_moe_cnn",
        "vmoe",
    ]:
        issues.append("cross-pipeline learned classification changed")
    return issues


def run(raw_path: Path, output_path: Path) -> dict[str, Any]:
    raw_path = _inside(raw_path, MOE_ROOT)
    output_path = _inside(output_path, PROJECT_ROOT)
    if output_path.exists():
        raise RuntimeError(f"audit output already exists: {output_path}")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    issues = validate(raw)
    result = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "raw_result": {"path": str(raw_path), "sha256": _sha256(raw_path)},
        "recomputed": {
            "pipelines": 3,
            "source_level_learned_router_pipelines": 2,
            "released_static_router_training_pipelines": 1,
            "robust_moe_cnn_anchor_count": len(
                raw["pipelines"]["robust_moe_cnn"].get("anchors", [])
            ),
            "vmoe_anchor_count": len(
                raw["pipelines"]["vmoe"].get("anchors", [])
            ),
        },
        "interpretation": (
            "The audit establishes pinned source-level gradient paths. Only RT-ER "
            "also has dynamic tensor and optimizer-state evidence in this project. "
            "No accuracy, robustness, or checkpoint result is inferred for the two "
            "newly audited pipelines."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if issues:
        raise RuntimeError(f"router-gradient audit found {len(issues)} issue(s)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.raw, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
