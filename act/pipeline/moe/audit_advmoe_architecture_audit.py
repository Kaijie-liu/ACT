"""Independently audit the pinned AdvMoE architecture-semantics result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from act.pipeline.moe.advmoe_architecture_audit import (
    PINNED_ANCHORS,
    REPO,
    REPOSITORY,
    _attach_router,
    _external_models,
)
from act.pipeline.moe.published_moe_router_gradient_audit import (
    BASELINE_ROOT,
    MOE_ROOT,
    PROJECT_ROOT,
    _git,
    _inside,
    _sha256,
)


EXPECTED_CLASSIFICATION = {
    "router_input_domain": "RAW_OR_OPTIONALLY_MODEL_NORMALIZED_IMAGE",
    "router_is_hidden_state_router": False,
    "router_network": "CIFAR_RESNET20_STYLE_CNN_3_TO_16_32_64_TO_E",
    "route_semantics": "ONE_SHARED_GLOBAL_PYTORCH_ARGMAX_TOP1",
    "route_controls_hidden_computation": True,
    "deep_route_conditioned_pathway": True,
    "independent_layerwise_route_decisions": False,
    "prefix_hz_before_router_applicable": False,
    "literal_tie_semantics": "FIRST_MAX_LOWEST_INDEX",
    "sound_overapproximation_tie_policy": "ALL_TIED_ROUTES_CONSERVATIVE",
    "router_update_mechanism": "EXPLICIT_SUPERVISED_AND_ROBUST_ROUTER_OBJECTIVE",
    "classification_ste_updates_router": False,
}


def _independent_dynamic_architecture() -> dict[str, Any]:
    torch.manual_seed(9102)
    with _external_models() as (resnet, router_module, moe_layer):
        model = resnet.resnet18_cifar_moe(num_classes=10, n_expert=2, ratio=0.5)
        router = router_module.build_router(num_experts=2)
        _attach_router(model, router)
        routed = [m for m in model.modules() if isinstance(m, moe_layer.MoEConv)]
        model.eval()
        with torch.no_grad():
            router_shape = list(router(torch.zeros(3, 3, 32, 32)).shape)
            model_shape = list(model(torch.zeros(3, 3, 32, 32)).shape)
            tie, _ = moe_layer.GetMask.apply(torch.zeros(1, 2))
        return {
            "router_output_shape": router_shape,
            "model_output_shape": model_shape,
            "routed_moe_convolutions": len(routed),
            "unique_router_object_ids": len({id(m.router) for m in routed}),
            "expert_widths": sorted(int(m.expert_width) for m in routed),
            "tie_index": int(tie.item()),
            "router_parameters": sum(p.numel() for p in router.parameters()),
        }


def validate(
    raw: dict[str, Any], *, replay: bool = True
) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    if raw.get("status") != "COMPLETED_NOT_INDEPENDENTLY_AUDITED":
        issues.append("raw result is not audit-ready")

    repo = _inside(REPO, BASELINE_ROOT)
    repository = raw.get("repository", {})
    if _git(repo, "rev-parse", "HEAD") != REPOSITORY["commit"]:
        issues.append("external repository commit changed")
    if _git(repo, "status", "--porcelain"):
        issues.append("external repository worktree is dirty")
    if repository.get("commit") != REPOSITORY["commit"]:
        issues.append("recorded repository commit changed")
    if repository.get("license") != "NOT_FOUND":
        issues.append("repository license classification changed")

    expected_anchor_keys = {item[1] for item in PINNED_ANCHORS}
    anchors = raw.get("anchors", [])
    if {item.get("key") for item in anchors} != expected_anchor_keys:
        issues.append("source anchor set changed")
    for anchor in anchors:
        path = _inside(repo / anchor.get("path", ""), repo)
        lines = path.read_text(encoding="utf-8").splitlines()
        line = anchor.get("line")
        if not isinstance(line, int) or line < 1 or line > len(lines):
            issues.append(f"invalid source anchor: {anchor.get('key')}")
            continue
        observed = hashlib.sha256(lines[line - 1].encode("utf-8")).hexdigest()
        if observed != anchor.get("matched_line_sha256"):
            issues.append(f"source anchor changed: {anchor.get('key')}")
        if _sha256(path) != anchor.get("file_sha256"):
            issues.append(f"source file changed: {anchor.get('path')}")

    for item in raw.get("files", []):
        path = _inside(repo / item.get("path", ""), repo)
        if _sha256(path) != item.get("sha256"):
            issues.append(f"source identity changed: {item.get('path')}")

    if raw.get("classification") != EXPECTED_CLASSIFICATION:
        issues.append("architecture classification changed")
    architecture = raw.get("dynamic_confirmation", {}).get("architecture", {})
    expected_architecture = {
        "router_input_shape": [2, 3, 32, 32],
        "router_output_shape": [2, 2],
        "model_output_shape": [2, 10],
        "residual_blocks": 8,
        "routed_moe_convolutions": 16,
        "unique_router_object_ids_across_moe_convolutions": 1,
        "literal_equal_score_selected_index": 0,
        "router_parameter_count": 269202,
        "attached_model_parameter_count": 5834652,
        "routed_expert_width_histogram": {
            "32": 4,
            "64": 4,
            "128": 4,
            "256": 4,
        },
    }
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            issues.append(f"dynamic architecture changed: {key}")

    training = raw.get("dynamic_confirmation", {}).get("training_schedule", {})
    if training.get("main_optimizer_router_parameter_overlap") != 0:
        issues.append("main optimizer unexpectedly includes router parameters")
    if training.get("router_tensors_changed_by_main_optimizer") != 0:
        issues.append("main optimizer unexpectedly changed router tensors")
    if training.get("router_tensors_with_nonzero_gradient_after_router_zero_grad") != 0:
        issues.append("router zero_grad did not clear classification STE gradients")
    if int(training.get("router_tensors_with_nonzero_ste_gradient_after_main_loss", 0)) <= 0:
        issues.append("classification STE gradient control is absent")
    if int(training.get("router_tensors_changed_by_explicit_router_optimizer", 0)) <= 0:
        issues.append("explicit router update control is absent")

    artifact = raw.get("training_artifact_findings", {})
    if artifact.get("separate_router_loader_name_occurrences_in_training_entrypoint") != 1:
        issues.append("separate router loader occurrence count changed")
    if artifact.get("separate_router_loader_passed_to_trainer") is not False:
        issues.append("separate router loader usage classification changed")
    if artifact.get("router_specific_cli_options_used_by_optimizer_construction") is not False:
        issues.append("router optimizer option-use classification changed")
    if artifact.get("main_and_router_updates_use_same_minibatch_in_released_trainer") is not True:
        issues.append("same-minibatch classification changed")

    policy = raw.get("artifact_policy", {})
    if policy.get("source_copied_into_act") is not False:
        issues.append("unlicensed-source policy changed")
    if policy.get("training_started") is not False:
        issues.append("audit unexpectedly claims training started")

    independent = _independent_dynamic_architecture() if replay else {}
    if replay and independent != {
        "router_output_shape": [3, 2],
        "model_output_shape": [3, 10],
        "routed_moe_convolutions": 16,
        "unique_router_object_ids": 1,
        "expert_widths": [32] * 4 + [64] * 4 + [128] * 4 + [256] * 4,
        "tie_index": 0,
        "router_parameters": 269202,
    }:
        issues.append("independent dynamic architecture replay changed")
    return issues, independent


def run(raw_path: Path, output_path: Path) -> dict[str, Any]:
    raw_path = _inside(raw_path, MOE_ROOT)
    output_path = _inside(output_path, PROJECT_ROOT)
    if output_path.exists():
        raise RuntimeError(f"audit output already exists: {output_path}")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    issues, independent = validate(raw)
    result = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "raw_result": {"path": str(raw_path), "sha256": _sha256(raw_path)},
        "independent_replay": independent,
        "conclusion": {
            "router_input": "CIFAR image tensor before the ResNet stem",
            "router_output": "one shared hard top-1 route",
            "routed_layers": 16,
            "verification_target": "full deep route-specialized ResNet pathway",
            "hidden_state_router": False,
        },
        "claim_boundary": (
            "This is a pinned architecture/training-semantics audit, not a trained "
            "checkpoint, accuracy, robustness, or certification result."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if issues:
        raise RuntimeError(f"AdvMoE architecture audit found {len(issues)} issue(s)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.raw, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
