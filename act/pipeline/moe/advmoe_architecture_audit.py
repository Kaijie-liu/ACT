"""Audit the pinned AdvMoE architecture without copying unlicensed source.

The external repository has no located license.  This module therefore stores
only file hashes, line hashes, semantic classifications, and independently
computed tensor/module facts.  It never embeds source text from that repository.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

from act.pipeline.moe.published_moe_router_gradient_audit import (
    BASELINE_ROOT,
    MOE_ROOT,
    REPOSITORIES,
    _file_records,
    _git,
    _hash_anchor,
    _inside,
)


REPOSITORY = REPOSITORIES["robust_moe_cnn"]
REPO = Path(REPOSITORY["path"])


PINNED_ANCHORS = [
    ("README.md", "official_cifar10_resnet18_e2_r05", 38, "78bdba9c678ca0cc3a49eb19d17a97df3d92efc6b7398f6e2508464161595740"),
    ("models/resnet_cifar_moe.py", "basic_block_first_moe_conv", 13, "6dc8a36f0601bdf8a4243ef93ecbdb234fde1a8e1d15de416cf7e7a13b5c7150"),
    ("models/resnet_cifar_moe.py", "basic_block_second_moe_conv", 17, "a69a85066ee56924c7be4834772c1d178d38935739f9f7afe1b039c174170d20"),
    ("models/resnet_cifar_moe.py", "fourth_residual_stage", 101, "3e52f699004bbd75beed5d9972d559a0244c62e250dc77b9eff0be30a7c9c94e"),
    ("models/resnet_cifar_moe.py", "router_called_on_model_input", 118, "3c41176c66589ad87b9a5a7d5c03406f4f306e5fae0c10a4d6b190a1b2342051"),
    ("models/resnet_cifar_moe.py", "stem_runs_after_router", 119, "9713a28f8247386a9b9badbc910bc89568ff47cd76581c012568c99f4d6895f3"),
    ("models/resnet_cifar_moe.py", "resnet18_has_eight_blocks", 131, "3ee9839e9a35313702ea8bb0cf84dbadfb22981624d8eb7b816d98aa6afc5e79"),
    ("models/layers/router.py", "router_accepts_three_channel_image", 43, "a515a357766ea53ae9cb34719cc95e08dca0a06f6e78031162bd0cf99d327796"),
    ("models/layers/router.py", "router_stage_one", 45, "6ce2164263a843de0a8eb62a5f7d0562174cc077527f4601fab94f04191014f2"),
    ("models/layers/router.py", "router_stage_two", 46, "28467c1f3cb969ca31d33a2f16255ea87bfac5081b9f764e0ddb1e8c3490fbd0"),
    ("models/layers/router.py", "router_stage_three", 47, "601882742e24352b1ab631d3ce9a2ca1b053e6cecf37e6ab485afbf97e765981"),
    ("models/layers/router.py", "router_score_head", 48, "7d70d99d612c97e8b2e628e22fd3c51ed2f3050ebedd32f6e0f1ed52cdb4f45d"),
    ("models/layers/router.py", "router_resnet20_depth", 72, "c65034db484f15fb33dc6d6f29c25585d061821a654856f1e2e6e0a31fb68bd5"),
    ("models/layers/moe_layer.py", "pytorch_argmax_top1", 9, "7817c30762b69773e86aa874525dc114ce514aa1a0a3b5e4b4ec04319c9cd826"),
    ("models/layers/moe_layer.py", "straight_through_backward", 16, "bc6d02226d4f26b949eefa1dee46ba8d5452eef5fa341ff7b17f44be083c1ab9"),
    ("models/layers/moe_layer.py", "one_score_is_shared_to_scored_modules", 34, "c5b3091bf1c0531d784e4a720b829108dbd7d1a13cded36f2903da5933cbb42c"),
    ("models/layers/moe_layer.py", "expert_channels_are_contiguous", 40, "adeaa2342ecd51e2a73655b66a6c551f37381eeec6fd3519a67f953a011c5cdf"),
    ("models/layers/moe_layer.py", "same_hard_mask_applied_per_layer", 61, "4e7cbf53a5fcf87ed55a4fb2a9685c48f2ef5b198de0824dbcbf7d8d482c32ce"),
    ("utils/general_utils.py", "router_attached_to_all_router_attributes", 20, "7412f26a6b1be3dc1695ad8a6aa346fd29fe65710a309645f475223a9d9629ae"),
    ("train_moe.py", "separate_router_loader_is_constructed", 189, "300259706e3ec4c30fcd7f60552d49262470f6a3936d632da4d27be00a8570fd"),
    ("train_moe.py", "main_optimizer_precedes_router_attachment", 191, "d515584648f82510454b25e1802b9a3333323b31c1840ed9256cbf1d5949dc9c"),
    ("train_moe.py", "router_attached_after_main_optimizer", 195, "2e66baa77c0090b7fc755ce7c6507974a62b2be273a8cb234a23edf6bc0b7095"),
    ("train_moe.py", "separate_router_optimizer", 196, "31dde889cdfccb04319a546abadd218e4e69ab3338696ba34d81f970544a941f"),
    ("train_moe.py", "trainer_receives_only_main_loader", 235, "5dbaa0157fd000248530d0595dc7f0873564d7c5883be5c00060fe4fd1f073d3"),
    ("train_moe.py", "classification_loss_backward", 100, "d9a37a96596b81c6beb8f245e05788ab21de52a4561100750baf9fe6f6326313"),
    ("train_moe.py", "main_optimizer_step", 101, "cdf624930347aab933f7ecb21fd067333b0ec7d2869498c65605e80a96fd9c7f"),
    ("train_moe.py", "explicit_router_objective", 143, "6b910c3039d3f67b2ead8bf9e4e2613a28eaed32aba2c3cb1134f47aa14e538a"),
    ("train_moe.py", "ste_gradient_cleared_before_router_loss", 155, "b69d0f0c995632e1c2835bc7a2d760030b70c42f2cfd84314de65cae7f0872cb"),
    ("train_moe.py", "router_optimizer_step", 157, "da696ae7c1e3c072f26849fdbb407d5fcd33c48e9e4db0a8d6d9630678654b69"),
    ("args.py", "router_optimizer_option_declared", 33, "d2ac8f24b6a8445bb888155263abbadc3c7136578ba8c44271a2813befd51ef0"),
    ("args.py", "router_lr_option_declared", 34, "596c59121f48b1aa8264b020ab313d78dad8127412fa5024ac2384ac823129f1"),
    ("args.py", "router_schedule_option_declared", 35, "83785d30ab4cd39ef967608c851f0bceee4e0dce230fbc7b4e82ce556ab1877e"),
    ("utils/schedules.py", "optimizer_helper_reads_generic_optimizer", 17, "0d4dbdaaf5b38d78448ee7e9bd1c047c5124e5c4773a6c7be7f0fe747a01f881"),
    ("utils/schedules.py", "optimizer_helper_reads_generic_lr", 20, "9cf2d9aecee04537885070e9073ec362fee5d9a3e2d84c642fa95598d478bd70"),
]


def _anchors() -> list[dict[str, Any]]:
    return [
        _hash_anchor(
            repo=REPO,
            relative=relative,
            key=key,
            line=line,
            matched_line_sha256=line_hash,
        )
        for relative, key, line, line_hash in PINNED_ANCHORS
    ]


@contextmanager
def _external_models() -> Iterator[tuple[Any, Any, Any]]:
    """Import the pinned external model locally, then remove its modules."""
    old_path = list(sys.path)
    preexisting = set(sys.modules)
    sys.path.insert(0, str(REPO))
    try:
        resnet = importlib.import_module("models.resnet_cifar_moe")
        router = importlib.import_module("models.layers.router")
        moe_layer = importlib.import_module("models.layers.moe_layer")
        yield resnet, router, moe_layer
    finally:
        sys.path[:] = old_path
        for name in list(sys.modules):
            if name not in preexisting and (name == "models" or name.startswith("models.")):
                del sys.modules[name]


def _attach_router(model: torch.nn.Module, router: torch.nn.Module) -> None:
    model.router = router
    for module in model.modules():
        if hasattr(module, "router"):
            module.router = router


def _changed(before: list[torch.Tensor], parameters: list[torch.nn.Parameter]) -> int:
    return sum(not torch.equal(old, new.detach()) for old, new in zip(before, parameters))


def _dynamic_confirmation() -> dict[str, Any]:
    torch.manual_seed(1234)
    with _external_models() as (resnet, router_module, moe_layer):
        model = resnet.resnet18_cifar_moe(num_classes=10, n_expert=2, ratio=0.5)
        router = router_module.build_router(num_experts=2)
        _attach_router(model, router)
        moe_modules = [m for m in model.modules() if isinstance(m, moe_layer.MoEConv)]
        input_shapes: list[list[int]] = []
        hook = router.register_forward_pre_hook(
            lambda _module, values: input_shapes.append(list(values[0].shape))
        )
        model.eval()
        with torch.no_grad():
            output = model(torch.rand(2, 3, 32, 32))
            tied_index, _ = moe_layer.GetMask.apply(torch.zeros(1, 2))
        hook.remove()
        widths = collections.Counter(int(m.expert_width) for m in moe_modules)

        architecture = {
            "official_configuration": {
                "dataset": "CIFAR10",
                "model": "resnet18_cifar_moe",
                "experts": 2,
                "ratio": 0.5,
            },
            "router_input_shape": input_shapes[0],
            "router_output_shape": [2, 2],
            "model_output_shape": list(output.shape),
            "router_parameter_count": sum(p.numel() for p in router.parameters()),
            "attached_model_parameter_count": sum(p.numel() for p in model.parameters()),
            "residual_blocks": 8,
            "routed_moe_convolutions": len(moe_modules),
            "routed_expert_width_histogram": {str(k): widths[k] for k in sorted(widths)},
            "unique_router_object_ids_across_moe_convolutions": len(
                {id(m.router) for m in moe_modules}
            ),
            "literal_equal_score_selected_index": int(tied_index.item()),
        }

        torch.manual_seed(4321)
        model = resnet.resnet18_cifar_moe(num_classes=10, n_expert=2, ratio=0.5)
        main_optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        router = router_module.build_router(num_experts=2)
        _attach_router(model, router)
        router_optimizer = torch.optim.SGD(router.parameters(), lr=0.1, momentum=0.9)
        main_ids = {id(p) for group in main_optimizer.param_groups for p in group["params"]}
        router_parameters = list(router.parameters())
        router_ids = {id(p) for p in router_parameters}

        model.train()
        x = torch.rand(2, 3, 32, 32)
        targets = torch.tensor([0, 1])
        main_optimizer.zero_grad()
        main_loss = torch.nn.functional.cross_entropy(model(x), targets)
        main_loss.backward()
        ste_nonzero = sum(
            p.grad is not None and bool(torch.count_nonzero(p.grad))
            for p in router_parameters
        )
        before_main_step = [p.detach().clone() for p in router_parameters]
        main_optimizer.step()
        changed_by_main = _changed(before_main_step, router_parameters)

        router_optimizer.zero_grad()
        remaining_after_clear = sum(
            p.grad is not None and bool(torch.count_nonzero(p.grad))
            for p in router_parameters
        )
        before_router_step = [p.detach().clone() for p in router_parameters]
        router_loss = torch.nn.functional.cross_entropy(router(x), targets % 2)
        router_loss.backward()
        router_optimizer.step()
        changed_by_router = _changed(before_router_step, router_parameters)

        training = {
            "main_optimizer_router_parameter_overlap": len(main_ids & router_ids),
            "router_tensors_with_nonzero_ste_gradient_after_main_loss": ste_nonzero,
            "router_tensors_changed_by_main_optimizer": changed_by_main,
            "router_tensors_with_nonzero_gradient_after_router_zero_grad": remaining_after_clear,
            "router_tensors_changed_by_explicit_router_optimizer": changed_by_router,
            "router_parameter_tensors": len(router_parameters),
        }
    return {"architecture": architecture, "training_schedule": training}


def collect() -> dict[str, Any]:
    repo = _inside(REPO, BASELINE_ROOT)
    head = _git(repo, "rev-parse", "HEAD")
    if head != REPOSITORY["commit"]:
        raise RuntimeError(f"repository commit drift: {head}")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("external repository worktree is dirty")
    root_names = {p.name.lower() for p in repo.iterdir() if p.is_file()}
    license_found = any(name.startswith(("license", "copying")) for name in root_names)
    if license_found:
        raise RuntimeError("repository license classification changed")

    train_text = (repo / "train_moe.py").read_text(encoding="utf-8")
    dynamic = _dynamic_confirmation()
    return {
        "schema_version": 1,
        "status": "COMPLETED_NOT_INDEPENDENTLY_AUDITED",
        "repository": {
            "name": REPOSITORY["name"],
            "url": REPOSITORY["url"],
            "commit": head,
            "branch": _git(repo, "branch", "--show-current"),
            "worktree_clean": True,
            "license": "NOT_FOUND",
        },
        "classification": {
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
        },
        "dynamic_confirmation": dynamic,
        "training_artifact_findings": {
            "separate_router_loader_name_occurrences_in_training_entrypoint": train_text.count(
                "train_router_loader"
            ),
            "separate_router_loader_passed_to_trainer": False,
            "router_specific_cli_options_declared": True,
            "router_specific_cli_options_used_by_optimizer_construction": False,
            "main_and_router_updates_use_same_minibatch_in_released_trainer": True,
        },
        "verification_implications": {
            "router_feasibility": "NONLINEAR_CNN_FROM_PIXELS",
            "route_branch_object": "FULL_RESNET_PATH_SPECIALIZED_BY_ONE_GLOBAL_ROUTE",
            "route_family_count_at_official_e2": 2,
            "not_claimed": [
                "hidden-state router input",
                "prefix-HZ reuse before router",
                "independent per-layer routing",
                "checkpoint robustness or accuracy",
            ],
        },
        "anchors": _anchors(),
        "files": _file_records(
            repo,
            [
                "README.md",
                "args.py",
                "models/resnet_cifar_moe.py",
                "models/layers/router.py",
                "models/layers/moe_layer.py",
                "train_moe.py",
                "utils/general_utils.py",
                "utils/schedules.py",
            ],
        ),
        "artifact_policy": {
            "source_copied_into_act": False,
            "training_started": False,
            "checkpoint_redistribution": "UNRESOLVED_NO_REPOSITORY_LICENSE_FOUND",
        },
    }


def write_result(output: Path) -> dict[str, Any]:
    output = _inside(output, MOE_ROOT)
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    result = collect()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_result(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
