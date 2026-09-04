from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping

import torch


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside_workspace(path: Path, workspace: Path) -> bool:
    try:
        path.resolve().relative_to(workspace.resolve())
    except ValueError:
        return False
    return True


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _changed_parameter_count(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> int:
    return sum(not torch.equal(before[name], after[name].detach().cpu()) for name in before)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def build_official_args(config: Mapping[str, Any]) -> SimpleNamespace:
    run = config["run"]
    return SimpleNamespace(
        configs=None,
        exp_identifier="official_seed0_r1",
        seed=int(run["seed"]),
        arch=run["arch"],
        resume="",
        dataset=run["dataset"],
        data_dir=config["dataset"]["run_data_root"],
        num_workers=int(run["num_workers_argument"]),
        batch_size=int(run["batch_size"]),
        test_batch_size=int(run["test_batch_size"]),
        normalize=bool(run["normalize"]),
        data_fraction=0.9,
        epochs=int(run["epochs"]),
        optimizer=run["optimizer"],
        momentum=float(run["momentum"]),
        wd=float(run["weight_decay"]),
        lr=float(run["learning_rate"]),
        lr_schedule=run["learning_rate_schedule"],
        router_optimizer=run["router_optimizer_argument"],
        router_lr=float(run["router_learning_rate_argument"]),
        router_lr_schedule=run["router_learning_rate_schedule_argument"],
        epsilon=float(run["epsilon"]),
        num_steps=int(run["train_attack_steps"]),
        step_size=float(run["train_attack_step_size"]),
        epsilon_test=float(run["test_epsilon"]),
        num_steps_test=int(run["test_attack_steps"]),
        step_size_test=float(run["test_attack_step_size"]),
        beta=float(run["beta"]),
        alpha=float(run["alpha"]),
        evaluate=False,
        ratio=float(run["ratio"]),
        n_expert=int(run["n_expert"]),
    )


def _optimizer_state_entries(optimizer: torch.optim.Optimizer) -> int:
    return len(optimizer.state_dict()["state"])


def _official_one_batch_loader_item(
    images: torch.Tensor, target: torch.Tensor
) -> list[torch.Tensor]:
    # The released helper handles dict/list containers but has no tuple branch.
    # Current PyTorch's default collation returns a list for this dataset.
    return [images, target]


def run_smoke(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    workspace = Path(config["workspace_boundary"])
    source = Path(config["official_source"]["repository"])
    run_root = Path(config["run"]["root"])
    data_root = Path(config["dataset"]["run_data_root"])
    source_data = Path(config["dataset"]["source_root"])
    for path in (config_path, output_path, source, run_root, data_root, source_data):
        if not _inside_workspace(path, workspace):
            raise RuntimeError(f"path escapes workspace: {path}")
    if output_path.exists():
        raise FileExistsError(output_path)
    if _git(source, "rev-parse", "HEAD") != config["official_source"]["commit"]:
        raise RuntimeError("official source commit mismatch")
    if _git(source, "rev-parse", "HEAD^{tree}") != config["official_source"]["tree"]:
        raise RuntimeError("official source tree mismatch")
    if _git(source, "status", "--porcelain=v1"):
        raise RuntimeError("official source clone is dirty")
    expected_data = data_root / "CIFAR10" / "cifar-10-batches-py"
    if expected_data.resolve() != source_data.resolve() or not expected_data.is_dir():
        raise RuntimeError("official CIFAR10 run-data link is missing or mismatched")

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    from models.layers.router import build_router  # type: ignore[import-not-found]
    from utils.general_utils import (  # type: ignore[import-not-found]
        get_data_model,
        initialize_weights,
        set_router,
        set_seed,
    )
    from utils.schedules import get_optimizer  # type: ignore[import-not-found]
    from train_moe import trainer  # type: ignore[import-not-found]

    args = build_official_args(config)
    device = torch.device("cuda:0")
    free_before, total = torch.cuda.mem_get_info(device)
    required = int(config["preflight_gates"]["minimum_free_gpu_memory_gib"] * 1024**3)
    if free_before < required:
        raise RuntimeError(
            f"GPU gate failed: free={free_before / 1024**3:.3f} GiB, "
            f"required={required / 1024**3:.3f} GiB"
        )

    set_seed(args.seed)
    model, train_loader, _, _, _ = get_data_model(args, device)
    initialize_weights(model)
    optimizer = get_optimizer(model, args)
    router = build_router(num_experts=args.n_expert).to(device)
    set_router(model, router)
    router_optimizer = get_optimizer(router, args)

    images, target = next(iter(train_loader))
    images = images.to(device)
    target = target.to(device)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    expert_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if id(parameter) in optimizer_ids
    }
    router_before = {
        name: parameter.detach().cpu().clone() for name, parameter in router.named_parameters()
    }
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    trainer(
        model,
        router,
        device,
        [_official_one_batch_loader_item(images, target)],
        0,
        optimizer,
        router_optimizer,
        args,
    )
    smoke_seconds = time.monotonic() - started
    torch.cuda.synchronize(device)
    expert_changed = _changed_parameter_count(
        expert_before,
        {name: parameter for name, parameter in model.named_parameters() if name in expert_before},
    )
    router_changed = _changed_parameter_count(
        router_before, dict(router.named_parameters())
    )
    model_state_hash = _state_hash(model.state_dict())
    router_state_hash = _state_hash(router.state_dict())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".checkpoint.pt")
    torch.save(
        {
            "epoch": 1,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "router": router.state_dict(),
            "best_acc": 0.0,
            "sa_record": 0.0,
            "optimizer": optimizer.state_dict(),
            "router_optimizer": router_optimizer.state_dict(),
        },
        checkpoint_path,
    )

    set_seed(args.seed)
    resumed_model, _, _, _, _ = get_data_model(args, device)
    initialize_weights(resumed_model)
    resumed_optimizer = get_optimizer(resumed_model, args)
    resumed_router = build_router(num_experts=args.n_expert).to(device)
    set_router(resumed_model, resumed_router)
    resumed_router_optimizer = get_optimizer(resumed_router, args)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    resumed_router.load_state_dict(checkpoint["router"])
    resumed_model.load_state_dict(checkpoint["state_dict"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    resumed_router_optimizer.load_state_dict(checkpoint["router_optimizer"])
    resumed_model.eval()
    model.eval()
    with torch.no_grad():
        reference_logits = model(images[:8])
        resumed_logits = resumed_model(images[:8])
        reference_scores = router(images[:8])
        resumed_scores = resumed_router(images[:8])
    maximum_logit_error = float((reference_logits - resumed_logits).abs().max().item())
    maximum_router_error = float((reference_scores - resumed_scores).abs().max().item())

    checks = {
        "real_cifar_batch_size": int(images.shape[0]) == args.batch_size,
        "finite_forward_after_update": bool(torch.isfinite(reference_logits).all()),
        "expert_parameters_updated": expert_changed > 0,
        "router_parameters_updated": router_changed > 0,
        "expert_optimizer_state_populated": _optimizer_state_entries(optimizer) > 0,
        "router_optimizer_state_populated": _optimizer_state_entries(router_optimizer) > 0,
        "model_state_resume_exact": _state_hash(resumed_model.state_dict()) == model_state_hash,
        "router_state_resume_exact": _state_hash(resumed_router.state_dict()) == router_state_hash,
        "optimizer_state_entry_count_preserved": _optimizer_state_entries(resumed_optimizer)
        == _optimizer_state_entries(optimizer),
        "router_optimizer_state_entry_count_preserved": _optimizer_state_entries(
            resumed_router_optimizer
        )
        == _optimizer_state_entries(router_optimizer),
        "resumed_logits_exact": maximum_logit_error == 0.0,
        "resumed_router_scores_exact": maximum_router_error == 0.0,
        "official_clone_clean_after": not bool(_git(source, "status", "--porcelain=v1")),
    }
    result = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "ADV_MOE_OFFICIAL_CODE_REAL_BATCH_UPDATE_AND_RESUME_SMOKE",
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "official_source": {
            "commit": _git(source, "rev-parse", "HEAD"),
            "tree": _git(source, "rev-parse", "HEAD^{tree}"),
            "clean": not bool(_git(source, "status", "--porcelain=v1")),
        },
        "device": {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "free_gib_before": free_before / 1024**3,
            "total_gib": total / 1024**3,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        },
        "batch": {"size": int(images.shape[0]), "shape": list(images.shape)},
        "updates": {
            "expert_parameters_changed": expert_changed,
            "router_parameters_changed": router_changed,
            "expert_optimizer_state_entries": _optimizer_state_entries(optimizer),
            "router_optimizer_state_entries": _optimizer_state_entries(router_optimizer),
        },
        "resume": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "maximum_logit_error": maximum_logit_error,
            "maximum_router_score_error": maximum_router_error,
        },
        "smoke_seconds": smoke_seconds,
        "checks": checks,
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    if result["status"] != "PASS":
        raise RuntimeError("AdvMoE training smoke failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run_smoke(arguments.config, arguments.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
