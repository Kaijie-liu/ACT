"""Locate the first non-finite router update in the unchanged AdvMoE trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import torch

from act.pipeline.moe.advmoe_training_smoke import build_official_args
from act.pipeline.moe.audit_advmoe_training import floating_tensor_summary


class NonfiniteDetected(RuntimeError):
    pass


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


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True
    ).strip()


def _floating_entries(value: Any) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    if torch.is_tensor(value):
        if value.is_floating_point() or value.is_complex():
            tensors.append(value)
    elif isinstance(value, dict):
        for child in value.values():
            tensors.extend(_floating_entries(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            tensors.extend(_floating_entries(child))
    return tensors


def all_finite(values: Iterable[torch.Tensor]) -> bool:
    tensors = list(values)
    if not tensors:
        return True
    flags = [torch.isfinite(tensor).all() for tensor in tensors]
    return bool(torch.stack(flags).all().item())


def _named_details(values: Iterable[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    rows = []
    elements = 0
    finite_elements = 0
    for name, tensor in values:
        if not (tensor.is_floating_point() or tensor.is_complex()):
            continue
        finite = torch.isfinite(tensor)
        count = tensor.numel()
        finite_count = int(finite.sum().item())
        elements += count
        finite_elements += finite_count
        if finite_count != count:
            rows.append(
                {
                    "name": name,
                    "elements": count,
                    "finite_elements": finite_count,
                    "nan_elements": int(torch.isnan(tensor).sum().item()),
                    "inf_elements": int(torch.isinf(tensor).sum().item()),
                }
            )
    return {
        "elements": elements,
        "finite_elements": finite_elements,
        "all_finite": elements == finite_elements,
        "nonfinite_entries": rows,
    }


def _router_snapshot(
    router: torch.nn.Module, router_optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    parameters = list(router.named_parameters())
    buffers = list(router.named_buffers())
    gradients = [
        (name, parameter.grad)
        for name, parameter in parameters
        if parameter.grad is not None
    ]
    optimizer_tensors = _floating_entries(router_optimizer.state)
    quick = {
        "parameters": all_finite(parameter for _name, parameter in parameters),
        "buffers": all_finite(buffer for _name, buffer in buffers),
        "gradients": all_finite(gradient for _name, gradient in gradients),
        "optimizer_state": all_finite(optimizer_tensors),
    }
    return {"all_finite": all(quick.values()), "quick": quick}


def _router_failure_details(
    router: torch.nn.Module, router_optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    parameters = list(router.named_parameters())
    buffers = list(router.named_buffers())
    gradients = [
        (name, parameter.grad)
        for name, parameter in parameters
        if parameter.grad is not None
    ]
    return {
        "parameters": _named_details(parameters),
        "buffers": _named_details(buffers),
        "gradients": _named_details(gradients),
        "optimizer_state": floating_tensor_summary(router_optimizer.state),
    }


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class TrackingLoader:
    def __init__(self, loader: Any, maximum_batches: int):
        self.loader = loader
        self.maximum_batches = maximum_batches
        self.current: dict[str, Any] | None = None

    def __len__(self) -> int:
        return min(len(self.loader), self.maximum_batches)

    def __iter__(self):
        for index, data in enumerate(self.loader):
            if index >= self.maximum_batches:
                break
            images, labels = data
            self.current = {
                "zero_based_batch_index": index,
                "batch_size": int(images.shape[0]),
                "images_sha256": _tensor_hash(images),
                "labels_sha256": _tensor_hash(labels),
            }
            yield data


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    workspace = Path(config["workspace_boundary"])
    config_path = _inside(config_path, workspace)
    act_repository = _inside(Path(config["act_repository"]), workspace)
    source = _inside(Path(config["official_source"]["repository"]), workspace)
    training_config_path = _inside(Path(config["training_config"]), workspace)
    exclusion = _inside(Path(config["numerical_exclusion"]), workspace)
    dataset_root = _inside(Path(config["dataset_root"]), workspace)
    output = _inside(Path(config["output"]), workspace)
    environment = _inside(Path(config["environment"]), workspace)
    if config.get("status") != "PREREGISTERED_NOT_RUN":
        raise RuntimeError("unexpected config status")
    expected_stages = [
        "BEFORE_MAIN_OPTIMIZER_STEP",
        "AFTER_MAIN_OPTIMIZER_STEP",
        "BEFORE_ROUTER_OPTIMIZER_STEP",
        "AFTER_ROUTER_OPTIMIZER_STEP",
    ]
    if config.get("stages") != expected_stages:
        raise RuntimeError("unexpected diagnosis stage list")
    if config.get("stop_at_first_nonfinite") is not True:
        raise RuntimeError("diagnosis must stop at the first non-finite state")
    expected_python = (environment / "bin" / "python").resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(
            f"wrong Python executable: {Path(sys.executable).resolve()} != "
            f"{expected_python}"
        )
    if output.exists():
        raise FileExistsError(output)
    if _git(act_repository, "branch", "--show-current") != config["required_branch"]:
        raise RuntimeError("ACT branch gate failed")
    if _git(act_repository, "status", "--porcelain=v1"):
        raise RuntimeError("ACT worktree is dirty")
    if _git(source, "status", "--porcelain=v1"):
        raise RuntimeError("official source clone is dirty")
    if _git(source, "rev-parse", "HEAD") != config["official_source"]["commit"]:
        raise RuntimeError("official source commit mismatch")
    if _git(source, "rev-parse", "HEAD^{tree}") != config["official_source"]["tree"]:
        raise RuntimeError("official source tree mismatch")
    if _sha256(training_config_path) != config["training_config_sha256"]:
        raise RuntimeError("training config hash mismatch")
    exclusion_payload = json.loads(exclusion.read_text(encoding="utf-8"))
    if exclusion_payload.get("status") != "EXCLUDED_NONFINITE_ROUTER":
        raise RuntimeError("numerical exclusion gate is missing")
    if not dataset_root.is_dir():
        raise RuntimeError("official run dataset root is missing")

    device = torch.device(config["device"])
    free, total = torch.cuda.mem_get_info(device)
    required = int(float(config["minimum_free_gpu_memory_gib"]) * 1024**3)
    if free < required:
        raise RuntimeError("GPU memory gate failed")
    torch.set_num_threads(int(config["torch_threads"]))
    torch.set_num_interop_threads(1)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    from models.layers.router import build_router  # type: ignore[import-not-found]
    from train_moe import trainer  # type: ignore[import-not-found]
    from utils.general_utils import (  # type: ignore[import-not-found]
        get_data_model,
        initialize_weights,
        set_router,
        set_seed,
    )
    from utils.schedules import get_optimizer  # type: ignore[import-not-found]

    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    arguments = build_official_args(training_config)
    if Path(arguments.data_dir).resolve() != dataset_root.resolve():
        raise RuntimeError("training args use a different dataset root")
    set_seed(arguments.seed)
    model, train_loader, _router_loader, _test_loader, _image_dim = get_data_model(
        arguments, device
    )
    initialize_weights(model)
    optimizer = get_optimizer(model, arguments)
    router = build_router(num_experts=arguments.n_expert).to(device)
    set_router(model, router)
    router_optimizer = get_optimizer(router, arguments)
    tracked_loader = TrackingLoader(train_loader, int(config["maximum_batches"]))

    phase_log: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    counters = {"main": 0, "router": 0}

    def inspect(stage: str, family: str) -> None:
        nonlocal failure
        snapshot = _router_snapshot(router, router_optimizer)
        row = {
            "stage": stage,
            "zero_based_batch_index": counters[family],
            **snapshot,
        }
        phase_log.append(row)
        if not snapshot["all_finite"]:
            failure = {
                **row,
                "batch_identity": tracked_loader.current,
                "details": _router_failure_details(router, router_optimizer),
            }
            raise NonfiniteDetected(stage)

    def main_pre(_optimizer, _args, _kwargs):
        inspect("BEFORE_MAIN_OPTIMIZER_STEP", "main")

    def main_post(_optimizer, _args, _kwargs):
        inspect("AFTER_MAIN_OPTIMIZER_STEP", "main")
        counters["main"] += 1

    def router_pre(_optimizer, _args, _kwargs):
        inspect("BEFORE_ROUTER_OPTIMIZER_STEP", "router")

    def router_post(_optimizer, _args, _kwargs):
        inspect("AFTER_ROUTER_OPTIMIZER_STEP", "router")
        counters["router"] += 1

    handles = [
        optimizer.register_step_pre_hook(main_pre),
        optimizer.register_step_post_hook(main_post),
        router_optimizer.register_step_pre_hook(router_pre),
        router_optimizer.register_step_post_hook(router_post),
    ]
    initial = _router_failure_details(router, router_optimizer)
    if not all(group["all_finite"] for group in initial.values()):
        for handle in handles:
            handle.remove()
        raise RuntimeError("router state is non-finite before the official trainer starts")
    started = time.monotonic()
    caught = None
    try:
        trainer(
            model,
            router,
            device,
            tracked_loader,
            0,
            optimizer,
            router_optimizer,
            arguments,
        )
    except NonfiniteDetected as error:
        caught = str(error)
    finally:
        for handle in handles:
            handle.remove()
    elapsed = time.monotonic() - started

    state_dict = model.state_dict()
    main_without_router = {
        name: tensor
        for name, tensor in state_dict.items()
        if not name.startswith("router.") and ".router." not in name
    }
    result = {
        "schema_version": 1,
        "status": (
            "COMPLETED_NONFINITE_DETECTED"
            if failure is not None
            else "COMPLETED_NO_NONFINITE_WITHIN_BUDGET"
        ),
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "official_source": {
            "commit": _git(source, "rev-parse", "HEAD"),
            "tree": _git(source, "rev-parse", "HEAD^{tree}"),
            "clean_after": not bool(_git(source, "status", "--porcelain=v1")),
        },
        "execution": {
            "device": str(device),
            "free_gpu_gib_before": free / 1024**3,
            "total_gpu_gib": total / 1024**3,
            "elapsed_seconds": elapsed,
            "maximum_batches": int(config["maximum_batches"]),
            "completed_main_steps": counters["main"],
            "completed_router_steps": counters["router"],
            "caught": caught,
        },
        "initial_router": initial,
        "phase_log": phase_log,
        "first_nonfinite": failure,
        "final_main_without_router": floating_tensor_summary(main_without_router),
        "interpretation": config["interpretation"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.config)
    print(
        json.dumps(
            {
                "status": result["status"],
                "execution": result["execution"],
                "first_nonfinite": result["first_nonfinite"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
