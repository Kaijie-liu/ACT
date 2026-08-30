"""Run numerical router-margin bounds in the isolated CROWN environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import typing
from typing import Any

from typing_extensions import override

if not hasattr(typing, "override"):
    typing.override = override  # type: ignore[attr-defined]

# This worker is intentionally launched as a file so that the Python 3.11
# compatibility shim above runs before importing ``act.pipeline``.  Direct
# file execution places this module's directory, rather than the repository
# root, on sys.path; establish the latter explicitly before any ACT import.
PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

import numpy as np
import torch

from act.pipeline.moe.advmoe_adapter import (
    CrownCompatibleAdvMoeRouter,
    adapter_equivalence,
    construct_official_init,
    state_dict_sha256,
)
from act.pipeline.moe.published_moe_router_gradient_audit import (
    MOE_ROOT,
    PROJECT_ROOT,
    _inside,
    _sha256,
)


def batchnorm_deployment_identity(module: torch.nn.Module) -> dict[str, Any]:
    layers = [item for item in module.modules() if isinstance(item, torch.nn.BatchNorm2d)]
    return {
        "layers": len(layers),
        "training_layers": sum(int(item.training) for item in layers),
        "maximum_abs_running_mean": max(
            (float(item.running_mean.abs().max().item()) for item in layers),
            default=0.0,
        ),
        "maximum_abs_running_variance_minus_one": max(
            (float((item.running_var - 1).abs().max().item()) for item in layers),
            default=0.0,
        ),
        "maximum_batches_tracked": max(
            (int(item.num_batches_tracked.item()) for item in layers),
            default=0,
        ),
    }


def crown_bound_options(config: dict[str, Any]) -> dict[str, Any]:
    """Translate the frozen scalable-CROWN controls into auto_LiRPA options."""
    return {
        "conv_mode": str(config.get("conv_mode", "patches")),
        "sparse_features_alpha": bool(config.get("sparse_alpha", False)),
        "sparse_spec_alpha": bool(config.get("sparse_alpha", False)),
        "sparse_intermediate_bounds": bool(config.get("sparse_intermediate", False)),
        "use_full_conv_alpha": bool(config.get("full_conv_alpha", True)),
        "crown_batch_size": int(config.get("crown_batch_size", int(1e9))),
        "max_crown_size": int(config.get("max_crown_size", int(1e9))),
        "batched_crown_max_vram_ratio": float(
            config.get("batched_crown_max_vram_ratio", 0.9)
        ),
        "optimize_bound_args": {
            "iteration": int(config.get("alpha_iterations", 20)),
            "lr_alpha": float(config.get("alpha_lr", 0.1)),
            "use_shared_alpha": bool(config.get("share_alphas", False)),
            "keep_best": True,
        },
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"worker refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _raw_frontend_probe(router: torch.nn.Module, inputs: torch.Tensor) -> dict[str, Any]:
    from auto_LiRPA import BoundedModule

    started = time.monotonic()
    try:
        BoundedModule(router, inputs, device=str(inputs.device))
        return {"status": "ACCEPTED", "seconds": time.monotonic() - started}
    except Exception as error:
        return {
            "status": "REJECTED",
            "error_type": type(error).__name__,
            "error_message": str(error)[:300],
            "seconds": time.monotonic() - started,
        }


def run(prepare_path: Path, output_path: Path) -> dict[str, Any]:
    prepare_path = _inside(prepare_path, MOE_ROOT)
    output_path = _inside(output_path, MOE_ROOT)
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    artifact = _inside(Path(prepare["input_artifact"]["path"]), MOE_ROOT)
    if _sha256(artifact) != prepare["input_artifact"]["sha256"]:
        raise RuntimeError("input artifact identity changed")
    with np.load(artifact, allow_pickle=False) as arrays:
        inputs = torch.from_numpy(arrays["inputs"].copy())
        clean_routes = torch.from_numpy(arrays["clean_routes"].copy()).long()
    config = prepare["config"]
    seed = int(config["model_seed"])
    _model, raw_router, _moe_type = construct_official_init(seed)
    if state_dict_sha256(raw_router) != prepare["router_sha256"]:
        raise RuntimeError("independent router construction changed")

    device = str(config["bound_worker"]["device"])
    method = str(config["bound_worker"]["method"])
    worker_config = config["bound_worker"]
    conv_mode = str(worker_config["conv_mode"])
    torch.set_num_threads(int(config["bound_worker"]["torch_threads"]))
    raw_router = raw_router.to(device).eval()
    inputs = inputs.to(device)
    clean_routes = clean_routes.to(device)
    raw_probe = _raw_frontend_probe(raw_router, inputs[:1])
    batchnorm_identity = batchnorm_deployment_identity(raw_router)
    if batchnorm_identity["training_layers"] != 0:
        raise RuntimeError("router BatchNorm layers are not in deployment eval mode")

    adapted = CrownCompatibleAdvMoeRouter(raw_router).to(device).eval()
    adapted.validate_input_shape(inputs)
    equivalence = adapter_equivalence(raw_router, inputs)
    if not equivalence["outputs_equal"] or not equivalence["routes_equal"]:
        raise RuntimeError("CROWN lowering failed concrete equivalence")

    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    bound_options = crown_bound_options(worker_config)
    sample_batch_size = int(worker_config.get("sample_batch_size", len(inputs)))
    bound_upper_enabled = bool(worker_config.get("bound_upper", True))
    epsilons = [float(value) for value in config["epsilons"]]
    row_accumulators = {
        epsilon: {
            "lower_bounds": [None] * len(inputs),
            "upper_bounds": [None] * len(inputs) if bound_upper_enabled else [],
            "seconds": 0.0,
            "sample_errors": [],
        }
        for epsilon in epsilons
    }
    build_seconds = 0.0
    peak_memory_bytes = 0
    for start in range(0, len(inputs), sample_batch_size):
        end = min(start + sample_batch_size, len(inputs))
        chunk_inputs = inputs[start:end]
        chunk_routes = clean_routes[start:end]
        chunk_adapter = CrownCompatibleAdvMoeRouter(raw_router).to(device).eval()
        C = torch.zeros(len(chunk_inputs), 1, 2, device=device, dtype=inputs.dtype)
        batch_indices = torch.arange(len(chunk_inputs), device=device)
        C[batch_indices, 0, chunk_routes] = 1
        C[batch_indices, 0, 1 - chunk_routes] = -1
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        build_started = time.monotonic()
        bounded = BoundedModule(
            chunk_adapter,
            chunk_inputs,
            device=device,
            bound_opts=bound_options,
        )
        build_seconds += time.monotonic() - build_started
        for epsilon in epsilons:
            lower = torch.clamp(chunk_inputs - epsilon, 0, 1)
            upper = torch.clamp(chunk_inputs + epsilon, 0, 1)
            bounded_input = BoundedTensor(
                chunk_inputs,
                PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper),
            )
            started = time.monotonic()
            try:
                bound_lower, bound_upper = bounded.compute_bounds(
                    x=(bounded_input,),
                    C=C,
                    method=method,
                    bound_upper=bound_upper_enabled,
                )
                lower_values = (
                    bound_lower.reshape(-1).detach().cpu().double().tolist()
                )
                row_accumulators[epsilon]["lower_bounds"][start:end] = lower_values
                if bound_upper_enabled:
                    upper_values = (
                        bound_upper.reshape(-1).detach().cpu().double().tolist()
                    )
                    row_accumulators[epsilon]["upper_bounds"][start:end] = upper_values
            except Exception as exc:
                row_accumulators[epsilon]["sample_errors"].append(
                    {
                        "start": start,
                        "end": end,
                        "type": type(exc).__name__,
                        "message": str(exc)[:500],
                    }
                )
                if isinstance(exc, torch.OutOfMemoryError) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            row_accumulators[epsilon]["seconds"] += time.monotonic() - started
        if device.startswith("cuda"):
            peak_memory_bytes = max(peak_memory_bytes, torch.cuda.max_memory_allocated())
            maximum_peak = worker_config.get("maximum_peak_memory_bytes")
            if maximum_peak is not None and peak_memory_bytes > int(maximum_peak):
                raise RuntimeError(
                    f"CROWN peak memory {peak_memory_bytes} exceeds frozen limit "
                    f"{int(maximum_peak)}"
                )
        del bounded, chunk_adapter
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    rows = []
    for epsilon in epsilons:
        accumulator = row_accumulators[epsilon]
        errors = accumulator["sample_errors"]
        rows.append(
            {
                "epsilon": epsilon,
                "status": (
                    "COMPLETED_NUMERICAL_FILTER" if not errors else "PARTIAL_ERROR"
                ),
                "lower_bounds": accumulator["lower_bounds"],
                "upper_bounds": accumulator["upper_bounds"],
                "seconds": accumulator["seconds"],
                "error": None if not errors else {"sample_errors": errors},
            }
        )
    result = {
        "schema_version": 1,
        "status": (
            "COMPLETED_NUMERICAL_FILTER"
            if all(row["status"] == "COMPLETED_NUMERICAL_FILTER" for row in rows)
            else "COMPLETED_WITH_BOUND_ERRORS"
        ),
        "prepare": {"path": str(prepare_path), "sha256": _sha256(prepare_path)},
        "router_sha256": state_dict_sha256(raw_router),
        "raw_frontend_probe": raw_probe,
        "adapter_equivalence": equivalence,
        "batchnorm_deployment_identity": batchnorm_identity,
        "graph_build_seconds": build_seconds,
        "method": method,
        "conv_mode": conv_mode,
        "device": device,
        "bound_options": bound_options,
        "sample_batch_size": sample_batch_size,
        "bound_upper": bound_upper_enabled,
        "peak_memory_bytes": peak_memory_bytes,
        "rows": rows,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "auto_lirpa": importlib.metadata.version("auto_LiRPA"),
        },
        "numerical_scope": (
            "Positive lower bounds are numerical conformance filters only. They "
            "are not outward-rounded formal route-stability certificates. Negative "
            "bounds are UNKNOWN, never route-unstable."
        ),
    }
    _write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.prepare, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
