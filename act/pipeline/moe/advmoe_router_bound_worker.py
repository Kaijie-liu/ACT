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
    conv_mode = str(config["bound_worker"]["conv_mode"])
    torch.set_num_threads(int(config["bound_worker"]["torch_threads"]))
    raw_router = raw_router.to(device).eval()
    inputs = inputs.to(device)
    clean_routes = clean_routes.to(device)
    raw_probe = _raw_frontend_probe(raw_router, inputs[:1])

    adapted = CrownCompatibleAdvMoeRouter(raw_router).to(device).eval()
    adapted.validate_input_shape(inputs)
    equivalence = adapter_equivalence(raw_router, inputs)
    if not equivalence["outputs_equal"] or not equivalence["routes_equal"]:
        raise RuntimeError("CROWN lowering failed concrete equivalence")

    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    C = torch.zeros(len(inputs), 1, 2, device=device, dtype=inputs.dtype)
    batch_indices = torch.arange(len(inputs), device=device)
    C[batch_indices, 0, clean_routes] = 1
    C[batch_indices, 0, 1 - clean_routes] = -1
    build_started = time.monotonic()
    bounded = BoundedModule(
        adapted,
        inputs,
        device=device,
        bound_opts={"conv_mode": conv_mode},
    )
    build_seconds = time.monotonic() - build_started
    rows = []
    for epsilon in config["epsilons"]:
        epsilon = float(epsilon)
        lower = torch.clamp(inputs - epsilon, 0, 1)
        upper = torch.clamp(inputs + epsilon, 0, 1)
        bounded_input = BoundedTensor(
            inputs,
            PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper),
        )
        started = time.monotonic()
        try:
            bound_lower, bound_upper = bounded.compute_bounds(
                x=(bounded_input,), C=C, method=method
            )
            lower_values = bound_lower.reshape(-1).detach().cpu().double().tolist()
            upper_values = bound_upper.reshape(-1).detach().cpu().double().tolist()
            error = None
            status = "COMPLETED_NUMERICAL_FILTER"
        except Exception as exc:
            lower_values = []
            upper_values = []
            error = {"type": type(exc).__name__, "message": str(exc)[:500]}
            status = "ERROR"
            if isinstance(exc, torch.OutOfMemoryError) and torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append(
            {
                "epsilon": epsilon,
                "status": status,
                "lower_bounds": lower_values,
                "upper_bounds": upper_values,
                "seconds": time.monotonic() - started,
                "error": error,
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
        "graph_build_seconds": build_seconds,
        "method": method,
        "conv_mode": conv_mode,
        "device": device,
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
