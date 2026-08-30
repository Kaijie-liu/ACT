"""Bracket float32 sparse-CROWN positive-bound reach on frozen init samples.

The output is deliberately numerical-only.  The installed backend is not
outward-rounded, and very small requested boxes can collapse under float32.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import typing
from typing import Any

from typing_extensions import override

if not hasattr(typing, "override"):
    typing.override = override  # type: ignore[attr-defined]

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
from act.pipeline.moe.certified_artifact_identity import represented_linf_box
from act.pipeline.moe.advmoe_router_bound_worker import (
    batchnorm_deployment_identity,
    crown_bound_options,
)
from act.pipeline.moe.published_moe_router_gradient_audit import (
    MOE_ROOT,
    PROJECT_ROOT,
    _inside,
    _sha256,
)


def geometric_midpoint(lower: float, upper: float) -> float:
    if not (0 < lower < upper):
        raise ValueError("geometric bracket must satisfy 0 < lower < upper")
    return math.exp((math.log(lower) + math.log(upper)) / 2.0)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _memory_gate(config: dict[str, Any]) -> dict[str, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("registered CUDA worker has no CUDA device")
    free, total = torch.cuda.mem_get_info()
    required = int(config["resource_gate"]["minimum_free_memory_bytes"])
    if free < required:
        raise RuntimeError(f"free CUDA memory {free} is below gate {required}")
    return {"free_bytes_before": int(free), "total_bytes": int(total)}


def run(config_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_path = _inside(Path(config["output"]), MOE_ROOT)
    if output_path.parent.exists():
        raise RuntimeError(f"refuses to reuse output directory {output_path.parent}")
    memory_gate = _memory_gate(config)

    source_dir = _inside(Path(config["source_result_dir"]), MOE_ROOT)
    input_path = source_dir / "inputs.npz"
    prepare_path = source_dir / "prepare.json"
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    with np.load(input_path, allow_pickle=False) as arrays:
        all_inputs = torch.from_numpy(arrays["inputs"].copy())
        all_routes = torch.from_numpy(arrays["clean_routes"].copy()).long()
        all_ranks = arrays["dataset_indices"].astype(np.int64)
    slots = np.asarray(config["sample_slots"], dtype=np.int64)
    if len(np.unique(slots)) != len(slots) or np.any(slots < 0) or np.any(
        slots >= len(all_inputs)
    ):
        raise RuntimeError("sample slots are invalid")

    _model, raw_router, _moe_type = construct_official_init(
        int(config["model_seed"])
    )
    del _model
    if state_dict_sha256(raw_router) != prepare["router_sha256"]:
        raise RuntimeError("router identity changed")
    device = str(config["device"])
    torch.set_num_threads(int(config["torch_threads"]))
    raw_router = raw_router.to(device).eval()
    inputs = all_inputs[slots].to(device)
    routes = all_routes[slots].to(device)
    equivalence = adapter_equivalence(raw_router, inputs)
    if not equivalence["outputs_equal"] or not equivalence["routes_equal"]:
        raise RuntimeError("adapter equivalence changed")
    bn_identity = batchnorm_deployment_identity(raw_router)
    if bn_identity["training_layers"] != 0:
        raise RuntimeError("BatchNorm deployment state changed")

    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    options = crown_bound_options(config["bound_options"])
    method = str(config["method"])
    maximum_peak = int(config["resource_gate"]["maximum_peak_memory_bytes"])
    overall_peak = 0

    def compute_bound(sample_slot: int, epsilon: float) -> dict[str, Any]:
        nonlocal overall_peak
        sample = inputs[sample_slot : sample_slot + 1]
        route = int(routes[sample_slot].item())
        C = torch.zeros(1, 1, 2, device=device, dtype=sample.dtype)
        C[0, 0, route] = 1
        C[0, 0, 1 - route] = -1
        adapter = CrownCompatibleAdvMoeRouter(raw_router).to(device).eval()
        torch.cuda.reset_peak_memory_stats()
        bounded = BoundedModule(adapter, sample, device=device, bound_opts=options)
        lower, upper, represented_set = represented_linf_box(sample, epsilon)
        bounded_input = BoundedTensor(
            sample,
            PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper),
        )
        started = time.monotonic()
        bound_lower = None
        try:
            bound_lower, _ = bounded.compute_bounds(
                x=(bounded_input,), C=C, method=method, bound_upper=False
            )
            value = float(bound_lower.reshape(-1)[0].detach().cpu().item())
        finally:
            seconds = time.monotonic() - started
            current_peak = int(torch.cuda.max_memory_allocated())
            del bounded, adapter, bounded_input, lower, upper, bound_lower, C
            gc.collect()
            torch.cuda.empty_cache()
        overall_peak = max(overall_peak, current_peak)
        if overall_peak > maximum_peak:
            raise RuntimeError(
                f"CROWN peak {overall_peak} exceeds resource gate {maximum_peak}"
            )
        return {
            "requested_epsilon": float(epsilon),
            "lower_bound": value,
            "positive_numerical_filter": bool(value > 0),
            "effective_lower_linf": represented_set.effective_lower_linf,
            "effective_upper_linf": represented_set.effective_upper_linf,
            "changed_lower_coordinates": (
                represented_set.coordinate_count
                - represented_set.unchanged_lower_coordinates
            ),
            "changed_upper_coordinates": (
                represented_set.coordinate_count
                - represented_set.unchanged_upper_coordinates
            ),
            "represented_set": represented_set.as_dict(),
            "seconds": seconds,
            "peak_memory_bytes": current_peak,
        }

    floor = float(config["minimum_positive_epsilon"])
    ceiling = float(config["maximum_epsilon"])
    iterations = int(config["log_bisection_iterations"])
    rows: list[dict[str, Any]] = []
    for local_slot, source_slot in enumerate(slots.tolist()):
        evaluations = [compute_bound(local_slot, 0.0)]
        floor_row = compute_bound(local_slot, floor)
        ceiling_row = compute_bound(local_slot, ceiling)
        evaluations.extend([floor_row, ceiling_row])
        if evaluations[0]["lower_bound"] <= 0:
            status = "NO_POSITIVE_BOUND_AT_ZERO"
            positive_epsilon = 0.0
            negative_epsilon = 0.0
        elif ceiling_row["positive_numerical_filter"]:
            status = "POSITIVE_THROUGH_CEILING"
            positive_epsilon = ceiling
            negative_epsilon = None
        elif not floor_row["positive_numerical_filter"]:
            status = "TRANSITION_BELOW_FLOOR"
            positive_epsilon = 0.0
            negative_epsilon = floor
        else:
            positive_epsilon = floor
            negative_epsilon = ceiling
            for _iteration in range(iterations):
                midpoint = geometric_midpoint(positive_epsilon, negative_epsilon)
                midpoint_row = compute_bound(local_slot, midpoint)
                evaluations.append(midpoint_row)
                if midpoint_row["positive_numerical_filter"]:
                    positive_epsilon = midpoint
                else:
                    negative_epsilon = midpoint
            status = "NUMERICAL_TRANSITION_BRACKETED"
        evaluations.sort(key=lambda row: row["requested_epsilon"])
        rows.append(
            {
                "sample_slot": int(source_slot),
                "dataset_rank": int(all_ranks[source_slot]),
                "clean_route": int(routes[local_slot].item()),
                "status": status,
                "positive_requested_epsilon": positive_epsilon,
                "negative_requested_epsilon": negative_epsilon,
                "evaluations": evaluations,
            }
        )

    result = {
        "schema_version": 1,
        "status": "COMPLETED_NUMERICAL_ONLY",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "source": {
            "inputs": {"path": str(input_path), "sha256": _sha256(input_path)},
            "prepare": {"path": str(prepare_path), "sha256": _sha256(prepare_path)},
        },
        "router_sha256": state_dict_sha256(raw_router),
        "adapter_equivalence": equivalence,
        "batchnorm_deployment_identity": bn_identity,
        "memory_gate": memory_gate,
        "peak_memory_bytes": overall_peak,
        "bound_method": method,
        "bound_options": options,
        "rows": rows,
        "interpretation": (
            "Positive bounds are float32 numerical filters only. The backend is not "
            "outward-rounded, and requested boxes can collapse or quantize in float32. "
            "No row may be labelled FORMAL_ROUTE_STABLE or sound reach."
        ),
    }
    _write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
