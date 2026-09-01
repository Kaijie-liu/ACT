"""Run the frozen lazy top-k expert-count and partial-MIP-start scaling study."""

from __future__ import annotations

import argparse
from itertools import combinations
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import torch

from act.back_end.core import Bounds
from act.back_end.moe.hz_routing import analyze_topk_sets, enumerate_topk_sets_lazy
from act.back_end.solver.solver_hz import sparse_hz_from_bounds, sparse_hz_linear
from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append(handle, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _write(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refuses to overwrite {path}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _router(config: dict[str, Any], experts: int, family: str, frame_id: int):
    dimension = int(config["input_dimension"])
    radius = float(config["random_box_radius"])
    domain = sparse_hz_from_bounds(
        Bounds(
            torch.full((1, dimension), -radius, dtype=torch.float64),
            torch.full((1, dimension), radius, dtype=torch.float64),
        ),
        frame_id=frame_id,
    )
    if family == "all_tied_worst_case":
        weight = np.zeros((experts, dimension), dtype=np.float64)
        weight[:, 0] = 1.0
        bias = np.zeros(experts, dtype=np.float64)
    elif family == "strictly_stable":
        weight = np.zeros((experts, dimension), dtype=np.float64)
        bias = np.linspace(1.0, -1.0, experts, dtype=np.float64)
    elif family == "random_affine_box":
        seed = int(config["random_seed"]) + 1009 * experts
        generator = np.random.default_rng(seed)
        weight = generator.normal(size=(experts, dimension)) / math.sqrt(dimension)
        bias = np.linspace(0.5, -0.5, experts, dtype=np.float64)
        bias += generator.normal(scale=0.02, size=experts)
    else:  # pragma: no cover - config validation
        raise ValueError(f"unknown family: {family}")
    return sparse_hz_linear(domain, weight, bias), weight, bias


def _condition_order(expert_index: int, family_index: int) -> tuple[bool, bool]:
    return (False, True) if (expert_index + family_index) % 2 == 0 else (True, False)


def run(config_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = _inside(Path(config["output_dir"]), WRITE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"refuses to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    rows_path = output_dir / "rows.jsonl"
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    with rows_path.open("x", encoding="utf-8") as handle:
        for expert_index, experts in enumerate(config["num_experts"]):
            experts = int(experts)
            for family_index, family in enumerate(config["families"]):
                router, weight, bias = _router(
                    config,
                    experts,
                    str(family),
                    frame_id=26090100 + expert_index * 100 + family_index,
                )
                exhaustive_sets = None
                exhaustive_seconds = 0.0
                if experts <= int(config["exhaustive_differential_max_experts"]):
                    exhaustive_started = time.monotonic()
                    exhaustive = analyze_topk_sets(
                        router,
                        int(config["top_k"]),
                        time_limit_per_set=float(config["time_limits_seconds"][str(experts)]),
                    )
                    exhaustive_seconds = time.monotonic() - exhaustive_started
                    if not exhaustive.exact:
                        raise RuntimeError("small-E exhaustive differential did not complete")
                    exhaustive_sets = [list(value) for value in exhaustive.feasible]
                for order_index, submit_start in enumerate(
                    _condition_order(expert_index, family_index)
                ):
                    result = enumerate_topk_sets_lazy(
                        router,
                        int(config["top_k"]),
                        time_limit=float(config["time_limits_seconds"][str(experts)]),
                        big_m_padding=float(config["big_m_padding"]),
                        big_m_support_mode=str(config["big_m_support_mode"]),
                        submit_mip_starts=bool(submit_start),
                    )
                    row = {
                        "experts": experts,
                        "top_k": int(config["top_k"]),
                        "family": family,
                        "pair_order_index": order_index,
                        "submit_partial_mip_starts": bool(submit_start),
                        "route_set_count": len(result.route_sets),
                        "route_sets": [list(value) for value in result.route_sets],
                        "complete": result.complete,
                        "status": result.status,
                        "reason": result.reason,
                        "solves": result.solves,
                        "no_good_cuts": result.no_good_cuts,
                        "selector_binaries": result.selector_binaries,
                        "pairwise_order_constraints": experts * (experts - 1),
                        "elapsed_seconds": result.elapsed,
                        "big_m_support_mode": result.big_m_support_mode,
                        "big_m_support_exact": result.big_m_support_exact,
                        "telemetry": dict(result.telemetry),
                        "router_weight_sha256": hashlib.sha256(weight.tobytes()).hexdigest(),
                        "router_bias_sha256": hashlib.sha256(bias.tobytes()).hexdigest(),
                        "exhaustive_sets": exhaustive_sets,
                        "exhaustive_seconds_shared_by_pair": exhaustive_seconds,
                    }
                    rows.append(row)
                    _append(handle, row)
    paired = []
    for experts in map(int, config["num_experts"]):
        for family in config["families"]:
            pair = [
                row for row in rows if row["experts"] == experts and row["family"] == family
            ]
            if len(pair) != 2:
                raise RuntimeError("paired scaling row is incomplete")
            by_start = {row["submit_partial_mip_starts"]: row for row in pair}
            no_start = by_start[False]
            with_start = by_start[True]
            paired.append(
                {
                    "experts": experts,
                    "family": family,
                    "sets_equal": no_start["route_sets"] == with_start["route_sets"],
                    "status_equal": no_start["status"] == with_start["status"],
                    "no_start_seconds": no_start["elapsed_seconds"],
                    "with_start_seconds": with_start["elapsed_seconds"],
                    "with_over_without_ratio": (
                        with_start["elapsed_seconds"] / no_start["elapsed_seconds"]
                        if no_start["elapsed_seconds"] > 0
                        else None
                    ),
                }
            )
    summary = {
        "schema_version": 1,
        "status": "COMPLETED_NOT_INDEPENDENTLY_AUDITED",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "thread_environment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
        },
        "rows": len(rows),
        "pairs": paired,
        "rows_jsonl": {"path": str(rows_path), "sha256": _sha256(rows_path)},
        "wall_seconds": time.monotonic() - started,
        "claim_boundary": config["claim_boundary"],
    }
    _write(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
