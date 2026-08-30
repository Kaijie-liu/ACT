"""Run the frozen E=8 lazy-enumeration and big-M correctness instance."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import os
from pathlib import Path

import torch

from act.back_end.core import Bounds
from act.back_end.moe.hz_routing import (
    analyze_topk_sets,
    condition_topk_membership,
    enumerate_topk_sets_lazy,
)
from act.back_end.solver.solver_hz import (
    hz_add_output_inequalities,
    sparse_hz_from_bounds,
    sparse_hz_linear,
)
from act.pipeline.moe.published_moe_router_gradient_audit import (
    MOE_ROOT,
    PROJECT_ROOT,
    _inside,
    _sha256,
)


def _write_json(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        raise RuntimeError(f"refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(config_path: Path) -> dict[str, object]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = _inside(Path(config["output_dir"]), MOE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"refuses to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    experts = int(config["num_experts"])
    top_k = int(config["top_k"])
    frame_id = int(config["frame_id"])
    domain = sparse_hz_from_bounds(
        Bounds(torch.tensor([[-1.0]]), torch.tensor([[1.0]])),
        frame_id=frame_id,
    )
    tied_router = sparse_hz_linear(
        domain,
        torch.ones(experts, 1, dtype=torch.float64),
    )
    exhaustive = analyze_topk_sets(
        tied_router,
        top_k,
        time_limit_per_set=float(config["exhaustive_time_limit_per_set"]),
    )
    lazy = enumerate_topk_sets_lazy(
        tied_router,
        top_k,
        time_limit=float(config["lazy_time_limit"]),
    )

    guarded = hz_add_output_inequalities(domain, [[-1.0]], [-0.5])
    guarded_router = sparse_hz_linear(
        guarded,
        [[1.0], [-1.0], [0.0]],
    )
    fast = condition_topk_membership(
        guarded_router,
        expert=0,
        top_k=2,
        big_m_support_mode="fast",
    )
    exact = condition_topk_membership(
        guarded_router,
        expert=0,
        top_k=2,
        big_m_support_mode="exact",
        big_m_support_time_limit=float(config["big_m_support_time_limit"]),
    )
    expected = tuple(combinations(range(experts), top_k))
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETED_NOT_INDEPENDENTLY_AUDITED",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "enumeration": {
            "expected_set_count": len(expected),
            "exhaustive_set_count": len(exhaustive.feasible),
            "exhaustive_exact": exhaustive.exact,
            "lazy_set_count": len(lazy.route_sets),
            "lazy_complete": lazy.complete,
            "lazy_status": lazy.status,
            "lazy_reason": lazy.reason,
            "lazy_solves": lazy.solves,
            "lazy_no_good_cuts": lazy.no_good_cuts,
            "lazy_model_builds": lazy.telemetry["model_builds"],
            "sets_equal": lazy.route_sets == exhaustive.feasible == expected,
            "route_sets": [list(route_set) for route_set in lazy.route_sets],
            "telemetry": dict(lazy.telemetry),
        },
        "big_m": {
            "fast_selection_binaries": fast.selection_binaries,
            "fast_values": {str(k): v for k, v in fast.big_m.items()},
            "exact_selection_binaries": exact.selection_binaries,
            "exact_values": {str(k): v for k, v in exact.big_m.items()},
            "exact_support_complete": exact.big_m_support_exact,
            "exact_upper_status": list(exact.big_m_upper_status),
        },
        "interpretation": {
            "enumeration": "correctness differential only; not an E-scaling result",
            "big_m": "constraint-aware support can remove binaries; timing rerun remains pending",
            "warm_start": "model reuse and basis submission are measured; solver-internal basis use is not claimed"
        }
    }
    _write_json(output_dir / "raw.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(result["enumeration"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
