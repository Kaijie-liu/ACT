"""Exact negative control for fixed-multiplier guard compilation."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from act.back_end.core import Bounds
from act.back_end.moe.hz_routing import guarded_input_domain
from act.back_end.moe.tie_safe_implication import LagrangianTop1GuardedProperty
from act.back_end.solver.solver_hz import (
    hz_support_bounds,
    sparse_hz_from_bounds,
    sparse_hz_linear,
)


WORKSPACE = Path("/data1/Kane/MOE")


def fixed_multiplier_exact_minimum(multiplier: float) -> float:
    """Return min over [-1,1] of 0.1-2 ReLU(-x)-mu*x."""

    value = float(multiplier)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("multiplier must be finite and nonnegative")
    negative_half_minimum = value - 1.9 if value <= 2.0 else 0.1
    positive_half_minimum = 0.1 - value
    return min(negative_half_minimum, positive_half_minimum)


def _support(hz) -> dict[str, Any]:
    result = hz_support_bounds(hz, [0], time_limit=5.0, relax_binaries=False)
    return {
        "lower": float(result.bounds.lb.item()),
        "upper": float(result.bounds.ub.item()),
        "exact": bool(result.exact),
        "lower_status": result.lower_status[0],
        "upper_status": result.upper_status[0],
    }


def _models() -> tuple[nn.Module, nn.Module]:
    router = nn.Linear(1, 2, dtype=torch.float64)
    expert = nn.Sequential(
        nn.Linear(1, 1, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(1, 1, dtype=torch.float64),
    )
    with torch.no_grad():
        router.weight.copy_(torch.tensor([[1.0], [0.0]], dtype=torch.float64))
        router.bias.zero_()
        expert[0].weight.fill_(-1.0)
        expert[0].bias.zero_()
        expert[2].weight.fill_(-2.0)
        expert[2].bias.fill_(0.1)
    return router, expert


def run() -> dict[str, Any]:
    domain = sparse_hz_from_bounds(
        Bounds(
            torch.tensor([[-1.0]], dtype=torch.float64),
            torch.tensor([[1.0]], dtype=torch.float64),
        ),
        frame_id=2026090602,
    )
    router_hz = sparse_hz_linear(
        domain,
        np.asarray([[1.0], [0.0]], dtype=np.float64),
        np.asarray([0.0, 0.0], dtype=np.float64),
    )
    guarded = guarded_input_domain(domain, router_hz, expert=0, top_k=1)
    guarded_negative_x = sparse_hz_linear(
        guarded.hz,
        np.asarray([[-1.0]], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
    )
    negative_x_support = _support(guarded_negative_x)
    # On x>=0, ReLU(-x)=0. Numerical support includes an explicit conservative
    # padding, so retain it in the derived lower bound rather than rounding it
    # silently to zero.
    retained_hz_safety_lower = 0.1 - 2.0 * max(
        0.0, negative_x_support["upper"]
    )

    router, expert = _models()
    multiplier_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 4.0]
    grid = torch.linspace(-1.0, 1.0, 20001, dtype=torch.float64).unsqueeze(1)
    compiler_rows = []
    for multiplier in multiplier_grid:
        compiler = LagrangianTop1GuardedProperty(
            router,
            expert,
            expert_index=0,
            property_matrix=[[1.0]],
            property_offset=[0.0],
            multipliers=[[multiplier]],
        )
        with torch.no_grad():
            _margins, _safety, compiled = compiler.forward_components(grid)
        observed = float(compiled.min().item())
        expected = fixed_multiplier_exact_minimum(multiplier)
        compiler_rows.append(
            {
                "multiplier": multiplier,
                "analytic_full_box_minimum": expected,
                "concrete_grid_minimum": observed,
                "absolute_error": abs(observed - expected),
                "status": "UNKNOWN_REDUCTION_GAP",
            }
        )

    optimum_multiplier = 1.0
    optimum_lower = fixed_multiplier_exact_minimum(optimum_multiplier)
    tolerance = 5e-9
    passed = bool(
        guarded.hz.exact
        and guarded.selection_binaries == 0
        and negative_x_support["exact"]
        and retained_hz_safety_lower >= 1e-7
        and math.isclose(optimum_lower, -0.9, rel_tol=0.0, abs_tol=tolerance)
        and all(row["absolute_error"] <= tolerance for row in compiler_rows)
        and all(row["analytic_full_box_minimum"] < 0.0 for row in compiler_rows)
    )
    return {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "scope": "FIXED_MULTIPLIER_LAGRANGIAN_REDUCTION_GAP_CONTROL",
        "problem": {
            "domain": "x in [-1,1]",
            "router": ["r0=x", "r1=0"],
            "legal_branch": "route 0 iff x>=0 under tie-inclusive semantics",
            "safety": "s(x)=0.1-2*ReLU(-x)",
            "guarded_exact_minimum": 0.1,
        },
        "retained_hz": {
            "selection_binaries": int(guarded.selection_binaries),
            "guarded_domain_exact": bool(guarded.hz.exact),
            "negative_x_support": negative_x_support,
            "conservative_safety_lower": retained_hz_safety_lower,
            "status": (
                "CERTIFIED_POSITIVE"
                if retained_hz_safety_lower >= 1e-7
                else "UNKNOWN"
            ),
        },
        "fixed_multiplier_compiler": {
            "exact_optimum_multiplier": optimum_multiplier,
            "supremum_of_exact_full_box_minimum": optimum_lower,
            "proof": (
                "For mu<=2 the endpoint minima are mu-1.9 at x=-1 and "
                "0.1-mu at x=1; their maximum intersection is mu=1 with "
                "value -0.9. For mu>2, 0.1-mu is smaller."
            ),
            "grid_controls": compiler_rows,
            "status": "UNKNOWN_INTRINSIC_SUFFICIENT_REDUCTION_GAP",
        },
        "comparison_tolerance": tolerance,
        "evidence_boundary": (
            "This exact one-dimensional control isolates incompleteness of the "
            "fixed nonnegative multiplier sufficient reduction. It contains no "
            "CROWN relaxation and does not estimate official-scale prevalence."
        ),
    }


def _write(path: Path, result: dict[str, Any]) -> None:
    path = path.resolve()
    path.relative_to(WORKSPACE)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run()
    _write(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
