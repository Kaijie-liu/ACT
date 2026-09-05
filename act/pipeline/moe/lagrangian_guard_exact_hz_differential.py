"""Exact retained-HZ differential for the hard-top1 Lagrangian compiler."""

from __future__ import annotations

import argparse
import json
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


def _support(hz) -> dict[str, Any]:
    result = hz_support_bounds(
        hz,
        [0],
        time_limit=5.0,
        relax_binaries=False,
    )
    return {
        "lower": float(result.bounds.lb.item()),
        "upper": float(result.bounds.ub.item()),
        "exact": bool(result.exact),
        "lower_status": result.lower_status[0],
        "upper_status": result.upper_status[0],
    }


def _case(name: str, safety_bias: float) -> dict[str, Any]:
    domain = sparse_hz_from_bounds(
        Bounds(
            torch.tensor([[-1.0]], dtype=torch.float64),
            torch.tensor([[1.0]], dtype=torch.float64),
        ),
        frame_id=20260906,
    )
    router_hz = sparse_hz_linear(
        domain,
        np.asarray([[1.0], [0.0]], dtype=np.float64),
        np.asarray([0.0, 0.0], dtype=np.float64),
    )
    guarded = guarded_input_domain(domain, router_hz, expert=0, top_k=1)
    unguarded_safety = sparse_hz_linear(
        domain,
        np.asarray([[1.0]], dtype=np.float64),
        np.asarray([safety_bias], dtype=np.float64),
    )
    guarded_safety = sparse_hz_linear(
        guarded.hz,
        np.asarray([[1.0]], dtype=np.float64),
        np.asarray([safety_bias], dtype=np.float64),
    )
    # For s=x+b, selected margin m=r_0-r_1=x, and mu=1, phi=s-m=b.
    compiled_hz = sparse_hz_linear(
        domain,
        np.asarray([[0.0]], dtype=np.float64),
        np.asarray([safety_bias], dtype=np.float64),
    )
    unguarded_support = _support(unguarded_safety)
    guarded_support = _support(guarded_safety)
    compiled_support = _support(compiled_hz)

    router = nn.Linear(1, 2, dtype=torch.float64)
    expert = nn.Linear(1, 1, dtype=torch.float64)
    with torch.no_grad():
        router.weight.copy_(torch.tensor([[1.0], [0.0]], dtype=torch.float64))
        router.bias.zero_()
        expert.weight.fill_(1.0)
        expert.bias.fill_(safety_bias)
    compiler = LagrangianTop1GuardedProperty(
        router,
        expert,
        0,
        [[1.0]],
        [0.0],
        [[1.0]],
    )
    grid = torch.linspace(-1.0, 1.0, 1001, dtype=torch.float64).unsqueeze(1)
    with torch.no_grad():
        margins, safety, compiled = compiler.forward_components(grid)
    legal = torch.all(margins >= 0.0, dim=1)
    maximum_formula_error = float((compiled - safety_bias).abs().max().item())
    maximum_legal_domination_error = float(
        torch.clamp(compiled[legal] - safety[legal], min=0.0).max().item()
    )
    # Exact support is solver-complete, while its serialized conservative
    # bounds include roughly 1.1e-9 of numerical padding on this control.  The
    # differential tolerance is explicit and remains far below the verifier's
    # registered 1e-7 positive-margin threshold.
    tolerance = 5e-9
    passed = bool(
        guarded.selection_binaries == 0
        and guarded.hz.exact
        and unguarded_support["exact"]
        and guarded_support["exact"]
        and compiled_support["exact"]
        and abs(guarded_support["lower"] - safety_bias) <= tolerance
        and abs(compiled_support["lower"] - safety_bias) <= tolerance
        and abs(guarded_support["lower"] - compiled_support["lower"]) <= tolerance
        and maximum_formula_error <= tolerance
        and maximum_legal_domination_error <= tolerance
    )
    return {
        "name": name,
        "safety_bias": float(safety_bias),
        "guard": "r_0-r_1=x>=0",
        "selection_binaries": int(guarded.selection_binaries),
        "guarded_hz_exact": bool(guarded.hz.exact),
        "unguarded_safety_support": unguarded_support,
        "retained_guard_safety_support": guarded_support,
        "compiled_phi_support": compiled_support,
        "maximum_formula_error": maximum_formula_error,
        "maximum_legal_domination_error": maximum_legal_domination_error,
        "comparison_tolerance": tolerance,
        "passed": passed,
    }


def run() -> dict[str, Any]:
    cases = [
        _case("safe_guarded_half_interval", 0.1),
        _case("unsafe_tie", -0.1),
    ]
    return {
        "schema_version": 1,
        "status": "PASS" if all(case["passed"] for case in cases) else "FAIL",
        "scope": "EXACT_RETAINED_HZ_VS_LAGRANGIAN_TOP1_TOY_DIFFERENTIAL",
        "cases": cases,
        "evidence_boundary": (
            "Exact one-dimensional linear differential only. It validates the "
            "compiler against retained-guard HZ on two analytic controls; it "
            "does not establish official-scale effect or CROWN formal soundness."
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
