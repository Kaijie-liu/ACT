"""Toy CROWN conformance for the hard-top1 Lagrangian guard compiler."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch
from torch import nn

from act.back_end.moe.tie_safe_implication import LagrangianTop1GuardedProperty
from act.pipeline.moe.crown_adapter_cohort import _crown_bounds


WORKSPACE = Path("/data1/Kane/MOE")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.relative_to(WORKSPACE)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _case(name: str, safety_bias: float, expected_filter: bool) -> dict[str, Any]:
    router = nn.Linear(1, 2)
    expert = nn.Linear(1, 1)
    with torch.no_grad():
        router.weight.copy_(torch.tensor([[1.0], [0.0]]))
        router.bias.zero_()
        expert.weight.fill_(1.0)
        expert.bias.fill_(float(safety_bias))
    center = torch.zeros((1, 1))
    lower = center - 1.0
    upper = center + 1.0
    original = _crown_bounds(
        expert,
        center,
        lower,
        upper,
        property_rows=((np.asarray([1.0]), 0.0),),
        device="cpu",
        tolerance=1e-7,
        method="CROWN",
        track_gradients=False,
    )
    compiled_module = LagrangianTop1GuardedProperty(
        router,
        expert,
        0,
        [[1.0]],
        [0.0],
        [[1.0]],
    )
    compiled = _crown_bounds(
        compiled_module,
        center,
        lower,
        upper,
        property_rows=None,
        device="cpu",
        tolerance=1e-7,
        method="CROWN",
        track_gradients=False,
    )
    points = torch.linspace(-1.0, 1.0, 1001).unsqueeze(1)
    with torch.no_grad():
        margins, safety, values = compiled_module.forward_components(points)
        legal = torch.all(margins >= 0.0, dim=1)
        implication_holds = bool(torch.all(values[legal] <= safety[legal]))
        tie_margin, tie_safety, tie_value = compiled_module.forward_components(center)
    observed_filter = compiled["status"] == "CERTIFIED_MARGIN_FILTER"
    exact_compiled = float(safety_bias)
    tolerance = 2e-6
    passed = bool(
        original["error"] is None
        and compiled["error"] is None
        and implication_holds
        and observed_filter == bool(expected_filter)
        and abs(float(compiled["minimum_lower_bound"]) - exact_compiled) <= tolerance
        and abs(float(tie_margin.item())) <= tolerance
        and abs(float(tie_value.item()) - float(tie_safety.item())) <= tolerance
    )
    return {
        "name": name,
        "safety_bias": float(safety_bias),
        "expected_filter": bool(expected_filter),
        "observed_filter": observed_filter,
        "original_box_lower": original["minimum_lower_bound"],
        "compiled_lower": compiled["minimum_lower_bound"],
        "exact_compiled_lower": exact_compiled,
        "tie_margin": float(tie_margin.item()),
        "tie_safety": float(tie_safety.item()),
        "tie_compiled": float(tie_value.item()),
        "legal_grid_points": int(legal.sum().item()),
        "compiled_no_greater_than_safety_on_legal_grid": implication_holds,
        "passed": passed,
    }


def run() -> dict[str, Any]:
    cases = [
        _case("guard_recovers_safe_half_interval", 0.1, True),
        _case("unsafe_tie_remains_unresolved", -0.1, False),
    ]
    return {
        "schema_version": 1,
        "status": "PASS" if all(case["passed"] for case in cases) else "FAIL",
        "scope": "TOY_LAGRANGIAN_TOP1_GUARD_CROWN_CONFORMANCE",
        "cases": cases,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "auto_lirpa": importlib.metadata.version("auto_LiRPA"),
        },
        "evidence_boundary": (
            "Analytic shared-input toy conformance only. Positive CROWN margins "
            "are numerical filters, not outward-rounded formal certificates."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run()
    _write_json(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
