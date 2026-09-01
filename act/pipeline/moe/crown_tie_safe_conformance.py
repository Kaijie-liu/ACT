"""Run a pinned alpha-beta-CROWN conformance check for the tie-safe compiler.

This is an adapter check, not a neural-network robustness experiment. It uses
constant affine routers and experts so the exact range of every compiled
scalar is known before CROWN is called.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from act.util.typing_compat import install_typing_override

install_typing_override()

import torch
from torch import nn

from act.back_end.moe.tie_safe_implication import TieSafeTop1Implication


DEFAULT_ETA = 1e-7


@dataclass(frozen=True)
class CaseResult:
    name: str
    guard: float
    safety: float
    concrete_compiled: float
    exact_compiled: float
    crown_lower: float
    crown_upper: float
    expected_nonnegative: bool
    crown_nonnegative: bool
    passed: bool


def _constant_affine(in_features: int, outputs: list[float], device: str) -> nn.Linear:
    layer = nn.Linear(in_features, len(outputs), device=device)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.copy_(torch.tensor(outputs, dtype=layer.bias.dtype, device=device))
    return layer


def _run_case(
    *,
    name: str,
    competitor_advantage: float,
    safety: float,
    eta: float,
    expected_nonnegative: bool,
    device: str,
) -> CaseResult:
    try:
        from auto_LiRPA import BoundedModule, BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm
    except ImportError as exc:  # pragma: no cover - environment gate
        raise RuntimeError("auto_LiRPA is required for this conformance run") from exc

    router = _constant_affine(1, [0.0, competitor_advantage], device)
    expert = _constant_affine(1, [safety], device)
    model = TieSafeTop1Implication(
        router,
        expert,
        0,
        torch.tensor([[1.0]], device=device),
        torch.tensor([0.0], device=device),
        eta=eta,
    ).eval()
    center = torch.zeros((1, 1), device=device)
    lower = torch.full_like(center, -1.0)
    upper = torch.full_like(center, 1.0)
    bounded_model = BoundedModule(model, center, device=device)
    bounded_input = BoundedTensor(
        center,
        PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper),
    )
    crown_lower, crown_upper = bounded_model.compute_bounds(
        x=(bounded_input,), method="CROWN"
    )
    guard, safety_value, concrete = model.forward_components(center)
    exact = max(competitor_advantage - eta, safety)
    lower_value = float(crown_lower.item())
    upper_value = float(crown_upper.item())
    concrete_value = float(concrete.item())
    tolerance = 2e-7
    exact_match = (
        abs(concrete_value - exact) <= tolerance
        and abs(lower_value - exact) <= tolerance
        and abs(upper_value - exact) <= tolerance
    )
    crown_nonnegative = lower_value >= 0.0
    return CaseResult(
        name=name,
        guard=float(guard.item()),
        safety=float(safety_value.item()),
        concrete_compiled=concrete_value,
        exact_compiled=float(exact),
        crown_lower=lower_value,
        crown_upper=upper_value,
        expected_nonnegative=expected_nonnegative,
        crown_nonnegative=crown_nonnegative,
        passed=bool(exact_match and crown_nonnegative == expected_nonnegative),
    )


def run_conformance(*, device: str, eta: float = DEFAULT_ETA) -> dict[str, object]:
    cases = [
        _run_case(name="legal_tie_safe", competitor_advantage=0.0, safety=1.0,
                  eta=eta, expected_nonnegative=True, device=device),
        _run_case(name="legal_tie_unsafe", competitor_advantage=0.0, safety=-1.0,
                  eta=eta, expected_nonnegative=False, device=device),
        _run_case(name="eta_overcheck_band", competitor_advantage=eta / 2.0,
                  safety=-1.0, eta=eta, expected_nonnegative=False, device=device),
        _run_case(name="strictly_outside_eta_band", competitor_advantage=2.0 * eta,
                  safety=-1.0, eta=eta, expected_nonnegative=True, device=device),
    ]
    naive_tie_value = max(0.0, -1.0)
    return {
        "schema_version": 1,
        "scope": "toy_adapter_conformance_only",
        "backend": "alpha-beta-CROWN/auto_LiRPA",
        "device": device,
        "eta": eta,
        "eta_policy": "safe_positive_margin",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "auto_lirpa": importlib.metadata.version("auto_LiRPA"),
        "python311_typing_override_shim": sys.version_info < (3, 12),
        "naive_tie_counterexample": {
            "guard": 0.0,
            "unsafe_safety": -1.0,
            "max_guard_safety": naive_tie_value,
            "incorrectly_nonnegative": naive_tie_value >= 0.0,
        },
        "cases": [asdict(case) for case in cases],
        "all_passed": all(case.passed for case in cases),
        "claim_limit": (
            "Checks graph lowering and tie semantics on analytically constant "
            "cases; it is not an official-model certificate or a numerical "
            "soundness validation of CROWN lower bounds."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    result = run_conformance(device=args.device, eta=args.eta)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
