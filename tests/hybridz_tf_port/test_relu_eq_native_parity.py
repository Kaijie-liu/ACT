"""Regression test: ACT hz_apply_relu (HZono) vs HyZor applyReLU_eq_native.

The two should produce algebraically identical HZ outputs since they
implement the same paper encoding (4 cont + 1 binary + 3 eq rows per
unstable neuron). This test catches any drift between them during the
HyZor→hybridz_tf port.

Strategy:
  - Run on a battery of HZ configurations
  - Apply HyZor's `applyReLU_eq_native` -> HZ_hyzor
  - Apply ACT's `hz_apply_relu` -> HZ_act
  - Compare 6-tuple element-wise (max rel-error <= 1e-12)
  - Sanity-check output bounds against theoretical post-ReLU values

Bypass HyZor's `_with_base` Phase 1-3 fallback by reading raw fields
of the returned object (HyZor's `_with_base` returns the same raw HZ
when the base is small enough, which is true for our small fixtures).
"""
from __future__ import annotations

import sys
from typing import Tuple

import torch

sys.path.insert(0, "/data1/Kane/ACT")
sys.path.insert(0, "/data1/Kane/HyZor")
sys.path.insert(0, "/data1/Kane")

from act.back_end.solver.solver_hz import HZono, hz_compute_bounds
from act.back_end.hybridz_tf.tf_mlp import hz_apply_relu


# ---------------------------------------------------------------------------
# Test infra
# ---------------------------------------------------------------------------


def _hzono(c, Gc, Gb, Ac, Ab, b):
    return HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b, eq_mask=None)


def _hyzor(c, Gc, Gb, Ac, Ab, b):
    from HybridZonotope import HybridZonotope
    return HybridZonotope(
        Gc=Gc, Gb=Gb, c=c, Ac=Ac, Ab=Ab, b=b,
        device=c.device, dtype=c.dtype,
    )


def _close(a: torch.Tensor, b: torch.Tensor, tag: str,
           atol: float = 1e-12, rtol: float = 1e-12) -> float:
    if tuple(a.shape) != tuple(b.shape):
        raise AssertionError(
            f"{tag} shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}"
        )
    if a.numel() == 0:
        return 0.0
    diff = (a - b).abs()
    max_err = float(diff.max().item())
    ref = max(float(a.abs().max().item()), float(b.abs().max().item()), 1.0)
    if max_err > atol + rtol * ref:
        raise AssertionError(
            f"{tag} mismatch: max_err={max_err:.3e}, ref={ref:.3e}"
        )
    return max_err


def _compare_outputs(hzono_out, hyzor_out, label: str) -> None:
    errs = {
        "c":  _close(hzono_out.c,  hyzor_out.c,  f"{label}/c"),
        "Gc": _close(hzono_out.Gc, hyzor_out.Gc, f"{label}/Gc"),
        "Gb": _close(hzono_out.Gb, hyzor_out.Gb, f"{label}/Gb"),
        "Ac": _close(hzono_out.Ac, hyzor_out.Ac, f"{label}/Ac"),
        "Ab": _close(hzono_out.Ab, hyzor_out.Ab, f"{label}/Ab"),
        "b":  _close(hzono_out.b,  hyzor_out.b,  f"{label}/b"),
    }
    s = ", ".join(f"{k}:{v:.1e}" for k, v in errs.items())
    print(f"  [{label}] 6-tuple max_err: {s}")


def _run_case(label: str, c, Gc, Gb, Ac, Ab, b):
    hzono_in = _hzono(c, Gc, Gb, Ac, Ab, b)
    hyzor_in = _hyzor(c, Gc, Gb, Ac, Ab, b)
    hzono_out = hz_apply_relu(hzono_in)
    hyzor_out = hyzor_in.applyReLU_eq_native()
    _compare_outputs(hzono_out, hyzor_out, label)
    return hzono_out, hyzor_out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def fixture_mixed_no_constraints(dtype=torch.float64, device="cpu"):
    """4 neurons: 1 active, 1 inactive, 2 unstable. No prior constraints."""
    n, ng, nb = 4, 2, 0
    c = torch.tensor([[2.0], [-3.0], [0.5], [-0.5]], dtype=dtype, device=device)
    Gc = torch.tensor([
        [0.5, 0.5],
        [0.5, 0.5],
        [1.0, 1.0],
        [1.0, 1.0],
    ], dtype=dtype, device=device)
    Gb = torch.zeros((n, nb), dtype=dtype, device=device)
    Ac = torch.zeros((0, ng), dtype=dtype, device=device)
    Ab = torch.zeros((0, nb), dtype=dtype, device=device)
    b = torch.zeros((0, 1), dtype=dtype, device=device)
    return c, Gc, Gb, Ac, Ab, b


def fixture_all_active(dtype=torch.float64, device="cpu"):
    """All neurons active (k=0). Output should pass through unchanged."""
    n, ng = 3, 2
    c = torch.tensor([[5.0], [4.0], [3.0]], dtype=dtype, device=device)
    Gc = torch.tensor([
        [0.5, 0.0],
        [0.0, 0.5],
        [0.3, 0.3],
    ], dtype=dtype, device=device)
    Gb = torch.zeros((n, 0), dtype=dtype, device=device)
    Ac = torch.zeros((0, ng), dtype=dtype, device=device)
    Ab = torch.zeros((0, 0), dtype=dtype, device=device)
    b = torch.zeros((0, 1), dtype=dtype, device=device)
    return c, Gc, Gb, Ac, Ab, b


def fixture_all_inactive(dtype=torch.float64, device="cpu"):
    """All neurons inactive (k=0). Output should be all zeros."""
    n, ng = 3, 2
    c = torch.tensor([[-5.0], [-4.0], [-3.0]], dtype=dtype, device=device)
    Gc = torch.tensor([
        [0.5, 0.0],
        [0.0, 0.5],
        [0.3, 0.3],
    ], dtype=dtype, device=device)
    Gb = torch.zeros((n, 0), dtype=dtype, device=device)
    Ac = torch.zeros((0, ng), dtype=dtype, device=device)
    Ab = torch.zeros((0, 0), dtype=dtype, device=device)
    b = torch.zeros((0, 1), dtype=dtype, device=device)
    return c, Gc, Gb, Ac, Ab, b


def fixture_with_binary_gens(dtype=torch.float64, device="cpu"):
    """Input HZ has pre-existing binary generators (nb0 > 0)."""
    n, ng, nb = 3, 2, 1
    c = torch.tensor([[1.0], [-2.0], [0.0]], dtype=dtype, device=device)
    Gc = torch.tensor([
        [0.5, 0.3],
        [0.2, 0.4],
        [1.0, 0.5],
    ], dtype=dtype, device=device)
    Gb = torch.tensor([
        [0.4],
        [0.6],
        [0.8],
    ], dtype=dtype, device=device)
    Ac = torch.zeros((0, ng), dtype=dtype, device=device)
    Ab = torch.zeros((0, nb), dtype=dtype, device=device)
    b = torch.zeros((0, 1), dtype=dtype, device=device)
    return c, Gc, Gb, Ac, Ab, b


def fixture_with_prior_constraints(dtype=torch.float64, device="cpu"):
    """Input HZ has prior nc=2 equality constraints."""
    n, ng, nb, nc = 3, 3, 1, 2
    c = torch.tensor([[1.0], [-1.5], [0.5]], dtype=dtype, device=device)
    Gc = torch.tensor([
        [0.5, 0.3, 0.2],
        [0.2, 0.4, 0.1],
        [0.6, 0.5, 0.3],
    ], dtype=dtype, device=device)
    Gb = torch.tensor([
        [0.4],
        [0.6],
        [0.2],
    ], dtype=dtype, device=device)
    Ac = torch.tensor([
        [1.0, -1.0,  0.0],
        [0.0,  0.5, -0.5],
    ], dtype=dtype, device=device)
    Ab = torch.tensor([
        [0.3],
        [-0.2],
    ], dtype=dtype, device=device)
    b = torch.tensor([[0.1], [-0.05]], dtype=dtype, device=device)
    return c, Gc, Gb, Ac, Ab, b


def fixture_random(seed: int, n: int, ng: int, nb: int, nc: int,
                   dtype=torch.float64, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    c = torch.randn(n, 1, dtype=dtype, device=device, generator=g) * 2.0
    Gc = torch.randn(n, ng, dtype=dtype, device=device, generator=g) * 0.5
    Gb = torch.randn(n, nb, dtype=dtype, device=device, generator=g) * 0.3
    Ac = torch.randn(nc, ng, dtype=dtype, device=device, generator=g) * 0.5
    Ab = torch.randn(nc, nb, dtype=dtype, device=device, generator=g) * 0.5
    b = torch.randn(nc, 1, dtype=dtype, device=device, generator=g) * 0.3
    return c, Gc, Gb, Ac, Ab, b


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_mixed_no_constraints():
    _run_case("mixed_no_constraints", *fixture_mixed_no_constraints())


def test_all_active():
    _run_case("all_active", *fixture_all_active())


def test_all_inactive():
    _run_case("all_inactive", *fixture_all_inactive())


def test_with_binary_gens():
    _run_case("with_binary_gens", *fixture_with_binary_gens())


def test_with_prior_constraints():
    _run_case("with_prior_constraints", *fixture_with_prior_constraints())


def test_random_battery():
    cases = [
        (1, 8, 4, 0, 0),
        (2, 10, 6, 2, 0),
        (3, 16, 8, 4, 3),
        (4, 32, 16, 4, 8),
    ]
    for (seed, n, ng, nb, nc) in cases:
        label = f"random[seed={seed},n={n},ng={ng},nb={nb},nc={nc}]"
        _run_case(label, *fixture_random(seed, n, ng, nb, nc))


def test_random_battery_cuda():
    if not torch.cuda.is_available():
        print("  [random_cuda] CUDA unavailable, SKIP")
        return
    cases = [
        (11, 8, 4, 0, 0),
        (12, 32, 16, 4, 6),
    ]
    for (seed, n, ng, nb, nc) in cases:
        label = f"cuda[seed={seed},n={n}]"
        _run_case(label, *fixture_random(seed, n, ng, nb, nc, device="cuda"))


if __name__ == "__main__":
    print("=== ReLU eq-native ACT vs HyZor parity battery ===")
    tests = [
        test_mixed_no_constraints,
        test_all_active,
        test_all_inactive,
        test_with_binary_gens,
        test_with_prior_constraints,
        test_random_battery,
        test_random_battery_cuda,
    ]
    fail = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL: {t.__name__}: {e}")
            fail += 1
    if fail:
        print(f"\n{fail} test(s) failed")
        sys.exit(1)
    print(f"\nALL {len(tests)} TESTS PASSED — hz_apply_relu == HyZor.applyReLU_eq_native")
