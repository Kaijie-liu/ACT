"""Parity test: ACT hz_apply_relu_triangle vs HyZor applyReLU_triangle.

Both implement DeepZ-style chord relaxation: per unstable neuron i with
bounds [l, u] (l<0<u),

    λ = u/(u-l),  μ = -lu/(2(u-l))
    y_i = λ x_i + μ + μ ε_new_i

No equality rows; binary slot unchanged. Pure 6-tuple element-wise
comparison (HyZor's _with_base does no transformation when base is
small, so we expect EXACT match).
"""
from __future__ import annotations
import sys
import torch

sys.path.insert(0, "/data1/Kane/ACT")
sys.path.insert(0, "/data1/Kane/HyZor")
sys.path.insert(0, "/data1/Kane")

from act.back_end.solver.solver_hz import HZono
from act.back_end.hybridz_tf.algorithms.relu_methods import hz_apply_relu_triangle


def _hzono(c, Gc, Gb, Ac, Ab, b, eq_mask=None):
    return HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b, eq_mask=eq_mask)


def _hyzor(c, Gc, Gb, Ac, Ab, b, eq_mask=None):
    from HybridZonotope import HybridZonotope
    if eq_mask is None and Ac.shape[0] > 0:
        eq_mask = torch.ones(Ac.shape[0], dtype=torch.bool, device=c.device)
    return HybridZonotope(
        Gc=Gc, Gb=Gb, c=c, Ac=Ac, Ab=Ab, b=b,
        device=c.device, dtype=c.dtype, eq_mask=eq_mask,
    )


def _close(a, b, tag, atol=1e-12):
    if tuple(a.shape) != tuple(b.shape):
        raise AssertionError(f"{tag} shape: {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.numel() == 0:
        return 0.0
    diff = (a - b).abs()
    me = float(diff.max().item())
    ref = max(float(a.abs().max().item()), float(b.abs().max().item()), 1.0)
    if me > atol * max(1.0, ref):
        raise AssertionError(f"{tag}: max_err={me:.3e}")
    return me


def _run_case(label, c, Gc, Gb, Ac, Ab, b, eq_mask=None):
    print(f"\n[{label}]")
    hzono_in = _hzono(c, Gc, Gb, Ac, Ab, b, eq_mask=eq_mask)
    hyzor_in = _hyzor(c, Gc, Gb, Ac, Ab, b, eq_mask=eq_mask)
    hzono_out = hz_apply_relu_triangle(hzono_in)
    hyzor_out = hyzor_in.applyReLU_triangle()
    errs = {
        "c": _close(hzono_out.c, hyzor_out.c, f"{label}/c"),
        "Gc": _close(hzono_out.Gc, hyzor_out.Gc, f"{label}/Gc"),
        "Gb": _close(hzono_out.Gb, hyzor_out.Gb, f"{label}/Gb"),
        "Ac": _close(hzono_out.Ac, hyzor_out.Ac, f"{label}/Ac"),
        "Ab": _close(hzono_out.Ab, hyzor_out.Ab, f"{label}/Ab"),
        "b": _close(hzono_out.b, hyzor_out.b, f"{label}/b"),
    }
    s = ", ".join(f"{k}:{v:.1e}" for k, v in errs.items())
    print(f"  6-tuple max_err: {s}")


def test_mixed_unstable():
    n, ng = 4, 2
    c = torch.tensor([[2.0], [-3.0], [0.5], [-0.5]], dtype=torch.float64)
    Gc = torch.tensor([
        [0.5, 0.5], [0.5, 0.5], [1.0, 1.0], [1.0, 1.0],
    ], dtype=torch.float64)
    Gb = torch.zeros((n, 0), dtype=torch.float64)
    Ac = torch.zeros((0, ng), dtype=torch.float64)
    Ab = torch.zeros((0, 0), dtype=torch.float64)
    b = torch.zeros((0, 1), dtype=torch.float64)
    _run_case("mixed_unstable", c, Gc, Gb, Ac, Ab, b)


def test_all_active():
    n, ng = 3, 2
    c = torch.tensor([[5.0], [4.0], [3.0]], dtype=torch.float64)
    Gc = torch.tensor([[0.5, 0.0], [0.0, 0.5], [0.3, 0.3]], dtype=torch.float64)
    Gb = torch.zeros((n, 0), dtype=torch.float64)
    Ac = torch.zeros((0, ng), dtype=torch.float64)
    Ab = torch.zeros((0, 0), dtype=torch.float64)
    b = torch.zeros((0, 1), dtype=torch.float64)
    _run_case("all_active", c, Gc, Gb, Ac, Ab, b)


def test_all_inactive():
    n, ng = 3, 2
    c = torch.tensor([[-5.0], [-4.0], [-3.0]], dtype=torch.float64)
    Gc = torch.tensor([[0.5, 0.0], [0.0, 0.5], [0.3, 0.3]], dtype=torch.float64)
    Gb = torch.zeros((n, 0), dtype=torch.float64)
    Ac = torch.zeros((0, ng), dtype=torch.float64)
    Ab = torch.zeros((0, 0), dtype=torch.float64)
    b = torch.zeros((0, 1), dtype=torch.float64)
    _run_case("all_inactive", c, Gc, Gb, Ac, Ab, b)


def test_with_binary_gens():
    n, ng, nb = 3, 2, 1
    c = torch.tensor([[1.0], [-2.0], [0.0]], dtype=torch.float64)
    Gc = torch.tensor([[0.5, 0.3], [0.2, 0.4], [1.0, 0.5]], dtype=torch.float64)
    Gb = torch.tensor([[0.4], [0.6], [0.8]], dtype=torch.float64)
    Ac = torch.zeros((0, ng), dtype=torch.float64)
    Ab = torch.zeros((0, nb), dtype=torch.float64)
    b = torch.zeros((0, 1), dtype=torch.float64)
    _run_case("with_binary_gens", c, Gc, Gb, Ac, Ab, b)


def test_with_prior_constraints():
    n, ng, nb = 3, 3, 1
    c = torch.tensor([[1.0], [-1.5], [0.5]], dtype=torch.float64)
    Gc = torch.tensor([
        [0.5, 0.3, 0.2], [0.2, 0.4, 0.1], [0.6, 0.5, 0.3],
    ], dtype=torch.float64)
    Gb = torch.tensor([[0.4], [0.6], [0.2]], dtype=torch.float64)
    Ac = torch.tensor([
        [1.0, -1.0, 0.0], [0.0, 0.5, -0.5],
    ], dtype=torch.float64)
    Ab = torch.tensor([[0.3], [-0.2]], dtype=torch.float64)
    b = torch.tensor([[0.1], [-0.05]], dtype=torch.float64)
    _run_case("with_prior_constraints", c, Gc, Gb, Ac, Ab, b,
              eq_mask=torch.tensor([True, True]))


def test_random_battery():
    cases = [(1, 8, 4, 0, 0), (2, 16, 8, 4, 3), (3, 32, 16, 4, 8)]
    for seed, n, ng, nb, nc in cases:
        g = torch.Generator().manual_seed(seed)
        c = torch.randn(n, 1, dtype=torch.float64, generator=g) * 2.0
        Gc = torch.randn(n, ng, dtype=torch.float64, generator=g) * 0.5
        Gb = torch.randn(n, nb, dtype=torch.float64, generator=g) * 0.3
        Ac = torch.randn(nc, ng, dtype=torch.float64, generator=g) * 0.5
        Ab = torch.randn(nc, nb, dtype=torch.float64, generator=g) * 0.5
        b = torch.randn(nc, 1, dtype=torch.float64, generator=g) * 0.3
        em = torch.ones(nc, dtype=torch.bool) if nc > 0 else None
        _run_case(f"random[seed={seed},n={n},ng={ng},nb={nb},nc={nc}]",
                  c, Gc, Gb, Ac, Ab, b, eq_mask=em)


if __name__ == "__main__":
    print("=== applyReLU_triangle ACT vs HyZor parity ===")
    tests = [
        test_mixed_unstable, test_all_active, test_all_inactive,
        test_with_binary_gens, test_with_prior_constraints,
        test_random_battery,
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
    print(f"\nALL {len(tests)} tests PASSED — hz_apply_relu_triangle == HyZor.applyReLU_triangle")
