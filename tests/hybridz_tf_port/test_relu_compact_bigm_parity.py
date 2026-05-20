"""Parity tests: ACT hz_apply_relu_compact / hz_apply_relu_bigM_fast
vs HyZor counterparts.

Both methods are sound over-approximations.
- compact: +k cont, +k bin, +2k sparse ineq
- bigM_fast: +k cont (eta), +k bin (z), +3k ineq

We compare 6-tuple element-wise (HyZor's _with_base is no-op for small
HZ, so direct match is expected).
"""
from __future__ import annotations
import sys
import torch

sys.path.insert(0, "/data1/Kane/ACT")
sys.path.insert(0, "/data1/Kane/HyZor")
sys.path.insert(0, "/data1/Kane")

from act.back_end.solver.solver_hz import HZono
from act.back_end.hybridz_tf.algorithms.relu_methods import (
    hz_apply_relu_compact,
    hz_apply_relu_bigM_fast,
)


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


def _run_pair(label, c, Gc, Gb, Ac, Ab, b, *, kind, eq_mask=None):
    print(f"\n[{label}] kind={kind}")
    hzono_in = _hzono(c, Gc, Gb, Ac, Ab, b, eq_mask=eq_mask)
    hyzor_in = _hyzor(c, Gc, Gb, Ac, Ab, b, eq_mask=eq_mask)
    if kind == "compact":
        hzono_out = hz_apply_relu_compact(hzono_in)
        hyzor_out = hyzor_in.applyReLU_compact()
    elif kind == "bigM":
        hzono_out = hz_apply_relu_bigM_fast(hzono_in)
        hyzor_out = hyzor_in.applyReLU_bigM_fast()
    else:
        raise ValueError(kind)
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


def _fixtures():
    """Shared fixtures to run with both compact and bigM."""
    out = {}

    out["mixed_unstable"] = dict(
        c=torch.tensor([[2.0], [-3.0], [0.5], [-0.5]], dtype=torch.float64),
        Gc=torch.tensor([
            [0.5, 0.5], [0.5, 0.5], [1.0, 1.0], [1.0, 1.0],
        ], dtype=torch.float64),
        Gb=torch.zeros((4, 0), dtype=torch.float64),
        Ac=torch.zeros((0, 2), dtype=torch.float64),
        Ab=torch.zeros((0, 0), dtype=torch.float64),
        b=torch.zeros((0, 1), dtype=torch.float64),
        eq_mask=None,
    )
    out["all_active"] = dict(
        c=torch.tensor([[5.0], [4.0], [3.0]], dtype=torch.float64),
        Gc=torch.tensor([[0.5, 0.0], [0.0, 0.5], [0.3, 0.3]], dtype=torch.float64),
        Gb=torch.zeros((3, 0), dtype=torch.float64),
        Ac=torch.zeros((0, 2), dtype=torch.float64),
        Ab=torch.zeros((0, 0), dtype=torch.float64),
        b=torch.zeros((0, 1), dtype=torch.float64),
        eq_mask=None,
    )
    out["all_inactive"] = dict(
        c=torch.tensor([[-5.0], [-4.0], [-3.0]], dtype=torch.float64),
        Gc=torch.tensor([[0.5, 0.0], [0.0, 0.5], [0.3, 0.3]], dtype=torch.float64),
        Gb=torch.zeros((3, 0), dtype=torch.float64),
        Ac=torch.zeros((0, 2), dtype=torch.float64),
        Ab=torch.zeros((0, 0), dtype=torch.float64),
        b=torch.zeros((0, 1), dtype=torch.float64),
        eq_mask=None,
    )
    out["with_binary_gens"] = dict(
        c=torch.tensor([[1.0], [-2.0], [0.0]], dtype=torch.float64),
        Gc=torch.tensor([[0.5, 0.3], [0.2, 0.4], [1.0, 0.5]], dtype=torch.float64),
        Gb=torch.tensor([[0.4], [0.6], [0.8]], dtype=torch.float64),
        Ac=torch.zeros((0, 2), dtype=torch.float64),
        Ab=torch.zeros((0, 1), dtype=torch.float64),
        b=torch.zeros((0, 1), dtype=torch.float64),
        eq_mask=None,
    )
    out["with_prior_eq"] = dict(
        c=torch.tensor([[1.0], [-1.5], [0.5]], dtype=torch.float64),
        Gc=torch.tensor([
            [0.5, 0.3, 0.2], [0.2, 0.4, 0.1], [0.6, 0.5, 0.3],
        ], dtype=torch.float64),
        Gb=torch.tensor([[0.4], [0.6], [0.2]], dtype=torch.float64),
        Ac=torch.tensor([
            [1.0, -1.0, 0.0], [0.0, 0.5, -0.5],
        ], dtype=torch.float64),
        Ab=torch.tensor([[0.3], [-0.2]], dtype=torch.float64),
        b=torch.tensor([[0.1], [-0.05]], dtype=torch.float64),
        eq_mask=torch.tensor([True, True]),
    )

    # Random battery
    for seed, n, ng, nb, nc in [(11, 8, 4, 0, 0), (12, 16, 8, 4, 3),
                                  (13, 32, 16, 4, 8)]:
        g = torch.Generator().manual_seed(seed)
        out[f"random_s{seed}_n{n}"] = dict(
            c=torch.randn(n, 1, dtype=torch.float64, generator=g) * 2.0,
            Gc=torch.randn(n, ng, dtype=torch.float64, generator=g) * 0.5,
            Gb=torch.randn(n, nb, dtype=torch.float64, generator=g) * 0.3,
            Ac=torch.randn(nc, ng, dtype=torch.float64, generator=g) * 0.5,
            Ab=torch.randn(nc, nb, dtype=torch.float64, generator=g) * 0.5,
            b=torch.randn(nc, 1, dtype=torch.float64, generator=g) * 0.3,
            eq_mask=torch.ones(nc, dtype=torch.bool) if nc > 0 else None,
        )
    return out


if __name__ == "__main__":
    print("=== compact / bigM_fast ACT vs HyZor parity ===")
    fixtures = _fixtures()
    fail = 0
    for label, fxt in fixtures.items():
        for kind in ("compact", "bigM"):
            try:
                _run_pair(f"{label}", **fxt, kind=kind)
            except AssertionError as e:
                print(f"  FAIL: {label}/{kind}: {e}")
                fail += 1
    n_cases = len(fixtures) * 2
    if fail:
        print(f"\n{fail}/{n_cases} cases FAILED")
        sys.exit(1)
    print(f"\nALL {n_cases} tests PASSED — compact == HyZor.applyReLU_compact, bigM_fast == HyZor.applyReLU_bigM_fast")
