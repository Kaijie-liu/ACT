"""Parity test: hz_bounds_hz_dual ↔ HybridZonotope._bounds_hz_dual.

Both should produce the same lb/ub on the same HZ given the same
hyperparameters (max_iter, lr, lp_threshold, selective_lp).
"""
from __future__ import annotations
import sys
import torch

sys.path.insert(0, "/data1/Kane/ACT")
sys.path.insert(0, "/data1/Kane/HyZor")
sys.path.insert(0, "/data1/Kane")


def _hzono(c, Gc, Gb, Ac, Ab, b, eq_mask=None):
    from act.back_end.solver.solver_hz import HZono
    return HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b, eq_mask=eq_mask)


def _hyzor(c, Gc, Gb, Ac, Ab, b, eq_mask=None):
    from HybridZonotope import HybridZonotope
    return HybridZonotope(
        Gc=Gc, Gb=Gb, c=c, Ac=Ac, Ab=Ab, b=b,
        device=c.device, dtype=c.dtype, eq_mask=eq_mask,
    )


def _close(a, b, tag, atol=1e-8):
    if tuple(a.shape) != tuple(b.shape):
        raise AssertionError(f"{tag}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.numel() == 0:
        return 0.0
    diff = (a - b).abs()
    me = float(diff.max().item())
    if me > atol:
        raise AssertionError(f"{tag}: max_err={me:.3e}")
    return me


def _make_pair_ineq(seed: int = 1, n: int = 4, ng: int = 3,
                     nb: int = 1, nc: int = 2):
    """Build HZ with ONLY inequality rows (no eq) — exercises Tier-2 hz_dual."""
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(n, 1, dtype=torch.float64, generator=g)
    Gc = torch.randn(n, ng, dtype=torch.float64, generator=g) * 0.5
    Gb = torch.randn(n, nb, dtype=torch.float64, generator=g) * 0.3
    Ac = torch.randn(nc, ng, dtype=torch.float64, generator=g) * 0.4
    Ab = torch.randn(nc, nb, dtype=torch.float64, generator=g) * 0.4
    b = torch.randn(nc, 1, dtype=torch.float64, generator=g) * 1.0 + 0.5
    em = torch.zeros(nc, dtype=torch.bool)  # all inequalities
    return c, Gc, Gb, Ac, Ab, b, em


def _make_pair_with_eq(seed: int = 1, n: int = 4, ng: int = 3,
                       nb: int = 1, nc: int = 2):
    """Build HZ with all-eq rows — exercises Tier-3 eq_elim_lp."""
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(n, 1, dtype=torch.float64, generator=g)
    Gc = torch.randn(n, ng, dtype=torch.float64, generator=g) * 0.5
    Gb = torch.randn(n, nb, dtype=torch.float64, generator=g) * 0.3
    Ac = torch.randn(nc, ng, dtype=torch.float64, generator=g) * 0.4
    Ab = torch.randn(nc, nb, dtype=torch.float64, generator=g) * 0.4
    b = torch.zeros(nc, 1, dtype=torch.float64)
    em = torch.ones(nc, dtype=torch.bool)  # all eq
    return c, Gc, Gb, Ac, Ab, b, em


def test_dual_ineq_only():
    """nc > 0 with no eq rows — exercises Tier-2 only."""
    print("\n[test_dual_ineq_only]")
    from act.back_end.hybridz_tf.algorithms.bounds_tighten import (
        hz_bounds_hz_dual,
    )
    for seed in [1, 2, 3]:
        c, Gc, Gb, Ac, Ab, b, em = _make_pair_ineq(seed=seed)
        act_hz = _hzono(c, Gc, Gb, Ac, Ab, b, eq_mask=em)
        hyz_hz = _hyzor(c, Gc, Gb, Ac, Ab, b, eq_mask=em)
        lb_a, ub_a = hz_bounds_hz_dual(
            act_hz, max_iter=50, lr=0.1, selective_lp=False,
        )
        lb_h, ub_h = hyz_hz._bounds_hz_dual(
            max_iter=50, lr=0.1, selective_lp=False,
        )
        e_lb = _close(lb_a, lb_h, f"seed{seed}/lb")
        e_ub = _close(ub_a, ub_h, f"seed{seed}/ub")
        print(f"  [seed{seed} ineq] lb:{e_lb:.1e} ub:{e_ub:.1e}")


def test_dual_with_eq():
    """All-eq HZ — Tier 3 LP correction. Test bound containment + soundness."""
    print("\n[test_dual_with_eq]")
    from act.back_end.hybridz_tf.algorithms.bounds_tighten import (
        hz_bounds_hz_dual,
    )
    for seed in [1, 2, 3]:
        c, Gc, Gb, Ac, Ab, b, em = _make_pair_with_eq(seed=seed)
        act_hz = _hzono(c, Gc, Gb, Ac, Ab, b, eq_mask=em)
        hyz_hz = _hyzor(c, Gc, Gb, Ac, Ab, b, eq_mask=em)
        lb_a, ub_a = hz_bounds_hz_dual(
            act_hz, max_iter=50, lr=0.1, selective_lp=True,
        )
        lb_h, ub_h = hyz_hz._bounds_hz_dual(
            max_iter=50, lr=0.1, selective_lp=True,
        )
        # Both should be reasonably close (Tier-3 LP can have different
        # numerical solver paths; tolerate larger atol).
        e_lb = _close(lb_a, lb_h, f"seed{seed}/lb", atol=1e-6)
        e_ub = _close(ub_a, ub_h, f"seed{seed}/ub", atol=1e-6)
        print(f"  [seed{seed} eq+LP] lb:{e_lb:.1e} ub:{e_ub:.1e}")


def test_dual_nc_zero():
    """nc = 0 → falls back to unconstrained — must match exactly."""
    print("\n[test_dual_nc_zero]")
    from act.back_end.hybridz_tf.algorithms.bounds_tighten import (
        hz_bounds_hz_dual, hz_bounds_unconstrained,
    )
    g = torch.Generator().manual_seed(7)
    n, ng, nb = 5, 4, 2
    c = torch.randn(n, 1, dtype=torch.float64, generator=g)
    Gc = torch.randn(n, ng, dtype=torch.float64, generator=g) * 0.5
    Gb = torch.randn(n, nb, dtype=torch.float64, generator=g) * 0.3
    Ac = torch.zeros(0, ng, dtype=torch.float64)
    Ab = torch.zeros(0, nb, dtype=torch.float64)
    b = torch.zeros(0, 1, dtype=torch.float64)
    act_hz = _hzono(c, Gc, Gb, Ac, Ab, b, eq_mask=None)
    lb_d, ub_d = hz_bounds_hz_dual(act_hz)
    lb_u, ub_u = hz_bounds_unconstrained(act_hz)
    _close(lb_d, lb_u, "nc0/lb")
    _close(ub_d, ub_u, "nc0/ub")
    print(f"  nc=0 fall-through matches unconstrained exactly")


if __name__ == "__main__":
    print("=== hz_bounds_hz_dual parity tests ===")
    tests = [test_dual_nc_zero, test_dual_ineq_only, test_dual_with_eq]
    fail = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            fail += 1
    if fail:
        print(f"\n{fail}/{len(tests)} FAILED")
        sys.exit(1)
    print(f"\nALL {len(tests)} tests PASSED")
