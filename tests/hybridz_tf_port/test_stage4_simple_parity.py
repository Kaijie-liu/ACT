"""Parity tests for Y2 Stage 4 simple ports.

  hz_bounds_unconstrained ↔ HybridZonotope._bounds_unconstrained
  hz_intersect_box        ↔ HybridZonotope.intersect_box
  hz_apply_relu(external_bounds=...) ↔ applyReLU_eq_native(external_bounds=...)
"""
from __future__ import annotations
import sys
import torch

sys.path.insert(0, "/data1/Kane/ACT")
sys.path.insert(0, "/data1/Kane/HyZor")
sys.path.insert(0, "/data1/Kane")

from act.back_end.solver.solver_hz import HZono
from act.back_end.hybridz_tf.algorithms.bounds_tighten import (
    hz_bounds_unconstrained, hz_intersect_box,
)
from act.back_end.hybridz_tf.tf_mlp import hz_apply_relu


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
        raise AssertionError(f"{tag}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.numel() == 0: return 0.0
    diff = (a - b).abs()
    me = float(diff.max().item())
    if me > atol:
        raise AssertionError(f"{tag}: max_err={me:.3e}")
    return me


def _compare(hzono_out, hyzor_out, label):
    errs = {
        "c": _close(hzono_out.c, hyzor_out.c, f"{label}/c"),
        "Gc": _close(hzono_out.Gc, hyzor_out.Gc, f"{label}/Gc"),
        "Gb": _close(hzono_out.Gb, hyzor_out.Gb, f"{label}/Gb"),
        "Ac": _close(hzono_out.Ac, hyzor_out.Ac, f"{label}/Ac"),
        "Ab": _close(hzono_out.Ab, hyzor_out.Ab, f"{label}/Ab"),
        "b": _close(hzono_out.b, hyzor_out.b, f"{label}/b"),
    }
    print(f"  [{label}] max_err: " + ", ".join(f"{k}:{v:.1e}" for k, v in errs.items()))


def _make_pair(seed: int = 1, n: int = 4, ng: int = 3, nb: int = 1, nc: int = 0):
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(n, 1, dtype=torch.float64, generator=g)
    Gc = torch.randn(n, ng, dtype=torch.float64, generator=g) * 0.5
    Gb = torch.randn(n, nb, dtype=torch.float64, generator=g) * 0.3 if nb > 0 else torch.zeros(n, 0, dtype=torch.float64)
    Ac = torch.randn(nc, ng, dtype=torch.float64, generator=g) * 0.4 if nc > 0 else torch.zeros(0, ng, dtype=torch.float64)
    Ab = torch.randn(nc, nb, dtype=torch.float64, generator=g) * 0.4 if nc > 0 and nb > 0 else torch.zeros(nc, nb, dtype=torch.float64)
    b = torch.randn(nc, 1, dtype=torch.float64, generator=g) * 0.3 if nc > 0 else torch.zeros(0, 1, dtype=torch.float64)
    em = torch.ones(nc, dtype=torch.bool) if nc > 0 else None
    return c, Gc, Gb, Ac, Ab, b, em


# --- Tests ---


def test_bounds_unconstrained():
    print("\n[test_bounds_unconstrained]")
    for seed in [1, 2, 3]:
        c, Gc, Gb, Ac, Ab, b, em = _make_pair(seed=seed)
        act_hz = _hzono(c, Gc, Gb, Ac, Ab, b, eq_mask=em)
        hyz_hz = _hyzor(c, Gc, Gb, Ac, Ab, b, eq_mask=em)
        lb_a, ub_a = hz_bounds_unconstrained(act_hz)
        lb_h, ub_h = hyz_hz._bounds_unconstrained()
        e_lb = _close(lb_a, lb_h, f"seed{seed}/lb")
        e_ub = _close(ub_a, ub_h, f"seed{seed}/ub")
        print(f"  [seed{seed}] lb:{e_lb:.1e} ub:{e_ub:.1e}")


def test_intersect_box():
    print("\n[test_intersect_box]")
    for seed in [1, 2, 3]:
        c, Gc, Gb, Ac, Ab, b, em = _make_pair(seed=seed)
        act_hz = _hzono(c, Gc, Gb, Ac, Ab, b, eq_mask=em)
        hyz_hz = _hyzor(c, Gc, Gb, Ac, Ab, b, eq_mask=em)
        # Tighter box for intersect
        lb_tight = c.flatten() - 0.5
        ub_tight = c.flatten() + 0.5
        out_a = hz_intersect_box(act_hz, lb_tight, ub_tight)
        out_h = hyz_hz.intersect_box(lb_tight, ub_tight)
        _compare(out_a, out_h, f"intersect_seed{seed}")


def test_relu_with_external_bounds():
    print("\n[test_relu_with_external_bounds]")
    # Mixed-unstable case from earlier parity tests.
    c = torch.tensor([[2.0], [-3.0], [0.5], [-0.5]], dtype=torch.float64)
    Gc = torch.tensor([[0.5, 0.5], [0.5, 0.5], [1.0, 1.0], [1.0, 1.0]],
                      dtype=torch.float64)
    Gb = torch.zeros((4, 0), dtype=torch.float64)
    Ac = torch.zeros((0, 2), dtype=torch.float64)
    Ab = torch.zeros((0, 0), dtype=torch.float64)
    b = torch.zeros((0, 1), dtype=torch.float64)

    # Tight external bounds (tighter than unconstrained).
    lb_ext = torch.tensor([1.5, -3.5, -1.0, -2.0], dtype=torch.float64)
    ub_ext = torch.tensor([2.5, -2.5, 2.0, 1.0], dtype=torch.float64)

    act_hz = _hzono(c, Gc, Gb, Ac, Ab, b, eq_mask=None)
    hyz_hz = _hyzor(c, Gc, Gb, Ac, Ab, b, eq_mask=None)
    out_a = hz_apply_relu(act_hz, external_bounds=(lb_ext, ub_ext))
    out_h = hyz_hz.applyReLU_eq_native(external_bounds=(lb_ext, ub_ext))
    _compare(out_a, out_h, "ext_bounds")


def test_relu_external_vs_unconstrained_same():
    """Without external_bounds, should match unconstrained-bounds case."""
    print("\n[test_relu_external_vs_unconstrained_same]")
    c, Gc, Gb, Ac, Ab, b, em = _make_pair(seed=10, n=4, ng=2, nb=0, nc=0)
    act_hz = _hzono(c, Gc, Gb, Ac, Ab, b, eq_mask=em)
    hyz_hz = _hyzor(c, Gc, Gb, Ac, Ab, b, eq_mask=em)
    # No external_bounds — should match HyZor's path.
    out_a = hz_apply_relu(act_hz)
    out_h = hyz_hz.applyReLU_eq_native()
    _compare(out_a, out_h, "no_ext")


def test_bounds_eq_elim_lp():
    """Test hz_bounds_eq_elim_lp matches HyZor._bounds_eq_elim_lp."""
    print("\n[test_bounds_eq_elim_lp]")
    from act.back_end.hybridz_tf.algorithms.bounds_tighten import hz_bounds_eq_elim_lp

    # Build an HZ with equality rows (mimics post-ReLU encoding).
    from act.back_end.hybridz_tf.tf_mlp import hz_apply_relu
    g = torch.Generator().manual_seed(42)
    n, ng = 4, 2
    c = torch.tensor([[2.0], [-3.0], [0.5], [-0.5]], dtype=torch.float64)
    Gc = torch.tensor([[0.5, 0.5], [0.5, 0.5], [1.0, 1.0], [1.0, 1.0]],
                      dtype=torch.float64)
    Gb = torch.zeros((4, 0), dtype=torch.float64)
    Ac = torch.zeros((0, 2), dtype=torch.float64)
    Ab = torch.zeros((0, 0), dtype=torch.float64)
    b = torch.zeros((0, 1), dtype=torch.float64)
    act_hz_pre = _hzono(c, Gc, Gb, Ac, Ab, b, eq_mask=None)
    hyz_hz_pre = _hyzor(c, Gc, Gb, Ac, Ab, b, eq_mask=None)
    # ReLU adds 3*k eq rows for the unstable neurons.
    act_hz_post = hz_apply_relu(act_hz_pre)
    hyz_hz_post = hyz_hz_pre.applyReLU_eq_native()

    # Compute LP-tight bounds on both.
    lb_a, ub_a = hz_bounds_eq_elim_lp(act_hz_post)
    lb_h, ub_h = hyz_hz_post._bounds_eq_elim_lp()
    e_lb = _close(lb_a, lb_h, "eq_elim/lb")
    e_ub = _close(ub_a, ub_h, "eq_elim/ub")
    print(f"  [eq_elim_full] lb:{e_lb:.1e} ub:{e_ub:.1e}")
    print(f"  bounds: lb={lb_a.flatten().tolist()}, ub={ub_a.flatten().tolist()}")


if __name__ == "__main__":
    print("=== Stage 4 simple parity tests ===")
    tests = [test_bounds_unconstrained, test_intersect_box,
             test_relu_with_external_bounds,
             test_relu_external_vs_unconstrained_same,
             test_bounds_eq_elim_lp]
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
