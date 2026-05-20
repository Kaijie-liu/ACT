"""Parity test: ACT hz_maxpool2d vs HyZor max_pool_node_evaluate.

Both implement solver-free maxpool with stable-winner row preservation
and per-block interval relaxation for unstable blocks. We compare
6-tuples element-wise.

HyZor entry point: ``HybridZonotope.max_pool_node_evaluate``.
HyZor exposes it externally via ``hz_maxpool2d(hz, kernel_size, stride,
padding, input_shape)``.
"""
from __future__ import annotations
import sys
import torch

sys.path.insert(0, "/data1/Kane/ACT")
sys.path.insert(0, "/data1/Kane/HyZor")
sys.path.insert(0, "/data1/Kane")

from act.back_end.solver.solver_hz import HZono
from act.back_end.hybridz_tf.algorithms.maxpool import hz_maxpool2d as act_maxpool


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
    if a.numel() == 0:
        return 0.0
    diff = (a - b).abs()
    me = float(diff.max().item())
    ref = max(float(a.abs().max().item()), float(b.abs().max().item()), 1.0)
    if me > atol * max(1.0, ref):
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
    s = ", ".join(f"{k}:{v:.1e}" for k, v in errs.items())
    print(f"  [{label}] 6-tuple max_err: {s}")


def _make_random_hz(*, n, ng, nb, nc, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(n, 1, dtype=torch.float64, generator=g)
    Gc = torch.randn(n, ng, dtype=torch.float64, generator=g) * 0.3
    Gb = torch.randn(n, nb, dtype=torch.float64, generator=g) * 0.2
    Ac = torch.randn(nc, ng, dtype=torch.float64, generator=g) * 0.5
    Ab = torch.randn(nc, nb, dtype=torch.float64, generator=g) * 0.5
    b = torch.randn(nc, 1, dtype=torch.float64, generator=g) * 0.3
    em = torch.ones(nc, dtype=torch.bool) if nc > 0 else None
    return c, Gc, Gb, Ac, Ab, b, em


def _run_case(label, kernel, stride, padding, C, H, W,
              *, ng=2, nb=0, nc=0, seed=1):
    n = C * H * W
    c, Gc, Gb, Ac, Ab, b, em = _make_random_hz(
        n=n, ng=ng, nb=nb, nc=nc, seed=seed)
    a_out = act_maxpool(
        _hzono(c, Gc, Gb, Ac, Ab, b, eq_mask=em),
        kernel_size=kernel, stride=stride, padding=padding,
        input_shape=(C, H, W),
    )
    # HyZor's hz_maxpool2d (the helper at __init__.py:1180).
    from HyZor import hz_maxpool2d as hyzor_maxpool
    h_out = hyzor_maxpool(
        _hyzor(c, Gc, Gb, Ac, Ab, b, eq_mask=em),
        kernel_size=kernel, stride=stride, padding=padding,
        input_shape=(C, H, W),
    )
    _compare(a_out, h_out, label)


def test_basic_k2_s2():
    _run_case("k2_s2", kernel=2, stride=2, padding=0,
              C=2, H=4, W=4, ng=3, seed=1)


def test_k2_s1():
    _run_case("k2_s1", kernel=2, stride=1, padding=0,
              C=2, H=4, W=4, ng=2, seed=2)


def test_with_padding_pad1():
    _run_case("pad1", kernel=3, stride=1, padding=1,
              C=1, H=4, W=4, ng=2, seed=3)


def test_with_binary_gens():
    _run_case("with_bin", kernel=2, stride=2, padding=0,
              C=2, H=4, W=4, ng=2, nb=2, seed=4)


def test_with_prior_constraints():
    _run_case("with_prior_eq", kernel=2, stride=2, padding=0,
              C=2, H=4, W=4, ng=3, nb=1, nc=2, seed=5)


def test_non_square_kernel():
    _run_case("k_1x3", kernel=(1, 3), stride=(1, 1), padding=0,
              C=1, H=3, W=4, ng=2, seed=6)


if __name__ == "__main__":
    print("=== hz_maxpool2d ACT vs HyZor parity ===")
    tests = [
        test_basic_k2_s2, test_k2_s1, test_with_padding_pad1,
        test_with_binary_gens, test_with_prior_constraints,
        test_non_square_kernel,
    ]
    fail = 0
    for t in tests:
        print(f"\n[{t.__name__}]")
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL: {e}")
            fail += 1
    if fail:
        print(f"\n{fail}/{len(tests)} FAILED")
        sys.exit(1)
    print(f"\nALL {len(tests)} tests PASSED — hz_maxpool2d ACT == HyZor")
