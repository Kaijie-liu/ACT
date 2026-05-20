"""Parity test: ACT hz_conv2d (tf_cnn.py) vs HyZor hz_conv2d.

ACT signature: ``hz_conv2d(hz, weight, bias, stride, padding, dilation, groups, input_shape)``
HyZor signature: ``hz_conv2d(hz, weight, bias=None, *, input_shape, stride=1, padding=0, dilation=1, groups=1)``

This test exercises the conv math on small (B=1, C=2, H=H_in, W=H_in)
images with a few weight configurations. Both implementations
convolve the center as one image and each generator as a per-batch
image (HyZor does the same — see HZ __init__.py:1007).

Eq_mask should pass through unchanged (conv only touches coordinates,
not factor-space constraint rows).
"""
from __future__ import annotations
import sys
import torch

sys.path.insert(0, "/data1/Kane/ACT")
sys.path.insert(0, "/data1/Kane/HyZor")
sys.path.insert(0, "/data1/Kane")

from act.back_end.solver.solver_hz import HZono
from act.back_end.hybridz_tf.tf_cnn import hz_conv2d as act_hz_conv2d


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


def _make_random_hz(*, n, ng, nb, nc, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(n, 1, dtype=dtype, generator=g)
    Gc = torch.randn(n, ng, dtype=dtype, generator=g) * 0.3
    Gb = torch.randn(n, nb, dtype=dtype, generator=g) * 0.2
    Ac = torch.randn(nc, ng, dtype=dtype, generator=g) * 0.5
    Ab = torch.randn(nc, nb, dtype=dtype, generator=g) * 0.5
    b = torch.randn(nc, 1, dtype=dtype, generator=g) * 0.3
    em = torch.ones(nc, dtype=torch.bool) if nc > 0 else None
    return c, Gc, Gb, Ac, Ab, b, em


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_conv_basic_2x2x2_kernel1():
    """2-channel 2×2 image, 1×1 kernel. Trivial spatial structure."""
    C_in, H_in, W_in = 2, 2, 2
    n = C_in * H_in * W_in  # 8
    c, Gc, Gb, Ac, Ab, b, em = _make_random_hz(
        n=n, ng=3, nb=0, nc=0, seed=1)
    C_out = 3
    weight = torch.tensor([
        [[[1.0]], [[2.0]]],   # filter 0
        [[[-1.0]], [[0.5]]],  # filter 1
        [[[0.5]], [[-0.5]]],  # filter 2
    ], dtype=torch.float64)  # shape (C_out, C_in, 1, 1)
    bias = torch.tensor([0.1, -0.2, 0.0], dtype=torch.float64)
    input_shape = (C_in, H_in, W_in)

    act_out = act_hz_conv2d(
        _hzono(c, Gc, Gb, Ac, Ab, b, eq_mask=em),
        weight, bias, stride=(1, 1), padding=0,
        dilation=(1, 1), groups=1, input_shape=input_shape,
    )
    from HyZor import hz_conv2d as hyzor_conv
    h_out = hyzor_conv(
        _hyzor(c, Gc, Gb, Ac, Ab, b, eq_mask=em),
        weight, bias, input_shape=input_shape,
        stride=1, padding=0, dilation=1, groups=1,
    )
    _compare(act_out, h_out, "basic_2x2x2_k1")


def test_conv_3channel_3x3_kernel3():
    """3-channel 3×3 image, 3×3 kernel, no padding."""
    C_in, H_in, W_in = 3, 3, 3
    n = C_in * H_in * W_in  # 27
    c, Gc, Gb, Ac, Ab, b, em = _make_random_hz(
        n=n, ng=4, nb=2, nc=2, seed=2)
    C_out = 2
    weight = torch.randn(C_out, C_in, 3, 3, dtype=torch.float64,
                          generator=torch.Generator().manual_seed(20))
    bias = torch.tensor([0.5, -0.5], dtype=torch.float64)
    input_shape = (C_in, H_in, W_in)

    act_out = act_hz_conv2d(
        _hzono(c, Gc, Gb, Ac, Ab, b, eq_mask=em),
        weight, bias, stride=(1, 1), padding=0,
        dilation=(1, 1), groups=1, input_shape=input_shape,
    )
    from HyZor import hz_conv2d as hyzor_conv
    h_out = hyzor_conv(
        _hyzor(c, Gc, Gb, Ac, Ab, b, eq_mask=em),
        weight, bias, input_shape=input_shape,
        stride=1, padding=0, dilation=1, groups=1,
    )
    _compare(act_out, h_out, "3ch_3x3_k3")


def test_conv_stride2_pad1():
    """Stride 2 + padding 1 to test output shape calculation."""
    C_in, H_in, W_in = 2, 4, 4
    n = C_in * H_in * W_in  # 32
    c, Gc, Gb, Ac, Ab, b, em = _make_random_hz(
        n=n, ng=6, nb=0, nc=3, seed=3)
    C_out = 3
    weight = torch.randn(C_out, C_in, 3, 3, dtype=torch.float64,
                          generator=torch.Generator().manual_seed(30))
    bias = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    input_shape = (C_in, H_in, W_in)

    act_out = act_hz_conv2d(
        _hzono(c, Gc, Gb, Ac, Ab, b, eq_mask=em),
        weight, bias, stride=(2, 2), padding=1,
        dilation=(1, 1), groups=1, input_shape=input_shape,
    )
    from HyZor import hz_conv2d as hyzor_conv
    h_out = hyzor_conv(
        _hyzor(c, Gc, Gb, Ac, Ab, b, eq_mask=em),
        weight, bias, input_shape=input_shape,
        stride=2, padding=1, dilation=1, groups=1,
    )
    _compare(act_out, h_out, "stride2_pad1")


def test_conv_no_bias():
    """No bias term."""
    C_in, H_in, W_in = 2, 3, 3
    n = C_in * H_in * W_in
    c, Gc, Gb, Ac, Ab, b, em = _make_random_hz(
        n=n, ng=3, nb=0, nc=0, seed=4)
    C_out = 2
    weight = torch.randn(C_out, C_in, 3, 3, dtype=torch.float64,
                          generator=torch.Generator().manual_seed(40))
    input_shape = (C_in, H_in, W_in)

    act_out = act_hz_conv2d(
        _hzono(c, Gc, Gb, Ac, Ab, b, eq_mask=em),
        weight, None, stride=(1, 1), padding=0,
        dilation=(1, 1), groups=1, input_shape=input_shape,
    )
    from HyZor import hz_conv2d as hyzor_conv
    h_out = hyzor_conv(
        _hyzor(c, Gc, Gb, Ac, Ab, b, eq_mask=em),
        weight, None, input_shape=input_shape,
        stride=1, padding=0, dilation=1, groups=1,
    )
    _compare(act_out, h_out, "no_bias")


def test_conv_eq_mask_preserved():
    """Ensure eq_mask passes through unchanged (the fix we just added)."""
    C_in, H_in, W_in = 2, 2, 2
    n = C_in * H_in * W_in
    c, Gc, Gb, Ac, Ab, b, _ = _make_random_hz(
        n=n, ng=2, nb=1, nc=3, seed=5)
    # Mixed eq_mask: first row eq, next two ineq.
    em_in = torch.tensor([True, False, False])
    weight = torch.randn(2, C_in, 1, 1, dtype=torch.float64,
                          generator=torch.Generator().manual_seed(50))
    bias = torch.tensor([0.0, 0.0], dtype=torch.float64)
    input_shape = (C_in, H_in, W_in)

    act_out = act_hz_conv2d(
        _hzono(c, Gc, Gb, Ac, Ab, b, eq_mask=em_in),
        weight, bias, stride=(1, 1), padding=0,
        dilation=(1, 1), groups=1, input_shape=input_shape,
    )
    if act_out.eq_mask is None:
        raise AssertionError("conv2d dropped eq_mask (regression)")
    if not torch.equal(act_out.eq_mask, em_in):
        raise AssertionError(
            f"conv2d changed eq_mask: {act_out.eq_mask.tolist()} vs {em_in.tolist()}"
        )
    print(f"  [eq_mask_preserved] OK: input {em_in.tolist()} == output {act_out.eq_mask.tolist()}")


if __name__ == "__main__":
    print("=== hz_conv2d ACT vs HyZor parity ===")
    tests = [
        test_conv_basic_2x2x2_kernel1,
        test_conv_3channel_3x3_kernel3,
        test_conv_stride2_pad1,
        test_conv_no_bias,
        test_conv_eq_mask_preserved,
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
    print(f"\nALL {len(tests)} tests PASSED — hz_conv2d ACT == HyZor")
