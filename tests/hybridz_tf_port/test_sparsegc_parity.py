"""Parity tests: ACT SparseGcZ methods vs HyZor SparseGcZ methods.

Each method should produce element-wise identical output (modulo
sparse layout permutation, which we normalise via .to_dense()).
"""
from __future__ import annotations
import sys
import torch

sys.path.insert(0, "/data1/Kane/ACT")
sys.path.insert(0, "/data1/Kane/HyZor")
sys.path.insert(0, "/data1/Kane")

from act.back_end.hybridz_tf.representations import SparseGcZ as ACT_SparseGcZ
from HyZor import SparseGcZ as HyZor_SparseGcZ


def _make_pair(seed: int = 1, n: int = 8, ng: int = 5, density: float = 0.4):
    """Build matching ACT and HyZor SparseGcZ instances."""
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(n, dtype=torch.float64, generator=g)
    Gc_dense = torch.randn(n, ng, dtype=torch.float64, generator=g)
    mask = (torch.rand(n, ng, generator=g) < density)
    Gc_dense = Gc_dense * mask.to(Gc_dense.dtype)
    Gc_sparse = Gc_dense.to_sparse_coo().coalesce()
    act = ACT_SparseGcZ(c=c, Gc_sparse=Gc_sparse,
                        dtype=torch.float64, device=torch.device("cpu"))
    hyz = HyZor_SparseGcZ(c=c.clone(), Gc_sparse=Gc_sparse.clone(),
                           dtype=torch.float64, device=torch.device("cpu"))
    return act, hyz


def _close(a, b, tag, atol=1e-12):
    if hasattr(a, "to_dense"): a = a.to_dense()
    if hasattr(b, "to_dense"): b = b.to_dense()
    if tuple(a.shape) != tuple(b.shape):
        raise AssertionError(f"{tag}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.numel() == 0: return 0.0
    diff = (a - b).abs()
    me = float(diff.max().item())
    if me > atol:
        raise AssertionError(f"{tag}: max_err={me:.3e}")
    return me


def _compare_sparse(act, hyz, label):
    err_c = _close(act.c, hyz.c, f"{label}/c")
    err_Gc = _close(act.Gc_sparse, hyz.Gc_sparse, f"{label}/Gc_sparse")
    print(f"  [{label}] c:{err_c:.1e} Gc_sparse:{err_Gc:.1e}")


def _compare_dense_hz(act_hz, hyz_hz, label):
    """ACT returns HZono, HyZor returns HybridZonotope. Both have .c/.Gc."""
    err_c = _close(act_hz.c, hyz_hz.c, f"{label}/c")
    err_Gc = _close(act_hz.Gc, hyz_hz.Gc, f"{label}/Gc")
    print(f"  [{label}] c:{err_c:.1e} Gc:{err_Gc:.1e}")


# --- Tests ---


def test_bounds():
    print("\n[test_bounds]")
    act, hyz = _make_pair(seed=1, n=6, ng=4)
    lb_a, ub_a = act.bounds()
    lb_h, ub_h = hyz.bounds()
    _close(lb_a, lb_h, "lb")
    _close(ub_a, ub_h, "ub")
    print("  bounds match")


def test_apply_scale():
    print("\n[test_apply_scale]")
    for seed in [1, 2, 3]:
        act, hyz = _make_pair(seed=seed, n=6, ng=4)
        a = torch.tensor([2.0, -1.0, 0.5, 3.0, -2.5, 0.0], dtype=torch.float64)
        out_a = act.apply_scale(a)
        out_h = hyz.apply_scale(a)
        _compare_sparse(out_a, out_h, f"scale_seed{seed}")


def test_apply_relu_triangle():
    print("\n[test_apply_relu_triangle]")
    for seed in [1, 2, 3, 4]:
        act, hyz = _make_pair(seed=seed, n=8, ng=4)
        out_a = act.apply_relu_triangle()
        out_h = hyz.apply_relu_triangle()
        _compare_sparse(out_a, out_h, f"relu_tri_seed{seed}")


def test_apply_dense():
    print("\n[test_apply_dense]")
    for seed in [1, 2]:
        act, hyz = _make_pair(seed=seed, n=6, ng=4)
        g = torch.Generator().manual_seed(seed + 100)
        W = torch.randn(5, 6, dtype=torch.float64, generator=g)
        b = torch.randn(5, dtype=torch.float64, generator=g)
        out_a = act.apply_dense(W, b)
        out_h = hyz.apply_dense(W, b)
        _compare_dense_hz(out_a, out_h, f"dense_seed{seed}")


def test_density_bytes():
    print("\n[test_density_bytes]")
    act, hyz = _make_pair(seed=1, n=10, ng=5, density=0.3)
    da = act.density_bytes()
    dh = hyz.density_bytes()
    if da != dh:
        raise AssertionError(f"density_bytes ACT={da} HyZor={dh}")
    print(f"  density_bytes ACT={da} HyZor={dh}")


def test_to_dense_hz():
    print("\n[test_to_dense_hz]")
    act, hyz = _make_pair(seed=1, n=6, ng=4)
    out_a = act.to_dense_hz()
    out_h = hyz.to_dense_hz()
    _compare_dense_hz(out_a, out_h, "to_dense_hz")


def test_reduce_generators():
    print("\n[test_reduce_generators]")
    # Make ng > target_ng so reduction triggers
    act, hyz = _make_pair(seed=1, n=6, ng=10, density=0.5)
    out_a = act.reduce_generators(target_ng=4)
    out_h = hyz.reduce_generators(target_ng=4)
    # Reduction can produce different column orderings but same SET of
    # generators (top-k by L2 norm is unique up to ties + slack ordering).
    # Compare via bounds (lb, ub) which are layout-invariant.
    lb_a, ub_a = out_a.bounds()
    lb_h, ub_h = out_h.bounds()
    _close(lb_a, lb_h, "reduce/lb")
    _close(ub_a, ub_h, "reduce/ub")
    print(f"  reduce_generators bounds match (ng {out_a.ng} vs {out_h.ng})")


def _make_image_pair(seed: int, C: int, H: int, W: int, ng: int, density: float = 0.3):
    """Build matching SparseGcZ instances with n = C*H*W (CHW flat)."""
    g = torch.Generator().manual_seed(seed)
    n = C * H * W
    c = torch.randn(n, dtype=torch.float64, generator=g)
    Gc_dense = torch.randn(n, ng, dtype=torch.float64, generator=g)
    mask = (torch.rand(n, ng, generator=g) < density)
    Gc_dense = Gc_dense * mask.to(Gc_dense.dtype)
    Gc_sparse = Gc_dense.to_sparse_coo().coalesce()
    act = ACT_SparseGcZ(c=c, Gc_sparse=Gc_sparse,
                        dtype=torch.float64, device=torch.device("cpu"))
    hyz = HyZor_SparseGcZ(c=c.clone(), Gc_sparse=Gc_sparse.clone(),
                           dtype=torch.float64, device=torch.device("cpu"))
    return act, hyz


def test_apply_conv():
    print("\n[test_apply_conv]")
    cases = [
        # (seed, Cin, Hin, Win, Cout, kH, kW, stride, pad, ng)
        (1, 2, 3, 3, 3, 2, 2, 1, 0, 4),
        (2, 2, 4, 4, 2, 3, 3, 1, 1, 5),
        (3, 3, 4, 4, 4, 2, 2, 2, 0, 6),
    ]
    for (seed, Cin, Hin, Win, Cout, kH, kW, stride, pad, ng) in cases:
        act, hyz = _make_image_pair(seed=seed, C=Cin, H=Hin, W=Win, ng=ng)
        g = torch.Generator().manual_seed(seed + 1000)
        weight = torch.randn(Cout, Cin, kH, kW, dtype=torch.float64, generator=g)
        bias = torch.randn(Cout, dtype=torch.float64, generator=g)
        out_a = act.apply_conv(weight, bias, (Cin, Hin, Win), stride, pad)
        out_h = hyz.apply_conv(weight, bias, (Cin, Hin, Win), stride, pad)
        _compare_sparse(out_a, out_h, f"conv_seed{seed}_{Cin}x{Hin}x{Win}_k{kH}_s{stride}_p{pad}")


if __name__ == "__main__":
    print("=== SparseGcZ ACT vs HyZor parity ===")
    tests = [test_bounds, test_apply_scale, test_apply_relu_triangle,
             test_apply_dense, test_density_bytes, test_to_dense_hz,
             test_reduce_generators, test_apply_conv]
    fails = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL {t.__name__}: {e}")
            fails += 1
    if fails:
        print(f"\n{fails}/{len(tests)} FAILED")
        sys.exit(1)
    print(f"\nALL {len(tests)} tests PASSED")
