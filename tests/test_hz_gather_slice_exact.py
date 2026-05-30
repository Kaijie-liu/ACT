"""Soundness tests for exact GATHER + SLICE HZ transfer.

These ops are linear row-selection / permutation maps. The HZ transfer must
EXACTLY preserve concretization: for any concrete xi feasible in the input HZ,
the output point must equal the output of applying the same gather/slice to
the concretized input point.
"""
import sys, os
sys.path.insert(0, '/data1/Kane/ACT')
import torch
import numpy as np

from act.back_end.solver.solver_hz import (
    HZono, _hz_gather_exact, _hz_slice_exact,
)


def _concretize(hz: HZono, xi_c: torch.Tensor, xi_b: torch.Tensor = None) -> torch.Tensor:
    """y = c + Gc xi_c + Gb xi_b for given factor space sample."""
    y = hz.c.clone()
    if hz.Gc.numel() > 0:
        y = y + hz.Gc @ xi_c
    if xi_b is not None and hz.Gb.numel() > 0:
        y = y + hz.Gb @ xi_b
    return y


def _make_simple_hz(n_feats: int, n_gens: int = 3):
    """Build a small HZ with non-trivial Gc, no constraints."""
    torch.manual_seed(42)
    c = torch.randn(n_feats, dtype=torch.float64)
    Gc = torch.randn(n_feats, n_gens, dtype=torch.float64)
    Gb = torch.zeros(n_feats, 0, dtype=torch.float64)
    Ac = torch.zeros(0, n_gens, dtype=torch.float64)
    Ab = torch.zeros(0, 0, dtype=torch.float64)
    b = torch.zeros(0, dtype=torch.float64)
    return HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b)


def test_gather_axis0_simple():
    """input_shape=(4,), axis=0, indices=[2,0,3]: output should reorder."""
    hz = _make_simple_hz(n_feats=4, n_gens=3)
    params = {
        "axis": 0,
        "indices": [2, 0, 3],
        "input_shape": (4,),
        "output_shape": (3,),
    }
    hz_out = _hz_gather_exact(hz, params)
    # Concrete check: for random xi_c, output should == input[indices]
    for seed in range(5):
        torch.manual_seed(seed)
        xi_c = torch.empty(3, dtype=torch.float64).uniform_(-1, 1)
        y_in = _concretize(hz, xi_c)
        y_out_expected = torch.tensor([y_in[2], y_in[0], y_in[3]], dtype=torch.float64)
        y_out_actual = _concretize(hz_out, xi_c)
        assert torch.allclose(y_out_actual, y_out_expected, atol=1e-12), \
            f"gather axis0 mismatch: {y_out_actual} vs {y_out_expected}"
    print("test_gather_axis0_simple PASSED")


def test_gather_axis1_multi_dim():
    """input_shape=(2,4), axis=1, indices=[1,3]: select cols 1,3 from each row."""
    hz = _make_simple_hz(n_feats=8, n_gens=2)
    params = {
        "axis": 1,
        "indices": [1, 3],
        "input_shape": (2, 4),
        "output_shape": (2, 2),
    }
    hz_out = _hz_gather_exact(hz, params)
    # Flat input (2,4): position (i, j) = i*4 + j
    # Output flat (2,2): position (i, k) = i*2 + k, where k indexes the chosen indices
    # Output[i, k] = Input[i, indices[k]] = Input flat (i*4 + indices[k])
    for seed in range(5):
        torch.manual_seed(seed)
        xi_c = torch.empty(2, dtype=torch.float64).uniform_(-1, 1)
        y_in = _concretize(hz, xi_c).reshape(2, 4)
        y_out_expected = y_in[:, [1, 3]].reshape(-1)
        y_out_actual = _concretize(hz_out, xi_c)
        assert torch.allclose(y_out_actual, y_out_expected, atol=1e-12), \
            f"gather axis1 mismatch: {y_out_actual} vs {y_out_expected}"
    print("test_gather_axis1_multi_dim PASSED")


def test_gather_scalar_index():
    """input_shape=(4,), axis=0, indices=[2] (single scalar)."""
    hz = _make_simple_hz(n_feats=4, n_gens=3)
    params = {
        "axis": 0,
        "indices": [2],
        "input_shape": (4,),
        "output_shape": (1,),
    }
    hz_out = _hz_gather_exact(hz, params)
    assert hz_out.dim == 1, f"expected dim 1, got {hz_out.dim}"
    for seed in range(3):
        torch.manual_seed(seed)
        xi_c = torch.empty(3, dtype=torch.float64).uniform_(-1, 1)
        y_in = _concretize(hz, xi_c)
        y_out_expected = y_in[[2]]
        y_out_actual = _concretize(hz_out, xi_c)
        assert torch.allclose(y_out_actual, y_out_expected, atol=1e-12)
    print("test_gather_scalar_index PASSED")


def test_slice_axis0_simple():
    """input_shape=(6,), axes=[0], starts=[1], ends=[5], steps=[1]."""
    hz = _make_simple_hz(n_feats=6, n_gens=2)
    params = {
        "starts": [1],
        "ends": [5],
        "axes": [0],
        "steps": [1],
        "input_shape": (6,),
    }
    hz_out = _hz_slice_exact(hz, params)
    assert hz_out.dim == 4, f"expected dim 4, got {hz_out.dim}"
    for seed in range(3):
        torch.manual_seed(seed)
        xi_c = torch.empty(2, dtype=torch.float64).uniform_(-1, 1)
        y_in = _concretize(hz, xi_c)
        y_out_expected = y_in[1:5]
        y_out_actual = _concretize(hz_out, xi_c)
        assert torch.allclose(y_out_actual, y_out_expected, atol=1e-12)
    print("test_slice_axis0_simple PASSED")


def test_slice_step_2():
    """input_shape=(8,), starts=[0], ends=[8], steps=[2]: select every other."""
    hz = _make_simple_hz(n_feats=8, n_gens=2)
    params = {
        "starts": [0], "ends": [8], "axes": [0], "steps": [2],
        "input_shape": (8,),
    }
    hz_out = _hz_slice_exact(hz, params)
    assert hz_out.dim == 4, f"expected dim 4, got {hz_out.dim}"
    for seed in range(3):
        torch.manual_seed(seed)
        xi_c = torch.empty(2, dtype=torch.float64).uniform_(-1, 1)
        y_in = _concretize(hz, xi_c)
        y_out_expected = y_in[0:8:2]
        y_out_actual = _concretize(hz_out, xi_c)
        assert torch.allclose(y_out_actual, y_out_expected, atol=1e-12)
    print("test_slice_step_2 PASSED")


def test_slice_multi_dim():
    """input_shape=(3,4), axes=[1], starts=[1], ends=[3], steps=[1].
    Output should be (3,2)."""
    hz = _make_simple_hz(n_feats=12, n_gens=2)
    params = {
        "starts": [1], "ends": [3], "axes": [1], "steps": [1],
        "input_shape": (3, 4),
    }
    hz_out = _hz_slice_exact(hz, params)
    assert hz_out.dim == 6, f"expected dim 6, got {hz_out.dim}"
    for seed in range(3):
        torch.manual_seed(seed)
        xi_c = torch.empty(2, dtype=torch.float64).uniform_(-1, 1)
        y_in = _concretize(hz, xi_c).reshape(3, 4)
        y_out_expected = y_in[:, 1:3].reshape(-1)
        y_out_actual = _concretize(hz_out, xi_c)
        assert torch.allclose(y_out_actual, y_out_expected, atol=1e-12)
    print("test_slice_multi_dim PASSED")


def test_gather_preserves_constraints():
    """GATHER should NOT touch Ac/Ab/b/eq_mask (factor space constraints stay)."""
    n_feats = 4; n_gens = 3; n_cons = 2
    torch.manual_seed(1)
    c = torch.randn(n_feats, dtype=torch.float64)
    Gc = torch.randn(n_feats, n_gens, dtype=torch.float64)
    Gb = torch.zeros(n_feats, 0, dtype=torch.float64)
    Ac = torch.randn(n_cons, n_gens, dtype=torch.float64)
    Ab = torch.zeros(n_cons, 0, dtype=torch.float64)
    b = torch.randn(n_cons, dtype=torch.float64)
    hz = HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b)
    params = {"axis": 0, "indices": [2, 0], "input_shape": (4,), "output_shape": (2,)}
    hz_out = _hz_gather_exact(hz, params)
    # Ac, Ab, b should be IDENTICAL
    assert torch.equal(hz_out.Ac, Ac)
    assert torch.equal(hz_out.Ab, Ab)
    assert torch.equal(hz_out.b, b)
    print("test_gather_preserves_constraints PASSED")


if __name__ == "__main__":
    test_gather_axis0_simple()
    test_gather_axis1_multi_dim()
    test_gather_scalar_index()
    test_slice_axis0_simple()
    test_slice_step_2()
    test_slice_multi_dim()
    test_gather_preserves_constraints()
    print("\nAll GATHER + SLICE exact tests PASSED ✓")
