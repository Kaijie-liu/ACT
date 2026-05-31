#!/usr/bin/env python3
"""Regression tests for SparseGcZ shared-generator residual add."""

from __future__ import annotations

import torch

from act.back_end.hybridz_tf.representations import SparseGcZ
from act.back_end.hybridz_tf.hz_routing import _sparse_sgm_add


def _sp(mat: torch.Tensor) -> torch.Tensor:
    idx = (mat != 0).nonzero(as_tuple=False).T
    val = mat[idx[0], idx[1]] if idx.numel() else mat.new_zeros(0)
    return torch.sparse_coo_tensor(
        idx, val, mat.shape, dtype=mat.dtype, device=mat.device
    ).coalesce()


def test_sparse_sgm_add_shares_prefix_and_dedupes_constraints():
    dtype = torch.float64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)

    n = 4
    shared_ng = 3
    x_tail = 2
    y_tail = 1
    shared_nc = 2
    x_nc_tail = 1
    y_nc_tail = 2

    gx = torch.randn(n, shared_ng + x_tail, dtype=dtype, device=device)
    gy = torch.randn(n, shared_ng + y_tail, dtype=dtype, device=device)
    gx[gx.abs() < 0.3] = 0
    gy[gy.abs() < 0.3] = 0

    a_shared = torch.randn(shared_nc, shared_ng, dtype=dtype, device=device)
    a_x_tail = torch.randn(x_nc_tail, shared_ng + x_tail, dtype=dtype, device=device)
    a_y_tail = torch.randn(y_nc_tail, shared_ng + y_tail, dtype=dtype, device=device)

    ax = torch.zeros(shared_nc + x_nc_tail, shared_ng + x_tail, dtype=dtype, device=device)
    ay = torch.zeros(shared_nc + y_nc_tail, shared_ng + y_tail, dtype=dtype, device=device)
    ax[:shared_nc, :shared_ng] = a_shared
    ay[:shared_nc, :shared_ng] = a_shared
    ax[shared_nc:] = a_x_tail
    ay[shared_nc:] = a_y_tail

    x = SparseGcZ(
        c=torch.randn(n, dtype=dtype, device=device),
        Gc_sparse=_sp(gx),
        dtype=dtype,
        device=device,
        Ac_sparse=_sp(ax),
        b=torch.randn(ax.shape[0], 1, dtype=dtype, device=device),
    )
    y = SparseGcZ(
        c=torch.randn(n, dtype=dtype, device=device),
        Gc_sparse=_sp(gy),
        dtype=dtype,
        device=device,
        Ac_sparse=_sp(ay),
        b=torch.randn(ay.shape[0], 1, dtype=dtype, device=device),
    )
    object.__setattr__(x, "_base_ng", shared_ng)
    object.__setattr__(y, "_base_ng", shared_ng)
    object.__setattr__(x, "_base_nc", shared_nc)
    object.__setattr__(y, "_base_nc", shared_nc)

    out = _sparse_sgm_add(x, y)

    expected_g = torch.cat(
        [gx[:, :shared_ng] + gy[:, :shared_ng], gx[:, shared_ng:], gy[:, shared_ng:]],
        dim=1,
    )
    expected_a = torch.zeros(
        shared_nc + x_nc_tail + y_nc_tail,
        shared_ng + x_tail + y_tail,
        dtype=dtype,
        device=device,
    )
    expected_a[:shared_nc, :shared_ng] = a_shared
    expected_a[shared_nc:shared_nc + x_nc_tail, :shared_ng + x_tail] = a_x_tail
    expected_a[shared_nc + x_nc_tail:, :shared_ng] = a_y_tail[:, :shared_ng]
    expected_a[shared_nc + x_nc_tail:, shared_ng + x_tail:] = a_y_tail[:, shared_ng:]

    assert torch.allclose(out.Gc_sparse.to_dense(), expected_g)
    assert torch.allclose(out.Ac_sparse.to_dense(), expected_a)
    assert out.ng == shared_ng + x_tail + y_tail
    assert out.nc == shared_nc + x_nc_tail + y_nc_tail
    assert out._base_ng == shared_ng
    assert out._base_nc == shared_nc


if __name__ == "__main__":
    test_sparse_sgm_add_shares_prefix_and_dedupes_constraints()
    print("OK: sparse SGM add regression passes")
