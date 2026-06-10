# ===- act/back_end/fchz_tf/conv2d_sparse_encoder.py - CONV2D as sparse affine ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   Convert CONV2D operation into sparse affine map (y = A @ x + b) suitable
#   for direct integration into forward LP/MILP encoding. Replaces the
#   "loose box" CONV2D handler in forward_global_lp.py with a TIGHT
#   equality-constraint-based encoding.
#
# Math:
#   For y[co, ho, wo] = sum_{ci, kh, kw} W[co, ci, kh, kw] * x[ci, hi, wi] + b[co]
#   where hi = ho*stride_h + kh - pad_h, wi = wo*stride_w + kw - pad_w
#   (with padding=0 contribution when out of bounds)
#
#   The sparse matrix A has:
#     - row_idx = co*Hout*Wout + ho*Wout + wo
#     - col_idx = ci*Hin*Win + hi*Win + wi  (when in bounds)
#     - value = W[co, ci, kh, kw]
#
# Memory:
#   For CIFAR (3,32,32) → (16,16,16): output 4096 vars, sparse density ~5%
#   nnz = 4096 * 9 * Cin ≈ 110K (vs dense 12.5M). Manageable for HiGHS.
#
# ===---------------------------------------------------------------------===#
"""Sparse affine encoding of CONV2D for forward LP/MILP."""
import numpy as np
from typing import Tuple, Optional
import scipy.sparse as sp


def conv2d_output_shape(input_shape: Tuple[int, int, int, int],
                                       weight_shape: Tuple[int, int, int, int],
                                       stride: Tuple[int, int] = (1, 1),
                                       padding: Tuple[int, int] = (0, 0)
                                       ) -> Tuple[int, int, int, int]:
    """Compute conv2d output shape from input/weight/stride/padding.

    Args:
      input_shape: (1, Cin, Hin, Win) — batch-first NCHW
      weight_shape: (Cout, Cin, kH, kW)
      stride: (stride_h, stride_w)
      padding: (pad_h, pad_w)

    Returns:
      (1, Cout, Hout, Wout)
    """
    _, Cin, Hin, Win = input_shape
    Cout, _, kH, kW = weight_shape
    sh, sw = stride
    ph, pw = padding
    Hout = (Hin + 2 * ph - kH) // sh + 1
    Wout = (Win + 2 * pw - kW) // sw + 1
    return (1, Cout, Hout, Wout)


def conv2d_to_sparse_affine(weight: np.ndarray,
                                                bias: Optional[np.ndarray],
                                                input_shape: Tuple[int, int, int, int],
                                                stride: Tuple[int, int] = (1, 1),
                                                padding: Tuple[int, int] = (0, 0)
                                                ) -> Tuple[sp.csr_matrix, np.ndarray, Tuple[int, int, int, int]]:
    """Convert CONV2D op into sparse affine: y_flat = A @ x_flat + b_flat.

    Args:
      weight: numpy array shape (Cout, Cin, kH, kW)
      bias: numpy array shape (Cout,) or None
      input_shape: (1, Cin, Hin, Win)
      stride: (stride_h, stride_w)
      padding: (pad_h, pad_w)

    Returns:
      (A_csr, b_flat, output_shape):
        A_csr: scipy.sparse.csr_matrix of shape (Cout*Hout*Wout, Cin*Hin*Win)
        b_flat: numpy array shape (Cout*Hout*Wout,)
        output_shape: (1, Cout, Hout, Wout)

    Soundness:
      Computes y = Conv(x) exactly as an affine map. No relaxation.
      The output_shape calculation matches torch.nn.functional.conv2d.
    """
    if weight.ndim != 4:
        raise ValueError(f"weight must be (Cout, Cin, kH, kW), got shape {weight.shape}")
    if len(input_shape) != 4 or input_shape[0] != 1:
        raise ValueError(f"input_shape must be (1, Cin, Hin, Win), got {input_shape}")

    Cout, Cin_w, kH, kW = weight.shape
    _, Cin, Hin, Win = input_shape
    if Cin != Cin_w:
        raise ValueError(f"weight Cin={Cin_w} doesn't match input Cin={Cin}")

    sh, sw = stride
    ph, pw = padding
    out_shape = conv2d_output_shape(input_shape, weight.shape, stride, padding)
    _, _, Hout, Wout = out_shape

    n_in = Cin * Hin * Win
    n_out = Cout * Hout * Wout

    # Build COO triplets
    rows = []
    cols = []
    vals = []

    for co in range(Cout):
        for ho in range(Hout):
            for wo in range(Wout):
                row_idx = co * Hout * Wout + ho * Wout + wo
                for ci in range(Cin):
                    for kh in range(kH):
                        for kw in range(kW):
                            hi = ho * sh + kh - ph
                            wi = wo * sw + kw - pw
                            # In-bounds check (out-of-bounds means zero-padded)
                            if 0 <= hi < Hin and 0 <= wi < Win:
                                w_val = float(weight[co, ci, kh, kw])
                                if w_val != 0.0:
                                    col_idx = ci * Hin * Win + hi * Win + wi
                                    rows.append(row_idx)
                                    cols.append(col_idx)
                                    vals.append(w_val)

    A_coo = sp.coo_matrix((vals, (rows, cols)), shape=(n_out, n_in), dtype=np.float64)
    A_csr = A_coo.tocsr()

    # Bias broadcast: b_flat[co*Hout*Wout + ho*Wout + wo] = bias[co]
    if bias is not None:
        if bias.shape[0] != Cout:
            raise ValueError(f"bias shape {bias.shape} doesn't match Cout={Cout}")
        b_flat = np.repeat(bias.astype(np.float64), Hout * Wout)
    else:
        b_flat = np.zeros(n_out, dtype=np.float64)

    return A_csr, b_flat, out_shape


def verify_sparse_affine_against_torch(A_csr: sp.csr_matrix, b_flat: np.ndarray,
                                                                 weight: np.ndarray, bias: Optional[np.ndarray],
                                                                 input_shape: Tuple[int, int, int, int],
                                                                 stride: Tuple[int, int] = (1, 1),
                                                                 padding: Tuple[int, int] = (0, 0),
                                                                 n_samples: int = 5,
                                                                 tol: float = 1e-5) -> bool:
    """Verify A @ x + b == Conv(x) for random x samples via torch.

    Returns True if all samples agree within tolerance.
    """
    import torch
    import torch.nn.functional as F

    _, Cin, Hin, Win = input_shape
    n_in = Cin * Hin * Win

    W_t = torch.from_numpy(weight.astype(np.float64))
    b_t = torch.from_numpy(bias.astype(np.float64)) if bias is not None else None

    rng = np.random.default_rng(42)
    for _ in range(n_samples):
        x = rng.standard_normal(n_in).astype(np.float64)
        # Sparse affine
        y_sparse = A_csr @ x + b_flat
        # Torch reference
        x_t = torch.from_numpy(x.reshape(1, Cin, Hin, Win))
        y_torch = F.conv2d(x_t, W_t, b_t, stride=stride, padding=padding).flatten().numpy()
        if not np.allclose(y_sparse, y_torch, atol=tol, rtol=tol):
            diff = np.abs(y_sparse - y_torch).max()
            return False, diff
    return True, 0.0


__all__ = ['conv2d_output_shape', 'conv2d_to_sparse_affine', 'verify_sparse_affine_against_torch']
