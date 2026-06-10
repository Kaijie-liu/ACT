# ===- act/back_end/fchz_tf/memory_estimator.py - LP/MILP memory estimator =====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   Memory estimator for LP/MILP encoding (v8_memaware concept ported to FCHZ).
#   Pre-flight check: estimate peak memory of sparse LP/MILP build to avoid OOM.
#
#   Estimates:
#     - Total vars (input + intermediate + spec)
#     - Total constraint rows (equality + inequality)
#     - Sparse matrix nnz (CONV2D blocks + triangle ReLU + bound rows)
#     - Peak dense float64 memory if no sparse storage
#     - Recommended path: dense OK / sparse required / TOO LARGE
#
# ===---------------------------------------------------------------------===#
"""Memory estimator for forward LP/MILP builds."""
import numpy as np
from typing import Dict, Tuple, Optional, List


def estimate_layer_var_count(net) -> Dict[int, int]:
    """Estimate number of variables introduced by each layer.

    Returns dict layer_id -> n_vars_added.
    """
    var_count: Dict[int, int] = {}
    for L in net.layers:
        if L.kind == 'INPUT':
            # Input vars
            in_shape = L.params.get('shape')
            if in_shape:
                var_count[L.id] = int(np.prod(in_shape[1:]) if len(in_shape) > 1 else in_shape[0])
            else:
                var_count[L.id] = 0
        elif L.kind == 'DENSE':
            W = L.params.get('weight')
            if W is not None:
                var_count[L.id] = int(W.shape[0]) if hasattr(W, 'shape') else 0
            else:
                var_count[L.id] = 0
        elif L.kind == 'CONV2D':
            out_shape = L.params.get('output_shape')
            if out_shape and len(out_shape) >= 4:
                var_count[L.id] = int(out_shape[1] * out_shape[2] * out_shape[3])
            else:
                var_count[L.id] = 0
        elif L.kind == 'RELU':
            preds = getattr(L, 'preds', []) if hasattr(L, 'preds') else []
            # Output dim = predecessor dim
            var_count[L.id] = 0   # filled in by consumer
        else:
            var_count[L.id] = 0
    return var_count


def estimate_conv2d_sparse_nnz(weight_shape: Tuple[int, int, int, int],
                                                       input_shape: Tuple[int, int, int, int],
                                                       stride: Tuple[int, int] = (1, 1),
                                                       padding: Tuple[int, int] = (0, 0)) -> Tuple[int, int, int]:
    """Estimate (n_out, n_in, nnz) of sparse CONV2D affine matrix.

    For each output element, contributes up to Cin*kH*kW nonzeros.
    """
    Cout, Cin, kH, kW = weight_shape
    _, _, Hin, Win = input_shape
    sh, sw = stride
    ph, pw = padding
    Hout = (Hin + 2 * ph - kH) // sh + 1
    Wout = (Win + 2 * pw - kW) // sw + 1
    n_out = Cout * Hout * Wout
    n_in = Cin * Hin * Win
    # Each output has up to Cin*kH*kW nonzeros (less near boundaries due to padding)
    nnz_per_out = Cin * kH * kW
    nnz = n_out * nnz_per_out
    return n_out, n_in, nnz


def estimate_build_memory(net,
                                          K_per_layer: int = 20,
                                          assume_all_relu_unstable: bool = False) -> Dict:
    """Estimate peak memory for build_forward_milp.

    Args:
      net: ACT net
      K_per_layer: binary indicators per RELU
      assume_all_relu_unstable: pessimistic case

    Returns:
      dict with keys:
        total_vars, total_eq_rows, total_ub_rows
        conv2d_blocks: list of (n_out, n_in, nnz) per CONV2D layer
        peak_sparse_bytes: estimated sparse storage in bytes
        peak_dense_bytes: what dense storage would cost
        recommendation: 'OK' / 'SPARSE_REQUIRED' / 'TOO_LARGE'
    """
    total_vars = 0
    total_eq_rows = 0
    total_ub_rows = 0
    conv2d_info = []
    relu_unstable_estimates = []

    for L in net.layers:
        if L.kind == 'INPUT':
            in_shape = L.params.get('shape', (1, 1))
            n_in_vars = int(np.prod(in_shape[1:])) if len(in_shape) > 1 else int(in_shape[0])
            total_vars += n_in_vars
        elif L.kind == 'DENSE':
            W = L.params.get('weight')
            if W is None: continue
            n_out_d = int(W.shape[0])
            total_vars += n_out_d
            total_eq_rows += n_out_d   # y = W @ x + b equality rows
        elif L.kind == 'CONV2D':
            W = L.params.get('weight')
            in_shape = L.params.get('input_shape')
            stride = L.params.get('stride', (1, 1))
            padding = L.params.get('padding', (0, 0))
            if W is None or in_shape is None: continue
            w_shape = tuple(W.shape) if hasattr(W, 'shape') else None
            if w_shape is None or len(w_shape) != 4: continue
            n_out_c, n_in_c, nnz_c = estimate_conv2d_sparse_nnz(w_shape, tuple(in_shape), stride, padding)
            total_vars += n_out_c
            total_eq_rows += n_out_c   # sparse equality block
            conv2d_info.append({
                'L_id': L.id,
                'n_out': n_out_c,
                'n_in': n_in_c,
                'nnz': nnz_c,
                'density_pct': round(100 * nnz_c / (n_out_c * n_in_c) if n_in_c else 0, 3),
            })
        elif L.kind == 'RELU':
            # Output dim = predecessor dim — estimate from prev layer
            n_relu = 0
            for prev_L in net.layers:
                if prev_L.id in getattr(L, 'preds', []) or L.id == prev_L.id + 1:
                    if prev_L.kind == 'DENSE':
                        W = prev_L.params.get('weight')
                        if W is not None: n_relu = int(W.shape[0])
                    elif prev_L.kind == 'CONV2D':
                        os_ = prev_L.params.get('output_shape')
                        if os_ and len(os_) >= 4: n_relu = int(os_[1] * os_[2] * os_[3])
                    break
            if n_relu == 0: continue
            total_vars += n_relu
            n_unstable_est = n_relu if assume_all_relu_unstable else min(n_relu, max(K_per_layer * 3, n_relu // 5))
            relu_unstable_estimates.append({'L_id': L.id, 'n': n_relu, 'unstable_est': n_unstable_est})
            # Per unstable neuron: 2 triangle ub rows + (if binary) 4 bigM ub rows = up to 6 ub rows
            total_ub_rows += 6 * n_unstable_est
            # Binary vars (K per layer)
            total_vars += min(K_per_layer, n_unstable_est)

    # Sparse storage: each nnz = 8 bytes (float64) + 4 bytes (int32 row) + 4 bytes (int32 col)
    bytes_per_nnz_sparse = 16
    sparse_eq_nnz = sum(b['nnz'] for b in conv2d_info) + total_eq_rows   # conv blocks + identity rows
    sparse_ub_nnz = total_ub_rows * 4   # avg 4 nonzeros per ub row
    peak_sparse_bytes = (sparse_eq_nnz + sparse_ub_nnz) * bytes_per_nnz_sparse

    # Dense storage: all_rows * total_vars * 8 bytes
    peak_dense_bytes = (total_eq_rows + total_ub_rows) * total_vars * 8

    # Recommendation
    if peak_dense_bytes < 100 * 1024 * 1024:   # < 100 MB
        rec = 'OK_DENSE'
    elif peak_sparse_bytes < 4 * 1024**3:   # < 4 GB
        rec = 'SPARSE_REQUIRED'
    else:
        rec = 'TOO_LARGE'

    return {
        'total_vars': total_vars,
        'total_eq_rows': total_eq_rows,
        'total_ub_rows': total_ub_rows,
        'conv2d_blocks': conv2d_info,
        'relu_layers': relu_unstable_estimates,
        'sparse_eq_nnz': sparse_eq_nnz,
        'sparse_ub_nnz': sparse_ub_nnz,
        'peak_sparse_bytes': peak_sparse_bytes,
        'peak_dense_bytes': peak_dense_bytes,
        'peak_sparse_mb': round(peak_sparse_bytes / 1024**2, 1),
        'peak_dense_gb': round(peak_dense_bytes / 1024**3, 2),
        'recommendation': rec,
    }


def fmt_mb(b): return f"{b / 1024**2:.1f}MB"


__all__ = ['estimate_layer_var_count', 'estimate_conv2d_sparse_nnz', 'estimate_build_memory', 'fmt_mb']
