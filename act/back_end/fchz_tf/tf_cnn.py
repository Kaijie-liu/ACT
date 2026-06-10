# ===- act/back_end/fchz_tf/tf_cnn.py - FCHZ CNN Transfer Functions -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   FCHZ CNN-layer transfer functions: CONV2D, BN, MAXPOOL2D, etc.
#
#   Sound propagation:
#     - Conv2D: c' = Conv(c), G' = Conv(G), tail' = Conv(|W|, tail)
#     - BN:     c' = α*c + β, G' = α*G, tail' = |α|*tail
#     - MaxPool: box relaxation (sound, lose G refinement)
#
#   Sparse-slack compression applied after Conv2D when K exceeds budget.
#
# ===---------------------------------------------------------------------===#

"""FCHZ CNN layer transfer functions."""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from act.back_end.fchz_tf.representations import FCHZState


def _normalize_pair(v, default):
    if v is None: return (default, default)
    if isinstance(v, int): return (v, v)
    if isinstance(v, (tuple, list)) and len(v) == 1:
        return (int(v[0]), int(v[0]))
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return (int(v[0]), int(v[1]))
    if isinstance(v, (tuple, list)) and len(v) == 4:
        # ONNX pad format (top, left, bottom, right) → just use first
        return (int(v[0]), int(v[1]))
    return (default, default)


def apply_conv2d(state: FCHZState, W: np.ndarray, b: Optional[np.ndarray],
                       input_shape: Tuple[int, int, int],
                       stride=(1, 1), padding=(0, 0), groups: int = 1) -> FCHZState:
    """Apply Conv2D to FCHZState.

    Args:
        state: input state with c.shape[0] = Ci*Hi*Wi
        W: kernel (Co, Ci/groups, kH, kW)
        b: bias (Co,) optional
        input_shape: (Ci, Hi, Wi)
        stride, padding: as ONNX
        groups: groups for depthwise

    Returns:
        new FCHZState with c.shape[0] = Co*Ho*Wo
    """
    import torch
    import torch.nn.functional as F

    Ci, Hi, Wi = input_shape
    n_in = Ci * Hi * Wi
    if state.c.shape[0] != n_in:
        raise ValueError(f"State n={state.c.shape[0]} != Ci*Hi*Wi={n_in}")

    # Pad / stride
    sh, sw = _normalize_pair(stride, 1)
    ph, pw = _normalize_pair(padding, 0)

    use_cuda = (os.environ.get('HYZOR_FCHZ_USE_CUDA', '0') == '1'
                       and torch.cuda.is_available())
    dev = torch.device('cuda') if use_cuda else torch.device('cpu')

    W_t = torch.from_numpy(W).to(torch.float64).to(dev)
    b_t = (torch.from_numpy(np.asarray(b).astype(np.float64).reshape(-1)).to(dev)
              if b is not None else None)

    # Center
    c_in = torch.from_numpy(state.c.reshape(1, Ci, Hi, Wi)).to(torch.float64).to(dev)
    c_out = F.conv2d(c_in, W_t, b_t, stride=(sh, sw), padding=(ph, pw), groups=groups)
    new_c = c_out.detach().cpu().numpy().reshape(-1)
    _, Co_p, Ho_p, Wo_p = c_out.shape

    # G in chunks — use float64 for parity with raw walker (advisor M1-fix)
    K_st = state.G.shape[1]
    n_out = Co_p * Ho_p * Wo_p
    new_G = np.zeros((n_out, K_st), dtype=np.float64)
    if K_st > 0:
        bytes_per_col = n_out * 8  # float64
        chunk_max = max(32, min(K_st, int(2.5e8 // max(bytes_per_col, 1))))
        G_T = state.G.T.astype(np.float64, copy=False)
        for start in range(0, K_st, chunk_max):
            end = min(K_st, start + chunk_max)
            chunk = G_T[start:end].reshape(end - start, Ci, Hi, Wi)
            G_chunk_t = torch.from_numpy(chunk).to(torch.float64).to(dev)
            G_out_chunk = F.conv2d(G_chunk_t, W_t, None,
                                              stride=(sh, sw), padding=(ph, pw), groups=groups)
            new_G[:, start:end] = G_out_chunk.detach().cpu().numpy().reshape(end - start, -1).T
            del G_chunk_t, G_out_chunk
        del G_T

    # Tail — also float64 (advisor M1-fix)
    new_tail = None
    if state.tail_radius is not None:
        abs_W_t = torch.abs(W_t)  # W_t already float64
        tail_in = torch.from_numpy(state.tail_radius.astype(np.float64).reshape(1, Ci, Hi, Wi)).to(dev)
        tail_out = F.conv2d(tail_in, abs_W_t, None,
                                  stride=(sh, sw), padding=(ph, pw), groups=groups)
        new_tail = tail_out.detach().cpu().numpy().reshape(-1).astype(np.float64)

    return FCHZState(c=new_c, G=new_G, n_root=state.n_root,
                          slack_records=state.slack_records,
                          tail_radius=new_tail), (Co_p, Ho_p, Wo_p)


def apply_bn(state: FCHZState, scale: np.ndarray, bias: np.ndarray,
                 input_shape: Tuple[int, int, int]) -> FCHZState:
    """Apply BatchNorm to FCHZState.

    Per-channel: y = scale * x + bias
    """
    Ci, Hi, Wi = input_shape
    n = Ci * Hi * Wi
    assert state.c.shape[0] == n, f"State {state.c.shape[0]} != {n}"
    # Per-channel broadcast: shape (Ci, 1, 1) → flat (Ci, Hi*Wi)
    a_full = np.repeat(scale.astype(np.float64), Hi * Wi)
    b_full = np.repeat(bias.astype(np.float64), Hi * Wi)
    new_c = a_full * state.c + b_full
    new_G = state.G * a_full[:, None]
    new_tail = (np.abs(a_full) * state.tail_radius) if state.tail_radius is not None else None
    return FCHZState(c=new_c, G=new_G, n_root=state.n_root,
                          slack_records=state.slack_records,
                          tail_radius=new_tail)


def apply_maxpool2d(state: FCHZState, kernel: Tuple[int, int],
                            stride: Tuple[int, int], padding: Tuple[int, int],
                            input_shape: Tuple[int, int, int]) -> FCHZState:
    """Sound MaxPool relaxation: box over each window. G linkage destroyed.

    Output per channel/pixel: in interval [max(lb in window), max(ub in window)].
    """
    Ci, Hi, Wi = input_shape
    kH, kW = kernel; sH, sW = stride; pH, pW = padding
    Ho = (Hi + 2*pH - kH) // sH + 1
    Wo = (Wi + 2*pW - kW) // sW + 1

    rad_in = np.abs(state.G).sum(axis=1)
    if state.tail_radius is not None:
        rad_in = rad_in + state.tail_radius
    c_in_img = state.c.reshape(Ci, Hi, Wi)
    rad_in_img = rad_in.reshape(Ci, Hi, Wi)

    if pH or pW:
        c_in_pad = np.pad(c_in_img, ((0,0),(pH,pH),(pW,pW)), constant_values=-1e30)
        rad_in_pad = np.pad(rad_in_img, ((0,0),(pH,pH),(pW,pW)), constant_values=0)
    else:
        c_in_pad = c_in_img; rad_in_pad = rad_in_img

    new_c_arr = np.zeros((Ci, Ho, Wo))
    new_tail_arr = np.zeros((Ci, Ho, Wo))
    for ho in range(Ho):
        for wo in range(Wo):
            hs = ho * sH; ws = wo * sW
            win_c = c_in_pad[:, hs:hs+kH, ws:ws+kW]
            win_r = rad_in_pad[:, hs:hs+kH, ws:ws+kW]
            win_lo = (win_c - win_r).reshape(Ci, -1)
            win_hi = (win_c + win_r).reshape(Ci, -1)
            out_lo = win_lo.max(axis=1)
            out_hi = win_hi.max(axis=1)
            new_c_arr[:, ho, wo] = (out_lo + out_hi) / 2
            new_tail_arr[:, ho, wo] = (out_hi - out_lo) / 2

    new_c = new_c_arr.reshape(-1)
    K_st = state.G.shape[1]
    new_G = np.zeros((Ci*Ho*Wo, K_st)) if K_st > 0 else state.G

    return FCHZState(c=new_c, G=new_G, n_root=state.n_root,
                          slack_records=state.slack_records,
                          tail_radius=new_tail_arr.reshape(-1)), (Ci, Ho, Wo)


# Module-level import of os (used by GPU flag)
import os
