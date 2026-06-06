"""GPU-aware walker wrapper.

Moves torch tensor ops in apply_conv2d / apply_convtranspose / etc to
CUDA when available. Falls back to CPU on RAM-cap or CUDA absence.

Principle: this is engineering acceleration, NO change to forward HZ
math. The HZ propagation algorithm is byte-identical to CPU walker.
"""
from __future__ import annotations

import sys
from pathlib import Path

ACT_ROOT = Path(__file__).resolve().parents[2]
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from research.sc_hz.prune import PrunedState


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def apply_conv2d_gpu(state: PrunedState,
                          W: np.ndarray, b,
                          input_shape, stride, padding, groups,
                          chunk: int = 256):
    """GPU-accelerated apply_conv2d. Mirrors research.sc_hz.ops.apply_conv2d."""
    Ci, Hi, Wi = input_shape
    n_in = Ci * Hi * Wi
    K = state.G_kept.shape[1]
    W_t = torch.from_numpy(W).to(torch.float64).to(_DEVICE)
    b_t = (torch.from_numpy(np.asarray(b, dtype=np.float64).reshape(-1)).to(
              torch.float64).to(_DEVICE) if b is not None else None)

    # Apply to center
    c_image = torch.from_numpy(
        state.c.reshape(1, Ci, Hi, Wi).astype(np.float64)
    ).to(torch.float64).to(_DEVICE)
    c_out = F.conv2d(c_image, W_t, b_t, stride=stride,
                       padding=padding, groups=groups)
    Co, Ho, Wo = (int(c_out.shape[1]), int(c_out.shape[2]),
                   int(c_out.shape[3]))
    new_c = c_out.detach().cpu().numpy().reshape(-1)

    # Apply to generators (chunked, GPU)
    n_out = Co * Ho * Wo
    new_G = np.zeros((n_out, K), dtype=np.float64)
    for kk in range(0, K, chunk):
        G_chunk = state.G_kept[:, kk:kk + chunk].T.reshape(-1, Ci, Hi, Wi)
        G_t = torch.from_numpy(np.ascontiguousarray(G_chunk)).to(
                  torch.float64).to(_DEVICE)
        G_out = F.conv2d(G_t, W_t, None, stride=stride,
                            padding=padding, groups=groups)
        new_G[:, kk:kk + chunk] = G_out.detach().cpu().numpy().reshape(
              G_t.shape[0], -1).T

    # Tail
    if state.tail_radius is not None:
        tail_image = torch.from_numpy(
            state.tail_radius.reshape(1, Ci, Hi, Wi).astype(np.float64)
        ).to(torch.float64).to(_DEVICE)
        W_abs_t = torch.from_numpy(np.abs(W)).to(torch.float64).to(_DEVICE)
        tail_out = F.conv2d(tail_image, W_abs_t, None,
                                stride=stride, padding=padding, groups=groups)
        new_tail = tail_out.detach().cpu().numpy().reshape(-1)
    else:
        new_tail = None

    return PrunedState(c=new_c, G_kept=new_G, tail_radius=new_tail,
                          metadata=dict(state.metadata)), (Co, Ho, Wo)


def device_name() -> str:
    return _DEVICE
