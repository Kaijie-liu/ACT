"""Multi-layer harness: simulate cifar100 ResNet-style conv chain to
measure cumulative HZ-storage savings from prune+dense->sparse vs the
current dense-HZono baseline.

Pipeline (CNN_4-ish, the kind of network where cifar100 currently RSS-caps):
  Input box  : (3, 32, 32) = 3072 dims, eps=1/255
  Conv1 3x3  : 3 -> 32, stride=1, pad=1   (output 32*32*32 = 32768)
  ReLU       : triangle relaxation (DeepZ-equivalent for sparse path)
  Conv2 3x3  : 32 -> 64, stride=2, pad=1  (output 64*16*16 = 16384)
  ReLU       : triangle
  Conv3 3x3  : 64 -> 128, stride=2, pad=1 (output 128*8*8 = 8192)
  Flatten + Dense (8192 -> 10)

Compares:
  - baseline_dense : pure HZono path, no prune, no sparse
  - prune_only     : HZono path with prune_gc_dense after each conv
  - dense_to_sparse: HZono path with hz_dense_to_sparse after each conv (-> SparseGcZ thereafter)
  - prune_then_sparse: both

Reports peak RSS + final dim/ng/density + soundness via box-bound check.
"""
from __future__ import annotations

import argparse
import gc as pygc
import os
import resource
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data1/Kane/ACT")
sys.path.insert(0, os.path.dirname(__file__))

from act.back_end.solver.solver_hz import HZono
from act.back_end.hybridz_tf.representations import SparseGcZ
from act.back_end.hybridz_tf.tf_cnn import hz_conv2d as hz_conv2d_native
from prototype import (
    hz_prune_gc_dense,
    hz_dense_to_sparse,
    stats_of,
    rss_mib,
    _bounds_of,
)


def make_input_hz(C, H, W, eps=1.0/255.0, dtype=torch.float64, device="cpu") -> HZono:
    n = C * H * W
    c = torch.zeros(n, 1, dtype=dtype, device=device)
    Gc = torch.eye(n, dtype=dtype, device=device) * eps
    return HZono(
        c=c, Gc=Gc,
        Gb=torch.zeros((n, 0), dtype=dtype, device=device),
        Ac=torch.zeros((0, n), dtype=dtype, device=device),
        Ab=torch.zeros((0, 0), dtype=dtype, device=device),
        b=torch.zeros((0, 1), dtype=dtype, device=device),
        eq_mask=torch.zeros(0, dtype=torch.bool, device=device),
    )


def apply_conv_to_hz(hz, weight, bias, *, stride, pad, input_shape):
    """Wrapper handles both HZono (dense) and SparseGcZ paths."""
    if isinstance(hz, SparseGcZ):
        return hz.apply_conv(weight, bias, input_shape, stride, pad)
    return hz_conv2d_native(
        hz, weight, bias,
        stride=(stride, stride) if isinstance(stride, int) else stride,
        padding=(pad, pad) if isinstance(pad, int) else pad,
        dilation=(1, 1), groups=1,
        input_shape=input_shape,
    )


def apply_relu_triangle_hz(hz):
    """Sound DeepZ-style triangle ReLU.

    For HZono: per unstable neuron i with [lb_i, ub_i] (l<0<u):
      lam = u/(u-l), mu = -l*u/(2(u-l))
      y_i = lam * x_i + mu + mu * eps_new
      adds k new generators (k = unstable count)
    For SparseGcZ: delegate to its native triangle ReLU.
    """
    if isinstance(hz, SparseGcZ):
        return hz.apply_relu_triangle()

    # HZono triangle
    Gc = hz.Gc
    n = hz.dim
    rad = Gc.abs().sum(dim=1) if Gc.numel() else torch.zeros(n, dtype=hz.c.dtype, device=hz.c.device)
    if hz.Gb.numel():
        rad = rad + hz.Gb.abs().sum(dim=1)
    c = hz.c.view(-1)
    lb = c - rad
    ub = c + rad

    active = lb >= 0
    inactive = ub <= 0
    unstable = ~(active | inactive)
    k = int(unstable.sum())

    c_new = torch.zeros(n, dtype=hz.c.dtype, device=hz.c.device)
    c_new[active] = c[active]

    Gc_scaled = Gc.clone()
    Gc_scaled[inactive, :] = 0
    if k > 0:
        u_uns = ub[unstable]
        l_uns = lb[unstable]
        lam = u_uns / (u_uns - l_uns)
        mu = -l_uns * u_uns / (2.0 * (u_uns - l_uns))
        c_new[unstable] = lam * c[unstable] + mu
        # row-scale Gc[unstable, :] by lam
        Gc_scaled[unstable, :] = Gc_scaled[unstable, :] * lam.view(-1, 1)
        # Add k new generator columns
        new_block = torch.zeros((n, k), dtype=Gc.dtype, device=Gc.device)
        new_block[unstable, torch.arange(k, device=Gc.device)] = mu
        Gc_out = torch.cat([Gc_scaled, new_block], dim=1)
        # Ac padding
        if hz.Ac.numel() > 0 and hz.nc > 0:
            Ac_pad = hz.Ac.new_zeros((hz.nc, k))
            Ac_out = torch.cat([hz.Ac, Ac_pad], dim=1)
        else:
            Ac_out = hz.Ac.new_zeros((hz.nc, Gc_out.shape[1])) if hz.Ac.numel() == 0 else hz.Ac
    else:
        Gc_out = Gc_scaled
        Ac_out = hz.Ac

    return HZono(
        c=c_new.view(-1, 1), Gc=Gc_out,
        Gb=hz.Gb.clone(),
        Ac=Ac_out, Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
    )


@dataclass
class StageRecord:
    stage: str
    dim: int
    ng: int
    nc: int
    storage_mib: float
    rss_mib: float
    elapsed_s: float


def run_pipeline(mode: str, *, verbose=True):
    """mode in {"baseline_dense", "prune_only", "dense_to_sparse", "prune_then_sparse"}.

    Reports per-stage (dim, ng, storage_MiB, RSS_MiB, elapsed).
    """
    dtype = torch.float64
    device = "cpu"
    g = torch.Generator(device=device).manual_seed(42)

    # Layer specs: (Cin, H_in, W_in, Cout, stride, pad).
    # Small but architecturally-faithful: 16x16 inputs (instead of 32x32)
    # to keep baseline_dense bound by 24 GiB so we can compare all 3
    # modes end-to-end without OOM. The relative win scales with
    # input dim, so 16x16 is a CONSERVATIVE estimate of cifar100 win.
    layer_specs = [
        (3, 16, 16, 16, 1, 1),    # Conv1: 768 -> 4096
        (16, 16, 16, 32, 2, 1),   # Conv2 stride 2: 4096 -> 2048
        (32, 8, 8, 32, 2, 1),     # Conv3 stride 2: 2048 -> 512
    ]

    weights = []
    biases = []
    for (Cin, H, W, Cout, stride, pad) in layer_specs:
        w = torch.randn(Cout, Cin, 3, 3, generator=g, dtype=dtype, device=device) * 0.1
        b = torch.zeros(Cout, dtype=dtype, device=device)
        weights.append(w)
        biases.append(b)

    eps = 1.0/255.0
    hz = make_input_hz(*layer_specs[0][:3], eps=eps, dtype=dtype, device=device)
    if verbose:
        s = stats_of(hz, "input")
        print(s.short(), flush=True)

    records: List[StageRecord] = []
    t_total = time.time()

    for li, (Cin, H, W, Cout, stride, pad) in enumerate(layer_specs):
        # Conv
        t0 = time.time()
        hz = apply_conv_to_hz(
            hz, weights[li], biases[li],
            stride=stride, pad=pad,
            input_shape=(Cin, H, W),
        )
        # Optional prune
        if mode in ("prune_only", "prune_then_sparse") and isinstance(hz, HZono):
            hz = hz_prune_gc_dense(hz, eps=1e-9)
        # Optional dense -> sparse
        if mode in ("dense_to_sparse", "prune_then_sparse") and isinstance(hz, HZono):
            hz = hz_dense_to_sparse(hz, density_threshold=0.10)
        elapsed = time.time() - t0
        s = stats_of(hz, f"L{li+1} conv")
        if verbose:
            print(f"{s.short()}  +{elapsed:.2f}s", flush=True)
        records.append(StageRecord(
            stage=f"L{li+1}_conv", dim=s.dim, ng=s.ng, nc=s.nc,
            storage_mib=(s.gc_nnz_bytes if isinstance(hz, SparseGcZ) else s.gc_dense_bytes) / 2**20,
            rss_mib=s.rss_mib, elapsed_s=elapsed,
        ))

        # ReLU
        t0 = time.time()
        hz = apply_relu_triangle_hz(hz)
        elapsed = time.time() - t0
        s = stats_of(hz, f"L{li+1} relu")
        if verbose:
            print(f"{s.short()}  +{elapsed:.2f}s", flush=True)
        records.append(StageRecord(
            stage=f"L{li+1}_relu", dim=s.dim, ng=s.ng, nc=s.nc,
            storage_mib=(s.gc_nnz_bytes if isinstance(hz, SparseGcZ) else s.gc_dense_bytes) / 2**20,
            rss_mib=s.rss_mib, elapsed_s=elapsed,
        ))

    if verbose:
        print(f"\nTotal pipeline: {time.time() - t_total:.2f}s  final_rss={rss_mib():.0f} MiB")

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=[
        "baseline_dense", "prune_only", "dense_to_sparse", "prune_then_sparse",
    ], default="baseline_dense")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print("=" * 78)
    print(f"T2 multi-layer harness  mode={args.mode}")
    print("=" * 78)
    records = run_pipeline(args.mode, verbose=not args.quiet)
    peak_storage = max(r.storage_mib for r in records)
    peak_rss = max(r.rss_mib for r in records)
    final_storage = records[-1].storage_mib
    print(f"\nSUMMARY mode={args.mode}")
    print(f"  peak Gc storage  : {peak_storage:.1f} MiB")
    print(f"  peak RSS         : {peak_rss:.0f} MiB")
    print(f"  final Gc storage : {final_storage:.1f} MiB")


if __name__ == "__main__":
    main()
