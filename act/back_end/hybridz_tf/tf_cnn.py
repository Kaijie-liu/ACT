#===- act/back_end/hybridz_tf/tf_cnn.py - HybridZ CNN Transfer Functions ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ CNN Transfer Functions. Implements HybridZ-based transfer functions
#   for CNN layers including convolution, pooling, and tensor reshaping
#   operations.
#
#===---------------------------------------------------------------------===#


from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, TYPE_CHECKING
from act.back_end.core import Bounds, Fact, Layer, ConSet

if TYPE_CHECKING:
    from act.back_end.hybridz_tf.hybridz_tf import HZono

from act.back_end.hybridz_tf.tf_mlp import (
    _hz_compute_bounds, _hz_from_bounds_fresh, _get_HZono,
)


def _parse_input_shape(input_shape):
    """Extract (C, H, W) from input_shape, which may be (N,C,H,W) or (C,H,W)."""
    if len(input_shape) == 4:
        _, C, H, W = input_shape
    elif len(input_shape) == 3:
        C, H, W = input_shape
    else:
        raise ValueError(f"Unexpected input_shape={input_shape}")
    return C, H, W


def _reshape_bounds_4d(lb, ub, in_shape):
    """Reshape flat/3D bounds to (1, C, H, W) for pooling/conv operations."""
    if lb.dim() == 1 and in_shape:
        return lb.view(1, *in_shape), ub.view(1, *in_shape)
    elif lb.dim() == 3:
        return lb.unsqueeze(0), ub.unsqueeze(0)
    return lb, ub


def _conv2d_generators(G, weight, C, H, W, stride, padding, dilation, groups, n_out, dtype, device):
    """Apply conv2d to a generator matrix (Gc or Gb). Returns convolved generator matrix."""
    if G.shape[1] == 0:
        return torch.zeros((n_out, 0), dtype=dtype, device=device)
    ncols = G.shape[1]
    imgs = G.t().contiguous().view(ncols, C, H, W)
    out = F.conv2d(imgs, weight, bias=None,
                   stride=stride, padding=padding, dilation=dilation, groups=groups)
    return out.permute(1, 2, 3, 0).contiguous().reshape(-1, ncols)


def _hz_conv2d(hz, weight, bias, stride, padding, dilation, groups, input_shape):
    """Apply conv2d to a hybrid zonotope: convolve center and each generator column."""
    dtype, device = hz.c.dtype, hz.c.device
    C, H, W = _parse_input_shape(input_shape)
    weight = weight.to(dtype=dtype, device=device)
    
    # Apply conv2d to center
    c_img = hz.c.view(C, H, W).unsqueeze(0)
    out_c = F.conv2d(c_img, weight,
                     bias=bias.to(dtype=dtype, device=device) if bias is not None else None,
                     stride=stride, padding=padding, dilation=dilation, groups=groups)
    new_c = out_c.reshape(-1, 1)
    n_out = new_c.shape[0]
    
    new_Gc = _conv2d_generators(hz.Gc, weight, C, H, W, stride, padding, dilation, groups, n_out, dtype, device)
    new_Gb = _conv2d_generators(hz.Gb, weight, C, H, W, stride, padding, dilation, groups, n_out, dtype, device)
    
    # Constraints unchanged (conv2d is linear)
    return _get_HZono()(c=new_c, Gc=new_Gc, Gb=new_Gb,
                        Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone())


@torch.no_grad()
def hybridz_tf_conv2d(L: Layer, Bin: Bounds):
    """2D convolution. Returns Fact."""
    weight = L.params["weight"]
    bias = L.params.get("bias", None)
    stride = L.params.get("stride", 1)
    padding = L.params.get("padding", 0)
    dilation = L.params.get("dilation", 1)
    groups = L.params.get("groups", 1)
    input_shape = L.params.get("input_shape", None)
    
    if Bin.lb.dim() == 1:
        if input_shape is None:
            raise ValueError("CONV2D got flat bounds but params.input_shape is missing")
        C, H, W = _parse_input_shape(input_shape)
        Bin_reshaped_lb = Bin.lb.view(1, C, H, W)
        Bin_reshaped_ub = Bin.ub.view(1, C, H, W)
    elif Bin.lb.dim() == 3:
        Bin_reshaped_lb = Bin.lb.unsqueeze(0)
        Bin_reshaped_ub = Bin.ub.unsqueeze(0)
    else:
        Bin_reshaped_lb = Bin.lb
        Bin_reshaped_ub = Bin.ub
    
    weight_pos = torch.clamp(weight, min=0)
    weight_neg = torch.clamp(weight, max=0)
    
    lb_conv = F.conv2d(Bin_reshaped_lb, weight_pos, bias=None, stride=stride,
                       padding=padding, dilation=dilation, groups=groups)
    lb_conv += F.conv2d(Bin_reshaped_ub, weight_neg, bias=None, stride=stride,
                        padding=padding, dilation=dilation, groups=groups)
    ub_conv = F.conv2d(Bin_reshaped_ub, weight_pos, bias=None, stride=stride,
                       padding=padding, dilation=dilation, groups=groups)
    ub_conv += F.conv2d(Bin_reshaped_lb, weight_neg, bias=None, stride=stride,
                        padding=padding, dilation=dilation, groups=groups)
    
    if bias is not None:
        lb_conv += bias.view(1, -1, 1, 1)
        ub_conv += bias.view(1, -1, 1, 1)
    
    lb = lb_conv.reshape(-1)
    ub = ub_conv.reshape(-1)
    assert lb.numel() == len(L.out_vars)
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_op( f"conv2d:{L.id}", list(L.out_vars + L.in_vars), weight=weight,
                bias=bias if bias is not None else torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype),
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                input_shape=L.params.get("input_shape"), output_shape=L.params.get("output_shape"))
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_maxpool2d(L: Layer, Bin: Bounds):
    """2D max pooling. Returns Fact."""
    kernel_size = L.params.get("kernel_size", 2)
    stride = L.params.get("stride", kernel_size)
    padding = L.params.get("padding", 0)
    in_shape = L.params.get("input_shape")
    
    Bin_lb, Bin_ub = _reshape_bounds_4d(Bin.lb, Bin.ub, in_shape)
    lb_pool = F.max_pool2d(Bin_lb, kernel_size, stride=stride, padding=padding)
    ub_pool = F.max_pool2d(Bin_ub, kernel_size, stride=stride, padding=padding)
    lb = lb_pool.squeeze(0).flatten() if len(L.out_vars) != lb_pool.numel() else lb_pool.squeeze(0)
    ub = ub_pool.squeeze(0).flatten() if len(L.out_vars) != ub_pool.numel() else ub_pool.squeeze(0)
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_op( f"maxpool2d:{L.id}", list(L.out_vars + L.in_vars), kernel_size=kernel_size,
                stride=stride, padding=padding, input_shape=in_shape,
                output_shape=L.params.get("output_shape"))
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_avgpool2d(L: Layer, Bin: Bounds):
    """2D average pooling. Returns Fact."""
    kernel_size = L.params.get("kernel_size", 2)
    stride = L.params.get("stride", kernel_size)
    padding = L.params.get("padding", 0)
    in_shape = L.params.get("input_shape")
    
    Bin_lb, Bin_ub = _reshape_bounds_4d(Bin.lb, Bin.ub, in_shape)
    
    lb_pool = F.avg_pool2d(Bin_lb, kernel_size, stride=stride, padding=padding)
    ub_pool = F.avg_pool2d(Bin_ub, kernel_size, stride=stride, padding=padding)
    lb = lb_pool.squeeze(0).flatten() if len(L.out_vars) != lb_pool.numel() else lb_pool.squeeze(0)
    ub = ub_pool.squeeze(0).flatten() if len(L.out_vars) != ub_pool.numel() else ub_pool.squeeze(0)
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_op(f"avgpool2d:{L.id}", list(L.out_vars + L.in_vars), kernel_size=kernel_size,
                stride=stride, padding=padding, input_shape=in_shape,
                output_shape=L.params.get("output_shape"))
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_flatten(L: Layer, Bin: Bounds):
    """Tensor flattening (HZ pass-through). Returns Fact."""
    start_dim = L.params.get("start_dim", 1)
    end_dim = L.params.get("end_dim", -1)
    
    lb = Bin.lb.flatten()
    ub = Bin.ub.flatten()
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_op(f"flatten:{L.id}", list(L.out_vars + L.in_vars),
                start_dim=start_dim, end_dim=end_dim,
                input_shape=L.params.get("input_shape"),
                output_shape=L.params.get("output_shape"))
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_reshape(L: Layer, Bin: Bounds):
    """Tensor reshaping (HZ pass-through). Returns Fact."""
    target_shape = L.params.get("target_shape")
    
    lb = Bin.lb.reshape(target_shape).flatten() if target_shape else Bin.lb.flatten()
    ub = Bin.ub.reshape(target_shape).flatten() if target_shape else Bin.ub.flatten()
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_op(f"reshape:{L.id}", list(L.out_vars + L.in_vars),
                target_shape=target_shape, input_shape=L.params.get("input_shape"),
                output_shape=L.params.get("output_shape"))
    return Fact(bounds=Bout, cons=cons)