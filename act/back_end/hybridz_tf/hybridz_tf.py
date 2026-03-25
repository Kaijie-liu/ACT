#===- act/back_end/hybridz_tf/hybridz_tf.py - HybridZ Transfer Function -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ Transfer Function Implementation. Implements the HybridzTF class
#   that provides zonotope-based transfer functions with enhanced precision
#   over interval methods.
#
#===---------------------------------------------------------------------===#

"""
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from act.back_end.core import Bounds, Fact, Layer, Net, ConSet
from act.back_end.transfer_functions import TransferFunction
from act.back_end.hybridz_tf.tf_mlp import *
from act.back_end.hybridz_tf.tf_cnn import *
from act.back_end.hybridz_tf.tf_rnn import *
from act.back_end.hybridz_tf.tf_transformer import *


@dataclass
class HZono:
    """Hybrid Zonotope data container.
    Represents the set {c + Gc·ξ_c + Gb·ξ_b | ξ_c ∈ [-1,1], ξ_b ∈ {-1,1}, Ac·ξ_c + Ab·ξ_b ≤ b}
    """
    c:  torch.Tensor   # (n, 1)   center vector
    Gc: torch.Tensor   # (n, ng)  continuous generator matrix
    Gb: torch.Tensor   # (n, nb)  binary generator matrix
    Ac: torch.Tensor   # (nc, ng) continuous constraint matrix
    Ab: torch.Tensor   # (nc, nb) binary constraint matrix
    b:  torch.Tensor   # (nc, 1)  constraint RHS vector


class HybridzTF(TransferFunction):
    """HybridZ-based transfer functions with zonotope operations."""

    def __init__(self):
        self._hz_cache: Dict[int, HZono] = {}
        self._cache_net_id: Optional[int] = None

    # ---- Registry: each function is pure (L, Bounds, hz_in) -> (Fact, hz_out) ----
    # Single-input ops
    _LAYER_REGISTRY = {
        # Identity/constraint layers — no HZ transformation
        "INPUT": None,
        "INPUT_SPEC": None,
        "ASSERT": None,

        # MLP operations
        "DENSE": hybridz_tf_dense,
        "BIAS": hybridz_tf_bias,
        "SCALE": hybridz_tf_scale,
        "RELU": hybridz_tf_relu,
        "LRELU": hybridz_tf_lrelu,
        "TANH": hybridz_tf_tanh,
        "SIGMOID": hybridz_tf_sigmoid,
        "ABS": hybridz_tf_abs,

        # CNN operations
        "CONV2D": hybridz_tf_conv2d,
        "MAXPOOL2D": hybridz_tf_maxpool2d,
        "AVGPOOL2D": hybridz_tf_avgpool2d,
        "FLATTEN": hybridz_tf_flatten,
        "RESHAPE": hybridz_tf_reshape,

        # RNN operations
        "LSTM": hybridz_tf_lstm,
        "GRU": hybridz_tf_gru,
        "RNN": hybridz_tf_rnn,
        "EMBEDDING": hybridz_tf_embedding,

        # Transformer operations
        "LAYERNORM": hybridz_tf_layernorm,
        "GELU": hybridz_tf_gelu,
        "SOFTMAX": hybridz_tf_softmax,
        "POSENC": hybridz_tf_posenc,
    }

    # Multi-input ops need special predecessor lookup
    _MULTI_INPUT_OPS = {
        "ADD": hybridz_tf_add,
        "MUL": hybridz_tf_mul,
    }

    @property
    def name(self) -> str:
        return "HybridzTF"

    def supports_layer(self, layer_kind: str) -> bool:
        """Check if HybridZ supports this layer kind."""
        k = layer_kind.upper()
        return k in self._LAYER_REGISTRY or k in self._MULTI_INPUT_OPS

    # Max input dimension for HZ tracking. Above this, fall back to interval
    # arithmetic to avoid memory issues (Gc diagonal is n×n).
    _HZ_MAX_INPUT_DIM = 1024

    def _hz_from_bounds(self, bounds: Bounds) -> Optional[HZono]:
        """Create initial HZ from Bounds: c=(lb+ub)/2, Gc=diag((ub-lb)/2).
        Returns None for large inputs to avoid O(n²) memory.
        """
        lb, ub = bounds.lb.flatten(), bounds.ub.flatten()
        n = lb.shape[0]
        if n > self._HZ_MAX_INPUT_DIM:
            return None
        dtype = lb.dtype
        device = lb.device
        c = ((lb + ub) / 2.0).view(-1, 1)
        rad = ((ub - lb) / 2.0)
        Gc = torch.diag(rad)
        Gb = torch.zeros((n, 0), dtype=dtype, device=device)
        Ac = torch.zeros((0, n), dtype=dtype, device=device)
        Ab = torch.zeros((0, 0), dtype=dtype, device=device)
        b  = torch.zeros((0, 1), dtype=dtype, device=device)
        return HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b)

    def apply(self, L: Layer, input_bounds: Bounds, net: Net,
              before: Dict[int, Fact], after: Dict[int, Fact]) -> Fact:
        """Apply HybridZ transfer function to layer L.

        Cache management is centralized here. Transfer functions are pure:
        they receive (L, Bounds, hz_in) and return (Fact, hz_out).
        """
        k = L.kind.upper()

        # Reset cache if network changed
        net_id = id(net)
        if self._cache_net_id != net_id:
            self._hz_cache.clear()
            self._cache_net_id = net_id

        # ---- Identity layers: pass through, optionally seed HZ cache ----
        if k in ("INPUT", "INPUT_SPEC", "ASSERT"):
            if k in ("INPUT", "INPUT_SPEC"):
                hz_init = self._hz_from_bounds(input_bounds)
                if hz_init is not None:
                    self._hz_cache[L.id] = hz_init
            return Fact(bounds=input_bounds, cons=ConSet())

        # ---- Resolve input HZ from predecessor ----
        preds = net.preds.get(L.id, [])
        hz_in = self._hz_cache.get(preds[0]) if preds else None

        # ---- Multi-input ops (ADD, MUL) ----
        if k in self._MULTI_INPUT_OPS:
            transfer_fn = self._MULTI_INPUT_OPS[k]
            Bin1 = net.get_predecessor_bounds(L.id, after, before, 0)
            Bin2 = net.get_predecessor_bounds(L.id, after, before, 1)
            hz_in2 = self._hz_cache.get(preds[1]) if len(preds) > 1 else None
            fact, hz_out = transfer_fn(L, Bin1, Bin2, hz_in, hz_in2)
            if hz_out is not None:
                self._hz_cache[L.id] = hz_out
            return fact

        # ---- Single-input ops ----
        if k not in self._LAYER_REGISTRY:
            raise NotImplementedError(f"HybridzTF: Unsupported layer kind '{k}'")

        transfer_fn = self._LAYER_REGISTRY[k]
        if transfer_fn is None:
            return Fact(bounds=input_bounds, cons=ConSet())

        fact, hz_out = transfer_fn(L, input_bounds, hz_in)
        if hz_out is not None:
            self._hz_cache[L.id] = hz_out
        return fact
