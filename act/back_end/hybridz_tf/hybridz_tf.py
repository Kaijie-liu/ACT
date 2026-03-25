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
from typing import Dict, List, Optional
from act.back_end.core import Bounds, Fact, Layer, Net, ConSet
from act.back_end.transfer_functions import TransferFunction
from act.back_end.hybridz_tf.tf_mlp import *
from act.back_end.hybridz_tf.tf_cnn import *
from act.back_end.hybridz_tf.tf_rnn import *
from act.back_end.hybridz_tf.tf_transformer import *


@dataclass
class HZono:
    """Hybrid Zonotope data container."""
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
        self._net: Optional[Net] = None
        self._before: Optional[Dict[int, Fact]] = None
        self._after: Optional[Dict[int, Fact]] = None
        self._tanh_K: int = 2      # Number of piecewise segments for tanh
        self._sigmoid_K: int = 2   # Number of piecewise segments for sigmoid

    # Layer kind to function mapping for HybridZ operations
    _LAYER_REGISTRY = {
        # Identity/constraint layers
        "INPUT": lambda L, bounds, tf: Fact(bounds=bounds, cons=ConSet()),
        "INPUT_SPEC": lambda L, bounds, tf: Fact(bounds=bounds, cons=ConSet()),
        "ASSERT": lambda L, bounds, tf: Fact(bounds=bounds, cons=ConSet()),
        
        # MLP operations (with HybridZ precision)
        "DENSE": lambda L, bounds, tf: hybridz_tf_dense(L, bounds, tf),
        "BIAS": lambda L, bounds, tf: hybridz_tf_bias(L, bounds, tf),
        "SCALE": lambda L, bounds, tf: hybridz_tf_scale(L, bounds, tf),
        "RELU": lambda L, bounds, tf: hybridz_tf_relu(L, bounds, tf),
        "LRELU": lambda L, bounds, tf: hybridz_tf_lrelu(L, bounds, tf),
        "TANH": lambda L, bounds, tf: hybridz_tf_tanh(L, bounds, tf),
        "SIGMOID": lambda L, bounds, tf: hybridz_tf_sigmoid(L, bounds, tf),
        "ABS": lambda L, bounds, tf: hybridz_tf_abs(L, bounds, tf),
        
        # Multi-input operations
        "ADD": lambda L, bounds, tf: hybridz_tf_add(L,
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
            tf),
        "MUL": lambda L, bounds, tf: hybridz_tf_mul(L,
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
            tf),
        
        # CNN operations
        "CONV2D": lambda L, bounds, tf: hybridz_tf_conv2d(L, bounds, tf),
        "MAXPOOL2D": lambda L, bounds, tf: hybridz_tf_maxpool2d(L, bounds, tf),
        "AVGPOOL2D": lambda L, bounds, tf: hybridz_tf_avgpool2d(L, bounds, tf),
        "FLATTEN": lambda L, bounds, tf: hybridz_tf_flatten(L, bounds, tf),
        "RESHAPE": lambda L, bounds, tf: hybridz_tf_reshape(L, bounds, tf),
        
        # RNN operations
        "LSTM": lambda L, bounds, tf: hybridz_tf_lstm(L, bounds, tf),
        "GRU": lambda L, bounds, tf: hybridz_tf_gru(L, bounds, tf),
        "RNN": lambda L, bounds, tf: hybridz_tf_rnn(L, bounds, tf),
        "EMBEDDING": lambda L, bounds, tf: hybridz_tf_embedding(L, bounds, tf),
        
        # Transformer operations
        "LAYERNORM": lambda L, bounds, tf: hybridz_tf_layernorm(L, bounds, tf),
        "GELU": lambda L, bounds, tf: hybridz_tf_gelu(L, bounds, tf),
        "SOFTMAX": lambda L, bounds, tf: hybridz_tf_softmax(L, bounds, tf),
        "POSENC": lambda L, bounds, tf: hybridz_tf_posenc(L, bounds, tf),
    }
    
    @property
    def name(self) -> str:
        return "HybridzTF"
    
    def supports_layer(self, layer_kind: str) -> bool:
        """Check if HybridZ supports this layer kind."""
        return layer_kind.upper() in self._LAYER_REGISTRY
    
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
        """Apply HybridZ transfer function to layer L."""
        k = L.kind.upper()
        if k not in self._LAYER_REGISTRY:
            raise NotImplementedError(f"HybridzTF: Unsupported layer kind '{k}'")
        
        # Reset cache if network changed
        net_id = id(net)
        if self._cache_net_id != net_id:
            self._hz_cache.clear()
            self._cache_net_id = net_id
        
        # Store context for lambdas
        self._net = net
        self._before = before
        self._after = after
        
        # For INPUT/INPUT_SPEC: create initial HZ from bounds and cache it
        if k in ("INPUT", "INPUT_SPEC"):
            hz_init = self._hz_from_bounds(input_bounds)
            if hz_init is not None:
                self._hz_cache[L.id] = hz_init
        
        # For non-identity layers: look up input HZ from predecessor
        if k not in ("INPUT", "INPUT_SPEC", "ASSERT"):
            preds = net.preds.get(L.id, [])
            if preds and preds[0] in self._hz_cache:
                self._hz_cache[L.id] = self._hz_cache[preds[0]]
        
        transfer_fn = self._LAYER_REGISTRY[k]
        return transfer_fn(L, input_bounds, self)