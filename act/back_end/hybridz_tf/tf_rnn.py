#===- act/back_end/hybridz_tf/tf_rnn.py - HybridZ RNN Transfer Functions ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ RNN Transfer Functions. Implements HybridZ-based transfer functions
#   for RNN layers including LSTM, GRU, and basic RNN cells.
#
#===---------------------------------------------------------------------===#

import torch
from act.back_end.core import Bounds, Fact, Layer, ConSet
from act.back_end.hybridz_tf.tf_mlp import _hz_from_bounds_fresh


@torch.no_grad()
def hybridz_tf_lstm(L: Layer, Bin: Bounds, tf=None):
    """LSTM cell (conservative)."""
    input_size = L.params.get("input_size")
    hidden_size = L.params.get("hidden_size")
    seq_len = L.params.get("seq_len", 1)
    batch_size = L.params.get("batch_size", 1)
    
    lb = torch.full((hidden_size,), -1.0, device=Bin.lb.device, dtype=Bin.lb.dtype)
    ub = torch.full((hidden_size,), 1.0, device=Bin.lb.device, dtype=Bin.lb.dtype)
    
    input_range = torch.max(Bin.ub) - torch.min(Bin.lb)
    if input_range < 2.0:
        scale_factor = min(1.0, input_range / 2.0)
        lb *= scale_factor
        ub *= scale_factor
    
    Bout = Bounds(lb=lb, ub=ub)
    if tf is not None:
        tf._hz_cache[L.id] = _hz_from_bounds_fresh(Bout, Bout.lb.dtype, Bout.lb.device)
    
    cons = ConSet()
    cons.add_op(f"lstm:{L.id}", list(L.out_vars + L.in_vars),
                input_size=input_size, hidden_size=hidden_size)
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_gru(L: Layer, Bin: Bounds, tf=None):
    """GRU cell (conservative)."""
    input_size = L.params.get("input_size")
    hidden_size = L.params.get("hidden_size")
    
    lb = torch.full((hidden_size,), -1.0, device=Bin.lb.device, dtype=Bin.lb.dtype)
    ub = torch.full((hidden_size,), 1.0, device=Bin.lb.device, dtype=Bin.lb.dtype)
    
    input_range = torch.max(Bin.ub) - torch.min(Bin.lb)
    if input_range < 2.0:
        scale_factor = min(1.0, input_range / 2.0)
        lb *= scale_factor
        ub *= scale_factor
    
    Bout = Bounds(lb=lb, ub=ub)
    if tf is not None:
        tf._hz_cache[L.id] = _hz_from_bounds_fresh(Bout, Bout.lb.dtype, Bout.lb.device)
    
    cons = ConSet()
    cons.add_op(f"gru:{L.id}", list(L.out_vars + L.in_vars),
                input_size=input_size, hidden_size=hidden_size)
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_rnn(L: Layer, Bin: Bounds, tf=None):
    """Basic RNN cell (conservative)."""
    input_size = L.params.get("input_size")
    hidden_size = L.params.get("hidden_size")
    nonlinearity = L.params.get("nonlinearity", "tanh")
    
    if nonlinearity == "tanh":
        lb = torch.full((hidden_size,), -1.0, device=Bin.lb.device, dtype=Bin.lb.dtype)
        ub = torch.full((hidden_size,), 1.0, device=Bin.lb.device, dtype=Bin.lb.dtype)
    elif nonlinearity == "relu":
        lb = torch.zeros((hidden_size,), device=Bin.lb.device, dtype=Bin.lb.dtype)
        input_max = torch.max(torch.abs(Bin.ub), torch.abs(Bin.lb))
        ub = torch.full((hidden_size,), float(input_max * 2), device=Bin.lb.device, dtype=Bin.lb.dtype)
    else:
        raise ValueError(f"Unsupported RNN nonlinearity: {nonlinearity}")
    
    Bout = Bounds(lb=lb, ub=ub)
    if tf is not None:
        tf._hz_cache[L.id] = _hz_from_bounds_fresh(Bout, Bout.lb.dtype, Bout.lb.device)
    
    cons = ConSet()
    cons.add_op(f"rnn:{L.id}", list(L.out_vars + L.in_vars),
                input_size=input_size, hidden_size=hidden_size, nonlinearity=nonlinearity)
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_embedding(L: Layer, Bin: Bounds, tf=None):
    """Embedding lookup (conservative)."""
    num_embeddings = L.params.get("num_embeddings")
    embedding_dim = L.params.get("embedding_dim")
    weight = L.params.get("weight")
    
    if weight is not None:
        lb = torch.min(weight, dim=0)[0]
        ub = torch.max(weight, dim=0)[0]
    else:
        lb = torch.full((embedding_dim,), -1.0, device=Bin.lb.device, dtype=Bin.lb.dtype)
        ub = torch.full((embedding_dim,), 1.0, device=Bin.lb.device, dtype=Bin.lb.dtype)
    
    Bout = Bounds(lb=lb, ub=ub)
    if tf is not None:
        tf._hz_cache[L.id] = _hz_from_bounds_fresh(Bout, Bout.lb.dtype, Bout.lb.device)
    
    cons = ConSet()
    cons.add_op(f"embedding:{L.id}", list(L.out_vars + L.in_vars),
                num_embeddings=num_embeddings, embedding_dim=embedding_dim, weight=weight)
    return Fact(bounds=Bout, cons=cons)