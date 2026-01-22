#===- act/back_end/hybridz_tf/tf_rnn.py - HybridZ RNN Transfer Functions ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ RNN Transfer Functions. For RNN/LSTM/GRU layers, we reuse the
#   validated interval transfer functions since:
#   1. Interval bounds for RNNs are already sound (verified with 840+ checks)
#   2. RNN recurrence is difficult to express as linear HybridZ constraints
#   3. For verification tasks, sound interval bounds are sufficient
#
#   This approach follows the principle of reusing verified components rather
#   than reimplementing complex logic that would be hard to verify.
#
#===---------------------------------------------------------------------===#

import torch
from typing import Tuple
from act.back_end.core import Bounds, Fact, Layer, ConSet, Con

# Import verified interval transfer functions
from act.back_end.interval_tf.tf_rnn import (
    tf_lstm as interval_tf_lstm,
    tf_gru as interval_tf_gru,
    tf_rnn as interval_tf_rnn,
    tf_embedding as interval_tf_embedding
)


@torch.no_grad()
def hybridz_tf_lstm(L: Layer, Bin: Bounds) -> Fact:
    """
    HybridZ transfer function for LSTM layers.

    Strategy: Reuse the validated interval transfer function for LSTM.
    The interval TF computes sound bounds through:
    - Multi-layer support (arbitrary num_layers)
    - Bidirectional processing
    - LSTM projection (proj_size > 0)
    - Gate-level bounds (sigmoid for i/f/o, tanh for g/c)

    Returns interval bounds + HybridZ-compatible constraint metadata.
    """
    # Use validated interval TF for bounds computation
    fact = interval_tf_lstm(L, Bin)

    # Extract metadata for HybridZ constraint system
    input_size = L.meta['input_size']
    hidden_size = L.meta['hidden_size']
    num_layers = L.meta['num_layers']
    bidirectional = L.meta['bidirectional']
    proj_size = L.meta.get('proj_size', 0)

    # Create HybridZ constraint with LSTM metadata
    # The "INEQ" tag indicates this is a general inequality constraint
    # HybridZ solver will use the bounds (not symbolic expansion)
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"lstm:{L.id}",
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "proj_size": proj_size,
        "method": "interval_bounds",  # Indicate we use interval analysis
        "note": "Sound bounds from validated interval TF"
    }))

    # Add box constraints from interval bounds
    C.add_box(L.id, L.out_vars, fact.bounds)

    return Fact(fact.bounds, C)


@torch.no_grad()
def hybridz_tf_gru(L: Layer, Bin: Bounds) -> Fact:
    """
    HybridZ transfer function for GRU layers.

    Strategy: Reuse the validated interval transfer function for GRU.
    The interval TF handles:
    - Multi-layer GRU stacks
    - Bidirectional processing
    - Reset, update, and new gates with proper bounds

    Returns interval bounds + HybridZ-compatible constraint metadata.
    """
    # Use validated interval TF for bounds computation
    fact = interval_tf_gru(L, Bin)

    # Extract metadata for HybridZ
    input_size = L.meta['input_size']
    hidden_size = L.meta['hidden_size']
    num_layers = L.meta['num_layers']
    bidirectional = L.meta['bidirectional']

    # Create HybridZ constraint with GRU metadata
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"gru:{L.id}",
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "method": "interval_bounds",
        "note": "Sound bounds from validated interval TF"
    }))

    C.add_box(L.id, L.out_vars, fact.bounds)

    return Fact(fact.bounds, C)


@torch.no_grad()
def hybridz_tf_rnn(L: Layer, Bin: Bounds) -> Fact:
    """
    HybridZ transfer function for basic RNN layers.

    Strategy: Reuse the validated interval transfer function for RNN.
    The interval TF handles:
    - Multi-layer RNN stacks
    - Bidirectional processing
    - Nonlinearity (tanh or relu)

    Returns interval bounds + HybridZ-compatible constraint metadata.
    """
    # Use validated interval TF for bounds computation
    fact = interval_tf_rnn(L, Bin)

    # Extract metadata for HybridZ
    input_size = L.meta['input_size']
    hidden_size = L.meta['hidden_size']
    num_layers = L.meta['num_layers']
    bidirectional = L.meta['bidirectional']
    nonlinearity = L.meta.get('nonlinearity', 'tanh')

    # Create HybridZ constraint with RNN metadata
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"rnn:{L.id}",
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "nonlinearity": nonlinearity,
        "method": "interval_bounds",
        "note": "Sound bounds from validated interval TF"
    }))

    C.add_box(L.id, L.out_vars, fact.bounds)

    return Fact(fact.bounds, C)


@torch.no_grad()
def hybridz_tf_embedding(L: Layer, Bin: Bounds) -> Fact:
    """
    HybridZ transfer function for embedding layers.

    Strategy: Reuse the validated interval transfer function for embeddings.
    Embeddings are straightforward lookup operations with bounds determined
    by the min/max values in the embedding table.

    Returns interval bounds + HybridZ-compatible constraint metadata.
    """
    # Use validated interval TF for bounds computation
    fact = interval_tf_embedding(L, Bin)

    # Extract metadata for HybridZ
    num_embeddings = L.meta['num_embeddings']
    embedding_dim = L.meta['embedding_dim']

    # Create HybridZ constraint with embedding metadata
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"embedding:{L.id}",
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "method": "interval_bounds",
        "note": "Sound bounds from validated interval TF"
    }))

    C.add_box(L.id, L.out_vars, fact.bounds)

    return Fact(fact.bounds, C)