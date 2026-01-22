#===- act/back_end/interval_tf/tf_rnn.py - RNN Interval Transfer Func ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Complete, fallback-free RNN Interval Transfer Functions with strict
#   metadata and weight validation. All parameters must be present or
#   the function will raise an error.


import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from act.back_end.core import Bounds, Con, ConSet, Fact, Layer
from act.back_end.utils import affine_bounds, pwl_meta, bound_var_interval, scale_interval


# =============================================================================
# Strict Metadata and Weight Validation
# =============================================================================

def _get_rnn_meta(L: Layer) -> Dict[str, any]:
    """
    Extract and validate RNN metadata. All required fields must be present.

    Args:
        L: Layer object

    Returns:
        Dictionary with validated metadata

    Raises:
        ValueError: If any required metadata is missing
    """
    required_fields = [
        'input_size', 'hidden_size', 'num_layers',
        'batch_first', 'bidirectional', 'input_shape', 'output_shape'
    ]

    meta = {}
    for field in required_fields:
        if field not in L.meta:
            raise ValueError(
                f"RNN layer {L.id} missing required metadata field '{field}'. "
                f"All fields {required_fields} must be present."
            )
        meta[field] = L.meta[field]

    # Optional fields
    meta['proj_size'] = L.meta.get('proj_size', 0)
    meta['nonlinearity'] = L.meta.get('nonlinearity', 'tanh')

    return meta


def _get_weights(
    L: Layer,
    layer_idx: int,
    direction: str = '',
    cell_type: str = 'rnn'
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Get weights for a specific layer and direction. Validates dimensions.

    Args:
        L: Layer object
        layer_idx: Layer index (0, 1, ...)
        direction: '' for forward, '_reverse' for backward
        cell_type: 'rnn', 'gru', or 'lstm'

    Returns:
        (weight_ih, weight_hh, bias_ih, bias_hh, weight_hr)

    Raises:
        ValueError: If required weights are missing or have wrong dimensions
    """
    suffix = f"_l{layer_idx}{direction}"

    # Get required weights
    weight_ih_key = f"weight_ih{suffix}"
    weight_hh_key = f"weight_hh{suffix}"

    if weight_ih_key not in L.params:
        raise ValueError(
            f"RNN layer {L.id} missing required parameter '{weight_ih_key}'"
        )
    if weight_hh_key not in L.params:
        raise ValueError(
            f"RNN layer {L.id} missing required parameter '{weight_hh_key}'"
        )

    weight_ih = L.params[weight_ih_key]
    weight_hh = L.params[weight_hh_key]

    # Optional biases
    bias_ih = L.params.get(f"bias_ih{suffix}", None)
    bias_hh = L.params.get(f"bias_hh{suffix}", None)

    # Optional projection weight for LSTM
    weight_hr = None
    if cell_type == 'lstm' and L.meta.get('proj_size', 0) > 0:
        weight_hr_key = f"weight_hr{suffix}"
        if weight_hr_key not in L.params:
            raise ValueError(
                f"LSTM layer {L.id} with proj_size>0 missing '{weight_hr_key}'"
            )
        weight_hr = L.params[weight_hr_key]

    # Validate dimensions
    # For layer 0: input_size from meta
    # For layer > 0: input comes from previous layer's output
    #   - If previous layer is bidirectional: hidden_size * 2
    #   - If previous layer has projection: proj_size * (2 if bidirectional else 1)
    #   - Otherwise: hidden_size * (2 if bidirectional else 1)
    if layer_idx == 0:
        expected_input_size = L.meta['input_size']
    else:
        hidden_size_meta = L.meta['hidden_size']
        proj_size_meta = L.meta.get('proj_size', 0)
        bidirectional_meta = L.meta['bidirectional']
        num_directions = 2 if bidirectional_meta else 1

        # Previous layer's output size
        if proj_size_meta > 0:
            expected_input_size = proj_size_meta * num_directions
        else:
            expected_input_size = hidden_size_meta * num_directions

    hidden_size = L.meta['hidden_size']

    # For LSTM/GRU, weight_ih has 4x/3x hidden_size rows
    multiplier = {'lstm': 4, 'gru': 3, 'rnn': 1}[cell_type]
    expected_ih_shape = (multiplier * hidden_size, expected_input_size)
    expected_hh_shape = (multiplier * hidden_size, hidden_size)

    if weight_ih.shape != expected_ih_shape:
        raise ValueError(
            f"weight_ih{suffix} shape {weight_ih.shape} does not match "
            f"expected {expected_ih_shape} for {cell_type.upper()}"
        )
    if weight_hh.shape != expected_hh_shape:
        raise ValueError(
            f"weight_hh{suffix} shape {weight_hh.shape} does not match "
            f"expected {expected_hh_shape} for {cell_type.upper()}"
        )

    return weight_ih, weight_hh, bias_ih, bias_hh, weight_hr


def _validate_input_shape(Bin: Bounds, meta: Dict) -> Bounds:
    """
    Validate and reshape input bounds to match expected 3D shape.

    Args:
        Bin: Input bounds (may be flattened)
        meta: Metadata dictionary with input_shape and batch_first

    Returns:
        Reshaped bounds as 3D tensor [B, T, input_size] or [T, B, input_size]

    Raises:
        ValueError: If input cannot be reshaped to expected shape
    """
    input_shape = tuple(meta['input_shape'])
    batch_first = meta['batch_first']

    # Check if Bin is flattened
    total_elements = torch.prod(torch.tensor(input_shape)).item()

    if Bin.lb.numel() != total_elements:
        raise ValueError(
            f"Input bounds size {Bin.lb.numel()} does not match "
            f"expected input_shape {input_shape} (total={total_elements})"
        )

    # Reshape to 3D
    if len(input_shape) != 3:
        raise ValueError(
            f"input_shape must be 3D [B, T, input_size] or [T, B, input_size], "
            f"got {input_shape}"
        )

    Bin_reshaped = Bounds(
        Bin.lb.view(*input_shape),
        Bin.ub.view(*input_shape)
    )

    return Bin_reshaped


# =============================================================================
# Complete Multi-Layer, Bidirectional RNN/GRU/LSTM Transfer Functions
# =============================================================================

def tf_lstm(L: Layer, Bin: Bounds) -> Fact:
    """
    Transfer function for LSTM layer with complete semantics.

    Handles:
    - Multiple layers (num_layers)
    - Bidirectional
    - LSTM projection (proj_size > 0)
    - Proper shape handling (batch_first)

    No fallback logic - all metadata and weights must be present.
    """
    # Extract and validate metadata
    meta = _get_rnn_meta(L)
    input_size = meta['input_size']
    hidden_size = meta['hidden_size']
    num_layers = meta['num_layers']
    batch_first = meta['batch_first']
    bidirectional = meta['bidirectional']
    proj_size = meta['proj_size']
    input_shape = tuple(meta['input_shape'])
    output_shape = tuple(meta['output_shape'])

    # Validate and reshape input
    Bin_3d = _validate_input_shape(Bin, meta)

    if batch_first:
        batch_size, seq_len, _ = input_shape
    else:
        seq_len, batch_size, _ = input_shape

    # Process each layer
    current_input = Bin_3d

    for layer_idx in range(num_layers):
        # Forward direction
        weight_ih_fwd, weight_hh_fwd, bias_ih_fwd, bias_hh_fwd, weight_hr_fwd = \
            _get_weights(L, layer_idx, '', 'lstm')

        output_fwd = _process_lstm_direction(
            current_input, weight_ih_fwd, weight_hh_fwd,
            bias_ih_fwd, bias_hh_fwd, weight_hr_fwd,
            hidden_size, proj_size, seq_len, batch_size,
            batch_first, forward=True
        )

        if bidirectional:
            # Backward direction
            weight_ih_bwd, weight_hh_bwd, bias_ih_bwd, bias_hh_bwd, weight_hr_bwd = \
                _get_weights(L, layer_idx, '_reverse', 'lstm')

            output_bwd = _process_lstm_direction(
                current_input, weight_ih_bwd, weight_hh_bwd,
                bias_ih_bwd, bias_hh_bwd, weight_hr_bwd,
                hidden_size, proj_size, seq_len, batch_size,
                batch_first, forward=False
            )

            # Concatenate forward and backward outputs
            concat_dim = 2 if batch_first else 2
            output_lb = torch.cat([output_fwd.lb, output_bwd.lb], dim=concat_dim)
            output_ub = torch.cat([output_fwd.ub, output_bwd.ub], dim=concat_dim)
            current_input = Bounds(output_lb, output_ub)
        else:
            current_input = output_fwd

    # Validate output shape
    expected_output_size = torch.prod(torch.tensor(output_shape)).item()
    actual_output_size = current_input.lb.numel()

    if actual_output_size != expected_output_size:
        raise ValueError(
            f"Output size mismatch: expected {expected_output_size} "
            f"(shape {output_shape}), got {actual_output_size}"
        )

    # Flatten output
    B_output = Bounds(current_input.lb.view(-1), current_input.ub.view(-1))

    # Create constraints with full metadata
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"lstm:{L.id}",
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "batch_first": batch_first,
        "proj_size": proj_size,
        "input_shape": input_shape,
        "output_shape": output_shape
    }))

    C.add_box(L.id, L.out_vars, B_output)
    return Fact(B_output, C)


def tf_gru(L: Layer, Bin: Bounds) -> Fact:
    """
    Transfer function for GRU layer with complete semantics.

    No fallback logic - all metadata and weights must be present.
    """
    # Extract and validate metadata
    meta = _get_rnn_meta(L)
    input_size = meta['input_size']
    hidden_size = meta['hidden_size']
    num_layers = meta['num_layers']
    batch_first = meta['batch_first']
    bidirectional = meta['bidirectional']
    input_shape = tuple(meta['input_shape'])
    output_shape = tuple(meta['output_shape'])

    # Validate and reshape input
    Bin_3d = _validate_input_shape(Bin, meta)

    if batch_first:
        batch_size, seq_len, _ = input_shape
    else:
        seq_len, batch_size, _ = input_shape

    # Process each layer
    current_input = Bin_3d

    for layer_idx in range(num_layers):
        # Forward direction
        weight_ih_fwd, weight_hh_fwd, bias_ih_fwd, bias_hh_fwd, _ = \
            _get_weights(L, layer_idx, '', 'gru')

        output_fwd = _process_gru_direction(
            current_input, weight_ih_fwd, weight_hh_fwd,
            bias_ih_fwd, bias_hh_fwd,
            hidden_size, seq_len, batch_size, batch_first, forward=True
        )

        if bidirectional:
            # Backward direction
            weight_ih_bwd, weight_hh_bwd, bias_ih_bwd, bias_hh_bwd, _ = \
                _get_weights(L, layer_idx, '_reverse', 'gru')

            output_bwd = _process_gru_direction(
                current_input, weight_ih_bwd, weight_hh_bwd,
                bias_ih_bwd, bias_hh_bwd,
                hidden_size, seq_len, batch_size, batch_first, forward=False
            )

            # Concatenate forward and backward outputs
            concat_dim = 2 if batch_first else 2
            output_lb = torch.cat([output_fwd.lb, output_bwd.lb], dim=concat_dim)
            output_ub = torch.cat([output_fwd.ub, output_bwd.ub], dim=concat_dim)
            current_input = Bounds(output_lb, output_ub)
        else:
            current_input = output_fwd

    # Validate output shape
    expected_output_size = torch.prod(torch.tensor(output_shape)).item()
    actual_output_size = current_input.lb.numel()

    if actual_output_size != expected_output_size:
        raise ValueError(
            f"Output size mismatch: expected {expected_output_size} "
            f"(shape {output_shape}), got {actual_output_size}"
        )

    # Flatten output
    B_output = Bounds(current_input.lb.view(-1), current_input.ub.view(-1))

    # Create constraints
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"gru:{L.id}",
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "batch_first": batch_first,
        "input_shape": input_shape,
        "output_shape": output_shape
    }))

    C.add_box(L.id, L.out_vars, B_output)
    return Fact(B_output, C)


def tf_rnn(L: Layer, Bin: Bounds) -> Fact:
    """
    Transfer function for vanilla RNN layer with complete semantics.

    No fallback logic - all metadata and weights must be present.
    """
    # Extract and validate metadata
    meta = _get_rnn_meta(L)
    input_size = meta['input_size']
    hidden_size = meta['hidden_size']
    num_layers = meta['num_layers']
    batch_first = meta['batch_first']
    bidirectional = meta['bidirectional']
    nonlinearity = meta['nonlinearity']
    input_shape = tuple(meta['input_shape'])
    output_shape = tuple(meta['output_shape'])

    # Validate and reshape input
    Bin_3d = _validate_input_shape(Bin, meta)

    if batch_first:
        batch_size, seq_len, _ = input_shape
    else:
        seq_len, batch_size, _ = input_shape

    # Process each layer
    current_input = Bin_3d

    for layer_idx in range(num_layers):
        # Forward direction
        weight_ih_fwd, weight_hh_fwd, bias_ih_fwd, bias_hh_fwd, _ = \
            _get_weights(L, layer_idx, '', 'rnn')

        output_fwd = _process_rnn_direction(
            current_input, weight_ih_fwd, weight_hh_fwd,
            bias_ih_fwd, bias_hh_fwd, nonlinearity,
            hidden_size, seq_len, batch_size, batch_first, forward=True
        )

        if bidirectional:
            # Backward direction
            weight_ih_bwd, weight_hh_bwd, bias_ih_bwd, bias_hh_bwd, _ = \
                _get_weights(L, layer_idx, '_reverse', 'rnn')

            output_bwd = _process_rnn_direction(
                current_input, weight_ih_bwd, weight_hh_bwd,
                bias_ih_bwd, bias_hh_bwd, nonlinearity,
                hidden_size, seq_len, batch_size, batch_first, forward=False
            )

            # Concatenate forward and backward outputs
            concat_dim = 2 if batch_first else 2
            output_lb = torch.cat([output_fwd.lb, output_bwd.lb], dim=concat_dim)
            output_ub = torch.cat([output_fwd.ub, output_bwd.ub], dim=concat_dim)
            current_input = Bounds(output_lb, output_ub)
        else:
            current_input = output_fwd

    # Validate output shape
    expected_output_size = torch.prod(torch.tensor(output_shape)).item()
    actual_output_size = current_input.lb.numel()

    if actual_output_size != expected_output_size:
        raise ValueError(
            f"Output size mismatch: expected {expected_output_size} "
            f"(shape {output_shape}), got {actual_output_size}"
        )

    # Flatten output
    B_output = Bounds(current_input.lb.view(-1), current_input.ub.view(-1))

    # Create constraints
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"rnn:{L.id}",
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "batch_first": batch_first,
        "nonlinearity": nonlinearity,
        "input_shape": input_shape,
        "output_shape": output_shape
    }))

    C.add_box(L.id, L.out_vars, B_output)
    return Fact(B_output, C)


def tf_embedding(L: Layer, Bin: Bounds) -> Fact:
    """
    Transfer function for Embedding layer with complete semantics.

    No fallback logic - all metadata and weights must be present.
    """
    # Strict metadata validation
    required_meta = ['num_embeddings', 'embedding_dim', 'input_shape', 'output_shape']
    for field in required_meta:
        if field not in L.meta:
            raise ValueError(
                f"Embedding layer {L.id} missing required metadata '{field}'"
            )

    if 'weight' not in L.params:
        raise ValueError(f"Embedding layer {L.id} missing required parameter 'weight'")

    weight = L.params['weight']
    num_embeddings = L.meta['num_embeddings']
    embedding_dim = L.meta['embedding_dim']
    input_shape = tuple(L.meta['input_shape'])
    output_shape = tuple(L.meta['output_shape'])

    # Validate weight dimensions
    if weight.shape != (num_embeddings, embedding_dim):
        raise ValueError(
            f"Embedding weight shape {weight.shape} does not match "
            f"expected ({num_embeddings}, {embedding_dim})"
        )

    # Compute bounds from embedding table
    # Use min/max across all embeddings as worst-case bounds
    weight_min = torch.min(weight, dim=0)[0]  # [embedding_dim]
    weight_max = torch.max(weight, dim=0)[0]  # [embedding_dim]

    # Compute output size and reshape bounds
    output_size = torch.prod(torch.tensor(output_shape)).item()
    embedding_elements = output_size // embedding_dim

    if output_size % embedding_dim != 0:
        raise ValueError(
            f"output_shape {output_shape} (size={output_size}) is not "
            f"divisible by embedding_dim {embedding_dim}"
        )

    # Tile bounds to match output shape
    output_lb = weight_min.repeat(embedding_elements)
    output_ub = weight_max.repeat(embedding_elements)

    B_output = Bounds(output_lb, output_ub)

    # Create constraints
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"embedding:{L.id}",
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "input_shape": input_shape,
        "output_shape": output_shape
    }))

    C.add_box(L.id, L.out_vars, B_output)
    return Fact(B_output, C)


# =============================================================================
# Direction Processing Functions (Forward/Backward)
# =============================================================================

def _process_lstm_direction(
    input_bounds: Bounds,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: Optional[torch.Tensor],
    bias_hh: Optional[torch.Tensor],
    weight_hr: Optional[torch.Tensor],
    hidden_size: int,
    proj_size: int,
    seq_len: int,
    batch_size: int,
    batch_first: bool,
    forward: bool
) -> Bounds:
    """Process LSTM in one direction (forward or backward)."""
    # Initialize hidden and cell states
    actual_hidden_size = proj_size if proj_size > 0 else hidden_size
    h_bounds = Bounds(
        torch.zeros(batch_size, actual_hidden_size),
        torch.zeros(batch_size, actual_hidden_size)
    )
    c_bounds = Bounds(
        torch.zeros(batch_size, hidden_size),
        torch.zeros(batch_size, hidden_size)
    )

    output_bounds_list = []
    time_steps = range(seq_len) if forward else range(seq_len - 1, -1, -1)

    for t in time_steps:
        # Extract input at time step t
        if batch_first:
            x_t_bounds = Bounds(
                input_bounds.lb[:, t, :],
                input_bounds.ub[:, t, :]
            )
        else:
            x_t_bounds = Bounds(
                input_bounds.lb[t, :, :],
                input_bounds.ub[t, :, :]
            )

        # LSTM cell computation
        h_bounds, c_bounds = _lstm_cell_bounds(
            x_t_bounds, h_bounds, c_bounds,
            weight_ih, weight_hh, bias_ih, bias_hh,
            weight_hr, hidden_size, proj_size
        )

        output_bounds_list.append(h_bounds)

    # Reverse output if backward pass
    if not forward:
        output_bounds_list = output_bounds_list[::-1]

    # Stack outputs
    if batch_first:
        output_lb = torch.stack([b.lb for b in output_bounds_list], dim=1)
        output_ub = torch.stack([b.ub for b in output_bounds_list], dim=1)
    else:
        output_lb = torch.stack([b.lb for b in output_bounds_list], dim=0)
        output_ub = torch.stack([b.ub for b in output_bounds_list], dim=0)

    return Bounds(output_lb, output_ub)


def _process_gru_direction(
    input_bounds: Bounds,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: Optional[torch.Tensor],
    bias_hh: Optional[torch.Tensor],
    hidden_size: int,
    seq_len: int,
    batch_size: int,
    batch_first: bool,
    forward: bool
) -> Bounds:
    """Process GRU in one direction (forward or backward)."""
    # Initialize hidden state
    h_bounds = Bounds(
        torch.zeros(batch_size, hidden_size),
        torch.zeros(batch_size, hidden_size)
    )

    output_bounds_list = []
    time_steps = range(seq_len) if forward else range(seq_len - 1, -1, -1)

    for t in time_steps:
        # Extract input at time step t
        if batch_first:
            x_t_bounds = Bounds(
                input_bounds.lb[:, t, :],
                input_bounds.ub[:, t, :]
            )
        else:
            x_t_bounds = Bounds(
                input_bounds.lb[t, :, :],
                input_bounds.ub[t, :, :]
            )

        # GRU cell computation
        h_bounds = _gru_cell_bounds(
            x_t_bounds, h_bounds,
            weight_ih, weight_hh, bias_ih, bias_hh,
            hidden_size
        )

        output_bounds_list.append(h_bounds)

    # Reverse output if backward pass
    if not forward:
        output_bounds_list = output_bounds_list[::-1]

    # Stack outputs
    if batch_first:
        output_lb = torch.stack([b.lb for b in output_bounds_list], dim=1)
        output_ub = torch.stack([b.ub for b in output_bounds_list], dim=1)
    else:
        output_lb = torch.stack([b.lb for b in output_bounds_list], dim=0)
        output_ub = torch.stack([b.ub for b in output_bounds_list], dim=0)

    return Bounds(output_lb, output_ub)


def _process_rnn_direction(
    input_bounds: Bounds,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: Optional[torch.Tensor],
    bias_hh: Optional[torch.Tensor],
    nonlinearity: str,
    hidden_size: int,
    seq_len: int,
    batch_size: int,
    batch_first: bool,
    forward: bool
) -> Bounds:
    """Process RNN in one direction (forward or backward)."""
    # Initialize hidden state
    h_bounds = Bounds(
        torch.zeros(batch_size, hidden_size),
        torch.zeros(batch_size, hidden_size)
    )

    output_bounds_list = []
    time_steps = range(seq_len) if forward else range(seq_len - 1, -1, -1)

    for t in time_steps:
        # Extract input at time step t
        if batch_first:
            x_t_bounds = Bounds(
                input_bounds.lb[:, t, :],
                input_bounds.ub[:, t, :]
            )
        else:
            x_t_bounds = Bounds(
                input_bounds.lb[t, :, :],
                input_bounds.ub[t, :, :]
            )

        # RNN cell: h_t = nonlinearity(W_ih * x_t + W_hh * h_{t-1} + b)
        ih_transform = _apply_linear_bounds(x_t_bounds, weight_ih, bias_ih)
        hh_transform = _apply_linear_bounds(h_bounds, weight_hh, bias_hh)

        combined_bounds = Bounds(
            ih_transform.lb + hh_transform.lb,
            ih_transform.ub + hh_transform.ub
        )

        if nonlinearity == 'tanh':
            h_bounds = _apply_tanh_bounds(combined_bounds)
        elif nonlinearity == 'relu':
            h_bounds = _apply_relu_bounds(combined_bounds)
        else:
            raise ValueError(f"Unsupported RNN nonlinearity: {nonlinearity}")

        output_bounds_list.append(h_bounds)

    # Reverse output if backward pass
    if not forward:
        output_bounds_list = output_bounds_list[::-1]

    # Stack outputs
    if batch_first:
        output_lb = torch.stack([b.lb for b in output_bounds_list], dim=1)
        output_ub = torch.stack([b.ub for b in output_bounds_list], dim=1)
    else:
        output_lb = torch.stack([b.lb for b in output_bounds_list], dim=0)
        output_ub = torch.stack([b.ub for b in output_bounds_list], dim=0)

    return Bounds(output_lb, output_ub)


# =============================================================================
# Cell-Level Bounds Computation
# =============================================================================

def _lstm_cell_bounds(
    x_bounds: Bounds,
    h_bounds: Bounds,
    c_bounds: Bounds,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: Optional[torch.Tensor],
    bias_hh: Optional[torch.Tensor],
    weight_hr: Optional[torch.Tensor],
    hidden_size: int,
    proj_size: int
) -> Tuple[Bounds, Bounds]:
    """Compute LSTM cell bounds for one time step with optional projection."""
    # Linear transformations
    ih_all = _apply_linear_bounds(x_bounds, weight_ih, bias_ih)
    hh_all = _apply_linear_bounds(h_bounds, weight_hh, bias_hh)

    # Split into 4 gates: input, forget, cell, output
    i_bounds = Bounds(
        ih_all.lb[:, :hidden_size] + hh_all.lb[:, :hidden_size],
        ih_all.ub[:, :hidden_size] + hh_all.ub[:, :hidden_size]
    )
    i_bounds = _apply_sigmoid_bounds(i_bounds)

    f_bounds = Bounds(
        ih_all.lb[:, hidden_size:2*hidden_size] + hh_all.lb[:, hidden_size:2*hidden_size],
        ih_all.ub[:, hidden_size:2*hidden_size] + hh_all.ub[:, hidden_size:2*hidden_size]
    )
    f_bounds = _apply_sigmoid_bounds(f_bounds)

    g_bounds = Bounds(
        ih_all.lb[:, 2*hidden_size:3*hidden_size] + hh_all.lb[:, 2*hidden_size:3*hidden_size],
        ih_all.ub[:, 2*hidden_size:3*hidden_size] + hh_all.ub[:, 2*hidden_size:3*hidden_size]
    )
    g_bounds = _apply_tanh_bounds(g_bounds)

    o_bounds = Bounds(
        ih_all.lb[:, 3*hidden_size:] + hh_all.lb[:, 3*hidden_size:],
        ih_all.ub[:, 3*hidden_size:] + hh_all.ub[:, 3*hidden_size:]
    )
    o_bounds = _apply_sigmoid_bounds(o_bounds)

    # Cell state: c_t = f * c_{t-1} + i * g
    fc_bounds = _multiply_bounds(f_bounds, c_bounds)
    ig_bounds = _multiply_bounds(i_bounds, g_bounds)
    new_c_bounds = Bounds(
        fc_bounds.lb + ig_bounds.lb,
        fc_bounds.ub + ig_bounds.ub
    )

    # Hidden state: h_t = o * tanh(c_t)
    tanh_c_bounds = _apply_tanh_bounds(new_c_bounds)
    new_h_bounds = _multiply_bounds(o_bounds, tanh_c_bounds)

    # Apply projection if proj_size > 0
    if proj_size > 0 and weight_hr is not None:
        new_h_bounds = _apply_linear_bounds(new_h_bounds, weight_hr, None)

    return new_h_bounds, new_c_bounds


def _gru_cell_bounds(
    x_bounds: Bounds,
    h_bounds: Bounds,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: Optional[torch.Tensor],
    bias_hh: Optional[torch.Tensor],
    hidden_size: int
) -> Bounds:
    """Compute GRU cell bounds for one time step."""
    # GRU formula (PyTorch variant):
    # r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
    # z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
    # n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
    # h' = (1 - z) * n + z * h

    # Linear transformation for input
    ih_all = _apply_linear_bounds(x_bounds, weight_ih, bias_ih)

    # For reset and update gates, use standard hh transformation
    weight_hh_rz = weight_hh[:2*hidden_size, :]  # Reset and update gate weights
    bias_hh_rz = bias_hh[:2*hidden_size] if bias_hh is not None else None
    hh_rz = _apply_linear_bounds(h_bounds, weight_hh_rz, bias_hh_rz)

    # Reset gate
    r_bounds = Bounds(
        ih_all.lb[:, :hidden_size] + hh_rz.lb[:, :hidden_size],
        ih_all.ub[:, :hidden_size] + hh_rz.ub[:, :hidden_size]
    )
    r_bounds = _apply_sigmoid_bounds(r_bounds)

    # Update gate
    z_bounds = Bounds(
        ih_all.lb[:, hidden_size:2*hidden_size] + hh_rz.lb[:, hidden_size:2*hidden_size],
        ih_all.ub[:, hidden_size:2*hidden_size] + hh_rz.ub[:, hidden_size:2*hidden_size]
    )
    z_bounds = _apply_sigmoid_bounds(z_bounds)

    # New gate: n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
    # First compute W_hn @ h + b_hn
    weight_hh_n = weight_hh[2*hidden_size:, :]
    bias_hh_n = bias_hh[2*hidden_size:] if bias_hh is not None else None
    hh_n = _apply_linear_bounds(h_bounds, weight_hh_n, bias_hh_n)

    # Then multiply by reset gate
    rh_n_bounds = _multiply_bounds(r_bounds, hh_n)

    # Add to input transformation
    n_bounds = Bounds(
        ih_all.lb[:, 2*hidden_size:] + rh_n_bounds.lb,
        ih_all.ub[:, 2*hidden_size:] + rh_n_bounds.ub
    )
    n_bounds = _apply_tanh_bounds(n_bounds)

    # Hidden state: h_t = (1 - z) * n + z * h
    one_minus_z_bounds = Bounds(1 - z_bounds.ub, 1 - z_bounds.lb)
    term1 = _multiply_bounds(one_minus_z_bounds, n_bounds)
    term2 = _multiply_bounds(z_bounds, h_bounds)

    new_h_bounds = Bounds(
        term1.lb + term2.lb,
        term1.ub + term2.ub
    )

    return new_h_bounds


# =============================================================================
# Helper Functions
# =============================================================================

def _apply_linear_bounds(
    input_bounds: Bounds,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor]
) -> Bounds:
    """Apply linear transformation to bounds, preserving batch dimension."""
    W_pos = torch.clamp(weight, min=0)
    W_neg = torch.clamp(weight, max=0)

    if bias is None:
        bias = torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)

    # Check if input has batch dimension
    if input_bounds.lb.dim() == 2:
        # Batch dimension present: [batch, features]
        # Compute bounds for each batch element
        batch_size = input_bounds.lb.shape[0]
        out_features = weight.shape[0]

        lb_out = torch.zeros(batch_size, out_features, device=weight.device, dtype=weight.dtype)
        ub_out = torch.zeros(batch_size, out_features, device=weight.device, dtype=weight.dtype)

        for i in range(batch_size):
            batch_bounds = Bounds(input_bounds.lb[i], input_bounds.ub[i])
            batch_result = affine_bounds(W_pos, W_neg, bias, batch_bounds)
            lb_out[i] = batch_result.lb
            ub_out[i] = batch_result.ub

        return Bounds(lb_out, ub_out)
    else:
        # No batch dimension: [features]
        result = affine_bounds(W_pos, W_neg, bias, input_bounds)
        return result


def _apply_sigmoid_bounds(bounds: Bounds) -> Bounds:
    """Apply sigmoid function to bounds (monotonic)."""
    return Bounds(torch.sigmoid(bounds.lb), torch.sigmoid(bounds.ub))


def _apply_tanh_bounds(bounds: Bounds) -> Bounds:
    """Apply tanh function to bounds (monotonic)."""
    return Bounds(torch.tanh(bounds.lb), torch.tanh(bounds.ub))


def _apply_relu_bounds(bounds: Bounds) -> Bounds:
    """Apply ReLU function to bounds."""
    return Bounds(
        torch.clamp(bounds.lb, min=0),
        torch.clamp(bounds.ub, min=0)
    )


def _multiply_bounds(bounds1: Bounds, bounds2: Bounds) -> Bounds:
    """Multiply two interval bounds."""
    a, b = bounds1.lb, bounds1.ub
    c, d = bounds2.lb, bounds2.ub

    products = [a*c, a*d, b*c, b*d]

    return Bounds(
        torch.min(torch.stack(products), dim=0)[0],
        torch.max(torch.stack(products), dim=0)[0]
    )
