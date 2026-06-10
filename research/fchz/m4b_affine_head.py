# ===- research/fchz/m4b_affine_head.py - M4B affine head extractor ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Per advisor 2026-06-08 P3 (M4B):
#   Extend M4 LP to small-dense + affine-head structure (e.g. linearizenn).
#
# Each head layer's output is linear in both:
#     xi[K_in]          : input noise vars (FCHZ INPUT state)
#     y_R[n_R]          : last ReLU's post-activation
#
# So output = W_xi_head @ xi + W_y_head @ y_R + b_head
#
# Allowed head ops:
#     DENSE, BIAS, SCALE
#     constant SLICE (axes/starts/ends/steps are static)
#     affine CONCAT
#     FLATTEN, RESHAPE, IDENTITY, TRANSPOSE (semantic-pure index ops)
#
# Forbidden:
#     MUL (unless one side is constant)
#     REDUCE_SUM (in head)
#     data-dependent SLICE / GATHER / nonlinear ops
#
# Returns: (W_xi_head, W_y_head, b_head) for the final output dim, or None
# if any forbidden op is encountered.
# ===---------------------------------------------------------------------===#

"""M4B affine head extractor — DAG walk with safety gate."""

import numpy as np


def _to_np(x, dtype=np.float64):
    if x is None: return None
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy().astype(dtype)
    return np.asarray(x, dtype=dtype)


def _identity_repr(n, K_in, n_R):
    """Identity affine: output = y (no contribution from xi)."""
    return (np.zeros((n, K_in), dtype=np.float64),   # W_xi
                np.eye(n, dtype=np.float64),                 # W_y
                np.zeros(n, dtype=np.float64))                # b


def _input_repr(input_state, K_in, n_R):
    """Input state repr: output = input.c + input.G @ xi."""
    c = input_state.c.astype(np.float64)
    G = input_state.G.astype(np.float64) if input_state.G is not None else np.zeros((c.shape[0], K_in))
    n = c.shape[0]
    return (G, np.zeros((n, n_R), dtype=np.float64), c)


def extract_affine_head(net, tf, last_relu_id: int, n_R: int, K_in: int, input_state):
    """Walk net topologically from after last_relu_id to ASSERT.

    Compute (W_xi, W_y, b) for each layer such that
        layer_output = W_xi @ xi + W_y @ y_R + b

    Returns:
        For the FINAL pre-ASSERT layer (= ASSERT predecessor), (W_xi, W_y, b).
        Or None if any forbidden op is encountered.
    """
    repr_map = {}    # layer_id → (W_xi, W_y, b)

    # Seed: last ReLU's output is y_R itself
    repr_map[last_relu_id] = _identity_repr(n_R, K_in, n_R)
    # Seed: INPUT_SPEC layer is input affine
    for L in net.layers:
        if L.kind in ('INPUT', 'INPUT_SPEC'):
            repr_map[L.id] = _input_repr(input_state, K_in, n_R)

    # Walk in id order (assumed topological)
    last_repr_id = last_relu_id
    for L in net.layers:
        if L.id <= last_relu_id: continue
        if L.kind == 'ASSERT': break
        preds = net.preds.get(L.id, [])

        def get_pred_repr(p):
            if p in repr_map: return repr_map[p]
            # If predecessor not in repr_map, fail
            return None

        if L.kind in ('DENSE',):
            if len(preds) != 1: return None
            pr = get_pred_repr(preds[0])
            if pr is None: return None
            W_xi_p, W_y_p, b_p = pr
            W = _to_np(L.params.get('weight'))
            b_l = _to_np(L.params.get('bias'))
            if W is None: return None
            if b_l is None: b_l = np.zeros(W.shape[0], dtype=np.float64)
            # output = W @ pred + b = W @ (W_xi_p @ xi + W_y_p @ y_R + b_p) + b
            new_W_xi = W @ W_xi_p
            new_W_y = W @ W_y_p
            new_b = W @ b_p + b_l
            repr_map[L.id] = (new_W_xi, new_W_y, new_b)
        elif L.kind == 'BIAS':
            if len(preds) != 1: return None
            pr = get_pred_repr(preds[0])
            if pr is None: return None
            W_xi_p, W_y_p, b_p = pr
            b_l = _to_np(L.params.get('c'))
            if b_l is None: return None
            new_b = b_p + b_l.flatten()
            repr_map[L.id] = (W_xi_p, W_y_p, new_b)
        elif L.kind == 'SCALE':
            if len(preds) != 1: return None
            pr = get_pred_repr(preds[0])
            if pr is None: return None
            W_xi_p, W_y_p, b_p = pr
            a = _to_np(L.params.get('a'))
            if a is None: return None
            a = a.flatten()
            new_W_xi = a[:, None] * W_xi_p
            new_W_y = a[:, None] * W_y_p
            new_b = a * b_p
            repr_map[L.id] = (new_W_xi, new_W_y, new_b)
        elif L.kind == 'SLICE':
            if len(preds) != 1: return None
            pr = get_pred_repr(preds[0])
            if pr is None: return None
            W_xi_p, W_y_p, b_p = pr
            starts = L.params.get('starts', [0])
            ends = L.params.get('ends', [None])
            axes = L.params.get('axes', [0])
            steps = L.params.get('steps', [1])
            # For our LP affine vector layout, slice acts on flat dimension.
            # Assume axes[0] is the only non-trivial axis (typical for ACT 1D slice).
            # Compute the slice on dim 0 of pred's vector representation.
            start = int(starts[0]) if len(starts) > 0 else 0
            end = int(ends[0]) if len(ends) > 0 and ends[0] is not None else b_p.shape[0]
            step = int(steps[0]) if len(steps) > 0 else 1
            # ACT might use different axis convention; here we assume axis matches flat layout
            indices = list(range(start, end, step))
            new_W_xi = W_xi_p[indices]
            new_W_y = W_y_p[indices]
            new_b = b_p[indices]
            repr_map[L.id] = (new_W_xi, new_W_y, new_b)
        elif L.kind == 'CONCAT':
            # Combine multiple predecessors via row stacking (axis 0 in flat repr)
            stacked_W_xi = []
            stacked_W_y = []
            stacked_b = []
            for p in preds:
                pr = get_pred_repr(p)
                if pr is None: return None
                W_xi_p, W_y_p, b_p = pr
                stacked_W_xi.append(W_xi_p)
                stacked_W_y.append(W_y_p)
                stacked_b.append(b_p)
            new_W_xi = np.concatenate(stacked_W_xi, axis=0)
            new_W_y = np.concatenate(stacked_W_y, axis=0)
            new_b = np.concatenate(stacked_b, axis=0)
            repr_map[L.id] = (new_W_xi, new_W_y, new_b)
        elif L.kind in ('FLATTEN', 'RESHAPE', 'IDENTITY'):
            if len(preds) != 1: return None
            pr = get_pred_repr(preds[0])
            if pr is None: return None
            repr_map[L.id] = pr   # semantic-pure for flat repr
        elif L.kind == 'TRANSPOSE':
            # Skip TRANSPOSE in our flat repr (not safe to assume)
            return None
        else:
            return None    # forbidden op

        last_repr_id = L.id

    # Return final layer's (W_xi, W_y, b)
    if last_repr_id not in repr_map: return None
    return repr_map[last_repr_id]


def has_affine_head(net, last_relu_id):
    """Quick check: are all ops AFTER last_relu_id in the allowed affine head set?"""
    allowed = {'DENSE', 'BIAS', 'SCALE', 'SLICE', 'CONCAT',
                  'FLATTEN', 'RESHAPE', 'IDENTITY', 'ASSERT',
                  'INPUT', 'INPUT_SPEC'}   # INPUT/INPUT_SPEC OK (might be SLICE pred)
    for L in net.layers:
        if L.id <= last_relu_id: continue
        if L.kind not in allowed: return False
    return True


def is_dense_body_with_affine_head(net):
    """Check: body (up to and including last ReLU) is dense-only;
                  head (after last ReLU) is allowed-affine-only."""
    last_relu_id = None
    body_allowed = {'INPUT', 'INPUT_SPEC', 'DENSE', 'RELU', 'BIAS', 'FLATTEN'}
    for L in net.layers:
        if L.kind == 'RELU': last_relu_id = L.id
    if last_relu_id is None: return False, None
    for L in net.layers:
        if L.id <= last_relu_id:
            if L.kind not in body_allowed: return False, last_relu_id
    if not has_affine_head(net, last_relu_id): return False, last_relu_id
    return True, last_relu_id
