# ===- act/back_end/fchz_tf/fchz_tf.py - FCHZ Transfer Function ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   FCHZ (Forward Constrained Hybrid Zonotope) Transfer Function.
#
#   Implements act.back_end.transfer_functions.TransferFunction interface.
#   Each layer type dispatched via _LAYER_REGISTRY following HybridzTF pattern.
#
#   Strict P1-P5 compliance:
#     - Forward propagation only (no backward)
#     - No gradient
#     - Continuous LP via HiGHS (no MILP)
#     - No input split
#     - No random certify (ORT post-audit only)
#
# ===---------------------------------------------------------------------===#

"""FCHZ Transfer Function — ACT-integrated forward HZ propagation."""

from __future__ import annotations
from typing import Dict, Optional, Any, Tuple
import torch
import numpy as np

from act.back_end.core import Bounds, Fact, Layer, Net, ConSet
from act.back_end.transfer_functions import TransferFunction
from act.back_end.layer_schema import LayerKind

from act.back_end.fchz_tf.representations import (
    FCHZState, initial_state, hz_closed_form_ub,
    apply_dense, compress_g_to_tail, SlackRecord,
)
from act.back_end.fchz_tf.sigmoid_chord import chord_params
from act.back_end.fchz_tf import tf_cnn


def _t2n(t) -> np.ndarray:
    """torch.Tensor -> numpy.ndarray (CPU, float64)."""
    if isinstance(t, np.ndarray):
        return t.astype(np.float64)
    return t.detach().cpu().numpy().astype(np.float64)


def _bounds_to_fchz(bounds: Bounds) -> FCHZState:
    """Build FCHZState from input bounds.

    ACT bounds are typically [B, n_in] (2D with batch dim). We flatten
    everything past dim 0 since FCHZState is always flat-state internally.
    """
    lb = _t2n(bounds.lb).reshape(-1)
    ub = _t2n(bounds.ub).reshape(-1)
    c = (lb + ub) / 2.0
    r = (ub - lb) / 2.0
    return initial_state(c, r)


def _fchz_to_bounds(state: FCHZState, dev=None, dtype=None,
                            batch_shape: Optional[tuple] = None) -> Bounds:
    """Extract Bounds (interval [lb, ub]) from FCHZState via HZ closed-form.

    For each dimension i:
        ub_i = c_i + Σ|G_ik| + tail_radius_i
        lb_i = c_i - Σ|G_ik| - tail_radius_i

    Args:
        state:       FCHZState (flat internal representation)
        dev, dtype:  output torch device/dtype
        batch_shape: if given, reshape output to (B, n_out). ACT's verify_once
                     expects (1, n_out). Default: preserve batch dim of 1.
    """
    rad = np.abs(state.G).sum(axis=1)
    if state.tail_radius is not None:
        rad = rad + state.tail_radius
    lb = state.c - rad
    ub = state.c + rad
    lb_t = torch.from_numpy(lb)
    ub_t = torch.from_numpy(ub)
    if dev is not None:
        lb_t = lb_t.to(dev); ub_t = ub_t.to(dev)
    if dtype is not None:
        lb_t = lb_t.to(dtype); ub_t = ub_t.to(dtype)
    # Add batch dimension for ACT verifier compatibility
    if batch_shape is None:
        # Default: assume B=1, single row
        lb_t = lb_t.unsqueeze(0)
        ub_t = ub_t.unsqueeze(0)
    else:
        lb_t = lb_t.reshape(batch_shape)
        ub_t = ub_t.reshape(batch_shape)
    return Bounds(lb=lb_t, ub=ub_t)


# -------------------------------------------------------------------------
# Per-layer transfer functions
# -------------------------------------------------------------------------

class FchzTF(TransferFunction):
    """FCHZ transfer function — strict P1-P5 forward HZ.

    State cache: maps layer.id -> FCHZState. Carried through analysis.

    Per layer, dispatch via _LAYER_REGISTRY following HybridzTF pattern.
    Fallback: interval-only bound from upstream bounds for unsupported ops.
    """

    def __init__(self, G_max_cols: Optional[int] = None):
        # Per layer FCHZState cache (key = layer.id)
        self._state_cache: Dict[int, FCHZState] = {}
        # Per layer shape cache for CNN layers (key = layer.id, value = (C, H, W))
        self._shape_cache: Dict[int, Tuple[int, ...]] = {}
        # Cache invalidation key = id(net)
        self._cache_net_id: Optional[int] = None
        # Sparse-slack compression budget (None = no compression)
        self.G_max_cols: Optional[int] = G_max_cols
        # Transient state for layer dispatch (set in apply)
        self._net: Optional[Net] = None
        self._before: Optional[Dict[int, Fact]] = None
        self._after: Optional[Dict[int, Fact]] = None

    @property
    def name(self) -> str:
        return "FchzTF"

    def supports_layer(self, layer_kind: str) -> bool:
        return layer_kind.upper() in self._LAYER_REGISTRY

    # Whitelist of single-input shape/index ops where multi-pred (data + CONSTANT
    # param feeds) is a known ONNX pattern. For these, _get_input_fchz may pick
    # the data pred. For ALL OTHER multi-input ops (CONCAT/ADD/MUL/SUB/DIV/MATMUL),
    # the op's own handler MUST resolve its multiple state inputs explicitly.
    _SHAPE_INDEX_OPS_WITH_DATA_PRED = {
        'SLICE', 'GATHER', 'RESHAPE', 'TRANSPOSE',
        'EXPAND', 'SQUEEZE', 'UNSQUEEZE', 'FLATTEN',
    }

    def _get_input_fchz(self, L: Layer, input_bounds: Bounds) -> FCHZState:
        """Get FCHZState for layer's input.

        Sound rules (advisor P5 2026-06-08 advisor harden):
          - Single pred + cached → use it.
          - Single pred missing → STATE_LOSS, fall back to fresh box.
          - Multi-pred + L.kind in SHAPE/INDEX whitelist → pick first cached
            non-CONSTANT pred (the data feed; other preds are param tensors).
          - Multi-pred + L.kind NOT in whitelist → STATE_LOSS, fall back to fresh
            box. Multi-input ops (CONCAT/ADD/MUL/etc.) MUST handle their preds
            themselves; this helper refuses to silently pick one of them.

        Every fallback/resolution is logged in self._state_loss_log for audit.
        """
        if not hasattr(self, '_state_loss_log'):
            self._state_loss_log = []

        if self._net is None or L.id not in self._net.preds:
            self._state_loss_log.append({
                'layer_id': L.id, 'kind': L.kind,
                'reason': 'no_preds_in_graph',
            })
            return _bounds_to_fchz(input_bounds)

        pred_ids = self._net.preds[L.id]
        # Single pred path
        if len(pred_ids) == 1:
            if pred_ids[0] in self._state_cache:
                return self._state_cache[pred_ids[0]]
            self._state_loss_log.append({
                'layer_id': L.id, 'kind': L.kind,
                'pred_ids': list(pred_ids),
                'reason': 'pred_state_missing',
            })
            return _bounds_to_fchz(input_bounds)

        # Multi-pred path — only allowed for shape/index whitelist.
        if L.kind not in self._SHAPE_INDEX_OPS_WITH_DATA_PRED:
            self._state_loss_log.append({
                'layer_id': L.id, 'kind': L.kind,
                'pred_ids': list(pred_ids),
                'cached_preds': [p for p in pred_ids if p in self._state_cache],
                'reason': 'multi_pred_not_whitelisted',
            })
            return _bounds_to_fchz(input_bounds)

        # Identify the data pred (non-CONSTANT, cached).
        data_pred = None
        constant_preds = []
        for p in pred_ids:
            pred_layer = next((LL for LL in self._net.layers if LL.id == p), None)
            if pred_layer is not None and pred_layer.kind == 'CONSTANT':
                constant_preds.append(p)
            elif p in self._state_cache and data_pred is None:
                data_pred = p
        if data_pred is not None:
            self._state_loss_log.append({
                'layer_id': L.id, 'kind': L.kind,
                'pred_ids': list(pred_ids),
                'data_pred': data_pred,
                'constant_preds': constant_preds,
                'reason': 'multi_pred_resolved_to_data',
            })
            return self._state_cache[data_pred]

        # Whitelist op but no data pred resolvable
        self._state_loss_log.append({
            'layer_id': L.id, 'kind': L.kind,
            'pred_ids': list(pred_ids),
            'cached_preds': [p for p in pred_ids if p in self._state_cache],
            'reason': 'multi_pred_data_pred_not_found',
        })
        return _bounds_to_fchz(input_bounds)

    def _store_state(self, layer_id: int, state: FCHZState,
                                allow_compress: bool = True) -> None:
        """Store FCHZState for layer; apply sparse-slack if budget exceeded.

        Compression timing fix (M1-fix advisor task 4):
          - Only compress after weighted ops (DENSE/CONV2D/BN/ReLU).
          - Skip compression at INPUT/INPUT_SPEC (matches raw walker:
            walker doesn't compress initial state; compress only after
            ops that increase G column count).
        """
        if allow_compress and self.G_max_cols is not None and state.G.shape[1] > self.G_max_cols:
            state = compress_g_to_tail(state, self.G_max_cols)
        self._state_cache[layer_id] = state

    # ---------------------------------------------------------------------
    # Per-layer-kind handlers (mirror HybridzTF.tf_mlp / tf_cnn pattern)
    # ---------------------------------------------------------------------

    def _tf_input(self, L: Layer, input_bounds: Bounds) -> Fact:
        """INPUT / INPUT_SPEC: initialize FCHZState from bounds.

        Per advisor M1-fix: do NOT compress at INPUT (matches raw walker).
        Compression only happens after Conv/Dense.
        """
        state = _bounds_to_fchz(input_bounds)
        self._store_state(L.id, state, allow_compress=False)
        out_bounds = _fchz_to_bounds(state, dev=input_bounds.lb.device,
                                              dtype=input_bounds.lb.dtype)
        return Fact(bounds=out_bounds, cons=ConSet())

    def _tf_dense(self, L: Layer, input_bounds: Bounds) -> Fact:
        """DENSE: y = Wx + b → propagate FCHZState through linear map.

        Shape contract (advisor P5 2026-06-08):
          s_in.c.shape[0] must equal W.shape[1] (in_features).
          On violation: log DENSE_INPUT_DIM_MISMATCH and fail-closed
          (return input_bounds, don't run matmul that would crash).
        """
        if not hasattr(self, '_state_loss_log'):
            self._state_loss_log = []
        W = _t2n(L.params.get('weight'))
        b = L.params.get('bias')
        b_n = _t2n(b).reshape(-1) if b is not None else None
        s_in = self._get_input_fchz(L, input_bounds)
        if W.shape[1] != s_in.c.shape[0]:
            self._state_loss_log.append({
                'layer_id': L.id, 'kind': 'DENSE',
                'reason': 'dense_input_dim_mismatch',
                'W_in_features': int(W.shape[1]),
                'state_c_dim': int(s_in.c.shape[0]),
            })
            return Fact(bounds=input_bounds, cons=ConSet())
        s_out = apply_dense(s_in, W, b_n)
        self._store_state(L.id, s_out)
        return Fact(bounds=_fchz_to_bounds(s_out, dev=input_bounds.lb.device,
                                                          dtype=input_bounds.lb.dtype),
                       cons=ConSet())

    def _tf_relu(self, L: Layer, input_bounds: Bounds) -> Fact:
        """RELU: triangle relaxation with CORRELATED slack columns (TIGHT).

        Adds one new G column per unstable neuron (slack variable xi ∈ [-1, 1]).
        This matches raw walker's apply_relu_triangle_with_record approach —
        TIGHTER than per-row independent tail_radius for deep networks.

        y = lam*z + mu + mu*xi  where xi is a per-neuron shared slack.
        """
        s_in = self._get_input_fchz(L, input_bounds)
        n = s_in.c.shape[0]
        K = s_in.G.shape[1]

        # SOUNDNESS NOTE (2026-06-11 fix):
        #   The bounds (l, u) used to classify neurons as stable-active /
        #   stable-inactive / unstable and to compute the triangle slope
        #   (lam) / intercept (mu) MUST be a sound box-hull of the pre-
        #   activation z = s_in.c + s_in.G @ xi (+ s_in.tail @ tail_noise).
        #
        #   The previous implementation used G ONLY (`G_rad = sum|G|`),
        #   which was correct under the design assumption that no tail
        #   was active. But sparse-slack mode (`HYZOR_FCHZ_G_MAX_COLS`)
        #   compresses dropped G columns into `tail_radius`; ignoring tail
        #   here gave artificially tight (l, u) bounds, causing some
        #   unstable neurons to be misclassified as stable-active. The
        #   downstream MILP then encoded `y = z` for those neurons, which
        #   is wrong when z can actually be negative — and produced unsound
        #   CERTs on benchmarks that triggered compression (tllverifybench,
        #   cersyve, etc.).
        #
        #   Fix: include the tail_radius in the per-neuron bound. This
        #   restores the sound box-hull and matches the bound returned by
        #   _fchz_to_bounds().
        G_rad = np.abs(s_in.G).sum(axis=1)
        if s_in.tail_radius is not None:
            G_rad = G_rad + s_in.tail_radius
        l = s_in.c - G_rad
        u = s_in.c + G_rad

        is_active = l >= 0
        is_inactive = u <= 0
        is_unstable = ~is_active & ~is_inactive

        unstable_idx = np.where(is_unstable)[0]
        n_unstable = len(unstable_idx)

        den = np.where(is_unstable, u - l, 1.0)
        lam = np.where(is_unstable, u / np.maximum(den, 1e-300), 0.0)
        lam = np.where(is_active, 1.0, lam)
        lam = np.where(is_inactive, 0.0, lam)
        mu = np.where(is_unstable, -lam * l / 2.0, 0.0)

        # New center: y_c = lam * z_c + mu
        new_c = lam * s_in.c + mu
        # Scale existing G by lam (per-row)
        new_G_base = s_in.G * lam[:, None]

        # Add unstable-neuron slack columns: y[i] = ... + mu[i] * xi_new_i, xi_new_i ∈ [-1, 1]
        if n_unstable > 0:
            slack_cols = np.zeros((n, n_unstable), dtype=s_in.G.dtype)
            for slot, i in enumerate(unstable_idx):
                slack_cols[i, slot] = mu[i]
            new_G = np.concatenate([new_G_base, slack_cols], axis=1)
        else:
            new_G = new_G_base

        # Match raw walker: drop tail_radius at ReLU (apply_relu_triangle_with_record
        # does not propagate tail). The slack columns now carry the unstable
        # behavior; downstream layers will rebuild tail if compression triggers.
        new_tail = None

        # Build slack record (for potential LP refinement downstream)
        from act.back_end.fchz_tf.representations import SlackRecord
        layer_idx = len(s_in.slack_records)
        rec = SlackRecord(
            layer_index=layer_idx,
            c_z=s_in.c.copy(),
            G_z=s_in.G.copy(),
            l=l.copy(),
            u=u.copy(),
            unstable_mask_arr=is_unstable.copy(),
            slack_indices=np.arange(K, K + n_unstable, dtype=np.int64),
        )
        new_records = list(s_in.slack_records) + [rec]

        s_out = FCHZState(c=new_c, G=new_G, n_root=s_in.n_root,
                                slack_records=new_records,
                                tail_radius=new_tail)
        self._store_state(L.id, s_out)
        return Fact(bounds=_fchz_to_bounds(s_out, dev=input_bounds.lb.device,
                                                          dtype=input_bounds.lb.dtype),
                       cons=ConSet())

    def _tf_sigmoid_tanh(self, L: Layer, input_bounds: Bounds, kind: str) -> Fact:
        """SIGMOID / TANH: analytical chord with sound radius."""
        s_in = self._get_input_fchz(L, input_bounds)
        rad = np.abs(s_in.G).sum(axis=1)
        if s_in.tail_radius is not None:
            rad = rad + s_in.tail_radius
        l = s_in.c - rad
        u = s_in.c + rad

        alpha, beta, radius = chord_params(l, u, kind)
        new_c = alpha * s_in.c + beta
        new_G = s_in.G * alpha[:, None]
        new_tail = None
        if s_in.tail_radius is not None:
            new_tail = np.abs(alpha) * s_in.tail_radius
        if np.any(radius > 0):
            new_tail = (new_tail + radius) if new_tail is not None else radius

        s_out = FCHZState(c=new_c, G=new_G, n_root=s_in.n_root,
                              slack_records=s_in.slack_records,
                              tail_radius=new_tail)
        self._store_state(L.id, s_out)
        return Fact(bounds=_fchz_to_bounds(s_out, dev=input_bounds.lb.device,
                                                          dtype=input_bounds.lb.dtype),
                       cons=ConSet())

    def _tf_bias(self, L: Layer, input_bounds: Bounds) -> Fact:
        """BIAS: c += b, G unchanged, tail_radius preserved.

        ACT BIAS layer uses param name 'c' (constant bias vector).
        Handle broadcast: if b.shape[0] != state.n, try:
          - if state.n == b.shape[0]: direct add
          - if state.n is a multiple of b.shape[0]: per-row tile
          - if b.shape[0] is a multiple of state.n: per-channel repeat (state expanded)
          - if input_shape is available: reshape state, broadcast bias, flatten
        """
        bias = L.params.get('c')
        if bias is None:
            bias = L.params.get('bias')
        if bias is None:
            return Fact(bounds=input_bounds, cons=ConSet())
        b = _t2n(bias).reshape(-1)
        s_in = self._get_input_fchz(L, input_bounds)

        # Broadcast handling
        if s_in.c.shape[0] != b.shape[0]:
            n_state = s_in.c.shape[0]
            n_b = b.shape[0]
            # Try input_shape-based broadcasting (most reliable)
            input_shape = L.params.get('input_shape')
            if input_shape is not None:
                in_shape_no_batch = tuple(input_shape[1:]) if input_shape[0] in (0, 1) else tuple(input_shape)
                target_n = int(np.prod(in_shape_no_batch))
                if n_state == target_n:
                    # state matches input_shape; broadcast b over it
                    try:
                        c_nd = s_in.c.reshape(in_shape_no_batch)
                        # Find broadcasting axis
                        b_shape = (1,) * (len(in_shape_no_batch) - 1) + (n_b,)
                        for sh in [b_shape,
                                       (n_b,) + (1,) * (len(in_shape_no_batch) - 1)]:
                            try:
                                b_broadcast = b.reshape(sh)
                                result_c = c_nd + b_broadcast
                                new_c = result_c.reshape(-1)
                                # G and tail unchanged (bias is constant)
                                s_out = FCHZState(c=new_c, G=s_in.G.copy(),
                                                        n_root=s_in.n_root,
                                                        slack_records=s_in.slack_records,
                                                        tail_radius=(s_in.tail_radius.copy()
                                                                            if s_in.tail_radius is not None else None))
                                self._store_state(L.id, s_out)
                                return Fact(bounds=_fchz_to_bounds(s_out, dev=input_bounds.lb.device,
                                                                              dtype=input_bounds.lb.dtype),
                                               cons=ConSet())
                            except ValueError:
                                continue
                    except Exception:
                        pass
            # Try simple divisibility
            if n_state % n_b == 0:
                # per-row repeat: b applies to every n_state/n_b group
                b = np.tile(b, n_state // n_b)
            elif n_b % n_state == 0:
                # state is per-row, b is full: tile state? this doesn't make sense
                # Drop bias (loose but sound)
                return Fact(bounds=input_bounds, cons=ConSet())
            else:
                # No clean broadcast; fall back
                return Fact(bounds=input_bounds, cons=ConSet())

        new_c = s_in.c + b
        s_out = FCHZState(c=new_c, G=s_in.G.copy(), n_root=s_in.n_root,
                              slack_records=s_in.slack_records,
                              tail_radius=(s_in.tail_radius.copy()
                                                  if s_in.tail_radius is not None else None))
        self._store_state(L.id, s_out)
        return Fact(bounds=_fchz_to_bounds(s_out, dev=input_bounds.lb.device,
                                                          dtype=input_bounds.lb.dtype),
                       cons=ConSet())

    def _tf_scale(self, L: Layer, input_bounds: Bounds) -> Fact:
        """SCALE: c *= a, G *= a, tail_radius *= |a|.

        ACT SCALE layer uses param 'a' (scale vector).
        """
        a_param = L.params.get('a')
        if a_param is None:
            a_param = L.params.get('scale')
        if a_param is None:
            return Fact(bounds=input_bounds, cons=ConSet())
        a = _t2n(a_param).reshape(-1)
        s_in = self._get_input_fchz(L, input_bounds)
        new_tail = None
        if s_in.tail_radius is not None:
            new_tail = np.abs(a) * s_in.tail_radius
        s_out = FCHZState(c=s_in.c * a, G=s_in.G * a[:, None],
                              n_root=s_in.n_root,
                              slack_records=s_in.slack_records,
                              tail_radius=new_tail)
        self._store_state(L.id, s_out)
        return Fact(bounds=_fchz_to_bounds(s_out, dev=input_bounds.lb.device,
                                                          dtype=input_bounds.lb.dtype),
                       cons=ConSet())

    def _tf_passthrough(self, L: Layer, input_bounds: Bounds) -> Fact:
        """Identity / no-op layer — propagate state unchanged."""
        s_in = self._get_input_fchz(L, input_bounds)
        self._store_state(L.id, s_in)
        return Fact(bounds=_fchz_to_bounds(s_in, dev=input_bounds.lb.device,
                                                          dtype=input_bounds.lb.dtype),
                       cons=ConSet())

    # ---------------------------------------------------------------------
    # CNN-layer handlers
    # ---------------------------------------------------------------------

    def _tf_conv2d(self, L: Layer, input_bounds: Bounds) -> Fact:
        """CONV2D: convolve c, G, tail via Pytorch F.conv2d."""
        W = _t2n(L.params.get('weight'))
        b = L.params.get('bias')
        b_n = _t2n(b).reshape(-1) if b is not None else None
        stride = L.params.get('stride', 1)
        padding = L.params.get('padding', 0)
        groups = int(L.params.get('groups', 1))

        # Determine input shape (Ci, Hi, Wi)
        Ci = int(L.params.get('in_channels', W.shape[1] * groups))
        input_shape = L.params.get('input_shape')
        if input_shape is None:
            # Try predecessor shape cache
            pred_id = self._net.preds.get(L.id, [None])[0] if self._net else None
            if pred_id is not None and pred_id in self._shape_cache:
                input_shape = self._shape_cache[pred_id]
            else:
                # Fall back to interval
                return Fact(bounds=input_bounds, cons=ConSet())
        if isinstance(input_shape, (tuple, list)) and len(input_shape) == 4:
            input_shape = tuple(input_shape[1:])  # strip batch
        s_in = self._get_input_fchz(L, input_bounds)
        s_out, out_shape = tf_cnn.apply_conv2d(s_in, W, b_n, input_shape,
                                                              stride=stride, padding=padding,
                                                              groups=groups)
        self._store_state(L.id, s_out)
        self._shape_cache[L.id] = out_shape
        return Fact(bounds=_fchz_to_bounds(s_out, dev=input_bounds.lb.device,
                                                          dtype=input_bounds.lb.dtype),
                       cons=ConSet())

    def _tf_conv1d(self, L: Layer, input_bounds: Bounds) -> Fact:
        """CONV1D: convolve over 1 spatial dim.

        Implementation: reshape state to 4D (treat as 2D conv with W_spatial=1),
        reuse apply_conv2d. Sound — 1D and 2D conv on same data are equivalent
        when one spatial dim is degenerate.
        """
        if not hasattr(self, '_state_loss_log'):
            self._state_loss_log = []
        W = _t2n(L.params.get('weight'))
        b = L.params.get('bias')
        b_n = _t2n(b).reshape(-1) if b is not None else None
        if W is None:
            self._state_loss_log.append({
                'layer_id': L.id, 'kind': 'CONV1D',
                'reason': 'missing_weight',
            })
            return Fact(bounds=input_bounds, cons=ConSet())
        # CONV1D weight: (out_channels, in_channels/groups, kernel)
        # Reshape to (out_channels, in_channels/groups, kernel, 1) for 2D
        if W.ndim == 3:
            W_2d = W[:, :, :, None]
        else:
            self._state_loss_log.append({
                'layer_id': L.id, 'kind': 'CONV1D',
                'reason': 'unexpected_weight_dim',
                'W_ndim': int(W.ndim),
            })
            return Fact(bounds=input_bounds, cons=ConSet())

        stride = L.params.get('stride', 1)
        if isinstance(stride, (list, tuple)) and len(stride) == 1:
            stride_2d = (int(stride[0]), 1)
        else:
            stride_2d = (int(stride) if isinstance(stride, int) else int(stride[0]), 1)
        padding = L.params.get('padding', 0)
        if isinstance(padding, (list, tuple)) and len(padding) == 1:
            padding_2d = (int(padding[0]), 0)
        elif isinstance(padding, (list, tuple)) and len(padding) == 2:
            padding_2d = (int(padding[0]), 0)
        else:
            padding_2d = (int(padding) if isinstance(padding, int) else int(padding[0]), 0)
        groups = int(L.params.get('groups', 1))

        # Input shape: (Ci, Li) for 1D → reshape to (Ci, Li, 1) for 2D
        input_shape = L.params.get('input_shape')
        if input_shape is None:
            pred_id = self._net.preds.get(L.id, [None])[0] if self._net else None
            if pred_id is not None and pred_id in self._shape_cache:
                input_shape = self._shape_cache[pred_id]
        if isinstance(input_shape, (tuple, list)) and len(input_shape) == 3:
            input_shape = tuple(input_shape[1:])   # strip batch
        if input_shape is None or len(input_shape) != 2:
            self._state_loss_log.append({
                'layer_id': L.id, 'kind': 'CONV1D',
                'reason': 'unexpected_input_shape',
                'input_shape': str(input_shape),
            })
            return Fact(bounds=input_bounds, cons=ConSet())
        Ci, Li = int(input_shape[0]), int(input_shape[1])
        input_shape_2d = (Ci, Li, 1)

        s_in = self._get_input_fchz(L, input_bounds)
        try:
            s_out, out_shape_2d = tf_cnn.apply_conv2d(
                s_in, W_2d, b_n, input_shape_2d,
                stride=stride_2d, padding=padding_2d, groups=groups)
        except Exception as e:
            self._state_loss_log.append({
                'layer_id': L.id, 'kind': 'CONV1D',
                'reason': 'apply_conv2d_exception',
                'exc': str(e)[:100],
            })
            return Fact(bounds=input_bounds, cons=ConSet())
        # out_shape_2d is (Co, Lo, 1). For CONV1D, output is (Co, Lo).
        Co, Lo, _ = out_shape_2d
        out_shape_1d = (Co, Lo)
        # Shape contract validation
        output_shape = L.params.get('output_shape')
        if output_shape is not None:
            expected_dim = int(np.prod(output_shape[1:])) if len(output_shape) > 1 and output_shape[0] in (0, 1) else int(np.prod(output_shape))
            if s_out.c.shape[0] != expected_dim:
                self._state_loss_log.append({
                    'layer_id': L.id, 'kind': 'CONV1D',
                    'reason': 'shape_contract_violation',
                    'expected_dim': expected_dim,
                    'actual_dim': int(s_out.c.shape[0]),
                })
                return Fact(bounds=input_bounds, cons=ConSet())
        self._store_state(L.id, s_out)
        self._shape_cache[L.id] = out_shape_1d
        return Fact(bounds=_fchz_to_bounds(s_out, dev=input_bounds.lb.device,
                                                          dtype=input_bounds.lb.dtype),
                       cons=ConSet())

    def _tf_bn(self, L: Layer, input_bounds: Bounds) -> Fact:
        """BatchNorm: y = A*x + c (precomputed in ACT schema)."""
        A = _t2n(L.params.get('A')).reshape(-1)
        c_param = _t2n(L.params.get('c')).reshape(-1)
        # Need input shape for per-channel broadcast
        pred_id = self._net.preds.get(L.id, [None])[0] if self._net else None
        input_shape = self._shape_cache.get(pred_id) if pred_id is not None else None
        s_in = self._get_input_fchz(L, input_bounds)
        if input_shape is not None and len(input_shape) == 3:
            s_out = tf_cnn.apply_bn(s_in, A, c_param, input_shape)
            self._shape_cache[L.id] = input_shape
        else:
            # 1D BN: just element-wise
            new_c = A * s_in.c + c_param
            new_G = s_in.G * A[:, None]
            new_tail = (np.abs(A) * s_in.tail_radius) if s_in.tail_radius is not None else None
            s_out = FCHZState(c=new_c, G=new_G, n_root=s_in.n_root,
                                  slack_records=s_in.slack_records,
                                  tail_radius=new_tail)
        self._store_state(L.id, s_out)
        return Fact(bounds=_fchz_to_bounds(s_out, dev=input_bounds.lb.device,
                                                          dtype=input_bounds.lb.dtype),
                       cons=ConSet())

    def _tf_maxpool2d(self, L: Layer, input_bounds: Bounds) -> Fact:
        """MaxPool2D: sound box relaxation."""
        kernel = L.params.get('kernel_size')
        if isinstance(kernel, int): kernel = (kernel, kernel)
        stride = L.params.get('stride', kernel)
        if isinstance(stride, int): stride = (stride, stride)
        padding = L.params.get('padding', 0)
        if isinstance(padding, int): padding = (padding, padding)

        pred_id = self._net.preds.get(L.id, [None])[0] if self._net else None
        input_shape = self._shape_cache.get(pred_id) if pred_id is not None else None
        if input_shape is None or len(input_shape) != 3:
            return Fact(bounds=input_bounds, cons=ConSet())

        s_in = self._get_input_fchz(L, input_bounds)
        s_out, out_shape = tf_cnn.apply_maxpool2d(s_in, tuple(kernel), tuple(stride),
                                                                  tuple(padding), input_shape)
        self._store_state(L.id, s_out)
        self._shape_cache[L.id] = out_shape
        return Fact(bounds=_fchz_to_bounds(s_out, dev=input_bounds.lb.device,
                                                          dtype=input_bounds.lb.dtype),
                       cons=ConSet())

    def _tf_flatten(self, L: Layer, input_bounds: Bounds) -> Fact:
        """FLATTEN: no-op in FCHZState (always flat internally). Pass state through."""
        s_in = self._get_input_fchz(L, input_bounds)
        # Just store same state under new layer id (no math change)
        self._store_state(L.id, s_in)
        # Shape: take input shape, flatten everything past dim 0
        in_shape = self._shape_cache.get(self._net.preds.get(L.id, [None])[0]) if self._net else None
        if in_shape is not None:
            self._shape_cache[L.id] = (int(np.prod(in_shape)),)
        return Fact(bounds=_fchz_to_bounds(s_in, dev=input_bounds.lb.device,
                                                          dtype=input_bounds.lb.dtype),
                       cons=ConSet())

    def _tf_add(self, L: Layer, input_bounds: Bounds) -> Fact:
        """ADD: residual connection. Two paths sharing the same INPUT ROOT.

        SOUNDNESS NOTE (2026-06-11 fix):
            The previous element-wise sum assumed that columns at the same
            index in s_a.G and s_b.G correspond to the SAME noise variable
            xi. This is true for the FIRST n_root columns (the input-box
            xi shared by both branches), and it is also true in the
            ResNet skip-pattern where one branch is a pure identity and
            therefore contributes no post-ReLU xi at all. But when BOTH
            branches have introduced their own post-ReLU xi (e.g., the
            cersyve pendulum_pretrain_inv model), the element-wise sum
            of column i for i >= n_root treats two INDEPENDENT xi as if
            they were the same variable, which is UNSOUND — the resulting
            G can produce ranges that artificially cancel (an actual
            counterexample falls outside the computed bound).

            The corrected handler:
              * Sums the first n_root columns element-wise (these are
                the shared input-box xi, genuinely correlated).
              * Concatenates the remaining columns horizontally (each
                branch's post-ReLU xi is treated as independent of the
                other branch's, which is sound — adding independent
                xi increases the zonotope's extent rather than canceling
                it).

            Behavior is UNCHANGED for the ResNet skip-pattern (one branch
            has only root xi, K == n_root). The change matters only when
            both branches have introduced new post-ReLU xi, in which case
            the new behavior is the sound interpretation.

            Matches raw walker (research/sc_hz/fchz_walker.py L271-309)
            only for the shared-root case; the multi-branch case is a
            soundness correction beyond what the raw walker handles.
        """
        if self._net is None or L.id not in self._net.preds:
            return Fact(bounds=input_bounds, cons=ConSet())
        pred_ids = self._net.preds[L.id]
        if len(pred_ids) != 2:
            return Fact(bounds=input_bounds, cons=ConSet())
        s_a = self._state_cache.get(pred_ids[0])
        s_b = self._state_cache.get(pred_ids[1])
        if s_a is None or s_b is None:
            return Fact(bounds=input_bounds, cons=ConSet())
        if s_a.c.shape != s_b.c.shape:
            return Fact(bounds=input_bounds, cons=ConSet())

        new_c = s_a.c + s_b.c
        n = new_c.shape[0]
        # n_root must agree between branches (both started from the same
        # InputSpec). If they disagree, take the minimum and treat the
        # remainder as independent — the conservative choice.
        n_root_a = int(getattr(s_a, 'n_root', 0) or 0)
        n_root_b = int(getattr(s_b, 'n_root', 0) or 0)
        n_root = min(n_root_a, n_root_b)
        K_a = s_a.G.shape[1]
        K_b = s_b.G.shape[1]

        # Shared root columns: sum element-wise. Both branches' first
        # n_root columns are coefficients on the SAME input-box xi, so
        # element-wise sum is sound and tight.
        K_shared = min(n_root, K_a, K_b)
        if K_shared > 0:
            G_root_sum = s_a.G[:, :K_shared] + s_b.G[:, :K_shared]
        else:
            G_root_sum = np.zeros((n, 0), dtype=s_a.G.dtype)

        # Independent post-root columns: concatenate (each branch's
        # post-ReLU xi is a fresh independent variable in the other
        # branch's frame).
        G_a_indep = s_a.G[:, K_shared:] if K_a > K_shared else np.zeros((n, 0), dtype=s_a.G.dtype)
        G_b_indep = s_b.G[:, K_shared:] if K_b > K_shared else np.zeros((n, 0), dtype=s_b.G.dtype)
        new_G = np.concatenate([G_root_sum, G_a_indep, G_b_indep], axis=1)

        # Tail radii: sum (independent box error sources)
        ta = s_a.tail_radius if s_a.tail_radius is not None else 0.0
        tb = s_b.tail_radius if s_b.tail_radius is not None else 0.0
        new_tail = (ta + tb) if (s_a.tail_radius is not None or s_b.tail_radius is not None) else None
        if new_tail is not None and not isinstance(new_tail, np.ndarray):
            new_tail = np.full(n, float(new_tail))

        # Slack records: union (each branch's records correspond to its
        # own post-ReLU xi, which are now distinct columns in new_G).
        merged_records = list(s_a.slack_records) + list(s_b.slack_records)

        s_out = FCHZState(c=new_c, G=new_G, n_root=n_root,
                                slack_records=merged_records,
                                tail_radius=new_tail)
        self._store_state(L.id, s_out)
        in_shape = self._shape_cache.get(pred_ids[0]) or self._shape_cache.get(pred_ids[1])
        if in_shape is not None:
            self._shape_cache[L.id] = in_shape
        return Fact(bounds=_fchz_to_bounds(s_out, dev=input_bounds.lb.device,
                                                          dtype=input_bounds.lb.dtype),
                       cons=ConSet())

    def _tf_sub(self, L: Layer, input_bounds: Bounds) -> Fact:
        """SUB: x - y where both are FCHZStates.

        SOUNDNESS NOTE (2026-06-11 fix): mirrors the ADD correction — only
        the first n_root columns (shared input-box xi) are subtracted
        element-wise; each branch's post-ReLU xi are kept independent
        (concat with appropriate sign).
        """
        if self._net is None or L.id not in self._net.preds:
            return Fact(bounds=input_bounds, cons=ConSet())
        pred_ids = self._net.preds[L.id]
        if len(pred_ids) != 2:
            return Fact(bounds=input_bounds, cons=ConSet())
        s_a = self._state_cache.get(pred_ids[0])
        s_b = self._state_cache.get(pred_ids[1])
        if s_a is None or s_b is None:
            return Fact(bounds=input_bounds, cons=ConSet())
        if s_a.c.shape != s_b.c.shape:
            return Fact(bounds=input_bounds, cons=ConSet())

        new_c = s_a.c - s_b.c
        n = new_c.shape[0]
        n_root_a = int(getattr(s_a, 'n_root', 0) or 0)
        n_root_b = int(getattr(s_b, 'n_root', 0) or 0)
        n_root = min(n_root_a, n_root_b)
        K_a = s_a.G.shape[1]; K_b = s_b.G.shape[1]
        K_shared = min(n_root, K_a, K_b)
        if K_shared > 0:
            G_root_diff = s_a.G[:, :K_shared] - s_b.G[:, :K_shared]
        else:
            G_root_diff = np.zeros((n, 0), dtype=s_a.G.dtype)
        G_a_indep = s_a.G[:, K_shared:] if K_a > K_shared else np.zeros((n, 0), dtype=s_a.G.dtype)
        # The s_b independent columns get a sign flip because we're subtracting.
        G_b_indep_neg = (-s_b.G[:, K_shared:]) if K_b > K_shared else np.zeros((n, 0), dtype=s_b.G.dtype)
        new_G = np.concatenate([G_root_diff, G_a_indep, G_b_indep_neg], axis=1)

        ta = s_a.tail_radius if s_a.tail_radius is not None else 0.0
        tb = s_b.tail_radius if s_b.tail_radius is not None else 0.0
        new_tail = (ta + tb) if (s_a.tail_radius is not None or s_b.tail_radius is not None) else None
        if new_tail is not None and not isinstance(new_tail, np.ndarray):
            new_tail = np.full(n, float(new_tail))

        merged_records = list(s_a.slack_records) + list(s_b.slack_records)
        s_out = FCHZState(c=new_c, G=new_G, n_root=n_root,
                                slack_records=merged_records,
                                tail_radius=new_tail)
        self._store_state(L.id, s_out)
        return Fact(bounds=_fchz_to_bounds(s_out, dev=input_bounds.lb.device,
                                                          dtype=input_bounds.lb.dtype),
                       cons=ConSet())

    def _tf_mul(self, L: Layer, input_bounds: Bounds) -> Fact:
        """MUL: element-wise. If one is a constant, like SCALE. If both are states, fall back to interval."""
        if self._net is None or L.id not in self._net.preds:
            return Fact(bounds=input_bounds, cons=ConSet())
        pred_ids = self._net.preds[L.id]
        if len(pred_ids) != 2:
            return Fact(bounds=input_bounds, cons=ConSet())
        s_a = self._state_cache.get(pred_ids[0])
        s_b = self._state_cache.get(pred_ids[1])
        # If both states: nonlinear, fall back to interval
        if s_a is None or s_b is None:
            return Fact(bounds=input_bounds, cons=ConSet())
        # Detect constant operand: G empty, tail None
        const_state = None; var_state = None
        if s_a.G.shape[1] == 0 and (s_a.tail_radius is None or float(s_a.tail_radius.max()) < 1e-12):
            const_state, var_state = s_a, s_b
        elif s_b.G.shape[1] == 0 and (s_b.tail_radius is None or float(s_b.tail_radius.max()) < 1e-12):
            const_state, var_state = s_b, s_a
        if const_state is None:
            return Fact(bounds=input_bounds, cons=ConSet())
        scale = const_state.c
        if scale.shape != var_state.c.shape:
            try:
                scale = np.broadcast_to(scale, var_state.c.shape)
            except Exception:
                return Fact(bounds=input_bounds, cons=ConSet())
        new_c = scale * var_state.c
        new_G = scale[:, None] * var_state.G
        new_tail = np.abs(scale) * var_state.tail_radius if var_state.tail_radius is not None else None
        s_out = FCHZState(c=new_c, G=new_G, n_root=var_state.n_root,
                                slack_records=var_state.slack_records,
                                tail_radius=new_tail)
        self._store_state(L.id, s_out)
        return Fact(bounds=_fchz_to_bounds(s_out, dev=input_bounds.lb.device,
                                                          dtype=input_bounds.lb.dtype),
                       cons=ConSet())

    def _tf_concat(self, L: Layer, input_bounds: Bounds) -> Fact:
        """CONCAT: stack states from N predecessors along given axis.

        Output: c' = [c_0, c_1, ...] (flat), G' = vstack of padded Gs.
        """
        if self._net is None or L.id not in self._net.preds:
            return Fact(bounds=input_bounds, cons=ConSet())
        pred_ids = self._net.preds[L.id]
        states = [self._state_cache.get(p) for p in pred_ids]
        if any(s is None for s in states):
            return Fact(bounds=input_bounds, cons=ConSet())
        # Stack centers
        new_c = np.concatenate([s.c for s in states], axis=0)
        # K_max = max K across predecessors
        K_max = max(s.G.shape[1] for s in states)
        # Pad each G to K_max, then vstack
        padded_Gs = []
        for s in states:
            if s.G.shape[1] < K_max:
                pad = np.zeros((s.c.shape[0], K_max - s.G.shape[1]), dtype=s.G.dtype)
                padded_Gs.append(np.concatenate([s.G, pad], axis=1))
            else:
                padded_Gs.append(s.G)
        new_G = np.concatenate(padded_Gs, axis=0)
        # tail_radius: concat
        tails = []
        any_tail = False
        for s in states:
            if s.tail_radius is not None:
                tails.append(s.tail_radius); any_tail = True
            else:
                tails.append(np.zeros(s.c.shape[0]))
        new_tail = np.concatenate(tails, axis=0) if any_tail else None
        s_out = FCHZState(c=new_c, G=new_G, n_root=states[0].n_root,
                                slack_records=states[0].slack_records,
                                tail_radius=new_tail)
        self._store_state(L.id, s_out)
        return Fact(bounds=_fchz_to_bounds(s_out, dev=input_bounds.lb.device,
                                                          dtype=input_bounds.lb.dtype),
                       cons=ConSet())

    def _tf_reduce_sum(self, L: Layer, input_bounds: Bounds) -> Fact:
        """REDUCE_SUM: sum along given axes.

        For flat state, sum equals sum of c, sum of each G col, sum of tail.
        Assumes axes form full reduction over a known input_shape dimension.
        """
        s_in = self._get_input_fchz(L, input_bounds)
        axes = L.params.get('axes', [])
        keepdims = L.params.get('keepdims', 1)
        input_shape = L.params.get('input_shape')
        output_shape = L.params.get('output_shape')

        # If we can determine input/output shapes, reshape, sum, flatten
        if input_shape is None or output_shape is None:
            return Fact(bounds=input_bounds, cons=ConSet())

        # Strip batch dim if present
        in_shape = tuple(input_shape[1:]) if len(input_shape) > 0 and input_shape[0] in (0, 1) else tuple(input_shape)
        out_shape = tuple(output_shape[1:]) if len(output_shape) > 0 and output_shape[0] in (0, 1) else tuple(output_shape)

        try:
            c_in_nd = s_in.c.reshape(in_shape)
            # Normalize axes (remove batch offset)
            norm_axes = tuple(a - 1 if a > 0 else a for a in axes)
            new_c = c_in_nd.sum(axis=norm_axes, keepdims=bool(keepdims)).reshape(-1)
            K = s_in.G.shape[1]
            if K > 0:
                G_in_nd = s_in.G.reshape(in_shape + (K,))
                # Sum over the input axes (not the K axis)
                new_G = G_in_nd.sum(axis=norm_axes, keepdims=bool(keepdims)).reshape(-1, K)
            else:
                new_G = np.zeros((new_c.shape[0], 0), dtype=s_in.G.dtype)
            new_tail = None
            if s_in.tail_radius is not None:
                tail_in_nd = s_in.tail_radius.reshape(in_shape)
                new_tail = np.abs(tail_in_nd).sum(axis=norm_axes, keepdims=bool(keepdims)).reshape(-1)
            s_out = FCHZState(c=new_c, G=new_G, n_root=s_in.n_root,
                                    slack_records=s_in.slack_records,
                                    tail_radius=new_tail)
            self._store_state(L.id, s_out)
            return Fact(bounds=_fchz_to_bounds(s_out, dev=input_bounds.lb.device,
                                                              dtype=input_bounds.lb.dtype),
                           cons=ConSet())
        except Exception:
            return Fact(bounds=input_bounds, cons=ConSet())

    def _tf_gather(self, L: Layer, input_bounds: Bounds) -> Fact:
        """GATHER: pick subset of indices along given axis.

        Shape contract (advisor P5 2026-06-08):
          new_c.shape[0] must equal product(output_shape[1:]).
          On any violation: log SHAPE_CONTRACT_VIOLATION and fail-closed
          (don't store wrong state; downstream fall-back gets honest fresh-box).
        """
        if not hasattr(self, '_state_loss_log'):
            self._state_loss_log = []
        s_in = self._get_input_fchz(L, input_bounds)
        indices = L.params.get('indices')
        axis = L.params.get('axis', 0)
        input_shape = L.params.get('input_shape')
        output_shape = L.params.get('output_shape')
        if indices is None or input_shape is None:
            self._state_loss_log.append({
                'layer_id': L.id, 'kind': 'GATHER',
                'reason': 'missing_params',
                'has_indices': indices is not None,
                'has_input_shape': input_shape is not None,
            })
            return Fact(bounds=input_bounds, cons=ConSet())
        idx = _t2n(indices).astype(np.int64).reshape(-1)
        in_shape = tuple(input_shape[1:]) if len(input_shape) > 0 and input_shape[0] in (0, 1) else tuple(input_shape)
        expected_out_dim = (int(np.prod(output_shape[1:])) if output_shape is not None
                                 and len(output_shape) > 1 and output_shape[0] in (0, 1)
                                 else None)
        try:
            ax = axis - 1 if axis > 0 else axis
            c_nd = s_in.c.reshape(in_shape)
            new_c = np.take(c_nd, idx, axis=ax).reshape(-1)
            K = s_in.G.shape[1]
            if K > 0:
                G_nd = s_in.G.reshape(in_shape + (K,))
                new_G = np.take(G_nd, idx, axis=ax).reshape(-1, K)
            else:
                new_G = np.zeros((new_c.shape[0], 0), dtype=s_in.G.dtype)
            new_tail = None
            if s_in.tail_radius is not None:
                tail_nd = s_in.tail_radius.reshape(in_shape)
                new_tail = np.take(tail_nd, idx, axis=ax).reshape(-1)
            # Shape contract validation
            if expected_out_dim is not None and new_c.shape[0] != expected_out_dim:
                self._state_loss_log.append({
                    'layer_id': L.id, 'kind': 'GATHER',
                    'reason': 'shape_contract_violation',
                    'expected_dim': expected_out_dim,
                    'actual_dim': int(new_c.shape[0]),
                    'in_shape': list(in_shape), 'axis': axis,
                    'n_indices': len(idx),
                })
                # Fail-closed: DO NOT store wrong state
                return Fact(bounds=input_bounds, cons=ConSet())
            if new_G.shape[0] != new_c.shape[0]:
                self._state_loss_log.append({
                    'layer_id': L.id, 'kind': 'GATHER',
                    'reason': 'G_c_dim_mismatch',
                    'c_dim': int(new_c.shape[0]), 'G_dim': int(new_G.shape[0]),
                })
                return Fact(bounds=input_bounds, cons=ConSet())
            s_out = FCHZState(c=new_c, G=new_G, n_root=s_in.n_root,
                                    slack_records=s_in.slack_records,
                                    tail_radius=new_tail)
            self._store_state(L.id, s_out)
            return Fact(bounds=_fchz_to_bounds(s_out, dev=input_bounds.lb.device,
                                                              dtype=input_bounds.lb.dtype),
                           cons=ConSet())
        except Exception as e:
            self._state_loss_log.append({
                'layer_id': L.id, 'kind': 'GATHER',
                'reason': 'tf_gather_exception',
                'exc': str(e)[:100],
            })
            return Fact(bounds=input_bounds, cons=ConSet())

    def _tf_reshape(self, L: Layer, input_bounds: Bounds) -> Fact:
        """RESHAPE: state is always flat, so RESHAPE is identity at FCHZ level.

        Only update shape_cache (in case input/output_shape differ for downstream).
        """
        s_in = self._get_input_fchz(L, input_bounds)
        self._store_state(L.id, s_in)
        # Update shape from output_shape
        output_shape = L.params.get('output_shape')
        if output_shape is not None:
            self._shape_cache[L.id] = tuple(output_shape[1:]) if output_shape[0] in (0, 1) else tuple(output_shape)
        return Fact(bounds=_fchz_to_bounds(s_in, dev=input_bounds.lb.device,
                                                          dtype=input_bounds.lb.dtype),
                       cons=ConSet())

    def _tf_transpose(self, L: Layer, input_bounds: Bounds) -> Fact:
        """TRANSPOSE: permute axes of state's logical shape."""
        s_in = self._get_input_fchz(L, input_bounds)
        perm = L.params.get('perm')
        input_shape = L.params.get('input_shape')
        if perm is None or input_shape is None:
            self._store_state(L.id, s_in)
            return Fact(bounds=input_bounds, cons=ConSet())
        try:
            perm_list = list(perm)
            # Strip batch from input_shape and perm
            in_shape_nb = tuple(input_shape[1:]) if len(input_shape) > 0 and input_shape[0] in (0, 1) else tuple(input_shape)
            # Adjust perm for stripped batch
            if perm_list[0] == 0 and len(perm_list) == len(input_shape):
                perm_nb = tuple(p - 1 for p in perm_list[1:])
            else:
                perm_nb = tuple(perm_list)

            c_nd = s_in.c.reshape(in_shape_nb).transpose(perm_nb)
            new_c = c_nd.reshape(-1).copy()
            K = s_in.G.shape[1]
            if K > 0:
                G_nd = s_in.G.reshape(in_shape_nb + (K,))
                perm_G = perm_nb + (len(in_shape_nb),)
                G_t = G_nd.transpose(perm_G)
                new_G = G_t.reshape(-1, K).copy()
            else:
                new_G = np.zeros((new_c.shape[0], 0), dtype=s_in.G.dtype)
            new_tail = None
            if s_in.tail_radius is not None:
                tail_nd = s_in.tail_radius.reshape(in_shape_nb).transpose(perm_nb)
                new_tail = tail_nd.reshape(-1).copy()

            s_out = FCHZState(c=new_c, G=new_G, n_root=s_in.n_root,
                                    slack_records=s_in.slack_records,
                                    tail_radius=new_tail)
            self._store_state(L.id, s_out)
            out_shape = tuple(np.array(in_shape_nb)[list(perm_nb)])
            self._shape_cache[L.id] = out_shape
            return Fact(bounds=_fchz_to_bounds(s_out, dev=input_bounds.lb.device,
                                                              dtype=input_bounds.lb.dtype),
                           cons=ConSet())
        except Exception:
            self._store_state(L.id, s_in)
            return Fact(bounds=input_bounds, cons=ConSet())

    def _tf_constant(self, L: Layer, input_bounds: Bounds) -> Fact:
        """CONSTANT: a state with constant c, no noise (G empty, tail None)."""
        val = L.params.get('value')
        if val is None:
            return Fact(bounds=input_bounds, cons=ConSet())
        c = _t2n(val).reshape(-1)
        n = c.shape[0]
        s_out = FCHZState(c=c, G=np.zeros((n, 0), dtype=np.float64), n_root=0,
                                slack_records=[], tail_radius=None)
        self._store_state(L.id, s_out)
        output_shape = L.params.get('output_shape')
        if output_shape is not None:
            self._shape_cache[L.id] = tuple(output_shape[1:]) if output_shape[0] in (0, 1) else tuple(output_shape)
        # Build interval bounds matching state
        lb_t = torch.from_numpy(c).reshape(1, -1).to(input_bounds.lb.dtype).to(input_bounds.lb.device)
        ub_t = torch.from_numpy(c).reshape(1, -1).to(input_bounds.lb.dtype).to(input_bounds.lb.device)
        return Fact(bounds=Bounds(lb=lb_t, ub=ub_t), cons=ConSet())

    def _tf_slice(self, L: Layer, input_bounds: Bounds) -> Fact:
        """SLICE: select [starts:ends] along given axes."""
        s_in = self._get_input_fchz(L, input_bounds)
        starts = L.params.get('starts')
        ends = L.params.get('ends')
        axes = L.params.get('axes')
        input_shape = L.params.get('input_shape')
        if starts is None or ends is None or input_shape is None:
            return Fact(bounds=input_bounds, cons=ConSet())
        starts = _t2n(starts).astype(np.int64).reshape(-1) if hasattr(starts, 'shape') else np.asarray(starts, dtype=np.int64).reshape(-1)
        ends = _t2n(ends).astype(np.int64).reshape(-1) if hasattr(ends, 'shape') else np.asarray(ends, dtype=np.int64).reshape(-1)
        if axes is None:
            axes = list(range(len(starts)))
        else:
            axes = _t2n(axes).astype(np.int64).reshape(-1).tolist() if hasattr(axes, 'shape') else list(axes)

        if not hasattr(self, '_state_loss_log'):
            self._state_loss_log = []
        in_shape = tuple(input_shape[1:]) if len(input_shape) > 0 and input_shape[0] in (0, 1) else tuple(input_shape)
        output_shape = L.params.get('output_shape')
        expected_out_dim = (int(np.prod(output_shape[1:])) if output_shape is not None
                                 and len(output_shape) > 1 and output_shape[0] in (0, 1)
                                 else None)
        try:
            c_nd = s_in.c.reshape(in_shape)
            slicer = [slice(None)] * len(in_shape)
            for ax, st, ed in zip(axes, starts, ends):
                ax = ax - 1 if ax > 0 else ax  # strip batch
                if 0 <= ax < len(in_shape):
                    slicer[ax] = slice(int(st), int(ed))
            new_c = c_nd[tuple(slicer)].reshape(-1)
            K = s_in.G.shape[1]
            if K > 0:
                G_nd = s_in.G.reshape(in_shape + (K,))
                slicer_G = slicer + [slice(None)]
                new_G = G_nd[tuple(slicer_G)].reshape(-1, K)
            else:
                new_G = np.zeros((new_c.shape[0], 0), dtype=s_in.G.dtype)
            new_tail = None
            if s_in.tail_radius is not None:
                tail_nd = s_in.tail_radius.reshape(in_shape)
                new_tail = tail_nd[tuple(slicer)].reshape(-1)
            # Shape contract validation (advisor P5)
            if expected_out_dim is not None and new_c.shape[0] != expected_out_dim:
                self._state_loss_log.append({
                    'layer_id': L.id, 'kind': 'SLICE',
                    'reason': 'shape_contract_violation',
                    'expected_dim': expected_out_dim,
                    'actual_dim': int(new_c.shape[0]),
                })
                return Fact(bounds=input_bounds, cons=ConSet())
            if new_G.shape[0] != new_c.shape[0]:
                self._state_loss_log.append({
                    'layer_id': L.id, 'kind': 'SLICE',
                    'reason': 'G_c_dim_mismatch',
                    'c_dim': int(new_c.shape[0]), 'G_dim': int(new_G.shape[0]),
                })
                return Fact(bounds=input_bounds, cons=ConSet())
            s_out = FCHZState(c=new_c, G=new_G, n_root=s_in.n_root,
                                    slack_records=s_in.slack_records,
                                    tail_radius=new_tail)
            self._store_state(L.id, s_out)
            return Fact(bounds=_fchz_to_bounds(s_out, dev=input_bounds.lb.device,
                                                              dtype=input_bounds.lb.dtype),
                           cons=ConSet())
        except Exception as e:
            self._state_loss_log.append({
                'layer_id': L.id, 'kind': 'SLICE',
                'reason': 'tf_slice_exception',
                'exc': str(e)[:100],
            })
            return Fact(bounds=input_bounds, cons=ConSet())

    def _tf_unsupported(self, L: Layer, input_bounds: Bounds) -> Fact:
        """For ops we don't yet support, fall back to interval bound."""
        return Fact(bounds=input_bounds, cons=ConSet())

    # ---------------------------------------------------------------------
    # Layer registry
    # ---------------------------------------------------------------------

    _LAYER_REGISTRY = {
        LayerKind.INPUT.value:      lambda L, b, tf: tf._tf_input(L, b),
        LayerKind.INPUT_SPEC.value: lambda L, b, tf: tf._tf_input(L, b),
        LayerKind.ASSERT.value:     lambda L, b, tf: Fact(bounds=b, cons=ConSet()),

        # MLP linear ops
        LayerKind.DENSE.value:      lambda L, b, tf: tf._tf_dense(L, b),
        LayerKind.BIAS.value:       lambda L, b, tf: tf._tf_bias(L, b),
        LayerKind.SCALE.value:      lambda L, b, tf: tf._tf_scale(L, b),

        # Activations
        LayerKind.RELU.value:       lambda L, b, tf: tf._tf_relu(L, b),
        LayerKind.SIGMOID.value:    lambda L, b, tf: tf._tf_sigmoid_tanh(L, b, "Sigmoid"),
        LayerKind.TANH.value:       lambda L, b, tf: tf._tf_sigmoid_tanh(L, b, "Tanh"),

        # CNN
        LayerKind.CONV1D.value:     lambda L, b, tf: tf._tf_conv1d(L, b),
        LayerKind.CONV2D.value:     lambda L, b, tf: tf._tf_conv2d(L, b),
        LayerKind.BN.value:         lambda L, b, tf: tf._tf_bn(L, b),
        LayerKind.MAXPOOL2D.value:  lambda L, b, tf: tf._tf_maxpool2d(L, b),

        # Structural
        LayerKind.FLATTEN.value:    lambda L, b, tf: tf._tf_flatten(L, b),

        # Multi-input
        LayerKind.ADD.value:        lambda L, b, tf: tf._tf_add(L, b),
        LayerKind.SUB.value:        lambda L, b, tf: tf._tf_sub(L, b),
        LayerKind.MUL.value:        lambda L, b, tf: tf._tf_mul(L, b),
        LayerKind.CONCAT.value:     lambda L, b, tf: tf._tf_concat(L, b),

        # Reductions / indexing
        LayerKind.REDUCE_SUM.value: lambda L, b, tf: tf._tf_reduce_sum(L, b),
        LayerKind.GATHER.value:     lambda L, b, tf: tf._tf_gather(L, b),
        LayerKind.SLICE.value:      lambda L, b, tf: tf._tf_slice(L, b),

        # Shape ops
        LayerKind.RESHAPE.value:    lambda L, b, tf: tf._tf_reshape(L, b),
        LayerKind.TRANSPOSE.value:  lambda L, b, tf: tf._tf_transpose(L, b),
        LayerKind.CONSTANT.value:   lambda L, b, tf: tf._tf_constant(L, b),
    }

    # ---------------------------------------------------------------------
    # TransferFunction.apply
    # ---------------------------------------------------------------------

    def apply(self, L: Layer, input_bounds: Bounds, net: Net,
                  before: Dict[int, Fact], after: Dict[int, Fact]) -> Fact:
        """Main dispatch — called by ACT analysis loop per layer."""
        # Invalidate cache if net changed
        if id(net) != self._cache_net_id:
            self._state_cache.clear()
            self._cache_net_id = id(net)

        # Transient pointers for handler dispatch
        self._net = net
        self._before = before
        self._after = after

        kind = L.kind.upper()
        if kind in self._LAYER_REGISTRY:
            return self._LAYER_REGISTRY[kind](L, input_bounds, self)
        else:
            return self._tf_unsupported(L, input_bounds)
