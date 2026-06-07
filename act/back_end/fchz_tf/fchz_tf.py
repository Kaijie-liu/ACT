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
    """Build FCHZState from input bounds."""
    lb = _t2n(bounds.lb).reshape(-1)
    ub = _t2n(bounds.ub).reshape(-1)
    c = (lb + ub) / 2.0
    r = (ub - lb) / 2.0
    return initial_state(c, r)


def _fchz_to_bounds(state: FCHZState, dev=None, dtype=None) -> Bounds:
    """Extract Bounds (interval [lb, ub]) from FCHZState via HZ closed-form.

    For each dimension i, compute:
        ub_i = c_i + Σ|G_ik| + tail_radius_i
        lb_i = c_i - Σ|G_ik| - tail_radius_i
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

    def _get_input_fchz(self, L: Layer, input_bounds: Bounds) -> FCHZState:
        """Get FCHZState for layer's input.

        If predecessor layer was processed and cached → use it.
        Otherwise build from input_bounds (lossy: treats it as fresh box).
        """
        # Predecessor layer: use net.preds graph
        if self._net is not None and L.id in self._net.preds:
            pred_ids = self._net.preds[L.id]
            if len(pred_ids) == 1 and pred_ids[0] in self._state_cache:
                return self._state_cache[pred_ids[0]]
        # Fall back to fresh box
        return _bounds_to_fchz(input_bounds)

    def _store_state(self, layer_id: int, state: FCHZState) -> None:
        """Store FCHZState for layer; apply sparse-slack if budget exceeded."""
        if self.G_max_cols is not None and state.G.shape[1] > self.G_max_cols:
            state = compress_g_to_tail(state, self.G_max_cols)
        self._state_cache[layer_id] = state

    # ---------------------------------------------------------------------
    # Per-layer-kind handlers (mirror HybridzTF.tf_mlp / tf_cnn pattern)
    # ---------------------------------------------------------------------

    def _tf_input(self, L: Layer, input_bounds: Bounds) -> Fact:
        """INPUT / INPUT_SPEC: initialize FCHZState from bounds."""
        state = _bounds_to_fchz(input_bounds)
        self._store_state(L.id, state)
        out_bounds = _fchz_to_bounds(state, dev=input_bounds.lb.device,
                                              dtype=input_bounds.lb.dtype)
        return Fact(bounds=out_bounds, cons=ConSet())

    def _tf_dense(self, L: Layer, input_bounds: Bounds) -> Fact:
        """DENSE: y = Wx + b → propagate FCHZState through linear map."""
        W = _t2n(L.params.get('weight'))
        b = L.params.get('bias')
        b_n = _t2n(b).reshape(-1) if b is not None else None
        s_in = self._get_input_fchz(L, input_bounds)
        s_out = apply_dense(s_in, W, b_n)
        self._store_state(L.id, s_out)
        return Fact(bounds=_fchz_to_bounds(s_out, dev=input_bounds.lb.device,
                                                          dtype=input_bounds.lb.dtype),
                       cons=ConSet())

    def _tf_relu(self, L: Layer, input_bounds: Bounds) -> Fact:
        """RELU: DeepZ triangle relaxation with per-row tail_radius update."""
        s_in = self._get_input_fchz(L, input_bounds)
        # Compute pre-act bounds from FCHZState
        rad = np.abs(s_in.G).sum(axis=1)
        if s_in.tail_radius is not None:
            rad = rad + s_in.tail_radius
        l = s_in.c - rad
        u = s_in.c + rad

        is_active = l >= 0
        is_inactive = u <= 0
        is_unstable = ~is_active & ~is_inactive

        den = np.where(is_unstable, u - l, 1.0)
        lam = np.where(is_unstable, u / np.maximum(den, 1e-300), 0.0)
        lam = np.where(is_active, 1.0, lam)
        lam = np.where(is_inactive, 0.0, lam)
        mu = np.where(is_unstable, -lam * l / 2.0, 0.0)

        new_c = lam * s_in.c + mu
        new_G = s_in.G * lam[:, None]
        new_tail = (lam * s_in.tail_radius) if s_in.tail_radius is not None else None
        if np.any(is_unstable):
            unstable_mu = np.abs(mu)
            new_tail = (new_tail + unstable_mu) if new_tail is not None else unstable_mu

        s_out = FCHZState(c=new_c, G=new_G, n_root=s_in.n_root,
                              slack_records=s_in.slack_records,
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
        """BIAS: c += b, G unchanged, tail_radius preserved."""
        b = _t2n(L.params.get('bias')).reshape(-1)
        s_in = self._get_input_fchz(L, input_bounds)
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
        """SCALE: c *= a, G *= a, tail_radius *= |a|."""
        a = _t2n(L.params.get('scale')).reshape(-1)
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
        LayerKind.CONV2D.value:     lambda L, b, tf: tf._tf_conv2d(L, b),
        LayerKind.BN.value:         lambda L, b, tf: tf._tf_bn(L, b),
        LayerKind.MAXPOOL2D.value:  lambda L, b, tf: tf._tf_maxpool2d(L, b),
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
