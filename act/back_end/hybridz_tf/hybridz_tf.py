# ===- act/back_end/hybridz_tf/hybridz_tf.py - HybridZ Transfer Function -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ Transfer Function Implementation.
#
#   Each hz_tf_* is a complete TF for one layer kind, combining
#   HZ zonotope propagation with interval_tf constraint generation.
#   hz_tf_* live in tf_mlp.py / tf_cnn.py alongside their layer types.
#   HZ domain ops co-locate with the hz_tf_* that use them.
#
# ===---------------------------------------------------------------------===#

""" """

import importlib
import torch
from typing import Dict, Optional
from act.back_end.core import Bounds, Fact, Layer, Net, ConSet
from act.back_end.transfer_functions import TransferFunction
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_hz import HZono, hz_from_bounds, hz_compute_bounds
from act.back_end.solver.sparse_hz import SparseHZono

import act.back_end.hybridz_tf.tf_mlp as hz_mlp
import act.back_end.hybridz_tf.tf_cnn as hz_cnn
import act.back_end.hybridz_tf.tf_rnn as hz_rnn
import act.back_end.hybridz_tf.tf_transformer as hz_transformer
import act.back_end.interval_tf.tf_mlp as interval_mlp
import act.back_end.interval_tf.tf_cnn as interval_cnn


_SPARSE_OPS_MODULE = None


def _sparse_ops_module():
    global _SPARSE_OPS_MODULE
    if _SPARSE_OPS_MODULE is None:
        _SPARSE_OPS_MODULE = importlib.import_module("act.back_end.hybridz_tf.sparse_ops")
    return _SPARSE_OPS_MODULE


class HybridzTF(TransferFunction):
    def __init__(self):
        self._hz_cache: Dict[int, HZono] = {}
        self._sparse_hz_cache: Dict[int, SparseHZono] = {}
        self._sparse_drop_reasons: Dict[int, str] = {}
        self._cache_net_id: Optional[int] = None
        self._tanh_K: int = 1
        self._sigmoid_K: int = 2
        self._scurve_domain_cuts: bool = False
        self._scurve_graph_cuts: bool = False
        self._enable_sparse_hz: bool = False
        # Canonical input factor ids + box, for share-merging residual roots.
        self._input_ids: Optional[torch.Tensor] = None
        self._input_box: Optional[tuple] = None
        self._sparse_input_hz: Optional[SparseHZono] = None

    def _input_box_matches(self, bounds: Bounds) -> bool:
        """True iff ``bounds`` equals the remembered network-input box, i.e. a
        floating layer is reading the same input (so it may reuse input ids)."""
        if self._input_box is None:
            return False
        lb, ub = bounds.lb.flatten(), bounds.ub.flatten()
        ilb, iub = self._input_box
        if lb.shape != ilb.shape:
            return False
        return bool(torch.allclose(lb, ilb.to(lb.device))
                    and torch.allclose(ub, iub.to(ub.device)))

    _LAYER_REGISTRY = {
        # Identity / spec
        LayerKind.INPUT.value: lambda L, b, tf: Fact(bounds=b, cons=ConSet()),
        LayerKind.INPUT_SPEC.value: lambda L, b, tf: Fact(bounds=b, cons=ConSet()),
        LayerKind.ASSERT.value: lambda L, b, tf: Fact(bounds=b, cons=ConSet()),
        # MLP: HZ + interval
        LayerKind.DENSE.value: lambda L, b, tf: hz_mlp.tf_dense(L, b, tf),
        LayerKind.BIAS.value: lambda L, b, tf: hz_mlp.tf_bias(L, b, tf),
        LayerKind.SCALE.value: lambda L, b, tf: hz_mlp.tf_scale(L, b, tf),
        LayerKind.RELU.value: lambda L, b, tf: hz_mlp.tf_relu(L, b, tf),
        LayerKind.LRELU.value: lambda L, b, tf: hz_mlp.tf_lrelu(L, b, tf),
        LayerKind.TANH.value: lambda L, b, tf: hz_mlp.tf_tanh(L, b, tf),
        LayerKind.SIGMOID.value: lambda L, b, tf: hz_mlp.tf_sigmoid(L, b, tf),
        LayerKind.ABS.value: lambda L, b, tf: hz_mlp.tf_abs(L, b, tf),
        LayerKind.BN.value: lambda L, b, tf: hz_mlp.tf_bn(L, b, tf),
        # Multi-input: HZ + interval
        LayerKind.ADD.value: lambda L, b, tf: hz_mlp.tf_add(L, b, tf),
        LayerKind.MUL.value: lambda L, b, tf: hz_mlp.tf_mul(L, b, tf),
        LayerKind.SUB.value: lambda L, b, tf: hz_mlp.tf_sub(L, b, tf),
        LayerKind.DIV.value: lambda L, b, tf: hz_mlp.tf_div(L, b, tf),
        LayerKind.CONCAT.value: lambda L, b, tf: hz_mlp.tf_concat(L, b, tf),
        # CNN: HZ + interval
        LayerKind.CONV2D.value: lambda L, b, tf: hz_cnn.tf_conv2d(L, b, tf),
        LayerKind.MAXPOOL2D.value: lambda L, b, tf: hz_cnn.tf_maxpool2d(L, b, tf),
        # Activations: interval-only
        LayerKind.CLIP.value: lambda L, b, tf: interval_mlp.tf_clip(L, b),
        LayerKind.SOFTPLUS.value: lambda L, b, tf: interval_mlp.tf_softplus(L, b),
        LayerKind.SILU.value: lambda L, b, tf: interval_mlp.tf_silu(L, b),
        LayerKind.RELU6.value: lambda L, b, tf: interval_mlp.tf_relu6(L, b),
        LayerKind.HARDTANH.value: lambda L, b, tf: interval_mlp.tf_hardtanh(L, b),
        LayerKind.HARDSIGMOID.value: lambda L, b, tf: interval_mlp.tf_hardsigmoid(L, b),
        LayerKind.HARDSWISH.value: lambda L, b, tf: interval_mlp.tf_hardswish(L, b),
        LayerKind.MISH.value: lambda L, b, tf: interval_mlp.tf_mish(L, b),
        LayerKind.SOFTSIGN.value: lambda L, b, tf: interval_mlp.tf_softsign(L, b),
        LayerKind.SQUARE.value: lambda L, b, tf: interval_mlp.tf_square(L, b),
        LayerKind.POWER.value: lambda L, b, tf: interval_mlp.tf_power(L, b),
        LayerKind.SIGN.value: lambda L, b, tf: hz_mlp.tf_sign(L, b, tf),
        LayerKind.REDUCE_SUM.value: lambda L, b, tf: hz_mlp.tf_reduce_sum(L, b, tf),
        LayerKind.CONSTANT.value: lambda L, b, tf: hz_mlp.tf_constant(L, b, tf),
        LayerKind.COMPARE.value: lambda L, b, tf: hz_mlp.tf_compare(L, b, tf),
        LayerKind.WHERE.value: lambda L, b, tf: hz_mlp.tf_where(L, b, tf),
        LayerKind.MATMUL.value: lambda L, b, tf: hz_mlp.tf_matmul(L, b, tf),
        LayerKind.ARG_EXTREMUM.value: lambda L, b, tf: hz_mlp.tf_arg_extremum(L, b, tf),
        LayerKind.UPSAMPLE.value: lambda L, b, tf: hz_mlp.tf_upsample(L, b, tf),
        LayerKind.SCATTER_ND.value: lambda L, b, tf: hz_mlp.tf_scatter_nd(L, b, tf),
        LayerKind.MAX.value: lambda L, b, tf: interval_mlp.tf_max(
            L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)
        ),
        LayerKind.MIN.value: lambda L, b, tf: interval_mlp.tf_min(
            L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)
        ),
        # CNN: interval-only
        LayerKind.AVGPOOL1D.value: lambda L, b, tf: interval_cnn.tf_avgpool1d(L, b),
        LayerKind.AVGPOOL2D.value: lambda L, b, tf: hz_cnn.tf_avgpool2d(L, b, tf),
        LayerKind.MAXPOOL3D.value: lambda L, b, tf: interval_cnn.tf_maxpool3d(L, b),
        LayerKind.PAD.value:      lambda L, b, tf: interval_cnn.tf_pad(L, b),
        LayerKind.CONV1D.value: lambda L, b, tf: interval_cnn.tf_conv1d(L, b),
        LayerKind.CONV3D.value: lambda L, b, tf: interval_cnn.tf_conv3d(L, b),
        LayerKind.CONVTRANSPOSE2D.value: lambda L, b, tf: hz_cnn.tf_convtranspose2d(L, b, tf),
        LayerKind.FLATTEN.value: lambda L, b, tf: hz_mlp.tf_flatten(L, b, tf),
        # Shape ops: structural pass-through (HZ row order = flattened layout)
        LayerKind.RESHAPE.value: lambda L, b, tf: hz_mlp.tf_reshape(L, b, tf),
        LayerKind.TRANSPOSE.value: lambda L, b, tf: hz_mlp.tf_transpose(L, b, tf),
        LayerKind.SQUEEZE.value: lambda L, b, tf: hz_mlp.tf_squeeze(L, b, tf),
        LayerKind.UNSQUEEZE.value: lambda L, b, tf: hz_mlp.tf_unsqueeze(L, b, tf),
        LayerKind.EXPAND.value: lambda L, b, tf: hz_mlp.tf_expand(L, b, tf),
        LayerKind.SLICE.value: lambda L, b, tf: hz_mlp.tf_slice(L, b, tf),
        LayerKind.GATHER.value: lambda L, b, tf: hz_mlp.tf_gather(L, b, tf),
        # RNN
        LayerKind.LSTM.value: lambda L, b, tf: hz_rnn.tf_lstm(L, b, tf),
        LayerKind.GRU.value: lambda L, b, tf: hz_rnn.tf_gru(L, b, tf),
        LayerKind.RNN.value: lambda L, b, tf: hz_rnn.tf_rnn(L, b, tf),
        LayerKind.EMBEDDING.value: lambda L, b, tf: hz_rnn.tf_embedding(L, b, tf),
        LayerKind.EMBEDDING_TF.value: lambda L, b, tf: hz_rnn.tf_embedding(L, b, tf),
        # Transformer
        LayerKind.POSENC.value: lambda L, b, tf: hz_transformer.tf_posenc(L, b, tf),
        LayerKind.LAYERNORM.value: lambda L, b, tf: hz_transformer.tf_layernorm(L, b, tf),
        LayerKind.GELU.value: lambda L, b, tf: hz_transformer.tf_gelu(L, b, tf),
        LayerKind.ATT_SCORES.value: lambda L, b, tf: hz_transformer.tf_att_scores(L, b, tf),
        LayerKind.SOFTMAX.value: lambda L, b, tf: hz_transformer.tf_softmax(L, b, tf),
        LayerKind.ATT_MIX.value: lambda L, b, tf: hz_transformer.tf_att_mix(L, b, tf),
        LayerKind.MHA_SPLIT.value: lambda L, b, tf: hz_transformer.tf_mha_split(L, b, tf),
        LayerKind.MHA_JOIN.value: lambda L, b, tf: hz_transformer.tf_mha_join(L, b, tf),
        LayerKind.MASK_ADD.value: lambda L, b, tf: hz_transformer.tf_mask_add(L, b, tf),
    }

    @property
    def name(self) -> str:
        return "HybridzTF"

    def supports_layer(self, layer_kind: str) -> bool:
        return layer_kind.upper() in self._LAYER_REGISTRY

    def get_hz(self, layer_id: int) -> Optional[HZono]:
        """Return the carried HZ for ``layer_id`` if propagation kept one.

        This is the public read-only boundary used by HybridZ verdict code.
        Callers must treat ``None`` as a representation drop and return UNKNOWN
        rather than silently using interval bounds as a proof.
        """
        return self._hz_cache.get(int(layer_id))

    def enable_sparse_hz(self, enabled: bool = True) -> None:
        """Enable the CSR exact-HZ side cache.

        Dense HZ propagation remains the authority for layer bounds.  The sparse
        cache is a verifier/productization path for large exact-HZ verdicts:
        unsupported sparse ops drop only the sparse cache, not the existing
        dense result.
        """
        self._enable_sparse_hz = bool(enabled)
        if not self._enable_sparse_hz:
            self._sparse_hz_cache.clear()
            self._sparse_drop_reasons.clear()
            self._sparse_input_hz = None

    def get_sparse_hz(self, layer_id: int) -> Optional[SparseHZono]:
        """Return the carried CSR HZ for ``layer_id`` if sparse propagation kept one."""
        return self._sparse_hz_cache.get(int(layer_id))

    def sparse_drop_reason(self, layer_id: int) -> Optional[str]:
        return self._sparse_drop_reasons.get(int(layer_id))

    _HZ_MAX_INPUT_DIM = 1024
    # A net whose every layer is a strictly-linear, generator-preserving HZ op
    # (no ReLU/sigmoid/etc., no column growth) can carry its EXACT input
    # zonotope at much higher dim safely: ng stays == input_dim, no binaries,
    # cost is one matmul per layer. The cap then only needs to bound the
    # one-time diagonal seed memory (n*n*dtype). 8192 => <=537MB fp64 / 268MB
    # fp32. Nets with ANY nonlinearity keep the protective _HZ_MAX_INPUT_DIM
    # (the cap exists to bound ReLU generator blow-up on wide feature maps).
    _HZ_MAX_AFFINE_DIM = 8192
    # Memory-based drop budget: keep the layer's HZ while n_out*(ng+nb) <= this
    # many cells. 100M was too conservative -- it dropped malbeware iid64 (a provable
    # HZ that builds in 6.5s at only 2.5GB peak) to UNKNOWN (gate-found regression
    # 2026-06-18). 200M recovers iid64 while keeping wide-conv (cifar/tiny) safe:
    # cifar100/0 peaks 7.9GB at 200M (vs 15GB at 500M), well under the 20GB worker
    # RLIMIT_AS hard guard. Monotone vs 100M (>=, so no NEW drops -- only recoveries).
    _hz_cell_budget = 200_000_000
    # Strictly-linear, generator-count-preserving kinds with a real HZ handler.
    # Deliberately EXCLUDES ADD/SUB/CONCAT (multi-input, can grow columns) and
    # any interval-only op (would drop the HZ anyway) -- conservative on purpose.
    _HZ_AFFINE_KINDS = frozenset({
        LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value, LayerKind.ASSERT.value,
        LayerKind.DENSE.value, LayerKind.BIAS.value, LayerKind.SCALE.value,
        LayerKind.BN.value, LayerKind.CONV2D.value, LayerKind.CONVTRANSPOSE2D.value,
        LayerKind.AVGPOOL2D.value, LayerKind.FLATTEN.value, LayerKind.RESHAPE.value,
        LayerKind.TRANSPOSE.value, LayerKind.SQUEEZE.value, LayerKind.UNSQUEEZE.value,
        LayerKind.EXPAND.value, LayerKind.SLICE.value, LayerKind.REDUCE_SUM.value,
        LayerKind.CONSTANT.value, LayerKind.GATHER.value, LayerKind.UPSAMPLE.value,
    })

    def _net_is_pure_affine(self, net: Net) -> bool:
        """True iff every layer kind is a strictly-linear generator-preserving
        HZ op -> the exact zonotope can be carried at high dim with no blow-up."""
        return all(L.kind.upper() in self._HZ_AFFINE_KINDS for L in net.layers)

    def _hz_from_bounds(self, bounds: Bounds, *, track_ids: bool = False,
                        reuse_ids: Optional[torch.Tensor] = None) -> Optional[HZono]:
        lb, ub = bounds.lb.flatten(), bounds.ub.flatten()
        n = lb.shape[0]
        rad = (ub - lb) / 2.0
        # Only ACTUALLY-perturbed dims get a generator column. A zero-radius dim
        # contributes an all-zero generator (redundant). So a POINT query (all
        # rad==0, e.g. metaroom) gets ng=0 -> propagates EXACTLY and costs nothing
        # even at huge n_out. The cap is then on the GENERATOR count (the real
        # memory driver, n_out*ng), NOT the raw input dimension.
        nz = rad > 0
        ng = int(nz.sum().item())
        if ng > getattr(self, "_effective_max_dim", self._HZ_MAX_INPUT_DIM):
            return None
        from act.back_end.solver.solver_hz import _fresh_col_ids, hz_mark_known_nonempty
        ids = None
        if reuse_ids is not None and reuse_ids.numel() == n:
            ids = reuse_ids.to(device=lb.device)
        elif track_ids:
            ids = _fresh_col_ids(n, device=lb.device)
        c = ((lb + ub) / 2.0).view(-1, 1)
        idx = torch.where(nz)[0]
        Gc = lb.new_zeros(n, ng)
        if ng > 0:
            Gc[idx, torch.arange(ng, device=lb.device)] = rad[idx]
        hz = HZono(
            c=c,
            Gc=Gc,
            Gb=lb.new_zeros(n, 0),
            Ac=lb.new_zeros(0, ng),
            Ab=lb.new_zeros(0, 0),
            b=lb.new_zeros(0, 1),
            col_ids=(ids[idx] if ids is not None else None),
            bcol_ids=(torch.zeros(0, dtype=torch.long, device=lb.device)
                      if ids is not None else None),
        )
        if ids is not None:
            # Full input-dimension ids are needed for witness replay. ``col_ids``
            # only stores nonzero-radius generator columns, but replay must
            # reconstruct all input dimensions, including constant ones.
            hz.full_col_ids = ids
        if bool(torch.all(lb <= ub).item()):
            hz_mark_known_nonempty(hz, "input_box")
        return hz

    def _sparse_from_bounds(self, bounds: Bounds) -> SparseHZono:
        return _sparse_ops_module().sparse_hz_from_bounds(bounds)

    def _sparse_box_matches(self, bounds: Bounds) -> bool:
        return self._input_box_matches(bounds)

    def _seed_sparse_cache(self, L: Layer, input_bounds: Bounds) -> None:
        if not self._enable_sparse_hz:
            return
        try:
            if L.kind.upper() in ("INPUT", "INPUT_SPEC"):
                hz = self._sparse_from_bounds(input_bounds)
                self._sparse_hz_cache[L.id] = hz
                if self._sparse_input_hz is None:
                    self._sparse_input_hz = hz
                self._sparse_drop_reasons.pop(L.id, None)
                return

            preds = self._net.preds.get(L.id, [])
            if preds and preds[0] in self._sparse_hz_cache:
                self._sparse_hz_cache[L.id] = self._sparse_hz_cache[preds[0]]
                self._sparse_drop_reasons.pop(L.id, None)
            elif not preds:
                if self._sparse_box_matches(input_bounds) and self._sparse_input_hz is not None:
                    self._sparse_hz_cache[L.id] = self._sparse_input_hz
                else:
                    self._sparse_hz_cache[L.id] = self._sparse_from_bounds(input_bounds)
                self._sparse_drop_reasons.pop(L.id, None)
        except Exception as exc:
            self._drop_sparse_hz(L.id, f"sparse_seed_failed:{type(exc).__name__}")

    def _drop_sparse_hz(self, layer_id: int, reason: str) -> None:
        lid = int(layer_id)
        self._sparse_hz_cache.pop(lid, None)
        self._sparse_drop_reasons[lid] = reason

    @staticmethod
    def _copy_sparse_input_metadata(out: SparseHZono, src: Optional[SparseHZono]) -> None:
        """Carry replay metadata through sparse exact-HZ operators.

        The metadata is not part of the HZ set semantics; it only maps a solver
        witness back to the original input box for audit.  Missing metadata
        simply disables ADV promotion and leaves the verdict UNKNOWN.
        """
        if src is None:
            return
        for name in ("input_center", "input_radius", "input_indices", "input_shape"):
            if getattr(out, name, None) is not None:
                continue
            val = getattr(src, name, None)
            if val is None:
                continue
            if hasattr(val, "copy"):
                val = val.copy()
            setattr(out, name, val)

    @staticmethod
    def _first_sparse_with_input_metadata(parts):
        for part in parts:
            if part is not None and getattr(part, "input_center", None) is not None:
                return part
        return None

    @staticmethod
    def _rows_to_numpy(rows: torch.Tensor):
        return rows.detach().cpu().numpy()

    def _propagate_sparse_hz(self, L: Layer, input_bounds: Bounds, result: Fact) -> None:
        if not self._enable_sparse_hz:
            return
        k = L.kind.upper()
        if k in ("INPUT", "INPUT_SPEC", "ASSERT"):
            return
        hz = self._sparse_hz_cache.get(L.id)
        if hz is None:
            return

        metadata_source = hz
        try:
            sparse_ops = _sparse_ops_module()
            if k == LayerKind.DENSE.value:
                out = sparse_ops.sparse_hz_apply_dense_layer(hz, L)
            elif k == LayerKind.BIAS.value:
                out = sparse_ops.sparse_hz_add_const(hz, L.params["c"])
            elif k == LayerKind.SCALE.value:
                out = sparse_ops.sparse_hz_scale(hz, L.params["a"])
            elif k == LayerKind.BN.value:
                out = sparse_ops.sparse_hz_add_const(
                    sparse_ops.sparse_hz_scale(hz, L.params["A"]),
                    L.params["c"],
                )
            elif k == LayerKind.CONV2D.value:
                if L.params.get("input_shape") is None:
                    self._drop_sparse_hz(L.id, "missing_conv2d_input_shape")
                    return
                out = sparse_ops.sparse_hz_apply_conv2d_layer(hz, L)
            elif k == LayerKind.CONVTRANSPOSE2D.value:
                if L.params.get("input_shape") is None:
                    self._drop_sparse_hz(L.id, "missing_convtranspose2d_input_shape")
                    return
                out = sparse_ops.sparse_hz_apply_convtranspose2d_layer(hz, L)
            elif k == LayerKind.AVGPOOL2D.value:
                if L.params.get("input_shape") is None:
                    self._drop_sparse_hz(L.id, "missing_avgpool2d_input_shape")
                    return
                out = sparse_ops.sparse_hz_apply_avgpool2d_layer(hz, L)
            elif k == LayerKind.MAXPOOL2D.value:
                if L.params.get("input_shape") is None:
                    self._drop_sparse_hz(L.id, "missing_maxpool2d_input_shape")
                    return
                out = sparse_ops.sparse_hz_apply_maxpool2d_layer(
                    hz,
                    L,
                    input_bounds,
                    compressed_relu=getattr(self, "_relu_compressed", False),
                    valid_cuts=getattr(self, "_relu_valid_cuts", False),
                )
            elif k == LayerKind.RELU.value:
                out = sparse_ops.sparse_hz_apply_relu_exact(
                    hz,
                    pre_bounds=input_bounds,
                    compressed=getattr(self, "_relu_compressed", False),
                    valid_cuts=getattr(self, "_relu_valid_cuts", False),
                )
            elif k == LayerKind.SIGMOID.value:
                out = sparse_ops.sparse_hz_apply_sigmoid_piecewise(
                    hz,
                    K=getattr(self, "_sigmoid_K", 2),
                    domain_cuts=getattr(self, "_scurve_domain_cuts", False),
                    graph_cuts=getattr(self, "_scurve_graph_cuts", False),
                )
            elif k == LayerKind.TANH.value:
                out = sparse_ops.sparse_hz_apply_tanh_piecewise(
                    hz,
                    K=getattr(self, "_tanh_K", 1),
                    domain_cuts=getattr(self, "_scurve_domain_cuts", False),
                    graph_cuts=getattr(self, "_scurve_graph_cuts", False),
                )
            elif k == LayerKind.MATMUL.value:
                preds = self._net.preds.get(L.id, [])
                if len(preds) != 2:
                    self._drop_sparse_hz(L.id, "unsupported_sparse_matmul_routing")
                    return
                left = self._sparse_hz_cache.get(preds[0])
                right = self._sparse_hz_cache.get(preds[1])
                if left is None or right is None:
                    self._drop_sparse_hz(L.id, "missing_sparse_matmul_input")
                    return
                if sparse_ops.sparse_hz_is_point(right):
                    metadata_source = left
                    out = sparse_ops.sparse_hz_apply_matmul_const_layer(
                        left, right, L, variable_is_left=True,
                    )
                elif sparse_ops.sparse_hz_is_point(left):
                    metadata_source = right
                    out = sparse_ops.sparse_hz_apply_matmul_const_layer(
                        right, left, L, variable_is_left=False,
                    )
                else:
                    self._drop_sparse_hz(L.id, "unsupported_sparse_matmul_var_var")
                    return
            elif k in {
                LayerKind.FLATTEN.value,
                LayerKind.RESHAPE.value,
                LayerKind.SQUEEZE.value,
                LayerKind.UNSQUEEZE.value,
                LayerKind.TRANSPOSE.value,
            }:
                out = hz
            elif k == LayerKind.SLICE.value:
                rows = hz_mlp._slice_row_idx(L, hz.n_out)
                if rows is None or rows.numel() != result.bounds.lb.numel():
                    self._drop_sparse_hz(L.id, "unsupported_slice_row_map")
                    return
                out = sparse_ops.sparse_hz_gather_rows(hz, self._rows_to_numpy(rows))
            elif k == LayerKind.GATHER.value:
                rows = hz_mlp._gather_row_idx(L, hz.n_out)
                if rows is None or rows.numel() != result.bounds.lb.numel():
                    self._drop_sparse_hz(L.id, "unsupported_gather_row_map")
                    return
                out = sparse_ops.sparse_hz_gather_rows(hz, self._rows_to_numpy(rows))
            elif k == LayerKind.EXPAND.value:
                rows = hz_mlp._expand_row_idx(L, hz.n_out)
                if rows is None or rows.numel() != result.bounds.lb.numel():
                    self._drop_sparse_hz(L.id, "unsupported_expand_row_map")
                    return
                out = sparse_ops.sparse_hz_gather_rows(hz, self._rows_to_numpy(rows))
            elif k == LayerKind.UPSAMPLE.value:
                rows = hz_mlp._upsample_nearest_row_idx(
                    L, hz.n_out, result.bounds.lb.numel(), input_bounds.lb.device)
                if rows is None:
                    self._drop_sparse_hz(L.id, "unsupported_upsample_row_map")
                    return
                out = sparse_ops.sparse_hz_gather_rows(hz, self._rows_to_numpy(rows))
            elif k == LayerKind.ADD.value:
                preds = self._net.preds.get(L.id, [])
                parts = [self._sparse_hz_cache.get(pid) for pid in preds[:2]]
                if len(parts) != 2 or any(part is None for part in parts):
                    self._drop_sparse_hz(L.id, "missing_sparse_add_input")
                    return
                metadata_source = self._first_sparse_with_input_metadata(parts)
                out = sparse_ops.sparse_hz_add_same_frame(parts[0], parts[1])
            elif k == LayerKind.SUB.value:
                preds = self._net.preds.get(L.id, [])
                parts = [self._sparse_hz_cache.get(pid) for pid in preds[:2]]
                if len(parts) != 2 or any(part is None for part in parts):
                    self._drop_sparse_hz(L.id, "missing_sparse_sub_input")
                    return
                metadata_source = self._first_sparse_with_input_metadata(parts)
                out = sparse_ops.sparse_hz_sub_same_frame(parts[0], parts[1])
            elif k == LayerKind.CONCAT.value:
                preds = self._net.preds.get(L.id, [])
                parts = [self._sparse_hz_cache.get(pid) for pid in preds]
                if not parts or any(part is None for part in parts):
                    self._drop_sparse_hz(L.id, "missing_sparse_concat_input")
                    return
                metadata_source = self._first_sparse_with_input_metadata(parts)
                out = sparse_ops.sparse_hz_concat(parts)
            elif k == LayerKind.CONSTANT.value:
                out = self._sparse_from_bounds(result.bounds)
            else:
                self._drop_sparse_hz(L.id, f"unsupported_sparse_op:{k}")
                return

            self._copy_sparse_input_metadata(out, metadata_source)
            self._sparse_hz_cache[L.id] = out
            self._sparse_drop_reasons.pop(L.id, None)
        except Exception as exc:
            self._drop_sparse_hz(L.id, f"sparse_op_failed:{k}:{type(exc).__name__}")

    def apply(
        self,
        L: Layer,
        input_bounds: Bounds,
        net: Net,
        before: Dict[int, Fact],
        after: Dict[int, Fact],
    ) -> Fact:
        k = L.kind.upper()
        if k not in self._LAYER_REGISTRY:
            raise NotImplementedError(f"HybridzTF: Unsupported layer kind '{k}'")

        net_id = id(net)
        if self._cache_net_id != net_id:
            self._hz_cache.clear()
            self._sparse_hz_cache.clear()
            self._sparse_drop_reasons.clear()
            self._cache_net_id = net_id
            self._input_ids = None
            self._input_box = None
            self._sparse_input_hz = None
            # Pure-affine nets carry the exact zonotope at high dim (no ReLU
            # blow-up); everything else keeps the protective 1024 cap.
            self._effective_max_dim = (
                self._HZ_MAX_AFFINE_DIM if self._net_is_pure_affine(net)
                else self._HZ_MAX_INPUT_DIM)

        self._net = net
        self._before = before
        self._after = after

        if k in ("INPUT", "INPUT_SPEC"):
            # Seed with FRESH factor ids so residual branches can later be
            # share-merged (hz_sgm_add). Remember the input HZ + box so a
            # floating root reading the same input can reuse these ids.
            reuse = self._input_ids if self._input_box_matches(input_bounds) else None
            hz_init = self._hz_from_bounds(
                input_bounds, track_ids=(reuse is None), reuse_ids=reuse)
            if hz_init is not None:
                self._hz_cache[L.id] = hz_init
                if self._input_ids is None:
                    self._input_ids = getattr(hz_init, "full_col_ids", hz_init.col_ids)
                    self._input_box = (
                        input_bounds.lb.flatten().clone(),
                        input_bounds.ub.flatten().clone(),
                    )
        elif k != "ASSERT":
            preds = net.preds.get(L.id, [])
            if preds and preds[0] in self._hz_cache:
                self._hz_cache[L.id] = self._hz_cache[preds[0]]
            elif not preds:
                # Floating root: a non-INPUT layer with no predecessors reads
                # the network input directly (e.g. the entry DENSE of a residual
                # block). Without seeding, its whole branch silently degrades to
                # interval and any downstream ADD collapses the zonotope to a box
                # (observed on cersyve residual nets). Seed from the layer's own
                # input bounds so the branch carries an HZ. If its box matches
                # the network input, REUSE the input factor ids so the residual
                # ADD share-merges correctly (sound: same input => same factor);
                # otherwise fresh ids (treated as independent — still sound).
                reuse = self._input_ids if self._input_box_matches(input_bounds) else None
                hz_init = self._hz_from_bounds(
                    input_bounds, track_ids=(reuse is None), reuse_ids=reuse)
                if hz_init is not None:
                    self._hz_cache[L.id] = hz_init

        self._seed_sparse_cache(L, input_bounds)

        n_out = len(L.out_vars)
        # Memory-based drop: the output HZ's Gc is ~ n_out * (ng+nb) cells; that
        # product (not the raw n_out) is what risks OOM. A point query carries
        # ng=nb=0, so n_out*0 = 0 -> never dropped, even through a 57k-dim conv
        # (metaroom). A genuinely wide perturbation (large ng) still drops to the
        # sound interval fallback. Budget defaults to cap^2 cells.
        hz_carried = self._hz_cache.get(L.id)
        ngnb = (hz_carried.Gc.shape[1] + hz_carried.Gb.shape[1]) if hz_carried is not None else 0
        cell_budget = getattr(self, "_hz_cell_budget",
                              self._HZ_MAX_INPUT_DIM * self._HZ_MAX_INPUT_DIM)
        if ngnb > 0 and n_out * ngnb > cell_budget and k not in (
            "INPUT",
            "INPUT_SPEC",
            "ASSERT",
        ):
            self._hz_cache.pop(L.id, None)

        hz_before = self._hz_cache.get(L.id)
        result = self._LAYER_REGISTRY[k](L, input_bounds, self)
        self._propagate_sparse_hz(L, input_bounds, result)

        # A handler that computed bounds but did NOT claim the cache leaves the
        # predecessor's (now stale) HZ in place; rebuild a sound box HZ from the
        # real result bounds. EXCEPT INPUT/INPUT_SPEC: their handlers never claim
        # the cache, but the seed above already installed the canonical input HZ
        # WITH col_ids (bounds-identical to result.bounds) -- reseeding here would
        # strip those ids and silently defeat residual share-merge (hz_sgm_add ->
        # minkowski) on every skip connection rooted at the input.
        if (hz_before is not None and self._hz_cache.get(L.id) is hz_before
                and k not in ("INPUT", "INPUT_SPEC")):
            self._hz_cache[L.id] = hz_from_bounds(
                result.bounds, result.bounds.lb.dtype, result.bounds.lb.device
            )

        return result


def _test_sparse_matmul_const_propagation() -> None:  # pragma: no cover
    import numpy as np
    import tempfile

    from act.back_end.analyze import analyze
    from act.back_end.net_factory import LAYER_TESTING_NAME_PREFIX, LAYER_TESTING_SPECS, NetFactory
    from act.back_end.solver.solver_hz_verdict import hz_row_max
    from act.back_end.transfer_functions import set_transfer_function
    from act.back_end.verifier import add_all_input_specs, get_input_ids, seed_from_input_specs

    with tempfile.TemporaryDirectory() as td:
        factory = NetFactory(output_dir=td, write_manifest=False)
        spec = LAYER_TESTING_SPECS[f"{LAYER_TESTING_NAME_PREFIX}matmul"]()
        net = factory.create_network("hybridz_sparse_matmul_const", spec)

    tf = HybridzTF()
    tf.enable_sparse_hz(True)
    set_transfer_function(tf)

    input_layer = next(L for L in net.layers if L.kind == LayerKind.INPUT.value)
    spec_layers = [L for L in net.layers if L.kind == LayerKind.INPUT_SPEC.value]
    seed = seed_from_input_specs(spec_layers)
    entry_fact = Fact(bounds=seed, cons=ConSet())
    add_all_input_specs(entry_fact.cons, get_input_ids(net), spec_layers)
    analyze(net, input_layer.id, entry_fact)

    matmul_layer = next(L for L in net.layers if L.kind == LayerKind.MATMUL.value)
    dense = tf._hz_cache.get(matmul_layer.id)
    sparse = tf.get_sparse_hz(matmul_layer.id)
    if dense is None or sparse is None:
        raise AssertionError(
            f"MATMUL sparse propagation dropped: reason={tf.sparse_drop_reason(matmul_layer.id)!r}"
        )
    rows = (
        np.array([1.0, 0.0, -0.5, 0.25], dtype=np.float64),
        np.array([0.0, -1.0, 0.5, 1.0], dtype=np.float64),
    )
    for row in rows:
        lhs = float(hz_row_max(dense, row))
        rhs = float(hz_row_max(sparse, row))
        if abs(lhs - rhs) > 1e-9:
            raise AssertionError(f"sparse MATMUL support mismatch: dense={lhs}, sparse={rhs}")
