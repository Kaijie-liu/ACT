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

import numpy as np
import torch
import weakref
from typing import Dict, Optional
from act.config.config import HybridZConfig
from act.back_end.core import Bounds, Fact, Layer, Net, ConSet
from act.back_end.transfer_functions import RegistryTF
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_hz import (
    HZono,
    SparseHZono,
    hz_bounds_are_liftable,
    hz_from_bounds,
    hz_lift_bounds,
    hz_tighten_bounds,
    sparse_hz_fast_bounds,
    sparse_hz_rebase_image_exact,
    sparse_hz_from_bounds,
)

import act.back_end.hybridz_tf.tf_mlp as hz_mlp
import act.back_end.hybridz_tf.tf_cnn as hz_cnn
import act.back_end.hybridz_tf.tf_rnn as hz_rnn
import act.back_end.hybridz_tf.tf_transformer as hz_transformer
import act.back_end.interval_tf.tf_mlp as interval_mlp
import act.back_end.interval_tf.tf_cnn as interval_cnn


class HybridzTF(RegistryTF):
    topological_single_pass = True

    def __init__(self, config: Optional[HybridZConfig] = None):
        super().__init__("HybridzTF")
        cfg = config or HybridZConfig()
        self._hz_cache: Dict[int, HZono] = {}
        self._sparse_hz_cache: Dict[int, SparseHZono] = {}
        self._sparse_drop_reasons: Dict[int, str] = {}
        self._sparse_precomputed_relu: Dict[int, tuple] = {}
        self._sparse_affine_expr_cache: Dict[int, object] = {}
        self._sparse_phase_output_bounds: Dict[int, Bounds] = {}
        self._sigmoid_affine_targets: Dict[int, int] = {}
        self._sigmoid_affine_inputs: Dict[int, HZono] = {}
        self._softmax_differences: Dict[int, object] = {}
        self._softmax_score_contexts: Dict[int, object] = {}
        self._cache_net_id: Optional[int] = None
        self._tanh_K: int = 2
        self._sigmoid_K: int = int(cfg.sigmoid_segments)
        if self._sigmoid_K < 1:
            raise ValueError("HybridZ sigmoid_segments must be positive")
        self._fuse_sigmoid_affine: bool = bool(cfg.fuse_sigmoid_affine)
        self._neural_hz_share_signed_relu: bool = bool(
            cfg.signed_relu_sharing and not cfg.signed_relu_compact
        )
        self._neural_hz_share_signed_compact_relu: bool = bool(
            cfg.signed_relu_sharing and cfg.signed_relu_compact
        )
        self._neural_hz_signed_cancellation: bool = bool(
            cfg.signed_relu_sharing
            and cfg.signed_relu_compact
            and cfg.signed_relu_cancellation
        )
        self._neural_hz_signed_cancellation_mixed_only: bool = bool(
            self._neural_hz_signed_cancellation
            and cfg.signed_relu_cancellation_mixed_only
        )
        self._neural_hz_signed_cancellation_pairs_only: bool = bool(
            self._neural_hz_signed_cancellation
            and cfg.signed_relu_cancellation_pairs_only
        )
        cancellation_max_cardinality = int(
            cfg.signed_relu_cancellation_elimination_max_cardinality
        )
        if cancellation_max_cardinality < 0:
            raise ValueError(
                "HybridZ cancellation elimination cardinality must be nonnegative"
            )
        self._neural_hz_signed_cancellation_elimination_max_cardinality = (
            cancellation_max_cardinality
            if self._neural_hz_signed_cancellation
            else 0
        )
        two_pair_min_outputs = int(
            cfg.signed_relu_cancellation_two_pair_min_outputs
        )
        if two_pair_min_outputs < 0:
            raise ValueError(
                "HybridZ two-pair output threshold must be nonnegative"
            )
        self._neural_hz_signed_cancellation_two_pair_min_outputs = (
            two_pair_min_outputs if self._neural_hz_signed_cancellation else 0
        )
        self._neural_hz_sparse_affine_nnz_guard: bool = bool(
            cfg.sparse_affine_nnz_guard
        )
        self._neural_hz_sparse_relu_nnz_guard: bool = bool(
            cfg.sparse_relu_nnz_guard
        )
        self._neural_hz_sparse_conv_csr_builder: bool = bool(
            cfg.sparse_conv_csr_builder
        )
        self._neural_hz_sparse_deferred_relu_materialization: bool = bool(
            cfg.sparse_deferred_relu_materialization
        )
        self._neural_hz_sparse_lazy_affine_dag: bool = bool(
            cfg.sparse_lazy_affine_dag
        )
        self._neural_hz_sparse_frontier_image_rebase: bool = bool(
            cfg.sparse_frontier_image_rebase
        )
        self._neural_hz_sparse_phase_separated_relu: bool = bool(
            cfg.sparse_phase_separated_relu
        )
        self._neural_hz_sparse_phase_selective_materialization: bool = bool(
            cfg.sparse_phase_selective_materialization
        )
        self._neural_hz_sparse_implicit_conv_dag: bool = bool(
            cfg.sparse_implicit_conv_dag
        )
        self._var_id_stride: int = 1
        setattr(self, "_HZ_MAX_INPUT_DIM", cfg.max_input_dim)
        self._sparse_next_frame_id: int = 0
        self._sparse_frame_widths: Dict[int, tuple[int, int]] = {}
        self._sparse_relu_slots: Dict[tuple[int, int, int], tuple[int, int, int]] = {}
        self._sparse_aux_slots: Dict[tuple[int, int], tuple[int, ...]] = {}
        self._neural_hz_cancellation_contexts: Dict[int, dict] = {}
        self._neural_hz_cancellation_profile: list[dict] = []
        self._neural_hz_deferred_relu_layers: int = 0
        self._neural_hz_deferred_zero_rows: int = 0
        self._neural_hz_lazy_affine_layers: int = 0
        self._neural_hz_lazy_affine_materializations: int = 0
        self._neural_hz_lazy_checkpoints: int = 0
        self._neural_hz_transient_relu_input: bool = False
        self._neural_hz_frontier_rebases: int = 0
        self._neural_hz_frontier_rebase_profile: list[dict] = []
        self._neural_hz_phase_separated_relus: int = 0
        self._neural_hz_phase_separated_profile: list[dict] = []
        self._neural_hz_phase_selective_relus: int = 0
        self._neural_hz_phase_selective_profile: list[dict] = []
        # Content interning must not keep dead operators alive. Reachability is
        # charged from live expressions/precomputed states, never from arena
        # membership.
        self._neural_hz_linear_op_arena = weakref.WeakValueDictionary()
        self._neural_hz_implicit_conv_ops: int = 0
        self._neural_hz_implicit_conv_profile: list[dict] = []
        self._neural_hz_sparse_consumer_gc: bool = bool(
            cfg.sparse_implicit_conv_dag
        )
        self._sparse_remaining_consumers: Dict[int, int] = {}
        self._sparse_pinned_layers: set[int] = set()
        self._neural_hz_released_sparse_states: int = 0

    @staticmethod
    def _net_var_id_stride(net: Net) -> int:
        max_id = -1
        for layer in net.layers:
            if layer.in_vars:
                max_id = max(max_id, max(layer.in_vars))
            if layer.out_vars:
                max_id = max(max_id, max(layer.out_vars))
        return max(max_id + 1, 1)

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
        LayerKind.ERF.value: lambda L, b, tf: hz_mlp.tf_erf(L, b, tf),
        LayerKind.SQRT.value: lambda L, b, tf: hz_mlp.tf_sqrt(L, b, tf),
        LayerKind.SIN.value: lambda L, b, tf: hz_mlp.tf_sin(L, b, tf),
        LayerKind.COS.value: lambda L, b, tf: hz_mlp.tf_cos(L, b, tf),
        LayerKind.QUANTIZE.value: lambda L, b, tf: hz_mlp.tf_quantize(L, b, tf),
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
        LayerKind.MEAN.value: lambda L, b, tf: hz_mlp.tf_mean(L, b, tf),
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

    def get_hz(self, layer_id: int) -> Optional[HZono]:
        return self._hz_cache.get(int(layer_id))

    def get_sparse_hz(self, layer_id: int) -> Optional[SparseHZono]:
        return self._sparse_hz_cache.get(int(layer_id))

    @staticmethod
    def _id_sig(ids: Optional[torch.Tensor]):
        if ids is None:
            return None
        vals = ids.detach().cpu().reshape(-1).tolist()
        return (tuple(ids.shape), tuple(int(v) for v in vals))

    @classmethod
    def _hz_sig(cls, hz: Optional[HZono]):
        if hz is None:
            return None
        eq_sig = None
        if hz.eq_mask is not None:
            eq_sig = (
                tuple(hz.eq_mask.shape),
                tuple(bool(v) for v in hz.eq_mask.detach().cpu().reshape(-1).tolist()),
            )
        return (
            tuple(hz.c.shape),
            tuple(hz.Gc.shape),
            tuple(hz.Gb.shape),
            tuple(hz.Ac.shape),
            tuple(hz.Ab.shape),
            tuple(hz.b.shape),
            eq_sig,
            cls._id_sig(hz.col_ids),
            cls._id_sig(hz.bcol_ids),
        )

    @staticmethod
    def _csr_sig(mat):
        return (tuple(mat.shape), int(mat.nnz))

    @classmethod
    def _sparse_hz_sig(cls, hz: Optional[SparseHZono]):
        if hz is None:
            return None
        return (
            int(hz.n_out),
            int(hz.n_cont),
            int(hz.n_bin),
            int(hz.n_eq),
            int(hz.n_ineq),
            cls._csr_sig(hz.Gc),
            cls._csr_sig(hz.Gb),
            cls._csr_sig(hz.Ac),
            cls._csr_sig(hz.Ab),
            cls._csr_sig(hz.Auc),
            cls._csr_sig(hz.Aub),
            hz.frame_id,
            bool(hz.exact),
        )

    def side_state_signature(self, layer_id: int):
        lid = int(layer_id)
        target = self._sigmoid_affine_targets.get(lid, lid)
        return (
            self._hz_sig(self._hz_cache.get(lid)),
            self._sparse_hz_sig(self._sparse_hz_cache.get(lid)),
            self._sparse_drop_reasons.get(lid),
            self._hz_sig(self._sigmoid_affine_inputs.get(target)),
        )

    _HZ_MAX_INPUT_DIM = 1024
    _SPARSE_MAX_AFFINE_CELLS = 64_000_000

    @staticmethod
    def _find_sigmoid_affine_targets(net: Net) -> Dict[int, int]:
        shape_only = {
            LayerKind.FLATTEN.value,
            LayerKind.RESHAPE.value,
        }
        targets: Dict[int, int] = {}
        for layer in net.layers:
            if layer.kind.upper() != LayerKind.SIGMOID.value:
                continue
            current = layer.id
            chain = [current]
            while True:
                successors = net.succs.get(current, [])
                if len(successors) != 1:
                    break
                successor = successors[0]
                if net.preds.get(successor, []) != [current]:
                    break
                kind = net.by_id[successor].kind.upper()
                if kind == LayerKind.DENSE.value:
                    targets.update((node, successor) for node in chain)
                    break
                if kind not in shape_only:
                    break
                current = successor
                chain.append(current)
        return targets

    def _col_ids_from_vars(self, bounds: Bounds, var_ids) -> Optional[torch.Tensor]:
        if not var_ids:
            return None
        n = len(var_ids)
        total = int(bounds.lb.numel())
        if n <= 0 or total % n != 0:
            return None
        base = torch.tensor(var_ids, dtype=torch.long, device=bounds.lb.device)
        B = total // n
        if B == 1:
            return base
        offsets = (
            torch.arange(B, dtype=torch.long, device=bounds.lb.device).view(-1, 1)
            * self._var_id_stride
        )
        return (offsets + base.view(1, -1)).reshape(-1)

    def _hz_from_bounds(
        self,
        bounds: Bounds,
        *,
        col_ids: Optional[torch.Tensor] = None,
    ) -> Optional[HZono]:
        lb, ub = bounds.lb.flatten(), bounds.ub.flatten()
        rad = (ub - lb) / 2.0
        ng = int((rad > 0).sum().item())
        if ng > self._HZ_MAX_INPUT_DIM:
            return None
        ids = col_ids.to(device=lb.device) if col_ids is not None else None
        return hz_from_bounds(
            bounds,
            lb.dtype,
            lb.device,
            col_ids=ids,
        )

    def _sparse_from_bounds(self, bounds: Bounds) -> SparseHZono:
        frame_id = self._sparse_next_frame_id
        self._sparse_next_frame_id += 1
        hz = sparse_hz_from_bounds(bounds, frame_id=frame_id)
        self._sparse_frame_widths[frame_id] = (hz.n_cont, hz.n_bin)
        return hz

    def _sparse_relu_slots_for(
        self,
        hz: SparseHZono,
        layer_id: int,
        neurons,
        *,
        compact: bool = False,
    ) -> Optional[tuple[list[tuple[int, int, int]], int, int]]:
        if hz.frame_id is None:
            raise ValueError("sparse ReLU requires a generator frame")
        frame_id = int(hz.frame_id)
        n_cont, n_bin = self._sparse_frame_widths.get(
            frame_id, (hz.n_cont, hz.n_bin)
        )
        n_cont = max(n_cont, hz.n_cont)
        n_bin = max(n_bin, hz.n_bin)
        missing = sum(
            (frame_id, int(layer_id), int(neuron)) not in self._sparse_relu_slots
            for neuron in neurons
        )
        new_variables = (2 if compact else 3) * missing
        if not self._neural_hz_transient_relu_input:
            if self._neural_hz_sparse_relu_nnz_guard:
                neuron_rows = [int(neuron) for neuron in neurons]
                selected_support = int(
                    hz.Gc[neuron_rows].nnz + hz.Gb[neuron_rows].nnz
                )
                estimated_entries = int(
                    self._sparse_storage_entries(hz)
                    + 8 * selected_support
                    + 32 * missing
                )
                if estimated_entries > self._SPARSE_MAX_AFFINE_CELLS:
                    return None
            elif (
                hz.n_out * (n_cont + n_bin + new_variables)
                > self._SPARSE_MAX_AFFINE_CELLS
            ):
                return None
        slots = []
        for neuron in neurons:
            key = (frame_id, int(layer_id), int(neuron))
            slot = self._sparse_relu_slots.get(key)
            if slot is None:
                slot = (
                    (n_cont, n_cont, n_bin)
                    if compact
                    else (n_cont, n_cont + 1, n_bin)
                )
                self._sparse_relu_slots[key] = slot
                n_cont += 1 if compact else 2
                n_bin += 1
            slots.append(slot)
        self._sparse_frame_widths[frame_id] = (n_cont, n_bin)
        return slots, n_cont, n_bin

    def _sparse_cont_slots_for(
        self,
        hz: SparseHZono,
        layer_id: int,
        count: int,
    ) -> Optional[tuple[tuple[int, ...], int]]:
        if hz.frame_id is None:
            raise ValueError("sparse auxiliary generators require a frame")
        frame_id = int(hz.frame_id)
        key = (frame_id, int(layer_id))
        slots = self._sparse_aux_slots.get(key)
        n_cont, n_bin = self._sparse_frame_widths.get(
            frame_id, (hz.n_cont, hz.n_bin)
        )
        n_cont = max(n_cont, hz.n_cont)
        n_bin = max(n_bin, hz.n_bin)
        if slots is None:
            count = int(count)
            if count * (n_cont + n_bin + count) > self._SPARSE_MAX_AFFINE_CELLS:
                return None
            slots = tuple(range(n_cont, n_cont + count))
            self._sparse_aux_slots[key] = slots
            n_cont += count
            self._sparse_frame_widths[frame_id] = (n_cont, n_bin)
        elif len(slots) != int(count):
            raise ValueError(
                f"sparse auxiliary slot count changed for layer {layer_id}: "
                f"{len(slots)} vs {count}"
            )
        return slots, n_cont

    @staticmethod
    def _sparse_fact(fact: Fact, hz: SparseHZono) -> Fact:
        hb = sparse_hz_fast_bounds(hz)
        return Fact(
            bounds=hz_tighten_bounds(fact.bounds, hb),
            cons=fact.cons,
        )

    def _seed_sparse_cache(self, L: Layer, input_bounds: Bounds) -> None:
        k = L.kind.upper()
        try:
            if k in ("INPUT", "INPUT_SPEC"):
                self._sparse_hz_cache[L.id] = self._sparse_from_bounds(input_bounds)
                self._sparse_drop_reasons.pop(L.id, None)
            elif k != "ASSERT":
                preds = self._net.preds.get(L.id, [])
                if preds and preds[0] in self._sparse_hz_cache:
                    self._sparse_hz_cache[L.id] = self._sparse_hz_cache[preds[0]]
                    self._sparse_drop_reasons.pop(L.id, None)
                elif not preds:
                    self._sparse_hz_cache[L.id] = self._sparse_from_bounds(input_bounds)
                    self._sparse_drop_reasons.pop(L.id, None)
        except Exception as exc:
            self._drop_sparse_hz(L.id, f"sparse_seed_failed:{type(exc).__name__}")

    def _drop_sparse_hz(self, layer_id: int, reason: str) -> None:
        lid = int(layer_id)
        self._sparse_hz_cache.pop(lid, None)
        self._sparse_phase_output_bounds.pop(lid, None)
        self._sparse_drop_reasons[lid] = reason

    def release_intermediate_hz(self) -> tuple[int, int]:
        """Release propagated HZ states after final/input states are retained.

        Callers must first take ordinary Python references to every HZ state
        needed by final solving or witness reconstruction. Clearing these
        caches does not mutate those retained objects; it only shortens the
        lifetime of intermediate DAG states before MILP lowering.
        """
        dense_states = len(self._hz_cache)
        sparse_states = len(self._sparse_hz_cache)
        self._hz_cache.clear()
        self._sparse_hz_cache.clear()
        self._sparse_drop_reasons.clear()
        self._sparse_precomputed_relu.clear()
        self._sparse_affine_expr_cache.clear()
        self._sparse_phase_output_bounds.clear()
        self._sigmoid_affine_targets.clear()
        self._sigmoid_affine_inputs.clear()
        self._softmax_differences.clear()
        self._softmax_score_contexts.clear()
        self._sparse_frame_widths.clear()
        self._sparse_relu_slots.clear()
        self._sparse_aux_slots.clear()
        self._neural_hz_cancellation_contexts.clear()
        self._neural_hz_linear_op_arena.clear()
        self._sparse_remaining_consumers.clear()
        self._sparse_pinned_layers.clear()
        self._cache_net_id = None
        self._sparse_next_frame_id = 0
        return dense_states, sparse_states

    def _release_consumed_sparse_predecessors(self, layer: Layer) -> None:
        """Drop a sparse cache entry only after its final graph consumer.

        INPUT/INPUT_SPEC states remain pinned for concrete-witness decoding,
        and the predecessor of ASSERT remains pinned for terminal solving.
        Expressions and deferred states retain ordinary Python references to
        every source/operator they still need, so removing an obsolete cache
        key cannot mutate the represented set.
        """
        if not self._neural_hz_sparse_consumer_gc:
            return
        for raw_pred in self._net.preds.get(layer.id, []):
            pred = int(raw_pred)
            remaining = self._sparse_remaining_consumers.get(pred)
            if remaining is None or remaining <= 0:
                raise ValueError(
                    f"invalid sparse consumer accounting for layer {pred}"
                )
            remaining -= 1
            self._sparse_remaining_consumers[pred] = remaining
            if remaining or pred in self._sparse_pinned_layers:
                continue
            released = int(pred in self._sparse_hz_cache) + int(
                pred in self._sparse_affine_expr_cache
            )
            self._sparse_hz_cache.pop(pred, None)
            self._sparse_affine_expr_cache.pop(pred, None)
            self._sparse_phase_output_bounds.pop(pred, None)
            self._neural_hz_released_sparse_states += released

    def _sparse_exceeds_limit(self, hz: SparseHZono, out_dim: int) -> bool:
        gen = int(hz.n_cont + hz.n_bin)
        return gen > 0 and int(out_dim) * gen > self._SPARSE_MAX_AFFINE_CELLS

    @staticmethod
    def _sparse_storage_entries(hz: SparseHZono) -> int:
        return int(
            hz.c.size
            + hz.b.size
            + hz.ub.size
            + hz.Gc.nnz
            + hz.Gb.nnz
            + hz.Ac.nnz
            + hz.Ab.nnz
            + hz.Auc.nnz
            + hz.Aub.nnz
        )

    _SPARSE_RESIDUAL_AFFINE_KINDS = frozenset(
        {
            "AVGPOOL2D",
            "BIAS",
            "BN",
            "CONV2D",
            "CONVTRANSPOSE2D",
            "DENSE",
            "FLATTEN",
            "RESHAPE",
            "SCALE",
            "SQUEEZE",
            "TRANSPOSE",
            "UNSQUEEZE",
        }
    )

    def _sparse_affine_source_terms(
        self,
        layer_id: int,
        current_relu_id: int,
        *,
        depth: int = 0,
        visiting: Optional[frozenset[int]] = None,
    ) -> int:
        """Count exact lazy sources behind one prospective residual input.

        Future nodes have no cache yet, so walk backwards through a purely
        affine chain until reaching the current ReLU or an already materialized
        exact HZ/lazy expression.  Zero means that the count is not structurally
        decidable and therefore disables the optimization.
        """
        lid = int(layer_id)
        if lid == int(current_relu_id):
            return 1
        if depth >= 16:
            return 0
        seen = frozenset() if visiting is None else visiting
        if lid in seen:
            return 0
        expression = self._sparse_affine_expr_cache.get(lid)
        if expression is not None:
            return int(len(expression.terms))
        hz = self._sparse_hz_cache.get(lid)
        if hz is not None and hz.exact and hz.frame_id is not None:
            return 1
        layer = self._net.by_id.get(lid)
        if layer is None:
            return 0
        kind = layer.kind.upper()
        predecessors = [int(value) for value in self._net.preds.get(lid, [])]
        next_seen = seen | {lid}
        if kind == "ADD":
            counts = [
                self._sparse_affine_source_terms(
                    pred,
                    current_relu_id,
                    depth=depth + 1,
                    visiting=next_seen,
                )
                for pred in predecessors
            ]
            return sum(counts) if counts and all(counts) else 0
        if kind not in self._SPARSE_RESIDUAL_AFFINE_KINDS or len(predecessors) != 1:
            return 0
        return self._sparse_affine_source_terms(
            predecessors[0],
            current_relu_id,
            depth=depth + 1,
            visiting=next_seen,
        )

    def _relu_next_residual_terms(self, layer_id: int) -> int:
        """Return the exact source count at the nearest downstream ADD.

        Only affine paths are traversed.  Multiple equally near residual
        targets, an unknown source, or a graph cycle fail closed with zero.
        """
        current = int(layer_id)
        frontier = [(int(value), 1) for value in self._net.succs.get(current, [])]
        seen_depth: Dict[int, int] = {}
        nearest_depth: Optional[int] = None
        nearest_adds = set()
        while frontier:
            lid, depth = frontier.pop(0)
            if depth > 16 or (
                nearest_depth is not None and depth > nearest_depth
            ):
                continue
            prior_depth = seen_depth.get(lid)
            if prior_depth is not None and prior_depth <= depth:
                continue
            seen_depth[lid] = depth
            layer = self._net.by_id.get(lid)
            if layer is None:
                continue
            kind = layer.kind.upper()
            if kind == "ADD":
                nearest_depth = depth
                nearest_adds.add(lid)
                continue
            if kind not in self._SPARSE_RESIDUAL_AFFINE_KINDS:
                continue
            frontier.extend(
                (int(value), depth + 1)
                for value in self._net.succs.get(lid, [])
            )
        if len(nearest_adds) != 1:
            return 0
        residual_id = nearest_adds.pop()
        counts = [
            self._sparse_affine_source_terms(pred, current)
            for pred in self._net.preds.get(residual_id, [])
        ]
        return sum(counts) if len(counts) >= 2 and all(counts) else 0

    def _maybe_rebase_sparse_frontier(
        self,
        L: Layer,
        hz: SparseHZono,
        bounds: Bounds,
    ) -> SparseHZono:
        """Install an exact sparse image interface before repeated residual use."""
        if (
            not self._neural_hz_sparse_frontier_image_rebase
            or L.kind.upper() != "RELU"
            or not hz.exact
            or hz.frame_id is None
        ):
            return hz

        lower = bounds.lb.detach().cpu().double().numpy().reshape(-1)
        upper = bounds.ub.detach().cpu().double().numpy().reshape(-1)
        if lower.size != hz.n_out or not np.isfinite(lower).all() or not np.isfinite(upper).all():
            return hz
        interface_width = int(np.count_nonzero(upper != lower))
        value_nnz = int(hz.Gc.nnz + hz.Gb.nnz)
        latent_width = int(hz.n_cont + hz.n_bin)
        before_storage = self._sparse_storage_entries(hz)
        residual_terms = self._relu_next_residual_terms(L.id)
        storage_pressure = (
            3 * before_storage
            >= 2 * min(self._SPARSE_MAX_AFFINE_CELLS, 64_000_000)
        )

        # A structural residual-source count is diagnostic, not sufficient to
        # justify rebasing: real traces show that an early dense link can cost
        # more than the next residual reuse saves.  Install the interface only
        # under the fixed, representation-wide storage-pressure gate.
        if (
            interface_width == 0
            or interface_width > latent_width
            or hz.n_out > latent_width
            or value_nnz < 1_000_000
            or value_nnz <= 4 * interface_width
            or not storage_pressure
        ):
            return hz

        try:
            candidate = sparse_hz_rebase_image_exact(hz, bounds)
        except (MemoryError, ValueError):
            return hz
        after_storage = self._sparse_storage_entries(candidate)
        if (
            after_storage > self._SPARSE_MAX_AFFINE_CELLS
            or candidate.n_bin != hz.n_bin
            or candidate.Gc.nnz >= value_nnz
        ):
            return hz

        frame_id = int(hz.frame_id)
        self._sparse_frame_widths[frame_id] = (
            candidate.n_cont,
            candidate.n_bin,
        )
        self._sparse_relu_slots = {
            key: value
            for key, value in self._sparse_relu_slots.items()
            if int(key[0]) != frame_id
        }
        self._sparse_aux_slots = {
            key: value
            for key, value in self._sparse_aux_slots.items()
            if int(key[0]) != frame_id
        }
        self._neural_hz_cancellation_contexts.clear()
        self._neural_hz_frontier_rebases += 1
        self._neural_hz_frontier_rebase_profile.append(
            {
                "layer_id": int(L.id),
                "frame_id": frame_id,
                "trigger": "storage_pressure",
                "residual_terms": int(residual_terms),
                "interface_width": interface_width,
                "n_cont_before": int(hz.n_cont),
                "n_cont_after": int(candidate.n_cont),
                "n_bin": int(candidate.n_bin),
                "value_nnz_before": value_nnz,
                "value_nnz_after": int(candidate.Gc.nnz + candidate.Gb.nnz),
                "storage_before": before_storage,
                "storage_after": after_storage,
            }
        )
        return candidate

    def _propagate_sparse_hz(self, L: Layer, input_bounds: Bounds, result: Fact) -> Fact:
        k = L.kind.upper()
        if k == "ASSERT" and any(
            int(pred) in self._sparse_affine_expr_cache
            for pred in self._net.preds.get(L.id, [])
        ):
            self._drop_sparse_hz(L.id, "lazy_affine_reached_terminal")
            return result
        if k in ("INPUT", "INPUT_SPEC", "ASSERT"):
            hz = self._sparse_hz_cache.get(L.id)
            return self._sparse_fact(result, hz) if hz is not None else result
        precomputed = self._sparse_precomputed_relu.pop(int(L.id), None)
        if precomputed is not None:
            phase_bounds = None
            if len(precomputed) == 3:
                out, expected_lb, expected_ub = precomputed
                out_expression = None
            elif len(precomputed) == 4:
                out, expected_lb, expected_ub, out_expression = precomputed
            else:
                (
                    out,
                    expected_lb,
                    expected_ub,
                    out_expression,
                    phase_bounds,
                ) = precomputed
            actual_lb = input_bounds.lb.detach().cpu()
            actual_ub = input_bounds.ub.detach().cpu()
            if not (
                torch.equal(actual_lb, expected_lb)
                and torch.equal(actual_ub, expected_ub)
            ):
                self._drop_sparse_hz(
                    L.id, "deferred_relu_interval_bounds_mismatch"
                )
                return result
            if out_expression is not None:
                if (
                    not out.exact
                    or out.frame_id != out_expression.frame_id
                    or out.n_out != out_expression.n_out
                ):
                    self._drop_sparse_hz(
                        L.id, "invalid_phase_separated_relu"
                    )
                    return result
                self._sparse_hz_cache.pop(L.id, None)
                self._sparse_affine_expr_cache[L.id] = out_expression
                self._sparse_drop_reasons[L.id] = "lazy_affine_expr"
                if phase_bounds is not None:
                    return Fact(
                        bounds=hz_tighten_bounds(result.bounds, phase_bounds),
                        cons=result.cons,
                    )
                return self._sparse_fact(result, out)
            out = self._maybe_rebase_sparse_frontier(L, out, result.bounds)
            if (
                self._sparse_storage_entries(out)
                > self._SPARSE_MAX_AFFINE_CELLS
            ):
                self._drop_sparse_hz(L.id, "sparse_storage_limit:RELU")
                return result
            self._sparse_hz_cache[L.id] = out
            self._sparse_drop_reasons.pop(L.id, None)
            return self._sparse_fact(result, out)
        expression = self._sparse_affine_expr_cache.get(int(L.id))
        predecessors = [int(value) for value in self._net.preds.get(L.id, [])]
        add_has_expression = k == "ADD" and any(
            pred in self._sparse_affine_expr_cache for pred in predecessors
        )
        if expression is not None or add_has_expression:
            handled, out, out_expression, reason = (
                hz_cnn.sparse_hz_apply_affine_expr_layer(
                    L,
                    expression,
                    input_bounds,
                    result,
                    self,
                )
            )
            if handled:
                self._sparse_hz_cache.pop(L.id, None)
                if out_expression is not None:
                    phase_bounds = self._sparse_phase_output_bounds.pop(
                        int(L.id), None
                    )
                    self._sparse_affine_expr_cache[L.id] = out_expression
                    self._sparse_drop_reasons[L.id] = "lazy_affine_expr"
                    if phase_bounds is not None:
                        return Fact(
                            bounds=hz_tighten_bounds(
                                result.bounds, phase_bounds
                            ),
                            cons=result.cons,
                        )
                    return (
                        self._sparse_fact(result, out)
                        if out is not None
                        else result
                    )
                if out is not None:
                    out = self._maybe_rebase_sparse_frontier(
                        L, out, result.bounds
                    )
                    self._sparse_affine_expr_cache.pop(L.id, None)
                    self._sparse_hz_cache[L.id] = out
                    self._sparse_drop_reasons.pop(L.id, None)
                    return self._sparse_fact(result, out)
                self._sparse_affine_expr_cache.pop(L.id, None)
                self._drop_sparse_hz(
                    L.id, reason or f"unsupported_lazy_affine_op:{k}"
                )
                return result
        hz = self._sparse_hz_cache.get(L.id)
        if hz is None:
            return result
        sparsity_priced_affine = bool(
            self._neural_hz_sparse_affine_nnz_guard
            and k in (
                "CONV2D",
                "CONVTRANSPOSE2D",
                "AVGPOOL2D",
                "SCALE",
                "BIAS",
                "BN",
                "ADD",
                "SUB",
            )
        )
        sparsity_priced_relu = bool(
            self._neural_hz_sparse_relu_nnz_guard and k == "RELU"
        )
        if (
            self._sparse_exceeds_limit(hz, result.bounds.lb.numel())
            and not (sparsity_priced_affine or sparsity_priced_relu)
        ):
            self._drop_sparse_hz(L.id, f"sparse_size_limit:{k}")
            return result
        try:
            for apply_sparse in (
                hz_mlp.sparse_hz_apply_layer,
                hz_cnn.sparse_hz_apply_layer,
                hz_transformer.sparse_hz_apply_layer,
            ):
                handled, out, drop_reason = apply_sparse(L, hz, input_bounds, result, self)
                if not handled:
                    continue
                if out is None:
                    self._drop_sparse_hz(L.id, drop_reason or f"unsupported_sparse_op:{k}")
                    return result
                if (
                    (sparsity_priced_affine or sparsity_priced_relu)
                    and self._sparse_storage_entries(out)
                    > self._SPARSE_MAX_AFFINE_CELLS
                ):
                    self._drop_sparse_hz(L.id, f"sparse_storage_limit:{k}")
                    return result
                out = self._maybe_rebase_sparse_frontier(
                    L, out, result.bounds
                )
                self._sparse_hz_cache[L.id] = out
                self._sparse_drop_reasons.pop(L.id, None)
                return self._sparse_fact(result, out)
            self._drop_sparse_hz(L.id, f"unsupported_sparse_op:{k}")
        except Exception as exc:
            self._drop_sparse_hz(L.id, f"sparse_op_failed:{k}:{type(exc).__name__}")
        return result

    def apply(
        self,
        L: Layer,
        input_bounds: Bounds,
        net: Net,
        before: Dict[int, Fact],
        after: Dict[int, Fact],
    ) -> Fact:
        k = self._check_supported(L.kind)

        net_id = id(net)
        if self._cache_net_id != net_id:
            self._hz_cache.clear()
            self._sparse_hz_cache.clear()
            self._sparse_drop_reasons.clear()
            self._sparse_precomputed_relu.clear()
            self._sparse_affine_expr_cache.clear()
            self._sparse_phase_output_bounds.clear()
            self._sigmoid_affine_inputs.clear()
            self._softmax_differences.clear()
            self._softmax_score_contexts.clear()
            self._sparse_frame_widths.clear()
            self._sparse_relu_slots.clear()
            self._sparse_aux_slots.clear()
            self._neural_hz_cancellation_contexts.clear()
            self._neural_hz_frontier_rebases = 0
            self._neural_hz_frontier_rebase_profile = []
            self._neural_hz_phase_separated_relus = 0
            self._neural_hz_phase_separated_profile = []
            self._neural_hz_phase_selective_relus = 0
            self._neural_hz_phase_selective_profile = []
            self._neural_hz_linear_op_arena.clear()
            self._neural_hz_implicit_conv_ops = 0
            self._neural_hz_implicit_conv_profile = []
            self._neural_hz_released_sparse_states = 0
            self._sparse_remaining_consumers = {
                int(layer.id): 0 for layer in net.layers
            }
            for consumer in net.layers:
                for predecessor in net.preds.get(consumer.id, []):
                    pred = int(predecessor)
                    self._sparse_remaining_consumers[pred] = (
                        self._sparse_remaining_consumers.get(pred, 0) + 1
                    )
            self._sparse_pinned_layers = {
                int(layer.id)
                for layer in net.layers
                if layer.kind.upper() in ("INPUT", "INPUT_SPEC")
                or any(
                    net.by_id[int(successor)].kind.upper() == "ASSERT"
                    for successor in net.succs.get(layer.id, [])
                )
            }
            self._cache_net_id = net_id
            self._var_id_stride = self._net_var_id_stride(net)
            self._sparse_next_frame_id = 0
            self._sigmoid_affine_targets = (
                self._find_sigmoid_affine_targets(net)
                if self._fuse_sigmoid_affine
                else {}
            )

        self._set_context(net, before, after)
        self._seed_sparse_cache(L, input_bounds)
        if k not in ("INPUT", "INPUT_SPEC", "ASSERT"):
            predecessors = net.preds.get(L.id, [])
            if (
                predecessors
                and int(predecessors[0]) in self._sparse_affine_expr_cache
            ):
                self._sparse_affine_expr_cache[L.id] = (
                    self._sparse_affine_expr_cache[int(predecessors[0])]
                )

        if k in ("INPUT", "INPUT_SPEC"):
            hz_init = self._hz_from_bounds(
                input_bounds,
                col_ids=self._col_ids_from_vars(input_bounds, L.out_vars),
            )
            if hz_init is not None:
                self._hz_cache[L.id] = hz_init
        elif k != "ASSERT":
            preds = net.preds.get(L.id, [])
            if preds and preds[0] in self._hz_cache:
                self._hz_cache[L.id] = self._hz_cache[preds[0]]
            elif not preds:
                hz_init = self._hz_from_bounds(
                    input_bounds,
                    col_ids=self._col_ids_from_vars(input_bounds, L.in_vars),
                )
                if hz_init is not None:
                    self._hz_cache[L.id] = hz_init

        n_out = len(L.out_vars)
        hz_carried = self._hz_cache.get(L.id)
        ngnb = (
            hz_carried.Gc.shape[1] + hz_carried.Gb.shape[1]
            if hz_carried is not None
            else 0
        )
        if max(n_out, ngnb) > self._HZ_MAX_INPUT_DIM and k not in (
            "INPUT",
            "INPUT_SPEC",
            "ASSERT",
        ):
            self._hz_cache.pop(L.id, None)

        hz_before = self._hz_cache.get(L.id)
        result = self._LAYER_REGISTRY[k](L, input_bounds, self)
        result = self._propagate_sparse_hz(L, input_bounds, result)

        if (
            hz_before is not None
            and self._hz_cache.get(L.id) is hz_before
            and k not in ("INPUT", "INPUT_SPEC")
        ):
            if (
                hz_bounds_are_liftable(result.bounds)
                and result.bounds.lb.numel() <= self._HZ_MAX_INPUT_DIM
            ):
                self._hz_cache[L.id] = hz_lift_bounds(hz_before, result.bounds)
            else:
                self._hz_cache.pop(L.id, None)

        self._release_consumed_sparse_predecessors(L)
        return result
