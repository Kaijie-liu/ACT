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

import torch
from typing import Dict, Optional
from act.back_end.core import Bounds, Fact, Layer, Net, ConSet
from act.back_end.transfer_functions import TransferFunction
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_hz import HZono, SparseHZono, hz_from_bounds

import act.back_end.hybridz_tf.tf_mlp as hz_mlp
import act.back_end.hybridz_tf.tf_cnn as hz_cnn
import act.back_end.hybridz_tf.tf_rnn as hz_rnn
import act.back_end.hybridz_tf.tf_transformer as hz_transformer
import act.back_end.interval_tf.tf_mlp as interval_mlp
import act.back_end.interval_tf.tf_cnn as interval_cnn

class HybridzTF(TransferFunction):
    def __init__(self):
        self._hz_cache: Dict[int, HZono] = {}
        self._sparse_hz_cache: Dict[int, SparseHZono] = {}
        self._sparse_drop_reasons: Dict[int, str] = {}
        self._cache_net_id: Optional[int] = None
        self._tanh_K: int = 2
        self._sigmoid_K: int = 2
        self._scurve_domain_cuts: bool = False
        self._scurve_graph_cuts: bool = False
        self._enable_sparse_hz: bool = False
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

    @staticmethod
    def _id_sig(ids):
        if ids is None:
            return None
        if isinstance(ids, torch.Tensor):
            flat = ids.detach().reshape(-1)
            return (
                int(flat.numel()),
                None if flat.numel() == 0 else int(flat[0].item()),
                None if flat.numel() == 0 else int(flat[-1].item()),
            )
        flat = ids.reshape(-1)
        return (
            int(flat.size),
            None if flat.size == 0 else int(flat[0]),
            None if flat.size == 0 else int(flat[-1]),
        )

    @classmethod
    def _hz_sig(cls, hz: Optional[HZono]):
        if hz is None:
            return None
        eq_sig = None
        if hz.eq_mask is not None:
            eq_sig = (tuple(hz.eq_mask.shape), int(hz.eq_mask.to(torch.int64).sum().item()))
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
        if mat is None:
            return None
        return (tuple(mat.shape), int(mat.nnz))

    @classmethod
    def _sparse_hz_sig(cls, hz: Optional[SparseHZono]):
        if hz is None:
            return None
        return (
            id(hz),
            tuple(hz.c.shape),
            cls._csr_sig(hz.Gc),
            cls._csr_sig(hz.Gb),
            cls._csr_sig(hz.Ac),
            cls._csr_sig(hz.Ab),
            tuple(hz.b.shape),
            cls._csr_sig(hz.Auc),
            cls._csr_sig(hz.Aub),
            None if hz.ub is None else tuple(hz.ub.shape),
            cls._id_sig(hz.col_ids),
            cls._id_sig(hz.bcol_ids),
        )

    def side_state_signature(self, layer_id: int):
        lid = int(layer_id)
        return (
            self._hz_sig(self._hz_cache.get(lid)),
            self._sparse_hz_sig(self._sparse_hz_cache.get(lid)),
            self._sparse_drop_reasons.get(lid),
        )

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
    _HZ_MAX_AFFINE_DIM = 8192
    _hz_cell_budget = 200_000_000
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
        nz = rad > 0
        ng = int(nz.sum().item())
        if ng > getattr(self, "_effective_max_dim", self._HZ_MAX_INPUT_DIM):
            return None
        from act.back_end.solver.solver_hz import hz_fresh_col_ids, hz_mark_known_nonempty
        ids = None
        if reuse_ids is not None and reuse_ids.numel() == n:
            ids = reuse_ids.to(device=lb.device)
        elif track_ids:
            ids = hz_fresh_col_ids(n, device=lb.device)
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
            hz.full_col_ids = ids
        if bool(torch.all(lb <= ub).item()):
            hz_mark_known_nonempty(hz, "input_box")
        return hz

    def _sparse_from_bounds(self, bounds: Bounds) -> SparseHZono:
        reuse = self._input_ids if self._input_box_matches(bounds) else None
        ids = None if reuse is None else reuse.detach().cpu().numpy()
        return hz_mlp.sparse_hz_from_bounds(bounds, col_ids=ids)

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
                if self._input_ids is None:
                    full_ids = getattr(hz, "full_col_ids", None)
                    if full_ids is not None:
                        self._input_ids = torch.as_tensor(
                            full_ids,
                            dtype=torch.long,
                            device=input_bounds.lb.device,
                        )
                        self._input_box = (
                            input_bounds.lb.flatten().clone(),
                            input_bounds.ub.flatten().clone(),
                        )
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

    def _propagate_sparse_hz(self, L: Layer, input_bounds: Bounds, result: Fact) -> None:
        if not self._enable_sparse_hz:
            return
        k = L.kind.upper()
        if k in ("INPUT", "INPUT_SPEC", "ASSERT"):
            return
        hz = self._sparse_hz_cache.get(L.id)
        if hz is None:
            return

        try:
            for apply_sparse in (
                hz_mlp.sparse_hz_apply_layer,
                hz_cnn.sparse_hz_apply_layer,
            ):
                handled, out, drop_reason = apply_sparse(L, hz, input_bounds, result, self)
                if not handled:
                    continue
                if drop_reason is not None:
                    self._drop_sparse_hz(L.id, drop_reason)
                    return
                self._sparse_hz_cache[L.id] = out
                self._sparse_drop_reasons.pop(L.id, None)
                return
            self._drop_sparse_hz(L.id, f"unsupported_sparse_op:{k}")
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
            self._effective_max_dim = (
                self._HZ_MAX_AFFINE_DIM if self._net_is_pure_affine(net)
                else self._HZ_MAX_INPUT_DIM)

        self._net = net
        self._before = before
        self._after = after

        if k in ("INPUT", "INPUT_SPEC"):
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
                reuse = self._input_ids if self._input_box_matches(input_bounds) else None
                hz_init = self._hz_from_bounds(
                    input_bounds, track_ids=(reuse is None), reuse_ids=reuse)
                if hz_init is not None:
                    self._hz_cache[L.id] = hz_init

        self._seed_sparse_cache(L, input_bounds)

        n_out = len(L.out_vars)
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

        if (hz_before is not None and self._hz_cache.get(L.id) is hz_before
                and k not in ("INPUT", "INPUT_SPEC")):
            self._hz_cache[L.id] = hz_from_bounds(
                result.bounds, result.bounds.lb.dtype, result.bounds.lb.device
            )

        return result
