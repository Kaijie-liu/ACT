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
from act.back_end.solver.solver_hz import HZono, hz_from_bounds, hz_compute_bounds

import act.back_end.hybridz_tf.tf_mlp as hz_mlp
import act.back_end.hybridz_tf.tf_cnn as hz_cnn
import act.back_end.hybridz_tf.tf_rnn as hz_rnn
import act.back_end.hybridz_tf.tf_transformer as hz_transformer
import act.back_end.interval_tf.tf_mlp as interval_mlp
import act.back_end.interval_tf.tf_cnn as interval_cnn


class HybridzTF(TransferFunction):
    def __init__(self):
        self._hz_cache: Dict[int, HZono] = {}
        self._cache_net_id: Optional[int] = None
        self._tanh_K: int = 1
        self._sigmoid_K: int = 1
        # Canonical input factor ids + box, for share-merging residual roots.
        self._input_ids: Optional[torch.Tensor] = None
        self._input_box: Optional[tuple] = None

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
        LayerKind.DIV.value: lambda L, b, tf: interval_mlp.tf_div(
            L,
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
        ),
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
        LayerKind.GATHER.value: lambda L, b, tf: interval_mlp.tf_gather(L, b),
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
    # many cells (~ fp32 bytes/4). 100M cells ~= 400MB, comfortably keeps narrow
    # nets + malbeware's 16M-cell affine HZ + any point query (ng=0 -> 0 cells),
    # while still dropping a genuinely exploding wide-perturbation conv stack.
    _hz_cell_budget = 100_000_000
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
        LayerKind.CONSTANT.value, LayerKind.UPSAMPLE.value,
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
        from act.back_end.solver.solver_hz import _fresh_col_ids
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
        return HZono(
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
            self._cache_net_id = net_id
            self._input_ids = None
            self._input_box = None
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
                    self._input_ids = hz_init.col_ids
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
