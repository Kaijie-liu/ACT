#===- act/back_end/analyze.py - Network Analysis Functions --------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Network analysis functions for ACT verification framework.
#   Provides analysis capabilities for neural network structures and properties.
#
#===---------------------------------------------------------------------===#

import torch
import os
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, cast
from act.back_end.core import Bounds, Fact, Net, ConSet
from act.back_end.layer_schema import LayerKind
from act.back_end.utils import box_join, changed_or_maskdiff, update_cache
from act.back_end.transfer_functions import dispatch_tf, set_transfer_function_mode

# Initialize default transfer function mode
def initialize_tf_mode(mode: str = "interval"):
    """Initialize transfer function mode. Call this before using analyze()."""
    set_transfer_function_mode(mode)


@dataclass
class AnalyzeCache:
    """Reusable dataflow state for monotone BaB input-box refinements."""
    before: Dict[int, Fact]
    after: Dict[int, Fact]
    globalC: ConSet


@torch.no_grad()
def analyze(
    net: Net,
    entry_id: int,
    entry_fact: Fact,
    eps: float = 1e-9,
    *,
    cache: Optional[AnalyzeCache] = None,
) -> Tuple[Dict[int, Fact], Dict[int, Fact], ConSet]:
    """
    Perform abstract interpretation on the network starting from entry_fact.
    Args:
        net: ACT network structure
        entry_id: ID of the entry (INPUT) layer
        entry_fact: Initial Fact containing bounds and constraints for the input
        eps: Convergence epsilon for fixpoint iteration
        cache: Optional state from a prior monotone-refinement analysis.

    Returns:
        Tuple of (before, after, globalC) containing propagated facts and global constraints
    """
    from act.back_end.transfer_functions import ensure_active_tf
    ensure_active_tf("interval")

    if cache is None:
        before: Dict[int, Fact] = {}
        after:  Dict[int, Fact] = {}
        globalC = ConSet()

        # Default bounds must carry the leading batch dim. Consumer TFs assume
        # bounds are [B, n] and reshape on that axis; a (n,) default would
        # silently broadcast in some paths and fail loudly in others.
        seed = entry_fact.bounds.lb
        B = seed.shape[0] if seed.dim() >= 2 else 1
        for layer in net.layers:
            n = len(layer.out_vars)
            hi = seed.new_full((B, n), float("inf"))
            lo = seed.new_full((B, n), -float("inf"))
            before[layer.id] = Fact(bounds=Bounds(lo.clone(), hi.clone()), cons=ConSet())
            after[layer.id] = Fact(bounds=Bounds(lo.clone(), hi.clone()), cons=ConSet())
            layer.cache.clear()

        # Seed entry with provided Fact (includes all input constraints)
        before[entry_id] = entry_fact

        # Seed every other zero-indegree source (e.g. CONSTANT layers emitted by
        # torch2act for ONNX initializers). Without this, source-layer bounds stay
        # at +/-inf forever because the worklist starts at entry_id and CONSTANTs
        # have no predecessor that would ever push them on. (Oracle finding #5.)
        seeds = [entry_id]
        for layer in net.layers:
            if layer.id == entry_id or net.preds.get(layer.id):
                continue
            if layer.kind == LayerKind.CONSTANT.value:
                B_size = entry_fact.bounds.lb.shape[0]
                raw_value = cast(object, layer.params["value"])
                if not isinstance(raw_value, torch.Tensor):
                    raise TypeError(
                        f"CONSTANT layer {layer.id} requires tensor param 'value', got {type(raw_value).__name__}."
                    )
                val = raw_value.reshape(-1).to(
                    device=entry_fact.bounds.lb.device,
                    dtype=entry_fact.bounds.lb.dtype,
                )
                val_b = val.unsqueeze(0).expand(B_size, -1).contiguous()  # [B, numel]
                before[layer.id] = Fact(bounds=Bounds(val_b.clone(), val_b.clone()), cons=ConSet())
            elif layer.kind == LayerKind.LUT_BOUNDS.value:
                # Zero-indegree LUT envelope layer (cctsdb_yolo_2023 dynamic
                # Slice). The bounds were precomputed at conversion and are
                # sealed into ``params['lb']`` and ``params['ub']``. Seed
                # `before` with the same (batched) interval so any later
                # ``Bjoin`` over a LUT predecessor reads finite bounds
                # rather than the +/-inf sentinel.
                B_size = entry_fact.bounds.lb.shape[0]
                lb_raw = layer.params.get("lb")
                ub_raw = layer.params.get("ub")
                if not isinstance(lb_raw, torch.Tensor) or not isinstance(ub_raw, torch.Tensor):
                    raise TypeError(
                        f"LUT_BOUNDS layer {layer.id} requires tensor params 'lb' and 'ub'"
                    )
                lb = lb_raw.reshape(-1).to(
                    device=entry_fact.bounds.lb.device,
                    dtype=entry_fact.bounds.lb.dtype,
                )
                ub = ub_raw.reshape(-1).to(
                    device=entry_fact.bounds.lb.device,
                    dtype=entry_fact.bounds.lb.dtype,
                )
                lb_b = lb.unsqueeze(0).expand(B_size, -1).contiguous()
                ub_b = ub.unsqueeze(0).expand(B_size, -1).contiguous()
                before[layer.id] = Fact(bounds=Bounds(lb_b.clone(), ub_b.clone()), cons=ConSet())
            # Other zero-indegree kinds (none today) would be seeded similarly.
            seeds.append(layer.id)
    else:
        before = cache.before
        after = cache.after
        globalC = cache.globalC
        before[entry_id] = entry_fact
        seeds = [entry_id]

    # R16 (advisor 2026-05-25): track which layers have been visited at
    # least once so we can defer multi-pred nodes until all their preds
    # are concretized. Otherwise a multi-input op (e.g. ViT's CONCAT)
    # may be popped from the worklist while one branch is still at the
    # default (-inf, +inf) sentinel; the box_join propagates ±inf into
    # an interior DENSE / matmul where ``±inf * 0_weight`` produces NaN,
    # poisoning the entire downstream chain irrecoverably.
    visited: set = {entry_id}
    # CONSTANT seeds are concretized at initialization (before is set
    # to the constant value above), so they count as visited too. Any
    # other zero-indegree kinds that got into ``seeds`` have ``before``
    # at the +/-inf default, but no consumer needs them to be ready
    # before its own first visit — they will be re-enqueued via the
    # changed_or_maskdiff path on subsequent passes.
    for layer in net.layers:
        if not net.preds.get(layer.id) and layer.kind in (
            LayerKind.CONSTANT.value, LayerKind.LUT_BOUNDS.value,
        ):
            visited.add(layer.id)

    WL = deque(seeds)
    deferred_counts: Dict[int, int] = {}
    DEFER_RETRY_LIMIT = max(64, 4 * len(net.layers))
    while WL:
        lid = WL.popleft(); layer = net.by_id[lid]

        # Ready-check: if any predecessor has not yet been visited, defer
        # this layer until its preds are concretized. The worklist is
        # FIFO; re-appending sends it to the back so other ready nodes
        # progress first. Bounded retry guards against pathological
        # cycles (the DAG check at conversion should prevent these, but
        # cheap insurance never hurts).
        preds_list = net.preds.get(lid, [])
        if preds_list and any(p not in visited for p in preds_list):
            tries = deferred_counts.get(lid, 0) + 1
            deferred_counts[lid] = tries
            if tries <= DEFER_RETRY_LIMIT:
                WL.append(lid)
                continue
            # Last-resort: give up deferring and run with whatever preds
            # have so far. Exceeding the bound implies a cycle or a
            # disconnected component the wrapper missed; we surface it
            # via the resulting +/-inf bounds rather than hanging.

        # merge predecessors into before[lid]
        if preds_list:
            # Initialize from first predecessor (not infinite bounds)
            first_bounds = after[preds_list[0]].bounds
            Bjoin = Bounds(lb=first_bounds.lb.clone(), ub=first_bounds.ub.clone())
            Cjoin = ConSet()
            for con in after[preds_list[0]].cons: Cjoin.replace(con)
            # Join with remaining predecessors when shapes match (DAG merge points).
            # Multi-input ops with heterogeneous predecessor shapes (MATMUL, CONCAT,
            # SCATTER_ND, etc.) ignore Bin and pull each predecessor explicitly via
            # get_predecessor_bounds; the join is meaningless for them so we skip
            # rather than crash.
            for pid in preds_list[1:]:
                pb = after[pid].bounds
                if pb.lb.shape == Bjoin.lb.shape:
                    Bjoin = box_join(Bjoin, pb)
                for con in after[pid].cons: Cjoin.replace(con)
            before[lid] = Fact(Bjoin, Cjoin)

        out_fact = dispatch_tf(layer, before, after, net)
        if os.environ.get("ACT_HZ_LAYER_PROGRESS", "0") == "1":
            try:
                b = out_fact.bounds
                print(
                    f"[ANALYZE-PROGRESS] L{layer.id} {layer.kind} "
                    f"lb_shape={tuple(b.lb.shape)}",
                    flush=True,
                )
            except Exception:
                print(f"[ANALYZE-PROGRESS] L{layer.id} {layer.kind}", flush=True)
        visited.add(lid)

        if changed_or_maskdiff(layer, out_fact.bounds, None, eps):
            after[lid] = out_fact
            update_cache(layer, out_fact.bounds, None)
            for con in out_fact.cons: globalC.replace(con)
            for sid in net.succs.get(lid, []): WL.append(sid)

    return before, after, globalC
