"""Forward-only witness extractor — replaces the backward W^T chain.

PRINCIPLE (per advisor 2026-06-04 review):
  The previous decoder computed d_at_input via backward W^T·...·d, which —
  although it ignores bounds/slopes/gradients — IS a backward linear-adjoint
  pass and could be challenged under a strict reading of P1 (forward-only).

  The forward-coefficient extractor here uses NO backward chain. Instead it
  relies on the fact that the INITIAL HZ generator matrix is G_0 = diag(r_in),
  so each root generator column j ↔ input coordinate j.  As HZ propagates
  forward, those columns evolve via standard forward ops (Dense / Conv / ReLU
  triangle); at the output, column j of G_out (if not pruned away) is the
  partial derivative of the network's output w.r.t. ξ_j, which is the unit
  perturbation of input coordinate j.

  For the unsafe direction d_out, the closed-form LP maximizer of d_out · y
  over the structured HZ is obtained by reading off the SIGN of the forward
  generator coefficient:
      alpha_j  =  d_out · G_out_kept[:, col_idx(j)]
      ξ_j*     =  sign(alpha_j)
      x*[j]    =  c_in[j]  +  r_in[j] · sign(alpha_j)

  No backward pass through weights is performed; no gradient is computed;
  the candidate is uniquely determined by (model, input box, d_out) AS A
  FUNCTION OF THE FORWARD-PROPAGATED HZ ALONE.

  For pure-linear networks this is provably identical to the previous decode
  (sign of W_L^T · ... · W_1^T · d times r_in is the same as the sign of the
  forward generator coefficient). For ReLU networks it is MORE accurate
  because the forward generators reflect the DeepZ-triangle linearization
  actually taken, while the W^T chain ignores ReLU non-linearity entirely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from research.sc_hz.prune import PrunedState


# ─── Lineage-tracking PRUNE ─────────────────────────────────────────


def prune_with_lineage(c: np.ndarray, G: np.ndarray, d: np.ndarray, K: int,
                         input_coord_origin: Optional[np.ndarray] = None,
                         ) -> Tuple[PrunedState, np.ndarray]:
    """Like prune() but tracks which kept columns correspond to root input coords.

    Returns (state, new_input_coord_origin), where new_input_coord_origin is
    an int array of length n_kept; entry k is the input-coord index of the
    k-th kept column, or -1 if the kept column does not correspond to a
    root input coord (e.g. ReLU slack).
    """
    n, ng = G.shape
    if input_coord_origin is None:
        # First call: assume columns 0..n_input-1 are input-coord generators.
        # In our use, the initial G is diag(r_in) with n_input cols.
        input_coord_origin = np.arange(ng, dtype=np.int64)

    if K >= ng or ng == 0:
        # Identity: no prune
        return PrunedState(c=c.copy(), G_kept=G.copy(),
                            tail_radius=None,
                            metadata={"input_coord_origin": input_coord_origin.copy()}), \
                input_coord_origin.copy()

    scores = np.abs(d @ G) if d is not None else np.abs(G).sum(axis=0)
    order = np.argsort(scores)[::-1]
    keep = order[:K]
    drop = order[K:]
    G_kept = G[:, keep]
    new_origin = input_coord_origin[keep]
    tail_radius = np.abs(G[:, drop]).sum(axis=1) if drop.size > 0 else np.zeros(n)
    state = PrunedState(c=c.copy(), G_kept=G_kept,
                          tail_radius=tail_radius,
                          metadata={"input_coord_origin": new_origin})
    return state, new_origin


# ─── Forward witness decoder ────────────────────────────────────────


def decode_xi_star_forward(state_out: PrunedState,
                              d_out: np.ndarray,
                              c_in: np.ndarray,
                              r_in: np.ndarray,
                              ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Closed-form forward-only LP maximizer.

    Inputs:
      state_out: PrunedState at network output (forward-propagated)
      d_out:     unsafe direction in output space
      c_in:      input box center  (n_in,)
      r_in:      input box radius  (n_in,)

    Method:
      For each kept generator column k, compute alpha_k = d_out · G_kept[:, k].
      Look up which input coordinate origin[k] each kept col corresponds to;
      for input coords whose generator is still kept, set
          ξ_j*  =  sign(alpha_at_input_coord_j)
      For input coords whose generator was pruned away, set ξ_j* = 0 (we
      cannot determine the optimal sign without that generator). This is
      sound (it just means we may miss those candidates).

    Returns (x_star, metadata).
    """
    n_in = c_in.shape[0]
    origin = state_out.metadata.get("input_coord_origin", None)
    if origin is None:
        # Fallback: assume first n_in kept cols are input-coord (only valid
        # when no PRUNE was applied between input and output)
        K_kept = state_out.G_kept.shape[1]
        origin = np.concatenate([
            np.arange(min(n_in, K_kept), dtype=np.int64),
            -np.ones(max(0, K_kept - n_in), dtype=np.int64),
        ])

    K_kept = state_out.G_kept.shape[1]
    assert origin.shape[0] == K_kept, \
        f"origin {origin.shape[0]} mismatch G_kept K {K_kept}"

    # Compute alpha for each kept column
    alpha_kept = d_out @ state_out.G_kept  # (K_kept,)

    # For each input coord, find the alpha if its generator was kept
    xi_star = np.zeros(n_in, dtype=np.float64)
    found_per_coord = np.zeros(n_in, dtype=bool)
    n_pruned = 0
    for k in range(K_kept):
        coord = int(origin[k])
        if coord < 0 or coord >= n_in:
            continue
        xi_star[coord] = np.sign(alpha_kept[k])
        found_per_coord[coord] = True
    n_pruned = int(np.sum(~found_per_coord))

    # x_star = c_in + r_in * xi_star; coords whose gen was pruned stay at center
    x_star = c_in + r_in * xi_star

    return x_star, {
        "method": "forward_coefficient",
        "n_input_coord_generators_kept": int(np.sum(found_per_coord)),
        "n_input_coord_generators_pruned": n_pruned,
        "max_abs_alpha": float(np.abs(alpha_kept).max() if alpha_kept.size else 0.0),
    }


def initial_state_with_lineage(c_in: np.ndarray, r_in: np.ndarray) -> PrunedState:
    """Build the initial PrunedState (box input) with input-coord lineage in metadata.

    Initial: c = c_in, G = diag(r_in), tail = 0
    Each column j ↔ input coordinate j (origin[j] = j).
    """
    n_in = c_in.shape[0]
    G_0 = np.diag(r_in)
    origin = np.arange(n_in, dtype=np.int64)
    return PrunedState(c=c_in.copy(), G_kept=G_0, tail_radius=None,
                        metadata={"input_coord_origin": origin})


# ─── Forward propagation without backward chain ─────────────────────


def forward_propagate_no_backward(
    state: PrunedState,
    layers,
    K_per_layer: int = 256,
    initial_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[PrunedState, List[Dict[str, Any]]]:
    """Forward HZ propagation with lineage tracking, NO backward d-chain.

    PRINCIPLE: this is the corrected forward-only path. Unlike the previous
    `forward_propagate`, this function does NOT take a d_per_layer chain
    (which required a backward W^T pass). Instead:
      - Pruning between layers (if needed) uses column L2 norm only —
        a direction-agnostic score derived from the forward generators.
      - input_coord_origin metadata is maintained so the final witness
        decode can read off forward generator coefficients.

    Returns (final_state, traces).
    """
    import research.sc_hz.ops as scops

    cur_shape = initial_shape or (state.c.shape[0],)
    traces: List[Dict[str, Any]] = []

    def _ensure_origin(s: PrunedState, prev_origin: np.ndarray,
                        op_kind: str) -> PrunedState:
        """After op s, fix up origin metadata. Most ops preserve column count;
        relu may add columns, in which case the new ones are NOT input-coord.
        """
        cur_K = s.G_kept.shape[1]
        prev_K = prev_origin.shape[0]
        if cur_K == prev_K:
            new_origin = prev_origin.copy()
        elif cur_K > prev_K:
            extra = cur_K - prev_K
            new_origin = np.concatenate([
                prev_origin,
                -np.ones(extra, dtype=np.int64),
            ])
        else:
            # cur_K < prev_K should only happen via prune; not in raw ops
            new_origin = prev_origin[:cur_K].copy()
        s.metadata["input_coord_origin"] = new_origin
        return s

    # Initial prune (if needed) by L2 norm — direction-agnostic
    if state.G_kept.shape[1] > K_per_layer:
        scores = np.linalg.norm(state.G_kept, axis=0)
        order = np.argsort(scores)[::-1]
        keep = order[:K_per_layer]
        drop = order[K_per_layer:]
        tail = np.abs(state.G_kept[:, drop]).sum(axis=1)
        prev_origin = state.metadata.get("input_coord_origin",
                                            np.arange(state.G_kept.shape[1], dtype=np.int64))
        new_origin = prev_origin[keep].copy()
        state = PrunedState(
            c=state.c.copy(), G_kept=state.G_kept[:, keep],
            tail_radius=(state.tail_radius + tail
                          if state.tail_radius is not None else tail),
            metadata={"input_coord_origin": new_origin},
        )

    for i, op in enumerate(layers):
        prev_origin = state.metadata.get("input_coord_origin")
        if prev_origin is None:
            prev_origin = np.arange(state.G_kept.shape[1], dtype=np.int64)

        k = op.kind
        if k == "sub":
            state = scops.apply_sub(state, op.params["const"])
        elif k == "flatten":
            state = scops.apply_flatten(state)
            cur_shape = (state.c.shape[0],)
        elif k == "dense":
            state = scops.apply_dense(state, op.params["W"],
                                        op.params.get("b"))
            cur_shape = (int(op.params["W"].shape[0]),)
        elif k == "conv2d":
            state, cur_shape = scops.apply_conv2d(
                state, op.params["W"], op.params.get("b"),
                input_shape=cur_shape,
                stride=op.params.get("stride", 1),
                padding=op.params.get("padding", 0),
                groups=op.params.get("groups", 1),
            )
        elif k == "bn":
            state = scops.apply_bn(state, op.params["scale"], op.params["shift"],
                                     input_shape=cur_shape)
        elif k == "relu":
            state, _ = scops.apply_relu_triangle(state)
        elif k == "maxpool":
            state, cur_shape = scops.apply_maxpool2d(
                state, input_shape=cur_shape,
                kernel_size=op.params.get("kernel_size", 2),
                stride=op.params.get("stride", None),
            )
        elif k == "add":
            raise NotImplementedError(
                "residual Add requires multi-parent tracking; not in Phase A"
            )
        else:
            raise NotImplementedError(f"forward for op '{k}' not implemented")

        # Reattach origin metadata (extend with -1 for any new cols)
        state = _ensure_origin(state, prev_origin, k)

        traces.append({
            "layer": i, "op": k, "ng": state.G_kept.shape[1],
            "tail_sum": (float(state.tail_radius.sum())
                          if state.tail_radius is not None else 0.0),
        })

        # PRUNE by L2 norm (no direction)
        if state.G_kept.shape[1] > K_per_layer:
            scores = np.linalg.norm(state.G_kept, axis=0)
            order = np.argsort(scores)[::-1]
            keep = order[:K_per_layer]
            drop = order[K_per_layer:]
            tail_add = np.abs(state.G_kept[:, drop]).sum(axis=1)
            cur_origin = state.metadata["input_coord_origin"]
            new_origin = cur_origin[keep].copy()
            state = PrunedState(
                c=state.c.copy(), G_kept=state.G_kept[:, keep],
                tail_radius=(state.tail_radius + tail_add
                              if state.tail_radius is not None else tail_add),
                metadata={"input_coord_origin": new_origin},
            )

    return state, traces
