"""Per-rival forward HZ propagation with PRUNE at every layer.

Per design lock §1.2 and EXECUTION §2.3 (Option A — full per-rival
forward HZ). Given a feedforward Dense+ReLU model and an input box,
this module runs N_RIVAL separate forward passes, each with its own
PrunedState pruned per-rival at every layer using the pre-computed
d_L^r direction.

Phase A scope:
  - Dense + ReLU (DeepZ triangle) only.
  - Conv2D / MaxPool / Add will be added when their adjoints
    (d_L computation) are implemented.

Output verdict per rival:
  - CERT: LP UB on (d_N · y) < 0  (the rival margin cannot reach 0)
  - FAL candidate: LP UB >= 0; the closed-form maximizer xi* can be
    decoded to an input x* via x* = c_in + r_in * sign(d_in)
    (each input coordinate's contribution is determined by xi_keep[i]
     for the corresponding input-generator column).
  - The driver runs strict ORT replay on the decoded x* to confirm FAL,
    else UNK.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from research.sc_hz.prune import PrunedState, prune
from research.sc_hz.precompute_direction import precompute_d_per_layer
from research.sc_hz.ops import (
    apply_dense, apply_relu_triangle,
    bounds, lp_ub_rival_margin,
)


@dataclass
class RivalForwardResult:
    """Per-rival result of pruned_forward_dense."""

    rival: int
    lp_ub_rival_margin: float
    verdict: str   # "CERT" | "FAL_CANDIDATE" | "UNK"
    xi_star_input: Optional[np.ndarray]    # closed-form max-direction in input box
    K_per_layer: List[int]
    layer_traces: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PrunedForwardResult:
    """Aggregate of all rivals for one iid."""

    iid_label: str
    n_rivals: int
    per_rival: List[RivalForwardResult]
    overall_verdict: str    # "CERT" if all rivals CERT; "FAL_CANDIDATE" if any FAL; else "UNK"
    max_lp_ub: float
    worst_rival: int


# ─── The driver ───────────────────────────────────────────────────


def pruned_forward_dense(
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    input_box_lb: np.ndarray,
    input_box_ub: np.ndarray,
    y_true: int,
    rivals: Sequence[int],
    K_per_layer: int = 256,
    iid_label: str = "",
) -> PrunedForwardResult:
    """Per-rival forward HZ on a Dense+ReLU model.

    Args:
      weights:  ordered [W_1, ..., W_{N+1}] where W_{N+1} is the output
                classifier. Dense + ReLU between each pair until the
                last; the output W_{N+1} produces the class scores y.
      biases:   matching list of biases (each same shape as the bias
                vector for that layer). May contain None for "no bias".
      input_box_lb, input_box_ub: (n_input,) arrays defining the L∞ box.
      y_true:   true class index.
      rivals:   list of rival class indices.
      K_per_layer: generator budget at each layer's PRUNE step.
      iid_label: optional string for receipts.

    Returns:
      PrunedForwardResult aggregating the per-rival outcomes.
    """
    input_box_lb = np.asarray(input_box_lb, dtype=np.float64).reshape(-1)
    input_box_ub = np.asarray(input_box_ub, dtype=np.float64).reshape(-1)
    n_input = input_box_lb.shape[0]
    if input_box_ub.shape != (n_input,):
        raise ValueError("input box lb/ub shape mismatch")

    # Pre-compute d_L^r for each rival
    weights_np = [np.asarray(w, dtype=np.float64) for w in weights]
    biases_np = [(np.asarray(b, dtype=np.float64).reshape(-1)
                  if b is not None else None) for b in biases]

    per_rival_results: List[RivalForwardResult] = []
    for r in rivals:
        # d-chain per rival: [d_0, d_1, ..., d_N], len = len(weights)
        # d_0: input space, d_N: pre-classifier (last-hidden) space.
        d_chain = precompute_d_per_layer(weights_np, rival=r, y_true=y_true)

        # Build initial input HZ as an axis-aligned box
        c0 = (input_box_lb + input_box_ub) / 2.0
        r0 = (input_box_ub - input_box_lb) / 2.0
        # Initial generators: diagonal box with ng = n_input
        G0 = np.diag(r0).astype(np.float64)
        state = PrunedState(
            c=c0, G_kept=G0, tail_radius=None,
            metadata={"layer": "input", "ng": n_input},
        )

        # Prune at input level using d_0
        state = prune(state.c, state.G_kept, d_chain[0], K_per_layer,
                       return_metadata=True)

        layer_traces: List[Dict[str, Any]] = [{
            "layer": "input",
            "ng": state.G_kept.shape[1],
            "tail_radius_sum": (float(state.tail_radius.sum())
                                 if state.tail_radius is not None else 0.0),
        }]

        # Walk weights L = 1..N: Dense → ReLU → Prune. The last layer (N+1)
        # is the classifier; only Dense, no ReLU after.
        N = len(weights_np)
        for L in range(N):
            W_L = weights_np[L]
            b_L = biases_np[L]
            # Dense
            state = apply_dense(state, W_L, b_L)
            # ReLU on every Dense except the LAST (classifier)
            if L < N - 1:
                state, _unstable = apply_relu_triangle(state)
                # Prune at this hidden layer using d_L (or d_{L+1} in our
                # chain indexing — see precompute_direction)
                # d_chain[L+1] is the direction at h_{L+1}'s space (post-this-layer)
                if (L + 1) < len(d_chain):
                    d_here = d_chain[L + 1]
                else:
                    d_here = None
                if d_here is not None:
                    state = prune(state.c, state.G_kept, d_here, K_per_layer,
                                   return_metadata=True)
            # No prune after the classifier layer; the rival margin LP
            # is solved directly on the post-classifier state.
            layer_traces.append({
                "layer": f"L{L+1}_Dense",
                "ng": state.G_kept.shape[1],
                "tail_radius_sum": (float(state.tail_radius.sum())
                                     if state.tail_radius is not None else 0.0),
            })

        # At this point `state` is the output set y. The rival direction
        # in output space is e_r - e_{y_t}.
        n_classes = weights_np[-1].shape[0]
        d_out = np.zeros(n_classes, dtype=np.float64)
        d_out[r] = 1.0
        d_out[y_true] = -1.0

        ub = lp_ub_rival_margin(state, d_out)

        # Verdict
        if ub < -1e-9:
            verdict = "CERT"
            xi_star = None
        else:
            verdict = "FAL_CANDIDATE"
            # Closed-form maximizer: xi_input[i] = sign(d_input[i]) per
            # the input-coordinate direction d_chain[0]. The input value
            # that produces the max-rival-margin is:
            #     x_star[i] = c_in[i] + r_in[i] * sign(d_chain[0][i])
            # (signs chosen to maximize d_chain[0] · x).
            sign_d = np.sign(d_chain[0])
            # Avoid 0-sign: default to +1
            sign_d = np.where(sign_d == 0, 1.0, sign_d)
            xi_star = c0 + r0 * sign_d

        per_rival_results.append(RivalForwardResult(
            rival=int(r),
            lp_ub_rival_margin=float(ub),
            verdict=verdict,
            xi_star_input=xi_star,
            K_per_layer=[t["ng"] for t in layer_traces],
            layer_traces=layer_traces,
        ))

    max_ub = max(r.lp_ub_rival_margin for r in per_rival_results)
    worst = int(np.argmax([r.lp_ub_rival_margin for r in per_rival_results]))
    worst_rival = per_rival_results[worst].rival
    if all(r.verdict == "CERT" for r in per_rival_results):
        overall = "CERT"
    elif any(r.verdict == "FAL_CANDIDATE" for r in per_rival_results):
        overall = "FAL_CANDIDATE"
    else:
        overall = "UNK"

    return PrunedForwardResult(
        iid_label=iid_label,
        n_rivals=len(rivals),
        per_rival=per_rival_results,
        overall_verdict=overall,
        max_lp_ub=max_ub,
        worst_rival=worst_rival,
    )
