"""PRUNE: per-rival generator pruning with sound interval-tail sidecar.

Per design lock dc_hz_phase_a_plan.md §1.3 and EXECUTION §2.1:

    PRUNE(c, G, d, K) keeps the top-K columns of G by relevance
    |d^T G[:, j]| and absorbs the dropped columns into an independent
    per-coordinate interval tail (NOT a single column r_tail[:, None]).

This implementation stores the tail as a sidecar `tail_radius` vector
of shape (n,), representing the conceptual `diag(tail_radius)`. This
keeps memory linear in n rather than n^2, which is the only viable
form for conv-sized hidden layers.

Soundness (proved in design lock §1.4):
    For any choice of d (including zero, random, adversarial), the
    pruned set
        { c + G_kept · xi_keep + tail_radius * xi_tail
          | xi_keep ∈ [-1, 1]^K,  xi_tail ∈ [-1, 1]^n }
    contains the original set { c + G · xi | xi ∈ [-1, 1]^ng }.

The proof uses the per-row triangle inequality:
    | Σ_{j ∈ drop} G[i, j] · xi_drop[j] |  ≤  Σ_{j ∈ drop} |G[i, j]| = tail_radius[i]

so each row's dropped contribution can be reproduced by xi_tail[i] ∈ [-1, +1].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class PrunedState:
    """Sidecar representation of a pruned HZ.

    Fields:
      c: (n,) center, unchanged from input.
      G_kept: (n, K_actual) explicit retained generators, where
              K_actual = min(K, ng) when pruning fires, else == ng.
      tail_radius: (n,) sound per-row interval remainder, or None when
              no pruning fired (K >= ng).
      metadata: bookkeeping dict with keys:
              - "keep": np.ndarray (K_actual,) of indices into original G
              - "drop": np.ndarray (ng - K_actual,) of dropped indices
              - "K_requested": int
              - "ng_original": int
              - other test-time helpers may be attached.

    Semantics:
      The set represented by this PrunedState is
          { c + G_kept · xi_keep + tail_radius * xi_tail
            | xi_keep ∈ [-1, 1]^K_actual,  xi_tail ∈ [-1, 1]^n }

      When tail_radius is None, no interval tail exists (pruning did
      not fire), and the set is just { c + G_kept · xi | xi ∈ [-1, 1]^K }.
    """

    c: np.ndarray
    G_kept: np.ndarray
    tail_radius: Optional[np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)


def prune(
    c: np.ndarray,
    G: np.ndarray,
    d: np.ndarray,
    K: int,
    *,
    return_metadata: bool = False,
    incoming_tail_radius: Optional[np.ndarray] = None,
) -> "PrunedState | Tuple[np.ndarray, np.ndarray]":
    """Top-K pruning with sound interval-tail sidecar.

    Args:
      c: (n,) center vector
      G: (n, ng) continuous generator matrix
      d: (n,) rival direction (heuristic only; soundness is INDEPENDENT
         of d's value — see test_adversarial_d_soundness)
      K: generator budget. If K >= ng, no pruning fires.
      incoming_tail_radius: (n,) prior tail to PRESERVE. The new tail is
        new = incoming + row_L1(dropped). Required for soundness when
        prune is called inside a forward loop that already accumulated
        tail from earlier prune steps. **SOUNDNESS-CRITICAL** —
        see 2026-06-04 advisor-caught bug.

    Returns:
      If return_metadata=True (default for tests and drivers):
        PrunedState dataclass as documented above.
      If return_metadata=False:
        (c, G) tuple when no pruning; (c, G_with_diag_tail) when pruned
        and the caller wants a dense expansion. NOT recommended for
        conv-sized layers.

    Raises:
      ValueError on shape mismatch.
    """
    c = np.asarray(c, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    if c.ndim != 1:
        raise ValueError(f"c must be 1-D, got shape {c.shape}")
    n = c.shape[0]
    if G.ndim != 2 or G.shape[0] != n:
        raise ValueError(
            f"G must be 2-D with shape[0] == n={n}; got {G.shape}"
        )
    if d.shape != (n,):
        raise ValueError(f"d must have shape ({n},); got {d.shape}")
    if incoming_tail_radius is not None:
        incoming_tail_radius = np.asarray(incoming_tail_radius, dtype=np.float64)
        if incoming_tail_radius.shape != (n,):
            raise ValueError(
                f"incoming_tail_radius must have shape ({n},); "
                f"got {incoming_tail_radius.shape}"
            )

    ng = G.shape[1]

    # Identity case: K >= ng → no pruning, no NEW tail; but PRESERVE incoming
    if K >= ng:
        keep = np.arange(ng, dtype=np.int64)
        drop = np.array([], dtype=np.int64)
        state = PrunedState(
            c=c.copy(),
            G_kept=G.copy(),
            tail_radius=(incoming_tail_radius.copy()
                          if incoming_tail_radius is not None else None),
            metadata={
                "keep": keep, "drop": drop,
                "K_requested": K, "ng_original": ng,
                "pruning_fired": False,
            },
        )
        if return_metadata:
            return state
        return state.c, state.G_kept

    # Compute relevance score for each generator
    # relevance[j] = |d · G[:, j]|
    relevance = np.abs(d @ G)        # (ng,)

    # Sort descending; keep top K
    # argsort is ascending, so flip
    order = np.argsort(-relevance, kind="stable")    # (ng,)
    keep = order[:K].copy()
    drop = order[K:].copy()
    # Sort the kept indices for reproducibility — relevance-tie ordering
    # is preserved by the original argsort.
    keep_sorted = np.sort(keep)
    drop_sorted = np.sort(drop)

    # Build G_kept and tail_radius
    G_kept = G[:, keep_sorted].copy()
    # tail_radius[i] = INCOMING_tail[i] + Σ_{j ∈ drop} |G[i, j]|
    new_drop_tail = np.abs(G[:, drop_sorted]).sum(axis=1)    # (n,)
    if incoming_tail_radius is not None:
        tail_radius = incoming_tail_radius + new_drop_tail
    else:
        tail_radius = new_drop_tail

    state = PrunedState(
        c=c.copy(),
        G_kept=G_kept,
        tail_radius=tail_radius,
        metadata={
            "keep": keep_sorted, "drop": drop_sorted,
            "K_requested": K, "ng_original": ng,
            "pruning_fired": True,
        },
    )
    if return_metadata:
        return state
    # Dense materialization for back-compat; raises if conv-sized
    return state.c, _materialize_dense_for_test(state)


def _materialize_dense_for_test(state: PrunedState) -> np.ndarray:
    """Build a dense G_full = [G_kept | diag(tail_radius)] for unit tests.

    DO NOT call this on conv-sized layers (n > 2048). The driver must
    use the sidecar form directly via downstream operators.
    """
    n = state.c.shape[0]
    if n > 2048:
        raise NotImplementedError(
            f"_materialize_dense_for_test refuses to build {n}x{n} diag; "
            f"this is the conv-size guard. Use the sidecar form."
        )
    if state.tail_radius is None:
        return state.G_kept.copy()
    diag_tail = np.diag(state.tail_radius)
    return np.concatenate([state.G_kept, diag_tail], axis=1)


# ─── Sound-image operator on PrunedState (for downstream layers) ──


def linear_image(state: PrunedState, W: np.ndarray,
                 b: Optional[np.ndarray] = None) -> PrunedState:
    """Apply a linear operator y = W @ x + b to the pruned HZ.

    Soundness: a linear operator is closed-form on each generator:
        new_c = W @ c + b
        new_G_kept[:, k] = W @ G_kept[:, k]
    For the interval-tail sidecar, the linear image is:
        new_tail_radius = |W| @ tail_radius     (row-sum of absolute weights)
    This is a sound over-approximation of the linear image of the
    diagonal interval box.

    For Conv2D, the caller should supply a callable representing the
    convolution; see `apply_linear_callable` for that path.
    """
    new_c = W @ state.c
    if b is not None:
        new_c = new_c + b
    new_G_kept = W @ state.G_kept
    new_tail = (np.abs(W) @ state.tail_radius
                if state.tail_radius is not None else None)
    return PrunedState(
        c=new_c, G_kept=new_G_kept, tail_radius=new_tail,
        metadata=dict(state.metadata),
    )
