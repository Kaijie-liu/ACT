"""Structural gate for the generic MLP end-cap profile.

The gate decides — purely from `net.layers`, the vnnlib `pair`, and the
already-computed CIFAR-narrow-profile flag — whether the generic MLP
end-cap sidecar should fire for this iid. The gate is FAIL-CLOSED: when
unsure, return ``False`` so the sidecar stays off and the canonical
verifier owns the verdict.

Conditions (all must hold):

1. There is a FLATTEN layer in the net.
2. The tail AFTER the last FLATTEN (ignoring ASSERT noise) matches
   one of:
   - ``DENSE`` (single-layer affine head — malbeware / soundnessbench / vgg)
   - ``DENSE -> DENSE`` (degenerate hidden-layer head, no ReLU)
   - ``DENSE -> RELU -> DENSE`` (canonical 2-layer MLP head — Tiny/CIFAR)

   DENSE here accepts the synonyms ``DENSE``, ``GEMM``, ``MATMUL``.
3. Final output dim (last NON-ASSERT layer) is small enough to be a
   classification head: ``1 <= out_dim <= 1024``. The 1024 cap rejects
   pixel/grid outputs (YOLO's 21125-dim detection head) while admitting
   all standard top-k classifiers up to ImageNet scale.
4. The vnnlib `pair` carries a ``labeled_tensor`` field AND the
   vnnlib file contains at least one `(>= Y_a Y_b)` style top-1-rival
   constraint. Some benchmarks (soundnessbench) populate
   ``labeled_tensor`` for any classification-shape model but use a
   different output-constraint format (``Y_i op const``) that the
   LP-endcap cannot decode; the second check rejects those.
5. CIFAR's narrow profile is not already active (it has its own
   FAL-only sidecar path).
6. The env knob ``ACT_HZ_MLP_ENDCAP_PROFILE`` is not explicitly off.

When all 6 hold, the sidecar may fire — but the sidecar itself runs a
SECOND fail-closed check on the snapshot (root_ng == input_dim AND
root_ng <= snap.ng), so this gate alone never triggers an unsound
upgrade.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple


DENSE_KIND_SYNONYMS = ("DENSE", "GEMM", "MATMUL")
ASSERT_KIND = "ASSERT"
FLATTEN_KIND = "FLATTEN"
RELU_KIND = "RELU"
DEFAULT_MAX_OUT_DIM = 1024


@dataclass(frozen=True)
class GateDiagnostic:
    """One-line summary of why the gate did or did not fire.

    Exposed via ``q_solver_stats["generic_mlp_endcap_gate"]`` for
    per-iid audit. The `enabled` field is the gate's verdict.
    """

    tail_supported: bool
    tail_kinds: Optional[Tuple[str, ...]]
    final_out_dim: int
    is_top1_robust: bool
    cifar_endcap_active: bool
    env_off: bool
    enabled: bool


def _last_flatten_index(layers: Sequence[Any]) -> Optional[int]:
    """Return index of the LAST FLATTEN layer, or None if absent."""
    idx: Optional[int] = None
    for i, L in enumerate(layers):
        if str(L.kind).upper() == FLATTEN_KIND:
            idx = i
    return idx


def _tail_after(
    layers: Sequence[Any], start_idx: int
) -> Tuple[str, ...]:
    """Layer kinds AFTER `start_idx`, excluding ASSERT noise."""
    return tuple(
        str(L.kind).upper()
        for L in layers[start_idx + 1:]
        if str(L.kind).upper() != ASSERT_KIND
    )


def _is_mlp_tail(tail_kinds: Tuple[str, ...]) -> bool:
    """True for single-Dense or 2-layer MLP heads.

    Supported patterns:
    - ``DENSE``                        — single-layer affine head
    - ``DENSE -> DENSE``               — 2-layer, no hidden activation
    - ``DENSE -> RELU -> DENSE``       — 2-layer with hidden ReLU
    """
    if len(tail_kinds) == 1:
        return tail_kinds[0] in DENSE_KIND_SYNONYMS
    if len(tail_kinds) == 2:
        return (
            tail_kinds[0] in DENSE_KIND_SYNONYMS
            and tail_kinds[1] in DENSE_KIND_SYNONYMS
        )
    if len(tail_kinds) == 3:
        return (
            tail_kinds[0] in DENSE_KIND_SYNONYMS
            and tail_kinds[1] == RELU_KIND
            and tail_kinds[2] in DENSE_KIND_SYNONYMS
        )
    return False


def _last_non_assert(layers: Sequence[Any]):
    """Return the last layer whose kind is not ASSERT."""
    for L in reversed(layers):
        if str(L.kind).upper() != ASSERT_KIND:
            return L
    return None


def supports_generic_mlp_endcap(
    *,
    layers: Sequence[Any],
    pair: Mapping[str, Any],
    cifar_endcap_active: bool,
    env: Optional[Mapping[str, str]] = None,
    max_out_dim: int = DEFAULT_MAX_OUT_DIM,
) -> GateDiagnostic:
    """Decide whether the generic MLP end-cap profile should fire.

    Parameters
    ----------
    layers : iterable of Layer-like
        ``net.layers`` from the ACT representation. Each layer must
        expose ``.kind`` (string-like) and ``.out_vars`` (sized).
    pair : Mapping
        The vnnlib pair (``load_vnnlib_pair`` return). Must carry a
        ``"labeled_tensor"`` key when top-1 robust.
    cifar_endcap_active : bool
        Whether the CIFAR narrow profile already activated for this iid.
        If so, defer to it.
    env : Mapping[str, str], optional
        Environment dict for the env-off check (defaults to ``os.environ``).
    max_out_dim : int
        Upper cap on the final non-ASSERT layer's output dim.

    Returns
    -------
    GateDiagnostic
        See class docstring. ``enabled`` is the gate's final verdict.
    """
    env_dict = os.environ if env is None else env
    env_off = env_dict.get(
        "ACT_HZ_MLP_ENDCAP_PROFILE", "1"
    ).strip().lower() in ("0", "false", "no", "off")

    flatten_idx = _last_flatten_index(layers)
    if flatten_idx is None:
        return GateDiagnostic(
            tail_supported=False,
            tail_kinds=None,
            final_out_dim=0,
            is_top1_robust=bool(pair.get("labeled_tensor") is not None),
            cifar_endcap_active=cifar_endcap_active,
            env_off=env_off,
            enabled=False,
        )

    tail_kinds = _tail_after(layers, flatten_idx)
    tail_supported = _is_mlp_tail(tail_kinds)

    last_layer = _last_non_assert(layers)
    final_out_dim = (
        int(len(last_layer.out_vars)) if last_layer is not None else 0
    )

    is_top1_robust = bool(pair.get("labeled_tensor") is not None)
    # Stricter check: ALSO require the vnnlib file to contain at least
    # one (>= Y_a Y_b) rival-vs-truth constraint. Some benchmarks
    # (soundnessbench) populate labeled_tensor for any classifier-shape
    # net but encode safety via `Y_i op const` predicates that the
    # generic LP-endcap cannot decode — refusing them here keeps the
    # snapshot-write step from running pointlessly on those iids.
    if is_top1_robust:
        vnn_path = pair.get("vnnlib_path") or pair.get("vnnlib_spec")
        if vnn_path:
            try:
                import re as _re
                from pathlib import Path as _GP
                _path_obj = _GP(str(vnn_path))
                if _path_obj.is_file():
                    _text = _path_obj.read_text()
                    if not _re.search(r"\(>=\s+Y_\d+\s+Y_\d+\)", _text):
                        is_top1_robust = False
            except Exception:
                # Reading the file shouldn't kill the gate; default to
                # the labeled_tensor-only check on any read error
                # (downstream sidecar will fail-closed on a bad parse).
                pass

    enabled = (
        (not env_off)
        and tail_supported
        and 1 <= final_out_dim <= max_out_dim
        and is_top1_robust
        and (not cifar_endcap_active)
    )

    return GateDiagnostic(
        tail_supported=tail_supported,
        tail_kinds=tail_kinds,
        final_out_dim=final_out_dim,
        is_top1_robust=is_top1_robust,
        cifar_endcap_active=cifar_endcap_active,
        env_off=env_off,
        enabled=enabled,
    )
