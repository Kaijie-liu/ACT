# ===- act/pipeline/hybridz_spec_utils.py - HybridZ spec helpers -------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
"""Spec and witness helpers shared by packaged HybridZ pipeline entry points."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def check_real_unsafe(
    onnx_path: Path,
    input_shape,
    x: np.ndarray,
    C: np.ndarray,
    t: np.ndarray,
) -> Tuple[bool, np.ndarray]:
    """Replay a HybridZ-produced witness through ONNX Runtime for audit only."""

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    y = sess.run(None, {input_name: x.reshape(input_shape).astype(np.float32)})[0]
    y = y.reshape(-1).astype(np.float64)
    cy = C.reshape(C.shape[0], -1) @ y
    return bool((cy <= t.reshape(-1) + 1e-9).all()), cy


def flatten_query_specs(queries, n_out: int) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """Return unsafe specs in the common form ``C y <= t``.

    ``UNSAFE_LINEAR`` specs from the VNNLIB parser already use that unsafe
    direction and may contain multiple conjunctive rows.  Those rows must be
    checked together as one unsafe polytope.  Other OutputSpec kinds encode
    violation rows as ``C y >= t`` via ``OutputSpec.encode_linear``; negating
    each row converts them to the same one-row unsafe form without changing the
    set being checked.
    """

    flat: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for _, spec in queries:
        kind = str(getattr(spec, "kind", ""))
        if "UNSAFE_LINEAR" in kind:
            if not (hasattr(spec, "c") and spec.c is not None and spec.d is not None):
                raise ValueError(f"unsupported UNSAFE_LINEAR spec layout: {kind}")
            C = spec.c.detach().cpu().numpy().astype(np.float64)
            t = spec.d.detach().cpu().numpy().astype(np.float64).reshape(-1)
            C = C.reshape(-1, C.shape[-1])
            if C.shape[0] != t.size:
                raise ValueError(f"UNSAFE_LINEAR row/threshold mismatch: {C.shape} vs {t.shape}")
            flat.append((C, t, kind))
            continue

        import torch

        encoded = spec.encode_linear(
            B=1,
            n_out=int(n_out),
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        C = encoded["C"].detach().cpu().numpy().astype(np.float64).reshape(-1, int(n_out))
        thresholds = encoded["thresholds"].detach().cpu().numpy().astype(np.float64).reshape(-1)
        if C.shape[0] != thresholds.size:
            raise ValueError(f"encoded spec row/threshold mismatch: {C.shape} vs {thresholds.shape}")
        for i in range(C.shape[0]):
            flat.append((-C[i:i + 1], -thresholds[i:i + 1], f"{kind}_ROW_AS_UNSAFE_LINEAR"))
    if not flat:
        raise ValueError("no output specs to verify")
    return flat


def interval_hard_rivals_from_specs(
    flat_specs: List[Tuple[np.ndarray, np.ndarray, str]],
    final_bounds,
) -> Tuple[int, List[float]]:
    lb = final_bounds.lb.detach().cpu().numpy().reshape(-1).astype(np.float64)
    ub = final_bounds.ub.detach().cpu().numpy().reshape(-1).astype(np.float64)
    hard = 0
    lows: List[float] = []
    for C, t, _ in flat_specs:
        c = C.reshape(C.shape[0], -1)
        lo = c.clip(min=0) @ lb + c.clip(max=0) @ ub - t
        margin = float(np.min(lo))
        lows.append(margin)
        if margin <= 0.0:
            hard += 1
    return hard, lows


__all__ = [
    "check_real_unsafe",
    "flatten_query_specs",
    "interval_hard_rivals_from_specs",
]
