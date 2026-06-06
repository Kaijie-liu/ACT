"""Minimal vnnlib parser for Phase A.

Extracts:
  - input box (per X_i lb/ub)
  - output spec as a list of "unsafe conditions". Each unsafe condition
    is a linear functional `d` on the output Y and a `threshold`:
        condition = "d · Y >= threshold is unsafe"
    The verifier must prove `d · Y < threshold` for all x in the input box.

Supported assertion forms:
  - Top-1 robustness: `(>= Y_r Y_t)`  → d = e_r - e_t, threshold = 0
  - Constant threshold (acasxu): `(>= Y_i constant)` → d = e_i, threshold = constant
  - `(<= Y_i constant)` → d = -e_i, threshold = -constant

OR/AND blocks: we flatten and collect every Y_? Y_? / Y_? const inequality
into the unsafe_conditions list. The verdict aggregator interprets:
  - CERT: ALL unsafe conditions cannot hold (LP UB on d·y < threshold for all).
  - FAL_CANDIDATE: ANY unsafe condition can hold (LP UB ≥ threshold).
For OR-structured assertions, only ONE branch needs to be unsafe → same as
"any unsafe condition reachable". For AND-structured, all must be jointly
reachable; our flatten approximation is conservative (treats AND as
multiple conditions to satisfy individually, which is sound for CERT but
may over-fire FAL).
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np


def parse_vnnlib(path: str, n_input: int, n_classes: int
                   ) -> Tuple[np.ndarray, np.ndarray,
                                List[Tuple[np.ndarray, float, str]]]:
    """Return (input_lb, input_ub, unsafe_conditions).

    unsafe_conditions: list of (d, threshold, label) where the unsafe
    condition is "d · Y >= threshold".
    """
    with open(path) as f:
        text = f.read()

    lb = np.full(n_input, -np.inf, dtype=np.float64)
    ub = np.full(n_input, np.inf, dtype=np.float64)

    pat_x_ge = re.compile(r"\(assert\s+\(>=\s+X_(\d+)\s+([-0-9eE.+]+)\s*\)\s*\)")
    pat_x_le = re.compile(r"\(assert\s+\(<=\s+X_(\d+)\s+([-0-9eE.+]+)\s*\)\s*\)")
    for m in pat_x_ge.finditer(text):
        i = int(m.group(1))
        if 0 <= i < n_input:
            v = float(m.group(2))
            lb[i] = v if not np.isfinite(lb[i]) else max(lb[i], v)
    for m in pat_x_le.finditer(text):
        i = int(m.group(1))
        if 0 <= i < n_input:
            v = float(m.group(2))
            ub[i] = v if not np.isfinite(ub[i]) else min(ub[i], v)

    unsafe: List[Tuple[np.ndarray, float, str]] = []

    # Y_r >= Y_t
    for m in re.finditer(r"\(>=\s+Y_(\d+)\s+Y_(\d+)\s*\)", text):
        r = int(m.group(1)); t = int(m.group(2))
        if r != t and 0 <= r < n_classes and 0 <= t < n_classes:
            d = np.zeros(n_classes, dtype=np.float64)
            d[r] = 1.0; d[t] = -1.0
            unsafe.append((d, 0.0, f"Y_{r}>=Y_{t}"))

    # Y_r <= Y_t  (equivalent to Y_t >= Y_r)
    for m in re.finditer(r"\(<=\s+Y_(\d+)\s+Y_(\d+)\s*\)", text):
        r = int(m.group(1)); t = int(m.group(2))
        if r != t and 0 <= r < n_classes and 0 <= t < n_classes:
            d = np.zeros(n_classes, dtype=np.float64)
            d[t] = 1.0; d[r] = -1.0
            unsafe.append((d, 0.0, f"Y_{r}<=Y_{t}"))

    # Y_i >= constant
    for m in re.finditer(r"\(>=\s+Y_(\d+)\s+([-0-9eE.+]+)\s*\)", text):
        i = int(m.group(1)); c = float(m.group(2))
        if 0 <= i < n_classes:
            d = np.zeros(n_classes, dtype=np.float64); d[i] = 1.0
            unsafe.append((d, c, f"Y_{i}>={c}"))

    # Y_i <= constant  (equivalent to -Y_i >= -constant)
    for m in re.finditer(r"\(<=\s+Y_(\d+)\s+([-0-9eE.+]+)\s*\)", text):
        i = int(m.group(1)); c = float(m.group(2))
        if 0 <= i < n_classes:
            d = np.zeros(n_classes, dtype=np.float64); d[i] = -1.0
            unsafe.append((d, -c, f"Y_{i}<={c}"))

    # Dedupe
    seen = set(); unique = []
    for u in unsafe:
        key = (tuple(u[0]), u[1])
        if key not in seen:
            seen.add(key); unique.append(u)

    if not np.isfinite(lb).all() or not np.isfinite(ub).all():
        raise ValueError(f"vnnlib {path}: some input dims unbounded")
    return lb, ub, unique
