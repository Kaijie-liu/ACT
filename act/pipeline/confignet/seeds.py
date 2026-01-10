#!/usr/bin/env python3
#===- act/pipeline/confignet/seeds.py - Reproducible Seeding ------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Reproducible seeding utilities for ConfigNet.
#   - seed_everything(seed): set Python/NumPy/Torch seeds
#   - derive_seed(base_seed, idx, instance_spec): stable derived seed
#
# Notes:
#   We avoid Python's built-in hash() because it is salted per-process.
#   We use sha256 to derive stable 32-bit seeds.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import hashlib
import os
import random
from typing import Optional

import torch
from contextlib import contextmanager

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: base seed
        deterministic: if True, request deterministic algorithms (can reduce performance).
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed)}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    if np is not None:
        np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Best-effort deterministic settings.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
@contextmanager
def seeded(seed: int, deterministic: bool = False):
    """Seed everything, then restore RNG/global states after."""
    old_py = random.getstate()
    old_hash = os.environ.get("PYTHONHASHSEED", None)

    old_np = None
    if np is not None:
        old_np = np.random.get_state()

    old_torch = torch.get_rng_state()
    old_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    old_det = torch.are_deterministic_algorithms_enabled()
    old_cudnn_det = torch.backends.cudnn.deterministic
    old_cudnn_bench = torch.backends.cudnn.benchmark

    try:
        seed_everything(seed, deterministic=deterministic)
        yield
    finally:
        random.setstate(old_py)
        if old_hash is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = old_hash

        if np is not None and old_np is not None:
            np.random.set_state(old_np)

        torch.set_rng_state(old_torch)
        if old_cuda is not None:
            torch.cuda.set_rng_state_all(old_cuda)
        
        torch.use_deterministic_algorithms(old_det, warn_only=True)
        torch.backends.cudnn.deterministic = old_cudnn_det
        torch.backends.cudnn.benchmark = old_cudnn_bench

def _stable_u32_from_bytes(data: bytes) -> int:
    # Take first 4 bytes as little-endian u32
    return int.from_bytes(data[:4], byteorder="little", signed=False)


def derive_seed(base_seed: int, idx: int, instance_id: Optional[str] = None) -> int:
    """
    Derive a stable per-instance seed from (base_seed, idx, instance_id).

    Args:
        base_seed: global base seed
        idx: instance index
        instance_id: optional string; if provided it is included in derivation

    Returns:
        Derived seed in [0, 2^32-1]
    """
    if not isinstance(base_seed, int):
        raise TypeError(f"base_seed must be int, got {type(base_seed)}")
    if not isinstance(idx, int):
        raise TypeError(f"idx must be int, got {type(idx)}")

    payload = f"{base_seed}|{idx}|{instance_id or ''}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return _stable_u32_from_bytes(digest)
