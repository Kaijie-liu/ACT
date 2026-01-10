#!/usr/bin/env python3
#===- tests/test_confignet_jsonl_tensor_digest.py - Tensor Digest Tests --====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

import torch

from act.pipeline.confignet.jsonl import tensor_digest


def test_tensor_digest_stable_sha() -> None:
    t = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    d1 = tensor_digest(t)
    d2 = tensor_digest(t)
    assert d1["sha256"] == d2["sha256"]


def test_tensor_digest_inline_values_size() -> None:
    small = torch.arange(4, dtype=torch.float32)
    small_digest = tensor_digest(small, max_inline_numel=8)
    assert "values" in small_digest

    large = torch.arange(100, dtype=torch.float32)
    large_digest = tensor_digest(large, max_inline_numel=8)
    assert "values" not in large_digest
