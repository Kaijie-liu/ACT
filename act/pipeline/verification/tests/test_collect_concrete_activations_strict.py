#!/usr/bin/env python3
#===- tests/test_collect_concrete_activations_strict.py - Activations --====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

import torch
import torch.nn as nn

from act.pipeline.verification.activations import collect_concrete_activations


class _ReuseReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.relu(x))


def test_collect_concrete_activations_strict_multiple_calls() -> None:
    model = _ReuseReLU()
    x = torch.tensor([[-1.0, 1.0]])
    events, errors, _warnings = collect_concrete_activations(
        model,
        x,
        strict_single_call_per_module=True,
    )
    assert len(events) == 2
    assert errors
