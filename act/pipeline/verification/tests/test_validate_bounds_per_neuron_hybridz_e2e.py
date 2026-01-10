#!/usr/bin/env python3
#===- tests/test_validate_bounds_per_neuron_hybridz_e2e.py - HybridZ ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

import torch

from act.pipeline.verification.validate_verifier import VerificationValidator


def test_validate_bounds_per_neuron_hybridz_e2e() -> None:
    dtype = torch.float64
    validator = VerificationValidator(device="cpu", dtype=dtype)
    networks = validator.factory.list_networks()
    if "control_conservative" in networks:
        networks = ["control_conservative"]
    else:
        networks = networks[:1]

    summary = validator.validate_bounds_per_neuron(
        networks=networks,
        tf_modes=["hybridz"],
        num_samples=1,
        atol=1e-6,
        rtol=0.0,
        topk=5,
        strict=True,
    )
    assert summary["total"] == 1

    results = [
        r for r in validator.validation_results
        if r.get("validation_type") == "bounds_per_neuron"
    ]
    assert len(results) == 1
    result = results[0]
    assert result["validation_status"] in ("PASSED", "FAILED", "ERROR")
    assert "layerwise_stats" in result
    assert "violations_topk" in result
    assert "alignment" in result
