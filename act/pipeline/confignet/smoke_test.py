#!/usr/bin/env python3
#===- act/pipeline/confignet/smoke_test.py - ConfigNet Smoke Test -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Smoke test for ConfigNet (pipeline-side instance generator).
#
# What it tests:
#   1) Sampling produces InstanceSpec list with stable instance_id/seed.
#   2) Builders can construct wrapped VerifiableModel without back_end.
#   3) Forward pass returns dict with required keys.
#   4) For each input spec kind, we build a feasible input x that satisfies it.
#   5) Reproducibility: building the same instance twice yields identical weights.
#
# Run:
#   python -m act.pipeline.confignet.smoke_test --num 5 --seed 0 --device cpu --dtype float64
#
# Exit codes:
#   0 - pass
#   1 - fail
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import argparse
import sys
import logging
from typing import Tuple, Optional

import torch

from act.pipeline.confignet import (
    ConfigNetConfig,
    sample_instances,
    build_wrapped_model,
)
from act.pipeline.confignet.input_sampling import make_feasible_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _prod(shape: Tuple[int, ...]) -> int:
    p = 1
    for s in shape:
        p *= int(s)
    return p




def assert_state_dict_equal(a: torch.nn.Module, b: torch.nn.Module) -> None:
    """
    Strict equality check for reproducibility: all tensors in state_dict must match exactly.
    """
    sa = a.state_dict()
    sb = b.state_dict()
    if sa.keys() != sb.keys():
        raise AssertionError(f"state_dict key mismatch: {sa.keys()} vs {sb.keys()}")
    for k in sa.keys():
        ta = sa[k]
        tb = sb[k]
        if torch.is_tensor(ta) and torch.is_tensor(tb):
            if ta.dtype != tb.dtype or ta.shape != tb.shape:
                raise AssertionError(f"tensor meta mismatch at '{k}': {ta.dtype}/{ta.shape} vs {tb.dtype}/{tb.shape}")
            if not torch.equal(ta, tb):
                # show max diff
                diff = (ta - tb).abs().max().item() if ta.numel() > 0 else 0.0
                raise AssertionError(f"tensor value mismatch at '{k}', max_abs_diff={diff}")
        else:
            if ta != tb:
                raise AssertionError(f"non-tensor value mismatch at '{k}': {ta} vs {tb}")


def run_smoke(num: int, base_seed: int, device: str, dtype_str: str) -> None:
    dtype = torch.float64 if dtype_str == "float64" else torch.float32
    dev = torch.device(device)

    cfg = ConfigNetConfig(num_instances=num, base_seed=base_seed)
    specs = sample_instances(cfg)

    # Basic sampler sanity
    if len(specs) != num:
        raise AssertionError(f"sample_instances returned {len(specs)} specs, expected {num}")

    logger.info("Sampled %d instances (base_seed=%d).", len(specs), base_seed)

    for i, inst_spec in enumerate(specs):
        logger.info("---- [%d/%d] instance_id=%s seed=%d family=%s ----",
                    i + 1, len(specs), inst_spec.instance_id, inst_spec.seed, inst_spec.family.value)

        # Build twice to test reproducibility (weights)
        m1 = build_wrapped_model(inst_spec, device=device, dtype=dtype)
        m2 = build_wrapped_model(inst_spec, device=device, dtype=dtype)
        assert_state_dict_equal(m1, m2)

        # Feasible input
        x = make_feasible_input(inst_spec, device=dev, dtype=dtype)

        # Forward
        with torch.no_grad():
            out = m1(x)

        # Check output structure from VerifiableModel
        required_keys = ["output", "input_satisfied", "input_explanation", "output_satisfied", "output_explanation"]
        for k in required_keys:
            if k not in out:
                raise AssertionError(f"VerifiableModel output missing key '{k}'")

        if not isinstance(out["input_satisfied"], bool):
            raise AssertionError("input_satisfied must be bool")
        if not isinstance(out["output_satisfied"], bool):
            raise AssertionError("output_satisfied must be bool")

        # For our feasible x, input_satisfied should be True (unless spec config is inconsistent)
        if out["input_satisfied"] is not True:
            raise AssertionError(
                f"Expected input_satisfied=True for feasible input, got False. explanation={out['input_explanation']}"
            )

        # Output spec may or may not hold (random weights), so we only log it.
        logger.info("input_ok=%s | output_ok=%s | y_shape=%s",
                    out["input_satisfied"], out["output_satisfied"], tuple(out["output"].shape))

    logger.info("✅ ConfigNet smoke test PASS.")


def main() -> int:
    p = argparse.ArgumentParser("ConfigNet smoke test")
    p.add_argument("--num", type=int, default=5, help="number of instances to sample")
    p.add_argument("--seed", type=int, default=0, help="base seed")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    args = p.parse_args()

    try:
        run_smoke(num=args.num, base_seed=args.seed, device=args.device, dtype_str=args.dtype)
        return 0
    except Exception as e:
        logger.error("❌ ConfigNet smoke test FAILED: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
