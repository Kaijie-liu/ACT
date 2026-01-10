#!/usr/bin/env python3
#===- act/pipeline/confignet/input_sampling.py - Feasible Input Sampler --====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Sample concrete inputs that satisfy the InputSpecConfig of an InstanceSpec.
#   This module is pipeline-side only and does NOT import back_end.
#
# Supported kinds:
#   - InKind.BOX
#   - InKind.LINF_BALL
#   - InKind.LIN_POLY (rejection sampling)
#
# Determinism:
#   Uses a provided seed to drive a CPU torch.Generator, then moves tensors to
#   target device. This is stable across runs and devices.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch

from act.front_end.specs import InKind

from .schema import InstanceSpec

logger = logging.getLogger(__name__)


def _resolve_input_shape(instance_spec: InstanceSpec) -> Tuple[int, ...]:
    model_cfg = instance_spec.model_cfg
    if not hasattr(model_cfg, "input_shape"):
        raise ValueError("instance_spec.model_cfg missing input_shape")
    shape = tuple(int(x) for x in getattr(model_cfg, "input_shape"))
    if len(shape) < 2:
        raise ValueError(f"input_shape must include batch dim, got {shape}")
    if shape[0] != 1:
        raise ValueError(f"ConfigNet assumes batch=1, got input_shape={shape}")
    return shape


def _prod(shape: Tuple[int, ...]) -> int:
    p = 1
    for s in shape:
        p *= int(s)
    return p


def _derive_lin_poly_from_box(
    in_cfg,
    shape: Tuple[int, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = _prod(shape[1:])
    I = torch.eye(n, device="cpu", dtype=torch.float64)
    A = torch.cat([I, -I], dim=0)
    if in_cfg.lb is not None and in_cfg.ub is not None:
        lb = in_cfg.lb.reshape(-1).to(device="cpu", dtype=torch.float64)
        ub = in_cfg.ub.reshape(-1).to(device="cpu", dtype=torch.float64)
        b = torch.cat([ub, -lb], dim=0)
    else:
        lb_val = float(in_cfg.lb_val if in_cfg.lb_val is not None else in_cfg.value_range[0])
        ub_val = float(in_cfg.ub_val if in_cfg.ub_val is not None else in_cfg.value_range[1])
        b = torch.cat(
            [
                torch.full((n,), ub_val, device="cpu", dtype=torch.float64),
                torch.full((n,), -lb_val, device="cpu", dtype=torch.float64),
            ],
            dim=0,
        )
    return A, b


def _get_lin_poly_ab(
    in_cfg,
    shape: Tuple[int, ...],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if in_cfg.A is not None and in_cfg.b is not None:
        A = in_cfg.A.to(device="cpu", dtype=torch.float64)
        b = in_cfg.b.to(device="cpu", dtype=torch.float64)
        return A, b
    if getattr(in_cfg, "derive_poly_from_box", False):
        return _derive_lin_poly_from_box(in_cfg, shape)
    return None


def make_feasible_input(
    instance_spec: InstanceSpec,
    *,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Construct a single deterministic feasible input (no randomness).
    Useful for smoke tests.

    Strategy:
      - BOX: midpoint
      - LINF_BALL: center
      - LIN_POLY: if derive_poly_from_box then midpoint; else zeros
    """
    dev = torch.device(device)
    shape = _resolve_input_shape(instance_spec)
    in_cfg = instance_spec.input_spec

    if in_cfg.kind == InKind.BOX:
        lb_val = float(in_cfg.lb_val if in_cfg.lb_val is not None else in_cfg.value_range[0])
        ub_val = float(in_cfg.ub_val if in_cfg.ub_val is not None else in_cfg.value_range[1])
        if ub_val < lb_val:
            lb_val, ub_val = ub_val, lb_val
        return torch.full(shape, (lb_val + ub_val) * 0.5, device=dev, dtype=dtype)

    if in_cfg.kind == InKind.LINF_BALL:
        if in_cfg.center is not None:
            c = in_cfg.center.reshape(shape).to(device=dev, dtype=dtype)
            return c
        center_val = float(in_cfg.center_val if in_cfg.center_val is not None else sum(in_cfg.value_range) / 2.0)
        return torch.full(shape, center_val, device=dev, dtype=dtype)

    if in_cfg.kind == InKind.LIN_POLY:
        if getattr(in_cfg, "derive_poly_from_box", False):
            lb_val = float(in_cfg.lb_val if in_cfg.lb_val is not None else in_cfg.value_range[0])
            ub_val = float(in_cfg.ub_val if in_cfg.ub_val is not None else in_cfg.value_range[1])
            if ub_val < lb_val:
                lb_val, ub_val = ub_val, lb_val
            return torch.full(shape, (lb_val + ub_val) * 0.5, device=dev, dtype=dtype)
        return torch.zeros(shape, device=dev, dtype=dtype)

    raise ValueError(f"Unsupported input spec kind: {in_cfg.kind}")


def sample_feasible_inputs(
    instance_spec: InstanceSpec,
    *,
    num_samples: int,
    seed: int,
    device: str,
    dtype: torch.dtype,
    strict_input: bool = True,
    max_tries: int = 200,
    return_status: bool = False,
) -> List[torch.Tensor]:
    """
    Sample a list of concrete inputs that satisfy InputSpecConfig.

    Args:
        instance_spec: instance specification
        num_samples: number of samples
        seed: RNG seed for sampling
        device: target torch device string
        dtype: torch dtype

    Returns:
        List[Tensor], each has shape instance_spec.model_cfg.input_shape (batch=1)

    Notes:
      - Sampling happens on CPU with a torch.Generator for determinism, then moved.
      - For LIN_POLY, uses rejection sampling within a box.
        If sampling fails, strict_input controls whether to raise or fallback.
    """
    if num_samples < 0:
        raise ValueError(f"num_samples must be >= 0, got {num_samples}")

    shape = _resolve_input_shape(instance_spec)
    in_cfg = instance_spec.input_spec
    dev = torch.device(device)

    # Deterministic CPU generator
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    xs: List[torch.Tensor] = []
    status = {
        "input_sampling_status": "ok",
        "tries_total": 0,
        "accepted": 0,
        "rejection_rate": 0.0,
    }

    if num_samples == 0:
        return (xs, status) if return_status else xs

    # Helper to create base tensors on CPU then move.
    def _move(x_cpu: torch.Tensor) -> torch.Tensor:
        return x_cpu.to(device=dev, dtype=dtype)

    if in_cfg.kind == InKind.BOX:
        # Prefer explicit lb/ub tensor if given, else use scalar bounds.
        if in_cfg.lb is not None and in_cfg.ub is not None:
            lb = in_cfg.lb.reshape(shape).to(device="cpu", dtype=torch.float64)
            ub = in_cfg.ub.reshape(shape).to(device="cpu", dtype=torch.float64)
        else:
            lb_val = float(in_cfg.lb_val if in_cfg.lb_val is not None else in_cfg.value_range[0])
            ub_val = float(in_cfg.ub_val if in_cfg.ub_val is not None else in_cfg.value_range[1])
            if ub_val < lb_val:
                lb_val, ub_val = ub_val, lb_val
            lb = torch.full(shape, lb_val, device="cpu", dtype=torch.float64)
            ub = torch.full(shape, ub_val, device="cpu", dtype=torch.float64)

        # Uniform in [lb, ub]
        for _ in range(num_samples):
            u = torch.rand(shape, generator=g, device="cpu", dtype=torch.float64)
            x = lb + (ub - lb) * u
            xs.append(_move(x))

        status["tries_total"] = int(num_samples)
        status["accepted"] = int(num_samples)
        status["rejection_rate"] = 0.0
        return (xs, status) if return_status else xs

    if in_cfg.kind == InKind.LINF_BALL:
        eps = float(in_cfg.eps if in_cfg.eps is not None else 0.0)
        if eps < 0:
            raise ValueError(f"LINF_BALL eps must be >=0, got {eps}")

        if in_cfg.center is not None:
            center = in_cfg.center.reshape(shape).to(device="cpu", dtype=torch.float64)
        else:
            center_val = float(in_cfg.center_val if in_cfg.center_val is not None else sum(in_cfg.value_range) / 2.0)
            center = torch.full(shape, center_val, device="cpu", dtype=torch.float64)

        # u ~ Uniform([-eps, eps]) elementwise; satisfies ||x-center||_inf <= eps
        for _ in range(num_samples):
            u = torch.rand(shape, generator=g, device="cpu", dtype=torch.float64) * 2.0 - 1.0
            x = center + eps * u
            xs.append(_move(x))

        status["tries_total"] = int(num_samples)
        status["accepted"] = int(num_samples)
        status["rejection_rate"] = 0.0
        return (xs, status) if return_status else xs

    if in_cfg.kind == InKind.LIN_POLY:
        ab = _get_lin_poly_ab(in_cfg, shape)
        if ab is None:
            msg = (
                "LIN_POLY sampling requires A/b or derive_poly_from_box=True. "
                f"instance_id={instance_spec.instance_id} value_range={in_cfg.value_range} "
                f"max_tries={max_tries} A_present={in_cfg.A is not None} b_present={in_cfg.b is not None}"
            )
            if strict_input:
                logger.error(msg)
                raise ValueError(msg)
            logger.warning("%s; falling back to deterministic feasible input.", msg)
            for _ in range(num_samples):
                xs.append(make_feasible_input(instance_spec, device=device, dtype=dtype))
            status["input_sampling_status"] = "fallback"
            status["tries_total"] = 0
            status["accepted"] = int(num_samples)
            status["rejection_rate"] = 1.0
            return (xs, status) if return_status else xs

        A, b = ab

        if in_cfg.lb is not None and in_cfg.ub is not None:
            lb = in_cfg.lb.reshape(shape).to(device="cpu", dtype=torch.float64)
            ub = in_cfg.ub.reshape(shape).to(device="cpu", dtype=torch.float64)
        else:
            lb_val = float(in_cfg.lb_val if in_cfg.lb_val is not None else in_cfg.value_range[0])
            ub_val = float(in_cfg.ub_val if in_cfg.ub_val is not None else in_cfg.value_range[1])
            if ub_val < lb_val:
                lb_val, ub_val = ub_val, lb_val
            lb = torch.full(shape, lb_val, device="cpu", dtype=torch.float64)
            ub = torch.full(shape, ub_val, device="cpu", dtype=torch.float64)

        def _satisfies(x_cpu: torch.Tensor) -> bool:
            x_flat = x_cpu.reshape(-1)
            return bool(torch.all(A.matmul(x_flat) <= b + 1e-8))

        tries_total = 0
        accepted = 0
        for _ in range(num_samples):
            found = False
            for _attempt in range(int(max_tries)):
                tries_total += 1
                u = torch.rand(shape, generator=g, device="cpu", dtype=torch.float64)
                cand = lb + (ub - lb) * u
                if _satisfies(cand):
                    xs.append(_move(cand))
                    found = True
                    accepted += 1
                    break

            if not found:
                msg = (
                    f"LIN_POLY rejection sampling failed after {max_tries} tries "
                    f"(instance_id={instance_spec.instance_id} A_shape={tuple(A.shape)} "
                    f"b_shape={tuple(b.shape)} value_range={in_cfg.value_range} max_tries={max_tries})"
                )
                if strict_input:
                    logger.error(msg)
                    raise ValueError(msg)
                logger.warning("%s; falling back to deterministic feasible input.", msg)
                xs.append(make_feasible_input(instance_spec, device=device, dtype=dtype))
                status["input_sampling_status"] = "fallback"

        status["tries_total"] = int(tries_total)
        status["accepted"] = int(accepted)
        if tries_total > 0:
            status["rejection_rate"] = float(1.0 - (accepted / tries_total))
        return (xs, status) if return_status else xs

    raise ValueError(f"Unsupported input spec kind: {in_cfg.kind}")
