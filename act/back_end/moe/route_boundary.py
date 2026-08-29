# ===- act/back_end/moe/route_boundary.py - Affine Route Oracle ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""Exact top-1 route-boundary geometry for an affine router."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AffineCompetitorBoundary:
    competitor: int
    clean_margin: float
    radius: float
    radius_lower: float
    radius_upper: float
    reachable: bool


@dataclass(frozen=True)
class AffineTop1RouteBoundary:
    """Minimum distance to any tie-inclusive alternative hard-top1 route."""

    clean_expert: int
    clean_scores: tuple[float, ...]
    radius: float
    radius_lower: float
    radius_upper: float
    boundary_competitor: int | None
    competitors: tuple[AffineCompetitorBoundary, ...]
    box_constrained: bool


def fold_affine_input_map(
    weight: Sequence[Sequence[float]],
    bias: Sequence[float] | None,
    scale: Sequence[float] | float,
    shift: Sequence[float] | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold ``z = scale*x + shift`` into affine router scores ``W*z+b``."""
    W = np.asarray(weight, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError("router weight must be a matrix")
    b = (
        np.zeros(W.shape[0], dtype=np.float64)
        if bias is None
        else np.asarray(bias, dtype=np.float64).reshape(-1)
    )
    if b.size != W.shape[0]:
        raise ValueError("router bias width does not match router outputs")
    input_scale = np.broadcast_to(
        np.asarray(scale, dtype=np.float64), (W.shape[1],)
    )
    input_shift = np.broadcast_to(
        np.asarray(shift, dtype=np.float64), (W.shape[1],)
    )
    return W * input_scale[None, :], b + W @ input_shift


def _outward_bracket(
    value: float,
    absolute: float,
    relative: float,
) -> tuple[float, float]:
    if not isfinite(value):
        return value, value
    slack = float(absolute) + float(relative) * max(1.0, abs(value))
    return (
        max(0.0, float(np.nextafter(value - slack, -np.inf))),
        float(np.nextafter(value + slack, np.inf)),
    )


def _minimum_capped_linf(
    target: float,
    coefficients: np.ndarray,
    capacities: np.ndarray,
) -> float:
    """Solve ``sum a_i min(radius, cap_i) >= target`` exactly by breakpoints."""
    if target <= 0.0:
        return 0.0
    keep = (coefficients > 0.0) & (capacities > 0.0)
    weights = coefficients[keep]
    caps = capacities[keep]
    if not weights.size:
        return inf
    total = float(np.sum(weights * caps))
    if isfinite(total) and total < target:
        return inf

    finite_caps = np.unique(caps[np.isfinite(caps)])
    active = float(np.sum(weights))
    accumulated = 0.0
    previous = 0.0
    for endpoint in finite_caps:
        endpoint = float(endpoint)
        available = active * (endpoint - previous)
        if accumulated + available >= target:
            return previous + (target - accumulated) / active
        accumulated += available
        active -= float(np.sum(weights[caps == endpoint]))
        previous = endpoint
    if active > 0.0:
        return previous + (target - accumulated) / active
    return inf


def affine_top1_route_boundary(
    weight: Sequence[Sequence[float]],
    bias: Sequence[float] | None,
    point: Sequence[float],
    *,
    input_lower: Sequence[float] | float | None = None,
    input_upper: Sequence[float] | float | None = None,
    outward_absolute: float = 1e-9,
    outward_relative: float = 1e-9,
) -> AffineTop1RouteBoundary:
    """Return the minimum L-infinity radius at which another route is legal.

    Ties follow ``ANY_LEGAL_TOPK`` semantics. With no input box, each competitor
    has radius ``margin / ||w_i-w_j||_1``. With a box, the same support function
    is clipped coordinate-wise and inverted exactly at its breakpoints.

    Returned lower/upper values apply an absolute-plus-relative outward slack
    and directed floating-point rounding. The defaults match the frozen MoE
    numerical policy.
    """
    W = np.asarray(weight, dtype=np.float64)
    x = np.asarray(point, dtype=np.float64).reshape(-1)
    if W.ndim != 2 or W.shape[1] != x.size or W.shape[0] < 2:
        raise ValueError("router weight, point, or expert count is inconsistent")
    b = (
        np.zeros(W.shape[0], dtype=np.float64)
        if bias is None
        else np.asarray(bias, dtype=np.float64).reshape(-1)
    )
    if b.size != W.shape[0]:
        raise ValueError("router bias width does not match router outputs")
    if not np.all(np.isfinite(W)) or not np.all(np.isfinite(b)) or not np.all(
        np.isfinite(x)
    ):
        raise ValueError("affine route oracle requires finite inputs")
    if outward_absolute < 0.0 or outward_relative < 0.0:
        raise ValueError("outward rounding tolerances must be nonnegative")

    box_constrained = input_lower is not None or input_upper is not None
    lower = np.broadcast_to(
        -np.inf if input_lower is None else np.asarray(input_lower, dtype=np.float64),
        x.shape,
    )
    upper = np.broadcast_to(
        np.inf if input_upper is None else np.asarray(input_upper, dtype=np.float64),
        x.shape,
    )
    if np.any(lower > upper) or np.any(x < lower) or np.any(x > upper):
        raise ValueError("point must lie inside the input box")

    scores = W @ x + b
    clean = int(np.argmax(scores))
    rows: list[AffineCompetitorBoundary] = []
    for competitor in range(W.shape[0]):
        if competitor == clean:
            continue
        margin = float(scores[clean] - scores[competitor])
        difference = W[clean] - W[competitor]
        coefficients = np.abs(difference)
        if margin <= 0.0:
            radius = 0.0
        elif not box_constrained:
            denominator = float(np.sum(coefficients))
            radius = margin / denominator if denominator > 0.0 else inf
        else:
            capacities = np.where(
                difference > 0.0,
                x - lower,
                np.where(difference < 0.0, upper - x, 0.0),
            )
            radius = _minimum_capped_linf(margin, coefficients, capacities)
        radius_lower, radius_upper = _outward_bracket(
            radius,
            float(outward_absolute),
            float(outward_relative),
        )
        rows.append(
            AffineCompetitorBoundary(
                competitor=competitor,
                clean_margin=margin,
                radius=radius,
                radius_lower=radius_lower,
                radius_upper=radius_upper,
                reachable=isfinite(radius),
            )
        )

    best = min(rows, key=lambda row: (row.radius, row.competitor))
    return AffineTop1RouteBoundary(
        clean_expert=clean,
        clean_scores=tuple(float(value) for value in scores),
        radius=best.radius,
        radius_lower=best.radius_lower,
        radius_upper=best.radius_upper,
        boundary_competitor=best.competitor if best.reachable else None,
        competitors=tuple(rows),
        box_constrained=box_constrained,
    )
