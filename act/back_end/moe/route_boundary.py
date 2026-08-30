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
    witness_delta: tuple[float, ...] | None


@dataclass(frozen=True)
class AffineTop1RouteBoundary:
    """Minimum distance to any tie-inclusive alternative hard-top1 route."""

    clean_expert: int
    clean_scores: tuple[float, ...]
    radius: float
    radius_lower: float
    radius_upper: float
    boundary_competitor: int | None
    boundary_witness_delta: tuple[float, ...] | None
    competitors: tuple[AffineCompetitorBoundary, ...]
    box_constrained: bool


@dataclass(frozen=True)
class AffineTop1RouteBoundaryBatch:
    """Vectorized route-boundary result for a batch of affine-router inputs."""

    clean_experts: np.ndarray
    radii: np.ndarray
    radius_lowers: np.ndarray
    radius_uppers: np.ndarray
    boundary_competitors: np.ndarray
    witness_deltas: np.ndarray | None
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


def fold_bilinear_resize_input_map(
    weight: Sequence[Sequence[float]],
    bias: Sequence[float] | None,
    *,
    channels: int,
    input_size: tuple[int, int],
    output_size: tuple[int, int],
    align_corners: bool = False,
    antialias: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold a deterministic bilinear resize into an affine map.

    The implementation applies the exact adjoint of PyTorch's real-arithmetic
    bilinear resize to each affine output row.  It never materializes the dense
    resize matrix, which is prohibitive for mappings such as ``64x64 ->
    224x224``.  Fold any pointwise affine normalization after the resize with
    :func:`fold_affine_input_map` before calling this function.
    """

    import torch
    import torch.nn.functional as functional

    W = np.asarray(weight, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError("router weight must be a matrix")
    if channels <= 0:
        raise ValueError("resize channels must be positive")
    input_height, input_width = map(int, input_size)
    output_height, output_width = map(int, output_size)
    if min(input_height, input_width, output_height, output_width) <= 0:
        raise ValueError("resize dimensions must be positive")
    expected_output = channels * output_height * output_width
    if W.shape[1] != expected_output:
        raise ValueError("router width does not match resized output shape")
    b = (
        np.zeros(W.shape[0], dtype=np.float64)
        if bias is None
        else np.asarray(bias, dtype=np.float64).reshape(-1)
    )
    if b.size != W.shape[0]:
        raise ValueError("router bias width does not match router outputs")
    if not np.all(np.isfinite(W)) or not np.all(np.isfinite(b)):
        raise ValueError("resize folding requires finite affine parameters")

    source = torch.zeros(
        (1, channels, input_height, input_width),
        dtype=torch.float64,
        requires_grad=True,
    )
    resized = functional.interpolate(
        source,
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=bool(align_corners),
        antialias=bool(antialias),
    )
    row_weights = torch.as_tensor(W, dtype=torch.float64).reshape(
        W.shape[0], channels, output_height, output_width
    )
    folded_rows: list[np.ndarray] = []
    for row_index in range(W.shape[0]):
        gradient = torch.autograd.grad(
            resized,
            source,
            grad_outputs=row_weights[row_index][None],
            retain_graph=row_index + 1 < W.shape[0],
            create_graph=False,
        )[0]
        folded_rows.append(gradient.detach().numpy().reshape(-1).copy())
    return np.stack(folded_rows), b.copy()


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


def _outward_brackets(
    values: np.ndarray,
    absolute: float,
    relative: float,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    slack = float(absolute) + float(relative) * np.maximum(
        1.0,
        np.abs(values),
    )
    lower = values.copy()
    upper = values.copy()
    lower[finite] = np.maximum(
        0.0,
        np.nextafter(values[finite] - slack[finite], -np.inf),
    )
    upper[finite] = np.nextafter(values[finite] + slack[finite], np.inf)
    return lower, upper


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


def _minimum_capped_linf_batch(
    targets: np.ndarray,
    coefficients: np.ndarray,
    capacities: np.ndarray,
    *,
    compute_device: str | None = None,
) -> np.ndarray:
    """Vectorized finite-capacity counterpart of ``_minimum_capped_linf``."""
    target_values = np.asarray(targets, dtype=np.float64).reshape(-1)
    weights = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    caps = np.asarray(capacities, dtype=np.float64)
    if caps.ndim != 2 or caps.shape != (target_values.size, weights.size):
        raise ValueError("batch capacities do not match targets and coefficients")
    if np.any(~np.isfinite(caps)):
        raise ValueError("batch capped support requires finite input bounds")

    result = np.full(target_values.shape, np.inf, dtype=np.float64)
    result[target_values <= 0.0] = 0.0
    positive_rows = np.flatnonzero(target_values > 0.0)
    keep = weights > 0.0
    if not positive_rows.size or not np.any(keep):
        return result

    active_targets = target_values[positive_rows]
    active_caps = caps[positive_rows][:, keep]
    active_weights = weights[keep]
    if compute_device is not None:
        import torch

        device = torch.device(compute_device)
        target_tensor = torch.as_tensor(
            active_targets,
            dtype=torch.float64,
            device=device,
        )
        weight_tensor = torch.as_tensor(
            active_weights,
            dtype=torch.float64,
            device=device,
        )
        cap_tensor = torch.as_tensor(
            active_caps,
            dtype=torch.float64,
            device=device,
        )
        sorted_caps, order = torch.sort(cap_tensor, dim=1)
        sorted_weights = weight_tensor[order]
        cumulative_weights = torch.cumsum(sorted_weights, dim=1)
        cumulative_weighted_caps = torch.cumsum(
            sorted_weights * sorted_caps,
            dim=1,
        )
        total_weight = torch.sum(weight_tensor)
        support_at_breakpoints = cumulative_weighted_caps + sorted_caps * (
            total_weight - cumulative_weights
        )
        reaches = support_at_breakpoints[:, -1] >= target_tensor
        if not bool(torch.any(reaches)):
            return result
        reached_support = support_at_breakpoints[reaches]
        reached_targets = target_tensor[reaches]
        first = torch.argmax(
            (reached_support >= reached_targets[:, None]).to(torch.int8),
            dim=1,
        )
        row_index = torch.arange(first.numel(), device=device)
        has_previous = first > 0
        previous_indices = torch.clamp(first - 1, min=0)
        previous_radius = torch.where(
            has_previous,
            sorted_caps[reaches][row_index, previous_indices],
            torch.zeros_like(reached_targets),
        )
        previous_support = torch.where(
            has_previous,
            reached_support[row_index, previous_indices],
            torch.zeros_like(reached_targets),
        )
        previous_weight = torch.where(
            has_previous,
            cumulative_weights[reaches][row_index, previous_indices],
            torch.zeros_like(reached_targets),
        )
        radius = previous_radius + (reached_targets - previous_support) / (
            total_weight - previous_weight
        )
        reached_rows = torch.nonzero(reaches, as_tuple=False).flatten().cpu().numpy()
        result[positive_rows[reached_rows]] = radius.cpu().numpy()
        return result

    order = np.argsort(active_caps, axis=1)
    sorted_caps = np.take_along_axis(active_caps, order, axis=1)
    sorted_weights = active_weights[order]
    cumulative_weights = np.cumsum(sorted_weights, axis=1)
    cumulative_weighted_caps = np.cumsum(
        sorted_weights * sorted_caps,
        axis=1,
    )
    total_weight = float(np.sum(active_weights))
    support_at_breakpoints = cumulative_weighted_caps + sorted_caps * (
        total_weight - cumulative_weights
    )
    reaches = support_at_breakpoints[:, -1] >= active_targets
    if not np.any(reaches):
        return result

    reached_rows = np.flatnonzero(reaches)
    reached_support = support_at_breakpoints[reaches]
    reached_targets = active_targets[reaches]
    first = np.argmax(
        reached_support >= reached_targets[:, None],
        axis=1,
    )
    row_index = np.arange(first.size)
    has_previous = first > 0
    previous_radius = np.zeros(first.size, dtype=np.float64)
    previous_support = np.zeros(first.size, dtype=np.float64)
    previous_weight = np.zeros(first.size, dtype=np.float64)
    previous_indices = first[has_previous] - 1
    selected_rows = row_index[has_previous]
    previous_radius[has_previous] = sorted_caps[reaches][
        selected_rows,
        previous_indices,
    ]
    previous_support[has_previous] = reached_support[
        selected_rows,
        previous_indices,
    ]
    previous_weight[has_previous] = cumulative_weights[reaches][
        selected_rows,
        previous_indices,
    ]
    radius = previous_radius + (reached_targets - previous_support) / (
        total_weight - previous_weight
    )
    result[positive_rows[reached_rows]] = radius
    return result


def _route_witness_delta(
    difference: np.ndarray,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    radius_upper: float,
    *,
    box_constrained: bool,
) -> tuple[float, ...] | None:
    """Construct a tie-inclusive concrete witness inside the upper bracket."""
    if not isfinite(radius_upper):
        return None
    if box_constrained:
        capacities = np.where(
            difference > 0.0,
            point - lower,
            np.where(difference < 0.0, upper - point, 0.0),
        )
        magnitude = np.minimum(float(radius_upper), capacities)
    else:
        magnitude = np.full(point.shape, float(radius_upper), dtype=np.float64)
    delta = -np.sign(difference) * magnitude
    return tuple(float(value) for value in delta)


def _minimum_grid_linf_torch(targets, weights, cap_indices, grid_steps: int):
    """Invert finite weighted grid support without a per-row breakpoint sort."""
    import torch

    result = torch.full_like(targets, torch.inf)
    result[targets <= 0.0] = 0.0
    positive = torch.nonzero(targets > 0.0, as_tuple=False).flatten()
    keep = weights > 0.0
    if not positive.numel() or not bool(torch.any(keep)):
        return result
    active_targets = targets[positive]
    active_weights = weights[keep]
    active_indices = cap_indices[positive][:, keep]
    histogram = torch.zeros(
        (active_targets.numel(), grid_steps + 1),
        dtype=torch.float64,
        device=targets.device,
    )
    histogram.scatter_add_(
        1,
        active_indices,
        active_weights.expand(active_targets.numel(), -1),
    )
    levels = torch.arange(
        grid_steps + 1,
        dtype=torch.float64,
        device=targets.device,
    ) / grid_steps
    cumulative_weights = torch.cumsum(histogram, dim=1)
    cumulative_weighted_caps = torch.cumsum(
        histogram * levels[None, :],
        dim=1,
    )
    total_weight = torch.sum(active_weights)
    support = cumulative_weighted_caps + levels[None, :] * (
        total_weight - cumulative_weights
    )
    reaches = support[:, -1] >= active_targets
    if not bool(torch.any(reaches)):
        return result
    reached_support = support[reaches]
    reached_targets = active_targets[reaches]
    first = torch.argmax(
        (reached_support >= reached_targets[:, None]).to(torch.int8),
        dim=1,
    )
    rows = torch.arange(first.numel(), device=targets.device)
    has_previous = first > 0
    previous_indices = torch.clamp(first - 1, min=0)
    previous_radius = torch.where(
        has_previous,
        levels[previous_indices],
        torch.zeros_like(reached_targets),
    )
    previous_support = torch.where(
        has_previous,
        reached_support[rows, previous_indices],
        torch.zeros_like(reached_targets),
    )
    reached_cumulative_weights = cumulative_weights[reaches]
    previous_weight = torch.where(
        has_previous,
        reached_cumulative_weights[rows, previous_indices],
        torch.zeros_like(reached_targets),
    )
    radius = previous_radius + (reached_targets - previous_support) / (
        total_weight - previous_weight
    )
    reached_rows = torch.nonzero(reaches, as_tuple=False).flatten()
    result[positive[reached_rows]] = radius
    return result


def _affine_top1_route_boundary_batch_grid_device(
    W: np.ndarray,
    b: np.ndarray,
    X: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    outward_absolute: float,
    outward_relative: float,
    include_witnesses: bool,
    compute_device: str,
    capacity_grid_steps: int,
) -> AffineTop1RouteBoundaryBatch:
    """Keep a quantized batch resident on one device for all expert pairs."""
    import torch

    grid_steps = int(capacity_grid_steps)
    if grid_steps <= 0:
        raise ValueError("capacity_grid_steps must be positive")
    device = torch.device(compute_device)
    point_tensor = torch.as_tensor(X, dtype=torch.float64, device=device)
    lower_tensor = torch.tensor(lower, dtype=torch.float64, device=device)
    upper_tensor = torch.tensor(upper, dtype=torch.float64, device=device)
    weight_tensor = torch.as_tensor(W, dtype=torch.float64, device=device)
    bias_tensor = torch.as_tensor(b, dtype=torch.float64, device=device)

    lower_scaled = (point_tensor - lower_tensor) * grid_steps
    upper_scaled = (upper_tensor - point_tensor) * grid_steps
    lower_indices = torch.round(lower_scaled)
    upper_indices = torch.round(upper_scaled)
    invalid_grid = (
        torch.any(lower_indices < 0.0)
        | torch.any(upper_indices < 0.0)
        | torch.any(lower_indices > grid_steps)
        | torch.any(upper_indices > grid_steps)
        | torch.any(torch.abs(lower_scaled - lower_indices) > 1e-6)
        | torch.any(torch.abs(upper_scaled - upper_indices) > 1e-6)
    )
    if bool(invalid_grid):
        raise ValueError("capacities do not lie on the declared finite grid")
    lower_indices = lower_indices.to(torch.int64)
    upper_indices = upper_indices.to(torch.int64)

    scores = point_tensor @ weight_tensor.T + bias_tensor
    clean = torch.argmax(scores, dim=1)
    radii = torch.full(
        (X.shape[0],),
        torch.inf,
        dtype=torch.float64,
        device=device,
    )
    competitors = torch.full(
        (X.shape[0],),
        -1,
        dtype=torch.int64,
        device=device,
    )
    for clean_expert in range(W.shape[0]):
        clean_rows = torch.nonzero(
            clean == clean_expert,
            as_tuple=False,
        ).flatten()
        if not clean_rows.numel():
            continue
        for competitor in range(W.shape[0]):
            if competitor == clean_expert:
                continue
            difference = weight_tensor[clean_expert] - weight_tensor[competitor]
            cap_indices = torch.where(
                difference > 0.0,
                lower_indices[clean_rows],
                torch.where(
                    difference < 0.0,
                    upper_indices[clean_rows],
                    torch.zeros_like(lower_indices[clean_rows]),
                ),
            )
            margin = (
                scores[clean_rows, clean_expert]
                - scores[clean_rows, competitor]
            )
            candidate_radii = _minimum_grid_linf_torch(
                margin,
                torch.abs(difference),
                cap_indices,
                grid_steps,
            )
            current = radii[clean_rows]
            current_competitor = competitors[clean_rows]
            improve = (candidate_radii < current) | (
                (candidate_radii == current)
                & ((current_competitor < 0) | (competitor < current_competitor))
            )
            selected_rows = clean_rows[improve]
            radii[selected_rows] = candidate_radii[improve]
            competitors[selected_rows] = competitor

    clean_array = clean.cpu().numpy()
    radii_array = radii.cpu().numpy()
    competitor_array = competitors.cpu().numpy()
    radius_lowers, radius_uppers = _outward_brackets(
        radii_array,
        float(outward_absolute),
        float(outward_relative),
    )
    witness_deltas = None
    if include_witnesses:
        witness_deltas = np.full(X.shape, np.nan, dtype=np.float64)
        for clean_expert in range(W.shape[0]):
            for competitor in range(W.shape[0]):
                rows = np.flatnonzero(
                    (clean_array == clean_expert)
                    & (competitor_array == competitor)
                    & np.isfinite(radius_uppers)
                )
                if not rows.size:
                    continue
                difference = W[clean_expert] - W[competitor]
                capacities = np.where(
                    difference > 0.0,
                    X[rows] - lower[rows],
                    np.where(
                        difference < 0.0,
                        upper[rows] - X[rows],
                        0.0,
                    ),
                )
                magnitude = np.minimum(radius_uppers[rows, None], capacities)
                witness_deltas[rows] = -np.sign(difference)[None, :] * magnitude

    for array in (
        clean_array,
        radii_array,
        radius_lowers,
        radius_uppers,
        competitor_array,
        witness_deltas,
    ):
        if array is not None:
            array.setflags(write=False)
    return AffineTop1RouteBoundaryBatch(
        clean_experts=clean_array,
        radii=radii_array,
        radius_lowers=radius_lowers,
        radius_uppers=radius_uppers,
        boundary_competitors=competitor_array,
        witness_deltas=witness_deltas,
        box_constrained=True,
    )


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
        witness_delta = _route_witness_delta(
            difference,
            x,
            lower,
            upper,
            radius_upper,
            box_constrained=box_constrained,
        )
        rows.append(
            AffineCompetitorBoundary(
                competitor=competitor,
                clean_margin=margin,
                radius=radius,
                radius_lower=radius_lower,
                radius_upper=radius_upper,
                reachable=isfinite(radius),
                witness_delta=witness_delta,
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
        boundary_witness_delta=best.witness_delta,
        competitors=tuple(rows),
        box_constrained=box_constrained,
    )


def affine_top1_route_boundary_batch(
    weight: Sequence[Sequence[float]],
    bias: Sequence[float] | None,
    points: Sequence[Sequence[float]],
    *,
    input_lower: Sequence[float] | float | None = None,
    input_upper: Sequence[float] | float | None = None,
    outward_absolute: float = 1e-9,
    outward_relative: float = 1e-9,
    include_witnesses: bool = False,
    compute_device: str | None = None,
    capacity_grid_steps: int | None = None,
) -> AffineTop1RouteBoundaryBatch:
    """Vectorize exact affine top-1 boundaries over a batch.

    The finite-box path groups points by clean expert and competitor, then uses
    one ``sort + cumsum`` breakpoint inversion per group. It never calls the
    scalar oracle per point. Set ``compute_device`` explicitly (for example,
    ``"cuda"``) to run those two primitives through PyTorch; ``None`` retains
    the pure NumPy path. Unbounded operation is supported only when neither
    input bound is supplied; a supplied box must be finite. Quantized pixel
    inputs may declare ``capacity_grid_steps=255``; every capacity is validated
    before a weighted histogram replaces the general sort.
    """
    W = np.asarray(weight, dtype=np.float64)
    X = np.asarray(points, dtype=np.float64)
    if W.ndim != 2 or X.ndim != 2 or W.shape[1] != X.shape[1] or W.shape[0] < 2:
        raise ValueError("router weight, points, or expert count is inconsistent")
    b = (
        np.zeros(W.shape[0], dtype=np.float64)
        if bias is None
        else np.asarray(bias, dtype=np.float64).reshape(-1)
    )
    if b.size != W.shape[0]:
        raise ValueError("router bias width does not match router outputs")
    if not np.all(np.isfinite(W)) or not np.all(np.isfinite(b)) or not np.all(
        np.isfinite(X)
    ):
        raise ValueError("affine route oracle requires finite inputs")
    if outward_absolute < 0.0 or outward_relative < 0.0:
        raise ValueError("outward rounding tolerances must be nonnegative")

    box_constrained = input_lower is not None or input_upper is not None
    if box_constrained and (input_lower is None or input_upper is None):
        raise ValueError("batch route oracle requires both finite input bounds")
    lower = np.broadcast_to(
        -np.inf if input_lower is None else np.asarray(input_lower, dtype=np.float64),
        X.shape,
    )
    upper = np.broadcast_to(
        np.inf if input_upper is None else np.asarray(input_upper, dtype=np.float64),
        X.shape,
    )
    if box_constrained and (
        np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper))
    ):
        raise ValueError("batch route oracle requires finite input bounds")
    if np.any(lower > upper) or np.any(X < lower) or np.any(X > upper):
        raise ValueError("every point must lie inside the input box")
    if capacity_grid_steps is not None and (
        not box_constrained or compute_device is None
    ):
        raise ValueError(
            "capacity grid acceleration requires a finite box and compute device"
        )
    if (
        box_constrained
        and compute_device is not None
        and capacity_grid_steps is not None
    ):
        return _affine_top1_route_boundary_batch_grid_device(
            W,
            b,
            X,
            lower,
            upper,
            outward_absolute=float(outward_absolute),
            outward_relative=float(outward_relative),
            include_witnesses=include_witnesses,
            compute_device=compute_device,
            capacity_grid_steps=int(capacity_grid_steps),
        )

    scores = X @ W.T + b
    clean = np.argmax(scores, axis=1).astype(np.int64)
    radii = np.full(X.shape[0], np.inf, dtype=np.float64)
    competitors = np.full(X.shape[0], -1, dtype=np.int64)
    for clean_expert in range(W.shape[0]):
        clean_rows = np.flatnonzero(clean == clean_expert)
        if not clean_rows.size:
            continue
        for competitor in range(W.shape[0]):
            if competitor == clean_expert:
                continue
            difference = W[clean_expert] - W[competitor]
            margin = scores[clean_rows, clean_expert] - scores[clean_rows, competitor]
            if box_constrained:
                capacities = np.where(
                    difference > 0.0,
                    X[clean_rows] - lower[clean_rows],
                    np.where(
                        difference < 0.0,
                        upper[clean_rows] - X[clean_rows],
                        0.0,
                    ),
                )
                candidate_radii = _minimum_capped_linf_batch(
                    margin,
                    np.abs(difference),
                    capacities,
                    compute_device=compute_device,
                )
            else:
                denominator = float(np.sum(np.abs(difference)))
                candidate_radii = np.where(
                    margin <= 0.0,
                    0.0,
                    margin / denominator if denominator > 0.0 else np.inf,
                )
            current = radii[clean_rows]
            current_competitor = competitors[clean_rows]
            improve = (candidate_radii < current) | (
                (candidate_radii == current)
                & ((current_competitor < 0) | (competitor < current_competitor))
            )
            selected_rows = clean_rows[improve]
            radii[selected_rows] = candidate_radii[improve]
            competitors[selected_rows] = competitor

    radius_lowers, radius_uppers = _outward_brackets(
        radii,
        float(outward_absolute),
        float(outward_relative),
    )
    witness_deltas = None
    if include_witnesses:
        witness_deltas = np.full(X.shape, np.nan, dtype=np.float64)
        for clean_expert in range(W.shape[0]):
            for competitor in range(W.shape[0]):
                rows = np.flatnonzero(
                    (clean == clean_expert)
                    & (competitors == competitor)
                    & np.isfinite(radius_uppers)
                )
                if not rows.size:
                    continue
                difference = W[clean_expert] - W[competitor]
                if box_constrained:
                    capacities = np.where(
                        difference > 0.0,
                        X[rows] - lower[rows],
                        np.where(
                            difference < 0.0,
                            upper[rows] - X[rows],
                            0.0,
                        ),
                    )
                    magnitude = np.minimum(radius_uppers[rows, None], capacities)
                else:
                    magnitude = np.broadcast_to(
                        radius_uppers[rows, None],
                        (rows.size, X.shape[1]),
                    )
                witness_deltas[rows] = -np.sign(difference)[None, :] * magnitude

    for array in (
        clean,
        radii,
        radius_lowers,
        radius_uppers,
        competitors,
        witness_deltas,
    ):
        if array is not None:
            array.setflags(write=False)
    return AffineTop1RouteBoundaryBatch(
        clean_experts=clean,
        radii=radii,
        radius_lowers=radius_lowers,
        radius_uppers=radius_uppers,
        boundary_competitors=competitors,
        witness_deltas=witness_deltas,
        box_constrained=box_constrained,
    )
