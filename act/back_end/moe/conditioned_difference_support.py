# ===- conditioned_difference_support.py - Retained path support -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""Segment an affine path condition and tighten a downstream support query.

This module deliberately does *not* segment or encode a sigmoid.  It partitions
the range of an affine router margin, retains each closed margin interval as two
linear constraints on the shared HZ factors, and recomputes a property-directed
expert-difference support bound in every segment.  The union of the closed
segments covers the original guarded domain, including all segment boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from act.back_end.moe.weighted_top2 import SharedInputPairHZ
from act.back_end.solver.solver_hz import (
    HZFeasibilityResult,
    HZSupportBoundsResult,
    SparseHZono,
    hz_check_feasibility,
    hz_support_bounds,
    sparse_hz_linear,
    sparse_pad_cols,
)


@dataclass(frozen=True)
class AffinePathSegment:
    """One closed interval of an affine retained path condition."""

    index: int
    lower: float
    upper: float

    def contains(self, value: float, *, tolerance: float = 0.0) -> bool:
        return bool(
            float(value) >= self.lower - float(tolerance)
            and float(value) <= self.upper + float(tolerance)
        )


@dataclass(frozen=True)
class ConditionedSegmentSupport:
    """Support result for one retained affine-margin segment."""

    segment: AffinePathSegment
    conditioned_target: SparseHZono = field(repr=False, compare=False)
    feasibility: HZFeasibilityResult
    raw_support: HZSupportBoundsResult | None = field(
        default=None, repr=False, compare=False
    )
    raw_bounds: tuple[float, float] | None = None
    tightened_bounds: tuple[float, float] | None = None
    fallback_reason: str | None = None
    constraint_rows_added: int = 2
    elapsed: float = 0.0

    @property
    def active(self) -> bool:
        """An unknown feasibility result must remain in the sound union."""
        return self.feasibility.status != "infeasible"


@dataclass(frozen=True)
class ConditionedSupportTelemetry:
    segmentation_axis: str
    gate_function_encoded: bool
    sigmoid_segments: int
    path_support_solves: int
    unconditional_support_solves: int
    feasibility_solves: int
    conditioned_support_solves: int
    feasible_segments: int
    infeasible_segments: int
    unknown_segments: int
    fallback_segments: int
    constraint_rows_added: int
    path_support_seconds: float
    unconditional_support_seconds: float
    feasibility_seconds: float
    conditioned_support_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class SegmentedConditionedSupport:
    """A sound interval union for a downstream affine expression."""

    path_bounds: tuple[float, float]
    unconditional_bounds: tuple[float, float]
    union_bounds: tuple[float, float] | None
    path_support: HZSupportBoundsResult = field(repr=False, compare=False)
    unconditional_support: HZSupportBoundsResult = field(
        repr=False, compare=False
    )
    segments: tuple[ConditionedSegmentSupport, ...]
    cut_points: tuple[float, ...]
    closed_boundary_overlap: bool
    coverage_complete: bool
    telemetry: ConditionedSupportTelemetry

    def segment_for_value(
        self, value: float, *, tolerance: float = 0.0
    ) -> tuple[ConditionedSegmentSupport, ...]:
        """Return every closed segment containing ``value`` (two at a cut)."""
        return tuple(
            result
            for result in self.segments
            if result.segment.contains(value, tolerance=tolerance)
        )


def _same_sparse(left: sp.csr_matrix, right: sp.csr_matrix) -> bool:
    if left.shape != right.shape:
        return False
    difference = (left - right).tocsr()
    difference.eliminate_zeros()
    return difference.nnz == 0


def _assert_retained_path_prefix(
    path_hz: SparseHZono,
    target_hz: SparseHZono,
) -> None:
    """Reject a frame-id match that does not retain the same path constraints."""
    if path_hz.frame_id is None or target_hz.frame_id != path_hz.frame_id:
        raise ValueError("path and target HZs must share a non-null frame identity")
    if target_hz.n_cont < path_hz.n_cont or target_hz.n_bin < path_hz.n_bin:
        raise ValueError("target HZ is narrower than the retained path frame")
    if target_hz.n_eq < path_hz.n_eq or target_hz.n_ineq < path_hz.n_ineq:
        raise ValueError("target HZ lost retained path constraints")

    target_eq_c = target_hz.Ac[: path_hz.n_eq, : path_hz.n_cont]
    target_eq_b = target_hz.Ab[: path_hz.n_eq, : path_hz.n_bin]
    target_le_c = target_hz.Auc[: path_hz.n_ineq, : path_hz.n_cont]
    target_le_b = target_hz.Aub[: path_hz.n_ineq, : path_hz.n_bin]
    private_eq_c = target_hz.Ac[: path_hz.n_eq, path_hz.n_cont :]
    private_eq_b = target_hz.Ab[: path_hz.n_eq, path_hz.n_bin :]
    private_le_c = target_hz.Auc[: path_hz.n_ineq, path_hz.n_cont :]
    private_le_b = target_hz.Aub[: path_hz.n_ineq, path_hz.n_bin :]
    if (
        not _same_sparse(target_eq_c, path_hz.Ac)
        or not _same_sparse(target_eq_b, path_hz.Ab)
        or not _same_sparse(target_le_c, path_hz.Auc)
        or not _same_sparse(target_le_b, path_hz.Aub)
        or private_eq_c.nnz
        or private_eq_b.nnz
        or private_le_c.nnz
        or private_le_b.nnz
        or not np.array_equal(target_hz.b[: path_hz.n_eq], path_hz.b)
        or not np.array_equal(target_hz.ub[: path_hz.n_ineq], path_hz.ub)
    ):
        raise ValueError("target HZ changed retained path constraint identity")


def _closed_segments(
    lower: float,
    upper: float,
    cut_points: Sequence[float],
) -> tuple[AffinePathSegment, ...]:
    lower, upper = float(lower), float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("path bounds must be finite and ordered")
    cuts = sorted(
        {
            float(value)
            for value in cut_points
            if np.isfinite(value) and lower < float(value) < upper
        }
    )
    edges = [lower, *cuts, upper]
    if len(edges) == 1:
        return (AffinePathSegment(0, lower, upper),)
    return tuple(
        AffinePathSegment(index, edges[index], edges[index + 1])
        for index in range(len(edges) - 1)
    )


def condition_on_affine_path_interval(
    target_hz: SparseHZono,
    path_scalar_hz: SparseHZono,
    lower: float,
    upper: float,
) -> SparseHZono:
    """Intersect ``target_hz`` with ``lower <= path_scalar <= upper``.

    The scalar and target must retain the same leading factors and path
    constraints.  The two weak inequalities make adjacent segments overlap at
    their boundary, which is required for tie-inclusive routing semantics.
    """
    if path_scalar_hz.n_out != 1:
        raise ValueError("retained path expression must be scalar")
    lower, upper = float(lower), float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("path interval must be finite and ordered")
    _assert_retained_path_prefix(path_scalar_hz, target_hz)

    path_gc = sparse_pad_cols(path_scalar_hz.Gc, target_hz.n_cont)
    path_gb = sparse_pad_cols(path_scalar_hz.Gb, target_hz.n_bin)
    extra_c = sp.vstack([path_gc, -path_gc], format="csr")
    extra_b = sp.vstack([path_gb, -path_gb], format="csr")
    center = float(path_scalar_hz.c[0])
    extra_rhs = np.asarray([upper - center, center - lower], dtype=np.float64)
    return SparseHZono(
        c=target_hz.c.copy(),
        Gc=target_hz.Gc.copy(),
        Gb=target_hz.Gb.copy(),
        Ac=target_hz.Ac.copy(),
        Ab=target_hz.Ab.copy(),
        b=target_hz.b.copy(),
        Auc=sp.vstack([target_hz.Auc, extra_c], format="csr"),
        Aub=sp.vstack([target_hz.Aub, extra_b], format="csr"),
        ub=np.concatenate([target_hz.ub, extra_rhs]),
        frame_id=target_hz.frame_id,
        exact=bool(target_hz.exact and path_scalar_hz.exact),
    )


def segmented_affine_conditioned_support(
    path_hz: SparseHZono,
    target_hz: SparseHZono,
    path_weights: Sequence[float],
    target_weights: Sequence[float],
    *,
    path_bias: float = 0.0,
    target_bias: float = 0.0,
    cut_points: Sequence[float] = (0.0,),
    path_time_limit: float,
    feasibility_time_limit: float,
    support_time_limit: float,
    relax_binaries: bool = True,
) -> SegmentedConditionedSupport:
    """Recompute downstream support under closed affine path segments.

    This is the reusable N1 mechanism.  It makes no assumption about a gate
    family and introduces no gate variable.  A segment proved infeasible is
    omitted; a segment with unknown feasibility remains in the union.  Failed
    support sides use the solver's sound generator fallback and are intersected
    with the unconditional bound, so every reported segment is monotonically no
    looser than the original support query.
    """
    started = time.monotonic()
    _assert_retained_path_prefix(path_hz, target_hz)
    path_w = np.asarray(path_weights, dtype=np.float64).reshape(1, -1)
    target_w = np.asarray(target_weights, dtype=np.float64).reshape(1, -1)
    if path_w.shape[1] != path_hz.n_out:
        raise ValueError("path weight width does not match path HZ output")
    if target_w.shape[1] != target_hz.n_out:
        raise ValueError("target weight width does not match target HZ output")

    path_scalar = sparse_hz_linear(path_hz, path_w, [float(path_bias)])
    target_scalar = sparse_hz_linear(target_hz, target_w, [float(target_bias)])
    path_support = hz_support_bounds(
        path_scalar,
        [0],
        time_limit=float(path_time_limit),
        relax_binaries=bool(relax_binaries),
    )
    unconditional = hz_support_bounds(
        target_scalar,
        [0],
        time_limit=float(support_time_limit),
        relax_binaries=bool(relax_binaries),
    )
    path_bounds = (
        float(path_support.bounds.lb.item()),
        float(path_support.bounds.ub.item()),
    )
    unconditional_bounds = (
        float(unconditional.bounds.lb.item()),
        float(unconditional.bounds.ub.item()),
    )
    segment_specs = _closed_segments(*path_bounds, cut_points)

    results: list[ConditionedSegmentSupport] = []
    feasibility_seconds = 0.0
    conditioned_seconds = 0.0
    conditioned_solves = 0
    fallbacks = 0
    for segment in segment_specs:
        segment_started = time.monotonic()
        conditioned = condition_on_affine_path_interval(
            target_scalar,
            path_scalar,
            segment.lower,
            segment.upper,
        )
        feasibility = hz_check_feasibility(
            conditioned,
            time_limit=float(feasibility_time_limit),
        )
        feasibility_seconds += feasibility.elapsed
        if feasibility.status == "infeasible":
            results.append(
                ConditionedSegmentSupport(
                    segment,
                    conditioned,
                    feasibility,
                    elapsed=time.monotonic() - segment_started,
                )
            )
            continue

        support = hz_support_bounds(
            conditioned,
            [0],
            time_limit=float(support_time_limit),
            relax_binaries=bool(relax_binaries),
        )
        conditioned_seconds += support.elapsed
        conditioned_solves += support.solves
        raw = (
            float(support.bounds.lb.item()),
            float(support.bounds.ub.item()),
        )
        # X_segment is a subset of X.  Intersecting two independently sound
        # outer intervals is sound and makes the support-monotonicity invariant
        # executable even when a segment side used the fast fallback.
        tightened = (
            max(unconditional_bounds[0], raw[0]),
            min(unconditional_bounds[1], raw[1]),
        )
        fallback_reason = None
        if tightened[0] > tightened[1]:
            # Do not silently trust inconsistent floating-point solver output.
            # The unconditional interval is the conservative sound fallback and
            # telemetry makes the event observable.
            tightened = unconditional_bounds
            fallback_reason = "numerical_support_intersection_conflict"
            fallbacks += 1
        elif (
            "fast_fallback" in support.lower_status
            or "fast_fallback" in support.upper_status
        ):
            fallback_reason = "support_side_fast_fallback"
            fallbacks += 1
        results.append(
            ConditionedSegmentSupport(
                segment,
                conditioned,
                feasibility,
                raw_support=support,
                raw_bounds=raw,
                tightened_bounds=tightened,
                fallback_reason=fallback_reason,
                elapsed=time.monotonic() - segment_started,
            )
        )

    active_bounds = [
        result.tightened_bounds
        for result in results
        if result.active and result.tightened_bounds is not None
    ]
    union_bounds = (
        (
            min(bounds[0] for bounds in active_bounds),
            max(bounds[1] for bounds in active_bounds),
        )
        if active_bounds
        else None
    )
    telemetry = ConditionedSupportTelemetry(
        segmentation_axis="affine_path_margin",
        gate_function_encoded=False,
        sigmoid_segments=0,
        path_support_solves=path_support.solves,
        unconditional_support_solves=unconditional.solves,
        feasibility_solves=len(segment_specs),
        conditioned_support_solves=conditioned_solves,
        feasible_segments=sum(
            result.feasibility.status == "feasible" for result in results
        ),
        infeasible_segments=sum(
            result.feasibility.status == "infeasible" for result in results
        ),
        unknown_segments=sum(
            result.feasibility.status == "unknown" for result in results
        ),
        fallback_segments=fallbacks,
        constraint_rows_added=2 * len(segment_specs),
        path_support_seconds=path_support.elapsed,
        unconditional_support_seconds=unconditional.elapsed,
        feasibility_seconds=feasibility_seconds,
        conditioned_support_seconds=conditioned_seconds,
        total_seconds=time.monotonic() - started,
    )
    return SegmentedConditionedSupport(
        path_bounds=path_bounds,
        unconditional_bounds=unconditional_bounds,
        union_bounds=union_bounds,
        path_support=path_support,
        unconditional_support=unconditional,
        segments=tuple(results),
        cut_points=tuple(
            segment.upper for segment in segment_specs[:-1]
        ),
        closed_boundary_overlap=True,
        coverage_complete=True,
        telemetry=telemetry,
    )


def conditioned_pair_difference_support(
    pair_hz: SharedInputPairHZ,
    conditioned_router: SparseHZono,
    pair: Sequence[int],
    property_row: Sequence[float],
    *,
    cut_points: Sequence[float] = (0.0,),
    margin_time_limit: float,
    feasibility_time_limit: float,
    difference_time_limit: float,
    relax_binaries: bool = True,
) -> SegmentedConditionedSupport:
    """N1 convenience wrapper for ``q^T(E_a-E_b)`` on a top-2 pair."""
    selected = tuple(sorted(int(value) for value in pair))
    if len(selected) != 2 or len(set(selected)) != 2:
        raise ValueError("conditioned pair support requires two distinct experts")
    if selected[0] < 0 or selected[1] >= conditioned_router.n_out:
        raise IndexError("expert pair is outside the router output")
    classes = len(pair_hz.a_rows)
    q = np.asarray(property_row, dtype=np.float64).reshape(-1)
    if q.size != classes or len(pair_hz.b_rows) != classes:
        raise ValueError("property row width does not match paired experts")

    margin_w = np.zeros(conditioned_router.n_out, dtype=np.float64)
    margin_w[selected[0]] = 1.0
    margin_w[selected[1]] = -1.0
    difference_w = np.zeros(pair_hz.output_hz.n_out, dtype=np.float64)
    difference_w[np.asarray(pair_hz.a_rows, dtype=np.int64)] = q
    difference_w[np.asarray(pair_hz.b_rows, dtype=np.int64)] = -q
    return segmented_affine_conditioned_support(
        conditioned_router,
        pair_hz.output_hz,
        margin_w,
        difference_w,
        cut_points=cut_points,
        path_time_limit=float(margin_time_limit),
        feasibility_time_limit=float(feasibility_time_limit),
        support_time_limit=float(difference_time_limit),
        relax_binaries=bool(relax_binaries),
    )
