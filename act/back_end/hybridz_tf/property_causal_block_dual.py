#!/usr/bin/env python3
# ===- property_causal_block_dual.py - causal block dual candidates --===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------===#
"""Proof-neutral property-conditioned causal block-dual candidates.

The primitive minimizes the original-frame Lagrangian support

``support_[rl,ru](d) + support_[lb,ub](q - A.T @ d)``

from an explicit full-dual warm start.  Version one optimizes only upper-bound
rows, whose maximization-certificate multipliers satisfy ``d >= 0``.  Rows not
named by an enabled semantic block remain fixed at their warm-start values.

Selection is intentionally out of scope.  The caller supplies complete
semantic blocks and explicit unions of those blocks.  Coordinate and block
directions are candidate-generation heuristics only.  Every returned state has
``proof_authority=False`` and must pass the existing independent full-frame
checker before it can affect a verification verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.gpu_dual_candidates import OriginalFrameLP


_FLOAT_EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class CausalDualBlock:
    """One selector-closed semantic row block.

    ``stable_row_keys`` provide row-permutation-independent identities.
    ``incident_columns`` are original-frame column positions used only by
    cycle detection; they do not select or remove constraints.
    """

    block_id: str
    family_id: str
    global_rows: Tuple[int, ...]
    stable_row_keys: Tuple[str, ...]
    incident_columns: Tuple[int, ...] = ()


@dataclass(frozen=True)
class CausalDirectionUnion:
    """An explicit union of complete semantic blocks used for one block ray."""

    union_id: str
    block_ids: Tuple[str, ...]


@dataclass(frozen=True)
class PropertyCausalBlockDualCandidates:
    """Best full-frame candidate retained across coordinate/block exploration."""

    d: np.ndarray
    fallback_support: np.ndarray
    warm_support: np.ndarray
    candidate_support: np.ndarray
    enabled_rows: np.ndarray
    enabled_families: Tuple[str, ...]
    updates: int
    coordinate_updates: int
    block_updates: int
    zero_gain_updates: int
    cycle_rejections: int
    projection_count: int
    projection_max: float
    deadline_reached: bool
    nnz_cap_reached: bool
    update_cap_reached: bool
    stop_reasons: Tuple[str, ...]
    elapsed_seconds: float
    proof_authority: bool = False
    method: str = "property_conditioned_causal_block_dual_exchange_v1"


@dataclass(frozen=True)
class _PreparedStructure:
    blocks: Tuple[CausalDualBlock, ...]
    unions: Tuple[CausalDirectionUnion, ...]
    block_by_id: dict[str, CausalDualBlock]
    enabled_rows: np.ndarray
    enabled_row_keys: dict[int, str]
    incident_columns: np.ndarray
    enabled_families: Tuple[str, ...]


@dataclass(frozen=True)
class _Move:
    move_id: str
    kind: str
    d: np.ndarray
    support: float
    gain: float


def _canonical_csr(value: object) -> sp.csr_matrix:
    matrix = sp.csr_matrix(value, dtype=np.float64)
    if not matrix.has_sorted_indices:
        matrix.sort_indices()
    if not matrix.has_canonical_format:
        raise ValueError("original-frame matrix has duplicate coefficients")
    if matrix.nnz and not np.all(np.isfinite(matrix.data)):
        raise ValueError("original-frame matrix has non-finite coefficients")
    return matrix


def _validate_frame_and_objectives(
    frame: OriginalFrameLP,
    q: np.ndarray,
    warm_d: np.ndarray,
) -> tuple[
    sp.csr_matrix,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    A = _canonical_csr(frame.A)
    rl = np.asarray(frame.rl, dtype=np.float64).reshape(-1)
    ru = np.asarray(frame.ru, dtype=np.float64).reshape(-1)
    lb = np.asarray(frame.lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(frame.ub, dtype=np.float64).reshape(-1)
    q64 = np.asarray(q, dtype=np.float64)
    warm64 = np.asarray(warm_d, dtype=np.float64)
    if (
        A.shape != (rl.size, lb.size)
        or ru.size != rl.size
        or ub.size != lb.size
        or len(frame.row_tags) != rl.size
    ):
        raise ValueError("original-frame LP shape mismatch")
    if q64.ndim != 2 or q64.shape[1] != A.shape[1]:
        raise ValueError("q must have shape [objectives, frame variables]")
    if warm64.shape != (q64.shape[0], A.shape[0]):
        raise ValueError("warm_d must have shape [objectives, frame rows]")
    if (
        not np.all(np.isfinite(q64))
        or not np.all(np.isfinite(warm64))
        or not np.all(np.isfinite(lb))
        or not np.all(np.isfinite(ub))
        or np.any(lb > ub)
        or np.any(np.isnan(rl))
        or np.any(np.isnan(ru))
        or np.any(rl > ru)
        or np.any(np.isposinf(rl))
        or np.any(np.isneginf(ru))
    ):
        raise ValueError("original-frame LP or warm dual contains invalid data")
    return A, rl, ru, lb, ub, q64.copy(), warm64.copy()


def _prepare_structure(
    *,
    n_rows: int,
    n_variables: int,
    rl: np.ndarray,
    ru: np.ndarray,
    blocks: Sequence[CausalDualBlock],
    direction_unions: Sequence[CausalDirectionUnion],
    enabled_families: Optional[Sequence[str]],
) -> _PreparedStructure:
    block_tuple = tuple(blocks)
    union_tuple = tuple(direction_unions)
    if len({block.block_id for block in block_tuple}) != len(block_tuple):
        raise ValueError("semantic block ids must be unique")
    if len({union.union_id for union in union_tuple}) != len(union_tuple):
        raise ValueError("direction union ids must be unique")
    block_by_id = {block.block_id: block for block in block_tuple}

    all_families = {str(block.family_id) for block in block_tuple}
    if enabled_families is None:
        enabled = tuple(sorted(all_families))
    else:
        requested = {str(value) for value in enabled_families}
        unknown = requested - all_families
        if unknown:
            raise ValueError(
                f"enabled_families contains unknown families: {sorted(unknown)}"
            )
        enabled = tuple(sorted(requested))
    enabled_set = set(enabled)

    seen_rows: set[int] = set()
    seen_keys: set[str] = set()
    enabled_rows: list[int] = []
    enabled_row_keys: dict[int, str] = {}
    incident_columns: set[int] = set()
    for block in block_tuple:
        rows = tuple(int(value) for value in block.global_rows)
        keys = tuple(str(value) for value in block.stable_row_keys)
        if not block.block_id or not block.family_id:
            raise ValueError("semantic block and family ids must be non-empty")
        if not rows or len(rows) != len(keys):
            raise ValueError(
                f"block {block.block_id!r} must bind one stable key per row"
            )
        if len(set(rows)) != len(rows) or len(set(keys)) != len(keys):
            raise ValueError(f"block {block.block_id!r} has duplicate rows or keys")
        if any(row < 0 or row >= n_rows for row in rows):
            raise IndexError(f"block {block.block_id!r} row is out of range")
        if seen_rows.intersection(rows):
            raise ValueError("semantic blocks must be row-disjoint")
        if seen_keys.intersection(keys):
            raise ValueError("stable row keys must be globally unique")
        seen_rows.update(rows)
        seen_keys.update(keys)
        columns = tuple(int(value) for value in block.incident_columns)
        if len(set(columns)) != len(columns):
            raise ValueError(
                f"block {block.block_id!r} has duplicate incident columns"
            )
        if any(column < 0 or column >= n_variables for column in columns):
            raise IndexError(
                f"block {block.block_id!r} incident column is out of range"
            )
        if block.family_id in enabled_set:
            for row, key in zip(rows, keys):
                enabled_rows.append(row)
                enabled_row_keys[row] = key
            incident_columns.update(columns)

    for union in union_tuple:
        if not union.union_id or not union.block_ids:
            raise ValueError("each direction union must name at least one block")
        if len(set(union.block_ids)) != len(union.block_ids):
            raise ValueError(
                f"direction union {union.union_id!r} repeats a block"
            )
        missing = set(union.block_ids) - set(block_by_id)
        if missing:
            raise ValueError(
                f"direction union {union.union_id!r} has unknown blocks: "
                f"{sorted(missing)}"
            )

    declared_rows = np.asarray(sorted(seen_rows), dtype=np.int64)
    if declared_rows.size:
        declared_upper_only = (
            ~np.isfinite(rl[declared_rows])
            & np.isfinite(ru[declared_rows])
        )
        if not bool(np.all(declared_upper_only)):
            raise ValueError(
                "PC-CBDE v1 semantic blocks must contain upper-only rows"
            )

    enabled_array = np.asarray(
        sorted(enabled_rows, key=lambda row: enabled_row_keys[row]),
        dtype=np.int64,
    )
    if not incident_columns and enabled_array.size:
        # The exact incidence is available from A at optimization time.  Empty
        # metadata remains valid; cycle keys conservatively use all columns.
        incident_array = np.arange(n_variables, dtype=np.int64)
    else:
        incident_array = np.asarray(sorted(incident_columns), dtype=np.int64)
    return _PreparedStructure(
        blocks=block_tuple,
        unions=union_tuple,
        block_by_id=block_by_id,
        enabled_rows=enabled_array,
        enabled_row_keys=enabled_row_keys,
        incident_columns=incident_array,
        enabled_families=enabled,
    )


def _project_dual_domain(
    d: np.ndarray,
    *,
    rl: np.ndarray,
    ru: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    projected = np.asarray(d, dtype=np.float64).copy()
    finite_l = np.isfinite(rl)
    finite_u = np.isfinite(ru)
    upper_only = (~finite_l) & finite_u
    lower_only = finite_l & (~finite_u)
    free = (~finite_l) & (~finite_u)
    before = projected.copy()
    if np.any(upper_only):
        projected[upper_only] = np.maximum(projected[upper_only], 0.0)
    if np.any(lower_only):
        projected[lower_only] = np.minimum(projected[lower_only], 0.0)
    if np.any(free):
        projected[free] = 0.0
    delta = np.abs(projected - before)
    return (
        projected,
        int(np.count_nonzero(delta)),
        float(np.max(delta) if delta.size else 0.0),
    )


def _support_value(
    *,
    A: sp.csr_matrix,
    rl: np.ndarray,
    ru: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    q: np.ndarray,
    d: np.ndarray,
) -> float:
    finite_l = np.isfinite(rl)
    finite_u = np.isfinite(ru)
    upper_only = (~finite_l) & finite_u
    lower_only = finite_l & (~finite_u)
    free = (~finite_l) & (~finite_u)
    domain_tol = 128.0 * _FLOAT_EPS * max(
        1.0, float(np.max(np.abs(d))) if d.size else 1.0
    )
    if (
        np.any(d[upper_only] < -domain_tol)
        or np.any(d[lower_only] > domain_tol)
        or np.any(np.abs(d[free]) > domain_tol)
    ):
        return float("inf")
    row_side = np.where(d >= 0.0, ru, rl)
    row_terms = np.zeros_like(d)
    finite_side = np.isfinite(row_side)
    row_terms[finite_side] = d[finite_side] * row_side[finite_side]
    residual = np.asarray(q - A.transpose() @ d, dtype=np.float64).reshape(-1)
    box_side = np.where(residual >= 0.0, ub, lb)
    value = float(np.sum(row_terms) + np.dot(residual, box_side))
    return value if np.isfinite(value) else float("inf")


def _gain_tolerance(old: float, new: float) -> float:
    return 128.0 * _FLOAT_EPS * (1.0 + abs(old) + abs(new))


def _line_minimizer(
    *,
    A: sp.csr_matrix,
    ru: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    residual: np.ndarray,
    d: np.ndarray,
    p: np.ndarray,
    nonnegative_step: bool,
) -> Optional[float]:
    """Exact binary64 minimizer of the convex PL support along ``d + t p``."""

    active = np.flatnonzero(p != 0.0)
    if active.size == 0:
        return None
    lower = 0.0 if nonnegative_step else -float("inf")
    upper = float("inf")
    for row in active:
        value = float(d[row])
        direction = float(p[row])
        boundary = -value / direction
        if direction > 0.0:
            lower = max(lower, boundary)
        else:
            upper = min(upper, boundary)
    domain_tol = 128.0 * _FLOAT_EPS * (
        1.0
        + abs(lower if np.isfinite(lower) else 0.0)
        + abs(upper if np.isfinite(upper) else 0.0)
    )
    if lower > upper + domain_tol:
        return None
    if lower > upper:
        lower = upper = 0.5 * (lower + upper)

    v = np.asarray(A.transpose() @ p, dtype=np.float64).reshape(-1)
    midpoint = 0.5 * (lb + ub)
    radius = 0.5 * (ub - lb)
    linear_slope = float(np.dot(ru[active], p[active]))
    linear_slope -= float(np.dot(midpoint, v))
    base_slope = linear_slope - float(np.dot(radius, np.abs(v)))
    if not np.isfinite(base_slope):
        return None

    nonzero = (v != 0.0) & (radius > 0.0)
    points = residual[nonzero] / v[nonzero]
    jumps = 2.0 * radius[nonzero] * np.abs(v[nonzero])
    keep = np.isfinite(points) & np.isfinite(jumps) & (jumps > 0.0)
    points = np.asarray(points[keep], dtype=np.float64)
    jumps = np.asarray(jumps[keep], dtype=np.float64)
    if points.size:
        order = np.argsort(points, kind="mergesort")
        points = points[order]
        jumps = jumps[order]
        unique_points, first = np.unique(points, return_index=True)
        unique_jumps = np.add.reduceat(jumps, first)
    else:
        unique_points = np.zeros(0, dtype=np.float64)
        unique_jumps = np.zeros(0, dtype=np.float64)

    slope_scale = (
        1.0
        + abs(base_slope)
        + float(np.sum(np.abs(unique_jumps)))
        + abs(linear_slope)
    )
    slope_tol = 128.0 * _FLOAT_EPS * slope_scale
    slope = float(base_slope)
    if np.isfinite(lower):
        before = unique_points < lower
        if np.any(before):
            slope += float(np.sum(unique_jumps[before]))
        at_lower = unique_points == lower
        if np.any(at_lower):
            slope += float(np.sum(unique_jumps[at_lower]))
        if slope > slope_tol:
            return float(lower)
        scan = np.flatnonzero(unique_points > lower)
    else:
        if slope > slope_tol:
            return None
        scan = np.arange(unique_points.size, dtype=np.int64)

    last_point: Optional[float] = float(lower) if np.isfinite(lower) else None
    for raw_index in scan:
        point = float(unique_points[int(raw_index)])
        if point > upper:
            break
        slope += float(unique_jumps[int(raw_index)])
        last_point = point
        # Deliberately cross a zero-slope interval to its far edge.  This
        # preserves the useful wavefront behavior; cycle caps bound flat travel.
        if slope > slope_tol:
            return min(max(point, lower), upper)

    if np.isfinite(upper):
        return float(upper)
    if abs(slope) <= slope_tol and last_point is not None:
        return float(last_point)
    return None


def _candidate_from_direction(
    *,
    move_id: str,
    kind: str,
    A: sp.csr_matrix,
    rl: np.ndarray,
    ru: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    q: np.ndarray,
    d: np.ndarray,
    residual: np.ndarray,
    support: float,
    p: np.ndarray,
    nonnegative_step: bool,
    nnz_cap: int,
    zero_tol: float,
) -> tuple[Optional[_Move], bool]:
    step = _line_minimizer(
        A=A,
        ru=ru,
        lb=lb,
        ub=ub,
        residual=residual,
        d=d,
        p=p,
        nonnegative_step=nonnegative_step,
    )
    if step is None or not np.isfinite(step):
        return None, False
    movement = step * p
    movement_tol = 128.0 * _FLOAT_EPS * max(
        1.0,
        float(np.max(np.abs(d))) if d.size else 1.0,
        float(np.max(np.abs(movement))) if movement.size else 1.0,
    )
    if not np.any(np.abs(movement) > movement_tol):
        return None, False
    candidate = d + movement
    domain_tol = 256.0 * _FLOAT_EPS * max(
        1.0, float(np.max(np.abs(candidate))) if candidate.size else 1.0
    )
    upper_only = (~np.isfinite(rl)) & np.isfinite(ru)
    if np.any(candidate[upper_only] < -domain_tol):
        return None, False
    candidate[upper_only] = np.maximum(candidate[upper_only], 0.0)
    candidate[np.abs(candidate) <= zero_tol] = 0.0
    if int(np.count_nonzero(np.abs(candidate) > zero_tol)) > int(nnz_cap):
        return None, True
    candidate_support = _support_value(
        A=A,
        rl=rl,
        ru=ru,
        lb=lb,
        ub=ub,
        q=q,
        d=candidate,
    )
    if not np.isfinite(candidate_support):
        return None, False
    gain = float(support - candidate_support)
    if gain < -_gain_tolerance(support, candidate_support):
        return None, False
    return (
        _Move(
            move_id=move_id,
            kind=kind,
            d=candidate,
            support=float(candidate_support),
            gain=float(max(gain, 0.0)),
        ),
        False,
    )


def _state_keys(
    *,
    d: np.ndarray,
    residual: np.ndarray,
    structure: _PreparedStructure,
    face_tol: float,
) -> tuple[bytes, bytes]:
    ordered_rows = sorted(
        structure.enabled_rows.tolist(),
        key=lambda row: structure.enabled_row_keys[int(row)],
    )
    exact = hashlib.sha256()
    face = hashlib.sha256()
    for family in structure.enabled_families:
        encoded = family.encode("utf-8")
        exact.update(encoded)
        exact.update(b"\0")
        face.update(encoded)
        face.update(b"\0")
    for row in ordered_rows:
        key = structure.enabled_row_keys[int(row)].encode("utf-8")
        exact.update(key)
        exact.update(np.float64(d[int(row)]).tobytes())
        face.update(key)
        face.update(b"+" if d[int(row)] > face_tol else b"0")
    columns = structure.incident_columns
    for column in columns.tolist():
        value = float(residual[int(column)])
        exact.update(np.int64(column).tobytes())
        exact.update(np.float64(value).tobytes())
        face.update(np.int64(column).tobytes())
        face.update(b"+" if value > face_tol else b"-" if value < -face_tol else b"0")
    return exact.digest(), face.digest()


def _coordinate_frontier(
    *,
    A: sp.csr_matrix,
    ru: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    residual: np.ndarray,
    d: np.ndarray,
    structure: _PreparedStructure,
    frontier_topk: int,
    face_tol: float,
) -> list[int]:
    if structure.enabled_rows.size == 0:
        return []
    midpoint = 0.5 * (lb + ub)
    radius = 0.5 * (ub - lb)
    signs = np.zeros_like(residual)
    signs[residual > face_tol] = 1.0
    signs[residual < -face_tol] = -1.0
    box_point = midpoint + radius * signs
    projected = np.asarray(A @ box_point, dtype=np.float64).reshape(-1)
    rows = structure.enabled_rows
    gradient = ru[rows] - projected[rows]
    active = d[rows] > face_tol
    score = np.where(active, np.abs(gradient), np.maximum(-gradient, 0.0))
    score[~np.isfinite(score)] = 0.0
    useful = np.flatnonzero(score > face_tol)
    chosen: set[int] = {
        int(rows[index]) for index in np.flatnonzero(active)
    }
    if useful.size:
        keep = min(int(frontier_topk), int(useful.size))
        ranked = sorted(
            useful.tolist(),
            key=lambda index: (
                -float(score[int(index)]),
                structure.enabled_row_keys[int(rows[int(index)])],
            ),
        )
        chosen.update(int(rows[index]) for index in ranked[:keep])
    return sorted(
        chosen, key=lambda row: structure.enabled_row_keys[int(row)]
    )


def _block_directions(
    *,
    A: sp.csr_matrix,
    ru: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    residual: np.ndarray,
    d: np.ndarray,
    structure: _PreparedStructure,
    face_tol: float,
) -> list[tuple[str, np.ndarray]]:
    if structure.enabled_rows.size == 0:
        return []
    midpoint = 0.5 * (lb + ub)
    radius = 0.5 * (ub - lb)
    signs = np.zeros_like(residual)
    signs[residual > face_tol] = 1.0
    signs[residual < -face_tol] = -1.0
    box_point = midpoint + radius * signs
    projected = np.asarray(A @ box_point, dtype=np.float64).reshape(-1)
    row_scale = np.asarray(np.abs(A) @ radius, dtype=np.float64).reshape(-1)
    row_scale = np.maximum(1.0, row_scale + np.abs(ru))
    enabled_set = set(structure.enabled_rows.tolist())
    directions: list[tuple[str, np.ndarray]] = []
    seen_vectors: set[bytes] = set()

    for union in sorted(structure.unions, key=lambda item: item.union_id):
        union_rows: list[int] = []
        for block_id in union.block_ids:
            block = structure.block_by_id[block_id]
            if block.family_id not in set(structure.enabled_families):
                continue
            union_rows.extend(int(row) for row in block.global_rows)
        rows = sorted(
            set(union_rows).intersection(enabled_set),
            key=lambda row: structure.enabled_row_keys[int(row)],
        )
        if not rows:
            continue
        rows_array = np.asarray(rows, dtype=np.int64)
        gradient = ru[rows_array] - projected[rows_array]
        local = -gradient / row_scale[rows_array]
        at_boundary = d[rows_array] <= face_tol
        local[at_boundary] = np.maximum(local[at_boundary], 0.0)
        local[np.abs(local) <= face_tol] = 0.0
        if np.any(local != 0.0):
            direction = np.zeros(A.shape[0], dtype=np.float64)
            direction[rows_array] = local
            digest = hashlib.sha256(direction.tobytes()).digest()
            if digest not in seen_vectors:
                seen_vectors.add(digest)
                directions.append(
                    (f"block:{union.union_id}:exchange", direction)
                )

        release_local = -d[rows_array]
        release_local[np.abs(release_local) <= face_tol] = 0.0
        if np.any(release_local != 0.0):
            direction = np.zeros(A.shape[0], dtype=np.float64)
            direction[rows_array] = release_local
            digest = hashlib.sha256(direction.tobytes()).digest()
            if digest not in seen_vectors:
                seen_vectors.add(digest)
                directions.append(
                    (f"block:{union.union_id}:release", direction)
                )
    return directions


def property_causal_block_duals(
    frame: OriginalFrameLP,
    q: np.ndarray,
    warm_d: np.ndarray,
    *,
    blocks: Sequence[CausalDualBlock],
    direction_unions: Sequence[CausalDirectionUnion],
    enabled_families: Optional[Sequence[str]] = None,
    max_updates: int = 64,
    max_zero_gain_updates: int = 16,
    face_visit_cap: int = 2,
    frontier_topk: int = 64,
    nnz_cap: int = 96,
    deadline: Optional[float] = None,
) -> PropertyCausalBlockDualCandidates:
    """Optimize proof-neutral full duals with coordinate and semantic block rays.

    ``warm_d`` uses the maximization-certificate convention and the same scale
    as ``q``.  If a caller normalizes ``q`` by a positive objective scale, it
    must divide ``warm_d`` by the same scale before calling this function.
    """

    if int(max_updates) < 0:
        raise ValueError("max_updates must be nonnegative")
    if int(max_zero_gain_updates) < 0:
        raise ValueError("max_zero_gain_updates must be nonnegative")
    if int(face_visit_cap) < 1:
        raise ValueError("face_visit_cap must be positive")
    if int(frontier_topk) < 1:
        raise ValueError("frontier_topk must be positive")
    if int(nnz_cap) < 1:
        raise ValueError("nnz_cap must be positive")
    if deadline is not None and not math.isfinite(float(deadline)):
        raise ValueError("deadline must be a finite absolute monotonic time")

    started = time.monotonic()
    A, rl, ru, lb, ub, q64, warm64 = _validate_frame_and_objectives(
        frame, q, warm_d
    )
    structure = _prepare_structure(
        n_rows=A.shape[0],
        n_variables=A.shape[1],
        rl=rl,
        ru=ru,
        blocks=blocks,
        direction_unions=direction_unions,
        enabled_families=enabled_families,
    )

    declared_rows = sorted(
        {
            int(row)
            for block in structure.blocks
            for row in block.global_rows
        }
    )
    enabled_set = set(structure.enabled_rows.tolist())
    disabled_rows = np.asarray(
        [row for row in declared_rows if row not in enabled_set],
        dtype=np.int64,
    )
    zero_tol = 256.0 * _FLOAT_EPS
    projection_count = 0
    projection_max = 0.0
    for rival in range(warm64.shape[0]):
        projected, count, maximum = _project_dual_domain(
            warm64[rival], rl=rl, ru=ru
        )
        warm64[rival] = projected
        projection_count += count
        projection_max = max(projection_max, maximum)
    if disabled_rows.size:
        warm64[:, disabled_rows] = 0.0

    fallback = warm64.copy()
    if declared_rows:
        fallback[:, np.asarray(declared_rows, dtype=np.int64)] = 0.0
    fallback_support = np.asarray(
        [
            _support_value(
                A=A, rl=rl, ru=ru, lb=lb, ub=ub,
                q=q64[rival], d=fallback[rival],
            )
            for rival in range(q64.shape[0])
        ],
        dtype=np.float64,
    )
    warm_support = np.asarray(
        [
            _support_value(
                A=A, rl=rl, ru=ru, lb=lb, ub=ub,
                q=q64[rival], d=warm64[rival],
            )
            for rival in range(q64.shape[0])
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(fallback_support)) or not np.all(
        np.isfinite(warm_support)
    ):
        raise ValueError("initial full-frame support is non-finite")

    best_d = fallback.copy()
    best_support = fallback_support.copy()
    use_warm = warm_support < best_support
    best_d[use_warm] = warm64[use_warm]
    best_support[use_warm] = warm_support[use_warm]

    updates_total = 0
    coordinate_updates = 0
    block_updates = 0
    zero_gain_updates = 0
    cycle_rejections = 0
    deadline_reached = False
    nnz_cap_reached = False
    update_cap_reached = False
    stop_reasons: list[str] = []

    for rival in range(q64.shape[0]):
        current = warm64[rival].copy()
        current_support = float(warm_support[rival])
        if int(np.count_nonzero(np.abs(current) > zero_tol)) > int(nnz_cap):
            current = fallback[rival].copy()
            current_support = float(fallback_support[rival])
            nnz_cap_reached = True
        residual = np.asarray(
            q64[rival] - A.transpose() @ current,
            dtype=np.float64,
        ).reshape(-1)
        face_tol = 256.0 * _FLOAT_EPS * max(
            1.0,
            float(np.max(np.abs(q64[rival]))) if q64.shape[1] else 1.0,
            float(np.max(np.abs(residual))) if residual.size else 1.0,
        )
        exact_key, face_key = _state_keys(
            d=current,
            residual=residual,
            structure=structure,
            face_tol=face_tol,
        )
        seen_exact: set[bytes] = {exact_key}
        face_visits: dict[bytes, int] = {face_key: 1}
        reason = "stationary"

        while updates_total < int(max_updates):
            if deadline is not None and time.monotonic() >= float(deadline):
                deadline_reached = True
                reason = "deadline"
                break

            candidates: list[_Move] = []
            coordinate_rows = _coordinate_frontier(
                A=A,
                ru=ru,
                lb=lb,
                ub=ub,
                residual=residual,
                d=current,
                structure=structure,
                frontier_topk=int(frontier_topk),
                face_tol=face_tol,
            )
            for row in coordinate_rows:
                direction = np.zeros(A.shape[0], dtype=np.float64)
                direction[int(row)] = 1.0
                move, cap_hit = _candidate_from_direction(
                    move_id=(
                        "coordinate:"
                        f"{structure.enabled_row_keys[int(row)]}"
                    ),
                    kind="coordinate",
                    A=A,
                    rl=rl,
                    ru=ru,
                    lb=lb,
                    ub=ub,
                    q=q64[rival],
                    d=current,
                    residual=residual,
                    support=current_support,
                    p=direction,
                    nonnegative_step=False,
                    nnz_cap=int(nnz_cap),
                    zero_tol=zero_tol,
                )
                nnz_cap_reached = nnz_cap_reached or cap_hit
                if move is not None:
                    candidates.append(move)

            for move_id, direction in _block_directions(
                A=A,
                ru=ru,
                lb=lb,
                ub=ub,
                residual=residual,
                d=current,
                structure=structure,
                face_tol=face_tol,
            ):
                move, cap_hit = _candidate_from_direction(
                    move_id=move_id,
                    kind="block",
                    A=A,
                    rl=rl,
                    ru=ru,
                    lb=lb,
                    ub=ub,
                    q=q64[rival],
                    d=current,
                    residual=residual,
                    support=current_support,
                    p=direction,
                    nonnegative_step=True,
                    nnz_cap=int(nnz_cap),
                    zero_tol=zero_tol,
                )
                nnz_cap_reached = nnz_cap_reached or cap_hit
                if move is not None:
                    candidates.append(move)

            strict = [
                move
                for move in candidates
                if move.gain
                > _gain_tolerance(current_support, move.support)
            ]
            if strict:
                chosen = sorted(
                    strict, key=lambda move: (-move.gain, move.move_id)
                )[0]
                seen_exact.clear()
                face_visits.clear()
            else:
                chosen = None
                if zero_gain_updates < int(max_zero_gain_updates):
                    for move in sorted(candidates, key=lambda item: item.move_id):
                        move_residual = np.asarray(
                            q64[rival] - A.transpose() @ move.d,
                            dtype=np.float64,
                        ).reshape(-1)
                        move_exact, move_face = _state_keys(
                            d=move.d,
                            residual=move_residual,
                            structure=structure,
                            face_tol=face_tol,
                        )
                        if (
                            move_exact in seen_exact
                            or face_visits.get(move_face, 0)
                            >= int(face_visit_cap)
                        ):
                            cycle_rejections += 1
                            continue
                        chosen = move
                        break
                if chosen is None:
                    if (
                        candidates
                        and zero_gain_updates >= int(max_zero_gain_updates)
                    ):
                        reason = "zero_gain_cap"
                    elif candidates:
                        reason = "cycle"
                    elif nnz_cap_reached:
                        reason = "nnz_cap"
                    else:
                        reason = "stationary"
                    break

            current = chosen.d.copy()
            current_support = float(chosen.support)
            residual = np.asarray(
                q64[rival] - A.transpose() @ current,
                dtype=np.float64,
            ).reshape(-1)
            updates_total += 1
            if chosen.kind == "coordinate":
                coordinate_updates += 1
            else:
                block_updates += 1
            if chosen.gain <= _gain_tolerance(
                current_support + chosen.gain, current_support
            ):
                zero_gain_updates += 1
            if current_support < best_support[rival] - _gain_tolerance(
                float(best_support[rival]), current_support
            ):
                best_support[rival] = current_support
                best_d[rival] = current
            exact_key, face_key = _state_keys(
                d=current,
                residual=residual,
                structure=structure,
                face_tol=face_tol,
            )
            seen_exact.add(exact_key)
            face_visits[face_key] = face_visits.get(face_key, 0) + 1

        if updates_total >= int(max_updates):
            update_cap_reached = True
            if reason == "stationary":
                reason = "update_cap"
        stop_reasons.append(reason)
        if deadline_reached:
            # Do not start another rival after the shared absolute deadline.
            stop_reasons.extend(
                "not_started_deadline"
                for _ in range(rival + 1, q64.shape[0])
            )
            break

    # A shared update cap may leave later rivals untouched.  Their preselected
    # fallback/warm best states are already present in ``best_d``.
    if len(stop_reasons) < q64.shape[0]:
        stop_reasons.extend(
            "not_started_update_cap"
            for _ in range(len(stop_reasons), q64.shape[0])
        )

    return PropertyCausalBlockDualCandidates(
        d=best_d,
        fallback_support=fallback_support,
        warm_support=warm_support,
        candidate_support=best_support,
        enabled_rows=structure.enabled_rows.copy(),
        enabled_families=structure.enabled_families,
        updates=int(updates_total),
        coordinate_updates=int(coordinate_updates),
        block_updates=int(block_updates),
        zero_gain_updates=int(zero_gain_updates),
        cycle_rejections=int(cycle_rejections),
        projection_count=int(projection_count),
        projection_max=float(projection_max),
        deadline_reached=bool(deadline_reached),
        nnz_cap_reached=bool(nnz_cap_reached),
        update_cap_reached=bool(update_cap_reached),
        stop_reasons=tuple(stop_reasons),
        elapsed_seconds=float(time.monotonic() - started),
    )


__all__ = [
    "CausalDirectionUnion",
    "CausalDualBlock",
    "PropertyCausalBlockDualCandidates",
    "property_causal_block_duals",
]
