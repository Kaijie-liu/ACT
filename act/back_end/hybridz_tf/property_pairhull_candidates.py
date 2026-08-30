#!/usr/bin/env python3
# ===- property_pairhull_candidates.py - sparse PairHull candidates ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===----------------------------------------------------------------===#
"""Sparse, candidate-only construction of property-bundled PairHull rows.

This module deliberately does not mutate an operator frame and never removes
the caller's foundation rows.  It chooses a small number of preactivation
pairs with a bounded float heuristic, then reconstructs every proposed
intercept with exact :class:`fractions.Fraction` arithmetic and the independent
four-phase core in :mod:`property_pairhull`.

For a stored foundation row

``q_i ReLU(z_i) <= a_i z_i + r_i``,

the caller promises that its outward-stored intercept contains the exact
singleton requirements for every neuron over the supplied stored ``lower``
and ``upper`` bounds.  Replacing two singleton requirements by an exact joint
requirement therefore gives

``new = stored_foundation_intercept - r_i - r_j + beta_ij``.

The stored foundation intercept, rather than a reconstructed bias/threshold
term, is the starting point.  Consequently all pre-existing outward slack is
retained.  The final value is again rounded toward positive infinity.

Pair selection has no proof authority.  In particular it samples only a
bounded number of large entries from each sparse generator row, bounds the
posting-list fanout, and bounds the global pair pool.  Any truncation is
recorded.  The authoritative construction uses the full sparse union of each
selected pair's generator rows, explicit independent rowwise errors, exact
stored binary64 property products, exact singleton requirements, and exact
four-phase PairHull beta values.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from numbers import Integral, Real
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.property_pairhull import (
    DEFAULT_PAIRHULL_DIRECTIONS,
    build_pairhull_projection,
    exact_pairhull_beta,
)


_ZERO = Fraction(0)


class _PairHullDeadline(RuntimeError):
    """Internal signal used to discard an incomplete candidate batch."""


class _PairHullResourceCap(RuntimeError):
    """Internal signal used to discard work exceeding an exact-work cap."""


@dataclass(frozen=True)
class PropertyPairHullCandidates:
    """At most one exact-audited candidate row for each property rival.

    ``planes`` and ``intercepts`` contain only new candidate rows.  The
    foundation rows are intentionally absent and must remain retained by the
    caller.  Rows are ordered by increasing ``rival_ids``.
    """

    rival_ids: np.ndarray
    foundation_indices: np.ndarray
    pair_indices: np.ndarray
    planes: np.ndarray
    intercepts: np.ndarray
    receipt: Dict[str, Any]

    @property
    def status(self) -> str:
        return str(self.receipt.get("status", "unknown"))


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray, *, dtype: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _csr_sha256(value: sp.csr_matrix) -> str:
    matrix = sp.csr_matrix(value, dtype=np.float64, copy=False)
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    digest.update(np.asarray(matrix.indptr, dtype=np.int64).tobytes())
    digest.update(np.asarray(matrix.indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(matrix.data, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _stored_fraction(value: Any, *, name: str) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{name}: bool is not a stored coefficient")
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Real):
        stored = float(value)
        if not math.isfinite(stored):
            raise ValueError(f"{name}: coefficient must be finite")
        return Fraction.from_float(stored)
    raise ValueError(f"{name}: expected a finite stored real")


def _outward_float(value: Fraction, *, name: str) -> float:
    """Store a finite Fraction as binary64 rounded toward positive infinity."""

    try:
        stored = float(value)
    except OverflowError as exc:
        raise OverflowError(f"{name}: exact value overflowed binary64") from exc
    if not math.isfinite(stored):
        raise OverflowError(f"{name}: exact value overflowed binary64")
    if Fraction.from_float(stored) < value:
        stored = math.nextafter(stored, math.inf)
    if not math.isfinite(stored) or Fraction.from_float(stored) < value:
        raise OverflowError(f"{name}: outward binary64 conversion failed")
    return stored


def _singleton_requirement(
    *,
    q: Fraction,
    slope: Fraction,
    lower: Fraction,
    upper: Fraction,
) -> Fraction:
    """Exact max of ``q*ReLU(z)-slope*z`` on a stored interval."""

    if lower > upper:
        raise ValueError("singleton interval is reversed")
    endpoints = [lower, upper]
    if lower < 0 < upper:
        endpoints.append(_ZERO)
    return max(
        q * max(_ZERO, point) - slope * point for point in endpoints
    )


def finalize_property_pairhull_candidates_receipt(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Copy a JSON-friendly payload and attach its canonical SHA-256."""

    receipt = dict(payload)
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def verify_property_pairhull_candidates_receipt(
    receipt: Mapping[str, Any],
) -> bool:
    """Detect any top-level or nested mutation of a batch receipt."""

    try:
        expected = receipt["receipt_sha256"]
        if not isinstance(expected, str) or len(expected) != 64:
            return False
        payload = dict(receipt)
        del payload["receipt_sha256"]
        return _canonical_sha256(payload) == expected
    except (KeyError, TypeError, ValueError):
        return False


# Concise alias used by the operator/verifier wrapper as well as inner
# candidate receipts.  The checksum format is schema-agnostic.
verify_pairhull_candidate_receipt = (
    verify_property_pairhull_candidates_receipt
)


def _empty_result(
    *,
    width: int,
    status: str,
    started: float,
    receipt: Dict[str, Any],
    whole_batch_complete: bool,
    partial_candidates_discarded: int = 0,
    error: Optional[BaseException] = None,
) -> PropertyPairHullCandidates:
    payload = {
        **receipt,
        "status": str(status),
        "whole_batch_complete": bool(whole_batch_complete),
        "partial_candidates_discarded": int(partial_candidates_discarded),
        "selected_candidates": 0,
        "foundation_rows_retained_by_caller": True,
        "elapsed_seconds": float(max(0.0, time.monotonic() - started)),
    }
    payload.setdefault("global_pair_count", 0)
    payload.setdefault("candidate_records", [])
    payload.setdefault(
        "candidate_records_sha256",
        _canonical_sha256(payload["candidate_records"]),
    )
    if error is not None:
        payload.update(
            {
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
            }
        )
    return PropertyPairHullCandidates(
        rival_ids=np.zeros(0, dtype=np.int64),
        foundation_indices=np.zeros(0, dtype=np.int64),
        pair_indices=np.zeros((0, 2), dtype=np.int64),
        planes=np.zeros((0, max(0, int(width))), dtype=np.float64),
        intercepts=np.zeros(0, dtype=np.float64),
        receipt=finalize_property_pairhull_candidates_receipt(payload),
    )


def _as_canonical_csr(value: Any, *, name: str) -> sp.csr_matrix:
    matrix = sp.csr_matrix(value, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError(f"{name} contains a non-finite stored coefficient")
    return matrix


def _row_top_entries(
    matrix: sp.csr_matrix,
    row: int,
    limit: int,
) -> List[Tuple[int, float]]:
    """Select deterministic largest entries without a matrix-width buffer."""

    start = int(matrix.indptr[row])
    end = int(matrix.indptr[row + 1])
    indices = matrix.indices[start:end]
    data = matrix.data[start:end]
    count = int(end - start)
    if count <= limit:
        positions = np.arange(count, dtype=np.int64)
    else:
        magnitudes = np.abs(data)
        threshold = float(np.partition(magnitudes, count - limit)[count - limit])
        strict = np.flatnonzero(magnitudes > threshold)
        tied = np.flatnonzero(magnitudes == threshold)
        remaining = int(limit - strict.size)
        if remaining < 0:
            # This cannot occur for an order-statistic threshold, but falling
            # back is safer than silently violating the selector cap.
            raise AssertionError("top-k order statistic selected too many rows")
        if tied.size > remaining:
            tied_columns = indices[tied]
            chosen_tied = np.argsort(
                tied_columns, kind="stable"
            )[:remaining]
            tied = tied[chosen_tied]
        positions = np.concatenate([strict, tied[:remaining]])
    order = np.argsort(indices[positions], kind="stable")
    positions = positions[order]
    return [
        (int(indices[position]), float(data[position]))
        for position in positions
    ]


def _select_sparse_pairs(
    *,
    generators: sp.csr_matrix,
    approximate_q: np.ndarray,
    pair_budget: int,
    row_topk: int,
    max_selector_postings: int,
    max_rows_per_generator: int,
    max_pair_pool: int,
    max_pair_updates: int,
    stop_at: float,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any], Dict[Tuple[int, int], Dict[str, Any]]]:
    """Bounded shared-generator/mixed-property pair heuristic."""

    rows = int(generators.shape[0])
    if rows * row_topk > max_selector_postings:
        raise _PairHullResourceCap(
            "selector posting cap is smaller than rows*row_topk"
        )

    postings: Dict[int, List[Tuple[int, float]]] = {}
    sampled_norm_sq = np.zeros(rows, dtype=np.float64)
    sampled_entries = 0
    for row in range(rows):
        if time.monotonic() >= stop_at:
            raise _PairHullDeadline("deadline during sparse pair selector")
        for column, value in _row_top_entries(generators, row, row_topk):
            postings.setdefault(column, []).append((row, value))
            sampled_norm_sq[row] += value * value
            sampled_entries += 1

    pair_dot: Dict[Tuple[int, int], float] = {}
    pair_shared: Dict[Tuple[int, int], int] = {}
    pair_updates = 0
    posting_fanout_truncated = 0
    pair_pool_truncated = False
    update_truncated = False
    for column in sorted(postings):
        if time.monotonic() >= stop_at:
            raise _PairHullDeadline("deadline during sparse pair pool")
        entries = postings[column]
        entries.sort(key=lambda item: (-abs(item[1]), item[0]))
        if len(entries) > max_rows_per_generator:
            posting_fanout_truncated += len(entries) - max_rows_per_generator
            entries = entries[:max_rows_per_generator]
        for left_position in range(len(entries)):
            left_row, left_value = entries[left_position]
            for right_position in range(left_position + 1, len(entries)):
                if pair_updates >= max_pair_updates:
                    update_truncated = True
                    break
                right_row, right_value = entries[right_position]
                pair = (
                    (left_row, right_row)
                    if left_row < right_row
                    else (right_row, left_row)
                )
                if pair not in pair_dot and len(pair_dot) >= max_pair_pool:
                    pair_pool_truncated = True
                    pair_updates += 1
                    continue
                product = float(left_value * right_value)
                if not math.isfinite(product):
                    raise ValueError(
                        "non-finite product in sparse pair selector"
                    )
                pair_dot[pair] = pair_dot.get(pair, 0.0) + product
                pair_shared[pair] = pair_shared.get(pair, 0) + 1
                pair_updates += 1
            if update_truncated:
                break
        if update_truncated:
            break

    scored: List[Tuple[float, int, int]] = []
    score_records: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for pair, dot in pair_dot.items():
        left, right = pair
        qi = approximate_q[:, left]
        qj = approximate_q[:, right]
        mixed = ((qi > 0.0) & (qj < 0.0)) | (
            (qi < 0.0) & (qj > 0.0)
        )
        if not np.any(mixed):
            continue
        mixed_strength = float(
            np.max(np.minimum(np.abs(qi[mixed]), np.abs(qj[mixed])))
        )
        denominator = math.sqrt(
            float(sampled_norm_sq[left] * sampled_norm_sq[right])
        )
        correlation = abs(float(dot)) / denominator if denominator > 0 else 0.0
        score = correlation * mixed_strength
        if not math.isfinite(score) or score <= 0.0:
            continue
        scored.append((score, left, right))
        score_records[pair] = {
            "pair": [int(left), int(right)],
            "selector_score_hex": float(score).hex(),
            "sampled_correlation_hex": float(correlation).hex(),
            "mixed_property_strength_hex": float(mixed_strength).hex(),
            "sampled_shared_generators": int(pair_shared[pair]),
        }
    scored.sort(key=lambda entry: (-entry[0], entry[1], entry[2]))
    # V1 deliberately uses disjoint global pairs.  This makes the eventual
    # operator interpretation simple (no neuron participates in two bundled
    # replacements) and bounds exact work more tightly than score truncation
    # alone.
    selected: List[Tuple[int, int]] = []
    used_rows = set()
    for _, left, right in scored:
        if left in used_rows or right in used_rows:
            continue
        selected.append((int(left), int(right)))
        used_rows.add(int(left))
        used_rows.add(int(right))
        if len(selected) >= pair_budget:
            break
    selector_receipt = {
        "algorithm": "bounded_sparse_topk_postings_mixed_sign_v1",
        "proof_authority": False,
        "generator_matrix_densified": False,
        "row_topk": int(row_topk),
        "sampled_entries": int(sampled_entries),
        "posting_columns": int(len(postings)),
        "max_selector_postings": int(max_selector_postings),
        "max_rows_per_generator": int(max_rows_per_generator),
        "posting_fanout_discarded": int(posting_fanout_truncated),
        "max_pair_pool": int(max_pair_pool),
        "pair_pool_size": int(len(pair_dot)),
        "pair_pool_truncated": bool(pair_pool_truncated),
        "max_pair_updates": int(max_pair_updates),
        "pair_updates": int(pair_updates),
        "pair_updates_truncated": bool(update_truncated),
        "eligible_scored_pairs": int(len(scored)),
        "selected_pairs": int(len(selected)),
        "selected_pairs_disjoint": True,
        "selected_pair_records": [
            score_records[pair] for pair in selected
        ],
    }
    selector_receipt["selected_pairs_sha256"] = _canonical_sha256(
        selector_receipt["selected_pair_records"]
    )
    return selected, selector_receipt, score_records


def _float_pair_beta(
    projection: Any,
    *,
    q: Tuple[float, float],
    slope: Tuple[float, float],
) -> float:
    """Cheap vertex/axis proxy; never used to authorize an output row."""

    constraints = [
        (float(row[0]), float(row[1]), float(row[2]))
        for row in projection.constraints
    ]
    boundaries = constraints + [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    scale = max(
        [1.0]
        + [
            abs(value)
            for row in constraints
            for value in row
            if math.isfinite(value)
        ]
    )
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    candidates: List[Tuple[float, float]] = []
    for left_position in range(len(boundaries)):
        left = boundaries[left_position]
        for right_position in range(left_position + 1, len(boundaries)):
            right = boundaries[right_position]
            determinant = left[0] * right[1] - left[1] * right[0]
            if determinant == 0.0 or not math.isfinite(determinant):
                continue
            z0 = (left[2] * right[1] - left[1] * right[2]) / determinant
            z1 = (left[0] * right[2] - left[2] * right[0]) / determinant
            if not math.isfinite(z0) or not math.isfinite(z1):
                continue
            if all(
                row[0] * z0 + row[1] * z1 <= row[2] + tolerance
                for row in constraints
            ):
                candidates.append((z0, z1))
    if not candidates:
        return math.inf
    values = [
        q[0] * max(0.0, point[0])
        + q[1] * max(0.0, point[1])
        - slope[0] * point[0]
        - slope[1] * point[1]
        for point in candidates
    ]
    result = max(values)
    return float(result) if math.isfinite(result) else math.inf


def _float_singleton_requirement(
    *,
    q: float,
    slope: float,
    lower: float,
    upper: float,
) -> float:
    endpoints = [float(lower), float(upper)]
    if lower < 0.0 < upper:
        endpoints.append(0.0)
    result = max(
        q * max(0.0, point) - slope * point for point in endpoints
    )
    return float(result) if math.isfinite(result) else math.inf


def _compact_pair_generators(
    matrix: sp.csr_matrix,
    left: int,
    right: int,
    *,
    max_union_nnz: int,
) -> Tuple[np.ndarray, Tuple[List[float], List[float]]]:
    """Merge two CSR rows without allocating anything by matrix column count."""

    left_start = int(matrix.indptr[left])
    left_end = int(matrix.indptr[left + 1])
    right_start = int(matrix.indptr[right])
    right_end = int(matrix.indptr[right + 1])
    if (left_end - left_start) + (right_end - right_start) > 2 * max_union_nnz:
        raise _PairHullResourceCap(
            "selected pair row nnz exceeds compact union pre-cap"
        )

    left_indices = matrix.indices[left_start:left_end]
    left_data = matrix.data[left_start:left_end]
    right_indices = matrix.indices[right_start:right_end]
    right_data = matrix.data[right_start:right_end]
    columns: List[int] = []
    first: List[float] = []
    second: List[float] = []
    left_position = 0
    right_position = 0
    while left_position < left_indices.size or right_position < right_indices.size:
        if len(columns) >= max_union_nnz:
            raise _PairHullResourceCap(
                "selected pair compact union exceeds max_pair_union_nnz"
            )
        if (
            right_position >= right_indices.size
            or (
                left_position < left_indices.size
                and int(left_indices[left_position])
                < int(right_indices[right_position])
            )
        ):
            columns.append(int(left_indices[left_position]))
            first.append(float(left_data[left_position]))
            second.append(0.0)
            left_position += 1
        elif (
            left_position >= left_indices.size
            or int(right_indices[right_position])
            < int(left_indices[left_position])
        ):
            columns.append(int(right_indices[right_position]))
            first.append(0.0)
            second.append(float(right_data[right_position]))
            right_position += 1
        else:
            columns.append(int(left_indices[left_position]))
            first.append(float(left_data[left_position]))
            second.append(float(right_data[right_position]))
            left_position += 1
            right_position += 1
    return np.asarray(columns, dtype=np.int64), (first, second)


def _exact_q_pair(
    *,
    property_matrix: sp.csr_matrix,
    output_weight: sp.csr_matrix,
    rival: int,
    pair: Tuple[int, int],
) -> Tuple[Fraction, Fraction, int]:
    """Exact stored-binary64 accumulation of two entries of ``C @ W``."""

    first = _ZERO
    second = _ZERO
    term_visits = 0
    c_start = int(property_matrix.indptr[rival])
    c_end = int(property_matrix.indptr[rival + 1])
    for c_position in range(c_start, c_end):
        output = int(property_matrix.indices[c_position])
        c_value = Fraction.from_float(
            float(property_matrix.data[c_position])
        )
        w_start = int(output_weight.indptr[output])
        w_end = int(output_weight.indptr[output + 1])
        indices = output_weight.indices[w_start:w_end]
        data = output_weight.data[w_start:w_end]
        for which, neuron in enumerate(pair):
            offset = int(np.searchsorted(indices, neuron))
            if offset < indices.size and int(indices[offset]) == neuron:
                product = c_value * Fraction.from_float(float(data[offset]))
                if which == 0:
                    first += product
                else:
                    second += product
                term_visits += 1
    return first, second, term_visits


def build_property_pairhull_candidates(
    *,
    property_matrix: Any,
    output_weight: Any,
    preactivation_center: Any,
    preactivation_generators: Any,
    preactivation_error: Any,
    lower: Any,
    upper: Any,
    foundation_planes: Any,
    foundation_intercepts: Any,
    pair_budget: int,
    time_limit: float,
    deadline: Optional[float] = None,
    foundation_names: Optional[Sequence[str]] = None,
    row_topk: int = 8,
    max_selector_postings: int = 65536,
    max_rows_per_generator: int = 16,
    max_pair_pool: int = 8192,
    max_pair_updates: int = 200000,
    max_property_cells: int = 2_000_000,
    max_pair_union_nnz: int = 250000,
    max_total_compact_nnz: int = 1_000_000,
    max_exact_evaluations: int = 4096,
) -> PropertyPairHullCandidates:
    """Construct sparse PairHull alternatives without replacing foundations.

    ``foundation_planes`` may have shape ``[rivals, neurons]`` (one
    foundation) or ``[foundations, rivals, neurons]``.  Intercepts use the
    corresponding shape without the final neuron dimension.

    Every disabled, deadline, resource-cap, or unexpected-error path returns
    an empty batch.  Incomplete candidates are discarded and counted in the
    receipt; no partial batch is labelled complete.
    """

    started = time.monotonic()
    width = 0
    partial_candidates = 0
    base_receipt: Dict[str, Any] = {
        "schema": "act.property_pairhull.candidates.v1",
        "candidate_only": True,
        "proof_authority": (
            "stored_binary64_fraction_q+singleton_endpoints+"
            "exact_pairhull_four_phase+outward_intercept"
        ),
        "pair_selector_proof_authority": False,
        "foundation_contract": (
            "stored_intercept_contains_exact_singleton_requirements_"
            "for_supplied_stored_bounds"
        ),
        "foundation_rows_must_remain_retained": True,
        "global_pair_count": 0,
        "requested_pair_budget": (
            int(pair_budget)
            if isinstance(pair_budget, (Integral, np.integer))
            and not isinstance(pair_budget, (bool, np.bool_))
            else repr(pair_budget)
        ),
        "time_limit_seconds": (
            float(time_limit)
            if isinstance(time_limit, Real)
            and not isinstance(time_limit, (bool, np.bool_))
            else repr(time_limit)
        ),
    }

    try:
        if (
            isinstance(pair_budget, (bool, np.bool_))
            or not isinstance(pair_budget, (Integral, np.integer))
            or not 0 <= int(pair_budget) <= 8
        ):
            raise ValueError("pair_budget must be an integer in [0, 8]")
        budget = int(pair_budget)
        local_seconds = float(time_limit)
        if (
            isinstance(time_limit, (bool, np.bool_))
            or not math.isfinite(local_seconds)
            or local_seconds < 0.0
        ):
            raise ValueError("time_limit must be finite and nonnegative")
        if deadline is not None and not math.isfinite(float(deadline)):
            raise ValueError("deadline must be finite")
        integer_limits = {
            "row_topk": row_topk,
            "max_selector_postings": max_selector_postings,
            "max_rows_per_generator": max_rows_per_generator,
            "max_pair_pool": max_pair_pool,
            "max_pair_updates": max_pair_updates,
            "max_property_cells": max_property_cells,
            "max_pair_union_nnz": max_pair_union_nnz,
            "max_total_compact_nnz": max_total_compact_nnz,
            "max_exact_evaluations": max_exact_evaluations,
        }
        parsed_limits: Dict[str, int] = {}
        for name, raw in integer_limits.items():
            if (
                isinstance(raw, (bool, np.bool_))
                or not isinstance(raw, (Integral, np.integer))
                or int(raw) <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
            parsed_limits[name] = int(raw)

        center = np.asarray(preactivation_center, dtype=np.float64).reshape(-1)
        error = np.asarray(preactivation_error, dtype=np.float64).reshape(-1)
        lower_array = np.asarray(lower, dtype=np.float64).reshape(-1)
        upper_array = np.asarray(upper, dtype=np.float64).reshape(-1)
        generators = _as_canonical_csr(
            preactivation_generators,
            name="preactivation_generators",
        )
        C = _as_canonical_csr(property_matrix, name="property_matrix")
        weight = _as_canonical_csr(output_weight, name="output_weight")
        planes = np.asarray(foundation_planes, dtype=np.float64)
        intercepts = np.asarray(
            foundation_intercepts, dtype=np.float64
        )
        if planes.ndim == 2:
            planes = planes.reshape(1, planes.shape[0], planes.shape[1])
        if intercepts.ndim == 1:
            intercepts = intercepts.reshape(1, intercepts.shape[0])
        if planes.ndim != 3 or intercepts.ndim != 2:
            raise ValueError(
                "foundation planes/intercepts must be 2D/1D or 3D/2D"
            )
        foundations, rivals, width = (
            int(planes.shape[0]),
            int(planes.shape[1]),
            int(planes.shape[2]),
        )
        if (
            foundations <= 0
            or rivals <= 0
            or width <= 1
            or intercepts.shape != (foundations, rivals)
            or center.size != width
            or error.size != width
            or lower_array.size != width
            or upper_array.size != width
            or generators.shape[0] != width
            or C.shape[0] != rivals
            or C.shape[1] != weight.shape[0]
            or weight.shape[1] != width
        ):
            raise ValueError("PairHull candidate input shapes are inconsistent")
        if (
            not np.all(np.isfinite(center))
            or not np.all(np.isfinite(error))
            or np.any(error < 0.0)
            or not np.all(np.isfinite(lower_array))
            or not np.all(np.isfinite(upper_array))
            or np.any(lower_array > upper_array)
            or not np.all(np.isfinite(planes))
            or not np.all(np.isfinite(intercepts))
        ):
            raise ValueError(
                "PairHull candidate arrays are malformed or non-finite"
            )
        if foundation_names is None:
            names = tuple(f"foundation_{index}" for index in range(foundations))
        else:
            names = tuple(str(value) for value in foundation_names)
            if len(names) != foundations or any(not value for value in names):
                raise ValueError(
                    "foundation_names must contain one nonempty name per foundation"
                )
            if len(set(names)) != len(names):
                raise ValueError("foundation_names must be unique")

        property_cells = int(rivals * width)
        if property_cells > parsed_limits["max_property_cells"]:
            raise _PairHullResourceCap(
                "approximate property matrix exceeds max_property_cells"
            )
        # A float proxy freezes at most one (pair, foundation) proposal per
        # rival.  Exact work never searches the Cartesian product.
        exact_evaluations = int(rivals)
        if exact_evaluations > parsed_limits["max_exact_evaluations"]:
            raise _PairHullResourceCap(
                "requested exact PairHull evaluations exceed cap"
            )

        base_receipt.update(
            {
                "rivals": rivals,
                "preactivation_rows": width,
                "continuous_columns": int(generators.shape[1]),
                "generator_nnz": int(generators.nnz),
                "foundations": foundations,
                "foundation_names": list(names),
                "property_cells": property_cells,
                "max_property_cells": parsed_limits["max_property_cells"],
                "max_pair_union_nnz": parsed_limits["max_pair_union_nnz"],
                "max_total_compact_nnz": parsed_limits[
                    "max_total_compact_nnz"
                ],
                "max_exact_evaluations": parsed_limits[
                    "max_exact_evaluations"
                ],
                "preactivation_generators_sha256": _csr_sha256(generators),
                "property_matrix_sha256": _csr_sha256(C),
                "output_weight_sha256": _csr_sha256(weight),
                "preactivation_center_sha256": _array_sha256(
                    center, dtype=np.float64
                ),
                "preactivation_error_sha256": _array_sha256(
                    error, dtype=np.float64
                ),
                "stored_bounds_sha256": _array_sha256(
                    np.vstack([lower_array, upper_array]), dtype=np.float64
                ),
                "foundation_planes_sha256": _array_sha256(
                    planes, dtype=np.float64
                ),
                "foundation_intercepts_sha256": _array_sha256(
                    intercepts, dtype=np.float64
                ),
            }
        )
        if budget == 0 or local_seconds == 0.0:
            return _empty_result(
                width=width,
                status="disabled",
                started=started,
                receipt=base_receipt,
                whole_batch_complete=True,
            )

        stop_at = started + local_seconds
        if deadline is not None:
            stop_at = min(stop_at, float(deadline))
        if time.monotonic() >= stop_at:
            raise _PairHullDeadline("candidate deadline already exhausted")

        approximate_sparse = (C @ weight).tocsr()
        approximate_sparse.sum_duplicates()
        approximate_sparse.eliminate_zeros()
        if not np.all(np.isfinite(approximate_sparse.data)):
            raise ValueError("approximate C@W contains a non-finite value")
        # Only the small rival-by-neuron property matrix is materialized.
        # The potentially million-column preactivation G is never densified.
        approximate_q = np.asarray(
            approximate_sparse.toarray(), dtype=np.float64
        )
        selected_pairs, selector_receipt, score_records = _select_sparse_pairs(
            generators=generators,
            approximate_q=approximate_q,
            pair_budget=budget,
            row_topk=parsed_limits["row_topk"],
            max_selector_postings=parsed_limits["max_selector_postings"],
            max_rows_per_generator=parsed_limits[
                "max_rows_per_generator"
            ],
            max_pair_pool=parsed_limits["max_pair_pool"],
            max_pair_updates=parsed_limits["max_pair_updates"],
            stop_at=stop_at,
        )
        base_receipt["selector"] = selector_receipt
        base_receipt["global_pair_count"] = int(len(selected_pairs))
        if not selected_pairs:
            return _empty_result(
                width=width,
                status="no_eligible_shared_mixed_sign_pairs",
                started=started,
                receipt=base_receipt,
                whole_batch_complete=True,
            )

        projections: Dict[Tuple[int, int], Any] = {}
        projection_records: List[Dict[str, Any]] = []
        total_compact_nnz = 0
        for pair in selected_pairs:
            if time.monotonic() >= stop_at:
                raise _PairHullDeadline(
                    "deadline while building selected pair projections"
                )
            columns, compact = _compact_pair_generators(
                generators,
                pair[0],
                pair[1],
                max_union_nnz=parsed_limits["max_pair_union_nnz"],
            )
            total_compact_nnz += int(columns.size)
            if total_compact_nnz > parsed_limits["max_total_compact_nnz"]:
                raise _PairHullResourceCap(
                    "selected projections exceed max_total_compact_nnz"
                )
            projection = build_pairhull_projection(
                center=(center[pair[0]], center[pair[1]]),
                generators=compact,
                error=(error[pair[0]], error[pair[1]]),
                directions=DEFAULT_PAIRHULL_DIRECTIONS,
            )
            projections[pair] = projection
            projection_records.append(
                {
                    **score_records[pair],
                    "compact_generator_columns": int(columns.size),
                    "compact_generator_column_indices_sha256": _array_sha256(
                        columns, dtype=np.int64
                    ),
                    "source_affine_sha256": projection.source_affine_sha256,
                    "constraints_sha256": projection.constraints_sha256,
                    "rowwise_error_hex": [
                        float(error[pair[0]]).hex(),
                        float(error[pair[1]]).hex(),
                    ],
                    "stored_supports_hex": [
                        value.hex() for value in projection.stored_supports
                    ],
                }
            )
        base_receipt["selected_projections"] = projection_records
        base_receipt["selected_projections_sha256"] = _canonical_sha256(
            projection_records
        )
        base_receipt["total_compact_generator_columns"] = int(
            total_compact_nnz
        )

        # First freeze at most one proposal per rival with a float-only proxy.
        # An exact rejection does not trigger a second-choice search: this is
        # the key V1 stop-loss which bounds exact beta calls by ``rivals``.
        proposals: Dict[int, Dict[str, Any]] = {}
        proposal_records: List[Dict[str, Any]] = []
        for rival in range(rivals):
            if time.monotonic() >= stop_at:
                raise _PairHullDeadline(
                    "deadline during float proposal selection"
                )
            best_proxy: Optional[Tuple[float, int, int, int]] = None
            best_payload: Optional[Dict[str, Any]] = None
            for pair_index, pair in enumerate(selected_pairs):
                q_proxy = (
                    float(approximate_q[rival, pair[0]]),
                    float(approximate_q[rival, pair[1]]),
                )
                if not (
                    (q_proxy[0] > 0.0 and q_proxy[1] < 0.0)
                    or (q_proxy[0] < 0.0 and q_proxy[1] > 0.0)
                ):
                    continue
                projection = projections[pair]
                for foundation in range(foundations):
                    slope_proxy = (
                        float(planes[foundation, rival, pair[0]]),
                        float(planes[foundation, rival, pair[1]]),
                    )
                    singleton0 = _float_singleton_requirement(
                        q=q_proxy[0],
                        slope=slope_proxy[0],
                        lower=float(lower_array[pair[0]]),
                        upper=float(upper_array[pair[0]]),
                    )
                    singleton1 = _float_singleton_requirement(
                        q=q_proxy[1],
                        slope=slope_proxy[1],
                        lower=float(lower_array[pair[1]]),
                        upper=float(upper_array[pair[1]]),
                    )
                    beta_proxy = _float_pair_beta(
                        projection,
                        q=q_proxy,
                        slope=slope_proxy,
                    )
                    gain_proxy = singleton0 + singleton1 - beta_proxy
                    if not math.isfinite(gain_proxy) or gain_proxy <= 0.0:
                        continue
                    # Maximize proxy intercept reduction; deterministic ties
                    # prefer the earlier foundation and lexicographic pair.
                    key = (
                        float(gain_proxy),
                        -int(foundation),
                        -int(pair[0]),
                        -int(pair[1]),
                    )
                    if best_proxy is None or key > best_proxy:
                        best_proxy = key
                        best_payload = {
                            "rival_id": int(rival),
                            "foundation": int(foundation),
                            "pair": pair,
                            "pair_index": int(pair_index),
                            "proxy_gain": float(gain_proxy),
                            "proxy_beta": float(beta_proxy),
                        }
            if best_payload is not None:
                proposals[rival] = best_payload
                proposal_records.append(
                    {
                        "rival_id": int(rival),
                        "foundation_index": int(best_payload["foundation"]),
                        "pair": [
                            int(best_payload["pair"][0]),
                            int(best_payload["pair"][1]),
                        ],
                        "proxy_intercept_reduction_hex": float(
                            best_payload["proxy_gain"]
                        ).hex(),
                        "proxy_pair_beta_hex": float(
                            best_payload["proxy_beta"]
                        ).hex(),
                        "selection_proof_authority": False,
                    }
                )
        base_receipt.update(
            {
                "global_pair_count": int(len(selected_pairs)),
                "global_pairs_disjoint": True,
                "float_proxy_algorithm": (
                    "polygon_vertex_and_relu_axis_intersections_v1"
                ),
                "float_proxy_proof_authority": False,
                "float_proposals": int(len(proposals)),
                "float_proposal_records": proposal_records,
                "float_proposal_records_sha256": _canonical_sha256(
                    proposal_records
                ),
                "exact_audit_policy": (
                    "one_frozen_float_proposal_per_rival_no_backtracking"
                ),
            }
        )
        if not proposals:
            return _empty_result(
                width=width,
                status="no_positive_float_proxy_proposal",
                started=started,
                receipt=base_receipt,
                whole_batch_complete=True,
            )

        best_by_rival: Dict[int, Dict[str, Any]] = {}
        exact_q_term_visits = 0
        exact_beta_evaluations = 0
        strict_improvements_considered = 0
        exact_rejected_proposals = 0
        for rival in sorted(proposals):
            proposal = proposals[rival]
            pair = proposal["pair"]
            foundation = int(proposal["foundation"])
            projection = projections[pair]
            if time.monotonic() >= stop_at:
                raise _PairHullDeadline(
                    "deadline during exact rival reconstruction"
                )
            q0, q1, visits = _exact_q_pair(
                property_matrix=C,
                output_weight=weight,
                rival=rival,
                pair=pair,
            )
            exact_q_term_visits += visits
            if not (
                (q0 > 0 and q1 < 0) or (q0 < 0 and q1 > 0)
            ):
                exact_rejected_proposals += 1
                continue
            slope0 = Fraction.from_float(
                float(planes[foundation, rival, pair[0]])
            )
            slope1 = Fraction.from_float(
                float(planes[foundation, rival, pair[1]])
            )
            lower0 = Fraction.from_float(float(lower_array[pair[0]]))
            upper0 = Fraction.from_float(float(upper_array[pair[0]]))
            lower1 = Fraction.from_float(float(lower_array[pair[1]]))
            upper1 = Fraction.from_float(float(upper_array[pair[1]]))
            requirement0 = _singleton_requirement(
                q=q0,
                slope=slope0,
                lower=lower0,
                upper=upper0,
            )
            requirement1 = _singleton_requirement(
                q=q1,
                slope=slope1,
                lower=lower1,
                upper=upper1,
            )
            pairhull = exact_pairhull_beta(
                projection,
                q=(q0, q1),
                candidate_slope=(slope0, slope1),
            )
            exact_beta_evaluations += 1
            if time.monotonic() >= stop_at:
                raise _PairHullDeadline(
                    "deadline crossed during exact PairHull beta audit"
                )
            old_stored = float(intercepts[foundation, rival])
            old_exact = Fraction.from_float(old_stored)
            reconstructed_exact = (
                old_exact
                - requirement0
                - requirement1
                + pairhull.beta_exact
            )
            reconstructed_stored = _outward_float(
                reconstructed_exact,
                name=(
                    f"pairhull intercept[{foundation},{rival},"
                    f"{pair[0]},{pair[1]}]"
                ),
            )
            reconstructed_stored_exact = Fraction.from_float(
                reconstructed_stored
            )
            if reconstructed_stored_exact < reconstructed_exact:
                raise AssertionError(
                    "PairHull candidate intercept was not outward"
                )
            if not reconstructed_stored < old_stored:
                exact_rejected_proposals += 1
                continue
            strict_improvements_considered += 1
            stored_gain = old_exact - reconstructed_stored_exact
            plane = np.asarray(
                planes[foundation, rival, :],
                dtype=np.float64,
            )
            candidate_record: Dict[str, Any] = {
                "rival_id": int(rival),
                "foundation_index": int(foundation),
                "foundation_name": names[foundation],
                "pair": [int(pair[0]), int(pair[1])],
                "candidate_selection_proof_authority": False,
                "candidate_selection_metric": (
                    "frozen_maximum_float_proxy_intercept_reduction"
                ),
                "proxy_intercept_reduction_hex": float(
                    proposal["proxy_gain"]
                ).hex(),
                "q_exact": [
                    _fraction_text(q0),
                    _fraction_text(q1),
                ],
                "stored_pair_slopes_hex": [
                    float(planes[foundation, rival, pair[0]]).hex(),
                    float(planes[foundation, rival, pair[1]]).hex(),
                ],
                "singleton_requirements_exact": [
                    _fraction_text(requirement0),
                    _fraction_text(requirement1),
                ],
                "foundation_intercept_hex": old_stored.hex(),
                "foundation_intercept_exact": _fraction_text(old_exact),
                "pair_beta_exact": _fraction_text(
                    pairhull.beta_exact
                ),
                "reconstructed_intercept_exact": _fraction_text(
                    reconstructed_exact
                ),
                "reconstructed_intercept_hex": (
                    reconstructed_stored.hex()
                ),
                "stored_intercept_reduction_exact": _fraction_text(
                    stored_gain
                ),
                "outward_intercept_validated": True,
                "candidate_plane_sha256": _array_sha256(
                    plane, dtype=np.float64
                ),
                "source_affine_sha256": (
                    projection.source_affine_sha256
                ),
                "constraints_sha256": projection.constraints_sha256,
                "exact_pairhull_receipt": pairhull.receipt,
            }
            candidate_record["record_sha256"] = _canonical_sha256(
                candidate_record
            )
            best_by_rival[rival] = {
                "foundation": foundation,
                "pair": pair,
                "plane": plane.copy(),
                "intercept": reconstructed_stored,
                "record": candidate_record,
            }
            partial_candidates = len(best_by_rival)

        if not best_by_rival:
            base_receipt.update(
                {
                    "exact_q_term_visits": int(exact_q_term_visits),
                    "exact_beta_evaluations": int(exact_beta_evaluations),
                    "exact_rejected_proposals": int(
                        exact_rejected_proposals
                    ),
                    "strict_improvements_considered": int(
                        strict_improvements_considered
                    ),
                }
            )
            return _empty_result(
                width=width,
                status="no_strict_intercept_improvement",
                started=started,
                receipt=base_receipt,
                whole_batch_complete=True,
            )

        ordered = [best_by_rival[key] for key in sorted(best_by_rival)]
        rival_ids = np.asarray(sorted(best_by_rival), dtype=np.int64)
        foundation_indices = np.asarray(
            [entry["foundation"] for entry in ordered], dtype=np.int64
        )
        pair_indices = np.asarray(
            [entry["pair"] for entry in ordered], dtype=np.int64
        ).reshape(-1, 2)
        candidate_planes = np.vstack(
            [entry["plane"] for entry in ordered]
        ).astype(np.float64, copy=False)
        candidate_intercepts = np.asarray(
            [entry["intercept"] for entry in ordered], dtype=np.float64
        )
        candidate_records = [entry["record"] for entry in ordered]
        complete_receipt = {
            **base_receipt,
            "status": "generated",
            "whole_batch_complete": True,
            "partial_candidates_discarded": 0,
            "selected_candidates": int(rival_ids.size),
            "at_most_one_candidate_per_rival": True,
            "foundation_rows_retained_by_caller": True,
            "exact_q_term_visits": int(exact_q_term_visits),
            "exact_beta_evaluations": int(exact_beta_evaluations),
            "exact_rejected_proposals": int(exact_rejected_proposals),
            "strict_improvements_considered": int(
                strict_improvements_considered
            ),
            "candidate_records": candidate_records,
            "candidate_records_sha256": _canonical_sha256(
                candidate_records
            ),
            "selected_rival_ids_sha256": _array_sha256(
                rival_ids, dtype=np.int64
            ),
            "selected_foundation_indices_sha256": _array_sha256(
                foundation_indices, dtype=np.int64
            ),
            "selected_pair_indices_sha256": _array_sha256(
                pair_indices, dtype=np.int64
            ),
            "candidate_planes_sha256": _array_sha256(
                candidate_planes, dtype=np.float64
            ),
            "candidate_intercepts_sha256": _array_sha256(
                candidate_intercepts, dtype=np.float64
            ),
            "elapsed_seconds": float(max(0.0, time.monotonic() - started)),
        }
        finalized_receipt = (
            finalize_property_pairhull_candidates_receipt(
                complete_receipt
            )
        )
        if time.monotonic() >= stop_at:
            raise _PairHullDeadline(
                "deadline crossed during final candidate materialization"
            )
        return PropertyPairHullCandidates(
            rival_ids=rival_ids,
            foundation_indices=foundation_indices,
            pair_indices=pair_indices,
            planes=candidate_planes,
            intercepts=candidate_intercepts,
            receipt=finalized_receipt,
        )
    except _PairHullDeadline as exc:
        return _empty_result(
            width=width,
            status="deadline_fallback_foundations",
            started=started,
            receipt=base_receipt,
            whole_batch_complete=False,
            partial_candidates_discarded=partial_candidates,
            error=exc,
        )
    except _PairHullResourceCap as exc:
        return _empty_result(
            width=width,
            status="resource_cap_fallback_foundations",
            started=started,
            receipt=base_receipt,
            whole_batch_complete=False,
            partial_candidates_discarded=partial_candidates,
            error=exc,
        )
    except Exception as exc:
        # Candidate generation is optional.  Any unexpected failure discards
        # every partially built alternative and leaves foundations untouched.
        return _empty_result(
            width=width,
            status="error_fallback_foundations",
            started=started,
            receipt=base_receipt,
            whole_batch_complete=False,
            partial_candidates_discarded=partial_candidates,
            error=exc,
        )


__all__ = [
    "PropertyPairHullCandidates",
    "build_property_pairhull_candidates",
    "finalize_property_pairhull_candidates_receipt",
    "verify_pairhull_candidate_receipt",
    "verify_property_pairhull_candidates_receipt",
]
