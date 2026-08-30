"""Proof-obligation toys for a possible GPU query-dual replay.

This module is deliberately isolated from the production verifier.  It grants
no proof authority and does not call CUDA.  Its two purposes are:

* show why an arbitrary, untrusted affine coefficient claimed by a GPU cannot
  be made useful with only a norm/triangle residual bound; and
* audit the algebra behind a stronger CPU alternative which computes one
  nominal affine product but compresses the roundoff guard to a matrix-vector
  precomputation plus one dot per query.

All audit comparisons use exact :class:`fractions.Fraction` arithmetic over
the stored binary64 inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, Tuple

import numpy as np


_U = Fraction(1, 2**53)
_ETA = Fraction(1, 2**1074)


def _f64(value: Any, *, name: str, ndim: int) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _fractions(value: np.ndarray) -> Tuple[Fraction, ...]:
    return tuple(Fraction.from_float(float(item)) for item in value.reshape(-1))


def _fraction_matrix(value: np.ndarray) -> Tuple[Tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Fraction.from_float(float(item)) for item in row)
        for row in value
    )


def _fraction_to_float_up(value: Fraction) -> float:
    nearest = float(value)
    if not math.isfinite(nearest):
        raise OverflowError("Fraction does not fit in binary64")
    if Fraction.from_float(nearest) < value:
        nearest = float(np.nextafter(np.float64(nearest), np.float64(math.inf)))
    if not math.isfinite(nearest):
        raise OverflowError("upward Fraction conversion overflowed")
    return nearest


@dataclass(frozen=True)
class DenseCompressedGuardAudit:
    """Exact audit of one CPU nominal dense product's compressed guard."""

    nominal: np.ndarray
    exact_product: Tuple[Tuple[Fraction, ...], ...]
    weighted_actual_error: Tuple[Fraction, ...]
    weighted_guard: Tuple[Fraction, ...]
    full_weighted_mass: Tuple[Fraction, ...]
    compressed_weighted_mass: Tuple[Fraction, ...]
    operation_count: int
    proof_authority: bool = False


def audit_dense_compressed_roundoff_guard(
    coefficients: Any,
    weight: Any,
    predecessor_max_abs: Any,
) -> DenseCompressedGuardAudit:
    """Audit a scalar box-absorption guard without ``abs(A) @ abs(W)``.

    Let ``N = fl(A @ W)`` and let ``m`` be the componentwise maximum absolute
    value of the predecessor box.  The ordinary componentwise mass is

    ``S_ij = sum_k |A_ik| |W_kj|``.

    The scalar mass needed after box absorption can be rearranged exactly:

    ``sum_j m_j S_ij = sum_k |A_ik| (sum_j |W_kj| m_j)``.

    Thus a production CPU implementation can outward-compute
    ``s_k = sum_j |W_kj|m_j`` once per layer/snapshot and use ``|A| @ s`` per
    query, instead of a second query-by-weight matrix product.  This toy uses
    Fractions so the rearrangement and the guard can be checked exactly.
    """

    a = _f64(coefficients, name="coefficients", ndim=2)
    w = _f64(weight, name="weight", ndim=2)
    m = _f64(predecessor_max_abs, name="predecessor_max_abs", ndim=1)
    if a.shape[1] != w.shape[0] or w.shape[1] != m.size:
        raise ValueError("dense dimensions do not agree")
    if np.any(m < 0.0):
        raise ValueError("predecessor_max_abs must be nonnegative")

    nominal = np.ascontiguousarray(np.asarray(a @ w, dtype=np.float64))
    if not np.all(np.isfinite(nominal)):
        raise ValueError("nominal dense product is non-finite")

    af = _fraction_matrix(a)
    wf = _fraction_matrix(w)
    mf = _fractions(m)
    exact_rows = []
    actual_errors = []
    full_masses = []
    compressed_masses = []
    guards = []
    operations = 2 * int(a.shape[1]) + 2
    product = operations * _U
    if product >= Fraction(1, 2):
        raise ValueError("operation count is too large for the gamma model")
    gamma = product / (1 - product)
    underflow = operations * _ETA

    precomputed = tuple(
        sum((abs(wf[k][j]) * mf[j] for j in range(w.shape[1])), Fraction(0))
        for k in range(w.shape[0])
    )
    max_abs_sum = sum(mf, Fraction(0))

    for row_index, row in enumerate(af):
        exact = tuple(
            sum(
                (row[k] * wf[k][j] for k in range(w.shape[0])),
                Fraction(0),
            )
            for j in range(w.shape[1])
        )
        exact_rows.append(exact)
        actual_errors.append(
            sum(
                (
                    abs(
                        Fraction.from_float(float(nominal[row_index, j]))
                        - exact[j]
                    )
                    * mf[j]
                    for j in range(w.shape[1])
                ),
                Fraction(0),
            )
        )
        full_mass = sum(
            (
                mf[j]
                * sum(
                    (abs(row[k]) * abs(wf[k][j]) for k in range(w.shape[0])),
                    Fraction(0),
                )
                for j in range(w.shape[1])
            ),
            Fraction(0),
        )
        compressed_mass = sum(
            (abs(row[k]) * precomputed[k] for k in range(w.shape[0])),
            Fraction(0),
        )
        if full_mass != compressed_mass:
            raise AssertionError("exact mass rearrangement failed")
        full_masses.append(full_mass)
        compressed_masses.append(compressed_mass)
        guards.append(gamma * compressed_mass + underflow * max_abs_sum)

    nominal.setflags(write=False)
    return DenseCompressedGuardAudit(
        nominal=nominal,
        exact_product=tuple(exact_rows),
        weighted_actual_error=tuple(actual_errors),
        weighted_guard=tuple(guards),
        full_weighted_mass=tuple(full_masses),
        compressed_weighted_mass=tuple(compressed_masses),
        operation_count=operations,
    )


@dataclass(frozen=True)
class UntrustedDenseClaimAudit:
    """Exact lower-bound audit for an arbitrary claimed predecessor adjoint."""

    claimed_lower: Fraction
    zero_claim_lower: Fraction
    exact_box_minimum: Fraction
    claim_radius_penalty: Fraction
    triangle_mass_penalty: Fraction
    proof_authority: bool = False


def audit_untrusted_dense_claim(
    coefficients: Any,
    weight: Any,
    bias: Any,
    predecessor_lower: Any,
    predecessor_upper: Any,
    claimed_predecessor: Any,
) -> UntrustedDenseClaimAudit:
    """Soundly correct an entirely untrusted affine coefficient claim.

    For ``y = W x + b``, incoming row ``a``, and arbitrary claim ``v``, the
    exact recurrence residual is ``e = aW-v``.  Without computing ``aW``, only
    the triangle enclosure

    ``|e| <= |v| + |a||W|``

    is available.  Center/radius box absorption yields a sound bound, but after
    concretising ``v`` it equals the zero-claim interval bound minus
    ``2 |v| radius``.  Therefore this certificate family can never make an
    untrusted GPU nominal useful; its optimum is ``v=0``.
    """

    a = _f64(coefficients, name="coefficients", ndim=1)
    w = _f64(weight, name="weight", ndim=2)
    b = _f64(bias, name="bias", ndim=1)
    lower = _f64(predecessor_lower, name="predecessor_lower", ndim=1)
    upper = _f64(predecessor_upper, name="predecessor_upper", ndim=1)
    claim = _f64(
        claimed_predecessor, name="claimed_predecessor", ndim=1
    )
    if (
        a.size != w.shape[0]
        or b.size != a.size
        or lower.size != w.shape[1]
        or upper.shape != lower.shape
        or claim.size != lower.size
    ):
        raise ValueError("dense claim dimensions do not agree")
    if np.any(lower > upper):
        raise ValueError("predecessor box is empty")

    af = _fractions(a)
    wf = _fraction_matrix(w)
    bf = _fractions(b)
    lf = _fractions(lower)
    uf = _fractions(upper)
    vf = _fractions(claim)
    center = tuple((lo + hi) / 2 for lo, hi in zip(lf, uf))
    radius = tuple((hi - lo) / 2 for lo, hi in zip(lf, uf))
    exact_coefficient = tuple(
        sum((af[i] * wf[i][j] for i in range(a.size)), Fraction(0))
        for j in range(w.shape[1])
    )
    affine_center = sum(
        (exact_coefficient[j] * center[j] for j in range(lower.size)),
        Fraction(0),
    )
    bias_value = sum((af[i] * bf[i] for i in range(a.size)), Fraction(0))
    triangle_mass = sum(
        (
            abs(af[i])
            * sum(
                (abs(wf[i][j]) * radius[j] for j in range(lower.size)),
                Fraction(0),
            )
            for i in range(a.size)
        ),
        Fraction(0),
    )
    claim_radius = sum(
        (abs(vf[j]) * radius[j] for j in range(lower.size)),
        Fraction(0),
    )
    zero_claim = bias_value + affine_center - triangle_mass
    claimed = zero_claim - 2 * claim_radius
    exact_minimum = (
        bias_value
        + affine_center
        - sum(
            (
                abs(exact_coefficient[j]) * radius[j]
                for j in range(lower.size)
            ),
            Fraction(0),
        )
    )
    if claimed > exact_minimum or claimed > zero_claim:
        raise AssertionError("untrusted-claim triangle audit is not sound")
    return UntrustedDenseClaimAudit(
        claimed_lower=claimed,
        zero_claim_lower=zero_claim,
        exact_box_minimum=exact_minimum,
        claim_radius_penalty=claim_radius,
        triangle_mass_penalty=triangle_mass,
    )


@dataclass(frozen=True)
class FractionReluUpperLine:
    slope: float
    intercept: float
    proof_authority: bool = False


def fraction_relu_upper_line(lower: float, upper: float) -> FractionReluUpperLine:
    """Construct and exactly audit one ambiguous-ReLU binary64 upper line."""

    lower = float(lower)
    upper = float(upper)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("ReLU endpoints must be finite")
    if not lower < 0.0 < upper:
        raise ValueError("ReLU endpoint pair must be ambiguous")
    lf = Fraction.from_float(lower)
    uf = Fraction.from_float(upper)
    slope = float(uf / (uf - lf))
    sf = Fraction.from_float(slope)
    required = max(Fraction(0), -sf * lf, (Fraction(1) - sf) * uf)
    intercept = _fraction_to_float_up(required)
    bf = Fraction.from_float(intercept)
    if sf * lf + bf < 0 or bf < 0 or sf * uf + bf < uf:
        raise AssertionError("stored binary64 ReLU line misses an endpoint")
    return FractionReluUpperLine(slope=slope, intercept=intercept)


def validate_fraction_relu_upper_line(
    lower: float,
    upper: float,
    slope: float,
    intercept: float,
) -> bool:
    try:
        lf = Fraction.from_float(float(lower))
        uf = Fraction.from_float(float(upper))
        sf = Fraction.from_float(float(slope))
        bf = Fraction.from_float(float(intercept))
        return bool(
            math.isfinite(float(slope))
            and math.isfinite(float(intercept))
            and lf < 0 < uf
            and sf >= 0
            and sf <= 1
            and sf * lf + bf >= 0
            and bf >= 0
            and sf * uf + bf >= uf
        )
    except (OverflowError, TypeError, ValueError):
        return False


__all__ = [
    "DenseCompressedGuardAudit",
    "FractionReluUpperLine",
    "UntrustedDenseClaimAudit",
    "audit_dense_compressed_roundoff_guard",
    "audit_untrusted_dense_claim",
    "fraction_relu_upper_line",
    "validate_fraction_relu_upper_line",
]
