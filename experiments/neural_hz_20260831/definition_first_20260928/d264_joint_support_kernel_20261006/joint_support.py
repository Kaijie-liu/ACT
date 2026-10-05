"""Reliable common-source residual support; no HZ/model/source certification.

The caller supplies the SAME ordered, radius-normalized source axes for
rho = delta + a*xi and e = e0 + b*xi, with xi in [-1, 1].  Correlations between
axes may be retained in the caller's H; using their box here only weakens the
bound.  Equal array shapes do not authenticate that source contract.

One vector construction forms four nonnegative statistics, not four affine
direction vectors.  Their sums share one NumPy reduction.  The directed error
certificate is the frozen D259 gamma_n/absolute-sum argument; nonnegative
statistics make each rounded sum its own rounded absolute sum.  Final scalar
encoding reuses D259's outward 2**-256 lattice, never a zero tolerance.

All computations are opt-in, charge one caller-owned meter before work and
allocation, and leave inputs untouched.  Costs include temporary allocations,
not only peak storage.  Ordinary schema rejection does not poison that meter;
its resource failures retain the meter's own sticky semantics and no charges
are refunded.  No partial Support is returned.  Source construction, native
rows, serialization, GPU execution and full physical costs are outside this
small arithmetic contract.
"""

from dataclasses import dataclass
import math

import numpy as np

from experiments.neural_hz_20260831.definition_first_20260928.d259_shared_difference_source_20261006 import source_bounds as sb


Interval = sb.Interval
Rejected = sb.Rejected


@dataclass(frozen=True)
class Support:
    upper: float
    independent_upper: float
    raw_upper: float
    radius_a_upper: float
    radius_b_upper: float
    credit_same_lower: float
    credit_opp_lower: float
    constant_abs_upper: float
    dimension: int


def _full_interval(value, ndim, budget):
    """Do not use the owned-token fast path for these public operands."""
    sb._pay(budget, 24, 24)
    if type(value) is not Interval:
        raise Rejected("D259 Interval required")
    lo, hi = value.lo, value.hi
    if (type(lo) is not np.ndarray or type(hi) is not np.ndarray
            or lo.dtype != np.float64 or hi.dtype != np.float64
            or lo.shape != hi.shape or lo.ndim != ndim):
        raise Rejected("matching float64 interval endpoints of required dimension")
    if ndim == 1 and lo.size > 65536:
        raise Rejected("joint source reduction exceeds fixed supported width")
    # These complete scans check finite payloads and D259's conservative
    # 512-bit point range even for an internally owned Interval.  They neither
    # retain the operand nor mark caller arrays as owned/read-only.
    sb._borrow_point_array(lo, budget)
    sb._borrow_point_array(hi, budget)
    sb._pay(budget, 2 * lo.size + 12, lo.size + 12)
    if not (lo <= hi).all():
        raise Rejected("unordered joint source interval")
    return value


def _checked_scalar(value, budget):
    sb._pay(budget, 12, 8)
    value = float(value)
    if (not math.isfinite(value) or abs(value) > sb._MAXIMUM
            or (value != 0.0 and abs(value) < sb._POINT_MIN)):
        raise Rejected("joint support scalar exceeds finite 512-bit support")
    return value


def _endpoint(value, upward, budget):
    sb._pay(budget, 8, 8)
    if not math.isfinite(value):
        raise Rejected("nonfinite joint support arithmetic")
    encoded = sb._round(np.array(value, dtype=np.float64), upward, budget)
    return _checked_scalar(float(encoded), budget)


def _up_add(left, right, budget):
    sb._pay(budget, 12, 8)
    raw = left + right
    # Equality of opposite stored operands certifies an exact zero, as in
    # D259._add.  This is not a tolerance or a common-source inference.
    if left == -right:
        return 0.0
    return _endpoint(raw, True, budget)


def _twice(value, budget):
    # Binary scaling by two is exact for supported finite input operands.
    # Reject overflow/unsupported output instead of weakening the certificate.
    sb._pay(budget, 8, 4)
    return _checked_scalar(math.ldexp(value, 1), budget)


def _statistics(a, b, budget):
    """Return Ra/E upper bounds and same/opposite overlap lower bounds."""
    n = a.lo.size
    sb._pay(budget, 40 * n + 128, 16 * n + 128)
    values = np.empty((n, 4), dtype=np.float64)
    scratch = np.empty(n, dtype=np.float64)
    np.absolute(a.lo, out=values[:, 0])
    np.absolute(a.hi, out=scratch)
    np.maximum(values[:, 0], scratch, out=values[:, 0])
    np.absolute(b.lo, out=values[:, 1])
    np.absolute(b.hi, out=scratch)
    np.maximum(values[:, 1], scratch, out=values[:, 1])

    # inf |[lo,hi]| = max(lo,-hi,0); extrema and negation are exact.
    amin = np.negative(a.hi)
    bmin = np.negative(b.hi)
    np.maximum(amin, a.lo, out=amin)
    np.maximum(bmin, b.lo, out=bmin)
    np.maximum(amin, 0.0, out=amin)
    np.maximum(bmin, 0.0, out=bmin)
    np.multiply(bmin, 2.0, out=bmin)
    # bmin <= 2**512, well inside binary64 range; multiplying by two is exact.
    # The retained credit is <= amin <= D259's supported endpoint maximum.
    np.minimum(amin, bmin, out=amin)

    apos, aneg = a.lo > 0.0, a.hi < 0.0
    bpos, bneg = b.lo > 0.0, b.hi < 0.0
    same = np.logical_and(apos, bpos)
    temporary = np.logical_and(aneg, bneg)
    np.logical_or(same, temporary, out=same)
    opposite = np.logical_and(apos, bneg)
    np.logical_and(aneg, bpos, out=temporary)
    np.logical_or(opposite, temporary, out=opposite)
    values[:, 2:] = 0.0
    np.copyto(values[:, 2], amin, where=same)
    np.copyto(values[:, 3], amin, where=opposite)

    # Four statistics, one reduction, no four source-direction coefficient
    # vectors.  Each column is nonnegative.  The error proof below is D259's
    # _fast_sum_endpoint with Ahat == total, saving its extra abs/reduction.
    sb._pay(budget, 4 * n + 640, 256)
    totals = np.sum(values, axis=0, dtype=np.float64)
    grid = math.ldexp(1.0, -sb._GRID_EXP)
    nu = math.ldexp(float(n), -53)
    gamma = math.nextafter(nu / (1.0 - nu), math.inf)
    answers = []
    for index in range(4):
        total = float(totals[index])
        if not math.isfinite(total) or total < 0.0:
            raise Rejected("nonfinite joint support reduction")
        if not n or total == 0.0:
            answers.append(0.0)
            continue
        spacing = math.nextafter(total, math.inf) - total
        abs_error = math.nextafter(float(n) * max(spacing, grid), math.inf)
        abs_upper = math.nextafter(total + abs_error, math.inf)
        product_error = math.nextafter(gamma * abs_upper, math.inf)
        error = math.nextafter(product_error + float(n) * grid, math.inf)
        upward = index < 2
        raw = math.nextafter(total + error if upward else total - error,
                             math.inf if upward else -math.inf)
        bound = _endpoint(raw, upward, budget)
        answers.append(bound if upward else max(0.0, bound))
    return tuple(answers)


def joint_support(a, b, delta, e0, *, budget, enabled=False):
    """Bound sup(|rho| + 2*ReLU(e)) for the declared common source box.

    Interval coefficient uncertainty is covered uniformly, including sign-
    crossing coefficients (which receive no overlap credit).  The caller must
    already have absorbed every source radius and center into these operands.
    The result is a bound, not a reconstructed input or source certificate.
    """
    if not sb._on(enabled):
        return None
    sb._pay(budget, 128, 128)
    a = _full_interval(a, 1, budget)
    b = _full_interval(b, 1, budget)
    delta = _full_interval(delta, 0, budget)
    e0 = _full_interval(e0, 0, budget)
    if a.lo.shape != b.lo.shape:
        raise Rejected("a and b require the same complete ordered source axis")
    ra, eb, same, opposite = _statistics(a, b, budget)

    sb._pay(budget, 128, 128)
    delta_lo, delta_hi = float(delta.lo), float(delta.hi)
    e0_hi = float(e0.hi)
    magnitude = max(abs(delta_lo), abs(delta_hi))
    base = _up_add(ra, magnitude, budget)
    twice_e0 = _twice(e0_hi, budget)
    twice_eb = _twice(eb, budget)
    plus = _up_add(delta_hi, twice_e0, budget)
    plus = _up_add(plus, ra, budget)
    plus = _up_add(plus, twice_eb, budget)
    plus = _up_add(plus, -_twice(opposite, budget), budget)
    minus = _up_add(-delta_lo, twice_e0, budget)
    minus = _up_add(minus, ra, budget)
    minus = _up_add(minus, twice_eb, budget)
    minus = _up_add(minus, -_twice(same, budget), budget)
    raw = max(base, plus, minus)
    positive_e = max(_up_add(e0_hi, eb, budget), 0.0)
    independent = _up_add(base, _twice(positive_e, budget), budget)
    upper = min(raw, independent)
    sb._pay(budget, 48, 32)
    return Support(upper, independent, raw, ra, eb, same, opposite,
                   magnitude, int(a.lo.size))
