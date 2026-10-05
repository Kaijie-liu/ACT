"""Outward source ranges and additive Conv ledgers, not a verification domain.

All public computations are explicitly opt-in and charge one caller-owned
``budget.charge(amount, entries=...)`` before array work/allocation.  Entries
are cumulative numeric/metadata allocations, not just retained arrays.  The
caller owns source identity, full graph coverage, BN epsilon identities and
physical peak accounting.  No model, solver, phase search or native HZ is used.

Basic interval operations enclose elementary results immediately; fused sums
instead pay a proved gamma_n times outward absolute-sum error.  Tiny
enclosure endpoints are rounded outward to the 2**-256 lattice: lower uses
floor(2**256*x)*2**-256, upper uses ceil, only when |x|<2**-204.
These are uniform directed encodings, never a zero tolerance.  This prevents
ordinary zero rounding from creating 1075-bit subnormal evidence.  Stored
point coefficients are not midpoint substitutions.  Fraction is used ONLY
for the small, explicitly supplied rational range endpoints, never weights.
Interval, ConvLedger and BNCarrier ownership tokens are trusted in-memory
results of their constructors, not serialization certificates.  Caller-forced
mutation of owned read-only arrays is unsupported.  External Interval arrays
receive complete checks.  A Conv ledger is not permission to subtract terms
from an arbitrary tight range.
"""

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np


class Rejected(ValueError):
    pass


_TOKEN = object()
_GRID_EXP = 256
_TINY = math.ldexp(1.0, -204)
_POINT_MIN = math.ldexp(1.0, -458)
_MAXIMUM = math.ldexp(1.0, 511)


def _on(enabled):
    if type(enabled) is not bool:
        raise Rejected("enabled must be bool")
    return enabled


def _pay(budget, work=0, entries=0):
    if not callable(getattr(budget, "charge", None)):
        raise Rejected("one caller-owned charge(amount, entries=...) meter required")
    budget.charge(int(work), entries=int(entries))


@dataclass(frozen=True)
class Interval:
    lo: np.ndarray
    hi: np.ndarray
    _token: object = None


@dataclass(frozen=True)
class ConvLedger:
    bounds: Interval
    mass_positive: Interval
    mass_negative: Interval
    input_bounds: Interval
    weights: np.ndarray
    bias: np.ndarray
    padding: tuple
    valid_mask: object
    _token: object


@dataclass(frozen=True)
class BNCarrier:
    nominal_a: np.ndarray
    nominal_b: np.ndarray
    error: np.ndarray
    bounds: Interval
    _token: object = None


def _check(value, budget):
    if type(value) is not Interval:
        raise Rejected("Interval required")
    lo, hi = value.lo, value.hi
    if (type(lo) is not np.ndarray or type(hi) is not np.ndarray
            or lo.dtype != np.float64 or hi.dtype != np.float64
            or lo.shape != hi.shape):
        raise Rejected("equal-shaped float64 endpoint arrays required")
    n = lo.size
    if value._token is _TOKEN:
        _pay(budget, 12, 12)
        if lo.flags.writeable or hi.flags.writeable:
            raise Rejected("owned source range was made writable")
        return value
    _pay(budget, 4 * n + 12, n + 12)
    if not n:
        return value
    lower_min, upper_max = float(np.min(lo)), float(np.max(hi))
    if (not math.isfinite(lower_min) or not math.isfinite(upper_max)
            or lower_min < -_MAXIMUM or upper_max > _MAXIMUM
            or not (lo <= hi).all()):
        raise Rejected("unordered, nonfinite or oversized source interval")
    return value


def _finish(lo, hi, budget):
    _pay(budget, 4, 4)
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    _check(Interval(lo, hi), budget)
    lo.setflags(write=False)
    hi.setflags(write=False)
    return Interval(lo, hi, _TOKEN)


def _round(raw, upward, budget):
    """One correctly rounded basic result, then an outward dyadic lattice."""
    _pay(budget, 2, 2)
    raw = np.asarray(raw, dtype=np.float64)
    n = raw.size
    _pay(budget, 5 * n + 8, 3 * n + 8)
    out = np.asarray(np.nextafter(raw, np.inf if upward else -np.inf), dtype=np.float64)
    magnitude = np.empty(out.shape, dtype=np.float64)
    tiny = np.empty(out.shape, dtype=np.bool_)
    np.absolute(out, out=magnitude)
    np.less(magnitude, _TINY, out=tiny)
    count = int(np.count_nonzero(tiny))
    if count:
        _pay(budget, 5 * count + n, count + 4)
        scaled = out[tiny]
        np.ldexp(scaled, _GRID_EXP, out=scaled)
        if upward:
            np.ceil(scaled, out=scaled)
        else:
            np.floor(scaled, out=scaled)
        np.ldexp(scaled, -_GRID_EXP, out=scaled)
        out[tiny] = scaled
    return out


def _shape_size(shapes, budget):
    _pay(budget, 8 + sum(len(s) for s in shapes), 8 + sum(len(s) for s in shapes))
    try:
        shape = np.broadcast_shapes(*shapes)
    except ValueError as exc:
        raise Rejected("source interval shapes do not broadcast") from exc
    return shape, math.prod(shape)


def _add(a, b, budget):
    shape, n = _shape_size((a.lo.shape, b.lo.shape), budget)
    _pay(budget, 8 * n + 8, 4 * n + 8)
    lo = _round(np.add(a.lo, b.lo), False, budget)
    hi = _round(np.add(a.hi, b.hi), True, budget)
    # Opposite *stored endpoint values* sum exactly to zero.  This does not
    # infer any unregistered equality between two uncertain source objects.
    opposite = np.empty(shape, dtype=np.float64)
    zero = np.empty(shape, dtype=np.bool_)
    np.negative(b.lo, out=opposite)
    np.equal(a.lo, opposite, out=zero)
    np.copyto(lo, 0.0, where=zero)
    np.negative(b.hi, out=opposite)
    np.equal(a.hi, opposite, out=zero)
    np.copyto(hi, 0.0, where=zero)
    return Interval(lo, hi)


def _neg(a, budget):
    _pay(budget, 2 * a.lo.size + 8, 2 * a.lo.size + 8)
    return Interval(np.asarray(np.negative(a.hi)), np.asarray(np.negative(a.lo)))


def _sub(a, b, budget):
    return _add(a, _neg(b, budget), budget)


def _mul(a, b, budget):
    shape, n = _shape_size((a.lo.shape, b.lo.shape), budget)
    if a.lo is a.hi:
        products = ((a.lo, b.lo), (a.lo, b.hi))
    elif b.lo is b.hi:
        products = ((a.lo, b.lo), (a.hi, b.lo))
    else:
        products = ((a.lo, b.lo), (a.lo, b.hi),
                    (a.hi, b.lo), (a.hi, b.hi))
    lower = upper = None
    for left, right in products:
        _pay(budget, 6 * n + 6, 3 * n + 6)
        raw = np.multiply(left, right)
        low, high = _round(raw, False, budget), _round(raw, True, budget)
        zero = np.empty(shape, dtype=np.bool_)
        other_zero = np.empty(shape, dtype=np.bool_)
        np.equal(left, 0.0, out=zero)
        np.equal(right, 0.0, out=other_zero)
        np.logical_or(zero, other_zero, out=zero)
        np.copyto(low, 0.0, where=zero)
        np.copyto(high, 0.0, where=zero)
        if lower is None:
            lower, upper = low, high
        else:
            _pay(budget, 2 * n)
            np.minimum(lower, low, out=lower)
            np.maximum(upper, high, out=upper)
    return Interval(lower, upper)


def interval(lo, hi, *, budget, enabled=False):
    if not _on(enabled):
        return None
    probe = _check(Interval(lo, hi), budget)
    _pay(budget, 2 * lo.size + 4, 2 * lo.size + 4)
    return _finish(probe.lo.copy(), probe.hi.copy(), budget)


def point(values, *, budget, enabled=False):
    if not _on(enabled):
        return None
    if type(values) is np.ndarray:
        if values.dtype != np.float64:
            raise Rejected("point arrays must already be exact float64 payloads")
        n = values.size
    elif type(values) in (float, np.float64):
        n = 1
    else:
        raise Rejected("float64 point array or scalar required")
    _pay(budget, 8 * n + 8, 6 * n + 8)
    owned = np.array(values, dtype=np.float64, copy=True)
    nonzero = owned != 0.0
    if (np.abs(owned[nonzero]) < _POINT_MIN).any():
        raise Rejected("point coefficient exceeds conservative 512-bit support")
    return _finish(owned, owned, budget)


def _rational_endpoint(value, upward, budget):
    if type(value) not in (Fraction, int):
        raise Rejected("rational endpoints must be Fraction or int, not floats")
    numerator, denominator = value.numerator, value.denominator
    if max(abs(numerator).bit_length(), denominator.bit_length()) > 512:
        raise Rejected("rational source endpoint exceeds 512 bits")
    _pay(budget, 32 + abs(numerator).bit_length() + denominator.bit_length(), 16)
    candidate = float(value)
    if not math.isfinite(candidate) or abs(candidate) > _MAXIMUM:
        raise Rejected("rational endpoint has no supported finite float enclosure")
    p, q = candidate.as_integer_ratio()
    # This is an exact check of the rounded conversion, not a float comparison.
    comparison = p * denominator - numerator * q
    _pay(budget, 16 + p.bit_length() + q.bit_length(), 12)
    if (upward and comparison < 0) or (not upward and comparison > 0):
        candidate = math.nextafter(candidate, math.inf if upward else -math.inf)
    if candidate != 0.0 and abs(candidate) < _TINY:
        scaled = math.ldexp(candidate, _GRID_EXP)
        candidate = math.ldexp(math.ceil(scaled) if upward else math.floor(scaled), -_GRID_EXP)
    p, q = candidate.as_integer_ratio()
    comparison = p * denominator - numerator * q
    _pay(budget, 16 + p.bit_length() + q.bit_length(), 12)
    if (upward and comparison < 0) or (not upward and comparison > 0):
        raise Rejected("outward rational conversion certificate failed")
    return candidate


def from_rationals(pairs, *, budget, enabled=False):
    if not _on(enabled):
        return None
    if type(pairs) not in (tuple, list):
        raise Rejected("finite sequence of rational (lower, upper) pairs required")
    _pay(budget, 8 * len(pairs) + 8, 8 * len(pairs) + 8)
    lower, upper = [], []
    for pair in pairs:
        if type(pair) not in (tuple, list) or len(pair) != 2:
            raise Rejected("rational interval pair required")
        lo, hi = pair
        if type(lo) not in (Fraction, int) or type(hi) not in (Fraction, int) or lo > hi:
            raise Rejected("unordered rational source interval")
        lower.append(_rational_endpoint(lo, False, budget))
        upper.append(_rational_endpoint(hi, True, budget))
    return _finish(np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64), budget)


def add(a, b, *, budget, enabled=False):
    if not _on(enabled):
        return None
    result = _add(_check(a, budget), _check(b, budget), budget)
    return _finish(result.lo, result.hi, budget)


def sub(a, b, *, budget, enabled=False):
    if not _on(enabled):
        return None
    result = _sub(_check(a, budget), _check(b, budget), budget)
    return _finish(result.lo, result.hi, budget)


def neg(a, *, budget, enabled=False):
    if not _on(enabled):
        return None
    result = _neg(_check(a, budget), budget)
    return _finish(result.lo, result.hi, budget)


def mul(a, b, *, budget, enabled=False):
    if not _on(enabled):
        return None
    result = _mul(_check(a, budget), _check(b, budget), budget)
    return _finish(result.lo, result.hi, budget)


def scale_half(a, *, budget, enabled=False):
    if not _on(enabled):
        return None
    _check(a, budget)
    _pay(budget, 2, 2)
    half = np.array(0.5, dtype=np.float64)
    result = _mul(a, Interval(half, half), budget)
    return _finish(result.lo, result.hi, budget)


def _axis_sum_endpoint(values, axis, upward, budget):
    """Batched version of the same certified absolute-sum reduction below."""
    count = values.shape[axis]
    if count > 65536:
        raise Rejected("source reduction exceeds fixed supported width")
    shape = values.shape[:axis] + values.shape[axis + 1:]
    n = math.prod(shape)
    # Two reductions and abs read every input.  Output arithmetic reuses one
    # scratch array, rather than creating one interval per reduced element.
    _pay(budget, 3 * values.size + 24 * n + 80,
         values.size + 6 * n + 32)
    total = np.asarray(np.sum(values, axis=axis, dtype=np.float64))
    absolute = np.asarray(np.sum(np.abs(values), axis=axis, dtype=np.float64))
    if not np.isfinite(total).all() or not np.isfinite(absolute).all():
        raise Rejected("nonfinite source reduction")
    if not count:
        return total, absolute == 0.0
    grid = math.ldexp(1.0, -_GRID_EXP)
    nu = math.ldexp(float(count), -53)
    gamma = math.nextafter(nu / (1.0 - nu), math.inf)
    scratch = np.asarray(np.nextafter(absolute, np.inf))
    np.subtract(scratch, absolute, out=scratch)
    np.maximum(scratch, grid, out=scratch)
    np.multiply(scratch, float(count), out=scratch)
    np.nextafter(scratch, np.inf, out=scratch)
    np.add(scratch, absolute, out=scratch)
    np.nextafter(scratch, np.inf, out=scratch)
    np.multiply(scratch, gamma, out=scratch)
    np.nextafter(scratch, np.inf, out=scratch)
    np.add(scratch, float(count) * grid, out=scratch)
    np.nextafter(scratch, np.inf, out=scratch)
    if upward:
        np.add(total, scratch, out=total)
    else:
        np.subtract(total, scratch, out=total)
    np.nextafter(total, np.inf if upward else -np.inf, out=total)
    # A zero absolute sum certifies all original stored summands were zero.
    zero = absolute == 0.0
    np.copyto(total, 0.0, where=zero)
    return total, zero


def _sum_axis(a, axis, keepdims, budget):
    if type(axis) is not int or not -a.lo.ndim <= axis < a.lo.ndim:
        raise Rejected("one valid integer reduction axis required")
    axis %= a.lo.ndim
    raw_lo, zero_lo = _axis_sum_endpoint(a.lo, axis, False, budget)
    raw_hi, zero_hi = _axis_sum_endpoint(a.hi, axis, True, budget)
    lo = _round(raw_lo, False, budget)
    hi = _round(raw_hi, True, budget)
    _pay(budget, 2 * lo.size + 4, 4)
    np.copyto(lo, 0.0, where=zero_lo)
    np.copyto(hi, 0.0, where=zero_hi)
    acc = Interval(lo, hi)
    if keepdims:
        _pay(budget, 8, 8)
        acc = Interval(np.expand_dims(acc.lo, axis), np.expand_dims(acc.hi, axis))
    return acc


def sum_axis(a, axis=0, keepdims=False, *, budget, enabled=False):
    if not _on(enabled):
        return None
    if type(keepdims) is not bool:
        raise Rejected("keepdims must be bool")
    result = _sum_axis(_check(a, budget), axis, keepdims, budget)
    return _finish(result.lo, result.hi, budget)


def take(a, indices, axis=0, *, budget, enabled=False):
    if not _on(enabled):
        return None
    _check(a, budget)
    if type(axis) is not int or not -a.lo.ndim <= axis < a.lo.ndim:
        raise Rejected("one valid integer index axis required")
    if type(indices) is int:
        count = 1
    elif type(indices) in (tuple, list) and all(type(i) is int for i in indices):
        count = len(indices)
    else:
        raise Rejected("integer or finite integer sequence index required")
    n = a.lo.size // max(1, a.lo.shape[axis]) * count
    _pay(budget, 3 * n + 12, 2 * n + 12)
    try:
        lo, hi = np.take(a.lo, indices, axis=axis), np.take(a.hi, indices, axis=axis)
    except (IndexError, ValueError) as exc:
        raise Rejected("source index out of range") from exc
    return _finish(lo, hi, budget)


def relu(a, *, budget, enabled=False):
    if not _on(enabled):
        return None
    _check(a, budget)
    _pay(budget, 2 * a.lo.size + 4, 2 * a.lo.size + 4)
    return _finish(np.maximum(a.lo, 0.0), np.maximum(a.hi, 0.0), budget)


def midpoint(a, *, budget, enabled=False):
    """A chosen finite point, NOT an enclosure or a replacement source."""
    if not _on(enabled):
        return None
    _check(a, budget)
    n = a.lo.size
    _pay(budget, 8 * n + 8, 6 * n + 8)
    answer = np.asarray(np.maximum(a.lo, np.minimum(a.hi, a.lo * 0.5 + a.hi * 0.5)))
    if not np.isfinite(answer).all():
        raise Rejected("nonfinite nominal midpoint")
    answer.setflags(write=False)
    return answer


def _kernel_mass(weights, mask, budget):
    """Two same-sign accumulators; no per-weight interval arrays.

    In a nonnegative floating sum every partial sum is <= its final value.
    Thus each addition's absolute rounding error is <= spacing(final); with
    n taps the exact sum is in final +/- n*spacing(abs(final)).  The negative
    accumulator is symmetric.  No final-nextafter-only sum claim is made.
    """
    co, ci, kh, kw = weights.shape
    n = co * ci
    _pay(budget, 2 * n + 12, 2 * n + 12)
    positive = np.zeros((co, ci), dtype=np.float64)
    negative = np.zeros((co, ci), dtype=np.float64)
    for ky in range(kh):
        for kx in range(kw):
            if mask is not None and not mask[ky, kx]:
                _pay(budget, 2)
                continue
            _pay(budget, 4 * n + 8, 2 * n + 8)
            tap = weights[:, :, ky, kx]
            np.add(positive, np.maximum(tap, 0.0), out=positive)
            np.add(negative, np.minimum(tap, 0.0), out=negative)
    result = []
    for total in (positive, negative):
        _pay(budget, 8 * n + 12, 7 * n + 12)
        magnitude = np.abs(total)
        spacing = np.nextafter(magnitude, np.inf) - magnitude
        # This error is a private intermediate, not a published lattice
        # endpoint.  One directed step suffices; only the final two bounds
        # need the public dyadic encoding.
        error = np.nextafter(spacing * float(kh * kw), np.inf)
        lo = _round(total - error, False, budget)
        hi = _round(total + error, True, budget)
        _pay(budget, 4 * n + 4, 3 * n + 4)
        zero = total == 0.0
        # Same-sign zero sums have only zero summands; no underflowing products
        # occur in this kernel-weight addition stage.
        lo, hi = np.where(zero, 0.0, lo), np.where(zero, 0.0, hi)
        result.append(_finish(lo, hi, budget))
    return tuple(result)


def _mass_channels(positive, negative, inputs, budget):
    n = positive.lo.size
    _pay(budget, 20 * n + 12, 14 * n + 4 * inputs.lo.size + 12)
    xl, xu = inputs.lo[None, :], inputs.hi[None, :]
    pl = np.where(xl >= 0.0, positive.lo, positive.hi)
    pu = np.where(xu >= 0.0, positive.hi, positive.lo)
    nl = np.where(xu >= 0.0, negative.lo, negative.hi)
    nu = np.where(xl >= 0.0, negative.hi, negative.lo)
    # Products are directed private endpoints.  Lattice conversion is done
    # after their addition, avoiding four full grid scans and scratch sets.
    lower_p = np.nextafter(pl * xl, -np.inf)
    lower_n = np.nextafter(nl * xu, -np.inf)
    upper_p = np.nextafter(pu * xu, np.inf)
    upper_n = np.nextafter(nu * xl, np.inf)
    lower = _round(lower_p + lower_n, False, budget)
    upper = _round(upper_p + upper_n, True, budget)
    # Preserve the empty kernel/channel contribution exactly.
    _pay(budget, 9 * n + 4, 4 * n + 4)
    empty = positive.lo == 0.0
    other = np.empty(empty.shape, dtype=np.bool_)
    for endpoint in (positive.hi, negative.lo, negative.hi):
        np.equal(endpoint, 0.0, out=other)
        np.logical_and(empty, other, out=empty)
    return Interval(np.where(empty, 0.0, lower), np.where(empty, 0.0, upper))


def conv_channel(weights, bias, input_bounds, padding, budget, *, enabled=False, valid_mask=None):
    if not _on(enabled):
        return None
    inputs = _check(input_bounds, budget)
    if (type(weights) is not np.ndarray or weights.dtype != np.float64
            or weights.ndim != 4 or any(n <= 0 for n in weights.shape)
            or inputs.lo.shape != (weights.shape[1],)):
        raise Rejected("Conv weights [out,in,kh,kw] and complete channel ranges required")
    if type(padding) is int and padding >= 0:
        pad = (padding,) * 4
    elif type(padding) is tuple and len(padding) in (2, 4) and all(type(v) is int and v >= 0 for v in padding):
        pad = padding * 2 if len(padding) == 2 else padding
    else:
        raise Rejected("nonnegative padding geometry required")
    W = point(weights, budget=budget, enabled=True)
    if bias is None:
        _pay(budget, weights.shape[0], weights.shape[0])
        bias = np.zeros(weights.shape[0], dtype=np.float64)
    B = point(bias, budget=budget, enabled=True)
    if B.lo.shape != (weights.shape[0],):
        raise Rejected("complete Conv bias vector required")
    mask = None
    if valid_mask is not None:
        if (type(valid_mask) is not np.ndarray or valid_mask.dtype != np.bool_
                or valid_mask.shape != weights.shape[2:]):
            raise Rejected("actual kernel-position validity mask required")
        _pay(budget, 2 * valid_mask.size + 8, valid_mask.size + 8)
        mask = valid_mask.copy()
        mask.setflags(write=False)
    elif any(pad):
        # A single channel envelope covers every real position, including zero
        # padding.  A supplied actual mask instead preserves valid-tap ranges.
        _pay(budget, 2 * inputs.lo.size + 4, 2 * inputs.lo.size + 4)
        inputs = _finish(np.minimum(inputs.lo, 0.0), np.maximum(inputs.hi, 0.0), budget)
    if inputs._token is not _TOKEN:
        _pay(budget, 2 * inputs.lo.size + 4, 2 * inputs.lo.size + 4)
        inputs = _finish(inputs.lo.copy(), inputs.hi.copy(), budget)
    positive, negative = _kernel_mass(W.lo, mask, budget)
    contributions = _mass_channels(positive, negative, inputs, budget)
    summed = _sum_axis(contributions, 1, False, budget)
    value = _add(summed, B, budget)
    bounds = _finish(value.lo, value.hi, budget)
    _pay(budget, 24, 24)
    return ConvLedger(bounds, positive, negative, inputs, W.lo, B.lo, pad, mask, _TOKEN)


def peel_conv_selected(ledger, indices, *, budget, enabled=False):
    if not _on(enabled):
        return None
    if type(ledger) is not ConvLedger or ledger._token is not _TOKEN:
        raise Rejected("owned additive Conv ledger required")
    if type(indices) is not tuple or len(indices) > 2:
        raise Rejected("at most two distinct selected Conv terms required")
    # All numeric ledger arrays were fully checked and made read-only by its
    # sole supported constructor.  Do not rescan the complete kernel per pair.
    _pay(budget, 24, 24)
    lower, upper = ledger.bounds.lo, ledger.bounds.hi
    ci, kh, kw = ledger.weights.shape[1:]
    seen = set()
    for index in indices:
        if (type(index) is not tuple or len(index) != 3
                or any(type(v) is not int for v in index)
                or not (0 <= index[0] < ci and 0 <= index[1] < kh and 0 <= index[2] < kw)):
            raise Rejected("selected Conv term index out of range")
        if index in seen:
            raise Rejected("duplicate selected Conv term")
        seen.add(index)
        if ledger.valid_mask is not None and not ledger.valid_mask[index[1], index[2]]:
            raise Rejected("a padded zero is not an actual selected parent term")
        n = lower.size
        _pay(budget, 12 * n + 16, 10 * n + 16)
        coefficient = ledger.weights[(slice(None),) + index]
        il, iu = ledger.input_bounds.lo[index[0]], ledger.input_bounds.hi[index[0]]
        lower_operand = np.where(coefficient >= 0.0, il, iu)
        upper_operand = np.where(coefficient >= 0.0, iu, il)
        # base.lower encloses the exact additive lower sum.  Remove an UPPER
        # enclosure of its selected min term; upper removes a LOWER enclosure
        # of the selected max.  This remains safe with aggregated mass errors.
        selected_min_upper = _round(coefficient * lower_operand, True, budget)
        selected_max_lower = _round(coefficient * upper_operand, False, budget)
        lower = _round(np.subtract(lower, selected_min_upper), False, budget)
        upper = _round(np.subtract(upper, selected_max_lower), True, budget)
    _pay(budget, 2 * lower.size + 4, 2 * lower.size + 4)
    return _finish(lower.copy(), upper.copy(), budget)


def bn_carrier(input_bounds, a_bounds, b_bounds, *, budget, enabled=False):
    if not _on(enabled):
        return None
    inputs, A, B = (_check(v, budget) for v in (input_bounds, a_bounds, b_bounds))
    if inputs.lo.shape != A.lo.shape or A.lo.shape != B.lo.shape:
        raise Rejected("BN parameters and complete channel ranges must match")
    nominal_a = midpoint(A, budget=budget, enabled=True)
    nominal_b = midpoint(B, budget=budget, enabled=True)
    pa = Interval(nominal_a, nominal_a)
    pb = Interval(nominal_b, nominal_b)
    da, db = _sub(A, pa, budget), _sub(B, pb, budget)
    n = A.lo.size
    _pay(budget, 10 * n + 8, 9 * n + 8)
    radius_a = np.maximum(np.abs(da.lo), np.abs(da.hi))
    radius_b = np.maximum(np.abs(db.lo), np.abs(db.hi))
    magnitude = np.maximum(np.abs(inputs.lo), np.abs(inputs.hi))
    product = _mul(Interval(radius_a, radius_a), Interval(magnitude, magnitude), budget)
    error_iv = _add(product, Interval(radius_b, radius_b), budget)
    _pay(budget, 3 * n + 8, 3 * n + 8)
    error = error_iv.hi.copy()
    noise = Interval(np.negative(error), error.copy())
    value = _add(_add(_mul(pa, inputs, budget), pb, budget), noise, budget)
    bounds = _finish(value.lo, value.hi, budget)
    error.setflags(write=False)
    _pay(budget, 12, 12)
    return BNCarrier(nominal_a, nominal_b, error, bounds, _TOKEN)


def _fast_point_mul(point_value, low, high, budget):
    """Trusted finite vector operands; no arrays escape before final checking."""
    n = low.size
    _pay(budget, 7 * n + 8, 3 * n + 8)
    positive = point_value >= 0.0
    lower = np.where(positive, low, high)
    upper = np.where(positive, high, low)
    np.multiply(lower, point_value, out=lower)
    np.multiply(upper, point_value, out=upper)
    np.nextafter(lower, -np.inf, out=lower)
    np.nextafter(upper, np.inf, out=upper)
    return lower, upper


def _fast_mul(al, au, bl, bu, budget):
    n = al.size
    _pay(budget, 13 * n + 12, 5 * n + 12)
    first = np.multiply(al, bl)
    lower, upper = first, first.copy()
    for left, right in ((al, bu), (au, bl), (au, bu)):
        value = np.multiply(left, right)
        np.minimum(lower, value, out=lower)
        np.maximum(upper, value, out=upper)
    # nextafter is monotone: min/max of the rounded corner products need only
    # one outward step after reduction, not eight separate corner enclosures.
    np.nextafter(lower, -np.inf, out=lower)
    np.nextafter(upper, np.inf, out=upper)
    return lower, upper


def _fast_sum_endpoint(values, upward, budget):
    n = values.size
    if n > 65536:
        raise Rejected("fused source reduction exceeds fixed supported width")
    _pay(budget, 3 * n + 128, n + 32)
    total = float(np.sum(values, dtype=np.float64))
    absolute = float(np.sum(np.abs(values), dtype=np.float64))
    if not math.isfinite(total) or not math.isfinite(absolute):
        raise Rejected("nonfinite fused source reduction")
    if not n or absolute == 0.0:
        return 0.0
    grid = math.ldexp(1.0, -_GRID_EXP)
    spacing = math.nextafter(absolute, math.inf) - absolute
    abs_error = math.nextafter(float(n) * max(spacing, grid), math.inf)
    abs_upper = math.nextafter(absolute + abs_error, math.inf)
    nu = math.ldexp(float(n), -53)
    gamma = math.nextafter(nu / (1.0 - nu), math.inf)
    product_error = math.nextafter(gamma * abs_upper, math.inf)
    error = math.nextafter(product_error + float(n) * grid, math.inf)
    answer = math.nextafter(total + error if upward else total - error,
                            math.inf if upward else -math.inf)
    if not math.isfinite(answer):
        raise Rejected("fused sum error enclosure overflow")
    return answer


def _fast_sum_interval(lower, upper, budget):
    lo = _fast_sum_endpoint(lower, False, budget)
    hi = _fast_sum_endpoint(upper, True, budget)
    _pay(budget, 8, 8)
    # Final scalar lattice conversion is outward; internal float64 directed
    # intermediates need not allocate separate lattice/validation arrays.
    return _finish(_round(np.array(lo), False, budget),
                   _round(np.array(hi), True, budget), budget)


def _borrow_point_array(value, budget):
    """Fully scan a synchronous operand without retaining or copying it."""
    if type(value) is not np.ndarray or value.dtype != np.float64:
        raise Rejected("exact float64 point array required")
    n = value.size
    _pay(budget, 5 * n + 24, 2 * n + 24)
    if n:
        low, high = float(np.min(value)), float(np.max(value))
        smallest = float(np.min(np.abs(value), where=(value != 0.0), initial=np.inf))
        if (not math.isfinite(low) or not math.isfinite(high)
                or low < -_MAXIMUM or high > _MAXIMUM or smallest < _POINT_MIN):
            raise Rejected("nonfinite or unsupported point operand")
    return value


def fused_h(weights_pair, nominal_a_pair, *, budget, enabled=False):
    """Full same-position final-Conv difference, before any independent bound.

    Inputs are the two complete output kernels and their BN nominal scales.
    Bias and the two ORIGINAL BN epsilon terms remain caller-owned separate
    readouts; neither is silently omitted or merged by channel identity.
    """
    if not _on(enabled):
        return None
    weights = _borrow_point_array(weights_pair, budget)
    scales = _borrow_point_array(nominal_a_pair, budget)
    if weights.ndim != 4 or weights.shape[0] != 2 or scales.shape != (2,):
        raise Rejected("two full [K,kh,kw] kernels and two nominal BN scales required")
    n = weights[0].size
    _pay(budget, 15 * n + 24, 8 * n + 24)
    first = np.multiply(weights[0], scales[0])
    second = np.multiply(weights[1], scales[1])
    first_lo, first_hi = np.nextafter(first, -np.inf), np.nextafter(first, np.inf)
    second_lo, second_hi = np.nextafter(second, -np.inf), np.nextafter(second, np.inf)
    lower = np.subtract(first_lo, second_hi)
    upper = np.subtract(first_hi, second_lo)
    np.nextafter(lower, -np.inf, out=lower)
    np.nextafter(upper, np.inf, out=upper)
    np.multiply(lower, 0.5, out=lower)
    np.multiply(upper, 0.5, out=upper)
    lower, upper = _round(lower, False, budget), _round(upper, True, budget)
    # Identical stored factors have exactly equal real products.  This is an
    # algebraic identity, not equality of rounded product estimates.
    if scales[0] == scales[1]:
        _pay(budget, 3 * n + 4, n + 4)
        equal = weights[0] == weights[1]
        np.copyto(lower, 0.0, where=equal)
        np.copyto(upper, 0.0, where=equal)
    return _finish(lower, upper, budget)


def fused_offset(ledger, selected, h, middle_carrier, skip_bounds, *, budget, enabled=False):
    """One COMPLETE intermediate-channel offset, no per-channel interval objects.

    Returns (sum(h*S_rest), (sum(h*a*W_i), sum(h*a*W_j))).  The latter are
    coefficients of the original unnormalised Q; caller applies parent scale.
    Add each original BN bias/error and skip exactly once, without inventing
    independent physical residuals.  Only range propagation drops correlations.
    """
    if not _on(enabled):
        return None
    if type(ledger) is not ConvLedger or ledger._token is not _TOKEN:
        raise Rejected("owned additive Conv ledger required")
    if type(middle_carrier) is not BNCarrier or middle_carrier._token is not _TOKEN:
        raise Rejected("the original intermediate BN carrier is required")
    H, skip = _check(h, budget), _check(skip_bounds, budget)
    k = ledger.weights.shape[0]
    if H.lo.shape != (k,) or skip.lo.shape != (k,):
        raise Rejected("complete intermediate channel population required")
    if type(selected) is not tuple or len(selected) != 2:
        raise Rejected("two selected tap descriptors (or None) required")
    ma, mb, error = middle_carrier.nominal_a, middle_carrier.nominal_b, middle_carrier.error
    if any(type(v) is not np.ndarray or v.dtype != np.float64 or v.shape != (k,) or v.flags.writeable
           for v in (ma, mb, error)):
        raise Rejected("read-only complete original BN parameters required")
    # The sole supported BN constructor checked all values and froze arrays;
    # its ownership token does not authorize externally manufactured records.
    _pay(budget, 2 * k + 48, 2 * k + 48)
    lower, upper = ledger.bounds.lo.copy(), ledger.bounds.hi.copy()
    weights = []
    seen = set()
    ci, kh, kw = ledger.weights.shape[1:]
    for index in selected:
        if index is None:
            _pay(budget, k + 4, k + 4)
            weights.append(np.zeros(k, dtype=np.float64))
            continue
        if (type(index) is not tuple or len(index) != 3 or any(type(v) is not int for v in index)
                or not (0 <= index[0] < ci and 0 <= index[1] < kh and 0 <= index[2] < kw)
                or index in seen):
            raise Rejected("invalid or repeated selected original Conv tap")
        if ledger.valid_mask is not None and not ledger.valid_mask[index[1], index[2]]:
            raise Rejected("selected parent is outside the actual padding mask")
        seen.add(index)
        coefficient = ledger.weights[(slice(None),) + index]
        weights.append(coefficient)
        _pay(budget, 11 * k + 12, 3 * k + 12)
        positive = coefficient >= 0.0
        min_term = np.where(positive, ledger.input_bounds.lo[index[0]], ledger.input_bounds.hi[index[0]])
        max_term = np.where(positive, ledger.input_bounds.hi[index[0]], ledger.input_bounds.lo[index[0]])
        np.multiply(min_term, coefficient, out=min_term)
        np.multiply(max_term, coefficient, out=max_term)
        np.nextafter(min_term, np.inf, out=min_term)
        np.nextafter(max_term, -np.inf, out=max_term)
        np.subtract(lower, min_term, out=lower)
        np.subtract(upper, max_term, out=upper)
        np.nextafter(lower, -np.inf, out=lower)
        np.nextafter(upper, np.inf, out=upper)
    lower, upper = _fast_point_mul(ma, lower, upper, budget)
    # Prepay the one new -E array before constructing the loop tuple.
    _pay(budget, 13 * k + 8, k + 8)
    for low_term, high_term in ((mb, mb), (np.negative(error), error), (skip.lo, skip.hi)):
        np.add(lower, low_term, out=lower)
        np.add(upper, high_term, out=upper)
        np.nextafter(lower, -np.inf, out=lower)
        np.nextafter(upper, np.inf, out=upper)
    product_lo, product_hi = _fast_mul(H.lo, H.hi, lower, upper, budget)
    rest = _fast_sum_interval(product_lo, product_hi, budget)
    common_lo, common_hi = _fast_point_mul(ma, H.lo, H.hi, budget)
    coefficients = []
    for coefficient in weights:
        lo, hi = _fast_point_mul(coefficient, common_lo, common_hi, budget)
        coefficients.append(_fast_sum_interval(lo, hi, budget))
    _pay(budget, 12, 12)
    return rest, tuple(coefficients)
