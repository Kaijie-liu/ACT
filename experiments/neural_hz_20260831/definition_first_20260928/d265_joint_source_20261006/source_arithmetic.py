"""Paid interval readouts of authenticated source expressions, not a domain.

These internal operations do not confer latent identity or ownership.  The
source observer authenticates original arrays; every complete result entering
a reduction is checked.  Directed vector intermediates may be subnormal, as
in D259's fused operations; scalar evidence is rounded onto its dyadic grid.
No operation mutates its arguments or resets the shared resource meter.
"""
from fractions import Fraction
import math

import numpy as np

from experiments.neural_hz_20260831.definition_first_20260928.d259_shared_difference_source_20261006 import source_bounds as sb
from experiments.neural_hz_20260831.definition_first_20260928.d264_joint_support_kernel_20261006 import joint_support as js

I = sb.Interval
Rejected = sb.Rejected


def _pay(budget, work, entries=0):
    sb._pay(budget, work, entries)


def _shape(a, b, budget):
    _pay(budget, 16, 16)
    shape = np.broadcast_shapes(a.lo.shape, b.lo.shape)
    return shape, math.prod(shape)


def point(value, budget):
    # Public construction, including exact-float range/bit support checks.
    return sb.point(value, budget=budget, enabled=True)


def add(a, b, budget):
    _, n = _shape(a, b, budget)
    _pay(budget, 4*n+8, 2*n+8)
    lo, hi = np.asarray(np.add(a.lo, b.lo)), np.asarray(np.add(a.hi, b.hi))
    np.nextafter(lo, -np.inf, out=lo)
    np.nextafter(hi, np.inf, out=hi)
    return I(lo, hi)


def neg(a, budget):
    _pay(budget, 2*a.lo.size+8, 2*a.lo.size+8)
    return I(np.asarray(np.negative(a.hi)), np.asarray(np.negative(a.lo)))


def sub(a, b, budget):
    return add(a, neg(b, budget), budget)


def mul(a, b, budget):
    shape, n = _shape(a, b, budget)
    _pay(budget, 8, 8)
    internal_shape = shape if shape else (1,)
    al, au = np.broadcast_to(a.lo, internal_shape), np.broadcast_to(a.hi, internal_shape)
    bl, bu = np.broadcast_to(b.lo, internal_shape), np.broadcast_to(b.hi, internal_shape)
    lo, hi = sb._fast_mul(al, au, bl, bu, budget)
    return I(lo.reshape(shape), hi.reshape(shape))


def scale(a, value, budget):
    _pay(budget, 16, 16)
    v = np.asarray(value, dtype=np.float64)
    shape = np.broadcast_shapes(a.lo.shape, v.shape)
    _pay(budget, 8, 8)
    lo, hi = sb._fast_point_mul(v, np.broadcast_to(a.lo, shape),
                               np.broadcast_to(a.hi, shape), budget)
    return I(lo, hi)


def half(a, budget):
    return scale(a, 0.5, budget)


def div(a, b, budget):
    if type(b) in (float, np.float64):
        b = point(b, budget)
    shape, n = _shape(a, b, budget)
    sb._check(b, budget)
    _pay(budget, 2*b.lo.size+8, b.lo.size+8)
    if not (b.lo > 0.0).all():
        raise Rejected('strictly positive denominator required')
    _pay(budget, 13*n+16, 6*n+16)
    first = np.asarray(np.divide(a.lo, b.lo))
    lo, hi = first, first.copy()
    for left, right in ((a.lo, b.hi), (a.hi, b.lo), (a.hi, b.hi)):
        product = np.asarray(np.divide(left, right))
        np.minimum(lo, product, out=lo)
        np.maximum(hi, product, out=hi)
    np.nextafter(lo, -np.inf, out=lo)
    np.nextafter(hi, np.inf, out=hi)
    return I(lo, hi)


def sum_iv(a, budget):
    sb._check(a, budget)
    return sb._fast_sum_interval(a.lo, a.hi, budget)


def sum_axis(a, axis, budget):
    return sb.sum_axis(a, axis=axis, budget=budget, enabled=True)


def stack(values, budget):
    n = sum(v.lo.size for v in values)
    _pay(budget, 2*n+32+8*len(values), 2*n+32+8*len(values))
    return I(np.stack([v.lo for v in values]), np.stack([v.hi for v in values]))


def concat(values, budget):
    n = sum(v.lo.size for v in values)
    _pay(budget, 2*n+32+8*len(values), 2*n+32+8*len(values))
    return I(np.concatenate([v.lo.reshape(-1) for v in values]),
             np.concatenate([v.hi.reshape(-1) for v in values]))


def midpoint(a, budget):
    sb._check(a, budget)
    n = a.lo.size
    _pay(budget, 6*n+24, 4*n+24)
    value = np.asarray(a.lo*0.5+a.hi*0.5, dtype=np.float64)
    if not np.isfinite(value).all():
        raise Rejected('nonfinite fixed midpoint')
    # This is a declared exact binary64 choice, not an exact projection claim.
    sb._borrow_point_array(value, budget)
    return value


def center_radius(bounds, budget):
    center = midpoint(bounds, budget)
    n = center.size
    _pay(budget, 6*n+16, 3*n+16)
    left = np.asarray(np.subtract(center, bounds.lo))
    right = np.asarray(np.subtract(bounds.hi, center))
    np.nextafter(left, np.inf, out=left)
    np.nextafter(right, np.inf, out=right)
    radius = np.asarray(np.maximum(left, right))
    # The extra radius is sound even for a degenerate interval.  Do not remove
    # it by a tolerance; scalar lattice accounting covers tiny roundoff.
    _pay(budget, n+8, n+8)
    sb._check(I(np.asarray(np.negative(radius)), radius), budget)
    return center, radius


def normalize_group(rho_coeff, e_coeff, bounds, budget):
    """One common box axis for each coordinate of an actual affine group."""
    if rho_coeff.lo.shape != e_coeff.lo.shape or rho_coeff.lo.shape != bounds.lo.shape:
        raise Rejected('complete same-shaped group operands required')
    center, radius = center_radius(bounds, budget)
    rc = sum_iv(scale(rho_coeff, center, budget), budget)
    ec = sum_iv(scale(e_coeff, center, budget), budget)
    ra = scale(rho_coeff, radius, budget)
    ea = scale(e_coeff, radius, budget)
    return rc, ec, ra, ea


def group_stats(m, h, bounds, budget, *, center_radius=None):
    """One paid weighted-statistic/center program on a common source group.

    It does not build two radius-normalized interval vectors.  Sign credit is
    obtained on the coefficient intervals and multiplied downward by the same
    nonnegative enclosing radius.  Centers use four directed products and one
    batched reduction.  The optional cached normalization is checked against
    the full bounds on every coordinate, so a tuple is never an owned token.
    """
    for value in (m, h, bounds):
        sb._check(value, budget)
    if m.lo.ndim != 1 or h.lo.shape != m.lo.shape or bounds.lo.shape != m.lo.shape:
        raise Rejected('one complete common grouped source axis required')
    n = m.lo.size
    if n > 65536:
        raise Rejected('group exceeds fixed reduction width')
    if center_radius is None:
        _pay(budget, 12*n+32, 6*n+32)
        center = bounds.lo*.5+bounds.hi*.5
        left, right = center-bounds.lo, bounds.hi-center
        np.nextafter(left, np.inf, out=left)
        np.nextafter(right, np.inf, out=right)
        radius = np.maximum(left, right)
    else:
        _pay(budget, 18*n+48, 9*n+48)
        center, radius = center_radius
        if (type(center) is not np.ndarray or type(radius) is not np.ndarray
                or center.dtype != np.float64 or radius.dtype != np.float64
                or center.shape != (n,) or radius.shape != (n,)
                or not np.isfinite(center).all() or not np.isfinite(radius).all()
                or not (radius >= 0.).all()):
            raise Rejected('complete finite cached group normalization required')
        # Upper roundoff makes this a sufficient enclosure check, never a
        # rounded subtraction that could understate a required radius.
        left = np.nextafter(center-bounds.lo, np.inf)
        right = np.nextafter(bounds.hi-center, np.inf)
        if not (radius >= np.maximum(left, right)).all():
            raise Rejected('cached radius does not enclose the actual group')

    _pay(budget, 49*n+128, 16*n+128)
    values = np.empty((n, 4), dtype=np.float64)
    scratch = np.empty(n, dtype=np.float64)
    for column, lo, hi in ((0, m.lo, m.hi), (1, h.lo, h.hi)):
        np.absolute(lo, out=values[:, column])
        np.absolute(hi, out=scratch)
        np.maximum(values[:, column], scratch, out=values[:, column])
        np.multiply(values[:, column], radius, out=values[:, column])
        np.nextafter(values[:, column], np.inf, out=values[:, column])
    amin, bmin = np.negative(m.hi), np.negative(h.hi)
    np.maximum(amin, m.lo, out=amin)
    np.maximum(bmin, h.lo, out=bmin)
    np.maximum(amin, 0., out=amin)
    np.maximum(bmin, 0., out=bmin)
    np.multiply(bmin, 2., out=bmin)
    np.minimum(amin, bmin, out=amin)
    np.multiply(amin, radius, out=amin)
    np.nextafter(amin, -np.inf, out=amin)
    np.maximum(amin, 0., out=amin)
    mp, mn, hp, hn = m.lo > 0., m.hi < 0., h.lo > 0., h.hi < 0.
    same, opposite = np.logical_and(mp, hp), np.logical_and(mp, hn)
    temp = np.logical_and(mn, hn)
    np.logical_or(same, temp, out=same)
    np.logical_and(mn, hp, out=temp)
    np.logical_or(opposite, temp, out=opposite)
    values[:, 2:] = 0.
    np.copyto(values[:, 2], amin, where=same)
    np.copyto(values[:, 3], amin, where=opposite)
    _pay(budget, 4*n+768, 320)
    totals = np.sum(values, axis=0, dtype=np.float64)
    stats = []
    grid = math.ldexp(1., -256)
    nu = math.ldexp(float(n), -53)
    gamma = math.nextafter(nu/(1.-nu), math.inf)
    for index, total in enumerate(totals):
        total = float(total)
        if not math.isfinite(total) or total < 0.:
            raise Rejected('nonfinite grouped statistic')
        if not n or total == 0.:
            stats.append(0.)
            continue
        spacing = math.nextafter(total, math.inf)-total
        abs_error = math.nextafter(float(n)*max(spacing, grid), math.inf)
        abs_upper = math.nextafter(total+abs_error, math.inf)
        error = math.nextafter(gamma*abs_upper, math.inf)
        error = math.nextafter(error+float(n)*grid, math.inf)
        upward = index < 2
        raw = math.nextafter(total+error if upward else total-error,
                             math.inf if upward else -math.inf)
        endpoint = js._endpoint(raw, upward, budget)
        stats.append(endpoint if upward else max(0., endpoint))

    _pay(budget, 18*n+48, 9*n+48)
    centers = np.empty((n, 4), dtype=np.float64)
    positive = center >= 0.
    for offset, coeff in ((0, m), (2, h)):
        np.multiply(np.where(positive, coeff.lo, coeff.hi), center, out=centers[:, offset])
        np.multiply(np.where(positive, coeff.hi, coeff.lo), center, out=centers[:, offset+1])
        np.negative(centers[:, offset], out=centers[:, offset])
    np.nextafter(centers, np.inf, out=centers)
    upper, _ = sb._axis_sum_endpoint(centers, 0, True, budget)
    _pay(budget, 32, 32)
    rc = I(np.asarray(-js._endpoint(float(upper[0]), True, budget)),
           np.asarray(js._endpoint(float(upper[1]), True, budget)))
    ec = I(np.asarray(-js._endpoint(float(upper[2]), True, budget)),
           np.asarray(js._endpoint(float(upper[3]), True, budget)))
    return rc, ec, tuple(stats)


def statistics(a, b, budget):
    """Full checks on internal finite arrays; no forged owned fast path.

    The proof requires bounded interval endpoints, not that every directed
    intermediate endpoint is a stored 512-bit scalar coefficient.  Only
    exported scalar evidence uses the D259 lattice.  Both arrays are read on
    all coordinates before reusing the already qualified four-statistic core.
    """
    sb._check(a, budget)
    sb._check(b, budget)
    _pay(budget, 32, 32)
    if a.lo.ndim != 1 or b.lo.shape != a.lo.shape or a.lo.size > 65536:
        raise Rejected('complete one-dimensional common source axis required')
    return js._statistics(a, b, budget)


def combine_stats(values, budget):
    _pay(budget, 20*len(values)+32, 12*len(values)+32)
    data = np.asarray(values, dtype=np.float64).reshape((-1, 4))
    if not np.isfinite(data).all() or not (data >= 0.0).all():
        raise Rejected('finite nonnegative source statistics required')
    result = []
    for i in range(4):
        upward = i < 2
        endpoint = sb._fast_sum_endpoint(data[:, i], upward, budget)
        endpoint = js._endpoint(endpoint, upward, budget)
        result.append(endpoint if upward else max(0.0, endpoint))
    return tuple(result)


def support(stats, r0, e0, budget):
    """Complete r range and D263-centered joint bound on the SAME sources."""
    _pay(budget, 160, 64)
    if len(stats) != 4 or not all(type(v) is float and math.isfinite(v) and v >= 0
                                 for v in stats):
        raise Rejected('four finite nonnegative statistics required')
    sb._check(r0, budget)
    sb._check(e0, budget)
    if r0.lo.shape != () or e0.lo.shape != ():
        raise Rejected('scalar residual centers required')
    ra, eb, same, opposite = stats
    lr = -js._up_add(-float(r0.lo), ra, budget)
    ur = js._up_add(float(r0.hi), ra, budget)
    mu = (Fraction.from_float(lr)+Fraction.from_float(ur))/2
    if max(abs(mu.numerator).bit_length(), mu.denominator.bit_length()) > 512:
        raise Rejected('residual midpoint exceeds bit cap')
    mi = sb.from_rationals([(mu, mu)], budget=budget, enabled=True)
    _pay(budget, 16, 16)
    delta = sub(r0, I(mi.lo.reshape(()), mi.hi.reshape(())), budget)
    dl, du = float(delta.lo), float(delta.hi)
    ehi = js._endpoint(float(e0.hi), True, budget)
    elo = js._endpoint(float(e0.lo), False, budget)
    magnitude = max(abs(dl), abs(du))
    base = js._up_add(ra, magnitude, budget)
    twice_e0, twice_e = js._twice(ehi, budget), js._twice(eb, budget)
    plus = js._up_add(du, twice_e0, budget)
    minus = js._up_add(-dl, twice_e0, budget)
    for value in (ra, twice_e):
        plus = js._up_add(plus, value, budget)
        minus = js._up_add(minus, value, budget)
    plus = js._up_add(plus, -js._twice(opposite, budget), budget)
    minus = js._up_add(minus, -js._twice(same, budget), budget)
    raw = max(base, plus, minus)
    ue = js._up_add(ehi, eb, budget)
    le = -js._up_add(-elo, eb, budget)
    independent = js._up_add(base, js._twice(max(ue, 0.0), budget), budget)
    _pay(budget, 48, 48)
    return dict(upper=min(raw, independent), independent_upper=independent,
                raw_upper=raw, r_bounds=(lr, ur), e_bounds=(le, ue),
                mu=(mu.numerator, mu.denominator), statistics=tuple(stats))


def _point_dots(p1, p2, z, budget):
    """Five exact-binary64 dot products enclosed by one matrix reduction.

    The gamma bound includes multiplication rounding, not just summation.
    The sum of rounded absolute products is first enclosed, then divided by
    1-u to cover exact products; the 2**-256 grid pays gradual underflow.
    """
    n = p1.size
    if n > 65536:
        raise Rejected('Gram stencil exceeds the reduction width')
    _pay(budget, 30*n+1536, 15*n+768)
    products = np.empty((n, 5), dtype=np.float64)
    for index, (left, right) in enumerate(((p1, p1), (p1, p2), (p2, p2), (p1, z), (p2, z))):
        np.multiply(left, right, out=products[:, index])
    totals = np.sum(products, axis=0, dtype=np.float64)
    absolutes = np.sum(np.abs(products), axis=0, dtype=np.float64)
    grid = math.ldexp(1., -256)
    nu = math.ldexp(float(n+1), -53)
    gamma = math.nextafter(nu/(1.-nu), math.inf)
    answers = []
    for total, absolute in zip(totals, absolutes):
        total, absolute = float(total), float(absolute)
        if not math.isfinite(total) or not math.isfinite(absolute):
            raise Rejected('nonfinite fixed-template Gram product')
        spacing = math.nextafter(absolute, math.inf)-absolute
        abs_error = math.nextafter(float(n)*max(spacing, grid), math.inf)
        abs_upper = math.nextafter(absolute+abs_error, math.inf)
        abs_upper = math.nextafter(abs_upper/math.nextafter(1., 0.), math.inf)
        error = math.nextafter(gamma*abs_upper, math.inf)
        error = math.nextafter(error+float(n)*grid, math.inf)
        lo = js._endpoint(math.nextafter(total-error, -math.inf), False, budget)
        hi = js._endpoint(math.nextafter(total+error, math.inf), True, budget)
        answers.append(I(np.asarray(lo), np.asarray(hi)))
    return tuple(answers)


def gram(p1, p2, z, budget):
    """Uniform interval-certified 2x2 Gram choice, with full residual paid later.

    Inputs share the entire radius-normalized parent stencil and original
    parent BN error axes. Fixed midpoint coefficients define the template
    Gram; this is not a claim of exact projection on uncertain actual sources.
    The caller may use the same declared choice on all
    padding masks; zero padding is still applied to each actual readout.
    A nonpositive determinant lower bound returns None, not a second policy.
    Resource failures and other malformed arithmetic are never caught here.
    """
    if p1.lo.ndim != 1 or p1.lo.shape != p2.lo.shape or p1.lo.shape != z.lo.shape:
        raise Rejected('full common Gram frontier required')
    vectors = tuple(midpoint(value, budget) for value in (p1, p2, z))
    g11, g12, g22, h1, h2 = _point_dots(*vectors, budget)
    determinant = sub(mul(g11, g22, budget), mul(g12, g12, budget), budget)
    sb._check(determinant, budget)
    _pay(budget, 16, 8)
    if float(determinant.lo) <= 0.0:
        return None
    first = sub(mul(h1, g22, budget), mul(h2, g12, budget), budget)
    second = sub(mul(h2, g11, budget), mul(h1, g12, budget), budget)
    _pay(budget, 16, 8)
    return (float(midpoint(div(first, determinant, budget), budget)),
            float(midpoint(div(second, determinant, budget), budget)))
