"""Exact scalar JP row and an ordinary-old-hull redundancy certificate.

The caller owns source certification: a/b/tau, the complete r range, both
support upper bounds, and the old child ranges must refer to the SAME declared
H and mask.  No scalar value or boolean here authenticates that provenance.
In particular, independent_upper is supplied, never invented from missing e.

The comparison uses ONLY the old [-1, 1] physical ReLU triangles and
0 <= Y2, Y1 <= child_upper[0].  It does not infer a narrower triangle from a
forward output bound.  Failure of this sufficient redundancy test means
not_excluded, never strict strengthening, a verified network, or a new solve.

Every enabled operation uses the caller's shared meter before work/allocation.
Small exact arithmetic is checked after EVERY operation against the existing
512-bit reduced numerator/denominator cap.  Fraction comparisons have at most
1024-bit cross-products of those bounded operands; this is bounded temporary
integer scratch, not an exemption for a stored oversized rational.  Ordinary
schema rejection does not itself poison the meter; its resource failure
semantics are not intercepted.  There is no solver, phase search, or fallback.
"""

from fractions import Fraction
import math

from experiments.neural_hz_20260831.definition_first_20260928.d259_shared_difference_source_20261006 import source_bounds as sb


Rejected = sb.Rejected
_ZERO = Fraction(0)
_ONE = Fraction(1)


class _Exact:
    """Fixed-width logical arithmetic charges; not a new resource budget."""

    def __init__(self, budget):
        # Covers scalar schema checks, this wrapper, fixed argument tuples,
        # result dictionaries/strings/booleans and the small control loops.
        budget.charge(128, entries=64)
        self.budget = budget

    @staticmethod
    def checked(value):
        if max(abs(value.numerator).bit_length(),
               value.denominator.bit_length()) > 512:
            raise Rejected("JP rational exceeds the original 512-bit cap")
        return value

    def scalar(self, value):
        self.budget.charge(16, entries=4)
        if type(value) is not float or not math.isfinite(value):
            raise Rejected("JP parameters must be finite stored binary64 floats")
        return self.checked(Fraction.from_float(value))

    def add(self, left, right):
        self.budget.charge(8, entries=4)
        return self.checked(left + right)

    def sub(self, left, right):
        self.budget.charge(8, entries=4)
        return self.checked(left - right)

    def half(self, value):
        self.budget.charge(8, entries=4)
        return self.checked(value / 2)

    def twice(self, value):
        self.budget.charge(8, entries=4)
        return self.checked(value * 2)

    def neg(self, value):
        self.budget.charge(4, entries=4)
        return self.checked(-value)

    def absolute(self, value):
        self.budget.charge(4, entries=4)
        return self.checked(abs(value))

    def maximum(self, *values):
        self.budget.charge(8 * (len(values) - 1), entries=1)
        return max(values)

    def minimum(self, *values):
        self.budget.charge(8 * (len(values) - 1), entries=1)
        return min(values)

    def le(self, left, right):
        self.budget.charge(8, entries=0)
        return left <= right

    def triangle_support(self, x_coefficient, q_coefficient):
        # Vertices are (-1, 0), (0, 0), (1, 1).  Negative q coefficients
        # are deliberately allowed: s0 need not be nonnegative.
        left = self.neg(x_coefficient)
        right = self.add(x_coefficient, q_coefficient)
        self.budget.charge(16, entries=5)
        values = (left, _ZERO, right)
        index, value = 0, values[0]
        for candidate in (1, 2):
            if values[candidate] > value:
                index, value = candidate, values[candidate]
        return value, index

    def pair(self, value):
        self.budget.charge(4, entries=2)
        return (value.numerator, value.denominator)


def certify(a, b, tau, r_bounds, joint_upper, independent_upper, child_upper,
            *, budget, enabled=False):
    """Return an exact row plus a sufficient old-hull exclusion certificate.

    All seven positional operands are fixed-size scalar records, not source
    arrays.  a, b, r_bounds and child_upper must be two-tuples of floats; the
    three other operands must be floats.  The row order is
    (x1, x2, q1, q2, Y1, Y2), with the caller's already-registered orientation.
    Exact numbers in the result are JSON-friendly (numerator, denominator)
    pairs.  The two child upper bounds are validated even though this one
    registered direction uses only Y1's upper bound and Y2's zero lower bound.

    A successful call has fixed logical charges of 952 work and 369 entries.
    This includes conversion, checked arithmetic, comparisons and the returned
    small record, not source construction or subsequent JSON serialization.
    The figures follow the explicit calls below, not a reserve or a wall-time
    claim.  Rejected calls retain whatever charges preceded the rejection.
    """
    if type(enabled) is not bool:
        raise Rejected("JP enabled must be a bool")
    if not enabled:
        return None
    m = _Exact(budget)
    if any(type(value) is not tuple or len(value) != 2
           for value in (a, b, r_bounds, child_upper)):
        raise Rejected("JP a/b/r/child ranges must be complete two-tuples")
    a1, a2 = (m.scalar(value) for value in a)
    b1, b2 = (m.scalar(value) for value in b)
    t = m.scalar(tau)
    lr, ur = (m.scalar(value) for value in r_bounds)
    jraw, jsep = m.scalar(joint_upper), m.scalar(independent_upper)
    y1upper, y2upper = (m.scalar(value) for value in child_upper)
    valid = (
        m.le(lr, ur), not m.le(t, _ZERO),
        m.le(_ZERO, jraw), m.le(_ZERO, jsep),
        m.le(_ZERO, y1upper), m.le(_ZERO, y2upper),
    )
    if not all(valid):
        raise Rejected("unordered ranges, nonpositive tau, or negative upper bound")

    j = m.minimum(jraw, jsep)
    payment_credit = not m.le(jsep, j)
    amean = m.half(m.add(a1, a2))
    aphase = m.half(m.sub(a1, a2))
    bmean = m.half(m.add(b1, b2))
    bphase = m.half(m.sub(b1, b2))
    mu = m.half(m.add(lr, ur))  # Exact center, never a rounded float midpoint.
    c = m.add(t, mu)
    chalf = m.half(c)
    aplus = m.maximum(amean, _ZERO)
    aminus = m.maximum(m.neg(amean), _ZERO)
    jplus = m.maximum(aphase, _ZERO)
    jminus = m.maximum(m.neg(aphase), _ZERO)

    l10 = m.sub(m.add(lr, m.minimum(_ZERO, m.add(a1, b1))),
                m.maximum(_ZERO, a2))
    u01 = m.add(m.sub(ur, m.minimum(_ZERO, a1)),
                m.maximum(_ZERO, m.add(a2, b2)))
    eta10minus = m.maximum(m.sub(m.neg(l10), t), _ZERO)
    eta01plus = m.maximum(m.sub(u01, t), _ZERO)
    c1 = m.add(m.add(chalf, aminus), jminus)
    c2 = m.add(m.neg(chalf), aminus)
    c3 = m.add(m.add(m.add(m.neg(chalf), aplus), jminus), m.twice(t))
    c4 = m.add(chalf, aplus)
    big_m = m.maximum(c1, c2, c3, c4)
    s0 = m.sub(big_m, jplus)
    qconstant = m.add(m.add(m.add(m.add(m.absolute(bmean),
        m.maximum(bphase, _ZERO)), eta10minus), eta01plus), j)
    rhs = m.add(m.twice(big_m), qconstant)
    wx1 = m.sub(m.neg(s0), chalf)
    wx2 = m.add(m.neg(s0), chalf)
    wq = m.twice(s0)

    support1, vertex1 = m.triangle_support(wx1, wq)
    support2, vertex2 = m.triangle_support(wx2, wq)
    support = m.add(m.add(support1, support2), y1upper)
    redundant = m.le(support, rhs)
    margin = m.sub(rhs, support)
    return {
        "physical_order": ("x1", "x2", "q1", "q2", "Y1", "Y2"),
        "row_coefficients": tuple(m.pair(value) for value in
                                  (wx1, wx2, wq, wq, _ONE, m.neg(_ONE))),
        "row_rhs": m.pair(rhs),
        "J": m.pair(j),
        "Jsep": m.pair(jsep),
        "payment_credit": payment_credit,
        "redundant": redundant,
        "status": "excluded" if redundant else "not_excluded",
        "support_witness": {
            "kind": "unit_parent_triangles_plus_Y1_upper_and_Y2_zero_lower",
            "parent_vertices": (vertex1, vertex2),
            "vertex_order": ((-1, 0), (0, 0), (1, 1)),
            "upper": m.pair(support),
            "rhs_minus_support": m.pair(margin),
        },
    }
