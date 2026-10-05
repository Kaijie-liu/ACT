"""Bounded rational exponential intervals; no floating-point oracle.

The fixed positive Taylor sum has terms 0 through 63.  Every recurrence is
rounded outwards to 72 binary fractional places.  A geometric tail, repeated
squaring, and (for negative inputs) an outward reciprocal complete the proof.
The reciprocal uses 168 places so the lower endpoint at -64 remains positive.
Work units are charged arithmetic/rounding operations, not bit complexity or
physical runtime.  All produced Fraction intermediates pass the 512-bit gate.
"""
from fractions import Fraction as F


MAX_BITS = 512
MAX_WORK = 256_000_000
PRECISION = 72
RECIPROCAL_PRECISION = 168
TERMS = 64
ZERO, ONE, HALF = F(0), F(1), F(1, 2)


class Budget:
    __slots__ = ('max_work', 'work')

    def __init__(self, max_work=MAX_WORK):
        if type(max_work) is not int or not 0 < max_work <= MAX_WORK:
            raise ValueError('work cap')
        self.max_work = max_work
        self.work = 0

    def charge(self, amount=1):
        if type(amount) is not int or amount < 0:
            raise ValueError('invalid work charge')
        if self.work + amount > self.max_work:
            raise ValueError('work cap')
        self.work += amount


def rational(value):
    if type(value) is int:
        value = F(value)
    if type(value) is not F:
        raise ValueError('exact rational required')
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_BITS:
        raise ValueError('rational bit cap')
    return value


def add(a, b, budget):
    budget.charge()
    return rational(rational(a) + rational(b))


def sub(a, b, budget):
    budget.charge()
    return rational(rational(a) - rational(b))


def mul(a, b, budget):
    budget.charge()
    return rational(rational(a) * rational(b))


def div(a, b, budget):
    budget.charge()
    a, b = rational(a), rational(b)
    if not b:
        raise ValueError('zero denominator')
    return rational(a / b)


def _round(value, bits, upward, budget):
    value = rational(value)
    budget.charge(2)
    numerator = value.numerator << bits
    denominator = value.denominator
    rounded = -((-numerator) // denominator) if upward else numerator // denominator
    return rational(F(rounded, 1 << bits))


def exp_bounds(value, *, budget=None):
    """Return rational lo <= exp(value) <= hi for Fraction value in [-64,64].

    Invalid types, arithmetic width, range, and resource failures raise
    ValueError; they never switch to a float approximation or another oracle.
    """
    if type(value) is not F:
        raise ValueError('Fraction required')
    value = rational(value)
    if budget is None:
        budget = Budget()
    if (type(budget) is not Budget or type(budget.work) is not int
            or type(budget.max_work) is not int
            or not 0 < budget.max_work <= MAX_WORK
            or not 0 <= budget.work <= budget.max_work):
        raise ValueError('valid work budget required')
    budget.charge(3)
    if not F(-64) <= value <= F(64):
        raise ValueError('exponential range')
    if not value:
        return ONE, ONE
    negative = value < ZERO
    reduced = -value if negative else value
    squarings = 0
    while reduced > HALF:
        budget.charge()
        reduced = div(reduced, F(2), budget)
        squarings += 1
        if squarings > 7:
            raise ValueError('range-reduction cap')
    rl = _round(reduced, PRECISION, False, budget)
    ru = _round(reduced, PRECISION, True, budget)
    term_lo = term_hi = sum_lo = sum_hi = ONE
    for k in range(1, TERMS):
        term_lo = _round(div(mul(term_lo, rl, budget), F(k), budget),
                         PRECISION, False, budget)
        term_hi = _round(div(mul(term_hi, ru, budget), F(k), budget),
                         PRECISION, True, budget)
        sum_lo = add(sum_lo, term_lo, budget)
        sum_hi = add(sum_hi, term_hi, budget)
    # The first omitted term is r^64/64!.  Every subsequent term ratio is
    # at most ru/65; positive-term monotonicity also covers rounded ru >= r.
    first = div(mul(term_hi, ru, budget), F(TERMS), budget)
    denominator = sub(ONE, div(ru, F(TERMS + 1), budget), budget)
    if denominator <= ZERO:
        raise ValueError('invalid Taylor tail denominator')
    tail = div(first, denominator, budget)
    lower = _round(sum_lo, PRECISION, False, budget)
    upper = _round(add(sum_hi, tail, budget), PRECISION, True, budget)
    for _ in range(squarings):
        lower = _round(mul(lower, lower, budget), PRECISION, False, budget)
        upper = _round(mul(upper, upper, budget), PRECISION, True, budget)
    if negative:
        return (_round(div(ONE, upper, budget), RECIPROCAL_PRECISION, False, budget),
                _round(div(ONE, lower, budget), RECIPROCAL_PRECISION, True, budget))
    return lower, upper
