"""Opt-in rational reference for certified local phase capacities, not a verifier."""
from dataclasses import dataclass
from fractions import Fraction


class Rejected(ValueError):
    """An unsupported or uncertified input never produces capacity rows."""


ZERO = Fraction(0)
ONE = Fraction(1)
WORK_CAP = 256_000_000
NESTED_CAP = 200_000_000
ENTRY_CAP = 64_000_000
BIT_CAP = 512
DIM_CAP = 128


@dataclass(frozen=True)
class AffineRow:
    frame_id: str
    coefficients: tuple
    bias: Fraction
    phase_id: str


@dataclass(frozen=True)
class Frame:
    identity: str
    lower: tuple
    upper: tuple
    rows: tuple


@dataclass(frozen=True)
class PairCertificate:
    positive: int
    negative: int
    amount: Fraction
    difference_lower: Fraction
    difference_upper: Fraction
    positive_capacity: Fraction
    negative_capacity: Fraction


@dataclass(frozen=True)
class CapacityRows:
    frame_id: str
    phase_ids: tuple
    positive_constant: Fraction
    negative_constant: Fraction
    positive: tuple
    negative: tuple
    predecessor_bounds: tuple
    pairs: tuple
    added_nnz: int
    reference_retained_entries: int
    arithmetic_work: int


class Arithmetic:
    def __init__(self, work_limit):
        if type(work_limit) is not int or not 0 <= work_limit <= WORK_CAP:
            raise Rejected('invalid whole work limit')
        self.limit = min(work_limit, NESTED_CAP)
        self.work = 0

    def tick(self):
        if self.work >= self.limit:
            raise Rejected('numerical work cap reached')
        self.work += 1

    def rational(self, value):
        self.tick()
        if type(value) is not Fraction:
            raise Rejected('exact Fraction values required')
        if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > BIT_CAP:
            raise Rejected('rational bit cap reached')
        return value

    def add(self, a, b):
        self.tick()
        return self.rational(a + b)

    def sub(self, a, b):
        self.tick()
        return self.rational(a - b)

    def mul(self, a, b):
        self.tick()
        return self.rational(a * b)

    def less(self, a, b):
        self.tick()
        return a < b

    def minimum(self, a, b):
        return a if self.less(a, b) else b

    def positive_part(self, value):
        return value if self.less(ZERO, value) else ZERO


def _interval(coefficients, bias, lower, upper, arithmetic):
    lo = hi = bias
    for coefficient, left, right in zip(coefficients, lower, upper):
        endpoints = (right, left) if arithmetic.less(coefficient, ZERO) else (left, right)
        lo = arithmetic.add(lo, arithmetic.mul(coefficient, endpoints[0]))
        hi = arithmetic.add(hi, arithmetic.mul(coefficient, endpoints[1]))
    return lo, hi


def compile_capacities(frame, weights, bias, *, enabled=False, work_limit=WORK_CAP):
    """Return two valid receiver rows; never deletes gates/bits or decides a property."""
    if enabled is not True:
        raise Rejected('explicit enabled=True required')
    arithmetic = Arithmetic(work_limit)
    if type(frame) is not Frame or type(frame.identity) is not str or not frame.identity:
        raise Rejected('one identified source frame required')
    if any(type(values) is not tuple for values in (frame.lower, frame.upper, frame.rows, weights)):
        raise Rejected('immutable tuples required')
    dimension, width = len(frame.lower), len(frame.rows)
    if not 1 <= dimension <= DIM_CAP or not 1 <= width <= DIM_CAP:
        raise Rejected('reference dimension cap reached')
    if len(frame.upper) != dimension or len(weights) != width:
        raise Rejected('shape mismatch')
    bias = arithmetic.rational(bias)
    for lower, upper in zip(frame.lower, frame.upper):
        arithmetic.rational(lower)
        arithmetic.rational(upper)
        if arithmetic.less(upper, lower):
            raise Rejected('reversed source interval')
    identities = set()
    bounds = []
    for row, weight in zip(frame.rows, weights):
        if (type(row) is not AffineRow or type(row.frame_id) is not str
                or row.frame_id != frame.identity):
            raise Rejected('shared source identity mismatch')
        if (type(row.phase_id) is not str or not row.phase_id or row.phase_id in identities
                or type(row.coefficients) is not tuple or len(row.coefficients) != dimension):
            raise Rejected('invalid original phase identity or row shape')
        identities.add(row.phase_id)
        arithmetic.rational(weight)
        arithmetic.rational(row.bias)
        for coefficient in row.coefficients:
            arithmetic.rational(coefficient)
        lo, hi = _interval(row.coefficients, row.bias, frame.lower, frame.upper, arithmetic)
        if not arithmetic.less(lo, ZERO) or not arithmetic.less(ZERO, hi):
            raise Rejected('only strictly crossing predecessor bounds qualified')
        bounds.append((lo, hi))
    positive = [arithmetic.positive_part(weight) for weight in weights]
    negative = [arithmetic.positive_part(arithmetic.sub(ZERO, weight)) for weight in weights]
    matched_positive = [ZERO] * width
    matched_negative = [ZERO] * width
    pairs = []
    for first in range(0, width - 1, 2):
        second = first + 1
        if arithmetic.less(ZERO, weights[first]) and arithmetic.less(weights[second], ZERO):
            pos, neg = first, second
        elif arithmetic.less(weights[first], ZERO) and arithmetic.less(ZERO, weights[second]):
            pos, neg = second, first
        else:
            continue
        amount = arithmetic.minimum(positive[pos], negative[neg])
        left, right = frame.rows[pos], frame.rows[neg]
        coefficients = tuple(arithmetic.sub(a, b) for a, b in zip(left.coefficients, right.coefficients))
        difference_bias = arithmetic.sub(left.bias, right.bias)
        lo, hi = _interval(coefficients, difference_bias, frame.lower, frame.upper, arithmetic)
        if not arithmetic.less(lo, ZERO) or not arithmetic.less(ZERO, hi):
            raise Rejected('only strictly crossing selected difference bounds qualified')
        up = arithmetic.minimum(hi, bounds[pos][1])
        down = arithmetic.minimum(arithmetic.sub(ZERO, lo), bounds[neg][1])
        positive[pos] = arithmetic.sub(positive[pos], amount)
        negative[neg] = arithmetic.sub(negative[neg], amount)
        matched_positive[pos] = arithmetic.mul(amount, up)
        matched_negative[neg] = arithmetic.mul(amount, down)
        pairs.append(PairCertificate(pos, neg, amount, lo, hi, up, down))
    positive = tuple(arithmetic.add(arithmetic.mul(remainder, bound[1]), matched)
                     for remainder, bound, matched in zip(positive, bounds, matched_positive))
    negative = tuple(arithmetic.add(arithmetic.mul(remainder, bound[1]), matched)
                     for remainder, bound, matched in zip(negative, bounds, matched_negative))
    positive_constant = arithmetic.positive_part(bias)
    negative_constant = arithmetic.positive_part(arithmetic.sub(ZERO, bias))
    # Input values, returned rational values, pair rational and index fields, row counts.
    entries = 2 * dimension + width * (dimension + 2) + 1 + 4 * width + 2 + 7 * len(pairs) + 3
    if entries > ENTRY_CAP:
        raise Rejected('retained entry cap reached')
    nnz = (3 + sum(arithmetic.less(ZERO, value) for value in positive)
           + sum(arithmetic.less(ZERO, value) for value in negative))
    return CapacityRows(frame.identity, tuple(row.phase_id for row in frame.rows),
                        positive_constant, negative_constant, positive, negative,
                        tuple(bounds), tuple(pairs), nnz, entries, arithmetic.work)
