"""Bounded native-attention queries; no model or source qualification.

Positive token supports are exact geometrically.  Negative supports use an
explicit rectangle upper bound.  All arithmetic shares the caller's D127
Budget; this module never creates or resets one.  Frozen prepared handles are
authenticated by weak object identity, not by caller-supplied metadata/hash.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from functools import cmp_to_key
from weakref import WeakKeyDictionary

from experiments.neural_hz_20260831.definition_first_20260928.d127_native_attention_component_20261002 import exp_interval as ei


ZERO, ONE = F(0), F(1)
MAX_ENTRIES = 64_000_000
TERMS = 16


def _budget(budget):
    if (type(budget) is not ei.Budget or type(budget.work) is not int
            or type(budget.max_work) is not int
            or not 0 < budget.max_work <= ei.MAX_WORK
            or not 0 <= budget.work <= budget.max_work):
        raise ValueError('shared D127 Budget required')
    return budget


def _fraction(value):
    if type(value) is not F:
        raise ValueError('Fraction required')
    return ei.rational(value)


def exp16_bounds(value, budget):
    """Certified 16-term variant, not a precision-equivalent D127 oracle."""
    budget = _budget(budget)
    value = _fraction(value)
    budget.charge(3)
    if not F(-64) <= value <= F(64):
        raise ValueError('exponential range')
    if not value:
        return ONE, ONE
    negative = value < ZERO
    reduced = -value if negative else value
    squarings = 0
    while reduced > F(1, 2):
        budget.charge()
        reduced = ei.div(reduced, F(2), budget)
        squarings += 1
        if squarings > 7:
            raise ValueError('range-reduction cap')
    rl = ei._round(reduced, ei.PRECISION, False, budget)
    ru = ei._round(reduced, ei.PRECISION, True, budget)
    term_lo = term_hi = sum_lo = sum_hi = ONE
    for k in range(1, TERMS):
        term_lo = ei._round(ei.div(ei.mul(term_lo, rl, budget), F(k), budget),
                            ei.PRECISION, False, budget)
        term_hi = ei._round(ei.div(ei.mul(term_hi, ru, budget), F(k), budget),
                            ei.PRECISION, True, budget)
        sum_lo = ei.add(sum_lo, term_lo, budget)
        sum_hi = ei.add(sum_hi, term_hi, budget)
    # First omitted term r^16/16!; all later successive ratios <= ru/17.
    first = ei.div(ei.mul(term_hi, ru, budget), F(TERMS), budget)
    denominator = ei.sub(ONE, ei.div(ru, F(TERMS+1), budget), budget)
    if denominator <= ZERO:
        raise ValueError('invalid Taylor tail denominator')
    tail = ei.div(first, denominator, budget)
    lower = ei._round(sum_lo, ei.PRECISION, False, budget)
    upper = ei._round(ei.add(sum_hi, tail, budget), ei.PRECISION, True, budget)
    for _ in range(squarings):
        lower = ei._round(ei.mul(lower, lower, budget), ei.PRECISION, False, budget)
        upper = ei._round(ei.mul(upper, upper, budget), ei.PRECISION, True, budget)
    if negative:
        return (ei._round(ei.div(ONE, upper, budget), ei.RECIPROCAL_PRECISION, False, budget),
                ei._round(ei.div(ONE, lower, budget), ei.RECIPROCAL_PRECISION, True, budget))
    return lower, upper


@dataclass(frozen=True, eq=False)
class Prepared:
    upper: tuple
    slopes: tuple
    rightmost_max: int
    value_min: F
    value_max: F
    score_min: F
    score_max: F
    entries: int
    entry_upper: int


class _Ticket:
    __slots__ = ('budget', 'limit', 'snapshot', 'last_work', 'shift', 'endpoints')

    def __init__(self, budget, snapshot):
        self.budget = budget
        self.limit = budget.max_work
        self.snapshot = snapshot
        self.last_work = budget.work
        self.shift = None
        self.endpoints = None


_ISSUED = WeakKeyDictionary()


def _snapshot(prepared):
    return (prepared.upper, prepared.slopes, prepared.rightmost_max,
            prepared.value_min, prepared.value_max, prepared.score_min,
            prepared.score_max, prepared.entries, prepared.entry_upper)


def _check(prepared, budget):
    _budget(budget)
    if type(prepared) is not Prepared:
        raise ValueError('issued Prepared required')
    ticket = _ISSUED.get(prepared)
    if (ticket is None or ticket.budget is not budget
            or ticket.limit != budget.max_work or budget.work < ticket.last_work
            or any(a is not b for a, b in zip(_snapshot(prepared), ticket.snapshot))):
        raise ValueError('untrusted or altered prepared geometry/budget')
    budget.charge(15)
    ticket.last_work = budget.work
    return ticket


def prepare_polygon(vertices, budget):
    """Prepare the convex hull of all supplied rational points, once.

    A caller binding a model must independently authenticate that these
    points enclose its score/value image.  This function grants no such claim.
    Vertical duplicates retain their upper endpoint; the global value minimum
    still includes every input point for a safe initial root bracket.
    """
    _budget(budget)
    if type(vertices) is not tuple or not vertices:
        raise ValueError('nonempty immutable vertices required')
    entry_upper = 32*len(vertices)+128
    if entry_upper > MAX_ENTRIES:
        raise ValueError('entry cap')
    # Input checks, container visits, sorted-copy/hull storage and registry
    # allocation.  Arithmetic and every sort/cross comparison are additional.
    budget.charge(8*len(vertices)+48)
    for point in vertices:
        if type(point) is not tuple or len(point) != 2:
            raise ValueError('immutable two-coordinate point required')
        _fraction(point[0])
        _fraction(point[1])

    def compare(a, b):
        budget.charge(3)
        if a[0] != b[0]:
            return -1 if a[0] < b[0] else 1
        return -1 if a[1] < b[1] else (1 if a[1] > b[1] else 0)

    ordered = sorted(vertices, key=cmp_to_key(compare))
    unique = []
    value_min = value_max = vertices[0][1]
    for score, value in ordered:
        budget.charge(4)
        value_min, value_max = min(value_min, value), max(value_max, value)
        if unique and unique[-1][0] == score:
            unique[-1] = (score, value)
        else:
            unique.append((score, value))
    upper = []
    for point in unique:
        budget.charge()
        while len(upper) >= 2:
            a, b = upper[-2], upper[-1]
            left = ei.mul(ei.sub(b[0], a[0], budget),
                          ei.sub(point[1], b[1], budget), budget)
            right = ei.mul(ei.sub(b[1], a[1], budget),
                           ei.sub(point[0], b[0], budget), budget)
            budget.charge()
            if ei.sub(left, right, budget) < ZERO:
                break
            upper.pop()
            budget.charge()
        upper.append(point)
    upper = tuple(upper)
    slopes = []
    rightmost_max = 0
    for index, point in enumerate(upper):
        budget.charge(3)
        if point[1] >= upper[rightmost_max][1]:
            rightmost_max = index
        if index:
            previous = upper[index-1]
            slopes.append(ei.div(ei.sub(point[1], previous[1], budget),
                                 ei.sub(point[0], previous[0], budget), budget))
    slopes = tuple(slopes)
    entries = 3*len(upper)+32
    if entries > entry_upper:
        raise ValueError('invalid entry reservation')
    prepared = Prepared(upper, slopes, rightmost_max, value_min, value_max,
                        upper[0][0], upper[-1][0], entries, entry_upper)
    budget.charge(12)
    _ISSUED[prepared] = _Ticket(budget, _snapshot(prepared))
    return prepared


def _endpoints(prepared, shift, budget, ticket):
    budget.charge(3)
    if ticket.shift == shift and ticket.endpoints is not None:
        return ticket.endpoints
    left = exp16_bounds(ei.sub(prepared.score_min, shift, budget), budget)
    if prepared.score_min == prepared.score_max:
        right = left
    else:
        right = exp16_bounds(ei.sub(prepared.score_max, shift, budget), budget)
    # One bounded immutable cache entry per issued handle, never an unbounded
    # per-threshold dictionary.  Changing shift discards the former pair.
    ticket.shift = shift
    ticket.endpoints = (left, right)
    budget.charge(8)
    return ticket.endpoints


def _scale_interval(interval, scalar, budget):
    budget.charge()
    if scalar >= ZERO:
        return ei.mul(interval[0], scalar, budget), ei.mul(interval[1], scalar, budget)
    return ei.mul(interval[1], scalar, budget), ei.mul(interval[0], scalar, budget)


def _rectangle(prepared, t, endpoints, budget):
    residual = ei.sub(prepared.value_max, t, budget)
    budget.charge()
    return _scale_interval(endpoints[1] if residual > ZERO else endpoints[0],
                           residual, budget)


def _positive_point(prepared, t, budget):
    upper, slopes = prepared.upper, prepared.slopes
    k, last = prepared.rightmost_max, len(upper)-1
    budget.charge(3)
    if k == last:
        return upper[k], 'vertex'

    def derivative(index):
        return ei.add(ei.sub(upper[index][1], t, budget), slopes[index], budget)

    budget.charge()
    if derivative(k) <= ZERO:
        return upper[k], 'vertex'
    # On this decreasing upper-hull tail the right derivatives decrease.
    # The final vertex is a sentinel; its incoming derivative decides whether
    # the last maximum lies in the last edge or at the endpoint.
    lo, hi = k+1, last
    while lo < hi:
        budget.charge(4)
        mid = (lo+hi)//2
        if derivative(mid) <= ZERO:
            hi = mid
        else:
            lo = mid+1
    j = lo
    incoming = ei.add(ei.sub(upper[j][1], t, budget), slopes[j-1], budget)
    budget.charge()
    if incoming >= ZERO:
        return upper[j], 'vertex'
    b = slopes[j-1]
    left_d = derivative(j-1)
    budget.charge(2)
    if not b < ZERO or not left_d > ZERO:
        raise ValueError('inconsistent positive-tail derivative')
    score = ei.sub(upper[j-1][0], ei.div(left_d, b, budget), budget)
    value = ei.sub(t, b, budget)
    return (score, value), 'stationary'


def threshold(prepared, t, shift, budget):
    """Enclose exp(-shift) H(t), not the exact general-negative support.

    Only a nonpositive upper endpoint certifies a threshold.  A positive
    lower endpoint is never an ADV witness or a correlated-domain lower bound.
    """
    ticket = _check(prepared, budget)
    t, shift = _fraction(t), _fraction(shift)
    budget.charge(3)
    endpoints = _endpoints(prepared, shift, budget, ticket)
    rectangle = _rectangle(prepared, t, endpoints, budget)
    if t >= prepared.value_max:
        interval = rectangle
        point, kind = None, 'zero' if t == prepared.value_max else 'negative_rectangle'
        exact = t == prepared.value_max or prepared.score_min == prepared.score_max
    else:
        point, kind = _positive_point(prepared, t, budget)
        budget.charge(3)
        if point[0] == prepared.score_min:
            exponential = endpoints[0]
        elif point[0] == prepared.score_max:
            exponential = endpoints[1]
        else:
            exponential = exp16_bounds(ei.sub(point[0], shift, budget), budget)
        residual = ei.sub(point[1], t, budget)
        if residual <= ZERO:
            raise ValueError('nonpositive positive-branch maximizer')
        interval = _scale_interval(exponential, residual, budget)
        exact = True
    ticket.last_work = budget.work
    return dict(h_interval=interval, rectangle_interval=rectangle,
                branch=kind, maximizer=point, exact_polygon_support=exact,
                quantity='shifted_piecewise_H_not_general_exact_F',
                work=budget.work, model_binding_qualified=False,
                native_binding_qualified=False, adv_witness_returned=False)


def _sum_intervals(intervals, budget):
    lower = upper = ZERO
    for lo, hi in intervals:
        budget.charge()
        lower, upper = ei.add(lower, lo, budget), ei.add(upper, hi, budget)
    return lower, upper


def bound(tokens, budget, steps=12):
    """Safe upper via the root of sum H, plus a matched rectangle comparator.

    lo brackets the ROOT OF sum H, not the native attention's attainable
    output.  To bound actual outputs below, query the opposite direction.
    Unresolved interval signs retain a safe bracket without a precision claim.
    """
    _budget(budget)
    if (type(tokens) is not tuple or not tokens or type(steps) is not int
            or not 1 <= steps <= 12):
        raise ValueError('immutable tokens and 1..12 steps required')
    budget.charge(16*len(tokens)+32)
    tickets = tuple(_check(token, budget) for token in tokens)
    entries = sum(token.entries for token in tokens)+32*len(tokens)+64
    entry_upper = sum(token.entry_upper for token in tokens)+64*len(tokens)+64
    if entry_upper > MAX_ENTRIES:
        raise ValueError('aggregate query entry cap')
    shift = max(token.score_max for token in tokens)
    initial_lo = min(token.value_min for token in tokens)
    initial_hi = max(token.value_max for token in tokens)
    endpoints = tuple(_endpoints(token, shift, budget, ticket)
                      for token, ticket in zip(tokens, tickets))
    lo, hi = initial_lo, initial_hi
    rectangle_lo, rectangle_hi = lo, hi
    completed = rectangle_completed = 0
    uncertain = rectangle_uncertain = False
    h_last = rectangle_last = None
    for _ in range(steps):
        budget.charge()
        if lo == hi:
            break
        mid = ei.div(ei.add(lo, hi, budget), F(2), budget)
        results = tuple(threshold(token, mid, shift, budget) for token in tokens)
        budget.charge(2*len(results))
        h_last = _sum_intervals(tuple(row['h_interval'] for row in results), budget)
        if h_last == (ZERO, ZERO):
            lo = hi = mid
        elif h_last[1] <= ZERO:
            hi = mid
        elif h_last[0] > ZERO:
            lo = mid
        else:
            uncertain = True
            break
        completed += 1
    # The rectangle root has its own bisection points.  Its endpoint exp
    # intervals are already certified and cached; no new exp call is needed.
    for _ in range(steps):
        budget.charge()
        if rectangle_lo == rectangle_hi:
            break
        mid = ei.div(ei.add(rectangle_lo, rectangle_hi, budget), F(2), budget)
        rows = tuple(_rectangle(token, mid, endpoint, budget)
                     for token, endpoint in zip(tokens, endpoints))
        budget.charge(len(rows))
        rectangle_last = _sum_intervals(rows, budget)
        if rectangle_last == (ZERO, ZERO):
            rectangle_lo = rectangle_hi = mid
        elif rectangle_last[1] <= ZERO:
            rectangle_hi = mid
        elif rectangle_last[0] > ZERO:
            rectangle_lo = mid
        else:
            rectangle_uncertain = True
            break
        rectangle_completed += 1
    raw_h_hi = hi
    budget.charge(16+3*len(tokens))
    hi = min(hi, rectangle_hi)
    if lo > hi:
        raise ValueError('inconsistent root enclosures')
    for ticket in tickets:
        ticket.last_work = budget.work
    return dict(lo=lo, hi=hi, raw_h_hi=raw_h_hi, shift=shift,
                initial_interval=(initial_lo, initial_hi),
                precision_certified=not uncertain and (completed == steps or lo == hi),
                steps_completed=completed, uncertain=uncertain,
                rectangle_lo=rectangle_lo, rectangle_hi=rectangle_hi,
                rectangle_precision_certified=not rectangle_uncertain and
                    (rectangle_completed == steps or rectangle_lo == rectangle_hi),
                rectangle_steps_completed=rectangle_completed,
                rectangle_uncertain=rectangle_uncertain,
                last_h_interval=h_last, last_rectangle_interval=rectangle_last,
                quantity='root_of_sum_H_not_actual_attention_lower',
                work=budget.work, entries=entries, entry_upper=entry_upper,
                exponential_terms=TERMS,
                model_binding_qualified=False, native_binding_qualified=False,
                complete_physical_qualification=False, gpu_qualified=False,
                adv_witness_returned=False, formal_gain=0)
