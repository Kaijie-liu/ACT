"""Opt-in, fail-closed rational interval primitives for a source-only census.

Draft only: no solver, model loader, shield selection, phase search, or score
updates. Intervals contain Fraction endpoints; an affine form is
(constant_interval, {shared_original_input_id: coefficient_interval}).
Coefficient intervals enclose fixed source coefficients, not independent new
source variables. Their arithmetic is an outward enclosure, not a claim of
exact dependence between uncertain coefficients.

The 512-bit bound applies to every stored rational endpoint (numerator and
denominator). Intermediate Fraction arithmetic may briefly use larger integers.
The work ledger counts bounded-size scalar operations and container visits,
not elapsed time or bit operations. Parsing, cache/proof metadata, output
encoding, and per-model 200M work limits remain the caller's responsibility.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from math import isqrt


class KernelError(ValueError):
    """Unsupported or invalid evidence; the caller must fail closed."""


class BudgetExceeded(RuntimeError):
    """The predeclared scalar-work cap would be exceeded."""


class KernelDisabled(RuntimeError):
    """Numeric work requires explicit opt-in."""


@dataclass
class WorkBudget:
    enabled: bool = False
    limit: int = 256_000_000
    max_bits: int = 512
    used: int = field(default=0, init=False)

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise KernelError("enabled must be a bool")
        if type(self.limit) is not int or not 0 <= self.limit <= 256_000_000:
            raise KernelError("invalid whole-work limit")
        if type(self.max_bits) is not int or not 1 <= self.max_bits <= 512:
            raise KernelError("endpoint limit must be between 1 and 512 bits")

    def charge(self, amount):
        if not self.enabled:
            raise KernelDisabled("explicit enabled=True is required")
        if type(amount) is not int or amount < 0:
            raise KernelError("work charge must be a nonnegative integer")
        if amount > self.limit - self.used:
            raise BudgetExceeded("whole-work limit exceeded before operation")
        self.used += amount


def _checked(interval, budget):
    budget.charge(3)
    if type(interval) is not tuple or len(interval) != 2:
        raise KernelError("an interval must be a pair of Fraction endpoints")
    for endpoint in interval:
        if not isinstance(endpoint, Fraction):
            raise KernelError("non-Fraction endpoint")
        if max(abs(endpoint.numerator).bit_length(),
               endpoint.denominator.bit_length()) > budget.max_bits:
            raise KernelError("rational endpoint exceeds bit bound")
    if interval[0] > interval[1]:
        raise KernelError("reversed interval")
    return interval


def point(value, budget):
    budget.charge(1)
    if type(value) is not int and not isinstance(value, Fraction):
        raise KernelError("point requires an exact int or Fraction")
    if type(value) is int and abs(value).bit_length() > budget.max_bits:
        raise KernelError("integer exceeds bit bound")
    value = Fraction(value)
    return _checked((value, value), budget)


def add(left, right, budget):
    left, right = _checked(left, budget), _checked(right, budget)
    budget.charge(2)
    return _checked((left[0] + right[0], left[1] + right[1]), budget)


def neg(value, budget):
    value = _checked(value, budget)
    budget.charge(2)
    return _checked((-value[1], -value[0]), budget)


def mul(left, right, budget):
    left, right = _checked(left, budget), _checked(right, budget)
    budget.charge(10)
    products = tuple(a * b for a in left for b in right)
    for product in products:
        _checked((product, product), budget)
    return _checked((min(products), max(products)), budget)


def div(numerator, denominator, budget):
    numerator = _checked(numerator, budget)
    denominator = _checked(denominator, budget)
    budget.charge(4)
    if denominator[0] <= 0 <= denominator[1]:
        raise KernelError("division by an interval containing zero")
    reciprocal = _checked((1 / denominator[1], 1 / denominator[0]), budget)
    return mul(numerator, reciprocal, budget)


def nonnegative_part(value, budget):
    """Enclose ReLU of an interval; its upper endpoint is a valid cap."""
    value = _checked(value, budget)
    budget.charge(2)
    zero = Fraction(0)
    return _checked((max(zero, value[0]), max(zero, value[1])), budget)


def dyadic_enclose(value, budget, bits=64):
    """Round outward to a fixed dyadic grid; never alter exact add/mul."""
    value = _checked(value, budget)
    if type(bits) is not int or not 0 <= bits <= 64:
        raise KernelError("dyadic precision must be between 0 and 64 bits")
    budget.charge(9)
    scale = 1 << bits
    lower = (value[0].numerator * scale) // value[0].denominator
    upper = -((-value[1].numerator * scale) // value[1].denominator)
    return _checked((Fraction(lower, scale), Fraction(upper, scale)), budget)


def sqrt_interval(value, budget):
    """Enclose sqrt(nonnegative Fraction) on the 2**-64 dyadic grid.

    For k = floor(sqrt(floor(n*2**128/d))), k**2*d <= n*2**128
    < (k+1)**2*d. The upper endpoint equals the lower only at equality.
    """
    point(value, budget)
    if not isinstance(value, Fraction) or value < 0:
        raise KernelError("sqrt requires a nonnegative Fraction point")
    budget.charge(12)
    scale = 1 << 64
    scaled_numerator = value.numerator << 128
    lower_integer = isqrt(scaled_numerator // value.denominator)
    exact = lower_integer * lower_integer * value.denominator == scaled_numerator
    upper_integer = lower_integer if exact else lower_integer + 1
    return _checked((Fraction(lower_integer, scale),
                     Fraction(upper_integer, scale)), budget)


def _form(form, budget):
    budget.charge(1)
    if type(form) is not tuple or len(form) != 2 or type(form[1]) is not dict:
        raise KernelError("affine form must be (interval, coefficient dict)")
    _checked(form[0], budget)
    budget.charge(len(form[1]))
    for source_id, coefficient in form[1].items():
        if type(source_id) is not int:
            raise KernelError("source IDs must be shared original-input integers")
        _checked(coefficient, budget)
    return form


def affine_add(left, right, budget):
    left, right = _form(left, budget), _form(right, budget)
    constant = add(left[0], right[0], budget)
    budget.charge(len(left[1]) + len(right[1]))
    coefficients = dict(left[1])
    zero = (Fraction(0), Fraction(0))
    for source_id, coefficient in right[1].items():
        value = add(coefficients.get(source_id, zero), coefficient, budget)
        budget.charge(1)
        if value == zero:
            coefficients.pop(source_id, None)
        else:
            coefficients[source_id] = value
    return constant, coefficients


def affine_scale(form, scalar, budget):
    form, scalar = _form(form, budget), _checked(scalar, budget)
    constant = mul(form[0], scalar, budget)
    coefficients = {}
    budget.charge(len(form[1]))
    for source_id, coefficient in form[1].items():
        value = mul(coefficient, scalar, budget)
        budget.charge(1)
        if value != (Fraction(0), Fraction(0)):
            coefficients[source_id] = value
    return constant, coefficients


def _source_bounds(box, source_id, budget):
    budget.charge(1)
    if type(box) is not dict or source_id not in box:
        raise KernelError("missing original-input box coordinate")
    return _checked(box[source_id], budget)


def source_box_bounds(form, box, budget):
    form = _form(form, budget)
    result = form[0]
    budget.charge(len(form[1]))
    for source_id, coefficient in form[1].items():
        bounds = _source_bounds(box, source_id, budget)
        result = add(result, mul(coefficient, bounds, budget), budget)
    return result


def evaluate_center(form, box, budget):
    form = _form(form, budget)
    result = form[0]
    budget.charge(len(form[1]))
    for source_id, coefficient in form[1].items():
        bounds = _source_bounds(box, source_id, budget)
        budget.charge(2)
        center = point((bounds[0] + bounds[1]) / 2, budget)
        result = add(result, mul(coefficient, center, budget), budget)
    return result


def orient_error(form, box, budget):
    """Return (tau, oriented source, ReLU source enclosure).

    tau is fixed, never a new phase bit. A center interval straddling zero
    chooses tau=0 conservatively. Then e(center) need NOT be zero: a caller
    cannot replace its chosen baseline at the center by the network value.
    For every fixed tau, ReLU(g) = tau*g + ReLU((1-2*tau)*g).
    """
    center = evaluate_center(form, box, budget)
    budget.charge(1)
    tau = 1 if center[0] >= 0 else 0
    oriented = affine_scale(form, point(1 - 2 * tau, budget), budget)
    cap = nonnegative_part(source_box_bounds(oriented, box, budget), budget)
    return tau, oriented, cap


def pair_conflict(oriented_left, oriented_right, box, budget):
    """Certify amplitude exclusivity, NEVER a constraint on original bits."""
    combined = affine_add(oriented_left, oriented_right, budget)
    bounds = source_box_bounds(combined, box, budget)
    budget.charge(1)
    return bounds[1] <= 0
