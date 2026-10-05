"""Default-off, conditional interval phase-capacity arithmetic.

The caller must establish all interval premises from the same original source.
This module is not a model importer, certificate authority, or verifier.  It
never creates phase identities or changes an original gate.  All work uses the
inherited ledger; scalar Fraction operations avoid interval multiplication's
four-product expansion.  One endpoint validation is a bounded scalar check,
as in the inherited kernel, rather than a count of individual bit operations.
"""

from fractions import Fraction

from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import shield_kernel_v1 as kernel


KernelError = kernel.KernelError
BudgetExceeded = kernel.BudgetExceeded
WorkBudget = kernel.WorkBudget
MAX_SLOTS = 65_536
ZERO = Fraction(0)


def _fraction(value, max_bits):
    """Endpoint inspection; callers prepay this bounded scalar check."""
    if type(value) is not Fraction:
        raise KernelError("exact Fraction endpoints required")
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > max_bits:
        raise KernelError("rational endpoint exceeds bit bound")
    return value


def _interval(value, budget):
    # Container/shape/access checks, two endpoint inspections, and ordering.
    budget.charge(8)
    if type(value) is not tuple or len(value) != 2:
        raise KernelError("an interval must be a tuple of two Fraction endpoints")
    lower = _fraction(value[0], budget.max_bits)
    upper = _fraction(value[1], budget.max_bits)
    if lower > upper:
        raise KernelError("reversed interval")
    return lower, upper


def _neg(value, budget):
    budget.charge(2)  # Negation and result endpoint inspection, before arithmetic.
    return _fraction(-value, budget.max_bits)


def _sub(left, right, budget):
    budget.charge(2)
    return _fraction(left - right, budget.max_bits)


def _mul(left, right, budget):
    budget.charge(2)
    return _fraction(left * right, budget.max_bits)


def _positive(value, budget):
    budget.charge(1)
    return value if value > ZERO else ZERO


def _minimum(left, right, budget):
    budget.charge(1)
    return left if left < right else right


def _active_budget(budget):
    if budget is None:
        budget = WorkBudget(enabled=True)
    if not isinstance(budget, WorkBudget):
        raise KernelError("the inherited WorkBudget is required")
    if (type(budget.enabled) is not bool
            or type(budget.limit) is not int or not 0 <= budget.limit <= 256_000_000
            or type(budget.max_bits) is not int or not 1 <= budget.max_bits <= 512
            or type(budget.used) is not int or not 0 <= budget.used <= budget.limit):
        raise KernelError("invalid inherited work ledger")
    budget.charge(0)  # Also rejects a disabled inherited ledger.
    return budget


def compile_capacities(weights, bias, bounds, pair_bounds, *, enabled=False, budget=None):
    """Return conditional capacity coefficients, or None without opt-in.

    ``pair_bounds[p]`` bounds f[2*p]-f[2*p+1], in that fixed orientation.
    Every canonical slot stays present, including zero/padding slots.  Padding
    has zero coefficients in the returned rows and needs no invented phase.
    Actual model/source validation and whole-model retained-entry accounting
    remain the caller's responsibility.
    """
    if type(enabled) is not bool:
        raise KernelError("enabled must be a bool")
    if not enabled:
        return None
    budget = _active_budget(budget)
    budget.charge(8)
    if type(weights) is not tuple or type(bounds) is not tuple or type(pair_bounds) is not tuple:
        raise KernelError("immutable tuples of intervals required")
    width = len(weights)
    if width > MAX_SLOTS:
        raise KernelError("canonical slot limit exceeded")
    if len(bounds) != width or len(pair_bounds) != width // 2:
        raise KernelError("weights, bounds, or canonical pair shape mismatch")
    bias_lower, bias_upper = _interval(bias, budget)

    # Prepay list containers and every visit/access/store in the source loop.
    budget.charge(4 + 9 * width)
    source_caps, positive, negative, matches = [], [], [], []
    for weight, bound in zip(weights, bounds):
        weight_lower, weight_upper = _interval(weight, budget)
        _, source_upper = _interval(bound, budget)
        source_cap = _positive(source_upper, budget)
        positive_magnitude = _positive(weight_upper, budget)
        negative_magnitude = _positive(_neg(weight_lower, budget), budget)
        source_caps.append(source_cap)
        positive.append(_mul(positive_magnitude, source_cap, budget))
        negative.append(_mul(negative_magnitude, source_cap, budget))

    budget.charge(2 * width + 2)
    unpaired_positive, unpaired_negative = tuple(positive), tuple(negative)

    # Pairing positions never depend on values.  Interval signs determine only
    # which of the two theorem-defined guaranteed amounts can be positive.
    budget.charge(12 * len(pair_bounds))
    for pair_index, difference in enumerate(pair_bounds):
        first, second = 2 * pair_index, 2 * pair_index + 1
        lower, upper = _interval(difference, budget)
        forward = _minimum(_positive(weights[first][0], budget),
                           _positive(_neg(weights[second][1], budget), budget), budget)
        reverse = _minimum(_positive(weights[second][0], budget),
                           _positive(_neg(weights[first][1], budget), budget), budget)
        budget.charge(2)
        if forward > ZERO:
            pos, neg, amount = first, second, forward
            difference_upper, reversed_upper = upper, _neg(lower, budget)
        elif reverse > ZERO:
            pos, neg, amount = second, first, reverse
            difference_upper, reversed_upper = _neg(lower, budget), upper
        else:
            continue
        up = _minimum(_positive(difference_upper, budget), source_caps[pos], budget)
        down = _minimum(_positive(reversed_upper, budget), source_caps[neg], budget)
        # Exact rearrangement of unmatched*scalar_cap + matched*pair_cap.
        saved_positive = _mul(amount, _sub(source_caps[pos], up, budget), budget)
        saved_negative = _mul(amount, _sub(source_caps[neg], down, budget), budget)
        positive[pos] = _sub(positive[pos], saved_positive, budget)
        negative[neg] = _sub(negative[neg], saved_negative, budget)
        budget.charge(2)
        if positive[pos] < ZERO or negative[neg] < ZERO:
            raise KernelError("negative certified capacity")
        matches.append((pos, neg, amount))

    bias_positive = _positive(bias_upper, budget)
    bias_negative = _positive(_neg(bias_lower, budget), budget)
    # Both generator visits and exact nonzero comparisons are prepaid.
    budget.charge(4 * width)
    nnz = 3 + sum(value != ZERO for value in positive) + sum(value != ZERO for value in negative)
    budget.charge(2 * width + len(matches) + 24)
    return {
        "positive": tuple(positive),
        "negative": tuple(negative),
        "unpaired_positive": unpaired_positive,
        "unpaired_negative": unpaired_negative,
        "bias_positive": bias_positive,
        "bias_negative": bias_negative,
        "matches": tuple(matches),
        "nnz": nnz,
    }
