"""Default-off D029 arithmetic, not an abstract domain or a verifier.

The caller proves that every interval encloses the actual coefficient in one
shared source frame.  Midpoints define only a mathematical reference function;
the explicit error envelope certifies the actual function.  No reference phase
is attached to an original bit.  Observation projection likewise requires the
caller to bind its premises to the actual gate and source.

All variable-size loops, containers, and bounded-size Fraction operations are
precharged to the inherited D015 ledger.  As in that ledger, this counts scalar
work, not individual integer bit operations or wall time.  Import constants,
the fixed-size default-ledger bootstrap, parsing, complete retained-memory
accounting, and proof/frame binding are not a substitute for the caller's full
transaction accounting.  No solver, model loader, or phase search is used.
"""

from fractions import Fraction

from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import shield_kernel_v1 as kernel


KernelError = kernel.KernelError
BudgetExceeded = kernel.BudgetExceeded
KernelDisabled = kernel.KernelDisabled
WorkBudget = kernel.WorkBudget
MAX_SLOTS = 65_536
ZERO = Fraction(0)
ONE = Fraction(1)
NEG_ONE = Fraction(-1)
HALF = Fraction(1, 2)
_PREPARED_SEAL = object()


def _fraction(value, max_bits):
    """One precharged, bounded endpoint inspection."""
    if type(value) is not Fraction:
        raise KernelError("exact Fraction values required")
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > max_bits:
        raise KernelError("rational value exceeds bit bound")
    return value


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
    budget.charge(0)
    return budget


def _enabled(enabled):
    if type(enabled) is not bool:
        raise KernelError("enabled must be a bool")
    return enabled


def _scalar(value, budget):
    budget.charge(1)
    return _fraction(value, budget.max_bits)


def _interval(value, budget):
    budget.charge(8)
    if type(value) is not tuple or len(value) != 2:
        raise KernelError("intervals must be pairs of exact Fraction endpoints")
    lower = _fraction(value[0], budget.max_bits)
    upper = _fraction(value[1], budget.max_bits)
    if lower > upper:
        raise KernelError("reversed interval")
    return lower, upper


def _add(left, right, budget):
    budget.charge(2)
    return _fraction(left + right, budget.max_bits)


def _sub(left, right, budget):
    budget.charge(2)
    return _fraction(left - right, budget.max_bits)


def _mul(left, right, budget):
    budget.charge(2)
    return _fraction(left * right, budget.max_bits)


def _div(left, right, budget):
    budget.charge(3)
    if right == ZERO:
        raise KernelError("division by zero")
    return _fraction(left / right, budget.max_bits)


def _neg(value, budget):
    budget.charge(2)
    return _fraction(-value, budget.max_bits)


def _abs(value, budget):
    budget.charge(2)
    return _fraction(abs(value), budget.max_bits)


def _max(left, right, budget):
    budget.charge(1)
    return left if left >= right else right


def _midrad(value, budget):
    lower, upper = _interval(value, budget)
    middle = _mul(_add(lower, upper, budget), HALF, budget)
    radius = _mul(_sub(upper, lower, budget), HALF, budget)
    budget.charge(1)
    return middle, radius


def _source_id(value, budget):
    budget.charge(2)
    if type(value) is not int or value < 0:
        raise KernelError("source IDs must be nonnegative exact integers")


def _prepare_box(box, budget):
    budget.charge(2)
    if type(box) is not dict:
        raise KernelError("source box must be an integer-keyed dict")
    # Includes dict/list construction, item visits, record tuples and stores.
    budget.charge(3 + 8 * len(box))
    lookup, records = {}, []
    for source_id, bounds in box.items():
        _source_id(source_id, budget)
        lower, upper = _interval(bounds, budget)
        middle = _mul(_add(lower, upper, budget), HALF, budget)
        radius = _mul(_sub(upper, lower, budget), HALF, budget)
        magnitude = _max(_abs(lower, budget), _abs(upper, budget), budget)
        record = (middle, radius, magnitude)
        lookup[source_id] = record
        records.append((source_id, record))
    budget.charge(1 + len(records))
    return lookup, tuple(records)


def _nominal_form(form, box, budget):
    """Return immutable (bias, sparse coefficients, center, support, error)."""
    budget.charge(4)
    if type(form) is not tuple or len(form) != 2 or type(form[1]) is not dict:
        raise KernelError("form must be (constant interval, coefficient dict)")
    constant, error = _midrad(form[0], budget)
    center, spread = constant, ZERO
    budget.charge(2 + 8 * len(form[1]))
    coefficients = []
    for source_id, interval in form[1].items():
        _source_id(source_id, budget)
        if source_id not in box:
            raise KernelError("source coefficient is missing its shared box coordinate")
        middle, radius = _midrad(interval, budget)
        x_center, x_radius, x_magnitude = box[source_id]
        center = _add(center, _mul(middle, x_center, budget), budget)
        spread = _add(spread, _mul(_abs(middle, budget), x_radius, budget), budget)
        error = _add(error, _mul(radius, x_magnitude, budget), budget)
        if middle != ZERO:
            coefficients.append((source_id, middle))
    support = _add(center, spread, budget)
    budget.charge(6 + len(coefficients))
    return constant, tuple(coefficients), center, support, error


class _PreparedSource:
    """Factory-only immutable cache; not a security or certificate authority.

    The roots are deeply immutable builtin tuples, integers and Fractions.
    ``evidence_roots`` returns the actual retained objects without a copy.
    Callers additionally account for this capsule's instance header.  Work
    paid during preparation remains part of the caller's complete transaction.
    """

    __slots__ = ("_roots", "_seal")

    def __init__(self, roots, seal):
        if seal is not _PREPARED_SEAL:
            raise KernelError("prepared sources must come from prepare_source")
        object.__setattr__(self, "_roots", roots)
        object.__setattr__(self, "_seal", seal)

    def __setattr__(self, name, value):
        raise AttributeError("prepared source is immutable")

    def evidence_roots(self):
        return self._roots


def prepare_source(forms, box, *, enabled=False, budget=None):
    """Copy/validate a shared source once; disabled calls inspect no inputs.

    The returned cache is independent of subsequent mutations of input dicts.
    It does not certify the caller's raw-model/frame provenance.  The source
    forms are affine in this same box, not arbitrary later-layer activations.
    """
    if not _enabled(enabled):
        return None
    budget = _active_budget(budget)
    budget.charge(4)
    if type(forms) is not tuple or len(forms) > MAX_SLOTS:
        raise KernelError("forms must be a tuple of at most 65536 slots")
    box_lookup, box_records = _prepare_box(box, budget)
    budget.charge(2 + 4 * len(forms))
    prepared_forms = []
    for form in forms:
        nominal = _nominal_form(form, box_lookup, budget)
        amplitude = _max(ZERO, _add(nominal[3], nominal[4], budget), budget)
        prepared_forms.append((nominal, amplitude))
    budget.charge(8 + len(prepared_forms))
    roots = (tuple(prepared_forms), box_records, budget.max_bits)
    return _PreparedSource(roots, _PREPARED_SEAL)


def _unpack_prepared(prepared, budget):
    budget.charge(5)
    if type(prepared) is not _PreparedSource or prepared._seal is not _PREPARED_SEAL:
        raise KernelError("an unmodified prepare_source cache is required")
    forms, records, source_max_bits = prepared.evidence_roots()
    if budget.max_bits < source_max_bits:
        raise KernelError("current bit budget is smaller than source preparation budget")
    # Rebuild only the coordinate index; never repeat source Fraction work.
    budget.charge(2 + 4 * len(records))
    box = {}
    for source_id, record in records:
        box[source_id] = record
    budget.charge(1)
    return forms, box


def _merge_scaled(coefficients, source, scale, budget):
    """Merge on shared source ID before any absolute value is taken."""
    budget.charge(2 + 8 * len(source))
    for source_id, coefficient in source:
        increment = _mul(scale, coefficient, budget)
        value = _add(coefficients.get(source_id, ZERO), increment, budget)
        if value == ZERO:
            coefficients.pop(source_id, None)
        else:
            coefficients[source_id] = value


def _base_support(constant, coefficients, divisor, box, budget):
    """Normalize one base in place and compute its support once."""
    constant = _div(constant, divisor, budget)
    center, spread = constant, ZERO
    budget.charge(2 + 6 * len(coefficients))
    for source_id, coefficient in coefficients.items():
        value = _div(coefficient, divisor, budget)
        coefficients[source_id] = value
        x_center, x_radius, _ = box[source_id]
        center = _add(center, _mul(value, x_center, budget), budget)
        spread = _add(spread, _mul(_abs(value, budget), x_radius, budget), budget)
    return _add(center, spread, budget)


def _patch(form, weight, budget):
    budget.charge(2)
    coefficients = {}
    _merge_scaled(coefficients, form[1], weight, budget)
    center = _mul(weight, form[2], budget)
    budget.charge(1)
    return center, coefficients


def _increment_support(base_support, base, patch_center, patch, box, scale, budget):
    """Support of scale*base+patch, visiting only patch coordinates."""
    value = _add(_mul(scale, base_support, budget), patch_center, budget)
    budget.charge(2 + 7 * len(patch))
    for source_id, increment in patch.items():
        old = _mul(scale, base.get(source_id, ZERO), budget)
        new = _add(old, increment, budget)
        difference = _sub(_abs(new, budget), _abs(old, budget), budget)
        value = _add(value, _mul(difference, box[source_id][1], budget), budget)
    return value


def _upper_reference(forms, weights, affine, box, sign, budget):
    """Compile the same uniform rule for the reference F or -F."""
    constant = _mul(sign, affine[0], budget)
    budget.charge(3)
    base, positives = {}, []
    _merge_scaled(base, affine[1], sign, budget)
    budget.charge(2 + 9 * len(weights))
    for index, weight in enumerate(weights):
        weight = _mul(sign, weight, budget)
        form = forms[index][0]
        # The reference term is algebraically zero (including padding), not
        # merely stable.  Its actual interval error was already paid above.
        # No original slot, phase, or actual source fact is removed.
        if form[0] == ZERO and not form[1]:
            continue
        if weight < ZERO:
            scale = _mul(weight, HALF, budget)
            constant = _add(constant, _mul(scale, form[0], budget), budget)
            _merge_scaled(base, form[1], scale, budget)
        elif weight > ZERO:
            positives.append((index, weight))

    budget.charge(8)
    pairs, singleton = len(positives) // 2, len(positives) % 2
    groups = pairs + singleton
    if groups == 0:
        bound = _base_support(constant, base, ONE, box, budget)
        budget.charge(1)
        return bound, bound, 0, 0
    budget.charge(2)
    divisor = _fraction(Fraction(groups), budget.max_bits)
    base_bound = _base_support(constant, base, divisor, box, budget)
    half_base_bound = _mul(HALF, base_bound, budget)
    bound, separate = ZERO, ZERO

    # Only one/two sparse patches are materialized per group, never a base copy.
    budget.charge(3 + 15 * groups)
    for start in range(0, len(positives), 2):
        first_index, first_weight = positives[start]
        first_center, first = _patch(forms[first_index][0], first_weight, budget)
        first_bound = _increment_support(base_bound, base, first_center, first,
                                         box, ONE, budget)
        group_bound = _max(base_bound, first_bound, budget)
        if start + 1 == len(positives):
            # A singleton has exactly the same support in both comparisons.
            bound = _add(bound, group_bound, budget)
            separate = _add(separate, group_bound, budget)
            continue

        second_index, second_weight = positives[start + 1]
        second_center, second = _patch(forms[second_index][0], second_weight, budget)
        second_bound = _increment_support(base_bound, base, second_center, second,
                                          box, ONE, budget)
        group_bound = _max(group_bound, second_bound, budget)
        first_separate = _max(half_base_bound,
                              _increment_support(base_bound, base, first_center,
                                                 first, box, HALF, budget), budget)
        second_separate = _max(half_base_bound,
                               _increment_support(base_bound, base, second_center,
                                                  second, box, HALF, budget), budget)

        # first is a disposable local patch.  Merge second into it in place.
        budget.charge(2 + 8 * len(second))
        for source_id, coefficient in second.items():
            value = _add(first.get(source_id, ZERO), coefficient, budget)
            if value == ZERO:
                first.pop(source_id, None)
            else:
                first[source_id] = value
        both_center = _add(first_center, second_center, budget)
        both_bound = _increment_support(base_bound, base, both_center, first,
                                        box, ONE, budget)
        group_bound = _max(group_bound, both_bound, budget)
        bound = _add(bound, group_bound, budget)
        separate = _add(separate, _add(first_separate, second_separate, budget), budget)
    budget.charge(1)
    return bound, separate, pairs, singleton


def compile_prepared(prepared, weights, affine, *, enabled=False, budget=None):
    """Bound a readout using a previously paid, immutable source cache.

    Weights and affine coefficients are validated and paid for on every call.
    The caller must charge preparation and each readout to its full work/memory
    transaction, and retain the actual source-frame/provenance binding.
    """
    if not _enabled(enabled):
        return None
    budget = _active_budget(budget)
    forms, box = _unpack_prepared(prepared, budget)
    budget.charge(5)
    if type(weights) is not tuple or len(weights) != len(forms):
        raise KernelError("weight tuple must match the original source slots")
    nominal_affine = _nominal_form(affine, box, budget)
    error = nominal_affine[4]
    budget.charge(2 + 6 * len(weights))
    nominal_weights = []
    for weight, (form, amplitude) in zip(weights, forms):
        middle, radius = _midrad(weight, budget)
        error = _add(error,
                     _add(_mul(radius, amplitude, budget),
                          _mul(_abs(middle, budget), form[4], budget), budget), budget)
        nominal_weights.append(middle)

    upper, separate_upper, upper_pairs, upper_singletons = _upper_reference(
        forms, nominal_weights, nominal_affine, box, ONE, budget)
    negative, separate_negative, lower_pairs, lower_singletons = _upper_reference(
        forms, nominal_weights, nominal_affine, box, NEG_ONE, budget)
    upper = _add(upper, error, budget)
    separate_upper = _add(separate_upper, error, budget)
    lower = _sub(_neg(negative, budget), error, budget)
    separate_lower = _sub(_neg(separate_negative, budget), error, budget)
    budget.charge(4)
    if lower > upper or separate_lower > lower or upper > separate_upper:
        raise KernelError("support or comparison invariant failed")
    budget.charge(20)
    return {
        "lower": lower,
        "upper": upper,
        "separate_lower": separate_lower,
        "separate_upper": separate_upper,
        "error": error,
        "upper_pairs": upper_pairs,
        "lower_pairs": lower_pairs,
        "upper_singletons": upper_singletons,
        "lower_singletons": lower_singletons,
    }


def compile_support(forms, weights, affine, box, *, enabled=False, budget=None):
    """Uniform signed reference support plus an explicit actual-model error."""
    if not _enabled(enabled):
        return None
    budget = _active_budget(budget)
    prepared = prepare_source(forms, box, enabled=True, budget=budget)
    return compile_prepared(prepared, weights, affine, enabled=True, budget=budget)


def project_observation(a, L, U, A, B, *, enabled=False, budget=None):
    """Project D029's six rows; return four (q, v, alpha, rhs) <= rows.

    This algebra assumes existing 0<=alpha<=1, L<=v<=U and the proven actual
    bounds A<=a*g+v<=B.  It retains q/g/the original bit.  It neither establishes
    these premises nor authorizes elimination of a lift with other consumers.
    """
    if not _enabled(enabled):
        return None
    budget = _active_budget(budget)
    a = _scalar(a, budget)
    L, U = _scalar(L, budget), _scalar(U, budget)
    A, B = _scalar(A, budget), _scalar(B, budget)
    budget.charge(2)
    if L > U or A > B:
        raise KernelError("observation intervals are reversed")
    negative_a = _neg(a, budget)
    first_alpha = _sub(A, U, budget)
    second_alpha = _sub(L, B, budget)
    third_alpha = _sub(A, L, budget)
    fourth_alpha = _sub(U, B, budget)
    negative_L = _neg(L, budget)
    budget.charge(21)
    return ((negative_a, ZERO, first_alpha, ZERO),
            (a, ZERO, second_alpha, ZERO),
            (negative_a, NEG_ONE, third_alpha, negative_L),
            (a, ONE, fourth_alpha, U))
