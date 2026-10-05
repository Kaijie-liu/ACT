"""Default-off interval D042/D044 arithmetic; not a verifier or a new domain.

The caller certifies SAME-window original q_i=ReLU(g_i), every original bit,
source/frame identity, coefficient enclosures, and bounds on g_i and g_j-g_k.
No string or interval equality proves that binding.  All original HZ variables,
EQ/LE and decoder remain outside and unchanged.  False real_slots mean literal
zero padding, not an invented gate.  Equal non-point coefficient intervals do
NOT cancel: coefficients are interval-subtracted without midpoint substitution.

For each fixed pair (2j,2j+1), its even original bit is the anchor.  Conditional
seed contradictions close only that observation; a stable anchor may have an
empty phase that cannot be encoded by a nonempty interval.  Such failure is
neither invalid-SAT nor UNSAT.  All pairs remain in the returned population.

Residuals subtract exactly the stored contributions from their OWN independent
sum, lower from lower and upper from upper.  No archive/tighter bound is used
as a subtraction base.  The independent and unconditional d+p comparators may
be incomparable.  Tightening counts are only POTENTIAL endpoint improvements,
not nonredundancy, solver success, or formal gain.

The inherited D025/D015 ledger, 512-bit arithmetic and 256M ceiling are used
unchanged.  Preparation is O(width), each fixed pair O(1).  A conservative
temporary numeric-entry reserve is 256*width+8192, including current record
construction and Fraction numerator/denominator occurrences.  Returned roots
and any previous result still held by the caller are separately accounted.
This reserve is not a native/GPU/full-verifier resource qualification.
"""

from fractions import Fraction

from experiments.neural_hz_20260831.definition_first_20260928.d025_interval_capacity_20260930 import interval_capacity as arithmetic


KernelError = arithmetic.KernelError
BudgetExceeded = arithmetic.BudgetExceeded
WorkBudget = arithmetic.WorkBudget
MAX_SLOTS = arithmetic.MAX_SLOTS
ZERO = Fraction(0)
_ZERO_INTERVAL = (ZERO, ZERO)


def _add(a, b, budget):
    budget.charge(2)
    return arithmetic._fraction(a + b, budget.max_bits)


def _iv_add(a, b, budget):
    budget.charge(2)
    return _add(a[0], b[0], budget), _add(a[1], b[1], budget)


def _iv_sub(a, b, budget):
    budget.charge(2)
    return arithmetic._sub(a[0], b[1], budget), arithmetic._sub(a[1], b[0], budget)


def _product(a, b, budget):
    # Four products, six extrema comparisons, tuple/access bookkeeping.
    budget.charge(14)
    products = (arithmetic._mul(a[0], b[0], budget),
                arithmetic._mul(a[0], b[1], budget),
                arithmetic._mul(a[1], b[0], budget),
                arithmetic._mul(a[1], b[1], budget))
    return min(products), max(products)


def _positive_interval(interval, budget):
    budget.charge(4)
    return max(ZERO, interval[0]), max(ZERO, interval[1])


def _difference_relu(interval, budget):
    budget.charge(4)
    return min(ZERO, interval[0]), max(ZERO, interval[1])


def _residual(total, first, second, budget):
    """Exclude exactly the two already-summed endpoint contributions."""
    budget.charge(4)
    lo = arithmetic._sub(total[0], _add(first[0], second[0], budget), budget)
    hi = arithmetic._sub(total[1], _add(first[1], second[1], budget), budget)
    if lo > hi:
        raise KernelError("own-contribution residual reconstruction failed")
    return lo, hi


def _basis(residual, d_coefficient, p_coefficient, d_bounds, p_bounds, budget):
    return _iv_add(_iv_add(residual, _product(d_coefficient, d_bounds, budget), budget),
                   _product(p_coefficient, p_bounds, budget), budget)


def _seed(g, f, difference, budget):
    # Fixed visits, endpoint extrema/order tests and four interval containers.
    budget.charge(48)
    lg, ug = g
    lf, uf = f
    lower, upper = difference
    p0 = (max(ZERO, lf), max(ZERO, min(uf, arithmetic._neg(lower, budget))))
    p1 = (max(ZERO, max(lf, arithmetic._neg(upper, budget))), max(ZERO, uf))
    d0 = (arithmetic._neg(p0[1], budget), arithmetic._neg(p0[0], budget))
    d1 = (max(arithmetic._sub(max(ZERO, lg), p1[1], budget), min(ZERO, lower)),
          min(arithmetic._sub(ug, p1[0], budget), max(ZERO, upper)))
    if any(lo > hi for lo, hi in (p0, p1, d0, d1)):
        return None
    return (d0, d1), (p0, p1)


def _anchor_state(real, source, budget):
    budget.charge(8)
    if not real:
        return "P"
    lo, hi = source
    if lo > ZERO:
        return "A"
    if hi < ZERO:
        return "I"
    if lo == hi == ZERO:
        return "Z0"
    if lo == ZERO:
        return "Z+"
    if hi == ZERO:
        return "Z-"
    return "C"


def _transfer_bands(d_bands, p_bands, d_coefficient, p_coefficient,
                    w_d_coefficient, w_p_coefficient, d_residual, w_residual, budget):
    """Internal fixed-basis lemma on already-validated SAME-anchor bands.

    This does not reseed a new gate bit.  Callers of this internal lemma must
    retain and certify the old anchor of both input bands.  The public compiler
    obtains all its arguments from checked inputs and the uniform seed.
    """
    budget.charge(24)
    d_result, w_result, r_result, t_result = [], [], [], []
    for state in (0, 1):
        d_value = _basis(d_residual, d_coefficient, p_coefficient,
                         d_bands[state], p_bands[state], budget)
        w_value = _basis(w_residual, w_d_coefficient, w_p_coefficient,
                         d_bands[state], p_bands[state], budget)
        d_result.append(d_value)
        w_result.append(w_value)
        r_result.append(_difference_relu(d_value, budget))
        t_result.append(_positive_interval(w_value, budget))
    return tuple(d_result), tuple(w_result), tuple(r_result), tuple(t_result)


def _tightening(conditional, references, budget):
    # Four families, two states, two strict endpoint comparisons per state.
    budget.charge(64)
    result = []
    for states, reference in zip(conditional, references):
        count = 0
        for lo, hi in states:
            count += int(lo > reference[0]) + int(hi < reference[1])
        result.append(count)
    return tuple(result)


def compile_pair(left_weights, left_bias, right_weights, right_bias,
                 source_bounds, pair_bounds, real_slots, *, enabled=False, budget=None):
    """Compile every fixed even-source-anchor pair; default disabled is None.

    All inputs are immutable tuples; coefficients, biases and preactivation
    bounds are two-Fraction intervals. pair_bounds[j] bounds g[2j]-g[2j+1].
    The whole population, including padding/stable/failed anchors, is returned.
    Tightening tuples count strictly tighter endpoints in the order
    (difference, ReLU-difference, w, companion); each entry is in 0..4.
    """
    if type(enabled) is not bool:
        raise KernelError("enabled must be a bool")
    if not enabled:
        return None
    budget = arithmetic._active_budget(budget)
    budget.charge(16)
    if any(type(value) is not tuple for value in
           (left_weights, right_weights, source_bounds, pair_bounds, real_slots)):
        raise KernelError("immutable tuple inputs required")
    width = len(source_bounds)
    population = width // 2
    if (width > MAX_SLOTS or len(left_weights) != width or len(right_weights) != width
            or len(real_slots) != width or len(pair_bounds) != population):
        raise KernelError("source/weight/pair/flag shapes or slot limit do not match")
    # Prepay every whole-population visit and array/container store before work.
    budget.charge(24 + 20 * width + 20 * population)
    a_bias = arithmetic._interval(left_bias, budget)
    b_bias = arithmetic._interval(right_bias, budget)
    delta_bias = _iv_sub(a_bias, b_bias, budget)
    total_h, total_w, total_d = a_bias, b_bias, delta_bias
    bounds, activations, coefficients, right, d_contributions, w_contributions = [], [], [], [], [], []
    valid = 0
    for index in range(width):
        a = arithmetic._interval(left_weights[index], budget)
        b = arithmetic._interval(right_weights[index], budget)
        bound = arithmetic._interval(source_bounds[index], budget)
        real = real_slots[index]
        if type(real) is not bool:
            raise KernelError("real_slots must contain exact bools")
        if not real and bound != _ZERO_INTERVAL:
            raise KernelError("padding must be a literal zero preactivation")
        valid += int(real)
        activation = _positive_interval(bound, budget)
        coefficient = _iv_sub(a, b, budget)
        h_part = _product(a, activation, budget)
        w_part = _product(b, activation, budget)
        d_part = _product(coefficient, activation, budget)
        total_h = _iv_add(total_h, h_part, budget)
        total_w = _iv_add(total_w, w_part, budget)
        total_d = _iv_add(total_d, d_part, budget)
        bounds.append(bound)
        activations.append(activation)
        coefficients.append(coefficient)
        right.append(b)
        d_contributions.append(d_part)
        w_contributions.append(w_part)
    base_r, base_t = _difference_relu(total_d, budget), _positive_interval(total_w, budget)
    records = []
    statuses = {"OK": 0, "PADDING": 0, "SEED_CONTRADICTION": 0}
    for pair_index in range(population):
        # Record creation, field/index accesses, decisions and retained counts.
        budget.charge(64)
        first, second = 2 * pair_index, 2 * pair_index + 1
        difference_bound = arithmetic._interval(pair_bounds[pair_index], budget)
        lower, upper = difference_bound
        g, f = bounds[first], bounds[second]
        budget.charge(6)
        if lower > arithmetic._sub(g[1], f[0], budget) or upper < arithmetic._sub(g[0], f[1], budget):
            raise KernelError("source ranges and difference premise contradict")
        p_star = activations[second]
        d_star = (max(min(ZERO, lower), arithmetic._sub(activations[first][0], p_star[1], budget)),
                  min(max(ZERO, upper), arithmetic._sub(activations[first][1], p_star[0], budget)))
        budget.charge(9)
        if d_star[0] > d_star[1]:
            raise KernelError("global activation-difference premises contradict")
        d_rest = _residual(total_d, d_contributions[first], d_contributions[second], budget)
        w_rest = _residual(total_w, w_contributions[first], w_contributions[second], budget)
        c_first, b_first = coefficients[first], right[first]
        c_sum = _iv_add(c_first, coefficients[second], budget)
        b_sum = _iv_add(b_first, right[second], budget)
        unconditional_d = _basis(d_rest, c_first, c_sum, d_star, p_star, budget)
        unconditional_w = _basis(w_rest, b_first, b_sum, d_star, p_star, budget)
        unconditional_r = _difference_relu(unconditional_d, budget)
        unconditional_t = _positive_interval(unconditional_w, budget)
        anchor_state = _anchor_state(real_slots[first], g, budget)
        conditional_d = conditional_w = conditional_r = conditional_t = None
        tight_independent = tight_paired = (0, 0, 0, 0)
        if anchor_state == "P":
            status = "PADDING"
        else:
            seed = _seed(g, f, difference_bound, budget)
            if seed is None:
                status = "SEED_CONTRADICTION"
            else:
                status = "OK"
                conditional_d, conditional_w, conditional_r, conditional_t = _transfer_bands(
                    seed[0], seed[1], c_first, c_sum, b_first, b_sum, d_rest, w_rest, budget)
                conditional = (conditional_d, conditional_r, conditional_w, conditional_t)
                tight_independent = _tightening(conditional, (total_d, base_r, total_w, base_t), budget)
                tight_paired = _tightening(conditional,
                                            (unconditional_d, unconditional_r, unconditional_w, unconditional_t), budget)
        statuses[status] += 1
        records.append({"index": (first, second), "anchor_state": anchor_state, "status": status,
                        "unconditional_difference": unconditional_d,
                        "unconditional_w": unconditional_w,
                        "unconditional_relu_difference": unconditional_r,
                        "unconditional_companion": unconditional_t,
                        "conditional_difference": conditional_d,
                        "conditional_w": conditional_w,
                        "relu_difference": conditional_r,
                        "relu_companion": conditional_t,
                        "tightening_independent": tight_independent,
                        "tightening_paired": tight_paired})
    budget.charge(32 + population)
    return {"baseline_h": total_h, "baseline_w": total_w, "baseline_difference": total_d,
            "baseline_relu_difference": base_r, "baseline_companion": base_t,
            "pairs": tuple(records), "canonical_slots": width, "valid_slots": valid,
            "padding_slots": width - valid, "pair_count": population, "status_counts": statuses}
