"""Paid, exact scalar consequences of common curvature for original weights.

This source diagnostic does not replace a HZ or certify its native binding.
All arithmetic is Fraction arithmetic; intervals enclose fixed original
parameters. No source sampling, optimization, or coefficient midpoint occurs.
"""
from fractions import Fraction as F

from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import shield_kernel_v1 as k
from experiments.neural_hz_20260831.definition_first_20260928.d119_curvature_component_20261002 import curvature_transfer as ct

ZERO, ONE = F(0), F(1)
TRANSIENT_ENTRIES_PER_SLOT, TRANSIENT_ENTRY_BASE = 128, 8192


def _enabled(enabled, budget):
    if enabled is False:
        return False
    if enabled is not True or type(budget) is not k.WorkBudget or budget.enabled is not True:
        raise ValueError('strict opt-in and original enabled budget required')
    return True


def scalar(value, budget):
    if type(value) is not F:
        raise ValueError('exact Fraction required')
    return k._checked((value, value), budget)[0]


def add(a, b, budget):
    # Internal operands have already passed the enclosing operation's checks.
    budget.charge(1)
    return scalar(a + b, budget)


def mul(a, b, budget):
    budget.charge(1)
    return scalar(a * b, budget)


def interval_add(a, b, budget):
    return k.add(a, b, budget)


def intersection(a, b, budget):
    a, b = k._checked(a, budget), k._checked(b, budget)
    budget.charge(2)
    return k._checked((max(a[0], b[0]), min(a[1], b[1])), budget)


def template(weights, *, enabled=False, budget=None):
    if not _enabled(enabled, budget):
        return {'enabled': False}
    if type(weights) is not tuple or len(weights) not in (4, 5):
        raise ValueError('canonical four/five original weights required')
    n = len(weights)
    # Includes tuple/zip visits, Fraction knot divisions, and ct.support's
    # two prefix loops, comparisons, rational checks and intermediate sums.
    # The reused support routine additionally enforces every 512-bit result.
    budget.charge(64 + 64 * n)
    for weight in weights:
        scalar(weight, budget)
    knots = tuple(F(i, n - 1) for i in range(n))
    lower, upper = ct.support(knots[1:-1], weights[1:-1])
    scalar(lower, budget); scalar(upper, budget)
    af, ah, li, ui = weights[0], weights[-1], ZERO, ZERO
    for t, w in zip(knots[1:-1], weights[1:-1]):
        omt = add(ONE, -t, budget)
        af = add(af, mul(w, omt, budget), budget)
        ah = add(ah, mul(w, t, budget), budget)
        cap = mul(t, omt, budget)
        li = add(li, mul(min(ZERO, w), cap, budget), budget)
        ui = add(ui, mul(max(ZERO, w), cap, budget), budget)
    if not li <= lower <= ZERO <= upper <= ui:
        raise ValueError('common kernel support containment failed')
    return dict(enabled=True, weights=weights, knots=knots, L=lower, U=upper,
                L_ind=li, U_ind=ui, A_f=af, A_h=ah)


def endpoint_range(a, b, bounds, budget):
    """Exact extrema of a*ReLU(z)+b*z at interval endpoints and zero."""
    a, b = scalar(a, budget), scalar(b, budget)
    lo, hi = k._checked(bounds, budget)
    budget.charge(8)
    positive_slope = add(a, b, budget)
    values = tuple(mul(positive_slope if z >= ZERO else b, z, budget)
                   for z in (lo, hi))
    if lo <= ZERO <= hi:
        values = (*values, ZERO)
    return k._checked((min(values), max(values)), budget)


def _weighted_interval(w, bounds, budget):
    w = scalar(w, budget)
    lo, hi = k._checked(bounds, budget)
    budget.charge(1)
    left, right = mul(w, lo, budget), mul(w, hi, budget)
    return k._checked((min(left, right), max(left, right)), budget)


def _bounds(plan, sources, residuals, lower, upper, budget):
    budget.charge(8 + 4 * len(residuals))
    # A-L*S and A-U*S share the actual endpoints, not independent p/f nodes.
    a_fu = add(plan['A_f'], mul(F(-2), lower, budget), budget)
    a_hu = add(plan['A_h'], mul(F(-2), lower, budget), budget)
    a_fl = add(plan['A_f'], mul(F(-2), upper, budget), budget)
    a_hl = add(plan['A_h'], mul(F(-2), upper, budget), budget)
    fu = endpoint_range(a_fu, lower, sources[0], budget)[1]
    hu = endpoint_range(a_hu, lower, sources[-1], budget)[1]
    fl = endpoint_range(a_fl, upper, sources[0], budget)[0]
    hl = endpoint_range(a_hl, upper, sources[-1], budget)[0]
    result_lo, result_hi = add(fl, hl, budget), add(fu, hu, budget)
    for w, residual in zip(plan['weights'][1:-1], residuals):
        wr = _weighted_interval(w, residual, budget)
        # eta is between min(0,r) and max(0,r); each signed term has its
        # own compensation. Never replace their sum by ReLU(sum residuals).
        result_lo = add(result_lo, min(ZERO, wr[0]), budget)
        result_hi = add(result_hi, max(ZERO, wr[1]), budget)
    return k._checked((result_lo, result_hi), budget)


def group_bounds(plan, source_bounds, residual_bounds, *, enabled=False, budget=None):
    if not _enabled(enabled, budget):
        return {'enabled': False}
    if type(plan) is not dict or type(plan.get('weights')) is not tuple:
        raise ValueError('original-weight template required')
    # Do not trust mutable externally supplied cached support bounds.
    checked = template(plan['weights'], enabled=True, budget=budget)
    budget.charge(32 + 8 * len(checked))
    if plan != checked:
        raise ValueError('template does not follow the original weights')
    return _compiled_group_bounds(checked, source_bounds, residual_bounds, budget)


def _compiled_group_bounds(plan, source_bounds, residual_bounds, budget):
    """Internal use only after this execution's template(raw_weights).

    The worker creates the plans itself from authenticated original slices;
    it accepts no supplied support certificate or mutable external cache.
    Public callers use group_bounds, which reconstructs the complete plan.
    """
    n = len(plan['weights'])
    if (type(source_bounds) is not tuple or len(source_bounds) != n
            or type(residual_bounds) is not tuple or len(residual_bounds) != n - 2):
        raise ValueError('complete original source/residual population required')
    budget.charge(4 * n)
    for bounds in (*source_bounds, *residual_bounds):
        k._checked(bounds, budget)
    ordinary = (ZERO, ZERO)
    for w, bounds in zip(plan['weights'], source_bounds):
        ordinary = k.add(ordinary, _weighted_interval(w, k.nonnegative_part(bounds, budget), budget), budget)
    common = _bounds(plan, source_bounds, residual_bounds, plan['L'], plan['U'], budget)
    independent = _bounds(plan, source_bounds, residual_bounds, plan['L_ind'], plan['U_ind'], budget)
    return dict(enabled=True, original=ordinary,
                common=intersection(common, ordinary, budget),
                independent=intersection(independent, ordinary, budget))


def post_affine(bounds, alpha, shift, budget):
    """Same original channel BN interval transform in every comparison arm."""
    return k.add(k.mul(alpha, k._checked(bounds, budget), budget), shift, budget)
