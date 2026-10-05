"""Exact breakpoint sweep for the common-kappa E-plus family only.

The finite source box is explicit; no constrained support oracle or solver is
used. Infinity is an independent-cap limit, never a finite certificate.
This is retained-entry-limited mathematical code, not a full-work/GPU claim.
"""
from fractions import Fraction as F

from experiments.neural_hz_20260831.definition_first_20260928.d112_shared_endpoint_forward_20261002 import endpoint_forward as ef
from experiments.neural_hz_20260831.definition_first_20260928.d119_curvature_component_20261002 import curvature_transfer as ct


ZERO, ONE = F(0), F(1)
MAX_ENTRIES = 64_000_000


def _event(events, position, change):
    if change:
        events[position] = ct._add(events.get(position, ZERO), change)


def _rho_events(system, h, term, lower):
    """rho(lower), right derivative, and slope jumps for max(0,box(h+k*t))."""
    h_terms, t_terms = dict(h.terms), dict(term.terms)
    indices = sorted(h_terms.keys() | t_terms.keys())
    slope = term.bias
    coefficient_events = {}
    for index in indices:
        hvalue, tvalue = h_terms.get(index, ZERO), t_terms.get(index, ZERO)
        lo, hi = system.bounds[index]
        value = ct._add(hvalue, ct._mul(lower, tvalue))
        endpoint = hi if value > ZERO or (value == ZERO and tvalue > ZERO) else lo
        slope = ct._add(slope, ct._mul(tvalue, endpoint))
        if tvalue and lo != hi:
            position = ct._div(-hvalue, tvalue)
            if position > lower:
                _event(coefficient_events, position,
                       ct._mul(ct._add(hi, -lo), abs(tvalue)))
    value = ct._box(system, ct._linear(((ONE, h), (lower, term))))[1]
    initial_value = max(ZERO, value)
    initial_slope = slope if value > ZERO or (value == ZERO and slope > ZERO) else ZERO
    events = {}
    current = lower
    for stop in (*sorted(coefficient_events), None):
        if slope:
            root = ct._add(current, -ct._div(value, slope))
            if root > current and (stop is None or root < stop):
                _event(events, root, abs(slope))
        if stop is None:
            break
        value = ct._add(value, ct._mul(slope, ct._add(stop, -current)))
        next_slope = ct._add(slope, coefficient_events[stop])
        if value > ZERO:
            jump = ct._add(next_slope, -slope)
        elif value == ZERO:
            jump = ct._add(max(ZERO, next_slope), -min(ZERO, slope))
        else:
            jump = ZERO
        _event(events, stop, jump)
        current, slope = stop, next_slope
    return initial_value, initial_slope, events


def common_multiplier(system, h, terms, weights, *, enabled=False,
                      max_entries=MAX_ENTRIES):
    """Minimize E_plus in the common finite kappa>A family.

    ``kappa is None`` with boundary=infinity means a nonattained infimum;
    callers must NOT use that limit as a source certificate. The separate
    finite_candidate fields disclose an examined actual finite alternative.
    Negative weights are validated but do not optimize the positive budget.
    """
    if enabled is False:
        return {'enabled': False}
    if enabled is not True:
        raise ValueError('strict boolean opt-in required')
    original_entries, _ = ct._schema(system, max_entries)
    ct._form(system, h)
    if type(terms) is not tuple or type(weights) is not tuple or len(terms) != len(weights):
        raise ValueError('immutable common-family shape required')
    for term in terms:
        ct._form(system, term)
    weights = tuple(ct._fraction(weight) for weight in weights)
    support = len(h.terms) * len(terms) + sum(len(term.terms) for term in terms)
    entry_upper = original_entries + 32*(support + len(terms) + 1)
    if entry_upper > max_entries:
        raise ValueError('entry cap')
    positive = tuple((term, weight) for term, weight in zip(terms, weights) if weight > ZERO)
    lower = ZERO
    for _, weight in positive:
        lower = ct._add(lower, weight)
    initial_value = initial_slope = independent = ZERO
    events = {}
    for term, weight in positive:
        value, slope, changes = _rho_events(system, h, term, lower)
        initial_value = ct._add(initial_value, ct._mul(weight, value))
        initial_slope = ct._add(initial_slope, ct._mul(weight, slope))
        independent = ct._add(independent,
                              ct._mul(weight, max(ZERO, ct._box(system, term)[1])))
        for position, change in changes.items():
            _event(events, position, ct._mul(weight, change))
    knots = tuple(sorted(events))
    finite = []
    value, slope, current = initial_value, initial_slope, lower
    for position in knots:
        value = ct._add(value, ct._mul(slope, ct._add(position, -current)))
        finite.append((ct._div(value, ct._add(position, -lower)), position))
        slope = ct._add(slope, events[position])
        current = position
    # A legal point in the first interval also covers the otherwise easily
    # missed globally-constant family and finite lower-boundary plateaus.
    witness = (ct._div(ct._add(lower, knots[0]), F(2)) if knots
               else ct._add(lower, ONE))
    witness_numerator = ct._add(initial_value,
                               ct._mul(initial_slope, ct._add(witness, -lower)))
    finite.append((ct._div(witness_numerator, ct._add(witness, -lower)), witness))
    finite_value, finite_kappa = min(finite)
    lower_limit = initial_slope if initial_value == ZERO else None
    infimum = min(finite_value, independent,
                  lower_limit if lower_limit is not None else finite_value)
    attained = finite_value == infimum
    boundary = 'finite' if attained else ('infinity' if independent == infimum else 'lower')
    return dict(enabled=True, kappa=finite_kappa if attained else None,
                infimum=infimum, attained=attained, boundary=boundary,
                finite_value=finite_value if attained else None,
                finite_candidate_kappa=finite_kappa, finite_candidate_value=finite_value,
                lower_endpoint=lower, lower_limit=lower_limit,
                lower_limit_infinite=lower_limit is None,
                independent_cap=independent, breakpoints=knots,
                entry_upper=entry_upper, exact_rational=True,
                objective='common_kappa_E_plus_only',
                infinity_is_finite_certificate=False,
                complete_physical_qualification=False, whole_work_qualified=False,
                gpu_qualified=False)
