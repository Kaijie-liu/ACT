"""Opt-in shared ReLU curvature rows on the exact-rational D112 System.

This is a mathematical component, not native/model admission.  Every gate is
checked against its existing four literal inequalities; every original column,
signed bit and predicate is retained.  The fixed-knot support scan is not an
input/phase split.  Entry accounting below concerns the returned System, not a
complete transient-memory, whole-work or GPU qualification.
"""
from dataclasses import dataclass, replace
from fractions import Fraction as F

from experiments.neural_hz_20260831.definition_first_20260928.d112_shared_endpoint_forward_20261002 import endpoint_forward as ef


Form = ef.Form
System = ef.System
KernelError = ValueError
MAX_BITS = 512
MAX_ENTRIES = 64_000_000
ZERO, ONE = F(0), F(1)


@dataclass(frozen=True)
class Gate:
    g: Form
    q: Form
    bit: int
    lo: F
    hi: F


def _rat(value):
    return ef.rational(value)


def _fraction(value):
    if not isinstance(value, F):
        raise ValueError('Fraction required')
    return _rat(value)


def _add(a, b):
    return _rat(a + b)


def _mul(a, b):
    return _rat(a * b)


def _div(a, b):
    if not b:
        raise ValueError('zero denominator')
    return _rat(a / b)


def _linear(parts, bias=ZERO):
    """Merge on original column identity, checking each rational operation."""
    result_bias = _rat(bias)
    terms = {}
    for coefficient, form in parts:
        coefficient = _rat(coefficient)
        result_bias = _add(result_bias, _mul(coefficient, form.bias))
        for index, value in form.terms:
            merged = _add(terms.get(index, ZERO), _mul(coefficient, value))
            if merged:
                terms[index] = merged
            else:
                terms.pop(index, None)
    return Form(result_bias, tuple(sorted(terms.items())))


def _form(system, form):
    if type(form) is not Form or type(form.terms) is not tuple:
        raise ValueError('canonical Form required')
    _fraction(form.bias)
    previous = -1
    for term in form.terms:
        if type(term) is not tuple or len(term) != 2:
            raise ValueError('canonical term required')
        index, value = term
        if type(index) is not int or not previous < index < len(system.bounds):
            raise ValueError('unbound or unordered column')
        if not _fraction(value):
            raise ValueError('zero term is not canonical')
        previous = index


def _schema(system, max_entries):
    if type(max_entries) is not int or not 0 < max_entries <= MAX_ENTRIES:
        raise ValueError('entry cap')
    if type(system) is not System or type(system.frame) is not int or system.frame < 0:
        raise ValueError('System/frame required')
    if any(type(items) is not tuple for items in
           (system.bounds, system.binary, system.eq, system.le)):
        raise ValueError('immutable System required')
    count = 2 * len(system.bounds) + len(system.binary)
    if count > max_entries:
        raise ValueError('entry cap')
    for interval in system.bounds:
        if type(interval) is not tuple or len(interval) != 2:
            raise ValueError('bound pair required')
        if _rat(interval[0]) > _rat(interval[1]):
            raise ValueError('reversed bounds')
    binary = set()
    for index in system.binary:
        if (type(index) is not int or not 0 <= index < len(system.bounds)
                or index in binary or system.bounds[index] != (F(-1), ONE)):
            raise ValueError('distinct signed binary columns required')
        binary.add(index)
    for rows in (system.eq, system.le):
        for row in rows:
            _form(system, row)
            count += 1 + 2 * len(row.terms)
            if count > max_entries:
                raise ValueError('entry cap')
    return count, binary


def _box(system, form):
    # ef.box checks the sums; check products individually as well, before any
    # cancellation could conceal an over-wide intermediate rational.
    for index, coefficient in form.terms:
        lo, hi = system.bounds[index]
        _mul(coefficient, lo)
        _mul(coefficient, hi)
    return ef.box(system, form)


def _output(system, form, binary):
    _form(system, form)
    if (form.bias != ZERO or len(form.terms) != 1
            or form.terms[0][1] != ONE or form.terms[0][0] in binary):
        raise ValueError('existing unit nonbinary output column required')
    return form.terms[0][0]


def _gate(system, gate, binary, rows):
    if type(gate) is not Gate:
        raise ValueError('Gate required')
    _form(system, gate.g)
    output = _output(system, gate.q, binary)
    if type(gate.bit) is not int or gate.bit not in binary:
        raise ValueError('original signed gate bit required')
    lo, hi = _fraction(gate.lo), _fraction(gate.hi)
    actual_lo, actual_hi = _box(system, gate.g)
    if lo > actual_lo or hi < actual_hi or lo > hi:
        raise ValueError('gate bounds do not cover the original readout box')
    active = Form(F(1, 2), ((gate.bit, F(1, 2)),))
    lower, upper = min(ZERO, lo), max(ZERO, hi)
    guards = (
        _linear(((-ONE, gate.q),)),
        _linear(((ONE, gate.g), (-ONE, gate.q))),
        _linear(((ONE, gate.q), (-upper, active))),
        _linear(((ONE, gate.q), (-ONE, gate.g), (-lower, active)), lower),
    )
    if any(row not in rows for row in guards):
        raise ValueError('missing literal original gate inequality')
    return output


def _secant(system, form):
    lo, hi = _box(system, form)
    if hi <= ZERO:
        return Form()
    if lo >= ZERO:
        return form
    slope = _div(hi, _add(hi, -lo))
    return _linear(((slope, form),), -_mul(slope, lo))


def _compensated(system, base, coefficients, residuals):
    # The sound quantity is SUM ReLU(c_i*r_i), never ReLU(SUM c_i*r_i).
    # Each secant is constructed first; their affine forms are then merged on
    # the shared original coordinates, retaining every certified cancellation.
    parts = [(ONE, base)]
    for index, coefficient in coefficients:
        if coefficient and index in residuals:
            term = _linear(((coefficient, residuals[index]),))
            parts.append((-ONE, _secant(system, term)))
    return _linear(parts)


def _receipt(original, result, rows, **extra):
    return dict(enabled=True, old_columns=len(original.bounds), new_columns=0,
                new_eq=0, new_le=len(rows), rows=rows,
                nnz=sum(len(row.terms) for row in rows), entries=ef.entries(result),
                exact_rational_rows=True, original_bits_deleted=0,
                native_binding_qualified=False, actual_model_qualified=False,
                complete_physical_qualification=False, whole_work_qualified=False,
                **extra)


def append_curvature(system, gates, knots, *, frames, enabled=False,
                     max_entries=MAX_ENTRIES):
    if enabled is False:
        return system, {'enabled': False}
    if enabled is not True:
        raise ValueError('strict boolean opt-in required')
    old_entries, binary = _schema(system, max_entries)
    if (type(gates) is not tuple or len(gates) < 3 or type(knots) is not tuple
            or len(knots) != len(gates) or type(frames) is not tuple
            or len(frames) != len(gates)):
        raise ValueError('group shape')
    if any(type(frame) is not int or frame != system.frame for frame in frames):
        raise ValueError('source/frame mismatch')
    previous = F(-1)
    for knot in knots:
        knot = _fraction(knot)
        if not previous < knot <= ONE or knot < ZERO:
            raise ValueError('strict increasing knots required')
        previous = knot
    if knots[0] != ZERO or knots[-1] != ONE:
        raise ValueError('endpoint knots required')
    original_rows = set(system.le)
    outputs, phases = set(), set()
    for gate in gates:
        output = _gate(system, gate, binary, original_rows)
        if output in outputs or gate.bit in phases:
            raise ValueError('distinct original gates required')
        outputs.add(output)
        phases.add(gate.bit)
    m = len(gates) - 2
    first, last = gates[0], gates[-1]
    # Each residual has support at most nnz(g_i)+nnz(f)+nnz(h).  It occurs in
    # at most three stencils and one budget row.  Secants add no new columns.
    # Bias entries are counted separately for all m+1 final rows.  This bounds
    # retained System entries BEFORE residual and result-row construction.
    residual_support = sum(len(gate.g.terms) + len(first.g.terms) + len(last.g.terms)
                           for gate in gates[1:-1])
    new_nnz_upper = 3*m + 4 + len(first.g.terms) + len(last.g.terms) + 4*residual_support
    entry_upper = old_entries + (m+1) + 2*new_nnz_upper
    if entry_upper > max_entries:
        raise ValueError('entry cap')
    gaps = tuple(_add(knots[i], -knots[i-1]) for i in range(1, len(knots)))
    residuals = {}
    residual_boxes = []
    for i in range(1, m+1):
        residual = _linear(((ONE, gates[i].g),
                            (-_add(ONE, -knots[i]), first.g), (-knots[i], last.g)))
        residuals[i] = residual
        residual_boxes.append(_box(system, residual))
    rows = []
    for i in range(1, m+1):
        coefficients = ((i-1, -gaps[i]),
                        (i, _add(gaps[i-1], gaps[i])), (i+1, -gaps[i-1]))
        base = _linear(tuple((coefficient, gates[index].q)
                             for index, coefficient in coefficients))
        rows.append(_compensated(system, base, coefficients, residuals))
    left, right = _div(ONE, gaps[0]), _div(ONE, gaps[-1])
    coefficients = ((0, _add(left, F(-2))), (1, -left),
                    (m, -right), (m+1, _add(right, F(-2))))
    # For m=1 the two middle entries intentionally refer to the same original
    # gate.  Merge their coefficients BEFORE applying its residual secant.
    combined = {}
    for index, coefficient in coefficients:
        combined[index] = _add(combined.get(index, ZERO), coefficient)
    coefficients = tuple(sorted(combined.items()))
    base = _linear(tuple((coefficient, gates[index].q)
                         for index, coefficient in coefficients)
                   + ((ONE, first.g), (ONE, last.g)))
    rows.append(_compensated(system, base, coefficients, residuals))
    rows = tuple(rows)
    result = replace(system, le=system.le + rows)
    if ef.entries(result) > min(max_entries, entry_upper):
        raise ValueError('entry cap')
    return result, _receipt(system, result, rows,
                           residual_boxes=tuple(residual_boxes), budget_row=rows[-1],
                           entry_upper=entry_upper)


def support(knots_interior, weights):
    """Exact signed support of the declared UNIT tent family, not a source LP."""
    if (type(knots_interior) is not tuple or type(weights) is not tuple
            or len(knots_interior) != len(weights)):
        raise ValueError('support shape')
    previous = ZERO
    total_weight = total_moment = ZERO
    for knot, weight in zip(knots_interior, weights):
        knot, weight = _fraction(knot), _fraction(weight)
        if not previous < knot < ONE:
            raise ValueError('strict interior knot order required')
        previous = knot
        total_weight = _add(total_weight, weight)
        total_moment = _add(total_moment, _mul(weight, knot))
    prefix_weight = prefix_moment = ZERO
    lower = upper = ZERO
    for knot, weight in zip(knots_interior, weights):
        prefix_weight = _add(prefix_weight, weight)
        prefix_moment = _add(prefix_moment, _mul(weight, knot))
        tail = _add(total_weight, -prefix_weight)
        value = _add(prefix_moment, _mul(knot, _add(tail, -total_moment)))
        lower, upper = min(lower, value), max(upper, value)
    return lower, upper


def append_budget_child(system, gate, group_outputs, budget_row, *, frame,
                        enabled=False, max_entries=MAX_ENTRIES):
    if enabled is False:
        return system, {'enabled': False}
    if enabled is not True:
        raise ValueError('strict boolean opt-in required')
    old_entries, binary = _schema(system, max_entries)
    if type(frame) is not int or frame != system.frame:
        raise ValueError('source/frame mismatch')
    _form(system, budget_row)
    if budget_row not in system.le:
        raise ValueError('budget row must already be a literal original predicate')
    if type(group_outputs) is not tuple or not group_outputs:
        raise ValueError('group outputs required')
    columns = tuple(_output(system, form, binary) for form in group_outputs)
    if len(set(columns)) != len(columns):
        raise ValueError('distinct group outputs required')
    child = _gate(system, gate, binary, set(system.le))
    if child in columns:
        raise ValueError('next gate must have a distinct output')
    entry_upper = old_entries + 1 + 2*(1 + len(gate.g.terms) + len(budget_row.terms))
    if entry_upper > max_entries:
        raise ValueError('entry cap')
    g_terms, budget_terms = dict(gate.g.terms), dict(budget_row.terms)
    multiplier = None
    for column in columns:
        numerator = g_terms.get(column, ZERO)
        denominator = budget_terms.get(column, ZERO)
        if not denominator:
            if numerator:
                raise ValueError('group output cannot be cancelled')
            continue
        ratio = _div(numerator, denominator)
        if ratio <= ZERO or (multiplier is not None and ratio != multiplier):
            raise ValueError('one common positive budget multiplier required')
        multiplier = ratio
    if multiplier is None:
        raise ValueError('no group readout coefficient')
    upper = _linear(((ONE, gate.g), (-multiplier, budget_row)))
    if any(index in columns for index, _ in upper.terms):
        raise ValueError('group output cancellation failed')
    if _box(system, upper)[0] < ZERO:
        raise ValueError('nonnegative upper envelope required')
    row = _linear(((ONE, gate.q), (-ONE, upper)))
    result = replace(system, le=system.le + (row,))
    if ef.entries(result) > min(max_entries, entry_upper):
        raise ValueError('entry cap')
    return result, _receipt(system, result, (row,),
                           **{'lambda': multiplier, 'upper': upper, 'row': row,
                              'budget_row': budget_row, 'entry_upper': entry_upper})
