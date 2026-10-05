"""Default-off six-row projection of a certified owned simplex interface.

All readouts refer to one retained System.  The required literal predicates
certify the finite local relaxation used by the projection theorem; they do
not establish that arbitrary caller-supplied t/u/delta really are products in
an original neural model.  That same-source, original-gate interpretation is
a caller proof obligation.  No original factor, bit, row or decoder is
replaced, and no solver is called here.

The entry preflight is a retained-System bound, not a certificate of peak
Python memory, whole-pipeline arithmetic work or native model qualification.
"""
from dataclasses import replace
from fractions import Fraction as F

from experiments.neural_hz_20260831.definition_first_20260928.d112_shared_endpoint_forward_20261002 import endpoint_forward as ef


MAX_ENTRIES = 64_000_000


def _form(form):
    if not isinstance(form, ef.Form) or type(form.terms) is not tuple:
        raise ValueError('canonical rational Form required')
    ef.rational(form.bias)
    previous = -1
    for item in form.terms:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError('canonical Form term required')
        index, coefficient = item
        if type(index) is not int or index <= previous:
            raise ValueError('strictly increasing nonnegative columns required')
        if ef.rational(coefficient) == 0:
            raise ValueError('zero coefficient is not canonical')
        previous = index


def _arguments(v, t, u, alpha, beta, delta, B):
    for vector in (v, t, u):
        if type(vector) is not tuple or len(vector) != 3:
            raise ValueError('exactly three immutable observations required')
        for form in vector:
            _form(form)
    for form in (alpha, beta, delta):
        _form(form)
    B = ef.rational(B)
    if B <= 0:
        raise ValueError('positive finite certified budget required')
    return B


def reference_rows(v, t, u, alpha, beta, delta, B):
    """Pure, canonical local reference rows, each interpreted as <= 0.

    Each coordinate and the total have explicit range [0,B], two complete
    single-bit box McCormick interfaces and the four D035 triangle rows.
    Coordinate nonnegativity and the total McCormick rows include both
    complete single-phase simplex interfaces and the source total budget.
    The shared overlap has all four McCormick rows.  Duplicate normalized
    rows are retained only once, in deterministic first-occurrence order.
    This helper constructs premises; it does not certify their neural origin.
    """
    B = _arguments(v, t, u, alpha, beta, delta, B)
    zero = ef.Form()
    rows = [-delta, delta-alpha, delta-beta, alpha+beta-delta-1]
    V, T, U = (sum(vector, zero) for vector in (v, t, u))
    for value, old, new in (*zip(v, t, u), (V, T, U)):
        rows.extend((-value, value-B))
        rows.extend(ef.mc_rows(old, alpha, value, F(0), F(1), F(0), B))
        rows.extend(ef.mc_rows(new, beta, value, F(0), F(1), F(0), B))
        rows.extend((
            old+new-value-B*delta,
            old-new-B*alpha+B*delta,
            new-old-B*beta+B*delta,
            value-old-new+B*alpha+B*beta-B*delta-B,
        ))
    return tuple(dict.fromkeys(rows))


# Descriptive alias for callers that distinguish these literal premises from
# any stronger, non-literal implication checker (which this module lacks).
literal_reference_rows = reference_rows


def _phase_column(system, phase):
    if phase.bias != F(1, 2) or len(phase.terms) != 1:
        raise ValueError('original signed-bit active view required')
    index, coefficient = phase.terms[0]
    if (coefficient not in (F(-1, 2), F(1, 2))
            or index not in system.binary
            or system.bounds[index] != (F(-1), F(1))):
        raise ValueError('original signed-bit active view required')
    return index


def append_owned_simplex(system, v, t, u, alpha, beta, delta, B,
                         *, frames, enabled=False, max_entries=MAX_ENTRIES):
    """Append the six owned-face rows without creating any coordinates.

    All reference_rows must already occur literally in system.le.  The own
    identity t[0]=v[0] must be literal or one signed orientation must occur in
    system.eq.  Semantic implication, numeric tolerances and metadata tokens
    are not substitutes for these explicit finite premises.
    """
    if enabled is False:
        return system, {'enabled': False}
    if enabled is not True or type(max_entries) is not int or not 0 < max_entries <= MAX_ENTRIES:
        raise ValueError('invalid opt-in or entry cap')
    if not isinstance(system, ef.System):
        raise ValueError('System required')
    if (type(frames) is not tuple or len(frames) != 7
            or any(type(frame) is not int or frame != system.frame for frame in frames)):
        raise ValueError('source/frame mismatch')
    old_columns = len(system.bounds)
    old_entries = ef.entries(system)
    entry_upper = old_entries + 6*(1+2*old_columns)
    if entry_upper > max_entries:
        raise ValueError('entry cap')
    for interval in system.bounds:
        if type(interval) is not tuple or len(interval) != 2:
            raise ValueError('finite rational column bounds required')
        lo, hi = map(ef.rational, interval)
        if lo > hi:
            raise ValueError('reversed column bounds')
    B = _arguments(v, t, u, alpha, beta, delta, B)
    for form in (*v, *t, *u, alpha, beta, delta):
        ef.check_form(system, form)
    alpha_column = _phase_column(system, alpha)
    beta_column = _phase_column(system, beta)
    if alpha_column == beta_column:
        raise ValueError('distinct original phase columns required')
    own = t[0]-v[0]
    if own != ef.Form() and own not in system.eq and -own not in system.eq:
        raise ValueError('literal owned observation identity required')
    required = reference_rows(v, t, u, alpha, beta, delta, B)
    if any(row not in system.le for row in required):
        raise ValueError('complete literal simplex/box reference required')

    A2, A3 = t[1]+u[1]-v[1], t[2]+u[2]-v[2]
    C2, C3 = t[1]-u[1], t[2]-u[2]
    rows = (
        u[0]+A2-B*delta,
        u[0]+A3-B*delta,
        v[0]-u[0]+C2-B*(alpha-delta),
        v[0]-u[0]+C3-B*(alpha-delta),
        -A2-A3-B*(1-alpha-beta+delta),
        -C2-C3-B*(beta-delta),
    )
    result = replace(system, le=system.le+rows)
    result_entries = ef.entries(result)
    if result_entries > max_entries or result_entries > entry_upper:
        raise ValueError('entry cap')
    assert result.bounds == system.bounds and result.binary == system.binary
    assert result.eq == system.eq and result.le[:len(system.le)] == system.le
    return result, dict(
        enabled=True, old_columns=old_columns, new_columns=0, new_eq=0,
        new_le=len(rows), nnz=sum(len(row.terms) for row in rows),
        entries=result_entries, entry_upper=entry_upper,
        reference_row_count=len(required), alpha_column=alpha_column,
        beta_column=beta_column, owned_observation=0,
        exact_rational_rows=True, native_binding_qualified=False,
        actual_model_qualified=False, complete_physical_qualification=False,
        formal_gain=0,
    )
