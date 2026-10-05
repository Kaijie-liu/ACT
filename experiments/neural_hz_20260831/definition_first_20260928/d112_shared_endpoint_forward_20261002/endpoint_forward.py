"""Opt-in exact-rational endpoint lowering, not a production admission path.

The caller must certify common source and actual Softmax/value semantics.
Metadata checks do not prove those obligations. Original columns, signed bits,
and predicates are retained; this module never calls a solver or a network.
"""
from dataclasses import dataclass, replace
from fractions import Fraction as F


def rational(value):
    if type(value) is int:
        value = F(value)
    if not isinstance(value, F):
        raise ValueError('exact rational required')
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 512:
        raise ValueError('rational bit cap')
    return value


@dataclass(frozen=True)
class Form:
    bias: F = F(0)
    terms: tuple = ()

    def __post_init__(self):
        entries = {}
        for index, value in self.terms:
            if type(index) is not int or index < 0:
                raise ValueError('invalid column')
            entries[index] = rational(entries.get(index, F(0)) + rational(value))
        object.__setattr__(self, 'bias', rational(self.bias))
        object.__setattr__(self, 'terms', tuple((i, v) for i, v in sorted(entries.items()) if v))

    def __add__(self, other):
        if not isinstance(other, Form):
            other = Form(rational(other))
        return Form(self.bias + other.bias, self.terms + other.terms)

    __radd__ = __add__

    def __neg__(self):
        return self * F(-1)

    def __sub__(self, other):
        return self + (-other if isinstance(other, Form) else -rational(other))

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, value):
        value = rational(value)
        return Form(self.bias * value, tuple((i, v * value) for i, v in self.terms))

    __rmul__ = __mul__


@dataclass(frozen=True)
class System:
    bounds: tuple
    binary: tuple
    eq: tuple
    le: tuple
    frame: int

    def __post_init__(self):
        if type(self.frame) is not int or self.frame < 0:
            raise ValueError('invalid frame')
        if any(type(x) is not tuple for x in (self.bounds, self.binary, self.eq, self.le)):
            raise ValueError('immutable tuple contract')
        for lo, hi in self.bounds:
            if rational(lo) > rational(hi):
                raise ValueError('reversed bounds')
        if len(set(self.binary)) != len(self.binary):
            raise ValueError('duplicate binary column')
        for i in self.binary:
            if type(i) is not int or not 0 <= i < len(self.bounds) or self.bounds[i] != (F(-1), F(1)):
                raise ValueError('signed binary column required')
        for row in (*self.eq, *self.le):
            check_form(self, row)


def check_form(system, form):
    if not isinstance(form, Form) or any(i >= len(system.bounds) for i, _ in form.terms):
        raise ValueError('unbound readout')


def box(system, form):
    check_form(system, form)
    lo = hi = form.bias
    for i, value in form.terms:
        a, b = system.bounds[i]
        lo = rational(lo + min(value*a, value*b))
        hi = rational(hi + max(value*a, value*b))
    return lo, hi


def variable(system, lo, hi):
    lo, hi = rational(lo), rational(hi)
    if lo > hi:
        raise ValueError('reversed variable bounds')
    index = len(system.bounds)
    return replace(system, bounds=system.bounds + ((lo, hi),)), Form(F(0), ((index, F(1)),))


def mc_rows(z, x, y, lx, ux, ly, uy):
    lx, ux, ly, uy = map(rational, (lx, ux, ly, uy))
    if lx > ux or ly > uy:
        raise ValueError('reversed product bounds')
    return (lx*y + ly*x - lx*ly - z,
            ux*y + uy*x - ux*uy - z,
            z - ux*y - ly*x + ux*ly,
            z - lx*y - uy*x + lx*uy)


def product(system, x, y, x_bounds=None, y_bounds=None):
    """Explicit tighter bounds are caller-certified and installed as rows."""
    check_form(system, x)
    check_form(system, y)
    lx, ux = box(system, x) if x_bounds is None else map(rational, x_bounds)
    ly, uy = box(system, y) if y_bounds is None else map(rational, y_bounds)
    if lx > ux or ly > uy:
        raise ValueError('reversed product bounds')
    system = replace(system, le=system.le + (lx-x, x-ux, ly-y, y-uy))
    if lx == ux:
        return system, lx*y
    if ly == uy:
        return system, ly*x
    corners = (lx*ly, lx*uy, ux*ly, ux*uy)
    system, z = variable(system, min(corners), max(corners))
    return replace(system, le=system.le + mc_rows(z, x, y, lx, ux, ly, uy)), z


def append_value_readout(system, probs, values, outputs):
    """Shared V centering is also provided to the baseline, not a gain claim."""
    n, channels = len(probs), len(outputs)
    if n < 2 or len(values) != n or not channels or any(len(row) != channels for row in values):
        raise ValueError('value readout shape')
    for form in (*probs, *outputs, *(v for row in values for v in row)):
        check_form(system, form)
    system = replace(system, eq=system.eq + (sum(probs, Form())-1,))
    for k in range(channels):
        total = values[0][k]
        for i in range(n):
            system, term = product(system, probs[i], values[i][k]-values[0][k])
            total = total + term
        system = replace(system, eq=system.eq + (outputs[k]-total,))
    return system


def entries(system):
    return (2*len(system.bounds)+len(system.binary)
            +sum(1+2*len(row.terms) for row in (*system.eq, *system.le)))


def append_endpoint(system, s, t, p, q, values, yp, yq, prob_bounds,
                    *, frames, enabled=False, max_entries=64_000_000):
    if enabled is False:
        return system, {'enabled': False}
    if enabled is not True or type(max_entries) is not int or not 0 < max_entries <= 64_000_000:
        raise ValueError('invalid opt-in or entry cap')
    if type(frames) is not tuple or len(frames) != 7 or any(type(f) is not int or f != system.frame for f in frames):
        raise ValueError('source/frame mismatch')
    n, channels = len(s), len(yp)
    if (n < 2 or any(len(rows) != n for rows in (t, p, q, values))
            or not channels or len(yq) != channels or any(len(row) != channels for row in values)
            or len(prob_bounds) != 2 or any(len(row) != n for row in prob_bounds)):
        raise ValueError('endpoint shape mismatch')
    for form in (*s, *t, *p, *q, *yp, *yq, *(v for row in values for v in row)):
        check_form(system, form)
    for bounds in prob_bounds:
        for lo, hi in bounds:
            if not 0 < rational(lo) <= rational(hi) <= 1:
                raise ValueError('positive probability bounds required')
    # Bound every possible new row by the entire final column population.
    # Point products use fewer rows/columns but never more. This deliberately
    # loose retained-entry preflight is NOT a transient Python-memory or
    # inherited whole-pipeline scalar-work certificate.
    extra_columns = 1 + n + n*channels
    extra_rows = 2 + channels + 13*n + 8*n*channels
    entry_upper = (entries(system) + 2*extra_columns
                   + extra_rows*(1+2*(len(system.bounds)+extra_columns)))
    if entry_upper > max_entries:
        raise ValueError('entry cap')
    original = system
    d = tuple(a-b for a, b in zip(s, t))
    dbox = tuple(box(system, form) for form in d)
    cl, cu = min(a for a, _ in dbox), max(b for _, b in dbox)
    c_index = len(system.bounds)
    system, c = variable(system, cl, cu)
    system = replace(system, eq=system.eq + (sum(p, Form())-1, sum(q, Form())-1))
    lambdas, delta_bounds = [], []
    for i in range(n):
        pl, pu = prob_bounds[0][i]
        ql, qu = prob_bounds[1][i]
        ll, lu = min(pl, ql), max(pu, qu)
        lambdas.append(len(system.bounds))
        system, lam = variable(system, ll, lu)
        h, z = d[i]-c, p[i]-q[i]
        hl, hu = dbox[i][0]-cu, dbox[i][1]-cl
        corners = (ll*hl, ll*hu, lu*hl, lu*hu)
        zl, zu = max(pl-qu, min(corners)), min(pu-ql, max(corners))
        if zl > zu:
            raise ValueError('inconsistent certified endpoint bounds')
        delta_bounds.append((zl, zu))
        system = replace(system, le=system.le + (
            pl-p[i], p[i]-pu, ql-q[i], q[i]-qu,
            2*lam-p[i]-q[i], hl-h, h-hu, zl-z, z-zu,
        ) + mc_rows(z, h, lam, hl, hu, ll, lu))
    for k in range(channels):
        total = Form()
        for i in range(n):
            system, term = product(system, p[i]-q[i], values[i][k]-values[0][k],
                                   x_bounds=delta_bounds[i])
            total = total + term
        system = replace(system, eq=system.eq + (yp[k]-yq[k]-total,))
    if entries(system) > max_entries:
        raise ValueError('entry cap')
    assert system.binary == original.binary
    assert system.bounds[:len(original.bounds)] == original.bounds
    assert system.eq[:len(original.eq)] == original.eq
    assert system.le[:len(original.le)] == original.le
    return system, dict(enabled=True, c_index=c_index, lambda_indices=tuple(lambdas),
        old_columns=len(original.bounds), new_columns=len(system.bounds)-len(original.bounds),
        new_eq=len(system.eq)-len(original.eq), new_le=len(system.le)-len(original.le),
        entries=entries(system), exact_rational_rows=True, native_binding_qualified=False,
        complete_physical_qualification=False)
