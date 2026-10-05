"""Default-off common-source certificates for two existing ReLU gates.

Implements the frozen D052 paper rule, including nonpoint parameters.  Midpoints
select exact nonnegative certificate functions P and N; support is then taken
from the ORIGINAL interval forms, never from a midpoint substitute network.
The output contains the complete one-consumer McCormick projection, at most two
LE rows.  No overlap variable or new phase is created, and nothing in HZ is
deleted.  Original phase ownership and source identities are checked formally;
the caller still certifies the real model, affine parameter enclosure, source
box, native columns, active direction, and decoder.  This is not a verifier.

All public calls require enabled=True; disabled calls inspect no task input.
Every rational input and intermediate result is limited to 512 bits.  Input
support occurrences are preflighted at 65536 before merged containers are made.
Proof storage, row copies, sorting, native expansion and the original HZ remain
payable.  This module establishes no complete physical-memory or GPU gate.
"""

from dataclasses import dataclass
from fractions import Fraction

from experiments.neural_hz_20260831.definition_first_20260928.d049_mixed_source_envelopes_20260930 import mixed_source as ms


mp = ms.mp
KernelError = ms.KernelError
MAX_BITS, MAX_SUPPORT = ms.MAX_BITS, ms.MAX_SUPPORT
ZERO, ONE, TWO = Fraction(0), Fraction(1), Fraction(2)
_f, _add, _sub, _mul, _neg = ms._f, ms._add, ms._sub, ms._mul, ms._neg


@dataclass(frozen=True)
class _Affine:
    context: ms._Context
    bias: tuple
    terms: tuple


@dataclass(frozen=True)
class _Certificate:
    f: _Affine
    g: _Affine
    alpha: mp._Phase
    beta: mp._Phase
    P: ms._Form
    N: ms._Form
    kappa0: Fraction
    kappa1: Fraction
    upper: Fraction
    delta: Fraction
    rows: tuple

    @property
    def context(self):
        return self.f.context


def _interval(value):
    if type(value) is not tuple or len(value) != 2:
        raise KernelError('immutable two-endpoint interval required')
    lower, upper = _f(value[0]), _f(value[1])
    if lower > upper:
        raise KernelError('reversed parameter interval')
    return lower, upper


def _shape(value):
    if type(value) is not _Affine or type(value.terms) is not tuple:
        raise KernelError('an immutable interval affine form is required')
    if len(value.terms) > MAX_SUPPORT:
        raise KernelError('interval form exceeds support limit')
    return len(value.terms)


def _checked(value, context=None):
    _shape(value)
    actual = ms._context(value.context)
    if context is not None and actual is not context:
        raise KernelError('different original source contexts')
    _interval(value.bias)
    previous = -1
    for item in value.terms:
        if type(item) is not tuple or len(item) != 3:
            raise KernelError('source terms are (source, lower, upper)')
        source = ms._source(item[0], actual)
        lower, upper = _interval((item[1], item[2]))
        if source.ordinal <= previous or lower == upper == ZERO:
            raise KernelError('source terms must be canonical and nonzero')
        previous = source.ordinal
    return value


def affine(context, bias, terms, *, enabled=False):
    """Declare a caller-certified affine enclosure on an existing source frame.

    Duplicate source tokens are rejected rather than silently assigning new
    parameter dependence. Fixed sources remain in terms and in the context.
    """
    if not ms._on(enabled):
        return None
    context, bias = ms._context(context), _interval(bias)
    if type(terms) is not tuple or len(terms) > MAX_SUPPORT:
        raise KernelError('bounded immutable source term tuple required')
    seen, result = set(), []
    for item in terms:
        if type(item) is not tuple or len(item) != 3:
            raise KernelError('source terms are (source, lower, upper)')
        source = ms._source(item[0], context)
        lower, upper = _interval((item[1], item[2]))
        if source in seen:
            raise KernelError('duplicate original source')
        seen.add(source)
        if lower != ZERO or upper != ZERO:
            result.append((source, lower, upper))
    return _Affine(context, bias, tuple(sorted(result, key=lambda item: item[0].ordinal)))


def _middle(lower, upper):
    return _f(_add(lower, upper) / TWO)


def _upper_shift(value, subtract, add):
    """Box support of the original interval form minus/add exact affine forms."""
    if len(value.terms) + len(subtract.source_terms) + len(add.source_terms) > MAX_SUPPORT:
        raise KernelError('aggregate shifted support exceeds limit')
    original = {s: (lo, hi) for s, lo, hi in value.terms}
    minus, plus = dict(subtract.source_terms), dict(add.source_terms)
    total = _add(_sub(value.bias[1], subtract.bias), add.bias)
    for source in sorted(original.keys() | minus.keys() | plus.keys(), key=lambda s: s.ordinal):
        lower, upper = original.get(source, (ZERO, ZERO))
        correction = _sub(plus.get(source, ZERO), minus.get(source, ZERO))
        lower, upper = _add(lower, correction), _add(upper, correction)
        contribution = max(_mul(lower, source.lower), _mul(lower, source.upper),
                           _mul(upper, source.lower), _mul(upper, source.upper))
        total = _add(total, contribution)
    return total


def _row(readout, P, alpha, beta, a, b, c):
    values = ms._row_values(readout, P.source_terms, ONE, -ONE)
    phases = mp._make_form(readout.frame, 'phase', ZERO,
                          ((alpha, _neg(a)), (beta, _neg(b)))).terms
    if len(values) + len(phases) > MAX_SUPPORT:
        raise KernelError('aggregate returned LE support exceeds limit')
    return values, phases, _add(P.bias, c)


def generate(f, g, q, p, alpha, beta, *, enabled=False):
    """Generate D052 static matching rows for caller-certified original gates.

    Returns proofs and <=2 rows (value_terms, phase_terms, rhs). The same source
    value namespace is used by readouts and source terms. It is not a native
    matrix installer, gate selector, feasibility decision or new model.
    """
    if not ms._on(enabled):
        return None
    if _shape(f) + _shape(g) + 2 > MAX_SUPPORT:
        raise KernelError('aggregate pair support exceeds limit')
    f = _checked(f)
    context = f.context
    g = _checked(g, context)
    q, p = mp._value(q, context.frame), mp._value(p, context.frame)
    alpha, beta = mp._phase(alpha, context.frame), mp._phase(beta, context.frame)
    if (q is p or alpha.original_output is not q or beta.original_output is not p):
        raise KernelError('distinct original outputs and their own phases required')
    # Reject circular local declarations; topology remains a caller obligation.
    for value in (f, g):
        if any(s.original_value is q or s.original_value is p for s, _, _ in value.terms):
            raise KernelError('pair source cannot be either target gate output')
    coefficients_f = {s: (lo, hi) for s, lo, hi in f.terms}
    coefficients_g = {s: (lo, hi) for s, lo, hi in g.terms}
    sources = sorted(coefficients_f.keys() | coefficients_g.keys(), key=lambda s: s.ordinal)
    P_terms, N_terms = [], []
    P_bias = N_bias = ZERO
    for source in sources:
        width = _sub(source.upper, source.lower)
        if width == ZERO:
            continue
        cf = _middle(*coefficients_f.get(source, (ZERO, ZERO)))
        cg = _middle(*coefficients_g.get(source, (ZERO, ZERO)))
        reverse = cf < ZERO
        A = _mul(_neg(cf) if reverse else cf, width)
        B = _mul(cg, _neg(width) if reverse else width)
        amount = min(A, _neg(B) if B < ZERO else B)
        if amount == ZERO:
            continue
        coefficient = _f(amount / width)
        if reverse:
            offset = _mul(coefficient, source.upper)
            coefficient = _neg(coefficient)
        else:
            offset = _neg(_mul(coefficient, source.lower))
        if B > ZERO:
            P_terms.append((source, coefficient))
            P_bias = _add(P_bias, offset)
        else:
            N_terms.append((source, coefficient))
            N_bias = _add(N_bias, offset)
    P = ms._make_form(context, P_bias, tuple(P_terms))
    N = ms._make_form(context, N_bias, tuple(N_terms))
    zero = ms._make_form(context, ZERO)
    kappa0 = _upper_shift(f, P, zero)
    kappa1 = _upper_shift(f, N, zero)
    upper = _upper_shift(g, P, N)
    delta = _sub(kappa1, kappa0)
    read = mp._make_form(context.frame, 'value', ZERO, ((q, ONE), (p, ONE)))
    if delta > ZERO:
        rows = (_row(read, P, alpha, beta, kappa1, upper, ZERO),
                _row(read, P, alpha, beta, kappa0, _add(upper, delta), ZERO))
    elif delta < ZERO:
        rows = (_row(read, P, alpha, beta, kappa0, upper, ZERO),
                _row(read, P, alpha, beta, kappa1, _add(upper, delta), _neg(delta)))
    else:
        rows = (_row(read, P, alpha, beta, kappa0, upper, ZERO),)
    return _Certificate(f, g, alpha, beta, P, N, kappa0, kappa1, upper, delta, rows)
