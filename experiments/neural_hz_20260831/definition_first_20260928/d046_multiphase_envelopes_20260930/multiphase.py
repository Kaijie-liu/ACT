"""Default-off, exact-rational multi-original-phase envelope arithmetic.

This is a finite mathematical component, not a network verifier, certificate
authority, native HZ binding, or completed new abstract domain.  The caller
certifies the original common frame, real readouts/gates, every premise, and
the active direction and structural ordinal of every original phase.  Tokens
prevent accidental mixing; they do not establish those network facts.  The
original continuous factors, ALL original bits and their legal zero choices,
predicates and decoder remain in the caller's nonconvex HZ unchanged.

Value readouts and phase-affine bounds are distinct immutable form types.
Affine transfer first combines identical whole readouts, identified by their
actual tokens and coefficients, not by labels or equal bounds.  For repeated
readouts it uses the first certified envelope deterministically; no claim of
optimal reduction is made.  Positive-part upper envelopes use one fixed
original-ordinal prefix order.  Lower envelopes use singleton marginals and
are valid for binary phases, NOT pointwise on the relaxed continuous cube.

Only exact Fraction data are accepted, with 512-bit inputs and arithmetic
results.  A public combination preflights at most 65,536 sparse input-term
occurrences in aggregate before building its output containers; individual
forms, observation triples, input collections, and frame declarations are
also bounded.  Returned two-row storage can have twice the input readout
support.  Sorting, identity maps, all arithmetic, retained premises, row
copies and caller certificates are costs, not free metadata.  No whole-work,
GPU, model-census or full physical-memory qualification is claimed here.

All public operations are explicit opt-in.  Disabled calls inspect no task
input.  Rejection never means UNSAT/SAFE and never removes an original bit.
Admission additionally requires lower<=upper on the entire independent bit
cube.  This conservative finite subclass can reject envelopes valid only
under the original predicates; it is not a total constructor for THEORY.
"""

from dataclasses import dataclass, field
from fractions import Fraction


MAX_BITS = 512
MAX_SUPPORT = 65_536
ZERO = Fraction(0)
ONE = Fraction(1)


class KernelError(ValueError):
    """Malformed, inconsistent or unsupported mathematical premises."""


def _on(enabled):
    if type(enabled) is not bool:
        raise KernelError("enabled must be a bool")
    return enabled


def _f(value):
    if type(value) is not Fraction:
        raise KernelError("exact Fraction values required")
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > MAX_BITS:
        raise KernelError("rational bit limit exceeded")
    return value


def _add(left, right):
    return _f(left + right)


def _neg(value):
    return _f(-value)


def _sub(left, right):
    return _add(left, _neg(right))


def _mul(left, right):
    return _f(left * right)


def _label(value):
    if type(value) is not str or not 0 < len(value) <= 256:
        raise KernelError("a bounded nonempty diagnostic label is required")
    return value


def _ordinal(value):
    if type(value) is not int or value < 0 or value.bit_length() > MAX_BITS:
        raise KernelError("a bounded nonnegative original ordinal is required")
    return value


@dataclass(frozen=True, eq=False)
class _Frame:
    label: str
    _values: dict = field(default_factory=dict, repr=False)
    _phases: dict = field(default_factory=dict, repr=False)
    _source_phases: dict = field(default_factory=dict, repr=False)


@dataclass(frozen=True, eq=False)
class _Value:
    frame: _Frame
    label: str
    _serial: int


@dataclass(frozen=True, eq=False)
class _Phase:
    original_output: _Value
    ordinal: int
    label: str

    @property
    def frame(self):
        return self.original_output.frame


@dataclass(frozen=True)
class _Form:
    frame: _Frame
    kind: str
    bias: Fraction
    terms: tuple


@dataclass(frozen=True)
class _Observation:
    readout: _Form
    lower: _Form
    upper: _Form
    rule: str

    @property
    def frame(self):
        return self.readout.frame


def _frame(value):
    if type(value) is not _Frame:
        raise KernelError("an explicit shared-frame token is required")
    _label(value.label)
    if any(type(items) is not dict or len(items) > MAX_SUPPORT
           for items in (value._values, value._phases, value._source_phases)):
        raise KernelError("invalid or oversized frame registry")
    return value


def _value(value, frame=None):
    if type(value) is not _Value:
        raise KernelError("an existing original-value token is required")
    _frame(value.frame)
    _label(value.label)
    if type(value._serial) is not int or value.frame._values.get(value._serial) is not value:
        raise KernelError("undeclared value identity")
    if frame is not None and value.frame is not frame:
        raise KernelError("different original frames")
    return value


def _phase(value, frame=None):
    if type(value) is not _Phase:
        raise KernelError("an original-phase token is required")
    _value(value.original_output, frame)
    _ordinal(value.ordinal)
    _label(value.label)
    if (value.frame._phases.get(value.ordinal) is not value or
            value.frame._source_phases.get(value.original_output) is not value):
        raise KernelError("phase ordinal or original output binding mismatch")
    return value


def _kind(value):
    if type(value) is not str or value not in ("value", "phase"):
        raise KernelError("form kind must be value or phase")
    return value


def _token(token, frame, kind):
    return _value(token, frame) if kind == "value" else _phase(token, frame)


def _order(token, kind):
    return token._serial if kind == "value" else token.ordinal


def make_frame(label, *, enabled=False):
    if not _on(enabled):
        return None
    return _Frame(_label(label))


def make_value(frame, label, *, enabled=False):
    """Declare an already-existing formal readout, not a new network node."""
    if not _on(enabled):
        return None
    frame, label = _frame(frame), _label(label)
    if len(frame._values) >= MAX_SUPPORT:
        raise KernelError("frame value declaration limit exceeded")
    serial = len(frame._values)
    result = _Value(frame, label, serial)
    frame._values[serial] = result
    return result


def make_phase(original_output, ordinal, label, *, enabled=False):
    """Declare its existing original bit and unique structural ordinal."""
    if not _on(enabled):
        return None
    source = _value(original_output)
    ordinal, label = _ordinal(ordinal), _label(label)
    frame = source.frame
    if ordinal in frame._phases or source in frame._source_phases:
        raise KernelError("original ordinal or output phase already declared")
    if len(frame._phases) >= MAX_SUPPORT:
        raise KernelError("frame phase declaration limit exceeded")
    result = _Phase(source, ordinal, label)
    frame._phases[ordinal] = result
    frame._source_phases[source] = result
    return result


def _pre_forms(forms):
    count = 0
    for value in forms:
        if type(value) is not _Form or type(value.terms) is not tuple:
            raise KernelError("immutable typed forms are required")
        count += len(value.terms)
        if count > MAX_SUPPORT:
            raise KernelError("aggregate sparse input limit exceeded")
    return count


def _checked_form(value, frame=None, kind=None):
    if type(value) is not _Form:
        raise KernelError("an immutable affine form is required")
    _frame(value.frame)
    _kind(value.kind)
    _f(value.bias)
    if frame is not None and value.frame is not frame:
        raise KernelError("different original frames")
    if kind is not None and value.kind != kind:
        raise KernelError("readout and phase forms cannot be interchanged")
    if type(value.terms) is not tuple or len(value.terms) > MAX_SUPPORT:
        raise KernelError("bounded immutable sparse terms required")
    previous = -1
    for item in value.terms:
        if type(item) is not tuple or len(item) != 2:
            raise KernelError("each sparse term is (token, Fraction)")
        token, coefficient = _token(item[0], value.frame, value.kind), _f(item[1])
        order = _order(token, value.kind)
        if order <= previous or coefficient == ZERO:
            raise KernelError("form terms must be unique, nonzero and canonical")
        previous = order
    return value


def _make_form(frame, kind, bias, terms):
    frame, kind, bias = _frame(frame), _kind(kind), _f(bias)
    if type(terms) is not tuple or len(terms) > MAX_SUPPORT:
        raise KernelError("bounded immutable sparse terms required")
    merged = {}
    for item in terms:
        if type(item) is not tuple or len(item) != 2:
            raise KernelError("each sparse term is (token, Fraction)")
        token, coefficient = _token(item[0], frame, kind), _f(item[1])
        merged[token] = _add(merged.get(token, ZERO), coefficient)
    ordered = sorted(((token, coefficient) for token, coefficient in merged.items()
                      if coefficient != ZERO), key=lambda item: _order(item[0], kind))
    return _Form(frame, kind, bias, tuple(ordered))


def form(frame, kind, bias, terms, *, enabled=False):
    if not _on(enabled):
        return None
    return _make_form(frame, kind, bias, terms)


def _sum_forms(frame, kind, terms, bias=ZERO):
    if type(terms) is not tuple or len(terms) > MAX_SUPPORT:
        raise KernelError("bounded immutable form combinations required")
    count = 0
    for item in terms:
        if type(item) is not tuple or len(item) != 2 or type(item[1]) is not _Form:
            raise KernelError("form combination terms are (Fraction, form)")
        if type(item[1].terms) is not tuple:
            raise KernelError("immutable form support required")
        count += len(item[1].terms)
        if count > MAX_SUPPORT:
            raise KernelError("aggregate sparse input limit exceeded")
    total_bias = _f(bias)
    merged = {}
    for coefficient, source in terms:
        coefficient = _f(coefficient)
        source = _checked_form(source, frame, kind)
        total_bias = _add(total_bias, _mul(coefficient, source.bias))
        for token, amount in source.terms:
            merged[token] = _add(merged.get(token, ZERO), _mul(coefficient, amount))
    return _make_form(frame, kind, total_bias, tuple(merged.items()))


def _cube_bounds(value):
    value = _checked_form(value, kind="phase")
    lower = upper = value.bias
    for _, coefficient in value.terms:
        if coefficient < ZERO:
            lower = _add(lower, coefficient)
        else:
            upper = _add(upper, coefficient)
    return lower, upper


def _validate_observation_forms(readout, lower, upper):
    _pre_forms((readout, lower, upper))
    readout = _checked_form(readout, kind="value")
    lower = _checked_form(lower, readout.frame, "phase")
    upper = _checked_form(upper, readout.frame, "phase")
    gap = _sum_forms(readout.frame, "phase", ((ONE, upper), (-ONE, lower)))
    if _cube_bounds(gap)[0] < ZERO:
        raise KernelError("envelope lower exceeds upper on the original-bit cube")
    return readout, lower, upper


def _make_observation(readout, lower, upper, rule, derived=False):
    readout, lower, upper = _validate_observation_forms(readout, lower, upper)
    if not readout.terms:
        if not derived:
            if _cube_bounds(lower)[1] > readout.bias or _cube_bounds(upper)[0] < readout.bias:
                raise KernelError("constant readout contradicts supplied envelope")
        # An exact formal identity proves a derived constant independently of
        # possible loose, separately propagated envelopes of its components.
        lower = upper = _make_form(readout.frame, "phase", readout.bias, ())
    return _Observation(readout, lower, upper, _label(rule))


def _obs(value, frame=None):
    if type(value) is not _Observation:
        raise KernelError("a caller-certified observation is required")
    _label(value.rule)
    _validate_observation_forms(value.readout, value.lower, value.upper)
    if frame is not None and value.frame is not frame:
        raise KernelError("different original frames")
    return value


def observe(readout, lower, upper, *, enabled=False):
    """Accept caller-certified premises; tokens cannot certify their truth."""
    if not _on(enabled):
        return None
    return _make_observation(readout, lower, upper, "caller-certified premise")


def affine(frame, terms, bias=ZERO, *, enabled=False):
    """Fixed signed affine transfer, merging identical whole readouts first."""
    if not _on(enabled):
        return None
    frame, bias = _frame(frame), _f(bias)
    if type(terms) is not tuple or len(terms) > MAX_SUPPORT:
        raise KernelError("bounded immutable affine observation terms required")
    count = 0
    for item in terms:
        if type(item) is not tuple or len(item) != 2 or type(item[1]) is not _Observation:
            raise KernelError("affine terms are (Fraction, observation)")
        observation = item[1]
        count += _pre_forms((observation.readout, observation.lower, observation.upper))
        if count > MAX_SUPPORT:
            raise KernelError("aggregate affine sparse input limit exceeded")
    groups = {}
    for coefficient, source in terms:
        coefficient, source = _f(coefficient), _obs(source, frame)
        key = (source.readout.bias, source.readout.terms)
        if key in groups:
            groups[key][0] = _add(groups[key][0], coefficient)
        else:
            groups[key] = [coefficient, source]
    readout_terms, lower_terms, upper_terms = [], [], []
    for coefficient, source in groups.values():
        if coefficient == ZERO:
            continue
        readout_terms.append((coefficient, source.readout))
        lower_terms.append((coefficient, source.lower if coefficient > ZERO else source.upper))
        upper_terms.append((coefficient, source.upper if coefficient > ZERO else source.lower))
    readout = _sum_forms(frame, "value", tuple(readout_terms), bias)
    lower = _sum_forms(frame, "phase", tuple(lower_terms), bias)
    upper = _sum_forms(frame, "phase", tuple(upper_terms), bias)
    return _make_observation(readout, lower, upper, "fixed signed affine transfer", derived=True)


def _positive(value, upper):
    value = _checked_form(value, kind="phase")
    base = value.bias
    for _, coefficient in value.terms:
        if coefficient < ZERO:
            base = _add(base, coefficient)
    output_bias = max(ZERO, base)
    prefix = base
    terms = []
    # Input terms are already in the unique original ordinal order.  A
    # complement literal is only a view of its original bit, never a new bit.
    for phase, coefficient in value.terms:
        magnitude = _neg(coefficient) if coefficient < ZERO else coefficient
        if upper:
            prefix = _add(prefix, magnitude)
            marginal = min(magnitude, max(ZERO, prefix))
        else:
            marginal = min(magnitude, max(ZERO, _add(base, magnitude)))
        if coefficient < ZERO:
            output_bias = _add(output_bias, marginal)
            amount = _neg(marginal)
        else:
            amount = marginal
        if amount != ZERO:
            terms.append((phase, amount))
    return _make_form(value.frame, "phase", output_bias, tuple(terms))


def positive_majorant(value, *, enabled=False):
    """Fixed-prefix affine upper bound on ReLU(value), including the cube."""
    if not _on(enabled):
        return None
    return _positive(value, upper=True)


def positive_minorant(value, *, enabled=False):
    """Singleton affine lower bound for BINARY phases, not cube-pointwise."""
    if not _on(enabled):
        return None
    return _positive(value, upper=False)


def relu_pair(difference, companion, r, t, *, enabled=False):
    """Caller binds difference=h-w, companion=w, r=R(h), t=R(w).

    r and t must already be original value tokens in the same frame.  This
    function declares no original/output phase and constructs no shadow gate.
    """
    if not _on(enabled):
        return None
    if type(difference) is not _Observation or type(companion) is not _Observation:
        raise KernelError("difference and right companion observations required")
    _pre_forms((difference.readout, difference.lower, difference.upper,
                companion.readout, companion.lower, companion.upper))
    difference = _obs(difference)
    frame = difference.frame
    companion = _obs(companion, frame)
    r, t = _value(r, frame), _value(t, frame)
    negative_lower = _sum_forms(frame, "phase", ((-ONE, difference.lower),))
    lower = _sum_forms(frame, "phase", ((-ONE, _positive(negative_lower, True)),))
    upper = _positive(difference.upper, True)
    companion_lower = _positive(companion.lower, False)
    companion_upper = _positive(companion.upper, True)
    difference_readout = _make_form(frame, "value", ZERO, ((r, ONE), (t, -ONE)))
    companion_readout = _make_form(frame, "value", ZERO, ((t, ONE),))
    return (_make_observation(difference_readout, lower, upper,
                              "multiphase ReLU difference", derived=True),
            _make_observation(companion_readout, companion_lower, companion_upper,
                              "multiphase ReLU companion", derived=True))


def compile_rows(observation, *, enabled=False):
    """Return upper/lower rows (value_terms, phase_terms, rhs), each <= rhs."""
    if not _on(enabled):
        return None
    observation = _obs(observation)
    readout, lower, upper = observation.readout, observation.lower, observation.upper
    return ((readout.terms,
             tuple((token, _neg(coefficient)) for token, coefficient in upper.terms),
             _sub(upper.bias, readout.bias)),
            (tuple((token, _neg(coefficient)) for token, coefficient in readout.terms),
             lower.terms, _sub(readout.bias, lower.bias)))


def evaluate(value, assignments, *, enabled=False):
    """Evaluate a form at supplied exact data; this does not run a network.

    Phase values in [0,1] are accepted for mathematical LP-row checks.  That
    convenience does not turn the original domain into a continuous cube, or
    make positive_minorant a cube-pointwise lower bound for the hinge.
    """
    if not _on(enabled):
        return None
    value = _checked_form(value)
    if type(assignments) is not dict or len(assignments) > MAX_SUPPORT:
        raise KernelError("a bounded exact token-to-Fraction assignment is required")
    for token, amount in assignments.items():
        if type(token) is _Value:
            _value(token, value.frame)
        elif type(token) is _Phase:
            _phase(token, value.frame)
        else:
            raise KernelError("assignment key is not an original token")
        amount = _f(amount)
        if type(token) is _Phase and not ZERO <= amount <= ONE:
            raise KernelError("phase assignment must lie in [0,1]")
    result = value.bias
    for token, coefficient in value.terms:
        if token not in assignments:
            raise KernelError("missing original token assignment")
        result = _add(result, _mul(coefficient, assignments[token]))
    return result
