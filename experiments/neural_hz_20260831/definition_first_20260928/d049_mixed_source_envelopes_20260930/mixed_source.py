"""Default-off exact mixed continuous-source/original-phase observations.

The caller certifies the common original frame, source intervals, phase
direction, real gates/readouts and observation premises. Formal tokens do not
establish those facts or native columns. Original factors, bits, predicates
and decoder are untouched. No source is treated as a new independent noise.

Upper hinges use continuous-source ordinals, then original-phase ordinals.
Lower hinges keep a fixed continuous supporting plane and add ONLY binary
singleton marginals. A lower bound need not hold pointwise at fractional bits.
Admission requires compatible bounds on the entire source box and bit cube;
failure is not an UNSAT verdict. Pure-phase arithmetic recovers frozen D046.

Inputs/intermediate rational results are limited to 512 bits. Sparse input
occurrences are preflighted in aggregate at 65536 before combinations build
containers. Returned LE rows may duplicate readout support. Source scales,
normalization, sorting, copies and actual latent expansion have costs: this
module claims no whole-work, complete physical-memory, native or GPU gate.
"""
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType

from experiments.neural_hz_20260831.definition_first_20260928.d046_multiphase_envelopes_20260930 import multiphase as mp

MAX_BITS, MAX_SUPPORT = mp.MAX_BITS, mp.MAX_SUPPORT
ZERO, ONE = Fraction(0), Fraction(1)
KernelError = mp.KernelError
make_frame, make_value, make_phase = mp.make_frame, mp.make_value, mp.make_phase
_on, _f, _add, _sub, _mul, _neg = mp._on, mp._f, mp._add, mp._sub, mp._mul, mp._neg


@dataclass(frozen=True, eq=False)
class _Source:
    original_value: mp._Value
    ordinal: int
    lower: Fraction
    upper: Fraction


@dataclass(frozen=True, eq=False)
class _Context:
    frame: mp._Frame
    sources: tuple
    _by_ordinal: MappingProxyType
    _by_value: MappingProxyType


@dataclass(frozen=True)
class _Form:
    context: _Context
    bias: Fraction
    source_terms: tuple
    phase_terms: tuple

    @property
    def frame(self):
        return self.context.frame


@dataclass(frozen=True)
class _Observation:
    readout: mp._Form
    lower: _Form
    upper: _Form
    rule: str

    @property
    def context(self):
        return self.lower.context

    @property
    def frame(self):
        return self.context.frame


def _context(value):
    if type(value) is not _Context:
        raise KernelError('an explicit immutable source context is required')
    mp._frame(value.frame)
    if (type(value.sources) is not tuple or len(value.sources) > MAX_SUPPORT
            or type(value._by_ordinal) is not MappingProxyType
            or type(value._by_value) is not MappingProxyType
            or len(value._by_ordinal) != len(value.sources)
            or len(value._by_value) != len(value.sources)):
        raise KernelError('invalid immutable source registry')
    return value


def _source(value, context):
    if type(value) is not _Source:
        raise KernelError('a declared continuous-source token is required')
    mp._value(value.original_value, context.frame)
    mp._ordinal(value.ordinal)
    lower, upper = _f(value.lower), _f(value.upper)
    if (lower > upper or context._by_ordinal.get(value.ordinal) is not value
            or context._by_value.get(value.original_value) is not value):
        raise KernelError('source interval, ordinal or context identity mismatch')
    return value


def source_context(frame, entries, *, enabled=False):
    """Declare existing source values as (mp.Value, ordinal, lower, upper).

    The returned context and its source wrappers are immutable and identity
    distinct. Equal labels/intervals in another context never permit a merge.
    Fixed sources remain in this registry even when forms substitute them.
    """
    if not _on(enabled):
        return None
    frame = mp._frame(frame)
    if type(entries) is not tuple or len(entries) > MAX_SUPPORT:
        raise KernelError('bounded immutable source declarations required')
    by_ordinal, by_value = {}, {}
    for item in entries:
        if type(item) is not tuple or len(item) != 4:
            raise KernelError('source declarations are (value, ordinal, lower, upper)')
        value, ordinal = mp._value(item[0], frame), mp._ordinal(item[1])
        lower, upper = _f(item[2]), _f(item[3])
        if lower > upper or ordinal in by_ordinal or value in by_value:
            raise KernelError('reversed bounds or duplicate original source declaration')
        token = _Source(value, ordinal, lower, upper)
        by_ordinal[ordinal], by_value[value] = token, token
    sources = tuple(by_ordinal[i] for i in sorted(by_ordinal))
    return _Context(frame, sources, MappingProxyType(by_ordinal), MappingProxyType(by_value))


def _size(value):
    if (type(value) is not _Form or type(value.source_terms) is not tuple
            or type(value.phase_terms) is not tuple):
        raise KernelError('immutable typed mixed forms required')
    return len(value.source_terms) + len(value.phase_terms)


def _pre_forms(values, readouts=()):
    count = 0
    for value in readouts:
        if type(value) is not mp._Form or type(value.terms) is not tuple:
            raise KernelError('immutable D046 readout forms required')
        count += len(value.terms)
        if count > MAX_SUPPORT:
            raise KernelError('aggregate sparse input limit exceeded')
    for value in values:
        count += _size(value)
        if count > MAX_SUPPORT:
            raise KernelError('aggregate sparse input limit exceeded')
    return count


def _checked(value, context=None):
    _pre_forms((value,))
    actual = _context(value.context)
    if context is not None and actual is not context:
        raise KernelError('different original source contexts')
    _f(value.bias)
    for terms, continuous in ((value.source_terms, True), (value.phase_terms, False)):
        previous = -1
        for item in terms:
            if type(item) is not tuple or len(item) != 2:
                raise KernelError('sparse terms are (typed token, Fraction)')
            token = _source(item[0], actual) if continuous else mp._phase(item[0], actual.frame)
            coefficient = _f(item[1])
            if (token.ordinal <= previous or coefficient == ZERO
                    or (continuous and token.lower == token.upper)):
                raise KernelError('mixed terms must be canonical, nonzero and nonfixed')
            previous = token.ordinal
    return value


def _make_form(context, bias, source_terms=(), phase_terms=()):
    context, bias = _context(context), _f(bias)
    if (type(source_terms) is not tuple or type(phase_terms) is not tuple
            or len(source_terms) + len(phase_terms) > MAX_SUPPORT):
        raise KernelError('aggregate immutable sparse terms exceed support limit')
    merged_source, merged_phase = {}, {}
    for terms, continuous, merged in ((source_terms, True, merged_source),
                                      (phase_terms, False, merged_phase)):
        for item in terms:
            if type(item) is not tuple or len(item) != 2:
                raise KernelError('sparse terms are (typed token, Fraction)')
            token = _source(item[0], context) if continuous else mp._phase(item[0], context.frame)
            amount = _f(item[1])
            merged[token] = _add(merged.get(token, ZERO), amount)
    for token, amount in tuple(merged_source.items()):
        if token.lower == token.upper:
            bias = _add(bias, _mul(amount, token.lower))
            del merged_source[token]
    sources = tuple(sorted(((t, a) for t, a in merged_source.items() if a != ZERO),
                           key=lambda item: item[0].ordinal))
    phases = tuple(sorted(((t, a) for t, a in merged_phase.items() if a != ZERO),
                          key=lambda item: item[0].ordinal))
    return _Form(context, bias, sources, phases)


def form(context, bias, source_terms=(), phase_terms=(), *, enabled=False):
    if not _on(enabled):
        return None
    return _make_form(context, bias, source_terms, phase_terms)


def readout(frame, bias, terms, *, enabled=False):
    if not _on(enabled):
        return None
    return mp.form(frame, 'value', bias, terms, enabled=True)


def _linear(context, terms, bias=ZERO):
    context = _context(context)
    if type(terms) is not tuple or len(terms) > MAX_SUPPORT:
        raise KernelError('bounded immutable mixed-form combinations required')
    count = 0
    for item in terms:
        if type(item) is not tuple or len(item) != 2:
            raise KernelError('linear terms are (Fraction, mixed form)')
        count += _size(item[1])
        if count > MAX_SUPPORT:
            raise KernelError('aggregate sparse input limit exceeded')
    total = _f(bias)
    sources, phases = {}, {}
    for coefficient, value in terms:
        coefficient, value = _f(coefficient), _checked(value, context)
        total = _add(total, _mul(coefficient, value.bias))
        for items, merged in ((value.source_terms, sources), (value.phase_terms, phases)):
            for token, amount in items:
                merged[token] = _add(merged.get(token, ZERO), _mul(coefficient, amount))
    return _make_form(context, total, tuple(sources.items()), tuple(phases.items()))


def linear(context, terms, bias=ZERO, *, enabled=False):
    if not _on(enabled):
        return None
    return _linear(context, terms, bias)


def _bounds(value):
    value = _checked(value)
    lower = upper = value.bias
    for token, coefficient in value.source_terms:
        a, b = _mul(coefficient, token.lower), _mul(coefficient, token.upper)
        lower, upper = _add(lower, min(a, b)), _add(upper, max(a, b))
    for _, coefficient in value.phase_terms:
        if coefficient < ZERO:
            lower = _add(lower, coefficient)
        else:
            upper = _add(upper, coefficient)
    return lower, upper


def _validate(read, lower, upper):
    _pre_forms((lower, upper), (read,))
    read = mp._checked_form(read, kind='value')
    lower = _checked(lower)
    upper = _checked(upper, lower.context)
    if read.frame is not lower.frame:
        raise KernelError('readout and source context belong to different frames')
    gap = _linear(lower.context, ((ONE, upper), (-ONE, lower)))
    if _bounds(gap)[0] < ZERO:
        raise KernelError('lower exceeds upper on the whole source box and bit cube')
    return read, lower, upper


def _observation(read, lower, upper, rule, derived=False):
    read, lower, upper = _validate(read, lower, upper)
    if not read.terms:
        if not derived and (_bounds(lower)[1] > read.bias or _bounds(upper)[0] < read.bias):
            raise KernelError('constant readout contradicts supplied envelope')
        lower = upper = _make_form(lower.context, read.bias)
    return _Observation(read, lower, upper, mp._label(rule))


def _obs(value, context=None):
    if type(value) is not _Observation:
        raise KernelError('a caller-certified mixed observation is required')
    mp._label(value.rule)
    _validate(value.readout, value.lower, value.upper)
    if context is not None and value.context is not context:
        raise KernelError('different original source contexts')
    return value


def observe(read, lower, upper, *, enabled=False):
    if not _on(enabled):
        return None
    return _observation(read, lower, upper, 'caller-certified mixed premise')


def affine(context, terms, bias=ZERO, *, enabled=False):
    """Combine identical whole physical readouts BEFORE intervalization."""
    if not _on(enabled):
        return None
    context, bias = _context(context), _f(bias)
    if type(terms) is not tuple or len(terms) > MAX_SUPPORT:
        raise KernelError('bounded immutable observation combination required')
    count = 0
    for item in terms:
        if type(item) is not tuple or len(item) != 2 or type(item[1]) is not _Observation:
            raise KernelError('affine terms are (Fraction, observation)')
        value = item[1]
        count += _pre_forms((value.lower, value.upper), (value.readout,))
        if count > MAX_SUPPORT:
            raise KernelError('aggregate affine sparse input limit exceeded')
    groups = {}
    for coefficient, value in terms:
        coefficient, value = _f(coefficient), _obs(value, context)
        key = (value.readout.bias, value.readout.terms)
        if key in groups:
            groups[key][0] = _add(groups[key][0], coefficient)
        else:
            groups[key] = [coefficient, value]
    reads, lowers, uppers = [], [], []
    for coefficient, value in groups.values():
        if coefficient != ZERO:
            reads.append((coefficient, value.readout))
            lowers.append((coefficient, value.lower if coefficient > ZERO else value.upper))
            uppers.append((coefficient, value.upper if coefficient > ZERO else value.lower))
    read = mp._sum_forms(context.frame, 'value', tuple(reads), bias)
    lower, upper = _linear(context, tuple(lowers), bias), _linear(context, tuple(uppers), bias)
    return _observation(read, lower, upper, 'fixed signed mixed affine transfer', derived=True)


def _positive(value, upper):
    value = _checked(value)
    base, continuous_sum = value.bias, ZERO
    literals = []
    for source, coefficient in value.source_terms:
        width = _sub(source.upper, source.lower)
        magnitude = _mul(_neg(coefficient) if coefficient < ZERO else coefficient, width)
        minimum = _mul(coefficient, source.upper if coefficient < ZERO else source.lower)
        base, continuous_sum = _add(base, minimum), _add(continuous_sum, magnitude)
        literals.append((source, coefficient < ZERO, magnitude, width))
    for phase, coefficient in value.phase_terms:
        negative = coefficient < ZERO
        if negative:
            base = _add(base, coefficient)
        literals.append((phase, negative, _neg(coefficient) if negative else coefficient, None))
    # A source with equal bounds was already folded without deleting its token.
    theta = ONE if _add(base, _mul(continuous_sum, Fraction(1, 2))) > ZERO else ZERO
    result_bias = max(ZERO, base) if upper else _mul(theta, base)
    prefix, sources, phases = base, [], []
    for token, negative, magnitude, width in literals:
        if upper:
            prefix = _add(prefix, magnitude)
            marginal = min(magnitude, max(ZERO, prefix))
        elif width is not None:
            marginal = _mul(theta, magnitude)
        else:
            marginal = min(magnitude, max(ZERO, _add(base, magnitude)))
        if width is not None:
            amount = _f(marginal / width)
            if negative:
                result_bias = _add(result_bias, _mul(amount, token.upper))
                amount = _neg(amount)
            else:
                result_bias = _sub(result_bias, _mul(amount, token.lower))
            if amount != ZERO:
                sources.append((token, amount))
        else:
            if negative:
                result_bias = _add(result_bias, marginal)
                marginal = _neg(marginal)
            if marginal != ZERO:
                phases.append((token, marginal))
    return _make_form(value.context, result_bias, tuple(sources), tuple(phases))


def positive_majorant(value, *, enabled=False):
    if not _on(enabled):
        return None
    return _positive(value, True)


def positive_minorant(value, *, enabled=False):
    """Continuous supporting plane plus BINARY-only singleton increments."""
    if not _on(enabled):
        return None
    return _positive(value, False)


def relu_pair(difference, companion, r, t, *, enabled=False):
    """Caller binds h-w, w and the existing true outputs R(h), R(w)."""
    if not _on(enabled):
        return None
    if type(difference) is not _Observation or type(companion) is not _Observation:
        raise KernelError('difference and companion observations required')
    _pre_forms((difference.lower, difference.upper, companion.lower, companion.upper),
               (difference.readout, companion.readout))
    difference = _obs(difference)
    context = difference.context
    companion = _obs(companion, context)
    r, t = mp._value(r, context.frame), mp._value(t, context.frame)
    minus_lower = _linear(context, ((-ONE, difference.lower),))
    lower = _linear(context, ((-ONE, _positive(minus_lower, True)),))
    upper = _positive(difference.upper, True)
    lower_t, upper_t = _positive(companion.lower, False), _positive(companion.upper, True)
    read_difference = mp._make_form(context.frame, 'value', ZERO, ((r, ONE), (t, -ONE)))
    read_companion = mp._make_form(context.frame, 'value', ZERO, ((t, ONE),))
    return (_observation(read_difference, lower, upper, 'mixed-source ReLU difference', derived=True),
            _observation(read_companion, lower_t, upper_t, 'mixed-source ReLU companion', derived=True))


def _row_values(read, source_terms, read_sign, source_sign):
    # Input occurrences, not merely merged output nnz, are checked before the
    # new row dictionary/tuple is allocated. Each row shares one real value
    # namespace; a source is NEVER given an independent assignment slot.
    if len(read.terms) + len(source_terms) > MAX_SUPPORT:
        raise KernelError('aggregate LE value/source support limit exceeded')
    merged = {}
    for token, coefficient in read.terms:
        merged[token] = _mul(read_sign, coefficient)
    for source, coefficient in source_terms:
        token = mp._value(source.original_value, read.frame)
        merged[token] = _add(merged.get(token, ZERO), _mul(source_sign, coefficient))
    return mp._make_form(read.frame, 'value', ZERO, tuple(merged.items())).terms


def compile_rows(observation, *, enabled=False):
    """Formal upper/lower LE rows (value_terms, phase_terms, rhs).

    Source.original_value is merged with identical readout value tokens before
    returning either row. No separate source assignment or fresh value is
    introduced. A native consumer still owes actual latent expansion, bit
    encoding, frame lifetime and decoder authentication; these formal rows
    neither certify native columns nor install a terminal matrix.
    """
    if not _on(enabled):
        return None
    observation = _obs(observation)
    read, lower, upper = observation.readout, observation.lower, observation.upper
    return ((_row_values(read, upper.source_terms, ONE, -ONE),
             tuple((t, _neg(a)) for t, a in upper.phase_terms), _sub(upper.bias, read.bias)),
            (_row_values(read, lower.source_terms, -ONE, ONE),
             lower.phase_terms, _sub(read.bias, lower.bias)))


def evaluate(value, assignments, *, enabled=False):
    """Exact mixed-form evaluation; fractional bits only assist row audits."""
    if not _on(enabled):
        return None
    value = _checked(value)
    if type(assignments) is not dict or len(assignments) > MAX_SUPPORT:
        raise KernelError('bounded exact token assignments required')
    for token, amount in assignments.items():
        amount = _f(amount)
        if type(token) is _Source:
            token = _source(token, value.context)
            if not token.lower <= amount <= token.upper:
                raise KernelError('source assignment outside its certified box')
        elif type(token) is mp._Phase:
            mp._phase(token, value.frame)
            if not ZERO <= amount <= ONE:
                raise KernelError('phase assignment outside [0,1]')
        else:
            raise KernelError('mixed assignment needs declared source or original phase tokens')
    result = value.bias
    for terms in (value.source_terms, value.phase_terms):
        for token, coefficient in terms:
            if token not in assignments:
                raise KernelError('missing original source/phase assignment')
            result = _add(result, _mul(coefficient, assignments[token]))
    return result
