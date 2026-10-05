"""Default-off, shared-original-phase four-slot relation interface.

The caller certifies the original affine readouts, parameter enclosures,
frontier box, gate topology, phases and decoder. Typed identity is NOT that
certificate. All original HZ variables, bits, EQ/LE and decoder remain present;
this module neither installs rows nor invokes a solver or a native verifier.

Preparation caches whole-source affine supports and dyadic24 majorants. A
condition uses two ORIGINAL gate indices, never a phase-restricted subproblem.
The midpoint anchor expression is actual_bit * midpoint_preactivation, NOT a
new midpoint ReLU. One uniform error covers actual/midpoint phase disagreement.
Real population selection must independently preregister fixed original pairs.

Every public operation is opt-in. Exact Fraction inputs and intermediate
arithmetic have the inherited 512-bit cap. Preparation counts all original
source/gate/term occurrences before merged containers; subsequent operations
preflight their immediate inputs, sparse merges and returned rows against
65536. These are finite component limits, not whole-run work/memory or GPU
qualification. Preparation retains O(source declarations + affine occurrences)
numeric entries; its four sparse bases also occupy O(source declarations).
Per pair, sparse updates visit only the two anchor supports (a fixed number of
times); the complete premises/caches remain retained and must be ledgered.

Frozen results retain caller-owned original identity tokens, whose registries
are not changed here. Cache mappings themselves are read-only. Premises are a
shared proof DAG, not copies of source variables and not free whole-network
storage. Reversed conditional endpoints can describe an empty original phase;
they never cause deletion/fixing of a bit or a SAFE/UNSAT conclusion.
"""

from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType

from experiments.neural_hz_20260831.definition_first_20260928.d063_mixed_block_component_20261001 import mixed_block as mb


ss, ms, mp = mb.ss, mb.ms, mb.mp
KernelError = ms.KernelError
MAX_BITS, MAX_SUPPORT = ms.MAX_BITS, ms.MAX_SUPPORT
ZERO, ONE = Fraction(0), Fraction(1)
DYADIC_DENOMINATOR = Fraction(1 << 24)
_f, _add, _sub, _mul, _neg = ms._f, ms._add, ms._sub, ms._mul, ms._neg
_SEAL = object()


@dataclass(frozen=True)
class _Point:
    bias: Fraction
    terms: tuple


@dataclass(frozen=True)
class _Base:
    bias: Fraction
    coefficients: MappingProxyType
    support: Fraction


@dataclass(frozen=True)
class Prepared:
    context: ms._Context
    v: ss._Affine
    gates: tuple
    receiver: mp._Value
    error: Fraction
    midpoint_weights: tuple
    midpoint_forms: tuple
    midpoint_bounds: tuple
    slopes: tuple
    majorants: tuple
    _geometry: MappingProxyType
    _upper_bases: tuple
    _lower_bases: tuple
    _seal: object


@dataclass(frozen=True)
class Table:
    context: ms._Context
    alpha: mp._Phase
    beta: mp._Phase
    receiver: mp._Value
    lower: tuple
    upper: tuple
    premises: tuple
    _seal: object


@dataclass(frozen=True)
class Overlap:
    """A continuous declaration; equality keys are original identity tokens."""
    context: ms._Context
    alpha: mp._Phase
    beta: mp._Phase


@dataclass(frozen=True)
class Compiled:
    context: object
    overlaps: tuple
    rows: tuple
    tables: tuple


def _bounded(count):
    if count > MAX_SUPPORT:
        raise KernelError('aggregate interface support limit exceeded')


def _point(value):
    bias, _ = mb._center_radius(value.bias)
    terms = []
    for source, lower, upper in value.terms:
        coefficient, _ = mb._center_radius((lower, upper))
        if coefficient != ZERO:
            terms.append((source, coefficient))
    return _Point(bias, tuple(terms))


def _point_bounds(point, geometry):
    center, radius = point.bias, ZERO
    for source, coefficient in point.terms:
        source_center, source_radius, _ = geometry[source]
        center = _add(center, _mul(coefficient, source_center))
        radius = _add(radius, _mul(mb._absolute(coefficient), source_radius))
    return _sub(center, radius), _add(center, radius)


def _majorant(point, bounds):
    lower, upper = bounds
    if lower >= ZERO:
        return ONE, point
    if upper <= ZERO:
        return ZERO, _Point(ZERO, ())
    ratio = mb._divide(upper, _sub(upper, lower))
    scaled = _mul(DYADIC_DENOMINATOR, ratio)
    integer = _f(Fraction(scaled.numerator // scaled.denominator))
    slope = mb._divide(integer, DYADIC_DENOMINATOR)
    intercept = _mul(_sub(ONE, slope), upper)
    bias = _add(_mul(slope, point.bias), intercept)
    terms = []
    for source, coefficient in point.terms:
        coefficient = _mul(slope, coefficient)
        if coefficient != ZERO:
            terms.append((source, coefficient))
    return slope, _Point(bias, tuple(terms))


def _accumulate(bias, coefficients, point, scale):
    _bounded(len(coefficients) + len(point.terms))
    bias = _add(bias, _mul(scale, point.bias))
    for source, coefficient in point.terms:
        updated = _add(coefficients.get(source, ZERO), _mul(scale, coefficient))
        if updated == ZERO:
            coefficients.pop(source, None)
        else:
            coefficients[source] = updated
    return bias


def _bases(v, forms, majorants, weights, geometry, sign):
    coefficients = {}
    bias = _accumulate(ZERO, coefficients, v, sign)
    for point, weight in zip(majorants, weights):
        signed = _mul(sign, weight)
        if signed > ZERO:
            bias = _accumulate(bias, coefficients, point, signed)
    support = mb._support(bias, coefficients, geometry)
    first = _Base(bias, MappingProxyType(coefficients), support)
    second_coefficients = dict(coefficients)
    second_bias = bias
    for point, weight in zip(forms, weights):
        signed = _mul(sign, weight)
        if signed < ZERO:
            second_bias = _accumulate(second_bias, second_coefficients, point, signed)
    second_support = mb._support(second_bias, second_coefficients, geometry)
    second = _Base(second_bias, MappingProxyType(second_coefficients), second_support)
    return first, second


def prepare(v, gates, receiver, *, enabled=False):
    """Prepare a certified existing F=v+sum(weight*original_ReLU) readout.

    gates uses D063's (ss._Affine, original output, own phase, weight interval)
    tuples with strictly increasing original phase ordinals. All inputs, even
    zero-weight gates and unused/fixed sources, are checked and retained.
    """
    if not ms._on(enabled):
        return None
    context = mb._preflight(v, gates)
    v, receiver = mb._validate(v, gates, receiver, context)
    geometry = MappingProxyType(mb._geometry(context))
    _, error = mb._reference_info(v, geometry)
    weights, forms, bounds, slopes, majorants = [], [], [], [], []
    for preactivation, _, _, interval in gates:
        weight, weight_radius = mb._center_radius(interval)
        support, epsilon = mb._reference_info(preactivation, geometry)
        amplitude = max(ZERO, _add(support, epsilon))
        error = _add(error, _mul(weight_radius, amplitude))
        error = _add(error, _mul(mb._absolute(weight), epsilon))
        point = _point(preactivation)
        bound = _point_bounds(point, geometry)
        slope, majorant = _majorant(point, bound)
        weights.append(weight)
        forms.append(point)
        bounds.append(bound)
        slopes.append(slope)
        majorants.append(majorant)
    weights, forms, majorants = tuple(weights), tuple(forms), tuple(majorants)
    reference_v = _point(v)
    upper_bases = _bases(reference_v, forms, majorants, weights, geometry, ONE)
    lower_bases = _bases(reference_v, forms, majorants, weights, geometry, _neg(ONE))
    return Prepared(context, v, gates, receiver, _f(error), weights, forms,
                    tuple(bounds), tuple(slopes), majorants, geometry,
                    upper_bases, lower_bases, _SEAL)


def _prepared(value):
    if type(value) is not Prepared or value._seal is not _SEAL:
        raise KernelError('an owned prepared interface is required')
    ms._context(value.context)
    mp._value(value.receiver, value.context.frame)
    return value


def _table(value):
    if type(value) is not Table or value._seal is not _SEAL:
        raise KernelError('an owned four-slot table is required')
    context = ms._context(value.context)
    mp._value(value.receiver, context.frame)
    alpha = mp._phase(value.alpha, context.frame)
    beta = mp._phase(value.beta, context.frame)
    if alpha is beta or alpha.ordinal >= beta.ordinal:
        raise KernelError('two ordered distinct original phases are required')
    for endpoints in (value.lower, value.upper):
        if type(endpoints) is not tuple or len(endpoints) != 4:
            raise KernelError('four immutable endpoints are required')
        for endpoint in endpoints:
            _f(endpoint)
    # Do not reinterpret an empty original phase as an invalid network or
    # remove an original label. Empty-state facts remain conditional facts.
    return value


def _direction(prepared, i, j, sign, bases):
    weights = tuple(_mul(sign, prepared.midpoint_weights[index]) for index in (i, j))
    positive = tuple(max(ZERO, weight) for weight in weights)
    points = tuple(prepared.midpoint_forms[index] for index in (i, j))
    majorants = tuple(prepared.majorants[index] for index in (i, j))
    _bounded(sum(len(point.terms) for point in points + majorants))
    increment = {}
    bias = ZERO
    for majorant, weight in zip(majorants, positive):
        bias = _accumulate(bias, increment, majorant, _neg(weight))
    base = bases[0]
    support00 = mb._increment_support(base.coefficients, base.support, bias,
                                      increment, prepared._geometry)
    first = dict(increment)
    bias10 = _accumulate(bias, first, points[0], weights[0])
    support10 = mb._increment_support(base.coefficients, base.support, bias10,
                                      first, prepared._geometry)
    second = dict(increment)
    bias01 = _accumulate(bias, second, points[1], weights[1])
    support01 = mb._increment_support(base.coefficients, base.support, bias01,
                                      second, prepared._geometry)
    joint = {}
    bias11 = ZERO
    for point, majorant, weight in zip(points, majorants, positive):
        bias11 = _accumulate(bias11, joint, point, weight)
        bias11 = _accumulate(bias11, joint, majorant, _neg(weight))
    base = bases[1]
    support11 = mb._increment_support(base.coefficients, base.support, bias11,
                                      joint, prepared._geometry)
    return support00, support10, support01, support11


def condition(prepared, i, j, *, enabled=False):
    """Seed one fixed ordered original anchor pair; do not search for a pair."""
    if not ms._on(enabled):
        return None
    prepared = _prepared(prepared)
    if (type(i) is not int or type(j) is not int
            or not 0 <= i < j < len(prepared.gates)):
        raise KernelError('indices must be ordered distinct original gate slots')
    alpha = mp._phase(prepared.gates[i][2], prepared.context.frame)
    beta = mp._phase(prepared.gates[j][2], prepared.context.frame)
    upper = _direction(prepared, i, j, ONE, prepared._upper_bases)
    negative = _direction(prepared, i, j, _neg(ONE), prepared._lower_bases)
    upper = tuple(_add(bound, prepared.error) for bound in upper)
    lower = tuple(_sub(_neg(bound), prepared.error) for bound in negative)
    return Table(prepared.context, alpha, beta, prepared.receiver, lower, upper,
                 ('whole-box seed', prepared, i, j), _SEAL)


def _table_population(tables):
    if type(tables) is not tuple:
        raise KernelError('an immutable table tuple is required')
    # Every endpoint occurrence counts even when coefficients later cancel.
    _bounded(9 * len(tables))
    for table in tables:
        _table(table)
    return tables


def _receiver(receiver, context, inputs):
    receiver = mp._value(receiver, context.frame)
    if receiver in context._by_value or any(receiver is item.receiver for item in inputs):
        raise KernelError('a new existing forward receiver cannot be its own source')
    return receiver


def affine(tables, weights, bias, receiver, *, enabled=False):
    """Propagate same-anchor tables along a caller-certified affine operation.

    weights and bias are exact Fractions. Repeated receiver identities are
    combined first; the first valid table for that readout supplies its bounds.
    No new pointwise minimum, phase selector or renamed anchor is introduced.
    """
    if not ms._on(enabled):
        return None
    tables = _table_population(tables)
    if (not tables or type(weights) is not tuple or len(weights) != len(tables)):
        raise KernelError('nonempty tables and equally many Fraction weights required')
    _bounded(10 * len(tables))
    bias = _f(bias)
    first = tables[0]
    for table, weight in zip(tables, weights):
        _f(weight)
        if (table.context is not first.context or table.alpha is not first.alpha
                or table.beta is not first.beta):
            raise KernelError('affine inputs must share context and original anchor identities')
    receiver = _receiver(receiver, first.context, tables)
    groups = {}
    for table, weight in zip(tables, weights):
        previous = groups.get(table.receiver)
        if previous is None:
            groups[table.receiver] = (table, weight)
        else:
            groups[table.receiver] = (previous[0], _add(previous[1], weight))
    lower, upper = [bias] * 4, [bias] * 4
    for table, weight in groups.values():
        lows, highs = ((table.lower, table.upper) if weight >= ZERO
                       else (table.upper, table.lower))
        for slot in range(4):
            lower[slot] = _add(lower[slot], _mul(weight, lows[slot]))
            upper[slot] = _add(upper[slot], _mul(weight, highs[slot]))
    return Table(first.context, first.alpha, first.beta, receiver,
                 tuple(lower), tuple(upper), ('affine', tables, weights, bias), _SEAL)


def relu(table, bias, receiver, own_phase, *, enabled=False):
    """Propagate R(original_readout+bias); keep the new original gate's bit."""
    if not ms._on(enabled):
        return None
    table = _table(table)
    bias = _f(bias)
    receiver = _receiver(receiver, table.context, (table,))
    own_phase = mp._phase(own_phase, table.context.frame)
    if own_phase.original_output is not receiver:
        raise KernelError('ReLU receiver and its original own phase must agree')
    lower = tuple(max(ZERO, _add(value, bias)) for value in table.lower)
    upper = tuple(max(ZERO, _add(value, bias)) for value in table.upper)
    return Table(table.context, table.alpha, table.beta, receiver, lower, upper,
                 ('ReLU', table, bias, own_phase), _SEAL)


def _coefficients(endpoints):
    constant, first, second, joint = endpoints
    first, second = _sub(first, constant), _sub(second, constant)
    interaction = _sub(_sub(_sub(joint, constant), first), second)
    return constant, first, second, interaction


def _nonzero_terms(pairs):
    return tuple((token, coefficient) for token, coefficient in pairs if coefficient != ZERO)


def compile_rows(tables, *, enabled=False):
    """Return immutable named LE rows and shared continuous declarations only.

    A row is (value_terms, phase_terms, overlap_terms, rhs), meaning sum<=rhs.
    Four McCormick rows per original anchor pair precede two rows per table.
    Equal Overlap keys across calls denote the same context/anchor identities;
    native allocation and authentication are not performed by this component.
    """
    if not ms._on(enabled):
        return None
    tables = _table_population(tables)
    if not tables:
        return Compiled(None, (), (), ())
    context = tables[0].context
    for table in tables:
        if table.context is not context:
            raise KernelError('compiled tables must share the same source context')
    # At worst every table has a distinct pair: 8 McCormick + 8 bound nnz.
    # Preflight before allocating the pair registry or output row containers.
    _bounded(16 * len(tables))
    by_pair = {}
    for table in tables:
        key = (table.alpha, table.beta)
        if key not in by_pair:
            by_pair[key] = Overlap(context, table.alpha, table.beta)
    overlaps = tuple(by_pair.values())
    rows = []
    for delta in overlaps:
        rows.extend((
            ((), (), ((delta, _neg(ONE)),), ZERO),
            ((), ((delta.alpha, _neg(ONE)),), ((delta, ONE),), ZERO),
            ((), ((delta.beta, _neg(ONE)),), ((delta, ONE),), ZERO),
            ((), ((delta.alpha, ONE), (delta.beta, ONE)), ((delta, _neg(ONE)),), ONE),
        ))
    for table in tables:
        delta = by_pair[(table.alpha, table.beta)]
        constant, first, second, joint = _coefficients(table.upper)
        phases = _nonzero_terms(((table.alpha, _neg(first)), (table.beta, _neg(second))))
        overlap = _nonzero_terms(((delta, _neg(joint)),))
        rows.append((((table.receiver, ONE),), phases, overlap, constant))
        constant, first, second, joint = _coefficients(table.lower)
        phases = _nonzero_terms(((table.alpha, first), (table.beta, second)))
        overlap = _nonzero_terms(((delta, joint),))
        rows.append((((table.receiver, _neg(ONE)),), phases, overlap, _neg(constant)))
    return Compiled(context, overlaps, tuple(rows), tables)
