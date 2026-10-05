"""Default-off fixed four-plane bounds for an existing mixed ReLU readout.

The caller authenticates the real source frontier, parameter enclosures, every
original gate/phase and the EXISTING receiver's affine readout. Formal tokens
check shared identities, not model topology, native columns or decoder facts.
All original sources, gates, bits (including both legal zero choices), EQ/LE
predicates and the input decoder remain untouched. No new phase is declared.

Positive midpoint weights are paired in original phase-ordinal order. Each
direction shares two affine bases and computes whole-box plane supports by
sparse corrections, not phase-restricted searches. Nonpoint parameters are
covered by an explicit error on the actual readout. Midpoint activations are
proof expressions only, never replacement gates. The only returned LP rows
bound the original receiver by two constants; no max-epigraph is installed.

Every rational input and arithmetic result is limited to 512 bits. Before any
merged container is built, source declarations, gates and ALL input affine
term occurrences together must fit 65536. Zero weights/forms are still checked
and retained. This is a mathematical component, not a whole-work, physical
memory, native, GPU or verification qualification. Original premises, source
lookup/normalization, row storage and evidence remain costs to the caller.
"""

from dataclasses import dataclass
from fractions import Fraction

from experiments.neural_hz_20260831.definition_first_20260928.d053_static_source_component_20260930 import static_source as ss


ms, mp = ss.ms, ss.mp
KernelError = ms.KernelError
MAX_BITS, MAX_SUPPORT = ms.MAX_BITS, ms.MAX_SUPPORT
ZERO, ONE, TWO = Fraction(0), Fraction(1), Fraction(2)
_f, _add, _sub, _mul, _neg = ms._f, ms._add, ms._sub, ms._mul, ms._neg


@dataclass(frozen=True)
class _Certificate:
    context: ms._Context
    v: ss._Affine
    gates: tuple
    receiver: mp._Value
    lower: Fraction
    upper: Fraction
    error: Fraction
    upper_groups: tuple
    lower_groups: tuple
    rows: tuple
    observation: ms._Observation


def _absolute(value):
    value = _f(value)
    return _neg(value) if value < ZERO else value


def _divide(value, divisor):
    value, divisor = _f(value), _f(divisor)
    if divisor == ZERO:
        raise KernelError('zero divisor in support construction')
    return _f(value / divisor)


def _center_radius(interval):
    lower, upper = ss._interval(interval)
    return (_divide(_add(lower, upper), TWO),
            _divide(_sub(upper, lower), TWO))


def _preflight(v, gates):
    """Count occurrences, including unused/fixed sources and zero weights."""
    count = ss._shape(v)
    context = ms._context(v.context)
    if type(gates) is not tuple or len(gates) > MAX_SUPPORT:
        raise KernelError('a bounded immutable tuple of original gates is required')
    count += len(context.sources) + len(gates)
    if count > MAX_SUPPORT:
        raise KernelError('aggregate block input limit exceeded')
    for gate in gates:
        if type(gate) is not tuple or len(gate) != 4:
            raise KernelError('each gate is (affine, original output, own phase, weight interval)')
        count += ss._shape(gate[0])
        if count > MAX_SUPPORT:
            raise KernelError('aggregate block input limit exceeded')
    return context


def _validate(v, gates, receiver, context):
    v = ss._checked(v, context)
    receiver = mp._value(receiver, context.frame)
    previous_source = -1
    for source in context.sources:
        source = ms._source(source, context)
        if source.ordinal <= previous_source:
            raise KernelError('source context must have canonical original ordinals')
        if source.original_value is receiver:
            raise KernelError('receiver cannot be its own registered source')
        previous_source = source.ordinal
    seen_outputs, seen_phases = set(), set()
    previous_phase = -1
    for preactivation, output, phase, weight in gates:
        preactivation = ss._checked(preactivation, context)
        output = mp._value(output, context.frame)
        phase = mp._phase(phase, context.frame)
        ss._interval(weight)
        if (output is receiver or output in seen_outputs or phase in seen_phases
                or phase.ordinal <= previous_phase
                or phase.original_output is not output):
            raise KernelError('distinct original outputs with ordered own phases required')
        if any(source.original_value is output for source, _, _ in preactivation.terms):
            raise KernelError('a preactivation cannot directly use its own output')
        # Other gate outputs may be shared frontier coordinates. Their true
        # topology is a caller premise, not a reason to clone or reject them.
        seen_outputs.add(output)
        seen_phases.add(phase)
        previous_phase = phase.ordinal
    return v, receiver


def _geometry(context):
    result = {}
    for source in context.sources:
        center, radius = _center_radius((source.lower, source.upper))
        result[source] = (center, radius,
                          max(_absolute(source.lower), _absolute(source.upper)))
    return result


def _reference_info(value, geometry):
    """Midpoint support and uniform error; fixed sources are NOT discarded."""
    bias, error = _center_radius(value.bias)
    support = bias
    for source, lower, upper in value.terms:
        coefficient, radius = _center_radius((lower, upper))
        center, width_radius, magnitude = geometry[source]
        support = _add(support, _mul(coefficient, center))
        support = _add(support, _mul(_absolute(coefficient), width_radius))
        error = _add(error, _mul(radius, magnitude))
    return _f(support), _f(error)


def _accumulate_midpoint(bias, coefficients, value, scale):
    """Add one certified midpoint form without caching per-gate copies."""
    scale = _f(scale)
    middle, _ = _center_radius(value.bias)
    bias = _add(bias, _mul(scale, middle))
    for source, lower, upper in value.terms:
        middle, _ = _center_radius((lower, upper))
        amount = _add(coefficients.get(source, ZERO), _mul(scale, middle))
        if amount == ZERO:
            coefficients.pop(source, None)
        else:
            coefficients[source] = amount
        if len(coefficients) > MAX_SUPPORT:
            raise KernelError('merged support exceeds limit')
    return bias


def _support(bias, coefficients, geometry):
    if len(coefficients) > MAX_SUPPORT:
        raise KernelError('box support exceeds limit')
    total = _f(bias)
    for source, coefficient in coefficients.items():
        coefficient = _f(coefficient)
        center, radius, _ = geometry[source]
        total = _add(total, _mul(coefficient, center))
        total = _add(total, _mul(_absolute(coefficient), radius))
    return _f(total)


def _increment_support(base, base_support, bias, increment, geometry):
    """sigma(B+t), including t's bias and SAME-source merged coefficients."""
    if len(base) + len(increment) > MAX_SUPPORT:
        raise KernelError('aggregate cached support exceeds limit')
    result = _add(_f(base_support), _f(bias))
    for source, amount in increment.items():
        amount = _f(amount)
        old = _f(base.get(source, ZERO))
        updated = _add(old, amount)
        center, radius, _ = geometry[source]
        result = _add(result, _mul(amount, center))
        correction = _sub(_absolute(updated), _absolute(old))
        result = _add(result, _mul(correction, radius))
    return _f(result)


def _direction(v, gates, weights, geometry, sign):
    """Bound sign*Fbar, sharing B0=v/G and B1=(v-Hminus)/G."""
    sign = _f(sign)
    positive = []
    for index, weight in enumerate(weights):
        if _mul(sign, weight) > ZERO:
            positive.append(index)
    if not positive:
        coefficients = {}
        bias = _accumulate_midpoint(ZERO, coefficients, v, sign)
        support = _support(bias, coefficients, geometry)
        # A one-plane structural case, not a dummy gate or omitted premise.
        return support, ((support,),)

    group_count = (len(positive) + 1) // 2
    inverse_groups = _divide(ONE, _f(Fraction(group_count)))
    base_scale = _mul(sign, inverse_groups)
    base_zero = {}
    bias_zero = _accumulate_midpoint(ZERO, base_zero, v, base_scale)
    base_one = dict(base_zero)
    bias_one = bias_zero
    for gate, weight in zip(gates, weights):
        signed_weight = _mul(sign, weight)
        if signed_weight < ZERO:
            # signed_weight is -d_j, hence this subtracts Hminus/G.
            scale = _mul(signed_weight, inverse_groups)
            bias_one = _accumulate_midpoint(bias_one, base_one, gate[0], scale)
    support_zero = _support(bias_zero, base_zero, geometry)
    support_one = _support(bias_one, base_one, geometry)

    groups, result = [], ZERO
    for position in range(0, len(positive), 2):
        first = positive[position]
        increment = {}
        bias = _accumulate_midpoint(ZERO, increment, gates[first][0],
                                    _mul(sign, weights[first]))
        if position + 1 == len(positive):
            planes = (support_zero,
                      _increment_support(base_one, support_one, bias, increment, geometry))
        else:
            support_first = _increment_support(base_zero, support_zero, bias,
                                                increment, geometry)
            second = positive[position + 1]
            second_increment = {}
            second_bias = _accumulate_midpoint(ZERO, second_increment, gates[second][0],
                                               _mul(sign, weights[second]))
            support_second = _increment_support(base_zero, support_zero, second_bias,
                                                 second_increment, geometry)
            # Merge BEFORE the norm correction. Taking two independent support
            # corrections would lose cancellation and would not be this rule.
            if len(increment) + len(second_increment) > MAX_SUPPORT:
                raise KernelError('aggregate two-gate increment exceeds limit')
            bias = _add(bias, second_bias)
            for source, amount in second_increment.items():
                merged = _add(increment.get(source, ZERO), amount)
                if merged == ZERO:
                    increment.pop(source, None)
                else:
                    increment[source] = merged
            support_joint = _increment_support(base_one, support_one, bias,
                                                increment, geometry)
            planes = (support_zero, support_first, support_second, support_joint)
        for bound in planes:
            _f(bound)
        result = _add(result, max(planes))
        groups.append(planes)
    return _f(result), tuple(groups)


def _check_rows(rows, context):
    if type(rows) is not tuple or len(rows) != 2:
        raise KernelError('two immutable receiver LE rows required')
    count = 0
    for values, phases, rhs in rows:
        count += len(values) + len(phases)
        if count > MAX_SUPPORT:
            raise KernelError('aggregate returned row support exceeds limit')
        _f(rhs)
        for token, coefficient in values:
            mp._value(token, context.frame)
            _f(coefficient)
        for token, coefficient in phases:
            mp._phase(token, context.frame)
            _f(coefficient)


def generate(v, gates, receiver, *, enabled=False):
    """Certify bounds on a caller-bound existing receiver; never install them.

    gates is the immutable tuple (ss._Affine, original output, own original
    phase, Fraction interval), strictly ordered by original phase ordinal.
    All forms use v.context. The frozen result retains these original objects.
    upper_groups/lower_groups contain plane supports for Fbar/-Fbar: four per
    pair, two per odd tail, or one for the no-positive structural case. Bounds
    include the shared interval error; the group supports themselves do not.
    observation is D049 and rows are (value_terms, phase_terms, rhs), upper
    first. They reference only receiver, never an unauthenticated midpoint
    consumer or independently assigned clone of an original source.
    """
    if not ms._on(enabled):
        return None
    context = _preflight(v, gates)
    v, receiver = _validate(v, gates, receiver, context)
    geometry = _geometry(context)
    _, error = _reference_info(v, geometry)
    weights = []
    for preactivation, _, _, interval in gates:
        weight, radius = _center_radius(interval)
        support, epsilon = _reference_info(preactivation, geometry)
        amplitude = max(ZERO, _add(support, epsilon))
        error = _add(error, _mul(radius, amplitude))
        error = _add(error, _mul(_absolute(weight), epsilon))
        weights.append(weight)
    weights = tuple(weights)
    reference_upper, upper_groups = _direction(v, gates, weights, geometry, ONE)
    negative_lower, lower_groups = _direction(v, gates, weights, geometry, _neg(ONE))
    upper = _add(reference_upper, error)
    lower = _sub(_neg(negative_lower), error)
    if lower > upper:
        raise KernelError('inconsistent constructed receiver bounds')
    read = ms.readout(context.frame, ZERO, ((receiver, ONE),), enabled=True)
    lower_form = ms.form(context, lower, enabled=True)
    upper_form = ms.form(context, upper, enabled=True)
    observation = ms._observation(read, lower_form, upper_form,
                                  'fixed mixed-block four-plane support', derived=True)
    rows = ms.compile_rows(observation, enabled=True)
    _check_rows(rows, context)
    return _Certificate(context, v, gates, receiver, _f(lower), _f(upper), _f(error),
                        upper_groups, lower_groups, rows, observation)
