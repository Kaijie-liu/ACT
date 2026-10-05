"""Default-off whole-box joint-negative seeds for the frozen D066 interface.

The caller certifies the existing network readout, original source frontier,
parameter enclosures, original gate/phase identities and decoder. Formal
tokens do not certify those premises or native HZ columns. All original HZ
continuous factors, bits (including both legal labels at zero), EQ/LE and
decoder remain present. No row is installed and no model or solver is run.

For each direction D066 supplies B and B-H. We recover H on the SAME source
identities and prepare one whole-box hinge-support base (B,H). A fixed pair of
original anchors changes only their sparse supports. Four queries bound
A_st-ReLU(Hrest); they do not restrict the source box by phases. The actual
anchor bits multiply the midpoint affine forms, with D066's unchanged uniform
parameter error. Reference midpoint phases are never introduced.

The original Prepared, two support bases and eight query results per table
are retained proof premises, not free caches. Persistent query paths and
source registries must be counted by a future complete physical ledger.
Whole-vector decode is deliberately not called here. Input/intermediate
Fractions retain the inherited 512-bit cap. Immediate sparse occurrences and
merged containers are preflighted at 65536, including unused/fixed source
declarations; these limits are NOT a whole-model work or memory qualification.

The private seals are trusted-construction markers, not a security boundary
against deliberate mutation/forgery of private objects. The reused D066
forward/terminal functions consume the original Table type and original seal;
no old globals or registries are changed. Reversed state endpoints remain
conditional empty-state facts, never a reason to delete or fix original bits.
"""

from dataclasses import dataclass
from fractions import Fraction

from experiments.neural_hz_20260831.definition_first_20260928.d066_wide_phase_interface_20261001 import phase_interface as pi
from experiments.neural_hz_20260831.definition_first_20260928.d070_joint_negative_component_20261001 import hinge_support as hs


ss, ms, mp = pi.ss, pi.ms, pi.mp
KernelError = pi.KernelError
MAX_BITS, MAX_SUPPORT = pi.MAX_BITS, pi.MAX_SUPPORT
ZERO, ONE = Fraction(0), Fraction(1)
_f, _add, _sub, _mul, _neg = pi._f, pi._add, pi._sub, pi._mul, pi._neg
affine, relu, compile_rows = pi.affine, pi.relu, pi.compile_rows
Table, Overlap, Compiled = pi.Table, pi.Overlap, pi.Compiled
_SEAL = object()


@dataclass(frozen=True)
class Prepared:
    original: pi.Prepared
    upper_base: object
    lower_base: object
    _seal: object

    @property
    def context(self):
        return self.original.context

    @property
    def v(self):
        return self.original.v

    @property
    def gates(self):
        return self.original.gates

    @property
    def receiver(self):
        return self.original.receiver

    @property
    def error(self):
        return self.original.error


def _bounded(count):
    if count > MAX_SUPPORT:
        raise KernelError('aggregate joint-negative support limit exceeded')


def _terms(coefficients):
    """Encode one bounded, already typed map without changing source identity."""
    _bounded(len(coefficients))
    # D066 maps use unique owned Source objects; their ordinals were checked
    # before preparation. Sorting here is an explicitly paid preparation or
    # anchor-sparse operation, never a complete-box copy inside a query.
    return tuple((source.ordinal, coefficient)
                 for source, coefficient in sorted(
                     coefficients.items(), key=lambda item: item[0].ordinal)
                 if coefficient != ZERO)


def _support_base(bounds, bases):
    first, second = bases                  # B and B-H from the same direction.
    _bounded(len(bounds) + len(first.coefficients) + len(second.coefficients))
    h_coefficients = {}
    for source, coefficient in first.coefficients.items():
        h_coefficients[source] = _f(coefficient)
    for source, coefficient in second.coefficients.items():
        updated = _sub(h_coefficients.get(source, ZERO), coefficient)
        if updated == ZERO:
            h_coefficients.pop(source, None)
        else:
            h_coefficients[source] = updated
    _bounded(len(bounds) + len(first.coefficients) + len(h_coefficients))
    h_bias = _sub(first.bias, second.bias)
    return hs.prepare(bounds, first.bias, _terms(first.coefficients),
                      h_bias, _terms(h_coefficients), enabled=True)


def prepare(v, gates, receiver, *, enabled=False):
    """Prepare the same certified F and actual-bit error as D066, default off.

    gates is an immutable tuple of (ss affine, original output, original own
    phase, weight interval), in strictly increasing original phase ordinals.
    No original slot, zero-weight gate, unused source or fixed source is
    discarded. Both hinge bases contain the full declared source box.
    """
    if not ms._on(enabled):
        return None
    original = pi.prepare(v, gates, receiver, enabled=True)
    _bounded(len(original.context.sources))
    bounds = tuple((source.ordinal, source.lower, source.upper)
                   for source in original.context.sources)
    upper = _support_base(bounds, original._upper_bases)
    lower = _support_base(bounds, original._lower_bases)
    return Prepared(original, upper, lower, _SEAL)


def _prepared(value):
    if type(value) is not Prepared or value._seal is not _SEAL:
        raise KernelError('an owned joint-negative prepared interface is required')
    pi._prepared(value.original)
    return value


def _accumulate(bias, coefficients, point, scale):
    """Add a checked reference point on its original sparse source support."""
    _bounded(len(coefficients) + len(point.terms))
    bias = _add(bias, _mul(scale, point.bias))
    for source, coefficient in point.terms:
        updated = _add(coefficients.get(source, ZERO), _mul(scale, coefficient))
        if updated == ZERO:
            coefficients.pop(source, None)
        else:
            coefficients[source] = updated
    return bias


def _direction(prepared, i, j, sign, base):
    original = prepared.original
    points = (original.midpoint_forms[i], original.midpoint_forms[j])
    majorants = (original.majorants[i], original.majorants[j])
    weights = (_mul(sign, original.midpoint_weights[i]),
               _mul(sign, original.midpoint_weights[j]))
    _bounded(sum(len(point.terms) for point in points + majorants))

    a_bias, a_coefficients = ZERO, {}
    h_bias, h_coefficients = ZERO, {}
    for point, majorant, weight in zip(points, majorants, weights):
        if weight > ZERO:
            a_bias = _accumulate(a_bias, a_coefficients, majorant, _neg(weight))
        elif weight < ZERO:
            # Hrest=H-|weight|*g. The signed negative weight is -|weight|.
            h_bias = _accumulate(h_bias, h_coefficients, point, weight)
    _bounded(len(a_coefficients) + len(h_coefficients))
    h_updates = _terms(h_coefficients)

    results = []
    # This is the frozen four-entry interface order, not four restricted
    # source domains or phase subnetwork executions: 00, 10, 01, 11.
    for state_i, state_j in ((0, 0), (1, 0), (0, 1), (1, 1)):
        _bounded(len(a_coefficients) + len(h_coefficients)
                 + state_i * len(points[0].terms)
                 + state_j * len(points[1].terms))
        updates = dict(a_coefficients)     # Anchor-sparse only; not the base.
        bias = a_bias
        if state_i:
            bias = _accumulate(bias, updates, points[0], weights[0])
        if state_j:
            bias = _accumulate(bias, updates, points[1], weights[1])
        _bounded(len(updates) + len(h_updates))
        results.append(hs.query(base, a_bias_delta=bias,
                                a_updates=_terms(updates),
                                h_bias_delta=h_bias,
                                h_updates=h_updates, enabled=True))
    return tuple(results)


def condition(prepared, i, j, *, enabled=False):
    """Seed one fixed pair of original gate slots with eight whole-box queries.

    The population of pairs is a caller's preregistered structural choice.
    This operation neither searches for a pair nor consults old condition,
    an LP state, model identity, or a fallback support/menu.
    """
    if not ms._on(enabled):
        return None
    prepared = _prepared(prepared)
    original = prepared.original
    if (type(i) is not int or type(j) is not int
            or not 0 <= i < j < len(original.gates)):
        raise KernelError('indices must be ordered distinct original gate slots')
    alpha = mp._phase(original.gates[i][2], original.context.frame)
    beta = mp._phase(original.gates[j][2], original.context.frame)
    if alpha is beta or alpha.ordinal >= beta.ordinal:
        raise KernelError('two ordered distinct original phases are required')

    upper_results = _direction(prepared, i, j, ONE, prepared.upper_base)
    lower_results = _direction(prepared, i, j, _neg(ONE), prepared.lower_base)
    upper = tuple(_add(_f(result.value), original.error)
                  for result in upper_results)
    lower = tuple(_sub(_neg(_f(result.value)), original.error)
                  for result in lower_results)
    return pi.Table(original.context, alpha, beta, original.receiver,
                    lower, upper,
                    ('whole-box joint-negative seed', prepared, i, j,
                     upper_results, lower_results), pi._SEAL)
