"""Default-off static shared-original-phase triangle certificate compiler.

This adds sound LE consequences to the ORIGINAL nonconvex HZ; it neither
replaces that domain nor constructs native columns. The caller certifies the
real affine enclosures, common source box, original gate/phase binding and
decoder. All original variables, zero phases and predicates remain present.
The classical Boolean triangle inequalities are not a novelty claim.

One invocation handles three distinct original gates in structural phase
ordinal order, with the same fixed forward orientation for each pair. Odd
signed triangles yield at most four rows and no auxiliary variables. Even
signs/zero edges or a strictly stable gate yield no NEW triangle rows, not a
verification verdict. Strict stability is not installed into any native LP.

All rational inputs/intermediates retain the inherited 512-bit limit. Sparse
occurrences are bounded before combining. This is not a complete work, GPU,
physical-storage or network qualification. Repeated pair construction across
triangles is real cost; no unimplemented cache is assumed.
"""
from dataclasses import dataclass

from experiments.neural_hz_20260831.definition_first_20260928.d053_static_source_component_20260930 import static_source as ss

ms, mp = ss.ms, ss.mp
ZERO, ONE, TWO = ss.ZERO, ss.ONE, ss.TWO
KernelError, MAX_SUPPORT = ss.KernelError, ss.MAX_SUPPORT
_add, _sub, _mul, _neg = ss._add, ss._sub, ss._mul, ss._neg
EDGES = ((0, 1), (0, 2), (1, 2))


@dataclass(frozen=True)
class _Triangle:
    forms: tuple
    outputs: tuple
    phases: tuple
    bounds: tuple
    pairs: tuple
    rows: tuple
    status: str


def _bounds(value):
    lower, upper = value.bias
    for source, lo, hi in value.terms:
        endpoints = (_mul(lo, source.lower), _mul(lo, source.upper),
                     _mul(hi, source.lower), _mul(hi, source.upper))
        lower, upper = _add(lower, min(endpoints)), _add(upper, max(endpoints))
    return lower, upper


def _planes(phases, deltas):
    """Internal exact projection formula, given already checked tokens/data.

    The Cartesian expansion below is of <=2 algebraic min terms, NOT an input
    or phase split. Original bits remain untouched; no literal is flipped.
    """
    if any(d == ZERO for d in deltas):
        return ()
    negatives = sum(d < ZERO for d in deltas)
    if negatives not in (1, 3):
        return ()
    tau = min(_neg(d) if d < ZERO else d for d in deltas)
    if negatives == 1:
        negative_edge = EDGES[next(i for i, d in enumerate(deltas) if d < ZERO)]
        middle = next(i for i in range(3) if i not in negative_edge)
        base = mp._make_form(phases[0].frame, 'phase', ZERO, ((phases[middle], tau),))
    else:
        base = mp._make_form(phases[0].frame, 'phase', tau,
                             tuple((phase, _neg(tau)) for phase in phases))
    planes = (base,)
    for (i, j), delta in zip(EDGES, deltas):
        residual = _sub(delta, tau) if delta > ZERO else _add(delta, tau)
        if residual == ZERO:
            continue
        if residual > ZERO:
            choices = ((ZERO, ((phases[i], residual),)),
                       (ZERO, ((phases[j], residual),)))
        else:
            choices = ((ZERO, ()), (_neg(residual),
                       ((phases[i], residual), (phases[j], residual))))
        planes = tuple(mp._make_form(base.frame, 'phase', _add(plane.bias, bias),
                                     plane.terms + terms)
                       for plane in planes for bias, terms in choices)
    return planes


def generate(forms, outputs, phases, *, enabled=False):
    """Three existing common-source gates -> <=4 additional static LE rows.

    Returns a record also for structurally skipped triples. Does not accept
    externally constructed pair certificates: all premises are reconstructed
    by D053 from the supplied original affine enclosures. Those enclosures
    themselves remain the caller's model-binding obligation.
    """
    if not ms._on(enabled):
        return None
    if any(type(items) is not tuple or len(items) != 3 for items in (forms, outputs, phases)):
        raise KernelError('exactly three immutable original gate declarations required')
    # Each affine appears in two pair constructions. Reserve all six output
    # occurrences and six phase occurrences before constructing pair objects.
    if 2 * sum(ss._shape(value) for value in forms) + 12 > MAX_SUPPORT:
        raise KernelError('aggregate triangle input support exceeds limit')
    context = ss._checked(forms[0]).context
    for value in forms:
        ss._checked(value, context)
    for output, phase in zip(outputs, phases):
        mp._value(output, context.frame)
        mp._phase(phase, context.frame)
        if phase.original_output is not output:
            raise KernelError('phase must own this original output')
    if len(set(outputs)) != 3 or len(set(phases)) != 3:
        raise KernelError('three distinct original gates required')
    if any(source.original_value in outputs for value in forms for source, _, _ in value.terms):
        raise KernelError('common frontier cannot contain any target gate output')
    ordered = sorted(zip(forms, outputs, phases), key=lambda item: item[2].ordinal)
    forms, outputs, phases = tuple(zip(*ordered))
    bounds = tuple(_bounds(value) for value in forms)
    if any(lo > ZERO or hi < ZERO for lo, hi in bounds):
        return _Triangle(forms, outputs, phases, bounds, (), (), 'strict_stable')
    pairs = tuple(ss.generate(forms[i], forms[j], outputs[i], outputs[j],
                              phases[i], phases[j], enabled=True) for i, j in EDGES)
    planes = _planes(phases, tuple(pair.delta for pair in pairs))
    if not planes:
        return _Triangle(forms, outputs, phases, bounds, pairs, (), 'no_odd_cycle')
    count = sum(len(pair.P.source_terms) for pair in pairs) + 12
    if count > MAX_SUPPORT:
        raise KernelError('aggregate triangle certificate support exceeds limit')
    P = ms._linear(context, tuple((ONE, pair.P) for pair in pairs))
    read = mp._make_form(context.frame, 'value', ZERO, tuple((q, TWO) for q in outputs))
    values = ms._row_values(read, P.source_terms, ONE, -ONE)
    linear = tuple(term for pair in pairs
                   for term in ((pair.alpha, pair.kappa0), (pair.beta, pair.upper)))
    # Count the physical copies in the result, not just one sparse row.
    if len(planes) * (len(values) + 3) > MAX_SUPPORT:
        raise KernelError('aggregate returned triangle support exceeds limit')
    rows = []
    for plane in planes:
        terms = mp._make_form(context.frame, 'phase', ZERO, linear + plane.terms).terms
        rows.append((values, tuple((phase, _neg(a)) for phase, a in terms),
                     _add(P.bias, plane.bias)))
    return _Triangle(forms, outputs, phases, bounds, pairs, tuple(rows), 'odd_cycle')
