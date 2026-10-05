"""Charged, default-off plain bridge to the frozen D053/D056 mathematics.

Every canonical receiver coefficient and PRE-ReLU source bound is validated,
including stable channels and literal padding.  Ordinary receiver bounds use
the inherited D025 receiver_interval on clipped activation intervals.  All
remaining three-channel combinations, including zero-touching channels and
no-odd-cycle results, are retained.  Within ONE window, every directed edge of
those channels is constructed exactly once by D053; no external certificate,
cross-window cache, solver, native column, or new original bit is accepted.

The caller certifies the original model/source/phase/decoder identities.  Local
tokens only prevent accidental mixing.  Typed objects never escape this call;
all pair premises and rows are serialized, and triangles reference the complete
window-local edge population.  Stable original bits remain in the caller's HZ.

Unmetered frozen calls are prepaid in inherited scalar-work units.  If T is the
sum of the two actual input supports, one D053 call reserves
4096 + 768*T + 32*T*ceil(log2(T+2)).  The linear allowance covers its two full
form validations, midpoint/normalization, P/N construction, three coefficient-
first supports, both row constructions and their repeated identity checks;
the sorting allowance covers all source/row canonicalizations.  A triangle
with H total P-support reserves 2048 + 384*H + 24*H*ceil(log2(H+2)), covering
the unchanged D056 planes, P/readout/phase merging and all <=4 returned rows.
Initial token/form construction and the once-per-gate D056 bound check have
their own prepaid bounds below.  These conservative charges count bounded
scalar operations, visits and sorting comparisons, not bit-level operations
or elapsed time.  They do not enlarge the inherited 256M/512-bit limits.

The transient_entries bound includes the typed frame, all open-gate forms,
the entire typed edge cache, and one in-flight construction/serialization.
It allows three numeric entries per Fraction (value and numerator/denominator),
48*open*width for forms, 192*edges*(width+8) for pair forms/rows and references,
512*width for source/one-call temporaries, and receiver/constant reserves.
Inputs and the full returned plain evidence must ALSO be ledgered by the
worker; Python headers/strings are covered by its physical process gates.
This module makes no native, complete-memory, GPU or verification claim.
The three registries of this call's private formal frame are cleared in a
finally clause, after serialization or on failure.  This breaks its token
reference cycles, with cleanup work reserved before frame construction.  No
caller frame or original HZ bit is removed; only ephemeral local wrappers die.
"""

from fractions import Fraction
from itertools import combinations

from experiments.neural_hz_20260831.definition_first_20260928.d025_interval_capacity_20260930 import interval_capacity as arithmetic
from experiments.neural_hz_20260831.definition_first_20260928.d025_interval_capacity_20260930.census import receiver_interval
from experiments.neural_hz_20260831.definition_first_20260928.d056_phase_triangle_component_20260930 import phase_triangle as pt

k = arithmetic.kernel
ss, ms, mp = pt.ss, pt.ms, pt.mp
WorkBudget = arithmetic.WorkBudget
KernelError = arithmetic.KernelError
BudgetExceeded = arithmetic.BudgetExceeded
KernelDisabled = k.KernelDisabled
MAX_SLOTS = arithmetic.MAX_SLOTS
ZERO, ONE, TWO = Fraction(0), Fraction(1), Fraction(2)


def _choose(n, degree):
    if n < degree:
        return 0
    return n * (n - 1) // 2 if degree == 2 else n * (n - 1) * (n - 2) // 6


def transient_entries(width, receiver_count, open_count):
    """Conservative metadata bound; no candidate or model operation is run."""
    if any(type(n) is not int or not 0 <= n <= MAX_SLOTS
           for n in (width, receiver_count, open_count)) or open_count > receiver_count:
        raise KernelError('invalid transient-bound dimensions')
    edges = _choose(open_count, 2) if open_count >= 3 else 0
    return (65_536 + 512 * width + 48 * open_count * width
            + 192 * edges * (width + 8) + 256 * receiver_count)


def _prepay(budget, support, base, linear, sorting):
    budget.charge(base + linear * support
                  + sorting * support * (support + 1).bit_length())


def _ordinal(value, budget):
    budget.charge(5)
    if type(value) is not int or value < 0 or value.bit_length() > 512:
        raise KernelError('an original nonnegative integer ordinal is required')
    return value


def _plain_affine(value, budget):
    budget.charge(24 + 12 * len(value.terms))
    return {'bias': value.bias,
            'terms': tuple((source.ordinal, lo, hi) for source, lo, hi in value.terms)}


def _plain_exact(value, budget):
    budget.charge(24 + 10 * len(value.source_terms) + 8 * len(value.phase_terms))
    if value.phase_terms:
        raise KernelError('P/N certificates must have no phase terms')
    return {'bias': value.bias,
            'source_terms': tuple((source.ordinal, a) for source, a in value.source_terms),
            'phase_terms': ()}


def _plain_rows(rows, roles, phase_roles, budget):
    budget.charge(8 + 8 * len(rows))
    result, nnz = [], 0
    for values, phases, rhs in rows:
        budget.charge(16 + 10 * (len(values) + len(phases)))
        if any(token not in roles for token, _ in values) or any(token not in phase_roles for token, _ in phases):
            raise KernelError('returned row lost an original identity')
        result.append((tuple((roles[token], a) for token, a in values),
                       tuple((phase_roles[token], a) for token, a in phases), rhs))
        nnz += len(values) + len(phases)
    return tuple(result), nnz


def _plain_pair(pair, channels, output_ordinals, roles, phase_roles, budget):
    budget.charge(48)
    rows, nnz = _plain_rows(pair.rows, roles, phase_roles, budget)
    return {'channels': channels,
            'outputs': tuple(output_ordinals[i] for i in channels),
            'f': _plain_affine(pair.f, budget), 'g': _plain_affine(pair.g, budget),
            'P': _plain_exact(pair.P, budget), 'N': _plain_exact(pair.N, budget),
            'kappa0': pair.kappa0, 'kappa1': pair.kappa1,
            'upper': pair.upper, 'delta': pair.delta, 'rows': rows}, nnz


def _compose(forms, outputs, phases, pairs, budget):
    """The D056 row formula, with internally owned already-validated edges.

    Original ordinal ordering, full triangle input/support caps, _planes,
    sparse P/read/A merging and bias signs are the frozen D056 operations.
    All gate bounds were checked once against ordinary bounds before edges.
    """
    budget.charge(32)
    if 2 * sum(len(value.terms) for value in forms) + 12 > pt.MAX_SUPPORT:
        raise KernelError('aggregate triangle input support exceeds limit')
    support = sum(len(pair.P.source_terms) for pair in pairs)
    _prepay(budget, support, 2048, 384, 24)
    planes = pt._planes(phases, tuple(pair.delta for pair in pairs))
    if not planes:
        return (), 'no_odd_cycle', ZERO
    if support + 12 > pt.MAX_SUPPORT:
        raise KernelError('aggregate triangle certificate support exceeds limit')
    context = forms[0].context
    P = ms._linear(context, tuple((ONE, pair.P) for pair in pairs))
    read = mp._make_form(context.frame, 'value', ZERO, tuple((q, TWO) for q in outputs))
    values = ms._row_values(read, P.source_terms, ONE, -ONE)
    linear = tuple(term for pair in pairs
                   for term in ((pair.alpha, pair.kappa0), (pair.beta, pair.upper)))
    if len(planes) * (len(values) + 3) > pt.MAX_SUPPORT:
        raise KernelError('aggregate returned triangle support exceeds limit')
    rows = []
    for plane in planes:
        terms = mp._make_form(context.frame, 'phase', ZERO, linear + plane.terms).terms
        rows.append((values, tuple((phase, ss._neg(a)) for phase, a in terms),
                     ss._add(P.bias, plane.bias)))
    # This is a cube-envelope diagnostic, NOT a feasible physical gain.
    budget.charge(16)
    tau = min(arithmetic._neg(pair.delta, budget) if pair.delta < ZERO else pair.delta
              for pair in pairs)
    budget.charge(2)
    gap = arithmetic._fraction(tau / TWO, budget.max_bits)
    return tuple(rows), 'odd_cycle', gap


def compile_window(receiver_weights, receiver_biases, source_bounds, source_ordinals,
                   output_ordinals, *, enabled=False, budget=None):
    """Compile a full canonical second-bank window without a native binding.

    ``source_bounds`` are FIRST-bank preactivation intervals; real ordinals
    identify distinct original source activations.  ``None`` is literal
    padding and requires exactly (0,0).  Output ordinals identify distinct
    original second-bank gates.  All returned objects are plain builtins or
    Fraction.  A rejected window is neither SAFE nor a partial success.
    """
    if type(enabled) is not bool:
        raise KernelError('enabled must be a bool')
    if not enabled:
        return None
    budget = arithmetic._active_budget(budget)
    start = budget.used
    budget.charge(48)
    if budget.max_bits != 512:
        raise KernelError('the frozen D053/D056 bridge requires the inherited 512-bit setting')
    if any(type(items) is not tuple for items in
           (receiver_weights, receiver_biases, source_bounds, source_ordinals, output_ordinals)):
        raise KernelError('immutable canonical tuples required')
    width, receivers = len(source_bounds), len(receiver_weights)
    if width > MAX_SLOTS or receivers > MAX_SLOTS:
        raise KernelError('window dimension exceeds limit')
    if len(source_ordinals) != width or len(receiver_biases) != receivers or len(output_ordinals) != receivers:
        raise KernelError('window shape mismatch')
    # Includes population visits, ID sets, activation containers and all row
    # lengths BEFORE any large canonical scan or coefficient container.
    budget.charge(48 + 20 * width + 16 * receivers + 16 * width * receivers)
    seen_sources, seen_outputs, activations = set(), set(), []
    real = 0
    for bound, ordinal in zip(source_bounds, source_ordinals):
        checked = arithmetic._interval(bound, budget)
        if ordinal is None:
            if checked != (ZERO, ZERO):
                raise KernelError('padding requires a literal-zero preactivation bound')
        else:
            _ordinal(ordinal, budget)
            if ordinal in seen_sources:
                raise KernelError('duplicate original source ordinal')
            seen_sources.add(ordinal)
            real += 1
        activations.append(k.nonnegative_part(checked, budget))
    activation_bounds = tuple(activations)
    ordinary, states = [], []
    for weights, bias, ordinal in zip(receiver_weights, receiver_biases, output_ordinals):
        if type(weights) is not tuple or len(weights) != width:
            raise KernelError('receiver canonical width mismatch')
        _ordinal(ordinal, budget)
        if ordinal in seen_outputs:
            raise KernelError('duplicate original output ordinal')
        seen_outputs.add(ordinal)
        arithmetic._interval(bias, budget)
        for coefficient in weights:
            arithmetic._interval(coefficient, budget)
        bound = receiver_interval(weights, bias, activation_bounds, k, budget)
        ordinary.append(bound)
        states.append('A' if bound[0] > ZERO else ('I' if bound[1] < ZERO else 'O'))
    budget.charge(64 + 8 * receivers + 8 * receivers * (receivers + 1).bit_length())
    ordinary, states = tuple(ordinary), tuple(states)
    open_channels = tuple(sorted((i for i, state in enumerate(states) if state == 'O'),
                                 key=lambda i: output_ordinals[i]))
    opened = len(open_channels)
    total, candidates = _choose(receivers, 3), _choose(opened, 3)
    edge_count = _choose(opened, 2) if candidates else 0
    reserve = transient_entries(width, receivers, opened)
    typed_forms, typed_outputs, typed_phases = {}, {}, {}
    roles, phase_roles = {}, {}
    typed_pairs, plain_pairs, pair_index, triangles = [], [], {}, []
    row_count = nnz = odd = no_odd = pair_row_count = pair_nnz = 0
    frame = None
    try:
        if candidates:
            if real + opened > MAX_SLOTS:
                raise KernelError('window formal value registry exceeds limit')
            # Covers frame/value/phase/source registry creation, context sorting
            # and source-role maps.  No first-bank phase tokens are invented.
            _prepay(budget, real, 4096 + 256 * opened, 256, 16)
            # Prepay three clears and all key/value visits before allocation.
            # A failed numeric operation cannot prevent cycle cleanup.
            budget.charge(32 + 4 * (real + 3 * opened))
            frame = ms.make_frame('D057 original window', enabled=True)
            entries, slot_values = [], [None] * width
            for slot, ordinal in enumerate(source_ordinals):
                if ordinal is not None:
                    value = ms.make_value(frame, 'original source activation', enabled=True)
                    slot_values[slot] = value
                    roles[value] = ('source', ordinal)
                    lo, hi = activation_bounds[slot]
                    entries.append((value, ordinal, lo, hi))
            context = ms.source_context(frame, tuple(entries), enabled=True)
            budget.charge(24 + 8 * width + 8 * opened)
            slot_sources = tuple(None if value is None else context._by_value[value] for value in slot_values)
            for channel in open_channels:
                value = ms.make_value(frame, 'original receiver output', enabled=True)
                phase = ms.make_phase(value, output_ordinals[channel], 'original receiver phase', enabled=True)
                roles[value], phase_roles[phase] = ('output', output_ordinals[channel]), ('phase', output_ordinals[channel])
                budget.charge(24 + 12 * width)
                terms = tuple((source, lo, hi) for source, (lo, hi) in
                              zip(slot_sources, receiver_weights[channel]) if source is not None)
                _prepay(budget, len(terms), 1024, 192, 8)
                form = ss.affine(context, receiver_biases[channel], terms, enabled=True)
                budget.charge(64 + 64 * len(form.terms))
                if pt._bounds(form) != ordinary[channel]:
                    raise KernelError('D056 source bounds differ from full ordinary bounds')
                typed_forms[channel], typed_outputs[channel], typed_phases[channel] = form, value, phase
            budget.charge(32 + 24 * edge_count)
            for i, j in combinations(open_channels, 2):
                support = len(typed_forms[i].terms) + len(typed_forms[j].terms)
                _prepay(budget, support, 4096, 768, 32)
                pair = ss.generate(typed_forms[i], typed_forms[j], typed_outputs[i], typed_outputs[j],
                                   typed_phases[i], typed_phases[j], enabled=True)
                pair_index[(i, j)] = len(typed_pairs)
                typed_pairs.append(pair)
                plain, count = _plain_pair(pair, (i, j), output_ordinals, roles, phase_roles, budget)
                plain_pairs.append(plain)
                pair_row_count += len(pair.rows)
                pair_nnz += count
            budget.charge(32 + 48 * candidates)
            for channels in combinations(open_channels, 3):
                i, j, h = channels
                indices = (pair_index[(i, j)], pair_index[(i, h)], pair_index[(j, h)])
                pairs = tuple(typed_pairs[index] for index in indices)
                forms = tuple(typed_forms[c] for c in channels)
                outputs = tuple(typed_outputs[c] for c in channels)
                phases = tuple(typed_phases[c] for c in channels)
                generated, status, gap = _compose(forms, outputs, phases, pairs, budget)
                plain_rows, count = _plain_rows(generated, roles, phase_roles, budget)
                budget.charge(48)
                triangles.append({'channels': channels,
                    'outputs': tuple(output_ordinals[c] for c in channels),
                    'bounds': tuple(ordinary[c] for c in channels),
                    'pair_indices': indices, 'status': status, 'rows': plain_rows,
                    'row_count': len(plain_rows), 'nnz': count, 'cube_midpoint_gap': gap})
                row_count += len(plain_rows)
                nnz += count
                odd += status == 'odd_cycle'
                no_odd += status == 'no_odd_cycle'
    except pt.KernelError as exc:
        raise KernelError('frozen triangle premise rejected: ' + str(exc)) from exc
    finally:
        if frame is not None:
            frame._source_phases.clear()
            frame._phases.clear()
            frame._values.clear()
    budget.charge(96 + 3 * len(plain_pairs) + 3 * len(triangles))
    if len(plain_pairs) != edge_count or len(triangles) != candidates or odd + no_odd != candidates:
        raise KernelError('incomplete window population')
    return {'schema': 'd057_window_v1', 'canonical_slots': width, 'real_slots': real,
        'padding_slots': width - real, 'receiver_count': receivers,
        'receiver_canonical_slots': receivers * width, 'source_ordinals': source_ordinals,
        'source_bounds': source_bounds, 'activation_bounds': activation_bounds,
        'output_ordinals': output_ordinals, 'ordinary_bounds': ordinary,
        'channel_states': states, 'open_channels': open_channels,
        'total_triangles': total, 'stable_skipped_triangles': total - candidates,
        'candidate_triangles': candidates, 'pairs': tuple(plain_pairs),
        'triangles': tuple(triangles), 'pair_proof_count': len(plain_pairs),
        'pair_row_count': pair_row_count, 'pair_nnz': pair_nnz,
        'odd_triangles': odd, 'no_odd_triangles': no_odd,
        'row_count': row_count, 'nnz': nnz, 'work_used': budget.used - start,
        'transient_numeric_entries': reserve, 'original_bits_deleted': 0}
