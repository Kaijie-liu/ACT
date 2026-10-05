"""Canonical sparse W/C interface for the frozen D092 relay mathematics.

Every matrix row is a tuple of (logical_index, nonzero Fraction), strictly
increasing and duplicate-free. Empty rows mean zero; they do not remove an
observation or next gate. No dense W or C is constructed. The complete child
bank and all original graph predicates are still authenticated and retained.

This changes matrix traversal, not ideal rows, common-realization semantics,
normalization, or outward rounding. All original gates already belong to H0.
Each original integer state retains one simultaneous canonical extension;
arbitrary encoded auxiliary points need not be exact products. No original
phase, source coordinate, EQ/LE, output readout, frame, or exact flag is changed.

Sparse modes and all parameter/index occurrences are checked before native
authentication or large coefficient maps. Coefficient entries are charged as
2*K+nnz(W)+nnz(C); sparse indices add another nnz(W)+nnz(C) occurrences. The
65536 local input/combination/generated-row guards and 512-bit arithmetic
remain. These are not whole-work, peak-memory, model, or GPU qualification.
"""

from fractions import Fraction

import numpy as np
import scipy.sparse as sp

from experiments.neural_hz_20260831.definition_first_20260928.d086_native_shared_transfer_20261001 import native_shared_transfer as nt
from experiments.neural_hz_20260831.definition_first_20260928.d092_native_observation_relay_20261001.native_observation_relay import RelayResult


nb = nt.nb
NativeAffine, Graph, KernelError = nt.NativeAffine, nt.Graph, nt.KernelError
MAX_BITS, MAX_SUPPORT = nt.MAX_BITS, nt.MAX_SUPPORT
ZERO, ONE, HALF = Fraction(0), Fraction(1), Fraction(1, 2)
_f, _add, _sub, _mul, _neg = nb._f, nb._add, nb._sub, nb._mul, nb._neg
STATES = ((0, 0), (1, 0), (0, 1), (1, 1))


def _sparse_row(row, width):
    """Check supplied storage in place; do not normalize or allocate a map."""
    if type(row) is not tuple:
        raise KernelError('sparse matrix rows must be immutable tuples')
    previous = -1
    for item in row:
        if type(item) is not tuple or len(item) != 2:
            raise KernelError('sparse entries require index,Fraction')
        index, amount = item
        if type(index) is not int or not previous < index < width:
            raise KernelError('sparse indices must be sorted, distinct and in range')
        if _f(amount) == ZERO:
            raise KernelError('sparse coefficients must be nonzero Fractions')
        previous = index


def append_sparse_observation_relay(hz, parents, consumers, observations,
                                   next_consumers, *, enabled=False):
    """Construct the complete D092 relay using only explicit W/C entries.

    consumers is tuple((Graph,a,b),...), with a,b on normalized parent outputs.
    observations is a nonempty tuple of canonical sparse rows indexed into
    the COMPLETE child bank. next_consumers is tuple((Graph,C),...), with C
    a canonical sparse row indexed into the COMPLETE observation bank.

    The actual next preactivation determines its constant and full original
    residual. Returned RelayResult has the frozen D092 schema; exact_rows and
    rounding receipts start with D086 rows. An empty W/C row is allowed and
    handled by the same zero-scale/residual rules, never an identity shortcut.
    """
    if not nb.ms._on(enabled):
        return None
    if type(parents) is not tuple or len(parents) != 2:
        raise KernelError('exactly two immutable parent graphs required')
    if type(consumers) is not tuple or not consumers:
        raise KernelError('nonempty immutable consumer bank required')
    if type(observations) is not tuple or not observations:
        raise KernelError('nonempty immutable sparse observation matrix required')
    if type(next_consumers) is not tuple or not next_consumers:
        raise KernelError('nonempty immutable next consumer bank required')
    k, p, jn = len(consumers), len(observations), len(next_consumers)
    parameter_entries = nb._limit(2 * k)
    index_entries = 0
    input_occurrences = nb._limit(2 + k + p + jn + parameter_entries)

    # Finish ALL sparse/type/parameter validation before schema population
    # scans, graph extraction, lookup construction, or the D086 HZ copy.
    for item in consumers:
        if type(item) is not tuple or len(item) != 3:
            raise KernelError('consumer entries require Graph,a,b')
        _f(item[1]), _f(item[2])
    for row in observations:
        if type(row) is not tuple:
            raise KernelError('sparse matrix rows must be immutable tuples')
        size = len(row)
        parameter_entries = nb._limit(parameter_entries + size)
        index_entries = nb._limit(index_entries + size)
        input_occurrences = nb._limit(input_occurrences + 2 * size)
        _sparse_row(row, k)
    for item in next_consumers:
        if type(item) is not tuple or len(item) != 2 or type(item[1]) is not tuple:
            raise KernelError('next entries require Graph and a canonical sparse C row')
        size = len(item[1])
        parameter_entries = nb._limit(parameter_entries + size)
        index_entries = nb._limit(index_entries + size)
        input_occurrences = nb._limit(input_occurrences + 2 * size)
        _sparse_row(item[1], p)

    nc, nbin = nb._hz_schema(hz)
    graphs = [*parents, *(item[0] for item in consumers),
              *(item[0] for item in next_consumers)]
    for graph in graphs:
        input_occurrences = nb._limit(input_occurrences
                                     + nb._graph_shape(graph, hz, nc, nbin))
    if parents[0].slots[2] == parents[1].slots[2]:
        raise KernelError('parent original phase columns must be distinct')
    extracted = []
    for graph in graphs:
        g, q, error = nb._extract_graph(hz, graph)
        nb._checked(g, nc, nbin), nb._checked(q, nc, nbin)
        if error != ZERO:
            raise KernelError('every protected original gate must have zero graph error')
        extracted.append((g, q))

    combination_occurrences = 0

    def combine(*scaled, bias=ZERO):
        nonlocal combination_occurrences
        appearances = 1
        for scale, affine in scaled:
            _f(scale)
            appearances = nb._limit(appearances + 1 + nb._shape(affine))
        combination_occurrences = nb._limit(combination_occurrences + appearances)
        return nb._combine(_f(bias), tuple(scaled))

    zero = nt._constant(ZERO)
    u = nt._continuous(parents[0].slots[1], -HALF, HALF)
    v = nt._continuous(parents[1].slots[1], -HALF, HALF)
    children, child_bounds, slopes = [], [], []
    for (_, a, b), (g, q) in zip(consumers, extracted[2:2 + k]):
        remainder = combine((ONE, g), (_neg(a), u), (_neg(b), v))
        residual = NativeAffine(ZERO, remainder.continuous, remainder.binary)
        c, radius = remainder.bias, nb._norm(residual)
        bounds, lines = [], []
        for s, t in STATES:
            lower = _sub(_add(_add(c, _mul(min(a, ZERO), Fraction(s))),
                              _mul(min(b, ZERO), Fraction(t))), radius)
            upper = _add(_add(_add(c, _mul(max(a, ZERO), Fraction(s))),
                              _mul(max(b, ZERO), Fraction(t))), radius)
            bounds.append((lower, upper))
            lines.append(nt._secant(lower, upper))
        children.append((c, a, b, residual, radius, q))
        child_bounds.append(tuple(bounds))
        slopes.append(tuple(lines))

    core_occurrences = 0
    for row in observations:
        for child_index, _ in row:
            core_occurrences = nb._limit(core_occurrences
                                        + 4 * (1 + nb._shape(children[child_index][3])))

    base = nt.append_shared_upper(hz, parents, consumers, enabled=True)
    middle = base.hz
    base_nc, base_count = middle.n_cont, len(base.exact_rows)
    new_nc = base_nc
    alpha = nt._binary(parents[0].slots[2], -HALF, HALF)
    beta = nt._binary(parents[1].slots[2], -HALF, HALF)
    delta, zu, zv = tuple(nt._continuous(column, HALF, HALF)
                          for column in base.shared_columns)
    masses = (combine((-ONE, alpha), (-ONE, beta), (ONE, delta), bias=ONE),
              combine((ONE, alpha), (-ONE, delta)),
              combine((ONE, beta), (-ONE, delta)), delta)
    us = (zero, combine((ONE, u), (-ONE, zu)), zero, zu)
    vs = (zero, zero, combine((ONE, v), (-ONE, zv)), zv)
    residual_map = {index: (radius, cols)
                    for index, radius, cols in base.residual_bindings}
    cores = []
    for index, (c, a, b, residual, radius, _) in enumerate(children):
        if radius == ZERO:
            residuals = (zero, zero, zero, zero)
        else:
            recorded_radius, columns = residual_map[index]
            if recorded_radius != radius:
                raise KernelError('frozen base residual binding mismatch')
            first = tuple(nt._continuous(column, radius) for column in columns)
            residuals = (*first, combine((ONE, residual),
                                        *((-ONE, value) for value in first)))
        cores.append(tuple(combine((c, mass), (a, uc), (b, vc), (ONE, rc))
                           for mass, uc, vc, rc in zip(masses, us, vs, residuals)))

    rows = []
    row_occurrences = nb._limit(sum(len(row.continuous) + len(row.binary)
                                    for row, _ in base.exact_rows))

    def le(left, right):
        nonlocal row_occurrences
        difference = combine((ONE, left), (-ONE, right))
        row_occurrences = nb._limit(row_occurrences + nb._shape(difference))
        nb._limit(base_count + len(rows) + 1)
        rows.append((NativeAffine(ZERO, difference.continuous, difference.binary),
                     _neg(difference.bias)))

    observation_bindings, readouts, observed_slots = [], [], []
    slot_bounds, observation_ranges = [], []
    for oi, weights in enumerate(observations):
        readout = combine(*((weight, children[ci][5]) for ci, weight in weights))
        readouts.append(readout)
        bounds, envelopes = [], []
        for si, (s, t) in enumerate(STATES):
            ilo = ihi = dc = da = db = dminus = dplus = ZERO
            common, lifted = [], []
            for ci, weight in weights:
                c, a, b, residual, _, _ = children[ci]
                lower, upper = child_bounds[ci][si]
                qlo, qhi = nt._positive_part(lower), nt._positive_part(upper)
                ilo = _add(ilo, _mul(weight, qlo if weight >= ZERO else qhi))
                ihi = _add(ihi, _mul(weight, qhi if weight >= ZERO else qlo))
                slope, intercept = slopes[ci][si]
                amount = _mul(weight, slope)
                dc, da, db = (_add(dc, _mul(amount, c)),
                              _add(da, _mul(amount, a)),
                              _add(db, _mul(amount, b)))
                common.append((amount, residual))
                lifted.append((amount, cores[ci][si]))
                if weight < ZERO:
                    dminus = _add(dminus, _mul(weight, intercept))
                else:
                    dplus = _add(dplus, _mul(weight, intercept))
            common_radius = nb._norm(combine(*common))
            blo = _add(_sub(_add(_add(dc, _mul(min(da, ZERO), Fraction(s))),
                                _mul(min(db, ZERO), Fraction(t))), common_radius), dminus)
            bhi = _add(_add(_add(_add(dc, _mul(max(da, ZERO), Fraction(s))),
                                _mul(max(db, ZERO), Fraction(t))), common_radius), dplus)
            lower, upper = max(ilo, blo), min(ihi, bhi)
            if lower > upper:
                raise KernelError('inconsistent derived observation slot bounds')
            bounds.append((lower, upper))
            core = combine(*lifted)
            envelopes.append((combine((ONE, core), (dminus, masses[si])),
                              combine((ONE, core), (dplus, masses[si]))))
        radius = max(nb._absolute(endpoint) for bound in bounds for endpoint in bound)
        if radius == ZERO:
            values = (zero, zero, zero, zero)
        else:
            nb._limit(new_nc - nc + 3)
            columns = (new_nc, new_nc + 1, new_nc + 2)
            new_nc += 3
            first = tuple(nt._continuous(column, radius) for column in columns)
            values = (*first, combine((ONE, readout), *((-ONE, value) for value in first)))
            observation_bindings.append((oi, radius, columns))
        start = base_count + len(rows)
        for value, (lower, upper) in zip(values, envelopes):
            le(lower, value)
            le(value, upper)
        observation_ranges.append((start, base_count + len(rows)))
        observed_slots.append(values)
        slot_bounds.append(tuple(bounds))

    next_bindings, next_bounds, upper_indices = [], [], []
    for ni, ((_, coefficients), (g, q)) in enumerate(
            zip(next_consumers, extracted[2 + k:])):
        remainder = combine((ONE, g),
                            *((_neg(amount), readouts[ci]) for ci, amount in coefficients))
        c = remainder.bias
        residual = NativeAffine(ZERO, remainder.continuous, remainder.binary)
        radius = nb._norm(residual)
        if radius == ZERO:
            residuals = (zero, zero, zero, zero)
        else:
            nb._limit(new_nc - nc + 3)
            columns = (new_nc, new_nc + 1, new_nc + 2)
            new_nc += 3
            first = tuple(nt._continuous(column, radius) for column in columns)
            residuals = (*first, combine((ONE, residual), *((-ONE, value) for value in first)))
            next_bindings.append((ni, radius, columns))
            for mass, value in zip(masses, residuals):
                capacity = combine((radius, mass))
                le(value, capacity)
                le(combine((-ONE, value)), capacity)
        bounds, uppers = [], []
        for si, (mass, rc) in enumerate(zip(masses, residuals)):
            lower, upper = _sub(c, radius), _add(c, radius)
            for ci, amount in coefficients:
                lo, hi = slot_bounds[ci][si]
                lower = _add(lower, _mul(amount, lo if amount >= ZERO else hi))
                upper = _add(upper, _mul(amount, hi if amount >= ZERO else lo))
            slope, intercept = nt._secant(lower, upper)
            core = combine((c, mass), (ONE, rc),
                           *((amount, observed_slots[ci][si]) for ci, amount in coefficients))
            uppers.append(combine((slope, core), (intercept, mass)))
            bounds.append((lower, upper))
        upper_indices.append(base_count + len(rows))
        le(q, combine(*((ONE, value) for value in uppers)))
        next_bounds.append(tuple(bounds))

    expected_rows = 12 + 8 * base.extra_residual_count + k + 8 * p + 8 * len(next_bindings) + jn
    expected_columns = 3 + 3 * base.extra_residual_count + 3 * len(observation_bindings) + 3 * len(next_bindings)
    if base_count + len(rows) != expected_rows or new_nc - nc != expected_columns:
        raise KernelError('complete relay population accounting mismatch')
    ac, ab, rhs, errors, stored, row_bytes = nt._rounded_rows(rows, new_nc, nbin)
    extra = new_nc - base_nc

    def padded_copy(matrix):
        return sp.hstack((matrix.copy(), sp.csr_matrix((matrix.shape[0], extra))), format='csr')

    result = nb.SparseHZono(
        c=middle.c.copy(), Gc=padded_copy(middle.Gc), Gb=middle.Gb.copy(),
        Ac=padded_copy(middle.Ac), Ab=middle.Ab.copy(), b=middle.b.copy(),
        Auc=sp.vstack((padded_copy(middle.Auc), ac), format='csr'),
        Aub=sp.vstack((middle.Aub.copy(), ab), format='csr'),
        ub=np.concatenate((middle.ub.copy(), rhs)), frame_id=hz.frame_id, exact=hz.exact)
    input_bytes, input_nnz = nt._storage(hz)
    middle_bytes, middle_nnz = nt._storage(middle)
    output_bytes, output_nnz = nt._storage(result)
    physical = {'input': input_bytes, 'intermediate': middle_bytes, 'output': output_bytes,
                'input_plus_intermediate_plus_output': input_bytes + middle_bytes + output_bytes,
                'relay_generated_row_buffers': row_bytes,
                'base_generated_row_buffers': base.physical_bytes['generated_row_buffers']}
    nnz = {'input': input_nnz, 'intermediate': middle_nnz, 'output': output_nnz,
           'added': base.nnz['added'] + int(ac.nnz + ab.nnz),
           'relay_added': int(ac.nnz + ab.nnz), 'exact_added': row_occurrences}
    local_cost = {'selected_input_occurrences': input_occurrences,
                  'matrix_parameter_entries': parameter_entries,
                  'matrix_index_entries': index_entries,
                  'observation_core_input_occurrences': core_occurrences,
                  'new_combination_input_occurrences': combination_occurrences,
                  'graph_extractions_including_base': 4 + 2 * k + jn,
                  'generated_row_occurrences': row_occurrences,
                  'whole_work_qualified': False, 'complete_physical_qualified': False,
                  'native_model_qualified': False, 'gpu_qualified': False}
    return RelayResult(
        result, nc, base_nc, base_count, base.shared_columns, base.residual_bindings,
        tuple(observation_bindings), tuple(readouts), tuple(observed_slots), tuple(child_bounds),
        tuple(slot_bounds), tuple(next_bindings), tuple(next_bounds), tuple(observation_ranges),
        tuple(upper_indices), base.exact_rows + tuple(rows), base.row_errors + errors,
        base.installed_rhs + stored, base.extra_residual_count + len(next_bindings),
        physical, nnz, local_cost)
