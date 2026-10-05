"""Certified D096 birth readouts feeding the unchanged D086/D093 relay.

The public boundary authenticates the complete native HZ schema and exactly
one BornGate for every parent, child and next-child role. Certificates may be
out of order, but their Graph set must match the complete role set. Duplicate
graphs, output eta coordinates and original phase columns are rejected.
Each certificate is verified against actual carrier/EQ/LE rows once. Only its
certified (g,q) pair enters the two private mathematical stages; raw graph
extraction, global patches and temporary predicate projection are not used.

The two fixed-one carriers are quotiented only in the authenticated readout.
Every original HZ row, factor prefix, output, binary, frame and exact flag is
copied unchanged. One simultaneous canonical extension protects original
integer states, not arbitrary previously rounded auxiliary assignments.
The result schema is the frozen D092 RelayResult. This is a single complete
structural component, not automatic discovery, model or GPU qualification.

All work is prepaid in caller-owned hard-pool batches: schema and certificate
checks, scalar slot work, actual combination occurrences and sorting, norms,
secants, exact-row rounding, and both native installations. The charge units
are conservative scalar/container-operation accounting, not bit complexity,
wall time or peak storage. Existing 512-bit and 65536 local guards remain.
The storage receipt counts native ndarray buffers, not all live Python roots.
"""

from fractions import Fraction

import numpy as np
import scipy.sparse as sp

from experiments.neural_hz_20260831.definition_first_20260928.d086_native_shared_transfer_20261001 import native_shared_transfer as nt
from experiments.neural_hz_20260831.definition_first_20260928.d092_native_observation_relay_20261001.native_observation_relay import RelayResult
from experiments.neural_hz_20260831.definition_first_20260928.d096_verified_phase_birth_20261001 import phase_birth as pb

nb = nt.nb
NativeAffine, Graph, KernelError = nt.NativeAffine, nt.Graph, nt.KernelError
MAX_BITS, MAX_SUPPORT = nt.MAX_BITS, nt.MAX_SUPPORT
ZERO, ONE, HALF = Fraction(0), Fraction(1), Fraction(1, 2)
_f, _add, _sub, _mul, _neg = nb._f, nb._add, nb._sub, nb._mul, nb._neg
_constant, _continuous, _binary = nt._constant, nt._continuous, nt._binary
_storage = nt._storage
STATES = ((0, 0), (1, 0), (0, 1), (1, 1))


class _Meter:
    """Prepay bounded stages; pass this same wrapper to pb.verify_gate.

    combine: S counts factors, biases and all unmerged source occurrences.
    128*(S+1)*(1+bit_length(S+1)) covers two validation/merge scans, checked
    scalar operations, dictionary updates and comparison sorting.
    norm: 64*(support+1); secant: 256. Outer scalar loops separately prepay
    1024 per child/observation/next row or explicit W/C term; each four-state
    base loop also prepays 256. These include argument/container preparation.
    Rounding prepays 256 per generated coefficient/RHS/row entry. Native
    copies and sparse stacks prepay 64 per complete old/new buffer occurrence
    plus row/column headers. These accounting bounds are deliberately local.
    """
    def __init__(self, pool):
        callback = getattr(pool, 'charge', None)
        if not callable(callback):
            raise KernelError('a caller-owned hard charging pool is required')
        self._callback, self.used = callback, 0

    def charge(self, name, amount):
        if (type(amount) is not int or amount < 0
                or amount.bit_length() > MAX_BITS
                or (self.used + amount).bit_length() > MAX_BITS):
            raise KernelError('invalid certified relay work charge')
        self._callback(name, amount)
        self.used += amount

    def combine(self, *scaled, bias=ZERO):
        self.charge('certified_combination_header', 64 * (len(scaled) + 1))
        appearances = 1
        for scale, affine in scaled:
            _f(scale)
            appearances = nb._limit(appearances + 1 + nb._shape(affine))
        self.charge('certified_combination_population',
                    128 * (appearances + 1) * (1 + (appearances + 1).bit_length()))
        return nb._combine(_f(bias), tuple(scaled))

    def norm(self, value):
        support = nb._shape(value)
        self.charge('certified_norm', 64 * (support + 1))
        return nb._norm(value)

    def secant(self, lower, upper):
        self.charge('certified_secant', 256)
        return nt._secant(lower, upper)

    def round_rows(self, rows, nc, nbin):
        self.charge('certified_round_row_headers', 128 * (len(rows) + 1))
        population = sum(nb._shape(row) + 2 for row, _ in rows)
        self.charge('certified_round_row_population', 256 * (population + 1))
        return nt._rounded_rows(rows, nc, nbin)

    def copy_install(self, hz, new_nc, rows):
        self.charge('certified_copy_headers', 128 * (len(rows) + 1))
        nc, nbin, population = pb._headers(hz)
        added = sum(nb._shape(row) for row, _ in rows)
        self.charge('certified_complete_copy_install',
                    64 * (population + nc + nbin + new_nc + added
                          + 10 * (len(rows) + 1) + 128))


def _sparse_row(row, width):
    """Caller prepays the complete supplied row; never normalize its identity."""
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


def _authenticate(hz, certificates, carriers, parents, consumers,
                  observations, next_consumers, meter):
    if type(parents) is not tuple or len(parents) != 2:
        raise KernelError('exactly two immutable parent graphs required')
    if type(consumers) is not tuple or not consumers:
        raise KernelError('nonempty immutable consumer bank required')
    if type(observations) is not tuple or not observations:
        raise KernelError('nonempty immutable sparse observation matrix required')
    if type(next_consumers) is not tuple or not next_consumers:
        raise KernelError('nonempty immutable next consumer bank required')
    if type(certificates) is not tuple:
        raise KernelError('an immutable complete birth-certificate tuple is required')
    k, p, jn = len(consumers), len(observations), len(next_consumers)
    if len(certificates) != 2 + k + jn:
        raise KernelError('birth certificates must exactly cover all graph roles')
    parameter_entries = nb._limit(2 * k)
    index_entries = 0
    input_occurrences = nb._limit(2 + k + p + jn + parameter_entries)
    meter.charge('certified_parameter_headers',
                 128 * (2 + k + p + jn + len(certificates)))
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
        meter.charge('certified_sparse_W', 128 * (size + 1))
        _sparse_row(row, k)
    for item in next_consumers:
        if type(item) is not tuple or len(item) != 2 or type(item[1]) is not tuple:
            raise KernelError('next entries require Graph and a canonical sparse C row')
        size = len(item[1])
        parameter_entries = nb._limit(parameter_entries + size)
        index_entries = nb._limit(index_entries + size)
        input_occurrences = nb._limit(input_occurrences + 2 * size)
        meter.charge('certified_sparse_C', 128 * (size + 1))
        _sparse_row(item[1], p)

    meter.charge('certified_native_headers', 128)
    nc, nbin, population = pb._headers(hz)
    meter.charge('certified_full_native_schema', 32 * (population + nc + nbin + 1))
    nb._hz_schema(hz)
    graphs = (*parents, *(item[0] for item in consumers),
              *(item[0] for item in next_consumers))
    role_graphs, role_phases, role_etas, footprints = set(), set(), set(), []
    for graph in graphs:
        meter.charge('certified_role_descriptor', 256)
        if (type(graph) is not Graph or type(graph.kind) is not str
                or graph.kind != 'extended'):
            raise KernelError('every role must use a D096 extended birth graph')
        footprint = nb._graph_shape(graph, hz, nc, nbin)
        input_occurrences = nb._limit(input_occurrences + footprint)
        phase, eta = graph.slots[2], graph.slots[1]
        if graph in role_graphs or phase in role_phases or eta in role_etas:
            raise KernelError('duplicate or ambiguous graph, phase or output eta role')
        role_graphs.add(graph)
        role_phases.add(phase)
        role_etas.add(eta)
        footprints.append(footprint)

    # Header/index checks precede any hashing of caller-supplied Graph fields.
    certificate_graphs, certificate_phases, certificate_etas = set(), set(), set()
    for certificate in certificates:
        meter.charge('certified_birth_descriptor', 256)
        if type(certificate) is not pb.BornGate:
            raise KernelError('only immutable D096 BornGate records are accepted')
        graph = certificate.graph
        if (type(graph) is not Graph or type(graph.kind) is not str
                or graph.kind != 'extended'):
            raise KernelError('birth certificate requires an extended graph')
        nb._graph_shape(graph, hz, nc, nbin)
        phase, eta = graph.slots[2], graph.slots[1]
        if (graph in certificate_graphs or phase in certificate_phases
                or eta in certificate_etas):
            raise KernelError('duplicate or ambiguous birth certificate')
        certificate_graphs.add(graph)
        certificate_phases.add(phase)
        certificate_etas.add(eta)
    if certificate_graphs != role_graphs:
        raise KernelError('missing or extra birth graph certificate')
    meter.charge('certified_lookup_and_order', 128 * (len(certificates) + 1))
    lookup = {}
    for certificate in certificates:
        lookup[certificate.graph] = pb.verify_gate(
            hz, certificate, carriers, pool=meter, enabled=True)
    extracted = tuple(lookup[graph] for graph in graphs)
    return extracted, tuple(footprints), (
        input_occurrences, parameter_entries, index_entries, len(certificates))


def append_certified_observation_relay(hz, certificates, carriers, parents,
                                      consumers, observations, next_consumers,
                                      *, pool, enabled=False):
    """Append the sparse relay after one complete, actual birth-row binding.

    W/C are D093 canonical sparse rows. Empty rows retain zero observations
    or complete next-gate residuals. Certificates may be reordered, never
    omitted, repeated or supplemented. All original graph roles are distinct.
    Disabled calls inspect only the strict-bool enabled argument.
    """
    if not nb.ms._on(enabled):
        return None
    meter = _Meter(pool)
    meter.charge('certified_public_metadata', 128)
    extracted, footprints, metadata = _authenticate(
        hz, certificates, carriers, parents, consumers, observations, next_consumers, meter)
    return _relay(hz, parents, consumers, observations, next_consumers,
                  extracted, footprints, metadata, meter)


def _append_certified_base(hz, parents, consumers, extracted, footprints, meter):
    """Private D086 body; never re-extract a carrier-augmented raw graph."""
    nc, nbin = hz.n_cont, hz.n_bin
    meter.charge('certified_base_headers', 2048 + 128 * (len(consumers) + 1))
    # Preserve the original base selected-occurrence guard as well as the
    # stricter public complete-role guard.
    count = 2 + len(consumers)
    for footprint in footprints:
        count = nb._limit(count + footprint)
    u = _continuous(parents[0].slots[1], -HALF, HALF)
    v = _continuous(parents[1].slots[1], -HALF, HALF)
    alpha = _binary(parents[0].slots[2], -HALF, HALF)
    beta = _binary(parents[1].slots[2], -HALF, HALF)
    shared = (nc, nc + 1, nc + 2)
    delta, zu, zv = tuple(_continuous(column, HALF, HALF) for column in shared)
    new_nc = nc + 3
    rows, row_occurrences, residual_bindings = [], 0, []

    def le(left, right):
        nonlocal row_occurrences
        difference = meter.combine((ONE, left), (-ONE, right))
        row_occurrences = nb._limit(row_occurrences + len(difference.continuous)
                                   + len(difference.binary))
        rows.append((NativeAffine(ZERO, difference.continuous, difference.binary),
                     _neg(difference.bias)))
        nb._limit(len(rows))

    zero = _constant(ZERO)
    le(zero, delta)
    le(delta, alpha)
    le(delta, beta)
    le(meter.combine((ONE, alpha), (ONE, beta), bias=-ONE), delta)
    for amplitude, split, phase in ((u, zu, alpha), (v, zv, beta)):
        le(zero, split)
        le(split, delta)
        rest = meter.combine((ONE, amplitude), (-ONE, split))
        le(zero, rest)
        le(rest, meter.combine((ONE, phase), (-ONE, delta)))
    masses = (meter.combine((-ONE, alpha), (-ONE, beta), (ONE, delta), bias=ONE),
              meter.combine((ONE, alpha), (-ONE, delta)),
              meter.combine((ONE, beta), (-ONE, delta)), delta)
    us = (zero, meter.combine((ONE, u), (-ONE, zu)), zero, zu)
    vs = (zero, zero, meter.combine((ONE, v), (-ONE, zv)), zv)
    states = ((0, 0), (1, 0), (0, 1), (1, 1))

    for consumer_index, ((_, a, b), (g, q)) in enumerate(zip(consumers, extracted[2:])):
        meter.charge('certified_base_consumer_scalars', 1024)
        remainder = meter.combine((ONE, g), (_neg(a), u), (_neg(b), v))
        c = remainder.bias
        residual = NativeAffine(ZERO, remainder.continuous, remainder.binary)
        radius = meter.norm(residual)
        if radius == ZERO:
            residuals = (zero, zero, zero, zero)
        else:
            nb._limit(new_nc - nc + 3)
            columns = (new_nc, new_nc + 1, new_nc + 2)
            new_nc += 3
            first = tuple(_continuous(column, radius) for column in columns)
            last = meter.combine((ONE, residual), *((-ONE, item) for item in first))
            residuals = (*first, last)
            residual_bindings.append((consumer_index, radius, columns))
            for mass, value in zip(masses, residuals):
                capacity = meter.combine((radius, mass))
                le(value, capacity)
                le(meter.combine((-ONE, value)), capacity)
        lifted = []
        for (s, t), mass, uc, vc, rc in zip(states, masses, us, vs, residuals):
            meter.charge('certified_base_slot_scalars', 256)
            lower = _sub(_add(_add(c, _mul(min(a, ZERO), Fraction(s))),
                              _mul(min(b, ZERO), Fraction(t))), radius)
            upper = _add(_add(_add(c, _mul(max(a, ZERO), Fraction(s))),
                              _mul(max(b, ZERO), Fraction(t))), radius)
            slope, intercept = meter.secant(lower, upper)
            preactivation = meter.combine((c, mass), (a, uc), (b, vc), (ONE, rc))
            lifted.append(meter.combine((slope, preactivation), (intercept, mass)))
        le(q, meter.combine(*((ONE, item) for item in lifted)))

    ac, ab, rhs, errors, stored_rhs, row_bytes = meter.round_rows(rows, new_nc, nbin)
    meter.copy_install(hz, new_nc, rows)
    extra = new_nc - nc

    def padded_copy(matrix):
        return sp.hstack((matrix.copy(), sp.csr_matrix((matrix.shape[0], extra))),
                         format='csr')

    # Copy every original array: neither installation nor future caller edits
    # may mutate the supplied source HZ through shared numeric buffers.
    result = nb.SparseHZono(
        c=hz.c.copy(), Gc=padded_copy(hz.Gc), Gb=hz.Gb.copy(),
        Ac=padded_copy(hz.Ac), Ab=hz.Ab.copy(), b=hz.b.copy(),
        Auc=sp.vstack((padded_copy(hz.Auc), ac), format='csr'),
        Aub=sp.vstack((hz.Aub.copy(), ab), format='csr'),
        ub=np.concatenate((hz.ub.copy(), rhs)), frame_id=hz.frame_id, exact=hz.exact)
    input_bytes, input_nnz = _storage(hz)
    output_bytes, output_nnz = _storage(result)
    physical = {'input': input_bytes, 'output': output_bytes,
                'input_plus_output': input_bytes + output_bytes,
                'generated_row_buffers': row_bytes}
    nnz = {'input': input_nnz, 'output': output_nnz,
           'added': int(ac.nnz + ab.nnz), 'exact_added': row_occurrences}
    return nt.SharedUpperResult(result, nc, shared, tuple(residual_bindings),
                             tuple(rows), errors, stored_rhs,
                             len(residual_bindings), physical, nnz)


def _relay(hz, parents, consumers, observations, next_consumers,
           extracted, footprints, metadata, meter):
    """Private D093 body sharing the exact same certified original readouts."""
    nc, nbin = hz.n_cont, hz.n_bin
    k, p, jn = len(consumers), len(observations), len(next_consumers)
    input_occurrences, parameter_entries, index_entries, certificate_count = metadata
    meter.charge('certified_relay_headers', 2048 + 128 * (k + p + jn + 1))
    combination_occurrences = 0

    def combine(*scaled, bias=ZERO):
        nonlocal combination_occurrences
        meter.charge('certified_relay_combination_guard', 64 * (len(scaled) + 1))
        appearances = 1
        for scale, affine in scaled:
            _f(scale)
            appearances = nb._limit(appearances + 1 + nb._shape(affine))
        combination_occurrences = nb._limit(combination_occurrences + appearances)
        return meter.combine(*scaled, bias=bias)

    zero = nt._constant(ZERO)
    u = nt._continuous(parents[0].slots[1], -HALF, HALF)
    v = nt._continuous(parents[1].slots[1], -HALF, HALF)
    children, child_bounds, slopes = [], [], []
    for (_, a, b), (g, q) in zip(consumers, extracted[2:2 + k]):
        meter.charge('certified_relay_child_scalars', 1024)
        remainder = combine((ONE, g), (_neg(a), u), (_neg(b), v))
        residual = NativeAffine(ZERO, remainder.continuous, remainder.binary)
        c, radius = remainder.bias, meter.norm(residual)
        bounds, lines = [], []
        for s, t in STATES:
            lower = _sub(_add(_add(c, _mul(min(a, ZERO), Fraction(s))),
                              _mul(min(b, ZERO), Fraction(t))), radius)
            upper = _add(_add(_add(c, _mul(max(a, ZERO), Fraction(s))),
                              _mul(max(b, ZERO), Fraction(t))), radius)
            bounds.append((lower, upper))
            lines.append(meter.secant(lower, upper))
        children.append((c, a, b, residual, radius, q))
        child_bounds.append(tuple(bounds))
        slopes.append(tuple(lines))

    core_occurrences = 0
    for row in observations:
        meter.charge('certified_core_occurrence_guard', 128 * (len(row) + 1))
        for child_index, _ in row:
            core_occurrences = nb._limit(core_occurrences
                                        + 4 * (1 + nb._shape(children[child_index][3])))

    base = _append_certified_base(hz, parents, consumers, extracted[:2 + k],
                                  footprints[:2 + k], meter)
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
        meter.charge('certified_relay_core_headers', 1024)
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
        meter.charge('certified_observation_scalar_population', 1024 * (len(weights) + 1))
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
            common_radius = meter.norm(combine(*common))
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
        meter.charge('certified_next_scalar_population', 1024 * (len(coefficients) + 1))
        remainder = combine((ONE, g),
                            *((_neg(amount), readouts[ci]) for ci, amount in coefficients))
        c = remainder.bias
        residual = NativeAffine(ZERO, remainder.continuous, remainder.binary)
        radius = meter.norm(residual)
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
            slope, intercept = meter.secant(lower, upper)
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
    ac, ab, rhs, errors, stored, row_bytes = meter.round_rows(rows, new_nc, nbin)
    meter.copy_install(middle, new_nc, rows)
    meter.charge('certified_result_receipts',
                 128 * (base_count + len(rows) + k + p + jn + certificate_count + 1))
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
                  'graph_extractions_including_base': 0,
                  'raw_graph_extractions': 0,
                  'unique_birth_certificates': certificate_count,
                  'certificate_verifications': certificate_count,
                  'work_charged': meter.used,
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
