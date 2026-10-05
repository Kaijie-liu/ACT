"""Default-off, single-step shared two-parent upper rows on an actual HZ.

Every gate is authenticated against stored EQ/LE predicates by frozen D064.
The original HZ, original bits, readouts, predicates, frame and exact flag are
retained. Three shared continuous coordinates encode delta, zu, zv; a nonzero
wide residual additionally uses three continuous coordinates per consumer.
These are not phase splits, new phase bits, or a next-layer cell interface.

Exact rational rows are compiled outward: each rounded coefficient contributes
its absolute error to the RHS because ALL native latent factors lie in [-1,1].
Thus every original integer-phase state has a canonical feasible extension;
rounded added rows need not uniquely enforce their exact auxiliary meanings.
This is a theorem about the supplied native predicates, not model provenance.

Selected graph/input and generated-row occurrences are separately bounded by
65536; every exact scalar/intermediate uses D064's 512-bit checks. Full input
CSR validation and copies, sparse padding/stacking, exact-row metadata and
conversion temporaries are real additional costs. Reported array bytes/nnz
are descriptive retained storage, NOT whole-work/peak-memory/GPU qualification.
"""

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np
import scipy.sparse as sp

from experiments.neural_hz_20260831.definition_first_20260928.d064_native_predicate_binding_20261001 import native_binding as nb


NativeAffine, Graph, KernelError = nb.NativeAffine, nb.Graph, nb.KernelError
MAX_BITS, MAX_SUPPORT = nb.MAX_BITS, nb.MAX_SUPPORT
ZERO, ONE, TWO = Fraction(0), Fraction(1), Fraction(2)
HALF = Fraction(1, 2)
_f, _add, _sub, _mul, _neg = nb._f, nb._add, nb._sub, nb._mul, nb._neg


@dataclass(frozen=True)
class SharedUpperResult:
    hz: nb.SparseHZono
    old_n_cont: int
    shared_columns: tuple[int, int, int]
    residual_bindings: tuple
    exact_rows: tuple
    row_errors: tuple
    installed_rhs: tuple
    extra_residual_count: int
    physical_bytes: dict
    nnz: dict

    @property
    def rows_added(self):
        return len(self.exact_rows)


def _constant(value):
    return NativeAffine(_f(value), (), ())


def _continuous(column, scale=ONE, bias=ZERO):
    return NativeAffine(_f(bias), ((column, _f(scale)),), ())


def _binary(column, scale=ONE, bias=ZERO):
    return NativeAffine(_f(bias), (), ((column, _f(scale)),))


def _sum(*scaled, bias=ZERO):
    return nb._combine(_f(bias), tuple(scaled))


def _positive_part(value):
    return value if value > ZERO else ZERO


def _secant(lower, upper):
    _f(lower), _f(upper)
    if lower > upper:
        raise KernelError('inconsistent slot interval')
    if upper <= ZERO:
        return ZERO, ZERO
    if lower >= ZERO:
        return ONE, ZERO
    slope = _f(upper / _sub(upper, lower))
    return slope, _neg(_mul(slope, lower))


def _storage(hz):
    matrices = (hz.Gc, hz.Gb, hz.Ac, hz.Ab, hz.Auc, hz.Aub)
    arrays = [hz.c, hz.b, hz.ub]
    for matrix in matrices:
        arrays.extend((matrix.data, matrix.indices, matrix.indptr))
    # Identity-sharing is counted once. This is retained ndarray storage only.
    seen, size = set(), 0
    for array in arrays:
        if id(array) not in seen:
            seen.add(id(array))
            size += int(array.nbytes)
    return size, sum(int(matrix.nnz) for matrix in matrices)


def _rounded_rows(rows, nc, nbin):
    """Convert canonical exact LE rows, compensating every coefficient error."""
    cdata, cindices, cptr = [], [], [0]
    bdata, bindices, bptr = [], [], [0]
    errors, rhs_float, rhs_exact = [], [], []
    for affine, rhs in rows:
        nb._checked(affine, nc, nbin)
        if affine.bias != ZERO:
            raise KernelError('compiled LE row must have zero bias')
        error = ZERO
        for terms, data, indices, ptr in (
                (affine.continuous, cdata, cindices, cptr),
                (affine.binary, bdata, bindices, bptr)):
            for index, amount in terms:
                try:
                    rounded = float(amount)
                except (OverflowError, ValueError) as exc:
                    raise KernelError('non-finite native coefficient') from exc
                if not math.isfinite(rounded):
                    raise KernelError('non-finite native coefficient')
                error = _add(error, nb._absolute(_sub(nb._fraction(rounded), amount)))
                if rounded != 0.0:
                    data.append(rounded)
                    indices.append(index)
            ptr.append(len(data))
        stored, exact = nb._round_upper(_add(_f(rhs), error))
        errors.append(error)
        rhs_float.append(stored)
        rhs_exact.append(exact)
    shape_c, shape_b = (len(rows), nc), (len(rows), nbin)
    ac = sp.csr_matrix((np.asarray(cdata, dtype=np.float64),
                        np.asarray(cindices, dtype=np.int64),
                        np.asarray(cptr, dtype=np.int64)), shape=shape_c)
    ab = sp.csr_matrix((np.asarray(bdata, dtype=np.float64),
                        np.asarray(bindices, dtype=np.int64),
                        np.asarray(bptr, dtype=np.int64)), shape=shape_b)
    rhs = np.asarray(rhs_float, dtype=np.float64)
    buffers = int(rhs.nbytes)
    for matrix in (ac, ab):
        buffers += int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    return ac, ab, rhs, tuple(errors), tuple(rhs_exact), buffers


def append_shared_upper(hz, parents, consumers, *, enabled=False):
    """Append 12 shared rows, 8 per nonzero residual, and one per consumer.

    parents is exactly two D064 Graphs. consumers is a nonempty immutable tuple
    of (Graph, Fraction a, Fraction b), where a,b multiply NORMALIZED parent
    readouts u=(1-etaA)/2 and v=(1-etaB)/2. The complete residual is derived
    from the actual child preactivation; no other parent is omitted.

    shared_columns stores the native zeta columns, with delta/zu/zv=(1+zeta)/2.
    residual_bindings entries are (consumer_index, M, (col00,col10,col01));
    r_s=M*xi_s and r11=original_residual-r00-r10-r01. exact_rows entries are
    (zero-bias NativeAffine, Fraction RHS), all in the returned HZ frame.
    Disabled calls inspect only the strict-bool enabled argument.
    """
    if not nb.ms._on(enabled):
        return None
    if type(parents) is not tuple or len(parents) != 2:
        raise KernelError('exactly two immutable parent graph descriptors required')
    if type(consumers) is not tuple or not consumers:
        raise KernelError('a nonempty immutable consumer bank is required')
    nb._limit(2 + len(consumers))
    nc, nbin = nb._hz_schema(hz)
    count = 2 + len(consumers)
    for graph in parents:
        count = nb._limit(count + nb._graph_shape(graph, hz, nc, nbin))
    for item in consumers:
        if type(item) is not tuple or len(item) != 3:
            raise KernelError('consumers are (Graph, Fraction a, Fraction b)')
        graph, a, b = item
        _f(a), _f(b)
        count = nb._limit(count + nb._graph_shape(graph, hz, nc, nbin))
    if parents[0].slots[2] == parents[1].slots[2]:
        raise KernelError('the two parents require distinct original binary columns')
    # Authenticate ALL supplied graph rows before constructing any new factors.
    extracted = []
    for graph in (*parents, *(item[0] for item in consumers)):
        g, q, error = nb._extract_graph(hz, graph)
        nb._checked(g, nc, nbin), nb._checked(q, nc, nbin)
        if error != ZERO:
            raise KernelError('this component requires zero native graph error')
        extracted.append((g, q))

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
        difference = _sum((ONE, left), (-ONE, right))
        row_occurrences = nb._limit(row_occurrences + len(difference.continuous)
                                   + len(difference.binary))
        rows.append((NativeAffine(ZERO, difference.continuous, difference.binary),
                     _neg(difference.bias)))
        nb._limit(len(rows))

    zero = _constant(ZERO)
    le(zero, delta)
    le(delta, alpha)
    le(delta, beta)
    le(_sum((ONE, alpha), (ONE, beta), bias=-ONE), delta)
    for amplitude, split, phase in ((u, zu, alpha), (v, zv, beta)):
        le(zero, split)
        le(split, delta)
        rest = _sum((ONE, amplitude), (-ONE, split))
        le(zero, rest)
        le(rest, _sum((ONE, phase), (-ONE, delta)))
    masses = (_sum((-ONE, alpha), (-ONE, beta), (ONE, delta), bias=ONE),
              _sum((ONE, alpha), (-ONE, delta)),
              _sum((ONE, beta), (-ONE, delta)), delta)
    us = (zero, _sum((ONE, u), (-ONE, zu)), zero, zu)
    vs = (zero, zero, _sum((ONE, v), (-ONE, zv)), zv)
    states = ((0, 0), (1, 0), (0, 1), (1, 1))

    for consumer_index, ((_, a, b), (g, q)) in enumerate(zip(consumers, extracted[2:])):
        remainder = _sum((ONE, g), (_neg(a), u), (_neg(b), v))
        c = remainder.bias
        residual = NativeAffine(ZERO, remainder.continuous, remainder.binary)
        radius = nb._norm(residual)
        if radius == ZERO:
            residuals = (zero, zero, zero, zero)
        else:
            nb._limit(new_nc - nc + 3)
            columns = (new_nc, new_nc + 1, new_nc + 2)
            new_nc += 3
            first = tuple(_continuous(column, radius) for column in columns)
            last = _sum((ONE, residual), *((-ONE, item) for item in first))
            residuals = (*first, last)
            residual_bindings.append((consumer_index, radius, columns))
            for mass, value in zip(masses, residuals):
                capacity = _sum((radius, mass))
                le(value, capacity)
                le(_sum((-ONE, value)), capacity)
        lifted = []
        for (s, t), mass, uc, vc, rc in zip(states, masses, us, vs, residuals):
            lower = _sub(_add(_add(c, _mul(min(a, ZERO), Fraction(s))),
                              _mul(min(b, ZERO), Fraction(t))), radius)
            upper = _add(_add(_add(c, _mul(max(a, ZERO), Fraction(s))),
                              _mul(max(b, ZERO), Fraction(t))), radius)
            slope, intercept = _secant(lower, upper)
            preactivation = _sum((c, mass), (a, uc), (b, vc), (ONE, rc))
            lifted.append(_sum((slope, preactivation), (intercept, mass)))
        le(q, _sum(*((ONE, item) for item in lifted)))

    ac, ab, rhs, errors, stored_rhs, row_bytes = _rounded_rows(rows, new_nc, nbin)
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
    return SharedUpperResult(result, nc, shared, tuple(residual_bindings),
                             tuple(rows), errors, stored_rhs,
                             len(residual_bindings), physical, nnz)
