"""Default-off binding of stored nonconvex HZ predicates to D063 bounds.

This certifies inequalities on the supplied SparseHZono, NOT correspondence
with a decoded network, an original model port, or a concrete input decoder.
Actual EQ/LE rows establish the gate relations; formal names alone do not.
All original factors, predicates, value rows, frame and exact flag survive.
No phase is searched, created, fixed or deleted. Compact rounding discrepancies
are retained as an explicit uniform error, including negative discrepancies.

Native float64 scalars are interpreted as exact binary rationals. Every
rational input/intermediate is limited to 512 bits. Sparse input occurrences
are preflighted together at 65536 before native combinations are allocated;
each expansion and combination is separately preflighted before merging.
Full original CSR validation/copies, reference construction and installation
are real costs; no complete work, memory, GPU or verifier qualification follows.
"""

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.definition_first_20260928.d063_mixed_block_component_20261001 import mixed_block as mb


ss, ms, mp = mb.ss, mb.ms, mb.mp
KernelError = mb.KernelError
MAX_BITS, MAX_SUPPORT = mb.MAX_BITS, mb.MAX_SUPPORT
ZERO, ONE, TWO = Fraction(0), Fraction(1), Fraction(2)
_f, _add, _sub, _mul, _neg = ms._f, ms._add, ms._sub, ms._mul, ms._neg


@dataclass(frozen=True)
class NativeAffine:
    bias: Fraction
    continuous: tuple[tuple[int, Fraction], ...]
    binary: tuple[tuple[int, Fraction], ...]


@dataclass(frozen=True)
class Graph:
    kind: str
    slots: tuple[int, int, int]
    eq_row: int | None
    le_rows: tuple[int, ...]


@dataclass(frozen=True)
class BindingResult:
    hz: SparseHZono
    lower: Fraction
    upper: Fraction
    reference: object
    graph_errors: tuple
    preactivation_errors: tuple
    receiver_error: Fraction
    installed_rhs: tuple
    rows_added: int = 2


def _limit(count):
    if count > MAX_SUPPORT:
        raise KernelError('aggregate native sparse occurrence limit exceeded')
    return count


def _index(value, width):
    if type(value) is not int or not 0 <= value < width:
        raise KernelError('native row/column index out of range')
    return value


def _absolute(value):
    value = _f(value)
    return _neg(value) if value < ZERO else value


def _fraction(value):
    value = float(value)
    if not math.isfinite(value):
        raise KernelError('native float64 value is not finite')
    return _f(Fraction.from_float(value))


def _midpoint(interval):
    lower, upper = ss._interval(interval)
    return _f(_add(lower, upper) / TWO)


def _shape(value):
    if (type(value) is not NativeAffine or type(value.continuous) is not tuple
            or type(value.binary) is not tuple):
        raise KernelError('immutable NativeAffine required')
    return _limit(len(value.continuous) + len(value.binary))


def _checked(value, nc, nb):
    _shape(value)
    _f(value.bias)
    for terms, width in ((value.continuous, nc), (value.binary, nb)):
        previous = -1
        for term in terms:
            if type(term) is not tuple or len(term) != 2:
                raise KernelError('native terms are (column, Fraction)')
            index, amount = _index(term[0], width), _f(term[1])
            if index <= previous or amount == ZERO:
                raise KernelError('native terms must be sorted, distinct and nonzero')
            previous = index
    return value


def _vector(value, size):
    if (type(value) is not np.ndarray or value.dtype != np.dtype('float64')
            or value.shape != (size,)):
        raise KernelError('native vectors must be one-dimensional float64 arrays')
    for item in value:
        if not math.isfinite(float(item)):
            raise KernelError('non-finite native vector')


def _csr(value, rows, cols):
    """Validate the stored object without normalizing or mutating it."""
    if (type(value) is not sp.csr_matrix or value.shape != (rows, cols)
            or value.data.dtype != np.dtype('float64')
            or value.data.ndim != 1 or value.indices.ndim != 1
            or value.indptr.ndim != 1 or value.indices.dtype.kind not in 'iu'
            or value.indptr.dtype.kind not in 'iu'
            or len(value.indptr) != rows + 1
            or len(value.indices) != len(value.data)
            or int(value.indptr[0]) != 0
            or int(value.indptr[-1]) != len(value.data)):
        raise KernelError('canonical native float64 CSR required')
    start = 0
    for row in range(rows):
        stop = int(value.indptr[row + 1])
        if not start <= stop <= len(value.data):
            raise KernelError('invalid native CSR row pointers')
        previous = -1
        for position in range(start, stop):
            column = int(value.indices[position])
            amount = float(value.data[position])
            if not previous < column < cols or not math.isfinite(amount) or amount == 0.0:
                raise KernelError('native CSR must be finite, sorted and zero-free')
            previous = column
        start = stop


def _hz_schema(hz):
    if type(hz) is not SparseHZono:
        raise KernelError('an actual SparseHZono is required')
    if (type(hz.frame_id) is not int or hz.frame_id < 0
            or hz.frame_id.bit_length() > MAX_BITS or type(hz.exact) is not bool):
        raise KernelError('native HZ requires its original frame and exact flag')
    if type(hz.c) is not np.ndarray or hz.c.ndim != 1:
        raise KernelError('invalid native output center')
    if type(hz.Gc) is not sp.csr_matrix or type(hz.Gb) is not sp.csr_matrix:
        raise KernelError('native readouts require CSR matrices')
    no, nc, nb = len(hz.c), hz.Gc.shape[1], hz.Gb.shape[1]
    if (type(hz.b) is not np.ndarray or type(hz.ub) is not np.ndarray
            or hz.b.ndim != 1 or hz.ub.ndim != 1):
        raise KernelError('native predicate right-hand sides must be arrays')
    _vector(hz.c, no)
    _vector(hz.b, len(hz.b))
    _vector(hz.ub, len(hz.ub))
    for matrix, rows, cols in ((hz.Gc, no, nc), (hz.Gb, no, nb),
                              (hz.Ac, len(hz.b), nc), (hz.Ab, len(hz.b), nb),
                              (hz.Auc, len(hz.ub), nc), (hz.Aub, len(hz.ub), nb)):
        _csr(matrix, rows, cols)
    return nc, nb


def _row_size(cont, binary, index):
    return (int(cont.indptr[index + 1]) - int(cont.indptr[index])
            + int(binary.indptr[index + 1]) - int(binary.indptr[index]))


def _row(cont, binary, index, bias=ZERO):
    _limit(_row_size(cont, binary, index))
    terms = []
    for matrix in (cont, binary):
        start, stop = int(matrix.indptr[index]), int(matrix.indptr[index + 1])
        terms.append(tuple((int(matrix.indices[p]), _fraction(matrix.data[p]))
                           for p in range(start, stop)))
    return NativeAffine(_f(bias), terms[0], terms[1])


def _graph_shape(graph, hz, nc, nb):
    if (type(graph) is not Graph or graph.kind not in ('extended', 'compact')
            or type(graph.slots) is not tuple or len(graph.slots) != 3
            or type(graph.le_rows) is not tuple):
        raise KernelError('immutable extended/compact graph descriptor required')
    s, eta, z = graph.slots
    _index(s, nc), _index(eta, nc), _index(z, nb)
    if graph.kind == 'extended':
        if s == eta or len(graph.le_rows) != 2:
            raise KernelError('extended graph requires two distinct continuous slots')
        _index(graph.eq_row, len(hz.b))
        count = _row_size(hz.Ac, hz.Ab, graph.eq_row)
    else:
        if s != eta or graph.eq_row is not None or len(graph.le_rows) != 3:
            raise KernelError('compact graph requires one slot and three LE rows')
        count = 0
    for index in graph.le_rows:
        _index(index, len(hz.ub))
        count = _limit(count + _row_size(hz.Auc, hz.Aub, index))
    if len(set(graph.le_rows)) != len(graph.le_rows):
        raise KernelError('graph LE row references must be distinct')
    return count


def _combine(bias, scaled):
    """Preflight expanded occurrences BEFORE constructing coefficient maps."""
    if type(scaled) is not tuple:
        raise KernelError('immutable native combination required')
    count = 0
    for scale, value in scaled:
        _f(scale)
        count = _limit(count + _shape(value))
    bias = _f(bias)
    cont, binary = {}, {}
    for scale, value in scaled:
        bias = _add(bias, _mul(scale, value.bias))
        for terms, merged in ((value.continuous, cont), (value.binary, binary)):
            for index, amount in terms:
                result = _add(merged.get(index, ZERO), _mul(scale, amount))
                if result == ZERO:
                    merged.pop(index, None)
                else:
                    merged[index] = result
                _limit(len(cont) + len(binary))
    return NativeAffine(bias, tuple(sorted(cont.items())), tuple(sorted(binary.items())))


def _norm(value):
    total = _absolute(value.bias)
    for terms in (value.continuous, value.binary):
        for _, coefficient in terms:
            total = _add(total, _absolute(coefficient))
    return total


def _mid_native(value, lookup):
    count = 0
    for source, _, _ in value.terms:
        count = _limit(count + _shape(lookup[source]))
    scaled = tuple((_midpoint((lo, hi)), lookup[source])
                   for source, lo, hi in value.terms)
    return _combine(_midpoint(value.bias), scaled)


def _guard(hz, index, cont, binary):
    actual = _row(hz.Auc, hz.Aub, index)
    expected = NativeAffine(ZERO, tuple(sorted(cont)), tuple(sorted(binary)))
    if actual != expected or _fraction(hz.ub[index]) != ZERO:
        raise KernelError('original phase guard row mismatch')


def _extract_graph(hz, graph):
    s, eta, z = graph.slots
    if graph.kind == 'extended':
        equation = _row(hz.Ac, hz.Ab, graph.eq_row)
        cont, binary = dict(equation.continuous), dict(equation.binary)
        L, Q = cont.pop(s, ZERO), _neg(cont.pop(eta, ZERO))
        if L >= ZERO or Q <= ZERO or binary.pop(z, ZERO) != L:
            raise KernelError('extended original equality template mismatch')
        _guard(hz, graph.le_rows[0], ((s, -ONE),), ((z, -ONE),))
        _guard(hz, graph.le_rows[1], ((eta, -ONE),), ((z, ONE),))
        remaining = NativeAffine(ZERO, tuple(sorted(cont.items())),
                                 tuple(sorted(binary.items())))
        native = _combine(_add(_fraction(hz.b[graph.eq_row]), Q), ((-ONE, remaining),))
        error = ZERO
    else:
        _guard(hz, graph.le_rows[0], ((eta, -ONE),), ((z, ONE),))
        lower = _row(hz.Auc, hz.Aub, graph.le_rows[1])
        upper = _row(hz.Auc, hz.Aub, graph.le_rows[2])
        low_c, low_b = dict(lower.continuous), dict(lower.binary)
        high_c, high_b = dict(upper.continuous), dict(upper.binary)
        Q = low_c.pop(eta, ZERO)
        L = high_b.pop(z, ZERO)
        if (Q <= ZERO or L >= ZERO or z in low_b
                or high_c.pop(eta, ZERO) != _neg(Q)):
            raise KernelError('compact original inequality template mismatch')
        P = NativeAffine(ZERO, tuple(sorted(low_c.items())), tuple(sorted(low_b.items())))
        opposite = _combine(ZERO, ((-ONE, P),))
        if (opposite.continuous != tuple(sorted(high_c.items()))
                or opposite.binary != tuple(sorted(high_b.items()))):
            raise KernelError('compact remaining coefficients are not exact negatives')
        R1, R2 = (_fraction(hz.ub[index]) for index in graph.le_rows[1:])
        delta = _add(_add(R1, R2), L)
        half = _f(delta / TWO)
        native = NativeAffine(_add(_sub(Q, R1), half), P.continuous, P.binary)
        error = _absolute(half)
    q = NativeAffine(Q, ((eta, _neg(Q)),), ())
    return native, q, error


def _round_upper(value):
    value = _f(value)
    try:
        rounded = float(value)
    except (OverflowError, ValueError) as exc:
        raise KernelError('native RHS cannot be represented finitely') from exc
    if not math.isfinite(rounded):
        raise KernelError('native RHS cannot be represented finitely')
    if _fraction(rounded) < value:
        rounded = math.nextafter(rounded, math.inf)
    exact = _fraction(rounded)
    if exact < value:
        raise KernelError('native outward RHS conversion failed')
    return rounded, exact


def bind_and_append(hz, v, gates, receiver, source_readouts, graphs,
                    receiver_index, *, enabled=False):
    """Derive a D063 reference once, bind actual predicates, append two LE.

    Graph LE order is (two guards) for extended and (guard, lower, upper)
    for compact. installed_rhs contains exact Fractions of the two stored
    float64 RHS values. Disabled calls inspect no argument besides enabled.
    """
    if not ms._on(enabled):
        return None
    context = mb._preflight(v, gates)
    mb._validate(v, gates, receiver, context)
    nc, nb = _hz_schema(hz)
    _index(receiver_index, hz.n_out)
    if (type(source_readouts) is not tuple or len(source_readouts) != len(context.sources)
            or type(graphs) is not tuple or len(graphs) != len(gates)):
        raise KernelError('complete ordered source and gate bindings required')

    # One aggregate input scan, before lookup maps or extracted rows exist.
    count = len(context.sources) + len(gates) + ss._shape(v)
    for gate in gates:
        count = _limit(count + ss._shape(gate[0]))
    for source in source_readouts:
        count = _limit(count + _shape(source))
    count = _limit(count + _row_size(hz.Gc, hz.Gb, receiver_index))
    for graph, gate in zip(graphs, gates):
        count = _limit(count + _graph_shape(graph, hz, nc, nb))
        if gate[2].ordinal != graph.slots[2]:
            raise KernelError('formal original phase ordinal is not the native binary column')
    for source, actual in zip(context.sources, source_readouts):
        _checked(actual, nc, nb)
        radius = _sub(_norm(actual), _absolute(actual.bias))
        if (source.lower > _sub(actual.bias, radius)
                or source.upper < _add(actual.bias, radius)):
            raise KernelError('formal source box does not cover its native cube readout')

    lookup = dict(zip(context.sources, source_readouts))
    receiver_native = _row(hz.Gc, hz.Gb, receiver_index, _fraction(hz.c[receiver_index]))
    native_v = _mid_native(v, lookup)
    q_values, weights, graph_errors, errors = [], [], [], []
    for gate, graph in zip(gates, graphs):
        native_g, native_q, graph_error = _extract_graph(hz, graph)
        reference_g = _mid_native(gate[0], lookup)
        difference = _combine(ZERO, ((ONE, native_g), (-ONE, reference_g)))
        errors.append(_add(graph_error, _norm(difference)))
        graph_errors.append(graph_error)
        q_values.append(native_q)
        weights.append(_midpoint(gate[3]))
        # If the same formal value also occurs as a source, it must have one
        # actual native readout, not two unrelated interpretations.
        declared = context._by_value.get(gate[1])
        if declared is not None and lookup[declared] != native_q:
            raise KernelError('shared source/gate output has inconsistent native readouts')
    predictor = _combine(ZERO, ((ONE, native_v), *tuple(zip(weights, q_values))))
    receiver_error = _norm(_combine(ZERO, ((ONE, receiver_native), (-ONE, predictor))))
    error = receiver_error
    for weight, gate_error in zip(weights, errors):
        error = _add(error, _mul(_absolute(weight), gate_error))

    reference = mb.generate(v, gates, receiver, enabled=True)
    upper = _add(_sub(reference.upper, reference.error), error)
    lower = _sub(_add(reference.lower, reference.error), error)
    if lower > upper:
        raise KernelError('inconsistent bound on original native receiver')
    upper_float, upper_exact = _round_upper(_sub(upper, receiver_native.bias))
    lower_float, lower_exact = _round_upper(_sub(receiver_native.bias, lower))
    _limit(2 * _row_size(hz.Gc, hz.Gb, receiver_index))
    row_c, row_b = hz.Gc[receiver_index], hz.Gb[receiver_index]
    result = SparseHZono(
        c=hz.c, Gc=hz.Gc, Gb=hz.Gb, Ac=hz.Ac, Ab=hz.Ab, b=hz.b,
        Auc=sp.vstack((hz.Auc, row_c, -row_c), format='csr'),
        Aub=sp.vstack((hz.Aub, row_b, -row_b), format='csr'),
        ub=np.concatenate((hz.ub, np.asarray((upper_float, lower_float), dtype=np.float64))),
        frame_id=hz.frame_id, exact=hz.exact,
    )
    return BindingResult(result, lower, upper, reference, tuple(graph_errors),
                         tuple(errors), receiver_error, (upper_exact, lower_exact))
