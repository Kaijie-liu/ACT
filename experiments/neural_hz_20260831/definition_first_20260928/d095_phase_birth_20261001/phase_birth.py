"""Exact native ReLU birth using two authenticated continuous-one carriers.

The original output is read completely as an exact binary-rational affine
form. Bounds come from its full unit latent box, not from an external oracle.
The new equality stores -c*rho_c+Q*rho_Q separately: it never rounds c-Q.
For an unstable gate, the signed bit is -1 on the active side and +1 on the
inactive side. At zero both original choices have a canonical extension.

All old predicates, latent prefixes, frame and exact flag survive unchanged.
Carrier/gate verification concerns actual native rows, not an owner token or
model provenance. Per-row support is limited to 65536 and rationals to 512
bits; the caller's hard pool prepays complete scans and construction batches.
Numeric buffers are reported, but complete Python/temporary memory, model,
decoder provenance, GPU and whole-physical qualification are not asserted.
"""

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np
import scipy.sparse as sp

from experiments.neural_hz_20260831.definition_first_20260928.d086_native_shared_transfer_20261001 import native_shared_transfer as nt

nb = nt.nb
NativeAffine, Graph, KernelError = nb.NativeAffine, nb.Graph, nb.KernelError
MAX_BITS, MAX_SUPPORT = nb.MAX_BITS, nb.MAX_SUPPORT
ZERO, ONE = Fraction(0), Fraction(1)


@dataclass(frozen=True)
class CarrierBinding:
    columns: tuple
    eq_rows: tuple


@dataclass(frozen=True)
class BornGate:
    output_row: int
    graph: Graph
    preactivation: NativeAffine
    readout: NativeAffine
    lower: Fraction
    upper: Fraction


@dataclass(frozen=True)
class BirthResult:
    hz: nb.SparseHZono
    old_n_cont: int
    old_n_bin: int
    carriers: CarrierBinding | None
    gates: tuple
    bounds: tuple
    physical_bytes: dict
    nnz: dict
    local_cost: dict


def _charger(pool):
    charge = getattr(pool, 'charge', None)
    if not callable(charge):
        raise KernelError('a caller-owned hard charging pool is required')
    paid = [0]

    def pay(name, amount):
        if type(amount) is not int or amount < 0 or amount.bit_length() > MAX_BITS:
            raise KernelError('invalid birth work charge')
        charge(name, amount)
        paid[0] += amount
    return pay, paid


def _headers(hz):
    """Constant-size native headers only; selected/full population follows."""
    if type(hz) is not nb.SparseHZono:
        raise KernelError('actual SparseHZono required')
    if (type(hz.frame_id) is not int or hz.frame_id < 0
            or hz.frame_id.bit_length() > MAX_BITS or type(hz.exact) is not bool):
        raise KernelError('original frame and exact flag required')
    if type(hz.Gc) is not sp.csr_matrix or type(hz.Gb) is not sp.csr_matrix:
        raise KernelError('native CSR readouts required')
    if any(type(v) is not np.ndarray or v.ndim != 1 or v.dtype != np.dtype('float64')
           for v in (hz.c, hz.b, hz.ub)):
        raise KernelError('native float64 vectors required')
    no, nc, nbin = len(hz.c), hz.Gc.shape[1], hz.Gb.shape[1]
    specs = ((hz.Gc, no, nc), (hz.Gb, no, nbin),
             (hz.Ac, len(hz.b), nc), (hz.Ab, len(hz.b), nbin),
             (hz.Auc, len(hz.ub), nc), (hz.Aub, len(hz.ub), nbin))
    for matrix, rows, cols in specs:
        if (type(matrix) is not sp.csr_matrix or matrix.shape != (rows, cols)
                or matrix.data.dtype != np.dtype('float64') or matrix.data.ndim != 1
                or matrix.indices.ndim != 1 or matrix.indptr.ndim != 1
                or matrix.indices.dtype.kind not in 'iu' or matrix.indptr.dtype.kind not in 'iu'
                or len(matrix.indptr) != rows + 1 or len(matrix.indices) != len(matrix.data)
                or int(matrix.indptr[0]) != 0 or int(matrix.indptr[-1]) != len(matrix.data)):
            raise KernelError('invalid native CSR header')
    population = (sum(len(m.data) + len(m.indptr) for m, _, _ in specs)
                  + len(hz.c) + len(hz.b) + len(hz.ub))
    return nc, nbin, population


def _selected_row(hz, inequality, index, pay):
    pay('birth_selected_row_header', 64)
    ac, ab, rhs = (hz.Auc, hz.Aub, hz.ub) if inequality else (hz.Ac, hz.Ab, hz.b)
    nb._index(index, len(rhs))
    count = 0
    for matrix in (ac, ab):
        start, stop = int(matrix.indptr[index]), int(matrix.indptr[index + 1])
        if not 0 <= start <= stop <= len(matrix.data):
            raise KernelError('invalid selected native row pointers')
        count += stop - start
    nb._limit(count)
    pay('birth_selected_row_population', 128 * (count + 8))
    row = nb._row(ac, ab, index)
    nb._checked(row, hz.n_cont, hz.n_bin)
    return row, nb._fraction(rhs[index])


def _carriers(hz, binding, pay):
    pay('birth_carrier_binding', 64)
    if (type(binding) is not CarrierBinding or type(binding.columns) is not tuple
            or type(binding.eq_rows) is not tuple
            or len(binding.columns) != 2 or len(binding.eq_rows) != 2):
        raise KernelError('two immutable carrier columns and equality rows required')
    if len(set(binding.columns)) != 2 or len(set(binding.eq_rows)) != 2:
        raise KernelError('carrier columns and equality rows must each be distinct')
    for column, row_index in zip(binding.columns, binding.eq_rows):
        nb._index(column, hz.n_cont)
        row, rhs = _selected_row(hz, False, row_index, pay)
        if row != NativeAffine(ZERO, ((column, ONE),), ()) or rhs != ONE:
            raise KernelError('carrier must be the actual literal equality rho=1')
    return binding


def _box(g):
    radius = ZERO
    for terms in (g.continuous, g.binary):
        for _, amount in terms:
            radius = nb._add(radius, nb._absolute(amount))
    return nb._sub(g.bias, radius), nb._add(g.bias, radius)


def _exact_float(value):
    value = nb._f(value)
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise KernelError('native scalar cannot be represented finitely') from exc
    if not math.isfinite(result) or nb._fraction(result) != value:
        raise KernelError('birth coefficient requires exact float64 storage')
    return result


def _halves(lower, upper):
    low, high = _exact_float(lower), _exact_float(upper)
    L, Q = nb._f(lower / 2), nb._f(upper / 2)
    if (not lower < ZERO < upper or L >= ZERO or Q <= ZERO
            or nb._fraction(low / 2.0) != L or nb._fraction(high / 2.0) != Q):
        raise KernelError('unstable endpoints require exact finite nonzero halves')
    return L, Q


def _literal_equation(g, graph, carriers, L, Q, nc, nbin):
    s, eta, bit = graph.slots
    forbidden = (*carriers.columns, s, eta)
    if len(set(forbidden)) != 4:
        raise KernelError('carrier and gate continuous roles must be distinct')
    if any(column in forbidden for column, _ in g.continuous) or any(
            column == bit for column, _ in g.binary):
        raise KernelError('preactivation cannot reference its gate slots or carriers')
    nb._limit(len(g.continuous) + len(g.binary) + 4 + int(g.bias != ZERO))
    continuous = [(column, nb._neg(amount)) for column, amount in g.continuous]
    continuous.extend(((s, L), (eta, nb._neg(Q)), (carriers.columns[1], Q)))
    if g.bias != ZERO:
        continuous.append((carriers.columns[0], nb._neg(g.bias)))
    binary = [(column, nb._neg(amount)) for column, amount in g.binary]
    binary.append((bit, L))
    row = NativeAffine(ZERO, tuple(sorted(continuous)), tuple(sorted(binary)))
    return nb._checked(row, nc, nbin)


def _matrix(rows, columns):
    data, indices, pointers = [], [], [0]
    for row in rows:
        nb._limit(len(row))
        for column, amount in row:
            indices.append(column)
            data.append(_exact_float(amount))
        pointers.append(len(data))
    return sp.csr_matrix((np.asarray(data, dtype=np.float64),
                          np.asarray(indices, dtype=np.int64),
                          np.asarray(pointers, dtype=np.int64)),
                         shape=(len(rows), columns))


def relu_birth(hz, *, carriers=None, pool, enabled=False):
    """Apply ReLU to every output using bounds derived from its full readout.

    bounds records exact unit-box intervals for ALL original output rows.
    gates records only the unstable rows, in original output order; its stored
    lower/upper endpoints are outward float64 bounds represented as Fractions.
    Existing carriers are authenticated even if no unstable gate is born.
    """
    if not nb.ms._on(enabled):
        return None
    pay, paid = _charger(pool)
    pay('birth_metadata', 128)
    nc, nbin, population = _headers(hz)
    pay('birth_full_native_schema', 32 * (population + nc + nbin + 1))
    nb._hz_schema(hz)
    if carriers is not None:
        _carriers(hz, carriers, pay)
    pay('birth_all_output_headers', 64 * (hz.n_out + 1))
    plans, bounds = [], []
    unstable = active = inactive = input_terms = 0
    largest = 1
    for index in range(hz.n_out):
        support = nb._row_size(hz.Gc, hz.Gb, index)
        nb._limit(support)
        pay('birth_complete_output_readout_and_bounds', 128 * (support + 8))
        g = nb._row(hz.Gc, hz.Gb, index, nb._fraction(hz.c[index]))
        nb._checked(g, nc, nbin)
        if carriers is not None and any(column in carriers.columns for column, _ in g.continuous):
            raise KernelError('every original output must have zero carrier coefficients')
        lower, upper = _box(g)
        bounds.append((lower, upper))
        input_terms += support
        largest = max(largest, support + 8)
        if lower >= ZERO:
            mode, endpoints = 'active', None
            active += 1
        elif upper <= ZERO:
            mode, endpoints = 'inactive', None
            inactive += 1
        else:
            _, minus_lower = nb._round_upper(nb._neg(lower))
            _, stored_upper = nb._round_upper(upper)
            stored_lower = nb._neg(minus_lower)
            L, Q = _halves(stored_lower, stored_upper)
            nb._limit(support + 4 + int(g.bias != ZERO))
            mode, endpoints = 'unstable', (stored_lower, stored_upper, L, Q)
            unstable += 1
        plans.append((index, g, mode, endpoints))

    new_carriers = unstable > 0 and carriers is None
    carrier_count = 2 if new_carriers else 0
    if new_carriers:
        carriers = CarrierBinding((nc, nc + 1), (hz.n_eq, hz.n_eq + 1))
    new_nc, new_nb = nc + carrier_count + 2 * unstable, nbin + unstable
    # This prepays all unmerged source occurrences, sorted literal rows,
    # per-output metadata, guard entries and their exact scalar checks.
    construction_population = input_terms + hz.n_out + 16 * unstable + 8
    pay('birth_full_literal_construction',
        128 * construction_population * (1 + largest.bit_length()))
    equality_rows, equality_rhs = [], []
    if new_carriers:
        for column in carriers.columns:
            equality_rows.append(NativeAffine(ZERO, ((column, ONE),), ()))
            equality_rhs.append(1.0)
    out_c, out_cont, out_binary = [], [], []
    negative, positive, gates = [], [], []
    for index, g, mode, endpoints in plans:
        if mode == 'active':
            readout = g
        elif mode == 'inactive':
            readout = NativeAffine(ZERO, (), ())
        else:
            lower, upper, L, Q = endpoints
            position = len(gates)
            s, eta = nc + carrier_count + 2 * position, nc + carrier_count + 2 * position + 1
            bit = nbin + position
            graph = Graph('extended', (s, eta, bit),
                          hz.n_eq + carrier_count + position,
                          (hz.n_ineq + position, hz.n_ineq + unstable + position))
            equality_rows.append(_literal_equation(g, graph, carriers, L, Q, new_nc, new_nb))
            equality_rhs.append(0.0)
            negative.append(NativeAffine(ZERO, ((s, -ONE),), ((bit, -ONE),)))
            positive.append(NativeAffine(ZERO, ((eta, -ONE),), ((bit, ONE),)))
            readout = NativeAffine(Q, ((eta, nb._neg(Q)),), ())
            gates.append(BornGate(index, graph, g, readout, lower, upper))
        out_c.append(_exact_float(readout.bias))
        out_cont.append(readout.continuous)
        out_binary.append(readout.binary)
    inequalities = (*negative, *positive)
    added_eq_nnz = sum(len(row.continuous) + len(row.binary) for row in equality_rows)
    added_le_nnz = 4 * unstable
    output_nnz = sum(len(row) for row in out_cont) + sum(len(row) for row in out_binary)
    added_eq, added_le = carrier_count + unstable, 2 * unstable
    copy_population = (population + nc + nbin + new_nc + new_nb + output_nnz
                       + added_eq_nnz + added_le_nnz + 8 * (added_eq + added_le + hz.n_out + 1))
    pay('birth_complete_native_copy_and_receipts',
        32 * copy_population * (1 + largest.bit_length()))

    def padded(matrix, width):
        return sp.hstack((matrix.copy(), sp.csr_matrix((matrix.shape[0], width - matrix.shape[1]))),
                         format='csr')

    ac = _matrix(tuple(row.continuous for row in equality_rows), new_nc)
    ab = _matrix(tuple(row.binary for row in equality_rows), new_nb)
    auc = _matrix(tuple(row.continuous for row in inequalities), new_nc)
    aub = _matrix(tuple(row.binary for row in inequalities), new_nb)
    result = nb.SparseHZono(
        c=np.asarray(out_c, dtype=np.float64), Gc=_matrix(out_cont, new_nc), Gb=_matrix(out_binary, new_nb),
        Ac=sp.vstack((padded(hz.Ac, new_nc), ac), format='csr'),
        Ab=sp.vstack((padded(hz.Ab, new_nb), ab), format='csr'),
        b=np.concatenate((hz.b.copy(), np.asarray(equality_rhs, dtype=np.float64))),
        Auc=sp.vstack((padded(hz.Auc, new_nc), auc), format='csr'),
        Aub=sp.vstack((padded(hz.Aub, new_nb), aub), format='csr'),
        ub=np.concatenate((hz.ub.copy(), np.zeros(added_le, dtype=np.float64))),
        frame_id=hz.frame_id, exact=hz.exact)
    input_bytes, input_nnz = nt._storage(hz)
    output_bytes, total_nnz = nt._storage(result)
    local = dict(work_charged=paid[0], output_rows=hz.n_out, input_readout_terms=input_terms,
                 active_rows=active, inactive_rows=inactive, unstable_rows=unstable,
                 added_eq=added_eq, added_le=added_le, added_cont=new_nc - nc,
                 added_binary=unstable, new_carrier_columns=carrier_count,
                 whole_work_qualified=False, complete_physical_qualified=False,
                 actual_model_qualified=False, gpu_qualified=False)
    return BirthResult(result, nc, nbin, carriers, tuple(gates), tuple(bounds),
        dict(input=input_bytes, output=output_bytes, input_plus_output=input_bytes + output_bytes),
        dict(input=input_nnz, output=total_nnz, added_eq=added_eq_nnz,
             added_le=added_le_nnz, output_readout=output_nnz), local)


def verify_gate(hz, gate, carriers, *, pool, enabled=False):
    """Authenticate one stored birth relation modulo the literal carriers.

    Only selected predicates are scanned. output_row is birth metadata, not a
    demand that a later affine consumer retain the original output bank. The
    actual equality and guards, not an owner identity, establish returned g,q.
    """
    if not nb.ms._on(enabled):
        return None
    pay, _ = _charger(pool)
    pay('birth_verify_metadata', 128)
    nc, nbin, _ = _headers(hz)
    if (type(gate) is not BornGate or type(gate.output_row) is not int
            or gate.output_row < 0 or gate.output_row.bit_length() > MAX_BITS
            or type(gate.graph) is not Graph or gate.graph.kind != 'extended'):
        raise KernelError('immutable extended BornGate required')
    _carriers(hz, carriers, pay)
    graph = gate.graph
    if (type(graph.slots) is not tuple or len(graph.slots) != 3
            or type(graph.le_rows) is not tuple or len(graph.le_rows) != 2
            or graph.le_rows[0] == graph.le_rows[1]):
        raise KernelError('birth gate requires distinct slots and two guard rows')
    s, eta, bit = graph.slots
    nb._index(s, nc), nb._index(eta, nc), nb._index(bit, nbin)
    if s == eta or graph.eq_row in carriers.eq_rows:
        raise KernelError('birth gate and carrier roles overlap')
    support = nb._shape(gate.preactivation) + nb._shape(gate.readout)
    pay('birth_verify_saved_readouts_and_literal',
        128 * (support + 16) * (1 + (support + 8).bit_length()))
    g, q = nb._checked(gate.preactivation, nc, nbin), nb._checked(gate.readout, nc, nbin)
    lower, upper = nb._f(gate.lower), nb._f(gate.upper)
    L, Q = _halves(lower, upper)
    lo, hi = _box(g)
    if lower > lo or upper < hi:
        raise KernelError('saved birth bounds do not cover the full source unit box')
    expected = _literal_equation(g, graph, carriers, L, Q, nc, nbin)
    actual, rhs = _selected_row(hz, False, graph.eq_row, pay)
    if actual != expected or rhs != ZERO:
        raise KernelError('stored birth equality differs from its literal source relation')
    for index, expected_guard in zip(graph.le_rows, (
            NativeAffine(ZERO, ((s, -ONE),), ((bit, -ONE),)),
            NativeAffine(ZERO, ((eta, -ONE),), ((bit, ONE),)))):
        actual, rhs = _selected_row(hz, True, index, pay)
        if actual != expected_guard or rhs != ZERO:
            raise KernelError('stored birth phase guard differs')
    if q != NativeAffine(Q, ((eta, nb._neg(Q)),), ()):
        raise KernelError('saved output is not the literal original born readout')
    return g, q
