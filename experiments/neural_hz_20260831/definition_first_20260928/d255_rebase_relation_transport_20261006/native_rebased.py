"""Default-off, content-certified rebase readouts for the D254 relation.

Only exact substitutions of stored linking equalities are added. No original
factor, phase, predicate, value map or decoder is removed. Rebase receipts
certify stored before/after structure, NOT a model, reliable input bounds,
a production invocation trace, or equality of the two integer sets.
Trusted in-memory records, no untrusted deserialization or private mutation.
"""
from dataclasses import dataclass, field
from fractions import Fraction as F
from functools import wraps
from types import MappingProxyType
import hashlib

import numpy as np
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono, _lower_hz_milp
from experiments.neural_hz_20260831.definition_first_20260928.d064_native_predicate_binding_20261001 import native_binding as nb
from experiments.neural_hz_20260831.definition_first_20260928.d086_native_shared_transfer_20261001 import native_shared_transfer as rounding
from experiments.neural_hz_20260831.definition_first_20260928.d243_operator_source_relation_20261005 import source_relation as base

Budget, Limits, Rejected = base.Budget, base.Limits, base.Rejected
NativeAffine, Graph = nb.NativeAffine, nb.Graph
ZERO, ONE, TWO, HALF, NEG = F(0), F(1), F(2), F(1, 2), F(-1)
_exact, _OWNED = base._exact, object()


def _legacy_failure(exc, budget):
    # These two messages are pinned by the frozen D046/D064 sources. Ordinary
    # malformed predicates do not poison a still-usable original snapshot.
    if isinstance(exc, nb.KernelError) and str(exc) in (
            "rational bit limit exceeded", "aggregate native sparse occurrence limit exceeded"):
        budget._reject("legacy native resource limit: " + str(exc))


def _op(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        meter = self.budget._branch()
        try:
            return method(self, *args, _meter=meter, **kwargs)
        except MemoryError:
            self.budget._reject("native terminal operation allocation failed")
        except (nb.KernelError, TypeError, ValueError, IndexError, KeyError,
                AttributeError, OverflowError) as exc:
            _legacy_failure(exc, self.budget)
            raise Rejected("invalid native terminal operation: " + str(exc)) from exc
    return call


def _on(enabled):
    if type(enabled) is not bool:
        raise Rejected("enabled must be bool")
    return enabled


def _charge_hz(hz, m, copies=1):
    if type(hz) is not SparseHZono:
        raise Rejected("actual SparseHZono required")
    arrays = [hz.c, hz.b, hz.ub]
    for matrix in (hz.Gc, hz.Gb, hz.Ac, hz.Ab, hz.Auc, hz.Aub):
        arrays.extend((matrix.data, matrix.indices, matrix.indptr))
    size = sum(int(a.size) for a in arrays)
    m.charge(work=copies * size + 20, entries=copies * size + 20)
    return arrays


def _stamp(hz, m):
    arrays = _charge_hz(hz, m, 2)
    nb._hz_schema(hz)
    m.charge(work=sum(int(a.nbytes) for a in arrays))
    digest = hashlib.sha256()
    digest.update(repr((hz.frame_id, hz.exact, hz.n_out, hz.n_cont, hz.n_bin,
                        len(hz.b), len(hz.ub))).encode())
    for a in arrays:
        digest.update(repr((a.shape, a.dtype.str)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def _readonly(hz):
    for a in (hz.c, hz.b, hz.ub):
        a.setflags(write=False)
    for a in (hz.Gc, hz.Gb, hz.Ac, hz.Ab, hz.Auc, hz.Aub):
        for v in (a.data, a.indices, a.indptr):
            v.setflags(write=False)
    return hz


def _pad(matrix, width):
    return sp.hstack((matrix.copy(), sp.csr_matrix((matrix.shape[0],
                       width - matrix.shape[1]), dtype=np.float64)), format="csr")


def _flat(value, nc, width, m):
    return base._form(value.bias, value.continuous + tuple(
        (nc + i, a) for i, a in value.binary), width, m)


def _native(row, nc, nbin, width, m):
    old = nc + nbin
    cont, binary = [], []
    m.charge(work=3 * len(row.terms) + 1, entries=3 * len(row.terms) + 1)
    for i, a in row.terms:
        if i < nc:
            cont.append((i, a))
        elif i < old:
            binary.append((i - nc, a))
        else:
            cont.append((nc + i - old, a))
    cont.sort()
    return NativeAffine(ZERO, tuple(cont), tuple(binary)), row.rhs


def _stored_flat(affine, nc, old_nc, old_nb, width, m):
    # Native continuous auxiliaries precede native binary columns. The proof
    # ordering instead preserves the complete old signed tuple as a prefix.
    old = old_nc + old_nb
    terms = tuple((i if i < old_nc else old + i - old_nc, a)
                  for i, a in affine.continuous)
    terms += tuple((old_nc + i, a) for i, a in affine.binary)
    return base._form(affine.bias, terms, width, m)



@dataclass(frozen=True)
class LinkEpoch:
    old_cont: int
    new_cont: int
    eq_rows: tuple


@dataclass(frozen=True)
class RebaseReceipt:
    epoch: LinkEpoch
    frame_id: int
    n_bin: int
    link_rows: tuple
    offsets: tuple
    all_added_eq: int
    before_fingerprint: str
    after_fingerprint: str
    budget: Budget = field(repr=False, compare=False)
    _authority: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._authority is not _OWNED:
            raise Rejected("rebase receipt must be produced by record_rebase")


@dataclass(frozen=True)
class RebaseBinding:
    receipt: RebaseReceipt
    eq_rows: tuple


@dataclass(frozen=True)
class Definition:
    column: int
    eq_row: int
    raw: base.Affine
    normal: base.Affine


def _csr_prefix(left, right, rows, m):
    count = int(left.indptr[rows])
    m.charge(work=3 * count + rows + 2, entries=2)
    return (np.array_equal(left.indptr[:rows + 1], right.indptr[:rows + 1])
            and np.array_equal(left.indices[:count], right.indices[:count])
            and np.array_equal(left.data[:count], right.data[:count]))


def record_rebase(before, after, *, budget=None, enabled=False):
    """Check complete stored rebase content; do not infer model provenance.

    A faithful linking format does not prove that the new image boxes enclose
    all old points. The producer's reliable-bound proof remains separately
    required for model admission. All normalization is on the final stored H.
    """
    if not _on(enabled):
        return None
    if budget is None:
        budget = Budget()
    if type(budget) is not Budget:
        raise Rejected("shared Budget required")
    m = budget._branch()
    try:
        first, last = _stamp(before, m), _stamp(after, m)
        if (not before.exact or not after.exact
                or before.frame_id != after.frame_id
                or before.n_out != after.n_out or before.n_bin != after.n_bin
                or after.n_cont < before.n_cont or len(after.b) < len(before.b)
                or len(after.ub) != len(before.ub)):
            raise Rejected("rebase must preserve complete old frame and predicates")
        _charge_hz(before, m, 4)
        _charge_hz(after, m, 4)
        if (not np.array_equal(before.b, after.b[:len(before.b)])
                or not np.array_equal(before.ub, after.ub)
                or not all(_csr_prefix(left, right, rows, m) for left, right, rows in (
                    (before.Ac, after.Ac, len(before.b)),
                    (before.Ab, after.Ab, len(before.b)),
                    (before.Auc, after.Auc, len(before.ub)),
                    (before.Aub, after.Aub, len(before.ub))))
                or after.Gb.nnz != 0):
            raise Rejected("rebase changed original predicate content or binary readout")
        next_col, next_row = before.n_cont, len(before.b)
        rows, links, offsets = [], [], []
        for i in range(before.n_out):
            m.charge(work=8, entries=4)
            start, stop = int(after.Gc.indptr[i]), int(after.Gc.indptr[i + 1])
            if stop - start not in (0, 1):
                raise Rejected("rebase output must be diagonal in appended image columns")
            col, radius = None, ZERO
            if stop != start:
                col = int(after.Gc.indices[start])
                radius = _exact._number(nb._fraction(after.Gc.data[start]), m)
                if col != next_col or _exact._cmp(radius, ZERO, m) <= 0:
                    raise Rejected("rebase image columns must be ordered, unique and positive")
                next_col += 1
            count = nb._row_size(before.Gc, before.Gb, i)
            m.charge(work=64 * count + 20, entries=12 * count + 20)
            original = nb._row(before.Gc, before.Gb, i)
            cont = original.continuous + (() if col is None else
                                          ((col, _exact._neg(radius, m)),))
            expected = NativeAffine(ZERO, cont, original.binary)
            # Audit the actual producer's binary64 subtraction, while keeping
            # its rational discrepancy. This is NOT exact-real subtraction.
            floating_rhs = float(after.c[i]) - float(before.c[i])
            rhs = _exact._number(nb._fraction(floating_rhs), m)
            used = bool(cont or original.binary or rhs != ZERO)
            if used:
                if next_row >= len(after.b):
                    raise Rejected("missing rebase linking equality")
                count = nb._row_size(after.Ac, after.Ab, next_row)
                m.charge(work=64 * count + 20, entries=12 * count + 20)
                actual = nb._row(after.Ac, after.Ab, next_row)
                if actual != expected or nb._fraction(after.b[next_row]) != rhs:
                    raise Rejected("stored rebase linking row differs")
                if col is not None:
                    rows.append(next_row)
                    links.append((actual, rhs))
                next_row += 1
            elif col is not None:
                raise Rejected("image column lacks a linking row")
            offset = _exact._sub(_exact._sub(
                _exact._number(nb._fraction(after.c[i]), m),
                _exact._number(nb._fraction(before.c[i]), m), m), rhs, m)
            offsets.append(offset)
        if next_col != after.n_cont or next_row != len(after.b):
            raise Rejected("unaccounted rebase columns or equalities")
        if _stamp(before, m) != first or _stamp(after, m) != last:
            raise Rejected("rebase states changed during content recording")
        return RebaseReceipt(LinkEpoch(before.n_cont, after.n_cont, tuple(rows)),
            before.frame_id, before.n_bin, tuple(links), tuple(offsets),
            len(after.b) - len(before.b), first, last, budget, _OWNED)
    except MemoryError:
        budget._reject("rebase receipt allocation failed")
    except (nb.KernelError, TypeError, ValueError, IndexError, KeyError,
            AttributeError, OverflowError) as exc:
        _legacy_failure(exc, budget)
        raise Rejected("invalid stored rebase: " + str(exc)) from exc


def _normalize_form(form, lookup, width, m):
    # Definitions are already fully normalized in increasing append order.
    # Charge both the lookup and every expanded occurrence before allocation.
    m.charge(work=2 * len(form.terms) + 1, entries=len(form.terms) + 1)
    occurrences = sum(len(lookup[i].terms) if i in lookup else 1 for i, _ in form.terms)
    m.charge(work=occurrences + 1, entries=2 * occurrences + 1)
    nb._limit(occurrences)
    constant, terms = _exact._number(form.constant, m), []
    for i, a in form.terms:
        if i not in lookup:
            terms.append((i, _exact._number(a, m)))
            continue
        value = lookup[i]
        constant = _exact._add(constant, _exact._mul(a, value.constant, m), m)
        terms.extend((j, _exact._mul(a, b, m)) for j, b in value.terms)
    return base._form(constant, tuple(terms), width, m)


def _definitions(hz, bindings, budget, m):
    if type(bindings) is not tuple:
        raise Rejected("immutable complete rebase bindings required")
    m.charge(work=16 * len(bindings) + 1, entries=12 * len(bindings) + 1)
    result, lookup, used_rows, previous_end = [], {}, set(), 0
    width = hz.n_cont + hz.n_bin
    for binding in bindings:
        if type(binding) is not RebaseBinding:
            raise Rejected("actual RebaseBinding required")
        receipt, rows = binding.receipt, binding.eq_rows
        if (type(receipt) is not RebaseReceipt or receipt._authority is not _OWNED
                or receipt.budget is not budget or receipt.frame_id != hz.frame_id
                or not 0 <= receipt.n_bin <= hz.n_bin):
            raise Rejected("owned same-frame receipt and shared Budget required")
        epoch = receipt.epoch
        if (type(epoch) is not LinkEpoch or type(rows) is not tuple
                or type(epoch.old_cont) is not int or type(epoch.new_cont) is not int
                or not previous_end <= epoch.old_cont <= epoch.new_cont <= hz.n_cont
                or len(rows) != epoch.new_cont - epoch.old_cont
                or len(receipt.link_rows) != len(rows)):
            raise Rejected("rebase epochs must be ordered, disjoint and complete")
        previous_end = epoch.new_cont
        m.charge(work=12 * len(rows) + 1, entries=8 * len(rows) + 1)
        for col, eq, (recorded, rhs) in zip(range(epoch.old_cont, epoch.new_cont),
                                           rows, receipt.link_rows):
            if type(eq) is not int or not 0 <= eq < len(hz.b) or eq in used_rows:
                raise Rejected("unique actual rebase equality rows required")
            used_rows.add(eq)
            count = nb._row_size(hz.Ac, hz.Ab, eq)
            m.charge(work=64 * count + 20, entries=12 * count + 20)
            actual = nb._row(hz.Ac, hz.Ab, eq)
            if actual != recorded or nb._fraction(hz.b[eq]) != rhs:
                raise Rejected("current equality does not match owned rebase content")
            pivot = ZERO
            terms = []
            for i, a in actual.continuous:
                if i == col:
                    pivot = _exact._number(a, m)
                elif i >= epoch.old_cont:
                    raise Rejected("rebase equality depends on a same-epoch or future column")
                else:
                    terms.append((i, a))
            if pivot == ZERO or any(i >= receipt.n_bin for i, _ in actual.binary):
                raise Rejected("missing image pivot or future binary support")
            inverse = _exact._div(ONE, pivot, m)
            raw = base._form(_exact._mul(rhs, inverse, m),
                tuple((i, _exact._mul(_exact._neg(a, m), inverse, m)) for i, a in terms)
                + tuple((hz.n_cont + i, _exact._mul(_exact._neg(a, m), inverse, m))
                        for i, a in actual.binary), width, m)
            normal = _normalize_form(raw, lookup, width, m)
            result.append(Definition(col, eq, raw, normal))
            m.charge(work=3, entries=3)
            lookup[col] = normal
    m.charge(work=len(result) + 1, entries=len(result) + 1)
    return tuple(result), MappingProxyType(lookup)


@dataclass(frozen=True)
class Snapshot:
    hz: SparseHZono
    decoder: tuple
    budget: Budget
    original_dimensions: tuple
    fingerprint: str
    definitions: tuple
    normal_forms: object
    bindings: tuple
    _authority: object = field(default=None, repr=False)

    @property
    def n_columns(self):
        return self.hz.n_cont + self.hz.n_bin

    def __post_init__(self):
        if self._authority is not _OWNED:
            raise Rejected("snapshot must be produced by capture")


def _verify(snapshot, m):
    if type(snapshot) is not Snapshot or snapshot._authority is not _OWNED:
        raise Rejected("owned detached terminal snapshot required")
    if _stamp(snapshot.hz, m) != snapshot.fingerprint:
        raise Rejected("terminal snapshot was mutated")


def capture(hz, decoder, *, global_widths, bindings=(), budget=None, enabled=False):
    if not _on(enabled):
        return None
    if budget is None:
        budget = Budget()
    if type(budget) is not Budget:
        raise Rejected("shared Budget required")
    m = budget._branch()
    try:
        before = _stamp(hz, m)
        if (type(global_widths) is not tuple or len(global_widths) != 2
                or any(type(v) is not int or v < 0 for v in global_widths)):
            raise Rejected("complete global widths must be two nonnegative integers")
        nc, nbin = global_widths
        m.integer(nc), m.integer(nbin)
        if nc < hz.n_cont or nbin < hz.n_bin:
            raise Rejected("global high water cannot truncate old factors")
        m.charge(work=nc + nbin, entries=nc + nbin + 20)
        if type(decoder) is not tuple or not decoder:
            raise Rejected("complete nonempty immutable original input readouts required")
        decoded = []
        for value in decoder:
            nb._checked(value, hz.n_cont, hz.n_bin)
            decoded.append(_flat(value, nc, nc + nbin, m))
        _charge_hz(hz, m, 3)
        copied = SparseHZono(hz.c.copy(), _pad(hz.Gc, nc), _pad(hz.Gb, nbin),
            _pad(hz.Ac, nc), _pad(hz.Ab, nbin), hz.b.copy(),
            _pad(hz.Auc, nc), _pad(hz.Aub, nbin), hz.ub.copy(),
            frame_id=hz.frame_id, exact=hz.exact)
        if _stamp(hz, m) != before:
            raise Rejected("source changed during terminal capture")
        definitions, normal_forms = _definitions(copied, bindings, budget, m)
        frozen = _readonly(copied)
        return Snapshot(frozen, tuple(decoded), budget, (hz.n_cont, hz.n_bin),
                        _stamp(frozen, m), definitions, normal_forms, bindings, _OWNED)
    except MemoryError:
        budget._reject("native terminal capture allocation failed")
    except (nb.KernelError, TypeError, ValueError, IndexError, AttributeError, OverflowError) as exc:
        _legacy_failure(exc, budget)
        raise Rejected("invalid terminal capture: " + str(exc)) from exc



def normalize(snapshot, value, *, enabled=False):
    if not _on(enabled):
        return None
    if type(snapshot) is not Snapshot or snapshot._authority is not _OWNED:
        raise Rejected("owned detached terminal snapshot required")
    m = snapshot.budget._branch()
    try:
        _verify(snapshot, m)
        count = nb._shape(value)
        m.charge(work=64 * count + 20, entries=12 * count + 20)
        nb._checked(value, snapshot.hz.n_cont, snapshot.hz.n_bin)
        return _normalize_form(_flat(value, snapshot.hz.n_cont, snapshot.n_columns, m),
                               snapshot.normal_forms, snapshot.n_columns, m)
    except MemoryError:
        snapshot.budget._reject("readout normalization allocation failed")
    except (nb.KernelError, TypeError, ValueError, IndexError, KeyError,
            AttributeError, OverflowError) as exc:
        _legacy_failure(exc, snapshot.budget)
        raise Rejected("invalid readout normalization: " + str(exc)) from exc


@dataclass(frozen=True)
class Product:
    source: base.Affine
    phase_index: int
    lower: F
    upper: F
    value: base.Affine
    column: int
    mid: F
    radius: F


@dataclass(frozen=True)
class Relation:
    x: tuple
    q: tuple
    y: tuple
    alphas: tuple
    scales: tuple
    tau: F
    a: tuple
    b: tuple
    r: base.Affine
    e: base.Affine
    r_bounds: tuple
    e_bounds: tuple
    mixed_bounds: tuple
    event_bounds: tuple
    tail_errors: tuple
    K: base.Affine
    products: tuple
    rows: tuple


def _relation(snapshot, specification, offset, width, m):
    hz, nc, nbin = snapshot.hz, snapshot.hz.n_cont, snapshot.hz.n_bin
    if type(specification) is not tuple or len(specification) != 2:
        raise Rejected("each complete relation has two parent and two child graphs")
    parents, children = specification
    if any(type(g) is not tuple or len(g) != 2 for g in specification):
        raise Rejected("immutable two-parent/two-child populations required")
    graphs = parents + children
    count = 0
    for graph in graphs:
        count += nb._graph_shape(graph, hz, nc, nbin)
    m.charge(work=count * 64 + 20, entries=count * 12 + 20)
    nb._limit(count)
    if len({g.slots[2] for g in graphs}) != 4:
        raise Rejected("four distinct original phase identities required")
    extracted, bounds = [], []
    for graph in graphs:
        f, q, error = nb._extract_graph(hz, graph)
        if error != ZERO:
            raise Rejected("nonzero native gate error is not an exact mixed relation")
        if graph.kind == "extended":
            row = nb._row(hz.Ac, hz.Ab, graph.eq_row)
            L = dict(row.continuous)[graph.slots[0]]
        else:
            row = nb._row(hz.Auc, hz.Aub, graph.le_rows[2])
            L = dict(row.binary)[graph.slots[2]]
        Q = q.bias
        bounds.append((_exact._mul(TWO, L, m), _exact._mul(TWO, Q, m)))
        extracted.append((_flat(f, nc, width, m), _flat(q, nc, width, m)))
    all_private = {i for g in graphs for i in (g.slots[0], g.slots[1], nc + g.slots[2])}
    child_private = {i for g in children for i in (g.slots[0], g.slots[1], nc + g.slots[2])}
    for index, (f, _) in enumerate(extracted):
        forbidden = all_private if index < 2 else child_private
        if any(i in forbidden for i, _ in f.terms):
            raise Rejected("selected internal factor remains in an earlier source")
    m.charge(work=len(all_private) + 1)
    if any(i in snapshot.normal_forms for i in all_private):
        raise Rejected("rebase substitution cannot target selected gate-private columns")
    extracted = [(_normalize_form(f, snapshot.normal_forms, width, m), q)
                 for f, q in extracted]
    for index, (f, _) in enumerate(extracted):
        forbidden = all_private if index < 2 else child_private
        if any(i in forbidden for i, _ in f.terms):
            raise Rejected("normalized source reveals a forbidden internal dependency")
    def linear(*pieces, constant=ZERO):
        return base._linear(pieces, width, m, constant=constant)
    scales = tuple(base._max(_exact._neg(lo, m), hi, m) for lo, hi in bounds[:2])
    inverses = tuple(_exact._div(ONE, s, m) for s in scales)
    x = tuple(linear((inv, v[0])) for inv, v in zip(inverses, extracted[:2]))
    q = tuple(linear((inv, v[1])) for inv, v in zip(inverses, extracted[:2]))
    y = tuple(v[1] for v in extracted[2:])
    alphas = tuple(base._form(HALF, ((nc + g.slots[2], -HALF),), width, m) for g in parents)
    g1, g2 = tuple(v[0] for v in extracted[2:])
    mean, diff = linear((HALF, g1), (HALF, g2)), linear((HALF, g1), (-HALF, g2))
    def qcoeff(form, i):
        column = parents[i].slots[1]
        return _exact._div(base._coef(form, column, m), base._coef(q[i], column, m), m)
    b = tuple(qcoeff(mean, i) for i in range(2))
    source = linear((ONE, mean), (-b[0], q[0]), (-b[1], q[1]))
    source_occurrences = sum(len(v.terms) for v in (*x, source))
    m.charge(work=source_occurrences * (source_occurrences.bit_length() + 8) + 10,
             entries=12 * source_occurrences + 20)
    maps = tuple(dict(v.terms) for v in (*x, source))
    columns = tuple(sorted(set(maps[0]) | set(maps[1]) | set(maps[2])))
    sums = [ZERO] * 5
    for col in columns:
        v1, v2, vs = (v.get(col, ZERO) for v in maps)
        for i, (left, right) in enumerate(((v1, v1), (v1, v2), (v2, v2), (v1, vs), (v2, vs))):
            sums[i] = _exact._add(sums[i], _exact._mul(left, right, m), m)
    G11, G12, G22, h1, h2 = sums
    det = _exact._sub(_exact._mul(G11, G22, m), _exact._mul(G12, G12, m), m)
    if _exact._cmp(det, ZERO, m) <= 0:
        raise Rejected("native complete source Gram is not rank two")
    a = (_exact._div(_exact._sub(_exact._mul(G22, h1, m), _exact._mul(G12, h2, m), m), det, m),
         _exact._div(_exact._sub(_exact._mul(G11, h2, m), _exact._mul(G12, h1, m), m), det, m))
    tau = _exact._mul(HALF, _exact._sub(qcoeff(diff, 0), qcoeff(diff, 1), m), m)
    if _exact._cmp(tau, ZERO, m) <= 0:
        raise Rejected("fixed child difference requires positive tau")
    r = linear((ONE, source), (-a[0], x[0]), (-a[1], x[1]))
    e = linear((ONE, diff), (-tau, q[0]), (tau, q[1]))
    m.charge(work=snapshot.n_columns, entries=snapshot.n_columns)
    cube = ((NEG, ONE),) * snapshot.n_columns
    rb, eb = tuple(base._bound_terms(v.constant, v.terms, cube, m) for v in (r, e))
    ab = tuple(_exact._add(aa, bb, m) for aa, bb in zip(a, b))
    l10 = _exact._sub(base._min(ZERO, ab[0], m), base._max(ZERO, a[1], m), m)
    u10 = _exact._sub(base._max(ZERO, ab[0], m), base._min(ZERO, a[1], m), m)
    l01 = _exact._add(-base._max(ZERO, a[0], m), base._min(ZERO, ab[1], m), m)
    u01 = _exact._add(-base._min(ZERO, a[0], m), base._max(ZERO, ab[1], m), m)
    event_bounds = (
        (_exact._add(rb[0], l10, m), _exact._add(rb[1], u10, m)),
        (_exact._add(rb[0], l01, m), _exact._add(rb[1], u01, m)),
    )
    mixed = (base._min(event_bounds[0][0], event_bounds[1][0], m),
             base._max(event_bounds[0][1], event_bounds[1][1], m))
    # This one formula also covers the old guard: all four entries then equal
    # zero. Bounds are complete same-H bounds, never conditional solver calls.
    tail_errors = tuple((
        base._max(ZERO, _exact._sub(_exact._neg(lo, m), tau, m), m),
        base._max(ZERO, _exact._sub(hi, tau, m), m),
    ) for lo, hi in event_bounds)
    sources = (x[1], x[0], q[1], q[0], r, r)
    intervals = ((NEG, ONE), (NEG, ONE), (ZERO, ONE), (ZERO, ONE), rb, rb)
    products, rows = [], []
    for i, (v, (lo, hi)) in enumerate(zip(sources, intervals)):
        low, high = base._min(ZERO, lo, m), base._max(ZERO, hi, m)
        mid = _exact._mul(HALF, _exact._add(low, high, m), m)
        rad = _exact._mul(HALF, _exact._sub(high, low, m), m)
        col = offset + i
        value = base._form(mid, () if rad == ZERO else ((col, rad),), width, m)
        alpha = alphas[i % 2]
        products.append(Product(v, nc + parents[i % 2].slots[2], lo, hi, value, col, mid, rad))
        mc = ((linear((NEG, value), (lo, alpha)), ZERO),
              (linear((ONE, value), (-hi, alpha)), ZERO),
              (linear((NEG, value), (ONE, v), (hi, alpha)), hi),
              (linear((ONE, value), (NEG, v), (-lo, alpha)), -lo))
        rows.extend(base._le(form, rhs, width, m) for form, rhs in mc)
    c12, c21, d12, d21, v1, v2 = tuple(p.value for p in products)
    K = linear((tau, alphas[0]), (-tau, alphas[1]), (a[0], q[0]), (-a[0], c21),
        (a[1], c12), (-a[1], q[1]), (b[0], q[0]), (-b[0], d21),
        (b[1], d12), (-b[1], q[1]), (ONE, v1), (NEG, v2))
    difference = linear((ONE, y[0]), (NEG, y[1]), (NEG, K))
    p = tuple(linear((ONE, alpha), (NEG, qq)) for alpha, qq in zip(alphas, q))
    two_tau = _exact._mul(TWO, tau, m)
    rows.extend((base._le(linear((ONE, difference), (-two_tau, p[1]),
                            (_exact._neg(tail_errors[0][0], m), alphas[0]),
                            (_exact._neg(tail_errors[1][1], m), alphas[1])),
                         _exact._mul(TWO, base._max(ZERO, eb[1], m), m), width, m),
                 base._le(linear((NEG, difference), (-two_tau, p[0]),
                            (_exact._neg(tail_errors[0][1], m), alphas[0]),
                            (_exact._neg(tail_errors[1][0], m), alphas[1])),
                         _exact._mul(-TWO, base._min(ZERO, eb[0], m), m), width, m)))
    T = linear((ONE, q[0]), (NEG, q[1]), (ONE, c12), (NEG, c21))
    deficits = tuple(linear((-TWO, qq), (ONE, xx), constant=ONE) for qq, xx in zip(q, x))
    U = linear((ONE, q[0]), (ONE, q[1]), (NEG, c12), (NEG, c21))
    D = linear((TWO, q[0]), (TWO, q[1]), (NEG, x[0]), (NEG, x[1]))
    rows.extend(base._le(v, ZERO, width, m) for v in (
        linear((ONE, T), (NEG, deficits[1])), linear((NEG, T), (NEG, deficits[0])),
        linear((ONE, U), (NEG, D))))
    return Relation(x, q, y, alphas, scales, tau, a, b, r, e, rb, eb, mixed,
                    event_bounds, tail_errors, K, tuple(products), tuple(rows))


def _holds_old(snapshot, values, integral, m):
    hz, n = snapshot.hz, snapshot.n_columns
    if any(not NEG <= v <= ONE for v in values[:n]):
        return False
    if integral and any(v not in (NEG, ONE) for v in values[hz.n_cont:n]):
        return False
    for ac, ab, rhs, equality in ((hz.Ac, hz.Ab, hz.b, True), (hz.Auc, hz.Aub, hz.ub, False)):
        for i in range(len(rhs)):
            form = _flat(nb._row(ac, ab, i), hz.n_cont, n, m)
            at = base._at(form, values, m)
            bound = nb._fraction(rhs[i])
            if (at != bound if equality else at > bound):
                return False
    return True


def _audit_lowering(hz, model, m):
    """The legacy signed-to-0/1 translation is accepted only when exact.

    In particular, fl(sum(Ab)) can change an old equality even for ordinary
    .1/.2 coefficients. Sound signed rows alone do not certify that conversion.
    This is validation of the one normal lowering, never a second solve/path.
    """
    nc, nbin = hz.n_cont, hz.n_bin
    if (model.n_cont != nc or model.n_bin != nbin
            or tuple(model.cont_source) != tuple(range(nc))
            or tuple(model.bin_source) != tuple(range(nbin))
            or model.bin_fixes or model.cont_eliminations
            or model.A.shape != (len(hz.b) + len(hz.ub), nc + nbin)
            or model.value_matrix.shape != (hz.n_out, nc + nbin)):
        raise Rejected("normal terminal lowering changed the factor population")
    m.charge(work=10 * (nc + nbin), entries=8 * (nc + nbin))
    if (tuple(model.var_lb) != (-1.0,) * nc + (0.0,) * nbin
            or tuple(model.var_ub) != (1.0,) * (nc + nbin)
            or tuple(model.integrality) != (0,) * nc + (1,) * nbin):
        raise Rejected("normal terminal lowering changed domains or integrality")
    def translated_row(ac, ab, row):
        original = nb._row(ac, ab, row)
        m.charge(work=4 * (len(original.continuous) + len(original.binary)) + 1,
                 entries=4 * (len(original.continuous) + len(original.binary)) + 1)
        shift = ZERO
        binary = []
        for i, value in original.binary:
            shift = _exact._add(shift, value, m)
            binary.append((nc + i, _exact._mul(TWO, value, m)))
        return original.continuous + tuple(binary), shift
    def actual_row(matrix, row):
        start, end = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
        m.charge(work=2 * (end - start) + 1, entries=2 * (end - start) + 1)
        return tuple((int(matrix.indices[p]), nb._fraction(matrix.data[p]))
                     for p in range(start, end) if matrix.data[p] != 0.0)
    for i in range(hz.n_out):
        expected, shift = translated_row(hz.Gc, hz.Gb, i)
        if (actual_row(model.value_matrix, i) != expected
                or nb._fraction(model.value_center[i]) != _exact._sub(nb._fraction(hz.c[i]), shift, m)):
            raise Rejected("normal terminal output translation is not exact")
    offset = 0
    for ac, ab, rhs, equality in ((hz.Ac, hz.Ab, hz.b, True), (hz.Auc, hz.Aub, hz.ub, False)):
        for i in range(len(rhs)):
            expected, shift = translated_row(ac, ab, i)
            target = _exact._add(nb._fraction(rhs[i]), shift, m)
            j = offset + i
            if (actual_row(model.A, j) != expected or nb._fraction(model.row_ub[j]) != target
                    or (nb._fraction(model.row_lb[j]) != target if equality
                        else model.row_lb[j] != -np.inf)):
                raise Rejected("normal terminal predicate translation is not exact")
        offset += len(rhs)


@dataclass(frozen=True)
class Result:
    snapshot: Snapshot
    hz: SparseHZono
    relations: tuple
    exact_rows: tuple
    row_errors: tuple
    installed_rhs: tuple
    stored_rows: tuple
    fingerprint: str
    _authority: object = field(default=None, repr=False)

    def __post_init__(self):
        if self._authority is not _OWNED:
            raise Rejected("result must be produced by attach")

    @property
    def budget(self):
        return self.snapshot.budget

    @property
    def n_columns(self):
        return self.snapshot.n_columns + 6 * len(self.relations)

    def _verify(self, m):
        _verify(self.snapshot, m)
        if _stamp(self.hz, m) != self.fingerprint:
            raise Rejected("enhanced terminal state was mutated")

    def _holds(self, values, integral, m):
        if not _holds_old(self.snapshot, values, integral, m):
            return False
        if any(not NEG <= v <= ONE for v in values[self.snapshot.n_columns:]):
            return False
        return all(base._row_holds(row, values, False, m) for row in self.stored_rows)

    @_op
    def canonical_extension(self, values, *, _meter):
        self._verify(_meter)
        values = _exact._vector(values, self.snapshot.n_columns, "complete old signed tuple", _meter)
        if not _holds_old(self.snapshot, values, True, _meter):
            raise Rejected("canonical extension requires a full old integer witness")
        extension = []
        for relation in self.relations:
            for p in relation.products:
                alpha = _exact._mul(HALF, _exact._sub(ONE, values[p.phase_index], _meter), _meter)
                true = _exact._mul(alpha, base._at(p.source, values, _meter), _meter)
                extension.append(ZERO if p.radius == ZERO else _exact._div(
                    _exact._sub(true, p.mid, _meter), p.radius, _meter))
        extended = values + tuple(extension)
        if not self._holds(extended, True, _meter):
            raise Rejected("canonical native extension violated stored predicates")
        return extended

    @_op
    def satisfied(self, values, *, integral=False, _meter):
        self._verify(_meter)
        if type(integral) is not bool:
            raise Rejected("integral must be bool")
        values = _exact._vector(values, self.n_columns, "complete extended tuple", _meter)
        return self._holds(values, integral, _meter)

    @_op
    def decode(self, values, *, _meter):
        self._verify(_meter)
        values = _exact._vector(values, self.n_columns, "complete extended tuple", _meter)
        if not self._holds(values, True, _meter):
            raise Rejected("decoder requires a valid complete integer witness")
        return tuple(base._at(f, values, _meter) for f in self.snapshot.decoder)

    @_op
    def terminal_model(self, *, _meter):
        self._verify(_meter)
        _charge_hz(self.hz, _meter, 8)
        _meter.charge(work=10 * self.n_columns, entries=10 * self.n_columns)
        model = _lower_hz_milp(self.hz, prune_unused=False, coalesce_rows=False,
                               project_inactive_cont=False, fix_implied_binary=False)
        _audit_lowering(self.hz, model, _meter)
        return model


def attach(snapshot, *, population, enabled=False):
    if not _on(enabled):
        return None
    if type(snapshot) is not Snapshot or snapshot._authority is not _OWNED:
        raise Rejected("owned detached terminal snapshot required")
    m = snapshot.budget._branch()
    try:
        _verify(snapshot, m)
        if type(population) is not tuple or not population:
            raise Rejected("fixed nonempty whole relation population required")
        m.charge(work=64 * len(population), entries=60 * len(population))
        if len(set(population)) != len(population):
            raise Rejected("duplicate relation population entry")
        hz = snapshot.hz
        old, count = snapshot.n_columns, 6 * len(population)
        width = old + count
        m.integer(width)
        m.charge(work=width, entries=width)
        relations = tuple(_relation(snapshot, spec, old + 6 * i, width, m)
                          for i, spec in enumerate(population))
        exact_rows = tuple(row for relation in relations for row in relation.rows)
        native = tuple(_native(row, hz.n_cont, hz.n_bin, width, m) for row in exact_rows)
        occurrences = sum(len(row.terms) for row in exact_rows)
        nb._limit(occurrences)
        m.charge(work=200 * occurrences + 100, entries=24 * occurrences + 100)
        ac, ab, rhs, errors, installed, _ = rounding._rounded_rows(native, hz.n_cont + count, hz.n_bin)
        for value in errors + installed:
            _exact._number(value, m)
        stored = tuple(base._le(_stored_flat(nb._row(ac, ab, i), hz.n_cont + count,
            hz.n_cont, hz.n_bin, width, m), nb._fraction(rhs[i]), width, m)
            for i in range(len(rhs)))
        _charge_hz(hz, m, 4)
        enhanced = SparseHZono(hz.c.copy(), _pad(hz.Gc, hz.n_cont + count), hz.Gb.copy(),
            _pad(hz.Ac, hz.n_cont + count), hz.Ab.copy(), hz.b.copy(),
            sp.vstack((_pad(hz.Auc, hz.n_cont + count), ac), format="csr"),
            sp.vstack((hz.Aub, ab), format="csr"), np.concatenate((hz.ub, rhs)),
            frame_id=hz.frame_id, exact=hz.exact)
        _verify(snapshot, m)
        _readonly(enhanced)
        return Result(snapshot, enhanced, relations, exact_rows, errors, installed, stored,
                      _stamp(enhanced, m), _OWNED)
    except MemoryError:
        snapshot.budget._reject("native mixed batch allocation failed")
    except (nb.KernelError, TypeError, ValueError, IndexError, KeyError, AttributeError, OverflowError) as exc:
        _legacy_failure(exc, snapshot.budget)
        raise Rejected("native mixed batch rejected: " + str(exc)) from exc
