"""Default-off exact pullback of D228 into one unchanged native HZ.

This is a mathematical native-coordinate bridge, not ONNX/model binding,
native installation, a global factor allocator, GPU qualification, or a new
domain/verification result.  D064's original predicate templates establish
four exact ReLU graphs.  Its D063 bound generator is NEVER called here.
Native float64 values mean their exact stored binary rationals.

Physical order is (f1, f2, q1, q2, v1, v2, y1, y2).  Each v is the COMPLETE
actual child preactivation minus the declared rational parent readout and
bias.  Cube intervals build only a local envelope: v is never replaced in H
by an independent source.  Local vertices need not have original-input
witnesses.  All H rows, outputs, consumers, original bits and decoder remain
untouched; no per-vertex original-source allocation is imposed.

``frame`` must be the original HZ object itself, by identity.  A charged
SHA-256 fingerprint of all native numerical fields is checked before and
after each public operation.  This detects ordinary source mutation; it is
not authenticated provenance or protection against adversarial hash attacks.
Concurrent mutation and concurrent use of a Budget are unsupported.
Schema-validated buffers must be one-dimensional and C-contiguous; hashing
uses byte views, not full byte copies.  The retained original H scalar root
is charged once per bind; later scans charge work and their new metadata and
views, not nonexistent copies of the original source.

Returned rows use the old native continuous and binary columns, followed by
fresh LOCAL continuous xi columns; lambda=(1+xi)/2 and xi in [-1,1].  Only
the separate D228 signed phase slots change convention: sigma=-native_bit.
NativeAffine binary coefficients already use native convention.

The 65536 sparse-occurrence limit applies to aggregate selected graph input,
each affine combination and each expanded row, not to the entire returned
matrix.  All source scans, expansion, arithmetic, queries and returned data
are charged to the persistent D228 Budget, including nested operations.
These are logical counters, NOT complete Python/time/storage accounting.
The result records are trusted in-memory products of the public functions;
direct construction, frozen bypasses and untrusted deserialization are not
supported certificate-production paths.
"""

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import math

from experiments.neural_hz_20260831.definition_first_20260928.d064_native_predicate_binding_20261001 import native_binding as native
from experiments.neural_hz_20260831.definition_first_20260928.d228_joint_bank_component_20261005 import bank as joint


Rejected = joint.Rejected
Budget = joint.Budget
Limits = joint.Limits
NativeAffine = native.NativeAffine
Graph = native.Graph
ZERO, ONE, HALF = Fraction(0), Fraction(1), Fraction(1, 2)
MAX_OCCURRENCES = 65536


@dataclass(frozen=True)
class Bound:
    source_hz: object = field(repr=False, compare=False)
    bank: joint.Bank
    physical: tuple
    phase_columns: tuple
    old_n_cont: int
    old_n_bin: int
    budget: Budget = field(repr=False, compare=False)
    _fingerprint: str = field(repr=False, compare=False)


@dataclass(frozen=True)
class RatRow:
    """An EQ/LE left affine (zero bias), interpreted by its containing tuple."""

    affine: NativeAffine
    rhs: Fraction


@dataclass(frozen=True)
class Pulled:
    """Only NEW rows; old H and all its original rows must remain present."""

    eq: tuple
    le: tuple
    n_cont: int
    n_bin: int
    old_n_cont: int
    n_lambda: int


@dataclass(frozen=True)
class ReceiverBounds:
    lower: Fraction
    upper: Fraction
    remainder: NativeAffine


def _limit(count, meter):
    meter.charge(work=1)
    if count > MAX_OCCURRENCES:
        meter.budget._reject("native sparse occurrence limit exceeded")
    return count


def _size(value):
    return len(value.continuous) + len(value.binary)


def _tuple(value, length, name):
    if type(value) is not tuple or len(value) != length:
        raise Rejected(name + " must be a tuple of length " + str(length))
    return value


def _stored(value, meter):
    """Only native schema-checked float64 scalars use float conversion."""
    meter.charge(work=2, entries=1)
    value = float(value)
    if not math.isfinite(value):
        raise Rejected("nonfinite stored native coefficient")
    return joint._number(Fraction.from_float(value), meter)


def _source(hz, meter, *, retain_source=False):
    """Charge complete scans; hash checked contiguous buffers without copies."""
    meter.charge(work=64, entries=128)
    if type(hz) is not native.SparseHZono:
        raise Rejected("an actual SparseHZono is required")
    arrays = []
    for name in ("c", "b", "ub"):
        value = getattr(hz, name)
        if type(value) is not native.np.ndarray:
            raise Rejected("native vectors must be ndarrays")
        arrays.append((name, value))
    for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
        matrix = getattr(hz, name)
        if type(matrix) is not native.sp.csr_matrix:
            raise Rejected("native matrices must be CSR")
        for part in ("data", "indices", "indptr"):
            value = getattr(matrix, part)
            if type(value) is not native.np.ndarray:
                raise Rejected("native CSR buffers must be ndarrays")
            arrays.append((name + "." + part, value))
    scalar_count = sum(int(value.size) for _, value in arrays)
    byte_count = sum(int(value.nbytes) for _, value in arrays)
    # Covers native._hz_schema's complete scans and this fingerprint scan.
    meter.charge(work=8 * scalar_count + byte_count + 64,
                 entries=(scalar_count if retain_source else 0) + 64)
    nc, nb = native._hz_schema(hz)
    if any(value.ndim != 1 or not value.flags.c_contiguous for _, value in arrays):
        raise Rejected("fingerprinting requires C-contiguous native buffers")
    digest = hashlib.sha256()

    def header(value):
        # One transient string and its encoded metadata; no source byte copy.
        meter.charge(work=len(value) + 1, entries=2 * len(value) + 3)
        digest.update(value.encode("ascii"))

    header(str(hz.frame_id) + ":" + str(hz.exact))
    for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
        header(name + str(getattr(hz, name).shape))
    for name, value in arrays:
        header(name + str(value.shape) + value.dtype.str)
        if value.nbytes:
            meter.charge(work=2, entries=2)
            digest.update(memoryview(value).cast("B"))
    meter.charge(work=1, entries=65)
    return nc, nb, digest.hexdigest()


def _row(cont, binary, index, bias, meter):
    count = _limit(native._row_size(cont, binary, index), meter)
    meter.charge(work=count + 2, entries=2 * count + 3)
    groups = []
    for matrix in (cont, binary):
        start, stop = int(matrix.indptr[index]), int(matrix.indptr[index + 1])
        groups.append(tuple((int(matrix.indices[p]), _stored(matrix.data[p], meter))
                            for p in range(start, stop)))
    return NativeAffine(joint._number(bias, meter), groups[0], groups[1])


def _affine(bias, cont, binary, meter):
    count = _limit(len(cont) + len(binary), meter)
    meter.charge(work=(count + 1) * (count.bit_length() + 1),
                 entries=2 * count + 3)
    return NativeAffine(bias, tuple(sorted(cont.items())),
                        tuple(sorted(binary.items())))


def _combine(bias, scaled, meter):
    """Budgeted D064 affine arithmetic, including low caller bit limits."""
    count = 0
    for scale, value in scaled:
        joint._number(scale, meter)
        count = _limit(count + _size(value), meter)
    meter.charge(work=len(scaled), entries=len(scaled))
    bias = joint._number(bias, meter)
    cont, binary = {}, {}
    for scale, value in scaled:
        bias = joint._add(bias, joint._mul(scale, value.bias, meter), meter)
        for terms, merged in ((value.continuous, cont), (value.binary, binary)):
            for index, amount in terms:
                meter.charge(work=1, entries=1)
                result = joint._add(merged.get(index, ZERO),
                                    joint._mul(scale, amount, meter), meter)
                if result == ZERO:
                    merged.pop(index, None)
                else:
                    merged[index] = result
    return _affine(bias, cont, binary, meter)


def _cube(value, meter):
    radius = ZERO
    for terms in (value.continuous, value.binary):
        for _, amount in terms:
            meter.charge(work=1)
            magnitude = joint._neg(amount, meter) if amount < ZERO else amount
            radius = joint._add(radius, magnitude, meter)
    return (joint._sub(value.bias, radius, meter),
            joint._add(value.bias, radius, meter))


def _guard(hz, index, cont, binary, meter):
    actual = _row(hz.Auc, hz.Aub, index, ZERO, meter)
    meter.charge(work=_size(actual) + 3, entries=len(cont) + len(binary) + 3)
    expected = NativeAffine(ZERO, tuple(sorted(cont)), tuple(sorted(binary)))
    if actual != expected or _stored(hz.ub[index], meter) != ZERO:
        raise Rejected("original phase guard row mismatch")


def _extract(hz, graph, meter):
    """D064's exact templates, with every rational operation metered here."""
    s, eta, z = graph.slots
    minus_one = joint._neg(ONE, meter)
    if graph.kind == "extended":
        equation = _row(hz.Ac, hz.Ab, graph.eq_row, ZERO, meter)
        meter.charge(work=_size(equation), entries=_size(equation))
        cont, binary = dict(equation.continuous), dict(equation.binary)
        lower = cont.pop(s, ZERO)
        amplitude = joint._neg(cont.pop(eta, ZERO), meter)
        if lower >= ZERO or amplitude <= ZERO or binary.pop(z, ZERO) != lower:
            raise Rejected("extended original equality template mismatch")
        _guard(hz, graph.le_rows[0], ((s, minus_one),), ((z, minus_one),), meter)
        _guard(hz, graph.le_rows[1], ((eta, minus_one),), ((z, ONE),), meter)
        remaining = _affine(ZERO, cont, binary, meter)
        center = joint._add(_stored(hz.b[graph.eq_row], meter), amplitude, meter)
        preactivation = _combine(center, ((minus_one, remaining),), meter)
    else:
        _guard(hz, graph.le_rows[0], ((eta, minus_one),), ((z, ONE),), meter)
        lower = _row(hz.Auc, hz.Aub, graph.le_rows[1], ZERO, meter)
        upper = _row(hz.Auc, hz.Aub, graph.le_rows[2], ZERO, meter)
        meter.charge(work=_size(lower) + _size(upper),
                     entries=_size(lower) + _size(upper))
        low_c, low_b = dict(lower.continuous), dict(lower.binary)
        high_c, high_b = dict(upper.continuous), dict(upper.binary)
        amplitude = low_c.pop(eta, ZERO)
        lower_bound = high_b.pop(z, ZERO)
        if (amplitude <= ZERO or lower_bound >= ZERO or z in low_b
                or high_c.pop(eta, ZERO) != joint._neg(amplitude, meter)):
            raise Rejected("compact original inequality template mismatch")
        positive = _affine(ZERO, low_c, low_b, meter)
        negative = _combine(ZERO, ((minus_one, positive),), meter)
        expected = _affine(ZERO, high_c, high_b, meter)
        if negative != expected:
            raise Rejected("compact coefficients are not exact negatives")
        rhs1 = _stored(hz.ub[graph.le_rows[1]], meter)
        rhs2 = _stored(hz.ub[graph.le_rows[2]], meter)
        delta = joint._add(joint._add(rhs1, rhs2, meter), lower_bound, meter)
        if delta != ZERO:
            raise Rejected("nonzero native graph_error is not supported")
        preactivation = NativeAffine(joint._sub(amplitude, rhs1, meter),
                                     positive.continuous, positive.binary)
        meter.charge(work=1, entries=3)
    output = NativeAffine(amplitude, ((eta, joint._neg(amplitude, meter)),), ())
    meter.charge(work=1, entries=5)
    return preactivation, output


def _budget(value):
    if value is None:
        return Budget()
    if type(value) is not Budget:
        raise Rejected("a D228 Budget is required")
    return value


def _finish_source(hz, fingerprint, meter):
    if _source(hz, meter)[2] != fingerprint:
        raise Rejected("native source changed during the operation")
    # Nested D228 calls have their own branch meters; enforce this whole call.
    meter.charge(work=0)


def bind(hz, graphs, coefficients, biases, *, enabled=False, budget=None):
    """Bind four original exact graphs (parent, parent, child, child)."""
    if enabled is not True:
        raise Rejected("source-bound bank is disabled unless enabled=True")
    budget = _budget(budget)
    meter = budget._branch()
    try:
        _tuple(graphs, 4, "graphs")
        coefficients = tuple(
            tuple(joint._number(v, meter) for v in _tuple(row, 4, "coefficient row"))
            for row in _tuple(coefficients, 2, "coefficients"))
        biases = tuple(joint._number(v, meter) for v in _tuple(biases, 2, "biases"))
        meter.charge(work=14, entries=14)
        nc, nb, fingerprint = _source(hz, meter, retain_source=True)
        count = 0
        for graph in graphs:
            meter.charge(work=16, entries=4)
            count = _limit(count + native._graph_shape(graph, hz, nc, nb), meter)
        phases = tuple(graph.slots[2] for graph in graphs)
        if len(set(phases)) != 4:
            raise Rejected("four distinct original phase columns are required")
        extracted = tuple(_extract(hz, graph, meter) for graph in graphs)
        base = (extracted[0][0], extracted[1][0],
                extracted[0][1], extracted[1][1])
        residuals = []
        for j in range(2):
            scaled = ((ONE, extracted[j + 2][0]),) + tuple(
                (joint._neg(amount, meter), value)
                for amount, value in zip(coefficients[j], base))
            residuals.append(_combine(joint._neg(biases[j], meter), scaled, meter))
        physical = base + tuple(residuals) + (extracted[2][1], extracted[3][1])
        parents = tuple(_cube(value, meter) for value in base[:2])
        residual_bounds = tuple(_cube(value, meter) for value in residuals)
        spec_coefficients = (coefficients[0] + (ONE, ZERO),
                             coefficients[1] + (ZERO, ONE))
        meter.charge(work=40, entries=40)
        spec = joint.Spec(parents, residual_bounds, spec_coefficients, biases)
        identity = joint.Binding(
            hz, ("parent:f1", "parent:f2", "parent:q1", "parent:q2",
                 "actual:v1", "actual:v2", "child:y1", "child:y2"),
            tuple("native:phase:" + str(index) for index in phases))
        bank = joint.build(spec, identity, enabled=True, budget=budget)
        _finish_source(hz, fingerprint, meter)
        meter.charge(work=1, entries=16)
        return Bound(hz, bank, physical, phases, nc, nb, budget, fingerprint)
    except native.KernelError as exc:
        raise Rejected(str(exc)) from exc
    except (AttributeError, IndexError, TypeError, OverflowError) as exc:
        raise Rejected("invalid native binding: " + str(exc)) from exc


def _start(bound, frame):
    if type(bound) is not Bound:
        raise Rejected("a trusted Bound returned by bind is required")
    if frame is not bound.source_hz:
        raise Rejected("frame must be the identical original native HZ")
    meter = bound.budget._branch()
    nc, nb, fingerprint = _source(bound.source_hz, meter)
    if (nc != bound.old_n_cont or nb != bound.old_n_bin
            or fingerprint != bound._fingerprint):
        raise Rejected("native source changed after binding")
    return meter


def _pull_row(row, bound, n_lambda, meter):
    scaled = []
    expanded = 0
    for index, coefficient in row.terms:
        meter.charge(work=1, entries=2)
        if index < 8:
            value = bound.physical[index]
        elif index < 12:
            value = NativeAffine(ZERO, (),
                                 ((bound.phase_columns[index - 8], joint._neg(ONE, meter)),))
            meter.charge(work=1, entries=5)
        elif index < 12 + n_lambda:
            value = NativeAffine(HALF,
                                 ((bound.old_n_cont + index - 12, HALF),), ())
            meter.charge(work=1, entries=5)
        else:
            raise Rejected("bank row column outside its local layout")
        expanded = _limit(expanded + _size(value), meter)
        scaled.append((coefficient, value))
    expression = _combine(ZERO, tuple(scaled), meter)
    rhs = joint._sub(row.rhs, expression.bias, meter)
    meter.charge(work=1, entries=5)
    return RatRow(NativeAffine(ZERO, expression.continuous, expression.binary), rhs)


def pullback(bound, *, frame):
    """Return NEW exact EQ/LE; do not install, round or allocate runtime slots."""
    try:
        meter = _start(bound, frame)
        compiled = bound.bank.compile(frame=frame)
        meter.charge(work=0)
        eq = tuple(_pull_row(row, bound, compiled.n_lambda, meter) for row in compiled.eq)
        le = tuple(_pull_row(row, bound, compiled.n_lambda, meter) for row in compiled.le)
        _finish_source(bound.source_hz, bound._fingerprint, meter)
        meter.charge(work=len(eq) + len(le) + 1, entries=len(eq) + len(le) + 6)
        return Pulled(eq, le, bound.old_n_cont + compiled.n_lambda,
                      bound.old_n_bin, bound.old_n_cont, compiled.n_lambda)
    except native.KernelError as exc:
        raise Rejected(str(exc)) from exc
    except (AttributeError, IndexError, TypeError, OverflowError) as exc:
        raise Rejected("invalid native pullback: " + str(exc)) from exc


def receiver_bounds(bound, receiver_index, physical_coefficients, *, frame):
    """Bound a COMPLETE original H output using one fixed bank decomposition.

    receiver = coefficients dot physical + remainder is an exact native
    identity.  The local bank support and the full remainder cube bound are
    both paid.  No coefficient search, LP, correction helper or rescue is
    used.  Correlations between the two summands are relaxed, so this is a
    sound bound, not the exact support of the complete correlated H.
    """
    try:
        meter = _start(bound, frame)
        native._index(receiver_index, len(bound.source_hz.c))
        weights = tuple(joint._number(value, meter) for value in
                        _tuple(physical_coefficients, 8, "physical_coefficients"))
        meter.charge(work=8, entries=8)
        hz = bound.source_hz
        receiver = _row(hz.Gc, hz.Gb, receiver_index,
                        _stored(hz.c[receiver_index], meter), meter)
        scaled = ((ONE, receiver),) + tuple(
            (joint._neg(weight, meter), value)
            for weight, value in zip(weights, bound.physical))
        remainder = _combine(ZERO, scaled, meter)
        rem_lower, rem_upper = _cube(remainder, meter)
        upper_local = bound.bank.support(weights, frame=frame).value
        opposite = tuple(joint._neg(value, meter) for value in weights)
        lower_local = joint._neg(bound.bank.support(opposite, frame=frame).value, meter)
        lower = joint._add(lower_local, rem_lower, meter)
        upper = joint._add(upper_local, rem_upper, meter)
        if joint._cmp(lower, upper, meter) > 0:
            raise Rejected("inconsistent receiver bounds")
        _finish_source(hz, bound._fingerprint, meter)
        meter.charge(work=1, entries=3)
        return ReceiverBounds(lower, upper, remainder)
    except native.KernelError as exc:
        raise Rejected(str(exc)) from exc
    except (AttributeError, IndexError, TypeError, OverflowError) as exc:
        raise Rejected("invalid receiver binding: " + str(exc)) from exc
