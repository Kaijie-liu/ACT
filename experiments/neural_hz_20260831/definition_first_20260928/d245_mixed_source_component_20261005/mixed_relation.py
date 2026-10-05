"""Default-off mixed source/phase capacity on one sealed D243 mathematical H.

This is not an ONNX/native adapter, solver, phase pool or model certificate.
The original source, binary identities, predicates, consumers and decoder are
retained.  A fixed exact source Gram extraction is followed by one mixed guard;
there is no X/Q fallback, optimized tau, support query or alternative rescue.

Only the already owned D243 source representation and its paid exact helpers
are reused.  Full source expansion, every Gram entry, all residual terms and
terminal materialization remain chargeable.  The logical Budget does not claim
complete Python/native/GPU accounting or qualification on the three real models.
Source declarations and BN intervals retain D243's mathematical-only scope.

Relation records are trusted in-memory results of attach, not an untrusted
serialization format.  Ordinary attachment rejection leaves the sealed H usable;
resource failure remains sticky and no partially constructed result is published.
"""

from dataclasses import dataclass, field

from experiments.neural_hz_20260831.definition_first_20260928.d243_operator_source_relation_20261005 import source_relation as base


Source, H = base.Source, base.H
Scalar, Gate, Tensor = base.Scalar, base.Gate, base.Tensor
Affine, Row, Materialized, Product = base.Affine, base.Row, base.Materialized, base.Product
Budget, Limits, Rejected = base.Budget, base.Limits, base.Rejected
ZERO, ONE, TWO, HALF, NEG = base.ZERO, base.ONE, base.TWO, base.HALF, base.NEG
_exact = base._exact
_OWNED = object()


def _independent_parents(h, targets, selected_q, m):
    """Reject selected-parent dependency through the complete original DAG.

    Unlike the affine frontier expansion, this check also traverses intervening
    ReLU/phase definitions.  It is structural, not a claim of statistical source
    independence.  Two parent forms may share any number of original sources.
    """
    stack, seen = list(targets), set()
    m.charge(work=len(targets) + 1, entries=len(targets) + 2)
    while stack:
        col = stack.pop()
        m.charge(work=3)
        if col in selected_q:
            raise Rejected("a selected parent depends on a selected parent output")
        if col in seen:
            continue
        seen.add(col)
        m.charge(entries=1)
        definition = h.definitions[col]
        if definition.kind in ("affine", "conv"):
            form = base._definition_form(h, col, m)
            m.charge(work=len(form.terms), entries=len(form.terms))
            for child, coefficient in form.terms:
                if coefficient != ZERO:
                    stack.append(child)
        elif definition.kind == "relu":
            stack.append(definition.data[0])
            m.charge(work=1, entries=1)
        elif definition.kind == "phase":
            stack.append(definition.data)
            m.charge(work=1, entries=1)
        elif definition.kind not in ("input", "error"):
            raise Rejected("unrecognized original source definition")


def _source_gram(h, forms, m):
    """The fixed radius-weighted 2x2 extraction; constants stay in the residual."""
    count = sum(len(f.terms) for f in forms)
    m.charge(work=count * (count.bit_length() + 2) + 12,
             entries=6 * count + 20)
    maps = tuple(dict(f.terms) for f in forms)
    columns = tuple(sorted(set(maps[0]) | set(maps[1]) | set(maps[2])))
    g11 = g12 = g22 = h1 = h2 = ZERO
    radii = []
    for col in columns:
        m.charge(work=5, entries=2)
        lower, upper = h.column_bounds[col]
        radius = _exact._mul(HALF, _exact._sub(upper, lower, m), m)
        radii.append(radius)
        u1 = _exact._mul(maps[0].get(col, ZERO), radius, m)
        u2 = _exact._mul(maps[1].get(col, ZERO), radius, m)
        uv = _exact._mul(maps[2].get(col, ZERO), radius, m)
        g11 = _exact._add(g11, _exact._mul(u1, u1, m), m)
        g12 = _exact._add(g12, _exact._mul(u1, u2, m), m)
        g22 = _exact._add(g22, _exact._mul(u2, u2, m), m)
        h1 = _exact._add(h1, _exact._mul(u1, uv, m), m)
        h2 = _exact._add(h2, _exact._mul(u2, uv, m), m)
    det = _exact._sub(_exact._mul(g11, g22, m), _exact._mul(g12, g12, m), m)
    if _exact._cmp(det, ZERO, m) <= 0:
        raise Rejected("the fixed complete-source Gram is not rank two")
    a1 = _exact._div(_exact._sub(_exact._mul(g22, h1, m),
                                _exact._mul(g12, h2, m), m), det, m)
    a2 = _exact._div(_exact._sub(_exact._mul(g11, h2, m),
                                _exact._mul(g12, h1, m), m), det, m)
    m.charge(work=len(columns) + 5, entries=len(columns) + 15)
    return columns, tuple(radii), ((g11, g12), (g12, g22)), (h1, h2), det, (a1, a2)


@dataclass(frozen=True)
class Relation:
    parent: H
    parents: tuple
    children: tuple
    scales: tuple
    normalized: tuple
    child_forms: tuple
    source_forms: tuple
    source_columns: tuple
    source_radii: tuple
    gram: tuple
    gram_rhs: tuple
    gram_det: object
    tau: object
    a: tuple
    b: tuple
    z: Affine
    d: Affine
    r: Affine
    e: Affine
    r_bounds: tuple
    e_bounds: tuple
    cone_bounds: tuple
    mixed_bounds: tuple
    K: Affine
    products: tuple
    additional_le: tuple
    product_rows: tuple
    defect_rows: tuple
    capacity_rows: tuple
    column_bounds: tuple
    _authority: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._authority is not _OWNED:
            raise Rejected("Relation is produced only by mixed attach")

    @property
    def frame(self):
        return self.parent.frame

    @property
    def budget(self):
        return self.parent.budget

    @property
    def anchor_kind(self):
        return "mixed"

    @property
    def n_columns(self):
        return len(self.column_bounds)

    @property
    def a1(self):
        return self.a[0]

    @property
    def a2(self):
        return self.a[1]

    @property
    def b1(self):
        return self.b[0]

    @property
    def b2(self):
        return self.b[1]

    def _holds(self, values, integral, m):
        n = self.parent.n_columns
        m.charge(work=n + 12, entries=n + 12)
        if not base._holds_h(self.parent, values[:n], integral, m):
            return False
        for value, (lo, hi) in zip(values[n:], self.column_bounds[n:]):
            if _exact._cmp(value, lo, m) < 0 or _exact._cmp(value, hi, m) > 0:
                return False
        for row in self.additional_le:
            m.charge(work=1 + len(row.terms))
            if not base._row_holds(row, values, False, m):
                return False
        return True

    @base._op
    def canonical_extension(self, values, *, frame, _meter):
        base._check_frame(self, frame)
        m = _meter
        values = _exact._vector(values, self.parent.n_columns, "original H witness", m)
        if not base._holds_h(self.parent, values, True, m):
            raise Rejected("extension requires a complete original integer witness")
        m.charge(work=6, entries=13)
        products = []
        for product in self.products:
            alpha = _exact._mul(HALF, _exact._add(values[product.phase_index], ONE, m), m)
            products.append(_exact._mul(alpha, base._at(product.source, values, m), m))
        m.charge(work=self.n_columns, entries=self.n_columns + 7)
        result = values + tuple(products)
        if not self._holds(result, True, m):
            raise Rejected("canonical mixed extension failed its complete predicates")
        return result

    @base._op
    def satisfied(self, values, *, frame, integral=False, _meter):
        base._check_frame(self, frame)
        if type(integral) is not bool:
            raise Rejected("integral must be bool")
        values = _exact._vector(values, self.n_columns, "extended assignment", _meter)
        return self._holds(values, integral, _meter)

    @base._op
    def decode(self, values, *, frame, _meter):
        base._check_frame(self, frame)
        values = _exact._vector(values, self.n_columns, "extended assignment", _meter)
        if not self._holds(values, True, _meter):
            raise Rejected("decoder requires a complete extended integer witness")
        _meter.charge(work=len(self.parent.input_columns), entries=len(self.parent.input_columns) + 1)
        return tuple(values[i] for i in self.parent.input_columns)

    @base._op
    def materialize(self, *, frame, _meter):
        base._check_frame(self, frame)
        old = base._materialize(self.parent, _meter)
        _meter.charge(work=len(old.eq) + len(old.le) + 41,
                      entries=len(old.eq) + 2 * len(old.le) + 70)
        new_bounds = tuple(base._bound_row(self.column_bounds, 2 * i + side,
                                           self.n_columns, _meter)
                           for i in range(self.parent.n_columns, self.n_columns)
                           for side in (0, 1))
        return Materialized(old.eq, old.le + new_bounds + self.additional_le,
                            self.n_columns, self.column_bounds, self.parent.binary_columns)


def attach(h, *, parents, children, enabled=False, frame):
    """Attach the one fixed source-Gram/mixed-guard factor; no caller tau/menu."""
    if type(h) is not H or h._authority is not base._OWNED:
        raise Rejected("attach requires a sealed owned D243 H")
    m = h.budget._branch()
    try:
        if enabled is not True:
            raise Rejected("mixed relation attachment is default-off")
        base._check_frame(h, frame)
        parents = base._tuple(parents, 2, "parent gates")
        children = base._tuple(children, 2, "child gates")
        m.charge(work=4 * len(h.gates) + 16, entries=24)
        for gate in parents + children:
            if (type(gate) is not Gate or gate.frame is not h.frame
                    or not any(old is gate for old in h.gates)):
                raise Rejected("foreign or nonowned gate")
            base._check_scalar(h, gate.f, m)
            base._check_scalar(h, gate.q, m)
        if len({id(g) for g in parents + children}) != 4:
            raise Rejected("four distinct original gates are required")
        if any(g.status != "crossing" or g.phase_column is None for g in parents):
            raise Rejected("the two original parents must be crossing")
        selected_q = tuple(g.q.column for g in parents)
        if len(set(selected_q)) != 2:
            raise Rejected("parent output identities cannot be separated")
        _independent_parents(h, tuple(g.f.column for g in parents), selected_q, m)
        scales = tuple(base._max(_exact._neg(g.lower, m), g.upper, m) for g in parents)
        inverses = tuple(_exact._div(ONE, s, m) for s in scales)
        old_width, width = h.n_columns, h.n_columns + 6
        m.integer(width)
        x1, x2 = tuple(base._linear(((inverses[i], base._var(g.f.column, width, m)),),
                                    width, m) for i, g in enumerate(parents))
        q1, q2 = tuple(base._linear(((inverses[i], base._var(g.q.column, width, m)),),
                                    width, m) for i, g in enumerate(parents))
        y1, y2 = tuple(base._var(g.q.column, width, m) for g in children)
        alpha1, alpha2 = tuple(base._form(HALF, ((g.phase_column, HALF),), width, m)
                               for g in parents)

        # Distinct stop sets must never share an expansion memo.  Both parents
        # share one no-selected-output frontier; both children share the other.
        parent_memo, child_memo = {}, {}
        m.charge(work=4, entries=8)
        expanded_x = tuple(base._linear(((inverses[i], base._expand(
            h, g.f.column, {}, m, parent_memo)),), width, m)
                           for i, g in enumerate(parents))
        for form in expanded_x:
            for col, coefficient in form.terms:
                m.charge(work=1)
                if col in selected_q and coefficient != ZERO:
                    raise Rejected("selected outputs remain in parent source forms")
        stops = {col: col for col in selected_q}
        g1, g2 = tuple(base._expand(h, g.f.column, stops, m, child_memo) for g in children)
        z = base._linear(((HALF, g1), (HALF, g2)), width, m)
        d = base._linear(((HALF, g1), (_exact._neg(HALF, m), g2)), width, m)
        b1, b2 = tuple(_exact._mul(base._coef(z, selected_q[i], m), scales[i], m)
                       for i in range(2))
        source_z = base._linear(((ONE, z), (_exact._neg(b1, m), q1),
                                  (_exact._neg(b2, m), q2)), width, m)
        forms = expanded_x + (source_z,)
        columns, radii, gram, rhs, det, a = _source_gram(h, forms, m)
        a1, a2 = a
        A, B = tuple(_exact._mul(base._coef(d, selected_q[i], m), scales[i], m)
                     for i in range(2))
        tau = _exact._mul(HALF, _exact._sub(A, B, m), m)
        if _exact._cmp(tau, ZERO, m) <= 0:
            raise Rejected("fixed child coefficients do not give positive tau")
        r = base._linear(((ONE, source_z), (_exact._neg(a1, m), expanded_x[0]),
                           (_exact._neg(a2, m), expanded_x[1])), width, m)
        e = base._linear(((ONE, d), (_exact._neg(tau, m), q1), (tau, q2)), width, m)
        rb = base._bound_terms(r.constant, r.terms, h.column_bounds, m)
        eb = base._bound_terms(e.constant, e.terms, h.column_bounds, m)
        ab1, ab2 = _exact._add(a1, b1, m), _exact._add(a2, b2, m)
        l10 = _exact._sub(base._min(ZERO, ab1, m), base._max(ZERO, a2, m), m)
        u10 = _exact._sub(base._max(ZERO, ab1, m), base._min(ZERO, a2, m), m)
        l01 = _exact._add(_exact._neg(base._max(ZERO, a1, m), m), base._min(ZERO, ab2, m), m)
        u01 = _exact._add(_exact._neg(base._min(ZERO, a1, m), m), base._max(ZERO, ab2, m), m)
        mixed = (_exact._add(rb[0], base._min(l10, l01, m), m),
                 _exact._add(rb[1], base._max(u10, u01, m), m))
        if (_exact._cmp(mixed[0], _exact._neg(tau, m), m) < 0
                or _exact._cmp(mixed[1], tau, m) > 0):
            raise Rejected("the complete same-H mixed mismatch guard is not certified")

        sources = (x2, x1, q2, q1, r, r)
        phases = tuple(parents[i % 2].phase_column for i in range(6))
        intervals = ((NEG, ONE), (NEG, ONE), (ZERO, ONE), (ZERO, ONE), rb, rb)
        products, bounds, rows, product_rows = [], [], [], []
        m.charge(work=6, entries=36)
        for offset, (source, phase, interval) in enumerate(zip(sources, phases, intervals)):
            lo, hi = interval
            col = old_width + offset
            products.append(Product(col, phase, source, lo, hi))
            bounds.append((base._min(ZERO, lo, m), base._max(ZERO, hi, m)))
            value = base._var(col, width, m)
            alpha = base._form(HALF, ((phase, HALF),), width, m)
            mc = (
                base._linear(((NEG, value), (lo, alpha)), width, m),
                base._linear(((ONE, value), (_exact._neg(hi, m), alpha)), width, m),
                base._linear(((NEG, value), (ONE, source), (hi, alpha)), width, m),
                base._linear(((ONE, value), (NEG, source), (_exact._neg(lo, m), alpha)), width, m),
            )
            product_rows.append(tuple(range(len(rows), len(rows) + 4)))
            rows.extend(base._le(f, value, width, m) for f, value in
                        zip(mc, (ZERO, ZERO, hi, _exact._neg(lo, m))))
            m.charge(work=4, entries=24)
        c12, c21, d12, d21, v1, v2 = tuple(base._var(old_width + i, width, m)
                                          for i in range(6))
        K = base._linear(((tau, alpha1), (_exact._neg(tau, m), alpha2),
                          (a1, q1), (_exact._neg(a1, m), c21),
                          (a2, c12), (_exact._neg(a2, m), q2),
                          (b1, q1), (_exact._neg(b1, m), d21),
                          (b2, d12), (_exact._neg(b2, m), q2),
                          (ONE, v1), (NEG, v2)), width, m)
        difference = base._linear(((ONE, y1), (NEG, y2), (NEG, K)), width, m)
        p1, p2 = tuple(base._linear(((ONE, alpha), (NEG, q)), width, m)
                       for alpha, q in ((alpha1, q1), (alpha2, q2)))
        two_tau = _exact._mul(TWO, tau, m)
        rows.append(base._le(base._linear(((ONE, difference), (_exact._neg(two_tau, m), p2)),
                                         width, m), _exact._mul(TWO, base._max(eb[1], ZERO, m), m), width, m))
        rows.append(base._le(base._linear(((NEG, difference), (_exact._neg(two_tau, m), p1)),
                                         width, m), _exact._neg(_exact._mul(TWO, base._min(eb[0], ZERO, m), m), m), width, m))
        T = base._linear(((ONE, q1), (NEG, q2), (ONE, c12), (NEG, c21)), width, m)
        deficits = tuple(base._linear(((_exact._neg(TWO, m), q), (ONE, x)),
                                      width, m, constant=ONE) for q, x in ((q1, x1), (q2, x2)))
        U = base._linear(((ONE, q1), (ONE, q2), (NEG, c12), (NEG, c21)), width, m)
        D = base._linear(((TWO, q1), (NEG, x1), (TWO, q2), (NEG, x2)), width, m)
        rows.extend(base._le(f, ZERO, width, m) for f in (
            base._linear(((ONE, T), (NEG, deficits[1])), width, m),
            base._linear(((NEG, T), (NEG, deficits[0])), width, m),
            base._linear(((ONE, U), (NEG, D)), width, m)))
        m.charge(work=width + len(columns) + 60, entries=width + len(columns) + 160)
        return Relation(h, parents, children, scales, (x1, x2, q1, q2), (g1, g2),
                        forms, columns, radii, gram, rhs, det, tau, (a1, a2), (b1, b2),
                        z, d, r, e, rb, eb, ((l10, u10), (l01, u01)), mixed, K,
                        tuple(products), tuple(rows), tuple(product_rows), (24, 25),
                        (26, 27, 28), h.column_bounds + tuple(bounds), _OWNED)
    except Rejected:
        raise
    except MemoryError:
        h.budget._reject("allocation failed during mixed attachment")
    except (TypeError, ValueError, IndexError, KeyError, AttributeError, OverflowError) as exc:
        raise Rejected("invalid mixed relation attachment: " + str(exc)) from exc
