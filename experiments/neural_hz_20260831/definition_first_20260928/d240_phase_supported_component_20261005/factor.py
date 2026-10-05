"""Default-off source-supported difference rows on an owned declared HZ graph.

This is an exact-rational mathematical component, NOT an ACT/ONNX adapter.
make_block declares the complete original graph; its frame is an in-memory
identity, not a model-binding certificate.  attach retains that object, all
original columns, signed bits, predicates and input coordinates.  It adds four
continuous products and finite valid rows, never a phase pool, clip gate,
solver, support query, or replacement source.  The input decoder returns the
declared inputs only; it does not certify an original-model ADV.

Column order is sources, q1,q2,y1,y2, four original signed bits, then (only in
Compiled) four products.  Alpha=(signed_bit+1)/2.  Extra binary sources also
remain signed.  Fractional binary values are admitted by satisfied only for
LP-row inspection; canonical_extension and decode require integer originals.

Only Budget/Limits, exact arithmetic and sparse Row construction are reused
from D228.  No D228 Bank, geometry, support, or compiler is called.  All owned
operations share one persistent budget.  These logical counters are not a
claim about full Python/native/GPU storage or performance.  External input
object construction is outside the ledger; reads, expansions, arithmetic,
sorting and retained results of these operations are charged.

Block, Compiled, Product and Affine are trusted in-memory records.  Supported
construction paths are make_block and attach; private-state mutation,
dataclass-construction bypasses and untrusted deserialization are unsupported.
The complete parent is immutable through this API.  There is no native row
installation or global factor allocator, and no arbitrary-depth qualification.
"""

from dataclasses import dataclass, field
from fractions import Fraction

from experiments.neural_hz_20260831.definition_first_20260928.d228_joint_bank_component_20261005 import bank as _exact


Rejected = _exact.Rejected
Budget = _exact.Budget
Limits = _exact.Limits
Row = _exact.Row
ZERO, ONE, TWO = Fraction(0), Fraction(1), Fraction(2)
HALF, NEG_ONE = Fraction(1, 2), Fraction(-1)
_OWNED = object()


@dataclass(frozen=True)
class Affine:
    constant: Fraction = ZERO
    terms: tuple = ()

    def __post_init__(self):
        constant = _exact._hard_number(self.constant)
        row = Row(self.terms, ZERO)
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "terms", row.terms)


@dataclass(frozen=True)
class Product:
    column: int
    phase_index: int
    source: Affine
    lower: Fraction
    upper: Fraction


def _budget(value):
    if value is None:
        return Budget()
    if type(value) is not Budget:
        raise Rejected("one shared D228 Budget is required")
    return value


def _tuple(value, length, name):
    if type(value) is not tuple or (length is not None and len(value) != length):
        raise Rejected(name + " must be a tuple of the declared length")
    return value


def _row(terms, rhs, width, meter):
    terms = tuple(terms)
    count = len(terms)
    meter.charge(work=count * (count.bit_length() + 1) + 1,
                 entries=count + 1)
    return _exact._row(terms, rhs, width, meter)


def _form(constant, terms, width, meter):
    constant = _exact._number(constant, meter)
    row = _row(terms, ZERO, width, meter)
    meter.charge(work=1 + len(row.terms), entries=4 + 2 * len(row.terms))
    return Affine(constant, row.terms)


def _var(index, width, meter):
    return _form(ZERO, ((index, ONE),), width, meter)


def _linear(pieces, width, meter, constant=ZERO):
    pieces = tuple(pieces)
    meter.charge(work=len(pieces) + 1, entries=2 * len(pieces) + 1)
    offset = _exact._number(constant, meter)
    terms = []
    for scale, form in pieces:
        scale = _exact._number(scale, meter)
        if type(form) is not Affine:
            raise Rejected("affine arithmetic requires a declared Affine")
        offset = _exact._add(offset, _exact._mul(scale, form.constant, meter), meter)
        meter.charge(work=len(form.terms), entries=2 * len(form.terms))
        for index, coefficient in form.terms:
            terms.append((index, _exact._mul(scale, coefficient, meter)))
    return _form(offset, terms, width, meter)


def _le(form, rhs, width, meter):
    return _row(form.terms, _exact._sub(_exact._number(rhs, meter),
                                      form.constant, meter), width, meter)


def _at(form, values, meter):
    value = _exact._number(form.constant, meter)
    for index, coefficient in form.terms:
        meter.charge(work=1)
        if not 0 <= index < len(values):
            raise Rejected("affine source lies outside the declared assignment")
        value = _exact._add(value, _exact._mul(coefficient, values[index], meter), meter)
    return value


def _bounds(form, bounds, meter):
    lower = upper = _exact._number(form.constant, meter)
    for index, coefficient in form.terms:
        meter.charge(work=2)
        if not 0 <= index < len(bounds):
            raise Rejected("bound source lies outside the complete declared frame")
        lo, hi = bounds[index]
        if coefficient.numerator < 0:
            lo, hi = hi, lo
        lower = _exact._add(lower, _exact._mul(coefficient, lo, meter), meter)
        upper = _exact._add(upper, _exact._mul(coefficient, hi, meter), meter)
    meter.charge(work=1, entries=2)
    return lower, upper


def _min(a, b, meter):
    return a if _exact._cmp(a, b, meter) <= 0 else b


def _max(a, b, meter):
    return a if _exact._cmp(a, b, meter) >= 0 else b


def _coefficient(form, index, meter):
    answer = ZERO
    for column, value in form.terms:
        meter.charge(work=1)
        if column == index:
            answer = value
    return answer


def _input_rows(rows, width, name, meter):
    rows = _tuple(rows, None, name)
    meter.charge(work=len(rows), entries=len(rows))
    for row in rows:
        if type(row) is not Row:
            raise Rejected(name + " must contain canonical Row objects")
        meter.charge(work=1 + len(row.terms), entries=3 + 2 * len(row.terms))
        _exact._number(row.rhs, meter)
        previous = -1
        for index, coefficient in row.terms:
            meter.charge(work=2)
            if type(index) is not int or not previous < index < width:
                raise Rejected("original predicate references an invalid column")
            if _exact._number(coefficient, meter) == ZERO:
                raise Rejected("original predicate contains a zero coefficient")
            previous = index
    return rows


def _holds(values, bounds, eq, le, binary_columns, integral, meter):
    if len(values) != len(bounds):
        return False
    for value, (lower, upper) in zip(values, bounds):
        meter.charge(work=1)
        if (_exact._cmp(value, lower, meter) < 0
                or _exact._cmp(value, upper, meter) > 0):
            return False
    if integral:
        for index in binary_columns:
            meter.charge(work=1)
            if values[index] != NEG_ONE and values[index] != ONE:
                return False
    for rows, equality in ((eq, True), (le, False)):
        for row in rows:
            meter.charge(work=1)
            value = ZERO
            for index, coefficient in row.terms:
                value = _exact._add(value, _exact._mul(coefficient, values[index], meter), meter)
            comparison = _exact._cmp(value, row.rhs, meter)
            if (equality and comparison != 0) or (not equality and comparison > 0):
                return False
    return True


@dataclass(frozen=True)
class Block:
    bounds: tuple
    coefficients: tuple
    biases: tuple
    source_kinds: tuple
    source_eq: tuple
    source_le: tuple
    child_forms: tuple
    child_bounds: tuple
    column_bounds: tuple
    eq: tuple
    le: tuple
    bound_rows: tuple
    graph_rows: tuple
    binary_columns: tuple
    frame: object = field(repr=False, compare=False)
    budget: Budget = field(repr=False, compare=False)
    _authority: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._authority is not _OWNED:
            raise Rejected("Block must be constructed by make_block")

    @property
    def n_sources(self):
        return len(self.bounds)

    @property
    def n_columns(self):
        return self.n_sources + 8

    @property
    def q_indices(self):
        return self.n_sources, self.n_sources + 1

    @property
    def y_indices(self):
        return self.n_sources + 2, self.n_sources + 3

    @property
    def phase_indices(self):
        return tuple(range(self.n_sources + 4, self.n_sources + 8))

    @property
    def parent_rows(self):
        return self.graph_rows[:2]

    @property
    def child_rows(self):
        return self.graph_rows[2:]

    def _start(self, frame):
        meter = self.budget._branch()
        meter.charge(work=1)
        if frame is not self.frame:
            raise Rejected("frame must be the identical declared Block frame")
        return meter

    def satisfied(self, values, *, frame, integral=False):
        meter = self._start(frame)
        if type(integral) is not bool:
            raise Rejected("integral must be bool")
        values = _exact._vector(values, self.n_columns, "original assignment", meter)
        return _holds(values, self.column_bounds, self.eq, self.le,
                      self.binary_columns, integral, meter)

    def evaluate(self, sources, *, phases=None, frame):
        """Construct one actual declared-graph state, not a support witness."""
        meter = self._start(frame)
        sources = _exact._vector(sources, self.n_sources, "all original sources", meter)
        q = tuple(_max(ZERO, value, meter) for value in sources[:2])
        meter.charge(work=self.n_sources + 2, entries=self.n_sources + 4)
        inputs = sources + q
        g = tuple(_at(form, inputs, meter) for form in self.child_forms)
        y = tuple(_max(ZERO, value, meter) for value in g)
        preactivations = sources[:2] + g
        if phases is None:
            phases = tuple(ONE if value.numerator > 0 else NEG_ONE for value in preactivations)
            meter.charge(work=4, entries=4)
        else:
            phases = _exact._vector(phases, 4, "four original signed phases", meter)
        meter.charge(work=self.n_columns, entries=self.n_columns)
        values = sources + q + y + phases
        if not _holds(values, self.column_bounds, self.eq, self.le,
                      self.binary_columns, True, meter):
            raise Rejected("source/phases do not satisfy the complete original graph")
        return values

    def decode(self, values, *, frame):
        meter = self._start(frame)
        values = _exact._vector(values, self.n_columns, "original assignment", meter)
        if not _holds(values, self.column_bounds, self.eq, self.le,
                      self.binary_columns, True, meter):
            raise Rejected("decoder requires a legal integer original assignment")
        meter.charge(work=self.n_sources, entries=self.n_sources)
        return values[:self.n_sources]


def make_block(bounds, coefficients, biases, *, source_kinds=None,
               eq=(), le=(), enabled=False, budget=None):
    if enabled is not True:
        raise Rejected("declared graph is disabled unless enabled=True")
    budget = _budget(budget)
    meter = budget._branch()
    bounds = _tuple(bounds, None, "source bounds")
    n = len(bounds)
    if n < 2:
        raise Rejected("two original continuous source coordinates are required")
    meter.charge(work=n, entries=3 * n)
    parsed = []
    for pair in bounds:
        lo, hi = (_exact._number(v, meter) for v in _tuple(pair, 2, "source interval"))
        if _exact._cmp(lo, hi, meter) > 0:
            raise Rejected("reversed original source interval")
        parsed.append((lo, hi))
    bounds = tuple(parsed)
    if bounds[:2] != ((NEG_ONE, ONE), (NEG_ONE, ONE)):
        raise Rejected("the two original parent inputs must have bounds [-1,1]")
    if source_kinds is None:
        source_kinds = ("continuous",) * n
        meter.charge(work=n, entries=n)
    source_kinds = _tuple(source_kinds, n, "source kinds")
    for index, kind in enumerate(source_kinds):
        meter.charge(work=2)
        if kind not in ("continuous", "binary") or type(kind) is not str:
            raise Rejected("unsupported original source kind")
        if kind == "binary" and (index < 2 or bounds[index] != (NEG_ONE, ONE)):
            raise Rejected("extra signed binary sources require [-1,1] bounds")
    coefficients = tuple(_exact._vector(row, n + 2, "complete child coefficients", meter)
                         for row in _tuple(coefficients, 2, "child coefficients"))
    biases = tuple(_exact._number(v, meter) for v in _tuple(biases, 2, "child biases"))
    width = n + 8
    source_eq = _input_rows(eq, width, "original EQ", meter)
    source_le = _input_rows(le, width, "original LE", meter)
    child_forms = tuple(_form(bias, tuple(enumerate(row)), width, meter)
                        for bias, row in zip(biases, coefficients))
    base_bounds = bounds + ((ZERO, ONE), (ZERO, ONE))
    meter.charge(work=n + 2, entries=n + 2)
    child_bounds = tuple(_bounds(form, base_bounds, meter) for form in child_forms)
    output_bounds = tuple((_max(ZERO, lo, meter), _max(ZERO, hi, meter))
                          for lo, hi in child_bounds)
    column_bounds = base_bounds + output_bounds + ((NEG_ONE, ONE),) * 4
    rows = list(source_le)
    meter.charge(work=len(rows) + width, entries=len(rows) + width)
    bound_rows = []
    for index, (lo, hi) in enumerate(column_bounds):
        first = len(rows)
        rows.append(_row(((index, NEG_ONE),), _exact._neg(lo, meter), width, meter))
        rows.append(_row(((index, ONE),), hi, width, meter))
        bound_rows.append((first, first + 1))
    graph_rows = []
    inputs = (_var(0, width, meter), _var(1, width, meter)) + child_forms
    graph_bounds = ((NEG_ONE, ONE), (NEG_ONE, ONE)) + child_bounds
    for offset, (g, interval) in enumerate(zip(inputs, graph_bounds)):
        output = _var(n + offset, width, meter)
        alpha = _form(HALF, ((n + 4 + offset, HALF),), width, meter)
        lower = _min(ZERO, interval[0], meter)
        upper = _max(ZERO, interval[1], meter)
        forms = (
            _linear(((NEG_ONE, output),), width, meter),
            _linear(((ONE, g), (NEG_ONE, output)), width, meter),
            _linear(((ONE, output), (_exact._neg(upper, meter), alpha)), width, meter),
            _linear(((ONE, output), (NEG_ONE, g), (_exact._neg(lower, meter), alpha)), width, meter),
        )
        indices = tuple(range(len(rows), len(rows) + 4))
        rows.extend(_le(form, rhs, width, meter)
                    for form, rhs in zip(forms, (ZERO, ZERO, ZERO, _exact._neg(lower, meter))))
        graph_rows.append(indices)
    binary_columns = tuple(i for i, kind in enumerate(source_kinds) if kind == "binary")
    binary_columns += tuple(range(n + 4, n + 8))
    meter.charge(work=width + len(rows),
                 entries=30 + 3 * width + len(rows) + 4 * len(graph_rows) + len(binary_columns))
    return Block(bounds, coefficients, biases, source_kinds, source_eq, source_le,
                 child_forms, child_bounds, column_bounds, source_eq, tuple(rows),
                 tuple(bound_rows), tuple(graph_rows), binary_columns, object(), budget,
                 _authority=_OWNED)


@dataclass(frozen=True)
class Compiled:
    parent: Block
    anchor_kind: str
    tau: Fraction
    a: Fraction
    b: Fraction
    z: Affine
    d: Affine
    e: Affine
    r: Affine
    e_bounds: tuple
    r_bounds: tuple
    products: tuple
    additional_le: tuple
    product_rows: tuple
    defect_rows: tuple
    capacity_rows: tuple
    column_bounds: tuple
    eq: tuple
    le: tuple
    _authority: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._authority is not _OWNED:
            raise Rejected("Compiled must be constructed by attach")

    @property
    def frame(self):
        return self.parent.frame

    @property
    def budget(self):
        return self.parent.budget

    @property
    def n_columns(self):
        return self.parent.n_columns + 4

    @property
    def additional_eq(self):
        return ()

    def satisfied(self, values, *, frame, integral=False):
        meter = self.parent._start(frame)
        if type(integral) is not bool:
            raise Rejected("integral must be bool")
        values = _exact._vector(values, self.n_columns, "extended assignment", meter)
        return _holds(values, self.column_bounds, self.eq, self.le,
                      self.parent.binary_columns, integral, meter)

    def canonical_extension(self, original_values, *, frame):
        meter = self.parent._start(frame)
        original_values = _exact._vector(original_values, self.parent.n_columns,
                                         "complete original integer assignment", meter)
        if not _holds(original_values, self.parent.column_bounds, self.parent.eq,
                      self.parent.le, self.parent.binary_columns, True, meter):
            raise Rejected("products require the complete legal original integer graph")
        extra = []
        for product in self.products:
            alpha = _exact._mul(HALF, _exact._add(ONE, original_values[product.phase_index], meter), meter)
            extra.append(_exact._mul(alpha, _at(product.source, original_values, meter), meter))
        meter.charge(work=self.n_columns, entries=self.n_columns + 4)
        values = original_values + tuple(extra)
        if not _holds(values, self.column_bounds, self.eq, self.le,
                      self.parent.binary_columns, True, meter):
            raise Rejected("canonical products violate the declared factor")
        return values

    def decode(self, values, *, frame):
        meter = self.parent._start(frame)
        values = _exact._vector(values, self.n_columns, "extended assignment", meter)
        if not _holds(values, self.column_bounds, self.eq, self.le,
                      self.parent.binary_columns, True, meter):
            raise Rejected("decoder requires a legal integer extended assignment")
        meter.charge(work=self.parent.n_sources, entries=self.parent.n_sources)
        return values[:self.parent.n_sources]


def attach(block, *, anchor_kind="x", tau=1, enabled=False, frame):
    if enabled is not True:
        raise Rejected("difference factor is disabled unless enabled=True")
    if type(block) is not Block or block._authority is not _OWNED:
        raise Rejected("attach requires an owned make_block result")
    meter = block._start(frame)
    if type(anchor_kind) is not str or anchor_kind not in ("x", "q"):
        raise Rejected("anchor_kind must be the preregistered x or q structure")
    tau = _exact._number(tau, meter)
    if _exact._cmp(tau, ZERO, meter) <= 0:
        raise Rejected("tau must be strictly positive")
    old_width, width = block.n_columns, block.n_columns + 4
    q1, q2 = (_var(i, width, meter) for i in block.q_indices)
    x1, x2 = _var(0, width, meter), _var(1, width, meter)
    y1, y2 = (_var(i, width, meter) for i in block.y_indices)
    alpha1, alpha2 = (_form(HALF, ((i, HALF),), width, meter)
                      for i in block.phase_indices[:2])
    g1, g2 = block.child_forms
    z = _linear(((HALF, g1), (HALF, g2)), width, meter)
    d = _linear(((HALF, g1), (_exact._neg(HALF, meter), g2)), width, meter)
    e = _linear(((ONE, d), (_exact._neg(tau, meter), q1), (tau, q2)), width, meter)
    anchors = (0, 1) if anchor_kind == "x" else block.q_indices
    a, b = (_coefficient(z, index, meter) for index in anchors)
    anchor1, anchor2 = (x1, x2) if anchor_kind == "x" else (q1, q2)
    r = _linear(((ONE, z), (_exact._neg(a, meter), anchor1),
                 (_exact._neg(b, meter), anchor2)), width, meter)
    e_bounds, r_bounds = (_bounds(form, block.column_bounds, meter) for form in (e, r))
    if anchor_kind == "x":
        ap, am = _max(a, ZERO, meter), _max(_exact._neg(a, meter), ZERO, meter)
        bp, bm = _max(b, ZERO, meter), _max(_exact._neg(b, meter), ZERO, meter)
        magnitude = _max(_exact._add(ap, bm, meter), _exact._add(am, bp, meter), meter)
        guard_lower = _exact._sub(r_bounds[0], magnitude, meter)
        guard_upper = _exact._add(r_bounds[1], magnitude, meter)
    else:
        guard_lower = _exact._add(r_bounds[0], _min(ZERO, _min(a, b, meter), meter), meter)
        guard_upper = _exact._add(r_bounds[1], _max(ZERO, _max(a, b, meter), meter), meter)
    if (_exact._cmp(guard_lower, _exact._neg(tau, meter), meter) < 0
            or _exact._cmp(guard_upper, tau, meter) > 0):
        raise Rejected("complete source does not certify the declared mismatch scope")
    source_forms = (anchor2, anchor1, r, r)
    phases = (block.phase_indices[0], block.phase_indices[1],
              block.phase_indices[0], block.phase_indices[1])
    source_ranges = (((NEG_ONE, ONE), (NEG_ONE, ONE)) if anchor_kind == "x"
                     else ((ZERO, ONE), (ZERO, ONE))) + (r_bounds, r_bounds)
    products, product_bounds, rows, product_rows = [], [], [], []
    for offset, (source, phase, interval) in enumerate(zip(source_forms, phases, source_ranges)):
        lo, hi = interval
        column = old_width + offset
        products.append(Product(column, phase, source, lo, hi))
        product_bounds.append((_min(ZERO, lo, meter), _max(ZERO, hi, meter)))
        m = _var(column, width, meter)
        alpha = _form(HALF, ((phase, HALF),), width, meter)
        forms = (
            _linear(((NEG_ONE, m), (lo, alpha)), width, meter),
            _linear(((ONE, m), (_exact._neg(hi, meter), alpha)), width, meter),
            _linear(((NEG_ONE, m), (ONE, source), (hi, alpha)), width, meter),
            _linear(((ONE, m), (NEG_ONE, source), (_exact._neg(lo, meter), alpha)), width, meter),
        )
        product_rows.append(tuple(range(len(rows), len(rows) + 4)))
        rows.extend(_le(form, rhs, width, meter)
                    for form, rhs in zip(forms, (ZERO, ZERO, hi, _exact._neg(lo, meter))))
        meter.charge(work=4, entries=14)
    c12, c21, v1, v2 = (_var(old_width + i, width, meter) for i in range(4))
    reference = _linear(((tau, alpha1), (_exact._neg(tau, meter), alpha2),
                         (a, q1), (_exact._neg(a, meter), c21),
                         (b, c12), (_exact._neg(b, meter), q2),
                         (ONE, v1), (NEG_ONE, v2)), width, meter)
    difference = _linear(((ONE, y1), (NEG_ONE, y2), (NEG_ONE, reference)), width, meter)
    p1 = _linear(((ONE, alpha1), (NEG_ONE, q1)), width, meter)
    p2 = _linear(((ONE, alpha2), (NEG_ONE, q2)), width, meter)
    twice_tau = _exact._mul(TWO, tau, meter)
    upper = _linear(((ONE, difference), (_exact._neg(twice_tau, meter), p2)), width, meter)
    lower = _linear(((NEG_ONE, difference), (_exact._neg(twice_tau, meter), p1)), width, meter)
    defect_rows = (len(rows), len(rows) + 1)
    rows.append(_le(upper, _exact._mul(TWO, _max(e_bounds[1], ZERO, meter), meter), width, meter))
    rows.append(_le(lower, _exact._neg(_exact._mul(TWO, _min(e_bounds[0], ZERO, meter), meter), meter), width, meter))
    capacity_rows = ()
    if anchor_kind == "x":
        T = _linear(((ONE, q1), (NEG_ONE, q2), (ONE, c12), (NEG_ONE, c21)), width, meter)
        minus_two = _exact._neg(TWO, meter)
        deficit1 = _linear(((minus_two, q1), (ONE, x1)), width, meter, constant=ONE)
        deficit2 = _linear(((minus_two, q2), (ONE, x2)), width, meter, constant=ONE)
        U = _linear(((ONE, q1), (ONE, q2), (NEG_ONE, c12), (NEG_ONE, c21)), width, meter)
        D = _linear(((TWO, q1), (NEG_ONE, x1), (TWO, q2), (NEG_ONE, x2)), width, meter)
        capacity_rows = tuple(range(len(rows), len(rows) + 3))
        rows.extend(_le(form, ZERO, width, meter) for form in (
            _linear(((ONE, T), (NEG_ONE, deficit2)), width, meter),
            _linear(((NEG_ONE, T), (NEG_ONE, deficit1)), width, meter),
            _linear(((ONE, U), (NEG_ONE, D)), width, meter),
        ))
    column_bounds = block.column_bounds + tuple(product_bounds)
    all_le = block.le + tuple(rows)
    meter.charge(work=len(all_le) + width,
                 entries=45 + len(all_le) + 3 * width + len(rows) + 4 * len(product_rows))
    return Compiled(block, anchor_kind, tau, a, b, z, d, e, r, e_bounds, r_bounds,
                    tuple(products), tuple(rows), tuple(product_rows), defect_rows,
                    capacity_rows, column_bounds, block.eq, all_le, _authority=_OWNED)
