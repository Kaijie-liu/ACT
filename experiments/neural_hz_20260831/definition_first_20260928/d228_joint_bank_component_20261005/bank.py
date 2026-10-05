"""Opt-in, exact local two-parent/two-child ReLU bank (D227).

This is a mathematical primitive, NOT a native-HZ/model adapter.  ``frame``
and the IDs are caller-supplied identity tokens, not certificates of a model
binding.  All four ORIGINAL signed phase columns are retained by ``compile``.

Base coordinates are (f1, f2, q1, q2, *residuals); physical coordinates append
(y1, y2).  Residuals are actual caller readouts: the interval product is only
a local envelope.  In particular, its vertices are not original-input ADV
witnesses.  Do not attach per-vertex original-input allocations or require
each vertex to satisfy the complete correlated parent H.  Original readout
equalities, predicates and the input decoder belong to the caller and must
remain in that caller's system.  Local ideality does not imply full-source
ideality or arbitrary-depth closure.

Construction visits true product faces of dimensions 0, 1 and 2, and solves
at most two affine zero equations on each face.  It does not call an LP,
support oracle, phase subproblem, sampling routine or rescue path.  Exact
integer/Fraction data only; no float coercion or tolerance-based decisions.

Budget counters are persistent, conservative logical work/entry counters;
they are NOT a claim about complete Python storage, bit-operation cost,
native-HZ cost, evidence size, decoder cost, device storage or performance.
Input object construction is outside that accounting; build validates and
charges its own reads and construction.  Each public build/query/compile/
satisfied call is one branch, with shared lifetime work/entry caps.  Failed
budget charges are never refunded, and an exhausted budget stays rejected.

Bank, Compiled and Support are trusted in-memory result records.  Their only
supported production paths are build(), Bank.compile() and Bank.support().
Direct result-record construction, mutation of private state, object-level
frozen-dataclass bypasses and untrusted deserialization are outside this API;
these records are not authenticated or independently validated certificates.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from functools import cmp_to_key
from itertools import product


class Rejected(ValueError):
    """Fail-closed rejection; no partial certificate is returned."""


_HARD_WORK = 256_000_000
_HARD_BRANCH = 200_000_000
_HARD_ENTRIES = 64_000_000
_HARD_BITS = 512
_ZERO = Fraction(0)
_ONE = Fraction(1)
_TWO = Fraction(2)


def _hard_number(value):
    if type(value) not in (int, Fraction):
        raise Rejected("only int/Fraction values are accepted (not bool/float)")
    result = value if type(value) is Fraction else Fraction(value)
    if (abs(result.numerator).bit_length() > _HARD_BITS
            or result.denominator.bit_length() > _HARD_BITS):
        raise Rejected("input exceeds the hard rational bit limit")
    return result


def _tuple(value, length, name):
    if type(value) is not tuple or len(value) != length:
        raise Rejected("%s must be a tuple of length %d" % (name, length))
    return value


@dataclass(frozen=True)
class Limits:
    max_work: int = _HARD_WORK
    max_branch: int = _HARD_BRANCH
    max_entries: int = _HARD_ENTRIES
    max_bits: int = _HARD_BITS

    def __post_init__(self):
        for name, cap, minimum in (
            ("max_work", _HARD_WORK, 0),
            ("max_branch", _HARD_BRANCH, 0),
            ("max_entries", _HARD_ENTRIES, 0),
            ("max_bits", _HARD_BITS, 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= cap:
                raise Rejected("invalid or enlarged limit: " + name)


class Budget:
    """Shared lifetime counters; not thread-safe or a process resource limit."""

    def __init__(self, limits=None, *, max_work=None, max_branch=None,
                 max_entries=None, max_bits=None):
        overrides = (max_work, max_branch, max_entries, max_bits)
        if limits is not None:
            if type(limits) is not Limits or any(v is not None for v in overrides):
                raise Rejected("use either Limits or limit keywords")
            self._limits = limits
        else:
            self._limits = Limits(
                _HARD_WORK if max_work is None else max_work,
                _HARD_BRANCH if max_branch is None else max_branch,
                _HARD_ENTRIES if max_entries is None else max_entries,
                _HARD_BITS if max_bits is None else max_bits,
            )
        self._work = 0
        self._entries = 0
        self._failure = None

    @property
    def limits(self):
        return self._limits

    @property
    def work(self):
        return self._work

    @property
    def entries(self):
        return self._entries

    @property
    def failed(self):
        return self._failure is not None

    @property
    def max_work(self):
        return self._limits.max_work

    @property
    def max_branch(self):
        return self._limits.max_branch

    @property
    def max_entries(self):
        return self._limits.max_entries

    @property
    def max_bits(self):
        return self._limits.max_bits

    def _reject(self, reason):
        self._failure = reason
        raise Rejected(reason)

    def _branch(self):
        if self._failure is not None:
            raise Rejected("budget was previously rejected: " + self._failure)
        if self._work > self.max_work or self._entries > self.max_entries:
            self._reject("budget is already over its cumulative limit")
        meter = _Meter(self, self._work)
        meter.charge(work=1, entries=1)
        return meter


class _Meter:
    def __init__(self, budget, start):
        self.budget = budget
        self.start = start

    def charge(self, *, work=0, entries=0):
        budget = self.budget
        if budget._failure is not None:
            raise Rejected("budget was previously rejected: " + budget._failure)
        if type(work) is not int or type(entries) is not int or work < 0 or entries < 0:
            budget._reject("invalid accounting charge")
        budget._work += work
        budget._entries += entries
        if budget._work > budget.max_work:
            budget._reject("cumulative work limit exceeded")
        if budget._work - self.start > budget.max_branch:
            budget._reject("per-operation branch work limit exceeded")
        if budget._entries > budget.max_entries:
            budget._reject("cumulative entry limit exceeded")

    def integer(self, value):
        if abs(value).bit_length() > self.budget.max_bits:
            self.budget._reject("rational intermediate bit limit exceeded")
        return value


def _fraction(numerator, denominator, meter):
    meter.charge(work=1, entries=1)
    meter.integer(numerator)
    meter.integer(denominator)
    if denominator == 0:
        raise Rejected("zero rational denominator")
    result = Fraction(numerator, denominator)
    meter.integer(result.numerator)
    meter.integer(result.denominator)
    return result


def _number(value, meter):
    if type(value) not in (int, Fraction):
        raise Rejected("only int/Fraction values are accepted (not bool/float)")
    if type(value) is int:
        return _fraction(value, 1, meter)
    meter.charge(work=1, entries=1)
    meter.integer(value.numerator)
    meter.integer(value.denominator)
    return value


def _add(a, b, meter):
    meter.charge(work=4)
    left = meter.integer(a.numerator * b.denominator)
    right = meter.integer(b.numerator * a.denominator)
    denominator = meter.integer(a.denominator * b.denominator)
    return _fraction(meter.integer(left + right), denominator, meter)


def _neg(a, meter):
    meter.charge(work=1)
    return _fraction(meter.integer(-a.numerator), a.denominator, meter)


def _sub(a, b, meter):
    return _add(a, _neg(b, meter), meter)


def _mul(a, b, meter):
    meter.charge(work=2)
    numerator = meter.integer(a.numerator * b.numerator)
    denominator = meter.integer(a.denominator * b.denominator)
    return _fraction(numerator, denominator, meter)


def _div(a, b, meter):
    if b.numerator == 0:
        raise Rejected("division by a zero pivot")
    meter.charge(work=2)
    numerator = meter.integer(a.numerator * b.denominator)
    denominator = meter.integer(a.denominator * b.numerator)
    return _fraction(numerator, denominator, meter)


def _cmp(a, b, meter):
    meter.charge(work=3)
    left = meter.integer(a.numerator * b.denominator)
    right = meter.integer(b.numerator * a.denominator)
    return (left > right) - (left < right)


def _vector(values, length, name, meter):
    if type(values) not in (tuple, list) or len(values) != length:
        raise Rejected("%s has the wrong length or container type" % name)
    meter.charge(work=length, entries=length)
    return tuple(_number(value, meter) for value in values)


def _dot(a, b, meter):
    if len(a) != len(b):
        raise Rejected("internal dot-product dimension mismatch")
    result = _ZERO
    for left, right in zip(a, b):
        meter.charge(work=1)
        if left.numerator != 0 and right.numerator != 0:
            result = _add(result, _mul(left, right, meter), meter)
    return result


@dataclass(frozen=True)
class Spec:
    parent_bounds: tuple
    residual_bounds: tuple
    coefficients: tuple
    biases: tuple

    def __post_init__(self):
        parents = []
        for pair in _tuple(self.parent_bounds, 2, "parent_bounds"):
            lo, hi = (_hard_number(v) for v in _tuple(pair, 2, "parent bound"))
            if not lo < 0 < hi:
                raise Rejected("parents must have strict crossing bounds")
            parents.append((lo, hi))
        if type(self.residual_bounds) is not tuple or len(self.residual_bounds) > 2:
            raise Rejected("zero, one or two residual bounds are required")
        residuals = []
        for pair in self.residual_bounds:
            lo, hi = (_hard_number(v) for v in _tuple(pair, 2, "residual bound"))
            # Check the comparison's unreduced integer intermediates as well.
            if (abs(lo.numerator * hi.denominator).bit_length() > _HARD_BITS
                    or abs(hi.numerator * lo.denominator).bit_length() > _HARD_BITS):
                raise Rejected("bound comparison exceeds the hard bit limit")
            if lo > hi:
                raise Rejected("reversed residual bounds")
            residuals.append((lo, hi))
        base_dim = 4 + len(residuals)
        coefficients = tuple(
            tuple(_hard_number(v) for v in _tuple(row, base_dim, "coefficient row"))
            for row in _tuple(self.coefficients, 2, "coefficients")
        )
        biases = tuple(_hard_number(v) for v in _tuple(self.biases, 2, "biases"))
        object.__setattr__(self, "parent_bounds", tuple(parents))
        object.__setattr__(self, "residual_bounds", tuple(residuals))
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "biases", biases)


@dataclass(frozen=True)
class Binding:
    frame: object
    physical_ids: tuple
    phase_ids: tuple

    def __post_init__(self):
        if type(self.physical_ids) is not tuple or not 6 <= len(self.physical_ids) <= 8:
            raise Rejected("physical_ids must contain six to eight IDs")
        _tuple(self.phase_ids, 4, "phase_ids")
        ids = self.physical_ids + self.phase_ids
        if any(type(value) is not str or not value.strip() for value in ids):
            raise Rejected("IDs must be nonempty strings")
        if len(set(ids)) != len(ids):
            raise Rejected("all physical and phase IDs must be distinct")


@dataclass(frozen=True)
class Support:
    value: Fraction
    vertex_index: int
    phases: tuple
    physical: tuple


@dataclass(frozen=True)
class Row:
    """Canonical sparse row, interpreted as EQ or LE by its containing list."""

    terms: tuple
    rhs: Fraction

    def __post_init__(self):
        if type(self.terms) is not tuple:
            raise Rejected("row terms must be a tuple")
        normalized = []
        previous = -1
        for term in self.terms:
            index, value = _tuple(term, 2, "row term")
            if type(index) is not int or index <= previous:
                raise Rejected("row columns must be nonnegative, strictly increasing")
            value = _hard_number(value)
            if value.numerator == 0:
                raise Rejected("canonical rows do not contain zero terms")
            previous = index
            normalized.append((index, value))
        object.__setattr__(self, "terms", tuple(normalized))
        object.__setattr__(self, "rhs", _hard_number(self.rhs))


def _row(terms, rhs, n_columns, meter):
    combined = {}
    for index, value in terms:
        meter.charge(work=1, entries=1)
        if type(index) is not int or not 0 <= index < n_columns:
            raise Rejected("row column outside the compiled layout")
        value = _number(value, meter)
        if index in combined:
            value = _add(combined[index], value, meter)
        combined[index] = value
    meter.charge(work=len(combined), entries=len(combined) + 1)
    canonical = tuple((index, combined[index]) for index in sorted(combined)
                      if combined[index].numerator != 0)
    return Row(canonical, _number(rhs, meter))


@dataclass(frozen=True)
class Compiled:
    """Trusted result of Bank.compile; not an untrusted row-file loader."""

    eq: tuple
    le: tuple
    n_columns: int
    physical_ids: tuple
    phase_ids: tuple
    n_lambda: int
    budget: Budget = field(repr=False, compare=False)

    def satisfied(self, physical, phases, lambdas):
        """Exact LP-row check; fractional signed phases are allowed here only."""
        meter = self.budget._branch()
        p = _vector(physical, len(self.physical_ids), "physical", meter)
        b = _vector(phases, 4, "phases", meter)
        weights = _vector(lambdas, self.n_lambda, "lambdas", meter)
        meter.charge(work=self.n_columns, entries=self.n_columns)
        values = p + b + weights
        if len(values) != self.n_columns:
            raise Rejected("compiled column layout is inconsistent")
        for rows, equality in ((self.eq, True), (self.le, False)):
            for row in rows:
                meter.charge(work=1)
                result = _ZERO
                for index, coefficient in row.terms:
                    if not 0 <= index < len(values):
                        raise Rejected("compiled row has an invalid column")
                    result = _add(result, _mul(coefficient, values[index], meter), meter)
                comparison = _cmp(result, row.rhs, meter)
                if (equality and comparison != 0) or (not equality and comparison > 0):
                    return False
        return True


@dataclass(frozen=True)
class Bank:
    """Trusted result of build; direct dataclass construction is unsupported."""

    spec: Spec
    binding: Binding
    vertices: tuple
    signs: tuple
    budget: Budget = field(repr=False, compare=False)
    original_grid_count: int
    candidate_count: int

    @property
    def physical_dim(self):
        return 6 + len(self.spec.residual_bounds)

    @property
    def vertex_limit(self):
        effective = sum(lo != hi for lo, hi in self.spec.residual_bounds)
        return (37, 104, 277)[effective]

    def _frame(self, frame):
        if frame is not self.binding.frame:
            raise Rejected("frame identity mismatch")

    def support(self, physical_coefficients, phase_coefficients=(0, 0, 0, 0), *, frame):
        """Exact local-envelope support, never an original-input ADV witness."""
        self._frame(frame)
        meter = self.budget._branch()
        physical = _vector(physical_coefficients, self.physical_dim,
                           "physical_coefficients", meter)
        phases = _vector(phase_coefficients, 4, "phase_coefficients", meter)
        best_value = None
        best_index = None
        best_phases = None
        for index, (vertex, signs) in enumerate(zip(self.vertices, self.signs)):
            meter.charge(work=1, entries=4)
            value = _dot(physical, vertex, meter)
            assignment = []
            for coefficient, sign in zip(phases, signs):
                signed = sign if sign != 0 else (1 if _cmp(coefficient, _ZERO, meter) > 0 else -1)
                assignment.append(signed)
                value = _add(value, _mul(coefficient, _number(signed, meter), meter), meter)
            if best_value is None or _cmp(value, best_value, meter) > 0:
                best_value, best_index, best_phases = value, index, tuple(assignment)
        if best_index is None:
            raise Rejected("empty bank")
        meter.charge(work=1, entries=4)
        return Support(best_value, best_index, best_phases, self.vertices[best_index])

    def compile(self, *, frame):
        """Appendable rows: physical, four original signed bits, then lambda."""
        self._frame(frame)
        meter = self.budget._branch()
        d = self.physical_dim
        n = len(self.vertices)
        offset = d + 4
        n_columns = offset + n
        eq = [_row(((offset + j, _ONE) for j in range(n)), _ONE, n_columns, meter)]
        for coordinate in range(d):
            terms = [(coordinate, _ONE)]
            meter.charge(work=1, entries=1)
            for j, vertex in enumerate(self.vertices):
                meter.charge(work=1)
                if vertex[coordinate].numerator != 0:
                    terms.append((offset + j, _neg(vertex[coordinate], meter)))
            eq.append(_row(terms, _ZERO, n_columns, meter))
        le = [_row(((offset + j, Fraction(-1)),), _ZERO, n_columns, meter)
              for j in range(n)]
        for phase in range(4):
            # 2 sum_positive lambda - signed_bit <= 1;
            # signed_bit + 2 sum_negative lambda <= 1.
            lower = [(d + phase, Fraction(-1))]
            upper = [(d + phase, _ONE)]
            meter.charge(work=2, entries=2)
            for j, signs in enumerate(self.signs):
                meter.charge(work=1)
                if signs[phase] > 0:
                    lower.append((offset + j, _TWO))
                elif signs[phase] < 0:
                    upper.append((offset + j, _TWO))
            # Always emit all eight rows, including empty sign masks.
            le.append(_row(lower, _ONE, n_columns, meter))
            le.append(_row(upper, _ONE, n_columns, meter))
        meter.charge(work=len(eq) + len(le), entries=len(eq) + len(le) + 7)
        return Compiled(tuple(eq), tuple(le), n_columns, self.binding.physical_ids,
                        self.binding.phase_ids, n, self.budget)


@dataclass(frozen=True)
class _Piece:
    values: tuple
    direction: tuple
    extent: object


def _factor_pieces(spec, meter):
    factors = []
    for i, (lo, hi) in enumerate(spec.parent_bounds):
        meter.charge(work=5, entries=25)
        f, q = i, i + 2
        factors.append((
            _Piece(((f, lo), (q, _ZERO)), (), None),
            _Piece(((f, _ZERO), (q, _ZERO)), (), None),
            _Piece(((f, hi), (q, hi)), (), None),
            _Piece(((f, lo), (q, _ZERO)), ((f, _ONE),), _neg(lo, meter)),
            _Piece(((f, _ZERO), (q, _ZERO)), ((f, _ONE), (q, _ONE)), hi),
        ))
    for j, (lo, hi) in enumerate(spec.residual_bounds):
        coordinate = 4 + j
        meter.charge(work=1, entries=3)
        if _cmp(lo, hi, meter) == 0:
            factors.append((_Piece(((coordinate, lo),), (), None),))
        else:
            meter.charge(work=3, entries=9)
            factors.append((
                _Piece(((coordinate, lo),), (), None),
                _Piece(((coordinate, hi),), (), None),
                _Piece(((coordinate, lo),), ((coordinate, _ONE),), _sub(hi, lo, meter)),
            ))
    return tuple(factors)


def _affine(spec, child, point, meter):
    return _add(_dot(spec.coefficients[child], point, meter), spec.biases[child], meter)


def _point(origin, directions, coordinates, meter):
    result = list(origin)
    meter.charge(work=len(result), entries=len(result))
    for direction, coordinate in zip(directions, coordinates):
        for i, coefficient in enumerate(direction):
            meter.charge(work=1)
            if coefficient.numerator != 0:
                result[i] = _add(result[i], _mul(coefficient, coordinate, meter), meter)
    return tuple(result)


def _inside(coordinates, extents, meter):
    for coordinate, extent in zip(coordinates, extents):
        if _cmp(coordinate, _ZERO, meter) < 0 or _cmp(coordinate, extent, meter) > 0:
            return False
    return True


def build(spec, binding, *, enabled=False, budget=None):
    """Build one local bank.  Disabled unless ``enabled is True``.

    ``candidate_count`` counts accepted feasible face/subset intersections
    before deduplication, including the original grid.  It is not the number
    of face tests.  Constant residual coordinates remain in physical output.
    """
    if enabled is not True:
        raise Rejected("the joint-bank primitive is opt-in and defaults off")
    if type(spec) is not Spec or type(binding) is not Binding:
        raise Rejected("build requires Spec and Binding")
    if len(binding.physical_ids) != 6 + len(spec.residual_bounds):
        raise Rejected("binding dimension does not match Spec")
    if budget is None:
        budget = Budget()
    if type(budget) is not Budget:
        raise Rejected("budget must be Budget")
    meter = budget._branch()
    for pair in spec.parent_bounds:
        lo, hi = (_number(v, meter) for v in pair)
        if _cmp(lo, _ZERO, meter) >= 0 or _cmp(hi, _ZERO, meter) <= 0:
            raise Rejected("invalid crossing interval")
    for pair in spec.residual_bounds:
        lo, hi = (_number(v, meter) for v in pair)
        if _cmp(lo, hi, meter) > 0:
            raise Rejected("reversed residual interval")
    for row in spec.coefficients:
        for value in row:
            _number(value, meter)
    for value in spec.biases:
        _number(value, meter)
    meter.charge(work=len(binding.physical_ids) + 4,
                 entries=len(binding.physical_ids) + 4)
    base_dim = 4 + len(spec.residual_bounds)
    factors = _factor_pieces(spec, meter)
    effective = sum(lo != hi for lo, hi in spec.residual_bounds)
    original_grid_count = 9 * (2 ** effective)
    vertex_limit = (37, 104, 277)[effective]
    pool = {}
    candidate_count = 0

    def add_candidate(point, selected):
        nonlocal candidate_count
        child_values = tuple(_affine(spec, j, point, meter) for j in range(2))
        for child in selected:
            if _cmp(child_values[child], _ZERO, meter) != 0:
                raise Rejected("intersection does not satisfy its exact zero equations")
        signs = tuple(_cmp(value, _ZERO, meter)
                      for value in (point[0], point[1]) + child_values)
        outputs = tuple(value if sign > 0 else _ZERO
                        for value, sign in zip(child_values, signs[2:]))
        physical = point + outputs
        meter.charge(work=len(physical) + 1, entries=len(physical) + 5)
        candidate_count += 1
        if physical in pool and pool[physical] != signs:
            raise Rejected("inconsistent signs for a duplicate graph point")
        pool[physical] = signs

    for pieces in product(*factors):
        meter.charge(work=len(pieces), entries=len(pieces))
        dimension = sum(piece.extent is not None for piece in pieces)
        if dimension > 2:
            continue
        meter.charge(work=base_dim, entries=base_dim)
        origin = [_ZERO] * base_dim
        directions = []
        extents = []
        for piece in pieces:
            for coordinate, value in piece.values:
                meter.charge(work=1)
                origin[coordinate] = value
            if piece.extent is not None:
                meter.charge(work=base_dim, entries=base_dim + 1)
                direction = [_ZERO] * base_dim
                for coordinate, value in piece.direction:
                    meter.charge(work=1)
                    direction[coordinate] = value
                directions.append(tuple(direction))
                extents.append(piece.extent)
        origin = tuple(origin)
        if dimension == 0:
            add_candidate(origin, ())
            continue
        intercepts = tuple(_affine(spec, j, origin, meter) for j in range(2))
        matrix = tuple(tuple(_dot(spec.coefficients[j], direction, meter)
                             for direction in directions) for j in range(2))
        meter.charge(work=2 * dimension, entries=2 * dimension + 2)
        if dimension == 1:
            for child in range(2):
                pivot = matrix[child][0]
                if _cmp(pivot, _ZERO, meter) == 0:
                    continue
                coordinate = _div(_neg(intercepts[child], meter), pivot, meter)
                if _inside((coordinate,), extents, meter):
                    add_candidate(_point(origin, directions, (coordinate,), meter), (child,))
        else:
            a, b = matrix[0]
            c, d = matrix[1]
            determinant = _sub(_mul(a, d, meter), _mul(b, c, meter), meter)
            if _cmp(determinant, _ZERO, meter) == 0:
                continue
            rhs0, rhs1 = (_neg(value, meter) for value in intercepts)
            first = _div(_sub(_mul(rhs0, d, meter), _mul(b, rhs1, meter), meter),
                         determinant, meter)
            second = _div(_sub(_mul(a, rhs1, meter), _mul(rhs0, c, meter), meter),
                          determinant, meter)
            coordinates = (first, second)
            if _inside(coordinates, extents, meter):
                add_candidate(_point(origin, directions, coordinates, meter), (0, 1))

    def compare(left, right):
        for a, b in zip(left, right):
            comparison = _cmp(a, b, meter)
            if comparison != 0:
                return comparison
        return 0

    if not pool or candidate_count > vertex_limit or len(pool) > vertex_limit:
        raise Rejected("vertex-count invariant failed")
    meter.charge(work=len(pool), entries=len(pool))
    vertices = tuple(sorted(pool, key=cmp_to_key(compare)))
    signs = tuple(pool[vertex] for vertex in vertices)
    meter.charge(work=len(vertices), entries=len(vertices) + 8)
    return Bank(spec, binding, vertices, signs, budget, original_grid_count, candidate_count)
