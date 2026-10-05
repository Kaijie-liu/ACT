"""Default-off exact D009 energy certificates; not a verifier or source adapter.

Rows use source x followed by ReLU values r.  The integration reference must
retain its own source equations, bounds, every original bit, gate and predicate.
Energy facts describe the vector expression identified by their proof DAG;
they do not identify that expression with an external network automatically.
"""

from dataclasses import dataclass, field
from fractions import Fraction


MAX_BITS = 512
MAX_ROWS = 64
MAX_PORTS = 32
MAX_VECTOR = 64
MAX_PROOF_NODES = 64
MAX_PROOF_DEPTH = 32
ZERO = Fraction(0)
ONE = Fraction(1)
TWO = Fraction(2)


def _q(value):
    if type(value) not in (int, Fraction):
        raise TypeError("only int and Fraction are accepted")
    if type(value) is int:
        if value.bit_length() > MAX_BITS:
            raise ValueError("rational numerator exceeds 512 bits")
        value = Fraction(value)
    if (value.numerator.bit_length() > MAX_BITS
            or value.denominator.bit_length() > MAX_BITS):
        raise ValueError("rational exceeds 512 bits")
    return value


def _add(a, b):
    return _q(a + b)


def _sub(a, b):
    return _q(a - b)


def _mul(a, b):
    return _q(a * b)


def _div(a, b):
    if b == ZERO:
        raise ValueError("zero divisor")
    return _q(a / b)


def _sum(values):
    total = ZERO
    for value in values:
        total = _add(total, value)
    return total


def _take(values, cap, label):
    """Read at most cap+1 items, including from an unbounded iterator."""
    result = []
    iterator = iter(values)
    for _ in range(cap + 1):
        try:
            value = next(iterator)
        except StopIteration:
            return tuple(result)
        if len(result) == cap:
            raise ValueError(label + " exceeds its size limit")
        result.append(value)
    raise ValueError(label + " exceeds its size limit")


def _vector(values, size, label, nonnegative=False):
    result = _take(values, size, label)
    if len(result) != size:
        raise ValueError(label + " has wrong arity")
    result = tuple(_q(value) for value in result)
    if nonnegative and any(value < ZERO for value in result):
        raise ValueError(label + " must be nonnegative")
    return result


def _matrix(values, cols, label):
    rows = _take(values, MAX_ROWS, label)
    if not rows:
        raise ValueError(label + " must have at least one row")
    return tuple(_vector(row, cols, label + " row") for row in rows)


def _dot(a, b):
    return _sum(_mul(x, y) for x, y in zip(a, b))


@dataclass(frozen=True, eq=False)
class Bank:
    weights: tuple
    bias: tuple
    lower: tuple
    upper: tuple
    owner: object = field(repr=False)

    @property
    def rows(self):
        return len(self.weights)

    @property
    def cols(self):
        return len(self.lower)


@dataclass(frozen=True)
class MassCert:
    bank: Bank = field(repr=False)
    weights: tuple
    gram: tuple
    center: tuple
    radius: tuple
    B: Fraction
    c: Fraction
    q: Fraction


@dataclass(frozen=True)
class Row:
    terms: tuple
    rhs: Fraction
    size: int = field(repr=False)

    def satisfies(self, assignment):
        if type(self.size) is not int or not 1 <= self.size <= MAX_PORTS + MAX_ROWS:
            raise ValueError("invalid row assignment size")
        if type(self.terms) is not tuple or len(self.terms) > self.size:
            raise ValueError("row terms exceed size limit")
        _q(self.rhs)
        for term in self.terms:
            if type(term) is not tuple or len(term) != 2:
                raise ValueError("malformed row term")
            index, coefficient = term
            if type(index) is not int or not 0 <= index < self.size:
                raise ValueError("row index out of range")
            _q(coefficient)
        values = _vector(assignment, self.size, "row assignment")
        return _sum(_mul(coef, values[index])
                    for index, coef in self.terms) <= self.rhs


@dataclass(frozen=True, eq=False)
class Energy:
    bank: Bank = field(repr=False)
    kind: str
    center: tuple
    weights: tuple
    bound: Fraction
    parents: tuple = field(repr=False)
    matrix: tuple
    bias: tuple
    factor: Fraction

    @property
    def width(self):
        return len(self.center)

    @property
    def owner(self):
        return self.bank.owner


def _bank_data(W, b, L, U):
    lower = _take(L, MAX_PORTS, "lower bounds")
    if not lower:
        raise ValueError("bank must have at least one source port")
    lower = _vector(lower, len(lower), "lower bounds")
    upper = _vector(U, len(lower), "upper bounds")
    if any(lo > hi for lo, hi in zip(lower, upper)):
        raise ValueError("reversed source bounds")
    weights = _matrix(W, len(lower), "source weights")
    bias = _vector(b, len(weights), "source bias")
    return weights, bias, lower, upper


def make_bank(W, b, L, U, *, enabled=False):
    if enabled is not True:
        return None
    return Bank(*_bank_data(W, b, L, U), object())


def _stored_vector(values, cap, label):
    if type(values) is not tuple or len(values) > cap:
        raise ValueError(label + " must be a bounded immutable tuple")
    for value in values:
        if type(value) is not Fraction:
            raise ValueError(label + " must contain stored Fractions")
        _q(value)


def _stored_matrix(values, cap, label):
    if type(values) is not tuple or len(values) > MAX_ROWS:
        raise ValueError(label + " must be a bounded immutable matrix")
    for row in values:
        _stored_vector(row, cap, label + " row")


def _require_bank(bank):
    if type(bank) is not Bank:
        raise TypeError("expected a Bank")
    if type(bank.owner) is not object:
        raise ValueError("invalid bank owner")
    _stored_matrix(bank.weights, MAX_PORTS, "stored source weights")
    for values in (bank.bias, bank.lower, bank.upper):
        _stored_vector(values, MAX_ROWS, "stored source vector")
    _bank_data(bank.weights, bank.bias, bank.lower, bank.upper)


def _gram(matrix, weights):
    width = len(matrix[0])
    result = [[ZERO for _ in range(width)] for _ in range(width)]
    for j in range(width):
        for k in range(j, width):
            value = _sum(_mul(_mul(w, row[j]), row[k])
                         for w, row in zip(weights, matrix))
            result[j][k] = value
            result[k][j] = value
    return tuple(tuple(row) for row in result)


def _gram_data(bank, weights):
    midpoint = tuple(_div(_add(lo, hi), TWO)
                     for lo, hi in zip(bank.lower, bank.upper))
    radius = tuple(_div(_sub(hi, lo), TWO)
                   for lo, hi in zip(bank.lower, bank.upper))
    center = tuple(_add(_dot(row, midpoint), bias)
                   for row, bias in zip(bank.weights, bank.bias))
    scaled = tuple(tuple(_mul(value, rad) for value, rad in zip(row, radius))
                   for row in bank.weights)
    gram = _gram(scaled, weights)
    diagonal = _sum(gram[j][j] for j in range(bank.cols))
    off_diagonal = _sum(abs(gram[j][k]) for j in range(bank.cols)
                        for k in range(j + 1, bank.cols))
    bound = _add(diagonal, _mul(TWO, off_diagonal))
    c = _sum(_mul(w, abs(z)) for w, z in zip(weights, center))
    return gram, center, radius, bound, c


def certify(bank, w, q):
    _require_bank(bank)
    weights = _vector(w, bank.rows, "mass weights", nonnegative=True)
    q = _q(q)
    if q < ZERO:
        raise ValueError("root upper bound must be nonnegative")
    gram, center, radius, bound, c = _gram_data(bank, weights)
    if _mul(q, q) < _mul(_sum(weights), bound):
        raise ValueError("root upper bound is insufficient")
    return MassCert(bank, weights, gram, center, radius, bound, c, q)


def compile_row(bank, cert, *, enabled=False):
    if enabled is not True:
        return None
    _require_bank(bank)
    if type(cert) is not MassCert:
        raise TypeError("expected a MassCert")
    if cert.bank is not bank:
        raise ValueError("certificate belongs to a different bank")
    for values in (cert.weights, cert.center, cert.radius):
        _stored_vector(values, MAX_ROWS, "stored certificate vector")
    _stored_matrix(cert.gram, MAX_PORTS, "stored certificate Gram")
    for value in (cert.B, cert.c, cert.q):
        if type(value) is not Fraction:
            raise ValueError("certificate scalar is not a stored Fraction")
        _q(value)
    if cert != certify(bank, cert.weights, cert.q):
        raise ValueError("certificate payload does not recompute")
    terms = []
    for j in range(bank.cols):
        coefficient = -_sum(_mul(w, row[j])
                            for w, row in zip(cert.weights, bank.weights))
        if coefficient:
            terms.append((j, coefficient))
    for i, w in enumerate(cert.weights):
        coefficient = _mul(TWO, w)
        if coefficient:
            terms.append((bank.cols + i, coefficient))
    rhs = _add(_add(cert.c, cert.q), _dot(cert.weights, bank.bias))
    return Row(tuple(terms), rhs, bank.cols + bank.rows)


def _derive(kind, bank, parents, weights=(), matrix=(), bias=()):
    """Recompute one node after all parent premises have been checked."""
    if kind == "gram":
        if parents or matrix or bias:
            raise ValueError("Gram fact has extraneous premises")
        weights = _vector(weights, bank.rows, "energy weights", True)
        _, center, _, bound, _ = _gram_data(bank, weights)
        factor = ONE
    elif kind == "affine":
        if len(parents) != 1:
            raise ValueError("affine fact needs one parent")
        parent = parents[0]
        matrix = _matrix(matrix, parent.width, "affine weights")
        bias = _vector(bias, len(matrix), "affine bias")
        weights = _vector(weights, len(matrix), "output weights", True)
        gram = _gram(matrix, weights)
        factor = ZERO
        for j, weight in enumerate(parent.weights):
            row_bound = _sum(abs(value) for value in gram[j])
            if weight == ZERO:
                if row_bound != ZERO:
                    raise ValueError("affine image uses an unweighted source")
            else:
                factor = max(factor, _div(row_bound, weight))
        center = tuple(_add(_dot(row, parent.center), value)
                       for row, value in zip(matrix, bias))
        bound = _mul(factor, parent.bound)
    else:
        if matrix or bias:
            raise ValueError("unexpected matrix or bias in energy fact")
        if kind == "relu" and len(parents) == 1:
            parent = parents[0]
            center = tuple(max(ZERO, value) for value in parent.center)
            weights, bound, factor = parent.weights, parent.bound, ONE
        elif kind == "add" and len(parents) == 2:
            left, right = parents
            if left.width != right.width or left.weights != right.weights:
                raise ValueError("Add needs identical diagonal weights and width")
            center = tuple(_add(a, b) for a, b in zip(left.center, right.center))
            weights = left.weights
            bound, factor = _mul(TWO, _add(left.bound, right.bound)), TWO
        elif kind == "concat" and len(parents) == 2:
            left, right = parents
            if left.width + right.width > MAX_VECTOR:
                raise ValueError("Concat exceeds vector limit")
            center, weights = left.center + right.center, left.weights + right.weights
            bound, factor = _add(left.bound, right.bound), ONE
        else:
            raise ValueError("unknown energy rule or wrong parent arity")
    return Energy(bank, kind, center, weights, bound, parents, matrix, bias, factor)


def _replay(fact):
    if type(fact) is not Energy:
        raise TypeError("expected an Energy fact")
    bank = fact.bank
    _require_bank(bank)
    heights, active = {}, set()

    def visit(node, depth):
        if depth > MAX_PROOF_DEPTH:
            raise ValueError("energy proof exceeds depth limit")
        if type(node) is not Energy:
            raise TypeError("expected an Energy proof node")
        if node.bank is not bank:
            raise ValueError("energy proof mixes bank identities")
        identity = id(node)
        if identity in active:
            raise ValueError("cyclic energy proof")
        if identity in heights:
            if depth + heights[identity] - 1 > MAX_PROOF_DEPTH:
                raise ValueError("energy proof exceeds depth limit")
            return heights[identity]
        if len(heights) + len(active) >= MAX_PROOF_NODES:
            raise ValueError("energy proof exceeds node limit")
        if type(node.parents) is not tuple or len(node.parents) > 2:
            raise ValueError("energy parents must be a bounded immutable tuple")
        _stored_vector(node.center, MAX_VECTOR, "stored energy center")
        _stored_vector(node.weights, MAX_VECTOR, "stored energy weights")
        _stored_vector(node.bias, MAX_VECTOR, "stored energy bias")
        _stored_matrix(node.matrix, MAX_VECTOR, "stored energy matrix")
        for value in (node.bound, node.factor):
            if type(value) is not Fraction:
                raise ValueError("energy scalar is not a stored Fraction")
            _q(value)
        active.add(identity)
        height = 1
        for parent in node.parents:
            height = max(height, 1 + visit(parent, depth + 1))
        expected = _derive(node.kind, bank, node.parents, node.weights,
                           node.matrix, node.bias)
        for name in ("center", "weights", "bound", "matrix", "bias", "factor"):
            if getattr(node, name) != getattr(expected, name):
                raise ValueError("energy payload does not recompute")
        active.remove(identity)
        heights[identity] = height
        return height

    visit(fact, 1)
    return fact


def _forward(kind, parents, weights=(), matrix=(), bias=()):
    for parent in parents:
        _replay(parent)
    bank = parents[0].bank
    if any(parent.bank is not bank for parent in parents):
        raise ValueError("forward operation mixes bank identities")
    result = _derive(kind, bank, parents, weights, matrix, bias)
    return _replay(result)


def gram_energy(bank, w, *, enabled=False):
    if enabled is not True:
        return None
    _require_bank(bank)
    return _replay(_derive("gram", bank, (), weights=w))


def relu_energy(fact, *, enabled=False):
    if enabled is not True:
        return None
    return _forward("relu", (fact,))


def affine_energy(fact, V, bias, out_weights, *, enabled=False):
    if enabled is not True:
        return None
    return _forward("affine", (fact,), out_weights, V, bias)


def add_energy(f1, f2, *, enabled=False):
    if enabled is not True:
        return None
    return _forward("add", (f1, f2))


def concat_energy(f1, f2, *, enabled=False):
    if enabled is not True:
        return None
    return _forward("concat", (f1, f2))
