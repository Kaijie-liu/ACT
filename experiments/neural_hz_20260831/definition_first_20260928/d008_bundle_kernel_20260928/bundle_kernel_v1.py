"""Default-off D008 conservation-row kernel, not a verifier or solver.

Rows strengthen an external exact ReLU encoding. This module neither replaces
its source equations/phase guards nor checks them. Constructors return immutable
exact data; all bounded arithmetic failures reject without a partial result.
"""

from dataclasses import dataclass, field
from fractions import Fraction


MAX_ROWS = 64
MAX_PORTS = 32
MAX_SUPPORT = 32
MAX_CERTIFICATES = 1
MAX_CLOSURE_ROWS = 33
MAX_ROW_NNZ = 2112
MAX_BITS = 512
GRID = 65536
PROPOSAL_NORM = 64


def _q(value):
    if type(value) not in (int, Fraction):
        raise TypeError("exact int or Fraction required; bool and float are forbidden")
    value = Fraction(value)
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_BITS:
        raise ValueError("512-bit rational limit exceeded")
    return value


def _add(left, right):
    return _q(left + right)


def _sub(left, right):
    return _q(left - right)


def _mul(left, right):
    return _q(left * right)


def _div(left, right):
    if not right:
        raise ValueError("zero divisor")
    return _q(left / right)


def _abs(value):
    return _q(abs(value))


def _collect(values, maximum):
    result = []
    for index, value in enumerate(values):
        if index >= maximum:
            raise ValueError("bounded collection limit exceeded")
        result.append(value)
    return tuple(result)


def _numbers(values, maximum, expected=None):
    result = []
    for index, value in enumerate(values):
        if index >= maximum:
            raise ValueError("vector arity or collection limit exceeded")
        result.append(_q(value))
    if expected is not None and len(result) != expected:
        raise ValueError("wrong vector arity")
    return tuple(result)


@dataclass(frozen=True, slots=True, eq=False)
class Bank:
    weights: tuple
    bias: tuple
    lower: tuple
    upper: tuple
    owner: object = field(default_factory=object, repr=False)

    @property
    def rows(self):
        return len(self.weights)

    @property
    def cols(self):
        return len(self.lower)


def make_bank(weights, bias, lower, upper, *, enabled=False):
    """Strict opt-in; consume at most the relevant cap plus one item."""
    if enabled is not True:
        return None
    matrix = []
    width = None
    for index, row in enumerate(weights):
        if index >= MAX_ROWS:
            raise ValueError("bank row limit exceeded")
        row = _numbers(row, MAX_PORTS, width)
        if not row:
            raise ValueError("at least one source port is required")
        width = len(row)
        matrix.append(row)
    if not matrix:
        raise ValueError("at least one bank row is required")
    bias = _numbers(bias, len(matrix), len(matrix))
    lower = _numbers(lower, width, width)
    upper = _numbers(upper, width, width)
    if any(lo > hi for lo, hi in zip(lower, upper)):
        raise ValueError("reversed source bounds")
    return Bank(tuple(matrix), bias, lower, upper)


def _require_bank(bank):
    if type(bank) is not Bank:
        raise TypeError("a Bank from make_bank is required")


def _residual(bank, a):
    residual = [Fraction(0)] * bank.cols
    for coefficient, row in zip(a, bank.weights):
        if coefficient:
            for column, value in enumerate(row):
                residual[column] = _add(residual[column], _mul(coefficient, value))
    return tuple(residual)


@dataclass(frozen=True, slots=True)
class Cert:
    bank: Bank
    a: tuple
    support: tuple
    d: tuple
    c: Fraction
    emin: Fraction
    emax: Fraction
    epsilon: Fraction


def certify(bank, a):
    """Certify a centered affine defect on this bank's original source box."""
    _require_bank(bank)
    a = _numbers(a, bank.rows, bank.rows)
    support = tuple(index for index, value in enumerate(a) if value)
    if len(support) > MAX_SUPPORT:
        raise ValueError("certificate support limit exceeded")
    residual = _residual(bank, a)
    offset = Fraction(0)
    for index in support:
        offset = _add(offset, _mul(a[index], bank.bias[index]))
    center = offset
    for coefficient, lo, hi in zip(residual, bank.lower, bank.upper):
        midpoint = _div(_add(lo, hi), Fraction(2))
        center = _add(center, _mul(coefficient, midpoint))
    minimum = maximum = _sub(offset, center)
    for coefficient, lo, hi in zip(residual, bank.lower, bank.upper):
        first, second = _mul(coefficient, lo), _mul(coefficient, hi)
        minimum = _add(minimum, min(first, second))
        maximum = _add(maximum, max(first, second))
    epsilon = max(_abs(minimum), _abs(maximum))
    return Cert(bank, a, support, residual, center, minimum, maximum, epsilon)


def _check_cert(bank, cert):
    if type(cert) is not Cert or cert.bank is not bank:
        raise ValueError("certificate belongs to a different bank object")
    if type(cert.a) is not tuple or type(cert.d) is not tuple or type(cert.support) is not tuple:
        raise ValueError("certificate vectors must be immutable tuples")
    if any(type(index) is not int for index in cert.support):
        raise TypeError("certificate support indices must be ints")
    checked = certify(bank, cert.a)
    claimed = (_numbers(cert.d, bank.cols, bank.cols), _q(cert.c), _q(cert.emin),
               _q(cert.emax), _q(cert.epsilon))
    expected = (checked.d, checked.c, checked.emin, checked.emax, checked.epsilon)
    if cert.support != checked.support or claimed != expected:
        raise ValueError("certificate does not match exact recomputation")
    return checked


@dataclass(frozen=True, slots=True)
class Row:
    """Sparse linear inequality: sum(coefficient * variable[index]) <= rhs."""

    terms: tuple
    rhs: Fraction


def _canonical(terms, rhs):
    ordered = tuple((index, coefficient) for index, coefficient in sorted(terms.items())
                    if coefficient)
    if not ordered:
        return None if rhs >= 0 else Row((), Fraction(-1))
    scale = _abs(ordered[0][1])
    return Row(tuple((index, _div(coefficient, scale)) for index, coefficient in ordered),
               _div(rhs, scale))


def _magnitude_row(bank, cert, member):
    # member=None is the anchor; every other row has one positive magnitude.
    rhs = (_add(_abs(cert.c), cert.epsilon) if member is not None
           else _sub(cert.epsilon, _abs(cert.c)))
    terms = {}
    for index in cert.support:
        coefficient = _abs(cert.a[index])
        if index != member:
            coefficient = _sub(Fraction(0), coefficient)
        terms[bank.cols + index] = _mul(Fraction(2), coefficient)
        for column, weight in enumerate(bank.weights[index]):
            terms[column] = _sub(terms.get(column, Fraction(0)), _mul(coefficient, weight))
        rhs = _add(rhs, _mul(coefficient, bank.bias[index]))
    return _canonical(terms, rhs)


@dataclass(frozen=True, slots=True)
class Closure:
    bank: Bank
    certs: tuple
    rows: tuple

    @property
    def counts(self):
        nnz = sum(len(row.terms) for row in self.rows)
        return {"n_ports": self.bank.cols, "n_activations": self.bank.rows,
                "n_certificates": len(self.certs), "n_rows": len(self.rows),
                "row_nnz": nnz, "row_rhs": len(self.rows),
                "row_coefficients": nnz + len(self.rows),
                "support_total": sum(len(cert.support) for cert in self.certs),
                "assignment_size": self.bank.cols + self.bank.rows}

    def satisfies_rows(self, x, r):
        """Check ONLY extra rows; box/source/ReLU/phase checks are external."""
        assignment = (_numbers(x, self.bank.cols, self.bank.cols)
                      + _numbers(r, self.bank.rows, self.bank.rows))
        feasible = True
        for row in self.rows:
            value = Fraction(0)
            for index, coefficient in row.terms:
                value = _add(value, _mul(coefficient, assignment[index]))
            feasible = (value <= row.rhs) and feasible
        return feasible


def compile_rows(bank, certs, *, enabled=False):
    """Recheck every certificate, emit all member/anchor rows, deduplicate."""
    if enabled is not True:
        return None
    _require_bank(bank)
    supplied = _collect(certs, MAX_CERTIFICATES)
    checked = tuple(_check_cert(bank, cert) for cert in supplied)
    rows, seen = [], set()
    nnz = 0
    for cert in checked:
        for member in cert.support + (None,):
            row = _magnitude_row(bank, cert, member)
            if row is None or row in seen:
                continue
            if len(rows) >= MAX_CLOSURE_ROWS or nnz + len(row.terms) > MAX_ROW_NNZ:
                raise ValueError("compiled row/nnz reference limit exceeded")
            rows.append(row)
            seen.add(row)
            nnz += len(row.terms)
    return Closure(bank, checked, tuple(rows))


def _quantize(value):
    """Nearest multiple of 2^-16; exact ties-to-even, including negatives."""
    scaled = _mul(value, Fraction(GRID))
    floor = scaled.numerator // scaled.denominator
    integer = _q(floor)
    remainder = _sub(scaled, integer)
    doubled = _mul(Fraction(2), remainder)
    if doubled > 1 or (doubled == 1 and floor % 2):
        integer = _add(integer, Fraction(1))
    return _div(integer, Fraction(GRID))


def _bounded_proposal(a):
    return sum(bool(value) for value in a) <= MAX_SUPPORT and all(
        _abs(value) <= PROPOSAL_NORM for value in a
    )


def select_certificate(bank, *, enabled=False):
    """First bounded quantized forward-elimination proposal; no scoring/search.

    At most 64 source rows, 32 basis rows, and 32 eliminations per source row.
    Quantization changes only a proposed certificate, never the source bank.
    A small residual is a proposal criterion, not an asserted exact identity.
    """
    if enabled is not True:
        return None
    _require_bank(bank)
    basis, pivots = [], set()
    threshold = Fraction(1, GRID)
    for row_index in range(bank.rows):
        a = tuple(Fraction(int(index == row_index)) for index in range(bank.rows))
        residual = _residual(bank, a)
        for basis_a, basis_v, pivot in basis:
            ratio = _div(residual[pivot], basis_v[pivot])
            a = tuple(_quantize(_sub(value, _mul(ratio, old)))
                      for value, old in zip(a, basis_a))
            residual = _residual(bank, a)
        available = tuple(column for column in range(bank.cols) if column not in pivots)
        pivot = (max(available, key=lambda column: (_abs(residual[column]), -column))
                 if available else None)
        bounded = _bounded_proposal(a)
        if pivot is None or _abs(residual[pivot]) <= threshold:
            if bounded:
                return (certify(bank, a),)
            continue
        if bounded:
            if len(basis) >= MAX_PORTS:
                raise ValueError("proposal basis limit exceeded")
            basis.append((a, residual, pivot))
            pivots.add(pivot)
    return ()
