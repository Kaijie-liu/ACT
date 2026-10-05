"""Default-off exact whole-box support for A(z) - max(0, H(z)).

This implements the D068 primal filling proof and D069 sparse update schedule.
It is not a network verifier: a decoded box maximizer need not satisfy the
caller's additional HZ predicates. No original source or phase is eliminated.

Owned immutable bases store all declared coordinates, including fixed/unused
ones. Each query derives a persistent AVL tree and a sparse coordinate overlay
from that SAME base. It does not copy the whole coordinate dictionary or sort
the whole source again. Retaining query roots retains their path copies; all
such proof storage remains payable. Full witness decoding costs O(d).

All rational inputs and arithmetic/comparison intermediates are checked at
512 bits. Inputs are canonical immutable tuples with at most 65536 aggregate
occurrences. These are component limits, not whole-memory or GPU qualification.
Private sealed objects are internal proof objects, not an untrusted certificate
format; arbitrary Python mutation/forgery is outside this constructor API.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from types import MappingProxyType

from experiments.neural_hz_20260831.definition_first_20260928.d049_mixed_source_envelopes_20260930 import mixed_source as ms

KernelError = ms.KernelError
MAX_BITS, MAX_SUPPORT = ms.MAX_BITS, ms.MAX_SUPPORT
ZERO, ONE = Fraction(0), Fraction(1)
_f, _on, _ordinal = ms._f, ms._on, ms.mp._ordinal
_SEAL = object()


def _integer(value):
    if abs(value).bit_length() > MAX_BITS:
        raise KernelError('rational intermediate bit limit exceeded')
    return value


def _add(left, right):
    left, right = _f(left), _f(right)
    divisor = gcd(left.denominator, right.denominator)
    ld, rd = left.denominator // divisor, right.denominator // divisor
    first = _integer(left.numerator * rd)
    second = _integer(right.numerator * ld)
    numerator = _integer(first + second)
    denominator = _integer(ld * right.denominator)
    return _f(Fraction(numerator, denominator))


def _neg(value):
    return _f(-_f(value))


def _sub(left, right):
    return _add(left, _neg(right))


def _mul(left, right):
    left, right = _f(left), _f(right)
    first = gcd(abs(left.numerator), right.denominator)
    second = gcd(abs(right.numerator), left.denominator)
    numerator = _integer((left.numerator // first) * (right.numerator // second))
    denominator = _integer((left.denominator // second) * (right.denominator // first))
    return _f(Fraction(numerator, denominator))


def _divide(left, right):
    left, right = _f(left), _f(right)
    if right == ZERO:
        raise KernelError('zero support divisor')
    return _mul(left, _f(Fraction(right.denominator, right.numerator)))


def _compare(left, right):
    left, right = _f(left), _f(right)
    # Comparisons are charged too, even when the final result is one bit.
    first = _integer(left.numerator * right.denominator)
    second = _integer(right.numerator * left.denominator)
    return (first > second) - (first < second)


def _minimum(left, right):
    return left if _compare(left, right) <= 0 else right


def _maximum(left, right):
    return left if _compare(left, right) >= 0 else right


def _count(*values):
    if any(type(value) is not tuple for value in values):
        raise KernelError('immutable support tuples required')
    if sum(len(value) for value in values) > MAX_SUPPORT:
        raise KernelError('aggregate support occurrence limit exceeded')


@dataclass(frozen=True, eq=False)
class _Node:
    rho: Fraction
    ordinal: int
    capacity: Fraction
    weighted: Fraction
    left: object
    right: object
    height: int
    size: int
    total_capacity: Fraction
    total_weighted: Fraction


@dataclass(frozen=True, eq=False)
class _Coordinate:
    ordinal: int
    lower: Fraction
    upper: Fraction
    a: Fraction
    h: Fraction
    endpoint: Fraction
    b0: Fraction
    h0: Fraction
    capacity: Fraction
    rho: object


@dataclass(frozen=True, eq=False)
class Prepared:
    bounds: tuple
    a_bias: Fraction
    a_terms: tuple
    h_bias: Fraction
    h_terms: tuple
    coordinates: MappingProxyType
    root: object
    b0: Fraction
    h0: Fraction
    _seal: object


@dataclass(frozen=True, eq=False)
class QueryResult:
    base: Prepared
    a_bias_delta: Fraction
    a_updates: tuple
    h_bias_delta: Fraction
    h_updates: tuple
    changes: MappingProxyType
    root: object
    b0: Fraction
    h0: Fraction
    target: Fraction
    value: Fraction
    _seal: object


def _height(node):
    return 0 if node is None else node.height


def _capacity(node):
    return ZERO if node is None else node.total_capacity


def _weighted(node):
    return ZERO if node is None else node.total_weighted


def _node(rho, ordinal, capacity, left=None, right=None):
    weighted = _mul(rho, capacity)
    total_capacity = _add(_add(_capacity(left), capacity), _capacity(right))
    total_weighted = _add(_add(_weighted(left), weighted), _weighted(right))
    size = 1 + (0 if left is None else left.size) + (0 if right is None else right.size)
    if size > MAX_SUPPORT:
        raise KernelError('merged support tree limit exceeded')
    return _Node(rho, ordinal, capacity, weighted, left, right,
                 1 + max(_height(left), _height(right)), size,
                 total_capacity, total_weighted)


def _key_compare(rho, ordinal, node):
    different = _compare(rho, node.rho)
    if different:
        return -different  # Descending ratio; ascending original ID for ties.
    return (ordinal > node.ordinal) - (ordinal < node.ordinal)


def _rotate_left(node):
    child = node.right
    if child is None:
        raise KernelError('invalid AVL left rotation')
    left = _node(node.rho, node.ordinal, node.capacity, node.left, child.left)
    return _node(child.rho, child.ordinal, child.capacity, left, child.right)


def _rotate_right(node):
    child = node.left
    if child is None:
        raise KernelError('invalid AVL right rotation')
    right = _node(node.rho, node.ordinal, node.capacity, child.right, node.right)
    return _node(child.rho, child.ordinal, child.capacity, child.left, right)


def _balance(node):
    delta = _height(node.left) - _height(node.right)
    if delta > 1:
        left = node.left
        if _height(left.left) < _height(left.right):
            left = _rotate_left(left)
            node = _node(node.rho, node.ordinal, node.capacity, left, node.right)
        return _rotate_right(node)
    if delta < -1:
        right = node.right
        if _height(right.right) < _height(right.left):
            right = _rotate_right(right)
            node = _node(node.rho, node.ordinal, node.capacity, node.left, right)
        return _rotate_left(node)
    return node


def _insert(node, coordinate):
    if node is None:
        return _node(coordinate.rho, coordinate.ordinal, coordinate.capacity)
    ordering = _key_compare(coordinate.rho, coordinate.ordinal, node)
    if ordering == 0:
        raise KernelError('duplicate support tree key')
    if ordering < 0:
        result = _node(node.rho, node.ordinal, node.capacity,
                       _insert(node.left, coordinate), node.right)
    else:
        result = _node(node.rho, node.ordinal, node.capacity,
                       node.left, _insert(node.right, coordinate))
    return _balance(result)


def _delete(node, rho, ordinal):
    if node is None:
        raise KernelError('missing support tree key')
    ordering = _key_compare(rho, ordinal, node)
    if ordering < 0:
        result = _node(node.rho, node.ordinal, node.capacity,
                       _delete(node.left, rho, ordinal), node.right)
    elif ordering > 0:
        result = _node(node.rho, node.ordinal, node.capacity,
                       node.left, _delete(node.right, rho, ordinal))
    else:
        if node.left is None:
            return node.right
        if node.right is None:
            return node.left
        successor = node.right
        while successor.left is not None:
            successor = successor.left
        result = _node(successor.rho, successor.ordinal, successor.capacity,
                       node.left, _delete(node.right, successor.rho, successor.ordinal))
    return _balance(result)


def _threshold_capacity(node, threshold):
    total = ZERO
    while node is not None:
        if _compare(node.rho, threshold) > 0:
            total = _add(total, _add(_capacity(node.left), node.capacity))
            node = node.right
        else:
            node = node.left
    return total


def _prefix_weight(node, target):
    if _compare(target, ZERO) < 0 or _compare(target, _capacity(node)) > 0:
        raise KernelError('prefix target outside total capacity')
    total = ZERO
    while node is not None and target != ZERO:
        left_capacity = _capacity(node.left)
        if _compare(target, left_capacity) <= 0:
            node = node.left
            continue
        total = _add(total, _weighted(node.left))
        target = _sub(target, left_capacity)
        amount = _minimum(target, node.capacity)
        total = _add(total, _mul(node.rho, amount))
        target = _sub(target, amount)
        node = node.right
    if target != ZERO:
        raise KernelError('incomplete support prefix')
    return total


def _coordinate(ordinal, lower, upper, a, h):
    if lower == upper:
        return _Coordinate(ordinal, lower, upper, a, h, lower,
                           _mul(a, lower), _mul(h, lower), ZERO, None)
    if h == ZERO:
        endpoint = upper if a > ZERO else lower
        return _Coordinate(ordinal, lower, upper, a, h, endpoint,
                           _mul(a, endpoint), ZERO, ZERO, None)
    endpoint = lower if h > ZERO else upper
    capacity = _mul(_neg(h) if h < ZERO else h, _sub(upper, lower))
    return _Coordinate(ordinal, lower, upper, a, h, endpoint,
                       _mul(a, endpoint), _mul(h, endpoint), capacity,
                       _divide(a, h))


def _terms(terms, declared):
    result, previous = {}, -1
    for item in terms:
        if type(item) is not tuple or len(item) != 2:
            raise KernelError('terms must be (original ordinal, Fraction)')
        ordinal, coefficient = _ordinal(item[0]), _f(item[1])
        if ordinal <= previous or ordinal not in declared:
            raise KernelError('terms need declared strictly ordered original ordinals')
        previous = ordinal
        if coefficient != ZERO:
            result[ordinal] = coefficient
    return result


def prepare(bounds, a_bias, a_terms, h_bias, h_terms, *, enabled=False):
    """Prepare a fixed complete source box and two exact affine forms.

    bounds: immutable (ordinal, lower, upper) declarations in increasing order.
    terms: immutable (ordinal, coefficient) tuples; zero coefficients are valid
    occurrences and count toward the cap even when absent from the merged map.
    """
    if not _on(enabled):
        return None
    _count(bounds, a_terms, h_terms)
    a_bias, h_bias = _f(a_bias), _f(h_bias)
    declared, previous = {}, -1
    for item in bounds:
        if type(item) is not tuple or len(item) != 3:
            raise KernelError('bounds must be (original ordinal, lower, upper)')
        ordinal, lower, upper = _ordinal(item[0]), _f(item[1]), _f(item[2])
        if ordinal <= previous or _compare(lower, upper) > 0:
            raise KernelError('bounds require ordered unique ordinals and nonempty intervals')
        previous = ordinal
        declared[ordinal] = (lower, upper)
    a_values, h_values = _terms(a_terms, declared), _terms(h_terms, declared)
    coordinates, root, b0, h0 = {}, None, a_bias, h_bias
    for ordinal, lower, upper in bounds:
        coordinate = _coordinate(ordinal, lower, upper,
                                 a_values.get(ordinal, ZERO), h_values.get(ordinal, ZERO))
        coordinates[ordinal] = coordinate
        b0, h0 = _add(b0, coordinate.b0), _add(h0, coordinate.h0)
        if coordinate.rho is not None:
            root = _insert(root, coordinate)
    return Prepared(bounds, a_bias, a_terms, h_bias, h_terms,
                    MappingProxyType(coordinates), root, b0, h0, _SEAL)


def _prepared(value):
    if type(value) is not Prepared or value._seal is not _SEAL:
        raise KernelError('an owned support base is required')
    if type(value.coordinates) is not MappingProxyType:
        raise KernelError('immutable support coordinate registry required')
    return value


def query(base, a_bias_delta=ZERO, a_updates=(), h_bias_delta=ZERO,
          h_updates=(), *, enabled=False):
    """Add sparse coefficient/bias deltas to the fixed base, not prior queries."""
    if not _on(enabled):
        return None
    base = _prepared(base)
    _count(a_updates, h_updates)
    a_bias_delta, h_bias_delta = _f(a_bias_delta), _f(h_bias_delta)
    a_values = _terms(a_updates, base.coordinates)
    h_values = _terms(h_updates, base.coordinates)
    indices = sorted(a_values.keys() | h_values.keys())
    changes, root = {}, base.root
    b0, h0 = _add(base.b0, a_bias_delta), _add(base.h0, h_bias_delta)
    for ordinal in indices:
        old = base.coordinates[ordinal]
        a, h = _add(old.a, a_values.get(ordinal, ZERO)), _add(old.h, h_values.get(ordinal, ZERO))
        new = _coordinate(ordinal, old.lower, old.upper, a, h)
        if old.rho is not None:
            root = _delete(root, old.rho, ordinal)
        b0, h0 = _add(_sub(b0, old.b0), new.b0), _add(_sub(h0, old.h0), new.h0)
        if new.rho is not None:
            root = _insert(root, new)
        changes[ordinal] = new
    p1, p0 = _threshold_capacity(root, ONE), _threshold_capacity(root, ZERO)
    target = _minimum(p0, _maximum(p1, _neg(h0)))
    value = _sub(_add(b0, _prefix_weight(root, target)),
                 _maximum(ZERO, _add(h0, target)))
    return QueryResult(base, a_bias_delta, a_updates, h_bias_delta, h_updates,
                       MappingProxyType(changes), root, b0, h0, target, value, _SEAL)


def _inorder(root):
    stack, node = [], root
    while stack or node is not None:
        while node is not None:
            stack.append(node)
            node = node.left
        node = stack.pop()
        yield node
        node = node.right


def decode(result, *, enabled=False):
    """Decode and check a complete box maximizer; this is NOT an HZ/ADV witness."""
    if not _on(enabled):
        return None
    if type(result) is not QueryResult or result._seal is not _SEAL:
        raise KernelError('an owned support result is required')
    base = _prepared(result.base)
    _count(base.bounds)
    amounts, remaining = {}, _f(result.target)
    for node in _inorder(result.root):
        amount = _minimum(remaining, node.capacity)
        amounts[node.ordinal] = amount
        remaining = _sub(remaining, amount)
    if remaining != ZERO:
        raise KernelError('support witness has unfilled capacity')
    point = []
    a_value = _add(base.a_bias, result.a_bias_delta)
    h_value = _add(base.h_bias, result.h_bias_delta)
    for ordinal, lower, upper in base.bounds:
        coordinate = result.changes.get(ordinal, base.coordinates[ordinal])
        value = coordinate.endpoint
        if coordinate.rho is not None:
            value = _add(value, _divide(amounts[ordinal], coordinate.h))
        if _compare(value, lower) < 0 or _compare(value, upper) > 0:
            raise KernelError('decoded support witness outside original box')
        a_value = _add(a_value, _mul(coordinate.a, value))
        h_value = _add(h_value, _mul(coordinate.h, value))
        point.append((ordinal, value))
    if _sub(a_value, _maximum(ZERO, h_value)) != result.value:
        raise KernelError('decoded support objective disagrees with certificate')
    return tuple(point)
