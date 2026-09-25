"""Owned, recursively read-only parser data under the trusted Python runtime.

Not a sandbox against object.__setattr__, ctypes or hostile runtime reflection.
Normal assignment, container mutation and Fraction private-slot assignment are
blocked. Slices/concatenation deliberately return detached lists: the unchanged
projection checker edits a slice, not its source. Arithmetic stays Fraction.
"""
from fractions import Fraction
from types import MappingProxyType


class ReadOnlyFraction(Fraction):
    __slots__ = ()

    def __new__(cls, numerator=0, denominator=None):
        q = Fraction(numerator) if denominator is None else Fraction(numerator, denominator)
        result = object.__new__(cls)
        object.__setattr__(result, '_numerator', q.numerator)
        object.__setattr__(result, '_denominator', q.denominator)
        return result

    def __setattr__(self, name, value):
        raise TypeError('read-only exact coefficient')

    def __delattr__(self, name):
        raise TypeError('read-only exact coefficient')


class ReadOnlySequence(tuple):
    __slots__ = ()
    __hash__ = None  # preserve list, not tuple, comparison semantics

    def __getitem__(self, key):
        value = tuple.__getitem__(self, key)
        return list(value) if isinstance(key, slice) else value

    def __eq__(self, other):
        if not isinstance(other, (list, tuple)):
            return NotImplemented
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))

    def __ne__(self, other):
        result = self.__eq__(other)
        return NotImplemented if result is NotImplemented else not result

    def __add__(self, other):
        if not isinstance(other, (list, tuple)):
            return NotImplemented
        return list(self) + list(other)

    def __radd__(self, other):
        if not isinstance(other, (list, tuple)):
            return NotImplemented
        return list(other) + list(self)


def seal(value, tick):
    """Copy once from a private validated parse; never expose its mutable aliases."""
    tick()
    if type(value) is Fraction:
        return ReadOnlyFraction(value), 1
    if type(value) in (tuple, list):
        items = []; cells = 1
        for item in value:
            frozen, count = seal(item, tick); items.append(frozen); cells += count
        return (tuple(items) if type(value) is tuple else ReadOnlySequence(items)), cells
    if type(value) is dict:
        items = {}; cells = 1
        for key, item in value.items():
            frozen_key, nk = seal(key, tick); frozen_item, nv = seal(item, tick)
            items[frozen_key] = frozen_item; cells += nk + nv
        return MappingProxyType(items), cells
    if type(value) in (str, int):
        return value, 1
    raise ValueError('unexpected reference parser result')


def borrow(value, tick):
    """Return owned immutable data, not a materialized copy or a proof fact."""
    tick()
    return value
