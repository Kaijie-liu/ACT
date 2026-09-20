"""Owned immutable data: only tuples, exact integer pairs and strings.

No caller-owned dict or Fraction object survives snapshot admission.
"""
from fractions import Fraction
from typing import NamedTuple


class ExactRational(NamedTuple):
    numerator: int
    denominator: int

    def __str__(self):
        return str(Fraction(self.numerator,self.denominator))


class Row(NamedTuple):
    entries: tuple

    def items(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)


class Source(NamedTuple):
    system: tuple
    scope: str
    binding: str
    units: int


class OwnedPlan(NamedTuple):
    binding: str
    initial: tuple
    steps: tuple
    checksum: str
    units: int


class Handle(NamedTuple):
    owner: object
    generation: int


class Admission(NamedTuple):
    source: Source
    plan: OwnedPlan
    handle: Handle
