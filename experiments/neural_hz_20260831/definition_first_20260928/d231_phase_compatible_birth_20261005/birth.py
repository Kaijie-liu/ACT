"""Opt-in exact next-ReLU transform of a complete D228 true-graph bank.

The supported entry is ``from_bank`` applied to the unmodified D228 Bank.
All vertices of every old phase cell, not merely convex-hull extreme points,
are retained.  A birth adds zero crossings of old-sign-compatible chords;
there is no phase/input subdivision, adjacency solver or external oracle.

The source domain remains the original declared local bank domain.  Keeping
an arbitrary correlated H alongside these rows is sound but does not make
the local hull ideal for H.  A support maximizer is not an input ADV.  IDs
and frame are caller identity tokens, not authenticated native/ONNX bindings.
Every original physical coordinate and signed phase column is retained.

Exact old preactivation values are certified at entry.  On a new chord they
are affinely interpolated with exactly the same parameter as every physical
coordinate.  Thus their signs are those of the old affine forms, without a
dense old-form evaluation at every new point.  Each new zero equation is
also checked directly against the complete physical point.

BirthBank and Compiled are trusted in-memory results, not arbitrary vertex
loaders or authenticated serialized certificates.  Direct construction,
private-state mutation, bypassing frozen records and untrusted deserialization
are unsupported.  Repeated births can grow quadratically per step; there is
no arbitrary-depth cost or full-network qualification.  Shared D228 Budget
counters include this implementation's logical arithmetic/construction work,
not complete Python/physical/device storage or bit-operation performance.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from functools import cmp_to_key

from experiments.neural_hz_20260831.definition_first_20260928.d228_joint_bank_component_20261005 import bank as _base


Rejected = _base.Rejected
Budget = _base.Budget
Limits = _base.Limits
Support = _base.Support
Row = _base.Row
_ZERO = Fraction(0)
_ONE = Fraction(1)
_TWO = Fraction(2)
_NEG_ONE = Fraction(-1)


def _compare(left, right, meter):
    if len(left) != len(right):
        raise Rejected("inconsistent physical dimensions")
    for a, b in zip(left, right):
        result = _base._cmp(a, b, meter)
        if result:
            return result
    return 0


def _affine(coefficients, bias, point, meter):
    return _base._add(_base._dot(coefficients, point, meter), bias, meter)


def _check_ids(physical_ids, phase_ids, meter):
    ids = physical_ids + phase_ids
    meter.charge(work=len(ids), entries=len(ids))
    for index, value in enumerate(ids):
        if type(value) is not str or not value.strip():
            raise Rejected("physical and phase IDs must be nonempty strings")
        meter.charge(work=len(value) + index, entries=index + 1)
        if value in ids[:index]:
            raise Rejected("all original physical and phase IDs must be distinct")


def _row(terms, rhs, n_columns, meter):
    # The shared canonicalizer accounts for rational arithmetic and row
    # storage.  Charge its integer-key sorting conservatively as well.
    terms = tuple(terms)
    n = len(terms)
    meter.charge(work=n * (n.bit_length() + 1), entries=n)
    return _base._row(terms, rhs, n_columns, meter)


@dataclass(frozen=True)
class Compiled:
    """Trusted rows with layout physical, original signed bits, lambda."""

    eq: tuple
    le: tuple
    n_columns: int
    physical_ids: tuple
    phase_ids: tuple
    n_lambda: int
    budget: Budget = field(repr=False, compare=False)

    def satisfied(self, physical, phases, lambdas):
        """Exact LP-row check; fractional signed phases are permitted here."""
        meter = self.budget._branch()
        p = _base._vector(physical, len(self.physical_ids), "physical", meter)
        b = _base._vector(phases, len(self.phase_ids), "phases", meter)
        weights = _base._vector(lambdas, self.n_lambda, "lambdas", meter)
        meter.charge(work=self.n_columns, entries=2 * self.n_columns)
        values = p + b + weights
        if len(values) != self.n_columns:
            raise Rejected("inconsistent compiled column layout")
        for rows, equality in ((self.eq, True), (self.le, False)):
            for row in rows:
                meter.charge(work=1)
                result = _ZERO
                for index, coefficient in row.terms:
                    if not 0 <= index < len(values):
                        raise Rejected("compiled row column is out of range")
                    result = _base._add(
                        result, _base._mul(coefficient, values[index], meter), meter)
                comparison = _base._cmp(result, row.rhs, meter)
                if (equality and comparison != 0) or (not equality and comparison > 0):
                    return False
        return True


@dataclass(frozen=True)
class BirthBank:
    """Trusted complete-cell-vertex bank returned by from_bank/advance.

    ``preactivations`` contains (dense physical coefficients, bias) pairs in
    phase-ID order; ``preactivation_values`` is aligned with ``vertices``.
    ``output_columns`` names the corresponding original ReLU readouts.
    ``candidate_count`` includes old points and accepted strict compatible
    crossings before deduplication in the last operation, not rejected pairs.
    """

    frame: object = field(repr=False, compare=False)
    physical_ids: tuple
    phase_ids: tuple
    vertices: tuple
    signs: tuple
    preactivations: tuple
    preactivation_values: tuple
    output_columns: tuple
    budget: Budget = field(repr=False, compare=False)
    candidate_count: int
    previous_vertex_count: int

    @property
    def physical_dim(self):
        return len(self.physical_ids)

    @property
    def phase_count(self):
        return len(self.phase_ids)

    @property
    def vertex_limit(self):
        n = self.previous_vertex_count
        return n + n * n // 4

    def _frame(self, frame):
        if frame is not self.frame:
            raise Rejected("frame identity mismatch")

    def support(self, physical_coefficients, phase_coefficients=None, *, frame):
        """Exact local labelled support, never an original-input witness."""
        self._frame(frame)
        meter = self.budget._branch()
        physical = _base._vector(physical_coefficients, self.physical_dim,
                                 "physical_coefficients", meter)
        if phase_coefficients is None:
            meter.charge(work=self.phase_count, entries=self.phase_count)
            phase_coefficients = (_ZERO,) * self.phase_count
        phases = _base._vector(phase_coefficients, self.phase_count,
                               "phase_coefficients", meter)
        best_value = None
        best_index = None
        best_phases = None
        for index, (vertex, signs) in enumerate(zip(self.vertices, self.signs)):
            meter.charge(work=1, entries=self.phase_count)
            value = _base._dot(physical, vertex, meter)
            assignment = []
            for coefficient, sign in zip(phases, signs):
                chosen = sign if sign else (
                    1 if _base._cmp(coefficient, _ZERO, meter) > 0 else -1)
                assignment.append(chosen)
                value = _base._add(value, _base._mul(
                    coefficient, _base._number(chosen, meter), meter), meter)
            if best_value is None or _base._cmp(value, best_value, meter) > 0:
                meter.charge(work=1, entries=self.phase_count)
                best_value, best_index, best_phases = value, index, tuple(assignment)
        if best_index is None:
            raise Rejected("empty complete graph bank")
        meter.charge(work=1, entries=4)
        return Support(best_value, best_index, best_phases, self.vertices[best_index])

    def compile(self, *, frame):
        """Compile every original bit; even empty sign masks get two rows."""
        self._frame(frame)
        meter = self.budget._branch()
        d, b, n = self.physical_dim, self.phase_count, len(self.vertices)
        offset = d + b
        n_columns = offset + n
        eq = [_row(((offset + j, _ONE) for j in range(n)),
                   _ONE, n_columns, meter)]
        for coordinate in range(d):
            meter.charge(work=1, entries=2)
            terms = [(coordinate, _ONE)]
            for j, vertex in enumerate(self.vertices):
                meter.charge(work=1)
                if vertex[coordinate].numerator:
                    meter.charge(entries=2)
                    terms.append((offset + j, _base._neg(vertex[coordinate], meter)))
            eq.append(_row(terms, _ZERO, n_columns, meter))
        le = []
        for j in range(n):
            meter.charge(work=1, entries=3)
            le.append(_row(((offset + j, _NEG_ONE),),
                           _ZERO, n_columns, meter))
        for phase in range(b):
            meter.charge(work=2, entries=4)
            lower = [(d + phase, _NEG_ONE)]
            upper = [(d + phase, _ONE)]
            for j, signs in enumerate(self.signs):
                meter.charge(work=1)
                if signs[phase] > 0:
                    meter.charge(entries=2)
                    lower.append((offset + j, _TWO))
                elif signs[phase] < 0:
                    meter.charge(entries=2)
                    upper.append((offset + j, _TWO))
            le.append(_row(lower, _ONE, n_columns, meter))
            le.append(_row(upper, _ONE, n_columns, meter))
        meter.charge(work=len(eq) + len(le), entries=2 * (len(eq) + len(le)) + 7)
        return Compiled(tuple(eq), tuple(le), n_columns, self.physical_ids,
                        self.phase_ids, n, self.budget)


def from_bank(bank, *, enabled=False, frame):
    """Enter only from a trusted D228 build result; defaults off."""
    if enabled is not True:
        raise Rejected("the phase-compatible birth primitive defaults off")
    if type(bank) is not _base.Bank:
        raise Rejected("from_bank requires the original D228 Bank type")
    bank._frame(frame)
    meter = bank.budget._branch()
    d = bank.physical_dim
    physical_ids = bank.binding.physical_ids
    phase_ids = bank.binding.phase_ids
    _check_ids(physical_ids, phase_ids, meter)
    forms = []
    for coordinate in (0, 1):
        meter.charge(work=d, entries=2 * d + 2)
        coefficients = [_ZERO] * d
        coefficients[coordinate] = _ONE
        forms.append((tuple(coefficients), _ZERO))
    for coefficients, bias in zip(bank.spec.coefficients, bank.spec.biases):
        coefficients = _base._vector(coefficients, d - 2, "old affine form", meter)
        bias = _base._number(bias, meter)
        meter.charge(work=d, entries=d + 2)
        forms.append((coefficients + (_ZERO, _ZERO), bias))
    output_columns = (2, 3, d - 2, d - 1)
    values = []
    if not bank.vertices or len(bank.vertices) != len(bank.signs):
        raise Rejected("invalid trusted bank dimensions")
    for vertex, signs in zip(bank.vertices, bank.signs):
        if len(signs) != 4:
            raise Rejected("invalid original phase dimension")
        point = _base._vector(vertex, d, "old graph point", meter)
        row = []
        for phase, (coefficients, bias) in enumerate(forms):
            value = _affine(coefficients, bias, point, meter)
            sign = _base._cmp(value, _ZERO, meter)
            if sign != signs[phase]:
                raise Rejected("old preactivation sign certification failed")
            output = value if sign > 0 else _ZERO
            if _base._cmp(output, point[output_columns[phase]], meter):
                raise Rejected("old ReLU graph certification failed")
            row.append(value)
        meter.charge(work=4, entries=9)
        values.append(tuple(row))
    meter.charge(work=len(values), entries=len(values) + 16)
    return BirthBank(frame, physical_ids, phase_ids, bank.vertices, bank.signs,
                     tuple(forms), tuple(values), output_columns, bank.budget,
                     len(bank.vertices), len(bank.vertices))


def advance(bank, coefficients, bias, physical_id, phase_id, *, enabled=False, frame):
    """Append one actual ReLU and its original bit, with no phase subproblems.

    All physical coordinates remain live.  Every accepted chord is in one
    old phase cell.  Points are sorted/deduplicated exactly, never pruned by
    convex extremality.  A failed shared resource charge returns no result.
    """
    if enabled is not True:
        raise Rejected("the phase-compatible birth primitive defaults off")
    if type(bank) is not BirthBank:
        raise Rejected("advance requires a trusted BirthBank")
    bank._frame(frame)
    meter = bank.budget._branch()
    d, b, n = bank.physical_dim, bank.phase_count, len(bank.vertices)
    if type(coefficients) is not tuple:
        raise Rejected("coefficients must be an exact tuple")
    coefficients = _base._vector(coefficients, d, "coefficients", meter)
    bias = _base._number(bias, meter)
    meter.charge(work=d + b + 2, entries=d + b + 2)
    physical_ids = bank.physical_ids + (physical_id,)
    phase_ids = bank.phase_ids + (phase_id,)
    _check_ids(physical_ids, phase_ids, meter)
    forms = []
    for old_coefficients, old_bias in bank.preactivations:
        meter.charge(work=d + 1, entries=d + 3)
        forms.append((old_coefficients + (_ZERO,), old_bias))
    meter.charge(work=d + 1, entries=d + 3)
    forms.append((coefficients + (_ZERO,), bias))
    candidates = []
    heights = []
    positive, negative = [], []
    for index, (vertex, old_signs, old_values) in enumerate(zip(
            bank.vertices, bank.signs, bank.preactivation_values)):
        height = _affine(coefficients, bias, vertex, meter)
        sign = _base._cmp(height, _ZERO, meter)
        meter.charge(work=d + 2 * b + 4, entries=d + 2 * b + 9)
        lifted = vertex + (height if sign > 0 else _ZERO,)
        candidates.append((lifted, old_signs + (sign,), old_values + (height,)))
        heights.append(height)
        if sign > 0:
            positive.append(index)
        elif sign < 0:
            negative.append(index)
    if len(heights) != n or not n:
        raise Rejected("invalid trusted complete-bank dimensions")

    for left in positive:
        for right in negative:
            meter.charge(work=1)
            compatible = True
            for a, c in zip(bank.signs[left], bank.signs[right]):
                meter.charge(work=1)
                if a and c and a != c:
                    compatible = False
                    break
            if not compatible:
                continue
            t = _base._div(_base._neg(heights[left], meter),
                          _base._sub(heights[right], heights[left], meter), meter)
            if _base._cmp(t, _ZERO, meter) <= 0 or _base._cmp(t, _ONE, meter) >= 0:
                raise Rejected("strict crossing has an invalid interpolation parameter")
            complement = _base._sub(_ONE, t, meter)
            point = []
            for a, c in zip(bank.vertices[left], bank.vertices[right]):
                point.append(_base._add(_base._mul(complement, a, meter),
                                        _base._mul(t, c, meter), meter))
            meter.charge(work=d, entries=2 * d)
            point = tuple(point)
            if _base._cmp(_affine(coefficients, bias, point, meter), _ZERO, meter):
                raise Rejected("new crossing failed its exact zero equation")
            values, signs = [], []
            for phase, (a, c) in enumerate(zip(bank.preactivation_values[left],
                                                bank.preactivation_values[right])):
                value = _base._add(_base._mul(complement, a, meter),
                                   _base._mul(t, c, meter), meter)
                sign = _base._cmp(value, _ZERO, meter)
                expected = bank.signs[left][phase] or bank.signs[right][phase]
                if sign != expected:
                    raise Rejected("compatible chord lost an old exact phase")
                expected_output = value if sign > 0 else _ZERO
                if _base._cmp(point[bank.output_columns[phase]], expected_output, meter):
                    raise Rejected("compatible chord left the old true graph")
                values.append(value)
                signs.append(sign)
            meter.charge(work=d + 2 * b + 3, entries=d + 4 * b + 8)
            candidates.append((point + (_ZERO,), tuple(signs) + (0,),
                               tuple(values) + (_ZERO,)))

    candidate_count = len(candidates)
    if candidate_count > n + n * n // 4:
        raise Rejected("single-birth candidate-count invariant failed")
    # Sorting, rather than Fraction hashing, keeps all rational comparison
    # intermediates under the same explicit bit/work checks.
    meter.charge(work=candidate_count, entries=3 * candidate_count)
    candidates.sort(key=cmp_to_key(lambda a, c: _compare(a[0], c[0], meter)))
    unique = []
    for candidate in candidates:
        meter.charge(work=1, entries=1)
        if unique and _compare(unique[-1][0], candidate[0], meter) == 0:
            meter.charge(work=2 * (b + 1))
            if unique[-1][1] != candidate[1]:
                raise Rejected("duplicate graph point has inconsistent phase signs")
            if _compare(unique[-1][2], candidate[2], meter):
                raise Rejected("duplicate graph point has inconsistent affine values")
        else:
            unique.append(candidate)
    meter.charge(work=3 * len(unique), entries=3 * len(unique) + b + 17)
    vertices = tuple(candidate[0] for candidate in unique)
    signs = tuple(candidate[1] for candidate in unique)
    values = tuple(candidate[2] for candidate in unique)
    return BirthBank(frame, physical_ids, phase_ids, vertices, signs, tuple(forms),
                     values, bank.output_columns + (d,), bank.budget,
                     candidate_count, n)
