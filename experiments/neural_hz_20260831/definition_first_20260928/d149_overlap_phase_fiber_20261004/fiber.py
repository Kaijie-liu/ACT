"""Default-off small rational reference for shared overlapping phase fibers.

This is a lossy domain, not an exact ReLU graph plus a helper.  Fresh gate
errors have one identity each; several norms may constrain the same errors.
Queries return two valid phase-affine certificates, not an affine encoding of
their minimum.  No model importer, solver, GPU or physical qualification is
provided.  The explicit reference caps below are not real-network admission.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt


MAX_BITS = 512
MAX_SOURCES = 32
MAX_PHASES = 64
MAX_ERRORS = 128
MAX_OUTPUTS = 64
MAX_BANKS = 16
MAX_GROUPS = 128
MAX_PREDICATES = 512
MAX_MATRIX_ENTRIES = 8192
MAX_WORK = 20_000_000
SQRT_BITS = 32
_AUTH = object()


class DisabledError(ValueError):
    pass


def rational(value):
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise ValueError("only explicit integers and Fractions are supported")
    answer = Fraction(value)
    if max(answer.numerator.bit_length(), answer.denominator.bit_length()) > MAX_BITS:
        raise ValueError("512-bit rational cap exceeded")
    return answer


def _qs(values):
    return tuple(rational(value) for value in values)


def _name(value):
    if not isinstance(value, str) or not value:
        raise ValueError("a nonempty identity name is required")
    return value


class Work:
    """One conservative algebra-work meter shared by an entire frame lineage."""

    def __init__(self):
        self.used = 0

    def charge(self, amount):
        if type(amount) is not int or amount < 0 or amount > MAX_WORK - self.used:
            raise ValueError("reference algebra-work cap exceeded")
        self.used += amount


def _dot(left, right, work):
    if len(left) != len(right):
        raise ValueError("dot-product shape mismatch")
    work.charge(1 + 4 * len(left))
    result = Fraction(0)
    for a, b in zip(left, right):
        result = rational(result + rational(a * b))
    return result


def sqrt_upper(value, work=None):
    """A checked outward rational square root; never a floating-point sqrt."""
    value = rational(value)
    if value < 0:
        raise ValueError("negative squared norm")
    if work is not None:
        work.charge(32 + 4 * (value.numerator.bit_length() + value.denominator.bit_length()) ** 2)
    p, q = isqrt(value.numerator), isqrt(value.denominator)
    if p * p == value.numerator and q * q == value.denominator:
        result = rational(Fraction(p, q))
    else:
        scale = 1 << SQRT_BITS
        scaled = rational(value * rational(scale * scale))
        ceiling = (scaled.numerator + scaled.denominator - 1) // scaled.denominator
        rational(ceiling)
        upper = isqrt(ceiling)
        if upper * upper < ceiling:
            upper += 1
        result = rational(Fraction(upper, scale))
    if result < 0 or rational(result * result) < value:
        raise ValueError("outward square-root certificate failed")
    return result


@dataclass(frozen=True, eq=False)
class Identity:
    name: str

    def __post_init__(self):
        _name(self.name)


@dataclass(frozen=True)
class PhaseAffine:
    constant: Fraction
    phase: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "constant", rational(self.constant))
        object.__setattr__(self, "phase", _qs(self.phase))
        if len(self.phase) > MAX_PHASES:
            raise ValueError("phase-affine reference cap exceeded")

    def pad(self, count):
        if count < len(self.phase):
            raise ValueError("cannot discard original phase coefficients")
        return PhaseAffine(self.constant, self.phase + (Fraction(0),) * (count - len(self.phase)))

    def plus(self, other, work):
        count = max(len(self.phase), len(other.phase))
        left, right = self.pad(count), other.pad(count)
        work.charge(4 + 3 * count)
        return PhaseAffine(rational(left.constant + right.constant),
                           tuple(rational(a + b) for a, b in zip(left.phase, right.phase)))

    def scale(self, value, work):
        value = rational(value)
        work.charge(4 + 3 * len(self.phase))
        return PhaseAffine(rational(value * self.constant),
                           tuple(rational(value * a) for a in self.phase))

    def at(self, phase01, work):
        if len(phase01) < len(self.phase):
            raise ValueError("phase assignment too short")
        return rational(self.constant + _dot(self.phase, phase01[:len(self.phase)], work))

    def extrema(self, work):
        work.charge(2 + 4 * len(self.phase))
        lower = upper = self.constant
        for value in self.phase:
            lower = rational(lower + min(Fraction(0), value))
            upper = rational(upper + max(Fraction(0), value))
        return lower, upper


@dataclass(frozen=True)
class Form:
    constant: Fraction
    source: tuple
    phase: tuple
    error: tuple

    def __post_init__(self):
        object.__setattr__(self, "constant", rational(self.constant))
        for attr in ("source", "phase", "error"):
            object.__setattr__(self, attr, _qs(getattr(self, attr)))
        if (len(self.source) > MAX_SOURCES or len(self.phase) > MAX_PHASES
                or len(self.error) > MAX_ERRORS):
            raise ValueError("reference factor cap exceeded")

    def pad(self, phases, errors):
        if phases < len(self.phase) or errors < len(self.error):
            raise ValueError("cannot discard original factors")
        return Form(self.constant, self.source,
                    self.phase + (Fraction(0),) * (phases - len(self.phase)),
                    self.error + (Fraction(0),) * (errors - len(self.error)))

    def scale(self, value, work):
        value = rational(value)
        work.charge(4 + 3 * (len(self.source) + len(self.phase) + len(self.error)))
        return Form(rational(value * self.constant),
                    tuple(rational(value * a) for a in self.source),
                    tuple(rational(value * a) for a in self.phase),
                    tuple(rational(value * a) for a in self.error))

    def plus(self, other, work):
        if (len(self.source), len(self.phase), len(self.error)) != (
                len(other.source), len(other.phase), len(other.error)):
            raise ValueError("unaligned coefficient maps")
        work.charge(4 + 3 * (len(self.source) + len(self.phase) + len(self.error)))
        return Form(rational(self.constant + other.constant),
                    tuple(rational(a + b) for a, b in zip(self.source, other.source)),
                    tuple(rational(a + b) for a, b in zip(self.phase, other.phase)),
                    tuple(rational(a + b) for a, b in zip(self.error, other.error)))

    def at(self, source, phase01, error, work):
        if (len(source) != len(self.source) or len(phase01) < len(self.phase)
                or len(error) < len(self.error)):
            raise ValueError("assignment shape mismatch")
        value = rational(self.constant + _dot(self.source, source, work))
        value = rational(value + _dot(self.phase, phase01[:len(self.phase)], work))
        return rational(value + _dot(self.error, error[:len(self.error)], work))


@dataclass(frozen=True)
class Predicate:
    form: Form
    sense: str
    rhs: Fraction

    def __post_init__(self):
        if not isinstance(self.form, Form) or self.sense not in ("eq", "le"):
            raise ValueError("only linear EQ/LE predicates are supported")
        object.__setattr__(self, "rhs", rational(self.rhs))


@dataclass(frozen=True)
class SourceConstraint:
    source: tuple
    phase: tuple
    sense: str
    rhs: Fraction

    def predicate(self):
        return Predicate(Form(0, self.source, self.phase, ()), self.sense, self.rhs)


@dataclass(frozen=True)
class NormGroup:
    indices: tuple
    radii: tuple


@dataclass(frozen=True)
class Bank:
    name: str
    coordinates: tuple
    whole: NormGroup
    local: tuple


@dataclass(frozen=True, eq=False)
class Frame:
    bounds: tuple
    sources: tuple
    reference: tuple
    decoder_matrix: tuple
    decoder_bias: tuple
    work: Work


@dataclass(frozen=True)
class Readout:
    frame: Frame
    owner: Identity
    phases: tuple
    errors: tuple
    forms: tuple


def _matrix(values, rows, columns):
    result = tuple(_qs(row) for row in values)
    if (len(result) != rows or any(len(row) != columns for row in result)
            or rows * columns > MAX_MATRIX_ENTRIES):
        raise ValueError("reference matrix shape or entry cap exceeded")
    return result


def _gram(matrix, work):
    rows, columns = len(matrix), len(matrix[0]) if matrix else 0
    work.charge(16 + 4 * rows * columns * columns + columns * columns)
    result = []
    for j in range(columns):
        row = []
        for k in range(columns):
            value = Fraction(0)
            for i in range(rows):
                value = rational(value + rational(matrix[i][j] * matrix[i][k]))
            row.append(value)
        result.append(tuple(row))
    return tuple(result)


def matrix_norm_upper(matrix, work):
    """sqrt(max absolute Gram row sum) bounds the Euclidean operator norm."""
    gram = _gram(matrix, work)
    squared = Fraction(0)
    for row in gram:
        total = Fraction(0)
        for value in row:
            total = rational(total + abs(value))
        squared = max(squared, total)
    return sqrt_upper(squared, work)


def source_norm_upper(matrix, bounds, work):
    """Gram absolute sum for M*(source-midpoint), with true box half-widths."""
    work.charge(16 + 4 * len(matrix) * len(bounds))
    magnitudes = tuple(rational(rational(upper - lower) / 2) for lower, upper in bounds)
    scaled = tuple(tuple(rational(value * magnitude) for value, magnitude in zip(row, magnitudes))
                   for row in matrix)
    gram = _gram(scaled, work)
    squared = Fraction(0)
    for row in gram:
        for value in row:
            squared = rational(squared + abs(value))
    return sqrt_upper(squared, work)


class Fiber:
    """Owned immutable relation metadata, with an explicitly mutable work meter.

    Construction is through box/embed_hz and the transformers; raw relation
    construction is unsupported.  There is no arbitrary intersection/join API.
    """

    __slots__ = ("frame", "phases", "references", "errors", "banks", "predicates", "lineage", "_sealed")

    def __init__(self, frame, phases, errors, banks, predicates, lineage, *, references=(), _auth=None):
        if _auth is not _AUTH:
            raise ValueError("use the authenticated domain constructors")
        self.frame, self.phases, self.errors = frame, tuple(phases), tuple(errors)
        self.references = tuple(references)
        self.banks, self.predicates, self.lineage = tuple(banks), tuple(predicates), tuple(lineage)
        self._validate()
        self._sealed = True

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("relation metadata is immutable")
        object.__setattr__(self, name, value)

    @property
    def work(self):
        return self.frame.work

    def _validate(self):
        n = len(self.frame.bounds)
        if (not 1 <= n <= MAX_SOURCES or len(self.phases) > MAX_PHASES
                or len(self.errors) > MAX_ERRORS or len(self.banks) > MAX_BANKS
                or len(self.predicates) > MAX_PREDICATES or not self.lineage):
            raise ValueError("reference relation cap exceeded")
        if (len(self.references) != len(self.phases)
                or any(type(value) is not int or value not in (0, 1) for value in self.references)):
            raise ValueError("one fixed original-phase reference is required per bit")
        identities = self.frame.sources + self.phases + self.errors
        if (any(not isinstance(item, Identity) for item in identities)
                or len(set(identities)) != len(identities)
                or len({item.name for item in identities}) != len(identities)):
            raise ValueError("factor identities must be distinct and typed")
        if any(len(pair) != 2 or pair[0] > pair[1] for pair in self.frame.bounds):
            raise ValueError("invalid source box")
        if (len(self.frame.sources) != n or not 1 <= len(self.frame.decoder_matrix) <= MAX_OUTPUTS
                or len(self.frame.decoder_bias) != len(self.frame.decoder_matrix)
                or any(len(row) != n for row in self.frame.decoder_matrix)):
            raise ValueError("invalid source decoder")
        claimed = []
        groups = 0
        for bank in self.banks:
            claimed.extend(bank.coordinates)
            size = len(bank.coordinates)
            if bank.whole.indices != tuple(range(size)):
                raise ValueError("whole-bank norm is required")
            local_indices = [group.indices for group in bank.local]
            if len(set(local_indices)) != len(local_indices):
                raise ValueError("duplicate local groups")
            if bank.local and set().union(*(set(indices) for indices in local_indices)) != set(range(size)):
                raise ValueError("local groups must cover every bank coordinate")
            for group in (bank.whole,) + bank.local:
                groups += 1
                if (not group.indices or len(set(group.indices)) != len(group.indices)
                        or any(type(i) is not int or not 0 <= i < size for i in group.indices)
                        or not 1 <= len(group.radii) <= 2 or len(set(group.radii)) != len(group.radii)):
                    raise ValueError("invalid coordinate-norm constraint")
                for radius in group.radii:
                    if (not isinstance(radius, PhaseAffine) or len(radius.phase) > len(self.phases)
                            or radius.extrema(self.work)[0] < 0):
                        raise ValueError("radius must be nonnegative on the entire Boolean box")
        if groups > MAX_GROUPS or tuple(claimed) != tuple(range(len(self.errors))):
            raise ValueError("bank population does not partition the shared errors")
        for predicate in self.predicates:
            form = predicate.form
            if (len(form.source) != n or len(form.phase) > len(self.phases)
                    or len(form.error) > len(self.errors)):
                raise ValueError("predicate factor population mismatch")
        self.work.charge(16 + len(identities) + len(self.predicates) + groups)

    @classmethod
    def box(cls, bounds, *, phase_names=(), predicates=(), decoder_matrix=None,
            decoder_bias=None, enabled=False):
        if enabled is not True:
            raise DisabledError("explicit enabled=True is required")
        bounds = tuple(_qs(pair) for pair in bounds)
        n = len(bounds)
        if not 1 <= n <= MAX_SOURCES:
            raise ValueError("source reference cap exceeded")
        phases = tuple(Identity(name) for name in phase_names)
        sources = tuple(Identity("source:" + str(i)) for i in range(n))
        if decoder_matrix is None:
            decoder_matrix = tuple(tuple(int(i == j) for j in range(n)) for i in range(n))
        decoder_matrix = tuple(_qs(row) for row in decoder_matrix)
        decoder_bias = _qs((0,) * len(decoder_matrix) if decoder_bias is None else decoder_bias)
        if any(len(pair) != 2 or pair[0] > pair[1] for pair in bounds):
            raise ValueError("invalid source box")
        reference = tuple(rational(rational(lower + upper) / 2) for lower, upper in bounds)
        frame = Frame(bounds, sources, reference, decoder_matrix, decoder_bias, Work())
        rows = tuple(item.predicate() for item in predicates)
        return cls(frame, phases, (), (), rows, (Identity("root"),),
                   references=(0,) * len(phases), _auth=_AUTH)

    @classmethod
    def embed_hz(cls, bounds, center, continuous, signed, *, phase_names,
                 signed_predicates=(), decoder_matrix=None, decoder_bias=None, enabled=False):
        """Exact c+Gc*xi+Gb*sigma embedding; predicates use original sigma."""
        base = cls.box(bounds, phase_names=phase_names, decoder_matrix=decoder_matrix,
                       decoder_bias=decoder_bias, enabled=enabled)
        center = _qs(center)
        if not 1 <= len(center) <= MAX_OUTPUTS:
            raise ValueError("HZ output population exceeds reference scope")
        continuous = _matrix(continuous, len(center), len(bounds))
        signed = _matrix(signed, len(center), len(base.phases))
        forms = []
        for offset, source, binary in zip(center, continuous, signed):
            adjustment = _dot(binary, (Fraction(1),) * len(binary), base.work)
            forms.append(Form(rational(offset - adjustment), source,
                              tuple(rational(2 * value) for value in binary), ()))
        rows = []
        for source, binary, sense, rhs in signed_predicates:
            binary = _qs(binary)
            if len(binary) != len(base.phases):
                raise ValueError("signed predicate phase population mismatch")
            adjustment = _dot(binary, (Fraction(1),) * len(binary), base.work)
            rows.append(Predicate(Form(-adjustment, source,
                                  tuple(rational(2 * value) for value in binary), ()), sense, rhs))
        state = base._extend(predicates=tuple(rows)) if rows else base
        return state, state._readout(forms)

    def _extend(self, *, phases=None, references=None, errors=None, banks=None, predicates=None):
        return Fiber(self.frame, self.phases if phases is None else phases,
                     self.errors if errors is None else errors,
                     self.banks if banks is None else banks,
                     self.predicates if predicates is None else predicates,
                     self.lineage + (Identity("extension"),),
                     references=self.references if references is None else references, _auth=_AUTH)

    def _zero(self):
        return Form(0, (0,) * len(self.frame.bounds), (0,) * len(self.phases), (0,) * len(self.errors))

    def _constant(self, value):
        zero = self._zero()
        return Form(value, zero.source, zero.phase, zero.error)

    def _readout(self, forms):
        forms = tuple(forms)
        if not 1 <= len(forms) <= MAX_OUTPUTS:
            raise ValueError("readout population cap exceeded")
        expected = (len(self.frame.bounds), len(self.phases), len(self.errors))
        if any(not isinstance(form, Form) or (len(form.source), len(form.phase), len(form.error)) != expected
               for form in forms):
            raise ValueError("readout shape differs from its owner")
        self.work.charge(4 + len(forms) * (1 + sum(expected)))
        return Readout(self.frame, self.lineage[-1], self.phases, self.errors, forms)

    def _aligned(self, view):
        if (not isinstance(view, Readout) or view.frame is not self.frame
                or view.owner not in self.lineage
                or self.phases[:len(view.phases)] != view.phases
                or self.errors[:len(view.errors)] != view.errors):
            raise ValueError("unauthenticated parent or independent sibling readout")
        self.work.charge(4 + len(view.forms) * (1 + len(self.phases) + len(self.errors)))
        return tuple(form.pad(len(self.phases), len(self.errors)) for form in view.forms)

    def sources(self):
        n = len(self.frame.bounds)
        return self._readout(Form(0, tuple(int(i == j) for j in range(n)),
                                  (0,) * len(self.phases), (0,) * len(self.errors)) for i in range(n))

    def phases01(self):
        if not self.phases:
            raise ValueError("no original phases")
        return self._readout(Form(0, (0,) * len(self.frame.bounds),
                                  tuple(int(i == j) for j in range(len(self.phases))),
                                  (0,) * len(self.errors)) for i in range(len(self.phases)))

    def affine(self, view, matrix, bias):
        forms, bias = self._aligned(view), _qs(bias)
        if not 1 <= len(bias) <= MAX_OUTPUTS:
            raise ValueError("affine output cap exceeded")
        matrix = _matrix(matrix, len(bias), len(forms))
        outputs = []
        for row, offset in zip(matrix, bias):
            result = self._constant(offset)
            for value, form in zip(row, forms):
                if value:
                    result = result.plus(form.scale(value, self.work), self.work)
            outputs.append(result)
        return self._readout(outputs)

    def add(self, left, right):
        left, right = self._aligned(left), self._aligned(right)
        if len(left) != len(right):
            raise ValueError("Add requires matching output populations")
        return self._readout(a.plus(b, self.work) for a, b in zip(left, right))

    def concat(self, *views):
        return self._readout(form for view in views for form in self._aligned(view))

    def constrain(self, scalar, sense, rhs):
        forms = self._aligned(scalar)
        if len(forms) != 1:
            raise ValueError("a scalar linear predicate is required")
        return self._extend(predicates=self.predicates + (Predicate(forms[0], sense, rhs),))

    def _bank_bound(self, matrix, bank, mode):
        if mode not in ("whole", "cover"):
            raise ValueError("unknown fixed norm certificate")
        if mode == "whole" or not bank.local:
            radius = bank.whole.radii[0 if mode == "whole" else -1]
            return radius.scale(matrix_norm_upper(matrix, self.work), self.work)
        degrees = [0] * len(bank.coordinates)
        for group in bank.local:
            for index in group.indices:
                degrees[index] += 1
        if any(value == 0 for value in degrees):
            raise ValueError("incomplete declared cover")
        answer = PhaseAffine(0, (0,) * len(self.phases))
        for group in bank.local:
            self.work.charge(8 + 4 * len(matrix) * len(group.indices))
            piece = tuple(tuple(rational(row[index] / degrees[index]) for index in group.indices)
                          for row in matrix)
            contribution = group.radii[-1].scale(matrix_norm_upper(piece, self.work), self.work)
            answer = answer.plus(contribution, self.work)
        return answer

    def _vector_radius(self, forms, mode):
        source = tuple(form.source for form in forms)
        answer = PhaseAffine(source_norm_upper(source, self.frame.bounds, self.work),
                             (0,) * len(self.phases))
        coefficients, drift_constant = [], Fraction(0)
        for j in range(len(self.phases)):
            column = tuple(form.phase[j] for form in forms)
            magnitude = sqrt_upper(_dot(column, column, self.work), self.work)
            reference = self.references[j]
            drift_constant = rational(drift_constant + rational(magnitude * reference))
            coefficients.append(rational(magnitude * (1 - 2 * reference)))
        answer = answer.plus(PhaseAffine(drift_constant, tuple(coefficients)), self.work)
        for bank in self.banks:
            matrix = tuple(tuple(form.error[i] for i in bank.coordinates) for form in forms)
            answer = answer.plus(self._bank_bound(matrix, bank, mode), self.work)
        return answer

    def support_certificates(self, view, direction):
        """Two safe U(beta); predicates are retained but not optimized here."""
        forms, direction = self._aligned(view), _qs(direction)
        if len(forms) != len(direction):
            raise ValueError("support direction shape mismatch")
        form = self._zero()
        for coefficient, item in zip(direction, forms):
            if coefficient:
                form = form.plus(item.scale(coefficient, self.work), self.work)
        constant = form.constant
        for coefficient, (lower, upper) in zip(form.source, self.frame.bounds):
            constant = rational(constant + max(rational(coefficient * lower), rational(coefficient * upper)))
        base = PhaseAffine(constant, form.phase)
        results = []
        for mode in ("whole", "cover"):
            result = base
            for bank in self.banks:
                matrix = (tuple(form.error[i] for i in bank.coordinates),)
                result = result.plus(self._bank_bound(matrix, bank, mode), self.work)
            results.append(result)
        return tuple(results)

    def support(self, view, direction):
        return min(cert.extrema(self.work)[1] for cert in self.support_certificates(view, direction))

    def bounds(self, scalar):
        if len(self._aligned(scalar)) != 1:
            raise ValueError("a scalar readout is required")
        return -self.support(scalar, (-1,)), self.support(scalar, (1,))

    def error_bank(self, name, dimension, whole, *, local=()):
        """Extend by a freely constrained shared bank; no claimed source equality."""
        _name(name)
        if type(dimension) is not int or not 1 <= dimension <= MAX_ERRORS - len(self.errors):
            raise ValueError("new residual population cap exceeded")
        def radius(value):
            item = value if isinstance(value, PhaseAffine) else PhaseAffine(value)
            if len(item.phase) > len(self.phases):
                raise ValueError("radius refers to a nonexistent phase")
            return item.pad(len(self.phases))
        local = tuple(NormGroup(tuple(indices), (radius(value),)) for indices, value in local)
        offset = len(self.errors)
        identities = tuple(Identity(name + ":" + str(i)) for i in range(dimension))
        bank = Bank(name, tuple(range(offset, offset + dimension)),
                    NormGroup(tuple(range(dimension)), (radius(whole),)), local)
        state = self._extend(errors=self.errors + identities, banks=self.banks + (bank,))
        forms = []
        for index in bank.coordinates:
            zero = state._zero()
            errors = tuple(int(i == index) for i in range(len(state.errors)))
            forms.append(Form(0, zero.source, zero.phase, errors))
        return state, state._readout(forms)

    def relu(self, view, names, *, groups=(), bounds=None):
        forms, names = self._aligned(view), tuple(_name(name) for name in names)
        size = len(forms)
        if (len(names) != size or len(set(names)) != size or len(self.phases) + size > MAX_PHASES
                or len(self.errors) + size > MAX_ERRORS):
            raise ValueError("new original gate population mismatch")
        groups = tuple(tuple(group) for group in groups)
        for group in groups:
            if (not group or len(set(group)) != len(group)
                    or any(type(index) is not int or not 0 <= index < size for index in group)):
                raise ValueError("invalid local coordinate group")
        if (len(set(groups)) != len(groups)
                or groups and set().union(*(set(group) for group in groups)) != set(range(size))):
            raise ValueError("local cover must contain every new gate exactly by identity")
        if len(groups) + sum(1 + len(bank.local) for bank in self.banks) + 1 > MAX_GROUPS:
            raise ValueError("coordinate-norm group cap exceeded")
        certified = tuple(self.bounds(self._readout((form,))) for form in forms)
        if bounds is not None:
            bounds = tuple(_qs(pair) for pair in bounds)
            if (len(bounds) != size or any(len(pair) != 2 or pair[0] > lo or pair[1] < hi
                                          for pair, (lo, hi) in zip(bounds, certified))):
                raise ValueError("supplied preactivation bounds are uncertified")
            certified = bounds
        nphase, nerror = len(self.phases) + size, len(self.errors) + size
        nominal = []
        for form in forms:
            value = rational(form.constant + _dot(form.source, self.frame.reference, self.work))
            nominal.append(rational(value + _dot(form.phase, self.references, self.work)))
        nominal = tuple(nominal)
        norm_groups = []
        for indices in (tuple(range(size)),) + groups:
            selected = tuple(forms[index] for index in indices)
            radii = []
            for mode in ("whole", "cover"):
                item = self._vector_radius(selected, mode).scale(Fraction(1, 2), self.work).pad(nphase)
                if item not in radii:
                    radii.append(item)
            norm_groups.append(NormGroup(indices, tuple(radii)))
        offset = len(self.errors)
        bank = Bank("relu:" + ",".join(names), tuple(range(offset, nerror)),
                    norm_groups[0], tuple(norm_groups[1:]))
        phases = self.phases + tuple(Identity(name) for name in names)
        errors = self.errors + tuple(Identity("error:" + name) for name in names)
        references = self.references + tuple(int(value > 0) for value in nominal)
        state = self._extend(phases=phases, references=references,
                             errors=errors, banks=self.banks + (bank,))
        outputs, rows = [], list(self.predicates)
        for i, (old, bar, (lo, hi)) in enumerate(zip(forms, nominal, certified)):
            g = old.pad(nphase, nerror)
            alpha_coefficients = tuple(int(j == len(self.phases) + i) for j in range(nphase))
            alpha = Form(0, (0,) * len(self.frame.bounds), alpha_coefficients, (0,) * nerror)
            e = Form(0, (0,) * len(self.frame.bounds), (0,) * nphase,
                     tuple(int(j == offset + i) for j in range(nerror)))
            q = g.scale(Fraction(1, 2), self.work).plus(state._constant(rational(-bar / 2)), self.work)
            q = q.plus(alpha.scale(bar, self.work), self.work).plus(e, self.work)
            lower, upper = min(Fraction(0), lo), max(Fraction(0), hi)
            rows.extend((
                Predicate(g.plus(alpha.scale(-upper, self.work), self.work), "le", 0),
                Predicate(g.scale(-1, self.work).plus(alpha.scale(-lower, self.work), self.work), "le", -lower),
                Predicate(q.scale(-1, self.work), "le", 0),
                Predicate(g.plus(q.scale(-1, self.work), self.work), "le", 0),
                Predicate(q.plus(alpha.scale(-upper, self.work), self.work), "le", 0),
            ))
            outputs.append(q)
        final = state._extend(predicates=tuple(rows))
        return final, final._readout(outputs)

    def interval_affine(self, view, lower_matrix, upper_matrix, lower_bias, upper_bias, *, name):
        forms = self._aligned(view)
        lower_bias, upper_bias = _qs(lower_bias), _qs(upper_bias)
        rows = len(lower_bias)
        if not 1 <= rows <= MAX_OUTPUTS or len(upper_bias) != rows:
            raise ValueError("interval affine output population mismatch")
        lower_matrix = _matrix(lower_matrix, rows, len(forms))
        upper_matrix = _matrix(upper_matrix, rows, len(forms))
        if (any(a > b for a, b in zip(lower_bias, upper_bias))
                or any(a > b for lo, hi in zip(lower_matrix, upper_matrix) for a, b in zip(lo, hi))):
            raise ValueError("unordered coefficient enclosure")
        self.work.charge(16 + 12 * rows * (1 + len(forms)))
        midpoint = tuple(tuple(rational(rational(a + b) / 2) for a, b in zip(lo, hi))
                         for lo, hi in zip(lower_matrix, upper_matrix))
        bias = tuple(rational(rational(a + b) / 2) for a, b in zip(lower_bias, upper_bias))
        nominal = self.affine(view, midpoint, bias)
        exact = lower_matrix == upper_matrix and lower_bias == upper_bias
        if exact:
            return self, nominal
        magnitudes = tuple(max(abs(lo), abs(hi)) for lo, hi in
                           (self.bounds(self._readout((form,))) for form in forms))
        errors = []
        for lo, hi, blo, bhi in zip(lower_matrix, upper_matrix, lower_bias, upper_bias):
            value = rational(rational(bhi - blo) / 2)
            for left, right, magnitude in zip(lo, hi, magnitudes):
                delta = rational(rational(right - left) / 2)
                value = rational(value + rational(delta * magnitude))
            errors.append(value)
        whole = sqrt_upper(_dot(errors, errors, self.work), self.work)
        state, noise = self.error_bank(name, rows, whole,
                                       local=tuple(((i,), value) for i, value in enumerate(errors)))
        return state, state.add(nominal, noise)

    def contains(self, source, signed_phases, errors):
        """Exact native membership; fractional phases never count as native."""
        try:
            source, signed_phases, errors = _qs(source), _qs(signed_phases), _qs(errors)
            if (len(source) != len(self.frame.bounds) or len(signed_phases) != len(self.phases)
                    or len(errors) != len(self.errors) or any(value not in (-1, 1) for value in signed_phases)):
                return False
            phase01 = tuple(rational((value + 1) / 2) for value in signed_phases)
            if any(not lower <= value <= upper for value, (lower, upper) in zip(source, self.frame.bounds)):
                return False
            for bank in self.banks:
                for group in (bank.whole,) + bank.local:
                    values = tuple(errors[bank.coordinates[index]] for index in group.indices)
                    squared = _dot(values, values, self.work)
                    for radius in group.radii:
                        bound = radius.at(phase01, self.work)
                        if bound < 0 or squared > rational(bound * bound):
                            return False
            for row in self.predicates:
                value = row.form.at(source, phase01, errors, self.work)
                if value != row.rhs if row.sense == "eq" else value > row.rhs:
                    return False
            return True
        except (ValueError, TypeError, OverflowError):
            return False

    def evaluate(self, view, source, signed_phases, errors):
        forms = self._aligned(view)
        if not self.contains(source, signed_phases, errors):
            raise ValueError("assignment is not a native member")
        source, signed_phases, errors = _qs(source), _qs(signed_phases), _qs(errors)
        phase01 = tuple(rational((value + 1) / 2) for value in signed_phases)
        return tuple(form.at(source, phase01, errors, self.work) for form in forms)

    def decode(self, source, signed_phases, errors):
        if not self.contains(source, signed_phases, errors):
            raise ValueError("decoder requires a native member, not a relaxed point")
        source = _qs(source)
        return tuple(rational(offset + _dot(row, source, self.work))
                     for row, offset in zip(self.frame.decoder_matrix, self.frame.decoder_bias))

    def cost(self, *live_views):
        """Full logical entries for this root and DECLARED live views, not RSS.

        Old states held separately by a caller are not magically released or
        included here.  Norm incidences count every stored radius separately.
        Dense map slots (including zeros), predicates and decoder are charged.
        """
        forms = tuple(form for view in live_views for form in self._aligned(view))
        groups = tuple(group for bank in self.banks for group in (bank.whole,) + bank.local)
        norm_constraints = sum(len(group.radii) for group in groups)
        incidence = sum(len(group.indices) * len(group.radii) for group in groups)
        radius_entries = sum(1 + len(radius.phase) for group in groups for radius in group.radii)
        def entries(form):
            return 1 + len(form.source) + len(form.phase) + len(form.error)
        predicate_entries = sum(entries(row.form) + 1 for row in self.predicates)
        live_entries = sum(entries(form) for form in forms)
        decoder_entries = len(self.frame.decoder_bias) + sum(len(row) for row in self.frame.decoder_matrix)
        result = dict(source_factors=len(self.frame.sources), phase_factors=len(self.phases),
            residual_factors=len(self.errors), banks=len(self.banks), norm_groups=len(groups),
            norm_constraints=norm_constraints, norm_incidence=incidence,
            radius_coefficients=radius_entries, predicate_rows=len(self.predicates),
            predicate_coefficients=predicate_entries, decoder_coefficients=decoder_entries,
            source_bound_entries=2 * len(self.frame.bounds), lineage_entries=len(self.lineage),
            source_reference_entries=len(self.frame.reference),
            phase_reference_entries=len(self.references),
            bank_coordinate_entries=sum(len(bank.coordinates) for bank in self.banks),
            live_readouts=len(forms), live_readout_coefficients=live_entries,
            live_readout_nnz=sum(value != 0 for form in forms
                for value in (form.constant,) + form.source + form.phase + form.error),
            complete_physical_qualification=False)
        self.work.charge(32 + incidence + radius_entries + predicate_entries + live_entries + decoder_entries)
        result["algebra_work_used"] = self.work.used
        return result
