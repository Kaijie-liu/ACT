"""Opt-in, exact-rational, small PWA reference for the D135 common fiber.

This is not a GPU kernel, a production verifier, or a novelty claim. The owned
relation keeps all signed binary identities and exact gate rows. Bounds use a
constant-cap causal fiber plus a source-box envelope, ignoring other predicates
only in that *outer query*. Membership never drops those predicates. Dense
affine maps, shared Add/Concat and scalar ReLU are supported; general joins of
independently extended branches, convolution, smooth functions and terminal
solver lowering are deliberately absent.
"""

from dataclasses import dataclass
from fractions import Fraction


MAX_SOURCES = 16
MAX_AMPLITUDES = 32
MAX_PHASES = 48
MAX_READOUTS = 64
MAX_ROWS = 256
MAX_BITS = 512


class DisabledError(ValueError):
    """The reference component requires explicit opt-in."""


def _q(value):
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise ValueError("only explicit integers and Fractions are accepted")
    result = Fraction(value)
    if max(result.numerator.bit_length(), result.denominator.bit_length()) > MAX_BITS:
        raise ValueError("reference rational-size cap exceeded")
    return result


def _qs(values):
    return tuple(_q(value) for value in values)


def _dot(left, right):
    if len(left) != len(right):
        raise ValueError("coefficient dimension mismatch")
    result = Fraction(0)
    for a, b in zip(left, right):
        result = _q(result + _q(a * b))
    return result


def _name(value):
    if not isinstance(value, str) or not value:
        raise ValueError("identity names must be nonempty strings")
    return value


@dataclass(frozen=True, eq=False)
class Identity:
    name: str

    def __post_init__(self):
        _name(self.name)


@dataclass(frozen=True)
class Form:
    constant: Fraction
    source: tuple
    phase: tuple
    amplitude: tuple

    def __post_init__(self):
        object.__setattr__(self, "constant", _q(self.constant))
        for attr in ("source", "phase", "amplitude"):
            object.__setattr__(self, attr, _qs(getattr(self, attr)))
        if (len(self.source) > MAX_SOURCES or len(self.phase) > MAX_PHASES
                or len(self.amplitude) > MAX_AMPLITUDES):
            raise ValueError("reference factor cap exceeded")

    def pad(self, phases, amplitudes):
        if phases < len(self.phase) or amplitudes < len(self.amplitude):
            raise ValueError("cannot remove factors by padding")
        return Form(self.constant, self.source,
                    self.phase + (Fraction(0),) * (phases - len(self.phase)),
                    self.amplitude + (Fraction(0),) * (amplitudes - len(self.amplitude)))

    def scale(self, coefficient):
        coefficient = _q(coefficient)
        return Form(_q(coefficient * self.constant),
                    tuple(_q(coefficient * x) for x in self.source),
                    tuple(_q(coefficient * x) for x in self.phase),
                    tuple(_q(coefficient * x) for x in self.amplitude))

    def plus(self, other):
        if (len(self.source), len(self.phase), len(self.amplitude)) != (
                len(other.source), len(other.phase), len(other.amplitude)):
            raise ValueError("unaligned forms")
        return Form(_q(self.constant + other.constant),
                    tuple(_q(a + b) for a, b in zip(self.source, other.source)),
                    tuple(_q(a + b) for a, b in zip(self.phase, other.phase)),
                    tuple(_q(a + b) for a, b in zip(self.amplitude, other.amplitude)))

    def at(self, source, phase01, amplitude):
        if (len(source) != len(self.source) or len(phase01) < len(self.phase)
                or len(amplitude) < len(self.amplitude)):
            raise ValueError("state dimension mismatch")
        result = _q(self.constant + _dot(self.source, source))
        result = _q(result + _dot(self.phase, phase01[:len(self.phase)]))
        return _q(result + _dot(self.amplitude, amplitude[:len(self.amplitude)]))


@dataclass(frozen=True)
class Predicate:
    form: Form
    sense: str
    rhs: Fraction

    def __post_init__(self):
        if not isinstance(self.form, Form) or self.sense not in ("eq", "le"):
            raise ValueError("a linear EQ or LE predicate is required")
        object.__setattr__(self, "rhs", _q(self.rhs))

    def holds(self, source, phase01, amplitude):
        value = self.form.at(source, phase01, amplitude)
        return value == self.rhs if self.sense == "eq" else value <= self.rhs


@dataclass(frozen=True)
class SourceConstraint:
    source: tuple
    phase: tuple
    sense: str
    rhs: Fraction

    def predicate(self):
        return Predicate(Form(0, self.source, self.phase, ()), self.sense, self.rhs)


@dataclass(frozen=True, eq=False)
class Frame:
    bounds: tuple
    sources: tuple
    initial_phases: tuple
    source_predicates: tuple
    decoder_matrix: tuple
    decoder_bias: tuple

    def __post_init__(self):
        object.__setattr__(self, "bounds", tuple(_qs(pair) for pair in self.bounds))
        for attr in ("sources", "initial_phases", "source_predicates"):
            object.__setattr__(self, attr, tuple(getattr(self, attr)))
        object.__setattr__(self, "decoder_matrix", tuple(_qs(row) for row in self.decoder_matrix))
        object.__setattr__(self, "decoder_bias", _qs(self.decoder_bias))
        n = len(self.bounds)
        if (not 1 <= n <= MAX_SOURCES or len(self.sources) != n
                or any(len(pair) != 2 or pair[0] > pair[1] for pair in self.bounds)
                or len(self.initial_phases) > MAX_PHASES
                or len(self.source_predicates) > MAX_ROWS):
            raise ValueError("invalid owned source frame")
        for identities in (self.sources, self.initial_phases):
            if (any(not isinstance(item, Identity) for item in identities)
                    or len({id(item) for item in identities}) != len(identities)
                    or len({item.name for item in identities}) != len(identities)):
                raise ValueError("invalid source or original-phase identities")
        if {id(item) for item in self.sources} & {id(item) for item in self.initial_phases}:
            raise ValueError("continuous and binary identities must remain distinct")
        for item in self.source_predicates:
            if (not isinstance(item, Predicate) or len(item.form.source) != n
                    or len(item.form.phase) != len(self.initial_phases) or item.form.amplitude):
                raise ValueError("source predicate shape mismatch")
        if (not 1 <= len(self.decoder_matrix) <= MAX_READOUTS
                or len(self.decoder_bias) != len(self.decoder_matrix)
                or any(len(row) != n for row in self.decoder_matrix)):
            raise ValueError("original-input decoder shape mismatch")


@dataclass(frozen=True)
class Readout:
    frame: Frame
    phases: tuple
    amplitudes: tuple
    forms: tuple

    def __post_init__(self):
        if not isinstance(self.frame, Frame):
            raise ValueError("readout requires an owned source frame")
        for attr in ("phases", "amplitudes", "forms"):
            object.__setattr__(self, attr, tuple(getattr(self, attr)))
        if not 1 <= len(self.forms) <= MAX_READOUTS:
            raise ValueError("reference readout cap exceeded")
        for form in self.forms:
            if not isinstance(form, Form) or (
                    len(form.source), len(form.phase), len(form.amplitude)) != (
                    len(self.frame.bounds), len(self.phases), len(self.amplitudes)):
                raise ValueError("readout coefficient shape mismatch")


@dataclass(frozen=True)
class Support:
    coefficients: tuple
    value: Fraction


def causal_support(caps, lower, weights):
    """Exact support of 0 <= e <= caps + lower @ e, not of all predicates."""
    caps, weights = _qs(caps), _qs(weights)
    lower = tuple(_qs(row) for row in lower)
    n = len(caps)
    if n > MAX_AMPLITUDES or len(weights) != n or len(lower) != n:
        raise ValueError("causal support shape/cap mismatch")
    if any(c < 0 for c in caps):
        raise ValueError("negative fiber cap")
    for i, row in enumerate(lower):
        if len(row) != n or any(x < 0 for x in row) or any(row[j] for j in range(i, n)):
            raise ValueError("fiber matrix must be nonnegative and strictly lower triangular")
    k = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        current = weights[i]
        for j in range(i + 1, n):
            current = _q(current + _q(lower[j][i] * k[j]))
        k[i] = max(Fraction(0), current)
    return Support(tuple(k), _dot(tuple(k), caps))


@dataclass(frozen=True)
class Fiber:
    frame: Frame
    phases: tuple = ()
    amplitudes: tuple = ()
    caps: tuple = ()
    lower: tuple = ()
    predicates: tuple = ()
    enabled: bool = False

    def __post_init__(self):
        if self.enabled is not True:
            raise DisabledError("pass enabled=True for this isolated reference")
        if not isinstance(self.frame, Frame):
            raise ValueError("owned source frame required")
        for attr in ("phases", "amplitudes", "predicates"):
            object.__setattr__(self, attr, tuple(getattr(self, attr)))
        object.__setattr__(self, "caps", _qs(self.caps))
        object.__setattr__(self, "lower", tuple(_qs(row) for row in self.lower))
        if (len(self.phases) > MAX_PHASES
                or len(self.predicates) + len(self.frame.source_predicates) > MAX_ROWS
                or len(self.amplitudes) != len(self.caps)):
            raise ValueError("reference relation cap/shape mismatch")
        if self.phases[:len(self.frame.initial_phases)] != self.frame.initial_phases:
            raise ValueError("original phase identities changed")
        for identities in (self.phases, self.amplitudes):
            if (any(not isinstance(item, Identity) for item in identities)
                    or len({id(item) for item in identities}) != len(identities)
                    or len({item.name for item in identities}) != len(identities)):
                raise ValueError("identities must remain unique")
        all_identities = self.frame.sources + self.phases + self.amplitudes
        if len({id(item) for item in all_identities}) != len(all_identities):
            raise ValueError("source, binary and amplitude identities must remain distinct")
        causal_support(self.caps, self.lower, (0,) * len(self.caps))
        for predicate in self.frame.source_predicates + self.predicates:
            if (not isinstance(predicate, Predicate)
                    or len(predicate.form.source) != len(self.frame.bounds)
                    or len(predicate.form.phase) > len(self.phases)
                    or len(predicate.form.amplitude) > len(self.amplitudes)):
                raise ValueError("predicate shape mismatch")

    @classmethod
    def box(cls, bounds, *, names=None, phase_names=(), predicates=(),
            decoder_matrix=None, decoder_bias=None, enabled=False):
        if enabled is not True:
            raise DisabledError("pass enabled=True for this isolated reference")
        bounds = tuple(tuple(_q(x) for x in pair) for pair in bounds)
        n = len(bounds)
        if not 1 <= n <= MAX_SOURCES or any(len(pair) != 2 or pair[0] > pair[1] for pair in bounds):
            raise ValueError("invalid source box")
        names = tuple(names) if names is not None else tuple("z" + str(i) for i in range(n))
        if len(names) != n or len(set(names)) != n:
            raise ValueError("invalid source identities")
        sources = tuple(Identity(name) for name in names)
        phases = tuple(Identity(name) for name in phase_names)
        if len(phases) > MAX_PHASES or len({item.name for item in phases}) != len(phases):
            raise ValueError("invalid original phases")
        source_predicates = tuple(item.predicate() for item in predicates)
        for item in source_predicates:
            if len(item.form.source) != n or len(item.form.phase) != len(phases):
                raise ValueError("source predicate shape mismatch")
        if decoder_matrix is None:
            decoder_matrix = tuple(tuple(int(i == j) for j in range(n)) for i in range(n))
        decoder_matrix = tuple(_qs(row) for row in decoder_matrix)
        if not 1 <= len(decoder_matrix) <= MAX_READOUTS or any(len(row) != n for row in decoder_matrix):
            raise ValueError("invalid original-input decoder")
        decoder_bias = ((0,) * len(decoder_matrix) if decoder_bias is None else decoder_bias)
        decoder_bias = _qs(decoder_bias)
        if len(decoder_bias) != len(decoder_matrix):
            raise ValueError("decoder bias shape mismatch")
        frame = Frame(bounds, sources, phases, source_predicates, decoder_matrix, decoder_bias)
        return cls(frame, phases=phases, enabled=True)

    def _zero(self):
        return Form(0, (0,) * len(self.frame.bounds), (0,) * len(self.phases),
                    (0,) * len(self.amplitudes))

    def _readout(self, forms):
        return Readout(self.frame, self.phases, self.amplitudes, tuple(forms))

    def _aligned(self, view):
        if (not isinstance(view, Readout) or view.frame is not self.frame
                or len(view.phases) > len(self.phases)
                or len(view.amplitudes) > len(self.amplitudes)
                or self.phases[:len(view.phases)] != view.phases
                or self.amplitudes[:len(view.amplitudes)] != view.amplitudes):
            raise ValueError("misaligned frame or independently extended branch")
        return tuple(form.pad(len(self.phases), len(self.amplitudes)) for form in view.forms)

    def sources(self):
        n = len(self.frame.bounds)
        return self._readout(Form(0, tuple(int(i == j) for j in range(n)),
                                  (0,) * len(self.phases), (0,) * len(self.amplitudes))
                             for i in range(n))

    def signed_phases(self):
        if not self.phases:
            raise ValueError("no phase readout")
        return self._readout(Form(-1, (0,) * len(self.frame.bounds),
                                  tuple(2 * int(i == j) for j in range(len(self.phases))),
                                  (0,) * len(self.amplitudes)) for i in range(len(self.phases)))

    def affine(self, view, matrix, bias):
        forms = self._aligned(view)
        matrix, bias = tuple(_qs(row) for row in matrix), _qs(bias)
        if not 1 <= len(matrix) <= MAX_READOUTS or len(matrix) != len(bias):
            raise ValueError("affine output shape mismatch")
        outputs = []
        for row, offset in zip(matrix, bias):
            if len(row) != len(forms):
                raise ValueError("affine input shape mismatch")
            result = self._zero()
            for coefficient, form in zip(row, forms):
                result = result.plus(form.scale(coefficient))
            outputs.append(result.plus(Form(offset, (0,) * len(self.frame.bounds),
                                             (0,) * len(self.phases), (0,) * len(self.amplitudes))))
        return self._readout(outputs)

    def add(self, left, right):
        left, right = self._aligned(left), self._aligned(right)
        if len(left) != len(right):
            raise ValueError("Add shape mismatch")
        return self._readout(a.plus(b) for a, b in zip(left, right))

    def concat(self, *views):
        return self._readout(form for view in views for form in self._aligned(view))

    def _source_upper(self, form):
        result = form.constant
        for coefficient, (lower, upper) in zip(form.source, self.frame.bounds):
            result = _q(result + max(_q(coefficient * lower), _q(coefficient * upper)))
        for coefficient in form.phase:
            result = _q(result + max(Fraction(0), coefficient))
        return result

    def support(self, view, direction):
        forms, direction = self._aligned(view), _qs(direction)
        if len(forms) != len(direction):
            raise ValueError("query direction mismatch")
        form = self._zero()
        for coefficient, item in zip(direction, forms):
            form = form.plus(item.scale(coefficient))
        return _q(self._source_upper(form)
                  + causal_support(self.caps, self.lower, form.amplitude).value)

    def bounds(self, scalar):
        if len(self._aligned(scalar)) != 1:
            raise ValueError("scalar bound required")
        return -self.support(scalar, (-1,)), self.support(scalar, (1,))

    def relu(self, scalar, name, *, bounds=None):
        """Exact graph extension; source/fiber bounds only select sound rows."""
        forms = self._aligned(scalar)
        if len(forms) != 1:
            raise ValueError("reference ReLU is scalar")
        name = _name(name)
        if any(item.name == name for item in self.phases):
            raise ValueError("original gate identity cannot be reused")
        certified_lower, certified_upper = self.bounds(scalar)
        if bounds is not None:
            bounds = _qs(bounds)
            if len(bounds) != 2 or bounds[0] > certified_lower or bounds[1] < certified_upper:
                raise ValueError("supplied gate bounds are not certified by this element")
            certified_lower, certified_upper = bounds
        lower, upper = min(Fraction(0), certified_lower), max(Fraction(0), certified_upper)
        old = forms[0]
        inactive = certified_upper <= 0
        cap = Fraction(0) if inactive else max(Fraction(0), self._source_upper(old))
        phase = Identity(name)
        amplitude = Identity("amplitude:" + name)
        nphase, namp = len(self.phases) + 1, len(self.amplitudes) + 1
        g = old.pad(nphase, namp)
        q = Form(0, (0,) * len(self.frame.bounds), (0,) * nphase, (0,) * (namp - 1) + (1,))
        alpha = Form(0, (0,) * len(self.frame.bounds), (0,) * (nphase - 1) + (1,), (0,) * namp)
        predicates = self.predicates + (
            Predicate(g.plus(q.scale(-1)), "le", 0),
            Predicate(q.plus(alpha.scale(-upper)), "le", 0),
            Predicate(q.plus(g.scale(-1)).plus(alpha.scale(-lower)), "le", -lower),
        )
        matrix = tuple(row + (Fraction(0),) for row in self.lower) + (
            tuple(Fraction(0) if inactive else max(Fraction(0), coefficient)
                  for coefficient in old.amplitude) + (Fraction(0),),
        )
        extended = Fiber(self.frame, self.phases + (phase,), self.amplitudes + (amplitude,),
                         self.caps + (cap,), matrix, predicates, enabled=True)
        return extended, extended._readout((q,))

    def contains(self, source, signed_phases, amplitude):
        """Exact membership, including all predicates; fractional bits reject."""
        try:
            source, signed_phases, amplitude = _qs(source), _qs(signed_phases), _qs(amplitude)
            if (len(source) != len(self.frame.bounds) or len(signed_phases) != len(self.phases)
                    or len(amplitude) != len(self.amplitudes)
                    or any(value not in (-1, 1) for value in signed_phases)):
                return False
            phase01 = tuple((value + 1) / 2 for value in signed_phases)
            if any(not lower <= value <= upper for value, (lower, upper) in zip(source, self.frame.bounds)):
                return False
            if any(not 0 <= value <= _q(cap + _dot(row, amplitude))
                   for value, cap, row in zip(amplitude, self.caps, self.lower)):
                return False
            return all(item.holds(source, phase01, amplitude)
                       for item in self.frame.source_predicates + self.predicates)
        except (ValueError, TypeError, OverflowError):
            return False

    def evaluate(self, view, source, signed_phases, amplitude):
        forms = self._aligned(view)
        if not self.contains(source, signed_phases, amplitude):
            raise ValueError("state is not a member of the retained exact relation")
        source, signed_phases, amplitude = _qs(source), _qs(signed_phases), _qs(amplitude)
        phase01 = tuple((value + 1) / 2 for value in signed_phases)
        return tuple(form.at(source, phase01, amplitude) for form in forms)

    def decode(self, source, signed_phases, amplitude):
        if not self.contains(source, signed_phases, amplitude):
            raise ValueError("decoder requires a member, not a relaxed fiber point")
        source = _qs(source)
        return tuple(_q(offset + _dot(row, source))
                     for row, offset in zip(self.frame.decoder_matrix, self.frame.decoder_bias))
