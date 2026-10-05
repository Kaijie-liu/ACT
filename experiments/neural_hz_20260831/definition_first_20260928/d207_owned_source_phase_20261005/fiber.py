"""Default-off, exact-rational owned Neural-HZ reference calculus.

This module has no solver, model loader, execution entry point, or production
hook. Logical accounting is not physical-memory or real-network qualification.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from fractions import Fraction


MAX_WORK = 256_000_000
MAX_BRANCH_WORK = 200_000_000
MAX_ENTRIES = 64_000_000
MAX_BITS = 512
Q = Fraction
_OWNED_CONSTRUCTION = object()


class DomainError(ValueError):
    pass


class BudgetError(DomainError):
    pass


def _q(value):
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise DomainError("only exact integer/Fraction coefficients are admitted")
    value = Fraction(value)
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_BITS:
        raise BudgetError("rational bit budget exceeded")
    return value


@dataclass(frozen=True)
class Form:
    constant: Fraction = Q(0)
    terms: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "constant", _q(self.constant))
        result, previous = [], -1
        for index, coefficient in self.terms:
            if isinstance(index, bool) or not isinstance(index, int) or index <= previous:
                raise DomainError("Form indices must be nonnegative, sorted and unique")
            coefficient = _q(coefficient)
            if not coefficient:
                raise DomainError("Form must not store zero coefficients")
            result.append((index, coefficient))
            previous = index
        object.__setattr__(self, "terms", tuple(result))

    def at(self, assignment):
        """Exact inspection only; owned operations also charge their evaluation."""
        result = self.constant
        for index, coefficient in self.terms:
            result = _q(result + _q(coefficient * _q(assignment[index])))
        return result


@dataclass(frozen=True)
class Predicate:
    form: Form
    relation: str = "le"
    rhs: Fraction = Q(0)

    def __post_init__(self):
        if not isinstance(self.form, Form) or self.relation not in ("eq", "le"):
            raise DomainError("expected an exact EQ/LE predicate")
        object.__setattr__(self, "rhs", _q(self.rhs))


@dataclass(frozen=True, eq=False)
class Identity:
    name: str


@dataclass(frozen=True)
class Factor:
    identity: Identity
    kind: str
    lower: Fraction
    upper: Fraction
    producer: object = None

    @property
    def name(self):
        return self.identity.name


class _Work:
    def __init__(self, max_work, max_branch_work, max_entries):
        for value, ceiling in ((max_work, MAX_WORK),
                               (max_branch_work, MAX_BRANCH_WORK),
                               (max_entries, MAX_ENTRIES)):
            if type(value) is not int or not 0 < value <= ceiling:
                raise DomainError("resource caps may only tighten positive defaults")
        self.max_work, self.max_branch_work = max_work, max_branch_work
        self.max_entries = max_entries
        self.used = self.entries = self.branch_used = self.depth = self.readouts = 0
        self.failed = False

    @contextmanager
    def branch(self):
        if self.failed:
            raise BudgetError("this lineage has exhausted a resource budget")
        # Conservatively charge the entire shared lineage to the branch cap.
        # Neither a new query nor a layer/sibling resets either ledger.
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1

    def charge(self, work, entries=0):
        self.used += work
        self.branch_used += work
        self.entries += entries
        if (self.used > self.max_work or self.branch_used > self.max_branch_work
                or self.entries > self.max_entries):
            self.failed = True
            raise BudgetError("owned lineage logical resource budget exceeded")


@dataclass(frozen=True, eq=False)
class _Frame:
    work: _Work
    source_count: int
    initial_phase_count: int
    decoder: tuple


@dataclass(frozen=True)
class Readout:
    frame: _Frame
    identities: tuple
    forms: tuple


@dataclass(frozen=True)
class Gate:
    input: Form
    phase_index: int
    q_index: int
    lower: Fraction
    upper: Fraction
    triangle: Form
    stable: int


@dataclass(frozen=True)
class Pair:
    positions: tuple
    caps: tuple
    scale: Form
    lower: Fraction
    rows: tuple
    constant_h: tuple
    constant_m: tuple
    source_h: tuple
    source_m: tuple


@dataclass(frozen=True)
class Bank:
    parent_count: int
    gates: tuple
    pairs: tuple


@dataclass(frozen=True)
class Fiber:
    _frame: _Frame
    factors: tuple
    predicates: tuple
    banks: tuple = ()
    _events: tuple = ()
    _authority: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._authority is not _OWNED_CONSTRUCTION:
            raise DomainError("Fiber construction is owned; use explicit opt-in box")

    @classmethod
    def box(cls, bounds, *, names=None, phase_names=(), predicates=(),
            decoder_matrix=None, decoder_bias=None, enabled=False,
            max_work=MAX_WORK, max_branch_work=MAX_BRANCH_WORK,
            max_entries=MAX_ENTRIES):
        if enabled is not True:
            raise DomainError("D207 requires explicit enabled=True")
        meter = _Work(max_work, max_branch_work, max_entries)
        with meter.branch():
            bounds = tuple(bounds)
            names = tuple(names) if names is not None else tuple(
                "x%d" % index for index in range(len(bounds)))
            phase_names = tuple(phase_names)
            meter.charge(20 + 3 * len(bounds) + len(phase_names),
                         20 + 3 * len(bounds) + len(phase_names))
            cls._names(names + phase_names, (), len(bounds) + len(phase_names))
            factors = []
            for name, interval in zip(names, bounds):
                lower, upper = map(_q, interval)
                if lower > upper:
                    raise DomainError("reversed source bounds")
                factors.append(Factor(Identity(name), "source", lower, upper))
            if len(names) != len(bounds):
                raise DomainError("source name shape mismatch")
            factors.extend(Factor(Identity(name), "phase", Q(-1), Q(1))
                           for name in phase_names)
            decoder = []
            if decoder_matrix is None:
                if decoder_bias is not None:
                    raise DomainError("decoder bias requires its complete matrix")
                decoder = [Form(0, ((i, Q(1)),)) for i in range(len(bounds))]
            else:
                matrix = tuple(tuple(row) for row in decoder_matrix)
                bias = tuple(decoder_bias) if decoder_bias is not None else (0,) * len(matrix)
                meter.charge(sum(len(row) for row in matrix) + len(bias),
                             sum(len(row) for row in matrix) + len(bias))
                if len(bias) != len(matrix):
                    raise DomainError("decoder bias shape mismatch")
                for row, offset in zip(matrix, bias):
                    if len(row) != len(bounds):
                        raise DomainError("decoder consumes all original source coordinates")
                    row = tuple(map(_q, row))
                    decoder.append(Form(offset, tuple((i, c) for i, c in enumerate(row) if c)))
            frame = _Frame(meter, len(bounds), len(phase_names), tuple(decoder))
            fiber = cls(frame, tuple(factors), tuple(predicates), _authority=_OWNED_CONSTRUCTION)
            for predicate in fiber.predicates:
                if not isinstance(predicate, Predicate):
                    raise DomainError("invalid initial predicate")
                fiber._validate_form(predicate.form, len(factors))
            fiber._retain_state()
            meter.charge(sum(1 + 2 * len(f.terms) for f in decoder),
                         sum(1 + 2 * len(f.terms) for f in decoder))
            return fiber

    @staticmethod
    def _names(names, existing, expected):
        if (len(names) != expected or any(not isinstance(n, str) or not n for n in names)
                or len(set(names)) != len(names) or set(names).intersection(existing)):
            raise DomainError("factor names must be unique nonempty strings with full shape")

    @property
    def _work(self):
        return self._frame.work

    def _validate_form(self, form, width=None):
        if not isinstance(form, Form):
            raise DomainError("expected Form")
        limit = len(self.factors) if width is None else width
        self._work.charge(1 + len(form.terms))
        if form.terms and form.terms[-1][0] >= limit:
            raise DomainError("Form refers beyond owned prefix")

    def _align(self, view):
        if not isinstance(view, Readout) or view.frame is not self._frame:
            raise DomainError("foreign source frame")
        self._work.charge(1 + len(view.identities))
        if len(view.identities) > len(self.factors) or any(
                identity is not self.factors[i].identity
                for i, identity in enumerate(view.identities)):
            raise DomainError("independent sibling extensions cannot be joined")
        for form in view.forms:
            self._validate_form(form, len(view.identities))
        return view.forms

    def _view(self, forms):
        forms = tuple(forms)
        for form in forms:
            self._validate_form(form)
        entries = len(self.factors) + len(forms) + sum(1 + 2 * len(f.terms) for f in forms)
        self._work.charge(entries, entries)
        self._work.readouts += 1
        return Readout(self._frame, tuple(f.identity for f in self.factors), forms)

    def _retain_state(self):
        # Include identities/names, tuple references and every scalar metadata
        # field, with deliberate overcounting of shared prefix records.
        entries = 8 * len(self.factors) + 2 * len(self._events) + len(self.banks)
        entries += sum(3 + 2 * len(p.form.terms) for p in self.predicates)
        entries += sum(1 + 2 * len(f.producer.terms) for f in self.factors if f.producer is not None)
        for bank in self.banks:
            entries += 3 + len(bank.gates) + len(bank.pairs)
            entries += sum(12 + 2 * len(g.input.terms) + 2 * len(g.triangle.terms)
                           for g in bank.gates)
            for pair in bank.pairs:
                entries += 12 + 2 * len(pair.scale.terms)
                entries += sum(1 + 2 * len(f.terms) for f in pair.constant_h + pair.source_h)
                entries += len(pair.rows)
        self._work.charge(entries, entries)

    def _spawn(self, factors, predicates, banks, events):
        child = Fiber(self._frame, tuple(factors), tuple(predicates), tuple(banks),
                      tuple(events), _authority=_OWNED_CONSTRUCTION)
        child._retain_state()
        return child

    def _linear(self, constant=0, pieces=()):
        constant, coefficients = _q(constant), {}
        for multiplier, form in pieces:
            multiplier = _q(multiplier)
            self._work.charge(3 + 3 * len(form.terms))
            constant = _q(constant + _q(multiplier * form.constant))
            for index, value in form.terms:
                coefficients[index] = _q(coefficients.get(index, Q(0)) + _q(multiplier * value))
        ordered = sorted((i, c) for i, c in coefficients.items() if c)
        self._work.charge(len(coefficients) * max(1, len(coefficients).bit_length()))
        return Form(constant, tuple(ordered))

    def _canonical(self, form):
        """Substitute only exact owned alias producers; never erase factors."""
        for index in range(len(self.factors) - 1, -1, -1):
            factor = self.factors[index]
            self._work.charge(1 + len(form.terms))
            if factor.kind != "alias":
                continue
            coefficient = dict(form.terms).get(index, Q(0))
            if coefficient:
                rest = Form(form.constant, tuple((i, c) for i, c in form.terms if i != index))
                form = self._linear(pieces=((1, rest), (coefficient, factor.producer)))
        return form

    def sources(self):
        with self._work.branch():
            return self._view(Form(0, ((i, Q(1)),)) for i in range(self._frame.source_count))

    def signed_phases(self):
        with self._work.branch():
            return self._view(Form(0, ((i, Q(1)),)) for i, f in enumerate(self.factors)
                              if f.kind == "phase")

    def affine(self, view, matrix, bias=None):
        with self._work.branch():
            forms = self._align(view)
            matrix = tuple(tuple(row) for row in matrix)
            bias = tuple(bias) if bias is not None else (0,) * len(matrix)
            self._work.charge(sum(len(row) for row in matrix) + len(bias),
                              sum(len(row) for row in matrix) + len(bias))
            if len(bias) != len(matrix) or any(len(row) != len(forms) for row in matrix):
                raise DomainError("affine shape mismatch")
            return self._view(self._linear(b, zip(row, forms)) for row, b in zip(matrix, bias))

    def add(self, left, right):
        with self._work.branch():
            left, right = self._align(left), self._align(right)
            if len(left) != len(right):
                raise DomainError("Add shape mismatch")
            return self._view(self._linear(pieces=((1, a), (1, b))) for a, b in zip(left, right))

    def concat(self, *views):
        with self._work.branch():
            return self._view(form for view in views for form in self._align(view))

    def select(self, view, indices):
        with self._work.branch():
            forms, indices = self._align(view), tuple(indices)
            if any(type(i) is not int or not 0 <= i < len(forms) for i in indices):
                raise DomainError("selection outside complete readout")
            return self._view(forms[i] for i in indices)

    def alias(self, view, names):
        with self._work.branch():
            forms, names = self._align(view), tuple(names)
            self._names(names, tuple(f.name for f in self.factors), len(forms))
            ranges = [self.bounds(self._view((form,))) for form in forms]
            factors, predicates, events, output = list(self.factors), list(self.predicates), list(self._events), []
            for name, form, (lower, upper) in zip(names, forms, ranges):
                index = len(factors)
                factors.append(Factor(Identity(name), "alias", lower, upper, form))
                output.append(Form(0, ((index, Q(1)),)))
                predicates.append(Predicate(self._linear(pieces=((1, output[-1]), (-1, form))), "eq"))
                events.append(("alias", index))
            child = self._spawn(factors, predicates, self.banks, events)
            return child, child._view(output)

    def _source_scale(self, first, second):
        first, second = self._canonical(first), self._canonical(second)
        differences = (self._linear(pieces=((1, first), (-Q(1, 2), second))),
                       self._linear(pieces=((1, second), (-Q(1, 2), first))))
        centers, radii, normalized, caps = {}, {}, [], []
        for difference in differences:
            offset, values = difference.constant, {}
            for index, coefficient in difference.terms:
                factor = self.factors[index]
                center = _q((factor.lower + factor.upper) / 2)
                radius = _q((factor.upper - factor.lower) / 2)
                centers[index], radii[index] = center, radius
                offset = _q(offset + _q(coefficient * center))
                if radius:
                    values[index] = _q(coefficient * radius)
                self._work.charge(10)
            cap = offset
            for value in values.values():
                cap = _q(cap + abs(value))
            caps.append(cap)
            normalized.append(values)
        if min(caps) <= 0:
            return None
        common, omega = {}, Q(0)
        for index, value in normalized[0].items():
            other = normalized[1].get(index, Q(0))
            self._work.charge(8)
            if value and other and (value > 0) == (other > 0):
                magnitude = min(_q(abs(value) / caps[0]), _q(abs(other) / caps[1]))
                common[index] = magnitude if value > 0 else -magnitude
                omega = _q(omega + magnitude)
        tau = min(Q(1), _q(1 / _q(4 * omega))) if omega else Q(1)
        omega = _q(tau * omega)
        constant, terms = _q(1 - omega), []
        for index, value in sorted(common.items()):
            coefficient = _q(_q(tau * value) / radii[index])
            constant = _q(constant - _q(coefficient * centers[index]))
            terms.append((index, coefficient))
            self._work.charge(6)
        return tuple(caps), Form(constant, tuple(terms)), _q(1 - 2 * omega)

    def _pair(self, positions, gates):
        one, two = (gates[i] for i in positions)
        result = self._source_scale(one.input, two.input)
        if result is None:
            return None
        caps, scale, lower = result
        rows, constant_h, constant_m, source_h, source_m = [], [], [], [], []
        qs = tuple(Form(0, ((g.q_index, Q(1)),)) for g in (one, two))
        betas = tuple(Form(Q(1, 2), ((g.phase_index, Q(1, 2)),)) for g in (one, two))
        for i, gate in enumerate((one, two)):
            ci, cj = caps[i], caps[1 - i]
            upper = _q(_q(ci + cj / 2) / Q(3, 4))
            qi, qj, beta = qs[i], qs[1 - i], betas[i]
            rows.extend((
                Predicate(self._linear(pieces=((-1, qi),))),
                Predicate(self._linear(pieces=((1, qi), (-upper, beta)))),
                Predicate(self._linear(pieces=((1, qi), (-upper, scale),
                                              (-_q(upper * lower), beta))), rhs=-_q(upper * lower)),
                Predicate(self._linear(pieces=((1, qi), (-ci, beta), (-Q(1, 2), qj)))),
                Predicate(self._linear(pieces=((1, qi), (-ci, scale),
                                              (-_q(ci * lower), beta), (-Q(1, 2), qj))),
                          rhs=-_q(ci * lower)),
            ))
            ell = -gate.lower
            for lo, sc, hs, ms in ((Q(1), Form(1), constant_h, constant_m),
                                   (lower, scale, source_h, source_m)):
                denominator = _q(_q(ci * lo) + ell)
                hs.append(self._linear(pieces=((_q(ci * lo / denominator), gate.input),
                                               (_q(ci * ell / denominator), sc))))
                ms.append(_q(ell / _q(2 * denominator)))
                self._work.charge(12)
        return Pair(positions, caps, scale, lower, tuple(rows), tuple(constant_h),
                    tuple(constant_m), tuple(source_h), tuple(source_m))

    def relu_bank(self, view, phase_names):
        with self._work.branch():
            forms, phase_names = self._align(view), tuple(phase_names)
            self._names(phase_names, tuple(f.name for f in self.factors), len(forms))
            q_names = tuple(name + ":q" for name in phase_names)
            self._names(phase_names + q_names, tuple(f.name for f in self.factors), 2 * len(forms))
            ranges = [self.bounds(self._view((form,))) for form in forms]
            factors, predicates, gates, outputs = list(self.factors), list(self.predicates), [], []
            for name, q_name, form, (bound_lower, bound_upper) in zip(phase_names, q_names, forms, ranges):
                lower, upper = min(Q(0), bound_lower), max(Q(0), bound_upper)
                phase_index, q_index = len(factors), len(factors) + 1
                factors.extend((Factor(Identity(name), "phase", Q(-1), Q(1)),
                                Factor(Identity(q_name), "relu", Q(0), upper)))
                beta, q = Form(Q(1, 2), ((phase_index, Q(1, 2)),)), Form(0, ((q_index, Q(1)),))
                outputs.append(q)
                predicates.extend((
                    Predicate(self._linear(pieces=((-1, q),))),
                    Predicate(self._linear(pieces=((1, form), (-1, q)))),
                    Predicate(self._linear(pieces=((1, q), (-upper, beta)))),
                    Predicate(self._linear(pieces=((1, q), (-1, form), (-lower, beta))), rhs=-lower),
                ))
                stable = -1 if bound_upper <= 0 else (1 if bound_lower >= 0 else 0)
                triangle = (Form() if stable == -1 else form if stable == 1 else
                            self._linear(_q(-upper * lower / (upper - lower)),
                                         ((_q(upper / (upper - lower)), form),)))
                gates.append(Gate(form, phase_index, q_index, lower, upper, triangle, stable))
            pairs = []
            for position in range(0, len(gates) - 1, 2):
                if gates[position].stable == gates[position + 1].stable == 0:
                    pair = self._pair((position, position + 1), gates)
                    if pair is not None:
                        pairs.append(pair)
                        predicates.extend(pair.rows)
            bank = Bank(len(self.factors), tuple(gates), tuple(pairs))
            child = self._spawn(factors, predicates, self.banks + (bank,), self._events + (("bank", bank),))
            return child, child._view(outputs)

    def relu(self, scalar, name):
        with self._work.branch():
            if len(self._align(scalar)) != 1:
                raise DomainError("relu requires one scalar; use complete relu_bank")
            return self.relu_bank(scalar, (name,))

    def _certificate(self, form, mode):
        for kind, event in reversed(self._events):
            weights = dict(form.terms)
            self._work.charge(1 + len(form.terms))
            if kind == "alias":
                weight = weights.pop(event, Q(0))
                form = self._linear(pieces=((1, Form(form.constant, tuple(sorted(weights.items())))),
                                           (weight, self.factors[event].producer)))
                continue
            bank, pieces = event, []
            local = [weights.pop(g.q_index, Q(0)) for g in bank.gates]
            handled = set()
            if mode:
                for pair in bank.pairs:
                    i, j = pair.positions
                    hs = pair.constant_h if mode == 1 else pair.source_h
                    a, b = pair.constant_m if mode == 1 else pair.source_m
                    denominator = _q(1 - _q(a * b))
                    if denominator <= 0:
                        raise DomainError("invalid pair contraction")
                    first = max(Q(0), local[i], _q(_q(local[i] + _q(b * local[j])) / denominator))
                    second = max(Q(0), local[j], _q(_q(local[j] + _q(a * local[i])) / denominator))
                    pieces.extend(((first, hs[0]), (second, hs[1])))
                    handled.update((i, j))
                    self._work.charge(24)
            for i, gate in enumerate(bank.gates):
                if i not in handled:
                    # This fixed certificate uses q>=0 for negative crossing
                    # coefficients, not the optimal support of a triangle LP.
                    weight = local[i] if gate.stable == 1 else max(Q(0), local[i])
                    pieces.append((weight, gate.triangle))
            rest = Form(form.constant, tuple(sorted(weights.items())))
            form = self._linear(pieces=((1, rest), *pieces))
        upper = form.constant
        for index, coefficient in form.terms:
            factor = self.factors[index]
            if factor.kind in ("alias", "relu"):
                raise DomainError("unprocessed birth in support")
            value = factor.upper if coefficient >= 0 else factor.lower
            upper = _q(upper + _q(coefficient * value))
            self._work.charge(3)
        return upper

    def support_certificates(self, view, direction):
        with self._work.branch():
            forms, direction = self._align(view), tuple(direction)
            if len(forms) != len(direction):
                raise DomainError("support direction shape mismatch")
            combined = self._linear(pieces=zip(direction, forms))
            return tuple(self._certificate(combined, mode) for mode in (0, 1, 2))

    def support(self, view, direction):
        return min(self.support_certificates(view, direction))

    def bounds(self, scalar):
        with self._work.branch():
            if len(self._align(scalar)) != 1:
                raise DomainError("bounds requires one scalar")
            return -self.support(scalar, (-1,)), self.support(scalar, (1,))

    def _at(self, form, assignment):
        self._work.charge(1 + 3 * len(form.terms))
        return form.at(assignment)

    def contains(self, assignment):
        with self._work.branch():
            assignment = tuple(map(_q, assignment))
            if len(assignment) != len(self.factors):
                return False
            for factor, value in zip(self.factors, assignment):
                self._work.charge(3)
                if not factor.lower <= value <= factor.upper:
                    return False
                if factor.kind == "phase" and value not in (Q(-1), Q(1)):
                    return False
            for predicate in self.predicates:
                value = self._at(predicate.form, assignment)
                if ((predicate.relation == "eq" and value != predicate.rhs)
                        or (predicate.relation == "le" and value > predicate.rhs)):
                    return False
            return True

    def evaluate(self, view, assignment):
        with self._work.branch():
            forms, assignment = self._align(view), tuple(assignment)
            if not self.contains(assignment):
                raise DomainError("evaluation requires a complete legal member")
            return tuple(self._at(form, assignment) for form in forms)

    def decode(self, assignment):
        with self._work.branch():
            assignment = tuple(assignment)
            if not self.contains(assignment):
                raise DomainError("decoder requires a complete legal member")
            return tuple(self._at(form, assignment) for form in self._frame.decoder)

    def complete(self, source_values, signed_initial_phases=()):
        with self._work.branch():
            sources, phases = tuple(map(_q, source_values)), tuple(map(_q, signed_initial_phases))
            if len(sources) != self._frame.source_count or len(phases) != self._frame.initial_phase_count:
                raise DomainError("complete requires all original sources and signed phases")
            values = list(sources + phases)
            for kind, event in self._events:
                if kind == "alias":
                    if len(values) != event:
                        raise DomainError("alias birth order violated")
                    values.append(self._at(self.factors[event].producer, values))
                else:
                    if len(values) != event.parent_count:
                        raise DomainError("bank birth order violated")
                    for gate in event.gates:
                        value = self._at(gate.input, values)
                        if gate.phase_index != len(values) or gate.q_index != len(values) + 1:
                            raise DomainError("gate birth order violated")
                        values.extend((Q(1) if value > 0 else Q(-1), max(Q(0), value)))
            result = tuple(values)
            if not self.contains(result):
                raise DomainError("source assignment does not extend to a legal member")
            return result

    def cost_report(self):
        """Stored-state counts plus conservative cumulative lineage accounting.

        Retained entries never decrease, including temporary readouts and copied
        prefix metadata. This upper ledger is not a resident-memory estimate.
        """
        with self._work.branch():
            self._work.charge(len(self.factors) + len(self.predicates) + len(self.banks))
            return {
                "factors": len(self.factors),
                "continuous": sum(f.kind != "phase" for f in self.factors),
                "binary": sum(f.kind == "phase" for f in self.factors),
                "aliases": sum(f.kind == "alias" for f in self.factors),
                "predicates": len(self.predicates),
                "predicate_nnz": sum(len(p.form.terms) for p in self.predicates),
                "predicate_scalar_coefficients": sum(
                    len(p.form.terms) + bool(p.form.constant) + bool(p.rhs)
                    for p in self.predicates),
                "banks": len(self.banks),
                "pairs": sum(len(b.pairs) for b in self.banks),
                "readouts": self._work.readouts,
                "work_used": self._work.used,
                "branch_work_used": self._work.branch_used,
                "entries": self._work.entries,
                "whole_work_cap": self._work.max_work,
                "branch_work_cap": self._work.max_branch_work,
                "entries_cap": self._work.max_entries,
                "physical_qualified": False,
                "native_qualified": False,
                "formal_gain": 0,
            }
