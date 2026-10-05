"""Default-off owned slack closure; exact mathematics, not native qualification.

The frozen D207 graph and its first three fixed queries remain present.  A
fourth query consumes relations born from certified common-parent slacks.
There is no solver, model loader, external bound oracle, or execution entry.
"""

from dataclasses import dataclass, field

from experiments.neural_hz_20260831.definition_first_20260928.d207_owned_source_phase_20261005 import fiber as _base


Q = _base.Q
Form, Predicate, Readout = _base.Form, _base.Predicate, _base.Readout
DomainError, BudgetError = _base.DomainError, _base.BudgetError
MAX_WORK, MAX_BRANCH_WORK = _base.MAX_WORK, _base.MAX_BRANCH_WORK
MAX_ENTRIES, MAX_BITS = _base.MAX_ENTRIES, _base.MAX_BITS
_q = _base._q


@dataclass(frozen=True)
class Proof:
    bound: Q
    subject: Form
    le_weights: tuple
    eq_weights: tuple
    frame: object = field(repr=False, compare=False)
    identities: tuple = field(repr=False, compare=False)
    predicates: tuple = field(repr=False, compare=False)


@dataclass(frozen=True)
class ClosurePair:
    pair: _base.Pair
    row_indices: tuple
    parent_proofs: tuple
    scale_proofs: tuple
    common_weights: tuple
    common_form: Form
    common_box_upper: Q
    tau: Q


@dataclass(frozen=True)
class BankProof:
    bank: _base.Bank
    gate_rows: tuple
    old_pair_rows: tuple
    closure_pairs: tuple


@dataclass(frozen=True)
class Fiber(_base.Fiber):
    _proof_banks: tuple = ()
    _alias_rows: tuple = ()

    def __post_init__(self):
        super().__post_init__()
        if (len(self._proof_banks) != len(self.banks)
                or any(meta.bank is not bank for meta, bank in
                       zip(self._proof_banks, self.banks))):
            raise DomainError("owned bank proof metadata was lost")

    @staticmethod
    def _proof_entries(proof):
        return (8 + len(proof.identities) + len(proof.predicates)
                + 2 * len(proof.subject.terms) + 3 * len(proof.le_weights)
                + 2 * len(proof.eq_weights))

    def _retain_state(self):
        super()._retain_state()
        entries = 2 * len(self._alias_rows) + len(self._proof_banks)
        for meta in self._proof_banks:
            entries += 4 + 5 * len(meta.gate_rows) + 11 * len(meta.old_pair_rows)
            for closure in meta.closure_pairs:
                pair = closure.pair
                entries += (25 + len(closure.row_indices)
                            + 3 * len(closure.common_weights)
                            + 2 * len(closure.common_form.terms)
                            + 2 * len(pair.scale.terms))
                # pair.rows retains its freshly constructed objects even when
                # equal installed rows reuse earlier predicate indices.
                entries += len(pair.rows) + sum(3 + 2 * len(row.form.terms)
                                                for row in pair.rows)
                entries += sum(1 + 2 * len(form.terms)
                               for form in pair.constant_h + pair.source_h)
                entries += sum(self._proof_entries(proof) for proof in
                               closure.parent_proofs + closure.scale_proofs)
        self._work.charge(entries, entries)

    def _spawn(self, factors, predicates, banks, events, *, proof_banks=None,
               alias_rows=None):
        child = Fiber(self._frame, tuple(factors), tuple(predicates), tuple(banks),
                      tuple(events), _authority=_base._OWNED_CONSTRUCTION,
                      _proof_banks=self._proof_banks if proof_banks is None else tuple(proof_banks),
                      _alias_rows=self._alias_rows if alias_rows is None else tuple(alias_rows))
        child._retain_state()
        return child

    def _linear(self, constant=0, pieces=()):
        # Account for transient expansion, not just eventually retained forms.
        pieces = tuple(pieces)
        terms = sum(len(form.terms) for _, form in pieces)
        self._work.charge(2 + len(pieces) + terms, 2 * len(pieces) + 3 * terms)
        result = super()._linear(constant, pieces)
        self._work.charge(1 + len(result.terms), 1 + 2 * len(result.terms))
        return result

    def _atom(self, key, width=None, row_count=None):
        width = len(self.factors) if width is None else width
        row_count = len(self.predicates) if row_count is None else row_count
        self._work.charge(3)
        if (type(key) is not tuple or len(key) != 2 or type(key[1]) is not int
                or key[1] < 0 or key[0] not in ("le", "lo", "hi")):
            raise DomainError("invalid nonnegative slack identity")
        kind, index = key
        if kind == "le":
            if index >= row_count or self.predicates[index].relation != "le":
                raise DomainError("slack does not name a retained LE")
            row = self.predicates[index]
            self._validate_form(row.form, width)
            return self._linear(row.rhs, ((-1, row.form),))
        if index >= width:
            raise DomainError("bound slack is outside its owned prefix")
        factor = self.factors[index]
        self._work.charge(3, 3)
        return (Form(-factor.lower, ((index, Q(1)),)) if kind == "lo"
                else Form(factor.upper, ((index, Q(-1)),)))

    def _eq_atom(self, index, width, row_count):
        self._work.charge(2)
        if (type(index) is not int or not 0 <= index < row_count
                or self.predicates[index].relation != "eq"):
            raise DomainError("equality identity is outside its owned prefix")
        row = self.predicates[index]
        self._validate_form(row.form, width)
        return self._linear(row.rhs, ((-1, row.form),))

    def _weight(self, mapping, key, coefficient):
        coefficient = _q(coefficient)
        self._work.charge(4, 3 if key not in mapping and coefficient else 0)
        value = _q(mapping.get(key, Q(0)) + coefficient)
        if value:
            mapping[key] = value
        else:
            mapping.pop(key, None)

    def verify_proof(self, proof):
        """Check ownership and the complete rational identity; never import a cap."""
        with self._work.branch():
            if not isinstance(proof, Proof) or proof.frame is not self._frame:
                raise DomainError("foreign or unsupported proof")
            if (type(proof.identities) is not tuple or type(proof.predicates) is not tuple
                    or type(proof.le_weights) is not tuple or type(proof.eq_weights) is not tuple):
                raise DomainError("proof prefixes and sparse coefficients must be tuples")
            width, rows = len(proof.identities), len(proof.predicates)
            self._work.charge(5 + width + rows)
            if (width > len(self.factors) or rows > len(self.predicates)
                    or any(identity is not self.factors[i].identity
                           for i, identity in enumerate(proof.identities))
                    or any(row is not self.predicates[i]
                           for i, row in enumerate(proof.predicates))):
                raise DomainError("proof does not belong to this complete parent prefix")
            self._validate_form(proof.subject, width)
            bound = _q(proof.bound)
            pieces, previous = [], None
            for item in proof.le_weights:
                if type(item) is not tuple or len(item) != 2:
                    raise DomainError("malformed sparse LE proof")
                key, value = item
                atom = self._atom(key, width, rows)
                value = _q(value)
                if value <= 0 or (previous is not None and key <= previous):
                    raise DomainError("LE proof weights must be positive, sorted and unique")
                previous = key
                pieces.append((value, atom))
            previous = -1
            for item in proof.eq_weights:
                if type(item) is not tuple or len(item) != 2:
                    raise DomainError("malformed sparse EQ proof")
                index, value = item
                atom = self._eq_atom(index, width, rows)
                value = _q(value)
                if not value or index <= previous:
                    raise DomainError("EQ proof weights must be nonzero, sorted and unique")
                previous = index
                pieces.append((value, atom))
            self._work.charge(2 * len(pieces), 2 * len(pieces))
            actual = self._linear(pieces=pieces)
            expected = self._linear(bound, ((-1, proof.subject),))
            self._work.charge(2 + len(actual.terms) + len(expected.terms))
            if actual != expected:
                raise DomainError("nonnegative certificate identity does not hold")
            return True

    def _proof(self, bound, subject, le_weights, eq_weights):
        le = tuple(sorted((key, value) for key, value in le_weights.items() if value))
        eq = tuple(sorted((key, value) for key, value in eq_weights.items() if value))
        count = len(le) + len(eq)
        self._work.charge(count * max(1, count.bit_length()))
        proof = Proof(_q(bound), subject, le, eq, self._frame,
                      tuple(f.identity for f in self.factors), self.predicates)
        entries = self._proof_entries(proof)
        self._work.charge(entries, entries)
        self.verify_proof(proof)
        return proof

    def alias(self, view, names):
        with self._work.branch():
            forms, names = self._align(view), tuple(names)
            self._names(names, tuple(f.name for f in self.factors), len(forms))
            ranges = [self.bounds(self._view((form,))) for form in forms]
            factors, predicates = list(self.factors), list(self.predicates)
            events, aliases, output = list(self._events), list(self._alias_rows), []
            for name, form, (lower, upper) in zip(names, forms, ranges):
                index = len(factors)
                factors.append(_base.Factor(_base.Identity(name), "alias", lower, upper, form))
                output.append(Form(0, ((index, Q(1)),)))
                aliases.append((index, len(predicates)))
                predicates.append(Predicate(self._linear(pieces=((1, output[-1]), (-1, form))), "eq"))
                events.append(("alias", index))
                self._work.charge(6, 6)
            child = self._spawn(factors, predicates, self.banks, events, alias_rows=aliases)
            return child, child._view(output)

    def _box_upper(self, form):
        self._validate_form(form)
        value = form.constant
        for index, coefficient in form.terms:
            factor = self.factors[index]
            endpoint = factor.upper if coefficient >= 0 else factor.lower
            value = _q(value + _q(coefficient * endpoint))
            self._work.charge(4)
        return value

    def _closure(self, positions, gates):
        one, two = (gates[i] for i in positions)
        differences = (self._linear(pieces=((1, one.input), (-Q(1, 2), two.input))),
                       self._linear(pieces=((1, two.input), (-Q(1, 2), one.input))))
        proofs = tuple(self.support_proof(self._view((form,)), (1,)) for form in differences)
        for proof in proofs:
            self.verify_proof(proof)
        caps = tuple(max(Q(0), proof.bound) for proof in proofs)
        common, common_form, cap, tau, scale, lower = (), Form(), Q(0), Q(1), Form(1), Q(1)
        scale_proofs = ()
        if min(proof.bound for proof in proofs) > 0:
            first, second = (dict(proof.le_weights) for proof in proofs)
            self._work.charge(2 * (len(first) + len(second)),
                              3 * (len(first) + len(second)))
            shared = []
            for key, coefficient in first.items():
                self._work.charge(6)
                value = min(_q(coefficient / caps[0]),
                            _q(second.get(key, Q(0)) / caps[1]))
                if value:
                    shared.append((key, value))
            common = tuple(sorted(shared))
            self._work.charge(len(common) * max(1, len(common).bit_length()), 3 * len(common))
            common_form = self._linear(pieces=((value, self._atom(key)) for key, value in common))
            cap = max(Q(0), self._box_upper(common_form))
            tau = min(Q(1), _q(1 / _q(2 * cap))) if cap else Q(1)
            scale = self._linear(1, ((-tau, common_form),))
            lower = _q(1 - _q(tau * cap))
            if not Q(1, 2) <= lower <= 1:
                raise DomainError("uncertified positive source scale")
            checked = []
            for proof, ci in zip(proofs, caps):
                remaining = dict(proof.le_weights)
                self._work.charge(2 * len(remaining), 3 * len(remaining))
                for key, value in common:
                    self._weight(remaining, key, -_q(_q(ci * tau) * value))
                if any(value < 0 for value in remaining.values()):
                    raise DomainError("common slack exceeds a parent certificate")
                subject = self._linear(pieces=((1, proof.subject), (-ci, scale)))
                self._work.charge(2 * len(proof.eq_weights), 2 * len(proof.eq_weights))
                checked.append(self._proof(0, subject, remaining, dict(proof.eq_weights)))
            scale_proofs = tuple(checked)
        qs = tuple(Form(0, ((gate.q_index, Q(1)),)) for gate in (one, two))
        betas = tuple(Form(Q(1, 2), ((gate.phase_index, Q(1, 2)),)) for gate in (one, two))
        rows, constant_h, constant_m, source_h, source_m = [], [], [], [], []
        for i, gate in enumerate((one, two)):
            ci, cj = caps[i], caps[1 - i]
            upper = _q(_q(ci + _q(cj / 2)) / Q(3, 4))
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
                if denominator <= 0:
                    raise DomainError("closure birth requires crossing gates")
                hs.append(self._linear(pieces=((_q(_q(ci * lo) / denominator), gate.input),
                                               (_q(_q(ci * ell) / denominator), sc))))
                ms.append(_q(ell / _q(2 * denominator)))
                self._work.charge(16)
        pair = _base.Pair(positions, caps, scale, lower, tuple(rows), tuple(constant_h),
                          tuple(constant_m), tuple(source_h), tuple(source_m))
        return ClosurePair(pair, (), proofs, scale_proofs, common, common_form, cap, tau)

    def relu_bank(self, view, phase_names):
        with self._work.branch():
            forms, phase_names = self._align(view), tuple(phase_names)
            self._names(phase_names, tuple(f.name for f in self.factors), len(forms))
            q_names = tuple(name + ":q" for name in phase_names)
            self._names(phase_names + q_names, tuple(f.name for f in self.factors), 2 * len(forms))
            ranges = [self.bounds(self._view((form,))) for form in forms]
            factors, predicates, gates, outputs, gate_rows = list(self.factors), list(self.predicates), [], [], []
            for name, q_name, form, (bound_lower, bound_upper) in zip(phase_names, q_names, forms, ranges):
                lower, upper = min(Q(0), bound_lower), max(Q(0), bound_upper)
                phase_index, q_index = len(factors), len(factors) + 1
                factors.extend((_base.Factor(_base.Identity(name), "phase", Q(-1), Q(1)),
                                _base.Factor(_base.Identity(q_name), "relu", Q(0), upper)))
                beta = Form(Q(1, 2), ((phase_index, Q(1, 2)),))
                q = Form(0, ((q_index, Q(1)),))
                outputs.append(q)
                start = len(predicates)
                predicates.extend((
                    Predicate(self._linear(pieces=((-1, q),))),
                    Predicate(self._linear(pieces=((1, form), (-1, q)))),
                    Predicate(self._linear(pieces=((1, q), (-upper, beta)))),
                    Predicate(self._linear(pieces=((1, q), (-1, form), (-lower, beta))), rhs=-lower),
                ))
                gate_rows.append(tuple(range(start, start + 4)))
                stable = -1 if bound_upper <= 0 else (1 if bound_lower >= 0 else 0)
                triangle = (Form() if stable == -1 else form if stable == 1 else
                            self._linear(_q(-_q(upper * lower) / _q(upper - lower)),
                                         ((_q(upper / _q(upper - lower)), form),)))
                gates.append(_base.Gate(form, phase_index, q_index, lower, upper, triangle, stable))
                self._work.charge(10, 6)
            pairs, old_pair_rows, closures = [], [], []
            for position in range(0, len(gates) - 1, 2):
                if gates[position].stable == gates[position + 1].stable == 0:
                    positions = (position, position + 1)
                    pair = self._pair(positions, gates)
                    if pair is not None:
                        start = len(predicates)
                        predicates.extend(pair.rows)
                        pairs.append(pair)
                        old_pair_rows.append(tuple(range(start, start + len(pair.rows))))
                    # Both cap proofs refer only to self, never the newborn bank.
                    closures.append(self._closure(positions, gates))
            index = {}
            for row_index, row in enumerate(predicates):
                self._work.charge(3 + 2 * len(row.form.terms), 2)
                if row not in index:
                    index[row] = row_index
            installed = []
            for closure in closures:
                row_indices = []
                for row in closure.pair.rows:
                    self._work.charge(4 + 2 * len(row.form.terms), 1)
                    if row not in index:
                        index[row] = len(predicates)
                        predicates.append(row)
                        self._work.charge(2, 2)
                    row_indices.append(index[row])
                installed.append(ClosurePair(closure.pair, tuple(row_indices), closure.parent_proofs,
                                             closure.scale_proofs, closure.common_weights,
                                             closure.common_form, closure.common_box_upper, closure.tau))
            bank = _base.Bank(len(self.factors), tuple(gates), tuple(pairs))
            meta = BankProof(bank, tuple(gate_rows), tuple(old_pair_rows), tuple(installed))
            child = self._spawn(factors, predicates, self.banks + (bank,),
                                self._events + (("bank", bank),),
                                proof_banks=self._proof_banks + (meta,))
            return child, child._view(outputs)

    def _pair_step(self, pair, rows, gates, gate_rows, weights, source, le):
        i, j = pair.positions
        hs = pair.source_h if source else pair.constant_h
        a, b = pair.source_m if source else pair.constant_m
        denominator = _q(1 - _q(a * b))
        if denominator <= 0:
            raise DomainError("invalid owned pair contraction")
        first = max(Q(0), weights[i], _q(_q(weights[i] + _q(b * weights[j])) / denominator))
        second = max(Q(0), weights[j], _q(_q(weights[j] + _q(a * weights[i])) / denominator))
        residuals = (_q(_q(first - _q(b * second)) - weights[i]),
                     _q(_q(second - _q(a * first)) - weights[j]))
        if min(residuals) < 0:
            raise DomainError("pair multiplier is not a nonnegative certificate")
        for side, position, value, residual in ((0, i, first, residuals[0]),
                                                 (1, j, second, residuals[1])):
            gate, ci = gates[position], pair.caps[side]
            lo = pair.lower if source else Q(1)
            ell = -gate.lower
            divisor = _q(_q(ci * lo) + ell)
            if divisor <= 0:
                raise DomainError("invalid h proof denominator")
            self._weight(le, ("le", gate_rows[position][3]),
                         _q(value * _q(_q(ci * lo) / divisor)))
            self._weight(le, ("le", rows[5 * side + (4 if source else 3)]),
                         _q(value * _q(ell / divisor)))
            self._weight(le, ("le", gate_rows[position][0]), residual)
        self._work.charge(36)
        return ((first, hs[0]), (second, hs[1]))

    def _certificate_proof(self, subject, mode):
        form, le, eq = subject, {}, {}
        aliases = dict(self._alias_rows)
        self._work.charge(2 * len(aliases), 2 * len(aliases))
        bank_index = len(self.banks) - 1
        for kind, event in reversed(self._events):
            weights = dict(form.terms)
            self._work.charge(1 + 2 * len(form.terms), 2 * len(form.terms))
            if kind == "alias":
                if event not in aliases:
                    raise DomainError("alias has no retained producer equality")
                weight = weights.pop(event, Q(0))
                # e=rhs-lhs=producer-alias, hence the substitution adds +weight*e.
                self._weight(eq, aliases[event], weight)
                form = self._linear(pieces=((1, Form(form.constant, tuple(sorted(weights.items())))),
                                           (weight, self.factors[event].producer)))
                continue
            if kind != "bank" or bank_index < 0 or event is not self.banks[bank_index]:
                raise DomainError("owned event order does not match retained banks")
            bank, meta, pieces = event, self._proof_banks[bank_index], []
            bank_index -= 1
            local = [weights.pop(gate.q_index, Q(0)) for gate in bank.gates]
            self._work.charge(len(local), len(local))
            handled = set()
            if mode in (1, 2):
                for pair, rows in zip(bank.pairs, meta.old_pair_rows):
                    pieces.extend(self._pair_step(pair, rows, bank.gates, meta.gate_rows,
                                                  local, mode == 2, le))
                    handled.update(pair.positions)
            elif mode == 3:
                for closure in meta.closure_pairs:
                    pieces.extend(self._pair_step(closure.pair, closure.row_indices, bank.gates,
                                                  meta.gate_rows, local, True, le))
                    handled.update(closure.pair.positions)
            for i, gate in enumerate(bank.gates):
                if i in handled:
                    continue
                weight, rows = local[i], meta.gate_rows[i]
                if gate.stable == 1:
                    pieces.append((weight, gate.input))
                    self._weight(le, ("le", rows[3] if weight >= 0 else rows[1]), abs(weight))
                elif gate.stable == -1:
                    self._weight(le, ("le", rows[2] if weight >= 0 else rows[0]), abs(weight))
                elif weight >= 0:
                    pieces.append((weight, gate.triangle))
                    ell, divisor = -gate.lower, _q(gate.upper - gate.lower)
                    self._weight(le, ("le", rows[2]), _q(weight * _q(ell / divisor)))
                    self._weight(le, ("le", rows[3]), _q(weight * _q(gate.upper / divisor)))
                else:
                    self._weight(le, ("le", rows[0]), -weight)
            rest = Form(form.constant, tuple(sorted(weights.items())))
            form = self._linear(pieces=((1, rest), *pieces))
        if bank_index != -1:
            raise DomainError("unprocessed owned bank")
        upper = form.constant
        for index, coefficient in form.terms:
            factor = self.factors[index]
            if factor.kind in ("alias", "relu"):
                raise DomainError("unprocessed birth in support proof")
            endpoint = factor.upper if coefficient >= 0 else factor.lower
            upper = _q(upper + _q(coefficient * endpoint))
            self._weight(le, ("hi" if coefficient >= 0 else "lo", index), abs(coefficient))
            self._work.charge(3)
        return self._proof(upper, subject, le, eq)

    def support_proofs(self, view, direction):
        with self._work.branch():
            forms, direction = self._align(view), tuple(map(_q, direction))
            if len(forms) != len(direction):
                raise DomainError("support direction shape mismatch")
            combined = self._linear(pieces=zip(direction, forms))
            return tuple(self._certificate_proof(combined, mode) for mode in range(4))

    def support_proof(self, view, direction):
        proofs = self.support_proofs(view, direction)
        self._work.charge(4)
        return min(proofs, key=lambda proof: proof.bound)

    def support_certificates(self, view, direction):
        return tuple(proof.bound for proof in self.support_proofs(view, direction))

    def support(self, view, direction):
        return self.support_proof(view, direction).bound

    def cost_report(self):
        with self._work.branch():
            report = super().cost_report()
            closures = tuple(closure for meta in self._proof_banks for closure in meta.closure_pairs)
            proofs = tuple(proof for closure in closures
                           for proof in closure.parent_proofs + closure.scale_proofs)
            self._work.charge(4 + len(closures) + sum(self._proof_entries(proof) for proof in proofs),
                              len(closures) + len(proofs))
            report.update(
                fixed_query_certificates=4,
                closure_pairs=len(closures),
                closure_row_references=sum(len(closure.row_indices) for closure in closures),
                stored_parent_proofs=sum(len(closure.parent_proofs) for closure in closures),
                stored_scale_proofs=sum(len(closure.scale_proofs) for closure in closures),
                stored_proof_le_terms=sum(len(proof.le_weights) for proof in proofs),
                stored_proof_eq_terms=sum(len(proof.eq_weights) for proof in proofs),
                common_slack_terms=sum(len(closure.common_weights) for closure in closures),
                alias_equation_references=len(self._alias_rows),
                work_used=self._work.used, branch_work_used=self._work.branch_used,
                entries=self._work.entries, new_set_class_qualified=False,
                actual_model_qualified=False, gpu_qualified=False,
            )
            return report
