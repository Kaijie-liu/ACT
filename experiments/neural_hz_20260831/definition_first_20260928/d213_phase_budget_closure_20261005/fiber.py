"""Default-off D212 phase budgets inside the complete owned D209 domain.

All births and queries use five fixed certificates.  No model/solver imports,
external caps, new product factors, or production qualification are provided.
"""

from dataclasses import dataclass, replace

from experiments.neural_hz_20260831.definition_first_20260928.d209_owned_slack_closure_20261005 import fiber as _d209


Q, Form, Predicate = _d209.Q, _d209.Form, _d209.Predicate
Readout, Proof = _d209.Readout, _d209.Proof
DomainError, BudgetError = _d209.DomainError, _d209.BudgetError
MAX_WORK, MAX_BRANCH_WORK = _d209.MAX_WORK, _d209.MAX_BRANCH_WORK
MAX_ENTRIES, MAX_BITS = _d209.MAX_ENTRIES, _d209.MAX_BITS
_q = _d209._q


@dataclass(frozen=True)
class PhasePair:
    positions: tuple
    caps: tuple
    parent_proofs: tuple
    residual_proofs: tuple
    common_weights: tuple
    common_form: Form
    box_upper: Q
    box_proof: object
    lower_proofs: tuple
    intrinsic_proofs: tuple
    T_int: Q
    T: Q
    selected_T_proof: object
    rows: tuple
    row_indices: tuple
    h: tuple
    m: tuple


@dataclass(frozen=True)
class PhaseBank:
    bank: object
    pairs: tuple


@dataclass(frozen=True)
class Fiber(_d209.Fiber):
    _phase_banks: tuple = ()

    def __post_init__(self):
        super().__post_init__()
        if (len(self._phase_banks) != len(self.banks)
                or any(meta.bank is not bank for meta, bank in
                       zip(self._phase_banks, self.banks))):
            raise DomainError("owned phase-bank metadata was lost")
        self._work.charge(1 + len(self._phase_banks))

    def _retain_state(self):
        super()._retain_state()
        entries = len(self._phase_banks)
        for meta in self._phase_banks:
            entries += 2 + len(meta.pairs)
            for pair in meta.pairs:
                entries += (24 + 3 * len(pair.common_weights)
                            + 2 * len(pair.common_form.terms) + len(pair.row_indices))
                # Equal installed predicates may reuse indices, but these
                # separately retained row objects and forms remain payable.
                entries += len(pair.rows) + sum(3 + 2 * len(row.form.terms)
                                                for row in pair.rows)
                entries += sum(1 + 2 * len(form.terms) for form in pair.h)
                proofs = (pair.parent_proofs + pair.residual_proofs
                          + pair.lower_proofs + pair.intrinsic_proofs)
                entries += sum(self._proof_entries(proof) for proof in proofs)
                if pair.box_proof is not None:
                    entries += self._proof_entries(pair.box_proof)
                entries += int(pair.selected_T_proof is not None)
        self._work.charge(entries, entries)

    def _box_proof(self, subject):
        upper = self._box_upper(subject)
        le = {}
        for index, coefficient in subject.terms:
            self._weight(le, ("hi" if coefficient >= 0 else "lo", index), abs(coefficient))
        return self._proof(upper, subject, le, {})

    def _combine_proofs(self, bound, subject, pieces):
        pieces = tuple(pieces)
        self._work.charge(len(pieces), 2 * len(pieces))
        le, eq = {}, {}
        for multiplier, proof in pieces:
            multiplier = _q(multiplier)
            if multiplier < 0:
                raise DomainError("negative multiplier on a nonnegative premise")
            self.verify_proof(proof)
            for key, value in proof.le_weights:
                self._weight(le, key, _q(multiplier * value))
            for key, value in proof.eq_weights:
                self._weight(eq, key, _q(multiplier * value))
        return self._proof(bound, subject, le, eq)

    def _phase_pair(self, closure, gates):
        positions, proofs = closure.pair.positions, closure.parent_proofs
        if len(positions) != 2 or len(proofs) != 2:
            raise DomainError("phase birth requires one complete owned pair")
        chosen = tuple(gates[index] for index in positions)
        if any(gate.stable != 0 or gate.lower >= 0 or gate.upper <= 0 for gate in chosen):
            raise DomainError("phase budgets require crossing gates")
        for proof in proofs:
            self.verify_proof(proof)
        differences = (self._linear(pieces=((1, chosen[0].input), (-Q(1, 2), chosen[1].input))),
                       self._linear(pieces=((1, chosen[1].input), (-Q(1, 2), chosen[0].input))))
        if any(proof.subject != form for proof, form in zip(proofs, differences)):
            raise DomainError("phase cap proof names a different parent difference")
        caps = tuple(max(Q(0), proof.bound) for proof in proofs)
        if caps != closure.pair.caps:
            raise DomainError("phase and retained closure caps differ")
        self._work.charge(12, 8)
        common, common_form, box_upper = (), Form(), Q(0)
        residuals, lower_proofs, intrinsic = (), (), ()
        box_proof = selected = None
        T_int = T = Q(1)
        uppers = tuple(_q(_q(caps[i] + _q(caps[1-i] / 2)) / Q(3, 4)) for i in range(2))
        if min(proof.bound for proof in proofs) > 0:
            maps = tuple(dict(proof.le_weights) for proof in proofs)
            count = sum(len(mapping) for mapping in maps)
            self._work.charge(2 * count, 3 * count)
            shared = []
            for key, coefficient in maps[0].items():
                value = min(_q(coefficient / caps[0]),
                            _q(maps[1].get(key, Q(0)) / caps[1]))
                self._work.charge(6)
                if value:
                    shared.append((key, value))
            common = tuple(sorted(shared))
            self._work.charge(len(common) * max(1, len(common).bit_length()), 3 * len(common))
            common_form = self._linear(pieces=((value, self._atom(key)) for key, value in common))
            if common != closure.common_weights or common_form != closure.common_form:
                raise DomainError("unscaled common slack differs from the owned cap proofs")
            remaining_proofs = []
            for proof, ci in zip(proofs, caps):
                le, eq = dict(proof.le_weights), dict(proof.eq_weights)
                self._work.charge(2 * (len(le) + len(eq)), 3 * len(le) + 2 * len(eq))
                for key, value in common:
                    self._weight(le, key, -_q(ci * value))
                if any(value < 0 for value in le.values()):
                    raise DomainError("negative unscaled residual coefficient")
                subject = self._linear(-ci, ((1, proof.subject), (ci, common_form)))
                remaining_proofs.append(self._proof(0, subject, le, eq))
            residuals = tuple(remaining_proofs)
            box_proof = self._box_proof(common_form)
            box_upper = max(Q(0), box_proof.bound)
            if box_upper != closure.common_box_upper:
                raise DomainError("complete parent box bound differs")
            lower_proofs = tuple(self.support_proof(
                self._view((self._linear(pieces=((-1, gate.input),)),)), (1,)) for gate in chosen)
            intrinsic_list = []
            for i, (gate, lower_proof, upper) in enumerate(zip(chosen, lower_proofs, uppers)):
                self.verify_proof(lower_proof)
                if lower_proof.bound != -gate.lower:
                    raise DomainError("recomputed five-mode lower bound differs from birth")
                ti = _q(1 + _q(lower_proof.bound / upper))
                multiplier = _q(1 / _q(Q(3, 4) * upper))
                intrinsic_list.append(self._combine_proofs(ti, common_form, (
                    (multiplier, residuals[i]),
                    (_q(multiplier / 2), residuals[1-i]),
                    (_q(1 / upper), lower_proof),
                )))
            intrinsic = tuple(intrinsic_list)
            T_int = min(proof.bound for proof in intrinsic)
            T = max(Q(1), min(box_upper, T_int))
            selected = min((box_proof, *intrinsic), key=lambda proof: proof.bound)
            self._work.charge(8)
            self.verify_proof(selected)
            if selected.subject != common_form or selected.bound > T:
                raise DomainError("phase budget has no authenticated parent upper proof")
        qs = tuple(Form(0, ((gate.q_index, Q(1)),)) for gate in chosen)
        betas = tuple(Form(Q(1, 2), ((gate.phase_index, Q(1, 2)),)) for gate in chosen)
        self._work.charge(16, 16)
        rows, hs, ms = [], [], []
        budget = self._linear(T, ((-1, common_form),))
        for i, gate in enumerate(chosen):
            ci, qi, qj, beta = caps[i], qs[i], qs[1-i], betas[i]
            projected = self._linear(pieces=((1, budget), (_q(1 - T), beta)))
            rows.extend((
                Predicate(self._linear(pieces=((1, qi), (-uppers[i], projected)))),
                Predicate(self._linear(pieces=((1, qi), (-Q(1, 2), qj), (-ci, projected)))),
            ))
            divisor = _q(gate.upper + _q(ci * _q(T - 1)))
            if divisor <= 0:
                raise DomainError("phase consumption denominator is not positive")
            hs.append(self._linear(pieces=((_q(_q(ci * gate.upper) / divisor), budget),)))
            ms.append(_q(gate.upper / _q(2 * divisor)))
            self._work.charge(20, 6)
        self._work.charge(23, 23)
        return PhasePair(positions, caps, proofs, residuals, common, common_form,
                         box_upper, box_proof, lower_proofs, intrinsic, T_int, T,
                         selected, tuple(rows), (), tuple(hs), tuple(ms))

    def _spawn(self, factors, predicates, banks, events, *, proof_banks=None,
               alias_rows=None):
        factors, predicates, banks, events = tuple(factors), list(predicates), tuple(banks), tuple(events)
        proof_banks = self._proof_banks if proof_banks is None else tuple(proof_banks)
        alias_rows = self._alias_rows if alias_rows is None else tuple(alias_rows)
        self._work.charge(5 + len(banks) + len(predicates), len(predicates))
        if (len(banks) not in (len(self.banks), len(self.banks) + 1)
                or any(a is not b for a, b in zip(self.banks, banks))):
            raise DomainError("phase state is not an append-only bank extension")
        phase_banks = self._phase_banks
        if len(banks) > len(self.banks):
            bank = banks[-1]
            if len(proof_banks) != len(banks) or proof_banks[-1].bank is not bank:
                raise DomainError("new bank lost its complete old proof metadata")
            # self remains the complete pre-birth parent for every query here.
            pending = tuple(self._phase_pair(closure, bank.gates)
                            for closure in proof_banks[-1].closure_pairs)
            self._work.charge(len(pending), len(pending))
            indices = {}
            for index, row in enumerate(predicates):
                self._work.charge(3 + 2 * len(row.form.terms), 2)
                if row not in indices:
                    indices[row] = index
            installed = []
            for pair in pending:
                references = []
                for row in pair.rows:
                    self._work.charge(4 + 2 * len(row.form.terms), 1)
                    if row not in indices:
                        indices[row] = len(predicates)
                        predicates.append(row)
                        self._work.charge(2, 2)
                    references.append(indices[row])
                # replace retains a second shallow metadata object until this
                # birth returns; references and that temporary object are paid.
                self._work.charge(18 + len(references), 18 + len(references))
                installed.append(replace(pair, row_indices=tuple(references)))
            phase_banks += (PhaseBank(bank, tuple(installed)),)
            self._work.charge(2 + len(installed) + len(phase_banks),
                              2 + len(installed) + len(phase_banks))
        child = Fiber(self._frame, factors, tuple(predicates), banks, events,
                      _authority=_d209._base._OWNED_CONSTRUCTION,
                      _proof_banks=proof_banks, _alias_rows=alias_rows,
                      _phase_banks=phase_banks)
        child._retain_state()
        return child

    def _phase_step(self, pair, gates, gate_rows, weights, le):
        i, j = pair.positions
        a, b = pair.m
        denominator = _q(1 - _q(a * b))
        if denominator <= 0:
            raise DomainError("non-contractive phase budget")
        first = max(Q(0), weights[i], _q(_q(weights[i] + _q(b * weights[j])) / denominator))
        second = max(Q(0), weights[j], _q(_q(weights[j] + _q(a * weights[i])) / denominator))
        residuals = (_q(_q(first - _q(b * second)) - weights[i]),
                     _q(_q(second - _q(a * first)) - weights[j]))
        if min(residuals) < 0:
            raise DomainError("phase multiplier is not a nonnegative proof")
        for side, position, value, residual in ((0, i, first, residuals[0]),
                                                 (1, j, second, residuals[1])):
            upper, ci = gates[position].upper, pair.caps[side]
            correction = _q(ci * _q(pair.T - 1))
            divisor = _q(upper + correction)
            if divisor <= 0 or correction < 0:
                raise DomainError("invalid on/off consumption weights")
            self._weight(le, ("le", pair.row_indices[2 * side + 1]),
                         _q(value * _q(upper / divisor)))
            # The D212 identity uses graph row 2 (u*beta-q), NOT row 3.
            self._weight(le, ("le", gate_rows[position][2]),
                         _q(value * _q(correction / divisor)))
            self._weight(le, ("le", gate_rows[position][0]), residual)
        self._work.charge(36)
        return ((first, pair.h[0]), (second, pair.h[1]))

    def _certificate_proof(self, subject, mode):
        if mode in (0, 1, 2, 3):
            return super()._certificate_proof(subject, mode)
        if mode != 4:
            raise DomainError("only the five registered certificate modes exist")
        form, le, eq = subject, {}, {}
        aliases = dict(self._alias_rows)
        self._work.charge(2 * len(aliases), 2 * len(aliases))
        bank_index = len(self.banks) - 1
        for kind, event in reversed(self._events):
            weights = dict(form.terms)
            self._work.charge(1 + 2 * len(form.terms), 2 * len(form.terms))
            if kind == "alias":
                if event not in aliases:
                    raise DomainError("alias has no owned equality")
                weight = weights.pop(event, Q(0))
                self._weight(eq, aliases[event], weight)
                form = self._linear(pieces=((1, Form(form.constant, tuple(sorted(weights.items())))),
                                           (weight, self.factors[event].producer)))
                continue
            if kind != "bank" or bank_index < 0 or event is not self.banks[bank_index]:
                raise DomainError("phase query lost the complete event order")
            bank, old, phase = event, self._proof_banks[bank_index], self._phase_banks[bank_index]
            bank_index -= 1
            local = [weights.pop(gate.q_index, Q(0)) for gate in bank.gates]
            self._work.charge(len(local), len(local))
            pieces, handled = [], set()
            for pair in phase.pairs:
                pieces.extend(self._phase_step(pair, bank.gates, old.gate_rows, local, le))
                handled.update(pair.positions)
            for i, gate in enumerate(bank.gates):
                if i in handled:
                    continue
                weight, rows = local[i], old.gate_rows[i]
                if gate.stable == 1:
                    pieces.append((weight, gate.input))
                    self._weight(le, ("le", rows[3] if weight >= 0 else rows[1]), abs(weight))
                elif gate.stable == -1:
                    self._weight(le, ("le", rows[2] if weight >= 0 else rows[0]), abs(weight))
                elif weight >= 0:
                    pieces.append((weight, gate.triangle))
                    divisor = _q(gate.upper - gate.lower)
                    self._weight(le, ("le", rows[2]), _q(weight * _q(-gate.lower / divisor)))
                    self._weight(le, ("le", rows[3]), _q(weight * _q(gate.upper / divisor)))
                else:
                    self._weight(le, ("le", rows[0]), -weight)
            rest = Form(form.constant, tuple(sorted(weights.items())))
            form = self._linear(pieces=((1, rest), *pieces))
        if bank_index != -1:
            raise DomainError("unprocessed phase bank")
        upper = form.constant
        for index, coefficient in form.terms:
            factor = self.factors[index]
            if factor.kind in ("alias", "relu"):
                raise DomainError("unprocessed birth in phase support")
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
            return tuple(self._certificate_proof(combined, mode) for mode in range(5))

    def support_proof(self, view, direction):
        proofs = self.support_proofs(view, direction)
        self._work.charge(5)
        return min(proofs, key=lambda proof: proof.bound)

    def cost_report(self):
        with self._work.branch():
            report = super().cost_report()
            pairs = tuple(pair for meta in self._phase_banks for pair in meta.pairs)
            proofs = tuple(proof for pair in pairs for proof in
                           (pair.parent_proofs + pair.residual_proofs
                            + pair.lower_proofs + pair.intrinsic_proofs))
            self._work.charge(5 + len(pairs) + sum(self._proof_entries(proof) for proof in proofs),
                              len(pairs) + len(proofs))
            report.update(
                fixed_query_certificates=5, phase_pairs=len(pairs),
                phase_row_references=sum(len(pair.row_indices) for pair in pairs),
                phase_parent_proofs=sum(len(pair.parent_proofs) for pair in pairs),
                phase_residual_proofs=sum(len(pair.residual_proofs) for pair in pairs),
                phase_lower_proofs=sum(len(pair.lower_proofs) for pair in pairs),
                phase_intrinsic_proofs=sum(len(pair.intrinsic_proofs) for pair in pairs),
                phase_box_proofs=sum(pair.box_proof is not None for pair in pairs),
                phase_common_terms=sum(len(pair.common_weights) for pair in pairs),
                phase_product_factors=0, work_used=self._work.used,
                branch_work_used=self._work.branch_used, entries=self._work.entries,
            )
            return report
