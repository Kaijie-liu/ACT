"""Default-off, six-certificate consumption of owned phase relations.

The complete D213 element and its first five queries remain present.  A sixth
query consumes an admissible source-feedback parameter fixed by the combined
readout coefficients.  No solver, external cap, or new free factor is used.
"""

from dataclasses import dataclass, replace

from experiments.neural_hz_20260831.definition_first_20260928.d213_phase_budget_closure_20261005 import fiber as _d213


Q, Form, Predicate = _d213.Q, _d213.Form, _d213.Predicate
Readout, Proof = _d213.Readout, _d213.Proof
PhasePair, PhaseBank = _d213.PhasePair, _d213.PhaseBank
DomainError, BudgetError = _d213.DomainError, _d213.BudgetError
MAX_WORK, MAX_BRANCH_WORK = _d213.MAX_WORK, _d213.MAX_BRANCH_WORK
MAX_ENTRIES, MAX_BITS = _d213.MAX_ENTRIES, _d213.MAX_BITS
_q = _d213._q


@dataclass(frozen=True)
class Fiber(_d213.Fiber):
    def _spawn(self, factors, predicates, banks, events, *, proof_banks=None,
               alias_rows=None):
        # Preserve the complete D213 birth and its charges; only the resulting
        # owned type changes, so later bounds and cap births also use six modes.
        factors, predicates, banks, events = tuple(factors), list(predicates), tuple(banks), tuple(events)
        proof_banks = self._proof_banks if proof_banks is None else tuple(proof_banks)
        alias_rows = self._alias_rows if alias_rows is None else tuple(alias_rows)
        self._work.charge(5 + len(banks) + len(predicates), len(predicates))
        if (len(banks) not in (len(self.banks), len(self.banks) + 1)
                or any(a is not b for a, b in zip(self.banks, banks))):
            raise DomainError("parametric state is not an append-only bank extension")
        phase_banks = self._phase_banks
        if len(banks) > len(self.banks):
            bank = banks[-1]
            if len(proof_banks) != len(banks) or proof_banks[-1].bank is not bank:
                raise DomainError("new bank lost its complete old proof metadata")
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
                self._work.charge(18 + len(references), 18 + len(references))
                installed.append(replace(pair, row_indices=tuple(references)))
            phase_banks += (PhaseBank(bank, tuple(installed)),)
            self._work.charge(2 + len(installed) + len(phase_banks),
                              2 + len(installed) + len(phase_banks))
        child = Fiber(self._frame, factors, tuple(predicates), banks, events,
                      _authority=_d213._d209._base._OWNED_CONSTRUCTION,
                      _proof_banks=proof_banks, _alias_rows=alias_rows,
                      _phase_banks=phase_banks)
        child._retain_state()
        return child

    def _parametric_step(self, closure, gates, gate_rows, weights, le):
        pair = closure.pair
        first, second = pair.positions
        self._work.charge(8, 8)
        for side, position, other in ((0, first, second), (1, second, first)):
            wi, wj = weights[position], weights[other]
            self._work.charge(4)
            if not (wi > 0 and wj < 0):
                continue
            gate, cap, lower = gates[position], pair.caps[side], pair.lower
            ell = _q(-gate.lower)
            if cap < 0 or lower <= 0 or ell <= 0 or gate.stable != 0:
                raise DomainError("parametric relation has no crossing source contract")
            cap_lower = _q(cap * lower)
            denominator = _q(cap_lower + ell)
            r_min = _q(ell / denominator)
            # a=1/2 is part of the same fixed pair relation, not a searched slope.
            r = _q(-wj / _q(wi / 2))
            self._work.charge(18)
            if not r_min <= r <= 1:
                continue
            one_minus_r = _q(1 - r)
            correction = _q(_q(r * cap_lower) - _q(one_minus_r * ell))
            if correction < 0:
                raise DomainError("negative phase-bound multiplier")
            self._weight(le, ("le", gate_rows[position][3]),
                         _q(wi * one_minus_r))
            self._weight(le, ("le", closure.row_indices[5 * side + 4]),
                         _q(wi * r))
            # Endpoint atom is 1-sigma, whereas 1-beta=(1-sigma)/2.
            self._weight(le, ("hi", gate.phase_index),
                         _q(_q(wi * correction) / 2))
            source_weight = _q(_q(-_q(2 * wj)) * cap)
            input_weight = _q(wi + _q(2 * wj))
            if input_weight < 0 or source_weight < 0:
                raise DomainError("admissible parameter lost its source signs")
            self._work.charge(24, 4)
            return ((input_weight, gate.input), (source_weight, pair.scale))
        # This is the other branch of the same algebraic rule, not a rescue
        # after inspecting a terminal result.  Its complete work is retained.
        return self._pair_step(pair, closure.row_indices, gates, gate_rows,
                               weights, True, le)

    def _certificate_proof(self, subject, mode):
        if mode in (0, 1, 2, 3, 4):
            return super()._certificate_proof(subject, mode)
        if mode != 5:
            raise DomainError("only the six registered certificate modes exist")
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
                raise DomainError("parametric query lost the complete event order")
            bank, meta = event, self._proof_banks[bank_index]
            bank_index -= 1
            local = [weights.pop(gate.q_index, Q(0)) for gate in bank.gates]
            self._work.charge(len(local), len(local))
            pieces, handled = [], set()
            for closure in meta.closure_pairs:
                pieces.extend(self._parametric_step(closure, bank.gates, meta.gate_rows, local, le))
                handled.update(closure.pair.positions)
                self._work.charge(2, 2)
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
                    divisor = _q(gate.upper - gate.lower)
                    self._weight(le, ("le", rows[2]), _q(weight * _q(-gate.lower / divisor)))
                    self._weight(le, ("le", rows[3]), _q(weight * _q(gate.upper / divisor)))
                else:
                    self._weight(le, ("le", rows[0]), -weight)
            rest = Form(form.constant, tuple(sorted(weights.items())))
            form = self._linear(pieces=((1, rest), *pieces))
        if bank_index != -1:
            raise DomainError("unprocessed parametric bank")
        upper = form.constant
        for index, coefficient in form.terms:
            factor = self.factors[index]
            if factor.kind in ("alias", "relu"):
                raise DomainError("unprocessed birth in parametric support")
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
            return tuple(self._certificate_proof(combined, mode) for mode in range(6))

    def support_proof(self, view, direction):
        proofs = self.support_proofs(view, direction)
        self._work.charge(6)
        return min(proofs, key=lambda proof: proof.bound)

    def cost_report(self):
        with self._work.branch():
            report = super().cost_report()
            self._work.charge(4)
            report.update(
                fixed_query_certificates=6, parametric_rule=True,
                parametric_product_factors=0, parametric_relation_rows=0,
                work_used=self._work.used, branch_work_used=self._work.branch_used,
                entries=self._work.entries,
            )
            return report
