"""Default-off rational D178 reference-observed Neural-HZ component.

The shared coordinates are delta, not D157's a. One common proxy t satisfies
Ct=Cd, Ht=Hd, ||t||^2<=E, delta=C(D_beta-D_tau)t, where H=C D_tau.
All original signed integer phases, guards, owned sources and predicates remain.
This implements mathematical supplied-point membership, not an optimizer, phase
search, model verifier, concrete witness validator, or GPU lowering. D158's
fixed mass-query algebra is retained only after the explicit delta=a-Hd
coordinate substitution; it is never applied to delta as though it were a.
"""

from dataclasses import dataclass
from fractions import Fraction

from experiments.neural_hz_20260831.definition_first_20260928.d157_consumer_fiber_component_20261004 import consumer_fiber as cf
from experiments.neural_hz_20260831.definition_first_20260928.d158_joint_forward_support_20261004 import joint_support as js


base = cf.base
rational = cf.rational
Form = cf.Form
PhaseAffine = cf.PhaseAffine
Predicate = cf.Predicate
SourceConstraint = cf.SourceConstraint
Readout = cf.Readout
Frame = cf.Frame
Identity = cf.Identity
Work = cf.Work
DisabledError = cf.DisabledError
sqrt_upper = cf.sqrt_upper
MAX_BITS = cf.MAX_BITS
MAX_SOURCES = cf.MAX_SOURCES
MAX_PHASES = cf.MAX_PHASES
MAX_ERRORS = cf.MAX_ERRORS
MAX_OUTPUTS = cf.MAX_OUTPUTS
MAX_BANKS = cf.MAX_BANKS
MAX_GROUPS = cf.MAX_GROUPS
MAX_PREDICATES = cf.MAX_PREDICATES
MAX_MATRIX_ENTRIES = cf.MAX_MATRIX_ENTRIES
MAX_WORK = cf.MAX_WORK


@dataclass(frozen=True)
class Bank(cf.Bank):
    observed_projection: tuple


class Fiber(cf.Fiber):
    """Owned common-witness domain; new banks only through relu_packet."""

    __slots__ = ()

    def _validate(self):
        super()._validate()
        for bank in self.banks:
            if not isinstance(bank, Bank) or len(bank.observed_projection) != len(bank.carrier):
                raise ValueError("D180 banks require owned reference-source observations")
            first_error, first_phase = bank.coordinates[0], bank.phase_indices[0]
            if bank.phase_indices != tuple(range(first_phase, first_phase + len(bank.reference))):
                raise ValueError("original phase identities must remain a contiguous birth block")
            if any(self.references[index] != int(bar > 0)
                   for index, bar in zip(bank.phase_indices, bank.reference)):
                raise ValueError("bank reference phase differs from its original nominal value")
            if any(bank.energy.phase[first_phase:]):
                raise ValueError("energy must be certified on the whole parent before fresh phases")
            for form in bank.source_projection + bank.observed_projection:
                if (not isinstance(form, Form) or len(form.source) != len(self.frame.bounds)
                        or len(form.phase) > len(self.phases) or len(form.error) > len(self.errors)
                        or any(form.error[first_error:]) or any(form.phase[first_phase:])):
                    raise ValueError("source observations may reference only the owned parent")
            self.work.charge(16 + sum(1 + len(f.source) + len(f.phase) + len(f.error)
                                      for f in bank.observed_projection))

    def _extend(self, *, phases=None, references=None, errors=None, banks=None, predicates=None):
        return Fiber(self.frame, self.phases if phases is None else phases,
                     self.errors if errors is None else errors,
                     self.banks if banks is None else banks,
                     self.predicates if predicates is None else predicates,
                     self.lineage + (Identity("reference-observed-extension"),),
                     references=self.references if references is None else references,
                     _auth=base._AUTH)

    def _delta_forms(self, bank):
        zero = self._zero()
        return tuple(Form(0, zero.source, zero.phase,
                          tuple(int(j == index) for j in range(len(self.errors))))
                     for index in bank.coordinates)

    def _amplitude_forms(self, bank):
        """Internal D157-compatible a=Hd+delta; never relabel delta as a."""
        return tuple(observed.pad(len(self.phases), len(self.errors)).plus(delta, self.work)
                     for observed, delta in zip(bank.observed_projection, self._delta_forms(bank)))

    def amplitudes(self, index=-1):
        """Return the canonical shared delta coordinates, not Hd+delta."""
        return self._readout(self._delta_forms(self.banks[index]))

    def _flip_energy(self, matrix, bank):
        """Emax sum_i ||M C_i||^2 |beta_i-tau_i|, affine in original bits."""
        product = cf._multiply(matrix, bank.carrier, self.work)
        maximum = bank.energy.extrema(self.work)[1]
        constant = Fraction(0)
        coefficients = [Fraction(0)] * len(self.phases)
        for j, (index, bar) in enumerate(zip(bank.phase_indices, bank.reference)):
            column = tuple(row[j] for row in product)
            squared = base._dot(column, column, self.work)
            value = rational(maximum * squared)
            tau = int(bar > 0)
            constant = rational(constant + rational(value * tau))
            coefficients[index] = rational(value * (1 - 2 * tau))
        return PhaseAffine(constant, tuple(coefficients))

    def _canonical_certificate(self, form):
        """Canonical flip-energy norm bound, maximized over the phase box."""
        constant = form.constant
        for coefficient, (lo, hi) in zip(form.source, self.frame.bounds):
            constant = rational(constant + max(rational(coefficient * lo), rational(coefficient * hi)))
        for bank in self.banks:
            vector = tuple(form.error[i] for i in bank.coordinates)
            if any(vector):
                maximum = self._flip_energy((vector,), bank).extrema(self.work)[1]
                constant = rational(constant + sqrt_upper(maximum, self.work))
        return PhaseAffine(constant, form.phase)

    def _substitute_observation(self, form, bank):
        """Replace v*delta by v*a-v*Hd; current-bank coefficients now mean a.

        Hd references only earlier canonical coordinates. The fixed reverse
        traversal subsequently converts those, not the already converted a.
        """
        vector = tuple(form.error[index] for index in bank.coordinates)
        self.work.charge(8 + len(vector))
        for coefficient, observed in zip(vector, bank.observed_projection):
            if coefficient:
                aligned = observed.pad(len(self.phases), len(self.errors))
                form = form.plus(aligned.scale(-coefficient, self.work), self.work)
        return form, vector

    def _legacy_norm_certificate(self, form):
        for bank in reversed(self.banks):
            form, _ = self._substitute_observation(form, bank)
        # This private temporary readout denotes legacy a coordinates only
        # inside this explicit algebra call; it is never returned to a caller.
        return cf.Fiber.support_certificates(self, self._readout((form,)), (1,))[0]

    def _mass_certificates(self, bank):
        # Cd, branch caps and the energy certificate have unchanged semantics.
        return js.Fiber._mass_certificates(self, bank)

    def _joint_certificate(self, form):
        for bank in reversed(self.banks):
            form, vector = self._substitute_observation(form, bank)
            size = len(bank.phase_indices)
            dimension = 1 + len(form.source) + len(form.phase) + len(form.error)
            self.work.charge(24 + 4 * len(bank.carrier) * size + 5 * dimension)
            columns = tuple(tuple(row[i] for row in bank.carrier) for i in range(size))
            projected = tuple(base._dot(vector, column, self.work) for column in columns)
            kappa = Fraction(0)
            for coefficient, row in zip(vector, bank.carrier):
                endpoint = max(row) if coefficient >= 0 else min(row)
                kappa = rational(kappa + rational(coefficient * endpoint))
            phases, errors = list(form.phase), list(form.error)
            for value, bar, index in zip(projected, bank.reference, bank.phase_indices):
                phases[index] = rational(phases[index] - rational(value * bar))
            for index in bank.coordinates:
                errors[index] = Fraction(0)
            form = Form(form.constant, form.source, tuple(phases), tuple(errors))
            upper, lower = self._mass_certificates(bank)
            form = form.plus(upper.scale(max(Fraction(0), kappa), self.work), self.work)
            form = form.plus(lower.scale(min(Fraction(0), kappa), self.work), self.work)
        if any(form.error):
            raise ValueError("reference-aware mass sweep did not eliminate all shared deltas")
        self.work.charge(8 + 5 * len(form.source) + len(form.phase))
        constant = form.constant
        for coefficient, (lo, hi) in zip(form.source, self.frame.bounds):
            constant = rational(constant + max(rational(coefficient * lo), rational(coefficient * hi)))
        return PhaseAffine(constant, form.phase)

    def joint_certificate(self, view, direction):
        """D158 mass rule with the required delta=a-Hd coordinate change."""
        return self._joint_certificate(self._direction_form(view, direction))

    def support_certificates(self, view, direction):
        """Always compute three complete certificates; never failure fallback."""
        form = self._direction_form(view, base._qs(direction))
        canonical = self._canonical_certificate(form)
        legacy = self._legacy_norm_certificate(form)
        joint = self._joint_certificate(form)
        return canonical, legacy, joint

    def _vector_energy(self, forms):
        """Parent-wide deviation energy using fixed equal-weight Cauchy."""
        terms = []
        source = tuple(form.source for form in forms)
        radius = base.source_norm_upper(source, self.frame.bounds, self.work)
        squared = rational(radius * radius)
        if squared:
            terms.append(PhaseAffine(squared, (0,) * len(self.phases)))
        for j, reference in enumerate(self.references):
            column = tuple(form.phase[j] for form in forms)
            squared = base._dot(column, column, self.work)
            if squared:
                coefficients = [Fraction(0)] * len(self.phases)
                coefficients[j] = rational(squared * (1 - 2 * reference))
                terms.append(PhaseAffine(rational(squared * reference), tuple(coefficients)))
        for bank in self.banks:
            matrix = tuple(tuple(form.error[i] for i in bank.coordinates) for form in forms)
            if any(value for row in matrix for value in row):
                term = self._flip_energy(matrix, bank)
                if term.constant or any(term.phase):
                    terms.append(term)
        result = PhaseAffine(0, (0,) * len(self.phases))
        for term in terms:
            result = result.plus(term.scale(len(terms), self.work), self.work)
        return result

    def relu_packet(self, view, consumers, phase_names):
        """Uniform whole-consumer birth; caller may not supply energy or bounds.

        B must include ALL mathematical consumers, predicates and reconstruction
        uses of the new ReLU value. This algebra API does not certify ONNX graph
        completeness. The original source decoder remains the owned decoder.
        """
        forms = self._aligned(view)
        m = len(forms)
        names = tuple(base._name(name) for name in phase_names)
        consumers = tuple(base._qs(row) for row in consumers)
        if (len(names) != m or len(set(names)) != m or not 1 <= len(consumers) <= MAX_OUTPUTS
                or any(len(row) != m for row in consumers) or len(consumers) * m > MAX_MATRIX_ENTRIES):
            raise ValueError("complete consumer/phase population mismatch")
        ones = (Fraction(1),) * m
        mass_index = next((i for i, row in enumerate(consumers) if row == ones), len(consumers))
        carrier = consumers if mass_index < len(consumers) else consumers + (ones,)
        p = len(carrier)
        fresh_rows = 2 * m + 18 * p + 2 * len(consumers)
        if (p > MAX_OUTPUTS or p * m > MAX_MATRIX_ENTRIES
                or len(self.phases) + m > MAX_PHASES or len(self.errors) + p > MAX_ERRORS
                or len(self.banks) + 1 > MAX_BANKS or len(self.predicates) + fresh_rows > MAX_PREDICATES):
            raise ValueError("reference-observed transformation exceeds inherited caps")
        existing = {item.name for item in self.frame.sources + self.phases + self.errors}
        delta_names = tuple("reference-delta:" + names[0] + ":" + str(i) for i in range(p))
        if (set(names) & existing or set(delta_names) & (existing | set(names))
                or len(set(delta_names)) != p):
            raise ValueError("new original phase/delta identity collision")
        self.work.charge(48 + 2 * p * m + fresh_rows)
        certified = tuple(self.bounds(self._readout((form,))) for form in forms)
        nominal = []
        for form in forms:
            value = rational(form.constant + base._dot(form.source, self.frame.reference, self.work))
            nominal.append(rational(value + base._dot(form.phase, self.references, self.work)))
        nominal = tuple(nominal)
        tau = tuple(int(value > 0) for value in nominal)
        nphase, nerror = len(self.phases) + m, len(self.errors) + p
        phase_indices = tuple(range(len(self.phases), nphase))
        coordinates = tuple(range(len(self.errors), nerror))
        energy = self._vector_energy(forms).pad(nphase)
        gs = tuple(form.pad(nphase, nerror) for form in forms)
        deviations = tuple(Form(rational(form.constant - bar), form.source, form.phase, form.error)
                           for form, bar in zip(gs, nominal))
        source_projection = tuple(cf._sum_forms(deviations, row, 0, self.work) for row in carrier)
        observed_projection = tuple(cf._sum_forms(deviations,
                    tuple(rational(value * bit) for value, bit in zip(row, tau)), 0, self.work) for row in carrier)
        start, ranges = len(self.predicates), []
        for label, count in (("guards", 2 * m), ("caps", 4 * p), ("mass", 2 * len(consumers)),
                             ("energy", 6 * p), ("observed_energy", 6 * p), ("delta_range", 2 * p)):
            ranges.append((label, start, start + count))
            start += count
        bank = Bank("reference-packet:" + ",".join(names), coordinates, phase_indices, carrier,
                    consumers, mass_index, nominal, source_projection, energy, certified,
                    tuple(ranges), observed_projection)
        state = self._extend(phases=self.phases + tuple(Identity(name) for name in names),
                             references=self.references + tau,
                             errors=self.errors + tuple(Identity(name) for name in delta_names),
                             banks=self.banks + (bank,))
        zero = state._zero()
        bits = tuple(Form(0, zero.source, tuple(int(j == index) for j in range(nphase)), zero.error)
                     for index in phase_indices)
        amplitudes, deltas = state._amplitude_forms(bank), state._delta_forms(bank)
        packet = state._packet_forms(bank)
        rows = list(self.predicates)

        def le(form, rhs=0):
            rows.append(Predicate(Form(0, form.source, form.phase, form.error), "le",
                                  rational(rational(rhs) - form.constant)))

        def phase_form(constant, coefficients):
            return Form(constant, zero.source, tuple(coefficients), zero.error)

        for g, bit, (lo, hi) in zip(gs, bits, certified):
            le(g.plus(bit.scale(-hi, self.work), self.work))
            le(g.scale(-1, self.work).plus(bit.scale(-lo, self.work), self.work), -lo)
        for row, amplitude, source in zip(carrier, amplitudes, source_projection):
            active_lower, active_upper = [Fraction(0)] * nphase, [Fraction(0)] * nphase
            inactive_lower, inactive_upper = [Fraction(0)] * nphase, [Fraction(0)] * nphase
            lower_constant = upper_constant = Fraction(0)
            for coefficient, bar, (lo, hi), index in zip(row, nominal, certified, phase_indices):
                alo, ahi = rational(max(lo, 0) - bar), rational(max(hi, 0) - bar)
                ilo, ihi = rational(min(lo, 0) - bar), rational(min(hi, 0) - bar)
                av, iv = (rational(coefficient * alo), rational(coefficient * ahi)), (rational(coefficient * ilo), rational(coefficient * ihi))
                active_lower[index], active_upper[index] = min(av), max(av)
                low, high = min(iv), max(iv)
                lower_constant, upper_constant = rational(lower_constant + low), rational(upper_constant + high)
                inactive_lower[index], inactive_upper[index] = -low, -high
            le(phase_form(0, active_lower).plus(amplitude.scale(-1, self.work), self.work))
            le(amplitude.plus(phase_form(0, active_upper).scale(-1, self.work), self.work))
            complement = source.plus(amplitude.scale(-1, self.work), self.work)
            le(phase_form(lower_constant, inactive_lower).plus(complement.scale(-1, self.work), self.work))
            le(complement.plus(phase_form(upper_constant, inactive_upper).scale(-1, self.work), self.work))
        mass = packet[mass_index]
        for row, output in zip(consumers, packet):
            le(mass.scale(min(row), self.work).plus(output.scale(-1, self.work), self.work))
            le(output.plus(mass.scale(-max(row), self.work), self.work))

        maximum = energy.extrema(self.work)[1]
        scales = []
        for row in carrier:
            squared = base._dot(row, row, self.work)
            scales.append(sqrt_upper(rational(maximum / squared), self.work) if maximum and squared else Fraction(1))
        # Always keep both complete six-direction banks; this is not a fallback.
        for observed in (False, True):
            for row, amplitude, source, hd, scale in zip(carrier, amplitudes, source_projection, observed_projection, scales):
                for left, right in ((scale, 0), (-scale, 0), (0, scale), (0, -scale), (scale, -scale), (-scale, scale)):
                    left, right = rational(left), rational(right)
                    shift = rational(right - left) if observed else Fraction(0)
                    lhs = amplitude.scale(rational(2 * rational(left - right)), self.work)
                    lhs = lhs.plus(source.scale(rational(2 * right), self.work), self.work)
                    lhs = lhs.plus(hd.scale(rational(2 * shift), self.work), self.work)
                    coefficients, constant = list(energy.phase), energy.constant
                    for value, reference, index in zip(row, tau, phase_indices):
                        h = rational(shift * reference)
                        active = rational(rational(rational(left + h) * value) ** 2)
                        inactive = rational(rational(rational(right + h) * value) ** 2)
                        constant = rational(constant + inactive)
                        coefficients[index] = rational(coefficients[index] + rational(active - inactive))
                    le(lhs.plus(phase_form(constant, coefficients).scale(-1, self.work), self.work))
        radius = sqrt_upper(maximum, self.work)
        for bar, (lo, hi) in zip(nominal, certified):
            radius = max(radius, abs(rational(lo - bar)), abs(rational(hi - bar)))
        for row, delta in zip(carrier, deltas):
            constant, coefficients = Fraction(0), [Fraction(0)] * nphase
            for value, reference, index in zip(row, tau, phase_indices):
                bound = rational(radius * abs(value))
                constant = rational(constant + rational(bound * reference))
                coefficients[index] = rational(bound * (1 - 2 * reference))
            bound = phase_form(constant, coefficients)
            le(delta.plus(bound.scale(-1, self.work), self.work))
            le(delta.scale(-1, self.work).plus(bound.scale(-1, self.work), self.work))
        if len(rows) != len(self.predicates) + fresh_rows:
            raise ValueError("D178 fixed finite-row population mismatch")
        final = state._extend(predicates=tuple(rows))
        return final, final._readout(packet[:len(consumers)])

    def contains(self, source, signed_phases, errors):
        """Exact membership at ONE supplied original integer-phase assignment."""
        try:
            source, phase01, errors = self._assignment(source, signed_phases, errors)
            if not self._linear_holds(source, phase01, errors):
                return False
            for bank in self.banks:
                p, m = len(bank.carrier), len(bank.phase_indices)
                if 3 * p * m > MAX_MATRIX_ENTRIES:
                    raise ValueError("reference-observed membership carrier cap exceeded")
                self.work.charge(32 + 3 * p * m)
                tau = tuple(int(bar > 0) for bar in bank.reference)
                observed = tuple(tuple(rational(value * reference) for value, reference in zip(row, tau)) for row in bank.carrier)
                active = tuple(tuple(rational(value * phase01[index]) for value, index in zip(row, bank.phase_indices))
                               for row in bank.carrier)
                cd = tuple(form.at(source, phase01, errors, self.work) for form in bank.source_projection)
                hd = tuple(form.at(source, phase01, errors, self.work) for form in bank.observed_projection)
                a = tuple(rational(value + errors[index]) for value, index in zip(hd, bank.coordinates))
                used = cf._minimum_energy(bank.carrier + observed + active, tuple(range(m)), cd + hd + a, self.work)
                available = bank.energy.at(phase01, self.work)
                if available < 0 or used > available:
                    return False
            return True
        except (ValueError, TypeError, OverflowError):
            return False

    def cost(self, *live_views):
        """Include added Hd storage; logical counts are not physical qualification."""
        result = super().cost(*live_views)
        extra = sum(1 + len(form.source) + len(form.phase) + len(form.error)
                    for bank in self.banks for form in bank.observed_projection)
        self.work.charge(16 + extra)
        result.update(observed_projection_coefficients=extra, canonical_residual="delta",
                      implicit_reference_mask_entries=sum(len(bank.phase_indices) for bank in self.banks),
                      support_rule="fixed_canonical_legacy_norm_reference_aware_mass",
                      support_certificate_count=3, d158_joint_query_implemented=True,
                      algebra_work_used=self.work.used)
        return result
