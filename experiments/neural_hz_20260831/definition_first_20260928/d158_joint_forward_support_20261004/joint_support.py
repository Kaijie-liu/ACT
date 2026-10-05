"""Default-off fixed joint forward support for the unchanged D157 domain.

Every query computes both the inherited norm certificates and one shared-mass
certificate.  Their scalar upper bounds may be minimized; phase coefficients
are never minimized separately.  This is algebra on the owned forward state,
not a network backward pass, predicate optimizer, or failure-triggered rescue.
No new coordinates, native predicates, cache, or solver are introduced.
"""

from fractions import Fraction

from experiments.neural_hz_20260831.definition_first_20260928.d157_consumer_fiber_component_20261004 import consumer_fiber as cf


class Fiber(cf.Fiber):
    __slots__ = ()

    def _extend(self, *, phases=None, references=None, errors=None, banks=None, predicates=None):
        return Fiber(self.frame, self.phases if phases is None else phases,
                     self.errors if errors is None else errors,
                     self.banks if banks is None else banks,
                     self.predicates if predicates is None else predicates,
                     self.lineage + (cf.Identity("joint-support-extension"),),
                     references=self.references if references is None else references,
                     _auth=cf.base._AUTH)

    def _mass_certificates(self, bank):
        """Return complete affine (upper, lower) mass certificates."""
        zero = self._zero()
        dimension = 1 + len(zero.source) + len(zero.phase) + len(zero.error)
        self.work.charge(24 + 8 * len(bank.phase_indices) + 6 * dimension)
        uppers = tuple(max(Fraction(0), hi) for lo, hi in bank.preactivation_bounds)
        if not any(uppers):
            # Existing active branch caps, not an assumed stable phase, give Q=0.
            return zero, zero

        lower_phase = [Fraction(0)] * len(self.phases)
        for (lo, hi), index in zip(bank.preactivation_bounds, bank.phase_indices):
            lower_phase[index] = max(Fraction(0), lo)
        lower = cf.Form(0, zero.source, tuple(lower_phase), zero.error)

        energy = bank.energy.pad(len(self.phases))
        maximum = energy.extrema(self.work)[1]
        size = len(bank.phase_indices)
        scale = (cf.sqrt_upper(cf.rational(maximum / size), self.work)
                 if maximum > 0 else Fraction(1))
        rho = min(cf.rational(abs(bar) / upper)
                  for bar, upper in zip(bank.reference, uppers) if upper > 0)
        denominator = cf.rational(4 * scale)
        constant = cf.rational(cf.rational(energy.constant / denominator)
                               + cf.rational(cf.rational(scale * size) / 4))
        phase = [cf.rational(value / denominator) for value in energy.phase]
        for bar, upper, index in zip(bank.reference, uppers, bank.phase_indices):
            phase[index] = cf.rational(phase[index] + cf.rational(bar + cf.rational(rho * upper)))
        upper = cf.Form(constant, zero.source, tuple(phase), zero.error)
        source = bank.source_projection[bank.mass_index].pad(len(self.phases), len(self.errors))
        upper = upper.plus(source.scale(Fraction(1, 2), self.work), self.work)
        upper = upper.scale(cf.rational(Fraction(1) / cf.rational(1 + rho)), self.work)
        return upper, lower

    def joint_certificate(self, view, direction):
        """One phase-affine upper bound obtained by a fixed reverse bank sweep.

        Only banks in this state are traversed, once each.  A bank's source
        projection refers exclusively to earlier banks.  All source/phase
        coefficients are combined before the final physical-box maximization.
        """
        form = self._direction_form(view, direction)
        for bank in reversed(self.banks):
            size = len(bank.phase_indices)
            dimension = 1 + len(form.source) + len(form.phase) + len(form.error)
            self.work.charge(24 + 4 * len(bank.carrier) * size + 5 * dimension)
            vector = tuple(form.error[i] for i in bank.coordinates)
            columns = tuple(tuple(row[i] for row in bank.carrier) for i in range(size))
            projected = tuple(cf.base._dot(vector, column, self.work) for column in columns)
            kappa = Fraction(0)
            for coefficient, row in zip(vector, bank.carrier):
                endpoint = max(row) if coefficient >= 0 else min(row)
                kappa = cf.rational(kappa + cf.rational(coefficient * endpoint))

            phases, errors = list(form.phase), list(form.error)
            for value, bar, index in zip(projected, bank.reference, bank.phase_indices):
                phases[index] = cf.rational(phases[index] - cf.rational(value * bar))
            for index in bank.coordinates:
                errors[index] = Fraction(0)
            form = cf.Form(form.constant, form.source, tuple(phases), tuple(errors))
            upper, lower = self._mass_certificates(bank)
            form = form.plus(upper.scale(max(Fraction(0), kappa), self.work), self.work)
            form = form.plus(lower.scale(min(Fraction(0), kappa), self.work), self.work)

        if any(form.error):
            raise ValueError("joint substitution did not eliminate every owned amplitude")
        self.work.charge(8 + 5 * len(form.source) + len(form.phase))
        constant = form.constant
        for coefficient, (lo, hi) in zip(form.source, self.frame.bounds):
            constant = cf.rational(constant + max(cf.rational(coefficient * lo),
                                                  cf.rational(coefficient * hi)))
        return cf.PhaseAffine(constant, form.phase)

    def support_certificates(self, view, direction):
        # Both computations are unconditional; errors propagate, never fallback.
        # Materialize direction once so an iterable cannot mean different queries.
        direction = cf.base._qs(direction)
        inherited = cf.Fiber.support_certificates(self, view, direction)
        joint = self.joint_certificate(view, direction)
        return inherited + (joint,)

    def cost(self, *live_views):
        result = cf.Fiber.cost(self, *live_views)
        result["query_support_rule"] = "fixed_reverse_consumer_mass_v1"
        return result
