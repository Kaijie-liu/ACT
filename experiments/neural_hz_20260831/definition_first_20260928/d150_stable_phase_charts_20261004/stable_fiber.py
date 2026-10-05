"""Default-off stable charts and mixed affine norm fibers atop frozen D149.

Stable fresh coordinates alone may be eliminated, after imposing the certified
exact stable outputs.  Mixed norm predicates are substituted, never reduced to
independent crossing balls.  This is a small mathematical reference, not a
model importer, solver, GPU implementation or physical-cost qualification.
"""

from dataclasses import dataclass
from fractions import Fraction

from experiments.neural_hz_20260831.definition_first_20260928.d149_overlap_phase_fiber_20261004 import fiber as base


DisabledError = base.DisabledError
PhaseAffine = base.PhaseAffine
SourceConstraint = base.SourceConstraint
Form = base.Form
Predicate = base.Predicate
rational = base.rational


@dataclass(frozen=True)
class AffineNorm:
    """One vector, on the common frame, subject to ALL listed norm bounds."""

    forms: tuple
    radii: tuple

    def __post_init__(self):
        object.__setattr__(self, "forms", tuple(self.forms))
        object.__setattr__(self, "radii", tuple(self.radii))
        if (not 1 <= len(self.forms) <= base.MAX_OUTPUTS
                or any(not isinstance(form, Form) for form in self.forms)
                or not 1 <= len(self.radii) <= 2
                or any(not isinstance(radius, PhaseAffine) for radius in self.radii)):
            raise ValueError("invalid affine norm predicate")


@dataclass(frozen=True)
class ProjectionCost:
    projected_stable_coordinates: int = 0
    substituted_map_slots: int = 0
    substituted_norm_rows: int = 0
    temporary_peak_errors: int = 0
    temporary_peak_logical_entries: int = 0


class Fiber(base.Fiber):
    """Same original factors and lineage, plus source-conditioned norm rows."""

    __slots__ = ("affine_norms", "projection_cost")

    def __init__(self, frame, phases, errors, banks, predicates, lineage, *,
                 references=(), affine_norms=(), projection_cost=None, _auth=None):
        self.affine_norms = tuple(affine_norms)
        self.projection_cost = ProjectionCost() if projection_cost is None else projection_cost
        super().__init__(frame, phases, errors, banks, predicates, lineage,
                         references=references, _auth=_auth)

    def _validate(self):
        super()._validate()
        if not isinstance(self.projection_cost, ProjectionCost):
            raise ValueError("invalid projection cost record")
        values = tuple(vars(self.projection_cost).values())
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("negative or untyped projection accounting")
        if len(self.affine_norms) + sum(1 + len(bank.local) for bank in self.banks) > base.MAX_GROUPS:
            raise ValueError("combined coordinate/affine norm group cap exceeded")
        for norm in self.affine_norms:
            if not isinstance(norm, AffineNorm):
                raise ValueError("owned affine norm required")
            for form in norm.forms:
                if (len(form.source) != len(self.frame.bounds)
                        or len(form.phase) > len(self.phases) or len(form.error) > len(self.errors)):
                    raise ValueError("affine norm references absent factors")
                self.work.charge(4 + len(form.source) + len(form.phase) + len(form.error))
            for radius in norm.radii:
                if len(radius.phase) > len(self.phases) or radius.extrema(self.work)[0] < 0:
                    raise ValueError("affine norm radius is not certified nonnegative")

    def _extend(self, *, phases=None, references=None, errors=None, banks=None,
                predicates=None, affine_norms=None, projection_cost=None):
        return Fiber(self.frame, self.phases if phases is None else phases,
                     self.errors if errors is None else errors,
                     self.banks if banks is None else banks,
                     self.predicates if predicates is None else predicates,
                     self.lineage + (base.Identity("extension"),),
                     references=self.references if references is None else references,
                     affine_norms=self.affine_norms if affine_norms is None else affine_norms,
                     projection_cost=self.projection_cost if projection_cost is None else projection_cost,
                     _auth=base._AUTH)

    def _combined_form(self, view, direction):
        forms, direction = self._aligned(view), base._qs(direction)
        if len(forms) != len(direction):
            raise ValueError("support direction population mismatch")
        form = self._zero()
        for coefficient, row in zip(direction, forms):
            if coefficient:
                form = form.plus(row.scale(coefficient, self.work), self.work)
        return form

    def _outer_certificates(self, form):
        # Deliberately call the frozen method, not self.support/bounds.  There
        # is no recursive use of conditioned norms in a remainder certificate.
        return base.Fiber.support_certificates(self, self._readout((form,)), (1,))

    def support(self, view, direction):
        original = self._combined_form(view, direction)
        best = min(cert.extrema(self.work)[1] for cert in self._outer_certificates(original))
        for norm in self.affine_norms:
            remainder = original
            coefficients = []
            for stored in norm.forms:
                row = stored.pad(len(self.phases), len(self.errors))
                self.work.charge(8 + 3 * (len(row.source) + len(row.phase) + len(row.error)))
                pivot = None
                for index, value in enumerate(row.source):
                    if value:
                        pivot = ("source", index, value)
                        break
                if pivot is None:
                    for index, value in enumerate(row.error):
                        if value:
                            pivot = ("error", index, value)
                            break
                coefficient = Fraction(0)
                if pivot is not None:
                    kind, index, value = pivot
                    coefficient = rational(getattr(remainder, kind)[index] / value)
                    if coefficient:
                        remainder = remainder.plus(row.scale(-coefficient, self.work), self.work)
                coefficients.append(coefficient)
            squared = base._dot(coefficients, coefficients, self.work)
            alpha = base.sqrt_upper(squared, self.work)
            outer = self._outer_certificates(remainder)
            for radius in norm.radii:
                contribution = radius.scale(alpha, self.work)
                for cert in outer:
                    candidate = cert.plus(contribution, self.work).extrema(self.work)[1]
                    best = min(best, candidate)
        return best

    def bounds(self, scalar):
        forms = self._aligned(scalar)
        if len(forms) != 1:
            raise ValueError("direct predicate bounds require a scalar")
        form = forms[0]
        lower, upper = -self.support(scalar, (-1,)), self.support(scalar, (1,))
        negative = form.scale(-1, self.work)
        for row in self.predicates:
            saved = row.form.pad(len(self.phases), len(self.errors))
            self.work.charge(8 + 6 * (len(form.source) + len(form.phase) + len(form.error)))
            sign = 0
            if (saved.source, saved.phase, saved.error) == (form.source, form.phase, form.error):
                sign = 1
            elif (saved.source, saved.phase, saved.error) == (negative.source, negative.phase, negative.error):
                sign = -1
            if not sign:
                continue
            offset = rational(row.rhs - saved.constant)
            value = rational(form.constant + rational(sign * offset))
            if row.sense == "eq":
                lower, upper = max(lower, value), min(upper, value)
            elif sign == 1:
                upper = min(upper, value)
            else:
                lower = max(lower, value)
        if lower > upper:
            raise ValueError("inconsistent direct scalar interval; fail closed")
        return lower, upper

    def _groups(self, groups, size):
        groups = tuple(tuple(group) for group in groups)
        self.work.charge(8 + sum(4 + 4 * len(group) for group in groups))
        if len(groups) > base.MAX_GROUPS:
            raise ValueError("group population cap exceeded")
        for group in groups:
            if (not group or len(set(group)) != len(group)
                    or any(type(i) is not int or not 0 <= i < size for i in group)):
                raise ValueError("invalid original coordinate group")
        if (len(set(groups)) != len(groups)
                or groups and set().union(*(set(group) for group in groups)) != set(range(size))):
            raise ValueError("original local groups must form a complete cover")
        return groups

    def _nominal(self, form):
        answer = rational(form.constant + base._dot(form.source, self.frame.reference, self.work))
        return rational(answer + base._dot(form.phase, self.references, self.work))

    def _stable_layer(self, forms, names, certified, active):
        phases = self.phases + tuple(base.Identity(name) for name in names)
        nominal = tuple(self._nominal(form) for form in forms)
        state = self._extend(phases=phases,
                             references=self.references + tuple(int(value > 0) for value in nominal))
        outputs, rows = [], list(self.predicates)
        for i, (form, (lo, hi), positive) in enumerate(zip(forms, certified, active)):
            g = form.pad(len(phases), len(self.errors))
            q = g if positive else state._zero()
            alpha = Form(0, (0,) * len(self.frame.bounds),
                         tuple(int(j == len(self.phases) + i) for j in range(len(phases))),
                         (0,) * len(self.errors))
            lower, upper = min(Fraction(0), lo), max(Fraction(0), hi)
            rows.extend((
                Predicate(g.plus(alpha.scale(-upper, self.work), self.work), "le", 0),
                Predicate(g.scale(-1, self.work).plus(alpha.scale(-lower, self.work), self.work), "le", -lower),
                Predicate(q.scale(-1, self.work), "le", 0),
                Predicate(g.plus(q.scale(-1, self.work), self.work), "le", 0),
                Predicate(q.plus(alpha.scale(-upper, self.work), self.work), "le", 0),
            ))
            outputs.append(q)
        state = state._extend(predicates=tuple(rows))
        return state, state._readout(outputs)

    def relu(self, view, names, *, groups=(), bounds=None):
        forms = self._aligned(view)
        names = tuple(base._name(name) for name in names)
        if (len(names) != len(forms) or len(set(names)) != len(names)
                or len(self.phases) + len(names) > base.MAX_PHASES):
            raise ValueError("new original phase population mismatch")
        groups = self._groups(groups, len(forms))
        certified = tuple(self.bounds(self._readout((form,))) for form in forms)
        if bounds is not None:
            bounds = tuple(base._qs(pair) for pair in bounds)
            if (len(bounds) != len(forms) or any(len(pair) != 2 or pair[0] > lo or pair[1] < hi
                                                for pair, (lo, hi) in zip(bounds, certified))):
                raise ValueError("uncertified supplied preactivation bounds")
            certified = bounds
        active = tuple(lo >= 0 for lo, _ in certified)
        stable = tuple(is_active or hi <= 0 for is_active, (_, hi) in zip(active, certified))
        crossing = tuple(i for i, value in enumerate(stable) if not value)
        if not crossing:
            return self._stable_layer(forms, names, certified, active)
        temporary, old_q = base.Fiber.relu(self, view, names, groups=groups, bounds=certified)
        if len(crossing) == len(forms):
            return temporary, old_q

        # Mixed chart: keep every new phase in its original order, but eliminate
        # ONLY this layer's stable fresh errors after imposing q_stable=q_exact.
        old_count = len(self.errors)
        retained = self.errors + tuple(temporary.errors[old_count + i] for i in crossing)
        total_phases, total_errors = len(temporary.phases), len(retained)
        nominal = tuple(self._nominal(form) for form in forms)
        replacement = {}
        for i, is_stable in enumerate(stable):
            if not is_stable:
                continue
            g = forms[i].pad(total_phases, total_errors)
            sign = Fraction(1, 2) if active[i] else -Fraction(1, 2)
            value = g.scale(sign, self.work)
            correction = Form(rational(nominal[i] / 2), (0,) * len(self.frame.bounds),
                tuple(-nominal[i] if j == len(self.phases) + i else Fraction(0)
                      for j in range(total_phases)), (0,) * total_errors)
            replacement[i] = value.plus(correction, self.work)

        visited = 0

        def substitute(form):
            nonlocal visited
            padded = form.pad(total_phases, len(temporary.errors))
            slots = 1 + len(padded.source) + len(padded.phase) + len(padded.error)
            visited += slots
            self.work.charge(8 + 4 * slots)
            errors = padded.error[:old_count] + tuple(padded.error[old_count + i] for i in crossing)
            result = Form(padded.constant, padded.source, padded.phase, errors)
            for i, expression in replacement.items():
                coefficient = padded.error[old_count + i]
                if coefficient:
                    result = result.plus(expression.scale(coefficient, self.work), self.work)
            return result

        new_rows = tuple(Predicate(substitute(row.form), row.sense, row.rhs)
                         for row in temporary.predicates[len(self.predicates):])
        fresh = temporary.banks[-1]
        conditioned = []
        for group in (fresh.whole,) + fresh.local:
            rows = []
            for index in group.indices:
                row = Form(0, (0,) * len(self.frame.bounds), (0,) * total_phases,
                           tuple(int(j == old_count + index) for j in range(len(temporary.errors))))
                rows.append(substitute(row))
            conditioned.append(AffineNorm(tuple(rows), group.radii))
        # This standalone crossing ball is redundant. It does not replace the
        # source/old-factor energy spent in any original whole or local group.
        crossing_bank = base.Bank(fresh.name, tuple(range(old_count, total_errors)),
            base.NormGroup(tuple(range(len(crossing))), fresh.whole.radii), ())
        outputs = tuple(substitute(form) for form in old_q.forms)
        temporary_record = temporary.cost(old_q, view)
        keys = ("source_factors", "phase_factors", "residual_factors", "norm_incidence",
                "radius_coefficients", "predicate_coefficients", "decoder_coefficients",
                "source_bound_entries", "source_reference_entries", "phase_reference_entries",
                "lineage_entries", "bank_coordinate_entries", "live_readout_coefficients",
                "affine_norm_map_slots", "affine_norm_radius_coefficients", "affine_norm_incidence")
        logical_entries = sum(temporary_record.get(key, 0) for key in keys)
        old = self.projection_cost
        projection = ProjectionCost(
            old.projected_stable_coordinates + len(replacement),
            old.substituted_map_slots + visited,
            old.substituted_norm_rows + sum(len(norm.forms) for norm in conditioned),
            max(old.temporary_peak_errors, len(temporary.errors)),
            max(old.temporary_peak_logical_entries, logical_entries))
        final = temporary._extend(errors=retained, banks=self.banks + (crossing_bank,),
            predicates=self.predicates + new_rows,
            affine_norms=self.affine_norms + tuple(conditioned), projection_cost=projection)
        return final, final._readout(outputs)

    def contains(self, source, signed_phases, errors):
        if not base.Fiber.contains(self, source, signed_phases, errors):
            return False
        try:
            source, signed_phases, errors = base._qs(source), base._qs(signed_phases), base._qs(errors)
            phase01 = tuple(rational((value + 1) / 2) for value in signed_phases)
            for norm in self.affine_norms:
                values = tuple(form.at(source, phase01, errors, self.work) for form in norm.forms)
                squared = base._dot(values, values, self.work)
                for radius in norm.radii:
                    bound = radius.at(phase01, self.work)
                    if bound < 0 or squared > rational(bound * bound):
                        return False
            return True
        except (ValueError, TypeError, OverflowError):
            return False

    def cost(self, *live_views):
        record = base.Fiber.cost(self, *live_views)
        maps = sum(1 + len(form.source) + len(form.phase) + len(form.error)
                   for norm in self.affine_norms for form in norm.forms)
        radii = sum(1 + len(radius.phase) for norm in self.affine_norms for radius in norm.radii)
        incidence = sum(len(norm.forms) * len(norm.radii) for norm in self.affine_norms)
        self.work.charge(32 + 3 * maps + radii + incidence)
        record.update(affine_norms=len(self.affine_norms),
            affine_norm_rows=sum(len(norm.forms) for norm in self.affine_norms),
            affine_norm_constraints=sum(len(norm.radii) for norm in self.affine_norms),
            affine_norm_map_slots=maps, affine_norm_radius_coefficients=radii,
            affine_norm_incidence=incidence,
            projection_temporary_peak_errors=self.projection_cost.temporary_peak_errors,
            projection_temporary_peak_logical_entries=self.projection_cost.temporary_peak_logical_entries,
            projected_stable_coordinates=self.projection_cost.projected_stable_coordinates,
            substituted_map_slots=self.projection_cost.substituted_map_slots,
            substituted_norm_rows=self.projection_cost.substituted_norm_rows,
            algebra_work_used=self.work.used, complete_physical_qualification=False)
        return record
