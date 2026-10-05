"""Default-off rational two-branch consumer-fiber reference (D157).

Only the supplied complete mathematical consumer block is represented.  This
module does not certify a model graph, solve an LP, or execute a network.  It
reuses D149's immutable scalar/frame/readout primitives and affine operations,
NOT its ReLU or independent-error transformers.  Native membership performs
exact linear algebra at ONE supplied signed-phase assignment; it is not a
phase search, optimization oracle, or counterexample validator.
"""

from dataclasses import dataclass
from fractions import Fraction

from experiments.neural_hz_20260831.definition_first_20260928.d149_overlap_phase_fiber_20261004 import fiber as base


rational = base.rational
Form = base.Form
PhaseAffine = base.PhaseAffine
Predicate = base.Predicate
SourceConstraint = base.SourceConstraint
Readout = base.Readout
Frame = base.Frame
Identity = base.Identity
Work = base.Work
DisabledError = base.DisabledError
sqrt_upper = base.sqrt_upper
MAX_BITS = base.MAX_BITS
MAX_SOURCES = base.MAX_SOURCES
MAX_PHASES = base.MAX_PHASES
MAX_ERRORS = base.MAX_ERRORS
MAX_OUTPUTS = base.MAX_OUTPUTS
MAX_BANKS = base.MAX_BANKS
MAX_GROUPS = base.MAX_GROUPS
MAX_PREDICATES = base.MAX_PREDICATES
MAX_MATRIX_ENTRIES = base.MAX_MATRIX_ENTRIES
MAX_WORK = base.MAX_WORK


@dataclass(frozen=True)
class Bank:
    name: str
    coordinates: tuple
    phase_indices: tuple
    carrier: tuple
    consumers: tuple
    mass_index: int
    reference: tuple
    source_projection: tuple
    energy: PhaseAffine
    preactivation_bounds: tuple
    row_ranges: tuple


def _sum_forms(forms, coefficients, offset, work):
    if not forms or len(forms) != len(coefficients):
        raise ValueError("affine combination population mismatch")
    first = forms[0]
    result = Form(offset, (0,) * len(first.source), (0,) * len(first.phase),
                  (0,) * len(first.error))
    for coefficient, form in zip(coefficients, forms):
        if coefficient:
            result = result.plus(form.scale(coefficient, work), work)
    return result


def _multiply(left, right, work):
    if not left or not right or any(len(row) != len(right) for row in left):
        raise ValueError("matrix product population mismatch")
    columns = len(right[0])
    if any(len(row) != columns for row in right):
        raise ValueError("ragged matrix")
    if len(left) * columns > MAX_MATRIX_ENTRIES:
        raise ValueError("matrix product entry cap exceeded")
    work.charge(16 + len(left) * columns)
    transposed = tuple(tuple(row[j] for row in right) for j in range(columns))
    return tuple(tuple(base._dot(row, column, work) for column in transposed)
                 for row in left)


def _minimum_energy(carrier, selected, target, work):
    """Solve G z=target exactly and return target*z, or reject its range.

    G is constructed as the Gram matrix of the selected columns.  Free
    solution coordinates are set to zero; the quadratic value is independent
    of this choice because target is in range(G).  No inverse/float tolerance
    or approximate rank is used.
    """
    size = len(carrier)
    if len(target) != size or size * (size + 1) > MAX_MATRIX_ENTRIES:
        raise ValueError("native membership matrix cap exceeded")
    work.charge(32 + 12 * size ** 3 + 4 * size * size * len(selected))
    augmented = []
    for row in carrier:
        values = []
        for other in carrier:
            value = Fraction(0)
            for j in selected:
                value = rational(value + rational(row[j] * other[j]))
            values.append(value)
        augmented.append(values + [target[len(augmented)]])
    pivots = []
    rank = 0
    for column in range(size):
        pivot = next((i for i in range(rank, size) if augmented[i][column]), None)
        if pivot is None:
            continue
        augmented[rank], augmented[pivot] = augmented[pivot], augmented[rank]
        divisor = augmented[rank][column]
        augmented[rank] = [rational(value / divisor) for value in augmented[rank]]
        for i in range(size):
            if i == rank:
                continue
            factor = augmented[i][column]
            if factor:
                augmented[i] = [rational(value - rational(factor * pivot_value))
                                for value, pivot_value in zip(augmented[i], augmented[rank])]
        pivots.append(column)
        rank += 1
    if any(all(value == 0 for value in row[:size]) and row[size] != 0
           for row in augmented):
        raise ValueError("assignment is outside the branch range")
    solution = [Fraction(0)] * size
    for row, column in enumerate(pivots):
        solution[column] = augmented[row][size]
    energy = base._dot(target, solution, work)
    if energy < 0:
        raise ValueError("invalid Gram energy certificate")
    return energy


class Fiber(base.Fiber):
    """Owned source/phase relation with joint consumer amplitudes.

    box/embed_hz and affine/add/concat/alignment/evaluate/decode are the
    inherited algebra-only APIs.  New banks may ONLY be born by relu_packet;
    there is no public arbitrary energy/certificate injection API.
    """

    __slots__ = ()

    def _validate(self):
        frame = self.frame
        n = len(frame.bounds)
        if (not isinstance(frame, Frame) or not isinstance(frame.work, Work)
                or not 1 <= n <= MAX_SOURCES or len(self.phases) > MAX_PHASES
                or len(self.errors) > MAX_ERRORS or len(self.banks) > MAX_BANKS
                or len(self.predicates) > MAX_PREDICATES or not self.lineage):
            raise ValueError("consumer-fiber population cap exceeded")
        if (len(frame.sources) != n or len(frame.reference) != n
                or any(len(pair) != 2 or pair[0] > pair[1] for pair in frame.bounds)
                or frame.reference != tuple(rational(rational(lo + hi) / 2)
                                            for lo, hi in frame.bounds)):
            raise ValueError("invalid owned source box/reference")
        if (not 1 <= len(frame.decoder_matrix) <= MAX_OUTPUTS
                or len(frame.decoder_bias) != len(frame.decoder_matrix)
                or any(len(row) != n for row in frame.decoder_matrix)
                or n * len(frame.decoder_matrix) > MAX_MATRIX_ENTRIES):
            raise ValueError("decoder must be a bounded source-affine map")
        if (len(self.references) != len(self.phases)
                or any(type(value) is not int or value not in (0, 1)
                       for value in self.references)):
            raise ValueError("one nominal reference is required per original bit")
        identities = frame.sources + self.phases + self.errors
        if (any(not isinstance(item, Identity) for item in identities)
                or len(set(identities)) != len(identities)
                or len({item.name for item in identities}) != len(identities)):
            raise ValueError("source, phase and amplitude identities must be distinct")
        claimed = []
        previous_error_count = 0
        for bank in self.banks:
            if not isinstance(bank, Bank):
                raise ValueError("only authenticated consumer banks are supported")
            p, m = len(bank.carrier), len(bank.phase_indices)
            if (not 1 <= p <= MAX_OUTPUTS or not 1 <= m <= MAX_OUTPUTS
                    or p * m > MAX_MATRIX_ENTRIES or len(bank.coordinates) != p
                    or bank.coordinates != tuple(range(previous_error_count, previous_error_count + p))
                    or len(bank.reference) != m or len(bank.preactivation_bounds) != m
                    or any(len(row) != m for row in bank.carrier)
                    or not 1 <= len(bank.consumers) <= p
                    or bank.carrier[:len(bank.consumers)] != bank.consumers
                    or not 0 <= bank.mass_index < p
                    or bank.carrier[bank.mass_index] != (Fraction(1),) * m
                    or len(set(bank.phase_indices)) != m
                    or any(type(i) is not int or not 0 <= i < len(self.phases)
                           for i in bank.phase_indices)):
                raise ValueError("invalid complete consumer-bank metadata")
            if (not isinstance(bank.energy, PhaseAffine)
                    or len(bank.energy.phase) > len(self.phases)
                    or bank.energy.extrema(self.work)[0] < 0
                    or len(bank.source_projection) != p):
                raise ValueError("invalid nonnegative phase energy")
            for form in bank.source_projection:
                if (not isinstance(form, Form) or len(form.source) != n
                        or len(form.phase) > len(self.phases)
                        or len(form.error) > len(self.errors)
                        or any(form.error[i] for i in range(previous_error_count, len(form.error)))):
                    raise ValueError("bank source may only reference its parent amplitudes")
            claimed.extend(bank.coordinates)
            previous_error_count += p
        if tuple(claimed) != tuple(range(len(self.errors))):
            raise ValueError("consumer banks must partition all shared amplitudes")
        for predicate in self.predicates:
            if (not isinstance(predicate, Predicate)
                    or len(predicate.form.source) != n
                    or len(predicate.form.phase) > len(self.phases)
                    or len(predicate.form.error) > len(self.errors)):
                raise ValueError("predicate refers to an unavailable factor")
        self.work.charge(32 + len(identities) + len(self.predicates) + len(self.banks))

    def _extend(self, *, phases=None, references=None, errors=None, banks=None, predicates=None):
        return Fiber(self.frame, self.phases if phases is None else phases,
                     self.errors if errors is None else errors,
                     self.banks if banks is None else banks,
                     self.predicates if predicates is None else predicates,
                     self.lineage + (Identity("consumer-extension"),),
                     references=self.references if references is None else references,
                     _auth=base._AUTH)

    def relu(self, *args, **kwargs):
        raise DisabledError("use the complete relu_packet transformer")

    def error_bank(self, *args, **kwargs):
        raise DisabledError("arbitrary residual banks are not part of D157")

    def interval_affine(self, *args, **kwargs):
        raise DisabledError("interval coefficient lowering is outside D157 scope")

    def _bank_bound(self, *args, **kwargs):
        raise DisabledError("D149 norm-bank queries do not apply to consumer banks")

    def _vector_radius(self, *args, **kwargs):
        raise DisabledError("D157 uses its own squared-energy recurrence")

    def _amplitude_forms(self, bank):
        zero = self._zero()
        return tuple(Form(0, zero.source, zero.phase,
                          tuple(int(j == index) for j in range(len(self.errors))))
                     for index in bank.coordinates)

    def amplitudes(self, index=-1):
        return self._readout(self._amplitude_forms(self.banks[index]))

    def _packet_forms(self, bank):
        outputs = []
        for row, amplitude in zip(bank.carrier, self._amplitude_forms(bank)):
            coefficients = [Fraction(0)] * len(self.phases)
            for value, bar, index in zip(row, bank.reference, bank.phase_indices):
                coefficients[index] = rational(value * bar)
            center = Form(0, (0,) * len(self.frame.bounds), tuple(coefficients),
                          (0,) * len(self.errors))
            outputs.append(center.plus(amplitude, self.work))
        return tuple(outputs)

    def bank_readout(self, index=-1):
        """Return the entire Cq carrier, including the single mass row."""
        return self._readout(self._packet_forms(self.banks[index]))

    def _direction_form(self, view, direction):
        forms, direction = self._aligned(view), base._qs(direction)
        return _sum_forms(forms, direction, 0, self.work)

    def support_certificates(self, view, direction):
        """One safe affine-phase certificate; no predicate optimization."""
        form = self._direction_form(view, direction)
        constant = form.constant
        for coefficient, (lo, hi) in zip(form.source, self.frame.bounds):
            constant = rational(constant + max(rational(coefficient * lo),
                                               rational(coefficient * hi)))
        for bank in self.banks:
            vector = tuple(form.error[i] for i in bank.coordinates)
            columns = tuple(tuple(row[j] for row in bank.carrier)
                            for j in range(len(bank.phase_indices)))
            projected = tuple(base._dot(vector, column, self.work) for column in columns)
            squared = base._dot(projected, projected, self.work)
            maximum = bank.energy.extrema(self.work)[1]
            constant = rational(constant + sqrt_upper(rational(squared * maximum), self.work))
        return (PhaseAffine(constant, form.phase),)

    def _vector_energy(self, forms):
        """Fixed uniform weighted-Cauchy energy, retaining old phase terms."""
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
            if not any(value for row in matrix for value in row):
                continue
            product = _multiply(matrix, bank.carrier, self.work)
            norm = base.matrix_norm_upper(product, self.work)
            squared = rational(norm * norm)
            if squared:
                terms.append(bank.energy.scale(squared, self.work).pad(len(self.phases)))
        result = PhaseAffine(0, (0,) * len(self.phases))
        for term in terms:
            result = result.plus(term.scale(len(terms), self.work), self.work)
        return result

    def relu_packet(self, view, consumers, phase_names):
        """Replace a whole declared ReLU consumer block, returning its Bq.

        Caller-supplied B must describe ALL consumers of the mathematical
        block.  Actual model-graph coverage is deliberately not certified.
        The sole extra carrier is sum(q); it is reused if exactly present.
        """
        forms = self._aligned(view)
        m = len(forms)
        names = tuple(base._name(name) for name in phase_names)
        consumers = tuple(base._qs(row) for row in consumers)
        if (len(names) != m or len(set(names)) != m or not 1 <= len(consumers) <= MAX_OUTPUTS
                or any(len(row) != m for row in consumers)
                or len(consumers) * m > MAX_MATRIX_ENTRIES):
            raise ValueError("complete consumer/phase population mismatch")
        ones = (Fraction(1),) * m
        mass_index = next((i for i, row in enumerate(consumers) if row == ones), len(consumers))
        carrier = consumers if mass_index < len(consumers) else consumers + (ones,)
        p = len(carrier)
        fresh_rows = 2 * m + 10 * p + 2 * len(consumers)
        if (p > MAX_OUTPUTS or p * m > MAX_MATRIX_ENTRIES
                or len(self.phases) + m > MAX_PHASES or len(self.errors) + p > MAX_ERRORS
                or len(self.banks) + 1 > MAX_BANKS
                or len(self.predicates) + fresh_rows > MAX_PREDICATES):
            raise ValueError("complete consumer transformation exceeds reference caps")
        existing = {item.name for item in self.frame.sources + self.phases + self.errors}
        amplitude_names = tuple("consumer:" + names[0] + ":" + str(i) for i in range(p))
        if (set(names) & existing or set(amplitude_names) & (existing | set(names))
                or len(set(amplitude_names)) != p):
            raise ValueError("new original phase/amplitude identity collision")
        self.work.charge(32 + p * m + fresh_rows)
        certified = tuple(self.bounds(self._readout((form,))) for form in forms)
        nominal = []
        for form in forms:
            value = rational(form.constant + base._dot(form.source, self.frame.reference, self.work))
            nominal.append(rational(value + base._dot(form.phase, self.references, self.work)))
        nominal = tuple(nominal)
        nphase, nerror = len(self.phases) + m, len(self.errors) + p
        phase_indices = tuple(range(len(self.phases), nphase))
        coordinates = tuple(range(len(self.errors), nerror))
        energy = self._vector_energy(forms).pad(nphase)
        gs = tuple(form.pad(nphase, nerror) for form in forms)
        deviations = tuple(Form(rational(form.constant - bar), form.source, form.phase, form.error)
                           for form, bar in zip(gs, nominal))
        source_projection = tuple(_sum_forms(deviations, row, 0, self.work) for row in carrier)
        start = len(self.predicates)
        ranges = []
        for label, count in (("guards", 2 * m), ("caps", 4 * p),
                             ("mass", 2 * len(consumers)), ("energy", 6 * p)):
            ranges.append((label, start, start + count))
            start += count
        bank = Bank("packet:" + ",".join(names), coordinates, phase_indices, carrier,
                    consumers, mass_index, nominal, source_projection, energy,
                    certified, tuple(ranges))
        state = self._extend(phases=self.phases + tuple(Identity(name) for name in names),
                             references=self.references + tuple(int(value > 0) for value in nominal),
                             errors=self.errors + tuple(Identity(name) for name in amplitude_names),
                             banks=self.banks + (bank,))
        zero = state._zero()
        bits = tuple(Form(0, zero.source, tuple(int(j == index) for j in range(nphase)), zero.error)
                     for index in phase_indices)
        amplitudes = state._amplitude_forms(bank)
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
            active_lower = [Fraction(0)] * nphase
            active_upper = [Fraction(0)] * nphase
            inactive_lower = [Fraction(0)] * nphase
            inactive_upper = [Fraction(0)] * nphase
            lower_constant = upper_constant = Fraction(0)
            for coefficient, bar, (lo, hi), index in zip(row, nominal, certified, phase_indices):
                alo, ahi = rational(max(lo, 0) - bar), rational(max(hi, 0) - bar)
                ilo, ihi = rational(min(lo, 0) - bar), rational(min(hi, 0) - bar)
                av = (rational(coefficient * alo), rational(coefficient * ahi))
                iv = (rational(coefficient * ilo), rational(coefficient * ihi))
                active_lower[index], active_upper[index] = min(av), max(av)
                low, high = min(iv), max(iv)
                lower_constant = rational(lower_constant + low)
                upper_constant = rational(upper_constant + high)
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
        for row, amplitude, source in zip(carrier, amplitudes, source_projection):
            squared = base._dot(row, row, self.work)
            scale = (sqrt_upper(rational(maximum / squared), self.work)
                     if maximum and squared else Fraction(1))
            for left, right in ((scale, 0), (-scale, 0), (0, scale), (0, -scale),
                                (scale, -scale), (-scale, scale)):
                left, right = rational(left), rational(right)
                lhs = amplitude.scale(rational(2 * rational(left - right)), self.work)
                lhs = lhs.plus(source.scale(rational(2 * right), self.work), self.work)
                coefficients = list(energy.phase)
                difference = rational(rational(left * left) - rational(right * right))
                for value, index in zip(row, phase_indices):
                    coefficients[index] = rational(coefficients[index] +
                                                    rational(difference * rational(value * value)))
                constant = rational(energy.constant + rational(rational(right * right) * squared))
                rhs = phase_form(constant, coefficients)
                le(lhs.plus(rhs.scale(-1, self.work), self.work))
        if len(rows) != len(self.predicates) + fresh_rows:
            raise ValueError("fixed row contract was not fulfilled")
        final = state._extend(predicates=tuple(rows))
        return final, final._readout(packet[:len(consumers)])

    def _assignment(self, source, signed_phases, errors, relaxed=False):
        source, signed_phases, errors = base._qs(source), base._qs(signed_phases), base._qs(errors)
        if (len(source) != len(self.frame.bounds) or len(signed_phases) != len(self.phases)
                or len(errors) != len(self.errors)):
            raise ValueError("assignment population mismatch")
        if relaxed:
            if any(not -1 <= value <= 1 for value in signed_phases):
                raise ValueError("relaxed signed phases must be in [-1,1]")
        elif any(value not in (-1, 1) for value in signed_phases):
            raise ValueError("native phases must retain their signed binary type")
        if any(not lo <= value <= hi for value, (lo, hi) in zip(source, self.frame.bounds)):
            raise ValueError("assignment is outside the source box")
        phase01 = tuple(rational(rational(value + 1) / 2) for value in signed_phases)
        return source, phase01, errors

    def _linear_holds(self, source, phase01, errors):
        for row in self.predicates:
            value = row.form.at(source, phase01, errors, self.work)
            if (row.sense == "eq" and value != row.rhs) or (row.sense == "le" and value > row.rhs):
                return False
        return True

    def finite_rows_hold(self, source, signed_phases, errors, *, relax_phases=False):
        """Check only the declared finite outer polyhedron at one assignment."""
        try:
            if type(relax_phases) is not bool:
                return False
            assignment = self._assignment(source, signed_phases, errors, relax_phases)
            return self._linear_holds(*assignment)
        except (ValueError, TypeError, OverflowError):
            return False

    def contains(self, source, signed_phases, errors):
        """Exact signed native membership, including ALL inherited banks."""
        try:
            source, phase01, errors = self._assignment(source, signed_phases, errors)
            if not self._linear_holds(source, phase01, errors):
                return False
            for bank in self.banks:
                active = tuple(j for j, index in enumerate(bank.phase_indices) if phase01[index] == 1)
                inactive = tuple(j for j, index in enumerate(bank.phase_indices) if phase01[index] == 0)
                amplitude = tuple(errors[i] for i in bank.coordinates)
                source_value = tuple(form.at(source, phase01, errors, self.work)
                                     for form in bank.source_projection)
                complement = tuple(rational(a - b) for a, b in zip(source_value, amplitude))
                used = rational(_minimum_energy(bank.carrier, active, amplitude, self.work)
                                + _minimum_energy(bank.carrier, inactive, complement, self.work))
                available = bank.energy.at(phase01, self.work)
                if available < 0 or used > available:
                    return False
            return True
        except (ValueError, TypeError, OverflowError):
            return False

    def cost(self, *live_views):
        """Logical stored entries, not RSS, solver time, or model coverage."""
        forms = tuple(form for view in live_views for form in self._aligned(view))

        def entries(form):
            return 1 + len(form.source) + len(form.phase) + len(form.error)

        def variable_nnz(form):
            return sum(value != 0 for value in form.source + form.phase + form.error)

        predicate_entries = sum(entries(row.form) + 1 for row in self.predicates)
        carrier_entries = sum(len(row) for bank in self.banks for row in bank.carrier)
        consumer_entries = sum(len(row) for bank in self.banks for row in bank.consumers)
        projection_entries = sum(entries(form) for bank in self.banks for form in bank.source_projection)
        energy_entries = sum(1 + len(bank.energy.phase) for bank in self.banks)
        decoder_entries = len(self.frame.decoder_bias) + sum(len(row) for row in self.frame.decoder_matrix)
        result = dict(source_factors=len(self.frame.sources), phase_factors=len(self.phases),
            residual_factors=len(self.errors), banks=len(self.banks),
            predicate_rows=len(self.predicates), predicate_coefficients=predicate_entries,
            predicate_nnz=sum(variable_nnz(row.form) for row in self.predicates),
            predicate_rhs_nnz=sum(rational(row.rhs - row.form.constant) != 0 for row in self.predicates),
            carrier_coefficients=carrier_entries, consumer_coefficients=consumer_entries,
            source_projection_coefficients=projection_entries, energy_coefficients=energy_entries,
            bank_coordinate_entries=sum(len(bank.coordinates) for bank in self.banks),
            bank_phase_entries=sum(len(bank.phase_indices) for bank in self.banks),
            bank_reference_entries=sum(len(bank.reference) for bank in self.banks),
            bank_bound_entries=sum(2 * len(bank.preactivation_bounds) for bank in self.banks),
            bank_row_range_entries=sum(3 * len(bank.row_ranges) for bank in self.banks),
            decoder_coefficients=decoder_entries, source_bound_entries=2 * len(self.frame.bounds),
            source_reference_entries=len(self.frame.reference), phase_reference_entries=len(self.references),
            lineage_entries=len(self.lineage), live_readouts=len(forms),
            live_readout_coefficients=sum(entries(form) for form in forms),
            live_readout_nnz=sum(variable_nnz(form) + int(form.constant != 0) for form in forms),
            complete_physical_qualification=False, terminal_is_outer_approximation=True)
        self.work.charge(32 + predicate_entries + carrier_entries + consumer_entries
                         + projection_entries + energy_entries + decoder_entries
                         + result["live_readout_coefficients"])
        result["algebra_work_used"] = self.work.used
        return result
