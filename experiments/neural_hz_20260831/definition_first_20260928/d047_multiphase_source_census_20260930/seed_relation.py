"""Default-off interval seeds for the frozen D046 mathematical component.

The caller certifies one common source frame, the original source ordinals,
all interval premises, and (only when requested) actual consumer identity.
An ordinal is NOT a verified native phase column.  Stable-source tokens are
not manufactured; any original stable/zero-phase bits stay in the caller's
unchanged HZ, as do all predicates, continuous identities and the decoder.

Every canonical input slot is inspected and charged, including padding.
Interval arithmetic uses the existing D015 charged kernel.  In addition,
80*n+256 work units prepay this wrapper's slot traversal, identity maps,
counts and input/output list administration.  Exact scalar operations and
endpoint checks charge separately.  Unmetered D046 calls are prepaid before
entry: frame/value/phase declarations use 64+192*k units; a form uses
128+160*s+16*s*ceil(log2(s+1)); a positive-part operation uses
256+256*s+16*s*ceil(log2(s+1)).  Here s is ACTUAL input support, not a fanin
or MAX_SUPPORT surrogate.  The linear coefficients cover token/frame and
fraction checks, hash-map/container operations, canonical filtering, and
the bounded loops in frozen D046; the logarithmic term covers key extraction,
sorting comparisons and moves.  These prepaid charges are ADDITIONAL to all
D015 arithmetic charges, not a replacement for them.  Work is abstract scalar
and container work, not bytecode/instruction or wall-time accounting.

192*n+8192 numeric entries is a conservative temporary reserve for the
checked-slot maps, eight forms, D046 registries and overlapping list/tuple
copies (a Fraction occurrence may include its numerator and denominator).
The caller must separately ledger actual retained inputs, returned plain
roots, evidence, parser/model roots, RSS and all other live work.  This is not
a whole-system physical-memory or native/GPU qualification.

The last-anchor gaps are sufficient differences between algebraic envelopes
on the independent original-bit cube.  They are NOT feasible-network gaps,
nonredundancy witnesses, new solver results or formal capability gains.
"""

from fractions import Fraction

from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import shield_kernel_v1 as k
from experiments.neural_hz_20260831.definition_first_20260928.d046_multiphase_envelopes_20260930 import multiphase as mp


KernelError = k.KernelError
BudgetExceeded = k.BudgetExceeded
KernelDisabled = k.KernelDisabled
WorkBudget = k.WorkBudget
MAX_SLOTS = 65_536
MAX_BITS = 512
TRANSIENT_ENTRIES_PER_SLOT = 192
TRANSIENT_ENTRY_BASE = 8192
ZERO = Fraction(0)
ZERO_INTERVAL = (ZERO, ZERO)
SEED_NAMES = ("delta_lower", "delta_upper", "w_lower", "w_upper")
AFTER_NAMES = ("difference_lower", "difference_upper", "companion_lower", "companion_upper")


def _active_budget(budget):
    if budget is None:
        budget = WorkBudget(enabled=True)
    if not isinstance(budget, WorkBudget):
        raise KernelError("a D015 WorkBudget is required")
    if (type(budget.enabled) is not bool or type(budget.limit) is not int
            or not 0 <= budget.limit <= 256_000_000
            or type(budget.max_bits) is not int or not 1 <= budget.max_bits <= MAX_BITS
            or type(budget.used) is not int or not 0 <= budget.used <= budget.limit):
        raise KernelError("invalid work ledger")
    budget.charge(0)
    return budget


def _fraction(value, budget):
    budget.charge(3)
    if type(value) is not Fraction:
        raise KernelError("exact Fraction endpoints and coefficients are required")
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > budget.max_bits:
        raise KernelError("rational bit limit exceeded")
    return value


def _interval(value, budget):
    budget.charge(3)
    if type(value) is not tuple or len(value) != 2:
        raise KernelError("an immutable two-endpoint interval is required")
    lo, hi = _fraction(value[0], budget), _fraction(value[1], budget)
    if lo > hi:
        raise KernelError("reversed interval")
    return value


def _add(left, right, budget):
    budget.charge(2)
    return _fraction(left + right, budget)


def _neg(value, budget):
    budget.charge(2)
    return _fraction(-value, budget)


def _mp_charge(support, positive, budget):
    # bit_length(s) == ceil(log2(s+1)) for every integer s >= 0.
    budget.charge((256 if positive else 128)
                  + (256 if positive else 160) * support
                  + 16 * support * support.bit_length())


def _make_form(frame, bias, terms, budget):
    _mp_charge(len(terms), False, budget)
    try:
        return mp.form(frame, "phase", bias, tuple(terms), enabled=True)
    except mp.KernelError as error:
        raise KernelError(str(error)) from error


def _positive(source, upper, budget):
    _mp_charge(len(source.terms), True, budget)
    try:
        operation = mp.positive_majorant if upper else mp.positive_minorant
        return operation(source, enabled=True)
    except mp.KernelError as error:
        raise KernelError(str(error)) from error


def _plain(source, budget, negate=False):
    budget.charge(16 + 12 * len(source.terms))
    bias = _neg(source.bias, budget) if negate else _fraction(source.bias, budget)
    terms = []
    for phase, coefficient in source.terms:
        amount = _neg(coefficient, budget) if negate else _fraction(coefficient, budget)
        terms.append((phase.ordinal, amount))
    return bias, tuple(terms)


def _clamp_metrics(source, output, budget):
    """Output may be negated for a lower bound; its magnitudes are unchanged."""
    budget.charge(24 + 12 * (len(source.terms) + len(output.terms)))
    lower = upper = _fraction(source.bias, budget)
    for _, coefficient in source.terms:
        if coefficient < ZERO:
            lower = _add(lower, coefficient, budget)
        else:
            upper = _add(upper, coefficient, budget)
    last = source.terms[-1][0].ordinal if source.terms else None
    gap = ZERO
    for phase, coefficient in output.terms:
        if phase.ordinal != last:
            magnitude = _neg(coefficient, budget) if coefficient < ZERO else coefficient
            gap = _add(gap, magnitude, budget)
    return lower < ZERO < upper, last, gap


def compile_pair(left_weights, left_bias, right_weights, right_bias,
                 source_bounds, source_ordinals, *, same_consumer=False,
                 enabled=False, budget=None):
    """Compile fixed, caller-certified consumers h and w over one source frame.

    Weight collections, source bounds and source ordinals are tuples.  Biases
    and every bound/weight are exact Fraction intervals.  None ordinals denote
    literal padding and require bound (0,0); repeated real ordinals require
    identical certified bounds and reuse their original phase.  Contributions
    at repeated slots are conservatively combined, not treated as independent
    phase identities.  same_consumer=True requires identical input interval
    arrays/biases AND the caller's external real-consumer identity certificate.
    Equality of interval data without this flag never proves cancellation.

    Returned forms are (Fraction bias, tuple((original_ordinal, Fraction))).
    No D046 token, dataclass or hidden certificate root escapes this function.
    Disabled calls inspect no task inputs or budget.  Rejection is not UNSAT.
    """
    if type(enabled) is not bool:
        raise KernelError("enabled must be bool")
    if not enabled:
        return None
    budget = _active_budget(budget)
    start_work = budget.used
    budget.charge(32)
    if type(same_consumer) is not bool:
        raise KernelError("same_consumer must be bool")
    collections = (left_weights, right_weights, source_bounds, source_ordinals)
    if any(type(value) is not tuple for value in collections):
        raise KernelError("immutable input tuples are required")
    width = len(source_bounds)
    if width > MAX_SLOTS or any(len(value) != width for value in collections):
        raise KernelError("inconsistent or oversized canonical source population")
    budget.charge(256 + 80 * width)
    left_bias = _interval(left_bias, budget)
    right_bias = _interval(right_bias, budget)
    counts = dict(canonical_slots=width, padding_slots=0, crossing_slots=0,
                  distinct_crossing_phases=0, stable_active_slots=0,
                  stable_inactive_slots=0, zero_boundary_slots=0)
    known_bounds = {}
    crossing_ordinals = {}
    checked = []
    for left, right, bounds, ordinal in zip(*collections):
        left, right = _interval(left, budget), _interval(right, budget)
        bounds = _interval(bounds, budget)
        if ordinal is None:
            if bounds != ZERO_INTERVAL:
                raise KernelError("padding requires a literal zero source")
            counts["padding_slots"] += 1
        else:
            if (type(ordinal) is not int or ordinal < 0
                    or ordinal.bit_length() > budget.max_bits):
                raise KernelError("bounded nonnegative original source ordinals required")
            if ordinal in known_bounds and known_bounds[ordinal] != bounds:
                raise KernelError("one source identity has inconsistent interval premises")
            known_bounds[ordinal] = bounds
            if bounds[0] < ZERO < bounds[1]:
                counts["crossing_slots"] += 1
                crossing_ordinals[ordinal] = None
            elif bounds[0] > ZERO:
                counts["stable_active_slots"] += 1
            elif bounds[1] < ZERO:
                counts["stable_inactive_slots"] += 1
            else:
                counts["zero_boundary_slots"] += 1
        checked.append((left, right, bounds, ordinal))
    if same_consumer and (left_weights != right_weights or left_bias != right_bias):
        raise KernelError("self-pair premise requires matching interval data")
    counts["distinct_crossing_phases"] = len(crossing_ordinals)

    budget.charge(64 + 192 * len(crossing_ordinals))
    try:
        frame = mp.make_frame("caller-certified source frame", enabled=True)
        for ordinal in crossing_ordinals:
            value = mp.make_value(frame, "original source activation", enabled=True)
            crossing_ordinals[ordinal] = mp.make_phase(
                value, ordinal, "original active indicator", enabled=True)
    except mp.KernelError as error:
        raise KernelError(str(error)) from error

    delta_bias = (ZERO_INTERVAL if same_consumer
                  else k.add(left_bias, k.neg(right_bias, budget), budget))
    biases = [delta_bias[0], delta_bias[1], right_bias[0], right_bias[1]]
    terms = [[], [], [], []]
    ordinary_h, ordinary_w = left_bias, right_bias
    plain_difference = delta_bias
    for left, right, bounds, ordinal in checked:
        activation = k.nonnegative_part(bounds, budget)
        delta_weight = (ZERO_INTERVAL if same_consumer
                        else k.add(left, k.neg(right, budget), budget))
        left_product = k.mul(left, activation, budget)
        right_product = k.mul(right, activation, budget)
        delta_product = k.mul(delta_weight, activation, budget)
        ordinary_h = k.add(ordinary_h, left_product, budget)
        ordinary_w = k.add(ordinary_w, right_product, budget)
        plain_difference = k.add(plain_difference, delta_product, budget)
        contributions = (delta_product[0], delta_product[1],
                         right_product[0], right_product[1])
        if bounds[0] < ZERO < bounds[1]:
            phase = crossing_ordinals[ordinal]
            for index, contribution in enumerate(contributions):
                if contribution != ZERO:
                    terms[index].append((phase, contribution))
        else:
            for index, contribution in enumerate(contributions):
                biases[index] = _add(biases[index], contribution, budget)

    budget.charge(256)
    seeds = tuple(_make_form(frame, bias, items, budget)
                  for bias, items in zip(biases, terms))
    negative_lower_terms = []
    budget.charge(16 + 8 * len(seeds[0].terms))
    for phase, coefficient in seeds[0].terms:
        negative_lower_terms.append((phase, _neg(coefficient, budget)))
    negative_lower = _make_form(frame, _neg(seeds[0].bias, budget),
                                negative_lower_terms, budget)
    clamp_inputs = (negative_lower, seeds[1], seeds[2], seeds[3])
    clamped = tuple(_positive(source, index != 2, budget)
                    for index, source in enumerate(clamp_inputs))
    result = {name: _plain(source, budget)
              for name, source in zip(SEED_NAMES, seeds)}
    result.update({name: _plain(source, budget, negate=index == 0)
                   for index, (name, source) in enumerate(zip(AFTER_NAMES, clamped))})

    budget.charge(256)
    supports = {name: len(result[name][1]) for name in SEED_NAMES + AFTER_NAMES}
    hinge_crossing, last_anchor_ordinal, last_anchor_gap, last_anchor_strict = {}, {}, {}, {}
    for name, source, output in zip(AFTER_NAMES, clamp_inputs, clamped):
        hinge, last, gap = _clamp_metrics(source, output, budget)
        hinge_crossing[name] = hinge
        last_anchor_ordinal[name] = last
        last_anchor_gap[name] = gap
        last_anchor_strict[name] = gap > ZERO
    result.update(ordinary_h=ordinary_h, ordinary_w=ordinary_w,
                  plain_shared_difference=plain_difference, source_counts=counts,
                  supports=supports, hinge_crossing=hinge_crossing,
                  last_anchor_ordinal=last_anchor_ordinal,
                  last_anchor_gap=last_anchor_gap,
                  last_anchor_strict=last_anchor_strict,
                  nonconstant_relations=sum(supports[name] > 0 for name in AFTER_NAMES),
                  hinge_crossing_relations=sum(hinge_crossing.values()),
                  last_anchor_strict_relations=sum(last_anchor_strict.values()),
                  potential_coordinate_nnz=6 + sum(supports[name] for name in AFTER_NAMES),
                  binding_mathematical_only=True,
                  actual_phase_column_binding_verified=False,
                  same_consumer_certified=same_consumer,
                  work_used=budget.used - start_work)
    return result
