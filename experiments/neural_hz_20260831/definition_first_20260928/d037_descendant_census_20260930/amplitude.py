"""Default-off census of conditional descendant-amplitude inequalities.

This is bounded rational arithmetic, not a verifier or a source-certificate
authority.  The caller must bind every supplied interval to the same actual
receiver h=b+sum(a_i*q_i), retain its original gates and bits, and establish
q_i>=0 and q_i<=Q_i*alpha_i in the comparison system.  The residual bounds
below subtract a term from the SAME independent interval sum; they do not
subtract from an unrelated, possibly correlated receiver bound.

Each returned row-mask bit means UNRESOLVED by a sufficient redundancy proof,
not proven nonredundancy, a new neural phase, or a verification improvement.
No row, variable, gate, solver call, or concrete execution is created here.
Canonical padding slots remain present but receive mask zero and no bit.

The inherited ledger counts bounded scalar operations and container visits,
not bit operations or wall time.  Every rational endpoint/result is checked
against its (at most 512-bit) limit.  Parsing, original-source authentication,
physical-memory accounting, and evidence encoding remain caller costs.  A
successful call uses at most 180*n+160 ledger units for n canonical slots;
fixed-loop visits and container work are conservatively prepaid before use.
"""

from experiments.neural_hz_20260831.definition_first_20260928.d025_interval_capacity_20260930 import interval_capacity as arithmetic


KernelError = arithmetic.KernelError
BudgetExceeded = arithmetic.BudgetExceeded
KernelDisabled = arithmetic.kernel.KernelDisabled
WorkBudget = arithmetic.WorkBudget
MAX_SLOTS = arithmetic.MAX_SLOTS
ZERO = arithmetic.ZERO


def _add(left, right, budget):
    """Prepay scalar addition and the inherited bounded endpoint check."""
    budget.charge(2)
    return arithmetic._fraction(left + right, budget.max_bits)


def compile_receiver(weights, bias, bounds, real_slots, *, enabled=False, budget=None):
    """Return a conditional census dictionary, or None without opt-in.

    ``weights`` and PRE-ReLU ``bounds`` are equal-length immutable tuples of
    exact Fraction intervals.  ``real_slots`` is an equally long tuple of
    exact bools; a False slot must have preactivation bounds exactly (0, 0).
    All slots are validated and charged, even zero weights and padding.

    ``ordinary_bounds`` is the receiver interval. ``source_caps`` contains
    each Q_i=max(0,u_i), and ``residual_bounds`` contains each all-but-one
    (L_i,H_i), including padding. ``row_masks`` uses bits 0..3 for unresolved
    D036 rows R1..R4.  The row coordinate-nnz upper bounds are (2,3,3,4).
    Counts named ``zero_coeff_slots`` and ``interval_cross_zero_slots`` range
    over ALL canonical slots, including padding: respectively a_i=[0,0] and
    the strict condition lower(a_i)<0<upper(a_i).

    Every row classification is conditional on the original receiver/source
    equalities and bounds and q_i<=Q_i*alpha_i.  This function does not inspect
    or assert a production HZ binding.  No original coefficient is replaced
    by an interval endpoint or midpoint in the actual network.
    """
    if type(enabled) is not bool:
        raise KernelError("enabled must be a bool")
    if not enabled:
        return None
    budget = arithmetic._active_budget(budget)
    budget.charge(20)
    if (type(weights) is not tuple or type(bounds) is not tuple
            or type(real_slots) is not tuple):
        raise KernelError("immutable tuples of intervals and bool slots required")
    width = len(weights)
    if width > MAX_SLOTS:
        raise KernelError("canonical slot limit exceeded")
    if len(bounds) != width or len(real_slots) != width:
        raise KernelError("weights, bounds, or real-slot shape mismatch")
    total_lower, total_upper = arithmetic._interval(bias, budget)

    # Prepay containers and every loop/access/store/branch/counter operation.
    # Arithmetic helpers separately prepay exact arithmetic and bit inspection.
    budget.charge(12 + 48 * width)
    source_caps, contributions, checked_weights = [], [], []
    valid_slots = 0
    zero_coeff_slots = 0
    interval_cross_zero_slots = 0
    for weight, bound, real in zip(weights, bounds, real_slots):
        weight_lower, weight_upper = arithmetic._interval(weight, budget)
        source_lower, source_upper = arithmetic._interval(bound, budget)
        if type(real) is not bool:
            raise KernelError("real_slots entries must be exact bools")
        if not real and (source_lower != ZERO or source_upper != ZERO):
            raise KernelError("padding requires exactly zero preactivation bounds")
        activation_lower = arithmetic._positive(source_lower, budget)
        activation_upper = arithmetic._positive(source_upper, budget)
        contribution_lower = arithmetic._mul(
            weight_lower,
            activation_upper if weight_lower < ZERO else activation_lower,
            budget,
        )
        contribution_upper = arithmetic._mul(
            weight_upper,
            activation_lower if weight_upper < ZERO else activation_upper,
            budget,
        )
        total_lower = _add(total_lower, contribution_lower, budget)
        total_upper = _add(total_upper, contribution_upper, budget)
        source_caps.append(activation_upper)
        contributions.append((contribution_lower, contribution_upper))
        checked_weights.append((weight_lower, weight_upper))
        if real:
            valid_slots += 1
        if weight_lower == ZERO and weight_upper == ZERO:
            zero_coeff_slots += 1
        if weight_lower < ZERO and weight_upper > ZERO:
            interval_cross_zero_slots += 1

    # The full traversal is prepaid even when padding or an easy proof avoids
    # some helper arithmetic.  No data-dependent group or phase is selected.
    budget.charge(16 + 80 * width)
    residual_bounds, row_masks = [], []
    row_unresolved_counts = [0, 0, 0, 0]
    potential_edges = 0
    potential_rows = 0
    potential_coordinate_nnz = 0
    for weight, contribution, cap, real in zip(
            checked_weights, contributions, source_caps, real_slots):
        lower = arithmetic._sub(total_lower, contribution[0], budget)
        upper = arithmetic._sub(total_upper, contribution[1], budget)
        if lower > upper:
            raise KernelError("invalid all-but-one residual interval")
        residual_bounds.append((lower, upper))
        if not real:
            row_masks.append(0)
            continue

        lower_coefficient_cap = arithmetic._mul(weight[0], cap, budget)
        upper_coefficient_cap = arithmetic._mul(weight[1], cap, budget)

        # Sufficient proofs of redundancy against the original epigraphs,
        # interval source premises, and q<=Q*alpha.  Failed proofs stay unknown.
        redundant_one = lower <= ZERO
        if not redundant_one:
            negative_capacity = (lower_coefficient_cap
                                 if lower_coefficient_cap < ZERO else ZERO)
            redundant_one = _add(lower, negative_capacity, budget) >= ZERO

        redundant_two = lower >= ZERO
        if not redundant_two:
            redundant_two = _add(lower_coefficient_cap, lower, budget) <= ZERO

        redundant_three = upper >= ZERO
        if not redundant_three:
            positive_capacity = (upper_coefficient_cap
                                 if upper_coefficient_cap > ZERO else ZERO)
            redundant_three = _add(upper, positive_capacity, budget) <= ZERO

        redundant_four = upper <= ZERO
        if not redundant_four:
            # Equivalent to -ahi*Q-H<=0, without an extra exact negation.
            redundant_four = _add(upper_coefficient_cap, upper, budget) >= ZERO

        mask = 0
        edge_rows = 0
        edge_nnz = 0
        if not redundant_one:
            mask |= 1
            row_unresolved_counts[0] += 1
            edge_rows += 1
            edge_nnz += 2
        if not redundant_two:
            mask |= 2
            row_unresolved_counts[1] += 1
            edge_rows += 1
            edge_nnz += 3
        if not redundant_three:
            mask |= 4
            row_unresolved_counts[2] += 1
            edge_rows += 1
            edge_nnz += 3
        if not redundant_four:
            mask |= 8
            row_unresolved_counts[3] += 1
            edge_rows += 1
            edge_nnz += 4
        # L/H signs exclude R1+R2 and R3+R4.  R2+R4 would require
        # alo>0>ahi, impossible for an ordered weight interval; hence <=6 nnz.
        if not 0 <= mask < 16 or edge_rows > 2 or edge_nnz > 6:
            raise KernelError("conditional row-count invariant violated")
        row_masks.append(mask)
        if mask:
            potential_edges += 1
            potential_rows += edge_rows
            potential_coordinate_nnz += edge_nnz

    # Prepay all tuple traversals, result containers, and final scalar counts.
    budget.charge(64 + 3 * width)
    return {
        "ordinary_bounds": (total_lower, total_upper),
        "source_caps": tuple(source_caps),
        "residual_bounds": tuple(residual_bounds),
        "row_masks": tuple(row_masks),
        "canonical_slots": width,
        "valid_slots": valid_slots,
        "padding_slots": width - valid_slots,
        "potential_edges": potential_edges,
        "potential_rows": potential_rows,
        "potential_coordinate_nnz": potential_coordinate_nnz,
        "row_unresolved_counts": tuple(row_unresolved_counts),
        "zero_coeff_slots": zero_coeff_slots,
        "interval_cross_zero_slots": interval_cross_zero_slots,
    }
