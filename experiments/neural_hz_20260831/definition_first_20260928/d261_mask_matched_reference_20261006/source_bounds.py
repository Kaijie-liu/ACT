"""D261: an ordinary, mask-matched reference bound; no new HZ relation.

The frozen D259 arithmetic/types below are shared by identity, including its
trusted in-memory ownership token.  This module does not reload that module,
recompute candidate e/tau, or replace an original BN error carrier.

For each output o and actual kernel tap t, we enclose
    Tlo[o,t] = sum_k min(W[o,k,t]*L[k], W[o,k,t]*U[k]),
    Thi[o,t] = sum_k max(W[o,k,t]*L[k], W[o,k,t]*U[k]).
Each original weight is visited for the point-times-box operation once; two
gamma_n/absolute-sum reductions then produce all [Co,3,3] contributions.
The nine masks reuse only these small arrays.  Bias and the SAME original
nominal BN coefficients and error E are retained.  In particular, E is not
re-estimated after masking.  Ufair=min(old_U, max(0, masked_g_upper)) is a
valid ordinary old bound, not evidence of QG strengthening.

All allocations/scans use the caller's one cumulative D259-compatible meter.
The main weight-dependent cost is 13*Co*Ci*9 work; mask reductions cost
O(49*Co), not nine complete kernel scans.  Directed primitive endpoints and
the frozen certified reduction also account for floating error/underflow;
published endpoints keep the frozen outward dyadic encoding.

The caller supplies the actual connected Add/Conv/BN records.  A numerical
zero-hull check binds the Add range to the Conv ledger, but neither that check
nor an ownership token certifies ONNX provenance or recovers the lost identity
of an unpadded Add object.  This is the same declared-source, synchronous,
non-adversarial in-memory contract as D259, not a serialized certificate.
"""

from dataclasses import dataclass

import numpy as np

from experiments.neural_hz_20260831.definition_first_20260928.d259_shared_difference_source_20261006 import source_bounds as base


# Explicit aliases keep every old observer operation on the frozen arithmetic
# path and preserve the exact original class/token identity.
Rejected = base.Rejected
Interval = base.Interval
ConvLedger = base.ConvLedger
BNCarrier = base.BNCarrier
interval = base.interval
point = base.point
from_rationals = base.from_rationals
add = base.add
sub = base.sub
mul = base.mul
neg = base.neg
scale_half = base.scale_half
sum_axis = base.sum_axis
take = base.take
relu = base.relu
midpoint = base.midpoint
conv_channel = base.conv_channel
peel_conv_selected = base.peel_conv_selected
bn_carrier = base.bn_carrier
fused_h = base.fused_h
fused_offset = base.fused_offset


@dataclass(frozen=True)
class MaskedChildBounds:
    masks: tuple
    tap_bounds: Interval
    preactivation: Interval
    old_upper: np.ndarray
    upper: np.ndarray


def _masks(budget):
    # The three row/column states are top/interior/bottom and
    # left/interior/right.  The caller still matches masks, not numeric IDs.
    base._pay(budget, 640, 320)
    return tuple(
        tuple(tuple(not (row == 0 and ky == 0)
                    and not (row == 2 and ky == 2)
                    and not (column == 0 and kx == 0)
                    and not (column == 2 and kx == 2)
                    for kx in range(3)) for ky in range(3))
        for row in range(3) for column in range(3)
    )


def masked_child_bounds(ledger, add_bounds, carrier, *, budget, enabled=False):
    """Return all nine complete output-channel bounds, explicitly opt-in.

    Shapes: tap_bounds [Co,3,3], preactivation/upper [9,Co], old_upper [Co].
    The full symmetric-pad-one 3x3 ledger is required; per-mask ledgers and
    independently reconstructed/mutated ownership records are not accepted.
    Default-off returns None without examining source data or the budget.
    """
    if not base._on(enabled):
        return None
    if type(ledger) is not ConvLedger or ledger._token is not base._TOKEN:
        raise Rejected("the original owned outer Conv ledger is required")
    if type(carrier) is not BNCarrier or carrier._token is not base._TOKEN:
        raise Rejected("the original owned outer BN carrier is required")
    base._pay(budget, 96, 48)
    weights, bias = ledger.weights, ledger.bias
    if (type(weights) is not np.ndarray or weights.dtype != np.float64
            or weights.ndim != 4 or weights.shape[2:] != (3, 3)
            or weights.flags.writeable or ledger.padding != (1, 1, 1, 1)
            or ledger.valid_mask is not None):
        raise Rejected("one complete read-only pad-one 3x3 kernel is required")
    co, ci = weights.shape[:2]
    if co <= 0 or ci <= 0:
        raise Rejected("nonempty complete Conv channels are required")
    source = base._check(add_bounds, budget)
    original = base._check(ledger.input_bounds, budget)
    old_bn = base._check(carrier.bounds, budget)
    if source.lo.shape != (ci,) or original.lo.shape != (ci,):
        raise Rejected("complete original Add channel bounds are required")
    nominal_a, nominal_b, error = carrier.nominal_a, carrier.nominal_b, carrier.error
    if (old_bn.lo.shape != (co,)
            or any(type(v) is not np.ndarray or v.dtype != np.float64
                   or v.shape != (co,) or v.flags.writeable
                   for v in (bias, nominal_a, nominal_b, error))):
        raise Rejected("complete read-only original Conv/BN parameters are required")
    # D259's pad-one ledger contains the zero hull of its supplied Add box.
    # This necessary compatibility check does not prove semantic source ID.
    base._pay(budget, 6 * ci + 16, 4 * ci + 16)
    padded_lo = np.minimum(source.lo, 0.0)
    padded_hi = np.maximum(source.hi, 0.0)
    if (not np.array_equal(padded_lo, original.lo)
            or not np.array_equal(padded_hi, original.hi)):
        raise Rejected("Add range zero hull differs from the original Conv ledger")

    masks = _masks(budget)
    base._pay(budget, 18 * co + 48, 18 * co + 48)
    tap_lo = np.empty((co, 3, 3), dtype=np.float64)
    tap_hi = np.empty((co, 3, 3), dtype=np.float64)
    # Broadcasts are views, not replicated source columns or copied arrays.
    base._pay(budget, 16, 16)
    source_lo = np.broadcast_to(source.lo, (co, ci))
    source_hi = np.broadcast_to(source.hi, (co, ci))
    for ky in range(3):
        for kx in range(3):
            base._pay(budget, 16, 16)
            point_weights = weights[:, :, ky, kx]
            lower, upper = base._fast_point_mul(point_weights, source_lo, source_hi, budget)
            contribution = base._sum_axis(Interval(lower, upper), 1, False, budget)
            base._pay(budget, 2 * co + 8, 8)
            tap_lo[:, ky, kx] = contribution.lo
            tap_hi[:, ky, kx] = contribution.hi
    taps = base._finish(tap_lo, tap_hi, budget)

    base._pay(budget, 19 * co + 48, 19 * co + 48)
    pre_lo = np.empty((9, co), dtype=np.float64)
    pre_hi = np.empty((9, co), dtype=np.float64)
    negative_error = np.negative(error)
    flat_lo, flat_hi = taps.lo.reshape(co, 9), taps.hi.reshape(co, 9)
    for mask_index, mask in enumerate(masks):
        base._pay(budget, 32, 24)
        indices = tuple(index for index in range(9) if mask[index // 3][index % 3])
        size = co * len(indices)
        base._pay(budget, 2 * size + 16, 2 * size + 16)
        selected_lo = np.take(flat_lo, indices, axis=1)
        selected_hi = np.take(flat_hi, indices, axis=1)
        summed = base._sum_axis(Interval(selected_lo, selected_hi), 1, False, budget)
        lower, upper = summed.lo, summed.hi
        # The complete original Conv bias is added before the nominal BN.
        base._pay(budget, 4 * co + 8, 8)
        np.add(lower, bias, out=lower)
        np.add(upper, bias, out=upper)
        np.nextafter(lower, -np.inf, out=lower)
        np.nextafter(upper, np.inf, out=upper)
        lower, upper = base._fast_point_mul(nominal_a, lower, upper, budget)
        base._pay(budget, 8 * co + 8, 8)
        for lo_term, hi_term in ((nominal_b, nominal_b), (negative_error, error)):
            np.add(lower, lo_term, out=lower)
            np.add(upper, hi_term, out=upper)
            np.nextafter(lower, -np.inf, out=lower)
            np.nextafter(upper, np.inf, out=upper)
        lower = base._round(lower, False, budget)
        upper = base._round(upper, True, budget)
        base._pay(budget, 2 * co + 8, 8)
        pre_lo[mask_index] = lower
        pre_hi[mask_index] = upper
    preactivation = base._finish(pre_lo, pre_hi, budget)
    base._pay(budget, 19 * co + 32, 19 * co + 32)
    old_upper = np.maximum(old_bn.hi, 0.0)
    fair_upper = np.minimum(np.maximum(preactivation.hi, 0.0), old_upper[None, :])
    old_upper.setflags(write=False)
    fair_upper.setflags(write=False)
    return MaskedChildBounds(masks, taps, preactivation, old_upper, fair_upper)
