"""Exact integer-only Conv kernel preparation; full inverse and native domain."""
import numpy as np

from experiments.neural_hz_20260831.c85_exact_tile_algebra_v1 import recover_filter
from experiments.neural_hz_20260831.c88_inline_tile_v1 import Prepared


def _separable(aligned):
    """H @ aligned @ H.T, with exactly18+24 nontrivial integer operations."""
    a, b, c = aligned[..., 0, :], aligned[..., 1, :], aligned[..., 2, :]
    first = np.stack((2*a, a+b+c, a-b+c, 2*c), axis=-2)
    a, b, c = first[..., 0], first[..., 1], first[..., 2]
    return np.stack((2*a, a+b+c, a-b+c, 2*c), axis=-1)


def prepare_words(weights, *, pool, enabled=False):
    """Fresh full-kernel checks and density; no externally supplied receipt.

    C85's shift<=35 gives |aligned|<2^59 and |numerator|<9*2^59<2^63.
    Returned powers lie in [-151,102]. Thus numerator*2**power is finite
    normal binary64 whenever the odd significand has at most53 bits. Final
    HZ row scaling/window/pivot/box obligations remain in C88.construct.
    """
    if not enabled:
        return None
    w = np.asarray(weights)
    if (w.dtype != np.float32 or w.ndim != 4 or w.shape[-2:] != (3, 3)
            or w.shape[0] == 0 or w.shape[1] == 0):
        raise ValueError('ordinary nonempty original binary32 3x3 filters required')
    kernels = int(w.shape[0]*w.shape[1])
    pool.charge('c95_sparse_integer_filter_and_complete_inverse', 302*kernels+4*int(w.size))
    pool.charge('c95_original_density_from_existing_bit_mask', int(w.size))
    bits = w.view(np.uint32)
    exponent = ((bits >> 23) & 255).astype(np.int32)
    fraction = bits & np.uint32((1 << 23)-1)
    if np.any(exponent == 255) or np.any((exponent == 0) & (fraction != 0)):
        raise ValueError('nonfinite/subnormal source outside ordinary certified integer domain')
    nonzero = exponent != 0
    original_nonzero = int(np.count_nonzero(nonzero))
    mantissa = (fraction | np.uint32(1 << 23)).astype(np.int64)
    mantissa[~nonzero] = 0
    mantissa[(bits >> 31) != 0] *= -1
    powers = exponent-127-23
    minimum = np.where(nonzero, powers, 300).min(axis=(-2, -1))
    minimum = np.where(minimum == 300, 0, minimum)
    shifts = np.where(nonzero, powers-minimum[..., None, None], 0)
    if np.any(shifts > 35) or np.any(shifts < 0):
        raise ValueError('complete signed-int64 transform envelope not established')
    aligned = np.left_shift(mantissa, shifts)
    numerator = _separable(aligned)
    if not np.array_equal(recover_filter(numerator), 4*aligned):
        raise ValueError('independent original-kernel recovery differs')
    absolute = np.abs(numerator).astype(np.uint64)
    lowbit = absolute & (~absolute+np.uint64(1))
    odd = absolute//np.where(lowbit == 0, np.uint64(1), lowbit)
    representable = odd <= np.uint64(1 << 53)
    transformed_nonzero = int(np.count_nonzero(numerator))
    report = dict(kernels=kernels, original_weights=int(w.size),
        original_nonzero=original_nonzero, original_dense=original_nonzero == int(w.size),
        transformed_coefficients=int(numerator.size), transformed_nonzero=transformed_nonzero,
        maximum_alignment_shift=int(shifts.max(initial=0)),
        exact_binary64_failures=int(np.count_nonzero(~representable)),
        independent_original_integer_kernel_recovery=True, complete_signed_int64_envelope=True,
        binary64_finite_normal_scaling_proved_by_envelope=True,
        all_coefficients_exact_binary64=bool(representable.all()),
        unused_float_filter_and_channel_gauge_materialized=False,
        complete_HZ_row_window_proved=False, normalizing_pivot_and_RHS_checked=False,
        floating_channel_gauge_or_window_checked=False,
        transformed_density_from_same_complete_count=True,
        dense_forward_arithmetic_per_kernel=140, sparse_forward_arithmetic_per_kernel=42)
    if not report['all_coefficients_exact_binary64']:
        return report, None
    words = dict(numerator=numerator, exponent=minimum-2)
    return report, Prepared(words, transformed_nonzero == int(numerator.size))
