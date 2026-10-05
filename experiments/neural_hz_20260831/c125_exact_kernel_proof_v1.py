"""Independent complete F4 kernel contraction with source-owned numeric evidence.

No constructor transform, bit decoder, inverse or constants are imported.
Original values are decoded through exact integer ratios, restored to their
normal binary32 24-bit words, and contracted directly against independent
literal tensor coefficients.  ALL36 coefficients are rebuilt for EVERY kernel.
This is fresh preparation on each call, not a reusable receipt or a cache.
"""

from fractions import Fraction as F
import math

import numpy as np


_GN = ((1, 0, 0), (-1, -1, -1), (-1, 1, -1),
       (1, 2, 4), (1, -2, 4), (0, 0, 1))
_TERMS = tuple(tuple((i, j, _GN[a][i]*_GN[b][j])
                     for i in range(3) for j in range(3)
                     if _GN[a][i] and _GN[b][j])
               for a in range(6) for b in range(6))
HEADER_FEE = 4096
SOURCE_CUSTODY_FEE = 32*9
SOURCE_DECODE_FEE = 128*9
ALIGNMENT_FEE = 32*9
DIRECT_CONTRACTION_FEE = 12*196+8*160
MATERIALIZATION_FEE = 16*36
KERNEL_FEE = (SOURCE_CUSTODY_FEE+SOURCE_DECODE_FEE+ALIGNMENT_FEE
              +DIRECT_CONTRACTION_FEE+MATERIALIZATION_FEE)
REFERENCE_HEADER_FEE = 1024
REFERENCE_ENCODING_FEE = 64*36
MAX_ALIGNMENT_SHIFT = 33
ALIGNED_LIMIT = 1 << 57
TRANSFORM_LIMIT = 49*ALIGNED_LIMIT


def _headers(weights):
    if (type(weights) is not np.ndarray
            or weights.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
            or weights.ndim != 4 or weights.shape[-2:] != (3, 3)
            or min(weights.shape[:2]) < 1):
        raise ValueError('complete ordinary binary32/exact-binary32-lift kernel required')
    return int(weights.shape[0]), int(weights.shape[1])


def _canonical_ratio(numerator, denominator):
    """Lossless dyadic normalization; no source/constructor bit decoder."""
    numerator, denominator = int(numerator), int(denominator)
    if denominator <= 0 or denominator & (denominator-1):
        raise ValueError('exact power-of-two denominator required')
    if numerator == 0:
        return 0, 0
    magnitude = abs(numerator)
    trailing = (magnitude & -magnitude).bit_length()-1
    number = numerator >> trailing
    exponent = trailing-(denominator.bit_length()-1)
    if (abs(number).bit_length() > 512 or exponent < -511
            or abs(number).bit_length()+max(0, exponent) > 512):
        raise ValueError('complete canonical dyadic value outside 512-bit domain')
    return number, exponent


def _bounded(value):
    value = F(value)
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 512:
        raise MemoryError('unchanged 512-bit exact kernel reference domain exceeded')
    return value


def prepare(weights, *, pool, enabled=False):
    """Return (report, complete owned numeric evidence); default-off is inert.

    Prepay4096+5936*K*C before ANY source value, copy or allocation.  Per-kernel
    components are288 custody,1152 independent ratio decoding,288 alignment,
    3632 direct word products/additions and576 for the caller's immediate full
    36-Fraction materialization in the native row oracle.  Preparation itself
    retains only numeric arrays; the native oracle pays no duplicate fee for
    that already reserved materialization.  No repeated-call credit is given.

    A normal binary32 nonzero canonical n*2**e has bit_length(|n|)<=24. Restore
    m=n*2**(24-bit_length(|n|)), p=e-(24-bit_length(|n|)).  Then |m|<2**24,
    -149<=p<=104.  The raw exponent span<=33 is checked BEFORE alignment;
    hence |aligned|<2**57.  Each independent tensor row has absolute weight
    sum<=49, so every product and partial sum is strictly below49*2**57<2**63.
    The direct36 contractions perform196 products and160 additions/kernel.
    """
    if not enabled:
        return None
    kcount, ccount = _headers(weights)
    kernels = kcount*ccount
    fee = HEADER_FEE+KERNEL_FEE*kernels
    pool.charge('c125_complete_independent_integer_kernel_proof', fee)
    source = np.array(weights, copy=True, order='C')
    shape = source.shape
    canonical_mantissa = np.empty(shape, np.int64)
    canonical_exponent = np.empty(shape, np.int32)
    raw_mantissa = np.empty(shape, np.int64)
    raw_exponent = np.empty(shape, np.int32)
    aligned = np.empty(shape, np.int64)
    exponents = np.empty((kcount, ccount), np.int32)
    maximum_shift = source_nonzero = 0
    for k in range(kcount):
        for c in range(ccount):
            nonzero_powers = []
            for i in range(3):
                for j in range(3):
                    value = float(source[k, c, i, j])
                    if not math.isfinite(value):
                        raise ValueError('finite ordinary binary32 source required')
                    n, e = _canonical_ratio(*value.as_integer_ratio())
                    if n:
                        bits = abs(n).bit_length()
                        leading_exponent = e+bits-1
                        if bits > 24 or not -126 <= leading_exponent <= 127:
                            raise ValueError('source is not zero/normal exact binary32')
                        restoration = 24-bits
                        # This shift is proved<=23 with result magnitude<2**24.
                        raw_number, raw_power = n << restoration, e-restoration
                        if not -149 <= raw_power <= 104 or abs(raw_number) >= (1 << 24):
                            raise ValueError('independent restored binary32 word domain differs')
                        nonzero_powers.append(raw_power)
                        source_nonzero += 1
                    else:
                        raw_number = raw_power = 0
                    canonical_mantissa[k, c, i, j] = n
                    canonical_exponent[k, c, i, j] = e
                    raw_mantissa[k, c, i, j] = raw_number
                    raw_exponent[k, c, i, j] = raw_power
            exponent = min(nonzero_powers) if nonzero_powers else 0
            span = max(nonzero_powers)-exponent if nonzero_powers else 0
            if span > MAX_ALIGNMENT_SHIFT:
                raise ValueError('complete independent raw binary32 alignment span exceeds33')
            maximum_shift = max(maximum_shift, span)
            exponents[k, c] = exponent
            for i in range(3):
                for j in range(3):
                    number = int(raw_mantissa[k, c, i, j])
                    shift = int(raw_exponent[k, c, i, j])-exponent if number else 0
                    if not 0 <= shift <= 33 or abs(number).bit_length()+shift > 57:
                        raise ValueError('independent signed-word shift envelope not established')
                    word = number << shift
                    if not -ALIGNED_LIMIT < word < ALIGNED_LIMIT:
                        raise ValueError('independent aligned coefficient exceeds signed envelope')
                    aligned[k, c, i, j] = word
    # These immutable mathematical checks PRECEDE every machine multiplication.
    if (sum(map(len, _TERMS)) != 196
            or sum(len(terms)-1 for terms in _TERMS) != 160
            or any(sum(abs(m) for i, j, m in terms) > 49 for terms in _TERMS)
            or TRANSFORM_LIMIT >= (1 << 63)):
        raise ValueError('complete independent tensor intermediate bound differs')
    numerator = np.empty((kcount, ccount, 6, 6), np.int64)
    for component, terms in enumerate(_TERMS):
        i, j, multiplier = terms[0]
        accumulator = aligned[:, :, i, j]*np.int64(multiplier)
        for i, j, multiplier in terms[1:]:
            accumulator += aligned[:, :, i, j]*np.int64(multiplier)
        a, b = divmod(component, 6)
        numerator[:, :, a, b] = accumulator
    if np.any(numerator <= -TRANSFORM_LIMIT) or np.any(numerator >= TRANSFORM_LIMIT):
        raise ValueError('complete independent transformed word envelope differs')
    evidence = dict(
        source_snapshot=source, canonical_mantissa=canonical_mantissa,
        canonical_exponent=canonical_exponent, raw_mantissa=raw_mantissa,
        raw_exponent=raw_exponent, aligned=aligned, numerator=numerator,
        exponent=exponents)
    report = dict(
        kernels=kernels, original_kernel_coefficients_observed=int(source.size),
        original_nonzero=source_nonzero,
        all_transformed_kernel_coefficients_rebuilt=36*kernels,
        transformed_nonzero=int(np.count_nonzero(numerator)),
        complete_all36_proved=True, exact_original_binary32_domain=True,
        complete_signed_int64_envelope=True,
        maximum_raw_alignment_shift=maximum_shift,
        independent_as_integer_ratio_source_decoder=True,
        constructor_transform_or_inverse_used=False,
        direct_integer_products_per_kernel=196,
        direct_integer_additions_per_kernel=160,
        intermediate_abs_bound_exclusive=TRANSFORM_LIMIT,
        complete_original_source_snapshot_owned=True,
        all_preparation_arrays_owned=all(value.flags.owndata for value in evidence.values()),
        numeric_only_complete_preparation_evidence=True,
        no_prepared_cache_or_reuse=True, current_source_reuse_authorized=False,
        word_program_prepaid=(KERNEL_FEE-MATERIALIZATION_FEE)*kernels+HEADER_FEE,
        full36_fraction_materialization_prepaid=MATERIALIZATION_FEE*kernels,
        complete_prepaid_work=fee, native_row_or_full_source_proof=False,
        live_admission=False, formal_gain=0)
    return report, evidence


def full_fraction_reference(weights, *, pool, enabled=False):
    """Unchanged complete196-term Fraction proof, with lossless numeric custody.

    The original64*196*K*C contraction tariff is NOT reduced.  Additional
    prepaid1024+32*weights.size covers the complete owned source observation;
    64*36*K*C covers encoding EVERY exact result as a signed odd int64 and
    int32 exponent.  No Fraction/object arrays escape into the root ledger.
    Every retained pair losslessly represents its complete Fraction result.
    """
    if not enabled:
        return None
    kcount, ccount = _headers(weights)
    kernels = kcount*ccount
    header_fee = REFERENCE_HEADER_FEE+32*int(weights.size)
    fraction_fee, encoding_fee = 64*196*kernels, REFERENCE_ENCODING_FEE*kernels
    pool.charge('c125_complete_fraction_reference_headers_and_snapshot', header_fee)
    pool.charge('c124_independent_exact_kernel_transform', fraction_fee)
    pool.charge('c125_complete_fraction_reference_lossless_encoding', encoding_fee)
    source = np.array(weights, copy=True, order='C')
    if not np.all(np.isfinite(source)):
        raise ValueError('complete finite original Fraction-reference kernel required')
    numbers = np.empty((kcount, ccount, 6, 6), np.int64)
    exponents = np.empty((kcount, ccount, 6, 6), np.int32)
    for k in range(kcount):
        for c in range(ccount):
            for a in range(6):
                for b in range(6):
                    value = F(0)
                    # Deliberately the old independent full contraction loop,
                    # not the new word contraction or its precomputed terms.
                    for i in range(3):
                        for j in range(3):
                            if _GN[a][i] and _GN[b][j]:
                                w = F(float(source[k, c, i, j]))
                                term = _bounded(_GN[a][i]*_GN[b][j]*w)
                                value = _bounded(value+term)
                    number, exponent = _canonical_ratio(value.numerator, value.denominator)
                    if not -(1 << 63) <= number < (1 << 63) or not -511 <= exponent <= 511:
                        raise ValueError('complete Fraction reference cannot be encoded losslessly')
                    if ((exponent >= 0 and (value.denominator != 1
                                           or number << exponent != value.numerator))
                            or (exponent < 0 and (number != value.numerator
                                                or 1 << (-exponent) != value.denominator))):
                        raise ValueError('complete Fraction reference encoding lost an exact coefficient')
                    numbers[k, c, a, b] = number
                    exponents[k, c, a, b] = exponent
    evidence = dict(source_snapshot=source, canonical_numerator=numbers,
                    canonical_exponent=exponents)
    report = dict(
        kernels=kernels, original_kernel_coefficients_observed=int(source.size),
        all_transformed_kernel_coefficients_rebuilt=36*kernels,
        all_fraction_products_recomputed=196*kernels,
        complete_all36_proved=True, unchanged_fraction_contraction_tariff=True,
        exact_lossless_numeric_reference_retained=True,
        complete_original_source_snapshot_owned=True,
        no_prepared_cache_or_reuse=True, current_source_reuse_authorized=False,
        fraction_contraction_prepaid=fraction_fee,
        complete_reference_encoding_prepaid=encoding_fee,
        complete_prepaid_work=header_fee+fraction_fee+encoding_fee,
        live_admission=False, formal_gain=0)
    return report, evidence
