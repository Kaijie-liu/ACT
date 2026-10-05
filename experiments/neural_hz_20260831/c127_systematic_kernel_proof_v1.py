"""Independent ALL36 source-derived F4 kernel program with systematic anchors.

All nine original source coefficients are decoded independently with exact
frexp/ldexp identities, not the constructor's bit decoder.  Nine transformed
anchors at indices0,1,5 are computed directly from the source; all remaining27
cells are independently reconstructed by a triangular integer program.  No
constructor array, inverse, constants, helper or receipt is accepted/imported.

The complete nine-basis theorem runs AFTER payment on EACH call.  It compares
every36 output coefficient against independent literal tensor coefficients;
this is full independent recomputation, not an inverse-only candidate check.
There are no numerical arrays or numerical programs executed on module import.
"""

import numpy as np


_GN = ((1, 0, 0), (-1, -1, -1), (-1, 1, -1),
       (1, 2, 4), (1, -2, 4), (0, 0, 1))
_L1 = (1, 3, 3, 7, 7, 1)
HEADER_FEE = 32768
SOURCE_CUSTODY_FEE = 32*9
SOURCE_DECODE_FEE = 64*9
ALIGNMENT_FEE = 32*9
# Composite vector-program charges, not arithmetic instructions alone:
# each11-operation expansion reserves132 units per kernel for its arithmetic,
# all four input/middle envelope scans, gathers, temporary vectors and three
# retained dependent-cell stores. The160 anchor units include the source
# gathers, corner copies and nine anchor stores as well as20 signed operations.
# The final separate36-cell scan is paid below; fixed basis work is HEADER_FEE.
SYSTEMATIC_PROGRAM_FEE = 8*20+12*99
COMPLETE_COMPONENT_CHECK_FEE = 8*36
MATERIALIZATION_FEE = 16*36
KERNEL_FEE = (SOURCE_CUSTODY_FEE+SOURCE_DECODE_FEE+ALIGNMENT_FEE
              +SYSTEMATIC_PROGRAM_FEE+COMPLETE_COMPONENT_CHECK_FEE
              +MATERIALIZATION_FEE)
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


def _inside(value, limit):
    # Do not use abs on untrusted signed words: abs(INT64_MIN) would overflow.
    return not (np.any(value <= -limit) or np.any(value >= limit))


def _expand_three(a, first, c, *, scale):
    """Recover all6 entries from0,1,5 by a safe source-derived linear program.

If a,c have magnitude<s*A and first<3*s*A, recovering the missing middle
source b takes partials<5*s*A.  Here s<=7, so these remain<35*A<2**63 BEFORE
checking |b|<s*A.  Subsequent shared a+4*c,2*b and their sums stay<7*s*A<=49*A.
The eleven signed integer operations include all negations, products and sums.
"""
    if type(scale) is not int or not 1 <= scale <= 7:
        raise ValueError('fixed systematic other-axis envelope required')
    bound = scale*ALIGNED_LIMIT
    if not (_inside(a, bound) and _inside(first, 3*bound) and _inside(c, bound)):
        raise ValueError('systematic input envelope not established before arithmetic')
    middle = -a-first-c
    if not _inside(middle, bound):
        raise ValueError('source-derived recovered middle envelope differs')
    second = -a+middle-c
    shared = a+4*c
    twice_middle = 2*middle
    third = shared+twice_middle
    fourth = shared-twice_middle
    return a, first, second, third, fourth, c


def _systematic(aligned):
    """Independently RECOMPUTE all36 outputs from a complete aligned3x3 source."""
    if (type(aligned) is not np.ndarray or aligned.dtype != np.dtype(np.int64)
            or aligned.ndim < 2 or aligned.shape[-2:] != (3, 3)
            or not _inside(aligned, ALIGNED_LIMIT)):
        raise ValueError('complete bounded int64 aligned3x3 source required')
    s00, s01, s02 = aligned[..., 0, 0], aligned[..., 0, 1], aligned[..., 0, 2]
    s10, s11, s12 = aligned[..., 1, 0], aligned[..., 1, 1], aligned[..., 1, 2]
    s20, s21, s22 = aligned[..., 2, 0], aligned[..., 2, 1], aligned[..., 2, 2]
    result = np.empty((*aligned.shape[:-2], 6, 6), dtype=np.int64)
    # Four literal source corners, four negative3-term edge forms (12 signed
    # operations), and the complete nine-source center (8 additions).  All
    # intermediate magnitudes are<9*A, already within the signed64 envelope.
    result[..., 0, 0], result[..., 0, 5] = s00, s02
    result[..., 5, 0], result[..., 5, 5] = s20, s22
    result[..., 0, 1] = -s00-s01-s02
    result[..., 5, 1] = -s20-s21-s22
    result[..., 1, 0] = -s00-s10-s20
    result[..., 1, 5] = -s02-s12-s22
    result[..., 1, 1] = s00+s01+s02+s10+s11+s12+s20+s21+s22
    # Nine missing cells in the three anchor rows, then eighteen missing cells
    # in ALL six columns: every non-anchor cell has exactly one defining step.
    for row, scale in ((0, 1), (1, 3), (5, 1)):
        expanded = _expand_three(result[..., row, 0], result[..., row, 1],
                                 result[..., row, 5], scale=scale)
        for column in (2, 3, 4):
            result[..., row, column] = expanded[column]
    for column, scale in enumerate(_L1):
        expanded = _expand_three(result[..., 0, column], result[..., 1, column],
                                 result[..., 5, column], scale=scale)
        for row in (2, 3, 4):
            result[..., row, column] = expanded[row]
    return result


def _prove_basis():
    """All nine source basis vectors, ALL36 literal tensor identities per call.

The linear integer program has no value-selecting branch, only rejection
guards.  Its equality on every source basis therefore proves all36 full-source
linear forms, including the27 cells an inverse-only check could leave free.
The fixed32768 header pays this theorem and its tiny arrays each invocation.
"""
    if (_L1 != tuple(sum(abs(value) for value in row) for row in _GN)
            or TRANSFORM_LIMIT >= (1 << 63) or 35*ALIGNED_LIMIT >= (1 << 63)):
        raise ValueError('complete literal systematic signed64 theorem differs')
    comparisons = 0
    for i in range(3):
        for j in range(3):
            basis = np.zeros((3, 3), dtype=np.int64)
            basis[i, j] = 1
            actual = _systematic(basis)
            for row in range(6):
                for column in range(6):
                    if int(actual[row, column]) != _GN[row][i]*_GN[column][j]:
                        raise ValueError('complete nine-basis ALL36 source identity failed')
                    comparisons += 1
    return comparisons


def prepare(weights, *, pool, enabled=False):
    """Return the exact eight-array C125 preparation/evidence contract.

Prepay32768+3364*K*C BEFORE any numeric source read, copy or allocation.  The
per-kernel tariff is288 custody,576 independent vector decoding/canonical
checks,288 alignment,1348 new systematic program,288 complete component
envelopes, and576 reserved for a caller's immediate ALL36 Fraction material-
ization.  That last reserve is never refunded, even if a word-only caller does
not use it.  There is no external prepared receipt or repeated-call discount.
The1348 composite program reserve includes every per-triple envelope scan,
gather, temporary and retained store; it is not an arithmetic-only payment.

frexp gives signed f with0.5<=|f|<1 and exponent e for nonzero finite source.
Exact integrality of ldexp(f,24), normal range-125<=e<=128, and the complete
ldexp integer roundtrip prove the ORIGINAL source is a normal exact binary32
value.  This is independent of bitfield decoding and has no tolerance.  Raw
words satisfy |m|<2**24; complete alignment span<=33 is checked BEFORE shifts.
Hence |aligned|<2**57 and every program partial is<49*2**57<2**63.

The reduced512-bit rational domain follows before arithmetic as well: raw
source exponents are[-149,104], canonical source numerator bits<=128 and
denominator bits<=150.  A transformed signed word has<=63bits at a common
exponent in[-149,104], so reduced numerator bits<=167 and denominator<=150.
"""
    if not enabled:
        return None
    kcount, ccount = _headers(weights)
    kernels = kcount*ccount
    fee = HEADER_FEE+KERNEL_FEE*kernels
    pool.charge('c127_complete_independent_systematic_kernel_proof', fee)
    basis_comparisons = _prove_basis()
    source = np.array(weights, copy=True, order='C')
    source64 = np.array(source, dtype=np.float64, copy=True, order='C')
    if not np.isfinite(source64).all():
        raise ValueError('finite ordinary binary32 source required')
    nonzero = source64 != 0.
    fractions, binary_exponents = np.frexp(source64)
    raw_float = np.ldexp(fractions, 24)
    # This complete bound PRECEDES conversion to signed integers.  All normal
    # binary64 inputs meet the magnitude bound, but integrality is independently
    # checked next: an inexact binary32 lift must not be rounded into admission.
    if (not np.isfinite(raw_float).all()
            or np.any(raw_float <= -(1 << 24)) or np.any(raw_float >= (1 << 24))):
        raise ValueError('independent original mantissa conversion envelope differs')
    raw_mantissa = raw_float.astype(np.int64)
    if not np.array_equal(raw_mantissa.astype(np.float64), raw_float):
        raise ValueError('original source is not exactly binary32')
    magnitude = np.abs(raw_mantissa)
    if (np.any(nonzero & ((magnitude < (1 << 23))
                          | (binary_exponents < -125) | (binary_exponents > 128)))):
        raise ValueError('source is not zero/normal exact binary32')
    raw_exponent = np.where(nonzero, binary_exponents-24, 0).astype(np.int32)
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        restored = np.ldexp(raw_mantissa.astype(np.float64), raw_exponent)
    if not np.array_equal(restored, source64):
        raise ValueError('complete independent original integer roundtrip differs')

    # Lowbit is exact in this proved<2**24 integer domain.  frexp of a power of
    # two determines its exponent exactly; zero is explicitly canonicalized.
    lowbit = magnitude & -magnitude
    trailing = np.where(nonzero, np.frexp(lowbit.astype(np.float64))[1]-1, 0)
    if np.any(trailing < 0) or np.any(trailing > 23):
        raise ValueError('independent original canonical word shift differs')
    canonical_mantissa = np.right_shift(raw_mantissa, trailing).astype(np.int64)
    canonical_exponent = np.where(nonzero, raw_exponent+trailing, 0).astype(np.int32)
    if (np.any(nonzero & ((canonical_mantissa & 1) == 0))
            or not np.array_equal(np.left_shift(canonical_mantissa, trailing), raw_mantissa)):
        raise ValueError('complete independent canonical word reconstruction differs')

    minimum = np.where(nonzero, raw_exponent, 300).min(axis=(-2, -1))
    minimum = np.where(minimum == 300, 0).astype(np.int32)
    shifts = np.where(nonzero, raw_exponent-minimum[..., None, None], 0)
    if np.any(shifts < 0) or np.any(shifts > MAX_ALIGNMENT_SHIFT):
        raise ValueError('complete independent raw binary32 alignment span exceeds33')
    aligned = np.left_shift(raw_mantissa, shifts).astype(np.int64)
    if not _inside(aligned, ALIGNED_LIMIT):
        raise ValueError('independent aligned coefficient exceeds signed envelope')
    numerator = _systematic(aligned)
    for row, left in enumerate(_L1):
        for column, right in enumerate(_L1):
            if not _inside(numerator[..., row, column], left*right*ALIGNED_LIMIT):
                raise ValueError('complete transformed per-component envelope differs')
    evidence = dict(
        source_snapshot=source, canonical_mantissa=canonical_mantissa,
        canonical_exponent=canonical_exponent, raw_mantissa=raw_mantissa,
        raw_exponent=raw_exponent, aligned=aligned, numerator=numerator,
        exponent=minimum)
    if not all(value.flags.owndata for value in evidence.values()):
        raise ValueError('complete owned original and transformed numeric evidence required')
    report = dict(
        kernels=kernels, original_kernel_coefficients_observed=int(source.size),
        original_nonzero=int(np.count_nonzero(nonzero)),
        all_transformed_kernel_coefficients_rebuilt=int(numerator.size),
        transformed_nonzero=int(np.count_nonzero(numerator)),
        complete_all36_proved=True, exact_original_binary32_domain=True,
        complete_signed_int64_envelope=True,
        maximum_raw_alignment_shift=int(shifts.max(initial=0)),
        systematic_source_derived_program=True,
        source_decoder='independent_vector_frexp_ldexp',
        independent_as_integer_ratio_source_decoder=False,
        constructor_transform_or_inverse_used=False,
        external_transformed_candidate_or_receipt_accepted=False,
        source_anchor_indices=[0, 1, 5],
        source_anchor_cells_per_kernel=9,
        source_anchor_signed_operations_per_kernel=20,
        source_derived_expansion_triples_per_kernel=9,
        source_derived_expansion_operations_per_kernel=99,
        source_derived_remaining_cells_per_kernel=27,
        all_nine_source_basis_vectors_proved=True,
        all_source_basis_output_coefficients_proved=basis_comparisons,
        full_basis_theorem_executed_each_call=True,
        intermediate_abs_bound_exclusive=TRANSFORM_LIMIT,
        complete_reduced_512_bit_source_and_transform_domain=True,
        complete_original_source_snapshot_owned=True,
        all_preparation_arrays_owned=True,
        complete_prepaid_work=fee,
        word_program_prepaid=HEADER_FEE+(KERNEL_FEE-MATERIALIZATION_FEE)*kernels,
        source_custody_prepaid=SOURCE_CUSTODY_FEE*kernels,
        independent_vector_source_decode_prepaid=SOURCE_DECODE_FEE*kernels,
        complete_alignment_prepaid=ALIGNMENT_FEE*kernels,
        systematic_program_prepaid=SYSTEMATIC_PROGRAM_FEE*kernels,
        complete_component_checks_prepaid=COMPLETE_COMPONENT_CHECK_FEE*kernels,
        full_basis_header_prepaid=HEADER_FEE,
        full36_fraction_materialization_prepaid=MATERIALIZATION_FEE*kernels,
        no_prepared_cache_or_reuse=True, current_source_reuse_authorized=False,
        numeric_only_complete_preparation_evidence=True,
        native_row_or_full_source_proof=False, live_admission=False, formal_gain=0)
    return report, evidence
