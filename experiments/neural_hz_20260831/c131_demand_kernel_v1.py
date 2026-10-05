"""UNQUALIFIED, default-off, raw-only demand-born F4 kernel preparation.

This private primitive belongs inside ONE fresh construct-and-prove call.  It
accepts no prepared arrays, proof receipt, cache key or instance identity.  It
proves every original 3x3 source word, including residual channels, and every
one of the 36 transformed cells for each selected channel.  A transformed
residual tensor is never allocated.  The enclosing caller must still prove
the C122 selection rule, absence of residual transform consumers, actual
source-origin/maps, every native row, storage lifetime and the full inverse.
The later signed-int64 source-bit binder may consume source_snapshot directly
ONLY after its enclosing actual-operator binding proves float64 storage.
ImplicitConv2DOp's original _numeric_array constructor provides float64; the
generic float32 kernel-proof case does NOT authorize float64 reinterpretation
or a free later conversion.  No hidden ninth persistent float64 array is kept.

No numerical arrays or numerical program execute at import.  No old research
module is imported or modified.  This draft has NOT been numerically run.
"""

import numpy as np


HEADER_FEE = 1024
BASIS_FEE = 65536
MASK_FEE = 32
SELECTED_ID_FEE = 16
SOURCE_CUSTODY_FEE = 288
RAW_PRODUCER_FEE = 288
INDEPENDENT_SOURCE_FEE = 576
FULL_DOMAIN_FEE = 256
SOURCE_CENSUS_FEE = 72
SOURCE_KERNEL_FEE = 1480
SELECTED_GATHER_FEE = 288
PRODUCER_ALIGNMENT_FEE = 288
PRODUCER_PROGRAM_FEE = 972
PRODUCER_ENVELOPE_FEE = 288
INDEPENDENT_ALIGNMENT_FEE = 288
ALIGNMENT_COMPARISON_FEE = 80
SOURCE_ANCHOR_FEE = 160
INDEPENDENT_PROGRAM_FEE = 972
COMPLETE_COMPONENT_FEE = 288
TRANSFORM_CENSUS_FEE = 288
SELECTED_KERNEL_FEE = 3912
MAX_ALIGNMENT_SHIFT = 33
ALIGNED_LIMIT = 1 << 57
TRANSFORM_LIMIT = 49*ALIGNED_LIMIT
MAX_NEW_ENTRIES = 64_000_000
MAX_NEW_BYTES = 1 << 30

# Independent literal tensor theorem, not an imported constructor constant.
_GN = ((1, 0, 0), (-1, -1, -1), (-1, 1, -1),
       (1, 2, 4), (1, -2, 4), (0, 0, 1))
_L1 = (1, 3, 3, 7, 7, 1)


def _headers(weights, selected_channels):
    """Metadata only: no source values or mask entries are inspected here."""
    if (type(weights) is not np.ndarray
            or weights.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
            or weights.ndim != 4 or weights.shape[-2:] != (3, 3)
            or min(weights.shape[:2]) < 1):
        raise ValueError('ordinary nonempty native float32/float64 3x3 source required')
    kcount, ccount = map(int, weights.shape[:2])
    if (type(selected_channels) is not np.ndarray
            or selected_channels.dtype != np.dtype(np.bool_)
            or selected_channels.shape != (ccount,)
            or ccount > (1 << 31)-1):
        raise ValueError('complete bool channel mask with bounded int32 IDs required')
    return kcount, ccount


def _same_bits(left, right):
    """Exact same-width word comparison, including source signed-zero bits."""
    if (type(right) is not np.ndarray or left.shape != right.shape
            or left.dtype != right.dtype):
        return False
    if left.dtype == np.dtype(np.float32):
        dtype = np.uint32
    elif left.dtype == np.dtype(np.float64):
        dtype = np.uint64
    else:
        dtype = np.uint8
    return bool(np.array_equal(left.view(dtype), right.view(dtype)))


def _inside(values, limit):
    # All callers establish this signed interval BEFORE abs, shifts or products.
    return not (np.any(values <= -limit) or np.any(values >= limit))


def _raw_words(source):
    """Bitfield producer only; its words gain authority in _certify_source.

The new 32/source composite covers float32 working conversion, bitfield
extraction/casts, sign application, zero handling, exponent formation and
their temporary/retained stores.  It performs no canonical odd-word work.
"""
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        binary32 = np.array(source, dtype=np.float32, order='C', copy=True)
    bits = binary32.view(np.uint32)
    encoded_exponent = ((bits >> 23) & np.uint32(255)).astype(np.int32)
    fraction = bits & np.uint32((1 << 23)-1)
    nonzero = encoded_exponent != 0
    mantissa = (fraction | np.uint32(1 << 23)).astype(np.int64)
    mantissa[~nonzero] = 0
    mantissa[(bits >> 31) != 0] *= -1
    exponent = np.where(nonzero, encoded_exponent-150, 0).astype(np.int32)
    return mantissa, exponent


def _certify_source(source, mantissa, exponent):
    """Independent guarded ldexp identity against EVERY original coefficient.

This checker neither decodes source bitfields nor trusts producer range flags.
The unique normalized 24-bit word, finite source and exact ldexp identity
prove zero/normal exact binary32, including exact float64 lifts.  A zero word
has exponent zero; its sign is preserved separately in source_snapshot.
All integer range checks precede abs and all exponent bounds precede ldexp.
The 64/source composite includes all guards, reductions, conversions, stores
and exact full-source comparison; it is not a 64-ALU-instruction claim.
"""
    if (type(mantissa) is not np.ndarray or mantissa.dtype != np.dtype(np.int64)
            or type(exponent) is not np.ndarray or exponent.dtype != np.dtype(np.int32)
            or mantissa.shape != source.shape or exponent.shape != source.shape
            or not np.isfinite(source).all()
            or not _inside(mantissa, 1 << 24)):
        raise ValueError('finite bounded complete raw source words required')
    nonzero = mantissa != 0
    magnitude = np.abs(mantissa)
    if (np.any(nonzero & ((magnitude < (1 << 23))
                          | (exponent < -149) | (exponent > 104)))
            or np.any((~nonzero) & (exponent != 0))):
        raise ValueError('source words are not unique zero/normal binary32 words')
    with np.errstate(over='raise', under='raise', invalid='raise'):
        restored = np.ldexp(mantissa.astype(np.float64), exponent)
    if not np.array_equal(restored, source):
        raise ValueError('every raw word must restore its exact original source')


def _full_domain(mantissa, exponent):
    """Prove ALL KC spans, including residuals, without shifting residuals."""
    nonzero = mantissa != 0
    minimum = np.where(nonzero, exponent, 300).min(axis=(-2, -1))
    maximum = np.where(nonzero, exponent, -300).max(axis=(-2, -1))
    empty = minimum == 300
    minimum = np.where(empty, 0, minimum).astype(np.int32)
    maximum = np.where(empty, 0, maximum).astype(np.int32)
    span = maximum-minimum
    if np.any(span < 0) or np.any(span > MAX_ALIGNMENT_SHIFT):
        raise ValueError('complete original raw alignment span exceeds33')
    return minimum, int(span.max(initial=0))


def _producer_alignment(mantissa, exponent, minimum):
    shifts = np.where(mantissa != 0, exponent-minimum[..., None, None], 0)
    if np.any(shifts < 0) or np.any(shifts > MAX_ALIGNMENT_SHIFT):
        raise ValueError('selected producer shift outside proved domain')
    aligned = np.left_shift(mantissa, shifts).astype(np.int64)
    if not _inside(aligned, ALIGNED_LIMIT):
        raise ValueError('selected producer aligned magnitude exceeds envelope')
    return aligned


def _independent_alignment(mantissa, exponent):
    """Separate minimum derivation; no producer minimum/aligned input accepted."""
    minimum = np.full(mantissa.shape[:2], 300, dtype=np.int32)
    for row in range(3):
        for column in range(3):
            current = np.where(mantissa[..., row, column] != 0,
                               exponent[..., row, column], 300)
            minimum = np.minimum(minimum, current)
    minimum[minimum == 300] = 0
    delta = np.where(mantissa != 0, exponent-minimum[..., None, None], 0)
    if np.any(delta < 0) or np.any(delta > MAX_ALIGNMENT_SHIFT):
        raise ValueError('independent selected shift outside proved domain')
    aligned = np.left_shift(mantissa, delta).astype(np.int64)
    if not _inside(aligned, ALIGNED_LIMIT):
        raise ValueError('independent aligned source envelope differs')
    return aligned, minimum


def _forward_axis(a, b, c, *, axis):
    """Nine signed arithmetic operations per triple, no old99-op repricing."""
    negative_sum = -a-c
    twice_b = 2*b
    shared = a+4*c
    return np.stack((a, negative_sum-b, negative_sum+b,
                     shared+twice_b, shared-twice_b, c), axis=axis)


def _forward(aligned):
    if not _inside(aligned, ALIGNED_LIMIT):
        raise ValueError('producer source bound must precede its arithmetic')
    first = _forward_axis(aligned[..., 0, :], aligned[..., 1, :],
                          aligned[..., 2, :], axis=-2)
    # Row r has bound L1[r]*A; second-axis partials are <=7*L1[r]*A.
    # The initial proof, not a post-overflow test, bounds every signed operation.
    return _forward_axis(first[..., 0], first[..., 1], first[..., 2], axis=-1)


def _expand_anchors(a, first, c, *, scale):
    """Nine safe operations from independently source-derived anchors0,1,5.

Recovering b=-a-first-c has partial bound <5*scale*A <=35*A <2**63.
Check |b|<scale*A BEFORE doubling.  second=first+2*b is the same exact
linear form as -a+b-c; remaining partials are <7*scale*A <=49*A.
The 12/op composite includes input/middle guards, gathers and temporary/stores.
"""
    if type(scale) is not int or not 1 <= scale <= 7:
        raise ValueError('fixed other-axis source envelope required')
    bound = scale*ALIGNED_LIMIT
    if not (_inside(a, bound) and _inside(first, 3*bound) and _inside(c, bound)):
        raise ValueError('anchor envelope must precede independent arithmetic')
    middle = -a-first-c
    if not _inside(middle, bound):
        raise ValueError('recovered middle exceeds independently proved envelope')
    twice_middle = 2*middle
    second = first+twice_middle
    shared = a+4*c
    third = shared+twice_middle
    fourth = shared-twice_middle
    return second, third, fourth


def _independent_transform(aligned):
    """Twenty source-anchor operations plus nine independent9-op expansions."""
    if not _inside(aligned, ALIGNED_LIMIT):
        raise ValueError('independent source bound must precede arithmetic')
    result = np.empty((*aligned.shape[:-2], 6, 6), dtype=np.int64)
    a, b, c = aligned[..., 0, 0], aligned[..., 0, 1], aligned[..., 0, 2]
    d, e, f = aligned[..., 1, 0], aligned[..., 1, 1], aligned[..., 1, 2]
    g, h, i = aligned[..., 2, 0], aligned[..., 2, 1], aligned[..., 2, 2]
    result[..., 0, 0], result[..., 0, 5] = a, c
    result[..., 5, 0], result[..., 5, 5] = g, i
    result[..., 0, 1], result[..., 5, 1] = -a-b-c, -g-h-i
    result[..., 1, 0], result[..., 1, 5] = -a-d-g, -c-f-i
    result[..., 1, 1] = a+b+c+d+e+f+g+h+i
    for row, scale in ((0, 1), (1, 3), (5, 1)):
        second, third, fourth = _expand_anchors(
            result[..., row, 0], result[..., row, 1], result[..., row, 5],
            scale=scale)
        result[..., row, 2], result[..., row, 3], result[..., row, 4] = second, third, fourth
    for column, scale in enumerate(_L1):
        second, third, fourth = _expand_anchors(
            result[..., 0, column], result[..., 1, column], result[..., 5, column],
            scale=scale)
        result[..., 2, column], result[..., 3, column], result[..., 4, column] = second, third, fourth
    return result


def _component_envelopes(values):
    for row, left in enumerate(_L1):
        for column, right in enumerate(_L1):
            if not _inside(values[..., row, column], left*right*ALIGNED_LIMIT):
                raise ValueError('complete transformed per-component bound differs')


def _prove_basis():
    """Fresh paid ALL9 basis x ALL36 cells for BOTH complete programs."""
    if (_L1 != tuple(sum(abs(v) for v in row) for row in _GN)
            or TRANSFORM_LIMIT >= (1 << 63) or 35*ALIGNED_LIMIT >= (1 << 63)):
        raise ValueError('fixed full tensor overflow theorem differs')
    comparisons = 0
    for i in range(3):
        for j in range(3):
            basis = np.zeros((3, 3), dtype=np.int64)
            basis[i, j] = 1
            forward = _forward(basis)
            independent = _independent_transform(basis)
            for row in range(6):
                for column in range(6):
                    expected = _GN[row][i]*_GN[column][j]
                    if (int(forward[row, column]) != expected
                            or int(independent[row, column]) != expected):
                        raise ValueError('complete source-basis tensor theorem failed')
                    comparisons += 2
    return comparisons


def _freeze_owned(evidence):
    """Eight fixed C-contiguous owners: exact nonoverlap by byte intervals."""
    intervals = []
    for value in evidence.values():
        if (type(value) is not np.ndarray or not value.flags.owndata
                or value.base is not None or not value.flags.c_contiguous
                or value.dtype.hasobject or not value.dtype.isnative):
            raise ValueError('complete independent native owned evidence required')
        start = int(value.__array_interface__['data'][0])
        intervals.append((start, start+int(value.nbytes)))
    intervals.sort()
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError('owned evidence buffers overlap')
    for value in evidence.values():
        value.flags.writeable = False


def _prepare_selected(weights, selected_channels, *, pool, enabled=False):
    """Private fresh kernel proof: None, or (report, eight_arrays_or_None).

Default False inspects nothing.  Empty selection pays only1024+32C, checks
complete headers/mask custody, and returns a literal no-op before any source
value, basis or kernel access.  Nonempty calls prepay the exact full tariff
66560+32C+16S+1480KC+3912KS before source reads/copies.  No fee is refunded.

KC1480:288 custody +288 raw decode +576 independent raw/source certificate
+256 all-kernel domain-only scans +72 original census.  KS3912:288 certified
gathers +288 producer alignment +972 new81-op producer +288 envelopes
+288 independent alignment +80 full alignment comparison +160 anchors
+972 new81-op independent expansion +288 ALL36 equality/envelopes +288 census.
These composite charges include their guards, reductions, temporary vectors
and retained stores, not merely signed arithmetic instruction counts.

New owned evidence has27KC+46KS+C+S entries.  A conservative phased scratch
envelope adds32*9KC+8KC and16*9KS+18KS elements, each priced at8 bytes.
Including evidence (float64 worst case) gives323KC+208KS+C+S entries and
2548KC+1660KS+C+4S bytes.  Source/consumer/whole-process roots are NOT included
in this local bound; the enclosing complete source proof must account for them.
"""
    if enabled is False:
        return None
    if enabled is not True:
        raise ValueError('explicit bool opt-in required')
    kcount, ccount = _headers(weights, selected_channels)
    initial_fee = HEADER_FEE+MASK_FEE*ccount
    pool.charge('c131_complete_headers_and_mask_custody', initial_fee)
    selected_mask = np.array(selected_channels, dtype=np.bool_, order='C', copy=True)
    if not _same_bits(selected_mask, selected_channels):
        raise ValueError('initial complete mask snapshot differs')
    scount = int(np.count_nonzero(selected_mask))
    if scount == 0:
        if not _same_bits(selected_mask, selected_channels):
            raise ValueError('empty selection changed during no-op')
        return dict(literal_noop=True, selected_channels=0, residual_channels=ccount,
                    complete_prepaid_work=initial_fee, kernel_transform_prepaid=0,
                    original_source_values_observed=0, evidence_entries=0,
                    source_origin_or_selection_rule_proved=False,
                    no_prepared_cache_or_reuse=True, live_admission=False,
                    formal_gain=0), None

    kernels, selected_kernels = kcount*ccount, kcount*scount
    fee = (HEADER_FEE+BASIS_FEE+MASK_FEE*ccount+SELECTED_ID_FEE*scount
           +SOURCE_KERNEL_FEE*kernels+SELECTED_KERNEL_FEE*selected_kernels)
    peak_entries = 323*kernels+208*selected_kernels+ccount+scount
    peak_bytes = 2548*kernels+1660*selected_kernels+ccount+4*scount
    if peak_entries > MAX_NEW_ENTRIES or peak_bytes > MAX_NEW_BYTES:
        raise MemoryError('complete new preparation/scratch envelope exceeds local ceiling')
    pool.charge('c131_complete_fresh_raw_and_selected_kernel_proof', fee-initial_fee)
    basis_comparisons = _prove_basis()
    selected_ids = np.flatnonzero(selected_mask).astype(np.int32)
    reconstructed_mask = np.zeros(ccount, dtype=np.bool_)
    reconstructed_mask[selected_ids] = True
    if (selected_ids.shape != (scount,) or np.any(np.diff(selected_ids) <= 0)
            or not np.array_equal(selected_mask, reconstructed_mask)):
        raise ValueError('complete selected/residual channel partition differs')
    source = np.array(weights, order='C', copy=True)
    if not _same_bits(source, weights):
        raise ValueError('initial complete original source snapshot differs')
    if not np.isfinite(source).all():
        raise ValueError('complete finite source required before working conversion')
    raw_mantissa, raw_exponent = _raw_words(source)
    _certify_source(source, raw_mantissa, raw_exponent)
    minimum, maximum_span = _full_domain(raw_mantissa, raw_exponent)

    selected_mantissa = np.take(raw_mantissa, selected_ids, axis=1)
    selected_exponent = np.take(raw_exponent, selected_ids, axis=1)
    # This is a source-mapped gather certificate, not a second source decoder.
    if (not np.array_equal(selected_mantissa, raw_mantissa[:, selected_ids])
            or not np.array_equal(selected_exponent, raw_exponent[:, selected_ids])):
        raise ValueError('complete selected word gather differs from original channels')
    exponent = np.array(minimum[:, selected_ids], dtype=np.int32, order='C', copy=True)
    aligned = _producer_alignment(selected_mantissa, selected_exponent, exponent)
    numerator = _forward(aligned)
    _component_envelopes(numerator)
    independent_aligned, independent_exponent = _independent_alignment(
        selected_mantissa, selected_exponent)
    if (not np.array_equal(independent_aligned, aligned)
            or not np.array_equal(independent_exponent, exponent)):
        raise ValueError('every selected source alignment must agree independently')
    independent_numerator = _independent_transform(independent_aligned)
    _component_envelopes(independent_numerator)
    if not np.array_equal(independent_numerator, numerator):
        raise ValueError('every one of all36 selected transformed cells must agree')

    original_nonzero = int(np.count_nonzero(raw_mantissa))
    transformed_nonzero = int(np.count_nonzero(numerator))
    # All original coefficients and the full mask remain custodied, not just
    # selected words or a saved digest.  Signed zero bytes are compared too.
    if not _same_bits(source, weights) or not _same_bits(selected_mask, selected_channels):
        raise ValueError('original source or complete selection changed during proof')
    evidence = dict(source_snapshot=source, raw_mantissa=raw_mantissa,
                    raw_exponent=raw_exponent, aligned=aligned, numerator=numerator,
                    exponent=exponent, selected_mask=selected_mask, selected_ids=selected_ids)
    _freeze_owned(evidence)
    entries = 27*kernels+46*selected_kernels+ccount+scount
    numeric_bytes = (9*(int(source.dtype.itemsize)+12)*kernels
                     +364*selected_kernels+ccount+4*scount)
    if (sum(int(v.size) for v in evidence.values()) != entries
            or sum(int(v.nbytes) for v in evidence.values()) != numeric_bytes):
        raise ValueError('complete eight-array evidence population differs')
    report = dict(
        literal_noop=False, filters=kcount, channels=ccount,
        selected_channels=scount, residual_channels=ccount-scount,
        kernels=kernels, selected_kernels=selected_kernels,
        original_kernel_coefficients_observed=9*kernels,
        original_nonzero=original_nonzero,
        all_selected_transformed_coefficients_proved=36*selected_kernels,
        transformed_nonzero=transformed_nonzero,
        residual_transformed_cells_allocated=0,
        all_original_raw_spans_proved=True, maximum_raw_alignment_shift=maximum_span,
        exact_original_zero_or_normal_binary32_domain=True,
        complete_all36_selected_proved=True,
        producer_signed_operations_per_selected_kernel=81,
        independent_anchor_operations_per_selected_kernel=20,
        independent_expansion_operations_per_selected_kernel=81,
        source_basis_comparisons=basis_comparisons,
        full_basis_theorem_executed_each_nonempty_call=True,
        complete_signed_int64_envelope=True,
        intermediate_abs_bound_exclusive=TRANSFORM_LIMIT,
        complete_reduced_512_bit_source_and_transform_domain=True,
        canonical_odd_source_arrays_created=False,
        independent_raw_ldexp_source_certificate=True,
        source_and_complete_mask_bytes_unchanged=True,
        all_evidence_arrays_owned_readonly_nonoverlapping=True,
        source_snapshot_dtype=source.dtype.str,
        source_snapshot_is_float64=bool(source.dtype == np.dtype(np.float64)),
        private_float64_working_copy_retained=False,
        evidence_arrays=8, evidence_entries=entries, evidence_numeric_bytes=numeric_bytes,
        local_new_numeric_peak_entries_upper=peak_entries,
        local_new_numeric_peak_bytes_upper=peak_bytes,
        complete_source_or_process_peak_proved=False,
        complete_prepaid_work=fee,
        kernel_transform_prepaid=BASIS_FEE+SELECTED_KERNEL_FEE*selected_kernels,
        work_components=dict(headers=HEADER_FEE, basis=BASIS_FEE,
                             mask=MASK_FEE*ccount, selected_ids=SELECTED_ID_FEE*scount,
                             all_original_kernels=SOURCE_KERNEL_FEE*kernels,
                             all_selected_kernels=SELECTED_KERNEL_FEE*selected_kernels),
        source_origin_or_selection_rule_proved=False,
        residual_consumer_absence_proved=False,
        no_prepared_cache_or_reuse=True, external_prepared_receipt_accepted=False,
        current_source_reuse_authorized=False,
        native_row_or_full_source_proof=False, live_admission=False, formal_gain=0)
    return report, evidence
