"""Opt-in denominator-bearing F4 rows with a proved exact integer kernel program.

The rows, topology, gauges, boxes and odd defining denominators are C119's.
Only preparation of GNUM @ kernel @ GNUM.T changes.  No floating transformed
coefficient is computed, and no source outside the declared word domain falls
back to a different constructor.  Neither this module nor its report admits a
candidate into a source HZ or a verifier.
"""
from fractions import Fraction as F
import numpy as np

from experiments.neural_hz_20260831.c119_denominator_f4_v1 import (
    AT, BT, D, GNUM, RATIONAL_BITS, _bounded, _l1, _native, _normalization, _power)


WORD_PROGRAM_FEE = 2048
SOURCE_CONVERSION_FEE = 16*9
NUMERATOR_MATERIALIZATION_FEE = 16*36
KERNEL_FEE = WORD_PROGRAM_FEE+SOURCE_CONVERSION_FEE+NUMERATOR_MATERIALIZATION_FEE
MAX_ALIGNMENT_SHIFT = 33
ALIGNED_LIMIT = 1 << 57
TRANSFORM_LIMIT = 49*ALIGNED_LIMIT


def _axis(a, b, c, *, axis):
    """Six GNUM forms with eleven shared signed-integer operations/triple."""
    negative_a, twice_b, four_c = -a, 2*b, 4*c
    return np.stack((a, negative_a-b-c, negative_a+b-c,
                     a+twice_b+four_c, a-twice_b+four_c, c), axis=axis)


def _separable(aligned):
    """GNUM @ aligned @ GNUM.T: 11*(3+6) = 99 integer operations/kernel."""
    first = _axis(aligned[..., 0, :], aligned[..., 1, :], aligned[..., 2, :],
                  axis=-2)
    return _axis(first[..., 0], first[..., 1], first[..., 2], axis=-1)


def _recover(numerator):
    """Independent inverse: a=y0, c=y5, b=y2+y0+y5, on both axes.

    This reconstructs all nine source words, not a checksum.  It is a guard
    against decoding/axis errors, not a substitute for the unchanged independent
    C119 oracle's complete 36-coefficient and actual-row equality checks.
    """
    first = np.stack((numerator[..., 0],
                      numerator[..., 2]+numerator[..., 0]+numerator[..., 5],
                      numerator[..., 5]), axis=-1)
    return np.stack((first[..., 0, :],
                     first[..., 2, :]+first[..., 0, :]+first[..., 5, :],
                     first[..., 5, :]), axis=-2)


def prepare_words(weights, *, pool, enabled=False):
    """Return exact int64 numerators and their common binary scale per kernel.

    Domain: native binary32, or binary64 values exactly equal to a binary32
    round-trip; only zero or normal finite binary32 coefficients.  A nonzero
    decoded mantissa has magnitude <2**24.  A per-kernel alignment span <=33
    therefore gives |aligned|<2**57.  GNUM's row L1 norms are <=7, so all first
    axis intermediates are <7*2**57 and all second axis intermediates are
    <49*2**57 <2**63.  Recovery intermediates, using rows/columns 0,2,5, are
    bounded by 35*2**57 <2**63.  No signed operation may overflow.

    The fixed prepaid word tariff is 2048/kernel, conservatively covering
    complete bit decoding, domain checks, alignment, the 99-operation forward
    program, its gathers/stacks, the 18-operation inverse, and full source-word
    comparison/report reductions.  Conversion additionally prepays 16*9/kernel;
    16*36/kernel prepays subsequent exact numerator Fraction materialization.
    These are new word operations, not credits against C119 Fraction products.
    Every source coefficient is inspected; no external receipt is accepted.
    """
    if not enabled:
        return None
    weights = np.asarray(weights)
    if (weights.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
            or weights.ndim != 4 or weights.shape[-2:] != (3, 3)
            or min(weights.shape[:2]) < 1):
        raise ValueError('ordinary nonempty binary32 or exact binary64 3x3 filters required')
    kernels = int(weights.shape[0])*int(weights.shape[1])
    pool.charge('c120_exact_word_filter_conversion_transform_and_materialization',
                KERNEL_FEE*kernels)
    if not np.all(np.isfinite(weights)):
        raise ValueError('finite original filter coefficients required')
    # Conversion is not a rounded coefficient source: the complete original
    # array must equal its round-trip before any decoded word is used.
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        binary32 = np.array(weights, dtype=np.float32, copy=True, order='C')
    if weights.dtype == np.float64 and not np.array_equal(binary32.astype(np.float64), weights):
        raise ValueError('original binary64 coefficient is not exactly binary32')
    bits = binary32.view(np.uint32)
    exponent = ((bits >> 23) & np.uint32(255)).astype(np.int32)
    fraction = bits & np.uint32((1 << 23)-1)
    if np.any(exponent == 255) or np.any((exponent == 0) & (fraction != 0)):
        raise ValueError('nonfinite/subnormal source outside ordinary certified word domain')
    nonzero = exponent != 0
    mantissa = (fraction | np.uint32(1 << 23)).astype(np.int64)
    mantissa[~nonzero] = 0
    mantissa[(bits >> 31) != 0] *= -1
    powers = exponent-127-23
    minimum = np.where(nonzero, powers, 300).min(axis=(-2, -1))
    minimum = np.where(minimum == 300, 0, minimum).astype(np.int32)
    shifts = np.where(nonzero, powers-minimum[..., None, None], 0)
    if np.any(shifts > MAX_ALIGNMENT_SHIFT) or np.any(shifts < 0):
        raise ValueError('complete F4 signed-int64 alignment envelope not established')
    aligned = np.left_shift(mantissa, shifts)
    if np.any(aligned <= -ALIGNED_LIMIT) or np.any(aligned >= ALIGNED_LIMIT):
        raise ValueError('aligned F4 word outside proved signed-int64 envelope')
    numerator = _separable(aligned)
    if (numerator.dtype != np.int64 or numerator.shape != (*weights.shape[:2], 6, 6)
            or np.any(numerator <= -TRANSFORM_LIMIT) or np.any(numerator >= TRANSFORM_LIMIT)):
        raise ValueError('transformed F4 word outside proved signed-int64 envelope')
    if not np.array_equal(_recover(numerator), aligned):
        raise ValueError('independent original F4 integer kernel recovery differs')
    report = dict(
        kernels=kernels, original_weights=int(weights.size),
        original_nonzero=int(np.count_nonzero(nonzero)),
        transformed_coefficients=int(numerator.size),
        transformed_nonzero=int(np.count_nonzero(numerator)),
        maximum_alignment_shift=int(shifts.max(initial=0)),
        exact_original_binary32_domain=True,
        independent_original_integer_kernel_recovery=True,
        complete_signed_int64_envelope=True,
        forward_integer_operations_per_kernel=99,
        inverse_integer_operations_per_kernel=18,
        word_program_prepaid=WORD_PROGRAM_FEE*kernels,
        source_conversion_prepaid=SOURCE_CONVERSION_FEE*kernels,
        fraction_materialization_prepaid=NUMERATOR_MATERIALIZATION_FEE*kernels,
        kernel_transform_prepaid=KERNEL_FEE*kernels,
        complete_transformed_equality_requires_independent_oracle=True,
        complete_HZ_row_window_proved=False, live_admission=False)
    return report, dict(numerator=numerator, exponent=minimum)


def construct(weights, parent_ids, parent_powers, output_ids, output_powers,
              base_n_cont, *, pool, enabled=False):
    """Emit C119-identical rows, changing only exact kernel preparation.

    Input/output masks, shared identities, normalization, row prepayment,
    ordering, native conversion, odd denominators and 512-bit bounds remain
    unchanged.  Caller must perform independent proof, complete source binding,
    physical/resource accounting and all replay/admission gates.
    """
    if not enabled:
        return None
    weights = np.asarray(weights)
    if (weights.ndim != 4 or weights.shape[-2:] != (3, 3)
            or min(weights.shape[:2]) < 1
            or weights.dtype not in (np.dtype(np.float32), np.dtype(np.float64))):
        raise ValueError('ordinary finite binary32 or exact binary64 3x3 filter block required')
    kcount, ccount = map(int, weights.shape[:2])
    pool.charge('c119_complete_tile_topology',
                1024+128*(ccount+kcount)+16*(36*ccount+16*kcount))
    ids, powers = np.asarray(parent_ids), np.asarray(parent_powers)
    outputs, outpowers = np.asarray(output_ids), np.asarray(output_powers)
    if (not isinstance(base_n_cont, (int, np.integer))
            or isinstance(base_n_cont, (bool, np.bool_))
            or not 0 <= int(base_n_cont) <= np.iinfo(np.int32).max-36*(ccount+kcount)
            or ids.shape != (ccount, 6, 6) or powers.shape != ids.shape
            or outputs.shape != (kcount, 4, 4) or outpowers.shape != outputs.shape
            or any(a.dtype.kind not in 'iu' for a in (ids, powers, outputs, outpowers))
            or np.any(ids < -1) or np.any(outputs < -1)
            or np.any(ids >= base_n_cont) or np.any(outputs >= base_n_cont)
            or np.any(powers < -511) or np.any(powers > 511)
            or np.any(outpowers < -511) or np.any(outpowers > 511)):
        raise ValueError('complete bounded original coordinate maps and powers required')
    base_n_cont = int(base_n_cont)
    input_set = set(map(int, ids[ids >= 0]))
    output_list = list(map(int, outputs[outputs >= 0]))
    if len(set(output_list)) != len(output_list) or input_set.intersection(output_list):
        raise ValueError('distinct original outputs disjoint from original inputs required')

    word_report, words = prepare_words(weights, pool=pool, enabled=True)
    kernel_fee = int(word_report['kernel_transform_prepaid'])
    numerators = np.empty((kcount, ccount, 36), dtype=object)
    for k in range(kcount):
        for c in range(ccount):
            exponent = int(words['exponent'][k, c])
            denominator = 1 << max(0, -exponent)
            shift = max(0, exponent)
            for index, word in enumerate(words['numerator'][k, c].reshape(36)):
                numerators[k, c, index] = _bounded(F(int(word) << shift, denominator))

    ids, powers = ids.reshape(ccount, 36), powers.reshape(ccount, 36)
    outputs, outpowers = outputs.reshape(kcount, 16), outpowers.reshape(kcount, 16)
    users = {}
    for k in range(kcount):
        for t in range(6):
            for u in range(6):
                needed = [s for s in range(16) if outputs[k, s] >= 0
                          and AT[s//4][t]*AT[s % 4][u]]
                if needed:
                    users[k, 6*t+u] = needed
    candidate_v = {(c, index) for k, index in users for c in range(ccount)
                   if numerators[k, c, index]}
    forms, prepaid_row_fee = {}, 0
    for c, index in sorted(candidate_v):
        t, u = divmod(index, 6)
        positions = [p for p in range(36)
                     if ids[c, p] >= 0 and BT[t][p//6]*BT[u][p % 6]]
        row_fee = 64+16*(len(positions)+2)
        pool.charge('c119_complete_row_terms_and_native', row_fee)
        prepaid_row_fee += row_fee
        merged = {}
        for p in positions:
            col = int(ids[c, p])
            value = _bounded(BT[t][p//6]*BT[u][p % 6]*_power(powers[c, p]))
            merged[col] = _bounded(merged.get(col, F(0))+value)
        merged = {col: value for col, value in merged.items() if value}
        if merged:
            forms[c, index] = merged
    channels = {(k, index): [c for c in range(ccount)
                            if (c, index) in forms and numerators[k, c, index]]
                for k, index in users}
    channels = {key: cs for key, cs in channels.items() if cs}
    keep_v = {(c, index) for (k, index), cs in channels.items() for c in cs}
    rows, vmap, mmap = [], {}, {}
    for c, index in sorted(keep_v):
        form = forms[c, index]
        exponent = _normalization(_l1(form.values()))
        slot = base_n_cont+len(rows)
        rows.append(_native([(slot, _power(exponent)),
                             *((col, -value) for col, value in form.items())],
                            slot, exponent, 1, (0, c, index)))
        vmap[c, index] = (slot, exponent)

    for k, index in sorted(channels):
        cs = channels[k, index]
        row_fee = 64+16*(len(cs)+2)
        pool.charge('c119_complete_row_terms_and_native', row_fee)
        prepaid_row_fee += row_fee
        terms = [(vmap[c, index][0],
                  _bounded(numerators[k, c, index]*_power(vmap[c, index][1])))
                 for c in cs]
        denominator = D[index//6]*D[index % 6]
        exponent = _normalization(_bounded(_l1(v for _, v in terms)/denominator))
        slot = base_n_cont+len(rows)
        pivot = _bounded(denominator*_power(exponent))
        rows.append(_native([(slot, pivot), *((col, -value) for col, value in terms)],
                            slot, exponent, denominator, (1, k, index)))
        mmap[k, index] = (slot, exponent)
    auxiliary_count = len(rows)
    for k in range(kcount):
        for s in range(16):
            if outputs[k, s] < 0:
                continue
            used = [index for index in range(36) if (k, index) in mmap
                    and AT[s//4][index//6]*AT[s % 4][index % 6]]
            row_fee = 64+16*(len(used)+2)
            pool.charge('c119_complete_row_terms_and_native', row_fee)
            prepaid_row_fee += row_fee
            exponent = int(outpowers[k, s])
            slot = int(outputs[k, s])
            coefficients = [(slot, _power(exponent))]
            coefficients += [(mmap[k, index][0],
                              _bounded(-AT[s//4][index//6]*AT[s % 4][index % 6]
                                       *_power(mmap[k, index][1]))) for index in used]
            rows.append(_native(coefficients, slot, exponent, 1, (2, k, s)))

    sizes = np.array([len(row['coefficients']) for row in rows], dtype=np.int64)
    ptr = np.r_[0, np.cumsum(sizes)].astype(np.int64)
    entries = [term for row in rows for term in row['coefficients']]
    packet = dict(
        indptr=ptr,
        columns=np.array([col for col, value in entries], dtype=np.int32),
        native=np.array([value for col, value in entries], dtype=np.float64),
        pivots=np.array([row['slot'] for row in rows], dtype=np.int32),
        gauges=np.array([row['gauge'] for row in rows], dtype=np.int32),
        rhs=np.zeros(len(rows), dtype=np.float64),
        ab_indptr=np.zeros(len(rows)+1, dtype=np.int64),
        roles=np.array([row['role'] for row in rows], dtype=np.int32).reshape(-1, 3),
        semantic_powers=np.array([row['semantic_power'] for row in rows], dtype=np.int32),
        defining_denominators=np.array([row['denominator'] for row in rows], dtype=np.int64))
    emission_fee = 64*len(rows)+16*(int(ptr[-1])+len(rows))
    if prepaid_row_fee < emission_fee:
        raise ValueError('complete row construction tariff was not prepaid')
    report = dict(
        native_coefficients_pass=True, auxiliary_boxes_redundant=True,
        kept_v=len(vmap), kept_m=len(mmap), new_factors=auxiliary_count,
        inlined_v=0, inlined_m=0, rows=len(rows), nnz=int(ptr[-1]),
        base_n_cont=base_n_cont, n_cont=base_n_cont+auxiliary_count,
        kernel_transform_prepaid=kernel_fee, row_construction_prepaid=prepaid_row_fee,
        kernel_word_report=word_report,
        exact_integer_kernel_preparation=True,
        unchanged_c119_row_semantics=True,
        whole_circuit_emission=emission_fee,
        original_input_output_ids_retained=True, original_predicates_retained_externally=True,
        semantic_units_are_not_defining_pivots=True,
        source_bound=False, global_auxiliary_gate_proved=False,
        complete_physical_reduction_proved=False, live_admission=False,
        score_gain=0, all_gates_not_admitted=True)
    return report, packet
