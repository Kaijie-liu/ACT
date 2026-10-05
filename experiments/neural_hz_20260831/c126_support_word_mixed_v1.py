"""Complete mixed F4 equations using support-compiled exact dyadic words.

This is a new row algorithm, not a reduced price for the C124 Fraction loops.
All V, M, residual and native-row arithmetic uses bounded integer words.  The
residual parent support is compiled once for the sixteen spatial positions,
then shared by every filter.  Every original kernel coefficient is decoded
once independently for the direct tail.  The unchanged C120 full-kernel fee
is nevertheless paid in full, including its unused materialization reserve.

Numeric support/source arrays are owned, ephemeral, call-local workspaces;
they are neither reusable receipts nor persistent packet evidence.  The caller
still proves the complete source, selection, inverse, physical and resource
conditions.  No routine here authorizes an admission or changes an old HZ.
"""

import math

import numpy as np

from experiments.neural_hz_20260831.c119_denominator_f4_v1 import AT, BT, D
from experiments.neural_hz_20260831.c120_word_f4_v1 import prepare_words
from experiments.neural_hz_20260831.c123_support_word_binding_v1 import (
    RATIONAL_BITS, _add, _decode, _shift, _word)


def row_fee(raw_nonpivot_terms):
    """Prepay formation, coalescence, normalization, native proof and emission.

The fixed128 covers row/pivot/role metadata and bounded gcd/log decisions;
48 per raw term (including the pivot) covers word arithmetic, L1 accumulation,
native exactness/window comparisons, sorting and literal packet retention.
Cancelled terms are not refunded.  This dominates80*rows+16*nnz emission.
"""
    return 128+48*(int(raw_nonpivot_terms)+1)


def _negative(value):
    return -value[0], value[1]


def _l1_word(values):
    total = (0, 0)
    for number, exponent in values:
        total = _add(total, (abs(number), exponent))
    return total


def _normalization_word(norm, denominator=1):
    """Exact smallest NONNEGATIVE e with norm/denominator <= 2**e.

The quotient is a general reduced rational because an M denominator includes
odd factors.  Its numerator AND denominator obey the old512-bit domain before
any potentially large shift.  No floating approximation or dyadic-only
denominator assumption is used.
"""
    number, exponent = norm
    denominator = int(denominator)
    if number < 0 or denominator <= 0:
        raise ValueError('nonnegative exact norm and positive denominator required')
    if number == 0:
        return 0
    twos = (denominator & -denominator).bit_length()-1
    odd = denominator >> twos
    common = math.gcd(number, odd)
    numerator, reduced_denominator = number//common, odd//common
    power = exponent-twos
    if (numerator.bit_length()+max(0, power) > RATIONAL_BITS
            or reduced_denominator.bit_length()+max(0, -power) > RATIONAL_BITS):
        raise ValueError('normalized exact norm outside512-bit rational domain')
    numerator <<= max(0, power)
    reduced_denominator <<= max(0, -power)
    if numerator <= reduced_denominator:
        return 0
    result = max(0, numerator.bit_length()-reduced_denominator.bit_length())
    if reduced_denominator.bit_length()+result > RATIONAL_BITS+1:
        raise ValueError('normalization comparison shift outside bounded domain')
    if numerator > reduced_denominator << result:
        result += 1
    if not 0 <= result <= 511:
        raise ValueError('semantic power outside512-bit rational domain')
    return result


def _native_word(coefficients, pivot, semantic_power, denominator, role):
    """C86's identical positive binary gauge, checked without Fraction loops."""
    values = sorted((int(column), value) for column, value in coefficients.items()
                    if value[0])
    if not values or coefficients.get(int(pivot), (0, 0))[0] <= 0:
        raise ValueError('nonempty coalesced row with positive original pivot required')
    lower, upper = -2048, 2048
    for _, (number, exponent) in values:
        magnitude = abs(number)
        floor_log = magnitude.bit_length()-1+exponent
        lower = max(lower, -20-floor_log)
        upper = min(upper, 40-floor_log-int(magnitude != 1))
    # C86 first increases zero to satisfy the minimum, then decreases it to
    # satisfy the maximum.  The same final gauge follows from exact log bounds.
    gauge = min(max(0, lower), upper)
    if gauge < lower:
        raise ValueError('complete coefficient window incompatible')
    literals = []
    for column, (number, exponent) in values:
        if abs(number).bit_length() > 53:
            raise ValueError('complete row is not exactly binary64')
        scaled = _shift((number, exponent), gauge)
        literal = math.ldexp(float(number), exponent+gauge)
        if (not math.isfinite(literal) or not 2.**-20 <= abs(literal) <= 2.**40
                or _decode(literal) != scaled):
            raise ValueError('complete row is not exactly binary64 within native window')
        literals.append((column, literal))
    return dict(coefficients=tuple(literals), rhs=0., slot=int(pivot), gauge=gauge,
                semantic_power=int(semantic_power), denominator=int(denominator),
                role=tuple(map(int, role)))


def construct(weights, parent_ids, parent_powers, output_ids, output_powers,
              base_n_cont, *, selected_channels, pool, enabled=False):
    """Return C124-compatible(report, packet); empty selection is literal None.

Complete fees, before their operations:
      unchanged topology and mask; unchanged full C1202768*K*C;
      decode512+64*9*K*C;
      coverage1024+16*(576*K+72*K*S+484*S+1296);
      support1024+32*36*C+64*144*R;
      each attempted V/emitted M/output row128+48*(raw_nonpivot_terms+1).
    S/R are selected/residual channel counts.  Output raw terms use every LIVE
    support occurrence, including zero kernel coefficients.  The complete
    144*R potential-support scan is paid once, not per filter.  All declared
    row fees include complete numeric emission with no cancellation refund.
    """
    if not enabled:
        return None
    weights = np.asarray(weights)
    if (weights.ndim != 4 or weights.shape[-2:] != (3, 3)
            or min(weights.shape[:2]) < 1
            or weights.dtype not in (np.dtype(np.float32), np.dtype(np.float64))):
        raise ValueError('ordinary binary32 or exact binary64 3x3 filter block required')
    kcount, ccount = map(int, weights.shape[:2])
    topology_fee = 1024+128*(ccount+kcount)+16*(36*ccount+16*kcount)
    pool.charge('c119_complete_tile_topology', topology_fee)
    mask_fee = 16*ccount
    pool.charge('c124_complete_selection_mask_headers_and_copy', mask_fee)
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
    if (type(selected_channels) is not np.ndarray
            or selected_channels.dtype != np.dtype(bool)
            or selected_channels.shape != (ccount,)):
        raise ValueError('complete original-channel bool selection mask required')
    base_n_cont = int(base_n_cont)
    input_set = set(map(int, ids[ids >= 0]))
    output_list = list(map(int, outputs[outputs >= 0]))
    if len(set(output_list)) != len(output_list) or input_set.intersection(output_list):
        raise ValueError('distinct original outputs disjoint from original inputs required')
    count_selected = int(np.count_nonzero(selected_channels))
    if count_selected == 0:
        return dict(
            literal_noop=True, selected_channels=0, residual_channels=ccount,
            new_factors=0, kept_v=0, kept_m=0, inlined_v=0, inlined_m=0,
            rows=0, nnz=0, base_n_cont=base_n_cont, n_cont=base_n_cont,
            topology_prepaid=topology_fee, selection_mask_prepaid=mask_fee,
            selection_mask_bytes=0, selection_mask_entries=0,
            residual_positions_visited=0, residual_nonzero_terms=0,
            residual_row_prepaid=0, kernel_transform_prepaid=0,
            row_construction_prepaid=0, whole_circuit_emission=0,
            source_decode_prepaid=0, coverage_prepaid=0, support_compile_prepaid=0,
            support_plan_entries=0, support_plan_bytes=0,
            no_kernel_preparation=True, native_coefficients_pass=False,
            auxiliary_boxes_redundant=False, source_bound=False,
            caller_must_authenticate_unchanged_C122_rule=True,
            no_once_per_operator_cache=True, exact_word_row_algorithm=True,
            global_auxiliary_gate_proved=False, complete_physical_reduction_proved=False,
            live_admission=False, score_gain=0, all_gates_not_admitted=True), None
    selection = np.array(selected_channels, dtype=bool, copy=True, order='C')
    selected = [c for c in range(ccount) if bool(selection[c])]
    residual = [c for c in range(ccount) if not bool(selection[c])]
    word_report, words = prepare_words(weights, pool=pool, enabled=True)
    kernel_fee = int(word_report['kernel_transform_prepaid'])
    numerators = words['numerator'].reshape(kcount, ccount, 36)
    kernel_exponents = words['exponent']
    ids, powers = ids.reshape(ccount, 36), powers.reshape(ccount, 36)
    outputs, outpowers = outputs.reshape(kcount, 16), outpowers.reshape(kcount, 16)

    support_fee = 1024+32*36*ccount+64*144*len(residual)
    pool.charge('c126_complete_residual_support_compilation', support_fee)
    offsets, raw_support = [0], []
    for s in range(16):
        y, x = divmod(s, 4)
        for c in residual:
            for dy in range(3):
                for dx in range(3):
                    p = 6*(y+dy)+x+dx
                    source = int(ids[c, p])
                    if source >= 0:
                        raw_support.append((source, int(powers[c, p]), c, 3*dy+dx))
        offsets.append(len(raw_support))
    support_ptr = np.array(offsets, dtype=np.int64, copy=True)
    support = np.array(raw_support, dtype=np.int64).reshape(-1, 4).copy()
    del raw_support, offsets

    decode_fee = 512+64*9*kcount*ccount
    pool.charge('c126_complete_original_kernel_word_decode', decode_fee)
    source_mantissa = np.empty((kcount, ccount, 9), dtype=np.int64)
    source_exponent = np.empty((kcount, ccount, 9), dtype=np.int32)
    for k in range(kcount):
        for c in range(ccount):
            for p in range(9):
                number, exponent = _decode(weights[k, c, p//3, p % 3])
                if abs(number).bit_length() > 24:
                    raise ValueError('complete source kernel is not an exact binary32 lift')
                source_mantissa[k, c, p] = number
                source_exponent[k, c, p] = exponent

    # Composite per-cell tariffs cover BOTH demand membership and later
    # output-component lookup, the two selected-connection passes, fixed
    # tensor basis construction, and selected parent-support membership.
    coverage_fee = 1024+16*(576*kcount+72*kcount*count_selected
                            +484*count_selected+1296)
    pool.charge('c126_complete_selected_coverage_and_basis_compilation', coverage_fee)
    basis = []
    for index in range(36):
        t, u = divmod(index, 6)
        basis.append(tuple((p, BT[t][p//6]*BT[u][p % 6]) for p in range(36)
                           if BT[t][p//6]*BT[u][p % 6]))
    users = {}
    for k in range(kcount):
        for index in range(36):
            t, u = divmod(index, 6)
            needed = [s for s in range(16) if outputs[k, s] >= 0
                      and AT[s//4][t]*AT[s % 4][u]]
            if needed:
                users[k, index] = needed
    candidate_v = {(c, index) for k, index in users for c in selected
                   if int(numerators[k, c, index]) != 0}
    forms, prepaid_row_fee = {}, 0
    candidate_v_raw = 0
    for c, index in sorted(candidate_v):
        positions = [(p, factor) for p, factor in basis[index] if ids[c, p] >= 0]
        fee = row_fee(len(positions))
        pool.charge('c126_complete_selected_V_word_row', fee)
        prepaid_row_fee += fee
        candidate_v_raw += len(positions)
        merged = {}
        for p, factor in positions:
            source = int(ids[c, p])
            value = _word(factor, int(powers[c, p]))
            merged[source] = _add(merged.get(source, (0, 0)), value)
        merged = {column: value for column, value in merged.items() if value[0]}
        if merged:
            forms[c, index] = merged
    channels = {(k, index): [c for c in selected
                            if (c, index) in forms and int(numerators[k, c, index]) != 0]
                for k, index in users}
    channels = {key: cs for key, cs in channels.items() if cs}
    # Every candidate arose from at least one demanded, nonzero kernel edge.
    # Thus each nonempty form occurs in channels; no third edge scan is needed.
    keep_v = set(forms)
    rows, vmap, mmap = [], {}, {}
    for c, index in sorted(keep_v):
        form = forms[c, index]
        exponent = _normalization_word(_l1_word(form.values()))
        slot = base_n_cont+len(rows)
        coefficients = {slot: _word(1, exponent)}
        coefficients.update((column, _negative(value)) for column, value in form.items())
        rows.append(_native_word(coefficients, slot, exponent, 1, (0, c, index)))
        vmap[c, index] = (slot, exponent)

    m_raw = 0
    for k, index in sorted(channels):
        cs = channels[k, index]
        fee = row_fee(len(cs))
        pool.charge('c126_complete_selected_M_word_row', fee)
        prepaid_row_fee += fee
        m_raw += len(cs)
        terms = {}
        for c in cs:
            source, power = vmap[c, index]
            value = _word(int(numerators[k, c, index]), int(kernel_exponents[k, c]))
            terms[source] = _shift(value, power)
        denominator = D[index//6]*D[index % 6]
        exponent = _normalization_word(_l1_word(terms.values()), denominator)
        slot = base_n_cont+len(rows)
        coefficients = {slot: _word(denominator, exponent)}
        coefficients.update((column, _negative(value)) for column, value in terms.items())
        rows.append(_native_word(coefficients, slot, exponent, denominator, (1, k, index)))
        mmap[k, index] = (slot, exponent)
    auxiliary_count = len(rows)
    residual_visits = residual_terms = residual_row_fee = output_m_raw = 0
    for k in range(kcount):
        for s in range(16):
            if outputs[k, s] < 0:
                continue
            used = [index for index in range(36) if (k, index) in mmap
                    and AT[s//4][index//6]*AT[s % 4][index % 6]]
            lo, hi = int(support_ptr[s]), int(support_ptr[s+1])
            fee = row_fee(len(used)+hi-lo)
            pool.charge('c126_complete_live_mixed_output_word_row', fee)
            prepaid_row_fee += fee
            residual_row_fee += fee
            output_m_raw += len(used)
            exponent, slot = int(outpowers[k, s]), int(outputs[k, s])
            merged = {slot: _word(1, exponent)}
            for index in used:
                source, power = mmap[k, index]
                factor = -AT[s//4][index//6]*AT[s % 4][index % 6]
                value = _word(factor, power)
                merged[source] = _add(merged.get(source, (0, 0)), value)
            for position in range(lo, hi):
                source, power, c, p = map(int, support[position])
                residual_visits += 1
                value = _word(-int(source_mantissa[k, c, p]),
                              int(source_exponent[k, c, p])+power)
                if value[0]:
                    residual_terms += 1
                    merged[source] = _add(merged.get(source, (0, 0)), value)
            rows.append(_native_word(merged, slot, exponent, 1, (2, k, s)))

    sizes = np.array([len(row['coefficients']) for row in rows], dtype=np.int64)
    ptr = np.r_[0, np.cumsum(sizes)].astype(np.int64)
    entries = [term for row in rows for term in row['coefficients']]
    packet = dict(
        indptr=ptr,
        columns=np.array([column for column, value in entries], dtype=np.int32),
        native=np.array([value for column, value in entries], dtype=np.float64),
        pivots=np.array([row['slot'] for row in rows], dtype=np.int32),
        gauges=np.array([row['gauge'] for row in rows], dtype=np.int32),
        rhs=np.zeros(len(rows), dtype=np.float64),
        ab_indptr=np.zeros(len(rows)+1, dtype=np.int64),
        roles=np.array([row['role'] for row in rows], dtype=np.int32).reshape(-1, 3),
        semantic_powers=np.array([row['semantic_power'] for row in rows], dtype=np.int32),
        defining_denominators=np.array([row['denominator'] for row in rows], dtype=np.int64),
        selected_channels=selection, new_factors=int(auxiliary_count))
    emission_fee = 80*len(rows)+16*int(ptr[-1])
    if prepaid_row_fee < emission_fee:
        raise ValueError('complete mixed row construction/emission tariff was not prepaid')
    report = dict(
        literal_noop=False, native_coefficients_pass=True, auxiliary_boxes_redundant=True,
        kept_v=len(vmap), kept_m=len(mmap), new_factors=auxiliary_count,
        inlined_v=0, inlined_m=0, rows=len(rows), nnz=int(ptr[-1]),
        base_n_cont=base_n_cont, n_cont=base_n_cont+auxiliary_count,
        selected_channels=count_selected, residual_channels=len(residual),
        selection_mask_bytes=int(selection.nbytes), selection_mask_entries=int(selection.size),
        selection_mask_is_owned=True, selection_mask_cost_additional_to_packet_rows=True,
        topology_prepaid=topology_fee, selection_mask_prepaid=mask_fee,
        residual_positions_visited=residual_visits, residual_nonzero_terms=residual_terms,
        residual_row_prepaid=residual_row_fee,
        kernel_transform_prepaid=kernel_fee, row_construction_prepaid=prepaid_row_fee,
        source_decode_prepaid=decode_fee, coverage_prepaid=coverage_fee,
        support_compile_prepaid=support_fee,
        support_potential_positions_scanned=144*len(residual),
        support_live_occurrences=int(support.shape[0]),
        support_plan_entries=int(support.size+support_ptr.size),
        support_plan_bytes=int(support.nbytes+support_ptr.nbytes),
        support_plan_is_owned=bool(support.flags.owndata and support_ptr.flags.owndata),
        source_word_entries=int(source_mantissa.size+source_exponent.size),
        source_word_bytes=int(source_mantissa.nbytes+source_exponent.nbytes),
        source_words_are_owned=bool(source_mantissa.flags.owndata
                                    and source_exponent.flags.owndata),
        support_and_source_workspaces_ephemeral=True,
        candidate_v_rows=len(candidate_v), candidate_v_raw_terms=candidate_v_raw,
        m_raw_terms=m_raw, output_m_raw_terms=output_m_raw,
        exact_word_row_algorithm=True, row_hotpath_has_no_fraction_arithmetic=True,
        exact_general_rational_norm_including_odd_denominator=True,
        kernel_word_report=word_report, exact_integer_kernel_preparation=True,
        full_original_kernel_prepared_each_nonempty_call=True,
        full_unchanged_C120_fee_including_unused_materialization_reserve=True,
        no_once_per_operator_cache=True, whole_circuit_emission=emission_fee,
        original_input_output_ids_retained=True, original_predicates_retained_externally=True,
        semantic_units_are_not_defining_pivots=True,
        all_residual_direct_terms_in_same_original_output_equation=True,
        caller_must_authenticate_unchanged_C122_rule=True,
        source_bound=False, global_auxiliary_gate_proved=False,
        complete_physical_reduction_proved=False, live_admission=False,
        score_gain=0, all_gates_not_admitted=True)
    return report, packet
