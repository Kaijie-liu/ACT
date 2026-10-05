"""Exact mixed selected-channel F4 and residual direct convolution equations.

One M per filter/component jointly sums the selected channels.  Every other
channel stays direct in the SAME original output row.  This low-level theorem
constructor does not choose channels or authenticate a structural selection:
its caller must enforce the unchanged C122 rule and prove the actual source,
whole physical cost, shared reserves and all subsequent admission conditions.

Each nonempty invocation deliberately pays the complete unchanged C120 kernel
preparation and Fraction materialization for ALL K*C kernels.  There is no
external prepared receipt, subset preparation or once-per-operator cache.
"""

from fractions import Fraction as F

import numpy as np

from experiments.neural_hz_20260831.c119_denominator_f4_v1 import (
    AT, BT, D, _bounded, _l1, _native, _normalization, _power)
from experiments.neural_hz_20260831.c120_word_f4_v1 import prepare_words


def construct(weights, parent_ids, parent_powers, output_ids, output_powers,
              base_n_cont, *, selected_channels, pool, enabled=False):
    """Return (report, packet), or (literal-noop report, None) for an empty mask.

    Original map, signed-index, power and native row domains are C120's.  The
    additional bool[C] mask is copied into each nonempty packet, and its full
    persistent byte/entry costs are reported separately.  An empty mask still
    validates all maps/headers/mask, but does not inspect kernel values, prepare
    kernels, construct rows or claim source/native proof for the untouched HZ.
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
            no_kernel_preparation=True, native_coefficients_pass=False,
            auxiliary_boxes_redundant=False, source_bound=False,
            caller_must_authenticate_unchanged_C122_rule=True,
            global_auxiliary_gate_proved=False, complete_physical_reduction_proved=False,
            live_admission=False, score_gain=0, all_gates_not_admitted=True), None
    selection = np.array(selected_channels, dtype=bool, copy=True, order='C')
    selected = [c for c in range(ccount) if bool(selection[c])]
    residual = [c for c in range(ccount) if not bool(selection[c])]

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
    candidate_v = {(c, index) for k, index in users for c in selected
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
    channels = {(k, index): [c for c in selected
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
    residual_visits = residual_terms = residual_row_fee = 0
    for k in range(kcount):
        for s in range(16):
            if outputs[k, s] < 0:
                continue
            used = [index for index in range(36) if (k, index) in mmap
                    and AT[s//4][index//6]*AT[s % 4][index % 6]]
            # ALL residual positions are prepaid, including absent parents and
            # zero coefficients.  Exact cancellations do not refund work.
            potential_residual = 9*len(residual)
            row_fee = 64+16*(len(used)+potential_residual+2)
            pool.charge('c124_complete_mixed_output_row_and_all_residual_visits', row_fee)
            prepaid_row_fee += row_fee
            residual_row_fee += row_fee
            exponent = int(outpowers[k, s])
            slot = int(outputs[k, s])
            merged = {slot: _power(exponent)}
            for index in used:
                source, unit = mmap[k, index]
                value = _bounded(-AT[s//4][index//6]*AT[s % 4][index % 6]*_power(unit))
                merged[source] = _bounded(merged.get(source, F(0))+value)
            y, x = divmod(s, 4)
            for c in residual:
                for dy in range(3):
                    for dx in range(3):
                        residual_visits += 1
                        p = 6*(y+dy)+x+dx
                        source = int(ids[c, p])
                        if source < 0:
                            continue
                        value = _bounded(-_bounded(F(float(weights[k, c, dy, dx])))
                                         *_power(powers[c, p]))
                        if value:
                            residual_terms += 1
                            merged[source] = _bounded(merged.get(source, F(0))+value)
            coefficients = [(column, value) for column, value in merged.items() if value]
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
        defining_denominators=np.array([row['denominator'] for row in rows], dtype=np.int64),
        selected_channels=selection, new_factors=int(auxiliary_count))
    emission_fee = 64*len(rows)+16*(int(ptr[-1])+len(rows))
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
        kernel_word_report=word_report, exact_integer_kernel_preparation=True,
        full_original_kernel_prepared_each_nonempty_call=True,
        no_once_per_operator_cache=True, whole_circuit_emission=emission_fee,
        original_input_output_ids_retained=True, original_predicates_retained_externally=True,
        semantic_units_are_not_defining_pivots=True,
        all_residual_direct_terms_in_same_original_output_equation=True,
        caller_must_authenticate_unchanged_C122_rule=True,
        source_bound=False, global_auxiliary_gate_proved=False,
        complete_physical_reduction_proved=False, live_admission=False,
        score_gain=0, all_gates_not_admitted=True)
    return report, packet
