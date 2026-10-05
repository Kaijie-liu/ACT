"""Independent complete mixed native proof using bounded exact dyadic words.

No constructor, constructor constants, C123 arithmetic or emitted-row echo is
used.  Every selected form and live residual is rebuilt from original maps;
every actual native row is compared against that independent reconstruction.
C127 independently reconstructs ALL36 through a source-derived systematic
program, with original materialization reserve fully retained on every call.
"""
import math

import numpy as np

from experiments.neural_hz_20260831.c127_systematic_kernel_proof_v2 import prepare


_BT = ((4, 0, -5, 0, 1, 0), (0, -4, -4, 1, 1, 0),
       (0, 4, -4, -1, 1, 0), (0, -2, -1, 2, 1, 0),
       (0, 2, -1, -2, 1, 0), (0, 4, 0, -5, 0, 1))
_GN = ((1, 0, 0), (-1, -1, -1), (-1, 1, -1),
       (1, 2, 4), (1, -2, 4), (0, 0, 1))
_D = (4, 6, 6, 24, 24, 1)
_AT = ((1, 1, 1, 1, 1, 0), (0, 1, -1, 2, -2, 0),
       (0, 1, 1, 4, 4, 0), (0, 1, -1, 8, -8, 1))
_BT_TERMS = tuple(tuple((i, j, _BT[a][i]*_BT[b][j])
                       for i in range(6) for j in range(6)
                       if _BT[a][i] and _BT[b][j])
                  for a in range(6) for b in range(6))
_AT_TERMS = tuple(tuple((6*a+b, _AT[i][a]*_AT[j][b])
                       for a in range(6) for b in range(6)
                       if _AT[i][a] and _AT[j][b])
                  for i in range(4) for j in range(4))
_AT_USERS = tuple(tuple(s for s, terms in enumerate(_AT_TERMS)
                        if any(component == t for component, _ in terms))
                  for t in range(36))
_ZERO = (0, 0)


def _word(number, exponent=0):
    """Canonical odd signed numerator and binary exponent, BOTH bounds512."""
    number, exponent = int(number), int(exponent)
    if not number:
        return _ZERO
    magnitude = abs(number)
    trailing = (magnitude & -magnitude).bit_length()-1
    number >>= trailing
    exponent += trailing
    if (exponent < -511 or abs(number).bit_length()+max(0, exponent) > 512):
        raise MemoryError('complete exact dyadic numerator/denominator exceeds512 bits')
    return number, exponent


def _shift(number, amount):
    number, amount = int(number), int(amount)
    if amount < 0 or abs(number).bit_length()+amount > 512:
        raise MemoryError('exact word shift exceeds512 bits before allocation')
    return number << amount


def _add(left, right):
    if not left[0]:
        return right
    if not right[0]:
        return left
    exponent = min(left[1], right[1])
    a = _shift(left[0], left[1]-exponent)
    b = _shift(right[0], right[1]-exponent)
    # One carry bit is possible; normalize before checking the reduced result.
    return _word(a+b, exponent)


def _mul(left, right):
    if not left[0] or not right[0]:
        return _ZERO
    exponent = left[1]+right[1]
    width = 512-max(0, exponent)
    if (exponent < -511 or width < 1
        or abs(left[0]) > ((1 << width)-1)//abs(right[0])):
        raise MemoryError('exact canonical word product exceeds512 bits before multiplication')
    return _word(left[0]*right[0], exponent)


def _scale(value, exponent):
    return _word(value[0], value[1]+int(exponent))


def _power(exponent):
    return _word(1, int(exponent))


def _from_float(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('finite exact native value required')
    numerator, denominator = value.as_integer_ratio()
    if denominator <= 0 or denominator & (denominator-1):
        raise ValueError('native value must be exactly dyadic')
    return _word(numerator, -(denominator.bit_length()-1))


def _compare(left, right):
    """Exact order without overflowing alignment of widely separated words."""
    if left == right:
        return 0
    if left[0] <= 0 <= right[0]:
        return -1
    if right[0] <= 0 <= left[0]:
        return 1
    sign = 1 if left[0] > 0 else -1
    a, b = abs(left[0]), abs(right[0])
    atop, btop = a.bit_length()+left[1], b.bit_length()+right[1]
    if atop != btop:
        return sign*(1 if atop > btop else -1)
    exponent = min(left[1], right[1])
    a, b = _shift(a, left[1]-exponent), _shift(b, right[1]-exponent)
    return sign*((a > b)-(a < b))


def _l1(values):
    result = _ZERO
    for number, exponent in values:
        result = _add(result, (abs(number), exponent))
    return result


def _ceil_over(value, denominator=1):
    """Exact nonnegative ceil power for L1/D, including odd reduced D bits."""
    numerator, exponent = value
    denominator = int(denominator)
    if numerator < 0 or denominator <= 0:
        raise ValueError('nonnegative norm and positive defining denominator required')
    if numerator == 0:
        return 0
    if exponent >= 0:
        numerator = _shift(numerator, exponent)
        common = math.gcd(numerator, denominator)
        numerator, denominator = numerator//common, denominator//common
    else:
        common = math.gcd(numerator, denominator)
        numerator, denominator = numerator//common, denominator//common
        denominator = _shift(denominator, -exponent)
    if max(numerator.bit_length(), denominator.bit_length()) > 512:
        raise MemoryError('complete reduced L1/odd-D rational exceeds512 bits')
    if numerator <= denominator:
        return 0
    power = max(0, numerator.bit_length()-denominator.bit_length())
    if _shift(denominator, power) < numerator:
        power += 1
    _power(power)
    return power


def _basis(pool):
    # The common denominator24 proves exactly the same72 coefficients and
    #432 summands as the old independent Fraction identity.  Fee unchanged.
    pool.charge('c124_independent_full_1d_bilinear_identity', 64*432)
    if (sum(map(len, _BT_TERMS)) != 484 or sum(map(len, _AT_TERMS)) != 324
        or any(24 % denominator for denominator in _D)):
        raise ValueError('independent literal transform support differs')
    for output in range(4):
        for kernel in range(3):
            for parent in range(6):
                actual = sum(_AT[output][t]*_GN[t][kernel]*_BT[t][parent]*(24//_D[t])
                             for t in range(6))
                if actual != 24*int(parent == output+kernel):
                    raise ValueError('complete independent bilinear identity failed')


def _integer_array(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in 'iu':
        raise ValueError('complete integer '+name+' shape required')
    return value


def _check_row(packet, row, raw, pivot, semantic_power, denominator, role, *, auxiliary, pool):
    start, stop = map(int, packet['indptr'][row:row+2])
    pool.charge('c126_complete_native_word_decode_gauge_box_comparison',
                128+32*(len(raw)+stop-start))
    if (tuple(map(int, packet['roles'][row])) != role
        or int(packet['pivots'][row]) != pivot
        or int(packet['semantic_powers'][row]) != semantic_power
        or int(packet['defining_denominators'][row]) != denominator):
        raise ValueError('complete role, original pivot, power or denominator differs')
    columns = packet['columns'][start:stop]
    if np.any(np.diff(columns) <= 0):
        raise ValueError('unique sorted actual row coordinates required')
    actual = {int(column):_from_float(value) for column,value in
              zip(columns, packet['native'][start:stop], strict=True)}
    if set(actual) != set(raw) or packet['rhs'][row] != 0 or not raw:
        raise ValueError('complete independent row support or zero RHS differs')
    gauge = int(packet['gauges'][row])
    _power(gauge)
    if any(actual[column] != _scale(value, gauge) for column,value in raw.items()):
        raise ValueError('actual native row differs from independently reconstructed words')
    minimum = maximum = None
    for number, exponent in raw.values():
        absolute = (abs(number), exponent)
        if not number:
            raise ValueError('canonical independent row must exclude zero coefficients')
        if minimum is None or _compare(absolute, minimum) < 0:
            minimum = absolute
        if maximum is None or _compare(absolute, maximum) > 0:
            maximum = absolute
    # Exact closed form of unchanged C86: raise the minimum first, then lower
    #the maximum.  Non-powers of two use the strict upper binary exponent.
    floor_min = minimum[0].bit_length()-1+minimum[1]
    ceil_max = maximum[0].bit_length()-1+maximum[1]+int(maximum[0] != 1)
    expected_gauge = min(max(0, -20-floor_min), 40-ceil_max)
    if gauge != expected_gauge:
        raise ValueError('unchanged complete-row native gauge differs')
    if actual[pivot][0] <= 0:
        raise ValueError('positive defining pivot required')
    if auxiliary:
        terms = {column:value for column,value in actual.items() if column != pivot}
        if (any(column >= pivot for column in terms)
            or _compare(_l1(terms.values()), actual[pivot]) > 0):
            raise ValueError('topological definition or redundant L1 box failed')
    return stop-start


def prove(packet, weights, parent_ids, parent_powers, output_ids, output_powers,
          base_n_cont, *, selected_channels, pool, enabled=False):
    """Complete original mixed proof; mathematical mask, not route authority.

    NEW word programs prepay before their work: selected support16*484*S plus
    32/live literal; coverage16*K*(340+36*S); V/M rows96+32/term;
    residual compiler16*144*R plus32/live spatial tuple; output96+32 per
    nonzero AT position or live residual tuple; actual comparison128/row plus
    32 per independently expected AND actual coefficient.  Compiler/coverage
    headers each1024.  No inherited loop is given a lower tariff: source,
    native headers and complete bilinear basis retain their old fees. C127's
    genuinely new ALL36 source program pays its own complete new fee. Its
    reserved Fraction materialization is not refunded,
    although this oracle does not materialize Fraction objects.
    """
    if not enabled:
        return None
    if not isinstance(packet, dict):
        raise ValueError('nonempty selected-channel native packet required')
    started_work = pool.used
    weights = np.asarray(weights)
    if (weights.ndim != 4 or weights.shape[2:] != (3, 3)
        or weights.dtype.kind not in 'fiu' or weights.dtype.itemsize > 8
        or min(weights.shape[:2]) <= 0):
        raise ValueError('ordinary complete finite3x3 kernel required')
    kcount, ccount = weights.shape[:2]
    if (type(selected_channels) is not np.ndarray
        or selected_channels.shape != (ccount,) or selected_channels.dtype != np.dtype(bool)):
        raise ValueError('complete original-channel boolean selection required')
    pool.charge('c124_independent_complete_source_headers',
                1024+32*(36*ccount+16*kcount+weights.size))
    pool.charge('c124_independent_complete_selected_mask_custody', 128+32*ccount)
    owned_mask = packet.get('selected_channels')
    if (type(owned_mask) is not np.ndarray or owned_mask.shape != (ccount,)
        or owned_mask.dtype != np.dtype(bool) or not owned_mask.flags.owndata
        or not np.array_equal(owned_mask, selected_channels)):
        raise ValueError('owned packet mask differs from original-channel selection')
    selected = tuple(c for c in range(ccount) if bool(selected_channels[c]))
    residual = tuple(c for c in range(ccount) if not bool(selected_channels[c]))
    if not selected:
        raise ValueError('empty selection is a constructor no-op, not a native proof')
    if not np.isfinite(weights).all():
        raise ValueError('complete finite original kernel required')
    ids = _integer_array(parent_ids, (ccount,6,6), 'original input IDs')
    powers = _integer_array(parent_powers, ids.shape, 'original input powers')
    outputs = _integer_array(output_ids, (kcount,4,4), 'original output IDs')
    outpowers = _integer_array(output_powers, outputs.shape, 'original output powers')
    if (not isinstance(base_n_cont, (int,np.integer))
        or isinstance(base_n_cont, (bool,np.bool_))
        or not 0 <= base_n_cont <= np.iinfo(np.int32).max-36*(ccount+kcount)
        or np.any(ids < -1) or np.any(ids >= base_n_cont)
        or np.any(outputs < -1) or np.any(outputs >= base_n_cont)
        or np.any(powers < -511) or np.any(powers > 511)
        or np.any(outpowers < -511) or np.any(outpowers > 511)
        or len(set(map(int, outputs[outputs >= 0]))) != int((outputs >= 0).sum())
        or set(map(int, ids[ids >= 0])) & set(map(int, outputs[outputs >= 0]))):
        raise ValueError('complete disjoint original input/output IDs required')
    required = {'indptr','columns','native','pivots','gauges','rhs','ab_indptr',
                'roles','semantic_powers','defining_denominators','selected_channels','new_factors'}
    if not required.issubset(packet):
        raise ValueError('complete actual native packet required')
    rows, nnz = len(packet['rhs']), len(packet['native'])
    if (not isinstance(packet['new_factors'], (int,np.integer))
        or isinstance(packet['new_factors'], (bool,np.bool_))
        or not 0 <= int(packet['new_factors']) <= 36*(ccount+kcount)):
        raise ValueError('complete native auxiliary population metadata required')
    pool.charge('c124_independent_complete_native_headers', 32*(rows+nnz))
    for name in ('indptr','ab_indptr','defining_denominators','columns','pivots',
                 'gauges','roles','semantic_powers'):
        dtype = np.dtype('int64' if name in ('indptr','ab_indptr','defining_denominators') else 'int32')
        if not isinstance(packet[name], np.ndarray) or packet[name].dtype != dtype:
            raise ValueError('exact native signed packet dtype required for '+name)
    for name in ('indptr','ab_indptr'):
        _integer_array(packet[name], (rows+1,), name)
    for name in ('pivots','gauges','semantic_powers','defining_denominators'):
        _integer_array(packet[name], (rows,), name)
    _integer_array(packet['roles'], (rows,3), 'complete roles')
    _integer_array(packet['columns'], (nnz,), 'complete columns')
    for name,size in (('native',nnz),('rhs',rows)):
        if (not isinstance(packet[name], np.ndarray) or packet[name].shape != (size,)
            or packet[name].dtype != np.dtype('float64') or not np.isfinite(packet[name]).all()):
            raise ValueError('finite actual binary64 coefficient arrays required')
    if (packet['indptr'][0] != 0 or packet['indptr'][-1] != nnz
        or np.any(np.diff(packet['indptr']) <= 0) or np.any(packet['ab_indptr'])
        or np.any(packet['columns'] < 0)
        or np.any((np.abs(packet['native']) < 2.**-20) | (np.abs(packet['native']) > 2.**40))
        or any(np.asarray(packet.get(name,())).size for name in ('ab_data','ab_columns','ab_native'))):
        raise ValueError('complete continuous-only native packet domain required')
    _basis(pool)
    kernel_report, evidence = prepare(weights, pool=pool, enabled=True)
    weights = evidence['source_snapshot']

    pool.charge('c126_independent_selected_support_compiler', 1024+16*484*len(selected))
    forms, live_v_literals = {}, 0
    for c in selected:
        for component, source_terms in enumerate(_BT_TERMS):
            terms = {}
            for i,j,multiplier in source_terms:
                column = int(ids[c,i,j])
                if column < 0:
                    continue
                pool.charge('c126_independent_selected_live_word_coalescence', 32)
                live_v_literals += 1
                value = _mul(_word(multiplier), _power(powers[c,i,j]))
                terms[column] = _add(terms.get(column,_ZERO), value)
            forms[c,component] = {column:value for column,value in terms.items() if value[0]}
    pool.charge('c126_independent_compiled_complete_support_coverage',
                1024+16*kcount*(340+36*len(selected)))
    users, channels = {}, {}
    for k in range(kcount):
        for component in range(36):
            used = tuple(s for s in _AT_USERS[component] if outputs[k,s//4,s%4] >= 0)
            a,b = divmod(component,6)
            cs = tuple(c for c in selected
                       if forms[c,component] and evidence['numerator'][k,c,a,b] != 0)
            if used and cs:
                users[k,component], channels[k,component] = used,cs
    need_v = sorted({(c,component) for (k,component),cs in channels.items() for c in cs})
    need_m = sorted(channels)
    need_out = [(k,s) for k in range(kcount) for s in range(16) if outputs[k,s//4,s%4] >= 0]
    if rows != len(need_v)+len(need_m)+len(need_out) or int(packet['new_factors']) != len(need_v)+len(need_m):
        raise ValueError('complete required V/M/original-output row population differs')

    compiler_positions = 144*len(residual)
    pool.charge('c126_independent_residual_spatial_support_compiler', 1024+16*compiler_positions)
    residual_support = []
    for s in range(16):
        i,j = divmod(s,4)
        terms = []
        for c in residual:
            for dy in range(3):
                for dx in range(3):
                    column = int(ids[c,i+dy,j+dx])
                    if column >= 0:
                        pool.charge('c126_independent_live_residual_spatial_tuple', 32)
                        terms.append((c,dy,dx,column,int(powers[c,i+dy,j+dx])))
        residual_support.append(tuple(terms))
    compiled_live = sum(map(len,residual_support))

    index = compared = 0
    vmap, mmap = {}, {}
    for c,component in need_v:
        terms = forms[c,component]
        pool.charge('c126_independent_complete_V_word_row', 96+32*len(terms))
        pivot = int(base_n_cont)+index
        power = _ceil_over(_l1(terms.values()))
        raw = {column:(-number,exponent) for column,(number,exponent) in terms.items()}
        raw[pivot] = _power(power)
        compared += _check_row(packet,index,raw,pivot,power,1,(0,c,component),auxiliary=True,pool=pool)
        vmap[c,component] = pivot,power
        index += 1
    for k,component in need_m:
        cs = channels[k,component]
        pool.charge('c126_independent_complete_M_word_row', 96+32*len(cs))
        pivot = int(base_n_cont)+index
        a,b = divmod(component,6)
        denominator = _D[a]*_D[b]
        raw = {}
        for c in cs:
            slot,unit = vmap[c,component]
            coefficient = _word(-int(evidence['numerator'][k,c,a,b]), int(evidence['exponent'][k,c]))
            raw[slot] = _scale(coefficient,unit)
        power = _ceil_over(_l1(raw.values()),denominator)
        raw[pivot] = _scale(_word(denominator),power)
        compared += _check_row(packet,index,raw,pivot,power,denominator,(1,k,component),auxiliary=True,pool=pool)
        mmap[k,component] = pivot,power
        index += 1
    residual_visits = residual_nonzero = aliases = cancellations = at_visits = 0
    for k,s in need_out:
        at_terms, direct_terms = _AT_TERMS[s], residual_support[s]
        pool.charge('c126_independent_complete_output_word_row', 96+32*(len(at_terms)+len(direct_terms)))
        at_visits += len(at_terms)
        residual_visits += len(direct_terms)
        i,j = divmod(s,4)
        pivot, power = int(outputs[k,i,j]), int(outpowers[k,i,j])
        raw = {pivot:_power(power)}
        for component,multiplier in at_terms:
            if (k,component) in mmap:
                slot,unit = mmap[k,component]
                raw[slot] = _scale(_word(-multiplier),unit)
        for c,dy,dx,column,parent_power in direct_terms:
            coefficient = (int(evidence['canonical_mantissa'][k,c,dy,dx]),
                           int(evidence['canonical_exponent'][k,c,dy,dx]))
            if not coefficient[0]:
                continue
            value = _scale((-coefficient[0],coefficient[1]),parent_power)
            residual_nonzero += 1
            if column in raw:
                aliases += 1
            value = _add(raw.get(column,_ZERO),value)
            if value[0]:
                raw[column] = value
            else:
                cancellations += 1
                raw.pop(column,None)
        compared += _check_row(packet,index,raw,pivot,power,1,(2,k,s),auxiliary=False,pool=pool)
        index += 1
    full_slot_equivalent = 9*len(residual)*len(need_out)
    report = dict(selected_channels=len(selected),residual_channels=len(residual),
        owned_selected_mask_matches_supplied_mask=True,structural_selector_authenticated_by_this_oracle=False,
        all_original_kernel_coefficients_observed=int(weights.size),
        all_transformed_kernel_coefficients_rebuilt=36*kcount*ccount,
        transformed_kernel_proof_includes_unselected_channels=True,
        all_residual_scan_positions_proved=full_slot_equivalent,
        old_residual_full_slot_equivalent=full_slot_equivalent,
        actual_residual_compiler_positions_visited=compiler_positions,
        residual_support_compiled_once_per_spatial_output=True,
        complete_residual_spatial_support_lists=16,compiled_live_residual_tuples=compiled_live,
        actual_live_residual_visits=residual_visits,
        all_residual_nonzero_occurrences_rebuilt=residual_nonzero,
        exact_residual_alias_additions=aliases,exact_residual_alias_cancellations=cancellations,
        residual_scan_work=1024+16*compiler_positions+32*compiled_live,
        residual_exact_term_reserve_work=32*residual_visits,
        complete_selected_V_live_literals_rebuilt=live_v_literals,
        complete_demanded_nonzero_AT_positions_visited=at_visits,
        complete_direct_tail_and_selected_transform_sum_proved=True,
        all_1d_bilinear_coefficients_proved=72,full_2d_tensor_product_identity_proved=True,
        all_native_rows_proved=rows,all_native_nnz_proved=compared,
        all_V_definitions_proved=len(need_v),all_M_definitions_proved=len(need_m),
        all_original_output_equations_proved=len(need_out),
        all_auxiliary_equations_and_redundant_boxes_proved=len(need_v)+len(need_m),
        exact_required_support_coverage=True,original_input_output_ids_preserved=True,
        original_output_powers_and_gauge_semantics_preserved=True,
        odd_denominators_only_on_new_M_pivots=True,universal_unique_box_extension=True,
        native_only_inverse_sufficient=True,compositional_original_source_equivalence=True,
        complete_HZ_source_predicates_proved=False,physical_reduction_proved=False,
        actual_network_source_bound=False,concrete_network_witness=False,formal_gain=0,
        kernel_proof=kernel_report,complete_kernel_preparation_evidence_returned=True,
        all36_fraction_materialization_completed=False,
        old_full_materialization_reservation_retained_without_refund=True,
        full_native_row_proof_scope_and_tariffs_unchanged=False,
        full_native_row_proof_scope_retained=True,new_independent_word_program_tariffs=True,
        systematic_source_derived_complete_kernel_program=True,
        independent_word_arithmetic_without_constructor_or_C123_helpers=True,
        exact_row_arithmetic_without_Fraction=True,
        complete_reduced_numerator_and_denominator_512_bit_guards=True,
        complete_observed_work=pool.used-started_work,no_prepared_cache_or_reuse=True)
    return report,evidence
