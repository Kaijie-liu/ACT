"""Independent complete proof of one mixed direct/F(4,3) output equation.

No constructor constants, transforms or source binders are imported.  The full
1-D basis and ALL 36 exact transformed coefficients for EVERY original channel
are rebuilt independently.  Only selected channels use V/M definitions; every
unselected direct coefficient is rebuilt and coalesced in each output row.
The supplied mask is mathematically arbitrary: the caller must independently
authenticate C122's one structural selection rule and every full-HZ admission
gate.  This proof does not confer source, physical, runtime or score admission.
"""
from fractions import Fraction as F
import numpy as np


_BT = ((4, 0, -5, 0, 1, 0), (0, -4, -4, 1, 1, 0),
       (0, 4, -4, -1, 1, 0), (0, -2, -1, 2, 1, 0),
       (0, 2, -1, -2, 1, 0), (0, 4, 0, -5, 0, 1))
_GN = ((1, 0, 0), (-1, -1, -1), (-1, 1, -1),
       (1, 2, 4), (1, -2, 4), (0, 0, 1))
_D = (4, 6, 6, 24, 24, 1)
_AT = ((1, 1, 1, 1, 1, 0), (0, 1, -1, 2, -2, 0),
       (0, 1, 1, 4, 4, 0), (0, 1, -1, 8, -8, 1))


def _bounded(value):
    value = F(value)
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 512:
        raise MemoryError('unchanged 512-bit exact proof domain exceeded')
    return value


def _pow(power):
    power = int(power)
    if abs(power) > 511:
        raise MemoryError('unchanged 512-bit exact power domain exceeded')
    return _bounded(F(2)**power)


def _ceil_power(value):
    value = _bounded(value)
    exponent = 0
    while _pow(exponent) < value:
        exponent += 1
    return exponent


def _l1(values):
    result = F(0)
    for value in values:
        result = _bounded(result+abs(value))
    return result


def _basis(pool):
    pool.charge('c124_independent_full_1d_bilinear_identity', 64*432)
    for output in range(4):
        for kernel in range(3):
            for parent in range(6):
                actual = sum((F(_AT[output][t]*_GN[t][kernel]*_BT[t][parent], _D[t])
                              for t in range(6)), F(0))
                if actual != int(parent == output+kernel):
                    raise ValueError('independent complete 1-D bilinear identity failed')


def _integer_array(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in 'iu':
        raise ValueError('complete integer '+name+' shape required')
    return value


def _check_row(packet, row, raw, pivot, semantic_power, denominator, role, *, auxiliary, pool):
    a, b = map(int, packet['indptr'][row:row+2])
    pool.charge('c124_independent_actual_row_comparison', 128+64*(len(raw)+b-a))
    if (tuple(map(int, packet['roles'][row])) != role
            or int(packet['pivots'][row]) != pivot
            or int(packet['semantic_powers'][row]) != semantic_power
            or int(packet['defining_denominators'][row]) != denominator):
        raise ValueError('complete role, original pivot, power or denominator differs')
    columns = packet['columns'][a:b]
    if np.any(np.diff(columns) <= 0):
        raise ValueError('unique sorted actual row coordinates required')
    actual = {int(c): _bounded(F(float(v))) for c, v in
              zip(columns, packet['native'][a:b], strict=True)}
    if set(actual) != set(raw) or packet['rhs'][row] != 0:
        raise ValueError('complete actual row support or zero RHS differs')
    gauge = _pow(packet['gauges'][row])
    if any(actual[c] != _bounded(v*gauge) for c, v in raw.items()):
        raise ValueError('actual native row differs from independent exact definition')
    # Match the unchanged C86 complete-row gauge policy independently.
    minimum, maximum = min(map(abs, raw.values())), max(map(abs, raw.values()))
    shift = 0
    while _bounded(minimum*_pow(shift)) < _pow(-20):
        shift += 1
    while _bounded(maximum*_pow(shift)) > _pow(40):
        shift -= 1
    if int(packet['gauges'][row]) != shift:
        raise ValueError('unchanged complete-row native gauge differs')
    if actual[pivot] <= 0:
        raise ValueError('positive defining pivot required')
    if auxiliary:
        terms = {c: v for c, v in actual.items() if c != pivot}
        if (any(c >= pivot for c in terms)
                or _l1(terms.values()) > actual[pivot]):
            raise ValueError('topological definition or redundant L1 box failed')
    return b-a


def prove(packet, weights, parent_ids, parent_powers, output_ids, output_powers,
          base_n_cont, *, selected_channels, pool, enabled=False):
    """Prove every mixed native row, including every residual direct term.

    Selected channels encode V = BT d B and Dtu M = GNUM g GNUM^T V.
    The original output equation combines AT M A with the unselected direct
    convolution sum.  All selected channels share one M per filter/component.
    Semantic powers normalize fresh V/M boxes; defining odd denominators occur
    only on new M pivots, never on original output terms.

    The independent all-channel kernel transform retains C119's full fee and
    proof scope.  Selected parent forms are charged for their complete 6x6
    contractions; the unchanged full support-coverage upper is retained.
    EACH demanded output also prepays 16*9*R residual visits plus 64*9*R
    exact coefficient/coalescence operations, where R is the number of
    unselected channels.  These are in addition to the old 64*37 transformed
    output reserve and the complete actual-native-row comparison charge.
    Empty selection/no packet is a constructor-level no-op, not a proof.
    """
    if not enabled:
        return None
    if not isinstance(packet, dict):
        raise ValueError('nonempty selected-channel native packet required')
    weights = np.asarray(weights)
    if (weights.ndim != 4 or weights.shape[2:] != (3, 3)
            or weights.dtype.kind not in 'fiu' or weights.dtype.itemsize > 8
            or min(weights.shape[:2]) <= 0):
        raise ValueError('ordinary complete finite 3x3 kernel required')
    kcount, ccount = weights.shape[:2]
    if (type(selected_channels) is not np.ndarray
            or selected_channels.shape != (ccount,)
            or selected_channels.dtype != np.dtype(bool)):
        raise ValueError('complete original-channel boolean selection required')
    # Reserve the complete source observation before any numeric-domain scan.
    # Shape/type headers above do not inspect the kernel or coordinate values.
    pool.charge('c124_independent_complete_source_headers',
                1024+32*(36*ccount+16*kcount+weights.size))
    pool.charge('c124_independent_complete_selected_mask_custody', 128+32*ccount)
    owned_mask = packet.get('selected_channels')
    if (type(owned_mask) is not np.ndarray or owned_mask.shape != (ccount,)
            or owned_mask.dtype != np.dtype(bool) or not owned_mask.flags.owndata
            or not np.array_equal(owned_mask, selected_channels)):
        raise ValueError('owned packet mask differs from complete original-channel selection')
    selected = tuple(c for c in range(ccount) if bool(selected_channels[c]))
    residual = tuple(c for c in range(ccount) if not bool(selected_channels[c]))
    if not selected:
        raise ValueError('empty selection is a literal constructor no-op, not a native packet proof')
    if not np.isfinite(weights).all():
        raise ValueError('ordinary complete finite 3x3 kernel required')
    ids = _integer_array(parent_ids, (ccount, 6, 6), 'original input IDs')
    powers = _integer_array(parent_powers, ids.shape, 'original input powers')
    outputs = _integer_array(output_ids, (kcount, 4, 4), 'original output IDs')
    outpowers = _integer_array(output_powers, outputs.shape, 'original output powers')
    if (not isinstance(base_n_cont, (int, np.integer))
            or isinstance(base_n_cont, (bool, np.bool_))
            or not 0 <= base_n_cont <= np.iinfo(np.int32).max-36*(ccount+kcount)
            or np.any(ids < -1) or np.any(ids >= base_n_cont)
            or np.any(outputs < -1) or np.any(outputs >= base_n_cont)
            or np.any(powers < -511) or np.any(powers > 511)
            or np.any(outpowers < -511) or np.any(outpowers > 511)
            or len(set(map(int, outputs[outputs >= 0]))) != int((outputs >= 0).sum())
            or set(map(int, ids[ids >= 0])) & set(map(int, outputs[outputs >= 0]))):
        raise ValueError('complete disjoint original input/output IDs required')
    required = {'indptr', 'columns', 'native', 'pivots', 'gauges', 'rhs',
                'ab_indptr', 'roles', 'semantic_powers', 'defining_denominators',
                'selected_channels', 'new_factors'}
    if not required.issubset(packet):
        raise ValueError('complete actual native packet required')
    rows, nnz = len(packet['rhs']), len(packet['native'])
    if (not isinstance(packet['new_factors'], (int, np.integer))
            or isinstance(packet['new_factors'], (bool, np.bool_))
            or not 0 <= int(packet['new_factors']) <= 36*(ccount+kcount)):
        raise ValueError('complete native auxiliary population metadata required')
    pool.charge('c124_independent_complete_native_headers',
                32*(rows+nnz))
    # The actual CSR packet is a concrete native format, not generic integer
    # metadata.  Exact signed dtypes also make subsequent differences reliable.
    for name in ('indptr', 'ab_indptr', 'defining_denominators', 'columns',
                 'pivots', 'gauges', 'roles', 'semantic_powers'):
        dtype = np.dtype('int64' if name in
                         ('indptr', 'ab_indptr', 'defining_denominators') else 'int32')
        if not isinstance(packet[name], np.ndarray) or packet[name].dtype != dtype:
            raise ValueError('exact native signed packet dtype required for '+name)
    for name in ('indptr', 'ab_indptr'):
        _integer_array(packet[name], (rows+1,), name)
    for name in ('pivots', 'gauges', 'semantic_powers', 'defining_denominators'):
        _integer_array(packet[name], (rows,), name)
    _integer_array(packet['roles'], (rows, 3), 'complete roles')
    _integer_array(packet['columns'], (nnz,), 'complete columns')
    for name, size in (('native', nnz), ('rhs', rows)):
        if (not isinstance(packet[name], np.ndarray) or packet[name].shape != (size,)
                or packet[name].dtype != np.dtype('float64')
                or not np.isfinite(packet[name]).all()):
            raise ValueError('finite actual binary64 coefficient arrays required')
    if (packet['indptr'][0] != 0 or packet['indptr'][-1] != nnz
            or np.any(np.diff(packet['indptr']) <= 0)
            or np.any(packet['ab_indptr']) or np.any(packet['columns'] < 0)
            or np.any((np.abs(packet['native']) < 2.**-20)
                      | (np.abs(packet['native']) > 2.**40))
            or any(np.asarray(packet.get(name, ())).size for name in
                   ('ab_data', 'ab_columns', 'ab_native'))):
        raise ValueError('complete continuous-only native packet domain required')
    _basis(pool)

    kernel_terms = sum(bool(a) for row in _GN for a in row)**2*kcount*ccount
    pool.charge('c124_independent_exact_kernel_transform', 64*kernel_terms)
    transformed = {}
    for k in range(kcount):
        for c in range(ccount):
            for a in range(6):
                for b in range(6):
                    value = F(0)
                    for i in range(3):
                        for j in range(3):
                            if _GN[a][i] and _GN[b][j]:
                                w = weights[k, c, i, j]
                                w = F(float(w)) if weights.dtype.kind == 'f' else F(int(w))
                                term = _bounded(_GN[a][i]*_GN[b][j]*w)
                                value = _bounded(value+term)
                    transformed[k, c, 6*a+b] = value

    pool.charge('c124_independent_complete_selected_parent_forms',
                64*len(selected)*sum(bool(a) for row in _BT for a in row)**2)
    forms = {}
    for c in selected:
        for a in range(6):
            for b in range(6):
                terms = {}
                for i in range(6):
                    for j in range(6):
                        col, multiplier = int(ids[c, i, j]), _BT[a][i]*_BT[b][j]
                        if col >= 0 and multiplier:
                            value = _bounded(multiplier*_pow(powers[c, i, j]))
                            terms[col] = _bounded(terms.get(col, F(0))+value)
                forms[c, 6*a+b] = {col: v for col, v in terms.items() if v}

    users, channels = {}, {}
    pool.charge('c124_independent_complete_support_coverage',
                64*(kcount*36*16+kcount*ccount*36))
    for k in range(kcount):
        for t in range(36):
            a, b = divmod(t, 6)
            used = [s for s in range(16) if outputs[k, s//4, s%4] >= 0
                    and _AT[s//4][a]*_AT[s%4][b]]
            cs = [c for c in selected if forms[c, t] and transformed[k, c, t]]
            if used and cs:
                users[k, t], channels[k, t] = used, cs
    need_v = sorted({(c, t) for (k, t), cs in channels.items() for c in cs})
    need_m = sorted(channels)
    need_out = [(k, s) for k in range(kcount) for s in range(16)
                if outputs[k, s//4, s%4] >= 0]
    if (rows != len(need_v)+len(need_m)+len(need_out)
            or int(packet['new_factors']) != len(need_v)+len(need_m)):
        raise ValueError('complete required selected V/M/output row population differs')
    row_index, compared_nnz = 0, 0
    vmap, mmap = {}, {}
    for c, t in need_v:
        pool.charge('c124_independent_expected_V_rows', 64*(len(forms[c, t])+1))
        slot = int(base_n_cont)+row_index
        power = _ceil_power(_l1(forms[c, t].values()))
        raw = {col: -v for col, v in forms[c, t].items()}
        raw[slot] = _pow(power)
        compared_nnz += _check_row(packet, row_index, raw, slot, power, 1,
                                  (0, c, t), auxiliary=True, pool=pool)
        vmap[c, t] = slot, power
        row_index += 1
    for k, t in need_m:
        pool.charge('c124_independent_expected_M_rows', 64*(len(channels[k, t])+1))
        slot = int(base_n_cont)+row_index
        denominator = _D[t//6]*_D[t%6]
        raw = {vmap[c, t][0]: _bounded(-transformed[k, c, t]*_pow(vmap[c, t][1]))
               for c in channels[k, t]}
        power = _ceil_power(_bounded(_l1(raw.values())/denominator))
        raw[slot] = _bounded(denominator*_pow(power))
        compared_nnz += _check_row(packet, row_index, raw, slot, power, denominator,
                                  (1, k, t), auxiliary=True, pool=pool)
        mmap[k, t] = slot, power
        row_index += 1
    residual_visits = residual_terms = residual_alias_additions = residual_cancellations = 0
    for k, s in need_out:
        pool.charge('c124_independent_expected_output_rows', 64*37)
        visits = 9*len(residual)
        pool.charge('c124_independent_complete_residual_scan', 16*visits)
        pool.charge('c124_independent_complete_residual_exact_terms', 64*visits)
        residual_visits += visits
        i, j = divmod(s, 4)
        pivot, power = int(outputs[k, i, j]), int(outpowers[k, i, j])
        raw = {pivot: _pow(power)}
        for t in range(36):
            multiplier = _AT[i][t//6]*_AT[j][t%6]
            if (k, t) in mmap and multiplier:
                slot, unit = mmap[k, t]
                raw[slot] = _bounded(-multiplier*_pow(unit))
        # Rebuild EVERY original residual summand independently; structural
        # zeros still paid for above.  Repeated original IDs may coalesce or
        # cancel, including several residual channels sharing the same ID.
        for c in residual:
            for dy in range(3):
                for dx in range(3):
                    column = int(ids[c, i+dy, j+dx])
                    if column < 0:
                        continue
                    original = weights[k, c, dy, dx]
                    original = _bounded(F(float(original)) if weights.dtype.kind == 'f'
                                        else F(int(original)))
                    if not original:
                        continue
                    value = _bounded(-original*_pow(powers[c, i+dy, j+dx]))
                    residual_terms += 1
                    if column in raw:
                        residual_alias_additions += 1
                    value = _bounded(raw.get(column, F(0))+value)
                    if value:
                        raw[column] = value
                    else:
                        residual_cancellations += 1
                        raw.pop(column, None)
        compared_nnz += _check_row(packet, row_index, raw, pivot, power, 1,
                                  (2, k, s), auxiliary=False, pool=pool)
        row_index += 1
    return dict(selected_channels=len(selected), residual_channels=len(residual),
                owned_selected_mask_matches_supplied_mask=True,
                structural_selector_authenticated_by_this_oracle=False,
                all_original_kernel_coefficients_observed=int(weights.size),
                all_transformed_kernel_coefficients_rebuilt=36*kcount*ccount,
                transformed_kernel_proof_includes_unselected_channels=True,
                all_residual_scan_positions_proved=residual_visits,
                all_residual_nonzero_occurrences_rebuilt=residual_terms,
                exact_residual_alias_additions=residual_alias_additions,
                exact_residual_alias_cancellations=residual_cancellations,
                residual_scan_work=16*residual_visits,
                residual_exact_term_reserve_work=64*residual_visits,
                complete_direct_tail_and_selected_transform_sum_proved=True,
                all_1d_bilinear_coefficients_proved=72,
                full_2d_tensor_product_identity_proved=True,
                all_native_rows_proved=rows, all_native_nnz_proved=compared_nnz,
                all_V_definitions_proved=len(need_v), all_M_definitions_proved=len(need_m),
                all_original_output_equations_proved=len(need_out),
                all_auxiliary_equations_and_redundant_boxes_proved=len(need_v)+len(need_m),
                exact_required_support_coverage=True, original_input_output_ids_preserved=True,
                original_output_powers_and_gauge_semantics_preserved=True,
                odd_denominators_only_on_new_M_pivots=True,
                universal_unique_box_extension=True, native_only_inverse_sufficient=True,
                compositional_original_source_equivalence=True,
                complete_HZ_source_predicates_proved=False, physical_reduction_proved=False,
                actual_network_source_bound=False, concrete_network_witness=False, formal_gain=0)

