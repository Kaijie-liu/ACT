"""Read-only continuous-definition census; no candidate transformation."""

import numpy as np
import scipy.sparse as sp


def odd_significand(values):
    """Exact unsigned odd mantissa for finite nonzero float64 values."""
    mantissa, exponent = np.frexp(np.abs(np.asarray(values, dtype=np.float64)))
    if np.any(mantissa == 0.) or not np.isfinite(mantissa).all():
        raise ValueError('odd significand requires finite nonzero data')
    integer = np.ldexp(mantissa, 53).astype(np.uint64)
    low = integer & (~integer + np.uint64(1))
    trailing = np.frexp(low.astype(np.float64))[1] - 1
    odd = integer >> trailing.astype(np.uint64)
    return odd, np.frexp(odd.astype(np.float64))[1]


def exact_products(left, right):
    """Whether each nonzero float64 product is exact AND finite, incl subnormals."""
    left, right = np.broadcast_arrays(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))
    if not np.isfinite(left).all() or not np.isfinite(right).all() or np.any(left == 0.) or np.any(right == 0.):
        raise ValueError('exact-product census requires finite nonzero operands')
    a, abits = odd_significand(left)
    b, bbits = odd_significand(right)
    bits = abits + bbits
    precise = np.asarray(bits <= 53).copy()
    boundary = bits == 54
    # Products for this mask fit54 bits, so uint64 cannot overflow.
    precise[boundary] = a[boundary] * b[boundary] < np.uint64(1 << 53)
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        product = left * right
        # A representable normal product has at most53 odd significant bits.
        # Subnormal-range products additionally need the ORIGINAL exact dyadic
        # exponent to lie on the float64 grid, checked via operand decomposition.
        _, le = np.frexp(np.abs(left))
        _, re = np.frexp(np.abs(right))
        lm = np.ldexp(np.frexp(np.abs(left))[0], 53).astype(np.uint64)
        rm = np.ldexp(np.frexp(np.abs(right))[0], 53).astype(np.uint64)
        lt = np.frexp((lm & (~lm + np.uint64(1))).astype(np.float64))[1] - 1
        rt = np.frexp((rm & (~rm + np.uint64(1))).astype(np.float64))[1] - 1
        lowest_power = le.astype(np.int64) + re.astype(np.int64) - 106 + lt + rt
    return precise & np.isfinite(product) & (product != 0.) & (lowest_power >= -1074), product


def histogram(values):
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(v)): int(n) for v, n in zip(unique, counts)}


def census(hz, *, old_n_cont, logical_n_cont, old_n_eq, eq_roots, def_rows,
           max_work=256_000_000, max_entries=64_000_000, observe=None):
    if type(max_work) is not int or not 0 <= max_work <= 256_000_000:
        raise ValueError('invalid/increased census work ceiling')
    if type(max_entries) is not int or not 0 <= max_entries <= 64_000_000:
        raise ValueError('invalid/increased census entry ceiling')
    for v in (old_n_cont, logical_n_cont, old_n_eq):
        if type(v) is not int or v < 0:
            raise ValueError('invalid defining-prefix dimensions')
    if not old_n_cont <= logical_n_cont <= hz.n_cont or not hz.exact or hz.frame_id is None:
        raise ValueError('invalid exact main-variable frame')
    matrices = [getattr(hz, key) for key in ('Gc', 'Gb', 'Ac', 'Ab', 'Auc', 'Aub')]
    if any(not sp.isspmatrix_csr(m) or not m.has_canonical_format or not np.isfinite(m.data).all() for m in matrices):
        raise ValueError('census requires finite canonical CSR')
    entries = sum(m.nnz for m in matrices)
    if entries > max_entries or any(np.any(m.data == 0.) for m in matrices):
        raise MemoryError('stored input exceeds census ceiling or includes explicit zero payload')
    nc, main = hz.n_cont, logical_n_cont - old_n_cont
    roots = np.asarray(eq_roots)
    if roots.dtype != np.dtype(np.int64) or roots.shape != (old_n_eq + main,):
        raise ValueError('invalid complete logical equality map')
    if np.any(roots < 0) or np.any(roots >= hz.n_eq) or np.unique(roots).size != roots.size:
        raise ValueError('duplicate/missing/outside logical equality row')
    radix = np.asarray(def_rows)
    if radix.dtype != np.dtype(np.int64) or logical_n_cont + radix.size > nc:
        raise ValueError('invalid radix frame')
    if not np.array_equal(np.sort(np.concatenate((roots, radix))), np.arange(old_n_eq + main + radix.size)):
        raise ValueError('missing/duplicated/nonprefix C9 defining-row partition')
    # Eight full payload passes cover validation/liveness/degree/row maps;
    # classifying one main row is charged32, product incidence is charged64.
    work = 8 * entries + 32 * main
    if work > max_work:
        raise MemoryError('census structural preflight exceeds work ceiling')
    value_degree = np.bincount(hz.Gc.indices, minlength=nc)
    eq_degree = np.bincount(hz.Ac.indices, minlength=nc)
    ineq_degree = np.bincount(hz.Auc.indices, minlength=nc)
    columns = np.arange(old_n_cont, logical_n_cont, dtype=np.int64)
    row_ids = roots[old_n_eq:].copy()
    start, stop = hz.Ac.indptr[row_ids], hz.Ac.indptr[row_ids + 1]
    safe_last = np.maximum(stop - 1, 0)
    direct = (stop > start) & (hz.Ac.indices[safe_last] == columns)
    row_width = np.diff(hz.Ac.indptr)[row_ids] + np.diff(hz.Ab.indptr)[row_ids]
    degree = eq_degree[columns] + ineq_degree[columns]
    dead = value_degree[columns] == 0
    upper_delta = -row_width.astype(np.int64) + (degree - 1) * (row_width.astype(np.int64) - 2)
    homogeneous = hz.b[row_ids] == 0.
    alias = direct & dead & homogeneous & (stop - start == 2) & (np.diff(hz.Ab.indptr)[row_ids] == 0)
    ratio, parent = np.zeros(main), np.full(main, -1, dtype=np.int64)
    index = np.flatnonzero(alias)
    pivots = hz.Ac.data[stop[index] - 1]
    pm, pe = np.frexp(np.abs(pivots))
    if np.any(pm != .5) or np.any(pivots <= 0.):
        raise ValueError('registered direct MAIN pivot is not a positive power of two')
    parent[index] = hz.Ac.indices[start[index]]
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        ratio[index] = np.ldexp(-hz.Ac.data[start[index]], 1 - pe)
        restored = np.ldexp(ratio[index], pe - 1)
    if not np.array_equal(restored, -hz.Ac.data[start[index]]) or np.any(parent[index] >= columns[index]):
        raise ValueError('alias ratio is not reversible/topologically original')
    local_box = alias & (np.abs(ratio) <= 1.)
    power_two = alias & (np.frexp(np.abs(ratio))[0] == .5)
    eligible = np.zeros(nc, dtype=bool)
    eligible[columns] = local_box
    coefficient_work = int(np.sum((degree - 1)[local_box]))
    work += 64 * coefficient_work
    if observe is not None:
        observe({'event': 'structural_preflight', 'main_factors': main,
            'main_value_dead': int(dead.sum()), 'direct_definitions': int(direct.sum()),
            'dead_single_consumer': int(np.count_nonzero(dead & direct & (degree == 2))),
            'homogeneous_aliases': int(alias.sum()), 'locally_redundant_aliases': int(local_box.sum()),
            'power_two_aliases': int(np.count_nonzero(local_box & power_two)),
            'alias_incident_products': coefficient_work, 'whole_work_upper': work})
    if work > max_work:
        raise MemoryError('all alias coefficient products exceed frozen census work ceiling')
    factor_ratio, defining_position = np.ones(nc), np.full(nc, -1, dtype=np.int64)
    factor_ratio[columns] = ratio
    defining_position[columns[direct]] = stop[direct] - 1
    bad_exact, bad_window = np.zeros(nc, bool), np.zeros(nc, bool)
    observed = 0
    for matrix, equality in ((hz.Ac, True), (hz.Auc, False)):
        for begin in range(0, matrix.nnz, 65536):
            end = min(begin + 65536, matrix.nnz)
            positions = np.arange(begin, end, dtype=np.int64)
            cols = matrix.indices[begin:end]
            selected = eligible[cols]
            if equality:
                selected &= positions != defining_position[cols]
            cols, vals = cols[selected], matrix.data[begin:end][selected]
            if not cols.size:
                continue
            exact, product = exact_products(vals, factor_ratio[cols])
            np.logical_or.at(bad_exact, cols, ~exact)
            np.logical_or.at(bad_window, cols, (np.abs(product) < 2.**-20) | (np.abs(product) > 2.**40) | ~np.isfinite(product))
            observed += cols.size
    if observed != coefficient_work:
        raise ValueError('alias occurrence accounting mismatch')
    all_exact, window_safe = local_box & ~bad_exact[columns], local_box & ~bad_window[columns]
    report = {'status': 'READ_ONLY_STRUCTURAL_CENSUS', 'formal_gain': 0,
        'n_cont': nc, 'n_bin': hz.n_bin, 'n_eq': hz.n_eq, 'n_ineq': hz.n_ineq,
        'total_coefficient_nnz': entries, 'main_factors': main,
        'radix_factors': int(radix.size), 'protected_original_continuous': old_n_cont,
        'post_main_radix_continuous': nc - logical_n_cont - int(radix.size),
        'value_live_continuous': int(np.count_nonzero(value_degree)),
        'main_value_dead': int(dead.sum()), 'main_direct_definitions': int(direct.sum()),
        'main_non_direct_packed_definitions': int((~direct).sum()),
        'main_single_consumer': int(np.count_nonzero(dead & direct & (degree == 2))),
        'main_negative_individual_nnz_upper': int(np.count_nonzero(dead & direct & (upper_delta < 0))),
        'homogeneous_aliases': int(alias.sum()), 'local_box_redundant_aliases': int(local_box.sum()),
        'power_two_aliases': int(np.count_nonzero(local_box & power_two)),
        'aliases_all_incident_products_exact': int(all_exact.sum()),
        'aliases_all_incident_products_window_safe': int(window_safe.sum()),
        'aliases_exact_and_window_safe': int(np.count_nonzero(all_exact & window_safe)),
        'alias_incident_products_checked': observed, 'logical_work_upper': work,
        'main_defining_row_width_histogram': histogram(row_width),
        'main_predicate_degree_histogram': histogram(degree),
        'alias_ratio_exponent_histogram': histogram(np.frexp(np.abs(ratio[local_box]))[1] - 1),
        'predicate_addition_collision_exactness_proved': False,
        'simultaneous_chain_elimination_proved': False,
        'transformation_constructed': False, 'native_ingestion_executed': False, 'solver_executed': False}
    table = {'column': columns, 'defining_row': row_ids, 'row_width': row_width,
        'predicate_degree': degree, 'value_dead': dead, 'direct_pivot': direct,
        'single_elimination_nnz_upper': upper_delta, 'alias': alias, 'parent': parent,
        'ratio': ratio, 'local_box_redundant': local_box, 'power_two': power_two,
        'all_products_exact': all_exact, 'products_window_safe': window_safe}
    return report, table
