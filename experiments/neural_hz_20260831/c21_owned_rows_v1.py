"""Exact C17 owned quotient with sparse changed-column two-stream normalization."""

import math
import numpy as np

from experiments.neural_hz_20260831.c10_fused_rows_v1 import aliases
from experiments.neural_hz_20260831.c21_exact_products_v1 import window_products
from experiments.neural_hz_20260831.c10_alias_quotient_v1 import exact_sum, frontier
from experiments.neural_hz_20260831.c17_owned_rows_v1 import WorkPool, OwnedRowEncoder
from experiments.neural_hz_20260831.c18_sparse_rewrite_v1 import sort_rewritten_row, sort_cost

def fold_rows(encoder, eq_roots, eq_scales, *, old_nc, old_eq, output_slots, observe=None):
    """Own/rewrite RowEncoder buffers; no completed original HZ is created."""
    pool, nc = encoder.pool, encoder.nc
    main = nc - old_nc
    roots, scales = np.asarray(eq_roots, dtype=np.int64), np.asarray(eq_scales, dtype=np.int64)
    if (main < 0 or roots.shape != (old_eq + main,) or scales.shape != roots.shape
            or not np.array_equal(np.sort(np.r_[roots, encoder.def_rows]), np.arange(len(encoder.eq)))):
        raise ValueError('invalid complete pre-emission MAIN/radix row partition')
    pool.charge('main_metadata_frontier', 32 * main)
    protected = np.zeros(nc, dtype=bool)
    protected[np.asarray(output_slots, dtype=np.int64)] = True
    parents = np.full(main, -1, dtype=np.int64)
    ratios = np.zeros(main)
    eligible = np.zeros(main, dtype=bool)
    defining = roots[old_eq:]
    for index, physical in enumerate(defining):
        cc, cv, bc, bv, rhs = encoder.eq[int(physical)]
        col = old_nc + index
        if protected[col] or len(cc) != 2 or len(bc) or rhs != 0. or cc[-1] != col:
            continue
        parent, pivot = int(cc[0]), float(cv[-1])
        pm, pe = math.frexp(pivot)
        if not 0 <= parent < col or pm != .5:
            raise ValueError('non-topological/nonpositive-dyadic MAIN pivot')
        ratio = math.ldexp(-float(cv[0]), 1 - pe)
        if math.ldexp(ratio, pe - 1) != -float(cv[0]):
            raise ValueError('nonreversible emitted alias ratio')
        if 2.**-60 <= abs(ratio) <= 1.:
            parents[index], ratios[index], eligible[index] = parent, ratio, True
    local = np.flatnonzero(eligible)
    if not local.size:
        raise ValueError('no locally eligible fused alias')
    lookup = np.full(nc + len(encoder.def_rows), -1, dtype=np.int64)
    lookup[old_nc + local] = local
    own_alias = np.zeros(len(encoder.eq), dtype=bool)
    own_alias[defining[local]] = True
    # Hit tuples are bounded temporary incidence data, discarded before return.
    hits = ([], [])
    examined = products_checked = general_products = fast_right_rows = product_rows = 0
    for kind, rows in enumerate((encoder.eq, encoder.ineq)):
        for row, (cc, cv, bc, bv, rhs) in enumerate(rows):
            pool.charge('continuous_incidence_scan', len(cc))
            examined += len(cc)
            columns = cc[:-1] if kind == 0 and own_alias[row] else cc
            ids = lookup[columns]
            positions = np.flatnonzero(ids >= 0)
            if not positions.size:
                continue
            ids = ids[positions]
            pool.charge('product_operand_gathers', 2 * positions.size)
            good, products, product_stats = window_products(cv[positions], ratios[ids], pool=pool)
            general_products += product_stats['general']
            fast_right_rows += int(product_stats['right_batch_fast'])
            product_rows += 1
            np.logical_and.at(eligible, ids, good)
            hits[kind].append((row, positions, ids, products))
            products_checked += positions.size
    columns = np.arange(old_nc, nc, dtype=np.int64)
    chosen = frontier({'column': columns, 'parent': parents,
        'all_products_exact': eligible, 'products_window_safe': eligible}, nc)
    if not chosen.size:
        raise ValueError('no exact independent fused alias frontier')
    selected = np.zeros(main, dtype=bool)
    selected[chosen] = True
    erased = np.zeros(len(encoder.eq), dtype=bool)
    erased[defining[chosen]] = True
    old_entries = encoder.entries
    old_nnz = sum(len(r[0]) + len(r[2]) for rows in (encoder.eq, encoder.ineq) for r in rows)
    rewritten = collisions = occurrences = sorted_rows = ordered_rows = 0
    normalization_counts = {}
    c17_sort_work = 0
    for index in chosen:
        defining_row = int(defining[index])
        # The selected alias itself is retired once after the complete rewrite.
        # Its nonselected parent loses the old defining-row incidence now.
        encoder.ledger.change(encoder.eq[defining_row][0][:-1], encoder.eq_uids[defining_row], -1)
    if observe:
        observe('fused_frontier', {'local_aliases': int(local.size), 'eligible_aliases': int(eligible.sum()),
            'selected_aliases': int(chosen.size), 'products_checked': int(products_checked),
            'coupled_extra_used': pool.used, 'coupled_extra_capacity': pool.capacity})
    for kind, rows in enumerate((encoder.eq, encoder.ineq)):
        for row, positions, ids, products in hits[kind]:
            active = selected[ids]
            if (kind == 0 and erased[row]) or not active.any():
                continue
            cc, cv, bc, bv, rhs = rows[row]
            width = len(cc)
            pool.charge('rewrite_order_check_emit', 4 * width)
            uid = (encoder.eq_uids if kind == 0 else encoder.ineq_uids)[row]
            encoder.ledger.change(parents[ids[active]], uid, 1)
            # These are owned unpublished buffers, never original source HZs.
            cc[positions[active]] = parents[ids[active]]
            cv[positions[active]] = products[active]
            if np.any(np.diff(cc) < 0):
                c17_sort_work += sort_cost(width)
                cc, cv, method = sort_rewritten_row(cc, cv, positions[active], pool)
                normalization_counts[method] = normalization_counts.get(method, 0) + 1
                sorted_rows += 1
            else:
                ordered_rows += 1
            starts = np.r_[0, np.flatnonzero(np.diff(cc)) + 1, width]
            for begin, end in zip(starts[:-1], starts[1:]):
                if end - begin > 1:
                    pool.charge('collision_exact_sum', 32 * (end - begin))
                    cv[begin] = exact_sum(cv[begin:end])
                    removed_count = end - begin - int(cv[begin] != 0.)
                    encoder.ledger.change(np.repeat(cc[begin], removed_count), uid, -1)
                    cv[begin + 1:end] = 0.
                    collisions += 1
            nonzero = cv != 0.
            rows[row] = (cc[nonzero], cv[nonzero], bc, bv, rhs)
            rewritten += 1
            occurrences += int(active.sum())
    row_map = np.cumsum(~erased, dtype=np.int64) - 1
    tagged_roots = row_map[roots].copy()
    tagged_scales = scales.copy()
    tagged_roots[old_eq + chosen] = -(parents[chosen] + 1)
    tagged_scales.view(np.float64)[old_eq + chosen] = ratios[chosen]
    encoder.def_rows = [int(row_map[r]) for r in encoder.def_rows]
    encoder.eq = [value for index, value in enumerate(encoder.eq) if not erased[index]]
    encoder.eq_uids = [value for index, value in enumerate(encoder.eq_uids) if not erased[index]]
    encoder.ledger.retire_verified_frontier(chosen)
    encoder.ledger.finish()
    new_nnz = sum(len(r[0]) + len(r[2]) for rows in (encoder.eq, encoder.ineq) for r in rows)
    encoder.entries = new_nnz + len(encoder.eq) + len(encoder.ineq)
    if new_nnz > old_nnz - 2 * chosen.size or encoder.entries >= old_entries:
        raise ValueError('fused emission does not strictly reduce all predicate entries')
    report = {'schema': 'tagged_alias_lineage_v1', 'local_aliases': int(local.size),
        'eligible_aliases': int(eligible.sum()), 'selected_aliases': int(chosen.size),
        'owned_continuous_coefficients_inspected': examined, 'alias_products_checked': int(products_checked),
        'rewritten_rows': rewritten, 'already_ordered_rewrites': ordered_rows, 'actual_sort_rewrites': sorted_rows, 'rewritten_occurrences': occurrences, 'collision_groups': collisions,
        'normalization_counts': normalization_counts,
        'c17_sort_work_on_same_complete_rows': c17_sort_work,
        'normalization_work': sum(pool.parts.get(n, 0) for n in
            ('rewrite_sort', 'rewrite_changed_order', 'rewrite_stream_merge')),
        'product_certification': {'hits': int(products_checked), 'both_general': general_products,
            'all_right_dyadic_rows': fast_right_rows, 'rows': product_rows,
            'work': sum(pool.parts.get(n, 0) for n in
                ('product_operand_gathers', 'product_common', 'product_left_classification', 'product_general_exact'))},
        'old_predicate_nnz': old_nnz, 'new_predicate_nnz': new_nnz,
        'old_predicate_entries': old_entries, 'new_predicate_entries': encoder.entries,
        'persistent_extra_reconstruction_arrays': 0, 'coupled_extra_work': pool.used,
        'coupled_extra_capacity': pool.capacity, 'work_parts': dict(pool.parts),
        'original_completed_hz_constructed': False, 'formal_gain': 0}
    return tagged_roots, tagged_scales, report
