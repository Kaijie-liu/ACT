"""Complete independent original-row audit after eliminating radix factors."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c5_zero_suffix_audit_v1 import zero_transfer_reference
from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import equal_payload
from experiments.neural_hz_20260831.c9_radix_predicate_audit_v1 import row, dyadic_power, unscale, recover_row


def proxy(candidate):
    return SimpleNamespace(original=SimpleNamespace(n_cont=candidate.logical_n_cont, n_bin=candidate.old_n_bin),
        hz=candidate.hz, eq_roots=candidate.eq_roots, eq_scales=candidate.eq_scales,
        ineq_roots=candidate.ineq_roots, ineq_scales=candidate.ineq_scales, def_rows=candidate.def_rows)


def audit(candidate):
    candidate.validate()
    expr, hz, nodes = candidate.expression, candidate.hz, candidate.nodes
    base = zero_transfer_reference(expr)
    main_aux = candidate.logical_n_cont - candidate.old_n_cont
    if (hz.n_cont != candidate.logical_n_cont + candidate.def_rows.size or hz.n_bin != candidate.old_n_bin
            or hz.frame_id != expr.frame_id or not hz.exact or hz.n_eq != base.n_eq + main_aux + candidate.def_rows.size
            or hz.n_ineq != base.n_ineq or candidate.old_n_eq != base.n_eq
            or candidate.old_n_cont < base.n_cont or candidate.old_n_bin < base.n_bin):
        raise ValueError('logical/radix/original frame or dimensions changed')
    for mapping, count in ((candidate.eq_roots, base.n_eq + main_aux), (candidate.eq_scales, base.n_eq + main_aux),
            (candidate.ineq_roots, base.n_ineq), (candidate.ineq_scales, base.n_ineq)):
        if mapping.dtype != np.dtype(np.int64) or mapping.shape != (count,):
            raise ValueError('invalid complete logical row map')
    if (not np.array_equal(np.sort(np.concatenate((candidate.eq_roots, candidate.def_rows))), np.arange(hz.n_eq))
            or not np.array_equal(np.sort(candidate.ineq_roots), np.arange(hz.n_ineq))):
        raise ValueError('missing/duplicated/hidden encoded predicate')
    for matrix in (hz.Ac, hz.Ab, hz.Auc, hz.Aub):
        if matrix.data.size and (np.any(np.abs(matrix.data) < 2.**-20) or np.any(np.abs(matrix.data) > 2.**40)):
            raise ValueError('coefficient window violated')
    for index, definition in enumerate(candidate.def_rows):
        cc, cv = row(hz.Ac, definition)
        bc, bv = row(hz.Ab, definition)
        if not cc.size or cc[-1] != candidate.logical_n_cont + index or np.any(cc[:-1] >= cc[-1]) or hz.b[definition] != 0.:
            raise ValueError('radix definition is not homogeneous/triangular')
        pivot = dyadic_power(cv[-1])
        for coefficient in cv[:-1][cc[:-1] >= candidate.logical_n_cont]:
            dyadic_power(-coefficient)
        values = np.abs(np.concatenate((cv[:-1], bv)))
        if not values.size:
            raise ValueError('empty radix definition')
        exponents = np.frexp(values)[1]
        minimum = int(exponents.min())
        envelope = sum(1 << (int(e) - minimum) for e in exponents)
        if minimum + (envelope - 1).bit_length() > pivot:
            raise ValueError('radix box not proved redundant')
    expanded = []
    for node in nodes:
        if node['kind'] == 'source':
            paths = [(id(node['source']), ())]
        elif node['kind'] == 'op':
            paths = [(s, (*ops, id(node['op']))) for s, ops in expanded[node['parents'][0]]]
        else:
            paths = [p for parent in node['parents'] for p in expanded[parent]]
        if len(paths) > len(expr.terms):
            raise ValueError('extra source/operator term')
        expanded.append(paths)
    expected_terms = [(id(t.source), tuple(id(op) for op in t.operators)) for t in expr.terms]
    if Counter(expanded[candidate.root]) != Counter(expected_terms):
        raise ValueError('source/operator multiset changed')
    holder, used = proxy(candidate), set()
    recovered_count = 0
    def check(index, cc, cv, bc, bv, rhs, *, inequality=False):
        nonlocal recovered_count
        actual_c, actual_b, actual_rhs, reached = recover_row(holder, index, inequality=inequality)
        if used & reached:
            raise ValueError('unregistered cross-row radix sharing')
        used.update(reached)
        for actual, cols, vals in ((actual_c, cc, cv), (actual_b, bc, bv)):
            expected = {int(c): float(v) for c, v in zip(cols, vals) if v != 0.}
            if actual != expected:
                raise ValueError('eliminated row differs from ORIGINAL unfused coefficients')
            recovered_count += len(expected)
        if actual_rhs != rhs:
            raise ValueError('eliminated row differs from ORIGINAL RHS')
    for inequality, cmat, bmat, rhs in ((False, base.Ac, base.Ab, base.b), (True, base.Auc, base.Aub, base.ub)):
        for index in range(cmat.shape[0]):
            cc, cv = row(cmat, index)
            bc, bv = row(bmat, index)
            check(index, cc, cv, bc, bv, float(rhs[index]), inequality=inequality)
    main_count = streamed = 0
    for node in nodes:
        for coordinate in np.flatnonzero(node['needed']):
            slot = int(node['slots'][coordinate])
            if slot != candidate.old_n_cont + main_count:
                raise ValueError('main factors do not use the registered shared prefix')
            bc, bv = np.empty(0, dtype=np.int64), np.empty(0)
            constant = 0.
            if node['kind'] == 'source':
                source = node['source']
                cc, cv = row(source.Gc, coordinate)
                selected = cv != 0.
                cc, cv = cc[selected], cv[selected]
                bc, bv = row(source.Gb, coordinate)
                selected = bv != 0.
                bc, bv = bc[selected], bv[selected]
                constant = float(source.c[coordinate])
                powers = np.zeros(cv.size, dtype=np.int64)
                bound_values = np.concatenate((cv, bv, [constant] if constant else []))
                bound_exponents = np.frexp(np.abs(bound_values))[1]
            elif node['kind'] == 'op':
                parent = nodes[node['parents'][0]]
                op = node['op']
                coords, cv = op._row(int(coordinate)) if type(op) is ImplicitConv2DOp else row(op, coordinate)
                streamed += int(cv.size)
                if streamed > 256_000_000:
                    raise MemoryError('original streaming row audit cap exceeded')
                selected = parent['needed'][coords] & (cv != 0.)
                coords, cv = coords[selected], cv[selected]
                cc = parent['slots'][coords]
                powers = parent['exponents'][coords].astype(np.int64)
                bound_values = cv
                bound_exponents = np.frexp(np.abs(cv))[1] + powers
            else:
                items = [(nodes[p]['slots'][coordinate], nodes[p]['exponents'][coordinate], count)
                    for p, count in sorted(Counter(node['parents']).items()) if nodes[p]['support'][coordinate]]
                cc = np.array([t[0] for t in items], dtype=np.int64)
                cv = np.array([t[2] for t in items], dtype=np.float64)
                powers = np.array([t[1] for t in items], dtype=np.int64)
                bound_values = cv
                bound_exponents = np.frexp(cv)[1] + powers
            unit = int(node['exponents'][coordinate])
            if unit < 0 or unit > 1023 or not bound_values.size or np.any(cc >= slot) or np.any(cc < 0):
                raise ValueError('invalid triangular MAIN definition')
            minimum = int(bound_exponents.min())
            envelope = sum(1 << (int(e) - minimum) for e in bound_exponents)
            if minimum + (envelope - 1).bit_length() > unit:
                raise ValueError('MAIN coordinate box not proved redundant')
            with np.errstate(over='raise', invalid='raise', under='ignore'):
                coefficients = np.ldexp(-cv, powers)
                if not np.array_equal(np.ldexp(coefficients, -powers), -cv):
                    raise ValueError('ORIGINAL coefficient powers cannot be exactly represented')
            check(base.n_eq + main_count, np.append(cc, slot), np.append(coefficients, np.ldexp(1., unit)), bc, -bv, constant)
            main_count += 1
    if main_count != main_aux or used != set(range(candidate.def_rows.size)):
        raise ValueError('missing MAIN definition or unaccounted radix factor')
    root = nodes[candidate.root]
    selected = candidate.keep & root['support']
    rows = np.flatnonzero(selected)
    expected_gc = sp.csr_matrix((np.ldexp(np.ones(rows.size), root['exponents'][rows]),
        (rows, root['slots'][rows])), shape=hz.Gc.shape)
    if not equal_payload(expected_gc, hz.Gc) or hz.Gb.nnz or not equal_payload(expr.bias, hz.c):
        raise ValueError('original output map/bias changed')
    candidate.validate()
    return {'all_main_defining_rows_checked': main_count, 'all_original_equalities_checked': base.n_eq,
        'all_original_inequalities_checked': base.n_ineq, 'all_radix_defining_rows_checked': candidate.def_rows.size,
        'all_recovered_logical_nonzero_coefficients_checked': recovered_count,
        'original_streamed_operator_entries': streamed, 'all_original_coefficients_exact': True,
        'all_redundant_main_and_radix_boxes_proved': True, 'all_old_predicates_preserved': True,
        'exact_original_source_operator_multiset': True, 'unique_extension_and_original_prefix_projection': True,
        'rounded_composed_matrix_identity_claimed': False}
