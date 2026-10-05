"""Independent actual sparse-incidence oracle for generated MAIN ownership."""

import numpy as np


def row_uid_tables(candidate):
    hz = candidate.hz
    eq = np.full(hz.n_eq, -1, np.int64)
    le = np.full(hz.n_ineq, -1, np.int64)
    def assign(array, index, uid):
        if not 0 <= index < len(array) or array[index] != -1:
            raise ValueError('missing/duplicate physical row label')
        array[index] = uid
    for i in range(candidate.old_n_eq):
        assign(eq, int(candidate.eq_roots[i]), i)
    for i, r in enumerate(candidate.ineq_roots):
        assign(le, int(r), candidate.old_n_eq + i)
    cursor = candidate.old_n_eq + len(candidate.ineq_roots)
    main = 0
    for node in candidate.nodes:
        for coordinate in np.flatnonzero(node['needed']):
            col = int(node['slots'][coordinate])
            if col != candidate.old_n_cont + main:
                raise ValueError('original MAIN slot/UID order changed')
            r = int(candidate.eq_roots[candidate.old_n_eq + main])
            if r >= 0:
                assign(eq, r, cursor + int(coordinate))
            main += 1
        cursor += node['width']
    if main != candidate.logical_n_cont - candidate.old_n_cont or cursor != candidate.report['radix_uid_base']:
        raise ValueError('incomplete MAIN or radix UID layout')
    for i, r in enumerate(candidate.def_rows):
        assign(eq, int(r), cursor + i)
    joined = np.r_[eq, le]
    if np.any(joined < 0) or np.any(joined >= 2**20) or np.unique(joined).size != joined.size:
        raise ValueError('incomplete/nonunique/out-of-domain canonical row UIDs')
    return eq, le


def actual_words(hz, old_nc, logical_nc, eq_uids, ineq_uids):
    size = logical_nc - old_nc
    count, sums = np.zeros(size, np.int64), np.zeros(size, np.int64)
    for matrix, uids in ((hz.Ac, eq_uids), (hz.Auc, ineq_uids)):
        if len(uids) != matrix.shape[0] or not matrix.has_canonical_format or np.any(matrix.data == 0.):
            raise ValueError('oracle requires complete canonical nonzero physical incidence')
        for row, uid in enumerate(uids):
            start, end = matrix.indptr[row:row + 2]
            cols = matrix.indices[start:end]
            cols = cols[(cols >= old_nc) & (cols < logical_nc)] - old_nc
            count[cols] += 1
            sums[cols] += int(uid)
    if np.any(count > 2**20) or np.any(sums >= 2**40):
        raise ValueError('independent canonical incidence bound exceeded')
    return count * (2**40) + sums


def audit(candidate):
    candidate.validate()
    eq, le = row_uid_tables(candidate)
    expected = actual_words(candidate.hz, candidate.old_n_cont, candidate.logical_n_cont, eq, le)
    if not np.array_equal(expected, candidate.owners):
        wrong = np.flatnonzero(expected != candidate.owners)
        raise ValueError(f'complete generated ownership differs from actual sparse incidence: {len(wrong)} columns; first {int(wrong[0])}')
    candidate.validate()
    return {'all_MAIN_columns_checked': len(expected), 'all_EQ_rows_checked': len(eq),
        'all_INEQ_rows_checked': len(le), 'integer_count_and_UID_sum_equal': True,
        'zero_incidence_MAIN': int(np.count_nonzero(expected == 0)),
        'two_incidence_MAIN': int(np.count_nonzero(expected // (2**40) == 2)),
        'formal_gain': 0, 'live_path_proved': False}


def phase_event_audit(candidate, post, *, exact_proof):
    """Diagnostic append-event replay; no new native/NN/solver execution.

    Requires the complete independent original-DAG/quotient proof. The complete
    authenticated external post-ReLU prefix is checked before replay. The
    parent-coefficient norm premise transfers by C16; this is not a fresh live
    HZ publication or a call to an optimizer.
    """
    from fractions import Fraction as F
    import math
    from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import equal_payload
    from experiments.neural_hz_20260831.c17_packed_ownership_v1 import Ledger, append_owned_rows, unique_other
    from experiments.neural_hz_20260831.c10_fused_rows_v1 import WorkPool
    candidate.validate()
    original = exact_proof.get('original_affine_proof', {})
    if (exact_proof.get('status') != 'EXACT_ORIGINAL_DAG_AND_QUOTIENT'
            or exact_proof.get('tagged_physical_lineage_checked') is not True
            or original.get('all_redundant_main_and_radix_boxes_proved') is not True
            or original.get('all_main_defining_rows_checked') != len(candidate.owners)):
        raise ValueError('phase ownership lacks full independent generation/box proof')
    pre = candidate.hz
    if post.frame_id != pre.frame_id or post.n_cont < pre.n_cont or post.n_bin < pre.n_bin or not post.exact:
        raise ValueError('phase oracle changed shared frame or exactness')
    for name in ('Ac', 'Ab', 'Auc', 'Aub'):
        old, new = getattr(pre, name), getattr(post, name)
        if (new.shape[0] < old.shape[0]
                or not equal_payload(new[:old.shape[0], :old.shape[1]], old)
                or new[:old.shape[0], old.shape[1]:].nnz):
            raise ValueError('phase oracle did not append to the exact generated predicate prefix')
    if not equal_payload(post.b[:pre.n_eq], pre.b) or not equal_payload(post.ub[:pre.n_ineq], pre.ub):
        raise ValueError('phase oracle changed original RHS')
    ne, nl = post.n_eq - pre.n_eq, post.n_ineq - pre.n_ineq
    pool = WorkPool(0, 0)
    pool.charge('diagnostic_MAIN_copy', len(candidate.owners))
    ledger = Ledger(candidate.owners.copy(), candidate.old_n_cont, pool=pool)
    first = candidate.report['radix_uid_base'] + 16_384
    append_owned_rows(ledger, post.Ac[pre.n_eq:], first)
    append_owned_rows(ledger, post.Auc[pre.n_ineq:], first + ne)
    ledger.finish()
    eq, le = row_uid_tables(candidate)
    eq = np.r_[eq, np.arange(first, first + ne)]
    le = np.r_[le, np.arange(first + ne, first + ne + nl)]
    actual = actual_words(post, candidate.old_n_cont, candidate.logical_n_cont, eq, le)
    if not np.array_equal(actual, ledger.words):
        raise ValueError('phase ownership differs from complete actual post incidence')
    # Use the independently assembled physical UID map only as a diagnostic
    # query oracle. It is not an uncharged map retained by the generator.
    lookup = {int(uid): (False, r) for r, uid in enumerate(eq)}
    lookup.update({int(uid): (True, r) for r, uid in enumerate(le)})
    output_live = np.bincount(post.Gc.indices, minlength=post.n_cont) != 0
    accepted, candidates = [], 0
    for i, packed in enumerate(ledger.words):
        col, d = candidate.old_n_cont + i, int(candidate.eq_roots[candidate.old_n_eq + i])
        if d < 0 or output_live[col] or int(packed) // 2**40 != 2:
            continue
        a, b = post.Ac.indptr[d:d + 2]
        if a == b or post.Ac.indices[b - 1] != col:
            continue
        pivot = float(post.Ac.data[b - 1])
        if pivot <= 0. or math.frexp(pivot)[0] != .5:
            continue
        candidates += 1
        other = unique_other(packed, int(eq[d]))
        if other not in lookup:
            raise ValueError('packed unique consumer does not name a surviving physical row')
        inequality, r = lookup[other]
        cm, rhs = (post.Auc, post.ub) if inequality else (post.Ac, post.b)
        start, stop = cm.indptr[r:r + 2]
        cc, cv = cm.indices[start:stop], cm.data[start:stop]
        pos = int(np.searchsorted(cc, col))
        if pos == len(cc) or cc[pos] != col:
            raise ValueError('unique consumer UID lacks the counted coefficient')
        if abs(float(cv[pos])) != pivot:
            continue
        if post.Ab.indptr[d + 1] != post.Ab.indptr[d] or pos != 0:
            raise ValueError('unit pair lacks C15 ordered binary-free reconstruction shape')
        sign = -1 if cv[pos] > 0. else 1
        updated = float(rhs[r]) + sign * float(post.b[d])
        if F(updated) != F(float(rhs[r])) + sign * F(float(post.b[d])):
            raise ValueError('unit RHS update is not exact')
        accepted.append(col)
    candidate.validate()
    return {'all_post_MAIN_columns_checked': len(actual), 'new_EQ_rows': ne, 'new_INEQ_rows': nl,
        'complete_post_incidence_equal': True, 'structural_single_consumer_definitions': candidates,
        'unit_pairs_identified_from_owned_metadata': len(accepted),
        'append_event_work': pool.used, 'append_event_work_parts': dict(pool.parts),
        'diagnostic_UID_lookup_entries': len(lookup), 'additional_owner_vector_bytes': ledger.words.nbytes,
        'new_native_relu_executed': False, 'whole_live_path_proved': False, 'formal_gain': 0}, np.array(accepted, np.int64), ledger.words
