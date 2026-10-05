"""Dense logical MAIN-rank slabs, composed with existing tagged physical maps."""

import numpy as np
from experiments.neural_hz_20260831.c22_uid_runs_v1 import pack, unpack, validate, row_for_uid, uid_for_row, LIMIT


def build_slabs(counts, first_uid, *, pool):
    if type(first_uid) is not int or not 0 <= first_uid <= LIMIT:
        raise ValueError('invalid node UID prefix')
    pool.charge('dense_uid_slab_metadata', 64 * len(counts))
    records = []
    uid, main = first_uid, 0
    for node in counts:
        width, count = node['width'], node['auxiliaries']
        if (type(width) is not int or type(count) is not int
                or not 0 <= count <= width or uid + width > LIMIT or main + count > LIMIT):
            raise ValueError('invalid full-node reservation/dense MAIN count')
        if count:
            if records:
                a, b, n = unpack(records[-1])
                if a + n == uid and b + n == main:
                    records[-1] = pack(a, b, n + count)
                else:
                    records.append(pack(uid, main, count))
            else:
                records.append(pack(uid, main, count))
        main += count
        uid += width
    words = np.asarray(records, np.uint64)
    validate(words)
    return words


def node_uid_tables(candidate):
    """Independent dense UID oracle from complete original needed/slot fields."""
    hz = candidate.hz
    eq = np.full(hz.n_eq, -1, np.int64)
    le = np.full(hz.n_ineq, -1, np.int64)
    expected_main = []
    def assign(target, row, uid):
        if not 0 <= row < len(target) or target[row] != -1:
            raise ValueError('duplicate/missing physical row in dense UID oracle')
        target[row] = uid
    for i in range(candidate.old_n_eq): assign(eq, int(candidate.eq_roots[i]), i)
    for i, row in enumerate(candidate.ineq_roots): assign(le, int(row), candidate.old_n_eq + i)
    uid = candidate.old_n_eq + len(candidate.ineq_roots)
    main = 0
    for node in candidate.nodes:
        for rank, coordinate in enumerate(np.flatnonzero(node['needed'])):
            if int(node['slots'][coordinate]) != candidate.old_n_cont + main:
                raise ValueError('global MAIN slots no longer follow dense needed order')
            expected_main.append(uid + rank)
            row = int(candidate.eq_roots[candidate.old_n_eq + main])
            if row >= 0: assign(eq, row, uid + rank)
            main += 1
        uid += node['width']
    if main != candidate.logical_n_cont - candidate.old_n_cont or uid != candidate.report['radix_uid_base']:
        raise ValueError('dense UID oracle is incomplete')
    for i, row in enumerate(candidate.def_rows): assign(eq, int(row), uid + i)
    all_uids = np.r_[eq, le]
    if np.any(all_uids < 0) or np.any(all_uids >= LIMIT) or np.unique(all_uids).size != len(all_uids):
        raise ValueError('dense physical UID partition is invalid')
    return eq, le, np.asarray(expected_main, np.int64)


def closed_uid_tables(candidate):
    """Materialize a DIAGNOSTIC table using checked slabs, with no node fields."""
    eq = np.full(candidate.hz.n_eq, -1, np.int64)
    le = np.full(candidate.hz.n_ineq, -1, np.int64)
    eq[candidate.eq_roots[:candidate.old_n_eq]] = np.arange(candidate.old_n_eq)
    le[candidate.ineq_roots] = candidate.old_n_eq + np.arange(len(le))
    for raw in candidate.uid_slabs:
        uid, first, length = unpack(raw)
        roots = candidate.eq_roots[candidate.old_n_eq + first:candidate.old_n_eq + first + length]
        active = roots >= 0
        eq[roots[active]] = uid + np.flatnonzero(active)
    eq[candidate.def_rows] = candidate.report['radix_uid_base'] + np.arange(len(candidate.def_rows))
    if np.any(eq < 0) or np.any(le < 0):
        raise ValueError('checked slabs fail to cover all physical predicates')
    return eq, le


def resolve(candidate, uid, *, pool):
    """Inside a checked transaction: return(inequality,physical row) or None."""
    if type(uid) is not int or not 0 <= uid < LIMIT:
        raise ValueError('invalid row UID')
    pool.charge('dense_uid_prefix_radix_query', 12)
    if uid < candidate.old_n_eq:
        return False, int(candidate.eq_roots[uid])
    if uid < candidate.old_n_eq + len(candidate.ineq_roots):
        return True, int(candidate.ineq_roots[uid - candidate.old_n_eq])
    radix = candidate.report['radix_uid_base']
    if radix <= uid < radix + len(candidate.def_rows):
        return False, int(candidate.def_rows[uid - radix])
    main = row_for_uid(candidate.uid_slabs, uid, pool=pool)
    if main is None: return None
    row = int(candidate.eq_roots[candidate.old_n_eq + main])
    return None if row < 0 else (False, row)
