"""Read-only complete current UID/incidence proof; never an old Closed reader.

All dense arrays here are DIAGNOSTIC, charged, and not a runtime design.
The caller must authenticate the entire actual SplicedState first and last.
"""
import numpy as np
from experiments.neural_hz_20260831.c32_fresh_lineage_v1 import original_entry, decode
from experiments.neural_hz_20260831.c22_uid_runs_v1 import unpack, validate
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import Overlay
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import incidence_oracle
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import UID_LIMIT


def current_tables(state, *, pool):
    f, hz, lin = state.original_fields, state.hz, state.lineage
    lin.validate(); validate(f['uid_slabs'])
    main = f['logical_n_cont'] - f['old_n_cont']
    old_eq = hz.n_eq + len(lin.columns)
    ne, nl = old_eq-f['hz'].n_eq, hz.n_ineq-f['hz'].n_ineq
    if (main < 0 or len(f['eq_roots']) != f['old_n_eq']+main or ne < 0 or nl < 0
            or state.old_uid_ceiling+ne+nl > UID_LIMIT
            or lin.eq_roots is not f['eq_roots'] or lin.eq_scales is not f['eq_scales']):
        raise ValueError('incomplete current/source map geometry')
    pool.charge('current_UID_arrays_and_partition', 16*(old_eq+hz.n_eq+2*hz.n_ineq+main))
    eq0 = np.full(old_eq, -1, np.int64)
    le0 = np.full(hz.n_ineq, -1, np.int64)
    main_uid = np.full(main, -1, np.int64)

    def assign(arr, row, uid):
        if not 0 <= row < len(arr) or arr[row] != -1 or not 0 <= uid < UID_LIMIT:
            raise ValueError('duplicate/outside original physical UID assignment')
        arr[row] = uid

    pool.charge('inverse_all_original_source_slots', 32*(len(f['eq_roots'])+len(f['ineq_roots'])+len(f['def_rows'])))
    for at in range(f['old_n_eq']):
        row, _ = original_entry(f['eq_roots'][at], f['eq_scales'][at])
        assign(eq0, row, at)
    for at, row in enumerate(f['ineq_roots']):
        assign(le0, int(row), f['old_n_eq']+at)
    for raw in f['uid_slabs']:
        uid, first, length = unpack(raw)
        if first+length > main:
            raise ValueError('MAIN slab exceeds complete source frame')
        for i in range(first, first+length):
            if main_uid[i] != -1:
                raise ValueError('duplicate MAIN UID ownership')
            main_uid[i] = uid+i-first
            at = f['old_n_eq']+i
            row, _ = original_entry(f['eq_roots'][at], f['eq_scales'][at])
            if row >= 0:
                assign(eq0, row, int(main_uid[i]))
    for i, row in enumerate(f['def_rows']):
        assign(eq0, int(row), f['report']['radix_uid_base']+i)
    for i in range(ne):
        assign(eq0, f['hz'].n_eq+i, state.old_uid_ceiling+i)
    for i in range(nl):
        assign(le0, f['hz'].n_ineq+i, state.old_uid_ceiling+ne+i)
    if np.any(eq0 < 0) or np.any(le0 < 0) or np.any(main_uid < 0):
        raise ValueError('incomplete original physical/MAIN UID coverage')
    eq = np.full(hz.n_eq, -1, np.int64)
    le = np.full(hz.n_ineq, -1, np.int64)
    rank = np.full(old_eq, -1, np.int64)
    for oldrow, raw in enumerate(eq0):
        row = lin.eq_row(oldrow, pool=pool)
        if row is None:
            continue
        uid = int(raw); replacement = lin.retired_to(uid, pool=pool)
        assign(eq, row, uid if replacement is None else replacement)
        rank[oldrow] = row
    for row, raw in enumerate(le0):
        uid = int(raw); replacement = lin.retired_to(uid, pool=pool)
        assign(le, row, uid if replacement is None else replacement)
    lookup = {int(uid):(False,row) for row,uid in enumerate(eq)}
    lookup.update({int(uid):(True,row) for row,uid in enumerate(le)})
    if np.any(eq < 0) or np.any(le < 0) or len(lookup) != len(eq)+len(le):
        raise ValueError('incomplete/reused transferred physical UID')
    pool.charge('current_definition_rank_and_UID', 16*main)
    definitions = np.full(main, -1, np.int64)
    for i, uid in enumerate(main_uid):
        at = f['old_n_eq']+i
        info = decode(f['eq_roots'][at])
        if info[0] in ('alias','splice'):
            continue
        oldrow, _ = original_entry(f['eq_roots'][at], f['eq_scales'][at])
        if not 0 <= oldrow < len(rank) or rank[oldrow] < 0:
            raise ValueError('surviving MAIN lost its current definition')
        row = int(rank[oldrow]); replacement = lin.retired_to(int(uid), pool=pool)
        if int(eq[row]) != (int(uid) if replacement is None else replacement):
            raise ValueError('redirected MAIN definition has a stale UID')
        definitions[i] = row
    return dict(eq=eq, le=le, definitions=definitions, lookup=lookup), dict(
        all_original_EQ_ranks=old_eq, all_current_physical_UIDs=len(lookup),
        all_logical_MAIN_mappings=main, original_phase_EQ_rows=ne, original_phase_LE_rows=nl,
        complete_unique_current_UID_partition=True, obsolete_Closed_reader_used=False)


def checked_incidence(state, tables, *, pool):
    f = state.original_fields
    actual = incidence_oracle(state.hz, tables['eq'], tables['le'],
        f['old_n_cont'], f['logical_n_cont'], pool=pool)
    overlay = Overlay(f['owners'], state.events, state.old_uid_ceiling)
    overlay.validate()
    pool.charge('compare_every_transferred_MAIN_word', 8*len(actual))
    count = 0
    for i, value in enumerate(state.lineage.iter_words(overlay, pool=pool)):
        if value != int(actual[i]):
            raise ValueError('current predicate incidence differs from complete transferred ownership')
        count += 1
    if count != len(actual):
        raise ValueError('incomplete current ownership comparison')
    return actual
