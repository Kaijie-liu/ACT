"""C99 full-population proof for explicit circuit-source native splicing.

No function here issues a native receipt or selects on network identity. The
caller must independently authenticate complete source and phase proofs.
"""
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import json
import math
import numpy as np
import scipy.sparse as sp
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import RADIX, unique_other
from experiments.neural_hz_20260831.c22_uid_runs_v1 import row_for_uid
from experiments.neural_hz_20260831.c99_circuit_consumer_v2 import closed_uid_tables
from experiments.neural_hz_20260831.c99_circuit_journal_v2 import CircuitJournal
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import Plan
from experiments.neural_hz_20260831.c30_first_write_v1 import AppendView, row
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import incidence_oracle
from experiments.neural_hz_20260831.c62_local_equations_v1 import SCHEMA, decode
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import fraction

PACKET = 'c70_authenticated_offline_phase_packet_v1'


def equal(a, b):
    """Exact array bits, including signs of zero."""
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def same_matrix(a, b):
    return a.shape == b.shape and all(equal(getattr(a, k), getattr(b, k))
                                     for k in ('data', 'indices', 'indptr'))


def entries(hz):
    return sum(int(getattr(hz, k).size) for k in ('c', 'b', 'ub')) + sum(
        int(m.data.size + m.indices.size + m.indptr.size)
        for m in (hz.Gc, hz.Gb, hz.Ac, hz.Ab, hz.Auc, hz.Aub))


def digest(value):
    """Portable content hash for the explicit small packet/proof dictionaries."""
    h = hashlib.sha256()
    def visit(v):
        if type(v) is dict:
            h.update(b'dict')
            for k in sorted(v):
                visit(k); visit(v[k])
        elif type(v) in (list, tuple):
            h.update(str((type(v).__name__, len(v))).encode())
            for x in v: visit(x)
        elif type(v) is np.ndarray:
            h.update(str((v.shape, v.dtype.str)).encode()); h.update(v.tobytes())
        elif sp.isspmatrix_csr(v):
            visit(list(v.shape))
            for k in ('data', 'indices', 'indptr'): visit(getattr(v, k))
        elif v is None or type(v) in (str, int, float, bool):
            h.update(json.dumps(v, allow_nan=False).encode() + b'\0')
        else:
            raise ValueError('unregistered packet field: ' + type(v).__name__)
    visit(value)
    return h.hexdigest()


def extract(pre, post, *, old_n_cont, old_n_eq, logical_n_cont, first_uid,
            provenance, pool, enabled=False):
    """Copy only phase data after the caller's complete prefix/source proof."""
    if not enabled: return None
    count = (len(pre.c) + pre.Gc.nnz * 2 + pre.Gb.nnz * 2 +
             len(post.c) + post.Gc.nnz * 2 + post.Gb.nnz * 2 +
             sum(2 * (getattr(post, k).nnz - getattr(pre, k).nnz)
                 for k in ('Ac', 'Ab', 'Auc', 'Aub')) +
             4 * (post.n_eq - pre.n_eq + post.n_ineq - pre.n_ineq) +
             4 * (pre.n_out + post.n_out + 1))
    pool.charge('c70_copy_complete_phase_packet', 4 * int(count) + 512)
    return dict(schema=PACKET, offline_only=True, fresh_native_execution=False,
        pre_c=pre.c.copy(), pre_Gc=pre.Gc.copy(), pre_Gb=pre.Gb.copy(),
        frame_id=pre.frame_id, source_n_cont=pre.n_cont, source_n_bin=pre.n_bin,
        old_n_cont=old_n_cont, old_n_eq=old_n_eq, logical_n_cont=logical_n_cont,
        first_uid=first_uid, provenance=provenance,
        eq_c=post.Ac[pre.n_eq:].copy(), eq_b=post.Ab[pre.n_eq:].copy(),
        eq_rhs=post.b[pre.n_eq:].copy(), le_c=post.Auc[pre.n_ineq:].copy(),
        le_b=post.Aub[pre.n_ineq:].copy(), le_rhs=post.ub[pre.n_ineq:].copy(),
        c=post.c.copy(), Gc=post.Gc.copy(), Gb=post.Gb.copy())


def bind_phase(c, packet, *, pool, enabled=False):
    if not enabled: return None
    h = c.hz
    pool.charge('c70_complete_pre_output_frame_binding',
                4 * (len(h.c) + 2 * h.Gc.nnz + 2 * h.Gb.nnz + 2 * (h.n_out + 1)) + 256)
    if (packet['schema'] != PACKET or not packet['offline_only'] or packet['fresh_native_execution']
            or packet['frame_id'] != h.frame_id or packet['source_n_cont'] != h.n_cont
            or packet['source_n_bin'] != h.n_bin or packet['old_n_cont'] != c.old_n_cont
            or packet['old_n_eq'] != c.old_n_eq or packet['logical_n_cont'] != c.logical_n_cont
            or packet['first_uid'] != c.report['radix_uid_base'] + 16384
            or not equal(packet['pre_c'], h.c) or not same_matrix(packet['pre_Gc'], h.Gc)
            or not same_matrix(packet['pre_Gb'], h.Gb)):
        raise ValueError('complete actual phase/pre-output/global source binding differs')
    view = AppendView(h, *(packet[k] for k in
        ('eq_c', 'eq_b', 'eq_rhs', 'le_c', 'le_b', 'le_rhs', 'c', 'Gc', 'Gb')))
    view.validate_shape(pool)
    return view


def factor_plans(c, view, *, first_uid, pool):
    """All MAIN factors; certified old prefix plus literal complete append.

    This reference does not call the event builder or consumer discovery.
    The complete source owner vector must already be bound to its source proof.
    """
    main = c.logical_n_cont - c.old_n_cont
    pool.charge('c70_complete_source_UID_and_owner_partition',
                8 * (view.n_eq + view.n_ineq + main))
    eq, le = closed_uid_tables(c)
    eq = np.r_[eq, np.arange(first_uid, first_uid + len(view.eq_rhs), dtype=np.int64)]
    le = np.r_[le, np.arange(first_uid + len(view.eq_rhs),
                           first_uid + len(view.eq_rhs) + len(view.le_rhs), dtype=np.int64)]
    lookup = {int(u): (False, i) for i, u in enumerate(eq)}
    lookup.update({int(u): (True, i) for i, u in enumerate(le)})
    if len(lookup) != len(eq) + len(le): raise ValueError('duplicate full physical UID')
    words = c.owners.copy()
    for m, first in ((view.eq_c, first_uid), (view.le_c, first_uid + len(view.eq_rhs))):
        pool.charge('c70_independent_every_appended_incidence', 12 * int(m.nnz) + 8 * m.shape[0])
        fresh = type(m)((m.data, m.indices, m.indptr), shape=m.shape, copy=False)
        if not fresh.has_canonical_format or not np.isfinite(m.data).all() or np.any(m.data == 0):
            raise ValueError('noncanonical complete actual append')
        for r in range(m.shape[0]):
            cols, _ = row(m, r)
            for raw in cols:
                col = int(raw)
                if c.old_n_cont <= col < c.logical_n_cont:
                    words[col - c.old_n_cont] += RADIX + first + r
    pool.charge('c70_independent_full_factor_liveness_scan', 32 * main + 4 * int(view.Gc.nnz))
    live = set(map(int, view.Gc.indices))
    plans = []
    for i, packed in enumerate(words):
        col = c.old_n_cont + i
        d = int(c.eq_roots[c.old_n_eq + i])
        if d < 0 or col in live or int(packed) // RADIX != 2: continue
        pc, pv = row(c.hz.Ac, d)
        if not len(pc) or int(pc[-1]) != col or c.hz.Ab.indptr[d] != c.hz.Ab.indptr[d + 1]: continue
        pivot = float(pv[-1])
        if pivot <= 0 or not math.isfinite(pivot) or math.frexp(pivot)[0] != .5: continue
        u = int(eq[d]); v = unique_other(int(packed), u)
        if v not in lookup or v == u: raise ValueError('factor consumer UID not unique')
        kind, r = lookup[v]; m, rhs, at = view.locate(kind, r); cc, cv = row(m, at)
        if not len(cc) or int(cc[0]) != col or abs(float(cv[0])) != pivot: continue
        pool.charge('c70_independent_factor_plan', 160 + 4 * (len(cc) - 1))
        sign = -1 if cv[0] > 0 else 1
        origin = float(c.hz.b[d]); target = float(rhs[at]); updated = target + sign * origin
        if not math.isfinite(updated) or F(updated) != F(target) + sign * F(origin):
            raise ValueError('factor reference exact RHS failed')
        rank = row_for_uid(c.uid_slabs, v, pool=pool)
        consumer_main = None if rank is None else c.old_n_cont + rank
        plans.append(Plan(col, d, r, kind, u, v, pivot, sign, origin,
                          consumer_main, tuple(map(int, cc[1:]))))
    defs = {p.definition for p in plans}
    if (len(defs) != len(plans) or len({(p.inequality, p.consumer) for p in plans}) != len(plans)
            or any(not p.inequality and p.consumer in defs for p in plans)):
        raise ValueError('independent population has selected-definition dependencies')
    return plans, words, eq, le


def verify(c, view, overlay, plans, new, journal, *, pool):
    """Full row/add/delete, source-map, UID and actual new incidence proof."""
    if type(journal) is not CircuitJournal or journal.state is not c.state:
        raise ValueError('complete explicit circuit native journal required')
    circuit_journal=journal; journal=journal.local
    expected, words, eq, le = factor_plans(c, view, first_uid=overlay.old_uid_ceiling, pool=pool)
    if [asdict(p) for p in expected] != [asdict(p) for p in plans]:
        raise ValueError('all-factor and all-consumer populations differ')
    for i, word in enumerate(overlay.iter_words(pool=pool)):
        if word != int(words[i]): raise ValueError('actual append incidence differs from overlay')
    if (journal.eq_roots is not c.eq_roots or journal.eq_scales is not c.eq_scales
            or journal.source_schema != SCHEMA or journal.source_n_cont != c.hz.n_cont
            or journal.columns.tolist() != [p.column for p in plans]):
        raise ValueError('immutable source-local journal binding differs')
    pool.charge('c70_independent_complete_row_partition', 12 * (view.n_eq + view.n_ineq))
    deleted = {p.definition for p in plans}; changes = {(p.inequality, p.consumer): p for p in plans}
    keep = np.array([r not in deleted for r in range(view.n_eq)], bool)
    eq_new, le_new = eq[keep].copy(), le.copy()
    rowmap = np.cumsum(keep, dtype=np.int64) - 1
    retired = {}; checked = parents = 0
    if (new.n_cont != view.n_cont or new.n_bin != view.n_bin or new.n_eq != view.n_eq - len(plans)
            or new.n_ineq != view.n_ineq or not new.exact or new.frame_id != c.hz.frame_id
            or not same_matrix(new.Gc, view.Gc) or not same_matrix(new.Gb, view.Gb)
            or not equal(new.c, view.c)):
        raise ValueError('binary/frame/output or full row counts changed')
    for kind in (False, True):
        out_c, out_b, out_rhs = (new.Auc, new.Aub, new.ub) if kind else (new.Ac, new.Ab, new.b)
        for r in range(view.n_ineq if kind else view.n_eq):
            if not kind and r in deleted: continue
            target = r if kind else int(rowmap[r])
            m, rhs, at = view.locate(kind, r); cc, cv = row(m, at)
            old_rows = c.hz.n_ineq if kind else c.hz.n_eq
            bm = view.blocks('Aub' if kind else 'Ab')[int(r >= old_rows)]
            bc, bv = row(bm, at); nc, nv = row(out_c, target); nb, nbv = row(out_b, target)
            p = changes.get((kind, r))
            pool.charge('c70_independent_all_written_row_bits', 4 * (len(nc) + len(nb)) + 16)
            if p is None:
                good = equal(nc, cc) and equal(nv, cv) and equal(out_rhs[target:target+1], rhs[at:at+1])
            else:
                pc, pv = row(c.hz.Ac, p.definition); cut = len(pc) - 1
                good = (len(nc) == cut + len(cc) - 1 and equal(nc[:cut], pc[:-1])
                        and equal(nv[:cut], p.sign * pv[:-1]) and equal(nc[cut:], cc[1:])
                        and equal(nv[cut:], cv[1:])
                        and F(float(out_rhs[target])) == F(float(rhs[at])) + p.sign * F(p.offset))
                retired[p.consumer_uid] = p.producer_uid
                (le_new if kind else eq_new)[target] = p.producer_uid
                parents += cut
            if not good or not equal(nb, bc) or not equal(nbv, bv):
                raise ValueError('complete exact written row differs at ' + str((kind, r)))
            checked += len(nc) + len(nb)
    for uid in np.r_[eq, le]:
        if journal.retired_to(int(uid), pool=pool) != retired.get(int(uid)):
            raise ValueError('complete UID retirement differs')
    for r in range(view.n_eq):
        if journal.eq_row(r, pool=pool) != (int(rowmap[r]) if keep[r] else None):
            raise ValueError('complete physical EQ rank map differs')
    all_actual = incidence_oracle(new, eq_new, le_new, 0, new.n_cont, pool=pool)
    actual=all_actual[c.old_n_cont:c.logical_n_cont]
    circuit_actual=all_actual[c.state['old_source_n_cont']:c.hz.n_cont]
    for i,word in enumerate(circuit_journal.iter_circuit_words(pool=pool)):
        if word!=int(circuit_actual[i]):raise ValueError('all actual new circuit owners differ')
    for i, word in enumerate(journal.iter_words(overlay, pool=pool)):
        if word != int(actual[i]): raise ValueError('journal differs from all actual new MAIN incidences')
    before = sum(int(m.nnz) for k in ('Ac', 'Ab', 'Auc', 'Aub') for m in view.blocks(k))
    if checked != before - 2 * len(plans) or not checked < before:
        raise ValueError('strict complete predicate nnz decrease missing')
    return dict(all_circuit_factors=len(circuit_actual), all_circuit_incidence_equal=True,
        all_MAIN_factors=len(words), complete_plans=len(plans), all_original_UIDs=len(eq)+len(le),
        all_original_EQ_rank_queries=view.n_eq, all_written_predicate_coefficients=checked,
        all_parent_prefix_terms=parents, old_predicate_nnz=before, new_predicate_nnz=checked,
        source_local_equations=int(np.count_nonzero(c.eq_roots < 0)),
        certified_original_prefix_plus_literal_complete_append=True,
        all_actual_new_incidence_equal=True, immutable_source_maps_preserved=True,
        source_box_proof_transferred_by_exact_unit_row_identity=True,
        fresh_inverse_check_still_required=True, full_LIVE_admission=False, formal_gain=0)
