"""Independent exact two-way row-pair audit; no builder/decoder arithmetic."""

from fractions import Fraction as F
import numpy as np

from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import equal_payload


def row(matrix, index):
    start, stop = matrix.indptr[index:index + 2]
    return matrix.indices[start:stop], matrix.data[start:stop]


def audit(original, result, *, old_n_eq, eq_roots, input_hz=None):
    result.validate()
    before = source_digest(original)
    if before != result.original_digest:
        raise ValueError('wrong original for unit-splice proof')
    new = result.hz
    n = len(result.columns)
    if (new.n_cont != original.n_cont or new.n_bin != original.n_bin or new.n_out != original.n_out
            or new.n_eq != original.n_eq - n or new.n_ineq != original.n_ineq
            or new.frame_id != original.frame_id or not new.exact):
        raise ValueError('unit splice changed global frame/binaries/dimensions')
    if input_hz is not None and (input_hz.frame_id != new.frame_id or input_hz.n_cont > result.old_n_cont):
        raise ValueError('original input is outside the protected global prefix')
    for name in ('c', 'Gc', 'Gb', 'Aub'):
        if not equal_payload(getattr(original, name), getattr(new, name)):
            raise ValueError(f'protected value/binary map changed: {name}')
    selected = set(map(int, result.columns))
    for matrix in (original.Gc, new.Gc, new.Ac, new.Auc):
        if np.isin(matrix.indices, result.columns).any():
            raise ValueError('selected coordinate was output-live or survives in predicates')
    eqc, lec = original.Ac.tocsc(), original.Auc.tocsc()
    pairs, cursor = [], 0
    for col, raw in zip(result.columns, result.descriptors):
        col, raw = int(col), int(raw)
        flag = raw >> 32
        pivot = F(2) ** ((flag & 63) - 20)
        sign = -1 if flag & 64 else 1
        offset = F(float(result.offsets[cursor])) if flag & 128 else F(0)
        cursor += bool(flag & 128)
        new_consumer, inequality = raw & ((1 << 31) - 1), bool(raw & (1 << 31))
        d = int(eq_roots[old_n_eq + col - result.old_n_cont])
        if not 0 <= d < original.n_eq:
            raise ValueError('erased definition not in original registered MAIN map')
        pc, pv = row(original.Ac, d)
        pb, _ = row(original.Ab, d)
        if (not len(pc) or int(pc[-1]) != col or F(float(pv[-1])) != pivot or len(pb)
                or F(float(original.b[d])) != offset or any(int(c) in selected for c in pc[:-1])):
            raise ValueError('not an independent binary-free registered definition')
        if sum((abs(F(float(v))) for v in pv[:-1]), F(0)) + abs(offset) > pivot:
            raise ValueError('removed variable box was not redundant')
        incidence = []
        for kind, matrix in ((False, eqc), (True, lec)):
            begin, end = matrix.indptr[col:col + 2]
            for r in matrix.indices[begin:end]:
                if kind or int(r) != d:
                    incidence.append((kind, int(r)))
        if len(incidence) != 1 or incidence[0][0] != inequality:
            raise ValueError('not a unique original predicate consumer')
        consumer = incidence[0][1]
        cm = original.Auc if inequality else original.Ac
        tc, tv = row(cm, consumer)
        if (not len(tc) or int(tc[0]) != col or F(float(tv[0])) != -sign * pivot
                or np.any(pc[:-1] >= col) or np.any(tc[1:] <= col)):
            raise ValueError('source and consumer supports do not permit in-row reconstruction')
        pairs.append(dict(column=col, definition=d, consumer=consumer, new_consumer=new_consumer,
            inequality=inequality, pivot=pivot, sign=sign, offset=offset))
    if cursor != len(result.offsets):
        raise ValueError('unconsumed sparse offset metadata')
    defs = {p['definition'] for p in pairs}
    consumers = {(p['inequality'], p['consumer']) for p in pairs}
    if len(defs) != n or len(consumers) != n or any(not k and r in defs for k, r in consumers):
        raise ValueError('simultaneous row-pair interference')
    keep = np.ones(original.n_eq, bool)
    keep[list(defs)] = False
    if not equal_payload(new.Ab, original.Ab[keep]):
        raise ValueError('surviving binary equality payload changed')
    rowmap = np.cumsum(keep, dtype=np.int64) - 1
    by_consumer = {(p['inequality'], p['consumer']): p for p in pairs}
    unchanged, changed, fraction_terms = 0, 0, 0
    for kind, source, target, rhs, new_rhs, rows in (
        (False, original.Ac, new.Ac, original.b, new.b, np.flatnonzero(keep)),
        (True, original.Auc, new.Auc, original.ub, new.ub, np.arange(original.n_ineq))):
        for newrow, oldrow in enumerate(rows):
            cc, cv = row(source, int(oldrow))
            tc, tv = row(target, newrow)
            pair = by_consumer.get((kind, int(oldrow)))
            if pair is None:
                if (not equal_payload(cc, tc) or not equal_payload(cv, tv)
                        or not equal_payload(rhs[int(oldrow):int(oldrow) + 1], new_rhs[newrow:newrow + 1])):
                    raise ValueError('unaffected predicate row changed')
                unchanged += 1
                continue
            expected_target = pair['consumer'] if kind else int(rowmap[pair['consumer']])
            if newrow != pair['new_consumer'] or newrow != expected_target:
                raise ValueError('encoded reconstruction target does not follow row deletion')
            pc, pv = row(original.Ac, pair['definition'])
            expected = {int(c): F(float(v)) for c, v in zip(cc, cv)}
            for c, v in zip(pc, pv):
                c = int(c)
                expected[c] = expected.get(c, F(0)) + pair['sign'] * F(float(v))
            expected = {c: v for c, v in expected.items() if v}
            actual = {int(c): F(float(v)) for c, v in zip(tc, tv)}
            if len(actual) != len(tc) or actual != expected:
                raise ValueError('Fraction combined-row identity failed')
            if F(float(new_rhs[newrow])) != F(float(rhs[int(oldrow)])) + pair['sign'] * pair['offset']:
                raise ValueError('Fraction combined RHS identity failed')
            # The certificate reconstructs the ORIGINAL definition from only
            # the lower-index prefix of the new consumer, not an old-row copy.
            cut = int(np.searchsorted(tc, pair['column']))
            recovered = {int(c): pair['sign'] * F(float(v)) for c, v in zip(tc[:cut], tv[:cut])}
            if recovered != {int(c): F(float(v)) for c, v in zip(pc[:-1], pv[:-1])}:
                raise ValueError('stored consumer cannot reconstruct original definition')
            changed += 1
            fraction_terms += len(expected)
    if changed != n or source_digest(original) != before:
        raise ValueError('incomplete/mutating unit-splice audit')
    result.validate()
    return {'status': 'EXACT_SIMULTANEOUS_UNIT_ROW_SPLICE', 'definitions_checked': n,
        'fraction_changed_rows_checked': changed, 'fraction_coefficients_checked': fraction_terms,
        'unchanged_predicate_rows_checked': unchanged, 'redundant_boxes_proved': True,
        'reconstruction_from_surviving_rows_proved': True, 'original_input_prefix_unchanged': True,
        'all_binary_factors_retained': True, 'same_global_frame': True,
        'whole_live_path_proved': False, 'formal_gain': 0}
