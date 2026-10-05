"""Independent Fraction substitution audit; does not use quotient arithmetic."""

from fractions import Fraction
import numpy as np

from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import equal_payload


def _row(matrix, row):
    a, b = matrix.indptr[row:row + 2]
    return matrix.indices[a:b], matrix.data[a:b]


def audit(original, reduced):
    reduced.validate()
    if source_digest(original) != reduced.original_digest:
        raise ValueError('wrong original HZ for quotient proof')
    hz = reduced.hz
    if (hz.n_cont != original.n_cont or hz.n_bin != original.n_bin or hz.n_out != original.n_out
            or hz.frame_id != original.frame_id or not hz.exact
            or hz.n_eq != original.n_eq - reduced.columns.size or hz.n_ineq != original.n_ineq):
        raise ValueError('frame/binary/predicate dimensions changed')
    for name in ('c', 'Gc', 'Gb', 'Aub', 'ub'):
        if not equal_payload(getattr(hz, name), getattr(original, name)):
            raise ValueError(f'protected value/binary payload changed: {name}')
    cs, ps, rs, ds = reduced.columns, reduced.parents, reduced.ratios, reduced.defining_rows
    if (not 0 <= reduced.old_n_cont <= reduced.logical_n_cont <= hz.n_cont
            or np.any(cs < reduced.old_n_cont) or np.any(cs >= reduced.logical_n_cont)
            or np.any(np.diff(cs) <= 0) or np.unique(ds).size != ds.size
            or np.any(ds < 0) or np.any(ds >= original.n_eq)
            or np.any(ps < 0) or np.any(ps >= cs) or np.isin(ps, cs).any()
            or not np.isfinite(rs).all() or np.any(rs == 0.) or np.any(np.abs(rs) > 1.)):
        raise ValueError('invalid protected/independent reconstruction map')
    mapping = {int(c): (int(p), Fraction(float(r))) for c, p, r in zip(cs, ps, rs)}
    for c, p, r, d in zip(cs, ps, rs, ds):
        cc, vv = _row(original.Ac, int(d))
        bc, _ = _row(original.Ab, int(d))
        if cc.tolist() != [int(p), int(c)] or bc.size or original.b[d] != 0.:
            raise ValueError('erased row is not a unique homogeneous continuous definition')
        if vv[-1] <= 0. or np.frexp(vv[-1])[0] != .5:
            raise ValueError('defining pivot is not positive power of two')
        if Fraction(float(vv[0])) + Fraction(float(vv[1])) * Fraction(float(r)) != 0:
            raise ValueError('reconstruction ratio does not satisfy erased definition')
    for matrix in (original.Gc, hz.Gc, hz.Ac, hz.Auc):
        if np.isin(matrix.indices, cs).any():
            raise ValueError('selected factor is output-live or still predicate-active')
    keep = np.ones(original.n_eq, dtype=bool)
    keep[ds] = False
    if not equal_payload(hz.Ab, original.Ab[keep]) or not equal_payload(hz.b, original.b[keep]):
        raise ValueError('surviving binary equality coefficients or RHS changed')
    checked, changed, unchanged = 0, 0, 0
    for name, rowmap in (('Ac', np.flatnonzero(keep)), ('Auc', np.arange(original.n_ineq))):
        source, target = getattr(original, name), getattr(hz, name)
        for newrow, oldrow in enumerate(rowmap):
            cc, vv = _row(source, int(oldrow))
            tc, tv = _row(target, newrow)
            if not any(int(col) in mapping for col in cc):
                if not np.array_equal(cc, tc) or not np.array_equal(vv, tv):
                    raise ValueError('unaffected predicate row changed')
                unchanged += 1
            else:
                expected = {}
                for col, value in zip(cc, vv):
                    parent, ratio = mapping.get(int(col), (int(col), Fraction(1)))
                    expected[parent] = expected.get(parent, Fraction(0)) + Fraction(float(value)) * ratio
                expected = {col: value for col, value in expected.items() if value}
                actual = {int(col): Fraction(float(value)) for col, value in zip(tc, tv)}
                if len(actual) != len(tc) or actual != expected:
                    raise ValueError(f'Fraction substitution mismatch in {name} row {int(oldrow)}')
                changed += 1
            checked += len(cc)
    for name in ('Gc', 'Gb', 'Ac', 'Ab', 'Auc', 'Aub'):
        matrix = getattr(hz, name)
        if (not matrix.has_canonical_format or np.any(matrix.data == 0.)
                or not np.isfinite(matrix.data).all()
                or np.any(np.abs(matrix.data) < 2.**-20) or np.any(np.abs(matrix.data) > 2.**40)):
            raise ValueError('noncanonical/nonfinite/out-of-window quotient coefficient')
    reduced.validate()
    return {'status': 'EXACT_TWO_WAY_CONTINUOUS_QUOTIENT', 'all_definitions_checked': int(cs.size),
        'surviving_equalities_checked': hz.n_eq, 'inequalities_checked': hz.n_ineq,
        'fraction_changed_rows_checked': changed, 'unchanged_rows_checked': unchanged,
        'surviving_original_continuous_coefficients_checked': checked,
        'redundant_boxes_proved': True, 'original_input_prefix_unchanged': True,
        'all_binary_factors_retained': True, 'shared_frame_unchanged': True,
        'witness_extension': 'independent exact x_j = ratio_j * x_parent; input prefix unchanged',
        'whole_live_path_proved': False, 'formal_gain': 0}
