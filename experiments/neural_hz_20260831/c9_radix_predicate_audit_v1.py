"""Independent power-path elimination of ALL encoded HZ predicates."""

import numpy as np

from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import equal_payload


def row(matrix, index):
    start, stop = matrix.indptr[index:index + 2]
    return matrix.indices[start:stop], matrix.data[start:stop]


def dyadic_power(value, *, positive=True):
    mantissa, exponent = np.frexp(abs(float(value)))
    if mantissa != .5 or (positive and value <= 0.):
        raise ValueError('internal link/pivot is not a positive dyadic power')
    return int(exponent) - 1


def unscale(values, power):
    values = np.asarray(values, dtype=np.float64)
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        restored = np.ldexp(values, int(power))
        back = np.ldexp(restored, -int(power))
    if not np.isfinite(restored).all() or not np.array_equal(values, back):
        raise ValueError('elimination lost original coefficient bits')
    return restored


def recover_row(packed, index, *, inequality=False):
    old, hz = packed.original, packed.hz
    roots = packed.ineq_roots if inequality else packed.eq_roots
    scales = packed.ineq_scales if inequality else packed.eq_scales
    cmat, bmat, rhs = (hz.Auc, hz.Aub, hz.ub) if inequality else (hz.Ac, hz.Ab, hz.b)
    cc, cv = row(cmat, int(roots[index]))
    bc, bv = row(bmat, int(roots[index]))
    continuous, binary, reached = {}, {}, set()

    def leaves(columns, values, shift, sign, target):
        restored = unscale(sign * values, shift)
        for col, value in zip(columns, restored):
            if int(col) in target or value == 0.:
                raise ValueError('duplicated/zero original coefficient in radix tree')
            target[int(col)] = float(value)

    def walk(cc, cv, bc, bv, shift, sign):
        original = cc < old.n_cont
        leaves(cc[original], cv[original], shift, sign, continuous)
        leaves(bc, bv, shift, sign, binary)
        for col, coefficient in zip(cc[~original], cv[~original]):
            slot = int(col) - old.n_cont
            if slot in reached or not 0 <= slot < packed.def_rows.size:
                raise ValueError('shared/cyclic/unregistered radix auxiliary')
            reached.add(slot)
            link_power = dyadic_power(coefficient, positive=False)
            definition = int(packed.def_rows[slot])
            dc, dv = row(hz.Ac, definition)
            db, dw = row(hz.Ab, definition)
            pivot = dyadic_power(dv[-1])
            walk(dc[:-1], dv[:-1], db, dw, shift + link_power - pivot,
                -sign * (1 if coefficient > 0. else -1))

    scale = int(scales[index])
    walk(cc, cv, bc, bv, -scale, 1)
    value = float(unscale([rhs[int(roots[index])]], -scale)[0])
    return continuous, binary, value, reached


def audit(packed):
    packed.validate()
    old, hz = packed.original, packed.hz
    if (hz.n_cont != old.n_cont + packed.def_rows.size or hz.n_bin != old.n_bin
            or hz.frame_id != old.frame_id or not hz.exact or hz.n_eq != old.n_eq + packed.def_rows.size
            or hz.n_ineq != old.n_ineq):
        raise ValueError('original frame/dimensions/phase factors not retained')
    for mapping, count in ((packed.eq_roots, old.n_eq), (packed.eq_scales, old.n_eq),
                          (packed.ineq_roots, old.n_ineq), (packed.ineq_scales, old.n_ineq)):
        if mapping.dtype != np.dtype(np.int64) or mapping.shape != (count,):
            raise ValueError('invalid logical row map')
    if (packed.def_rows.dtype != np.dtype(np.int64) or packed.def_rows.ndim != 1
            or not np.array_equal(np.sort(np.concatenate((packed.eq_roots, packed.def_rows))), np.arange(hz.n_eq))
            or not np.array_equal(np.sort(packed.ineq_roots), np.arange(hz.n_ineq))):
        raise ValueError('missing, duplicated or hidden encoded predicate')
    if (not equal_payload(hz.c, old.c) or not equal_payload(hz.Gb, old.Gb)
            or not equal_payload(hz.Gc[:, :old.n_cont], old.Gc) or hz.Gc[:, old.n_cont:].nnz):
        raise ValueError('original value map changed')
    for matrix in (hz.Ac, hz.Ab, hz.Auc, hz.Aub):
        if matrix.data.size and (np.any(np.abs(matrix.data) < 2.**-20) or np.any(np.abs(matrix.data) > 2.**40)):
            raise ValueError('encoded coefficients violate fixed window')
    for slot, definition in enumerate(packed.def_rows):
        cc, cv = row(hz.Ac, definition)
        bc, bv = row(hz.Ab, definition)
        if not cc.size or cc[-1] != old.n_cont + slot or np.any(cc[:-1] >= cc[-1]) or hz.b[definition] != 0.:
            raise ValueError('radix factor is not uniquely homogeneously defined')
        pivot = dyadic_power(cv[-1])
        internal = cc[:-1] >= old.n_cont
        for coefficient in cv[:-1][internal]:
            dyadic_power(-coefficient)
        values = np.abs(np.concatenate((cv[:-1], bv)))
        if not values.size:
            raise ValueError('empty auxiliary definition')
        exponents = np.frexp(values)[1]
        minimum = int(exponents.min())
        envelope = sum(1 << (int(e) - minimum) for e in exponents)
        if minimum + (envelope - 1).bit_length() > pivot:
            raise ValueError('redundant auxiliary box not proved')
    checked = 0
    used = set()
    for inequality, cmat, bmat, rhs in ((False, old.Ac, old.Ab, old.b), (True, old.Auc, old.Aub, old.ub)):
        for index in range(cmat.shape[0]):
            continuous, binary, value, reached = recover_row(packed, index, inequality=inequality)
            if used & reached:
                raise ValueError('unregistered cross-row radix factor sharing')
            used |= reached
            for actual, matrix in ((continuous, cmat), (binary, bmat)):
                cols, vals = row(matrix, index)
                expected = {int(col): float(v) for col, v in zip(cols, vals) if v != 0.}
                if actual != expected:
                    raise ValueError('radix elimination differs from original coefficients')
                checked += len(expected)
            if value != rhs[index]:
                raise ValueError('logical predicate RHS changed')
    if used != set(range(packed.def_rows.size)):
        raise ValueError('unregistered or unused auxiliary definition')
    packed.validate()
    return {'all_logical_equalities_checked': old.n_eq, 'all_logical_inequalities_checked': old.n_ineq,
        'all_original_nonzero_coefficients_checked': checked, 'all_auxiliary_definitions_checked': packed.def_rows.size,
        'all_redundant_boxes_proved': True, 'exact_two_way_predicate_equivalence': True,
        'original_continuous_binary_frame_and_value_retained': True,
        'witness_reconstruction': 'unique radix extension and original continuous prefix projection',
        'full_affine_suffix_qualification_claimed': False}
