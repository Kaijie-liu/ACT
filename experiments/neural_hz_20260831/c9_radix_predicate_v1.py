"""Default-off exact radix factorization of complete nonconvex HZ predicates.

No solver, phase selection, instance identity or coefficient truncation.
Only new continuous factors are eliminated by the reconstruction proof.
"""

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono, sparse_pad_cols
from act.back_end.hybridz_tf.tf_cnn import _sparse_hz_storage_entries
from experiments.neural_hz_20260831.c7_factored_hz_v1 import scaled_exact
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c6_support_affine_plan_v1 import digest_arrays


def exponent_data(values, powers):
    values = np.asarray(values, dtype=np.float64)
    powers = np.asarray(powers)
    if powers.dtype.kind not in 'iu' or np.any(powers < -4096) or np.any(powers > 4096):
        raise ValueError('bounded integer coefficient powers required')
    powers = np.broadcast_to(powers.astype(np.int64), values.shape)
    if values.ndim != 1 or values.size > 64_000_000 or not np.isfinite(values).all() or np.any(values == 0.):
        raise ValueError('finite nonzero bounded coefficient vector required')
    mantissas, exponents = np.frexp(np.abs(values))
    return mantissas, exponents.astype(np.int64) + powers


def sum_unit(values, powers=0):
    _, exponents = exponent_data(values, powers)
    if not exponents.size:
        raise ValueError('empty partial sum')
    floor = int(exponents.max()) - 26
    total = int(np.left_shift(np.ones(exponents.size, dtype=np.int64),
        np.maximum(exponents - floor, 0)).sum(dtype=np.int64))
    unit = floor + (total - 1).bit_length()
    if not -1074 <= unit <= 1023:
        raise ValueError('nonfinite partial sum unit')
    return unit


def row_shift(values, powers=0):
    mantissas, exponents = exponent_data(values, powers)
    if not exponents.size:
        return 0
    lower = -19 - int(exponents.min())
    maximum = int(exponents.max())
    mantissa = float(mantissas[exponents == maximum].max())
    upper = (41 if mantissa == .5 else 40) - maximum
    if lower > upper:
        return None
    return min(max(0, lower), upper)


def _row(matrix, index):
    start, stop = matrix.indptr[index:index + 2]
    cols, vals = matrix.indices[start:stop], matrix.data[start:stop]
    selected = vals != 0.
    return cols[selected], vals[selected]


class RowEncoder:
    def __init__(self, n_cont, n_bin, base_entries, *, max_aux=16_384,
                 max_extra_entries=131_072, max_extra_work=16_000_000):
        for value, ceiling in ((max_aux, 16_384), (max_extra_entries, 131_072), (max_extra_work, 16_000_000)):
            if type(value) is not int or not 0 <= value <= ceiling:
                raise ValueError('invalid or increased radix reserve')
        self.nc, self.nb, self.base_entries = n_cont, n_bin, base_entries
        self.max_aux, self.max_extra_entries, self.max_extra_work = max_aux, max_extra_entries, max_extra_work
        self.eq, self.ineq, self.def_rows = [], [], []
        self.extra_work = self.entries = self.packed_rows = self.relays = 0

    def charge(self, work):
        if self.extra_work + work > self.max_extra_work:
            raise MemoryError('radix extra work reserve exhausted')
        self.extra_work += int(work)

    def emit(self, cc, cv, cp, bc, bv, bp, rhs, *, inequality=False, known_shift=None):
        shift = known_shift
        if shift is None:
            values, powers = np.concatenate((cv, bv)), np.concatenate((cp, bp))
            shift = row_shift(values, powers)
        if shift is None:
            raise ValueError('local radix definition exceeds fixed coefficient window')
        needed = int(cv.size + bv.size + 1)
        if self.entries + needed > self.base_entries + self.max_extra_entries:
            raise MemoryError('radix numeric entry reserve exhausted')
        cvals, bvals = scaled_exact(cv, cp + shift), scaled_exact(bv, bp + shift)
        result_rhs = float(scaled_exact([rhs], shift)[0])
        magnitudes = np.abs(np.concatenate((cvals, bvals)))
        if magnitudes.size and (magnitudes.min() < 2.**-20 or magnitudes.max() > 2.**40):
            raise ValueError('emitted row outside fixed window')
        rows = self.ineq if inequality else self.eq
        index = len(rows)
        rows.append((np.asarray(cc, dtype=np.int64).copy(), cvals,
            np.asarray(bc, dtype=np.int64).copy(), bvals, result_rhs))
        self.entries += needed
        return index, shift

    def auxiliary(self, cc, cv, cp, bc, bv, bp, *, forced_unit=None):
        if len(self.def_rows) >= self.max_aux:
            raise MemoryError('radix continuous auxiliary reserve exhausted')
        self.charge(12 * (cv.size + bv.size + 2))
        unit = sum_unit(np.concatenate((cv, bv)), np.concatenate((cp, bp))) if forced_unit is None else forced_unit
        if not -1074 <= unit <= 1023:
            raise ValueError('nonfinite auxiliary unit')
        slot = self.nc + len(self.def_rows)
        if np.any(cc >= slot):
            raise ValueError('radix definition not topological')
        row, unused_shift = self.emit(np.append(cc, slot), np.append(-cv, 1.), np.append(cp, unit),
            bc, -bv, bp, 0.)
        self.def_rows.append(row)
        return slot, unit

    def relay(self, term, target_unit):
        slot, unit = term
        while target_unit - unit > 24:
            next_unit = unit + 24
            slot, unit = self.auxiliary(np.array([slot]), np.ones(1), np.array([unit]),
                np.empty(0, dtype=np.int64), np.empty(0), np.empty(0, dtype=np.int64), forced_unit=next_unit)
            self.relays += 1
        return slot, unit

    def encode(self, cc, cv, bc, bv, rhs, *, cp=None, bp=None, inequality=False):
        cc, cv, bc, bv = map(np.asarray, (cc, cv, bc, bv))
        for power in (cp, bp):
            if power is not None:
                array = np.asarray(power)
                if array.dtype.kind not in 'iu' or np.any(array < -4096) or np.any(array > 4096):
                    raise ValueError('bounded integer coefficient powers required')
        cp = np.zeros(cv.size, dtype=np.int64) if cp is None else np.broadcast_to(cp, cv.shape).astype(np.int64)
        bp = np.zeros(bv.size, dtype=np.int64) if bp is None else np.broadcast_to(bp, bv.shape).astype(np.int64)
        if (cc.dtype.kind not in 'iu' or bc.dtype.kind not in 'iu'
                or cc.shape != cv.shape or bc.shape != bv.shape or cv.ndim != 1 or bv.ndim != 1
                or np.any(cc < 0) or np.any(cc >= self.nc) or np.any(bc < 0) or np.any(bc >= self.nb)
                or np.any(np.diff(cc) <= 0) or np.any(np.diff(bc) <= 0) or not np.isfinite(rhs)):
            raise ValueError('invalid original row coordinates')
        values, powers = np.concatenate((cv, bv)), np.concatenate((cp, bp))
        shift = row_shift(values, powers)
        if shift is not None:
            return self.emit(cc, cv, cp, bc, bv, bp, rhs, inequality=inequality, known_shift=shift)
        self.packed_rows += 1
        _, exponents = exponent_data(values, powers)
        buckets = (exponents - int(exponents.min())) // 24
        self.charge(values.size * ((int(values.size) - 1).bit_length() + 2))
        terms = []
        for bucket in np.unique(buckets):
            self.charge(values.size)
            cmask, bmask = buckets[:cv.size] == bucket, buckets[cv.size:] == bucket
            terms.append(self.auxiliary(cc[cmask], cv[cmask], cp[cmask], bc[bmask], bv[bmask], bp[bmask]))
        root = terms[0]
        for term in terms[1:]:
            high = max(root[1], term[1])
            root, term = self.relay(root, high), self.relay(term, high)
            ordered = sorted((root, term))
            root = self.auxiliary(np.array([v[0] for v in ordered]), np.ones(2),
                np.array([v[1] for v in ordered]), np.empty(0, dtype=np.int64), np.empty(0), np.empty(0, dtype=np.int64))
        slot, unit = root
        # A positive root-unit division preserves the inequality direction.
        root_rhs = float(scaled_exact([rhs], -unit)[0])
        self.charge(24)
        index, shift = self.emit(np.array([slot]), np.ones(1), np.zeros(1, dtype=np.int64),
            np.empty(0, dtype=np.int64), np.empty(0), np.empty(0, dtype=np.int64), root_rhs, inequality=inequality)
        return index, shift - unit

    def matrices(self, rows):
        nc = self.nc + len(self.def_rows)
        cptr = np.concatenate(([0], np.cumsum([r[0].size for r in rows], dtype=np.int64)))
        bptr = np.concatenate(([0], np.cumsum([r[2].size for r in rows], dtype=np.int64)))
        def concat(index, dtype):
            return np.concatenate([r[index] for r in rows]).astype(dtype, copy=False) if rows else np.empty(0, dtype=dtype)
        ac = sp.csr_matrix((concat(1, np.float64), concat(0, np.int64), cptr), shape=(len(rows), nc))
        ab = sp.csr_matrix((concat(3, np.float64), concat(2, np.int64), bptr), shape=(len(rows), self.nb))
        if not ac.has_canonical_format or not ab.has_canonical_format:
            raise ValueError('noncanonical radix matrix')
        return ac, ab, np.array([r[4] for r in rows], dtype=np.float64)


@dataclass
class Packed:
    original: object
    origin_digest: str
    hz: object
    eq_roots: np.ndarray
    eq_scales: np.ndarray
    ineq_roots: np.ndarray
    ineq_scales: np.ndarray
    def_rows: np.ndarray
    report: dict
    seal: str = ''

    def fingerprint(self):
        if set(vars(self)) != {'original', 'origin_digest', 'hz', 'eq_roots', 'eq_scales',
                'ineq_roots', 'ineq_scales', 'def_rows', 'report', 'seal'}:
            raise ValueError('unregistered packed state')
        h = hashlib.sha256(source_digest(self.hz).encode())
        h.update(digest_arrays(self.eq_roots, self.eq_scales, self.ineq_roots, self.ineq_scales, self.def_rows).encode())
        h.update(json.dumps(self.report, sort_keys=True, allow_nan=False).encode())
        return h.hexdigest()

    def validate(self):
        if self.origin_digest != source_digest(self.original) or self.seal != self.fingerprint():
            raise ValueError('source or radix predicate changed')

    def numeric_roots(self):
        self.validate()
        return {key: getattr(self, key) for key in ('original', 'hz', 'eq_roots', 'eq_scales', 'ineq_roots', 'ineq_scales', 'def_rows')}


def pack(hz, *, enabled=False, max_work=256_000_000, max_entries=64_000_000,
         max_aux=16_384, max_extra_entries=131_072, max_extra_work=16_000_000):
    if not enabled:
        return None
    if type(max_work) is not int or not 0 <= max_work <= 256_000_000 or type(max_entries) is not int or not 0 <= max_entries <= 64_000_000:
        raise ValueError('invalid or increased global ceiling')
    frozen = source_digest(hz)
    if not hz.exact:
        raise ValueError('exact HZ required')
    base_entries = sum(getattr(hz, key).nnz for key in ('Ac', 'Ab', 'Auc', 'Aub')) + hz.n_eq + hz.n_ineq
    base_work = 12 * base_entries
    original_entries = _sparse_hz_storage_entries(hz)
    if base_work + max_extra_work > max_work or original_entries + max_extra_entries > max_entries:
        raise MemoryError('complete predicate plus reserved preflight exceeds ceiling')
    encoder = RowEncoder(hz.n_cont, hz.n_bin, base_entries, max_aux=max_aux,
        max_extra_entries=max_extra_entries, max_extra_work=max_extra_work)
    mappings = []
    for cmat, bmat, rhs, inequality in ((hz.Ac, hz.Ab, hz.b, False), (hz.Auc, hz.Aub, hz.ub, True)):
        roots, scales = [], []
        for index in range(cmat.shape[0]):
            cc, cv = _row(cmat, index)
            bc, bv = _row(bmat, index)
            root, scale = encoder.encode(cc, cv, bc, bv, float(rhs[index]), inequality=inequality)
            roots.append(root)
            scales.append(scale)
        mappings.extend((np.array(roots, dtype=np.int64), np.array(scales, dtype=np.int64)))
    ac, ab, b = encoder.matrices(encoder.eq)
    auc, aub, ub = encoder.matrices(encoder.ineq)
    nc = hz.n_cont + len(encoder.def_rows)
    result = SparseHZono(hz.c.copy(), sparse_pad_cols(hz.Gc, nc), hz.Gb.copy(), ac, ab, b,
        auc, aub, ub, frame_id=hz.frame_id, exact=True)
    entries = _sparse_hz_storage_entries(result)
    if entries > max_entries or entries - original_entries > max_extra_entries:
        raise MemoryError('actual complete HZ entry ceiling exceeded')
    report = {'formal_gain': 0, 'default_off': True, 'base_work': base_work,
        'reserved_extra_work': max_extra_work, 'total_work_upper': base_work + max_extra_work,
        'actual_extra_work': encoder.extra_work, 'actual_work_upper': base_work + encoder.extra_work,
        'auxiliary_count': len(encoder.def_rows), 'packed_logical_rows': encoder.packed_rows,
        'scale_relays': encoder.relays, 'original_hz_entries': original_entries, 'actual_hz_entries': entries,
        'extra_hz_entries': entries - original_entries, 'coefficient_min': 2.**-20, 'coefficient_max': 2.**40,
        'solver_executed': False, 'full_suffix_executed': False, 'physical_reduction_claimed': False}
    packed = Packed(hz, frozen, result, *mappings, np.array(encoder.def_rows, dtype=np.int64), report)
    packed.seal = packed.fingerprint()
    packed.validate()
    return packed
