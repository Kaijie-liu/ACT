"""Default-off exact dyadic row arithmetic; original equations stay independent.

No change to a journal, HZ, source maps, binary factors or admission schema.
The old Fraction inverse remains an independent test oracle.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import numpy as np

from experiments.neural_hz_20260831.c68_local_splice_v1 import SCHEMA
from experiments.neural_hz_20260831.c62_local_equations_v1 import SCHEMA as LOCAL_SCHEMA, decode as decode_local
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import decode as decode_splice
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import fraction


def words(bits):
    return max(1, (int(bits) + 63) // 64)


class Point:
    """Exact point plus conservative dyadic metadata; never specializes zero."""
    def __init__(self, values, *, pool):
        pool.charge('c81_point_table_preparation', 8 * len(values))
        self.values = values
        self.numerators = [None] * len(values)
        self.exponents = [0] * len(values)
        self.bits = np.empty(len(values), np.int64)
        self.low = np.empty(len(values), np.int64)
        self.dyadic = np.empty(len(values), bool)
        for i, v in enumerate(values):
            self._set(i, v)

    def _set(self, i, value):
        n, d = value.numerator, value.denominator
        good = not (d & (d - 1))
        e = 1 - d.bit_length() if good else 0
        low = min(-52, e)
        self.numerators[i], self.exponents[i] = n, e
        self.dyadic[i] = good
        self.low[i] = low
        self.bits[i] = max(53, abs(n).bit_length() + e - low)

    def update(self, i, value, *, pool):
        pool.charge('c81_point_table_update', 8)
        self.values[i] = value
        self._set(i, value)


@dataclass
class Row:
    columns: np.ndarray
    values: np.ndarray
    mantissas: np.ndarray
    exponents: np.ndarray
    upper: int = 0
    low: int = 0
    bits: int = 0
    dyadic: bool = True

    @classmethod
    def compile(cls, columns, values, size, *, pool):
        n = len(values)
        pool.charge('c81_coefficient_bit_decode', 16 * n + 128)
        if (values.dtype != np.dtype(np.float64) or values.ndim != 1
                or columns.ndim != 1 or len(columns) != n
                or columns.dtype.kind not in 'iu'
                or (n and (int(columns.min()) < 0 or int(columns.max()) >= size))):
            raise ValueError('complete binary64 row and in-frame integer columns required')
        raw = values.view(np.uint64)
        field = ((raw >> 52) & 2047).astype(np.int16)
        if np.any(field == 2047):
            raise ValueError('nonfinite exact row coefficient')
        mantissa = ((raw & ((1 << 52) - 1)) | ((field != 0).astype(np.uint64) << 52)).astype(np.int64)
        mantissa[raw >> 63 != 0] *= -1
        exponent = np.where(field == 0, -1074, field - 1075).astype(np.int16)
        exponent[mantissa == 0] = 0
        return cls(columns, values, mantissa, exponent)

    def bound(self, point, *, pool):
        n = len(self.columns)
        pool.charge('c81_complete_row_bound', 4 * n + 128)
        self.dyadic = bool(np.all(point.dyadic[self.columns]))
        if not self.dyadic:
            self.upper = 64 * n
            return self.upper
        if not n:
            self.upper, self.low, self.bits = 192, -52, 53
            return self.upper
        bits = point.bits[self.columns]
        low = point.low[self.columns]
        logn = (n - 1).bit_length()
        running = np.maximum.accumulate(bits)
        num_words = (bits + 63) // 64
        bin_words = (running + 53 + logn + 63) // 64
        core = 8 * n + 2 * int(np.sum(num_words + bin_words))
        # Bounds describe integers at a common low exponent, not rounded values.
        self.low = int(np.min(self.exponents.astype(np.int64) + low))
        high = int(np.max(self.exponents))  # |x| <= 1 implies canonical q <= 0.
        span = high - self.low
        bins = min(n, span + 1)
        self.bits = int(bits.max()) + 53 + logn + span
        align_words = words(self.bits + max(1, bins).bit_length())
        self.upper = core + bins * (16 + 4 * align_words) + 128 + 64 * align_words
        return self.upper

    def execute(self, point, *, pool):
        pool.charge('c81_exact_binned_dot' if self.dyadic else 'c81_general_fraction_dot', self.upper)
        if not self.dyadic:
            return sum((F(float(v)) * point.values[int(k)]
                        for k, v in zip(self.columns, self.values)), F(0))
        bins = {}
        for k, m, e in zip(self.columns, self.mantissas, self.exponents):
            k = int(k)
            key = int(e) + point.exponents[k]
            bins[key] = bins.get(key, 0) + int(m) * point.numerators[k]
        if not bins:
            return F(0)
        base = min(bins)
        total = sum(value << (e - base) for e, value in bins.items())
        return F(total << base) if base >= 0 else F(total, 1 << -base)


def dot(columns, values, continuous, *, pool, enabled=False):
    """Standalone exact row, useful for independent nonzero oracle checks."""
    if not enabled:
        return None
    values_point = [F(v) for v in continuous]
    if any(abs(v) > 1 for v in values_point):
        raise ValueError('point outside original latent box')
    pool.charge('c68_full_frame_fraction_and_box', 16 * len(values_point))
    point = Point(values_point, pool=pool)
    row = Row.compile(columns, values, len(values_point), pool=pool)
    row.bound(point, pool=pool)
    return row.execute(point, pool=pool)


def matrix_row(matrix, i):
    a, b = map(int, matrix.indptr[i:i + 2])
    return matrix.indices[a:b], matrix.data[a:b]


def reconstruct(c, hz, journal, plans, continuous, *, pool, enabled=False, observe=None):
    """Preflight both complete row systems, restore, then check original rows.

The caller authenticates source/native state. This function makes no native
admission or feasibility claim, and never changes that state's schema/type.
"""
    if not enabled:
        return None
    if (journal.schema != SCHEMA or journal.source_schema != LOCAL_SCHEMA
            or len(continuous) != hz.n_cont or hz.n_cont < journal.source_n_cont
            or len(plans) != len(journal.columns)
            or c.old_n_cont != journal.old_n_cont or c.old_n_eq != journal.old_n_eq
            or c.eq_roots is not journal.eq_roots or c.eq_scales is not journal.eq_scales):
        raise ValueError('explicit shared local/journal source and complete frame required')
    pool.charge('c68_full_frame_fraction_and_box', 16 * hz.n_cont)
    full = [F(v) for v in continuous]
    if any(abs(v) > 1 for v in full):
        raise ValueError('point outside original latent box')
    point = Point(full, pool=pool)
    main = len(journal.eq_roots) - journal.old_n_eq
    pool.charge('c81_removed_local_mask', 4 * hz.n_cont)
    removed = np.zeros(hz.n_cont, bool)
    removed[journal.old_n_cont:journal.old_n_cont + main] = journal.eq_roots[journal.old_n_eq:] < 0
    local_count = int(removed.sum())
    units, originals = [], []
    remaining = 128 * len(plans) + 64 * len(plans) + 16 * main + 256 * local_count
    remaining += 8 * (len(plans) + local_count)
    for col, tag, offset in zip(journal.columns, journal.tags, journal.offsets):
        col = int(col)
        kind, _, consumer, inequality, pivot, sign = decode_splice(tag)
        if kind != 'splice':
            raise ValueError('journal contains a non-splice descriptor')
        target = consumer if inequality else journal.eq_row(consumer, pool=pool)
        matrix = hz.Auc if inequality else hz.Ac
        if target is None or not 0 <= target < matrix.shape[0]:
            raise ValueError('missing surviving inverse row')
        cc, cv = matrix_row(matrix, target)
        cut = int(np.searchsorted(cc, col))
        row = Row.compile(cc[:cut], cv[:cut], hz.n_cont, pool=pool)
        remaining += row.bound(point, pool=pool)
        units.append((col, row, F(float(offset)), F(pivot), sign))
        # A bound-only update; actual numerators and continuous input are untouched.
        if row.dyadic:
            off = F(float(offset))
            off_e = 1 - off.denominator.bit_length()
            base = min(row.low, off_e)
            width = max(row.bits + row.low - base, abs(off.numerator).bit_length() + off_e - base) + 1
            pv = F(pivot)
            pe = pv.numerator.bit_length() - pv.denominator.bit_length()
            e = base - pe
            point.low[col] = min(-52, e)
            # Accepted inverse must be inside its original box; otherwise abort.
            width = min(width, max(1, 1 - e))
            point.bits[col] = max(53, width + e - int(point.low[col]))
            point.dyadic[col] = True
        else:
            point.dyadic[col] = False
    for p in plans:
        cc, cv = matrix_row(c.hz.Ac, p.definition)
        row = Row.compile(cc, cv, hz.n_cont, pool=pool)
        # Physical source equations no longer contain eliminated local children.
        if np.any(removed[cc]):
            raise ValueError('source producer still references a removed local child')
        remaining += row.bound(point, pool=pool)
        originals.append((row, F(p.offset)))
    bound = dict(used_before_inverse=pool.used, remaining_upper=int(remaining),
                 complete_upper=pool.used + int(remaining), cap=pool.cap,
                 unit_rows=len(units), producer_rows=len(originals),
                 local_equations=local_count,
                 surviving_terms=sum(len(r.columns) for _, r, *_ in units),
                 producer_terms=sum(len(r.columns) for r, _ in originals))
    if observe:
        observe(dict(event='complete_inverse_work_preflight', **bound))
    if bound['complete_upper'] > pool.cap:
        raise MemoryError(f'complete inverse bound exceeds unchanged whole cap: {bound}')
    # Bound metadata may be overwritten on updates; row charges are already frozen.
    for col, row, offset, pivot, sign in units:
        pool.charge('c68_exact_splice_inverse', 128)
        value = (offset - sign * row.execute(point, pool=pool)) / pivot
        if abs(value) > 1:
            raise ValueError('unit inverse outside proved MAIN box')
        point.update(col, value, pool=pool)
    pool.charge('c68_all_source_local_tags', 8 * main)
    scales = journal.eq_scales.view(np.float64)
    for at in range(journal.old_n_eq, len(journal.eq_roots)):
        if journal.eq_roots[at] >= 0:
            continue
        pool.charge('c68_exact_local_inverse', 128)
        col = journal.old_n_cont + at - journal.old_n_eq
        parent, ratio = decode_local(journal.eq_roots[at], scales[at], column=col,
                                    n_cont=journal.source_n_cont, schema=journal.source_schema)
        value = fraction(ratio) * full[parent]
        if abs(value) > 1:
            raise ValueError('local inverse outside original box')
        point.update(col, value, pool=pool)
    if full[:c.old_n_cont] != [F(v) for v in continuous[:c.old_n_cont]]:
        raise ValueError('original coordinates changed')
    for row, offset in originals:
        pool.charge('c70_independent_every_unit_inverse_equation_header', 64)
        if row.execute(point, pool=pool) != offset:
            raise ValueError('original complete producer equation not restored')
    count = 0
    for at in range(c.old_n_eq, len(c.eq_roots)):
        pool.charge('c70_independent_local_equation_header', 8)
        if c.eq_roots[at] >= 0:
            continue
        pool.charge('c70_independent_local_equation', 128)
        col = c.old_n_cont + at - c.old_n_eq
        parent, ratio = decode_local(c.eq_roots[at], c.eq_scales.view(np.float64)[at],
                                    column=col, n_cont=c.hz.n_cont, schema=LOCAL_SCHEMA)
        if full[col] != fraction(ratio) * full[parent]:
            raise ValueError('original local inverse failed')
        count += 1
    if pool.used != bound['complete_upper'] or count != local_count:
        raise ValueError('complete inverse work/equation inventory differs from preflight')
    report = dict(unit_equations=len(plans), local_equations=count,
                  original_coordinates=c.old_n_cont, all_equations_exact=True,
                  feasibility_or_concrete_witness_claim=False, formal_gain=0,
                  binned_work_bound=bound)
    return full, report


def verify_inverse(c, new, journal, plans, *, pool, continuous=None, observe=None):
    point = [F(0)] * new.n_cont if continuous is None else continuous
    _, report = reconstruct(c, new, journal, plans, point, pool=pool,
                            enabled=True, observe=observe)
    return report
