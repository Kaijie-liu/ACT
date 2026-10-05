"""Owned temporary-row quotient, coupled work budget and tagged lineage."""

import math
import numpy as np

from experiments.neural_hz_20260831.c9_radix_predicate_v1 import RowEncoder
from experiments.neural_hz_20260831.c10_predicate_census_v1 import odd_significand
from experiments.neural_hz_20260831.c10_alias_quotient_v1 import exact_sum, frontier


class WorkPool:
    def __init__(self, whole_base, branch_base, max_work=256_000_000, max_branch=200_000_000):
        if (any(type(v) is not int or v < 0 for v in (whole_base, branch_base, max_work, max_branch))
                or max_work > 256_000_000 or max_branch > 200_000_000):
            raise ValueError('invalid/increased coupled work ceiling')
        self.whole_base, self.branch_base = whole_base, branch_base
        self.capacity = min(max_work - whole_base, max_branch - branch_base)
        self.used, self.parts = 0, {}
        if self.capacity < 0:
            raise MemoryError('base construction exceeds coupled work cap')

    def charge(self, name, amount):
        amount = int(amount)
        if amount < 0 or self.used + amount > self.capacity:
            raise MemoryError(f'coupled work exhausted: {name} requests {amount}; used {self.used}; capacity {self.capacity}')
        self.used += amount
        self.parts[name] = self.parts.get(name, 0) + amount


class FusedRowEncoder(RowEncoder):
    def __init__(self, *args, pool, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = pool

    def charge(self, work):
        self.pool.charge('radix', work)
        super().charge(work)


def window_products(values, ratios, ratio_odd=None, ratio_bits=None):
    """Exact-product predicate specialized to the registered NORMAL window.

    Precomputed right operands avoid the generic routine's second mantissa
    extraction and both-operand subnormal-grid reconstruction.
    """
    values, ratios = np.broadcast_arrays(np.asarray(values, dtype=np.float64), np.asarray(ratios, dtype=np.float64))
    if (not np.isfinite(values).all() or not np.isfinite(ratios).all()
            or np.any(np.abs(values) < 2.**-20) or np.any(np.abs(values) > 2.**40)
            or np.any(np.abs(ratios) < 2.**-60) or np.any(np.abs(ratios) > 1.)):
        raise ValueError('specialized product operand outside proved normal window')
    left_odd, left_bits = odd_significand(values)
    if ratio_odd is None:
        ratio_odd, ratio_bits = odd_significand(ratios)
    bits = left_bits + ratio_bits
    exact = np.asarray(bits <= 53).copy()
    boundary = bits == 54
    exact[boundary] = left_odd[boundary] * ratio_odd[boundary] < np.uint64(1 << 53)
    products = values * ratios
    return exact & (np.abs(products) >= 2.**-20) & (np.abs(products) <= 2.**40), products


def fold_rows(encoder, eq_roots, eq_scales, *, old_nc, old_eq, output_slots, observe=None):
    """Own/rewrite RowEncoder buffers; no completed original HZ is created."""
    pool, nc = encoder.pool, encoder.nc
    main = nc - old_nc
    roots, scales = np.asarray(eq_roots, dtype=np.int64), np.asarray(eq_scales, dtype=np.int64)
    if (main < 0 or roots.shape != (old_eq + main,) or scales.shape != roots.shape
            or not np.array_equal(np.sort(np.r_[roots, encoder.def_rows]), np.arange(len(encoder.eq)))):
        raise ValueError('invalid complete pre-emission MAIN/radix row partition')
    pool.charge('main_metadata_frontier', 32 * main)
    protected = np.zeros(nc, dtype=bool)
    protected[np.asarray(output_slots, dtype=np.int64)] = True
    parents = np.full(main, -1, dtype=np.int64)
    ratios = np.zeros(main)
    eligible = np.zeros(main, dtype=bool)
    defining = roots[old_eq:]
    for index, physical in enumerate(defining):
        cc, cv, bc, bv, rhs = encoder.eq[int(physical)]
        col = old_nc + index
        if protected[col] or len(cc) != 2 or len(bc) or rhs != 0. or cc[-1] != col:
            continue
        parent, pivot = int(cc[0]), float(cv[-1])
        pm, pe = math.frexp(pivot)
        if not 0 <= parent < col or pm != .5:
            raise ValueError('non-topological/nonpositive-dyadic MAIN pivot')
        ratio = math.ldexp(-float(cv[0]), 1 - pe)
        if math.ldexp(ratio, pe - 1) != -float(cv[0]):
            raise ValueError('nonreversible emitted alias ratio')
        if 2.**-60 <= abs(ratio) <= 1.:
            parents[index], ratios[index], eligible[index] = parent, ratio, True
    local = np.flatnonzero(eligible)
    if not local.size:
        raise ValueError('no locally eligible fused alias')
    lookup = np.full(nc + len(encoder.def_rows), -1, dtype=np.int64)
    lookup[old_nc + local] = local
    odd, bits = np.zeros(main, dtype=np.uint64), np.zeros(main, dtype=np.int32)
    odd[local], bits[local] = odd_significand(ratios[local])
    own_alias = np.zeros(len(encoder.eq), dtype=bool)
    own_alias[defining[local]] = True
    # Hit tuples are bounded temporary incidence data, discarded before return.
    hits = ([], [])
    examined = products_checked = 0
    for kind, rows in enumerate((encoder.eq, encoder.ineq)):
        for row, (cc, cv, bc, bv, rhs) in enumerate(rows):
            pool.charge('continuous_incidence_scan', len(cc))
            examined += len(cc)
            columns = cc[:-1] if kind == 0 and own_alias[row] else cc
            ids = lookup[columns]
            positions = np.flatnonzero(ids >= 0)
            if not positions.size:
                continue
            ids = ids[positions]
            pool.charge('normal_exact_products', 32 * positions.size)
            good, products = window_products(cv[positions], ratios[ids], odd[ids], bits[ids])
            np.logical_and.at(eligible, ids, good)
            hits[kind].append((row, positions, ids, products))
            products_checked += positions.size
    columns = np.arange(old_nc, nc, dtype=np.int64)
    chosen = frontier({'column': columns, 'parent': parents,
        'all_products_exact': eligible, 'products_window_safe': eligible}, nc)
    if not chosen.size:
        raise ValueError('no exact independent fused alias frontier')
    selected = np.zeros(main, dtype=bool)
    selected[chosen] = True
    erased = np.zeros(len(encoder.eq), dtype=bool)
    erased[defining[chosen]] = True
    old_entries = encoder.entries
    old_nnz = sum(len(r[0]) + len(r[2]) for rows in (encoder.eq, encoder.ineq) for r in rows)
    rewritten = collisions = occurrences = 0
    if observe:
        observe('fused_frontier', {'local_aliases': int(local.size), 'eligible_aliases': int(eligible.sum()),
            'selected_aliases': int(chosen.size), 'products_checked': int(products_checked),
            'coupled_extra_used': pool.used, 'coupled_extra_capacity': pool.capacity})
    for kind, rows in enumerate((encoder.eq, encoder.ineq)):
        for row, positions, ids, products in hits[kind]:
            active = selected[ids]
            if (kind == 0 and erased[row]) or not active.any():
                continue
            cc, cv, bc, bv, rhs = rows[row]
            width = len(cc)
            sort_work = 4 * width * max(1, (width - 1).bit_length())
            pool.charge('rewrite_sort', sort_work)
            # These are owned unpublished buffers, never original source HZs.
            cc[positions[active]] = parents[ids[active]]
            cv[positions[active]] = products[active]
            order = np.argsort(cc, kind='stable')
            cc, cv = cc[order], cv[order]
            starts = np.r_[0, np.flatnonzero(np.diff(cc)) + 1, width]
            for begin, end in zip(starts[:-1], starts[1:]):
                if end - begin > 1:
                    pool.charge('collision_exact_sum', 32 * (end - begin))
                    cv[begin] = exact_sum(cv[begin:end])
                    cv[begin + 1:end] = 0.
                    collisions += 1
            nonzero = cv != 0.
            rows[row] = (cc[nonzero], cv[nonzero], bc, bv, rhs)
            rewritten += 1
            occurrences += int(active.sum())
    row_map = np.cumsum(~erased, dtype=np.int64) - 1
    tagged_roots = row_map[roots].copy()
    tagged_scales = scales.copy()
    tagged_roots[old_eq + chosen] = -(parents[chosen] + 1)
    tagged_scales.view(np.float64)[old_eq + chosen] = ratios[chosen]
    encoder.def_rows = [int(row_map[r]) for r in encoder.def_rows]
    encoder.eq = [value for index, value in enumerate(encoder.eq) if not erased[index]]
    new_nnz = sum(len(r[0]) + len(r[2]) for rows in (encoder.eq, encoder.ineq) for r in rows)
    encoder.entries = new_nnz + len(encoder.eq) + len(encoder.ineq)
    if new_nnz > old_nnz - 2 * chosen.size or encoder.entries >= old_entries:
        raise ValueError('fused emission does not strictly reduce all predicate entries')
    report = {'schema': 'tagged_alias_lineage_v1', 'local_aliases': int(local.size),
        'eligible_aliases': int(eligible.sum()), 'selected_aliases': int(chosen.size),
        'owned_continuous_coefficients_inspected': examined, 'alias_products_checked': int(products_checked),
        'rewritten_rows': rewritten, 'rewritten_occurrences': occurrences, 'collision_groups': collisions,
        'old_predicate_nnz': old_nnz, 'new_predicate_nnz': new_nnz,
        'old_predicate_entries': old_entries, 'new_predicate_entries': encoder.entries,
        'persistent_extra_reconstruction_arrays': 0, 'coupled_extra_work': pool.used,
        'coupled_extra_capacity': pool.capacity, 'work_parts': dict(pool.parts),
        'original_completed_hz_constructed': False, 'formal_gain': 0}
    return tagged_roots, tagged_scales, report


def aliases(candidate):
    """Decode tagged maps to temporary arrays; never cache unaccounted payload."""
    roots, scales = candidate.eq_roots, candidate.eq_scales
    if roots.dtype != np.dtype(np.int64) or scales.dtype != np.dtype(np.int64) or roots.shape != scales.shape:
        raise ValueError('invalid tagged lineage dtype/shape')
    tagged = np.flatnonzero(roots < 0)
    if np.any(tagged < candidate.old_n_eq):
        raise ValueError('original predicate cannot carry a MAIN alias tag')
    cols = candidate.old_n_cont + tagged - candidate.old_n_eq
    ps, rs = -(roots[tagged] + 1), scales.view(np.float64)[tagged].copy()
    if (np.any(cols >= candidate.logical_n_cont) or np.any(ps < 0) or np.any(ps >= cols)
            or np.isin(ps, cols).any() or not np.isfinite(rs).all()
            or np.any(np.abs(rs) < 2.**-60) or np.any(np.abs(rs) > 1.)):
        raise ValueError('invalid independent tagged reconstruction')
    return cols, ps, rs, tagged
