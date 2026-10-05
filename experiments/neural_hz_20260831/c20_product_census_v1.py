"""Complete read-only original local-alias product classes; no product evaluation."""

from collections import Counter
import math
import numpy as np
import scipy.sparse as sp

from experiments.neural_hz_20260831.c10_fused_rows_v1 import WorkPool


def census(hz, *, old_nc, logical_nc, old_eq, eq_roots, def_rows, output_slots,
           max_work=256_000_000):
    for value in (old_nc, logical_nc, old_eq):
        if type(value) is not int or value < 0:
            raise ValueError('invalid original frame dimensions')
    if not old_nc <= logical_nc <= hz.n_cont:
        raise ValueError('invalid MAIN/global frame')
    matrices = [hz.Ac, hz.Ab, hz.Auc, hz.Aub]
    if any(not sp.isspmatrix_csr(m) or not m.has_canonical_format for m in matrices):
        raise ValueError('canonical original predicate required')
    nnz, main = sum(m.nnz for m in matrices), logical_nc - old_nc
    if nnz > 64_000_000 or hz.n_cont > 64_000_000:
        raise MemoryError('unchanged entry ceiling')
    pool = WorkPool(0, 0, max_work=max_work)
    pool.charge('complete_local_alias_geometry', 8 * nnz + 64 * main)
    roots, radix = np.asarray(eq_roots), np.asarray(def_rows)
    output = np.asarray(output_slots)
    if (roots.dtype != np.dtype(np.int64) or roots.shape != (old_eq + main,)
            or radix.dtype != np.dtype(np.int64) or radix.ndim != 1
            or logical_nc + len(radix) != hz.n_cont
            or output.ndim != 1 or output.dtype.kind != 'i'
            or np.any(output < 0) or np.any(output >= logical_nc)):
        raise ValueError('incomplete original row/frame/root partition')
    partition = np.r_[roots, radix]
    if (len(partition) != hz.n_eq or np.any(partition < 0) or np.any(partition >= hz.n_eq)
            or np.any(np.bincount(partition, minlength=hz.n_eq) != 1)):
        raise ValueError('incomplete original row/frame/root partition')
    if (any(not np.isfinite(m.data).all() or np.any(m.data == 0.) for m in matrices)
            or not np.isfinite(hz.b).all() or not np.isfinite(hz.ub).all()):
        raise ValueError('nonfinite or zero-filled original predicates')
    protected = np.zeros(logical_nc, bool)
    protected[output] = True
    local = np.zeros(main, bool)
    right_power_two = np.zeros(main, bool)
    ratios = np.zeros(main)
    exponent = np.zeros(main, np.int32)
    for i, physical in enumerate(roots[old_eq:]):
        col = old_nc + i
        a, b = hz.Ac.indptr[physical:physical+2]
        if (protected[col] or b-a != 2 or hz.Ab.indptr[physical+1] != hz.Ab.indptr[physical]
                or hz.b[physical] != 0. or hz.Ac.indices[b-1] != col):
            continue
        parent, pivot = int(hz.Ac.indices[a]), float(hz.Ac.data[b-1])
        pm, pe = math.frexp(pivot)
        if not 0 <= parent < col or pm != .5:
            raise ValueError('non-topological/nonpositive-dyadic MAIN pivot')
        ratio = math.ldexp(-float(hz.Ac.data[a]), 1-pe)
        if math.ldexp(ratio, pe-1) != -float(hz.Ac.data[a]):
            raise ValueError('nonreversible original alias ratio')
        if 2.**-60 <= abs(ratio) <= 1.:
            local[i], ratios[i] = True, ratio
            rm, re = math.frexp(abs(ratio))
            right_power_two[i] = rm == .5
            exponent[i] = re - 1
    indices = np.flatnonzero(local)
    if not len(indices): raise ValueError('no local original alias')
    lookup = np.full(hz.n_cont, -1, np.int64)
    lookup[old_nc + indices] = indices
    own = np.zeros(hz.n_eq, bool)
    own[roots[old_eq + indices]] = True
    counts, row_classes, size_hist = Counter(), Counter(), Counter()
    exp_hist = np.zeros(61, np.int64)
    for kind, matrix in enumerate((hz.Ac, hz.Auc)):
        for row in range(matrix.shape[0]):
            start, stop = matrix.indptr[row:row+2]
            if kind == 0 and own[row]: stop -= 1
            cc = matrix.indices[start:stop]
            ids = lookup[cc]
            positions = np.flatnonzero(ids >= 0)
            n = len(positions)
            if not n: continue
            pool.charge('complete_product_classification', 24 * n)
            ids = ids[positions]
            dyadic = right_power_two[ids]
            d = int(np.count_nonzero(dyadic))
            left = matrix.data[start:stop][positions]
            left_dyadic = np.frexp(np.abs(left))[0] == .5
            counts['hits'] += n
            counts['right_power_two_hits'] += d
            counts['right_general_hits'] += n-d
            counts['either_power_two_hits'] += int(np.count_nonzero(dyadic | left_dyadic))
            counts['general_right_power_two_left_hits'] += int(np.count_nonzero(~dyadic & left_dyadic))
            counts['negative_right_hits'] += int(np.count_nonzero(ratios[ids] < 0.))
            counts['INEQ_hits' if kind else 'EQ_hits'] += n
            mode = 'all_dyadic' if d == n else 'all_general' if d == 0 else 'mixed'
            row_classes[mode] += 1
            size_hist[f'{mode}:ceil_log2_hits{(n-1).bit_length()}'] += 1
            np.add.at(exp_hist, exponent[ids[dyadic]] + 60, 1)
    return {'complete': True, 'local_aliases': len(indices),
        'local_right_power_two_aliases': int(np.count_nonzero(right_power_two & local)),
        'counts': dict(counts), 'row_classes': dict(row_classes),
        'right_power_two_exponent_histogram': {str(i-60): int(n) for i, n in enumerate(exp_hist) if n},
        'hit_row_histogram': dict(size_hist),
        'diagnostic_work': pool.used, 'diagnostic_work_parts': dict(pool.parts),
        'coefficient_products_evaluated': False, 'candidate_generated': False,
        'selected_frontier_only': False, 'formal_gain': 0}
