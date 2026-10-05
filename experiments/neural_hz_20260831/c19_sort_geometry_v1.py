"""Complete read-only alias rewrite geometry census, not a generator or optimizer."""

from collections import Counter
import numpy as np
import scipy.sparse as sp

from experiments.neural_hz_20260831.c10_fused_rows_v1 import WorkPool
from experiments.neural_hz_20260831.c18_sparse_rewrite_v1 import sort_cost, merge_cost


def census(ac, auc, old_nc, columns, parents, erased, *, max_work=256_000_000):
    if (not sp.isspmatrix_csr(ac) or not sp.isspmatrix_csr(auc)
            or ac.shape[1] != auc.shape[1] or not ac.has_canonical_format
            or not auc.has_canonical_format or type(old_nc) is not int
            or not 0 <= old_nc <= ac.shape[1]):
        raise ValueError('invalid canonical source geometry')
    nc = ac.shape[1]
    if nc > 64_000_000 or ac.nnz + auc.nnz > 64_000_000:
        raise MemoryError('unchanged source entry ceiling')
    pool = WorkPool(0, 0, max_work=max_work)
    pool.charge('complete_source_geometry', 8 * (ac.nnz + auc.nnz) + 32 * nc)
    columns, parents, erased = (np.asarray(a, dtype=np.int64) for a in (columns, parents, erased))
    if (columns.ndim != 1 or parents.shape != columns.shape or erased.shape != columns.shape
            or np.any(columns < old_nc) or np.any(columns >= nc)
            or np.any(np.diff(columns) <= 0) or np.any(parents < 0)
            or np.any(parents >= columns) or np.isin(parents, columns).any()
            or np.any(erased < 0) or np.any(erased >= ac.shape[0])
            or np.unique(erased).size != erased.size):
        raise ValueError('invalid independent alias/definition lineage')
    mapping = np.arange(nc, dtype=np.int64)
    mapping[columns] = parents
    removed = np.zeros(ac.shape[0], bool)
    removed[erased] = True
    counts, modes, run_hist, shape_hist = Counter(), Counter(), Counter(), Counter()
    old_sort = new_sort = ordered_check = 0
    parent_events = int(np.count_nonzero(parents >= old_nc))
    for kind, matrix in enumerate((ac, auc)):
        for row in range(matrix.shape[0]):
            if kind == 0 and removed[row]:
                continue
            cc = matrix.indices[matrix.indptr[row]:matrix.indptr[row + 1]]
            updated = mapping[cc]
            active = updated != cc
            k = int(np.count_nonzero(active))
            if not k:
                continue
            w = len(cc)
            pool.charge('complete_changed_row_geometry', 32 * w)
            counts['rows'] += 1
            counts['inequality_rows' if kind else 'equality_rows'] += 1
            counts['occurrences'] += k
            counts['width_sum'] += w
            ordered_check += 4 * w
            right = updated[active]
            parent_events += int(np.count_nonzero(right >= old_nc))
            runs = 1 + int(np.count_nonzero(np.diff(right) < 0))
            run_hist[str(runs)] += 1
            whole_ordered = not bool(np.any(np.diff(updated) < 0))
            if whole_ordered:
                mode, charge = 'already_ordered', 0
            else:
                counts['disordered_rows'] += 1
                counts['disordered_width_sum'] += w
                counts['disordered_changed_sum'] += k
                counts['disordered_right_ordered_rows' if runs == 1 else 'disordered_right_unordered_rows'] += 1
                full, lower = sort_cost(w), merge_cost(w, k)
                old_sort += full
                if not 0 < k < w or 8*k + lower >= full:
                    mode, charge = 'full', full
                else:
                    extra = sort_cost(k) if runs > 1 else 0
                    if lower + extra >= full:
                        mode, charge = 'full_after_check', 8*k + full
                    else:
                        mode = 'merge_ordered' if runs == 1 else 'merge_sorted'
                        charge = 8*k + lower + extra
                new_sort += charge
            modes[mode] += 1
            key = f'{mode}:wlog{(w-1).bit_length()}:klog{(k-1).bit_length()}:runs{runs}'
            if key not in shape_hist and len(shape_hist) >= 65_536:
                raise MemoryError('diagnostic histogram ceiling')
            shape_hist[key] += 1
    return {'complete': True, 'aliases': len(columns), 'source_EQ': ac.shape[0],
        'source_INEQ': auc.shape[0], 'source_continuous_nnz': ac.nnz + auc.nnz,
        'counts': dict(counts), 'modes': dict(modes), 'right_run_histogram': dict(run_hist),
        'shape_histogram': dict(shape_hist), 'same_rows_C17_sort_work': old_sort,
        'same_rows_C18_normalization_work': new_sort,
        'same_rows_C18_minus_C17_work': new_sort - old_sort,
        'same_rows_whole_order_check_work': ordered_check,
        'ownership_known_parent_event_work_before_collision_adjustments': 4 * parent_events,
        'diagnostic_work': pool.used, 'diagnostic_work_parts': dict(pool.parts),
        'stable_sort_executed': False, 'candidate_generated': False,
        'numerical_quotient_reexecuted': False, 'formal_gain': 0}
