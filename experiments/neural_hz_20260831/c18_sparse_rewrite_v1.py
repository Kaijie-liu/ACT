"""Exact updated-column normalization; no predicate arithmetic or solver change."""

import numpy as np


def sort_cost(width):
    return 4 * width * max(1, (width - 1).bit_length())


def merge_cost(width, changed):
    # 6*w mask initialization/scans + 4*w unchanged gathers/scatters,
    # plus 10*k changed values, point masks, rank arithmetic and validation.
    # Two units per binary-search comparison; ceil(log2(u+1)) = bit_length(u).
    return 10 * width + 10 * changed + 2 * changed * (width - changed).bit_length()


def sort_rewritten_row(columns, values, positions, pool):
    """Canonicalize an owned row whose untouched subsequence was canonical.

    Caller has already paid/performed the whole-row order check and substitutions.
    positions is the sorted unique active hit list; no source HZ is modified.
    Equal-column terms may change order, but the caller's exact integer-grid
    collision sum is order-independent. No partial output is returned on failure.
    """
    if (type(columns) is not np.ndarray or type(values) is not np.ndarray
            or type(positions) is not np.ndarray or columns.ndim != 1
            or values.shape != columns.shape or positions.ndim != 1
            or columns.dtype.kind != 'i' or positions.dtype.kind != 'i'
            or values.dtype != np.dtype(np.float64)):
        raise ValueError('invalid owned row/position arrays')
    w, k = len(columns), len(positions)
    if k > w or (k and (positions[0] < 0 or positions[-1] >= w)):
        raise ValueError('updated positions outside owned row')
    full = sort_cost(w)
    lower = merge_cost(w, k)
    # Do not inspect V if even its best-case total tariff cannot beat full sort.
    # This is a structural choice BEFORE reserving/attempting the operation.
    if not 0 < k < w or 8 * k + lower >= full:
        pool.charge('rewrite_sort', full)
        order = np.argsort(columns, kind='stable')
        return columns[order], values[order], 'full'

    pool.charge('rewrite_changed_order', 8 * k)
    if np.any(np.diff(positions) <= 0):
        raise ValueError('changed positions are not sorted unique')
    right_columns = columns[positions]
    right_disordered = bool(np.any(np.diff(right_columns) < 0))
    right_sort = sort_cost(k) if right_disordered else 0
    # The 8*k inspection is sunk work in BOTH alternatives; never catch a cap.
    if lower + right_sort >= full:
        pool.charge('rewrite_sort', full)
        order = np.argsort(columns, kind='stable')
        return columns[order], values[order], 'full_after_check'

    # Reserve ALL merge allocations/searches/scatters plus the k-sort first.
    # A cap failure here cannot fall back to full sort or publish a row.
    pool.charge('rewrite_stream_merge', lower + right_sort)
    mask = np.ones(w, dtype=bool)
    mask[positions] = False
    left_columns, left_values = columns[mask], values[mask]
    right_values = values[positions]
    if right_disordered:
        order = np.argsort(right_columns, kind='stable')
        right_columns, right_values = right_columns[order], right_values[order]
    places = np.searchsorted(left_columns, right_columns, side='right') + np.arange(k)
    if (places[0] < 0 or places[-1] >= w or np.any(np.diff(places) <= 0)):
        raise ValueError('two-stream placement is not a complete injection')
    mask.fill(True)
    mask[places] = False
    out_columns, out_values = np.empty_like(columns), np.empty_like(values)
    out_columns[places], out_values[places] = right_columns, right_values
    out_columns[mask], out_values[mask] = left_columns, left_values
    return out_columns, out_values, 'merge_sorted' if right_disordered else 'merge_ordered'
