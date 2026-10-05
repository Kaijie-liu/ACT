"""Exact sparse source/output incidence of complete F(2,3) tile definitions."""
import numpy as np

from experiments.neural_hz_20260831.c85_exact_tile_algebra_v1 import T, R
from experiments.neural_hz_20260831.c85_exact_filter_transform_v1 import transform

COLUMNS = ('y', 'x', 'input_active', 'output_active', 'direct_nnz', 'factored_nnz',
           'v_factors', 'm_factors', 'v_nnz', 'm_nnz', 'output_nnz', 'strict_nnz_win')


def census(kernel, parent_mask, output_mask, padding, *, pool, enabled=False):
    """Return ALL tile costs, not a selected HZ or physical reduction receipt."""
    if not enabled:
        return None
    w, active, needed = np.asarray(kernel), np.asarray(parent_mask), np.asarray(output_mask)
    if (w.dtype != np.float64 or w.ndim != 4 or w.shape[-2:] != (3, 3)
            or active.dtype != np.dtype(bool) or needed.dtype != np.dtype(bool)
            or active.ndim != 3 or needed.ndim != 3):
        raise ValueError('original binary64 kernel and full boolean masks required')
    outputs, channels = w.shape[:2]
    _, height, width = active.shape
    _, oh, ow = needed.shape
    py, px = padding
    if (active.shape[0] != channels or needed.shape[0] != outputs or py < 0 or px < 0
            or oh != height+2*py-2 or ow != width+2*px-2 or min(height, width, oh, ow) <= 0):
        raise ValueError('complete ordinary stride1 padded geometry required')
    pool.charge('c87_complete_original_kernel_precision', 4*int(w.size))
    narrow = w.astype(np.float32)
    if not np.array_equal(narrow.astype(np.float64), w):
        raise ValueError('whole actual kernel is not exactly original binary32')
    proof, values = transform(narrow, pool=pool, enabled=True)
    if values is None or not proof['all_coefficients_exact_binary64']:
        raise ValueError('whole exact transformed kernel rejected')
    u = values['native']
    pool.charge('c87_complete_support_preparation',
                4*(int(w.size)+int(u.size))+64*(int(active.size)+int(needed.size)))
    original_nonzero, transformed_nonzero = w != 0, u != 0
    input_incidence, output_incidence = (T != 0).astype(np.int64), (R != 0).astype(np.int64)
    rows = []
    for y in range(0, oh, 2):
        for x in range(0, ow, 2):
            pool.charge('c87_complete_tile_masks_and_transform_incidence', 1024+512*channels+256*outputs)
            tile = np.zeros((channels, 4, 4), np.int64)
            for i in range(4):
                for j in range(4):
                    iy, ix = y-py+i, x-px+j
                    if 0 <= iy < height and 0 <= ix < width:
                        tile[:, i, j] = active[:, iy, ix]
            want = np.zeros((outputs, 2, 2), np.int64)
            want[:, :min(2, oh-y), :min(2, ow-x)] = needed[:, y:y+2, x:x+2]
            v_first = np.einsum('ai,cij->caj', input_incidence, tile, optimize=False)
            v_parents = np.einsum('caj,bj->cab', v_first, input_incidence, optimize=False)
            m_first = np.einsum('kij,ia->kaj', want, output_incidence, optimize=False)
            m_consumers = np.einsum('kaj,jb->kab', m_first, output_incidence, optimize=False)
            direct = 0
            for i in range(2):
                for j in range(2):
                    ks = np.flatnonzero(want[:, i, j])
                    for a in range(3):
                        for b in range(3):
                            cs = np.flatnonzero(tile[:, i+a, j+b])
                            pool.charge('c87_complete_direct_pairs', 4*int(ks.size*cs.size))
                            direct += int(np.count_nonzero(original_nonzero[ks[:, None], cs[None, :], a, b]))
            vrows = mrows = vnnz = mnnz = outnnz = 0
            for a in range(4):
                for b in range(4):
                    cs, ks = np.flatnonzero(v_parents[:, a, b]), np.flatnonzero(m_consumers[:, a, b])
                    pool.charge('c87_complete_transformed_pairs', 6*int(ks.size*cs.size))
                    connections = transformed_nonzero[ks[:, None], cs[None, :], a, b]
                    used_c, used_k = connections.any(axis=0), connections.any(axis=1)
                    nv, nm = int(used_c.sum()), int(used_k.sum())
                    vrows += nv
                    mrows += nm
                    vnnz += int(v_parents[cs[used_c], a, b].sum())+nv
                    mnnz += int(np.count_nonzero(connections))+nm
                    outnnz += int(m_consumers[ks[used_k], a, b].sum())
            outrows = int(want.sum())
            direct += outrows
            outnnz += outrows
            factored = vnnz+mnnz+outnnz
            rows.append((y, x, int(tile.sum()), outrows, direct, factored,
                         vrows, mrows, vnnz, mnnz, outnnz, int(factored < direct)))
    table = np.asarray(rows, dtype=np.uint64).reshape(-1, len(COLUMNS))
    wins = table[:, -1] != 0
    report = dict(tiles=len(table), nonempty_tiles=int(np.count_nonzero(table[:, 3])),
        winning_tiles=int(wins.sum()), direct_nnz=int(table[:, 4].sum()),
        blanket_factored_nnz=int(table[:, 5].sum()), blanket_new_factors=int(table[:, 6:8].sum()),
        winning_direct_nnz=int(table[wins, 4].sum()), winning_factored_nnz=int(table[wins, 5].sum()),
        winning_new_factors=int(table[wins, 6:8].sum()),
        necessary_only_nnz_saving=int(table[wins, 4].sum())-int(table[wins, 5].sum()),
        all_tiles_including_zero_and_losing_retained=True, original_kernel_proof=proof,
        native_normalized_rows_proved=False, complete_physical_reduction_proved=False)
    return report, table, values
