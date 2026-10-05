"""Independent ordinary masked tile edge oracle, no model or iid selectors."""
from fractions import Fraction as F
import numpy as np
import pytest

from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c85_exact_tile_algebra_v1 import H, T, R
from experiments.neural_hz_20260831.c87_masked_tile_cost_v1 import census


def independent(kernel, active, needed, padding):
    outputs, channels = kernel.shape[:2]
    height, width = active.shape[1:]
    oh, ow = needed.shape[1:]
    input_map, output_map = np.kron(T, T), np.kron(R, R)
    transformed = np.empty((outputs, channels, 16), object)
    for k, c in np.ndindex(outputs, channels):
        original = np.array([[F(float(v)) for v in row] for row in kernel[k, c]], object)
        transformed[k, c] = (H.astype(object)@original@H.T.astype(object)/4).reshape(-1)
    answer = []
    for y in range(0, oh, 2):
        for x in range(0, ow, 2):
            live = {}
            for c, i, j in np.ndindex(channels, 4, 4):
                sy, sx = y-padding[0]+i, x-padding[1]+j
                if 0 <= sy < height and 0 <= sx < width and active[c, sy, sx]:
                    live[c, 4*i+j] = (c*height+sy)*width+sx
            outputs_needed = [(k, i, j) for k, i, j in np.ndindex(outputs, 2, 2)
                              if y+i < oh and x+j < ow and needed[k, y+i, x+j]]
            original_edges = sum(1 for k, i, j in outputs_needed for c in range(channels)
                                 for a in range(3) for b in range(3)
                                 if (c, 4*(i+a)+j+b) in live and kernel[k, c, a, b] != 0)
            v = {(c, t): {col: int(input_map[t, p]) for (cc, p), col in live.items()
                         if cc == c and input_map[t, p]} for c in range(channels) for t in range(16)}
            wanted_m = {(k, t) for k, i, j in outputs_needed for t in range(16) if output_map[2*i+j, t]}
            m = {(k, t): {(c, t): transformed[k, c, t] for c in range(channels)
                          if v[c, t] and transformed[k, c, t]} for k, t in wanted_m}
            m = {key: terms for key, terms in m.items() if terms}
            used_v = {key for terms in m.values() for key in terms}
            vnnz = sum(len(v[key])+1 for key in used_v)
            mnnz = sum(len(terms)+1 for terms in m.values())
            outnnz = len(outputs_needed)+sum(1 for k, i, j in outputs_needed for t in range(16)
                                             if (k, t) in m and output_map[2*i+j, t])
            direct, factored = original_edges+len(outputs_needed), vnnz+mnnz+outnnz
            answer.append((y, x, len(live), len(outputs_needed), direct, factored,
                           len(used_v), len(m), vnnz, mnnz, outnnz, int(factored < direct)))
    return np.array(answer, np.uint64)


@pytest.mark.parametrize('channels,outputs', [(1, 2), (3, 4), (8, 8)])
@pytest.mark.parametrize('mode', ['dense', 'sparse', 'striped', 'output_mask', 'boundary_odd'])
def test_every_masked_tile_against_independent_actual_incidence(channels, outputs, mode):
    height, padding = (3, (1, 1)) if mode == 'boundary_odd' else (4, (0, 0))
    oh = height+2*padding[0]-2
    active = np.ones((channels, height, height), bool)
    needed = np.ones((outputs, oh, oh), bool)
    if mode == 'sparse':
        active = np.arange(active.size).reshape(active.shape) % 5 == 0
    if mode == 'striped':
        active[:, :, 1::2] = False
    if mode == 'output_mask':
        needed = np.arange(needed.size).reshape(needed.shape) % 3 == 0
    kernel = ((np.arange(channels*outputs*9) % 31)-15).reshape(outputs, channels, 3, 3).astype(np.float64)/32
    report, table, _ = census(kernel, active, needed, padding, pool=WorkPool(256_000_000), enabled=True)
    assert np.array_equal(table, independent(kernel, active, needed, padding))
    assert report['winning_tiles'] == int(table[:, -1].sum())
    assert not report['native_normalized_rows_proved'] and not report['complete_physical_reduction_proved']


def test_zero_demand_counts_no_transformed_rows():
    kernel, active, needed = np.ones((2, 1, 3, 3)), np.ones((1, 4, 4), bool), np.zeros((2, 2, 2), bool)
    report, table, _ = census(kernel, active, needed, (0, 0), pool=WorkPool(256_000_000), enabled=True)
    assert report['blanket_new_factors'] == report['direct_nnz'] == report['blanket_factored_nnz'] == 0


def test_whole_actual_kernel_precision_cannot_accept_rounded_prefix():
    kernel = np.ones((2, 1, 3, 3))
    kernel.flat[-1] = np.nextafter(1., 2.)
    with pytest.raises(ValueError, match='whole actual kernel'):
        census(kernel, np.ones((1, 4, 4), bool), np.ones((2, 2, 2), bool), (0, 0),
               pool=WorkPool(256_000_000), enabled=True)


def test_default_off_and_prepaid_rejection():
    kernel, active, needed = np.ones((2, 1, 3, 3)), np.ones((1, 4, 4), bool), np.ones((2, 2, 2), bool)
    pool = WorkPool(0)
    assert census(kernel, active, needed, (0, 0), pool=pool) is None and pool.used == 0
    with pytest.raises(MemoryError):
        census(kernel, active, needed, (0, 0), pool=pool, enabled=True)
