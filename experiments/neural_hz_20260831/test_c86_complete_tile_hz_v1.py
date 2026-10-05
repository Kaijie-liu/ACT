"""Complete ordinary exact HZ projection/extension, not a production trial."""
from fractions import Fraction as F
import numpy as np
import pytest

from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c86_complete_tile_hz_v1 import (
    build_tile, native_row, extend_actual, project_outputs)


def fixture(channels=1, mode='ordinary'):
    outputs = 2
    ids = np.arange(channels*16).reshape(channels, 4, 4)
    centers, scales = np.zeros(ids.shape, object), np.ones(ids.shape, object)
    if mode == 'shared':
        ids[:, 1, :] = ids[:, 0, :]
    if mode == 'padding':
        ids[:, 0, :] = -1
        scales[:, 0, :] = 0
    if mode == 'offset_radius':
        centers = np.array([F(i % 3-1, 8) for i in range(ids.size)], object).reshape(ids.shape)
        scales = np.array([F(i % 5+1, 8) for i in range(ids.size)], object).reshape(ids.shape)
    out_ids = np.arange(channels*16, channels*16+8).reshape(outputs, 2, 2)
    out_centers = np.full(out_ids.shape, F(1, 8), object)
    out_scales = np.full(out_ids.shape, F(64*channels), object)
    weights = ((np.arange(outputs*channels*9, dtype=np.float32) % 29)-14).reshape(outputs, channels, 3, 3)/64
    bias = [F(1, 8), F(-1, 4)]
    source = dict(n_cont=channels*16+8, frame=object(), binary_ids=(101, 102),
                  equalities=(((0, F(1)), ('binary:101', F(-1, 2))),),
                  inequalities=(((1, F(1)), ('binary:102', F(1, 4)), ('rhs', F(1))),))
    return source, ids, centers, scales, weights, bias, out_ids, out_centers, out_scales


def spatial_polynomials(args):
    """No transform matrices, new row recipes or production expression oracle."""
    source, ids, centers, scales, weights, bias, out_ids, out_centers, out_scales = args
    result = []
    for k, i, j in np.ndindex(out_ids.shape):
        poly = {int(out_ids[k, i, j]): F(out_scales[k, i, j]),
                -1: F(out_centers[k, i, j])-F(bias[k])}
        for c in range(weights.shape[1]):
            for a in range(3):
                for b in range(3):
                    col = int(ids[c, i+a, j+b])
                    if col < 0:
                        continue
                    weight = F(float(weights[k, c, a, b]))
                    poly[-1] -= weight*F(centers[c, i+a, j+b])
                    poly[col] = poly.get(col, F(0))-weight*F(scales[c, i+a, j+b])
        result.append({c: v for c, v in poly.items() if v})
    return result


@pytest.mark.parametrize('channels', [1, 3, 8])
@pytest.mark.parametrize('mode', ['ordinary', 'shared', 'padding', 'offset_radius'])
def test_complete_all_coordinate_projection_and_actual_extension(channels, mode, record_property):
    args = fixture(channels, mode)
    state = build_tile(*args, pool=WorkPool(256_000_000), enabled=True)
    direct = spatial_polynomials(args)
    assert project_outputs(state['auxiliary_rows'], state['output_rows'], state['base_n_cont']) == direct
    assert state['source'] is args[0] and state['binary_ids'] is args[0]['binary_ids']
    assert state['new_factors'] == 16*(channels+2)
    assert state['total_new_equation_nnz'] == sum(len(r['coefficients']) for r in
                                               [*state['auxiliary_rows'], *state['output_rows']])
    old = [F(i % 5-2, 4) for i in range(args[0]['n_cont'])]
    old[0], old[1] = F(1, 2), F(0)  # Feasible retained binary101=1/binary102=-1.
    for col, poly in zip(args[6].flat, direct, strict=True):
        old[col] = -(poly.get(-1, F(0))+sum((v*old[c] for c, v in poly.items()
                    if c not in (-1, int(col))), F(0)))/poly[int(col)]
    assert all(abs(v) <= 1 for v in old)
    point = extend_actual(state['auxiliary_rows'], old)
    assert point[:len(old)] == old
    for row in state['output_rows']:
        assert sum((F(v)*point[c] for c, v in row['coefficients']), F(0)) == F(row['rhs'])
    for row in [*state['auxiliary_rows'], *state['output_rows']]:
        assert all(F(2)**-20 <= abs(F(v)) <= F(2)**40 for _, v in row['coefficients'])
    record_property('new_factors', state['new_factors'])
    record_property('factored_equation_nnz', state['total_new_equation_nnz'])
    record_property('direct_equation_nnz', sum(sum(c != -1 for c in p) for p in direct))
    record_property('full_actual_polynomial_and_box_inverse_pass', True)


def test_two_overlapping_tiles_keep_same_global_source_and_old_output_ids():
    args = list(fixture())
    grid = np.arange(24).reshape(1, 4, 6)
    args[0] = {**args[0], 'n_cont': 40}
    args[1] = grid[:, :, :4]
    args[6] = np.arange(24, 32).reshape(2, 2, 2)
    left = build_tile(*args, pool=WorkPool(256_000_000), enabled=True)
    right_args = list(args)
    right_args[1], right_args[6] = grid[:, :, 2:], np.arange(32, 40).reshape(2, 2, 2)
    right = build_tile(*right_args, pool=WorkPool(256_000_000), enabled=True, first_aux=left['n_cont'])
    rows = left['auxiliary_rows']+right['auxiliary_rows']
    outputs = left['output_rows']+right['output_rows']
    assert project_outputs(rows, outputs, 40) == spatial_polynomials(args)+spatial_polynomials(right_args)
    assert left['source'] is right['source']
    assert set(grid[:, :, 2:4].flat) <= set(c for row in right['auxiliary_rows']
                                          for c, _ in row['coefficients'] if c < 40)


def test_actual_output_corruption_changes_full_polynomial_not_accepted_by_oracle():
    args = fixture(3)
    state = build_tile(*args, pool=WorkPool(256_000_000), enabled=True)
    changed = [dict(r) for r in state['output_rows']]
    changed[0]['rhs'] += 1.
    assert project_outputs(state['auxiliary_rows'], changed, state['base_n_cont']) != spatial_polynomials(args)


@pytest.mark.parametrize('rhs,coefficients,message', [
    (F(1, 3), [(0, F(1))], 'exactly binary64'),
    (F(0), [(0, F(2)**-24), (1, F(2)**40)], 'window incompatible'),
])
def test_complete_row_rejection_keeps_all_native_terms(rhs, coefficients, message):
    with pytest.raises(ValueError, match=message):
        native_row(coefficients, rhs)


def test_opt_in_and_complete_cost_precharge():
    args, pool = fixture(), WorkPool(0)
    assert build_tile(*args, pool=pool) is None and pool.used == 0
    with pytest.raises(MemoryError):
        build_tile(*args, pool=pool, enabled=True)
