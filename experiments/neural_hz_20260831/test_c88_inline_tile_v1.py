"""Independent full polynomial projection and box inverse for reduced circuits."""
from fractions import Fraction as F
import numpy as np
import pytest

from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c85_exact_filter_transform_v1 import transform
from experiments.neural_hz_20260831.c86_complete_tile_hz_v1 import extend_actual, project_outputs
from experiments.neural_hz_20260831.c88_inline_tile_v1 import construct, actual_rows, NativeUnproved, _row, prepare, Prepared


@pytest.mark.parametrize('channels,outputs', [(1, 2), (3, 4), (8, 8)])
@pytest.mark.parametrize('mode', ['dense', 'sparse', 'output_mask', 'boundary_odd', 'shared', 'scaled'])
def test_full_original_polynomial_and_actual_inverse(channels, outputs, mode):
    height, pad = (3, 1) if mode == 'boundary_odd' else (4, 0)
    oh = height+2*pad-2
    original_ids = np.arange(channels*height*height).reshape(channels, height, height)
    if mode == 'shared':
        original_ids[:, 1] = original_ids[:, 0]
    active = np.ones(original_ids.shape, bool)
    if mode == 'sparse':
        active = np.arange(active.size).reshape(active.shape) % 5 == 0
    powers = np.arange(active.size).reshape(active.shape) % 4 if mode == 'scaled' else np.zeros(active.shape, int)
    out = np.arange(original_ids.size, original_ids.size+outputs*oh*oh).reshape(outputs, oh, oh)
    base = original_ids.size+out.size
    if mode == 'output_mask':
        out[np.arange(out.size).reshape(out.shape) % 3 != 0] = -1
    weights = ((np.arange(outputs*channels*9) % 31)-15).reshape(outputs, channels, 3, 3).astype(np.float32)/32
    _, data = transform(weights, pool=WorkPool(256_000_000), enabled=True)
    for y in range(0, oh, 2):
        for x in range(0, oh, 2):
            ids = np.full((channels, 4, 4), -1, np.int64)
            exps = np.zeros(ids.shape, np.int32)
            output_ids = np.full((outputs, 2, 2), -1, np.int64)
            for c, i, j in np.ndindex(ids.shape):
                sy, sx = y-pad+i, x-pad+j
                if 0 <= sy < height and 0 <= sx < height and active[c, sy, sx]:
                    ids[c, i, j], exps[c, i, j] = original_ids[c, sy, sx], powers[c, sy, sx]
            output_ids[:, :min(2, oh-y), :min(2, oh-x)] = out[:, y:y+2, x:x+2]
            output_exps = np.full(output_ids.shape, 20, np.int32)
            report, packet = construct(data, ids, exps, output_ids, output_exps, base,
                                       pool=WorkPool(256_000_000), enabled=True)
            aux, emitted = actual_rows(report, packet)
            expected = []
            for k, i, j in np.ndindex(output_ids.shape):
                col = int(output_ids[k, i, j])
                if col < 0:
                    continue
                poly = {col: F(2)**20}
                for c, a, b in np.ndindex(channels, 3, 3):
                    parent = int(ids[c, i+a, j+b])
                    if parent >= 0:
                        v = F(float(weights[k, c, a, b]))*F(2)**int(exps[c, i+a, j+b])
                        poly[parent] = poly.get(parent, F(0))-v
                expected.append({c: v for c, v in poly.items() if v})
            assert project_outputs(aux, emitted, base) == expected
            old = [F(i % 5-2, 4) for i in range(base)]
            for row, poly in zip(emitted, expected, strict=True):
                col = row['slot']
                old[col] = -sum((v*old[c] for c, v in poly.items() if c != col), F(0))/poly[col]
            point = extend_actual(aux, old)
            for row in emitted:
                assert sum((F(v)*point[c] for c, v in row['coefficients']), F(0)) == 0
            assert np.array_equal(np.ldexp(packet['native'], -packet['powers']-
                                  np.repeat(packet['gauges'], np.diff(packet['indptr']))), packet['words'].astype(float))
            assert report['new_factors'] <= report['original_used_v']+report['original_used_m']


def test_nonrepresentable_coalescence_never_publishes_rounded_row():
    with pytest.raises(NativeUnproved, match='not binary64'):
        _row([(0, (1 << 53)+1, -40)], 1, 20)


def test_native_box_and_window_rejection():
    with pytest.raises(NativeUnproved, match='window'):
        _row([(0, 1, -40)], 1, 40)


def test_exact_cancellation_and_old_output_pivot():
    row = _row([(0, 3, -2), (0, -6, -3)], 1, 5)
    assert row['columns'].tolist() == [1]
    assert row['words'].tolist() == [1] and row['powers'].tolist() == [5]


def test_default_off_and_whole_precharge():
    _, data = transform(np.ones((1, 1, 3, 3), np.float32), pool=WorkPool(256_000_000), enabled=True)
    args = (data, np.arange(16).reshape(1, 4, 4), np.zeros((1, 4, 4), np.int32),
            np.arange(16, 20).reshape(1, 2, 2), np.full((1, 2, 2), 10, np.int32), 20)
    pool = WorkPool(0)
    assert construct(*args, pool=pool) is None and pool.used == 0
    with pytest.raises(MemoryError):
        construct(*args, pool=pool, enabled=True)


def test_dense_degree_factorization_matches_complete_pair_path():
    pool = WorkPool(256_000_000)
    _, data = transform(np.ones((3, 2, 3, 3), np.float32), pool=pool, enabled=True)
    prepared = prepare(data, pool=pool)
    assert prepared.dense
    args = (np.arange(32).reshape(2, 4, 4), np.zeros((2, 4, 4), np.int32),
            np.arange(32, 44).reshape(3, 2, 2), np.full((3, 2, 2), 10, np.int32), 44)
    fast_report, fast = construct(prepared, *args, pool=pool, enabled=True)
    slow_report, slow = construct(Prepared(data, False), *args, pool=WorkPool(256_000_000), enabled=True)
    assert fast_report == slow_report
    assert all(np.array_equal(fast[key], slow[key]) for key in fast)
    assert 'c88_dense_transform_factorized_degrees' in pool.parts
