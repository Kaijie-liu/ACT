"""Independent full exact coefficients and complete old/new tile behavior."""
from fractions import Fraction as F
import numpy as np
import pytest

from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c85_exact_filter_transform_v1 import transform
from experiments.neural_hz_20260831.c85_exact_tile_algebra_v1 import H
from experiments.neural_hz_20260831.c86_complete_tile_hz_v1 import extend_actual, project_outputs
from experiments.neural_hz_20260831.c88_inline_tile_v1 import construct, actual_rows, prepare
from experiments.neural_hz_20260831.c95_word_filter_v1 import prepare_words


@pytest.mark.parametrize('k,c', [(1, 1), (2, 3), (8, 8)])
@pytest.mark.parametrize('mode', ['dense', 'zero', 'sparse', 'cancel', 'scaled'])
def test_complete_integer_and_fraction_coefficients(k, c, mode):
    w = ((np.arange(k*c*9) % 29)+1).reshape(k, c, 3, 3).astype(np.float32)/64
    if mode == 'zero': w[:] = 0
    if mode == 'sparse': w[np.arange(w.size).reshape(w.shape) % 3 != 0] = 0
    if mode == 'cancel': w[:] = np.array([1, -2, 1, 2, -4, 2, 1, -2, 1]).reshape(3, 3)/64
    if mode == 'scaled': w = np.ldexp(w, np.arange(w.size).reshape(w.shape) % 9-4)
    frozen = w.tobytes()
    old_report, old = transform(w, pool=WorkPool(1000000), enabled=True)
    pool = WorkPool(1000000)
    report, new = prepare_words(w, pool=pool, enabled=True)
    assert old_report['all_coefficients_exact_binary64'] == report['all_coefficients_exact_binary64']
    assert new is not None and set(new.transformed) == {'numerator', 'exponent'}
    assert all(np.array_equal(new.transformed[n], old[n]) for n in new.transformed)
    assert new.dense == prepare(old, pool=WorkPool(1000000)).dense
    assert report['original_dense'] == bool(np.all(w != 0))
    assert pool.used == 302*k*c+5*w.size
    assert w.tobytes() == frozen
    for out, channel, a, b in np.ndindex(k, c, 4, 4):
        expected = sum((F(int(H[a, i])*int(H[b, j]), 4)*F(float(w[out, channel, i, j]))
                        for i in range(3) for j in range(3)), F(0))
        value = F(int(new.transformed['numerator'][out, channel, a, b]))*F(2)**int(new.transformed['exponent'][out, channel])
        assert value == expected
    assert report['binary64_finite_normal_scaling_proved_by_envelope']
    assert not report['complete_HZ_row_window_proved']


def test_default_off_and_complete_precharge():
    pool = WorkPool(0)
    w = np.ones((1, 1, 3, 3), np.float32)
    assert prepare_words(w, pool=pool) is None and pool.used == 0
    with pytest.raises(MemoryError): prepare_words(w, pool=pool, enabled=True)


@pytest.mark.parametrize('kind', ['nan', 'inf', 'subnormal', 'shift'])
def test_inherited_unproved_domain_rejection(kind):
    w = np.ones((1, 1, 3, 3), np.float32)
    w.flat[0] = {'nan': np.nan, 'inf': np.inf, 'subnormal': np.nextafter(np.float32(0), np.float32(1)), 'shift': 2.**-40}[kind]
    for function in (transform, prepare_words):
        with pytest.raises(ValueError): function(w, pool=WorkPool(1000000), enabled=True)


def test_normal_source_requires_exact_transformed_significand():
    w = np.ones((1, 1, 3, 3), np.float32)
    w.flat[0] = np.float32(1+2.**-23)
    w.flat[4] = np.float32((1+2.**-23)*2.**-31)
    old, a = transform(w, pool=WorkPool(1000000), enabled=True)
    new, b = prepare_words(w, pool=WorkPool(1000000), enabled=True)
    assert a is b is None
    assert old['exact_binary64_failures'] == new['exact_binary64_failures'] > 0


@pytest.mark.parametrize('channels,outputs', [(1, 2), (3, 4), (8, 8)])
@pytest.mark.parametrize('mode', ['dense', 'sparse', 'output_mask', 'boundary_odd', 'shared', 'scaled'])
def test_fused_full_original_polynomial_native_and_actual_inverse(channels, outputs, mode):
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
    _, legacy = transform(weights, pool=WorkPool(256_000_000), enabled=True)
    _, data = prepare_words(weights, pool=WorkPool(256_000_000), enabled=True)
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
            old_report, old_packet = construct(legacy, ids, exps, output_ids, output_exps, base,
                                              pool=WorkPool(256_000_000), enabled=True)
            assert report == old_report
            assert set(packet) == set(old_packet)
            assert all(packet[n].dtype == old_packet[n].dtype and packet[n].shape == old_packet[n].shape
                       and packet[n].tobytes() == old_packet[n].tobytes() for n in packet)
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
