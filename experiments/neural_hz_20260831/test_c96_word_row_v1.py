"""Exact old/new rows and full polynomial/native/inverse checks with aliases."""
from fractions import Fraction as F
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c85_exact_filter_transform_v1 import transform
from experiments.neural_hz_20260831.c86_complete_tile_hz_v1 import extend_actual, project_outputs
from experiments.neural_hz_20260831.c88_inline_tile_v1 import construct as old_construct, actual_rows, _row as old_row
from experiments.neural_hz_20260831.c95_word_filter_v1 import prepare_words
from experiments.neural_hz_20260831.c96_word_row_v1 import construct, _row


@pytest.mark.parametrize('width', [0, 1, 4, 32])
@pytest.mark.parametrize('mode', ['ordinary', 'alias', 'cancellation', 'mixed_power'])
def test_all_actual_fields_and_only_final_word_credits(width, mode):
    terms = [(i, i % 5+1, i % 3-4) for i in range(width)]
    if mode == 'alias': terms += [(i, -1, -6) for i in range(width)]
    if mode == 'cancellation': terms += [(c, -n, e) for c, n, e in terms.copy()]
    if mode == 'mixed_power': terms += [(i, 3, -7) for i in range(width)]
    pool = WorkPool(1000000)
    old = old_row(terms, width)
    new = _row(terms, width, pool=pool)
    assert set(old) == set(new)
    for key in old:
        if isinstance(old[key], np.ndarray):
            assert old[key].dtype == new[key].dtype and old[key].tobytes() == new[key].tobytes()
        else: assert old[key] == new[key]
    assert pool.used == 64+16*(len(terms)+1)-8*len(new['words'])
    assert pool.parts['c96_no_credit_for_coalesced_away_guards'] == 8*(len(terms)+1-len(new['words']))


@pytest.mark.parametrize('mode', ['word', 'exponent', 'precision', 'window', 'pivot'])
def test_all_existing_unproved_row_conditions(mode):
    terms, pivot, power = [(0, 1, 0)], 1, 8
    if mode == 'word': terms = [(0, (1 << 62)+1, 0)]
    if mode == 'exponent': terms = [(0, 1, -4097)]
    if mode == 'precision': terms = [(0, (1 << 53)+1, -40)]
    if mode == 'window': terms, power = [(0, 1, -40)], 40
    if mode == 'pivot': pivot = 0
    with pytest.raises(ValueError): old_row(terms, pivot, power)
    with pytest.raises(ValueError): _row(terms, pivot, power, pool=WorkPool(1000000))


def test_complete_row_precharge():
    pool = WorkPool(0)
    with pytest.raises(MemoryError): _row([(0, 1, 0)], 1, pool=pool)
    assert pool.used == 0


@pytest.mark.parametrize('channels,outputs', [(1, 2), (3, 4), (8, 8)])
@pytest.mark.parametrize('mode', ['dense', 'sparse', 'output_mask', 'boundary_odd', 'shared', 'scaled'])
def test_word_rows_full_polynomial_native_and_actual_inverse(channels, outputs, mode):
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
            new_pool, old_pool = WorkPool(256_000_000), WorkPool(256_000_000)
            report, packet = construct(data, ids, exps, output_ids, output_exps, base,
                                       pool=new_pool, enabled=True)
            old_report, old_packet = old_construct(legacy, ids, exps, output_ids, output_exps, base,
                                              pool=old_pool, enabled=True)
            new_row_work = sum(new_pool.parts.get(n, 0) for n in
                               ('c96_exact_row_prefix_and_numeric', 'c96_no_credit_for_coalesced_away_guards'))
            assert old_pool.parts['c88_exact_row_words_native_and_box']-new_row_work == 8*report['nnz']
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
