from fractions import Fraction
from itertools import combinations, product

import numpy as np
import pytest

from experiments.neural_hz_20260831.c18_sparse_rewrite_v1 import (
    sort_rewritten_row, sort_cost, merge_cost)
from experiments.neural_hz_20260831.c17_owned_rows_v1 import WorkPool
from experiments.neural_hz_20260831.c10_alias_quotient_v1 import exact_sum


def normalize_result(columns, values):
    result = []
    for col in sorted(set(map(int, columns))):
        terms = values[columns == col]
        value = exact_sum(terms) if len(terms) > 1 else float(terms[0])
        assert Fraction(value) == sum(map(Fraction, terms), Fraction())
        if value:
            result.append((col, value))
    return tuple((col, np.float64(value).tobytes()) for col, value in result)


def check(columns, values, positions):
    columns, values = np.array(columns, np.int64), np.array(values, np.float64)
    positions = np.array(positions, np.int64)
    old = columns.copy(), values.copy(), positions.copy()
    pool = WorkPool(0, 0)
    cc, cv, mode = sort_rewritten_row(columns, values, positions, pool)
    assert np.all(np.diff(cc) >= 0)
    assert sorted(zip(map(int, columns), map(float, values))) == sorted(zip(map(int, cc), map(float, cv)))
    order = np.argsort(columns, kind='stable')
    try:
        expected = normalize_result(columns[order], values[order])
    except ValueError:
        with pytest.raises(ValueError): normalize_result(cc, cv)
    else:
        assert normalize_result(cc, cv) == expected
    for a, b in zip((columns, values, positions), old):
        assert a.tobytes() == b.tobytes()
    return mode, pool


@pytest.mark.parametrize('w,k', [(0, 0), (1, 0), (1, 1), (3, 1), (64, 0), (64, 64)])
def test_empty_full_and_small_fallback(w, k):
    cols = np.arange(w, dtype=np.int64)
    positions = np.arange(w - k, w)
    cols[positions] = cols[positions][::-1]
    mode, pool = check(cols, np.ones(w), positions)
    assert mode == 'full' and pool.used == sort_cost(w)


@pytest.mark.parametrize('pattern', ['single', 'ordered', 'reversed', 'duplicates', 'interleaved'])
@pytest.mark.parametrize('sign', [-1., 1.])
def test_exact_sparse_streams_with_signed_non_dyadic_terms(pattern, sign):
    w = 256
    cols = 4 * np.arange(w)
    positions = np.array([128, 160, 192, 224])
    if pattern == 'single': positions = positions[:1]
    mapped = {'single': [5], 'ordered': [5, 101, 501, 777],
        'reversed': [777, 501, 101, 5], 'duplicates': [8, 8, 8, 8],
        'interleaved': [4, 12, 20, 28]}[pattern]
    cols[positions] = mapped
    values = np.full(w, sign * .3)
    mode, pool = check(cols, values, positions)
    assert mode == ('merge_sorted' if pattern == 'reversed' else 'merge_ordered')
    assert pool.used == 8 * len(positions) + merge_cost(w, len(positions)) + (
        sort_cost(len(positions)) if pattern == 'reversed' else 0)
    assert pool.used < sort_cost(w)


@pytest.mark.parametrize('terms', [(.3, -.3), (.3, .3), (.3, 1.),
    (2.**40, 2.**40), (2.**-20, -2.**-20 + 2.**-40)])
def test_collision_cancellation_and_exactness_window_rejections(terms):
    cols, values = np.arange(128), np.ones(128)
    cols[100] = 5
    values[5], values[100] = terms
    mode, _ = check(cols, values, [100])
    assert mode == 'merge_ordered'


def test_exhaustive_small_position_and_mapped_parent_patterns():
    # Deterministic exhaustive unit-test inputs; never attack/search NN inputs.
    for positions in combinations(range(4), 2):
        for mapped in product(range(4), repeat=2):
            cc = np.arange(4)
            cc[list(positions)] = mapped
            check(cc, [.5, -.5, .25, -.25], positions)


def test_dense_right_disorder_selects_full_after_check_before_attempt():
    w, k = 1024, 512
    cc = np.arange(w)
    p = np.arange(w - k, w)
    cc[p] = np.arange(k)[::-1]
    mode, pool = check(cc, np.ones(w), p)
    assert mode == 'full_after_check'
    assert pool.used == 8 * k + sort_cost(w)
    assert 'rewrite_stream_merge' not in pool.parts


@pytest.mark.parametrize('stage', ['full', 'check', 'merge', 'full_after_check'])
def test_cap_failure_before_operation_no_fallback_no_input_mutation(stage, monkeypatch):
    w, k = ((3, 1) if stage == 'full' else (1024, 512) if stage == 'full_after_check' else (256, 4))
    cc, cv, p = np.arange(w), np.ones(w), np.arange(w-k, w)
    cc[p] = np.arange(k)[::-1]
    before = cc.copy(), cv.copy(), p.copy()
    amount = sort_cost(w) if stage == 'full' else 8 * k
    if stage == 'merge': amount += merge_cost(w, k) + sort_cost(k)
    if stage == 'full_after_check': amount += sort_cost(w)
    pool = WorkPool(0, 0, max_work=amount - 1)
    def forbidden(*args, **kwargs): raise AssertionError('sort/search executed before reservation')
    monkeypatch.setattr(np, 'argsort', forbidden)
    monkeypatch.setattr(np, 'searchsorted', forbidden)
    with pytest.raises(MemoryError): sort_rewritten_row(cc, cv, p, pool)
    for a, b in zip((cc, cv, p), before): assert a.tobytes() == b.tobytes()
    expected = 8*k if stage in ('merge', 'full_after_check') else 0
    assert pool.used == expected


@pytest.mark.parametrize('kind', ['dtype', 'shape', 'negative', 'out_of_range', 'duplicate', 'unsorted'])
def test_invalid_stream_contract_fails_closed(kind):
    cc, cv, p = np.arange(256), np.ones(256), np.array([200, 220])
    cc[p] = [8, 4]
    if kind == 'dtype': p = p.astype(float)
    if kind == 'shape': cv = cv[:, None]
    if kind == 'negative': p[0] = -1
    if kind == 'out_of_range': p[-1] = 256
    if kind == 'duplicate': p[-1] = p[0]
    if kind == 'unsorted': p = p[::-1].copy()
    with pytest.raises(ValueError): sort_rewritten_row(cc, cv, p, WorkPool(0, 0))
