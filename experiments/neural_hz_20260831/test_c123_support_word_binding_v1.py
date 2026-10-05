"""Independent actual SparseHZ source-row tests for support-first word binding.

Fixtures retain continuous and binary factors, an old mixed equality, parent
definitions, and a mixed inequality.  Original convolution equations are built
with Fraction arithmetic, never the implementation's word or support helpers.
These are ordinary isolated sources, not restored target-network evidence.
"""
from fractions import Fraction as F

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c120_complete_source_fixture_v1 import (
    _bind_actual_rows as fraction_bind)
from experiments.neural_hz_20260831.c123_support_word_binding_v1 import bind_actual_rows


def _fraction_row(weights, maps, k, y, x):
    pivot = int(maps['outs'][k, y, x])
    expected = {pivot: F(2)**int(maps['output_powers'][k, y, x])}
    for c in range(weights.shape[1]):
        for dy in range(3):
            for dx in range(3):
                source = int(maps['ids'][c, y+dy, x+dx])
                if source < 0:
                    continue
                coefficient = F(float(weights[k, c, dy, dx]))
                coefficient *= F(2)**int(maps['powers'][c, y+dy, x+dx])
                expected[source] = expected.get(source, F(0))-coefficient
    return {column: value for column, value in expected.items() if value}


def _source(mode='dense', *, c=2, k=2, h=4):
    old_nc, old_ne = 2, 1
    oh = h-2
    parent_count, output_count = c*h*h, k*oh*oh
    ncont = old_nc+parent_count+output_count
    neq = old_ne+parent_count+output_count
    ids = np.arange(old_nc, old_nc+parent_count, dtype=np.int64).reshape(c, h, h)
    powers = (np.arange(parent_count, dtype=np.int32).reshape(c, h, h) % 3)-1
    raw_outs = np.arange(old_nc+parent_count, ncont, dtype=np.int64).reshape(k, oh, oh)
    outs = raw_outs.copy()
    opowers = np.full(outs.shape, 4, dtype=np.int32)
    kernel = np.array(((1, 2, 4), (2, 4, 8), (4, 8, 16)), dtype=np.float64)/32
    weights = np.broadcast_to(kernel, (k, c, 3, 3)).copy()
    if mode == 'masked':
        ids[np.arange(parent_count).reshape(ids.shape) % 4 == 0] = -1
    elif mode == 'heterogeneous':
        ids[1] = -1
        ids[1, 1, 1] = old_nc+h*h+1
    elif mode == 'partial_output':
        outs.reshape(-1)[1::2] = -1
    elif mode == 'no_output':
        outs[:] = -1
    elif mode == 'signed_scaled':
        weights[:, 1] *= -1
        weights[1, 0, 0, 1] = -3/64
        opowers[:] = np.arange(output_count).reshape(outs.shape) % 3+3
    elif mode == 'shared_coalesce':
        ids[1] = ids[0]
        powers[1] = powers[0]
        ids[0, 1, 2] = ids[0, 1, 1]
        ids[1, 1, 2] = ids[0, 1, 1]
    elif mode == 'shared_cancel':
        ids[1] = ids[0]
        powers[1] = powers[0]
        weights[:, 1] = -weights[:, 0]
    elif mode == 'outside_native_window':
        opowers[0, 0, 0] = 41
    maps = dict(ids=ids, powers=powers, outs=outs, output_powers=opowers)
    eq_roots = np.arange(neq, dtype=np.int64)
    eq_scales = np.zeros(neq, dtype=np.int32)
    rows = [{0: F(1)}]
    rhs = [0.0]
    binary_rows, binary_values = [0], [-.25]
    for parent in range(old_nc, old_nc+parent_count):
        rows.append({0: -F(1, 4), parent: F(1)})
        rhs.append(.03125)
        binary_rows.append(len(rows)-1)
        binary_values.append(-.125)
    expected_literals, expected_gauges, expected_pivots = [], [], {}
    for filter_index in range(k):
        for y in range(oh):
            for x in range(oh):
                physical = len(rows)
                raw_pivot = int(raw_outs[filter_index, y, x])
                if int(outs[filter_index, y, x]) < 0:
                    rows.append({raw_pivot: F(1)})
                    rhs.append(0.0)
                    continue
                gauge = (physical % 3)-1
                if mode == 'outside_native_window':
                    gauge = 0
                expected = _fraction_row(weights, maps, filter_index, y, x)
                native = {column: value*F(2)**gauge for column, value in expected.items()}
                rows.append(native)
                rhs.append(0.0)
                eq_scales[physical] = gauge
                literal = dict(coefficients=[(column, float(value))
                                             for column, value in sorted(native.items())],
                               rhs=0.0, pivot=raw_pivot)
                expected_literals.append(literal)
                expected_gauges.append(gauge)
                expected_pivots[raw_pivot] = (literal, gauge)
    rr, cc, vv = [], [], []
    for row_index, coefficients in enumerate(rows):
        for column, value in sorted(coefficients.items()):
            assert F(float(value)) == value
            rr.append(row_index)
            cc.append(column)
            vv.append(float(value))
    ac = sp.csr_matrix((vv, (rr, cc)), shape=(neq, ncont))
    ab = sp.csr_matrix((binary_values, (binary_rows, np.zeros(len(binary_rows), dtype=int))),
                       shape=(neq, 1))
    gc = sp.csr_matrix((np.full(output_count, .25),
                        (np.arange(output_count), raw_outs.reshape(-1))),
                       shape=(output_count, ncont))
    auc = sp.csr_matrix(([1.0], ([0], [1])), shape=(1, ncont))
    hz = SparseHZono(np.zeros(output_count), gc, sp.csr_matrix((output_count, 1)),
                    ac, ab, np.array(rhs), auc, sp.csr_matrix([[.125]]),
                    np.ones(1), frame_id=31, exact=True)
    fields = dict(hz=hz, old_n_cont=old_nc, old_n_eq=old_ne,
                  logical_n_cont=ncont, eq_roots=eq_roots, eq_scales=eq_scales)
    return fields, weights, maps, (expected_literals, expected_gauges, expected_pivots)


def _snapshot(fields, weights, maps):
    hz = fields['hz']
    arrays = [weights, *maps.values(), fields['eq_roots'], fields['eq_scales'],
              hz.c, hz.b, hz.ub]
    for name in ('Gc', 'Gb', 'Ac', 'Ab', 'Auc', 'Aub'):
        matrix = getattr(hz, name)
        arrays.extend((matrix.data, matrix.indices, matrix.indptr))
    return [(array, array.copy()) for array in arrays]


@pytest.mark.parametrize('mode', [
    'dense', 'masked', 'heterogeneous', 'partial_output', 'no_output',
    'signed_scaled', 'shared_coalesce', 'shared_cancel',
])
def test_every_actual_literal_and_gauge_matches_independent_fraction_source(mode):
    fields, weights, maps, expected = _source(mode, h=6 if mode == 'dense' else 4)
    report, originals, gauges, by_pivot = bind_actual_rows(
        fields, weights, maps, pool=WorkPool(256_000_000), enabled=True)
    assert (originals, gauges, by_pivot) == expected
    assert report['all_actual_source_output_rows_bound'] == len(expected[0])
    assert report['actual_direct_output_nnz'] == sum(len(row['coefficients']) for row in originals)
    assert report['all_actual_native_coefficients_compared'] == report['actual_direct_output_nnz']
    assert report['original_kernel_coefficients_decoded'] == weights.size
    assert report['exact_coefficients_and_original_row_gauges_equal']
    assert report['source_survival']['all_selected_original_coordinates_survive']
    assert not report['source_survival']['numeric_or_LIVE_admission']
    assert not report['numeric_admission'] and not report['actual_global_admission']
    assert report['formal_gain'] == 0
    hz = fields['hz']
    assert hz.n_bin == 1 and hz.n_ineq == 1 and hz.Ab[0, 0] == -.25
    assert hz.Aub[0, 0] == .125 and hz.frame_id == 31 and hz.exact
    if np.all(maps['outs'] >= 0):
        old_report, old_rows, old_gauges, old_pivots = fraction_bind(
            fields, weights, maps, WorkPool(256_000_000))
        assert (originals, gauges, by_pivot) == (old_rows, old_gauges, old_pivots)
        assert report['actual_direct_output_nnz'] == old_report['actual_direct_output_nnz']
    if mode == 'shared_cancel':
        assert originals and all(len(row['coefficients']) == 1 for row in originals)
    if mode == 'no_output':
        assert originals == gauges == [] and by_pivot == {}


@pytest.mark.parametrize('tamper', [
    'coefficient', 'support', 'pivot', 'rhs', 'binary',
    'parent_quotient', 'output_quotient', 'canonical',
])
def test_any_actual_source_coefficient_support_or_semantic_tamper_is_rejected(tamper):
    fields, weights, maps, _ = _source()
    hz = fields['hz']
    pivot = int(maps['outs'][0, 0, 0])
    physical = int(fields['eq_roots'][fields['old_n_eq']+pivot-fields['old_n_cont']])
    begin, end = map(int, hz.Ac.indptr[physical:physical+2])
    if tamper == 'coefficient':
        hz.Ac.data[begin] *= 2
    elif tamper == 'support':
        maps['ids'][0, 0, 0] = -1
    elif tamper == 'pivot':
        position = begin+int(np.flatnonzero(hz.Ac.indices[begin:end] == pivot)[0])
        hz.Ac.data[position] *= 2
    elif tamper == 'rhs':
        hz.b[physical] = .125
    elif tamper == 'binary':
        bad = hz.Ab.tolil()
        bad[physical, 0] = .125
        hz.Ab = bad.tocsr()
    elif tamper == 'parent_quotient':
        parent = int(maps['ids'][0, 0, 0])
        fields['eq_roots'][fields['old_n_eq']+parent-fields['old_n_cont']] = -1
    elif tamper == 'output_quotient':
        fields['eq_roots'][fields['old_n_eq']+pivot-fields['old_n_cont']] = -1
    else:
        # The mathematical row is unchanged, but its demanded native CSR is
        # deliberately noncanonical.  The binder must not silently repair it.
        hz.Ac.indices[begin:begin+2] = hz.Ac.indices[begin:begin+2][::-1]
        hz.Ac.data[begin:begin+2] = hz.Ac.data[begin:begin+2][::-1]
        hz.Ac.has_sorted_indices = False
        hz.Ac.has_canonical_format = False
    with pytest.raises(ValueError):
        bind_actual_rows(fields, weights, maps, pool=WorkPool(256_000_000), enabled=True)


def test_exact_but_outside_native_coefficient_window_does_not_get_bound():
    fields, weights, maps, _ = _source('outside_native_window')
    with pytest.raises(ValueError):
        bind_actual_rows(fields, weights, maps, pool=WorkPool(256_000_000), enabled=True)


def test_default_off_does_not_inspect_inputs_or_charge():
    pool = WorkPool(0)
    assert bind_actual_rows(None, None, None, pool=pool) is None
    assert pool.used == 0


def test_empty_budget_rejects_before_reading_a_bad_weight_value():
    fields, weights, maps, _ = _source()
    weights[0, 0, 0, 0] = np.nan
    pool = WorkPool(0)
    with pytest.raises(MemoryError):
        bind_actual_rows(fields, weights, maps, pool=pool, enabled=True)
    assert pool.used == 0


def test_actual_inputs_predicates_maps_and_frame_are_never_mutated():
    fields, weights, maps, expected = _source('signed_scaled')
    snapshot = _snapshot(fields, weights, maps)
    _, originals, gauges, by_pivot = bind_actual_rows(
        fields, weights, maps, pool=WorkPool(256_000_000), enabled=True)
    assert (originals, gauges, by_pivot) == expected
    assert all(np.array_equal(array, before) for array, before in snapshot)
    assert fields['hz'].frame_id == 31 and fields['hz'].n_bin == 1
    originals[0]['coefficients'][0] = (-1, 123.0)
    gauges[0] += 1
    assert all(np.array_equal(array, before) for array, before in snapshot)


def test_same_support_with_a_one_ulp_original_weight_change_cannot_reuse_old_rows():
    fields, weights, maps, _ = _source()
    weights[0, 0, 0, 0] = np.nextafter(weights[0, 0, 0, 0], np.inf)
    with pytest.raises(ValueError):
        bind_actual_rows(fields, weights, maps, pool=WorkPool(256_000_000), enabled=True)


def test_zero_output_demand_still_requires_every_original_kernel_coefficient():
    fields, weights, maps, _ = _source('no_output')
    weights[-1, -1, -1, -1] = np.nan
    with pytest.raises(ValueError):
        bind_actual_rows(fields, weights, maps, pool=WorkPool(256_000_000), enabled=True)
