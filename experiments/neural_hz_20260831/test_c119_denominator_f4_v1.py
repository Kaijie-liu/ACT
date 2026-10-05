"""Ordinary exact convolution, actual inverse and packet rejection tests."""
from fractions import Fraction as F
import numpy as np
import pytest

from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c86_complete_tile_hz_v1 import extend_actual, project_outputs
from experiments.neural_hz_20260831.c119_denominator_f4_v1 import construct
from experiments.neural_hz_20260831.c119_f4_oracle_v1 import prove


def _fixture(mode='dense'):
    channels = 2 if mode == 'shared' else 1
    ids = np.arange(channels*36, dtype=np.int64).reshape(channels, 6, 6)
    powers = np.zeros(ids.shape, np.int32)
    outputs = np.arange(ids.size, ids.size+16, dtype=np.int64).reshape(1, 4, 4)
    outpowers = np.full(outputs.shape, 8, np.int32)
    base = ids.size+16
    weights = ((np.arange(9*channels).reshape(1, channels, 3, 3) % 11)-5).astype(np.float64)/16
    if mode == 'mask':
        outputs[:, 1::2, ::2] = -1
        ids[:, 2::2, 1::2] = -1
    elif mode == 'padding':
        ids[:, 0, :] = -1
        ids[:, :, 0] = -1
    elif mode == 'shared':
        ids[:, 1] = ids[:, 0]
        ids[1] = ids[0]
    elif mode == 'scaled':
        powers[:] = np.arange(ids.size).reshape(ids.shape) % 5-2
        outpowers[:] = np.arange(outputs.size).reshape(outputs.shape) % 3+8
    elif mode == 'sparse_kernel':
        weights[:] = 0
        weights[0, 0, 1, 1] = F(3, 8)
    elif mode == 'zero_kernel':
        weights[:] = 0
    elif mode == 'zero_inputs':
        ids[:] = -1
    elif mode == 'zero_outputs':
        outputs[:] = -1
    return weights, ids, powers, outputs, outpowers, base


def _build(args):
    return construct(*args, pool=WorkPool(256_000_000), enabled=True)


def _rows(packet):
    auxiliary, outputs = [], []
    for i, role in enumerate(packet['roles']):
        a, b = map(int, packet['indptr'][i:i+2])
        row = dict(coefficients=tuple((int(c), float(v)) for c, v in
                   zip(packet['columns'][a:b], packet['native'][a:b], strict=True)),
                   rhs=float(packet['rhs'][i]), slot=int(packet['pivots'][i]),
                   gauge=int(packet['gauges'][i]))
        (outputs if role[0] == 2 else auxiliary).append(row)
    return auxiliary, outputs


def _direct(args):
    weights, ids, powers, outputs, outpowers, base = args
    result = []
    for k, i, j in np.ndindex(outputs.shape):
        pivot = int(outputs[k, i, j])
        if pivot < 0:
            continue
        row = {pivot: F(2)**int(outpowers[k, i, j])}
        for c, a, b in np.ndindex(weights.shape[1], 3, 3):
            parent = int(ids[c, i+a, j+b])
            if parent >= 0:
                value = F(float(weights[k, c, a, b]))*F(2)**int(powers[c, i+a, j+b])
                row[parent] = row.get(parent, F(0))-value
        result.append({col: value for col, value in row.items() if value})
    return result


def _old_point(args, emitted, direct):
    point = [F(i % 7-3, 4) for i in range(args[-1])]
    for native, row in zip(emitted, direct, strict=True):
        pivot = native['slot']
        point[pivot] = -sum((value*point[col] for col, value in row.items()
                            if col != pivot), F(0))/row[pivot]
    return point


@pytest.mark.parametrize('mode', ['dense', 'mask', 'padding', 'shared', 'scaled',
                                  'sparse_kernel', 'zero_kernel', 'zero_inputs'])
def test_exact_composition_against_independent_actual_elimination_and_inverse(mode):
    args = _fixture(mode)
    snapshots = [value.copy() for value in args[:-1]]
    report, packet = _build(args)
    proof = prove(packet, *args, pool=WorkPool(256_000_000), enabled=True)
    assert proof['all_1d_bilinear_coefficients_proved'] == 72
    assert proof['all_native_nnz_proved'] == report['nnz']
    assert proof['all_auxiliary_equations_and_redundant_boxes_proved'] == report['new_factors']
    auxiliary, emitted = _rows(packet)
    direct = _direct(args)
    assert project_outputs(auxiliary, emitted, args[-1]) == direct
    original = _old_point(args, emitted, direct)
    point = extend_actual(auxiliary, original)
    assert point[:args[-1]] == original
    for row in [*auxiliary, *emitted]:
        assert sum((F(v)*point[c] for c, v in row['coefficients']), F(0)) == F(row['rhs'])
    assert all(np.array_equal(old, value) for old, value in zip(snapshots, args[:-1], strict=True))
    assert report['row_construction_prepaid'] >= report['whole_circuit_emission']
    assert proof['formal_gain'] == 0 and not proof['actual_network_source_bound']


def test_odd_denominators_are_fresh_M_pivots_not_original_output_scaling():
    args = _fixture()
    report, packet = _build(args)
    prove(packet, *args, pool=WorkPool(256_000_000), enabled=True)
    m = packet['roles'][:, 0] == 1
    assert np.any(packet['defining_denominators'][m] % 3 == 0)
    assert np.all(packet['defining_denominators'][~m] == 1)
    for r in np.flatnonzero(m):
        a, b = packet['indptr'][r:r+2]
        pos = np.flatnonzero(packet['columns'][a:b] == packet['pivots'][r])[0]+a
        pivot = F(float(packet['native'][pos]))/F(2)**int(packet['gauges'][r])
        assert pivot == int(packet['defining_denominators'][r])*F(2)**int(packet['semantic_powers'][r])
    assert report['semantic_units_are_not_defining_pivots']


def test_both_binary_source_branches_keep_EQ_INEQ_and_original_coordinates():
    args = _fixture()
    _, packet = _build(args)
    auxiliary, emitted = _rows(packet)
    direct = _direct(args)
    # External original HZ predicates: x0-b=0, x1+b<=1, b in {-1,+1}.
    # Both branches survive unchanged under the actual unique extension.
    for binary in (-1, 1):
        original = [F(0)]*args[-1]
        original[0] = F(binary)
        for native, row in zip(emitted, direct, strict=True):
            pivot = native['slot']
            original[pivot] = -sum((v*original[c] for c, v in row.items() if c != pivot), F(0))/row[pivot]
        point = extend_actual(auxiliary, original)
        assert point[:args[-1]] == original
        assert point[0]-binary == 0 and point[1]+binary <= 1
        for row in emitted:
            assert sum((F(v)*point[c] for c, v in row['coefficients']), F(0)) == 0


def test_disabled_is_uninspected_and_unpaid():
    pool = WorkPool(0)
    assert construct(None, None, None, None, None, None, pool=pool) is None
    assert prove(None, None, None, None, None, None, None, pool=pool) is None
    assert pool.used == 0


def test_budget_failure_before_exact_work():
    args = _fixture()
    _, packet = _build(args)
    for operation in (lambda pool: construct(*args, pool=pool, enabled=True),
                      lambda pool: prove(packet, *args, pool=pool, enabled=True)):
        pool = WorkPool(0)
        with pytest.raises(MemoryError):
            operation(pool)
        assert pool.used == 0
    # Source numeric validation must itself be prepaid, not run before payment.
    invalid = list(args)
    invalid[0] = invalid[0].copy()
    invalid[0][0, 0, 0, 0] = np.nan
    with pytest.raises(MemoryError):
        prove(packet, *invalid, pool=WorkPool(0), enabled=True)


@pytest.mark.parametrize('mutation', ['coefficient', 'denominator', 'power', 'role',
                                      'output_pivot', 'binary_incidence', 'gauge'])
def test_independent_oracle_rejects_mutated_actual_packet(mutation):
    args = _fixture()
    _, packet = _build(args)
    packet = {key: value.copy() for key, value in packet.items()}
    mrow = int(np.flatnonzero(packet['roles'][:, 0] == 1)[0])
    orow = int(np.flatnonzero(packet['roles'][:, 0] == 2)[0])
    if mutation == 'coefficient':
        packet['native'][0] *= 2
    elif mutation == 'denominator':
        packet['defining_denominators'][mrow] += 1
    elif mutation == 'power':
        packet['semantic_powers'][mrow] += 1
    elif mutation == 'role':
        packet['roles'][mrow, 2] = 100
    elif mutation == 'output_pivot':
        packet['pivots'][orow] += 1
    elif mutation == 'binary_incidence':
        packet['ab_indptr'][-1] = 1
    elif mutation == 'gauge':
        packet['gauges'][orow] += 1
    with pytest.raises(ValueError):
        prove(packet, *args, pool=WorkPool(256_000_000), enabled=True)
    if mutation == 'binary_incidence':
        # The native packet contract excludes unsigned difference wraparound.
        _, unsigned = _build(args)
        unsigned['columns'] = unsigned['columns'].astype(np.uint32)
        with pytest.raises(ValueError, match='signed packet dtype'):
            prove(unsigned, *args, pool=WorkPool(256_000_000), enabled=True)


def test_no_outputs_empty_packet_still_proves_complete_support():
    args = _fixture('zero_outputs')
    report, packet = _build(args)
    proof = prove(packet, *args, pool=WorkPool(256_000_000), enabled=True)
    assert report['rows'] == report['nnz'] == report['new_factors'] == 0
    assert proof['all_original_output_equations_proved'] == 0
    assert proof['exact_required_support_coverage']


def test_missing_required_last_output_fails_full_support():
    args = _fixture()
    _, packet = _build(args)
    packet = {key: value.copy() for key, value in packet.items()}
    end = int(packet['indptr'][-2])
    for key in ('columns', 'native'):
        packet[key] = packet[key][:end].copy()
    for key in ('pivots', 'gauges', 'rhs', 'roles', 'semantic_powers', 'defining_denominators'):
        packet[key] = packet[key][:-1].copy()
    for key in ('indptr', 'ab_indptr'):
        packet[key] = packet[key][:-1].copy()
    with pytest.raises(ValueError, match='row population'):
        prove(packet, *args, pool=WorkPool(256_000_000), enabled=True)
