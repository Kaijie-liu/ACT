"""Independent exact mixed F4/direct composition and native packet guards.

Direct references include every original channel.  Actual emitted auxiliary
equations are eliminated independently, so shared-ID cancellation across the
transformed/direct boundary cannot be hidden by a common transform helper.
"""
from fractions import Fraction as F

import numpy as np
import pytest

from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c86_complete_tile_hz_v1 import extend_actual, project_outputs
from experiments.neural_hz_20260831.c120_word_f4_v1 import construct as all_f4
from experiments.neural_hz_20260831 import c124_mixed_f4_v1 as mixed
from experiments.neural_hz_20260831.c124_mixed_f4_oracle_v1 import prove


def _fixture(mode='dense'):
    ids = np.arange(72, dtype=np.int64).reshape(2, 6, 6)
    powers = np.zeros(ids.shape, dtype=np.int32)
    outputs = np.arange(72, 88, dtype=np.int64).reshape(1, 4, 4)
    opowers = np.full(outputs.shape, 8, dtype=np.int32)
    kernel = np.array(((1, 2, 4), (2, 4, 8), (4, 8, 16)), dtype=np.float64)/32
    weights = np.broadcast_to(kernel, (1, 2, 3, 3)).copy()
    selected = np.array([True, False], dtype=bool)
    if mode == 'masked':
        ids[:, 2::2, 1::2] = -1
        outputs[:, 1::2, ::2] = -1
    elif mode in ('shared', 'cancel'):
        ids[1] = ids[0]
        ids[:, 1] = ids[:, 0]
        if mode == 'cancel':
            weights[:, 1] = -weights[:, 0]
    elif mode == 'scaled':
        powers[:] = np.arange(ids.size).reshape(ids.shape) % 5-2
        opowers[:] = np.arange(outputs.size).reshape(outputs.shape) % 3+8
        weights[0, 1, 1, 1] = -3/16
    elif mode == 'sparse_selected':
        weights[:, 0] = 0
        weights[0, 0, 1, 1] = 3/8
    elif mode == 'zero_selected':
        weights[:, 0] = 0
    elif mode == 'no_outputs':
        outputs[:] = -1
    return (weights, ids, powers, outputs, opowers, 88), selected


def _build(args, selected, pool=None):
    return mixed.construct(*args, selected_channels=selected,
                           pool=pool if pool is not None else WorkPool(256_000_000),
                           enabled=True)


def _prove(packet, args, selected):
    return prove(packet, *args, selected_channels=selected,
                 pool=WorkPool(256_000_000), enabled=True)


def _rows(packet):
    auxiliary, output = [], []
    for index, role in enumerate(packet['roles']):
        begin, end = map(int, packet['indptr'][index:index+2])
        row = dict(coefficients=tuple((int(column), float(value)) for column, value in
                   zip(packet['columns'][begin:end], packet['native'][begin:end], strict=True)),
                   rhs=float(packet['rhs'][index]), slot=int(packet['pivots'][index]),
                   gauge=int(packet['gauges'][index]))
        (output if int(role[0]) == 2 else auxiliary).append(row)
    return auxiliary, output


def _direct(args):
    weights, ids, powers, outputs, opowers, _ = args
    rows = []
    for k, y, x in np.ndindex(outputs.shape):
        pivot = int(outputs[k, y, x])
        if pivot < 0:
            continue
        row = {pivot: F(2)**int(opowers[k, y, x])}
        for channel in range(weights.shape[1]):
            for a in range(3):
                for b in range(3):
                    source = int(ids[channel, y+a, x+b])
                    if source >= 0:
                        value = F(float(weights[k, channel, a, b]))
                        value *= F(2)**int(powers[channel, y+a, x+b])
                        row[source] = row.get(source, F(0))-value
        rows.append({column: value for column, value in row.items() if value})
    return rows


def _satisfying_original(args, emitted, direct, binary=None):
    original = [F(i % 7-3, 4) for i in range(args[-1])]
    if binary is not None:
        original[0], original[1] = F(binary), F(0)
    for native, row in zip(emitted, direct, strict=True):
        pivot = native['slot']
        original[pivot] = -sum((value*original[column] for column, value in row.items()
                                if column != pivot), F(0))/row[pivot]
    return original


@pytest.mark.parametrize('mode', [
    'dense', 'masked', 'shared', 'cancel', 'scaled', 'sparse_selected', 'zero_selected',
])
def test_every_mixed_output_eliminates_to_all_original_channels_exactly(mode):
    args, selected = _fixture(mode)
    snapshots = [value.copy() for value in args[:-1]]
    old_selection = selected.copy()
    report, packet = _build(args, selected)
    proof = _prove(packet, args, selected)
    auxiliary, emitted = _rows(packet)
    direct = _direct(args)
    assert project_outputs(auxiliary, emitted, args[-1]) == direct
    original = _satisfying_original(args, emitted, direct)
    point = extend_actual(auxiliary, original)
    assert point[:args[-1]] == original
    for row in [*auxiliary, *emitted]:
        assert sum((F(value)*point[column] for column, value in row['coefficients']), F(0)) == 0
    assert proof['all_native_nnz_proved'] == len(packet['native']) == report['nnz']
    assert proof['all_1d_bilinear_coefficients_proved'] == 72
    assert proof['all_auxiliary_equations_and_redundant_boxes_proved'] == report['new_factors']
    assert proof['all_original_output_equations_proved'] == len(direct)
    assert proof['exact_required_support_coverage']
    assert not proof['actual_network_source_bound'] and proof['formal_gain'] == 0
    assert not report['live_admission'] and report['score_gain'] == 0
    assert report['row_construction_prepaid'] >= report['whole_circuit_emission']
    assert report['residual_positions_visited'] == 9*len(direct)
    assert proof['all_residual_scan_positions_proved'] == report['residual_positions_visited']
    assert proof['all_residual_nonzero_occurrences_rebuilt'] == report['residual_nonzero_terms']
    assert proof['all_transformed_kernel_coefficients_rebuilt'] == 36*args[0].shape[0]*args[0].shape[1]
    assert packet['new_factors'] == report['new_factors']
    assert packet['selected_channels'].dtype == np.dtype(bool)
    assert packet['selected_channels'].flags.owndata
    assert not np.shares_memory(packet['selected_channels'], selected)
    assert np.array_equal(packet['selected_channels'], old_selection)
    assert np.array_equal(selected, old_selection)
    assert all(np.array_equal(before, value)
               for before, value in zip(snapshots, args[:-1], strict=True))
    if mode == 'cancel':
        assert all(len(row) == 1 for row in direct)
    if mode == 'zero_selected':
        assert report['new_factors'] == 0 and report['rows'] == len(direct)


def test_all_selected_preserves_every_unchanged_C120_native_packet_value():
    args, selected = _fixture('scaled')
    selected[:] = True
    report, packet = _build(args, selected)
    old_report, old_packet = all_f4(*args, pool=WorkPool(256_000_000), enabled=True)
    for name, expected in old_packet.items():
        assert packet[name].dtype == expected.dtype
        assert np.array_equal(packet[name], expected), name
    for name in ('kept_v', 'kept_m', 'new_factors', 'rows', 'nnz', 'whole_circuit_emission'):
        assert report[name] == old_report[name]
    assert report['residual_positions_visited'] == 0
    assert _prove(packet, args, selected)['exact_required_support_coverage']


def test_denominator_units_output_pivots_gauges_and_native_window_are_preserved():
    args, selected = _fixture('scaled')
    _, packet = _build(args, selected)
    _prove(packet, args, selected)
    assert np.all(np.abs(packet['native']) >= 2.**-20)
    assert np.all(np.abs(packet['native']) <= 2.**40)
    saw_odd_denominator = False
    for row_index, role in enumerate(packet['roles']):
        begin, end = map(int, packet['indptr'][row_index:row_index+2])
        columns = packet['columns'][begin:end]
        assert np.all(np.diff(columns) > 0)
        pivot = int(packet['pivots'][row_index])
        offset = int(np.flatnonzero(columns == pivot)[0])+begin
        native_pivot = F(float(packet['native'][offset]))/F(2)**int(packet['gauges'][row_index])
        denominator = int(packet['defining_denominators'][row_index])
        assert native_pivot == denominator*F(2)**int(packet['semantic_powers'][row_index])
        if int(role[0]) == 1:
            saw_odd_denominator |= denominator % 3 == 0
        else:
            assert denominator == 1
        if int(role[0]) == 2:
            k, position = map(int, role[1:])
            assert pivot == int(args[3][k, position//4, position % 4])
            assert int(packet['semantic_powers'][row_index]) == int(args[4][k, position//4, position % 4])
    assert saw_odd_denominator


def test_both_external_binary_branches_preserve_old_EQ_INEQ_and_original_inverse():
    args, selected = _fixture()
    _, packet = _build(args, selected)
    auxiliary, emitted = _rows(packet)
    direct = _direct(args)
    for binary in (-1, 1):
        original = _satisfying_original(args, emitted, direct, binary=binary)
        point = extend_actual(auxiliary, original)
        assert point[:args[-1]] == original
        assert point[0]-binary == 0 and point[1]+binary <= 1
        for row in emitted:
            assert sum((F(value)*point[column] for column, value in row['coefficients']), F(0)) == 0


@pytest.mark.parametrize('mutation', ['residual', 'mask', 'rhs', 'binary', 'gauge', 'denominator', 'pivot'])
def test_independent_oracle_rejects_mixed_tail_or_native_semantic_corruption(mutation):
    args, selected = _fixture()
    _, original = _build(args, selected)
    packet = {name: value.copy() if isinstance(value, np.ndarray) else value
              for name, value in original.items()}
    output_row = int(np.flatnonzero(packet['roles'][:, 0] == 2)[0])
    mrow = int(np.flatnonzero(packet['roles'][:, 0] == 1)[0])
    if mutation == 'residual':
        begin, end = map(int, packet['indptr'][output_row:output_row+2])
        residual_parent = int(args[1][1, 0, 0])
        position = begin+int(np.flatnonzero(packet['columns'][begin:end] == residual_parent)[0])
        packet['native'][position] *= 2
    elif mutation == 'mask':
        packet['selected_channels'][:] = True
    elif mutation == 'rhs':
        packet['rhs'][output_row] = .125
    elif mutation == 'binary':
        packet['ab_indptr'][-1] = 1
    elif mutation == 'gauge':
        packet['gauges'][output_row] += 1
    elif mutation == 'denominator':
        packet['defining_denominators'][mrow] += 1
    else:
        packet['pivots'][output_row] += 1
    with pytest.raises(ValueError):
        _prove(packet, args, selected)
    if mutation == 'mask':
        for invalid_mask in (original['selected_channels'].view(),
                             original['selected_channels'].astype(np.uint8)):
            altered = dict(original, selected_channels=invalid_mask)
            with pytest.raises(ValueError):
                _prove(altered, args, selected)


def test_empty_selection_is_literal_noop_before_any_kernel_preparation(monkeypatch):
    args, selected = _fixture()
    selected[:] = False
    args[0][:] = np.nan
    def forbidden_preparation(*values, **keywords):
        raise AssertionError('empty selection reached kernel preparation')
    monkeypatch.setattr(mixed, 'prepare_words', forbidden_preparation)
    report, packet = _build(args, selected)
    assert packet is None and report['literal_noop']
    assert report['new_factors'] == report['rows'] == report['nnz'] == 0
    assert report['kernel_transform_prepaid'] == report['whole_circuit_emission'] == 0
    assert report['selection_mask_bytes'] == report['selection_mask_entries'] == 0


@pytest.mark.parametrize('malformation', ['dtype', 'rank', 'list'])
def test_selection_requires_complete_boolean_channel_geometry(malformation):
    args, selected = _fixture()
    if malformation == 'dtype':
        selected = selected.astype(np.uint8)
    elif malformation == 'rank':
        selected = selected.reshape(2, 1)
    else:
        selected = selected.tolist()
    with pytest.raises((TypeError, ValueError)):
        _build(args, selected)


def test_disabled_and_zero_budget_paths_inspect_no_numeric_source():
    pool = WorkPool(0)
    assert mixed.construct(None, None, None, None, None, None,
                           selected_channels=None, pool=pool) is None
    assert prove(None, None, None, None, None, None, None,
                 selected_channels=None, pool=pool) is None
    args, selected = _fixture()
    args[0][0, 1, 0, 0] = np.nan
    with pytest.raises(MemoryError):
        _build(args, selected, pool)
    assert pool.used == 0


def test_entire_residual_row_fee_precedes_residual_fraction_materialization(monkeypatch):
    class RefuseResidual(WorkPool):
        def __init__(self):
            super().__init__(256_000_000)
            self.refused = False

        def charge(self, name, amount):
            if name == 'c124_complete_mixed_output_row_and_all_residual_visits':
                self.refused = True
                raise MemoryError('test refuses complete residual row before materialization')
            return super().charge(name, amount)
    floats_seen = []
    def watched_fraction(*values, **keywords):
        if values and type(values[0]) is float:
            floats_seen.append(values[0])
        return F(*values, **keywords)
    monkeypatch.setattr(mixed, 'F', watched_fraction)
    args, selected = _fixture()
    pool = RefuseResidual()
    with pytest.raises(MemoryError):
        _build(args, selected, pool)
    assert pool.refused and floats_seen == []


def test_no_output_demand_preserves_complete_zero_row_coverage():
    args, selected = _fixture('no_outputs')
    report, packet = _build(args, selected)
    proof = _prove(packet, args, selected)
    assert report['rows'] == report['nnz'] == report['new_factors'] == 0
    assert report['residual_positions_visited'] == 0
    assert proof['all_original_output_equations_proved'] == 0
    assert proof['exact_required_support_coverage']


def test_mixed_complete_row_still_rejects_unrepresentable_native_window():
    args, selected = _fixture()
    args[4][:] = 80
    with pytest.raises(ValueError, match='coefficient window'):
        _build(args, selected)
