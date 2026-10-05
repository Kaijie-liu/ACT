"""Independent ordinary tests for one exact-summand channel-routing rule.

The oracle unpacks coordinates and uses its own transform constants.  Exhaustive
subsets below prove a small synthetic separable-cost identity; they are not a
verification search, input-domain split, or target-instance experiment.  No
packet cost is interpreted as native HZ admission or a formal score gain.
"""
import itertools

import numpy as np
import pytest

from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c122_channel_route_v1 import route


_BT = ((4, 0, -5, 0, 1, 0), (0, -4, -4, 1, 1, 0),
       (0, 4, -4, -1, 1, 0), (0, -2, -1, 2, 1, 0),
       (0, 2, -1, -2, 1, 0), (0, 4, 0, -5, 0, 1))
_AT = ((1, 1, 1, 1, 1, 0), (0, 1, -1, 2, -2, 0),
       (0, 1, 1, 4, 4, 0), (0, 1, -1, 8, -8, 1))


def _pack(mask, dtype):
    return np.array([sum(1 << p for p, live in enumerate(channel.flat) if live)
                     for channel in mask], dtype=dtype)


def _ordinary(mode, c=None, k=None):
    if c is None:
        c = 7 if mode == 'dense' else 4
    if k is None:
        k = 128 if mode in ('dense', 'heterogeneous', 'channel_sparse') else 32
    inputs = np.ones((c, 6, 6), dtype=bool)
    outputs = np.ones((k, 4, 4), dtype=bool)
    if mode == 'masked':
        inputs[:] = np.arange(inputs.size).reshape(inputs.shape) % 4 != 0
    elif mode == 'heterogeneous':
        for channel in range(c):
            if channel % 3 == 1:
                inputs[channel] = False
                inputs[channel, 2, 2] = True
            elif channel % 3 == 2:
                inputs[channel] = False
    elif mode == 'partial_output':
        outputs[:] = np.arange(outputs.size).reshape(outputs.shape) % 3 != 0
    elif mode == 'border':
        inputs[:, 0, :] = False
        inputs[:, :, 0] = False
        outputs[:, 2:, :] = False
        outputs[:, :, 3:] = False
    elif mode == 'channel_sparse':
        inputs[1::2] = False
    elif mode == 'no_output':
        outputs[:] = False
    elif mode == 'no_input':
        inputs[:] = False
    return inputs, outputs


def _reference(inputs, outputs):
    """Return independent per-channel costs and complete mixed-row counts."""
    ccount, kcount = len(inputs), len(outputs)
    output_counts = [[sum(int(outputs[k, y, x]) for k in range(kcount))
                      for x in range(4)] for y in range(4)]
    output_rows = sum(sum(row) for row in output_counts)
    direct_channels = [sum(output_counts[y][x] * int(inputs[c, y+a, x+b])
                           for y in range(4) for x in range(4)
                           for a in range(3) for b in range(3))
                       for c in range(ccount)]
    support_counts = []
    filter_counts = []
    output_uses = []
    for t in range(6):
        for u in range(6):
            support_counts.append([
                sum(int(inputs[c, y, x]) for y in range(6) for x in range(6)
                    if _BT[t][y] and _BT[u][x])
                for c in range(ccount)])
            users = [sum(int(outputs[k, y, x]) for y in range(4) for x in range(4)
                         if _AT[y][t] and _AT[x][u])
                     for k in range(kcount)]
            filter_counts.append(sum(count > 0 for count in users))
            output_uses.append(sum(users))
    stats = []
    for c in range(ccount):
        forms = [t for t in range(36)
                 if support_counts[t][c] and filter_counts[t]]
        a = len(forms)
        l = sum(support_counts[t][c] for t in forms)
        b = sum(filter_counts[t] for t in forms)
        d = direct_channels[c]
        q = l+a+b-d
        stats.append((d, a, l, b, q, 12*q+88*a, 2*q+13*a))
    selected = tuple(row[5] < 0 for row in stats)
    active = tuple(bool(filter_counts[t]) and
                   any(selected[c] and support_counts[t][c] for c in range(ccount))
                   for t in range(36))
    direct = output_rows+sum(direct_channels)

    def bill(fixed):
        if not any(selected):
            return dict(kept_v=0, kept_m=0, new_factors=0,
                        output_rows=output_rows, rows=0,
                        direct_nnz=direct, nnz_upper=direct,
                        delta_nnz_upper=0, byte_delta_upper=0,
                        entry_delta_upper=0, new_emission_work_upper=0,
                        v_nnz_upper=0, m_nnz_upper=0, output_terms_upper=0,
                        residual_direct_terms=sum(direct_channels),
                        byte_saving_after_route_mask_lower=0,
                        entry_delta_with_route_mask_upper=0)
        residual = sum(direct_channels[c] for c in range(ccount) if not selected[c])
        vnnz = sum(stats[c][2]+stats[c][1] for c in range(ccount) if selected[c])
        connections = sum(stats[c][3] for c in range(ccount) if selected[c])
        if fixed:
            kept_v = sum(stats[c][1] for c in range(ccount) if selected[c])
            kept_m = sum(filter_counts)
            output_terms = sum(output_uses)
            nnz = direct+sum(stats[c][4] for c in range(ccount) if selected[c])
            nnz += kept_m+output_terms
        else:
            kept_v = sum(sum(selected[c] and bool(support_counts[t][c])
                             for c in range(ccount))
                         for t in range(36) if active[t])
            kept_m = sum(filter_counts[t] for t in range(36) if active[t])
            output_terms = sum(output_uses[t] for t in range(36) if active[t])
            nnz = output_rows+residual
            for t in range(36):
                if not active[t]:
                    continue
                channels = [c for c in range(ccount)
                            if selected[c] and support_counts[t][c]]
                nnz += sum(support_counts[t][c]+1 for c in channels)
                nnz += filter_counts[t]*(len(channels)+1)+output_uses[t]
        auxiliary = kept_v+kept_m
        delta = nnz-direct
        byte_delta = 12*delta+88*auxiliary+16*output_rows+64
        entries = 2*delta+13*auxiliary+3*output_rows+8
        return dict(kept_v=kept_v, kept_m=kept_m, new_factors=auxiliary,
                    output_rows=output_rows, rows=auxiliary+output_rows,
                    direct_nnz=direct, nnz_upper=nnz,
                    delta_nnz_upper=delta, byte_delta_upper=byte_delta,
                    entry_delta_upper=entries,
                    new_emission_work_upper=16*nnz+80*(auxiliary+output_rows),
                    v_nnz_upper=vnnz, m_nnz_upper=connections+kept_m,
                    output_terms_upper=output_terms, residual_direct_terms=residual,
                    byte_saving_after_route_mask_lower=-byte_delta-ccount,
                    entry_delta_with_route_mask_upper=entries+ccount)

    return dict(stats=stats, selected=selected, active=active, direct=direct,
                fixed=bill(True), tight=bill(False), output_rows=output_rows,
                filter_counts=filter_counts, output_uses=output_uses)


def _invoke(inputs, outputs, direct=None, *, use_reference_direct=True):
    expected = _reference(inputs, outputs)
    if use_reference_direct:
        direct = expected['direct']
    actual = route(_pack(inputs, np.uint64), _pack(outputs, np.uint16), direct,
                   pool=WorkPool(256_000_000), enabled=True)
    return expected, actual


@pytest.mark.parametrize('mode', [
    'dense', 'masked', 'heterogeneous', 'partial_output', 'border', 'channel_sparse',
])
def test_complete_mixed_support_and_costs_equal_unpacked_reference(mode):
    inputs, outputs = _ordinary(mode)
    expected, (report, evidence) = _invoke(inputs, outputs)
    assert evidence['channel_stats'].dtype == np.dtype(np.int64)
    assert evidence['channel_stats'].tolist() == [list(row) for row in expected['stats']]
    assert evidence['selected'].dtype == np.dtype(bool)
    assert evidence['selected'].tolist() == list(expected['selected'])
    assert evidence['active_components'].tolist() == list(expected['active'])
    assert report['selected_channels'] == sum(expected['selected'])
    assert report['residual_channels'] == len(inputs)-sum(expected['selected'])
    for field in ('fixed', 'tight'):
        for name, value in expected[field].items():
            assert report[field+'_bill'][name] == value, (field, name)
    mask_size = len(inputs) if any(expected['selected']) else 0
    assert report['route_mask_bytes'] == report['route_mask_entries'] == mask_size
    tight = expected['tight']
    qualifies = (any(expected['selected']) and tight['delta_nnz_upper'] < 0
                 and tight['byte_saving_after_route_mask_lower'] > 0
                 and tight['entry_delta_with_route_mask_upper'] <= 0)
    assert bool(report['conditional_candidate']) == qualifies
    assert bool(report['topology_qualified']) == qualifies
    assert report['packet_bill_only']
    assert report['actual_whole_HZ_physical_reduction_unproved']
    assert not report['numeric_admission'] and not report['actual_global_admission']
    assert report['formal_gain'] == 0


def test_seven_dense_channels_match_pre_registered_joint_sanity_not_native_admission():
    inputs, outputs = _ordinary('dense', c=7, k=128)
    expected, (report, evidence) = _invoke(inputs, outputs)
    assert all(row[4] == -13304 and row[5] == -156480 for row in expected['stats'])
    assert evidence['selected'].all()
    for label in ('fixed_bill', 'tight_bill'):
        bill = report[label]
        assert bill['delta_nnz_upper'] == -47048
        assert bill['byte_delta_upper'] == -104064
        assert bill['entry_delta_upper'] == -24764
        assert bill['new_factors'] == 4860
        assert bill['byte_saving_after_route_mask_lower'] == 104064-7
        assert bill['entry_delta_with_route_mask_upper'] == -24764+7
    assert report['conditional_candidate']
    assert not report['numeric_admission'] and not report['actual_global_admission']
    assert report['formal_gain'] == 0


@pytest.mark.parametrize('mode', ['dense', 'heterogeneous', 'partial_output'])
def test_all_negative_channel_weights_minimize_fixed_bound_over_tiny_subsets(mode):
    inputs, outputs = _ordinary(mode, c=3, k=128)
    expected, (report, evidence) = _invoke(inputs, outputs)
    stats = expected['stats']
    fixed_overhead = (12*(sum(expected['filter_counts'])+sum(expected['output_uses']))
                      +88*sum(expected['filter_counts'])+16*expected['output_rows']+64)
    candidate_costs = [fixed_overhead+sum(row[5] for row, keep in zip(stats, subset) if keep)
                       for subset in itertools.product((False, True), repeat=len(inputs))]
    selected_cost = fixed_overhead+sum(row[5] for row, keep in
                                       zip(stats, expected['selected']) if keep)
    assert selected_cost == min(candidate_costs)
    assert evidence['selected'].tolist() == [row[5] < 0 for row in stats]
    for row, keep in zip(stats, expected['selected']):
        if keep:
            assert row[4] < 0 and row[6] < 0
    if any(expected['selected']):
        assert report['fixed_bill']['byte_delta_upper'] == selected_cost
        assert report['tight_bill']['byte_delta_upper'] <= selected_cost
    # Empty selection is a literal no-op, not an allocation of this artificial
    # fixed overhead.  The subset statement is only about the frozen bound.


@pytest.mark.parametrize('mode', ['no_output', 'no_input', 'unprofitable'])
def test_empty_selection_keeps_literal_direct_representation_without_packet(mode):
    inputs, outputs = _ordinary(mode, c=1, k=1)
    expected, (report, evidence) = _invoke(inputs, outputs)
    assert not evidence['selected'].any() and not evidence['active_components'].any()
    assert report['selected_channels'] == 0 and report['residual_channels'] == 1
    assert report['route_mask_bytes'] == report['route_mask_entries'] == 0
    for label in ('fixed_bill', 'tight_bill'):
        bill = report[label]
        assert bill['nnz_upper'] == expected['direct']
        assert bill['new_factors'] == 0
        for name in ('delta_nnz_upper', 'byte_delta_upper', 'entry_delta_upper',
                     'new_emission_work_upper', 'byte_saving_after_route_mask_lower',
                     'entry_delta_with_route_mask_upper'):
            assert bill[name] == 0
    assert not report['conditional_candidate'] and not report['topology_qualified']


def test_unknown_direct_never_selects_or_authenticates_a_structural_candidate():
    inputs, outputs = _ordinary('dense', c=7, k=128)
    _, (report, evidence) = _invoke(inputs, outputs, use_reference_direct=False)
    assert not evidence['selected'].any()
    assert report['selected_channels'] == 0
    assert not report['conditional_candidate'] and not report['topology_qualified']
    assert not report['numeric_admission'] and not report['actual_global_admission']
    assert report['formal_gain'] == 0


def test_wrong_direct_count_is_rejected_not_used_to_manufacture_savings():
    inputs, outputs = _ordinary('dense', c=2, k=3)
    direct = _reference(inputs, outputs)['direct']
    with pytest.raises(ValueError):
        route(_pack(inputs, np.uint64), _pack(outputs, np.uint16), direct+1,
              pool=WorkPool(256_000_000), enabled=True)


def test_disabled_route_inspects_nothing_and_charges_nothing():
    pool = WorkPool(0)
    assert route(None, None, None, pool=pool) is None
    assert pool.used == 0


def test_zero_budget_rejects_before_reading_out_of_geometry_input_values():
    pool = WorkPool(0)
    with pytest.raises(MemoryError):
        route(np.array([1 << 36], dtype=np.uint64), np.array([1], dtype=np.uint16),
              1, pool=pool, enabled=True)
    assert pool.used == 0


@pytest.mark.parametrize('malformation', ['high_bit', 'input_dtype', 'output_dtype', 'rank'])
def test_invalid_bit_geometry_or_typed_shape_fails_closed(malformation):
    inputs = np.array([1], dtype=np.uint64)
    outputs = np.array([1], dtype=np.uint16)
    if malformation == 'high_bit':
        inputs[0] = 1 << 36
    elif malformation == 'input_dtype':
        inputs = inputs.astype(np.int64)
    elif malformation == 'output_dtype':
        outputs = outputs.astype(np.uint32)
    else:
        inputs = inputs.reshape(1, 1)
    with pytest.raises((TypeError, ValueError)):
        route(inputs, outputs, 2, pool=WorkPool(256_000_000), enabled=True)


def test_input_masks_are_immutable_and_new_evidence_does_not_alias_them():
    inputs, outputs = _ordinary('heterogeneous', c=3, k=128)
    ins, outs = _pack(inputs, np.uint64), _pack(outputs, np.uint16)
    before_in, before_out = ins.copy(), outs.copy()
    report, evidence = route(ins, outs, _reference(inputs, outputs)['direct'],
                             pool=WorkPool(256_000_000), enabled=True)
    assert np.array_equal(ins, before_in) and np.array_equal(outs, before_out)
    for array in evidence.values():
        assert isinstance(array, np.ndarray) and array.flags.owndata
        assert not np.shares_memory(array, ins) and not np.shares_memory(array, outs)
    assert report['packet_bill_only'] and report['formal_gain'] == 0
