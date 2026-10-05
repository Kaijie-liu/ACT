"""Independent ordinary F4 footprints, raw supports and conditional costs.

Bounds are never source authentication or physical admission.  References use
unpacked coordinate loops and independent BT/AT constants, not implementation
bitsets, class histograms or transform helpers.
"""
import numpy as np
import pytest

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c120_word_f4_v1 import construct
from experiments.neural_hz_20260831.c121_f4_mask_cost_v1 import (
    census, dense_direct, footprints, upper_bill)


_BT = ((4, 0, -5, 0, 1, 0), (0, -4, -4, 1, 1, 0),
       (0, 4, -4, -1, 1, 0), (0, -2, -1, 2, 1, 0),
       (0, 2, -1, -2, 1, 0), (0, 4, 0, -5, 0, 1))
_AT = ((1, 1, 1, 1, 1, 0), (0, 1, -1, 2, -2, 0),
       (0, 1, 1, 4, 4, 0), (0, 1, -1, 8, -8, 1))


def _pack(mask, dtype):
    return np.array([sum(1 << p for p, value in enumerate(channel.flat) if value)
                     for channel in mask], dtype=dtype)


def _tile_masks(c, k, mode):
    inputs = np.ones((c, 6, 6), dtype=bool)
    outputs = np.ones((k, 4, 4), dtype=bool)
    if mode == 'masked':
        inputs[:] = np.arange(inputs.size).reshape(inputs.shape) % 4 != 0
    elif mode == 'partial_output':
        outputs[:] = np.arange(outputs.size).reshape(outputs.shape) % 3 != 0
    elif mode == 'channel_sparse':
        inputs[1::2] = False
        outputs[1::2] = False
    elif mode == 'no_output':
        outputs[:] = False
    return inputs, outputs


def _direct_reference(inputs, outputs):
    total = 0
    for k, y, x in np.ndindex(outputs.shape):
        if outputs[k, y, x]:
            total += 1
            for c, a, b in np.ndindex(inputs.shape[0], 3, 3):
                total += int(inputs[c, y+a, x+b])
    return total


def _support_reference(inputs, outputs):
    ccount, kcount = inputs.shape[0], outputs.shape[0]
    v = {}
    m = {}
    for t, u in np.ndindex(6, 6):
        users = {k: [(i, j) for i, j in np.ndindex(4, 4)
                     if outputs[k, i, j] and _AT[i][t]*_AT[j][u]]
                 for k in range(kcount)}
        active_channels = []
        for c in range(ccount):
            terms = [(i, j) for i, j in np.ndindex(6, 6)
                     if inputs[c, i, j] and _BT[t][i]*_BT[u][j]]
            if terms and any(users.values()):
                v[c, t, u] = terms
                active_channels.append(c)
        for k, positions in users.items():
            if positions and active_channels:
                m[k, t, u] = (active_channels, positions)
    output_rows = int(outputs.sum())
    vnnz = sum(1+len(terms) for terms in v.values())
    mnnz = sum(1+len(channels) for channels, positions in m.values())
    output_nnz = output_rows+sum(len(positions) for channels, positions in m.values())
    auxiliary = len(v)+len(m)
    rows = auxiliary+output_rows
    nnz = vnnz+mnnz+output_nnz
    direct = _direct_reference(inputs, outputs)
    old_bytes = 12*direct+16*output_rows+8
    new_bytes = 12*nnz+88*auxiliary+32*output_rows+72
    return dict(kept_v=len(v), kept_m=len(m), new_factors=auxiliary,
                rows=rows, output_rows=output_rows, nnz_upper=nnz,
                direct_nnz=direct, nnz_saving_lower=direct-nnz,
                declared_old_bytes=old_bytes, declared_new_bytes_upper=new_bytes,
                byte_saving_lower=old_bytes-new_bytes,
                entry_delta_upper=2*(nnz-direct)+13*auxiliary+3*output_rows+8,
                new_emission_work_upper=64*rows+16*(nnz+rows))


@pytest.mark.parametrize('c,k,h,w,padding,mode', [
    (1, 2, 6, 6, 0, 'full'),
    (3, 4, 6, 6, 0, 'masked'),
    (1, 2, 7, 8, 0, 'partial_output'),
    (3, 4, 5, 6, 1, 'full'),
    (3, 4, 7, 7, 1, 'channel_sparse'),
    (1, 2, 6, 6, 0, 'no_output'),
])
def test_all_tile_footprints_equal_independent_unpacked_geometry(c, k, h, w, padding, mode):
    oh, ow = h+2*padding-2, w+2*padding-2
    inputs = np.ones((c, h, w), dtype=bool)
    outputs = np.ones((k, oh, ow), dtype=bool)
    if mode == 'masked':
        inputs[:] = np.arange(inputs.size).reshape(inputs.shape) % 4 != 0
    elif mode == 'partial_output':
        outputs[:] = np.arange(outputs.size).reshape(outputs.shape) % 3 != 0
    elif mode == 'channel_sparse':
        inputs[1::2] = False
        outputs[1::2] = False
    elif mode == 'no_output':
        outputs[:] = False
    before = inputs.copy(), outputs.copy()
    result = footprints(inputs, outputs, (padding, padding),
                        pool=WorkPool(256_000_000), enabled=True)
    expected_positions = [(y, x) for y in range(0, oh, 4) for x in range(0, ow, 4)]
    assert result['positions'].dtype == np.dtype(np.int32)
    assert result['positions'].tolist() == [list(position) for position in expected_positions]
    assert result['input_masks'].dtype == np.dtype(np.uint64)
    assert result['output_masks'].dtype == np.dtype(np.uint16)
    for p, (y, x) in enumerate(expected_positions):
        a = np.zeros((c, 6, 6), dtype=bool)
        o = np.zeros((k, 4, 4), dtype=bool)
        for channel, i, j in np.ndindex(a.shape):
            sy, sx = y-padding+i, x-padding+j
            if 0 <= sy < h and 0 <= sx < w:
                a[channel, i, j] = inputs[channel, sy, sx]
        for channel, i, j in np.ndindex(o.shape):
            if y+i < oh and x+j < ow:
                o[channel, i, j] = outputs[channel, y+i, x+j]
        assert np.array_equal(result['input_masks'][p], _pack(a, np.uint64))
        assert np.array_equal(result['output_masks'][p], _pack(o, np.uint16))
    assert np.array_equal(inputs, before[0]) and np.array_equal(outputs, before[1])


@pytest.mark.parametrize('c,k,mode', [(1, 2, 'dense'), (3, 4, 'dense'),
                                    (3, 4, 'masked'), (2, 3, 'partial_output'),
                                    (3, 4, 'channel_sparse'), (1, 2, 'no_output')])
def test_every_support_count_and_declared_cost_uses_complete_retained_rows(c, k, mode):
    a, o = _tile_masks(c, k, mode)
    expected = _support_reference(a, o)
    ins, outs = _pack(a, np.uint64), _pack(o, np.uint16)
    direct = dense_direct(ins, outs, pool=WorkPool(256_000_000), enabled=True)
    assert direct == expected['direct_nnz']
    result = upper_bill(ins, outs, direct, pool=WorkPool(256_000_000), enabled=True)
    for name, value in expected.items():
        assert result['bill'][name] == value, name
    expected_qualification = (expected['nnz_saving_lower'] > 0
                              and expected['byte_saving_lower'] > 0
                              and expected['entry_delta_upper'] <= 0)
    assert bool(result['topology_qualified']) == expected_qualification
    assert not result['numeric_admission'] and result['formal_gain'] == 0
    assert result['packet_bill_only']
    assert result['actual_whole_HZ_physical_reduction_unproved']


@pytest.mark.parametrize('c,k,mode', [(1, 2, 'dense'), (1, 2, 'masked'),
                                    (3, 4, 'channel_sparse')])
def test_actual_C120_rows_attain_structural_upper_for_noncancelling_geometric_kernel(c, k, mode):
    a, o = _tile_masks(c, k, mode)
    ids = np.arange(a.size, dtype=np.int64).reshape(a.shape)
    outputs = np.arange(a.size, a.size+o.size, dtype=np.int64).reshape(o.shape)
    ids[~a], outputs[~o] = -1, -1
    kernel = np.array([[1, 2, 4], [2, 4, 8], [4, 8, 16]], dtype=np.float64)/32
    weights = np.broadcast_to(kernel, (k, c, 3, 3)).copy()
    report, packet = construct(weights, ids, np.zeros(a.shape, dtype=np.int32),
                               outputs, np.full(o.shape, 10, dtype=np.int32),
                               a.size+o.size, pool=WorkPool(256_000_000), enabled=True)
    expected = _support_reference(a, o)
    assert report['new_factors'] == expected['new_factors']
    assert report['kept_v'] == expected['kept_v']
    assert report['kept_m'] == expected['kept_m']
    assert report['rows'] == expected['rows']
    assert report['nnz'] == len(packet['native']) == expected['nnz_upper']
    assert report['whole_circuit_emission'] == expected['new_emission_work_upper']
    assert not report['live_admission'] and report['score_gain'] == 0


def test_disabled_helpers_inspect_nothing_and_charge_nothing():
    pool = WorkPool(0)
    assert footprints(None, None, None, pool=pool) is None
    assert dense_direct(None, None, pool=pool) is None
    assert upper_bill(None, None, None, pool=pool) is None
    assert census(None, pool=pool) is None
    assert pool.used == 0


def test_zero_budget_fails_before_packing_histograms_or_kernel_scans():
    a, o = _tile_masks(1, 2, 'dense')
    ins, outs = _pack(a, np.uint64), _pack(o, np.uint16)
    for operation in (
        lambda pool: footprints(a, o, (0, 0), pool=pool, enabled=True),
        lambda pool: dense_direct(ins, outs, pool=pool, enabled=True),
        lambda pool: upper_bill(ins, outs, 320, pool=pool, enabled=True),
        lambda pool: census(_nodes(), pool=pool, enabled=True),
    ):
        pool = WorkPool(0)
        with pytest.raises(MemoryError):
            operation(pool)
        assert pool.used == 0


def test_unknown_direct_count_never_establishes_reduction_or_admission():
    a, o = _tile_masks(3, 4, 'dense')
    result = upper_bill(_pack(a, np.uint64), _pack(o, np.uint16), None,
                        pool=WorkPool(256_000_000), enabled=True)
    assert not result['topology_qualified']
    assert not result['numeric_admission'] and result['formal_gain'] == 0
    assert result['bill']['direct_nnz'] is None
    assert result['bill']['nnz_upper'] == _support_reference(a, o)['nnz_upper']


def test_input_bits_outside_six_by_six_geometry_are_rejected():
    ins = np.array([1 << 36], dtype=np.uint64)
    outs = np.array([1], dtype=np.uint16)
    for operation in (
        lambda: dense_direct(ins, outs, pool=WorkPool(256_000_000), enabled=True),
        lambda: upper_bill(ins, outs, 1, pool=WorkPool(256_000_000), enabled=True),
    ):
        with pytest.raises(ValueError):
            operation()


def _nodes(c=1, k=2, h=6, w=6, *, padding=0, stride=1, zero_kernel=False,
           no_output=False, row_mask=None):
    kernel = np.array([[1, 2, 4], [2, 4, 8], [4, 8, 16]], dtype=np.float64)/32
    weights = np.broadcast_to(kernel, (k, c, 3, 3)).copy()
    if zero_kernel:
        weights[0, 0, 0, 0] = 0
    op = ImplicitConv2DOp(weights, (1, c, h, w), padding=padding,
                          stride=stride, row_mask=row_mask)
    nout, nin = op.shape
    needed = np.full(nout, not no_output, dtype=bool)
    return [dict(kind='source', width=nin, needed=np.ones(nin, dtype=bool), parents=()),
            dict(kind='op', width=nout, needed=needed, parents=(0,), op=op)]


def _append_nodes(nodes, branch):
    offset = len(nodes)
    branch[1]['parents'] = (offset,)
    nodes.extend(branch)
    return offset+1


def test_census_covers_every_eligible_operator_tile_and_marks_sparse_kernel_direct_unknown():
    nodes = []
    first = _append_nodes(nodes, _nodes())
    masked = _nodes(c=3, k=4, h=7, w=7, padding=1)
    masked[0]['needed'][::4] = False
    masked[1]['needed'][::3] = False
    second = _append_nodes(nodes, masked)
    _append_nodes(nodes, _nodes(stride=2))
    _append_nodes(nodes, _nodes(no_output=True))
    sparse = _append_nodes(nodes, _nodes(zero_kernel=True))
    snapshots = [node['needed'].copy() for node in nodes]
    report, evidence = census(nodes, pool=WorkPool(256_000_000), enabled=True)
    assert report['eligible_operators'] == 4
    assert report['active_operators'] == 3
    assert report['total_tiles'] == 6
    assert len(report['eligibility']) == len(nodes)
    assert not report['actual_global_admission'] and report['selected_positions'] == []
    assert report['packet_bill_only']
    assert report['actual_whole_HZ_physical_reduction_unproved']
    expected_positions = [(first, 0, 0),
                          *[(second, y, x) for y in (0, 4) for x in (0, 4)],
                          (sparse, 0, 0)]
    assert [(record['node'], record['y'], record['x'])
            for record in report['records']] == expected_positions
    for record in report['records']:
        node_index = record['node']
        positions = evidence[node_index]['positions'].tolist()
        p = positions.index([record['y'], record['x']])
        expected_direct = dense_direct(evidence[node_index]['input_masks'][p],
                                       evidence[node_index]['output_masks'][p],
                                       pool=WorkPool(256_000_000), enabled=True)
        if node_index == sparse:
            assert record['cost']['bill']['direct_nnz'] is None
            assert not record['cost']['topology_qualified']
        else:
            assert record['cost']['bill']['direct_nnz'] == expected_direct
        assert not record['cost']['numeric_admission']
    assert all(np.array_equal(snapshot, node['needed'])
               for snapshot, node in zip(snapshots, nodes, strict=True))
    # Operator row masking is an ordinary source-semantic precondition: a
    # demanded disabled convolution row must not be counted as a dense row.
    row_mask = np.ones(32, dtype=bool)
    row_mask[0] = False
    incompatible = _nodes(row_mask=row_mask)
    with pytest.raises(ValueError):
        census(incompatible, pool=WorkPool(256_000_000), enabled=True)
    incompatible[1]['needed'][0] = False
    compatible, _ = census(incompatible, pool=WorkPool(256_000_000), enabled=True)
    assert compatible['total_tiles'] == 1
    assert compatible['records'][0]['cost']['bill']['output_rows'] == 31


def test_census_no_hit_is_complete_and_does_not_synthesize_a_candidate():
    nodes = []
    _append_nodes(nodes, _nodes(stride=2))
    _append_nodes(nodes, _nodes(no_output=True))
    report, evidence = census(nodes, pool=WorkPool(256_000_000), enabled=True)
    assert report['eligible_operators'] == 1
    assert report['active_operators'] == report['total_tiles'] == 0
    assert report['records'] == [] and report['topology_candidates'] == 0
    assert report['selected_positions'] == [] and not report['actual_global_admission']
    assert len(report['eligibility']) == len(nodes)
    assert evidence == {}
