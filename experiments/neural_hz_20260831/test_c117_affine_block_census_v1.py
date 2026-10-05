import copy
import io
import json

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c24_dense_graph_v1 import graph
from experiments.neural_hz_20260831.c24_dense_ownership_v1 import RADIX
from experiments.neural_hz_20260831 import c117_affine_block_census_v1 as census_module
from experiments.neural_hz_20260831.c117_affine_block_census_v1 import classify
from experiments.neural_hz_20260831.test_c5_live_value_contraction_v1 import source


def run(branches, keep=None):
    first_source, first_ops = branches[0]
    width = first_ops[-1].shape[0] if first_ops else first_source.n_out
    expr = cnn.SparseHZAffineExpr(tuple(cnn.SparseHZAffineTerm(s, tuple(ops)) for s, ops in branches),
                                  np.zeros(width), width, first_source.frame_id)
    keep = np.ones(width, bool) if keep is None else np.array(keep, dtype=bool)
    nodes, _, counts, owners, _ = graph(expr, keep, 96_000_000, uid_start=0)
    before = [(n['needed'].copy(), n['support'].copy(), o.copy()) for n, o in zip(nodes, owners)]
    pool = WorkPool(256_000_000)
    report, evidence = classify(nodes, counts, owners, pool=pool, enabled=True)
    for n, o, (needed, support, old_owned) in zip(nodes, owners, before):
        assert np.array_equal(n['needed'], needed)
        assert np.array_equal(n['support'], support)
        assert np.array_equal(o, old_owned)
    assert report['all_consumer_occurrences'] == sum(
        c['continuous_edges'] for c in counts['node_counts'] if c['kind'] != 'source')
    assert json.loads(json.dumps(report)) == report
    assert report['formal_gain'] == 0 and not report['source_or_live_admission']
    return report, evidence, nodes, counts, owners, pool


def op_index(nodes, op):
    return next(i for i, n in enumerate(nodes) if n['kind'] == 'op' and n['op'] is op)


def test_cloned_csr_full_program_and_complete_ownership():
    s = source(4)
    a = sp.csr_matrix([[1., .5, 0., 0.], [0., 1., 0., .5],
                       [0., 0., 1., 0.], [0., 0., .25, 1.]])
    b = a.copy()
    report, evidence, nodes, counts, owners, _ = run([(s, (a,)), (s, (b,))])
    ai, bi = op_index(nodes, a), op_index(nodes, b)
    assert evidence['operator_classes'][ai] == evidence['operator_classes'][bi]
    assert evidence['program_classes'][ai] == evidence['program_classes'][bi]
    group, = report['repeated_complete_programs']
    assert group['duplicate_needed_row_occurrences'] == 4
    expected = np.zeros(4, np.int64)
    for op in (a, b):
        for row in range(4):
            lo, hi = op.indptr[row:row + 2]
            expected[op.indices[lo:hi][op.data[lo:hi] != 0]] += 1
    assert np.array_equal(evidence['consumer_degrees'][0], expected)
    assert all(np.array_equal(d, o // RADIX) for d, o in zip(evidence['consumer_degrees'], owners))
    assert sum(report['full_consumer_degree_histogram'].values()) == sum(n['width'] for n in nodes)
    # Graph construction may legitimately populate SciPy caches. Remove only
    # these test-created cache flags, then verify the census does not add them.
    for op in (a, b):
        vars(op).pop('_has_canonical_format', None)
        vars(op).pop('_has_sorted_indices', None)
    attributes = [vars(op).copy() for op in (a, b)]
    repeated, _ = classify(nodes, counts, owners, pool=WorkPool(256_000_000), enabled=True)
    assert repeated == report
    for op, old_attributes in zip((a, b), attributes):
        assert set(vars(op)) == set(old_attributes)
        assert all(vars(op)[key] is value for key, value in old_attributes.items())


def test_equal_values_and_frame_do_not_merge_distinct_sources():
    s, a = source(4), sp.eye(4, format='csr')
    t, b = copy.deepcopy(s), a.copy()
    report, evidence, nodes, *_ = run([(s, (a,)), (t, (b,))])
    ai, bi = op_index(nodes, a), op_index(nodes, b)
    assert s.frame_id == t.frame_id and np.array_equal(s.c, t.c)
    assert evidence['operator_classes'][ai] == evidence['operator_classes'][bi]
    assert evidence['program_classes'][ai] != evidence['program_classes'][bi]
    assert not report['repeated_complete_programs']


def test_changed_csr_coefficient_is_different_complete_operator():
    s, a = source(4), sp.eye(4, format='csr')
    b = a.copy()
    b.data[0] = .5
    report, evidence, nodes, *_ = run([(s, (a,)), (s, (b,))])
    ai, bi = op_index(nodes, a), op_index(nodes, b)
    assert evidence['operator_classes'][ai] != evidence['operator_classes'][bi]
    assert evidence['program_classes'][ai] != evidence['program_classes'][bi]


def test_cloned_convolution_uses_full_unexpanded_payload(monkeypatch):
    s = source(6)
    a = ImplicitConv2DOp(np.array([[[[.5]]]]), (1, 1, 2, 3))
    b = ImplicitConv2DOp(a._kernel.copy(), a.input_shape)
    def forbidden(*args, **kwargs):
        raise AssertionError('complete content census must not expand convolution rows')
    monkeypatch.setattr(ImplicitConv2DOp, '_row', forbidden)
    monkeypatch.setattr(ImplicitConv2DOp, 'to_csr_reference', forbidden)
    report, evidence, nodes, *_ = run([(s, (a,)), (s, (b,))])
    ai, bi = op_index(nodes, a), op_index(nodes, b)
    assert evidence['program_classes'][ai] == evidence['program_classes'][bi]
    assert report['nodes'][ai]['operator_geometry']['kernel_shape'] == [1, 1, 1, 1]


@pytest.mark.parametrize('change', ['kernel', 'stride', 'padding', 'dilation', 'input_shape', 'mask', 'mask_presence'])
def test_full_convolution_geometry_and_mask_are_part_of_equivalence(change):
    kernel, shape, kwargs = np.array([[[[.5]]]]), (1, 1, 2, 3), {}
    a = ImplicitConv2DOp(kernel, shape)
    if change == 'kernel':
        kernel = np.array([[[[.25]]]])
    elif change == 'stride':
        kwargs['stride'] = (1, 2)
    elif change == 'padding':
        kwargs['padding'] = (1, 0)
    elif change == 'dilation':
        kwargs['dilation'] = (2, 1)
    elif change == 'input_shape':
        shape = (1, 1, 3, 2)
    elif change == 'mask':
        kwargs['row_mask'] = np.array([True, False, True, True, True, True])
    else:
        kwargs['row_mask'] = np.ones(6, bool)
    b = ImplicitConv2DOp(kernel, shape, **kwargs)
    s = source(6)
    tail_a = sp.csr_matrix(np.ones((1, a.shape[0])))
    tail_b = sp.csr_matrix(np.ones((1, b.shape[0])))
    report, evidence, nodes, *_ = run([(s, (a, tail_a)), (s, (b, tail_b))])
    ai, bi = op_index(nodes, a), op_index(nodes, b)
    assert evidence['operator_classes'][ai] != evidence['operator_classes'][bi]
    assert evidence['program_classes'][ai] != evidence['program_classes'][bi]


def test_convolution_groups_and_kernel_axes_are_recorded():
    s = source(8)
    a = ImplicitConv2DOp(np.ones((2, 1, 1, 1)), (1, 2, 2, 2), groups=2)
    b = ImplicitConv2DOp(np.ones((2, 2, 1, 1)), (1, 2, 2, 2), groups=1)
    report, evidence, nodes, *_ = run([(s, (a,)), (s, (b,))])
    ai, bi = op_index(nodes, a), op_index(nodes, b)
    assert evidence['operator_classes'][ai] != evidence['operator_classes'][bi]
    assert report['nodes'][ai]['operator_geometry']['groups'] == 2
    assert report['nodes'][bi]['operator_geometry']['groups'] == 1


@pytest.mark.parametrize('overlap', [False, True])
def test_complete_equivalent_program_needed_row_union(overlap):
    s, a = source(4), sp.eye(4, format='csr')
    b = a.copy()
    x = sp.csr_matrix([[1., 1., 0., 0.]])
    y = sp.csr_matrix([[0., 1., 1., 0.]] if overlap else [[0., 0., 1., 1.]])
    report, _, _, *_ = run([(s, (a, x)), (s, (b, y))])
    group, = report['repeated_complete_programs']
    assert group['needed_row_occurrences'] == 4
    assert group['needed_row_union'] == (3 if overlap else 4)
    assert group['duplicate_needed_row_occurrences'] == int(overlap)
    assert group['candidate_only']


@pytest.mark.parametrize('duplicate', [False, True])
def test_sum_equivalence_commutes_but_preserves_parent_multiplicity(duplicate):
    s, t = source(4), source(4)
    a, b = sp.eye(4, format='csr'), sp.eye(4, format='csr')
    branches = [(s, (a,)), (t, (a,)), (t, (b,)), (s, (b,))]
    if duplicate:
        branches.append((s, (b,)))
    report, evidence, nodes, *_ = run(branches)
    ai, bi = op_index(nodes, a), op_index(nodes, b)
    assert (evidence['program_classes'][ai] == evidence['program_classes'][bi]) == (not duplicate)
    pa, pb = nodes[ai]['parents'][0], nodes[bi]['parents'][0]
    assert nodes[pa]['kind'] == nodes[pb]['kind'] == 'sum'
    assert len(nodes[pb]['parents']) == (3 if duplicate else 2)
    assert (evidence['program_classes'][pa] == evidence['program_classes'][pb]) == (not duplicate)


def test_all_nodes_including_unneeded_rows_are_observed():
    s, a = source(4), sp.eye(4, format='csr')
    report, evidence, nodes, *_ = run([(s, (a,))], keep=[True, False, True, False])
    assert report['complete_node_count'] == 2
    assert sum(report['full_consumer_degree_histogram'].values()) == 8
    assert sum(report['needed_consumer_degree_histogram'].values()) == 4
    assert all(len(mask) == 4 for mask in evidence['needed_masks'])


def test_default_off_does_not_access_inputs():
    assert classify(object(), object(), object(), pool=object()) is None


def test_budget_is_charged_before_node_observation():
    nodes = [{'width': 4, 'parents': (), 'needed': object(), 'support': object()}]
    with pytest.raises(MemoryError):
        classify(nodes, [{}], [object()], pool=WorkPool(0), enabled=True)


def test_payload_budget_precedes_content_bytes_copy(monkeypatch):
    s, a = source(4), sp.eye(4, format='csr')
    _, _, nodes, counts, owners, _ = run([(s, (a,))])
    only_node_observation = sum(256 + 16 * (3 * n['width'] + len(n['parents'])) for n in nodes)
    def forbidden(array):
        raise AssertionError('payload bytes read before the full payload budget')
    monkeypatch.setattr(census_module, '_array_key', forbidden)
    with pytest.raises(MemoryError, match='operator_payload'):
        classify(nodes, counts, owners, pool=WorkPool(only_node_observation), enabled=True)


def test_consumer_population_must_equal_complete_non_source_edges():
    s, a = source(4), sp.eye(4, format='csr')
    _, _, nodes, counts, owners, _ = run([(s, (a,))])
    incomplete = copy.deepcopy(counts)
    incomplete['node_counts'][-1]['continuous_edges'] -= 1
    with pytest.raises(ValueError, match='consumer occurrence sum'):
        classify(nodes, incomplete, owners, pool=WorkPool(256_000_000), enabled=True)


def test_mixed_width_complete_evidence_archives_without_object_arrays():
    from experiments.neural_hz_20260831.c117_affine_block_worker_v1 import archive_arrays
    s, a = source(4), sp.eye(4, format='csr')
    b, tail = a.copy(), sp.csr_matrix([[1., .5, 0., 0.]])
    _, evidence, *_ = run([(s, (a, tail)), (s, (b, tail.copy()))])
    assert {len(mask) for mask in evidence['needed_masks']} == {1, 4}
    assert evidence['equivalent_program_needed_coverage']
    every_array = []
    def inspect(value):
        if type(value) is np.ndarray:
            every_array.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                inspect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                inspect(item)
    inspect(evidence)
    flattened = archive_arrays(evidence)
    assert sorted(map(id, flattened.values())) == sorted(map(id, every_array))
    assert all(not value.dtype.hasobject for value in flattened.values())
    with io.BytesIO() as stream:
        np.savez(stream, **flattened)
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as restored:
            assert set(restored.files) == set(flattened)
            for name, expected in flattened.items():
                actual = restored[name]
                assert actual.dtype == expected.dtype and actual.shape == expected.shape
                assert np.array_equal(actual, expected)
