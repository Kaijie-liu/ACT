"""Four fixed controls; no pretrained model, solver, archive or GPU call."""
from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.solver.solver_hz import sparse_hz_linear
from act.back_end.hybridz_tf.tf_mlp import sparse_hz_apply_relu_exact
from experiments.neural_hz_20260831.definition_first_20260928.d086_native_shared_transfer_20261001 import test_native_shared_transfer as old
from experiments.neural_hz_20260831.definition_first_20260928.d088_native_structure_discovery_20261001 import native_discovery as nd
from experiments.neural_hz_20260831.definition_first_20260928.d088_native_structure_discovery_20261001 import test_native_discovery as discovery_tests
from experiments.neural_hz_20260831.definition_first_20260928.d092_native_observation_relay_20261001 import test_native_observation_relay as dense_tests
from experiments.neural_hz_20260831.definition_first_20260928.d093_sparse_native_relay_20261001 import sparse_observation_relay as sparse
from experiments.neural_hz_20260831.definition_first_20260928.d093_sparse_native_relay_20261001 import native_relay_plan as planner

nb = sparse.nb
ZERO, ONE, HALF = F(0), F(1), F(1, 2)


def _sparse(row):
    return tuple((i, value) for i, value in enumerate(row) if value != ZERO)


def _call(fixture):
    return sparse.append_sparse_observation_relay(
        fixture['hz'], fixture['parents'], fixture['consumers'],
        tuple(_sparse(row) for row in fixture['observations']),
        tuple((graph, _sparse(row)) for graph, row in fixture['next_consumers']),
        enabled=True)


def _same_semantics(left, right):
    assert old._state(left.hz) == old._state(right.hz)
    for name in ('old_n_cont', 'base_n_cont', 'base_row_count', 'shared_columns',
                 'base_residual_bindings', 'observation_bindings',
                 'observation_readouts', 'observation_slots', 'child_slot_bounds',
                 'slot_bounds', 'next_residual_bindings', 'next_slot_bounds',
                 'observation_row_ranges', 'next_upper_row_indices',
                 'exact_rows', 'row_errors', 'installed_rhs',
                 'extra_residual_count', 'physical_bytes', 'nnz'):
        assert getattr(left, name) == getattr(right, name), name


def _plan(hz):
    pool = discovery_tests._Pool(200_000_000)
    reserve = nd.discovery_reserve(hz, pool.limit - pool.used, pool=pool, enabled=True)
    assert reserve > 0
    found = nd.discover(hz, pool=pool, enabled=True)
    planned_reserve = planner.planner_reserve(
        hz, found, pool.limit - pool.used, pool=pool, enabled=True)
    assert planned_reserve > 0
    # Controlled fixture reserve only; this is not a census of a model worker.
    assert (found.summary['discovery_numeric_entries_upper']
            + planned_reserve + old._buffers(hz)) < 64_000_000
    return planner.plan_native_relays(hz, found, pool=pool, enabled=True), found, pool


def _apply_group(hz, group):
    return sparse.append_sparse_observation_relay(
        hz, group.parents, group.consumers, group.observations,
        group.next_consumers, enabled=True)


def _fanout_fixture(count):
    base = old._fixture()
    child_hz, parents, consumers, meta = base
    pre = sparse_hz_linear(child_hz, np.ones((count, 2)), np.full(count, -.5))
    slots = tuple((pre.n_cont + 2*i, pre.n_cont + 2*i + 1, pre.n_bin + i)
                  for i in range(count))
    graphs = tuple(nb.Graph('extended', slot, pre.n_eq + i,
                           (pre.n_ineq + i, pre.n_ineq + count + i))
                   for i, slot in enumerate(slots))
    hz = sparse_hz_apply_relu_exact(pre, [-.5]*count, [1.]*count, slots,
                                    pre.n_cont + 2*count, pre.n_bin + count)
    return dict(base=base, hz=hz, parents=parents, consumers=consumers, meta=meta,
                graphs=graphs, observations=((ONE, ONE),)*count,
                next_consumers=tuple((g, tuple(ONE if i == j else ZERO
                                               for j in range(count)))
                                     for i, g in enumerate(graphs)))


def _duplicate_eta_with_new_phase(hz, graph):
    """Two real authenticated gates may share eta; planner must not choose."""
    bit = hz.n_bin
    eq = graph.eq_row
    neg, pos = graph.le_rows
    old_bit = graph.slots[2]
    padded_ab = sp.hstack((hz.Ab, sp.csr_matrix((hz.n_eq, 1))), format='csr')
    padded_aub = sp.hstack((hz.Aub, sp.csr_matrix((hz.n_ineq, 1))), format='csr')
    added_ab = sp.csr_matrix(([float(hz.Ab[eq, old_bit])], ([0], [bit])),
                             shape=(1, bit + 1))
    added_aub = sp.csr_matrix(([-1., 1.], ([0, 1], [bit, bit])),
                              shape=(2, bit + 1))
    return replace(hz,
        Gb=sp.hstack((hz.Gb, sp.csr_matrix((hz.n_out, 1))), format='csr'),
        Ac=sp.vstack((hz.Ac, hz.Ac[eq:eq+1]), format='csr'),
        Ab=sp.vstack((padded_ab, added_ab), format='csr'),
        b=np.concatenate((hz.b, hz.b[eq:eq+1])),
        Auc=sp.vstack((hz.Auc, hz.Auc[[neg, pos]]), format='csr'),
        Aub=sp.vstack((padded_aub, added_aub), format='csr'),
        ub=np.concatenate((hz.ub, hz.ub[[neg, pos]])))


def test_sparse_matches_dense_joint_semantics():
    ordinary = dense_tests._fixture()
    signed = dense_tests._fixture(wide=True)
    signed['hz'] = replace(signed['hz'], exact=False)
    zero_w = dict(ordinary, observations=((ZERO, ZERO),))
    zero_c = dict(ordinary, next_consumers=((ordinary['third'], (ZERO,)),))
    for fixture in (ordinary, signed, zero_w, zero_c):
        hz = fixture['hz']
        before = old._state(hz)
        expected = dense_tests._call(fixture)
        actual = _call(fixture)
        _same_semantics(actual, expected)
        dense_tests._preserved(hz, actual, before)
        dense_tests._check_installed(hz, actual)
        point = (ZERO, ZERO, ZERO) if fixture is signed else (ZERO, ZERO)
        parent_labels = set()
        for continuous, binary in dense_tests._states(fixture, point):
            extended = dense_tests._extend(fixture, actual, continuous, binary)
            assert extended[:hz.n_cont] == continuous
            assert old._holds(actual.hz, extended, binary)
            assert all(old._eval(row, extended, binary) <= rhs
                       for row, rhs in actual.exact_rows)
            offset = fixture['meta'][1]
            parent_labels.add(binary[offset:offset+2])
        assert len(parent_labels) == 4
    # Exact row/storage identity with the same D092 separating fixture above
    # transports its quantified all-auxiliary gap, not one guessed extension.
    ordinary_result = _call(ordinary)
    final = ordinary_result.next_upper_row_indices[0]
    loss = (ordinary_result.row_errors[final] + ordinary_result.installed_rhs[final]
            - ordinary_result.exact_rows[final][1])
    assert F(1, 24) - loss > F(1, 48)


def test_sparse_identity_avoids_dense_parameter_population():
    count = 64
    fixture = _fanout_fixture(count)
    hz = fixture['hz']
    before = old._state(hz)
    plan, found, pool = _plan(hz)
    assert len(found.graphs) == count + 4 and len(plan.groups) == 1
    group = plan.groups[0]
    assert group.parents == fixture['parents'] and group.consumers == fixture['consumers']
    assert group.observations == (((0, ONE), (1, ONE)),)*count
    assert group.next_consumers == tuple((graph, ((i, ONE),))
                                         for i, graph in enumerate(fixture['graphs']))
    result = _apply_group(hz, group)
    dense_tests._preserved(hz, result, before)
    assert result.local_cost['matrix_parameter_entries'] == 4 + 3*count
    assert result.local_cost['matrix_index_entries'] == 3*count
    assert result.local_cost['matrix_parameter_entries'] * 16 < 4 + 2*count + count*count
    assert len(result.observation_bindings) == len(result.next_upper_row_indices) == count
    assert result.next_residual_bindings == ()
    assert result.hz.n_cont - hz.n_cont == 3 + 3*count
    assert result.rows_added == 14 + 9*count
    assert pool.used < pool.limit
    for continuous, binary in old._original_states(fixture['base'], (ONE, ONE)):
        continuous = (*continuous, *((ONE, -ONE)*count))
        binary = (*binary, *((-ONE,)*count))
        assert old._holds(hz, continuous, binary)
        extended = dense_tests._extend(fixture, result, continuous, binary)
        assert old._holds(result.hz, extended, binary)
        assert all(old._eval(row, extended, binary) <= rhs
                   for row, rhs in result.exact_rows)
    dense_tests._check_installed(hz, result)
    assert all(result.local_cost[key] is False for key in (
        'whole_work_qualified', 'complete_physical_qualified',
        'native_model_qualified', 'gpu_qualified'))


def test_uniform_native_plan_uses_physical_child_scale():
    for wide in (False, True):
        fixture = dense_tests._fixture(wide=wide)
        hz = fixture['hz']
        before = old._state(hz)
        plan, found, pool = _plan(hz)
        assert len(found.graphs) == 5 and len(plan.groups) == 1
        assert plan.owner_id == id(hz)
        group = plan.groups[0]
        weights = (F(9, 8), -ONE) if wide else (ONE, ONE)
        assert group.parents == fixture['parents']
        assert group.consumers == fixture['consumers']
        assert group.observations == (_sparse(weights),)
        assert group.next_consumers == ((fixture['third'], ((0, ONE),)),)
        actual_g, _, _ = nb._extract_graph(hz, fixture['third'])
        for i, (graph, _, _) in enumerate(group.consumers):
            _, q, error = nb._extract_graph(hz, graph)
            assert error == ZERO and q.bias > ZERO
            coefficient = dict(actual_g.continuous)[graph.slots[1]]
            assert weights[i] == -coefficient/q.bias
            if not wide:
                assert q.bias == F(5, 8)
                assert weights[i] != -2*coefficient
        dense_fixture = dict(fixture, observations=(weights,),
                             next_consumers=((fixture['third'], (ONE,)),))
        result = _apply_group(hz, group)
        _same_semantics(result, dense_tests._call(dense_fixture))
        dense_tests._preserved(hz, result, before)
        if wide:
            assert result.base_residual_bindings[0][1] == F(3, 4)
            assert result.next_residual_bindings[0][1] == F(3, 8)
        point = (ONE, ONE, HALF) if wide else (ONE, ONE)
        for continuous, binary in dense_tests._states(dense_fixture, point):
            extended = dense_tests._extend(dense_fixture, result, continuous, binary)
            assert old._holds(result.hz, extended, binary)
            assert all(old._eval(row, extended, binary) <= rhs
                       for row, rhs in result.exact_rows)
        assert pool.used > 0


def test_sparse_plan_and_binding_fail_closed():
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError('disabled argument inspected')
    poison = Poison()
    assert sparse.append_sparse_observation_relay(poison, poison, poison, poison, poison) is None
    assert planner.plan_native_relays(poison, poison, pool=poison) is None
    assert planner.planner_reserve(poison, poison, poison, pool=poison) is None
    with pytest.raises(nb.KernelError):
        sparse.append_sparse_observation_relay(poison, poison, poison, poison, poison, enabled=1)
    with pytest.raises(nb.KernelError):
        planner.plan_native_relays(poison, poison, pool=poison, enabled=1)
    fixture = dense_tests._fixture()
    hz, parents, consumers = fixture['hz'], fixture['parents'], fixture['consumers']
    before = old._state(hz)
    successors = ((fixture['third'], ((0, ONE),)),)
    bad_rows = ([(0, ONE)], ((True, ONE),), ((-1, ONE),), ((2, ONE),),
                ((0, ONE), (0, ONE)), ((1, ONE), (0, ONE)), ((0, ZERO),),
                ((0, 1),), ((0, F(1 << nb.MAX_BITS)),))
    for row in bad_rows:
        with pytest.raises(nb.KernelError):
            sparse.append_sparse_observation_relay(hz, parents, consumers, (row,),
                                                    successors, enabled=True)
        with pytest.raises(nb.KernelError):
            sparse.append_sparse_observation_relay(hz, parents, consumers,
                (((0, ONE), (1, ONE)),), ((fixture['third'], row),), enabled=True)
    with pytest.raises(nb.KernelError):
        sparse.append_sparse_observation_relay(hz, parents, consumers,
            ((),)*(nb.MAX_SUPPORT + 1), successors, enabled=True)
    bad_graph = replace(fixture['third'], slots=(*fixture['third'].slots[:2], parents[0].slots[2]))
    with pytest.raises(nb.KernelError):
        sparse.append_sparse_observation_relay(hz, parents, consumers,
            (((0, ONE), (1, ONE)),), ((bad_graph, ((0, ONE),)),), enabled=True)
    plan, found, _ = _plan(hz)
    assert plan.groups
    with pytest.raises(nb.KernelError):
        planner.plan_native_relays(replace(hz), found, pool=discovery_tests._Pool(), enabled=True)
    with pytest.raises(nb.KernelError):
        planner.plan_native_relays(hz, replace(found, _seal=object()),
                                   pool=discovery_tests._Pool(), enabled=True)
    with pytest.raises(nb.KernelError):
        planner.plan_native_relays(hz, found, pool=discovery_tests._Pool(0), enabled=True)
    bad = hz.b.copy()
    bad[0] = np.inf
    bad_hz = replace(hz, b=bad)
    with pytest.raises(nb.KernelError):
        planner.plan_native_relays(bad_hz, replace(found, owner_id=id(bad_hz)),
                                   pool=discovery_tests._Pool(), enabled=True)
    ambiguous = _duplicate_eta_with_new_phase(hz, consumers[0][0])
    ambiguous_found = nd.discover(ambiguous, pool=discovery_tests._Pool(), enabled=True)
    assert len(ambiguous_found.graphs) == len(found.graphs) + 1
    assert len({g.slots[1] for g in ambiguous_found.graphs}) < len(ambiguous_found.graphs)
    with pytest.raises(nb.KernelError):
        planner.plan_native_relays(ambiguous, ambiguous_found,
                                   pool=discovery_tests._Pool(), enabled=True)
    no_next = old._fixture()[0]
    empty_plan, _, _ = _plan(no_next)
    assert empty_plan.groups == ()
    assert old._state(hz) == before
