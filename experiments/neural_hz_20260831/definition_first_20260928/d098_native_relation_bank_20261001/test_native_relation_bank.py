"""Four fixed mathematical controls; no model, archive, LP, or GPU run.

The finite state generators below enumerate only the legitimate zero labels
of these test fixtures. The candidate contains no phase enumeration/search.
Every new public call receives the same hard 256M fixture pool.
"""
from dataclasses import fields, replace
from fractions import Fraction as F
from types import MappingProxyType

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.solver.solver_hz import (
    sparse_hz_add_same_frame, sparse_hz_concat, sparse_hz_linear,
    sparse_hz_pad_frame,
)
from act.back_end.hybridz_tf.tf_mlp import sparse_hz_apply_relu_exact
from experiments.neural_hz_20260831.definition_first_20260928.d086_native_shared_transfer_20261001 import test_native_shared_transfer as old
from experiments.neural_hz_20260831.definition_first_20260928.d088_native_structure_discovery_20261001.test_native_discovery import _Pool
from experiments.neural_hz_20260831.definition_first_20260928.d092_native_observation_relay_20261001 import test_native_observation_relay as dense_tests
from experiments.neural_hz_20260831.definition_first_20260928.d093_sparse_native_relay_20261001 import test_sparse_native_relay as prior_tests
from experiments.neural_hz_20260831.definition_first_20260928.d098_native_relation_bank_20261001 import native_relation_bank as bank_module

nb = old.nb
ZERO, ONE, HALF = F(0), F(1), F(1, 2)
CAP = 256_000_000


def _build(hz, pool=None):
    return bank_module.build_native_relation_bank(
        hz, pool=_Pool(CAP) if pool is None else pool, enabled=True)


def _apply(hz, pool=None):
    return bank_module.append_native_observation_bank(
        hz, pool=_Pool(CAP) if pool is None else pool, enabled=True)


def _group_view(group):
    return (group.parents, group.consumers, group.observations,
            group.next_consumers)


def _catalog_matches_old(hz, bank):
    """Frozen general parser/discovery/planner is an independent reference."""
    planned, found, pool = prior_tests._plan(hz)
    assert pool.used <= pool.limit
    assert tuple(r.graph for r in bank.relations) == found.graphs
    for relation in bank.relations:
        g, q, error = nb._extract_graph(hz, relation.graph)
        assert error == ZERO
        assert (relation.preactivation, relation.readout) == (g, q)
        assert relation.footprint == nb._graph_shape(
            relation.graph, hz, hz.n_cont, hz.n_bin)
    assert tuple((g.parents, g.consumers) for g in bank.groups) == found.groups
    assert tuple(_group_view(g) for g in bank.groups if g.next_consumers) == tuple(
        _group_view(g) for g in planned.groups)
    assert bank.summary['graphs'] == bank.summary['catalog_graphs'] == len(found.graphs)
    assert bank.summary['groups'] == bank.summary['source_groups'] == len(found.groups)
    assert bank.summary['applicable_groups'] == len(planned.groups)
    assert bank.summary['catalog_group_visits'] == len(found.graphs)*len(found.groups)
    assert bank.summary['native_relation_authentications'] == len(found.graphs)
    assert bank.summary['raw_graph_extractions'] == len(found.graphs)
    assert bank.summary['generic_graph_parser_calls'] == 0


def _states(hz, relations, source_point, source_bits=()):
    """Topologically allocated fixture gates; all original factors preserved."""
    values = list(source_point)+[ZERO]*(hz.n_cont-len(source_point))
    bits = list(source_bits)+[ZERO]*(hz.n_bin-len(source_bits))

    def visit(position):
        if position == len(relations):
            yield tuple(values), tuple(bits)
            return
        relation = relations[position]
        graph = relation.graph
        s, eta, bit_index = graph.slots
        g = old._eval(relation.preactivation, values, bits)
        q = max(ZERO, g)
        L = F.from_float(float(hz.Ac[graph.eq_row, s]))
        Q = relation.readout.bias
        assert L < ZERO < Q
        for bit in old._choices(g):
            values[s] = (q-g)/(-L)-bit
            values[eta] = ONE-q/Q
            bits[bit_index] = bit
            yield from visit(position+1)
    yield from visit(0)


def _readouts(hz, cont, bits):
    return tuple(F.from_float(float(hz.c[i]))+old._dot(hz.Gc, i, cont)
                 +old._dot(hz.Gb, i, bits) for i in range(hz.n_out))


def _same_receipt(receipt, expected):
    assert not hasattr(receipt, 'hz')
    for name in ('old_n_cont', 'base_n_cont', 'base_row_count', 'shared_columns',
                 'base_residual_bindings', 'observation_bindings', 'observation_readouts',
                 'observation_slots', 'child_slot_bounds', 'slot_bounds',
                 'next_residual_bindings', 'next_slot_bounds', 'observation_row_ranges',
                 'next_upper_row_indices', 'exact_rows', 'row_errors', 'installed_rhs',
                 'extra_residual_count', 'physical_bytes', 'nnz'):
        assert getattr(receipt, name) == getattr(expected, name), name


def _extend_group(original_hz, group, receipt, target_width, cont, bits):
    """One group's Gamma, evaluated on the SAME complete old integer state."""
    assert len(cont) == receipt.old_n_cont
    values = list(cont)+[ZERO]*(target_width-len(cont))
    a_phase, b_phase = ((ONE-bits[g.slots[2]])/2 for g in group.parents)
    u, v = ((ONE-cont[g.slots[1]])/2 for g in group.parents)
    delta = a_phase*b_phase
    for column, value in zip(receipt.shared_columns, (delta, b_phase*u, a_phase*v)):
        values[column] = 2*value-ONE
    masses = (ONE-a_phase-b_phase+delta, a_phase-delta, b_phase-delta, delta)
    uf, vf = (nb.NativeAffine(HALF, ((g.slots[1], -HALF),), ())
              for g in group.parents)
    for index, radius, columns in receipt.base_residual_bindings:
        graph, a, b = group.consumers[index]
        g, _, error = nb._extract_graph(original_hz, graph)
        assert error == ZERO
        rest = dense_tests._lin((ONE, g), (-a, uf), (-b, vf))
        residual = nb.NativeAffine(ZERO, rest.continuous, rest.binary)
        value = old._eval(residual, cont, bits)
        assert -radius <= value <= radius
        for column, mass in zip(columns, masses[:3]):
            values[column] = mass*value/radius
    for index, radius, columns in receipt.observation_bindings:
        value = old._eval(receipt.observation_readouts[index], cont, bits)
        assert -radius <= value <= radius
        for column, mass in zip(columns, masses[:3]):
            values[column] = mass*value/radius
    for index, radius, columns in receipt.next_residual_bindings:
        graph, weights = group.next_consumers[index]
        g, _, error = nb._extract_graph(original_hz, graph)
        assert error == ZERO
        rest = dense_tests._lin((ONE, g), *((-weight, receipt.observation_readouts[i])
                                           for i, weight in weights))
        residual = nb.NativeAffine(ZERO, rest.continuous, rest.binary)
        value = old._eval(residual, cont, bits)
        assert -radius <= value <= radius
        for column, mass in zip(columns, masses[:3]):
            values[column] = mass*value/radius
    return tuple(values)


def _fork():
    source = nb.SparseHZono(np.zeros(2), sp.eye(2, format='csr'),
        sp.csr_matrix((2, 0)), sp.csr_matrix((0, 2)), sp.csr_matrix((0, 0)),
        np.zeros(0), frame_id=98003)
    parent = sparse_hz_apply_relu_exact(source, [-1., -1.], [1., 1.],
                                        ((2, 3, 0), (4, 5, 1)), 6, 2)
    left_pre = sparse_hz_linear(parent, np.array([[1., 1.]]), np.array([-.75]))
    left = sparse_hz_apply_relu_exact(left_pre, [-.75], [1.25], ((6, 7, 2),), 8, 3)
    # A SINGLE shared-frame high-water mark: the second fork is padded past
    # every previously allocated column/bit BEFORE its distinct birth.
    right_pre = sparse_hz_pad_frame(
        sparse_hz_linear(parent, np.array([[1., -1.]]), np.array([.25])), 8, 3)
    right = sparse_hz_apply_relu_exact(right_pre, [-.75], [1.25], ((8, 9, 3),), 10, 4)
    return left, right


def _two_applicable_groups():
    fixture = prior_tests._fanout_fixture(64)
    hz = fixture['hz']
    weights = np.zeros((1, 64))
    weights[0, 0], weights[0, 1] = 1., -.5
    pre = sparse_hz_linear(hz, weights, np.array([-.125]))
    # The independent complete readout box is [-5/8,7/8]; no bound oracle.
    slot = (pre.n_cont, pre.n_cont+1, pre.n_bin)
    return sparse_hz_apply_relu_exact(pre, [-.625], [.875], (slot,),
                                      pre.n_cont+2, pre.n_bin+1)


def test_native_catalog_matches_literal_relations():
    for wide in (False, True):
        fixture = dense_tests._fixture(wide=wide)
        hz = fixture['hz']
        before = old._state(hz)
        pool = _Pool(CAP)
        bank = _build(hz, pool)
        assert type(bank.summary) is MappingProxyType
        _catalog_matches_old(hz, bank)
        assert bank.summary['work_charged'] == pool.used > 0
        assert bank.summary['actual_model_qualified'] is False
        assert bank.summary['whole_physical_qualified'] is False
        assert bank.summary['formal_gain'] == 0
        assert old._state(hz) == before

    source = nb.SparseHZono(np.array([.1]), sp.eye(1, format='csr'),
        sp.csr_matrix((1, 0)), sp.csr_matrix((0, 1)), sp.csr_matrix((0, 0)),
        np.zeros(0), frame_id=98001)
    hz = sparse_hz_apply_relu_exact(source, [-.9], [1.1], ((1, 2, 0),), 3, 1)
    before = old._state(hz)
    bank = _build(hz)
    relation, = bank.relations
    graph, actual_g, q = relation.graph, relation.preactivation, relation.readout
    Q = F.from_float(float(hz.c[0]))
    assert actual_g.bias == F.from_float(float(hz.b[graph.eq_row]))+Q
    assert actual_g.bias-F.from_float(.1) == -F(1, 2**55)
    assert actual_g.continuous == ((0, ONE),) and actual_g.binary == ()
    assert q == nb.NativeAffine(Q, ((2, -Q),), ())
    zero_labels = set()
    for x in (-ONE, -actual_g.bias, ZERO, ONE):
        for cont, bits in _states(hz, bank.relations, (x,)):
            assert old._holds(hz, cont, bits)
            assert _readouts(hz, cont, bits) == (max(ZERO, x+actual_g.bias),)
            if x == -actual_g.bias:
                zero_labels.add(bits[0])
                assert x+F.from_float(.1) != ZERO  # not a source-rounding repair
    assert zero_labels == {-ONE, ONE}
    assert old._state(hz) == before


def test_native_bank_survives_actual_merges():
    left, right = _fork()
    left_before, right_before = old._state(left), old._state(right)
    lbank, rbank = _build(left), _build(right)
    known = {r.graph.slots[2]: (r.preactivation, r.readout)
             for bank in (lbank, rbank) for r in bank.relations}
    assert len(known) == 4 and left.n_bin == 3 and right.n_bin == 4
    merged = sparse_hz_add_same_frame(left, right)
    concatenated = sparse_hz_concat((left, right))
    for hz, is_concat in ((merged, False), (concatenated, True)):
        bank = _build(hz)
        _catalog_matches_old(hz, bank)
        assert len(bank.relations) == 4
        assert {r.graph.slots[2]: (r.preactivation, r.readout)
                for r in bank.relations} == known
        right_old = next(r.graph for r in rbank.relations if r.graph.slots[2] == 3)
        right_new = next(r.graph for r in bank.relations if r.graph.slots[2] == 3)
        assert right_new.eq_row != right_old.eq_row
        assert right_new.le_rows != right_old.le_rows
        if is_concat:
            assert bank.summary['duplicate_guard_rows'] >= 4
            assert bank.summary['duplicate_equation_rows'] >= 2
        parent_labels = set()
        for point in ((-ONE, -ONE), (ONE, -ONE), (ZERO, ZERO), (ONE, ONE)):
            for cont, bits in _states(hz, bank.relations, point):
                assert old._holds(hz, cont, bits)
                u, v = (max(ZERO, x) for x in point)
                r, t = max(ZERO, u+v-F(3, 4)), max(ZERO, u-v+F(1, 4))
                assert _readouts(hz, cont, bits) == ((r, t) if is_concat else (r+t,))
                if point == (ZERO, ZERO):
                    parent_labels.add(bits[:2])
        assert len(parent_labels) == 4
    assert old._state(left) == left_before and old._state(right) == right_before
    with pytest.raises(ValueError):
        sparse_hz_add_same_frame(left, replace(right, frame_id=98004))


def test_complete_native_relay_bank():
    # Strictness is inherited by exact equality with frozen D093 rows and
    # outward receipts, whose original quantified gap is 1/24 (not one aux).
    ordinary = dense_tests._fixture()
    applied = _apply(ordinary['hz'])
    plan, _, _ = prior_tests._plan(ordinary['hz'])
    assert len(applied.receipts) == len(plan.groups) == 1
    expected = prior_tests._apply_group(ordinary['hz'], plan.groups[0])
    _same_receipt(applied.receipts[0], expected)
    assert old._state(applied.hz) == old._state(expected.hz)
    final = expected.next_upper_row_indices[0]
    loss = expected.row_errors[final]+expected.installed_rhs[final]-expected.exact_rows[final][1]
    assert F(1, 24)-loss > F(1, 48)

    hz = _two_applicable_groups()
    before = old._state(hz)
    pool = _Pool(CAP)
    applied = _apply(hz, pool)
    bank = applied.bank
    _catalog_matches_old(hz, bank)
    groups = tuple(g for g in bank.groups if g.next_consumers)
    assert len(bank.relations) == 69 and len(bank.groups) == 3
    assert len(groups) == len(applied.receipts) == 2
    assert tuple(len(g.next_consumers) for g in groups) == (64, 1)
    assert len(groups[0].consumers) == 2 and len(groups[1].consumers) == 64
    assert bank.summary['recipients'] == bank.summary['observations'] == 65
    assert bank.summary['groups_without_relay'] == 1
    assert applied.summary['applied_groups'] == 2
    assert applied.summary['native_relation_authentications'] == 69
    assert applied.summary['raw_graph_extractions'] == 69
    assert applied.summary['work_charged'] == pool.used < CAP
    assert applied.hz.n_cont-hz.n_cont == applied.summary['added_cont'] == 201
    assert applied.hz.n_ineq-hz.n_ineq == applied.summary['added_rows'] == 675
    assert applied.hz.n_bin == hz.n_bin == 69
    assert old._state(hz) == before
    references, current = [], hz
    for group, receipt in zip(groups, applied.receipts):
        expected = prior_tests._apply_group(current, group)
        _same_receipt(receipt, expected)
        assert not any(isinstance(getattr(receipt, field.name), nb.SparseHZono)
                       for field in fields(receipt))
        references.append(expected)
        current = expected.hz
    assert old._state(applied.hz) == old._state(current)
    for point in ((-ONE, -ONE), (ZERO, ZERO), (ONE, ONE)):
        for original_cont, bits in _states(hz, bank.relations, point):
            assert old._holds(hz, original_cont, bits)
            extended = original_cont
            for group, receipt, expected in zip(groups, applied.receipts, references):
                extended = _extend_group(hz, group, receipt, expected.hz.n_cont, extended, bits)
                assert extended[:hz.n_cont] == original_cont
                assert old._holds(expected.hz, extended, bits)
                assert all(old._eval(row, extended, bits) <= rhs for row, rhs in receipt.exact_rows)
            assert old._holds(applied.hz, extended, bits)
    assert old._state(hz) == before


def test_native_bank_fail_closed():
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError('disabled path inspected a task argument')
    poison = Poison()
    for function in (bank_module.build_native_relation_bank, bank_module.append_native_observation_bank):
        assert function(poison, pool=poison) is None
        for enabled in (1, None, np.bool_(True)):
            with pytest.raises(nb.KernelError):
                function(poison, pool=poison, enabled=enabled)
    fixture = dense_tests._fixture()
    hz = fixture['hz']
    before = old._state(hz)
    for function in (_build, _apply):
        with pytest.raises(nb.KernelError):
            function(hz, _Pool(0))
        pool = _Pool(CAP)
        function(hz, pool)
        with pytest.raises(nb.KernelError):
            function(hz, _Pool(pool.used-1))
    bad_rhs = hz.b.copy()
    bad_rhs[0] = np.nan
    with pytest.raises(nb.KernelError):
        _build(replace(hz, b=bad_rhs))
    bad_guard = hz.ub.copy()
    bad_guard[fixture['third'].le_rows[0]] = .125
    missing = _build(replace(hz, ub=bad_guard))
    assert missing.summary['missing_guard_bits'] >= 1
    assert fixture['third'].slots[2] not in {r.graph.slots[2] for r in missing.relations}
    ambiguous = prior_tests._duplicate_eta_with_new_phase(hz, fixture['consumers'][0][0])
    for function in (_build, _apply):
        with pytest.raises(nb.KernelError):
            function(ambiguous)
    width = nb.MAX_SUPPORT+1
    wide_values = np.ones(width)
    first_parent = fixture['parents'][0]
    assert first_parent.eq_row == 0
    for column in first_parent.slots[:2]:
        wide_values[column] = float(hz.Ac[first_parent.eq_row, column])
    too_wide = replace(hz,
        Gc=sp.hstack((hz.Gc, sp.csr_matrix((hz.n_out, width-hz.n_cont))), format='csr'),
        Ac=sp.csr_matrix((wide_values, np.arange(width),
                         np.array([0, width]+[width]*(hz.n_eq-1))), shape=(hz.n_eq, width)),
        Auc=sp.hstack((hz.Auc, sp.csr_matrix((hz.n_ineq, width-hz.n_cont))), format='csr'))
    with pytest.raises(nb.KernelError):
        _build(too_wide)
    no_next = old._fixture()[0]
    empty = _apply(no_next)
    assert empty.bank.groups and not any(g.next_consumers for g in empty.bank.groups)
    assert empty.receipts == () and empty.summary['applied_groups'] == 0
    assert old._state(empty.hz) == old._state(no_next)
    assert old._state(hz) == before
