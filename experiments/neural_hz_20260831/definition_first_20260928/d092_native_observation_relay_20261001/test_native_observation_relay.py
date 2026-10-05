"""Four fixed mathematical controls on original SparseHZono predicates."""
from dataclasses import replace
from fractions import Fraction as F
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.solver.solver_hz import sparse_hz_linear
from act.back_end.hybridz_tf.tf_mlp import sparse_hz_apply_relu_exact
from experiments.neural_hz_20260831.definition_first_20260928.d086_native_shared_transfer_20261001 import test_native_shared_transfer as old
from experiments.neural_hz_20260831.definition_first_20260928.d092_native_observation_relay_20261001 import native_observation_relay as relay

nb, nt = old.nb, old.nt
ZERO, ONE, HALF = F(0), F(1), F(1, 2)


def _fixture(wide=False):
    base = old._fixture(wide=wide)
    parent_hz, parents, consumers, meta = base
    observations = ((ONE, -HALF), (F(-1, 4), ONE)) if wide else ((ONE, ONE),)
    coefficients = (ONE, -HALF) if wide else (ONE,)
    weights = tuple(sum((c * row[j] for c, row in zip(coefficients, observations)), ZERO)
                    for j in range(len(consumers)))
    bias = F(1, 8) if wide else -HALF
    pre = sparse_hz_linear(parent_hz, np.array([[float(w) for w in weights]]),
                           np.array([float(bias)]))
    if wide:
        # The NEXT residual, like the first residual, includes both original
        # continuous and original binary coordinates, not a fresh error box.
        extra_c = sp.csr_matrix(([.25], ([0], [2])), shape=pre.Gc.shape)
        extra_b = sp.csr_matrix(([.125], ([0], [0])), shape=pre.Gb.shape)
        pre = replace(pre, Gc=(pre.Gc + extra_c).tocsr(),
                      Gb=(pre.Gb + extra_b).tocsr())
    lower, upper = (F(-4), F(4)) if wide else (-HALF, ONE)
    slots = (pre.n_cont, pre.n_cont + 1, pre.n_bin)
    graph = nb.Graph('extended', slots, pre.n_eq, (pre.n_ineq, pre.n_ineq + 1))
    hz = sparse_hz_apply_relu_exact(pre, [float(lower)], [float(upper)],
                                    (slots,), pre.n_cont + 2, pre.n_bin + 1)
    return dict(base=base, hz=hz, parents=parents, consumers=consumers,
                observations=observations, next_consumers=((graph, coefficients),),
                third=graph, lower=lower, upper=upper, meta=meta)


def _call(fixture):
    return relay.append_observation_relay(
        fixture['hz'], fixture['parents'], fixture['consumers'],
        fixture['observations'], fixture['next_consumers'], enabled=True)


def _states(fixture, point, upstream=ONE):
    hz = fixture['hz']
    g, _, error = nb._extract_graph(hz, fixture['third'])
    assert error == ZERO
    for continuous, binary in old._original_states(fixture['base'], point, upstream):
        padded, padded_b = (*continuous, ZERO, ZERO), (*binary, ZERO)
        value = old._eval(g, padded, padded_b)
        output = max(ZERO, value)
        for bit in old._choices(value):
            xi1 = (output - value) / (-fixture['lower'] / 2) - bit
            xi2 = ONE - output / (fixture['upper'] / 2)
            yield (*continuous, xi1, xi2), (*binary, bit)


def _lin(*terms, bias=ZERO):
    return nb._combine(bias, tuple(terms))


def _constant(value):
    return nb.NativeAffine(value, (), ())


def _le(left, right):
    difference = _lin((ONE, left), (-ONE, right))
    return (nb.NativeAffine(ZERO, difference.continuous, difference.binary),
            -difference.bias)


def _extend(fixture, result, continuous, binary):
    original = (fixture['hz'], fixture['parents'], fixture['consumers'], fixture['meta'])
    proxy = SimpleNamespace(hz=result.hz, shared_columns=result.shared_columns,
                            residual_bindings=result.base_residual_bindings)
    extended = list(old._extend(original, proxy, continuous, binary))
    alpha, beta = ((ONE - binary[g.slots[2]]) / 2 for g in fixture['parents'])
    delta = alpha * beta
    masses = (ONE - alpha - beta + delta, alpha - delta, beta - delta, delta)
    for index, radius, columns in result.observation_bindings:
        value = old._eval(result.observation_readouts[index], continuous, binary)
        assert radius > ZERO
        for col, mass in zip(columns, masses[:3]):
            extended[col] = mass * value / radius
    for index, radius, columns in result.next_residual_bindings:
        graph, coefficients = fixture['next_consumers'][index]
        actual, _, _ = nb._extract_graph(fixture['hz'], graph)
        remainder = _lin((ONE, actual), *((-c, readout) for c, readout in
                         zip(coefficients, result.observation_readouts)))
        residual = nb.NativeAffine(ZERO, remainder.continuous, remainder.binary)
        value = old._eval(residual, continuous, binary)
        assert -radius <= value <= radius
        for col, mass in zip(columns, masses[:3]):
            extended[col] = mass * value / radius
    return tuple(extended)


def _preserved(hz, result, before):
    assert old._state(hz) == before
    assert result.old_n_cont == hz.n_cont
    assert result.hz.n_bin == hz.n_bin and result.hz.n_eq == hz.n_eq
    assert result.hz.frame_id == hz.frame_id and result.hz.exact is hz.exact
    for name in ('c', 'b'):
        assert np.array_equal(getattr(hz, name), getattr(result.hz, name))
    for name in ('Gc', 'Ac'):
        assert (getattr(result.hz, name)[:, :hz.n_cont] != getattr(hz, name)).nnz == 0
        assert getattr(result.hz, name)[:, hz.n_cont:].nnz == 0
    for name in ('Gb', 'Ab'):
        assert (getattr(result.hz, name) != getattr(hz, name)).nnz == 0
    assert (result.hz.Auc[:hz.n_ineq, :hz.n_cont] != hz.Auc).nnz == 0
    assert result.hz.Auc[:hz.n_ineq, hz.n_cont:].nnz == 0
    assert (result.hz.Aub[:hz.n_ineq] != hz.Aub).nnz == 0
    assert np.array_equal(result.hz.ub[:hz.n_ineq], hz.ub)


def _check_installed(hz, result):
    assert len(result.exact_rows) == len(result.row_errors) == len(result.installed_rhs)
    assert len(result.exact_rows) == result.hz.n_ineq - hz.n_ineq
    for index, ((row, rhs), error, installed) in enumerate(zip(
            result.exact_rows, result.row_errors, result.installed_rhs)):
        actual = nb._row(result.hz.Auc, result.hz.Aub, hz.n_ineq + index)
        assert row.bias == ZERO
        assert nb._norm(_lin((ONE, row), (-ONE, actual))) == error
        assert installed == F.from_float(float(result.hz.ub[hz.n_ineq + index]))
        assert installed >= rhs + error


def test_native_joint_realization_and_zero_phases():
    fixture = _fixture()
    hz = fixture['hz']
    before = old._state(hz)
    result = _call(fixture)
    assert hz.n_cont == 12 and hz.n_bin == 5  # Third gate belongs to H0 already.
    _preserved(hz, result, before)
    assert result.base_row_count == 14
    assert result.hz.n_cont == hz.n_cont + 6
    assert result.hz.n_ineq == hz.n_ineq + 23
    assert result.slot_bounds == (((F(1, 4), F(1, 4)), (F(1, 4), F(3, 2)),
                                  (ZERO, F(1, 4)), (ZERO, F(15, 8))),)
    assert result.next_slot_bounds == (((-F(1, 4), -F(1, 4)),
                                       (-F(1, 4), ONE), (-HALF, -F(1, 4)),
                                       (-HALF, F(11, 8))),)
    parent_labels, r_zero, t_zero, w_zero = set(), set(), set(), set()
    points = ((-ONE, -ONE), (ZERO, ZERO), (ONE, ONE),
              (F(1, 2), F(1, 4)), (ZERO, F(1, 4)), (F(1, 4), -HALF))
    for point in points:
        for continuous, binary in _states(fixture, point):
            assert old._holds(hz, continuous, binary)
            extended = _extend(fixture, result, continuous, binary)
            assert extended[:hz.n_cont] == continuous
            assert old._holds(result.hz, extended, binary)
            assert all(old._eval(row, extended, binary) <= rhs
                       for row, rhs in result.exact_rows)
            if point == (ZERO, ZERO):
                parent_labels.add(binary[:2])
            if point == (F(1, 2), F(1, 4)):
                r_zero.add(binary[2])
                w_zero.add(binary[4])
            if point == (ZERO, F(1, 4)):
                t_zero.add(binary[3])
    assert len(parent_labels) == 4
    assert r_zero == t_zero == w_zero == {-ONE, ONE}
    _check_installed(hz, result)


def test_third_relu_separates_all_auxiliary_extensions():
    fixture = _fixture()
    hz, parents = fixture['hz'], fixture['parents']
    continuous = (ZERO, ZERO, HALF, HALF, HALF, HALF,
                  ONE, HALF, F(1, 5), F(3, 10), ONE, ZERO)
    binary = (ZERO, ZERO, HALF, F(3, 10), ZERO)
    assert old._holds(hz, continuous, binary, integer=False)
    baseline = nt.append_shared_upper(hz, parents, fixture['consumers'], enabled=True)
    old_extended = list(continuous) + [ZERO] * (baseline.hz.n_cont - hz.n_cont)
    for col, value in zip(baseline.shared_columns, (ZERO, -HALF, -HALF)):
        old_extended[col] = value
    assert old._holds(baseline.hz, tuple(old_extended), binary, integer=False)
    result = _call(fixture)
    alpha = nb.NativeAffine(HALF, (), ((parents[0].slots[2], -HALF),))
    beta = nb.NativeAffine(HALF, (), ((parents[1].slots[2], -HALF),))
    u = nb.NativeAffine(HALF, ((parents[0].slots[1], -HALF),), ())
    delta, zu, _ = (nb.NativeAffine(HALF, ((col, HALF),), ())
                    for col in result.shared_columns)
    zero = _constant(ZERO)
    masses = (_lin((-ONE, alpha), (-ONE, beta), (ONE, delta), bias=ONE),
              _lin((ONE, alpha), (-ONE, delta)),
              _lin((ONE, beta), (-ONE, delta)), delta)
    u10 = _lin((ONE, u), (-ONE, zu))
    slots = result.observation_slots[0]
    lower = (_lin((F(1, 4), masses[0])),
             _lin((F(5, 4), u10), (F(1, 16), masses[1])),
             _lin((F(-1, 8), masses[2])),
             _lin((F(5, 4), zu), (F(-5, 16), masses[3])))
    upper = (_lin((F(1, 4), masses[0])),
             _lin((F(5, 4), u10), (F(1, 4), masses[1])),
             _lin((F(1, 4), masses[2])),
             _lin((F(5, 4), zu), (F(5, 8), masses[3])))
    observation_rows = result.exact_rows[result.base_row_count:result.base_row_count + 8]
    expected = tuple(row for slot, lo, hi in zip(slots, lower, upper)
                     for row in (_le(lo, slot), _le(slot, hi)))
    assert len(observation_rows) == len(expected)
    assert all(row in observation_rows for row in expected)
    z = result.observation_readouts[0]
    assert _lin(*((ONE, slot) for slot in slots)) == z
    summed_upper = _lin(*((ONE, bound) for bound in upper))
    assert summed_upper == _lin((F(5, 4), u), (F(3, 8), delta), bias=F(1, 4))
    assert _le(delta, alpha) in result.exact_rows[:result.base_row_count]
    assert _le(zero, u10) in result.exact_rows[:result.base_row_count]
    assert _le(u10, masses[1]) in result.exact_rows[:result.base_row_count]
    _, w, error = nb._extract_graph(hz, fixture['third'])
    assert error == ZERO
    final_index, = result.next_upper_row_indices
    assert final_index == len(result.exact_rows) - 1
    final_expected = _le(w, _lin((F(4, 5), slots[1]), (F(-1, 5), masses[1]),
                                 (F(11, 15), slots[3])))
    assert result.exact_rows[final_index] == final_expected
    # Every front row is exactly stored. Its algebra forces delta=1/2,
    # then pi10=pi01=u10=0, Z00=1/8 and Z11=5/8 for ANY extension.
    assert all(error == ZERO for error in result.row_errors[:final_index])
    assert all(stored == rhs for (_, rhs), stored in zip(
        result.exact_rows[:final_index], result.installed_rhs[:final_index]))
    z_value = old._eval(z, continuous, binary)
    u_value = old._eval(u, continuous, binary)
    forced_delta = (z_value - F(1, 4) - F(5, 4) * u_value) / F(3, 8)
    assert forced_delta == old._eval(alpha, continuous, binary) == HALF
    assert old._eval(beta, continuous, binary) == HALF
    forced_z00 = F(1, 4) * (ONE - HALF - HALF + forced_delta)
    forced_z11 = z_value - forced_z00
    assert forced_z00 == F(1, 8) and forced_z11 == F(5, 8)
    exact_gap = old._eval(w, continuous, binary) - F(11, 15) * forced_z11
    assert exact_gap == F(1, 24)
    # Coefficient error bounds hold for ALL unit latent assignments, not just
    # one guessed set of new columns; RHS rounding is charged separately.
    _, exact_rhs = result.exact_rows[final_index]
    rounding_loss = (result.row_errors[final_index]
                     + result.installed_rhs[final_index] - exact_rhs)
    assert exact_gap - rounding_loss > F(1, 48)
    _check_installed(hz, result)


def test_signed_observation_and_full_residuals():
    fixture = _fixture(wide=True)
    fixture['hz'] = replace(fixture['hz'], exact=False)
    hz = fixture['hz']
    before = old._state(hz)
    result = _call(fixture)
    _preserved(hz, result, before)
    assert len(result.base_residual_bindings) == 1
    assert result.base_residual_bindings[0][1] == F(3, 4)
    assert len(result.next_residual_bindings) == 1
    assert result.next_residual_bindings[0][1] == F(3, 8)
    assert len(result.observation_bindings) == 2
    assert result.hz.n_cont - hz.n_cont == 15
    assert result.hz.n_ineq - hz.n_ineq == 47
    assert any(error > ZERO for error in result.row_errors)
    assert any(any(index == 0 for index, _ in row.binary)
               for row, _ in result.exact_rows)
    for point in ((-ONE, -ONE, -ONE), (ZERO, ZERO, ZERO),
                  (ONE, ONE, ONE), (HALF, F(1, 4), -HALF)):
        for upstream in (-ONE, ONE):
            for continuous, binary in _states(fixture, point, upstream):
                assert old._holds(hz, continuous, binary)
                extended = _extend(fixture, result, continuous, binary)
                assert extended[:hz.n_cont] == continuous and extended[:3] == point
                assert old._holds(result.hz, extended, binary)
                assert all(old._eval(row, extended, binary) <= rhs
                           for row, rhs in result.exact_rows)
                alpha, beta = ((ONE - binary[g.slots[2]]) / 2 for g in fixture['parents'])
                state_index = {(ZERO, ZERO): 0, (ONE, ZERO): 1,
                               (ZERO, ONE): 2, (ONE, ONE): 3}[alpha, beta]
                for readout, bounds in zip(result.observation_readouts, result.slot_bounds):
                    value = old._eval(readout, continuous, binary)
                    lo, hi = bounds[state_index]
                    assert lo <= value <= hi
    _check_installed(hz, result)


def test_default_off_binding_and_resource_rejections():
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError('disabled argument was inspected')
    poison = Poison()
    assert relay.append_observation_relay(poison, poison, poison, poison, poison) is None
    with pytest.raises(nb.KernelError):
        relay.append_observation_relay(poison, poison, poison, poison, poison, enabled=1)
    fixture = _fixture()
    hz, parents, consumers = fixture['hz'], fixture['parents'], fixture['consumers']
    observations, successors = fixture['observations'], fixture['next_consumers']
    before = old._state(hz)
    rejected = (
        ((parents[0], parents[0]), consumers, observations, successors),
        (parents, consumers, ((ONE,),), successors),
        (parents, consumers, ((1, ONE),), successors),
        (parents, consumers, ((F(1 << nb.MAX_BITS), ONE),), successors),
        (parents, consumers, observations, ((fixture['third'], (ONE, ONE)),)),
        (parents, consumers, observations * (nb.MAX_SUPPORT + 1), successors),
    )
    for p, c, w, n in rejected:
        with pytest.raises(nb.KernelError):
            relay.append_observation_relay(hz, p, c, w, n, enabled=True)
    bad_graph = replace(fixture['third'], slots=(*fixture['third'].slots[:2], parents[0].slots[2]))
    with pytest.raises(nb.KernelError):
        relay.append_observation_relay(hz, parents, consumers, observations,
                                       ((bad_graph, (ONE,)),), enabled=True)
    bad_rhs = hz.ub.copy()
    bad_rhs[fixture['third'].le_rows[0]] += .125
    with pytest.raises(nb.KernelError):
        relay.append_observation_relay(replace(hz, ub=bad_rhs), parents, consumers,
                                       observations, successors, enabled=True)
    nonfinite = hz.c.copy()
    nonfinite[0] = np.nan
    with pytest.raises(nb.KernelError):
        relay.append_observation_relay(replace(hz, c=nonfinite), parents, consumers,
                                       observations, successors, enabled=True)
    assert old._state(hz) == before
    result = _call(fixture)
    assert result.physical_bytes['input'] == old._buffers(hz)
    assert result.physical_bytes['output'] == old._buffers(result.hz)
    added_c = result.hz.Auc[hz.n_ineq:]
    added_b = result.hz.Aub[hz.n_ineq:]
    assert result.nnz['input'] == old._nnz(hz)
    assert result.nnz['output'] == old._nnz(result.hz)
    assert result.nnz['added'] == added_c.nnz + added_b.nnz
    assert result.nnz['exact_added'] == sum(len(row.continuous) + len(row.binary)
                                           for row, _ in result.exact_rows)
    assert result.physical_bytes['input_plus_intermediate_plus_output'] == sum(
        result.physical_bytes[key] for key in ('input', 'intermediate', 'output'))
    for start, stop, key in (
            (hz.n_ineq, hz.n_ineq + result.base_row_count, 'base_generated_row_buffers'),
            (hz.n_ineq + result.base_row_count, result.hz.n_ineq, 'relay_generated_row_buffers')):
        matrices = (result.hz.Auc[start:stop], result.hz.Aub[start:stop])
        actual_bytes = result.hz.ub[start:stop].nbytes + sum(
            matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
            for matrix in matrices)
        assert result.physical_bytes[key] == actual_bytes
    for key in ('selected_input_occurrences', 'matrix_parameter_entries',
                'observation_core_input_occurrences', 'new_combination_input_occurrences',
                'generated_row_occurrences'):
        assert 0 < result.local_cost[key] <= nb.MAX_SUPPORT
    assert result.local_cost['graph_extractions_including_base'] == 9
    assert all(result.local_cost[key] is False for key in (
        'whole_work_qualified', 'complete_physical_qualified',
        'native_model_qualified', 'gpu_qualified'))
    _check_installed(hz, result)
    zero_fixture = dict(fixture, observations=((ZERO, ZERO),))
    zero_result = _call(zero_fixture)
    assert zero_result.observation_bindings == ()
    assert zero_result.observation_readouts == (_constant(ZERO),)
    assert zero_result.observation_slots == ((_constant(ZERO),) * 4,)
    assert zero_result.slot_bounds == (((ZERO, ZERO),) * 4,)
    assert len(zero_result.next_residual_bindings) == 1
    assert zero_result.hz.n_cont == zero_result.base_n_cont + 3
    for continuous, binary in _states(zero_fixture, (ONE, ONE)):
        extended = _extend(zero_fixture, zero_result, continuous, binary)
        assert old._holds(zero_result.hz, extended, binary)
        assert all(old._eval(row, extended, binary) <= rhs
                   for row, rhs in zero_result.exact_rows)
    assert old._state(hz) == before
