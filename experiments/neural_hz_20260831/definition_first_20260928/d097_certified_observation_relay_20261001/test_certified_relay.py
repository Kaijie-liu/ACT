"""Four fixed controls for birth-certified common observation consumption."""
from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.solver.solver_hz import sparse_hz_linear
from experiments.neural_hz_20260831.definition_first_20260928.d086_native_shared_transfer_20261001 import test_native_shared_transfer as old
from experiments.neural_hz_20260831.definition_first_20260928.d088_native_structure_discovery_20261001.test_native_discovery import _Pool
from experiments.neural_hz_20260831.definition_first_20260928.d092_native_observation_relay_20261001 import test_native_observation_relay as rt
from experiments.neural_hz_20260831.definition_first_20260928.d093_sparse_native_relay_20261001 import sparse_observation_relay as prior
from experiments.neural_hz_20260831.definition_first_20260928.d096_verified_phase_birth_20261001 import phase_birth as pb
from experiments.neural_hz_20260831.definition_first_20260928.d096_verified_phase_birth_20261001 import test_phase_birth as bt
from experiments.neural_hz_20260831.definition_first_20260928.d097_certified_observation_relay_20261001 import certified_relay as cr

nb = pb.nb
ZERO, ONE, HALF = F(0), F(1), F(1, 2)


def _fixture(wide=False, count=1):
    assert not wide or count == 1
    nc, nbin = (3, 1) if wide else (2, 0)
    source = nb.SparseHZono(np.zeros(2),
        sp.csr_matrix(([1., 1.], ([0, 1], [0, 1])), shape=(2, nc)),
        sp.csr_matrix((2, nbin)), sp.csr_matrix((0, nc)),
        sp.csr_matrix((0, nbin)), np.zeros(0), frame_id=97001, exact=not wide)
    first = bt._call(source)
    pre = sparse_hz_linear(first.hz, np.array([[1., 1.], [1., -1.]]), np.array([-.75, .25]))
    if wide:
        pre = replace(pre,
            Gc=(pre.Gc+sp.csr_matrix(([.5], ([0], [2])), shape=pre.Gc.shape)).tocsr(),
            Gb=(pre.Gb+sp.csr_matrix(([.25], ([0], [0])), shape=pre.Gb.shape)).tocsr())
    second = bt._call(pre, first.carriers)
    observations = (((0, ONE), (1, -HALF)), ((0, F(-1, 4)), (1, ONE))) if wide else (
        ((0, ONE), (1, ONE)),)*count
    coefficients = ((0, ONE), (1, -HALF)) if wide else ((0, ONE),)
    weights = (F(9, 8), -ONE) if wide else (ONE, ONE)
    pre_next = sparse_hz_linear(second.hz,
        np.array([[float(v) for v in weights]]*count), np.full(count, .125 if wide else -.5))
    if wide:
        pre_next = replace(pre_next,
            Gc=(pre_next.Gc+sp.csr_matrix(([.25], ([0], [2])), shape=pre_next.Gc.shape)).tocsr(),
            Gb=(pre_next.Gb+sp.csr_matrix(([.125], ([0], [0])), shape=pre_next.Gb.shape)).tocsr())
    third = bt._call(pre_next, first.carriers)
    parents = tuple(g.graph for g in first.gates)
    consumers = ((second.gates[0].graph, ONE, ONE), (second.gates[1].graph, ONE, -ONE))
    successors = tuple((g.graph, coefficients if wide else ((i, ONE),))
                       for i, g in enumerate(third.gates))
    return dict(hz=third.hz, certificates=(*first.gates, *second.gates, *third.gates),
                carriers=first.carriers, parents=parents, consumers=consumers,
                observations=observations, next_consumers=successors,
                births=(first, second, third), wide=wide, source=source)


def _call(fixture, pool=None):
    return cr.append_certified_observation_relay(
        fixture['hz'], fixture['certificates'], fixture['carriers'],
        fixture['parents'], fixture['consumers'], fixture['observations'],
        fixture['next_consumers'], pool=_Pool(256000000) if pool is None else pool,
        enabled=True)


def _reference_hz(fixture):
    # An independent test-only equivalent presentation with identical factor
    # indices: eliminate carrier coefficients in the BIRTH equations only.
    # Dyadic fixtures make c-Q exactly float64. No candidate input is mutated.
    hz = fixture['hz']
    rows = [dict(nb._row(hz.Ac, hz.Ab, i).continuous) for i in range(hz.n_eq)]
    rhs = hz.b.copy()
    for gate in fixture['certificates']:
        row = rows[gate.graph.eq_row]
        for column in fixture['carriers'].columns:
            row.pop(column, None)
        value = gate.preactivation.bias-gate.readout.bias
        assert F.from_float(float(value)) == value
        rhs[gate.graph.eq_row] = float(value)
    values, columns, pointers = [], [], [0]
    for row in rows:
        for column, amount in sorted(row.items()):
            columns.append(column)
            values.append(float(amount))
        pointers.append(len(values))
    ac = sp.csr_matrix((np.array(values), np.array(columns), np.array(pointers)), shape=hz.Ac.shape)
    reference = replace(hz, Ac=ac, b=rhs)
    for gate in fixture['certificates']:
        g, q, error = nb._extract_graph(reference, gate.graph)
        assert (g, q, error) == (gate.preactivation, gate.readout, ZERO)
    return reference


def _states(fixture, point, upstream=ONE):
    first, second, third = fixture['births']
    bits = (upstream,) if fixture['wide'] else ()
    for c1, b1 in bt._extensions(first, point, bits):
        for c2, b2 in bt._extensions(second, c1, b1):
            yield from bt._extensions(third, c2, b2)


def _extend(fixture, result, cont, bits):
    original_nc = fixture['hz'].n_cont
    values = list(cont)+[ZERO]*(result.hz.n_cont-original_nc)
    parents = fixture['parents']
    alpha, beta = ((ONE-bits[g.slots[2]])/2 for g in parents)
    u, v = ((ONE-cont[g.slots[1]])/2 for g in parents)
    delta = alpha*beta
    for col, value in zip(result.shared_columns, (delta, beta*u, alpha*v)):
        values[col] = 2*value-ONE
    masses = (ONE-alpha-beta+delta, alpha-delta, beta-delta, delta)
    certified = {g.graph: g.preactivation for g in fixture['certificates']}
    uf = nb.NativeAffine(HALF, ((parents[0].slots[1], -HALF),), ())
    vf = nb.NativeAffine(HALF, ((parents[1].slots[1], -HALF),), ())
    for index, radius, columns in result.base_residual_bindings:
        graph, a, b = fixture['consumers'][index]
        rest = rt._lin((ONE, certified[graph]), (-a, uf), (-b, vf))
        residual = nb.NativeAffine(ZERO, rest.continuous, rest.binary)
        value = old._eval(residual, cont, bits)
        assert -radius <= value <= radius
        for col, mass in zip(columns, masses[:3]):
            values[col] = mass*value/radius
    for index, radius, columns in result.observation_bindings:
        value = old._eval(result.observation_readouts[index], cont, bits)
        assert -radius <= value <= radius
        for col, mass in zip(columns, masses[:3]):
            values[col] = mass*value/radius
    for index, radius, columns in result.next_residual_bindings:
        graph, coefficients = fixture['next_consumers'][index]
        rest = rt._lin((ONE, certified[graph]),
            *((-amount, result.observation_readouts[i]) for i, amount in coefficients))
        residual = nb.NativeAffine(ZERO, rest.continuous, rest.binary)
        value = old._eval(residual, cont, bits)
        assert -radius <= value <= radius
        for col, mass in zip(columns, masses[:3]):
            values[col] = mass*value/radius
    return tuple(values)


def test_born_relations_match_zero_constant_reference():
    for wide in (False, True):
        fixture = _fixture(wide=wide)
        hz = fixture['hz']
        before = old._state(hz)
        pool = _Pool(256000000)
        result = _call(fixture, pool)
        reference = _reference_hz(fixture)
        expected = prior.append_sparse_observation_relay(reference, fixture['parents'],
            fixture['consumers'], fixture['observations'], fixture['next_consumers'], enabled=True)
        for name in ('old_n_cont', 'base_n_cont', 'base_row_count', 'shared_columns',
                     'base_residual_bindings', 'observation_bindings', 'observation_readouts',
                     'observation_slots', 'child_slot_bounds', 'slot_bounds', 'next_residual_bindings',
                     'next_slot_bounds', 'observation_row_ranges', 'next_upper_row_indices',
                     'exact_rows', 'row_errors', 'installed_rhs', 'extra_residual_count'):
            assert getattr(result, name) == getattr(expected, name), name
        rt._preserved(hz, result, before)
        rt._check_installed(hz, result)
        assert result.local_cost['certificate_verifications'] == len(fixture['certificates'])
        assert sum(name == 'birth_verify_metadata' for name, _ in pool.charges) == len(fixture['certificates'])
        assert result.local_cost['raw_graph_extractions'] == 0
        assert result.local_cost['work_charged'] == pool.used > 0
        if wide:
            assert result.base_residual_bindings[0][1] == F(3, 4)
            assert result.next_residual_bindings[0][1] == F(3, 8)
        else:
            assert not result.base_residual_bindings and not result.next_residual_bindings
            # Raw carrier expressions are valid but have a spurious 7/8 box
            # residual for the first child; the certified representative has 0.
            graph, a, b = fixture['consumers'][0]
            raw, _, _ = nb._extract_graph(hz, graph)
            u, v = (nb.NativeAffine(HALF, ((g.slots[1], -HALF),), ()) for g in fixture['parents'])
            remainder = rt._lin((ONE, raw), (-a, u), (-b, v))
            assert nb._norm(nb.NativeAffine(ZERO, remainder.continuous, remainder.binary)) == F(7, 8)
        reversed_result = _call(dict(fixture, certificates=fixture['certificates'][::-1]))
        assert reversed_result.exact_rows == result.exact_rows
        assert old._state(reversed_result.hz) == old._state(result.hz)


def test_common_extension_and_strict_next_layer_gain():
    for wide in (False, True):
        fixture = _fixture(wide=wide)
        hz, parents = fixture['hz'], fixture['parents']
        result = _call(fixture)
        points = ((-ONE, -ONE), (ZERO, ZERO), (ONE, ONE),
                  (HALF, F(1, 4)), (ZERO, F(1, 4)), (F(1, 4), -HALF))
        labels, r_zero, t_zero, w_zero = set(), set(), set(), set()
        for p in points:
            point = (*p, -HALF) if wide else p
            for upstream in ((-ONE, ONE) if wide else (ONE,)):
                for cont, bits in _states(fixture, point, upstream):
                    assert old._holds(hz, cont, bits)
                    extended = _extend(fixture, result, cont, bits)
                    assert extended[:hz.n_cont] == cont and extended[:len(point)] == point
                    assert old._holds(result.hz, extended, bits)
                    assert all(old._eval(row, extended, bits) <= rhs for row, rhs in result.exact_rows)
                    if not wide:
                        if p == (ZERO, ZERO):
                            labels.add(bits[:2])
                        if p == (HALF, F(1, 4)):
                            r_zero.add(bits[2]); w_zero.add(bits[4])
                        if p == (ZERO, F(1, 4)):
                            t_zero.add(bits[3])
        if not wide:
            assert len(labels) == 4 and r_zero == t_zero == w_zero == {-ONE, ONE}

    fixture = _fixture()
    hz, parents = fixture['hz'], fixture['parents']
    assert (hz.n_cont, hz.n_bin, hz.n_eq, hz.n_ineq) == (14, 5, 7, 10)
    assert fixture['births'][2].gates[0].upper == F(2)  # actual box, not old tight ub=1
    cont = (ZERO, ZERO, ONE, ONE, HALF, HALF, HALF, HALF,
            ONE, HALF, F(1, 5), F(3, 10), ONE, HALF)
    bits = (ZERO, ZERO, HALF, F(3, 10), ZERO)
    assert old._holds(hz, cont, bits, integer=False)
    baseline = old.nt.append_shared_upper(_reference_hz(fixture), parents, fixture['consumers'], enabled=True)
    previous = list(cont)+[ZERO]*(baseline.hz.n_cont-hz.n_cont)
    for col, value in zip(baseline.shared_columns, (ZERO, -HALF, -HALF)):
        previous[col] = value
    assert old._holds(baseline.hz, tuple(previous), bits, integer=False)
    result = _call(fixture)
    alpha = nb.NativeAffine(HALF, (), ((parents[0].slots[2], -HALF),))
    beta = nb.NativeAffine(HALF, (), ((parents[1].slots[2], -HALF),))
    u = nb.NativeAffine(HALF, ((parents[0].slots[1], -HALF),), ())
    delta, zu, _ = (nb.NativeAffine(HALF, ((col, HALF),), ()) for col in result.shared_columns)
    masses = (rt._lin((-ONE, alpha), (-ONE, beta), (ONE, delta), bias=ONE),
              rt._lin((ONE, alpha), (-ONE, delta)), rt._lin((ONE, beta), (-ONE, delta)), delta)
    u10 = rt._lin((ONE, u), (-ONE, zu))
    slots = result.observation_slots[0]
    lower = (rt._lin((F(1, 4), masses[0])), rt._lin((F(5, 4), u10), (F(1, 16), masses[1])),
             rt._lin((F(-1, 8), masses[2])), rt._lin((F(5, 4), zu), (F(-5, 16), masses[3])))
    upper = (rt._lin((F(1, 4), masses[0])), rt._lin((F(5, 4), u10), (F(1, 4), masses[1])),
             rt._lin((F(1, 4), masses[2])), rt._lin((F(5, 4), zu), (F(5, 8), masses[3])))
    rows = result.exact_rows[result.base_row_count:result.base_row_count+8]
    assert all(row in rows for slot, lo, hi in zip(slots, lower, upper)
               for row in (rt._le(lo, slot), rt._le(slot, hi)))
    z = result.observation_readouts[0]
    assert rt._lin(*((ONE, slot) for slot in slots)) == z
    assert rt._lin(*((ONE, bound) for bound in upper)) == rt._lin(
        (F(5, 4), u), (F(3, 8), delta), bias=F(1, 4))
    assert rt._le(delta, alpha) in result.exact_rows[:result.base_row_count]
    assert rt._le(rt._constant(ZERO), u10) in result.exact_rows[:result.base_row_count]
    assert rt._le(u10, masses[1]) in result.exact_rows[:result.base_row_count]
    w = fixture['births'][2].gates[0].readout
    final, = result.next_upper_row_indices
    assert result.exact_rows[final] == rt._le(w, rt._lin(
        (F(4, 5), slots[1]), (F(-1, 5), masses[1]), (F(11, 15), slots[3])))
    assert all(error == ZERO for error in result.row_errors[:final])
    assert all(stored == rhs for (_, rhs), stored in zip(result.exact_rows[:final], result.installed_rhs[:final]))
    z_value, u_value = old._eval(z, cont, bits), old._eval(u, cont, bits)
    forced_delta = (z_value-F(1, 4)-F(5, 4)*u_value)/F(3, 8)
    assert forced_delta == old._eval(alpha, cont, bits) == old._eval(beta, cont, bits) == HALF
    forced_z00 = F(1, 4)*(ONE-HALF-HALF+forced_delta)
    forced_z11 = z_value-forced_z00
    assert (forced_z00, forced_z11) == (F(1, 8), F(5, 8))
    gap = old._eval(w, cont, bits)-F(11, 15)*forced_z11
    assert gap == F(1, 24)
    loss = result.row_errors[final]+result.installed_rhs[final]-result.exact_rows[final][1]
    assert gap-loss > F(1, 48)
    rt._check_installed(hz, result)


def test_full_bank_reuses_authenticated_relations():
    fixture = _fixture(count=64)
    hz = fixture['hz']
    before = old._state(hz)
    pool = _Pool(256000000)
    result = _call(fixture, pool)
    assert len(fixture['certificates']) == 68
    assert result.local_cost['unique_birth_certificates'] == result.local_cost['certificate_verifications'] == 68
    assert sum(name == 'birth_verify_metadata' for name, _ in pool.charges) == 68
    assert result.local_cost['raw_graph_extractions'] == 0
    assert result.local_cost['graph_extractions_including_base'] == 0
    assert result.local_cost['work_charged'] == pool.used < 256000000
    assert len(result.next_upper_row_indices) == len(result.observation_bindings) == 64
    assert result.hz.n_cont-hz.n_cont == 195
    assert result.hz.n_ineq-hz.n_ineq == 590
    assert result.hz.n_bin == hz.n_bin == 68
    assert result.local_cost['matrix_parameter_entries'] == 196
    assert result.local_cost['matrix_index_entries'] == 192
    rt._preserved(hz, result, before)
    rt._check_installed(hz, result)
    assert result.physical_bytes['input'] == old._buffers(hz)
    assert result.physical_bytes['output'] == old._buffers(result.hz)
    for point in ((-ONE, -ONE), (ONE, ONE)):
        for cont, bits in _states(fixture, point):
            extended = _extend(fixture, result, cont, bits)
            assert old._holds(result.hz, extended, bits)
            assert all(old._eval(row, extended, bits) <= rhs for row, rhs in result.exact_rows)
    for key in ('whole_work_qualified', 'complete_physical_qualified', 'native_model_qualified', 'gpu_qualified'):
        assert result.local_cost[key] is False


def test_certified_binding_and_resource_fail_closed():
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError('disabled argument inspected')
    poison = Poison()
    assert cr.append_certified_observation_relay(poison, poison, poison, poison, poison, poison, poison, pool=poison) is None
    with pytest.raises(nb.KernelError):
        cr.append_certified_observation_relay(poison, poison, poison, poison, poison, poison, poison, pool=poison, enabled=1)
    fixture = _fixture()
    before = old._state(fixture['hz'])
    certs = fixture['certificates']
    for bad in (certs[:-1], (*certs, certs[0]), (certs[0],)*len(certs),
                (replace(certs[0], preactivation=replace(certs[0].preactivation, bias=ONE)), *certs[1:])):
        with pytest.raises(nb.KernelError):
            _call(dict(fixture, certificates=bad))
    for field, value in (('parents', (fixture['parents'][0],)*2),
                         ('observations', (((0, ONE), (0, ONE)),)),
                         ('observations', (((0, F(1 << nb.MAX_BITS)),),)),
                         ('observations', fixture['observations']*(nb.MAX_SUPPORT+1))):
        with pytest.raises(nb.KernelError):
            _call(dict(fixture, **{field: value}))
    carriers = fixture['carriers']
    for row in (*carriers.eq_rows, certs[-1].graph.eq_row):
        rhs = fixture['hz'].b.copy()
        rhs[row] += .125
        with pytest.raises(nb.KernelError):
            _call(dict(fixture, hz=replace(fixture['hz'], b=rhs)))
    rhs = fixture['hz'].ub.copy()
    rhs[certs[-1].graph.le_rows[0]] += .125
    with pytest.raises(nb.KernelError):
        _call(dict(fixture, hz=replace(fixture['hz'], ub=rhs)))
    with pytest.raises(nb.KernelError):
        _call(dict(fixture, carriers=replace(carriers, columns=([0], carriers.columns[1]))))
    with pytest.raises(nb.KernelError):
        _call(fixture, _Pool(0))
    full_pool = _Pool(256000000)
    _call(fixture, full_pool)
    with pytest.raises(nb.KernelError):
        _call(fixture, _Pool(full_pool.used-1))
    assert old._state(fixture['hz']) == before
