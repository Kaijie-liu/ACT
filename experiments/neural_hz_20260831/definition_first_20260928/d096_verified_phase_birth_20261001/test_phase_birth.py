"""Four fixed whole-population controls for literal nonconvex ReLU birth.

All numerical fixtures run only within the frozen full inherited math gate.
No model, attack, terminal search, GPU or production-default modification.
"""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.hybridz_tf.tf_mlp import sparse_hz_linear
from experiments.neural_hz_20260831.definition_first_20260928.d086_native_shared_transfer_20261001 import test_native_shared_transfer as old
from experiments.neural_hz_20260831.definition_first_20260928.d088_native_structure_discovery_20261001.test_native_discovery import _Pool
from experiments.neural_hz_20260831.definition_first_20260928.d096_verified_phase_birth_20261001 import phase_birth as pb

ZERO, ONE = F(0), F(1)
nb = pb.nb


def _source():
    # The old bit and its actual equality survive; none is fixed or removed.
    return nb.SparseHZono(
        c=np.array([.1, 2., -2., 0.]),
        Gc=sp.csr_matrix(np.array([[1., .125], [.25, 0.], [.5, 0.], [0., 0.]])),
        Gb=sp.csr_matrix(np.array([[.25], [0.], [0.], [0.]])),
        Ac=sp.csr_matrix(np.array([[0., 1.]])),
        Ab=sp.csr_matrix(np.array([[-.5]])), b=np.array([0.]),
        Auc=sp.csr_matrix(np.array([[1., 0.]])),
        Aub=sp.csr_matrix((1, 1)), ub=np.array([1.]),
        frame_id=96001, exact=False)


def _single():
    return nb.SparseHZono(np.array([.1]), sp.eye(1, format='csr'),
        sp.csr_matrix((1, 0)), sp.csr_matrix((0, 1)),
        sp.csr_matrix((0, 0)), np.zeros(0), frame_id=96002)


def _call(hz, carriers=None, pool=None):
    return pb.relu_birth(hz, carriers=carriers,
        pool=_Pool(256000000) if pool is None else pool, enabled=True)


def _verify(hz, gate, carriers):
    return pb.verify_gate(hz, gate, carriers, pool=_Pool(256000000), enabled=True)


def _readouts(hz, continuous, binary):
    return tuple(F.from_float(float(hz.c[i])) + old._dot(hz.Gc, i, continuous)
                 + old._dot(hz.Gb, i, binary) for i in range(hz.n_out))


def _extensions(result, continuous, binary):
    assert len(continuous) == result.old_n_cont
    assert len(binary) == result.old_n_bin
    pre = [old._eval(g.preactivation, continuous, binary) for g in result.gates]
    for choices in product(*(old._choices(g) for g in pre)):
        values = list(continuous) + [ZERO] * (result.hz.n_cont-len(continuous))
        bits = list(binary) + [ZERO] * (result.hz.n_bin-len(binary))
        if result.carriers is not None:
            for column in result.carriers.columns:
                if column < len(continuous):
                    assert continuous[column] == ONE
                values[column] = ONE
        for gate, g, bit in zip(result.gates, pre, choices):
            s, eta, b = gate.graph.slots
            q = max(ZERO, g)
            values[s] = 2*(q-g)/(-gate.lower)-bit
            values[eta] = ONE-2*q/gate.upper
            bits[b] = bit
        yield tuple(values), tuple(bits)


def _preserved_predicates(before, after):
    assert after.frame_id == before.frame_id and after.exact == before.exact
    assert after.n_cont >= before.n_cont and after.n_bin >= before.n_bin
    for cname, bname, count in (('Ac', 'Ab', before.n_eq),
                                ('Auc', 'Aub', before.n_ineq)):
        for name, width in ((cname, before.n_cont), (bname, before.n_bin)):
            original = getattr(before, name)
            extended = getattr(after, name)[:count]
            prefix = extended[:, :width]
            assert prefix.shape == original.shape
            assert prefix.data.tobytes() == original.data.tobytes()
            assert np.array_equal(prefix.indices, original.indices)
            assert np.array_equal(prefix.indptr, original.indptr)
            assert extended[:, width:].nnz == 0
    assert before.b.tobytes() == after.b[:before.n_eq].tobytes()
    assert before.ub.tobytes() == after.ub[:before.n_ineq].tobytes()


def _literal_check(hz, gate, carriers):
    # Independent coefficient maps, not the kernel's literal constructor.
    s, eta, bit = gate.graph.slots
    cont = {i: -a for i, a in gate.preactivation.continuous}
    binary = {i: -a for i, a in gate.preactivation.binary}
    cont[s], cont[eta] = gate.lower/2, -gate.upper/2
    cont[carriers.columns[1]] = gate.upper/2
    if gate.preactivation.bias:
        cont[carriers.columns[0]] = -gate.preactivation.bias
    binary[bit] = gate.lower/2
    actual = nb._row(hz.Ac, hz.Ab, gate.graph.eq_row)
    assert dict(actual.continuous) == cont and dict(actual.binary) == binary
    assert hz.b[gate.graph.eq_row] == 0.
    g, q = _verify(hz, gate, carriers)
    assert g == gate.preactivation and q == gate.readout
    assert q == nb.NativeAffine(gate.upper/2, ((eta, -gate.upper/2),), ())


def test_exact_birth_and_zero_phase_labels():
    source = _single()
    before = old._state(source)
    born = _call(source)
    gate, = born.gates
    actual_c = F.from_float(.1)
    assert born.bounds == ((actual_c-ONE, actual_c+ONE),)
    assert gate.upper == F.from_float(1.1)
    Q = gate.upper/2
    assert F.from_float(float(actual_c-Q))-(actual_c-Q) == -F(1, 2**55)
    _literal_check(born.hz, gate, born.carriers)
    assert old._state(source) == before
    _preserved_predicates(source, born.hz)
    zero_bits = set()
    for x in (-ONE, -actual_c, ZERO, ONE):
        for continuous, binary in _extensions(born, (x,), ()):
            assert old._holds(born.hz, continuous, binary)
            assert _readouts(born.hz, continuous, binary) == (max(ZERO, x+actual_c),)
            assert continuous[:1] == (x,)
            if x == -actual_c:
                zero_bits.add(binary[0])
    assert zero_bits == {-ONE, ONE}
    # Reverse direction: every feasible point in this literal gate grid has
    # exactly the ReLU readout, including both signs at the kink.
    feasible = 0
    for s, eta, bit in product((-ONE, ZERO, ONE), (-ONE, ZERO, ONE), (-ONE, ONE)):
        q = Q*(ONE-eta)
        g = gate.lower/2*(s+bit)-Q*eta+Q
        x = g-actual_c
        if not -ONE <= x <= ONE:
            continue
        continuous = (x, ONE, ONE, s, eta)
        if old._holds(born.hz, continuous, (bit,)):
            feasible += 1
            assert q == max(ZERO, g)
    assert feasible >= 4

    source = _source()
    before = old._state(source)
    born = _call(source)
    assert len(born.gates) == 1
    assert born.local_cost['active_rows'] == 2  # includes identically zero
    assert born.local_cost['inactive_rows'] == 1
    _preserved_predicates(source, born.hz)
    _literal_check(born.hz, born.gates[0], born.carriers)
    for bit in (-ONE, ONE):
        y = bit/2
        x_zero = -actual_c-bit/4-y/8
        for x in (-ONE, x_zero, ZERO, ONE):
            assert old._holds(source, (x, y), (bit,))
            wanted = tuple(max(ZERO, v) for v in _readouts(source, (x, y), (bit,)))
            labels = set()
            for continuous, binary in _extensions(born, (x, y), (bit,)):
                assert old._holds(born.hz, continuous, binary)
                assert _readouts(born.hz, continuous, binary) == wanted
                assert continuous[:2] == (x, y) and binary[:1] == (bit,)
                labels.add(binary[-1])
            if x == x_zero:
                assert labels == {-ONE, ONE}
    assert old._state(source) == before
    stable = replace(source, c=np.array([1., -1., 0.]),
                     Gc=sp.csr_matrix((3, 2)), Gb=sp.csr_matrix((3, 1)))
    stable_result = _call(stable)
    assert stable_result.carriers is None and not stable_result.gates
    assert stable_result.hz.n_cont == 2 and stable_result.hz.n_bin == 1
    assert stable_result.local_cost['added_eq'] == stable_result.local_cost['added_le'] == 0
    _preserved_predicates(stable, stable_result.hz)
    assert np.array_equal(stable_result.hz.c, [1., 0., 0.])


def test_reused_carriers_and_transported_gate_certificates():
    source = _source()
    first = _call(source)
    # An ordinary mixed affine changes the output bank but not latent rows.
    pre = sparse_hz_linear(first.hz,
        np.array([[1., 0., 0., 0.], [.5, -.25, 0., 0.]]), np.array([-.5, .125]))
    pre_before = old._state(pre)
    second = _call(pre, first.carriers)
    assert len(second.gates) == 2 and second.carriers == first.carriers
    assert second.local_cost['new_carrier_columns'] == 0
    assert second.hz.n_cont == pre.n_cont+4
    assert second.hz.n_bin == pre.n_bin+2
    _preserved_predicates(pre, second.hz)
    assert old._state(pre) == pre_before
    for gate in (*first.gates, *second.gates):
        _literal_check(second.hz, gate, second.carriers)
    # output_row is deliberately only a birth record, not a stale output
    # position that later affine transformations must preserve.
    shifted = replace(first.gates[0], output_row=1000)
    assert _verify(second.hz, shifted, second.carriers) == (
        first.gates[0].preactivation, first.gates[0].readout)
    for bit in (-ONE, ONE):
        for x in (-ONE, ZERO, ONE):
            original_cont = (x, bit/2)
            for first_cont, first_bits in _extensions(first, original_cont, (bit,)):
                wanted = tuple(max(ZERO, v) for v in _readouts(pre, first_cont, first_bits))
                for cont, bits in _extensions(second, first_cont, first_bits):
                    assert old._holds(second.hz, cont, bits)
                    assert _readouts(second.hz, cont, bits) == wanted
                    assert cont[:2] == original_cont and bits[:1] == (bit,)
    # Even an all-stable continuation authenticates the reused carriers.
    stable_pre = sparse_hz_linear(second.hz, np.zeros((1, 2)), np.array([1.]))
    stable = _call(stable_pre, second.carriers)
    assert not stable.gates and stable.carriers == second.carriers
    assert stable.local_cost['added_cont'] == stable.local_cost['added_binary'] == 0
    for gate in (*first.gates, *second.gates):
        _literal_check(stable.hz, gate, stable.carriers)


def test_birth_bank_population_and_literal_equations():
    width = 64
    source = nb.SparseHZono(
        np.array([(i % 5-2)/16 for i in range(width)]), sp.eye(width, format='csr'),
        sp.csr_matrix(np.full((width, 1), .25)), sp.csr_matrix((0, width)),
        sp.csr_matrix((0, 1)), np.zeros(0), frame_id=96003)
    before = old._state(source)
    pool = _Pool(256000000)
    result = _call(source, pool=pool)
    assert len(result.gates) == width
    assert tuple(g.output_row for g in result.gates) == tuple(range(width))
    assert result.hz.n_cont == width+2+2*width and result.hz.n_bin == 1+width
    assert result.hz.n_eq == width+2 and result.hz.n_ineq == 2*width
    assert result.local_cost['new_carrier_columns'] == 2
    assert result.local_cost['work_charged'] == pool.used > 0
    assert pool.used < pool.limit
    assert result.physical_bytes['input'] == old._buffers(source)
    assert result.physical_bytes['output'] == old._buffers(result.hz)
    assert result.nnz['input'] == old._nnz(source)
    assert result.nnz['output'] == old._nnz(result.hz)
    assert all(result.local_cost[key] is False for key in (
        'whole_work_qualified', 'complete_physical_qualified',
        'actual_model_qualified', 'gpu_qualified'))
    _preserved_predicates(source, result.hz)
    for gate in result.gates:
        _literal_check(result.hz, gate, result.carriers)
        assert any(i == 0 for i, _ in gate.preactivation.binary)
    assert old._state(source) == before
    for old_bit in (-ONE, ONE):
        point = tuple(ONE if i % 2 else -ONE for i in range(width))
        wanted = tuple(max(ZERO, v) for v in _readouts(source, point, (old_bit,)))
        states = tuple(_extensions(result, point, (old_bit,)))
        assert len(states) == 1
        for cont, bits in states:
            assert old._holds(result.hz, cont, bits)
            assert _readouts(result.hz, cont, bits) == wanted
            assert cont[:width] == point and bits[:1] == (old_bit,)


def test_birth_binding_and_resource_fail_closed():
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError('disabled path touched input')

    assert pb.relu_birth(Poison(), pool=Poison()) is None
    assert pb.verify_gate(Poison(), Poison(), Poison(), pool=Poison()) is None
    for value in (1, None, np.bool_(True)):
        with pytest.raises(nb.KernelError):
            pb.relu_birth(Poison(), pool=Poison(), enabled=value)
    source = _single()
    before = old._state(source)
    with pytest.raises(nb.KernelError):
        _call(source, pool=_Pool(0))
    with pytest.raises(nb.KernelError):
        pb.relu_birth(source, pool=None, enabled=True)
    bad_gc = source.Gc.copy()
    bad_gc.data[0] = np.nan
    with pytest.raises(nb.KernelError):
        _call(replace(source, Gc=bad_gc))
    result = _call(source)
    gate, = result.gates
    carriers = result.carriers
    for bad in (replace(carriers, columns=([0], carriers.columns[1])),
                replace(carriers, columns=(True, carriers.columns[1])),
                replace(carriers, columns=(carriers.columns[0],)*2),
                replace(carriers, eq_rows=([0], carriers.eq_rows[1])),
                replace(carriers, eq_rows=(carriers.eq_rows[0],)*2)):
        with pytest.raises(nb.KernelError):
            _verify(result.hz, gate, bad)
    bad_rhs = result.hz.b.copy()
    bad_rhs[carriers.eq_rows[0]] = .5
    with pytest.raises(nb.KernelError):
        _verify(replace(result.hz, b=bad_rhs), gate, carriers)
    with pytest.raises(nb.KernelError):
        _call(replace(result.hz, b=bad_rhs), carriers)
    for graph in (replace(gate.graph, eq_row=[]),
                  replace(gate.graph, le_rows=([], gate.graph.le_rows[1])),
                  replace(gate.graph, le_rows=(gate.graph.le_rows[0],)*2),
                  replace(gate.graph, slots=(gate.graph.slots[1],)*2+(gate.graph.slots[2],))):
        with pytest.raises(nb.KernelError):
            _verify(result.hz, replace(gate, graph=graph), carriers)
    for bad_gate in (replace(gate, lower=F(-1, 4)),
                     replace(gate, readout=replace(gate.readout, bias=ONE)),
                     replace(gate, preactivation=replace(gate.preactivation, bias=ZERO)),
                     replace(gate, lower=F(-1, 2**600))):
        with pytest.raises(nb.KernelError):
            _verify(result.hz, bad_gate, carriers)
    bad_guard = result.hz.ub.copy()
    bad_guard[gate.graph.le_rows[0]] = .125
    with pytest.raises(nb.KernelError):
        _verify(replace(result.hz, ub=bad_guard), gate, carriers)
    bad_eq = result.hz.b.copy()
    bad_eq[gate.graph.eq_row] = .125
    with pytest.raises(nb.KernelError):
        _verify(replace(result.hz, b=bad_eq), gate, carriers)
    carrier_readout = sp.csr_matrix(([1.], ([0], [carriers.columns[0]])),
                                    shape=result.hz.Gc.shape)
    with pytest.raises(nb.KernelError):
        _call(replace(result.hz, Gc=carrier_readout), carriers)
    # The existing per-form support boundary is retained, not shrunk to the
    # toy control widths. Complete batch populations use the caller hard pool.
    width = nb.MAX_SUPPORT+1
    too_wide = nb.SparseHZono(np.array([0.]),
        sp.csr_matrix((np.ones(width), np.arange(width), np.array([0, width])), shape=(1, width)),
        sp.csr_matrix((1, 0)), sp.csr_matrix((0, width)),
        sp.csr_matrix((0, 0)), np.zeros(0), frame_id=96004)
    with pytest.raises(nb.KernelError):
        _call(too_wide)
    assert old._state(source) == before
