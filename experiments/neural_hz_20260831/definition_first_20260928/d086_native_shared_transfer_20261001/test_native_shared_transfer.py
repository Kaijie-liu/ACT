"""Four frozen tests on real SparseHZono operators, not trained networks."""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono, sparse_hz_linear
from act.back_end.hybridz_tf.tf_mlp import (
    sparse_hz_apply_relu_exact, sparse_hz_apply_relu_compact_exact,
)
from experiments.neural_hz_20260831.definition_first_20260928.d086_native_shared_transfer_20261001 import native_shared_transfer as nt

nb = nt.nb
ZERO, ONE = F(0), F(1)


def _fixture(wide=False, compact=False):
    nc, nbits = (3, 1) if wide else (2, 0)
    source = SparseHZono(np.zeros(2),
        sp.csr_matrix(([1., 1.], ([0, 1], [0, 1])), shape=(2, nc)),
        sp.csr_matrix((2, nbits)), sp.csr_matrix((0, nc)),
        sp.csr_matrix((0, nbits)), np.zeros(0), frame_id=86001)
    width = 1 if compact else 2
    ps = tuple((nc+width*i, nc+width*i+(not compact), nbits+i) for i in range(2))
    kernel = sparse_hz_apply_relu_compact_exact if compact else sparse_hz_apply_relu_exact
    parent = kernel(source, [-1., -1.], [1., 1.], ps, nc+2*width, nbits+2)
    pre = sparse_hz_linear(parent, np.array([[1., 1.], [1., -1.]]),
                           np.array([-.75, .25]))
    if wide:
        extra_c = sp.csr_matrix(([.5], ([0], [2])), shape=pre.Gc.shape)
        extra_b = sp.csr_matrix(([.25], ([0], [0])), shape=pre.Gb.shape)
        pre = replace(pre, Gc=(pre.Gc+extra_c).tocsr(), Gb=(pre.Gb+extra_b).tocsr())
    cs = tuple((nc+2*width+width*i, nc+2*width+width*i+(not compact), nbits+2+i)
               for i in range(2))
    lower, upper = (F(-2), F(2)) if wide else (F(-3, 4), F(5, 4))
    post = kernel(pre, [float(lower)]*2, [float(upper)]*2, cs,
                  nc+4*width, nbits+4)
    if compact:
        parents = tuple(nb.Graph('compact', ps[i], None, (i, 2+i, 4+i)) for i in range(2))
        children = tuple(nb.Graph('compact', cs[i], None, (6+i, 8+i, 10+i)) for i in range(2))
    else:
        parents = tuple(nb.Graph('extended', ps[i], i, (i, 2+i)) for i in range(2))
        children = tuple(nb.Graph('extended', cs[i], 2+i, (4+i, 6+i)) for i in range(2))
    consumers = ((children[0], ONE, ONE), (children[1], ONE, -ONE))
    return post, parents, consumers, (nc, nbits, lower, upper, compact)


def _call(fixture, consumers=None):
    hz, parents, normal, _ = fixture
    return nt.append_shared_upper(hz, parents, normal if consumers is None else consumers, enabled=True)


def _dot(matrix, row, values):
    start, end = matrix.indptr[row:row+2]
    return sum((F.from_float(float(matrix.data[j]))*values[int(matrix.indices[j])]
                for j in range(int(start), int(end))), ZERO)


def _eval(form, continuous, binary):
    return form.bias + sum((a*continuous[i] for i, a in form.continuous), ZERO) + sum(
        (a*binary[i] for i, a in form.binary), ZERO)


def _holds(hz, continuous, binary, integer=True):
    assert len(continuous) == hz.n_cont and len(binary) == hz.n_bin
    assert all(-ONE <= x <= ONE for x in continuous)
    assert all(x in (-ONE, ONE) if integer else -ONE <= x <= ONE for x in binary)
    return (all(_dot(hz.Ac, i, continuous)+_dot(hz.Ab, i, binary) == F.from_float(float(rhs))
                for i, rhs in enumerate(hz.b)) and
            all(_dot(hz.Auc, i, continuous)+_dot(hz.Aub, i, binary) <= F.from_float(float(rhs))
                for i, rhs in enumerate(hz.ub)))


def _choices(g):
    return (-ONE,) if g > ZERO else ((ONE,) if g < ZERO else (-ONE, ONE))


def _original_states(fixture, point, upstream=ONE):
    hz, parents, consumers, (nc, nbits, lo, hi, compact) = fixture
    parent_g = point[:2]
    for pb in product(*(_choices(g) for g in parent_g)):
        continuous = list(point)
        binary = [upstream] if nbits else []
        binary.extend(pb)
        for g, bit in zip(parent_g, pb):
            q = max(ZERO, g)
            if not compact:
                continuous.append(2*(q-g)-bit)
            continuous.append(ONE-2*q)
        padded = continuous + [ZERO]*(hz.n_cont-len(continuous))
        padded_b = binary + [ZERO]*2
        child_g = tuple(_eval(nb._extract_graph(hz, c[0])[0], padded, padded_b) for c in consumers)
        for cb in product(*(_choices(g) for g in child_g)):
            values = list(continuous)
            for g, bit in zip(child_g, cb):
                q = max(ZERO, g)
                if not compact:
                    values.append((q-g)/(-lo/2)-bit)
                values.append(ONE-q/(hi/2))
            yield tuple(values), tuple([*binary, *cb])


def _extend(fixture, result, continuous, binary, consumers=None):
    hz, parents, ordinary, _ = fixture
    consumers = ordinary if consumers is None else consumers
    alpha, beta = ((ONE-binary[g.slots[2]])/2 for g in parents)
    u, v = ((ONE-continuous[g.slots[1]])/2 for g in parents)
    delta, zu, zv = alpha*beta, beta*u, alpha*v
    new = list(continuous) + [ZERO]*(result.hz.n_cont-len(continuous))
    for col, value in zip(result.shared_columns, (delta, zu, zv)):
        new[col] = 2*value-ONE
    pi = (ONE-alpha-beta+delta, alpha-delta, beta-delta, delta)
    for index, scale, columns in result.residual_bindings:
        graph, a, b = consumers[index]
        g, _, _ = nb._extract_graph(hz, graph)
        bias = g.bias-a/2-b/2
        residual = _eval(g, continuous, binary)-a*u-b*v-bias
        assert -scale <= residual <= scale
        for col, mass in zip(columns, pi[:3]):
            new[col] = mass*residual/scale
    return tuple(new)


def _state(hz):
    return (hz.frame_id, hz.exact, hz.c.tobytes(), hz.b.tobytes(), hz.ub.tobytes(),
        tuple((a.shape, a.data.tobytes(), a.indices.tobytes(), a.indptr.tobytes())
              for a in (hz.Gc, hz.Gb, hz.Ac, hz.Ab, hz.Auc, hz.Aub)))


def _buffers(hz):
    return sum(a.nbytes for a in (hz.c, hz.b, hz.ub)) + sum(
        a.data.nbytes+a.indices.nbytes+a.indptr.nbytes
        for a in (hz.Gc, hz.Gb, hz.Ac, hz.Ab, hz.Auc, hz.Aub))


def _nnz(hz):
    return sum(a.nnz for a in (hz.Gc, hz.Gb, hz.Ac, hz.Ab, hz.Auc, hz.Aub))


def test_shared_native_positive_and_projection():
    fixture = _fixture()
    hz, parents, consumers, _ = fixture
    before = _state(hz)
    result = _call(fixture)
    assert result.old_n_cont == hz.n_cont
    assert result.hz.n_cont == hz.n_cont+3 and result.hz.n_bin == hz.n_bin
    assert result.hz.n_ineq == hz.n_ineq+14 and result.hz.n_eq == hz.n_eq
    assert result.extra_residual_count == 0 and not result.residual_bindings
    assert _state(hz) == before
    zero_parent_labels = set()
    child_zero_labels = set()
    for point in ((-ONE, -ONE), (ZERO, ZERO), (ONE, ONE), (F(3, 4), ZERO),
                  (ZERO, F(1, 4)), (F(1, 2), F(1, 4)), (F(1, 4), F(3, 4))):
        for cont, bits in _original_states(fixture, point):
            assert _holds(hz, cont, bits)
            extended = _extend(fixture, result, cont, bits)
            assert _holds(result.hz, extended, bits)
            assert all(_eval(row, extended, bits) <= rhs for row, rhs in result.exact_rows)
            if point == (ZERO, ZERO):
                zero_parent_labels.add(bits[:2])
            if point == (F(1, 2), F(1, 4)):
                child_zero_labels.add(bits[2])
    assert len(zero_parent_labels) == 4 and child_zero_labels == {-ONE, ONE}
    # Old LP point is genuine feasible; two new upper rows plus
    # (3/8)*(delta-alpha)<=0 cancel every new factor and contradict it.
    cont = (ZERO, ZERO, F(1, 2), F(1, 2), F(1, 2), F(1, 2),
            ONE, F(1, 2), ONE, ZERO)
    bits = (ZERO, ZERO, F(1, 2), ZERO)
    assert _holds(hz, cont, bits, integer=False)
    upper_rows = result.exact_rows[-2:]
    delta_minus_alpha = nb.NativeAffine(ZERO,
        ((result.shared_columns[0], F(1, 2)),), ((parents[0].slots[2], F(1, 2)),))
    combined = nb._combine(ZERO, ((ONE, upper_rows[0][0]), (ONE, upper_rows[1][0]),
                                      (F(3, 8), delta_minus_alpha)))
    rhs = upper_rows[0][1]+upper_rows[1][1]
    assert all(i < hz.n_cont for i, _ in combined.continuous)
    assert _eval(combined, cont, bits)-rhs == F(3, 16)
    assert all(error == ZERO for error in result.row_errors)


def test_full_residual_and_binary_preservation():
    fixture = _fixture(wide=True)
    hz, parents, consumers, meta = fixture
    hz = replace(hz, exact=False)
    fixture = (hz, parents, consumers, meta)
    before = _state(hz)
    result = _call(fixture)
    assert result.extra_residual_count == 1
    assert result.hz.n_cont == hz.n_cont+6 and result.hz.n_bin == 5
    assert result.hz.n_ineq == hz.n_ineq+22
    assert result.hz.frame_id == hz.frame_id and result.hz.exact is False
    assert _state(hz) == before
    assert np.array_equal(hz.c, result.hz.c) and np.array_equal(hz.b, result.hz.b)
    for name in ('Gc', 'Ac'):
        assert (getattr(result.hz, name)[:, :hz.n_cont] != getattr(hz, name)).nnz == 0
        assert getattr(result.hz, name)[:, hz.n_cont:].nnz == 0
    for name in ('Gb', 'Ab'):
        assert (getattr(result.hz, name) != getattr(hz, name)).nnz == 0
    assert (result.hz.Auc[:hz.n_ineq, :hz.n_cont] != hz.Auc).nnz == 0
    assert (result.hz.Aub[:hz.n_ineq] != hz.Aub).nnz == 0
    assert np.array_equal(result.hz.ub[:hz.n_ineq], hz.ub)
    assert result.residual_bindings[0][1] == F(3, 4)
    assert any(row.binary and any(i == 0 for i, _ in row.binary)
               for row, _ in result.exact_rows)
    for point in ((-ONE, -ONE, -ONE), (ZERO, ZERO, ZERO), (ONE, ONE, ONE),
                  (F(1, 2), F(1, 4), F(-1, 2))):
        for upstream in (-ONE, ONE):
            for cont, bits in _original_states(fixture, point, upstream):
                assert _holds(hz, cont, bits)
                extended = _extend(fixture, result, cont, bits)
                assert _holds(result.hz, extended, bits)
                # Original input decoder remains this exact prefix; no new
                # auxiliary is substituted for any original coordinate.
                assert extended[:3] == point


def test_binding_rejection_and_default_off():
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError('disabled input inspected')
    poison = Poison()
    assert nt.append_shared_upper(poison, poison, poison) is None
    with pytest.raises(nb.KernelError):
        nt.append_shared_upper(poison, poison, poison, enabled=1)
    fixture = _fixture()
    hz, parents, consumers, _ = fixture
    before = _state(hz)
    with pytest.raises(nb.KernelError):
        nt.append_shared_upper(hz, (parents[0], parents[0]), consumers, enabled=True)
    bad = replace(parents[0], slots=(*parents[0].slots[:2], parents[1].slots[2]))
    with pytest.raises(nb.KernelError):
        nt.append_shared_upper(hz, (bad, parents[1]), consumers, enabled=True)
    with pytest.raises(nb.KernelError):
        nt.append_shared_upper(hz, parents, ((consumers[0][0], F(1 << nb.MAX_BITS), ONE),), enabled=True)
    assert _state(hz) == before
    compact = _fixture(compact=True)
    result = _call(compact)
    for cont, bits in _original_states(compact, (ZERO, ZERO)):
        assert _holds(result.hz, _extend(compact, result, cont, bits), bits)
    original, p, c, meta = compact
    rhs = original.ub.copy()
    rhs[4] += .125
    altered = replace(original, ub=rhs)
    with pytest.raises(nb.KernelError):
        nt.append_shared_upper(altered, p, c, enabled=True)
    assert original.ub[4] != altered.ub[4]


def test_outward_rows_and_cost_accounting():
    fixture = _fixture()
    hz, parents, consumers, _ = fixture
    changed = ((consumers[0][0], F(1, 3), F(-2, 5)), consumers[1])
    result = _call(fixture, changed)
    assert result.extra_residual_count == 1
    assert any(error > ZERO for error in result.row_errors)
    assert len(result.exact_rows) == result.hz.n_ineq-hz.n_ineq == 22
    assert len(result.row_errors) == len(result.installed_rhs) == len(result.exact_rows)
    for j, ((row, rhs), error, installed) in enumerate(zip(
            result.exact_rows, result.row_errors, result.installed_rhs)):
        assert row.bias == ZERO
        actual = nb._row(result.hz.Auc, result.hz.Aub, hz.n_ineq+j)
        difference = nb._combine(ZERO, ((ONE, row), (-ONE, actual)))
        assert nb._norm(difference) == error
        assert installed == F.from_float(float(result.hz.ub[hz.n_ineq+j]))
        assert installed >= rhs+error
    for point in ((ZERO, ZERO), (F(3, 4), F(1, 4)), (-ONE, ONE)):
        for cont, bits in _original_states(fixture, point):
            extended = _extend(fixture, result, cont, bits, changed)
            assert _holds(result.hz, extended, bits)
            assert all(_eval(row, extended, bits) <= rhs for row, rhs in result.exact_rows)
    assert result.physical_bytes['input'] == _buffers(hz)
    assert result.physical_bytes['output'] == _buffers(result.hz)
    assert result.physical_bytes['input_plus_output'] == _buffers(hz)+_buffers(result.hz)
    added_c, added_b = result.hz.Auc[hz.n_ineq:], result.hz.Aub[hz.n_ineq:]
    added_bytes = result.hz.ub[hz.n_ineq:].nbytes + sum(
        a.data.nbytes+a.indices.nbytes+a.indptr.nbytes for a in (added_c, added_b))
    assert result.physical_bytes['generated_row_buffers'] == added_bytes
    assert result.nnz['input'] == _nnz(hz) and result.nnz['output'] == _nnz(result.hz)
    assert result.nnz['added'] == added_c.nnz+added_b.nnz
    assert result.nnz['exact_added'] == sum(len(row.continuous)+len(row.binary)
                                           for row, _ in result.exact_rows)
