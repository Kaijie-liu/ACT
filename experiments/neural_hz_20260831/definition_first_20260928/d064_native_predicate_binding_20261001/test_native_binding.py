"""Four frozen native-predicate fixtures; no model or solver execution."""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product
import math

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono, sparse_hz_linear
from act.back_end.hybridz_tf.tf_mlp import (
    sparse_hz_apply_relu_exact, sparse_hz_apply_relu_compact_exact,
)
from experiments.neural_hz_20260831.definition_first_20260928.d064_native_predicate_binding_20261001 import native_binding as nb

ZERO, ONE = F(0), F(1)
ms, ss = nb.ms, nb.ss


def _fixture(compact=False, bias=F(1, 8), upstream_binary=False):
    old_bits = int(upstream_binary)
    gb = (sp.csr_matrix(([0.25], ([0], [0])), shape=(3, 1))
          if upstream_binary else sp.csr_matrix((3, 0)))
    source = SparseHZono(np.zeros(3), sp.eye(3, format="csr"), gb,
        sp.csr_matrix((0, 3)), sp.csr_matrix((0, old_bits)), np.zeros(0), frame_id=64001)
    weights = np.array([[1., 1., 0.], [0., 1., 1.], [1., 0., 1.]])
    pre = sparse_hz_linear(source, weights, np.full(3, float(bias)))
    slots = (tuple((3 + i, 3 + i, old_bits + i) for i in range(3)) if compact
             else tuple((3 + 2*i, 4 + 2*i, old_bits + i) for i in range(3)))
    kernel = sparse_hz_apply_relu_compact_exact if compact else sparse_hz_apply_relu_exact
    post = kernel(pre, np.full(3, -3.), np.full(3, 3.), slots,
                  6 if compact else 9, old_bits + 3)
    receiver = sparse_hz_linear(post, np.array([[1., 1., -1.]]))
    y = sp.csr_matrix(([1.], ([0], [1])), shape=receiver.Gc.shape)
    receiver = replace(receiver, Gc=(receiver.Gc-y).tocsr())

    frame = ms.make_frame("original native binding reference", enabled=True)
    values = tuple(ms.make_value(frame, "source " + str(i), enabled=True) for i in range(3))
    radius = F(5, 4) if upstream_binary else ONE
    ctx = ms.source_context(frame, tuple((v, i, -radius if i == 0 else -ONE,
                       radius if i == 0 else ONE) for i, v in enumerate(values)), enabled=True)
    forms = tuple(ss.affine(ctx, (bias, bias),
        tuple((s, F(int(a)), F(int(a))) for s, a in zip(ctx.sources, row)), enabled=True)
        for row in weights)
    v = ss.affine(ctx, (ZERO, ZERO), ((ctx.sources[1], -ONE, -ONE),), enabled=True)
    gates = []
    for i, (form, weight) in enumerate(zip(forms, (ONE, ONE, -ONE))):
        q = ms.make_value(frame, "original native q " + str(i), enabled=True)
        phase = ms.make_phase(q, slots[i][2], "original signed slot " + str(i), enabled=True)
        gates.append((form, q, phase, (weight, weight)))
    token = ms.make_value(frame, "actual receiver row zero", enabled=True)
    sources = tuple(nb.NativeAffine(ZERO, ((i, ONE),),
                    ((0, F(1, 4)),) if upstream_binary and i == 0 else ()) for i in range(3))
    graphs = tuple(nb.Graph("compact" if compact else "extended", slots[i],
                    None if compact else i,
                    (i, 3+i, 6+i) if compact else (i, 3+i)) for i in range(3))
    return dict(hz=receiver, v=v, gates=tuple(gates), receiver=token,
                source_readouts=sources, graphs=graphs, receiver_index=0)


def _call(fixture, **changes):
    args = dict(fixture)
    args.update(changes)
    return nb.bind_and_append(**args, enabled=True)


def _state(hz):
    return (hz.frame_id, hz.exact, hz.c.tobytes(), hz.b.tobytes(), hz.ub.tobytes(),
        tuple((a.shape, a.data.tobytes(), a.indices.tobytes(), a.indptr.tobytes())
              for a in (hz.Gc, hz.Gb, hz.Ac, hz.Ab, hz.Auc, hz.Aub)))


def _row(matrix, index, vector):
    start, end = matrix.indptr[index:index+2]
    return sum((F.from_float(float(matrix.data[k])) * vector[int(matrix.indices[k])]
                for k in range(int(start), int(end))), ZERO)


def _holds(hz, continuous, binary):
    assert len(continuous) == hz.n_cont and len(binary) == hz.n_bin
    assert all(-ONE <= x <= ONE for x in continuous)
    assert all(x in (-ONE, ONE) for x in binary)
    return (all(_row(hz.Ac, i, continuous)+_row(hz.Ab, i, binary) == F.from_float(float(rhs))
                for i, rhs in enumerate(hz.b))
            and all(_row(hz.Auc, i, continuous)+_row(hz.Aub, i, binary) <= F.from_float(float(rhs))
                    for i, rhs in enumerate(hz.ub)))


def test_native_extended_multigate_binding():
    f = _fixture()
    before = _state(f["hz"])
    result = _call(f)
    assert result.lower == -ONE and result.upper == F(9, 8)
    assert result.receiver_error == ZERO
    assert result.graph_errors == result.preactivation_errors == (ZERO, ZERO, ZERO)
    assert result.rows_added == 2 and result.hz.n_ineq == f["hz"].n_ineq + 2
    assert result.hz.n_cont == f["hz"].n_cont and result.hz.n_bin == 3
    assert _state(f["hz"]) == before
    points = ((-ONE, -ONE, -ONE), (ZERO, ZERO, ZERO), (ONE, ONE, ONE),
              (F(3, 4), F(1, 4), F(-1, 2)), (F(-1, 16),)*3)
    zero_choices = set()
    for x, y, z in points:
        pre = (x+y+F(1, 8), y+z+F(1, 8), x+z+F(1, 8))
        choices = tuple((-ONE,) if g > 0 else ((ONE,) if g < 0 else (-ONE, ONE)) for g in pre)
        for bits in product(*choices):
            continuous = [x, y, z]
            for g, bit in zip(pre, bits):
                q = max(ZERO, g)
                continuous.extend((ONE if bit == -ONE else -g/F(3, 2)-ONE,
                                   ONE-q/F(3, 2)))
            assert _holds(f["hz"], continuous, bits)
            assert _holds(result.hz, continuous, bits)
            value = F.from_float(float(result.hz.c[0])) + _row(result.hz.Gc, 0, continuous)
            assert result.lower <= value <= result.upper
            if all(g == ZERO for g in pre):
                zero_choices.add(bits)
    assert len(zero_choices) == 8


def test_native_compact_residual_and_outward_rows():
    f = _fixture(compact=True, upstream_binary=True)
    baseline = _call(f)
    assert baseline.graph_errors == baseline.preactivation_errors == (ZERO,)*3
    assert baseline.receiver_error == ZERO and baseline.hz.n_bin == 4
    rhs = f["hz"].ub.copy()
    rhs[6] += 0.125  # A fixed ordinary nonzero width in the actual compact relation.
    relaxed = replace(f["hz"], ub=rhs)
    widened = _call(f, hz=relaxed)
    assert widened.graph_errors == (F(1, 16), ZERO, ZERO)
    assert widened.preactivation_errors == (F(1, 8), ZERO, ZERO)
    assert widened.upper == baseline.upper+F(1, 8)
    assert widened.lower == baseline.lower-F(1, 8)
    positive_witness = (ZERO, ZERO, ZERO, F(2, 3), F(11, 12), F(3, 4))
    assert _holds(relaxed, positive_witness, (ONE, -ONE, -ONE, -ONE))
    assert _holds(widened.hz, positive_witness, (ONE, -ONE, -ONE, -ONE))
    # Negative Delta is a property of the STORED relation, not a license to
    # delete its original bit or repair the model. An inactive point survives.
    rhs_negative = f["hz"].ub.copy()
    rhs_negative[6] -= 0.125
    tightened = replace(f["hz"], ub=rhs_negative)
    negative = _call(f, hz=tightened)
    assert negative.graph_errors == (F(1, 16), ZERO, ZERO)
    assert negative.preactivation_errors == (F(1, 8), ZERO, ZERO)
    assert negative.hz.n_bin == tightened.n_bin == 4
    inactive_point = (-ONE, -ONE, -ONE, ONE, ONE, ONE)
    assert _holds(tightened, inactive_point, (-ONE, ONE, ONE, ONE))
    assert _holds(negative.hz, inactive_point, (-ONE, ONE, ONE, ONE))
    # Actual native bias rounding is paid, not accepted as exact by tolerance.
    rounded = _call(_fixture(bias=F.from_float(0.1)))
    assert rounded.graph_errors == (ZERO,)*3
    assert all(error > 0 for error in rounded.preactivation_errors)
    # Reference interval widths are replaced by the COMPLETE native residual,
    # not silently assumed to describe an exact native/model equality.
    wide_v = ss.affine(f["v"].context, (-ONE, ONE), f["v"].terms, enabled=True)
    same = _call(f, v=wide_v)
    assert same.reference.error > ZERO
    assert (same.lower, same.upper) == (baseline.lower, baseline.upper)
    gates = list(f["gates"])
    gates[0] = (*gates[0][:3], (F(1, 3), F(1, 3)))
    rounded_rows = _call(f, gates=tuple(gates))
    center = F.from_float(float(f["hz"].c[0]))
    exact = (rounded_rows.upper-center, center-rounded_rows.lower)
    for target, stored, value in zip(exact, rounded_rows.installed_rhs, rounded_rows.hz.ub[-2:]):
        assert stored == F.from_float(float(value)) and stored >= target
        assert F.from_float(math.nextafter(float(value), -math.inf)) < target
    assert rounded_rows.receiver_error > ZERO


def test_native_identity_and_predicate_rejection():
    f = _fixture()
    with pytest.raises(nb.KernelError):
        _call(f, receiver_index=-1)
    with pytest.raises(nb.KernelError):
        _call(f, hz=replace(f["hz"], frame_id=None))
    with pytest.raises(nb.KernelError):
        _call(f, graphs=f["graphs"][:2])
    bad = list(f["graphs"])
    bad[0] = replace(bad[0], slots=(3, 4, 1))
    with pytest.raises(nb.KernelError):
        _call(f, graphs=tuple(bad))
    bad = list(f["graphs"])
    bad[0] = replace(bad[0], le_rows=(1, 3))
    with pytest.raises(nb.KernelError):
        _call(f, graphs=tuple(bad))
    broken = f["hz"].ub.copy()
    broken[0] = 0.125
    with pytest.raises(nb.KernelError):
        _call(f, hz=replace(f["hz"], ub=broken))
    sources = list(f["source_readouts"])
    sources[0] = nb.NativeAffine(ZERO, ((0, F(2)),), ())
    with pytest.raises(nb.KernelError):
        _call(f, source_readouts=tuple(sources))
    nonfinite = f["hz"].c.copy()
    nonfinite[0] = math.inf
    with pytest.raises(nb.KernelError):
        _call(f, hz=replace(f["hz"], c=nonfinite))
    compact = _fixture(compact=True)
    bad_matrix = compact["hz"].Auc.copy()
    start, end = bad_matrix.indptr[6:8]
    assert end > start
    bad_matrix.data[start] += 0.125
    with pytest.raises(nb.KernelError):
        _call(compact, hz=replace(compact["hz"], Auc=bad_matrix))


def test_native_default_off_and_original_state_preservation():
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError("disabled bridge inspected input")
    poison = Poison()
    assert nb.bind_and_append(poison, poison, poison, poison, poison, poison, poison) is None
    with pytest.raises(nb.KernelError):
        nb.bind_and_append(poison, poison, poison, poison, poison, poison, poison, enabled=1)
    f = _fixture()
    native = replace(f["hz"], exact=False)
    before = _state(native)
    result = _call(f, hz=native)
    assert _state(native) == before and result.hz.exact is False
    assert result.hz.frame_id == native.frame_id
    assert np.array_equal(result.hz.c, native.c)
    assert np.array_equal(result.hz.b, native.b)
    for name in ("Gc", "Gb", "Ac", "Ab"):
        assert (getattr(result.hz, name) != getattr(native, name)).nnz == 0
    assert (result.hz.Auc[:-2] != native.Auc).nnz == 0
    assert (result.hz.Aub[:-2] != native.Aub).nnz == 0
    assert np.array_equal(result.hz.ub[:-2], native.ub)
    oversized = list(f["source_readouts"])
    oversized[0] = nb.NativeAffine(F(1 << nb.MAX_BITS), ((0, ONE),), ())
    with pytest.raises(nb.KernelError):
        _call(f, source_readouts=tuple(oversized))
