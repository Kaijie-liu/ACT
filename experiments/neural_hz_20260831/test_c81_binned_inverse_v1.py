"""Independent Fraction and complete original-row tests on ordinary nonzero inputs."""
from dataclasses import replace
from fractions import Fraction as F
from types import SimpleNamespace
import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c81_binned_inverse_v1 import dot, reconstruct, verify_inverse
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c62_local_equations_v1 import encode, SCHEMA
from experiments.neural_hz_20260831.c68_local_splice_v1 import compile_journal
from experiments.neural_hz_20260831.c30_append_discovery_v1 import discover_append
from experiments.neural_hz_20260831.c30_first_write_v1 import splice_append
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.test_c28_consumer_discovery_v1 import source
from experiments.neural_hz_20260831.test_c30_first_write_v1 import view_from
from experiments.neural_hz_20260831.test_c68_local_splice_v2 import journal
from experiments.neural_hz_20260831.test_c10_alias_quotient_v1 import feasible
from experiments.neural_hz_20260831.test_c74_native_binding_v1 import native


def pool():
    return WorkPool(256_000_000)


@pytest.mark.parametrize('n', [0, 1, 17, 6272])
@pytest.mark.parametrize('kind', ['dyadic', 'general', 'cancellation'])
def test_nonzero_full_row_matches_independent_fraction(n, kind):
    i = np.arange(n, dtype=np.int64)
    columns = (i % 31).astype(np.int32)
    values = np.ldexp(((i * 37 % 127) - 63).astype(np.float64), (i % 27 - 20).astype(np.int32))
    point = [F(j % 9 - 4, 8 if kind != 'general' else 7) for j in range(31)]
    if kind == 'cancellation':
        columns = (i // 2 % 31).astype(np.int32)
        values = np.ldexp((i // 2 % 19 + 1).astype(np.float64), (i // 2 % 21 - 20).astype(np.int32))
        values[1::2] *= -1
    before = (columns.tobytes(), values.tobytes(), tuple(point))
    expected = sum((F(float(v)) * point[int(k)] for k, v in zip(columns, values)), F(0))
    budget = pool()
    assert dot(columns, values, point, pool=budget, enabled=True) == expected
    assert before == (columns.tobytes(), values.tobytes(), tuple(point))
    if n and kind == 'general':
        assert 'c81_general_fraction_dot' in budget.parts
    else:
        assert 'c81_exact_binned_dot' in budget.parts
    if n == 6272 and kind == 'dyadic':
        assert budget.used < 64 * n


def test_ordinary_binary64_points_and_wider_exact_dyadics():
    values = np.array([.1, -.3, .7, 2.**-16, -3.25, 0., -0.], np.float64)
    columns = np.arange(len(values), dtype=np.int32)
    point = [F(.1), F(-.7), F(.9), F(2**78 + 17, 2**80), F(-.25), F(1), F(0)]
    expected = sum((F(float(v)) * x for v, x in zip(values, point)), F(0))
    assert dot(columns, values, point, pool=pool(), enabled=True) == expected


def test_zero_point_does_not_receive_a_special_work_discount():
    columns = np.arange(64, dtype=np.int32)
    values = np.linspace(-1., 1., 64)
    a, b = pool(), pool()
    assert dot(columns, values, [F(0)] * 64, pool=a, enabled=True) == 0
    dot(columns, values, [F(1, 2)] * 64, pool=b, enabled=True)
    assert a.used == b.used and a.parts == b.parts


@pytest.mark.parametrize('mixed', [False, True])
@pytest.mark.parametrize('subtract', [False, True])
@pytest.mark.parametrize('phase', [False, True])
def test_full_nonzero_unit_EQ_INEQ_phase_inverse_and_independent_rows(mixed, subtract, phase):
    c, h, overlay, _, _, _, info = source(mixed=mixed, subtract=subtract, phase=phase)
    plans, _ = discover_append(c, view_from(c, h), overlay, pool=pool(), enabled=True)
    new, _ = splice_append(view_from(c, h), plans, pool=pool(), enabled=True)
    j = journal(c, plans)
    before = (source_digest(h), source_digest(new), j.eq_roots.tobytes(), j.eq_scales.tobytes())
    for z in (-1, 1):
        for x in (F(-1, 2), F(1, 3), F(1, 2)):
            point = [F(0)] * h.n_cont
            point[0], point[1] = F(z), x
            for i, (_, out, _, _, _, _, sign, offset, beta) in enumerate(info):
                prefix = -F(.25) * point[0] - (F(.125) * point[1] if i % 2 else 0)
                point[out] = F(.125) + sign * (F(offset) - prefix) - F(beta) * z
            expected = j.reconstruct_fraction(new, point, pool=pool())
            budget = pool()
            full, report = reconstruct(c, new, j, plans, point, pool=budget, enabled=True)
            assert full == expected and report['all_equations_exact']
            assert feasible(new, point, (z,)) and feasible(h, full, (z,))
            assert budget.used == report['binned_work_bound']['complete_upper']
            assert not report['feasibility_or_concrete_witness_claim']
    assert before == (source_digest(h), source_digest(new), j.eq_roots.tobytes(), j.eq_scales.tobytes())


@pytest.mark.parametrize('subtract', [False, True])
def test_shared_local_children_restore_after_nonzero_unit(subtract):
    c, h, overlay, *_ = source(redirect=True, subtract=subtract)
    plans, _ = discover_append(c, view_from(c, h), overlay, pool=pool(), enabled=True)
    new, _ = splice_append(view_from(c, h), plans, pool=pool(), enabled=True)
    width = h.n_cont
    tags, nums = zip(encode(plans[0].column, (3, -2)), encode(width, (-1, -1)))
    roots = np.r_[c.eq_roots, np.array(tags, np.int64)]
    scales = np.r_[c.eq_scales, np.array(nums, np.float64).view(np.int64)]
    def widen(m):
        return sp.csr_matrix((m.data, m.indices, m.indptr), shape=(m.shape[0], width + 2))
    def expanded(hz):
        return SparseHZono(hz.c, widen(hz.Gc), hz.Gb, widen(hz.Ac), hz.Ab, hz.b,
            widen(hz.Auc), hz.Aub, hz.ub, frame_id=hz.frame_id, exact=True)
    new = expanded(new)
    c = SimpleNamespace(**{**vars(c), 'hz': expanded(c.hz), 'eq_roots': roots, 'eq_scales': scales})
    j = compile_journal(roots, scales, plans, old_n_cont=c.old_n_cont, old_n_eq=c.old_n_eq,
        source_n_cont=width + 2, source_schema=SCHEMA, pool=pool(), enabled=True)
    point = [F(0)] * new.n_cont
    point[0] = F(1)
    full, report = reconstruct(c, new, j, plans, point, pool=pool(), enabled=True)
    assert full == j.reconstruct_fraction(new, point, pool=pool())
    assert full[plans[0].column] != 0
    assert full[width] == F(3, 4) * full[plans[0].column]
    assert full[width + 1] == -F(1, 2) * full[width]
    assert report['local_equations'] == 2


def test_actual_native_schema_and_full_local_population_fixture():
    state, plans, _ = native()
    before = (state.source.validate()['identity'], source_digest(state.hz))
    point = [F(i % 3 - 1, 64) for i in range(state.hz.n_cont)]
    expected = state.lineage.reconstruct_fraction(state.hz, point, pool=pool())
    full, report = reconstruct(state.source, state.hz, state.lineage, plans, point,
                              pool=pool(), enabled=True)
    assert full == expected and report['local_equations'] > 0
    assert before == (state.source.validate()['identity'], source_digest(state.hz))
    assert state.validate()['full_source_and_native_content_bound']
    assert verify_inverse(state.source, state.hz, state.lineage, plans,
                          pool=pool())['all_equations_exact']


def test_complete_bound_rejects_before_any_inverse_arithmetic():
    c, h, overlay, *_ = source()
    plans, _ = discover_append(c, view_from(c, h), overlay, pool=pool(), enabled=True)
    new, _ = splice_append(view_from(c, h), plans, pool=pool(), enabled=True)
    j = journal(c, plans)
    _, report = reconstruct(c, new, j, plans, [F(0)] * new.n_cont, pool=pool(), enabled=True)
    limited = WorkPool(report['binned_work_bound']['complete_upper'] - 1)
    with pytest.raises(MemoryError, match='complete inverse bound'):
        reconstruct(c, new, j, plans, [F(0)] * new.n_cont, pool=limited, enabled=True)
    assert 'c81_exact_binned_dot' not in limited.parts
    assert 'c68_exact_local_inverse' not in limited.parts


def test_wrong_original_equation_is_not_hidden_by_successful_reconstruction():
    c, h, overlay, *_ = source()
    plans, _ = discover_append(c, view_from(c, h), overlay, pool=pool(), enabled=True)
    new, _ = splice_append(view_from(c, h), plans, pool=pool(), enabled=True)
    j = journal(c, plans)
    changed = [replace(plans[0], offset=plans[0].offset + .125), *plans[1:]]
    with pytest.raises(ValueError, match='original complete producer'):
        reconstruct(c, new, j, changed, [F(0)] * new.n_cont, pool=pool(), enabled=True)


def test_default_off_box_nonfinite_and_budget_guards():
    assert dot(None, None, None, pool=None) is None
    assert reconstruct(None, None, None, None, None, pool=None) is None
    for val in (np.inf, np.nan):
        with pytest.raises(ValueError, match='nonfinite'):
            dot(np.array([0]), np.array([val]), [F(1)], pool=pool(), enabled=True)
    with pytest.raises(ValueError, match='box'):
        dot(np.array([0]), np.array([1.]), [F(2)], pool=pool(), enabled=True)
    with pytest.raises(MemoryError):
        dot(np.array([0]), np.array([1.]), [F(1)], pool=WorkPool(0), enabled=True)
