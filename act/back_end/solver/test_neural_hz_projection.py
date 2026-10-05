import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")

from act.back_end.solver.neural_hz import (
    fix_predicate_implied_binary_phases,
    project_inactive_equality_factors,
    reconstruct_continuous_factors,
)
from act.back_end.solver.solver_hz import HZSolver, SparseHZono, _lower_hz_milp


def _chain_hz() -> SparseHZono:
    # y = x0; x1 and x2 are predicate-only auxiliaries:
    #   x1 + x0 = 0
    #   x2 + x1 = 0
    return SparseHZono(
        c=np.array([0.0]),
        Gc=sp.csr_matrix([[1.0, 0.0, 0.0]]),
        Gb=sp.csr_matrix((1, 0)),
        Ac=sp.csr_matrix([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
        Ab=sp.csr_matrix((2, 0)),
        b=np.zeros(2),
        frame_id=4,
        exact=True,
    )


def test_neural_hz_projection_is_opt_in_and_eliminates_a_predicate_chain():
    hz = _chain_hz()
    baseline = _lower_hz_milp(hz)
    projected = _lower_hz_milp(hz, project_inactive_cont=True)

    assert baseline.n_cont == 3
    assert baseline.cont_eliminations == ()
    assert projected.n_cont == 1
    assert projected.n_bin == 0
    assert projected.cont_source.tolist() == [0]
    assert [item.source for item in projected.cont_eliminations] == [2, 1]
    assert projected.A.nnz < baseline.A.nnz

    # Both predicate systems denote exactly x0 in [-1, 1]. The projected box
    # row is the image of both eliminated latent boxes.
    for x0 in np.linspace(-1.0, 1.0, 9):
        original = np.array([x0, -x0, x0])
        np.testing.assert_allclose(baseline.A @ original, baseline.row_lb)
        value = np.asarray(projected.A @ np.array([x0])).reshape(-1)
        assert np.all(value >= projected.row_lb - 1e-12)
        assert np.all(value <= projected.row_ub + 1e-12)


def test_neural_hz_projection_cost_gate_skips_small_models_uniformly():
    hz = _chain_hz()
    gated = _lower_hz_milp(
        hz,
        project_inactive_cont=True,
        neural_hz_min_problem_size=10_000,
    )

    assert gated.n_cont == 3
    assert gated.cont_eliminations == ()


def test_neural_hz_projection_reconstructs_eliminated_input_factors():
    model = _lower_hz_milp(_chain_hz(), project_inactive_cont=True)
    input_hz = SparseHZono(
        c=np.array([0.0, 0.0]),
        Gc=sp.csr_matrix([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        Gb=sp.csr_matrix((2, 0)),
        Ac=sp.csr_matrix((0, 3)),
        Ab=sp.csr_matrix((0, 0)),
        b=np.zeros(0),
        frame_id=4,
        exact=True,
    )

    recovered = HZSolver._recover_input(
        model,
        np.array([0.25]),
        input_hz,
        (1, 2),
        lane=0,
    )

    assert recovered is not None
    # x1 = -x0 and x2 = x0, reconstructed in reverse elimination order.
    np.testing.assert_allclose(recovered.numpy(), [-0.25, 0.25])


def test_neural_hz_projection_keeps_binary_factors_and_reconstructs_through_them():
    # HZ coordinates: x0 + x1 + beta = 0. Lowering uses beta = 2*z-1,
    # hence x1 = 1 - x0 - 2*z after projecting predicate-only x1.
    hz = SparseHZono(
        c=np.array([0.0]),
        Gc=sp.csr_matrix([[1.0, 0.0]]),
        Gb=sp.csr_matrix((1, 1)),
        Ac=sp.csr_matrix([[1.0, 1.0]]),
        Ab=sp.csr_matrix([[1.0]]),
        b=np.zeros(1),
        frame_id=5,
        exact=True,
    )
    model = _lower_hz_milp(hz, project_inactive_cont=True)

    assert model.n_cont == 1
    assert model.n_bin == 1
    assert model.bin_source.tolist() == [0]
    assert len(model.cont_eliminations) == 1
    for z in (0.0, 1.0):
        x0 = 0.25 if z == 0.0 else -0.25
        x1 = 1.0 - x0 - 2.0 * z
        assert -1.0 <= x1 <= 1.0
        assignment = np.array([x0, z])
        values = np.asarray(model.A @ assignment).reshape(-1)
        assert np.all(values >= model.row_lb - 1e-12)
        assert np.all(values <= model.row_ub + 1e-12)
        reconstructed = reconstruct_continuous_factors(
            np.array([x0]),
            np.array([z]),
            model.cont_source,
            model.bin_source,
            model.cont_eliminations,
        )
        np.testing.assert_allclose(reconstructed[1], x1)


def test_fill_increasing_projection_is_rejected():
    # Only x1 is output-inactive. Substituting its four-wide defining equality
    # into the second row would increase predicate nnz from 6 to 7.
    value = sp.csr_matrix(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    predicates = sp.csr_matrix(
        [[0.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0]]
    )
    projection = project_inactive_equality_factors(
        value,
        predicates,
        np.array([0.0, -np.inf]),
        np.array([0.0, 1.0]),
        np.arange(5),
        np.zeros(0, dtype=np.int64),
    )

    assert projection.removed_cont == 0
    assert projection.constraint_matrix.nnz == predicates.nnz
    assert projection.cont_sources.tolist() == list(range(5))


def _phase_reduce(constraints, lower, upper, *, value_center=None, value=None):
    constraints = sp.csr_matrix(constraints, dtype=np.float64)
    n_var = constraints.shape[1]
    if value_center is None:
        value_center = np.zeros(1)
    if value is None:
        value = sp.csr_matrix((1, n_var))
    return fix_predicate_implied_binary_phases(
        value_center,
        value,
        constraints,
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        np.array([0], dtype=np.int64),
        np.arange(n_var - 1, dtype=np.int64),
    )


def test_predicate_implied_phase_fixes_one_and_substitutes_output():
    # x + 2*z = 2.5 with x in [-1, 1] implies z = 1.
    reduced = _phase_reduce(
        [[1.0, 2.0]],
        [2.5],
        [2.5],
        value_center=np.array([10.0]),
        value=sp.csr_matrix([[3.0, 4.0]]),
    )

    assert [(fix.source, fix.value) for fix in reduced.fixes] == [(0, 1)]
    assert reduced.bin_sources.size == 0
    np.testing.assert_allclose(reduced.value_center, [14.0])
    np.testing.assert_allclose(reduced.value_matrix.toarray(), [[3.0]])
    np.testing.assert_allclose(reduced.constraint_matrix.toarray(), [[1.0]])
    np.testing.assert_allclose(reduced.row_lower, [0.5])
    np.testing.assert_allclose(reduced.row_upper, [0.5])


def test_predicate_implied_phase_fixes_zero_from_inequality():
    # x + 2*z <= 0.5 rules out z = 1 even after relaxing x to its box.
    reduced = _phase_reduce([[1.0, 2.0]], [-np.inf], [0.5])

    assert [(fix.source, fix.value) for fix in reduced.fixes] == [(0, 0)]
    assert reduced.bin_sources.size == 0
    np.testing.assert_allclose(reduced.constraint_matrix.toarray(), [[1.0]])
    assert np.isneginf(reduced.row_lower[0])
    np.testing.assert_allclose(reduced.row_upper, [0.5])


def test_predicate_implied_phase_fixing_cascades_without_branching():
    # The first row fixes a=1. Its substitution makes the second row fix b=0.
    reduced = _phase_reduce(
        [[1.0, 2.0, 0.0], [0.0, 2.0, 2.0]],
        [2.5, -np.inf],
        [2.5, 2.5],
    )

    assert [(fix.source, fix.value) for fix in reduced.fixes] == [(0, 1), (1, 0)]
    assert reduced.bin_sources.size == 0
    np.testing.assert_allclose(reduced.constraint_matrix.toarray(), [[1.0], [0.0]])
    np.testing.assert_allclose(reduced.row_lower, [0.5, -np.inf])
    np.testing.assert_allclose(reduced.row_upper, [0.5, 0.5])


def test_predicate_phase_is_retained_when_both_values_survive_relaxation():
    reduced = _phase_reduce([[1.0, 1.0]], [0.5], [0.5])

    assert reduced.fixes == ()
    assert reduced.bin_sources.tolist() == [0]
    np.testing.assert_allclose(reduced.constraint_matrix.toarray(), [[1.0, 1.0]])


def test_predicate_phase_detects_empty_integer_hz_fail_closed():
    # z = 0.5 has no binary solution. The explicit contradictory row lets the
    # downstream MILP prove empty; the verifier still reports UNKNOWN.
    reduced = _phase_reduce([[0.0, 1.0]], [0.5], [0.5])

    assert reduced.proven_empty
    assert reduced.fixes == ()
    np.testing.assert_allclose(reduced.constraint_matrix.toarray()[-1], [0.0, 0.0])
    assert reduced.row_lower[-1] > reduced.row_upper[-1]


def test_phase_fixing_is_opt_in_and_reconstructs_input_binary():
    # Original HZ predicate x + beta = 1.5 implies beta=+1 and x=0.5.
    hz = SparseHZono(
        c=np.array([0.0]),
        Gc=sp.csr_matrix([[0.0]]),
        Gb=sp.csr_matrix([[2.0]]),
        Ac=sp.csr_matrix([[1.0]]),
        Ab=sp.csr_matrix([[1.0]]),
        b=np.array([1.5]),
        frame_id=6,
        exact=True,
    )
    baseline = _lower_hz_milp(hz)
    fixed = _lower_hz_milp(hz, fix_implied_binary=True)

    assert baseline.n_bin == 1
    assert fixed.n_bin == 0
    assert [(item.source, item.value) for item in fixed.bin_fixes] == [(0, 1)]
    np.testing.assert_allclose(fixed.value_center, [2.0])
    input_hz = SparseHZono(
        c=np.zeros(2),
        Gc=sp.csr_matrix([[1.0], [0.0]]),
        Gb=sp.csr_matrix([[0.0], [1.0]]),
        Ac=sp.csr_matrix((0, 1)),
        Ab=sp.csr_matrix((0, 1)),
        b=np.zeros(0),
        frame_id=6,
        exact=True,
    )
    recovered = HZSolver._recover_input(
        fixed,
        np.array([0.5]),
        input_hz,
        (1, 2),
        lane=0,
    )
    assert recovered is not None
    np.testing.assert_allclose(recovered.numpy(), [0.5, 1.0])
