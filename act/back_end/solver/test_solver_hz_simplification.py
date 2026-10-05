import numpy as np
import pytest
import torch

sp = pytest.importorskip("scipy.sparse")

from act.back_end.solver.solver_hz import (
    HZSolver,
    SparseHZono,
    _coalesce_antiparallel_rows,
    _lower_hz_milp,
    hz_tighten_bounds,
    sparse_hz_prune_unused_factors,
)
from act.back_end.core import Bounds
from act.back_end.hybridz_tf import HybridzTF


def _make_sparse_hz() -> SparseHZono:
    # Continuous columns 1 and 4 and binary column 1 are deliberately absent
    # from both the value map and predicates. Column 0 is also absent but is
    # retained as an input-coordinate prefix by the simplifier call below.
    return SparseHZono(
        c=np.array([0.25, -0.5]),
        Gc=sp.csr_matrix(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -2.0, 0.0],
            ]
        ),
        Gb=sp.csr_matrix([[0.5, 0.0, 0.0], [0.0, 0.0, -0.75]]),
        Ac=sp.csr_matrix([[0.0, 0.0, 1.0, -1.0, 0.0]]),
        Ab=sp.csr_matrix([[1.0, 0.0, 0.0]]),
        b=np.array([0.25]),
        Auc=sp.csr_matrix([[0.0, 0.0, -1.0, 0.0, 0.0]]),
        Aub=sp.csr_matrix([[0.0, 0.0, 1.0]]),
        ub=np.array([1.0]),
        frame_id=7,
        exact=True,
    )


def test_prune_unused_factors_preserves_hz_milp_and_input_prefix():
    hz = _make_sparse_hz()
    compact, removed_cont, removed_bin = sparse_hz_prune_unused_factors(
        hz,
        preserve_cont_prefix=1,
        preserve_bin_prefix=1,
    )

    assert (removed_cont, removed_bin) == (2, 1)
    assert compact.n_cont == 3
    assert compact.n_bin == 2
    assert compact.frame_id == hz.frame_id
    assert compact.exact is True

    original = _lower_hz_milp(hz, prune_unused=False)
    reduced = _lower_hz_milp(compact, prune_unused=False)
    # The preserved prefix remains the first solver coordinate, and every
    # retained value/predicate coefficient is unchanged after deleting zeros.
    np.testing.assert_allclose(
        reduced.value_matrix.toarray(),
        original.value_matrix[:, [0, 2, 3, 5, 7]].toarray(),
    )
    np.testing.assert_allclose(
        reduced.A.toarray(),
        original.A[:, [0, 2, 3, 5, 7]].toarray(),
    )
    np.testing.assert_allclose(reduced.value_center, original.value_center)
    np.testing.assert_allclose(reduced.row_lb, original.row_lb)
    np.testing.assert_allclose(reduced.row_ub, original.row_ub)


def test_prune_unused_factors_is_identity_when_every_factor_is_used():
    hz = _make_sparse_hz()
    used = SparseHZono(
        c=hz.c,
        Gc=hz.Gc + sp.csr_matrix(
            ([1.0, 1.0, 1.0], ([0, 0, 1], [0, 1, 4])), shape=hz.Gc.shape
        ),
        Gb=hz.Gb + sp.csr_matrix(([1.0], ([0], [1])), shape=hz.Gb.shape),
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        Auc=hz.Auc,
        Aub=hz.Aub,
        ub=hz.ub,
        frame_id=hz.frame_id,
        exact=hz.exact,
    )
    compact, removed_cont, removed_bin = sparse_hz_prune_unused_factors(used)
    assert compact is used
    assert (removed_cont, removed_bin) == (0, 0)


@pytest.mark.parametrize(
    ("preserve_cont", "preserve_bin"),
    [(-1, 0), (6, 0), (0, -1), (0, 4)],
)
def test_prune_unused_factors_rejects_invalid_prefixes(preserve_cont, preserve_bin):
    with pytest.raises(ValueError):
        sparse_hz_prune_unused_factors(
            _make_sparse_hz(),
            preserve_cont_prefix=preserve_cont,
            preserve_bin_prefix=preserve_bin,
        )


def test_coalesce_parallel_and_antiparallel_rows_intersects_predicates():
    matrix = sp.csr_matrix(
        [
            [1.0, -2.0],
            [1.0, -2.0],
            [-1.0, 2.0],
            [0.0, 3.0],
        ]
    )
    lower = np.array([-3.0, -1.0, -1.0, -np.inf])
    upper = np.array([5.0, 2.0, 0.0, 7.0])
    compact, compact_lb, compact_ub = _coalesce_antiparallel_rows(
        matrix, lower, upper
    )

    assert compact.shape == (2, 2)
    np.testing.assert_allclose(compact.toarray(), [[1.0, -2.0], [0.0, 3.0]])
    # Same-direction intersection gives [-1, 2]; the negated row denotes
    # [0, 1] in the representative orientation, giving the final intersection.
    np.testing.assert_allclose(compact_lb[0], 0.0)
    np.testing.assert_allclose(compact_ub[0], 1.0)
    assert np.isneginf(compact_lb[1])
    np.testing.assert_allclose(compact_ub[1], 7.0)


def test_lowering_coalesces_duplicate_hz_predicates_without_relaxation():
    hz = _make_sparse_hz()
    duplicated = SparseHZono(
        c=hz.c,
        Gc=hz.Gc,
        Gb=hz.Gb,
        Ac=sp.vstack([hz.Ac, hz.Ac], format="csr"),
        Ab=sp.vstack([hz.Ab, hz.Ab], format="csr"),
        b=np.concatenate([hz.b, hz.b]),
        Auc=sp.vstack([hz.Auc, -hz.Auc], format="csr"),
        Aub=sp.vstack([hz.Aub, -hz.Aub], format="csr"),
        ub=np.array([1.0, 2.0]),
        frame_id=hz.frame_id,
        exact=hz.exact,
    )
    model = _lower_hz_milp(duplicated)
    assert model.A.shape[0] == 2
    assert np.count_nonzero(np.isclose(model.row_lb, model.row_ub)) == 1


def test_pruned_solver_coordinates_recover_a_valid_input_assignment():
    input_hz = SparseHZono(
        c=np.array([0.0]),
        Gc=sp.csr_matrix([[1.0, 10.0, 100.0]]),
        Gb=sp.csr_matrix([[5.0, 50.0]]),
        Ac=sp.csr_matrix((0, 3)),
        Ab=sp.csr_matrix((0, 2)),
        b=np.zeros(0),
        frame_id=11,
        exact=True,
    )
    output_hz = SparseHZono(
        c=np.array([0.0]),
        Gc=sp.csr_matrix([[1.0, 0.0, 1.0]]),
        Gb=sp.csr_matrix([[0.0, 1.0]]),
        Ac=sp.csr_matrix((0, 3)),
        Ab=sp.csr_matrix((0, 2)),
        b=np.zeros(0),
        frame_id=11,
        exact=True,
    )
    model = _lower_hz_milp(output_hz)
    np.testing.assert_array_equal(model.cont_source, [0, 2])
    np.testing.assert_array_equal(model.bin_source, [1])

    # Solver coordinates are continuous sources 0/2 followed by binary source
    # 1 in {0,1}. Missing continuous source 1 is reconstructed at 0, while
    # missing binary source 0 is reconstructed at the valid HZ value -1.
    recovered = HZSolver._recover_input(
        model,
        np.array([0.2, -0.3, 1.0]),
        input_hz,
        (1, 1),
        lane=0,
    )
    assert recovered is not None
    np.testing.assert_allclose(recovered.numpy(), [15.2])


def test_nonfinite_hz_candidate_endpoints_fall_back_to_interval_bounds():
    base = Bounds(
        lb=torch.tensor([[-2.0, -1.0, 0.0]], dtype=torch.float64),
        ub=torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float64),
    )
    candidate = Bounds(
        lb=torch.tensor([[float("nan"), -0.5, -float("inf")]], dtype=torch.float64),
        ub=torch.tensor([[1.5, float("inf"), 3.5]], dtype=torch.float64),
    )
    tightened = hz_tighten_bounds(base, candidate)
    torch.testing.assert_close(
        tightened.lb, torch.tensor([[-2.0, -0.5, 0.0]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        tightened.ub, torch.tensor([[1.5, 3.0, 3.5]], dtype=torch.float64)
    )


def test_release_intermediate_hz_keeps_retained_objects_alive_and_resets_frame():
    tf = HybridzTF()
    retained = _make_sparse_hz()
    tf._sparse_hz_cache[3] = retained
    tf._sparse_hz_cache[4] = _make_sparse_hz()
    tf._cache_net_id = 123
    tf._sparse_next_frame_id = 9
    tf._sparse_frame_widths[7] = (5, 3)
    tf._sparse_aux_slots[(7, 4)] = (5,)

    released = tf.release_intermediate_hz()

    assert released == (0, 2)
    assert retained.n_cont == 5
    assert tf._sparse_hz_cache == {}
    assert tf._cache_net_id is None
    assert tf._sparse_next_frame_id == 0
    assert tf._sparse_frame_widths == {}
    assert tf._sparse_aux_slots == {}
