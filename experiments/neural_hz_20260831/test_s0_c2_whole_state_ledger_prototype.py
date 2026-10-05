from __future__ import annotations

from dataclasses import fields
import gc
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds
from act.back_end.hybridz_tf.exact_linear_op import (
    CSRLinearOp,
    DiagonalLinearOp,
    ImplicitConv2DOp,
)
from act.back_end.hybridz_tf.tf_cnn import (
    SparseHZAffineExpr,
    SparseHZAffineTerm,
)
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.s0_c2_whole_state_ledger_prototype import (
    ConsumerGCStep,
    ROOT_MISSING,
    RootPath,
    RootSubstitution,
    StrongRetentionPlan,
    WHOLE_STATE_NO_CLAIMS,
    WholeStateReject,
    WholeStateRoots,
    assess_hypothetical_substitution,
    snapshot_whole_state,
)


def _hz(width: int = 4, *, frame_id: int = 91) -> SparseHZono:
    value_rows = np.arange(width, dtype=np.int64)
    value_cols = np.arange(width, dtype=np.int64)
    return SparseHZono(
        c=np.linspace(-0.5, 0.5, width, dtype=np.float64),
        Gc=sp.csr_matrix(
            (
                np.linspace(1.0, 2.0, width, dtype=np.float64),
                (value_rows, value_cols),
            ),
            shape=(width, width),
        ),
        Gb=sp.csr_matrix(
            (
                np.array([0.25, -0.5], dtype=np.float64),
                (
                    np.array([0, width - 1], dtype=np.int64),
                    np.array([0, 0], dtype=np.int64),
                ),
            ),
            shape=(width, 1),
        ),
        Ac=sp.csr_matrix(
            (np.array([1.0]), (np.array([0]), np.array([0]))),
            shape=(1, width),
        ),
        Ab=sp.csr_matrix(np.array([[1.0]], dtype=np.float64)),
        b=np.array([0.125], dtype=np.float64),
        Auc=sp.csr_matrix(
            (np.array([-1.0]), (np.array([0]), np.array([width - 1]))),
            shape=(1, width),
        ),
        Aub=sp.csr_matrix(np.array([[0.75]], dtype=np.float64)),
        ub=np.array([1.25], dtype=np.float64),
        frame_id=frame_id,
        exact=True,
    )


def _expr(
    source: SparseHZono,
    operators: tuple[object, ...] = (),
) -> SparseHZAffineExpr:
    output_width = source.n_out if not operators else int(operators[-1].shape[0])
    return SparseHZAffineExpr(
        terms=(SparseHZAffineTerm(source, operators),),
        bias=np.linspace(0.0, 0.25, output_width, dtype=np.float64),
        n_out=output_width,
        frame_id=int(source.frame_id),
    )


def _bounds(width: int = 4) -> Bounds:
    return Bounds(
        lb=torch.linspace(-1.0, 0.0, width, dtype=torch.float64),
        ub=torch.linspace(0.5, 1.5, width, dtype=torch.float64),
    )


def _roles(ledger) -> tuple[str, ...]:
    return tuple(
        role
        for storage in ledger.storage_provenance
        for role in storage.roles
    )


def _replace_active(old, new, *, key: str = "target") -> RootSubstitution:
    return RootSubstitution(
        RootPath("active", key), expected=old, replacement=new
    )


def test_complete_explicit_root_surface_and_operator_payloads_are_visited():
    source = _hz()
    raw_csr = sp.eye(4, format="csr", dtype=np.float64)
    csr_operator = CSRLinearOp(sp.eye(4, format="csr", dtype=np.float64))
    diagonal = DiagonalLinearOp(np.array([1.0, 2.0, 3.0, 4.0]))
    convolution = ImplicitConv2DOp(
        np.ones((1, 1, 1, 1), dtype=np.float64),
        (1, 1, 2, 2),
        row_mask=np.array([True, False, True, True]),
    )
    expression = _expr(
        source, (raw_csr, csr_operator, diagonal, convolution)
    )
    bounds = _bounds()
    expected_lb = torch.arange(4, dtype=torch.float64)
    expected_ub = expected_lb.clone() + 1.0
    descriptor_payload = np.arange(5, dtype=np.float32)
    artifact_payload = sp.csr_matrix(np.array([[0.0, 2.0, 0.0]]))
    active_payload = np.arange(3, dtype=np.int16)
    pending_payload = np.arange(2, dtype=np.float64)

    ledger = snapshot_whole_state(
        WholeStateRoots(
            sparse_hz={1: source},
            affine_expr={2: expression},
            precomputed_relu={3: (source, expected_lb, expected_ub, expression, bounds)},
            phase_bounds={4: bounds},
            descriptor={"payload": descriptor_payload, "conv": convolution},
            artifact={"csr": artifact_payload},
            active={"numeric": active_payload},
            pending={
                "plan": StrongRetentionPlan((pending_payload,), "c2-stage")
            },
        )
    )

    roles = _roles(ledger)
    assert any("value.Gc.data" in role for role in roles)
    assert any("predicate.Auc.data" in role for role in roles)
    assert any("operators[0].data" in role for role in roles)
    assert any("operators[1].matrix.data" in role for role in roles)
    assert any("operators[2].diagonal" in role for role in roles)
    assert any("operators[3].kernel" in role for role in roles)
    assert any("precomputed_relu[3].slot1" in role for role in roles)
    assert any("precomputed_relu[3].slot2" in role for role in roles)
    assert any("precomputed_relu[3].slot4.lb" in role for role in roles)
    assert any("descriptor['payload']" in role for role in roles)
    assert any("artifact['csr'].data" in role for role in roles)
    assert any("active['numeric']" in role for role in roles)
    assert any("pending['plan'].strong_roots[0]" in role for role in roles)
    assert ledger.object_count("SparseHZono") == 1
    assert ledger.object_count("SparseHZAffineExpr") == 1
    assert ledger.object_count("StrongRetentionPlan") == 1
    assert ledger.phase_bounds_entries_included
    assert not ledger.python_object_bytes_included
    assert ledger.measured_rss_bytes is None
    assert ledger.no_claims == WHOLE_STATE_NO_CLAIMS


def test_numpy_same_owner_same_full_span_views_are_deduplicated():
    owner = np.arange(12, dtype=np.float64)
    first = owner.view()
    second = owner.reshape(3, 4)

    ledger = snapshot_whole_state(
        WholeStateRoots(active={"first": first, "second": second})
    )

    assert ledger.resident_bytes == owner.nbytes
    assert ledger.resident_entries == owner.size
    assert ledger.numeric_storage_count == 1
    assert ledger.object_count("numpy.ndarray") == 2
    record = ledger.storage_provenance[0]
    assert record.token == "storage-0001"
    assert len(record.roles) == 2
    assert "0x" not in repr(record)


def test_numpy_small_view_charges_complete_final_owner_allocation():
    owner = np.arange(1024, dtype=np.float64)
    small_view = owner[17:21]

    ledger = snapshot_whole_state(
        WholeStateRoots(active={"small-view": small_view})
    )

    assert small_view.nbytes < owner.nbytes
    assert ledger.resident_bytes == owner.nbytes
    assert ledger.resident_entries == owner.size
    assert ledger.numeric_storage_count == 1


def test_numpy_disjoint_views_keep_one_complete_owner_allocation():
    owner = np.arange(128, dtype=np.float64)
    first = owner[:8]
    second = owner[-8:]

    ledger = snapshot_whole_state(
        WholeStateRoots(active={"first": first, "second": second})
    )

    assert ledger.resident_bytes == owner.nbytes
    assert ledger.resident_entries == owner.size
    assert ledger.numeric_storage_count == 1
    assert len(ledger.storage_provenance[0].roles) == 2


def test_numpy_same_owner_partial_overlap_fails_closed_without_addresses():
    owner = np.arange(12, dtype=np.float64)
    first = owner[:8]
    second = owner[5:]
    roots = WholeStateRoots(active={"first": first, "second": second})

    with pytest.raises(WholeStateReject, match="partial_numpy_overlap") as caught:
        snapshot_whole_state(roots)
    assert "0x" not in str(caught.value)

    decision = assess_hypothetical_substitution(roots, ())
    assert not decision.accepted
    assert decision.reason.startswith("partial_numpy_overlap")


def test_numpy_unknown_external_buffer_and_gapped_view_fail_closed():
    external = np.frombuffer(bytearray(64), dtype=np.float64)
    owner = np.arange(12, dtype=np.float64)
    gapped = owner[::2]

    with pytest.raises(WholeStateReject, match="unknown_numpy_external_buffer"):
        snapshot_whole_state(WholeStateRoots(active={"x": external}))
    with pytest.raises(WholeStateReject, match="gapped_numpy_view"):
        snapshot_whole_state(WholeStateRoots(active={"x": gapped}))


def test_equal_numpy_content_in_distinct_allocations_is_double_counted():
    first = np.ones(9, dtype=np.float64)
    second = np.ones(9, dtype=np.float64)
    assert np.array_equal(first, second)

    ledger = snapshot_whole_state(
        WholeStateRoots(active={"first": first, "second": second})
    )

    assert ledger.numeric_storage_count == 2
    assert ledger.resident_bytes == first.nbytes + second.nbytes
    assert ledger.resident_entries == first.size + second.size


def test_torch_untyped_storage_aliases_deduplicate_and_clones_do_not():
    owner = torch.arange(10, dtype=torch.float64)
    view = owner.reshape(2, 5)
    clone = owner.clone()

    aliased = snapshot_whole_state(
        WholeStateRoots(active={"owner": owner, "view": view})
    )
    distinct = snapshot_whole_state(
        WholeStateRoots(active={"owner": owner, "clone": clone})
    )

    assert aliased.numeric_storage_count == 1
    assert aliased.resident_bytes == owner.untyped_storage().nbytes()
    assert aliased.resident_entries == 10
    assert distinct.numeric_storage_count == 2
    assert distinct.resident_bytes == 2 * owner.untyped_storage().nbytes()
    assert all("0x" not in repr(item) for item in aliased.storage_provenance)


def test_csr_counts_data_entries_but_all_three_buffer_bytes():
    matrix = sp.csr_matrix(
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([0, 2, 1], dtype=np.int32),
            np.array([0, 2, 3], dtype=np.int32),
        ),
        shape=(2, 3),
    )
    ledger = snapshot_whole_state(WholeStateRoots(artifact={"m": matrix}))

    assert ledger.resident_entries == matrix.data.size
    assert ledger.resident_bytes == (
        matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    )
    assert ledger.numeric_storage_count == 3
    assert sorted(item.resident_entries for item in ledger.storage_provenance) == [0, 0, 3]
    assert "data.size_only" in ledger.csr_entry_convention


def test_sparse_hz_value_and_all_predicate_buffers_are_in_scope():
    source = _hz(5)
    ledger = snapshot_whole_state(WholeStateRoots(sparse_hz={7: source}))

    matrices = (
        source.Gc,
        source.Gb,
        source.Ac,
        source.Ab,
        source.Auc,
        source.Aub,
    )
    expected_entries = (
        source.c.size
        + source.b.size
        + source.ub.size
        + sum(matrix.data.size for matrix in matrices)
    )
    assert ledger.resident_entries == expected_entries
    roles = _roles(ledger)
    for field_name in ("c", "Gc", "Gb"):
        assert any(f"value.{field_name}" in role for role in roles)
    for field_name in ("Ac", "Ab", "b", "Auc", "Aub", "ub"):
        assert any(f"predicate.{field_name}" in role for role in roles)


def test_precomputed_slots_zero_through_four_and_top_level_phase_bounds():
    source = _hz()
    expression = _expr(source)
    expected_lb = torch.arange(4, dtype=torch.float64)
    expected_ub = torch.arange(4, dtype=torch.float64) + 1.0
    phase = _bounds()

    ledger = snapshot_whole_state(
        WholeStateRoots(
            precomputed_relu={9: (source, expected_lb, expected_ub, expression, phase)},
            phase_bounds={9: phase},
        )
    )
    roles = _roles(ledger)

    assert any("slot0.value.c" in role for role in roles)
    assert any("slot1" in role for role in roles)
    assert any("slot2" in role for role in roles)
    assert any("slot3.bias" in role for role in roles)
    assert any("slot4.lb" in role for role in roles)
    assert any("slot4.ub" in role for role in roles)
    assert ledger.object_count("Bounds") == 1
    assert ledger.object_count("precomputed_tuple") == 1


def test_unknown_operator_and_unknown_numeric_root_fail_closed():
    source = _hz()
    unknown_operator = SimpleNamespace(shape=(source.n_out, source.n_out))
    expression = _expr(source, (unknown_operator,))

    with pytest.raises(WholeStateReject, match="unknown_operator"):
        snapshot_whole_state(WholeStateRoots(affine_expr={1: expression}))
    numeric_decision = assess_hypothetical_substitution(
        WholeStateRoots(active={"foreign": memoryview(bytearray(8))}), ()
    )
    assert not numeric_decision.accepted
    assert numeric_decision.reason.startswith("unknown_numeric_root")


def test_identity_cas_strict_reduction_is_pure_and_result_retains_no_roots():
    old = np.arange(20, dtype=np.float64)
    new = np.arange(5, dtype=np.float64)
    active = {"target": old}
    original_items = tuple(active.items())

    decision = assess_hypothetical_substitution(
        WholeStateRoots(active=active), (_replace_active(old, new),)
    )

    assert decision.accepted
    assert decision.reason == "accepted_strict_whole_state_reduction"
    assert decision.before.resident_bytes == old.nbytes
    assert decision.after.resident_bytes == new.nbytes
    assert tuple(active.items()) == original_items
    assert active["target"] is old
    assert not {
        "roots",
        "substitutions",
        "expected",
        "replacement",
        "plan",
    } & {field.name for field in fields(type(decision))}


def test_path_substitution_does_not_replace_other_alias_roots():
    old = np.arange(20, dtype=np.float64)
    new = np.arange(2, dtype=np.float64)
    roots = WholeStateRoots(active={"selected": old, "other": old})

    decision = assess_hypothetical_substitution(
        roots,
        (
            RootSubstitution(
                RootPath("active", "selected"), old, replacement=new
            ),
        ),
    )

    assert not decision.accepted
    assert decision.reason == "resident_bytes_not_strictly_reduced"
    assert decision.before.resident_bytes == old.nbytes
    assert decision.after.resident_bytes == old.nbytes + new.nbytes


def test_path_cas_requires_expected_identity_and_missing_for_insert():
    old = np.arange(8, dtype=np.float64)
    impostor = old.copy()
    new = np.arange(2, dtype=np.float64)
    roots = WholeStateRoots(active={"target": old})

    mismatch = assess_hypothetical_substitution(
        roots,
        (
            RootSubstitution(
                RootPath("active", "target"), impostor, replacement=new
            ),
        ),
    )
    assert not mismatch.accepted
    assert mismatch.reason == "path_cas_expected_identity_mismatch"

    insert_without_missing = assess_hypothetical_substitution(
        roots,
        (
            RootSubstitution(
                RootPath("pending", "new"), old, replacement=new
            ),
        ),
    )
    assert not insert_without_missing.accepted
    assert insert_without_missing.reason == "path_cas_expected_identity_mismatch"

    valid_insert_and_remove = assess_hypothetical_substitution(
        roots,
        (
            RootSubstitution(
                RootPath("pending", "new"), ROOT_MISSING, replacement=new
            ),
            RootSubstitution(
                RootPath("active", "target"), old, remove=True
            ),
        ),
    )
    assert valid_insert_and_remove.accepted
    assert roots.pending == {}
    assert roots.active["target"] is old


def test_precomputed_slot_cas_rebuilds_only_private_tuple():
    source = _hz()
    lower = torch.zeros(16, dtype=torch.float64)
    upper = torch.ones(16, dtype=torch.float64)
    small_upper = torch.ones(4, dtype=torch.float64)
    original_tuple = (source, lower, upper)
    cache = {3: original_tuple}

    decision = assess_hypothetical_substitution(
        WholeStateRoots(precomputed_relu=cache),
        (
            RootSubstitution(
                RootPath("precomputed_relu", 3, slot=2),
                upper,
                replacement=small_upper,
            ),
        ),
    )

    assert decision.accepted
    assert cache[3] is original_tuple
    assert cache[3][2] is upper


def test_bytes_and_entries_must_each_strictly_decrease():
    bytes_down_entries_up_old = np.arange(10, dtype=np.float64)
    bytes_down_entries_up_new = np.arange(11, dtype=np.uint8)
    entry_reject = assess_hypothetical_substitution(
        WholeStateRoots(active={"target": bytes_down_entries_up_old}),
        (_replace_active(bytes_down_entries_up_old, bytes_down_entries_up_new),),
    )
    assert not entry_reject.accepted
    assert entry_reject.after.resident_bytes < entry_reject.before.resident_bytes
    assert entry_reject.after.resident_entries > entry_reject.before.resident_entries
    assert entry_reject.reason == "resident_entries_not_strictly_reduced"

    entries_down_bytes_up_old = np.arange(10, dtype=np.uint8)
    entries_down_bytes_up_new = np.arange(9, dtype=np.float64)
    byte_reject = assess_hypothetical_substitution(
        WholeStateRoots(active={"target": entries_down_bytes_up_old}),
        (_replace_active(entries_down_bytes_up_old, entries_down_bytes_up_new),),
    )
    assert not byte_reject.accepted
    assert byte_reject.after.resident_entries < byte_reject.before.resident_entries
    assert byte_reject.after.resident_bytes > byte_reject.before.resident_bytes
    assert byte_reject.reason == "resident_bytes_not_strictly_reduced"


def test_pending_plan_strongly_retaining_old_root_is_charged_and_rejected():
    old = np.arange(32, dtype=np.float64)
    new = np.arange(4, dtype=np.float64)
    roots = WholeStateRoots(active={"target": old})
    rewrite_only = assess_hypothetical_substitution(
        roots, (_replace_active(old, new),)
    )
    assert rewrite_only.accepted

    retaining_plan = StrongRetentionPlan((old,), "bad-plan")
    retained = assess_hypothetical_substitution(
        roots,
        (
            _replace_active(old, new),
            RootSubstitution(
                RootPath("pending", "plan"),
                ROOT_MISSING,
                replacement=retaining_plan,
            ),
        ),
    )
    assert not retained.accepted
    assert retained.reason == "resident_bytes_not_strictly_reduced"
    assert retained.before.resident_bytes == old.nbytes
    assert retained.after.resident_bytes == old.nbytes + new.nbytes
    assert retained.after.object_count("StrongRetentionPlan") == 1


def test_live_and_dead_weak_references_are_never_promoted_to_strong_roots():
    live_target = np.arange(100, dtype=np.float64)
    live_reference = weakref.ref(live_target)
    dead_target = np.arange(200, dtype=np.float64)
    dead_reference = weakref.ref(dead_target)
    del dead_target
    gc.collect()
    assert dead_reference() is None

    ledger = snapshot_whole_state(
        WholeStateRoots(
            pending={"live": live_reference, "dead": dead_reference}
        )
    )

    assert ledger.resident_bytes == 0
    assert ledger.resident_entries == 0
    assert ledger.live_weak_references_ignored == 1
    assert ledger.dead_weak_references_ignored == 1


def _gc_roots(*, remaining: int, pinned: bool):
    source = _hz()
    expression = _expr(source)
    phase = _bounds()
    old = np.arange(20, dtype=np.float64)
    new = np.arange(4, dtype=np.float64)
    roots = WholeStateRoots(
        sparse_hz={1: source},
        affine_expr={1: expression},
        phase_bounds={1: phase},
        active={"target": old},
        remaining_consumers={1: remaining},
        pinned_layers=frozenset({1}) if pinned else frozenset(),
    )
    return roots, old, new


def test_last_consumer_gc_is_applied_to_both_sides_before_comparison():
    roots, old, new = _gc_roots(remaining=1, pinned=False)
    decision = assess_hypothetical_substitution(
        roots,
        (_replace_active(old, new),),
        consumer_gc_steps=(ConsumerGCStep((1,)),),
    )

    assert decision.accepted
    assert decision.before.resident_bytes == old.nbytes
    assert decision.after.resident_bytes == new.nbytes
    assert decision.baseline_released_paths == (
        "sparse_hz[1]",
        "affine_expr[1]",
        "phase_bounds[1]",
    )
    assert decision.candidate_released_paths == decision.baseline_released_paths
    assert decision.remaining_consumers_after == (("1", 0),)
    assert 1 in roots.sparse_hz
    assert 1 in roots.affine_expr
    assert 1 in roots.phase_bounds
    assert roots.remaining_consumers[1] == 1


@pytest.mark.parametrize(
    ("remaining", "pinned", "expected_remaining"),
    ((2, False, 1), (1, True, 0)),
)
def test_remaining_or_pinned_consumer_state_keeps_all_predecessor_roots(
    remaining: int, pinned: bool, expected_remaining: int
):
    roots, old, new = _gc_roots(remaining=remaining, pinned=pinned)
    decision = assess_hypothetical_substitution(
        roots,
        (_replace_active(old, new),),
        consumer_gc_steps=(ConsumerGCStep((1,)),),
    )

    assert decision.accepted
    assert decision.baseline_released_paths == ()
    assert decision.candidate_released_paths == ()
    assert decision.remaining_consumers_after == (("1", expected_remaining),)
    assert decision.before.object_count("SparseHZono") == 1
    assert decision.before.object_count("SparseHZAffineExpr") == 1
    assert decision.before.object_count("Bounds") == 1


def test_invalid_consumer_count_and_malformed_precomputed_fail_closed():
    old = np.arange(8, dtype=np.float64)
    new = np.arange(2, dtype=np.float64)
    invalid_gc = assess_hypothetical_substitution(
        WholeStateRoots(
            active={"target": old}, remaining_consumers={1: 0}
        ),
        (_replace_active(old, new),),
        consumer_gc_steps=(ConsumerGCStep((1,)),),
    )
    assert not invalid_gc.accepted
    assert invalid_gc.reason == "invalid_sparse_consumer_accounting"

    malformed = assess_hypothetical_substitution(
        WholeStateRoots(precomputed_relu={2: (old, new)}), ()
    )
    assert not malformed.accepted
    assert malformed.reason.startswith("malformed_precomputed_tuple")


def test_mapping_memory_error_fails_closed_without_mutating_input():
    class AllocationFailingMap(dict):
        def items(self):
            raise MemoryError("synthetic allocation failure")

    active = AllocationFailingMap(target=np.arange(8, dtype=np.float64))
    decision = assess_hypothetical_substitution(
        WholeStateRoots(active=active), ()
    )

    assert not decision.accepted
    assert decision.reason == "active_snapshot_failed"
    assert tuple(dict.items(active))[0][0] == "target"


@pytest.mark.parametrize("failure", (KeyboardInterrupt, SystemExit))
def test_mapping_baseexception_propagates_without_mutating_input(failure):
    class InterruptingMap(dict):
        def items(self):
            raise failure("synthetic asynchronous interruption")

    active = InterruptingMap(target=np.arange(8, dtype=np.float64))
    with pytest.raises(failure, match="synthetic asynchronous interruption"):
        assess_hypothetical_substitution(
            WholeStateRoots(active=active), ()
        )

    assert tuple(dict.items(active))[0][0] == "target"
