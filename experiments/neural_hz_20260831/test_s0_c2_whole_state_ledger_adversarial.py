from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds
from act.back_end.hybridz_tf.tf_cnn import (
    SparseHZAffineExpr,
    SparseHZAffineTerm,
)
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831 import (
    s0_c2_whole_state_ledger_prototype as ledger_module,
)
from experiments.neural_hz_20260831.s0_c2_whole_state_ledger_prototype import (
    ConsumerGCStep,
    RootPath,
    RootSubstitution,
    StrongRetentionPlan,
    WholeStateReject,
    WholeStateRoots,
    assess_hypothetical_substitution,
    snapshot_whole_state,
)


def _hz(width: int = 4, *, frame_id: int = 17) -> SparseHZono:
    eye = sp.eye(width, format="csr", dtype=np.float64)
    no_binary = sp.csr_matrix((width, 0), dtype=np.float64)
    no_equalities_cont = sp.csr_matrix((0, width), dtype=np.float64)
    no_equalities_binary = sp.csr_matrix((0, 0), dtype=np.float64)
    return SparseHZono(
        c=np.linspace(-0.5, 0.5, width, dtype=np.float64),
        Gc=eye,
        Gb=no_binary,
        Ac=no_equalities_cont,
        Ab=no_equalities_binary,
        b=np.zeros(0, dtype=np.float64),
        Auc=no_equalities_cont.copy(),
        Aub=no_equalities_binary.copy(),
        ub=np.zeros(0, dtype=np.float64),
        frame_id=frame_id,
        exact=True,
    )


def _expr(width: int = 4, *, frame_id: int = 17) -> SparseHZAffineExpr:
    source = _hz(width, frame_id=frame_id)
    return SparseHZAffineExpr(
        terms=(SparseHZAffineTerm(source, ()),),
        bias=np.zeros(width, dtype=np.float64),
        n_out=width,
        frame_id=frame_id,
    )


def _bounds(width: int = 4) -> Bounds:
    return Bounds(
        lb=torch.linspace(-1.0, 0.0, width, dtype=torch.float64),
        ub=torch.linspace(0.0, 1.0, width, dtype=torch.float64),
    )


def _manual_csr(data: np.ndarray, width: int) -> sp.csr_matrix:
    """Build a valid one-row CSR while preserving the supplied data view."""

    if data.ndim != 1 or data.size != width:
        raise AssertionError("test fixture mismatch")
    matrix = sp.csr_matrix((1, width), dtype=data.dtype)
    matrix.data = data
    matrix.indices = np.arange(width, dtype=np.int32)
    matrix.indptr = np.array([0, width], dtype=np.int32)
    return matrix


def _graph_root(namespace: str, width: int):
    if namespace == "sparse_hz":
        return _hz(width)
    if namespace == "affine_expr":
        return _expr(width)
    if namespace == "phase_bounds":
        return _bounds(width)
    raise AssertionError(namespace)


def _roots_with_graph_value(
    namespace: str,
    value: object,
    *,
    key: int = 32,
    remaining: int = 0,
    pinned: bool = False,
    active: dict[object, object] | None = None,
) -> WholeStateRoots:
    values = {namespace: {key: value}}
    return WholeStateRoots(
        sparse_hz=values.get("sparse_hz", {}),
        affine_expr=values.get("affine_expr", {}),
        phase_bounds=values.get("phase_bounds", {}),
        active={} if active is None else active,
        remaining_consumers={key: remaining},
        pinned_layers=frozenset({key}) if pinned else frozenset(),
    )


def _all_roles(ledger) -> tuple[str, ...]:
    return tuple(
        role
        for storage in ledger.storage_provenance
        for role in storage.roles
    )


def test_csr_short_data_view_cannot_manufacture_an_entry_reduction():
    """CSR entries are data.size, while bytes retain the complete owner."""

    large_owner = np.arange(100, dtype=np.float64)
    old = _manual_csr(large_owner[10:12], 2)
    new = _manual_csr(np.arange(3, dtype=np.float64), 3)
    assert old.data.size == 2 < new.data.size == 3
    assert old.data.base is large_owner

    decision = assess_hypothetical_substitution(
        WholeStateRoots(artifact={"matrix": old}),
        (
            RootSubstitution(
                RootPath("artifact", "matrix"),
                expected=old,
                replacement=new,
            ),
        ),
    )

    # Counting large_owner.size as CSR entries would turn the real 2 -> 3
    # increase into an apparent 100 -> 3 decrease.  The ambiguous short view
    # must instead fail closed before either strict gate is evaluated.
    assert not decision.accepted
    assert "csr_data" in decision.reason


def test_later_numpy_partial_overlap_is_checked_against_every_prior_span():
    owner = np.arange(32, dtype=np.float64)
    first = owner[:2]
    second = owner[8:20]
    third = owner[16:28]
    assert not np.shares_memory(first, second)
    assert not np.shares_memory(first, third)
    assert np.shares_memory(second, third)

    with pytest.raises(WholeStateReject, match="partial_numpy_overlap"):
        snapshot_whole_state(
            WholeStateRoots(
                descriptor={"first": first},
                artifact={"second": second},
                active={"third": third},
            )
        )


def test_disjoint_views_across_all_transaction_namespaces_charge_owner_once():
    owner = np.arange(64, dtype=np.float64)
    descriptor_view = owner[:8]
    artifact_view = owner[16:24]
    active_view = owner[32:40]
    pending_view = owner[48:56]

    ledger = snapshot_whole_state(
        WholeStateRoots(
            descriptor={"d": descriptor_view},
            artifact={"a": artifact_view},
            active={"x": active_view},
            pending={
                "p": StrongRetentionPlan((pending_view,), "pending-c2")
            },
        )
    )

    assert ledger.resident_bytes == owner.nbytes
    assert ledger.resident_entries == owner.size
    assert ledger.numeric_storage_count == 1
    assert len(ledger.storage_provenance[0].roles) == 4


def test_two_csr_roots_sharing_full_data_owner_deduplicate_only_that_buffer():
    shared_data = np.arange(4, dtype=np.float64)
    first = _manual_csr(shared_data.view(), 4)
    second = _manual_csr(shared_data.reshape(-1), 4)

    ledger = snapshot_whole_state(
        WholeStateRoots(artifact={"first": first, "second": second})
    )

    expected_bytes = (
        shared_data.nbytes
        + first.indices.nbytes
        + first.indptr.nbytes
        + second.indices.nbytes
        + second.indptr.nbytes
    )
    assert ledger.resident_bytes == expected_bytes
    assert ledger.resident_entries == shared_data.size
    assert ledger.numeric_storage_count == 5
    data_roles = tuple(
        role
        for record in ledger.storage_provenance
        if record.resident_entries == shared_data.size
        for role in record.roles
    )
    assert len(data_roles) == 2


def test_torch_nonzero_offset_aliases_charge_complete_storage_once():
    owner = torch.arange(64, dtype=torch.float64)
    lower = owner[3:19]
    upper = owner[27:43]
    bounds = Bounds(lb=lower, ub=upper)

    ledger = snapshot_whole_state(
        WholeStateRoots(
            phase_bounds={1: bounds},
            active={"owner-tail": owner[48:]},
        )
    )

    assert ledger.resident_bytes == owner.untyped_storage().nbytes()
    assert ledger.resident_entries == owner.numel()
    assert ledger.numeric_storage_count == 1
    assert "0x" not in repr(ledger.storage_provenance)


def test_incompatible_torch_dtype_alias_fails_closed_without_pointer_text():
    owner = torch.arange(8, dtype=torch.float64)
    byte_alias = owner.view(torch.uint8)

    with pytest.raises(WholeStateReject) as caught:
        snapshot_whole_state(
            WholeStateRoots(active={"float": owner, "bytes": byte_alias})
        )

    assert "incompatible_storage_alias" in str(caught.value)
    assert str(owner.untyped_storage().data_ptr()) not in str(caught.value)
    assert hex(owner.untyped_storage().data_ptr()) not in str(caught.value)


@pytest.mark.parametrize(
    "namespace", ("sparse_hz", "affine_expr", "phase_bounds")
)
def test_second_consumer_blocks_manual_graph_root_removal(namespace: str):
    old = _graph_root(namespace, 8)
    roots = _roots_with_graph_value(namespace, old, remaining=2)

    decision = assess_hypothetical_substitution(
        roots,
        (
            RootSubstitution(
                RootPath(namespace, 32), expected=old, remove=True
            ),
        ),
        consumer_gc_steps=(ConsumerGCStep((32,)),),
    )

    assert not decision.accepted
    assert getattr(roots, namespace)[32] is old
    assert roots.remaining_consumers[32] == 2


@pytest.mark.parametrize(
    "namespace", ("sparse_hz", "affine_expr", "phase_bounds")
)
def test_pinned_graph_root_cannot_be_removed_by_a_substitution(namespace: str):
    old = _graph_root(namespace, 8)
    roots = _roots_with_graph_value(
        namespace, old, remaining=0, pinned=True
    )

    decision = assess_hypothetical_substitution(
        roots,
        (
            RootSubstitution(
                RootPath(namespace, 32), expected=old, remove=True
            ),
        ),
    )

    assert not decision.accepted
    assert getattr(roots, namespace)[32] is old


@pytest.mark.parametrize(
    "namespace", ("sparse_hz", "affine_expr", "phase_bounds")
)
def test_graph_managed_root_replacement_needs_a_separate_certified_api(
    namespace: str,
):
    old = _graph_root(namespace, 8)
    smaller = _graph_root(namespace, 2)
    roots = _roots_with_graph_value(namespace, old, remaining=0)

    decision = assess_hypothetical_substitution(
        roots,
        (
            RootSubstitution(
                RootPath(namespace, 32),
                expected=old,
                replacement=smaller,
            ),
        ),
    )

    assert not decision.accepted
    assert getattr(roots, namespace)[32] is old


def test_active_replacement_remains_legal_while_add_has_second_consumer():
    shared_add_root = _hz(8)
    old_active = np.arange(64, dtype=np.float64)
    new_active = np.arange(4, dtype=np.float64)
    roots = _roots_with_graph_value(
        "sparse_hz",
        shared_add_root,
        remaining=2,
        active={"target": old_active},
    )

    decision = assess_hypothetical_substitution(
        roots,
        (
            RootSubstitution(
                RootPath("active", "target"),
                expected=old_active,
                replacement=new_active,
            ),
        ),
        consumer_gc_steps=(ConsumerGCStep((32,)),),
    )

    assert decision.accepted
    assert decision.remaining_consumers_after == (("32", 1),)
    assert decision.baseline_released_paths == ()
    assert decision.candidate_released_paths == ()
    assert roots.active["target"] is old_active
    assert roots.sparse_hz[32] is shared_add_root


def test_cross_namespace_pending_alias_prevents_fake_release():
    old = np.arange(64, dtype=np.float64)
    new = np.arange(4, dtype=np.float64)
    plan = StrongRetentionPlan((old,), "still-pending")
    roots = WholeStateRoots(
        descriptor={"descriptor": old},
        active={"target": old},
        pending={"plan": plan},
    )

    decision = assess_hypothetical_substitution(
        roots,
        (
            RootSubstitution(
                RootPath("active", "target"), old, replacement=new
            ),
            RootSubstitution(
                RootPath("descriptor", "descriptor"), old, remove=True
            ),
        ),
    )

    assert not decision.accepted
    assert decision.reason == "resident_bytes_not_strictly_reduced"
    assert decision.before.resident_bytes == old.nbytes
    assert decision.after.resident_bytes == old.nbytes + new.nbytes
    assert roots.pending["plan"] is plan
    assert roots.active["target"] is old


def test_precomputed_torch_slot_alias_blocks_apparent_slot_reduction():
    source = _hz(16)
    owner = torch.arange(32, dtype=torch.float64)
    lower = owner[:16]
    upper = owner[16:]
    replacement = torch.arange(2, dtype=torch.float64)
    cached = (source, lower, upper)
    roots = WholeStateRoots(precomputed_relu={36: cached})

    decision = assess_hypothetical_substitution(
        roots,
        (
            RootSubstitution(
                RootPath("precomputed_relu", 36, slot=2),
                expected=upper,
                replacement=replacement,
            ),
        ),
    )

    assert not decision.accepted
    assert decision.reason == "resident_bytes_not_strictly_reduced"
    assert roots.precomputed_relu[36] is cached
    assert roots.precomputed_relu[36][2] is upper


def test_gc_removal_does_not_hide_precomputed_alias_strong_root():
    source = _hz(8)
    lower = torch.zeros(8, dtype=torch.float64)
    upper = torch.ones(8, dtype=torch.float64)
    old_active = np.arange(64, dtype=np.float64)
    new_active = np.arange(4, dtype=np.float64)
    roots = WholeStateRoots(
        sparse_hz={32: source},
        precomputed_relu={36: (source, lower, upper)},
        active={"target": old_active},
        remaining_consumers={32: 1},
    )

    decision = assess_hypothetical_substitution(
        roots,
        (
            RootSubstitution(
                RootPath("active", "target"),
                expected=old_active,
                replacement=new_active,
            ),
        ),
        consumer_gc_steps=(ConsumerGCStep((32,)),),
    )

    assert decision.accepted
    assert decision.baseline_released_paths == ("sparse_hz[32]",)
    assert decision.candidate_released_paths == ("sparse_hz[32]",)
    assert decision.before.object_count("SparseHZono") == 1
    assert decision.after.object_count("SparseHZono") == 1
    assert decision.before.resident_bytes - decision.after.resident_bytes == (
        old_active.nbytes - new_active.nbytes
    )
    assert roots.sparse_hz[32] is source


def test_hz_alias_provenance_lists_sparse_and_precomputed_root_paths():
    source = _hz(8)
    lower = torch.zeros(8, dtype=torch.float64)
    upper = torch.ones(8, dtype=torch.float64)

    ledger = snapshot_whole_state(
        WholeStateRoots(
            sparse_hz={32: source},
            precomputed_relu={36: (source, lower, upper)},
        )
    )

    center_roles = next(
        record.roles
        for record in ledger.storage_provenance
        if any(role == "sparse_hz[32].value.c" for role in record.roles)
    )
    assert "sparse_hz[32].value.c" in center_roles
    assert "precomputed_relu[36].slot0.value.c" in center_roles
    generator_roles = next(
        record.roles
        for record in ledger.storage_provenance
        if any(role == "sparse_hz[32].value.Gc.data" for role in record.roles)
    )
    assert "precomputed_relu[36].slot0.value.Gc.data" in generator_roles
    assert ledger.object_count("SparseHZono") == 1


def test_bounds_alias_provenance_lists_slot4_and_phase_root_paths():
    source = _hz(8)
    expression = SparseHZAffineExpr(
        terms=(SparseHZAffineTerm(source, ()),),
        bias=np.zeros(8, dtype=np.float64),
        n_out=8,
        frame_id=17,
    )
    lower = torch.zeros(8, dtype=torch.float64)
    upper = torch.ones(8, dtype=torch.float64)
    phase = _bounds(8)

    ledger = snapshot_whole_state(
        WholeStateRoots(
            precomputed_relu={
                36: (source, lower, upper, expression, phase)
            },
            phase_bounds={36: phase},
        )
    )

    lower_roles = next(
        record.roles
        for record in ledger.storage_provenance
        if any(
            role == "precomputed_relu[36].slot4.lb"
            for role in record.roles
        )
    )
    assert "precomputed_relu[36].slot4.lb" in lower_roles
    assert "phase_bounds[36].lb" in lower_roles
    upper_roles = next(
        record.roles
        for record in ledger.storage_provenance
        if any(
            role == "precomputed_relu[36].slot4.ub"
            for role in record.roles
        )
    )
    assert "phase_bounds[36].ub" in upper_roles
    assert ledger.object_count("Bounds") == 1


def test_expr_alias_provenance_lists_every_strong_root_path_once():
    expression = _expr(8)
    source = expression.terms[0].source
    lower = torch.zeros(8, dtype=torch.float64)
    upper = torch.ones(8, dtype=torch.float64)

    ledger = snapshot_whole_state(
        WholeStateRoots(
            affine_expr={33: expression},
            precomputed_relu={
                36: (source, lower, upper, expression, None)
            },
            active={"expr": expression},
        )
    )

    bias_roles = next(
        record.roles
        for record in ledger.storage_provenance
        if "affine_expr[33].bias" in record.roles
    )
    assert bias_roles == (
        "active['expr'].bias",
        "affine_expr[33].bias",
        "precomputed_relu[36].slot3.bias",
    )
    assert ledger.object_count("SparseHZAffineExpr") == 1
    assert ledger.object_count("SparseHZAffineTerm") == 1


def test_self_referential_retention_plan_fails_closed_instead_of_disappearing():
    plan = StrongRetentionPlan((), "self-cycle")
    object.__setattr__(plan, "roots", (plan,))

    with pytest.raises(WholeStateReject):
        snapshot_whole_state(WholeStateRoots(pending={"plan": plan}))


def test_two_plan_retention_cycle_fails_closed_instead_of_disappearing():
    first = StrongRetentionPlan((), "first")
    second = StrongRetentionPlan((first,), "second")
    object.__setattr__(first, "roots", (second,))

    with pytest.raises(WholeStateReject):
        snapshot_whole_state(
            WholeStateRoots(active={"first": first}, pending={"second": second})
        )


def test_descriptor_and_artifact_roots_survive_same_consumer_gc_boundary():
    source = _hz(8)
    descriptor_payload = np.arange(12, dtype=np.float64)
    artifact_payload = sp.eye(8, format="csr", dtype=np.float64)
    old_active = np.arange(64, dtype=np.float64)
    new_active = np.arange(4, dtype=np.float64)
    roots = WholeStateRoots(
        sparse_hz={32: source},
        descriptor={32: descriptor_payload},
        artifact={32: artifact_payload},
        active={"target": old_active},
        remaining_consumers={32: 1},
    )

    decision = assess_hypothetical_substitution(
        roots,
        (
            RootSubstitution(
                RootPath("active", "target"), old_active, new_active
            ),
        ),
        consumer_gc_steps=(ConsumerGCStep((32,)),),
    )

    assert decision.accepted
    before_roles = _all_roles(decision.before)
    after_roles = _all_roles(decision.after)
    assert any("descriptor[32]" in role for role in before_roles)
    assert any("descriptor[32]" in role for role in after_roles)
    assert any("artifact[32]" in role for role in before_roles)
    assert any("artifact[32]" in role for role in after_roles)
    assert 32 in roots.descriptor and 32 in roots.artifact


@pytest.mark.parametrize(
    "namespace", ("descriptor", "artifact", "active", "pending")
)
def test_unknown_numeric_holder_in_transaction_namespace_fails_closed(
    namespace: str,
):
    holder = SimpleNamespace(payload=np.arange(1_000, dtype=np.float64))
    kwargs = {namespace: {"holder": holder}}

    decision = assess_hypothetical_substitution(
        WholeStateRoots(**kwargs), ()
    )

    assert not decision.accepted
    assert decision.reason.startswith("unsupported_strong_root")


def test_ordinary_exception_after_private_staging_fails_closed_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
):
    old = np.arange(64, dtype=np.float64)
    new = np.arange(4, dtype=np.float64)
    active = {"target": old}
    calls = 0
    original_gc = ledger_module._simulate_consumer_gc

    def fail_on_candidate(snapshot, steps):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic post-staging failure")
        return original_gc(snapshot, steps)

    monkeypatch.setattr(
        ledger_module, "_simulate_consumer_gc", fail_on_candidate
    )
    decision = assess_hypothetical_substitution(
        WholeStateRoots(active=active),
        (
            RootSubstitution(
                RootPath("active", "target"), old, replacement=new
            ),
        ),
    )

    assert not decision.accepted
    assert decision.reason == "unexpected_ledger_error:RuntimeError"
    assert active == {"target": old}
    assert active["target"] is old


@pytest.mark.parametrize("failure", (KeyboardInterrupt, SystemExit))
def test_baseexception_after_private_staging_propagates_without_mutation(
    monkeypatch: pytest.MonkeyPatch, failure
):
    old = np.arange(64, dtype=np.float64)
    new = np.arange(4, dtype=np.float64)
    active = {"target": old}
    calls = 0
    original_gc = ledger_module._simulate_consumer_gc

    def interrupt_on_candidate(snapshot, steps):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure("synthetic post-staging interruption")
        return original_gc(snapshot, steps)

    monkeypatch.setattr(
        ledger_module, "_simulate_consumer_gc", interrupt_on_candidate
    )
    with pytest.raises(failure, match="post-staging interruption"):
        assess_hypothetical_substitution(
            WholeStateRoots(active=active),
            (
                RootSubstitution(
                    RootPath("active", "target"), old, replacement=new
                ),
            ),
        )

    assert active == {"target": old}
    assert active["target"] is old


def test_public_ledger_and_decision_do_not_retain_or_print_storage_addresses():
    numpy_owner = np.arange(32, dtype=np.float64)
    torch_owner = torch.arange(32, dtype=torch.float64)
    old = numpy_owner[:16]
    new = np.arange(2, dtype=np.float64)
    roots = WholeStateRoots(
        active={"target": old, "torch": torch_owner[4:20]}
    )

    ledger = snapshot_whole_state(roots)
    decision = assess_hypothetical_substitution(
        roots,
        (
            RootSubstitution(
                RootPath("active", "target"), old, replacement=new
            ),
        ),
    )
    rendered = repr((ledger, decision))

    forbidden = {
        hex(id(numpy_owner)),
        str(id(numpy_owner)),
        hex(torch_owner.untyped_storage().data_ptr()),
        str(torch_owner.untyped_storage().data_ptr()),
    }
    assert all(address not in rendered for address in forbidden)
    assert "numpy-owner" not in rendered
    assert "data_ptr" not in rendered
