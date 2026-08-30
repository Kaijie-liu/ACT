#!/usr/bin/env python3
"""One-use, receipt-only relaxed-LP owner for one fixed internal PCOH toy.

This module intentionally does *not* accept a caller-owned ``OperatorHZBuild``,
an ndarray/CSR frame, or a source-producing callback.  The sole public run
entry constructs a fixed K4 corner toy inside the transaction, keeps every
source buffer private, and returns only an immutable receipt.  That narrow
contract is sufficient to close ordinary owner-alias ABA for the toy; it is
not a generic real-model owner and is never verifier or verdict authority.

The lifetime is deliberately sequential:

* build and freeze a private source;
* close one parent split-CG model and independently certify a private feasible
  parent point;
* issue and consume a detached PCOH fresh build;
* drop the source, collect/trim, and enforce an RSS gate;
* close one fresh split-CG model, independently replay its dual upper, and
  compare the two binary64 bounds with exact ``Fraction`` arithmetic;
* seal and release the fresh build before creating the public result.

The materialized PCOH toy row contains sub-HiGHS-threshold coefficients.  The
fresh numeric LP therefore uses a positive ``2**24`` row scaling for every
new PCOH upper row.  Every changed coefficient and RHS is replayed as an
exact dyadic ``Fraction`` equality and the transform is receipt-bound.  This
is a toy numeric normalization, not a production large-frame strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
import ctypes
import gc
import hashlib
import json
import math
import os
import secrets
import threading
import time
from types import MappingProxyType, SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import weakref

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds, ConSet, Fact
from act.back_end.hybridz_tf.adaptive_phase_forest import (
    RivalSpec,
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
    derive_operator_exact_relu_property_phase_literals,
)
from act.back_end.hybridz_tf.operator_hz import (
    OperatorHZBuild,
    build_operator_hz,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_objective_bounds import (
    _build_complete_operator_phase_conditioned_objective_bounds_until,
)
from act.back_end.hybridz_tf import (
    operator_phase_conditioned_objective_hull_fresh_materializer as _fresh_module,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_objective_hull_fresh_materializer import (
    PCOHFreshMaterializationCaps,
    consume_live_phase_conditioned_objective_hull_fresh_build,
    discard_live_phase_conditioned_objective_hull_fresh_build,
    issue_live_phase_conditioned_objective_hull_fresh_build,
    verify_live_phase_conditioned_objective_hull_fresh_materialized_tightness,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_objective_hull_row_materializer import (
    PCOHRowMaterializationCaps,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_pair_infeasibility import (
    run_phase_conditioned_pair_infeasibility_candidate,
)
from act.back_end.hybridz_tf.preformed_split_primal_certificate import (
    PreformedSplitPrimalCertificateCaps,
    certify_preformed_split_primal_lower,
)
from act.back_end.hybridz_tf.split_constraint_generation_candidate import (
    SplitConstraintGenerationCandidate,
    propose_split_constraint_generation_candidate,
)
from act.back_end.hybridz_tf.strict_split_lp_improvement_certificate import (
    PreformedSplitLPProblem,
    SplitRelaxedLPFrame,
    _frame_sha256,
    _fresh_dual_anchor,
    _parent_primal_anchor,
    _prepare_problem,
    _recheck_authority_input_identity_and_readonly,
    _recheck_candidate_identity,
    _validate_candidate_receipt,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    _hz_form_exact_factor_objective_envelope_from_live_split_blocks,
    _hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope,
    _hz_read_exact_objective_binding_material_from_factor_envelope,
)


_SCHEMA = "act.hybridz_pcoh_relaxed_lp_live_owner.toy.v1"
_RECEIPT_SCHEMA = "act.hybridz_pcoh_relaxed_lp_live_owner_receipt.toy.v1"
_OBJECTIVE_ID = "toy:pcoh:private-live-owner:k4-corner:rival-10"
_FOCUSED_RIVAL_ID = 10
_ROW_SCALE_EXPONENT = 24
_ROW_SCALE = 1 << _ROW_SCALE_EXPONENT
_DTYPE = torch.float64
_GIB = 1024 * 1024 * 1024
_EXPECTED_PARENT_LOWER = float.fromhex("0x1.00000000001f4p-2")
_EXPECTED_FRESH_UPPER = float.fromhex("-0x1.ffffffffffc61p-3")
_RECEIPT_KEYS = frozenset(
    {
        "caller_can_retain_source_alias_through_public_api",
        "entry_rss_bytes",
        "exact_gap",
        "fresh_candidate_receipt_sha256",
        "fresh_cg_native_model_closed_before_dual_replay",
        "fresh_dimensions",
        "fresh_dual_anchor_sha256",
        "fresh_issuance_sha256",
        "fresh_materializer_receipt_sha256",
        "fresh_payload_bytes",
        "fresh_registry_cleanup_route",
        "fresh_row_scaling",
        "fresh_scaled_frame_sha256",
        "fresh_semantic_digest",
        "fresh_solver_dual_used_only_after_independent_replay",
        "fresh_terminal_semantic_digest",
        "fresh_upper",
        "fresh_upper_hex",
        "fresh_weakrefs_released_before_receipt",
        "hard_wall_deadline_guaranteed",
        "internal_toy_live_binding",
        "materialized_tightness_summary_sha256",
        "maximum_simultaneous_highs_models_by_construction",
        "objective_semantics_exactly_equal_parent_fresh",
        "objective_semantics_sha256",
        "owner_registry_contains_hz",
        "owner_registry_empty_before_return",
        "owner_registry_one_use",
        "owner_registry_pop_required_before_return",
        "parent_anchor_source",
        "parent_candidate_receipt_sha256",
        "parent_cg_native_model_closed_before_primal_replay",
        "parent_frame_sha256",
        "parent_lower",
        "parent_lower_hex",
        "parent_primal_anchor_sha256",
        "parent_solver_primal_used_as_authority",
        "post_source_release_rss_bytes",
        "post_source_release_rss_cap_bytes",
        "post_source_release_rss_cap_caller_tightenable_only",
        "post_source_release_rss_gate_passed",
        "post_source_release_rss_hard_ceiling_bytes",
        "production_ready",
        "proof_authority",
        "public_accepts_build_or_numeric_frame",
        "public_accepts_source_callback",
        "pure_checker_authenticates_opaque_binding_hashes",
        "pure_checker_authenticates_producer_provenance",
        "pure_checker_scope",
        "real_owner_blocker",
        "real_owner_minimum_upstream_api",
        "real_parent_binding_authority",
        "real_sound_extension_authority",
        "receipt_sha256",
        "receipt_sha256_keyed_authenticator",
        "row_scaling_exact_fraction_replay_required",
        "rss_with_source_and_fresh_bytes",
        "schema",
        "shared_absolute_deadline",
        "source_and_fresh_overlap_scope",
        "source_dimensions",
        "source_fresh_detachment_pairs_checked",
        "source_owner_scope",
        "source_payload_bytes",
        "source_release_gc_called",
        "source_release_malloc_trim_attempted",
        "source_release_malloc_trim_result",
        "source_semantic_digest",
        "source_terminal_semantic_digest",
        "source_weakrefs_released_before_fresh_lp",
        "stable_bit_ids",
        "status",
        "strict_comparison",
        "supplied_numeric_strict_ordering_theorem",
        "terminal_gc_called",
        "terminal_malloc_trim_attempted",
        "terminal_malloc_trim_result",
        "terminal_rss_bytes",
        "timings",
        "toy_numeric_frame_authority",
        "toy_only",
        "toy_sound_extension_claimed",
        "uses_sparse_hstack",
        "uses_sparse_vstack",
        "verdict_authority",
    }
)
_SCALING_RECEIPT_KEYS = frozenset(
    {
        "all_changed_coefficients_and_rhs_exactly_scaled",
        "block_hashes_authenticate_live_provenance",
        "exact_fraction_items_replayed",
        "fresh_semantic_digest",
        "fresh_upper_row_count",
        "original_Aub_sha256",
        "original_Auc_sha256",
        "original_ub_sha256",
        "production_ready",
        "proof_authority",
        "row_feasible_set_equivalence",
        "scale_exact_integer",
        "scale_exponent",
        "scale_positive",
        "scaled_Aub_sha256",
        "scaled_Auc_sha256",
        "scaled_numeric_frame_sha256",
        "scaled_ub_sha256",
        "scaled_upper_rows",
        "scaling_receipt_sha256",
        "scaling_receipt_sha256_keyed_authenticator",
        "schema",
        "source_upper_row_count",
        "unscaled_prefix_byte_equal",
        "verdict_authority",
    }
)
_FRESH_DISCARD_CLEANUP_AUTHORITY = (
    discard_live_phase_conditioned_objective_hull_fresh_build
)
_CORE_DENSE = ("c", "b", "ub", "col_ids", "bcol_ids")
_CORE_CSR = ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub")
_PROVENANCE = (
    "full_col_ids",
    "operator_input_center",
    "operator_input_radius",
    "_solver_continuous_column_layer_ids",
)


class PCOHRelaxedLPLiveOwnerError(ValueError):
    """The private toy transaction failed closed without returning an HZ."""


def _toy_fresh_caps() -> PCOHFreshMaterializationCaps:
    return PCOHFreshMaterializationCaps(
        max_parent_variables=64,
        max_parent_rows=128,
        max_parent_nonzeros=4096,
        max_parent_buffer_items=16384,
        max_tag_bytes=65536,
        max_registry_entries=1,
        capability_ttl_seconds=30.0,
        row_caps=PCOHRowMaterializationCaps(
            max_parent_continuous_columns=64,
            max_parent_binary_columns=8,
            max_eta_columns=16,
            max_rows=64,
            max_total_exact_nonzeros=4096,
            max_exact_bits=16384,
        ),
    )


def _toy_primal_caps() -> PreformedSplitPrimalCertificateCaps:
    return PreformedSplitPrimalCertificateCaps(
        max_columns=64,
        max_rows=128,
        max_constraint_nnz=4096,
        max_exact_objective_terms=64,
        max_exact_equality_rows=64,
        max_exact_equality_nnz=4096,
        max_exact_upper_rows=128,
        max_exact_upper_nnz=4096,
        chunk_elements=256,
    )


@dataclass(frozen=True)
class PCOHRelaxedLPToyOwnerCaps:
    """Pure-scalar/nested-frozen caps for the fixed internal toy only.

    ``max_post_source_release_rss_bytes`` may tighten but never relax the
    fixed 2.5 GiB process-RSS stop-loss.
    """

    max_post_source_release_rss_bytes: int = 5 * _GIB // 2
    max_selected_upper_rows: int = 64
    max_equality_rows: int = 64
    max_binary_change_coefficients: int = 4096
    scan_chunk_rows: int = 64
    fresh_caps: PCOHFreshMaterializationCaps = field(
        default_factory=_toy_fresh_caps
    )
    parent_primal_caps: PreformedSplitPrimalCertificateCaps = field(
        default_factory=_toy_primal_caps
    )


@dataclass(frozen=True, eq=False)
class PCOHRelaxedLPToyOwnerResult:
    """Pure receipt result; no build, HZ, solver, envelope, or capability."""

    schema: str
    status: str
    parent_lower: float
    fresh_upper: float
    exact_gap: Fraction
    source_semantic_digest: str
    fresh_semantic_digest: str
    receipt: Mapping[str, Any]
    receipt_sha256: str
    production_ready: bool = False
    proof_authority: bool = False
    verdict_authority: bool = False
    real_parent_binding_authority: bool = False

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA:
            raise ValueError("toy owner result schema mismatch")
        if self.status != "strict_private_toy_numeric_ordering_certified":
            raise ValueError("toy owner result status mismatch")
        if not self.exact_gap > 0:
            raise ValueError("toy owner exact gap must be positive")
        if any(
            value is not False
            for value in (
                self.production_ready,
                self.proof_authority,
                self.verdict_authority,
                self.real_parent_binding_authority,
            )
        ):
            raise ValueError("toy owner cannot acquire production authority")
        # Always detach and recursively freeze, even when the caller presents
        # an exact MappingProxyType: its hidden backing dict may still be
        # caller-owned and mutable after a successful verification.
        if not isinstance(self.receipt, Mapping):
            raise ValueError("toy owner receipt must be a mapping")
        object.__setattr__(self, "receipt", _deep_freeze(self.receipt))


@dataclass(frozen=True)
class _PrivateProblem:
    problem: PreformedSplitLPProblem
    envelope: Any
    lower: np.ndarray
    upper: np.ndarray
    q: np.ndarray
    objective_semantics: Tuple[Any, ...]


@dataclass(frozen=True)
class _SourcePhaseOutput:
    fresh_build: Optional[OperatorHZBuild]
    source_weakrefs: Tuple["weakref.ReferenceType[Any]", ...]
    parent_lower: float
    parent_anchor_sha256: str
    parent_candidate_receipt_sha256: str
    parent_frame_sha256: str
    parent_objective_semantics: Tuple[Any, ...]
    source_semantic_digest: str
    source_terminal_semantic_digest: str
    fresh_semantic_digest: str
    fresh_issuance_sha256: str
    fresh_receipt_sha256: str
    materialized_tightness_summary_sha256: str
    stable_bit_ids: Tuple[int, ...]
    source_dimensions: Tuple[int, ...]
    fresh_dimensions: Tuple[int, ...]
    source_payload_bytes: int
    fresh_payload_bytes: int
    detachment_pairs_checked: int
    timings: Mapping[str, float]


@dataclass(frozen=True)
class _FreshPhaseOutput:
    fresh_weakrefs: Tuple["weakref.ReferenceType[Any]", ...]
    fresh_upper: float
    fresh_anchor_sha256: str
    fresh_candidate_receipt_sha256: str
    fresh_frame_sha256: str
    fresh_semantic_digest: str
    fresh_terminal_semantic_digest: str
    row_scaling_receipt: Mapping[str, Any]
    objective_semantics_sha256: str
    timings: Mapping[str, float]


@dataclass(frozen=True)
class _ResultDraft:
    parent_lower: float
    fresh_upper: float
    exact_gap: Fraction
    source_semantic_digest: str
    fresh_semantic_digest: str
    payload: Mapping[str, Any]


_ACTIVE_LOCK = threading.Lock()
_ACTIVE_TRANSACTIONS: Dict[str, Tuple[int, str]] = {}


def _reserve_transaction() -> str:
    token = secrets.token_hex(32)
    with _ACTIVE_LOCK:
        if _ACTIVE_TRANSACTIONS:
            raise PCOHRelaxedLPLiveOwnerError(
                "one_private_toy_transaction_already_active"
            )
        _ACTIVE_TRANSACTIONS[token] = (os.getpid(), "reserved")
    return token


def _mark_transaction(token: str, phase: str) -> None:
    with _ACTIVE_LOCK:
        record = _ACTIVE_TRANSACTIONS.get(token)
        if record is None or record[0] != os.getpid():
            raise PCOHRelaxedLPLiveOwnerError("private_owner_registry_lost")
        _ACTIVE_TRANSACTIONS[token] = (record[0], phase)


def _release_transaction(token: Optional[str]) -> bool:
    if token is None:
        return True
    with _ACTIVE_LOCK:
        record = _ACTIVE_TRANSACTIONS.get(token)
        if record is None or record[0] != os.getpid():
            return False
        _ACTIVE_TRANSACTIONS.pop(token, None)
    return True


def _active_transaction_count() -> int:
    with _ACTIVE_LOCK:
        return len(_ACTIVE_TRANSACTIONS)


def _deadline(value: Any) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise PCOHRelaxedLPLiveOwnerError(
            "deadline_must_be_builtin_finite_absolute_monotonic_time"
        )
    result = float(value)
    if not math.isfinite(result) or time.monotonic() >= result:
        raise PCOHRelaxedLPLiveOwnerError(
            "deadline_must_be_builtin_finite_absolute_monotonic_time"
        )
    return result


def _check_deadline(deadline: float, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise PCOHRelaxedLPLiveOwnerError(
            f"deadline_exhausted:{stage}:private_state_released"
        )


def _normalize_caps(value: Any) -> PCOHRelaxedLPToyOwnerCaps:
    if type(value) is not PCOHRelaxedLPToyOwnerCaps:
        raise PCOHRelaxedLPLiveOwnerError(
            "caps_must_be_exact_frozen_toy_owner_caps"
        )
    scalar_names = (
        "max_post_source_release_rss_bytes",
        "max_selected_upper_rows",
        "max_equality_rows",
        "max_binary_change_coefficients",
        "scan_chunk_rows",
    )
    # Snapshot every caller-owned field once.  The returned cap record is a
    # new internal object, so even object.__setattr__ on the caller's frozen
    # dataclass after admission cannot relax the later RSS/resource gates.
    scalar_snapshot = {name: getattr(value, name) for name in scalar_names}
    fresh_caps = value.fresh_caps
    primal_caps = value.parent_primal_caps
    for name, item in scalar_snapshot.items():
        if type(item) is not int or item <= 0:
            raise PCOHRelaxedLPLiveOwnerError(f"caps_{name}_invalid")
    if (
        type(fresh_caps) is not PCOHFreshMaterializationCaps
        or type(primal_caps) is not PreformedSplitPrimalCertificateCaps
    ):
        raise PCOHRelaxedLPLiveOwnerError("nested_toy_caps_wrong_type")
    expected_fresh = _toy_fresh_caps()
    row_caps = fresh_caps.row_caps
    if type(row_caps) is not PCOHRowMaterializationCaps:
        raise PCOHRelaxedLPLiveOwnerError("nested_row_caps_wrong_type")
    for name in (
        "max_parent_variables",
        "max_parent_rows",
        "max_parent_nonzeros",
        "max_parent_buffer_items",
        "max_tag_bytes",
        "max_registry_entries",
    ):
        item = getattr(fresh_caps, name)
        if type(item) is not int or item != getattr(expected_fresh, name):
            raise PCOHRelaxedLPLiveOwnerError(
                f"nested_fresh_caps_{name}_must_match_fixed_toy_profile"
            )
    if (
        type(fresh_caps.capability_ttl_seconds) is not float
        or fresh_caps.capability_ttl_seconds
        != expected_fresh.capability_ttl_seconds
    ):
        raise PCOHRelaxedLPLiveOwnerError(
            "nested_fresh_caps_ttl_must_match_fixed_toy_profile"
        )
    for name in vars(expected_fresh.row_caps):
        item = getattr(row_caps, name)
        if (
            type(item) is not int
            or item != getattr(expected_fresh.row_caps, name)
        ):
            raise PCOHRelaxedLPLiveOwnerError(
                f"nested_row_caps_{name}_must_match_fixed_toy_profile"
            )
    expected_primal = _toy_primal_caps()
    for name in vars(expected_primal):
        item = getattr(primal_caps, name)
        if type(item) is not int or item != getattr(expected_primal, name):
            raise PCOHRelaxedLPLiveOwnerError(
                f"nested_primal_caps_{name}_must_match_fixed_toy_profile"
            )
    if (
        scalar_snapshot["max_selected_upper_rows"] > 1024
        or scalar_snapshot["max_equality_rows"] > 1024
    ):
        raise PCOHRelaxedLPLiveOwnerError("toy_lp_row_cap_exceeds_hard_limit")
    if scalar_snapshot["max_post_source_release_rss_bytes"] > 5 * _GIB // 2:
        raise PCOHRelaxedLPLiveOwnerError(
            "post_source_release_rss_cap_cannot_relax_2_5_gib_stoploss"
        )
    if scalar_snapshot["max_binary_change_coefficients"] > 65536:
        raise PCOHRelaxedLPLiveOwnerError(
            "toy_binary_change_cap_exceeds_hard_limit"
        )
    return PCOHRelaxedLPToyOwnerCaps(
        **scalar_snapshot,
        fresh_caps=expected_fresh,
        parent_primal_caps=expected_primal,
    )


def _canonical_form(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise PCOHRelaxedLPLiveOwnerError("canonical_nonfinite_float")
        return {"__binary64_hex__": value.hex()}
    if type(value) is Fraction:
        return {"__fraction__": [value.numerator, value.denominator]}
    if isinstance(value, np.generic):
        return _canonical_form(value.item())
    if type(value) in {tuple, list}:
        return [_canonical_form(item) for item in value]
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise PCOHRelaxedLPLiveOwnerError(
                    "canonical_mapping_key_not_string"
                )
            result[key] = _canonical_form(item)
        return result
    raise PCOHRelaxedLPLiveOwnerError(
        f"canonical_unsupported:{type(value).__name__}"
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _canonical_form(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if type(value) in {tuple, list}:
        return tuple(_deep_freeze(item) for item in value)
    return value


def _array_sha256(array: np.ndarray, *, name: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"act.hybridz.pcoh_live_owner.ndarray.toy.v1\0")
    digest.update(name.encode("ascii") + b"\0")
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def _csr_sha256(matrix: sp.csr_matrix, *, name: str) -> str:
    return _canonical_sha256(
        {
            "name": name,
            "shape": tuple(int(item) for item in matrix.shape),
            "nnz": int(matrix.nnz),
            "indptr_sha256": _array_sha256(matrix.indptr, name=name + ":indptr"),
            "indices_sha256": _array_sha256(matrix.indices, name=name + ":indices"),
            "data_sha256": _array_sha256(matrix.data, name=name + ":data"),
        }
    )


def _layer(
    layer_id: int,
    kind: str,
    params: Optional[Mapping[str, Any]] = None,
    *,
    width: int,
) -> Any:
    return SimpleNamespace(
        id=int(layer_id),
        kind=str(kind),
        params=dict(params or {}),
        in_vars=[],
        out_vars=[(int(layer_id), row) for row in range(int(width))],
    )


def _dense(
    layer_id: int,
    weight: Sequence[Sequence[float]],
    bias: Sequence[float],
) -> Any:
    weight_array = np.asarray(weight, dtype=np.float64)
    bias_array = np.asarray(bias, dtype=np.float64)
    return _layer(
        layer_id,
        "DENSE",
        {
            "weight": torch.tensor(weight_array, dtype=_DTYPE),
            "bias": torch.tensor(bias_array, dtype=_DTYPE),
            "in_features": int(weight_array.shape[1]),
            "out_features": int(weight_array.shape[0]),
        },
        width=int(weight_array.shape[0]),
    )


def _build_private_corner_toy_source() -> OperatorHZBuild:
    """Construct the source internally; there is no caller producer hook."""

    lower = torch.tensor([[-1.0, -1.0]], dtype=_DTYPE)
    upper = torch.tensor([[1.0, 1.0]], dtype=_DTYPE)
    layers = [
        _layer(0, "INPUT", {"shape": (1, 2)}, width=2),
        _layer(
            1,
            "INPUT_SPEC",
            {"kind": "BOX", "lb": lower, "ub": upper},
            width=2,
        ),
        _dense(
            2,
            (
                (1.0, 1.0),
                (1.0, -1.0),
                (-1.0, 1.0),
                (-1.0, -1.0),
            ),
            (-1.5, -1.5, -1.5, -1.5),
        ),
        _layer(3, "RELU", width=4),
        _dense(
            4,
            (
                (0.0, 0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0, 1.0),
                (0.5, 0.5, 0.5, 0.5),
            ),
            (0.75, 0.0, 0.0),
        ),
        _layer(5, "ASSERT", width=3),
    ]
    predecessors = {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]}
    successors = {layer.id: [] for layer in layers}
    for child, parents in predecessors.items():
        for parent in parents:
            successors[parent].append(child)
    network = SimpleNamespace(
        layers=layers,
        preds=predecessors,
        succs=successors,
        by_id={layer.id: layer for layer in layers},
    )
    facts: Dict[int, Fact] = {}
    for layer in layers:
        width = len(layer.out_vars)
        if layer.kind in {"INPUT", "INPUT_SPEC"}:
            fact_lower, fact_upper = lower.clone(), upper.clone()
        else:
            fact_lower = torch.full((1, width), -1.0e30, dtype=_DTYPE)
            fact_upper = torch.full((1, width), 1.0e30, dtype=_DTYPE)
        facts[layer.id] = Fact(Bounds(fact_lower, fact_upper), ConSet())
    build = build_operator_hz(
        network,
        facts,
        facts,
        exact_budget=4,
        materialize_add=True,
    )
    if type(build) is not OperatorHZBuild or type(build.hz) is not SparseHZono:
        raise PCOHRelaxedLPLiveOwnerError("internal_toy_builder_wrong_type")
    return build


def _rivals() -> Tuple[RivalSpec, RivalSpec]:
    return (
        RivalSpec(
            rival_id=10,
            objective=(-1.0, 1.0, 0.0),
            threshold=0.0,
            assert_digest="a" * 64,
        ),
        RivalSpec(
            rival_id=20,
            objective=(-1.0, 0.0, 1.0),
            threshold=0.0,
            assert_digest="b" * 64,
        ),
    )


def _owned_arrays(build: OperatorHZBuild) -> Tuple[np.ndarray, ...]:
    hz = build.hz
    raw = [build.input_col_ids]
    raw.extend(getattr(hz, name) for name in _CORE_DENSE)
    raw.extend(getattr(hz, name) for name in _PROVENANCE)
    for name in _CORE_CSR:
        matrix = getattr(hz, name)
        raw.extend((matrix.data, matrix.indices, matrix.indptr))
    result = []
    seen = set()
    for value in raw:
        if type(value) is not np.ndarray:
            raise PCOHRelaxedLPLiveOwnerError("private_hz_array_whitelist_failed")
        if id(value) not in seen:
            seen.add(id(value))
            result.append(value)
    return tuple(result)


def _freeze_private_build(build: OperatorHZBuild) -> None:
    for array in _owned_arrays(build):
        array.setflags(write=False)
    if any(array.flags.writeable for array in _owned_arrays(build)):
        raise PCOHRelaxedLPLiveOwnerError("private_hz_freeze_failed")


def _private_weakrefs(build: OperatorHZBuild) -> Tuple[weakref.ReferenceType[Any], ...]:
    objects: Tuple[Any, ...] = (build, build.hz, *_owned_arrays(build))
    result = []
    seen = set()
    for item in objects:
        if id(item) not in seen:
            seen.add(id(item))
            result.append(weakref.ref(item))
    return tuple(result)


def _assert_detached(source: OperatorHZBuild, fresh: OperatorHZBuild) -> int:
    checked = 0
    for left in _owned_arrays(source):
        for right in _owned_arrays(fresh):
            checked += 1
            if left is right or (
                left.size and right.size and np.shares_memory(left, right)
            ):
                raise PCOHRelaxedLPLiveOwnerError(
                    "fresh_materialization_borrowed_source_buffer"
                )
    return checked


def _shape(build: OperatorHZBuild) -> Tuple[int, int, int, int, int]:
    hz = build.hz
    return (hz.n_out, hz.n_cont, hz.n_bin, hz.n_eq, hz.n_ub)


def _payload_bytes(build: OperatorHZBuild) -> int:
    return sum(int(array.nbytes) for array in _owned_arrays(build))


def _emergency_drop_internal_fresh_token(issuance: Any) -> bool:
    """Last-resort cleanup for this owner's internally issued secret token.

    The normal and frozen public cleanup routes are attempted first.  This
    exact-token pop exists solely so a second ``BaseException`` in cleanup
    cannot leave the private fresh HZ in the upstream registry until TTL
    sweep.  No registry record or build is returned or inspected outside the
    upstream lock.
    """

    try:
        capability = issuance.capability
        token = capability.token
        if (
            type(token) is not str
            or len(token) != 64
            or capability.process_id != os.getpid()
        ):
            return False
        with _fresh_module._REGISTRY_LOCK:
            record = _fresh_module._REGISTRY.get(token)
            if record is None:
                return True
            # The issuance was created inside this transaction and never came
            # from the caller.  Still require all live owner identities before
            # the emergency pop so a corrupted module cannot target a foreign
            # registry entry.
            if (
                record.process_id != os.getpid()
                or record.capability_ref() is not capability
                or record.issuance_ref() is not issuance
                or record.issuance_sha256 != issuance.issuance_sha256
            ):
                return False
            _fresh_module._REGISTRY.pop(token, None)
        return True
    except BaseException as exc:
        _clear_exception_traceback(exc)
        return False


def _discard_unconsumed_internal_fresh(issuance: Any) -> bool:
    """Try two stable API bindings, then an exact-token locked pop."""

    for authority in (
        discard_live_phase_conditioned_objective_hull_fresh_build,
        _FRESH_DISCARD_CLEANUP_AUTHORITY,
    ):
        try:
            if authority(issuance, issuance.capability) is True:
                return True
        except BaseException as exc:
            _clear_exception_traceback(exc)
    return _emergency_drop_internal_fresh_token(issuance)


def _rss_bytes() -> int:
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as stream:
            fields = stream.read(256).split()
        if len(fields) < 2:
            raise ValueError("statm missing resident pages")
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, OverflowError) as exc:
        raise PCOHRelaxedLPLiveOwnerError("current_rss_measurement_failed") from exc


def _malloc_trim() -> Tuple[bool, Optional[bool]]:
    try:
        library = ctypes.CDLL(None)
        function = getattr(library, "malloc_trim")
        function.argtypes = [ctypes.c_size_t]
        function.restype = ctypes.c_int
        return True, bool(function(0))
    except (AttributeError, OSError, TypeError, ValueError):
        return True, None


def _fraction_pair(value: Fraction) -> Tuple[int, int]:
    return value.numerator, value.denominator


def _objective_semantics_sha256(semantics: Tuple[Any, ...]) -> str:
    center, continuous, binary = semantics
    payload = {
        "schema": "act.hybridz.pcoh_live_owner_objective_semantics.toy.v1",
        "objective_id": _OBJECTIVE_ID,
        "center": _fraction_pair(center),
        "continuous_terms": tuple(
            (int(stable_id), _fraction_pair(value))
            for stable_id, value in continuous
        ),
        "binary_terms": tuple(
            (int(stable_id), _fraction_pair(value))
            for stable_id, value in binary
        ),
    }
    return _canonical_sha256(payload)


def _make_problem(
    build: OperatorHZBuild,
    *,
    semantic_digest: str,
    Auc: Optional[sp.csr_matrix] = None,
    Aub: Optional[sp.csr_matrix] = None,
    ub: Optional[np.ndarray] = None,
    deadline: float,
) -> _PrivateProblem:
    hz = build.hz
    rival = _rivals()[0]
    objective_row = np.asarray(rival.objective, dtype=np.float64)
    objective_row.setflags(write=False)
    envelope, formation = (
        _hz_form_exact_factor_objective_envelope_from_live_split_blocks(
            c=hz.c,
            Gc=hz.Gc,
            Gb=hz.Gb,
            C_row=objective_row,
            threshold=float(rival.threshold),
            continuous_col_ids=hz.col_ids,
            binary_col_ids=hz.bcol_ids,
            objective_id=_OBJECTIVE_ID,
            parent_semantic_digest=semantic_digest,
            deadline=deadline,
        )
    )
    if envelope is None or formation.get("status") != "formed":
        raise PCOHRelaxedLPLiveOwnerError(
            "private_objective_envelope_formation_failed:"
            + str(formation.get("status"))[:160]
        )
    semantics = _hz_read_exact_objective_binding_material_from_factor_envelope(
        envelope,
        expected_parent_semantic_digest=semantic_digest,
        expected_objective_id=_OBJECTIVE_ID,
    )[:3]
    lower = -np.ones(hz.n_cont + hz.n_bin, dtype=np.float64)
    upper = np.ones(hz.n_cont + hz.n_bin, dtype=np.float64)
    q = np.concatenate(
        (envelope.q_continuous_hat, envelope.q_binary_hat)
    ).astype(np.float64, copy=False)
    for array in (lower, upper, q):
        array.setflags(write=False)
    chosen_Auc = hz.Auc if Auc is None else Auc
    chosen_Aub = hz.Aub if Aub is None else Aub
    chosen_ub = hz.ub if ub is None else ub
    problem = PreformedSplitLPProblem(
        objective_envelope=envelope,
        expected_parent_semantic_digest=semantic_digest,
        expected_exact_objective_sha256=envelope.exact_objective_sha256,
        expected_objective_binding_sha256=envelope.objective_binding_sha256,
        continuous_col_ids=hz.col_ids,
        binary_col_ids=hz.bcol_ids,
        continuous_lb=lower[: hz.n_cont],
        continuous_ub=upper[: hz.n_cont],
        binary_lb=lower[hz.n_cont :],
        binary_ub=upper[hz.n_cont :],
        frame=SplitRelaxedLPFrame(
            Auc=chosen_Auc,
            Aub=chosen_Aub,
            Ac=hz.Ac,
            Ab=hz.Ab,
            ub=chosen_ub,
            b=hz.b,
        ),
    )
    return _PrivateProblem(
        problem=problem,
        envelope=envelope,
        lower=lower,
        upper=upper,
        q=q,
        objective_semantics=semantics,
    )


def _candidate(
    private: _PrivateProblem,
    *,
    deadline: float,
    caps: PCOHRelaxedLPToyOwnerCaps,
) -> SplitConstraintGenerationCandidate:
    frame = private.problem.frame
    n_upper = int(frame.ub.size)
    n_equality = int(frame.b.size)
    if (
        n_upper > caps.max_selected_upper_rows
        or n_equality > caps.max_equality_rows
    ):
        raise PCOHRelaxedLPLiveOwnerError("private_toy_frame_exceeds_lp_caps")
    seed_duals = np.zeros(n_upper, dtype=np.float64)
    seed_duals.setflags(write=False)
    candidate = propose_split_constraint_generation_candidate(
        Auc=frame.Auc,
        Aub=frame.Aub,
        Ac=frame.Ac,
        Ab=frame.Ab,
        ub=frame.ub,
        b=frame.b,
        q=private.q,
        lower_bounds=private.lower,
        upper_bounds=private.upper,
        seed_upper_rows=tuple(range(n_upper)),
        seed_upper_duals=seed_duals,
        deadline=deadline,
        max_rounds=1,
        add_batch=1,
        max_selected_upper_rows=max(1, caps.max_selected_upper_rows),
        max_equality_rows=caps.max_equality_rows,
        max_binary_change_coefficients=caps.max_binary_change_coefficients,
        scan_chunk_rows=caps.scan_chunk_rows,
        threads=1,
    )
    receipt = candidate.receipt
    if (
        receipt.get("status") != "full_scan_candidate_feasible"
        or receipt.get("native_model_closed_before_return") is not True
        or receipt.get("full_split_scan_count") != 1
        or receipt.get("full_split_rows_scanned") != n_upper + n_equality
        or receipt.get("candidate_only") is not True
        or receipt.get("proof_authority") is not False
        or receipt.get("verdict_authority") is not False
    ):
        raise PCOHRelaxedLPLiveOwnerError(
            "split_cg_terminal_close_or_scan_contract_failed"
        )
    return candidate


def _prepare_and_validate_candidate(
    private: _PrivateProblem,
    candidate: SplitConstraintGenerationCandidate,
    *,
    label: str,
    deadline: float,
):
    prepared = _prepare_problem(
        private.problem,
        expected_objective_id=_OBJECTIVE_ID,
        label=label,
        deadline=deadline,
    )
    candidate_sha, identity = _validate_candidate_receipt(
        candidate,
        prepared,
        label=label,
        deadline=deadline,
    )
    return prepared, candidate_sha, identity


def _terminal_numeric_recheck(
    prepared: Any,
    candidate: SplitConstraintGenerationCandidate,
    candidate_sha: str,
    candidate_identity: Tuple[Any, ...],
    *,
    label: str,
    deadline: float,
) -> None:
    post = _frame_sha256(
        matrices=prepared.matrices,
        arrays=prepared.dense_frame,
        deadline=deadline,
        chunk_bytes=8 * 65536,
    )
    if post != prepared.frame_sha256:
        raise PCOHRelaxedLPLiveOwnerError(f"{label}_numeric_frame_changed")
    _recheck_authority_input_identity_and_readonly(prepared.identity_records)
    _recheck_candidate_identity(candidate_identity)
    terminal_sha, _ = _validate_candidate_receipt(
        candidate,
        prepared,
        label=label + "_terminal",
        deadline=deadline,
    )
    if terminal_sha != candidate_sha:
        raise PCOHRelaxedLPLiveOwnerError(f"{label}_candidate_changed")


def _scale_new_upper_rows_exactly(
    fresh: OperatorHZBuild,
    *,
    source_upper_rows: int,
    fresh_semantic_digest: str,
    deadline: float,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, Mapping[str, Any]]:
    hz = fresh.hz
    if (
        type(source_upper_rows) is not int
        or source_upper_rows < 0
        or source_upper_rows >= hz.n_ub
    ):
        raise PCOHRelaxedLPLiveOwnerError("fresh_pcoh_upper_row_range_invalid")
    scaled_rows = tuple(range(source_upper_rows, hz.n_ub))
    Auc = hz.Auc.copy()
    Aub = hz.Aub.copy()
    ub = hz.ub.copy()
    exact_replayed = 0
    scale_fraction = Fraction(_ROW_SCALE, 1)
    for row in scaled_rows:
        _check_deadline(deadline, "exact_positive_row_scaling")
        for source, target in ((hz.Auc, Auc), (hz.Aub, Aub)):
            source_start = int(source.indptr[row])
            source_stop = int(source.indptr[row + 1])
            target_start = int(target.indptr[row])
            target_stop = int(target.indptr[row + 1])
            if source_stop - source_start != target_stop - target_start:
                raise PCOHRelaxedLPLiveOwnerError("scaled_row_structure_changed")
            target.data[target_start:target_stop] *= float(_ROW_SCALE)
            for old, new in zip(
                source.data[source_start:source_stop],
                target.data[target_start:target_stop],
            ):
                if (
                    not math.isfinite(float(new))
                    or Fraction.from_float(float(new))
                    != Fraction.from_float(float(old)) * scale_fraction
                ):
                    raise PCOHRelaxedLPLiveOwnerError(
                        "scaled_row_coefficient_not_exact_dyadic_equivalent"
                    )
                exact_replayed += 1
        old_rhs = float(hz.ub[row])
        ub[row] *= float(_ROW_SCALE)
        if (
            not math.isfinite(float(ub[row]))
            or Fraction.from_float(float(ub[row]))
            != Fraction.from_float(old_rhs) * scale_fraction
        ):
            raise PCOHRelaxedLPLiveOwnerError(
                "scaled_row_rhs_not_exact_dyadic_equivalent"
            )
        exact_replayed += 1
    if (
        not np.array_equal(Auc.indptr, hz.Auc.indptr)
        or not np.array_equal(Auc.indices, hz.Auc.indices)
        or not np.array_equal(Aub.indptr, hz.Aub.indptr)
        or not np.array_equal(Aub.indices, hz.Aub.indices)
        or not np.array_equal(Auc.data[: int(Auc.indptr[source_upper_rows])],
                              hz.Auc.data[: int(hz.Auc.indptr[source_upper_rows])])
        or not np.array_equal(Aub.data[: int(Aub.indptr[source_upper_rows])],
                              hz.Aub.data[: int(hz.Aub.indptr[source_upper_rows])])
        or not np.array_equal(ub[:source_upper_rows], hz.ub[:source_upper_rows])
    ):
        raise PCOHRelaxedLPLiveOwnerError("unscaled_fresh_prefix_changed")
    for array in (Auc.data, Auc.indices, Auc.indptr, Aub.data, Aub.indices,
                  Aub.indptr, ub):
        array.setflags(write=False)
    payload = {
        "schema": "act.hybridz_pcoh_positive_upper_row_scaling.toy.v1",
        "fresh_semantic_digest": fresh_semantic_digest,
        "scale_exponent": _ROW_SCALE_EXPONENT,
        "scale_exact_integer": _ROW_SCALE,
        "scale_positive": True,
        "scaled_upper_rows": scaled_rows,
        "source_upper_row_count": source_upper_rows,
        "fresh_upper_row_count": hz.n_ub,
        "exact_fraction_items_replayed": exact_replayed,
        "all_changed_coefficients_and_rhs_exactly_scaled": True,
        "block_hashes_authenticate_live_provenance": False,
        "unscaled_prefix_byte_equal": True,
        "row_feasible_set_equivalence": "positive_exact_dyadic_scaling_v1",
        "original_Auc_sha256": _csr_sha256(hz.Auc, name="original_Auc"),
        "scaled_Auc_sha256": _csr_sha256(Auc, name="scaled_Auc"),
        "original_Aub_sha256": _csr_sha256(hz.Aub, name="original_Aub"),
        "scaled_Aub_sha256": _csr_sha256(Aub, name="scaled_Aub"),
        "original_ub_sha256": _array_sha256(hz.ub, name="original_ub"),
        "scaled_ub_sha256": _array_sha256(ub, name="scaled_ub"),
        "production_ready": False,
        "proof_authority": False,
        "verdict_authority": False,
        "scaling_receipt_sha256_keyed_authenticator": False,
    }
    payload["scaling_receipt_sha256"] = _canonical_sha256(payload)
    return Auc, Aub, ub, _deep_freeze(payload)


def _source_phase(
    *,
    deadline: float,
    caps: PCOHRelaxedLPToyOwnerCaps,
) -> _SourcePhaseOutput:
    started = time.monotonic()
    build = _build_private_corner_toy_source()
    _freeze_private_build(build)
    source_weakrefs = _private_weakrefs(build)
    source_digest = sparse_hz_semantic_digest(build.hz)
    source_shape = _shape(build)
    source_payload = _payload_bytes(build)
    _check_deadline(deadline, "private_source_built_and_frozen")

    parent_problem = _make_problem(
        build, semantic_digest=source_digest, deadline=deadline
    )
    parent_candidate = _candidate(parent_problem, deadline=deadline, caps=caps)
    prepared, candidate_sha, candidate_identity = (
        _prepare_and_validate_candidate(
            parent_problem,
            parent_candidate,
            label="private_parent",
            deadline=deadline,
        )
    )
    # The solver proposal is deliberately not used as primal authority: tiny
    # bound overshoots are common.  The private all-zero relaxed point is an
    # exact feasible interior anchor for this fixed K4 source and is replayed
    # by the independent primal checker over every supplied row.
    zero = np.zeros(build.hz.n_cont + build.hz.n_bin, dtype=np.float64)
    zero.setflags(write=False)
    lower, lower_receipt = certify_preformed_split_primal_lower(
        objective_envelope=parent_problem.envelope,
        expected_parent_semantic_digest=source_digest,
        expected_objective_id=_OBJECTIVE_ID,
        expected_objective_binding_sha256=(
            parent_problem.envelope.objective_binding_sha256
        ),
        continuous_col_ids=build.hz.col_ids,
        binary_col_ids=build.hz.bcol_ids,
        Auc=build.hz.Auc,
        Aub=build.hz.Aub,
        Ac=build.hz.Ac,
        Ab=build.hz.Ab,
        ub=build.hz.ub,
        b=build.hz.b,
        continuous_lb=parent_problem.problem.continuous_lb,
        continuous_ub=parent_problem.problem.continuous_ub,
        binary_lb=parent_problem.problem.binary_lb,
        binary_ub=parent_problem.problem.binary_ub,
        continuous_candidate=zero[: build.hz.n_cont],
        binary_candidate=zero[build.hz.n_cont :],
        deadline=deadline,
        caps=caps.parent_primal_caps,
    )
    lower, parent_anchor = _parent_primal_anchor(
        lower, lower_receipt, prepared
    )
    _terminal_numeric_recheck(
        prepared,
        parent_candidate,
        candidate_sha,
        candidate_identity,
        label="private_parent",
        deadline=deadline,
    )
    parent_frame_sha = prepared.frame_sha256
    parent_semantics = parent_problem.objective_semantics
    parent_seconds = time.monotonic() - started
    _check_deadline(deadline, "parent_model_closed_and_primal_replayed")

    rivals = _rivals()
    selection = derive_operator_exact_relu_property_phase_literals(build, rivals)
    stable_ids = tuple(mapping.stable_bcol_id for mapping in selection.mappings)
    if len(stable_ids) != 4 or len(set(stable_ids)) != 4:
        raise PCOHRelaxedLPLiveOwnerError("internal_k4_selection_incomplete")
    certificates = _build_complete_operator_phase_conditioned_objective_bounds_until(
        build,
        rivals,
        selection,
        focused_rival_id=_FOCUSED_RIVAL_ID,
        stable_bit_ids=stable_ids,
        deadline=deadline,
    )
    pair_bundle = run_phase_conditioned_pair_infeasibility_candidate(
        build,
        rivals,
        selection,
        stable_bit_ids=stable_ids,
        deadline=deadline,
    )
    if (
        len(certificates) != 16
        or pair_bundle.status != "complete"
        or len(pair_bundle.records) != 24
        or any(record.model_closed is not True for record in pair_bundle.records)
    ):
        raise PCOHRelaxedLPLiveOwnerError("internal_pcoh_evidence_incomplete")

    issuance = None
    consumed = False
    try:
        issuance = issue_live_phase_conditioned_objective_hull_fresh_build(
            build,
            rivals,
            selection,
            focused_rival_id=_FOCUSED_RIVAL_ID,
            stable_bit_ids=stable_ids,
            conditional_certificates=certificates,
            pair_bundle=pair_bundle,
            deadline=deadline,
            caps=caps.fresh_caps,
        )
        if not verify_live_phase_conditioned_objective_hull_fresh_materialized_tightness(
            issuance
        ):
            raise PCOHRelaxedLPLiveOwnerError(
                "fresh_materialized_tightness_replay_failed"
            )
        fresh = consume_live_phase_conditioned_objective_hull_fresh_build(
            issuance, issuance.capability, deadline=deadline
        )
        consumed = True
        if type(fresh) is not OperatorHZBuild or type(fresh.hz) is not SparseHZono:
            raise PCOHRelaxedLPLiveOwnerError("consumed_fresh_wrong_type")
        if any(array.flags.writeable for array in _owned_arrays(fresh)):
            raise PCOHRelaxedLPLiveOwnerError("consumed_fresh_not_readonly")
        detachment_checks = _assert_detached(build, fresh)
        fresh_digest = sparse_hz_semantic_digest(fresh.hz)
        source_terminal = sparse_hz_semantic_digest(build.hz)
        receipt = issuance.receipt
        if (
            source_terminal != source_digest
            or issuance.parent_semantic_digest != source_digest
            or issuance.terminal_parent_semantic_digest != source_digest
            or issuance.fresh_parent_prefix_semantic_digest != source_digest
            or issuance.fresh_semantic_digest != fresh_digest
            or receipt.get("source_buffers_borrowed_by_fresh") is not False
            or receipt.get("fresh_buffers_readonly") is not True
            or receipt.get("receipt_sha256") is None
        ):
            raise PCOHRelaxedLPLiveOwnerError(
                "source_fresh_terminal_cross_binding_failed"
            )
        result = _SourcePhaseOutput(
            fresh_build=fresh,
            source_weakrefs=source_weakrefs,
            parent_lower=lower,
            parent_anchor_sha256=parent_anchor,
            parent_candidate_receipt_sha256=candidate_sha,
            parent_frame_sha256=parent_frame_sha,
            parent_objective_semantics=parent_semantics,
            source_semantic_digest=source_digest,
            source_terminal_semantic_digest=source_terminal,
            fresh_semantic_digest=fresh_digest,
            fresh_issuance_sha256=issuance.issuance_sha256,
            fresh_receipt_sha256=receipt["receipt_sha256"],
            materialized_tightness_summary_sha256=(
                issuance.materialized_tightness_summary.summary_sha256
            ),
            stable_bit_ids=stable_ids,
            source_dimensions=tuple(receipt["source_dimensions"]),
            fresh_dimensions=tuple(receipt["fresh_dimensions"]),
            source_payload_bytes=int(receipt["source_payload_bytes"]),
            fresh_payload_bytes=int(receipt["fresh_payload_bytes"]),
            detachment_pairs_checked=detachment_checks,
            timings=MappingProxyType(
                {
                    "parent_lp_and_primal_seconds": float(parent_seconds),
                    "source_phase_total_seconds": float(time.monotonic() - started),
                }
            ),
        )
    finally:
        if issuance is not None and not consumed:
            cleanup_ok = _discard_unconsumed_internal_fresh(issuance)
            if cleanup_ok is not True:
                raise PCOHRelaxedLPLiveOwnerError(
                    "fresh_registry_cleanup_failed_after_all_routes"
                ) from None
        issuance = None
    _check_deadline(deadline, "source_phase_terminal_seals")
    return result


def _fresh_phase(
    fresh: OperatorHZBuild,
    source: _SourcePhaseOutput,
    *,
    deadline: float,
    caps: PCOHRelaxedLPToyOwnerCaps,
) -> _FreshPhaseOutput:
    started = time.monotonic()
    fresh_weakrefs = _private_weakrefs(fresh)
    if sparse_hz_semantic_digest(fresh.hz) != source.fresh_semantic_digest:
        raise PCOHRelaxedLPLiveOwnerError("fresh_changed_before_lp_phase")
    Auc, Aub, ub, scaling = _scale_new_upper_rows_exactly(
        fresh,
        source_upper_rows=int(source.source_dimensions[4]),
        fresh_semantic_digest=source.fresh_semantic_digest,
        deadline=deadline,
    )
    private = _make_problem(
        fresh,
        semantic_digest=source.fresh_semantic_digest,
        Auc=Auc,
        Aub=Aub,
        ub=ub,
        deadline=deadline,
    )
    if private.objective_semantics != source.parent_objective_semantics:
        raise PCOHRelaxedLPLiveOwnerError(
            "parent_fresh_exact_objective_semantics_differ"
        )
    semantics_sha = _objective_semantics_sha256(private.objective_semantics)
    candidate = _candidate(private, deadline=deadline, caps=caps)
    prepared, candidate_sha, candidate_identity = (
        _prepare_and_validate_candidate(
            private,
            candidate,
            label="private_fresh_scaled",
            deadline=deadline,
        )
    )
    upper_ld, dual_receipt = (
        _hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope(
            objective_envelope=private.envelope,
            expected_parent_semantic_digest=source.fresh_semantic_digest,
            expected_exact_objective_sha256=(
                private.envelope.exact_objective_sha256
            ),
            expected_objective_binding_sha256=(
                private.envelope.objective_binding_sha256
            ),
            Auc=Auc,
            Aub=Aub,
            Ac=fresh.hz.Ac,
            Ab=fresh.hz.Ab,
            ub=ub,
            b=fresh.hz.b,
            continuous_lb=private.problem.continuous_lb,
            continuous_ub=private.problem.continuous_ub,
            binary_lb=private.problem.binary_lb,
            binary_ub=private.problem.binary_ub,
            upper_row_dual=candidate.upper_row_dual,
            equality_row_dual=candidate.equality_row_dual,
            deadline=deadline,
        )
    )
    upper, fresh_anchor = _fresh_dual_anchor(
        upper_ld, dual_receipt, prepared
    )
    _terminal_numeric_recheck(
        prepared,
        candidate,
        candidate_sha,
        candidate_identity,
        label="private_fresh_scaled",
        deadline=deadline,
    )
    fresh_terminal = sparse_hz_semantic_digest(fresh.hz)
    if fresh_terminal != source.fresh_semantic_digest:
        raise PCOHRelaxedLPLiveOwnerError("fresh_terminal_semantic_seal_changed")
    _check_deadline(deadline, "fresh_dual_and_terminal_seals")
    scaling_payload = dict(scaling)
    scaling_payload["scaled_numeric_frame_sha256"] = prepared.frame_sha256
    scaling_payload["scaling_receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in scaling_payload.items()
         if key != "scaling_receipt_sha256"}
    )
    return _FreshPhaseOutput(
        fresh_weakrefs=fresh_weakrefs,
        fresh_upper=upper,
        fresh_anchor_sha256=fresh_anchor,
        fresh_candidate_receipt_sha256=candidate_sha,
        fresh_frame_sha256=prepared.frame_sha256,
        fresh_semantic_digest=source.fresh_semantic_digest,
        fresh_terminal_semantic_digest=fresh_terminal,
        row_scaling_receipt=_deep_freeze(scaling_payload),
        objective_semantics_sha256=semantics_sha,
        timings=MappingProxyType(
            {"fresh_lp_and_dual_seconds": float(time.monotonic() - started)}
        ),
    )


def _all_released(refs: Tuple[weakref.ReferenceType[Any], ...]) -> bool:
    return all(reference() is None for reference in refs)


def _run_transaction(
    token: str,
    *,
    deadline: float,
    caps: PCOHRelaxedLPToyOwnerCaps,
) -> _ResultDraft:
    started = time.monotonic()
    entry_rss = _rss_bytes()
    source_output = None
    fresh_build = None
    fresh_output = None
    try:
        _mark_transaction(token, "private_source_and_parent_lp")
        source_output = _source_phase(deadline=deadline, caps=caps)
        rss_with_source_and_fresh = _rss_bytes()
        fresh_build = source_output.fresh_build
        source_refs = source_output.source_weakrefs
        source_pure = replace(
            source_output, fresh_build=None, source_weakrefs=()
        )
        # Destroy the dataclass owner of the fresh build only after extracting
        # that one fresh owner.  No source object is returned by source_phase.
        source_output = None
        gc.collect()
        trim_attempted, trim_result = _malloc_trim()
        gc.collect()
        post_source_rss = _rss_bytes()
        if not _all_released(source_refs):
            raise PCOHRelaxedLPLiveOwnerError(
                "private_source_weakrefs_survived_release_gate"
            )
        if post_source_rss > caps.max_post_source_release_rss_bytes:
            raise PCOHRelaxedLPLiveOwnerError(
                "post_source_release_rss_gate_exceeded"
            )
        _check_deadline(deadline, "post_source_release_gc_trim_rss_gate")

        _mark_transaction(token, "fresh_lp_after_source_release")
        fresh_output = _fresh_phase(
            fresh_build, source_pure, deadline=deadline, caps=caps
        )
        fresh_refs = fresh_output.fresh_weakrefs
        fresh_build = None
        gc.collect()
        terminal_trim_attempted, terminal_trim_result = _malloc_trim()
        gc.collect()
        terminal_rss = _rss_bytes()
        if not _all_released(fresh_refs):
            raise PCOHRelaxedLPLiveOwnerError(
                "private_fresh_weakrefs_survived_terminal_release"
            )
        _check_deadline(deadline, "fresh_release_before_receipt")

        lower_fraction = Fraction.from_float(source_pure.parent_lower)
        upper_fraction = Fraction.from_float(fresh_output.fresh_upper)
        gap = lower_fraction - upper_fraction
        if not upper_fraction < lower_fraction:
            raise PCOHRelaxedLPLiveOwnerError(
                "valid_private_bounds_without_strict_fraction_ordering"
            )
        payload = {
            "schema": _RECEIPT_SCHEMA,
            "status": "strict_private_toy_numeric_ordering_certified",
            "toy_only": True,
            "production_ready": False,
            "proof_authority": False,
            "verdict_authority": False,
            "real_parent_binding_authority": False,
            "real_sound_extension_authority": False,
            "internal_toy_live_binding": True,
            "supplied_numeric_strict_ordering_theorem": True,
            "toy_numeric_frame_authority": True,
            "toy_sound_extension_claimed": False,
            "strict_comparison": (
                "Fraction.from_float(fresh_upper)<"
                "Fraction.from_float(parent_lower)"
            ),
            "parent_lower": source_pure.parent_lower,
            "parent_lower_hex": source_pure.parent_lower.hex(),
            "fresh_upper": fresh_output.fresh_upper,
            "fresh_upper_hex": fresh_output.fresh_upper.hex(),
            "exact_gap": _fraction_pair(gap),
            "source_semantic_digest": source_pure.source_semantic_digest,
            "source_terminal_semantic_digest": (
                source_pure.source_terminal_semantic_digest
            ),
            "fresh_semantic_digest": fresh_output.fresh_semantic_digest,
            "fresh_terminal_semantic_digest": (
                fresh_output.fresh_terminal_semantic_digest
            ),
            "parent_frame_sha256": source_pure.parent_frame_sha256,
            "fresh_scaled_frame_sha256": fresh_output.fresh_frame_sha256,
            "parent_candidate_receipt_sha256": (
                source_pure.parent_candidate_receipt_sha256
            ),
            "fresh_candidate_receipt_sha256": (
                fresh_output.fresh_candidate_receipt_sha256
            ),
            "parent_primal_anchor_sha256": source_pure.parent_anchor_sha256,
            "fresh_dual_anchor_sha256": fresh_output.fresh_anchor_sha256,
            "parent_cg_native_model_closed_before_primal_replay": True,
            "fresh_cg_native_model_closed_before_dual_replay": True,
            "maximum_simultaneous_highs_models_by_construction": 1,
            "parent_solver_primal_used_as_authority": False,
            "parent_anchor_source": (
                "private_fixed_toy_all_zero_factor_exact_replay_v1"
            ),
            "fresh_solver_dual_used_only_after_independent_replay": True,
            "objective_semantics_sha256": (
                fresh_output.objective_semantics_sha256
            ),
            "objective_semantics_exactly_equal_parent_fresh": True,
            "fresh_row_scaling": fresh_output.row_scaling_receipt,
            "fresh_issuance_sha256": source_pure.fresh_issuance_sha256,
            "fresh_materializer_receipt_sha256": (
                source_pure.fresh_receipt_sha256
            ),
            "materialized_tightness_summary_sha256": (
                source_pure.materialized_tightness_summary_sha256
            ),
            "stable_bit_ids": source_pure.stable_bit_ids,
            "source_dimensions": source_pure.source_dimensions,
            "fresh_dimensions": source_pure.fresh_dimensions,
            "source_payload_bytes": source_pure.source_payload_bytes,
            "fresh_payload_bytes": source_pure.fresh_payload_bytes,
            "source_fresh_detachment_pairs_checked": (
                source_pure.detachment_pairs_checked
            ),
            "public_accepts_build_or_numeric_frame": False,
            "public_accepts_source_callback": False,
            "receipt_sha256_keyed_authenticator": False,
            "pure_checker_authenticates_producer_provenance": False,
            "pure_checker_authenticates_opaque_binding_hashes": False,
            "pure_checker_scope": (
                "selected_fixed_toy_structural_and_known_numeric_"
                "invariants_only"
            ),
            "source_owner_scope": "fixed_internal_k4_toy_recipe_only",
            "caller_can_retain_source_alias_through_public_api": False,
            "source_weakrefs_released_before_fresh_lp": True,
            "fresh_weakrefs_released_before_receipt": True,
            "source_and_fresh_overlap_scope": (
                "fresh_materialization_and_detachment_check_only"
            ),
            "owner_registry_one_use": True,
            "owner_registry_contains_hz": False,
            "owner_registry_pop_required_before_return": True,
            "fresh_registry_cleanup_route": (
                "public_then_frozen_then_exact_internal_token_pop_v1"
            ),
            "shared_absolute_deadline": True,
            "hard_wall_deadline_guaranteed": False,
            "row_scaling_exact_fraction_replay_required": True,
            "uses_sparse_hstack": False,
            "uses_sparse_vstack": False,
            "entry_rss_bytes": entry_rss,
            "rss_with_source_and_fresh_bytes": rss_with_source_and_fresh,
            "post_source_release_rss_bytes": post_source_rss,
            "post_source_release_rss_cap_bytes": (
                caps.max_post_source_release_rss_bytes
            ),
            "post_source_release_rss_hard_ceiling_bytes": 5 * _GIB // 2,
            "post_source_release_rss_cap_caller_tightenable_only": True,
            "post_source_release_rss_gate_passed": True,
            "source_release_gc_called": True,
            "source_release_malloc_trim_attempted": trim_attempted,
            "source_release_malloc_trim_result": trim_result,
            "terminal_rss_bytes": terminal_rss,
            "terminal_gc_called": True,
            "terminal_malloc_trim_attempted": terminal_trim_attempted,
            "terminal_malloc_trim_result": terminal_trim_result,
            "timings": {
                **dict(source_pure.timings),
                **dict(fresh_output.timings),
                "total_seconds": float(time.monotonic() - started),
            },
            "real_owner_blocker": (
                "generic real source requires an upstream private producer "
                "capability/registry; accepting caller build/arrays/CSR/"
                "callback cannot establish no-alias authority"
            ),
            "real_owner_minimum_upstream_api": (
                "issuer builds source from immutable model/spec identity; "
                "one-use consume transfers it only to a trusted internal "
                "owner continuation; terminal pop/cleanup never exposes HZ"
            ),
        }
        _mark_transaction(token, "private_objects_released_receipt_draft")
        return _ResultDraft(
            parent_lower=source_pure.parent_lower,
            fresh_upper=fresh_output.fresh_upper,
            exact_gap=gap,
            source_semantic_digest=source_pure.source_semantic_digest,
            fresh_semantic_digest=fresh_output.fresh_semantic_digest,
            payload=_deep_freeze(payload),
        )
    finally:
        source_output = None
        fresh_build = None
        fresh_output = None
        gc.collect()


def _clear_exception_traceback(exc: BaseException) -> None:
    current_code = _clear_exception_traceback.__code__
    cursor = exc.__traceback__
    while cursor is not None:
        frame = cursor.tb_frame
        cursor = cursor.tb_next
        if frame.f_code is current_code:
            continue
        try:
            frame.clear()
        except RuntimeError:
            pass
    exc.__traceback__ = None
    exc.__cause__ = None
    exc.__context__ = None


def run_private_k4_pcoh_relaxed_lp_live_owner(
    *,
    deadline: float,
    caps: PCOHRelaxedLPToyOwnerCaps = PCOHRelaxedLPToyOwnerCaps(),
) -> PCOHRelaxedLPToyOwnerResult:
    """Run the fixed private K4 toy and return one receipt-only result.

    The signature is intentionally closed: no source build, numeric frame,
    objective, callback, ground truth, solver object, or capability is
    accepted.  Ordinary exceptions and ``BaseException`` interruptions are
    converted to a traceback-scrubbed owner error after both registries have
    followed their cleanup paths.
    """

    deadline_value = _deadline(deadline)
    normalized_caps = _normalize_caps(caps)
    token: Optional[str] = None
    draft: Optional[_ResultDraft] = None
    failure: Optional[str] = None
    interrupted = False
    registry_release_ok = True
    try:
        token = _reserve_transaction()
        draft = _run_transaction(
            token, deadline=deadline_value, caps=normalized_caps
        )
    except BaseException as exc:
        interrupted = not isinstance(exc, Exception)
        failure = f"{type(exc).__name__}:{str(exc)[:300]}"
        _clear_exception_traceback(exc)
    finally:
        registry_release_ok = _release_transaction(token)
        token = None
        gc.collect()
    if failure is not None:
        prefix = "interrupted" if interrupted else "failed_closed"
        if not registry_release_ok or _active_transaction_count() != 0:
            failure += ":owner_registry_cleanup_failed"
        raise PCOHRelaxedLPLiveOwnerError(prefix + ":" + failure) from None
    if (
        draft is None
        or not registry_release_ok
        or _active_transaction_count() != 0
    ):
        raise PCOHRelaxedLPLiveOwnerError(
            "failed_closed:owner_registry_terminal_state_invalid"
        ) from None
    payload = dict(draft.payload)
    payload["owner_registry_empty_before_return"] = True
    payload["receipt_sha256"] = _canonical_sha256(payload)
    frozen = _deep_freeze(payload)
    return PCOHRelaxedLPToyOwnerResult(
        schema=_SCHEMA,
        status="strict_private_toy_numeric_ordering_certified",
        parent_lower=draft.parent_lower,
        fresh_upper=draft.fresh_upper,
        exact_gap=draft.exact_gap,
        source_semantic_digest=draft.source_semantic_digest,
        fresh_semantic_digest=draft.fresh_semantic_digest,
        receipt=frozen,
        receipt_sha256=payload["receipt_sha256"],
        production_ready=False,
        proof_authority=False,
        verdict_authority=False,
        real_parent_binding_authority=False,
    )


def _receipt_is_pure(value: Any) -> bool:
    if value is None or type(value) in {str, bool, int, float}:
        return type(value) is not float or math.isfinite(value)
    if type(value) in {tuple, list}:
        return all(_receipt_is_pure(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            type(key) is str and _receipt_is_pure(item)
            for key, item in value.items()
        )
    return False


def _valid_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def verify_private_k4_pcoh_relaxed_lp_live_owner_result(value: Any) -> bool:
    """Check selected fixed-toy structural and known numeric invariants.

    This checker rejects unknown fields, authority-flag changes, and changes
    to the fixed known bounds/scaling shape even if a caller recomputes the
    unkeyed integrity hashes.  It cannot authenticate producer provenance or
    the values of opaque live binding/hash diagnostics from the pure receipt;
    those hashes are checked only for canonical form and cross-fields that are
    present in the same unkeyed payload.  Consequently a successful check is
    not proof, verdict, real-parent, sound-extension, or provenance authority.
    """

    try:
        if type(value) is not PCOHRelaxedLPToyOwnerResult:
            return False
        receipt = value.receipt
        if (
            type(receipt) is not MappingProxyType
            or set(receipt) != _RECEIPT_KEYS
            or not _receipt_is_pure(receipt)
        ):
            return False
        claimed = receipt.get("receipt_sha256")
        raw = dict(receipt)
        raw.pop("receipt_sha256", None)
        scaling = receipt.get("fresh_row_scaling")
        if (
            type(scaling) is not MappingProxyType
            or set(scaling) != _SCALING_RECEIPT_KEYS
        ):
            return False
        scaling_claimed = scaling.get("scaling_receipt_sha256")
        scaling_raw = dict(scaling)
        scaling_raw.pop("scaling_receipt_sha256", None)
        gap = Fraction.from_float(value.parent_lower) - Fraction.from_float(
            value.fresh_upper
        )
        source_digest = receipt.get("source_semantic_digest")
        fresh_digest = receipt.get("fresh_semantic_digest")
        source_dimensions = tuple(receipt.get("source_dimensions", ()))
        fresh_dimensions = tuple(receipt.get("fresh_dimensions", ()))
        expected_scaled_rows = tuple(
            range(source_dimensions[4], fresh_dimensions[4])
        ) if len(source_dimensions) == 5 and len(fresh_dimensions) == 5 else ()
        sha_fields = (
            "parent_frame_sha256",
            "fresh_scaled_frame_sha256",
            "parent_candidate_receipt_sha256",
            "fresh_candidate_receipt_sha256",
            "parent_primal_anchor_sha256",
            "fresh_dual_anchor_sha256",
            "objective_semantics_sha256",
            "fresh_issuance_sha256",
            "fresh_materializer_receipt_sha256",
            "materialized_tightness_summary_sha256",
        )
        timings = receipt.get("timings")
        return bool(
            value.schema == _SCHEMA
            and value.status
            == "strict_private_toy_numeric_ordering_certified"
            and value.production_ready is False
            and value.proof_authority is False
            and value.verdict_authority is False
            and value.real_parent_binding_authority is False
            and value.parent_lower == _EXPECTED_PARENT_LOWER
            and value.fresh_upper == _EXPECTED_FRESH_UPPER
            and value.exact_gap == gap
            and gap > 0
            and value.receipt_sha256 == claimed
            and type(claimed) is str
            and _canonical_sha256(raw) == claimed
            and type(scaling_claimed) is str
            and _canonical_sha256(scaling_raw) == scaling_claimed
            and receipt.get("schema") == _RECEIPT_SCHEMA
            and receipt.get("status")
            == "strict_private_toy_numeric_ordering_certified"
            and receipt.get("toy_only") is True
            and receipt.get("production_ready") is False
            and receipt.get("proof_authority") is False
            and receipt.get("verdict_authority") is False
            and receipt.get("real_parent_binding_authority") is False
            and receipt.get("real_sound_extension_authority") is False
            and receipt.get("toy_sound_extension_claimed") is False
            and receipt.get("internal_toy_live_binding") is True
            and receipt.get("supplied_numeric_strict_ordering_theorem") is True
            and receipt.get("toy_numeric_frame_authority") is True
            and receipt.get("strict_comparison")
            == (
                "Fraction.from_float(fresh_upper)<"
                "Fraction.from_float(parent_lower)"
            )
            and receipt.get("source_semantic_digest")
            == value.source_semantic_digest
            and receipt.get("fresh_semantic_digest")
            == value.fresh_semantic_digest
            and receipt.get("parent_lower") == value.parent_lower
            and receipt.get("fresh_upper") == value.fresh_upper
            and receipt.get("parent_lower_hex") == value.parent_lower.hex()
            and receipt.get("fresh_upper_hex") == value.fresh_upper.hex()
            and tuple(receipt.get("exact_gap", ())) == _fraction_pair(gap)
            and _valid_sha256(source_digest)
            and _valid_sha256(fresh_digest)
            and source_digest != fresh_digest
            and receipt.get("source_terminal_semantic_digest") == source_digest
            and receipt.get("fresh_terminal_semantic_digest") == fresh_digest
            and all(_valid_sha256(receipt.get(name)) for name in sha_fields)
            and receipt.get("owner_registry_empty_before_return") is True
            and receipt.get("owner_registry_one_use") is True
            and receipt.get("owner_registry_contains_hz") is False
            and receipt.get("owner_registry_pop_required_before_return") is True
            and receipt.get("fresh_registry_cleanup_route")
            == "public_then_frozen_then_exact_internal_token_pop_v1"
            and receipt.get("source_weakrefs_released_before_fresh_lp") is True
            and receipt.get("fresh_weakrefs_released_before_receipt") is True
            and receipt.get("maximum_simultaneous_highs_models_by_construction")
            == 1
            and receipt.get("parent_cg_native_model_closed_before_primal_replay")
            is True
            and receipt.get("fresh_cg_native_model_closed_before_dual_replay")
            is True
            and receipt.get("parent_solver_primal_used_as_authority") is False
            and receipt.get("parent_anchor_source")
            == "private_fixed_toy_all_zero_factor_exact_replay_v1"
            and receipt.get("fresh_solver_dual_used_only_after_independent_replay")
            is True
            and receipt.get("objective_semantics_exactly_equal_parent_fresh")
            is True
            and source_dimensions == (3, 9, 4, 0, 12)
            and fresh_dimensions == (3, 25, 4, 16, 13)
            and type(receipt.get("stable_bit_ids")) is tuple
            and len(receipt.get("stable_bit_ids")) == 4
            and len(set(receipt.get("stable_bit_ids"))) == 4
            and all(type(item) is int and item >= 0
                    for item in receipt.get("stable_bit_ids"))
            and all(
                receipt.get("stable_bit_ids")[offset + 1]
                == receipt.get("stable_bit_ids")[offset] + 1
                for offset in range(3)
            )
            and type(receipt.get("source_payload_bytes")) is int
            and type(receipt.get("fresh_payload_bytes")) is int
            and receipt.get("source_payload_bytes") == 1068
            and receipt.get("fresh_payload_bytes") == 2868
            and type(receipt.get("source_fresh_detachment_pairs_checked"))
            is int
            and receipt.get("source_fresh_detachment_pairs_checked") == 784
            and receipt.get("public_accepts_build_or_numeric_frame") is False
            and receipt.get("public_accepts_source_callback") is False
            and receipt.get("receipt_sha256_keyed_authenticator") is False
            and receipt.get("pure_checker_authenticates_producer_provenance")
            is False
            and receipt.get("pure_checker_authenticates_opaque_binding_hashes")
            is False
            and receipt.get("pure_checker_scope")
            == "selected_fixed_toy_structural_and_known_numeric_invariants_only"
            and receipt.get("caller_can_retain_source_alias_through_public_api")
            is False
            and receipt.get("source_owner_scope")
            == "fixed_internal_k4_toy_recipe_only"
            and receipt.get("source_and_fresh_overlap_scope")
            == "fresh_materialization_and_detachment_check_only"
            and receipt.get("shared_absolute_deadline") is True
            and receipt.get("hard_wall_deadline_guaranteed") is False
            and receipt.get("row_scaling_exact_fraction_replay_required") is True
            and receipt.get("uses_sparse_hstack") is False
            and receipt.get("uses_sparse_vstack") is False
            and receipt.get("post_source_release_rss_gate_passed") is True
            and type(receipt.get("entry_rss_bytes")) is int
            and receipt.get("entry_rss_bytes") > 0
            and type(receipt.get("rss_with_source_and_fresh_bytes")) is int
            and receipt.get("rss_with_source_and_fresh_bytes") > 0
            and type(receipt.get("post_source_release_rss_bytes")) is int
            and type(receipt.get("post_source_release_rss_cap_bytes")) is int
            and receipt.get("post_source_release_rss_bytes") > 0
            and receipt.get("post_source_release_rss_cap_bytes") > 0
            and receipt.get("post_source_release_rss_hard_ceiling_bytes")
            == 5 * _GIB // 2
            and receipt.get("post_source_release_rss_cap_caller_tightenable_only")
            is True
            and receipt.get("post_source_release_rss_cap_bytes")
            <= 5 * _GIB // 2
            and receipt.get("post_source_release_rss_bytes")
            <= receipt.get("post_source_release_rss_cap_bytes")
            and receipt.get("source_release_gc_called") is True
            and receipt.get("source_release_malloc_trim_attempted") is True
            and type(receipt.get("source_release_malloc_trim_result"))
            in {type(None), bool}
            and type(receipt.get("terminal_rss_bytes")) is int
            and receipt.get("terminal_rss_bytes") > 0
            and receipt.get("terminal_gc_called") is True
            and receipt.get("terminal_malloc_trim_attempted") is True
            and type(receipt.get("terminal_malloc_trim_result"))
            in {type(None), bool}
            and type(timings) is MappingProxyType
            and set(timings) == {
                "parent_lp_and_primal_seconds",
                "source_phase_total_seconds",
                "fresh_lp_and_dual_seconds",
                "total_seconds",
            }
            and all(type(item) is float and math.isfinite(item) and item >= 0.0
                    for item in timings.values())
            and timings.get("source_phase_total_seconds")
            >= timings.get("parent_lp_and_primal_seconds")
            and timings.get("total_seconds")
            >= timings.get("source_phase_total_seconds")
            and timings.get("total_seconds")
            >= timings.get("fresh_lp_and_dual_seconds")
            and receipt.get("real_owner_blocker")
            == (
                "generic real source requires an upstream private producer "
                "capability/registry; accepting caller build/arrays/CSR/"
                "callback cannot establish no-alias authority"
            )
            and receipt.get("real_owner_minimum_upstream_api")
            == (
                "issuer builds source from immutable model/spec identity; "
                "one-use consume transfers it only to a trusted internal "
                "owner continuation; terminal pop/cleanup never exposes HZ"
            )
            and scaling.get("scale_exponent") == _ROW_SCALE_EXPONENT
            and scaling.get("scale_exact_integer") == _ROW_SCALE
            and scaling.get("scale_positive") is True
            and scaling.get("all_changed_coefficients_and_rhs_exactly_scaled")
            is True
            and scaling.get("unscaled_prefix_byte_equal") is True
            and scaling.get("row_feasible_set_equivalence")
            == "positive_exact_dyadic_scaling_v1"
            and scaling.get("production_ready") is False
            and scaling.get("proof_authority") is False
            and scaling.get("verdict_authority") is False
            and scaling.get("scaling_receipt_sha256_keyed_authenticator")
            is False
            and scaling.get("block_hashes_authenticate_live_provenance")
            is False
            and scaling.get("schema")
            == "act.hybridz_pcoh_positive_upper_row_scaling.toy.v1"
            and scaling.get("fresh_semantic_digest") == fresh_digest
            and tuple(scaling.get("scaled_upper_rows", ()))
            == expected_scaled_rows
            and scaling.get("source_upper_row_count") == source_dimensions[4]
            and scaling.get("fresh_upper_row_count") == fresh_dimensions[4]
            and type(scaling.get("exact_fraction_items_replayed")) is int
            and scaling.get("exact_fraction_items_replayed") == 12
            and scaling.get("scaled_numeric_frame_sha256")
            == receipt.get("fresh_scaled_frame_sha256")
            and all(
                _valid_sha256(scaling.get(name))
                for name in (
                    "original_Auc_sha256",
                    "scaled_Auc_sha256",
                    "original_Aub_sha256",
                    "scaled_Aub_sha256",
                    "original_ub_sha256",
                    "scaled_ub_sha256",
                    "scaled_numeric_frame_sha256",
                )
            )
            and scaling.get("original_Auc_sha256")
            != scaling.get("scaled_Auc_sha256")
            and scaling.get("original_ub_sha256")
            != scaling.get("scaled_ub_sha256")
            and _active_transaction_count() == 0
        )
    except Exception:
        return False


__all__ = [
    "PCOHRelaxedLPLiveOwnerError",
    "PCOHRelaxedLPToyOwnerCaps",
    "PCOHRelaxedLPToyOwnerResult",
    "run_private_k4_pcoh_relaxed_lp_live_owner",
    "verify_private_k4_pcoh_relaxed_lp_live_owner_result",
]
