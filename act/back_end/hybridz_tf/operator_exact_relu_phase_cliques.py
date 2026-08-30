#!/usr/bin/env python3
"""Candidate-only exact phase-conflict cliques for live Operator-HZ builds.

The Operator-HZ exact-ReLU adapter proves only that an explicitly focused
ordered rival subset has a direct, unanimous signed effect through a mapped
ReLU output.  The focused subset may contain one rival.  This module treats
that selection strictly as a proposal:

* rank selected phases by the minimum exact absolute coefficient across all
  focused ordered rivals;
* bind the deterministic top-K subset to the live parent, property, Operator
  row tags, selection receipt, caller caps, and exact scores;
* probe every top-K pair in one persistent HiGHS relaxation;
* accept an edge only after a full-parent ``Fraction`` dual-ray replay; and
* append cuts only for deterministic graph cliques whose every edge has such
  a certificate.

This common-sign path does not claim coverage of all rivals in a large-class
property.  Mixed-sign rival signature grouping is a separate future stage.
Nevertheless, every accepted conflict is proved from the complete parent
feasible set, so an emitted clique cut is globally valid and may safely
tighten rivals that were not in the focused selection subset.

With the default ``emit_cut_hz=True``, the optional ``result.hz`` is a
semantic SparseHZ cut copy.  It intentionally does not preserve Operator
row-tag metadata or unrelated proof receipts; the live tagged
``OperatorHZBuild`` remains the sole selection input.  The pipeline-only
compact mode sets ``emit_cut_hz=False`` and returns only the exact certificates
and clique descriptor.  Its hardened verifier still reconstructs the unique
cut from a separate private parent snapshot before issuing any capability.

The result is isolated from verifier/BaB verdict paths and always carries
``proof_authority=False``.  Solver status, graph heuristics, and telemetry
never authorize an edge or a cut without exact certificate replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
import secrets
import threading
import time
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple
import weakref

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds as ScipyBounds
from scipy.optimize import LinearConstraint as ScipyLinearConstraint
from scipy.optimize import linprog, milp

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    RivalSpec,
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
    OperatorExactReLUPhaseMapping,
    OperatorExactReLUPhaseSelection,
    derive_operator_exact_relu_property_phase_literals,
)
from act.back_end.hybridz_tf.operator_hz import (
    OperatorHZBuild,
    validate_operator_hz_constructive_nonempty_seal,
)
from act.back_end.hybridz_tf.persistent_phase_conflict_oracle import (
    ExactDualRayConflictCertificate,
    ExactSourceTermV2,
    PersistentPairRecord,
    _CANDIDATE_DUST_ABS,
    _MAX_BINARY_CHANGE_COEFFICIENTS,
    _PersistentHighsPairLP,
    _ordered_source_frame_digest,
    _verify_exact_certificate_with_source_frame,
    exact_certificate_from_highs_dual_ray_candidate,
)
from act.back_end.hybridz_tf.property_phase_conflict_clique import (
    PhaseLiteral,
    _literal_binding_digest,
    _ordered_pair,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    _base_milp_matrices,
    hz_constructively_nonempty,
)


class OperatorExactReLUPhaseCliqueError(ValueError):
    """A live selection, cap, deadline, or exact closure failed closed."""


@dataclass(frozen=True)
class OperatorPhaseCliqueCaps:
    """Caller-bound resource limits for one exact graph closure."""

    max_parent_variables: int
    max_parent_rows: int
    max_parent_nonzeros: int
    max_parent_buffer_items: int
    max_top_literals: int
    max_total_pairs: int
    max_cliques: int
    max_clique_search_nodes: int
    max_source_terms: int
    max_multiplier_bits: int
    max_exact_bits: int
    max_exact_nonzeros: int


@dataclass(frozen=True)
class RankedOperatorPhase:
    """One top-K phase and its minimum exact cross-rival effect."""

    rank: int
    stable_bcol_id: int
    phase: int
    score_numerator: int
    score_denominator: int
    proof_authority: bool = False

    @property
    def score(self) -> Fraction:
        return Fraction(self.score_numerator, self.score_denominator)


@dataclass(frozen=True)
class OperatorCertifiedPhaseClique:
    """One deterministic clique backed by all of its exact edge proofs."""

    clique_id: str
    literals: Tuple[PhaseLiteral, ...]
    edge_certificate_sha256s: Tuple[str, ...]
    total_score_numerator: int
    total_score_denominator: int
    cut_applied: bool = True
    proof_authority: bool = False

    @property
    def total_score(self) -> Fraction:
        return Fraction(
            self.total_score_numerator,
            self.total_score_denominator,
        )


@dataclass(frozen=True)
class OperatorExactReLUPhaseCliqueResult:
    """Exact graph closure plus optional non-authoritative cut candidate."""

    status: str
    hz: Optional[SparseHZono]
    parent_semantic_digest: str
    focused_property_digest: str
    operator_row_tag_digest: str
    selection_digest: str
    subset_binding_digest: str
    ordered_source_frame_sha256: Optional[str]
    caps: OperatorPhaseCliqueCaps
    ranked_phases: Tuple[RankedOperatorPhase, ...]
    literals: Tuple[PhaseLiteral, ...]
    omitted_zero_bcol_ids: Tuple[int, ...]
    excluded_selected_bcol_ids: Tuple[int, ...]
    pair_records: Tuple[PersistentPairRecord, ...]
    certificates: Tuple[
        ExactDualRayConflictCertificate, ...
    ]
    cliques: Tuple[OperatorCertifiedPhaseClique, ...]
    telemetry: Mapping[str, Any]
    proof_authority: bool = False


@dataclass(frozen=True, eq=False)
class VerifiedOperatorPhaseCliqueCapability:
    """Opaque one-use handle for a verifier-owned private snapshot."""

    token: str
    snapshot_digest: str
    expires_monotonic: float
    proof_authority: bool = False


@dataclass(frozen=True)
class VerifiedOperatorPhaseCliqueSnapshot:
    """Unique deep snapshot transferred by one successful consume."""

    cut_hz: SparseHZono
    verified_cliques: Tuple[
        Tuple[
            str,
            Tuple[Tuple[int, int, str], ...],
        ],
        ...,
    ]
    parent_row_tags: Tuple[str, ...]
    continuous_layer_ids: np.ndarray
    full_col_ids: np.ndarray
    input_center: np.ndarray
    input_radius: np.ndarray
    build_input_col_ids: np.ndarray
    input_layer_id: int
    output_layer_id: int
    assert_layer_id: int
    original_parent_n_ub: int
    parent_semantic_digest: str
    ordered_source_frame_sha256: Optional[str]
    focused_property_digest: str
    selection_digest: str
    subset_binding_digest: str
    verified_result_digest: str
    caps_payload: Tuple[Tuple[str, int], ...]
    materializer_source_modes: Tuple[
        Tuple[str, Any], ...
    ]
    producer_nonempty_seal_verified: bool
    snapshot_digest: str
    proof_authority: bool = False


@dataclass
class _VerifiedSnapshotRecord:
    capability_ref: "weakref.ReferenceType[VerifiedOperatorPhaseCliqueCapability]"
    snapshot: VerifiedOperatorPhaseCliqueSnapshot
    expires_monotonic: float
    process_id: int


_DEFAULT_MAX_TOP_LITERALS = 16
_DEFAULT_MAX_TOTAL_PAIRS = 120
_DEFAULT_MAX_CLIQUES = 16
_DEFAULT_MAX_CLIQUE_SEARCH_NODES = 100000
_DEFAULT_MAX_PARENT_VARIABLES = 2_000_000
_DEFAULT_MAX_PARENT_ROWS = 2_000_000
_DEFAULT_MAX_PARENT_NONZEROS = 50_000_000
_DEFAULT_MAX_PARENT_BUFFER_ITEMS = 120_000_000
_DEFAULT_MAX_SOURCE_TERMS = 128
_DEFAULT_MAX_MULTIPLIER_BITS = 256
_DEFAULT_MAX_EXACT_BITS = 4096
_DEFAULT_MAX_EXACT_NONZEROS = 200000

_HARD_LIMITS = {
    "max_parent_variables": 10_000_000,
    "max_parent_rows": 10_000_000,
    "max_parent_nonzeros": 500_000_000,
    "max_parent_buffer_items": 1_200_000_000,
    "max_top_literals": 64,
    "max_total_pairs": 2016,
    "max_cliques": 64,
    "max_clique_search_nodes": 5_000_000,
    "max_source_terms": 256,
    "max_multiplier_bits": 2048,
    "max_exact_bits": 16384,
    "max_exact_nonzeros": 1000000,
}
_VALID_PAIR_STATUSES = {
    "certified_conflict",
    "feasible_or_unknown",
    "infeasible_without_ray",
    "exact_replay_rejected",
}
_VALID_SOURCE_KINDS = {
    "upper",
    "equality_pos",
    "equality_neg",
}
_MAX_INT64 = int(np.iinfo(np.int64).max)
_MAX_SELECTION_MAPPINGS = 65536
_TELEMETRY_KEY_COUNT = 14
_SNAPSHOT_REGISTRY_CAPACITY = 1024
_SNAPSHOT_DEFAULT_TTL_SECONDS = 300.0
_SNAPSHOT_HARD_TTL_SECONDS = 3600.0
_MAX_HZ_INSTANCE_ATTRIBUTES = 256
_LEGACY_SUCCESS_STATUS = "focused_rival_clique_cut_candidate"
_LEGACY_EMPTY_STATUS = "no_certified_focused_rival_phase_clique"
_COMPACT_SUCCESS_STATUS = "focused_rival_clique_compact_candidate"
_COMPACT_EMPTY_STATUS = (
    "no_certified_focused_rival_phase_clique_compact"
)
_LEGACY_TELEMETRY_SCHEMA = (
    "act.operator_exact_relu_phase_clique_candidate.v1"
)
_COMPACT_TELEMETRY_SCHEMA = (
    "act.operator_exact_relu_phase_clique_compact_candidate.v1"
)
_PROGRESS_SCHEMA = (
    "act.operator_exact_relu_phase_clique_progress.v1"
)
_SNAPSHOT_REGISTRY: dict[str, _VerifiedSnapshotRecord] = {}
_SNAPSHOT_REGISTRY_LOCK = threading.RLock()


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(
            character in "0123456789abcdef"
            for character in value
        )
    )


def _strict_int(value: Any, *, name: str) -> int:
    if type(value) is not int:
        raise OperatorExactReLUPhaseCliqueError(
            f"{name}_not_builtin_integer"
        )
    return value


def _normalize_deadline(deadline: Any) -> float:
    if (
        type(deadline) not in {int, float}
        or type(deadline) is bool
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "deadline_not_builtin_numeric"
        )
    result = float(deadline)
    if not math.isfinite(result):
        raise OperatorExactReLUPhaseCliqueError(
            "deadline_nonfinite"
        )
    return result


def _check_deadline(deadline: float, *, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise OperatorExactReLUPhaseCliqueError(
            f"deadline_expired_{stage}"
        )


def _normalize_caps(
    *,
    max_parent_variables: int,
    max_parent_rows: int,
    max_parent_nonzeros: int,
    max_parent_buffer_items: int,
    max_top_literals: int,
    max_total_pairs: int,
    max_cliques: int,
    max_clique_search_nodes: int,
    max_source_terms: int,
    max_multiplier_bits: int,
    max_exact_bits: int,
    max_exact_nonzeros: int,
) -> OperatorPhaseCliqueCaps:
    values = {
        "max_parent_variables": _strict_int(
            max_parent_variables, name="max_parent_variables"
        ),
        "max_parent_rows": _strict_int(
            max_parent_rows, name="max_parent_rows"
        ),
        "max_parent_nonzeros": _strict_int(
            max_parent_nonzeros, name="max_parent_nonzeros"
        ),
        "max_parent_buffer_items": _strict_int(
            max_parent_buffer_items,
            name="max_parent_buffer_items",
        ),
        "max_top_literals": _strict_int(
            max_top_literals, name="max_top_literals"
        ),
        "max_total_pairs": _strict_int(
            max_total_pairs, name="max_total_pairs"
        ),
        "max_cliques": _strict_int(
            max_cliques, name="max_cliques"
        ),
        "max_clique_search_nodes": _strict_int(
            max_clique_search_nodes,
            name="max_clique_search_nodes",
        ),
        "max_source_terms": _strict_int(
            max_source_terms, name="max_source_terms"
        ),
        "max_multiplier_bits": _strict_int(
            max_multiplier_bits, name="max_multiplier_bits"
        ),
        "max_exact_bits": _strict_int(
            max_exact_bits, name="max_exact_bits"
        ),
        "max_exact_nonzeros": _strict_int(
            max_exact_nonzeros, name="max_exact_nonzeros"
        ),
    }
    if any(
        value < 1 or value > _HARD_LIMITS[name]
        for name, value in values.items()
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "phase_clique_cap_out_of_range"
        )
    return OperatorPhaseCliqueCaps(**values)


def _caps_payload(caps: Any) -> Mapping[str, int]:
    if type(caps) is not OperatorPhaseCliqueCaps:
        raise OperatorExactReLUPhaseCliqueError(
            "phase_clique_caps_wrong_type"
        )
    normalized = _normalize_caps(
        max_parent_variables=caps.max_parent_variables,
        max_parent_rows=caps.max_parent_rows,
        max_parent_nonzeros=caps.max_parent_nonzeros,
        max_parent_buffer_items=caps.max_parent_buffer_items,
        max_top_literals=caps.max_top_literals,
        max_total_pairs=caps.max_total_pairs,
        max_cliques=caps.max_cliques,
        max_clique_search_nodes=(
            caps.max_clique_search_nodes
        ),
        max_source_terms=caps.max_source_terms,
        max_multiplier_bits=caps.max_multiplier_bits,
        max_exact_bits=caps.max_exact_bits,
        max_exact_nonzeros=caps.max_exact_nonzeros,
    )
    return {
        "max_parent_variables": (
            normalized.max_parent_variables
        ),
        "max_parent_rows": normalized.max_parent_rows,
        "max_parent_nonzeros": (
            normalized.max_parent_nonzeros
        ),
        "max_parent_buffer_items": (
            normalized.max_parent_buffer_items
        ),
        "max_top_literals": normalized.max_top_literals,
        "max_total_pairs": normalized.max_total_pairs,
        "max_cliques": normalized.max_cliques,
        "max_clique_search_nodes": (
            normalized.max_clique_search_nodes
        ),
        "max_source_terms": normalized.max_source_terms,
        "max_multiplier_bits": normalized.max_multiplier_bits,
        "max_exact_bits": normalized.max_exact_bits,
        "max_exact_nonzeros": normalized.max_exact_nonzeros,
    }


def _selection_timeout_value(value: Any) -> float:
    if type(value) not in {int, float} or type(value) is bool:
        raise OperatorExactReLUPhaseCliqueError(
            "selection_timeout_not_builtin_numeric"
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise OperatorExactReLUPhaseCliqueError(
            "selection_timeout_invalid"
        )
    return result


@dataclass(frozen=True)
class _ExactHZCoreLayout:
    dense: Tuple[Tuple[str, np.ndarray], ...]
    sparse: Tuple[
        Tuple[
            str,
            sp.csr_matrix,
            Tuple[int, int],
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        ...,
    ]
    n_out: int
    n_cont: int
    n_bin: int
    n_eq: int
    n_ub: int
    nonzeros: int
    buffer_items: int


@dataclass(frozen=True)
class _OperatorBuildSnapshot:
    build: OperatorHZBuild
    parent_row_tags: Tuple[str, ...]
    continuous_layer_ids: np.ndarray
    full_col_ids: np.ndarray
    input_center: np.ndarray
    input_radius: np.ndarray
    build_input_col_ids: np.ndarray
    source_modes: Tuple[Tuple[str, Any], ...]
    producer_nonempty_seal_verified: bool
    private_parent_semantic_digest: Optional[str]


@dataclass(frozen=True)
class _VerifiedCliqueCore:
    cut_hz: Optional[SparseHZono]
    verified_cliques: Tuple[
        Tuple[str, Tuple[Tuple[int, int, str], ...]],
        ...,
    ]
    parent_row_tags: Tuple[str, ...]
    continuous_layer_ids: np.ndarray
    full_col_ids: np.ndarray
    input_center: np.ndarray
    input_radius: np.ndarray
    build_input_col_ids: np.ndarray
    input_layer_id: int
    output_layer_id: int
    assert_layer_id: int
    original_parent_n_ub: int
    parent_semantic_digest: str
    ordered_source_frame_sha256: Optional[str]
    focused_property_digest: str
    selection_digest: str
    subset_binding_digest: str
    verified_result_digest: str
    caps_payload: Tuple[Tuple[str, int], ...]
    materializer_source_modes: Tuple[Tuple[str, Any], ...]
    producer_nonempty_seal_verified: bool


def _exact_hz_core_layout(
    hz: Any,
) -> _ExactHZCoreLayout:
    """Reject custom hooks before reading shape, nnz, or properties."""

    if type(hz) is not SparseHZono:
        raise OperatorExactReLUPhaseCliqueError(
            "parent_not_exact_sparse_hz_type"
        )
    live = vars(hz)
    if len(live) > _MAX_HZ_INSTANCE_ATTRIBUTES:
        raise OperatorExactReLUPhaseCliqueError(
            "parent_instance_attribute_cap_exceeded"
        )
    dense_specs = (
        ("c", np.dtype(np.float64)),
        ("b", np.dtype(np.float64)),
        ("ub", np.dtype(np.float64)),
        ("col_ids", np.dtype(np.int64)),
        ("bcol_ids", np.dtype(np.int64)),
    )
    dense = []
    for name, dtype in dense_specs:
        value = live.get(name)
        if (
            type(value) is not np.ndarray
            or value.dtype != dtype
            or value.ndim != 1
            or not value.flags.c_contiguous
        ):
            raise OperatorExactReLUPhaseCliqueError(
                f"parent_core_{name}_not_exact_array"
            )
        dense.append((name, value))

    sparse = []
    for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
        matrix = live.get(name)
        if type(matrix) is not sp.csr_matrix:
            raise OperatorExactReLUPhaseCliqueError(
                f"parent_core_{name}_not_exact_csr"
            )
        matrix_vars = vars(matrix)
        shape = matrix_vars.get("_shape")
        data = matrix_vars.get("data")
        indices = matrix_vars.get("indices")
        indptr = matrix_vars.get("indptr")
        if (
            type(shape) is not tuple
            or len(shape) != 2
            or any(
                type(item) is not int or item < 0
                for item in shape
            )
            or type(data) is not np.ndarray
            or data.dtype != np.dtype(np.float64)
            or data.ndim != 1
            or not data.flags.c_contiguous
            or type(indices) is not np.ndarray
            or indices.dtype
            not in {np.dtype(np.int32), np.dtype(np.int64)}
            or indices.ndim != 1
            or not indices.flags.c_contiguous
            or type(indptr) is not np.ndarray
            or indptr.dtype
            not in {np.dtype(np.int32), np.dtype(np.int64)}
            or indptr.ndim != 1
            or not indptr.flags.c_contiguous
        ):
            raise OperatorExactReLUPhaseCliqueError(
                f"parent_core_{name}_buffers_noncanonical"
            )
        sparse.append(
            (
                name,
                matrix,
                shape,
                data,
                indices,
                indptr,
            )
        )

    dense_map = dict(dense)
    sparse_map = {item[0]: item for item in sparse}
    n_out = int(dense_map["c"].size)
    n_cont = int(sparse_map["Gc"][2][1])
    n_bin = int(sparse_map["Gb"][2][1])
    n_eq = int(sparse_map["Ac"][2][0])
    n_ub = int(sparse_map["Auc"][2][0])
    expected_shapes = {
        "Gc": (n_out, n_cont),
        "Gb": (n_out, n_bin),
        "Ac": (n_eq, n_cont),
        "Ab": (n_eq, n_bin),
        "Auc": (n_ub, n_cont),
        "Aub": (n_ub, n_bin),
    }
    if (
        any(
            sparse_map[name][2] != shape
            for name, shape in expected_shapes.items()
        )
        or int(dense_map["b"].size) != n_eq
        or int(dense_map["ub"].size) != n_ub
        or int(dense_map["col_ids"].size) != n_cont
        or int(dense_map["bcol_ids"].size) != n_bin
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "parent_core_dimensions_inconsistent"
        )
    nonzeros = 0
    buffer_items = sum(
        int(value.size) for _, value in dense
    )
    for name, _, shape, data, indices, indptr in sparse:
        rows, _ = shape
        if (
            int(data.size) != int(indices.size)
            or int(indptr.size) != rows + 1
        ):
            raise OperatorExactReLUPhaseCliqueError(
                f"parent_core_{name}_buffer_sizes_invalid"
            )
        nonzeros += int(data.size)
        buffer_items += (
            int(data.size)
            + int(indices.size)
            + int(indptr.size)
        )
    return _ExactHZCoreLayout(
        dense=tuple(dense),
        sparse=tuple(sparse),
        n_out=n_out,
        n_cont=n_cont,
        n_bin=n_bin,
        n_eq=n_eq,
        n_ub=n_ub,
        nonzeros=nonzeros,
        buffer_items=buffer_items,
    )


def _exact_hz_core_identity(
    hz: SparseHZono,
    layout: _ExactHZCoreLayout,
) -> Tuple[int, ...]:
    """Match the producer seal's owner/core identity ordering."""

    if (
        type(hz) is not SparseHZono
        or type(layout) is not _ExactHZCoreLayout
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "operator_seal_core_identity_malformed"
        )
    objects: list[Any] = [hz]
    objects.extend(value for _name, value in layout.dense)
    for (
        _name,
        matrix,
        _shape,
        data,
        indices,
        indptr,
    ) in layout.sparse:
        objects.extend((matrix, data, indices, indptr))
    return tuple(id(value) for value in objects)


def _conditional_metadata_buffer_items(
    hz: SparseHZono,
    *,
    maximum: int,
    deadline: float,
) -> int:
    """Recursively bound everything the semantic digest may traverse."""

    used = [0]
    active: set[int] = set()
    visits = [0]

    def consume(amount: int) -> None:
        if (
            type(amount) is not int
            or amount < 0
            or used[0] + amount > maximum
        ):
            raise OperatorExactReLUPhaseCliqueError(
                "parent_conditional_metadata_cap_exceeded"
            )
        used[0] += amount

    def visit(value: Any, depth: int) -> None:
        visits[0] += 1
        if visits[0] % 256 == 0:
            _check_deadline(
                deadline, stage="conditional_metadata_audit"
            )
        if depth > 128:
            raise OperatorExactReLUPhaseCliqueError(
                "parent_conditional_metadata_depth_exceeded"
            )
        if value is None:
            consume(1)
            return
        if type(value) in {bool, np.bool_}:
            consume(1)
            return
        if type(value) is int or isinstance(value, np.integer):
            integer = int(value)
            decimal_bound = max(
                1,
                (
                    abs(integer).bit_length() * 30103
                    // 100000
                )
                + 2,
            )
            consume(decimal_bound)
            return
        if type(value) is float or isinstance(value, np.floating):
            consume(1)
            return
        if type(value) is str:
            consume(max(1, 4 * len(value)))
            return
        if type(value) is bytes:
            consume(max(1, len(value)))
            return
        if isinstance(value, np.generic):
            consume(1)
            return
        if type(value) is np.ndarray:
            consume(
                1 + int(value.size) + int(value.ndim)
            )
            return
        if type(value) is sp.csr_matrix:
            layout_vars = vars(value)
            data = layout_vars.get("data")
            indices = layout_vars.get("indices")
            indptr = layout_vars.get("indptr")
            if (
                type(data) is not np.ndarray
                or type(indices) is not np.ndarray
                or type(indptr) is not np.ndarray
            ):
                raise OperatorExactReLUPhaseCliqueError(
                    "parent_conditional_metadata_csr_noncanonical"
                )
            consume(
                1
                + int(data.size)
                + int(indices.size)
                + int(indptr.size)
            )
            return
        if type(value) is dict:
            identity = id(value)
            if identity in active:
                raise OperatorExactReLUPhaseCliqueError(
                    "parent_conditional_metadata_cycle"
                )
            length = len(value)
            consume(1 + int(length))
            active.add(identity)
            try:
                for key in value:
                    visit(key, depth + 1)
                    visit(value[key], depth + 1)
            finally:
                active.remove(identity)
            return
        if type(value) in {tuple, list}:
            identity = id(value)
            if identity in active:
                raise OperatorExactReLUPhaseCliqueError(
                    "parent_conditional_metadata_cycle"
                )
            consume(1 + len(value))
            active.add(identity)
            try:
                for item in value:
                    visit(item, depth + 1)
            finally:
                active.remove(identity)
            return
        raise OperatorExactReLUPhaseCliqueError(
            "parent_conditional_metadata_type_unsupported"
        )

    live_vars = vars(hz)
    if len(live_vars) > _MAX_HZ_INSTANCE_ATTRIBUTES:
        raise OperatorExactReLUPhaseCliqueError(
            "parent_instance_attribute_cap_exceeded"
        )
    consume(1 + len(live_vars))
    conditional_names = tuple(
        name
        for name in live_vars
        if "conditional" in name.lower()
    )
    consume(1 + len(conditional_names))
    for name in conditional_names:
        consume(max(1, 4 * len(name)))
        visit(live_vars[name], 0)
    _check_deadline(
        deadline, stage="after_conditional_metadata_audit"
    )
    return used[0]


def _check_parent_size(
    hz: SparseHZono,
    *,
    caps: OperatorPhaseCliqueCaps,
    deadline: float,
) -> _ExactHZCoreLayout:
    """Bound every later parent digest, source replay, and solver build."""

    _check_deadline(deadline, stage="before_parent_size_audit")
    layout = _exact_hz_core_layout(hz)
    variables = layout.n_cont + layout.n_bin
    rows = layout.n_out + layout.n_eq + layout.n_ub
    nonzeros = layout.nonzeros
    buffer_items = layout.buffer_items
    if buffer_items > caps.max_parent_buffer_items:
        raise OperatorExactReLUPhaseCliqueError(
            "parent_size_cap_exceeded"
        )
    buffer_items += _conditional_metadata_buffer_items(
        hz,
        maximum=(
            caps.max_parent_buffer_items - buffer_items
        ),
        deadline=deadline,
    )
    if (
        variables < 0
        or variables > caps.max_parent_variables
        or rows < 0
        or rows > caps.max_parent_rows
        or nonzeros < 0
        or nonzeros > caps.max_parent_nonzeros
        or buffer_items < 0
        or buffer_items > caps.max_parent_buffer_items
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "parent_size_cap_exceeded"
        )
    _check_deadline(deadline, stage="after_parent_size_audit")
    return layout


def _copy_exact_array_with_deadline(
    value: np.ndarray,
    *,
    deadline: float,
    stage: str,
) -> np.ndarray:
    """Copy one already exact-gated array without an unbounded C memcpy."""

    _check_deadline(deadline, stage=f"before_{stage}")
    result = _allocate_owned_exact_array(
        value.shape,
        dtype=value.dtype,
        stage=stage,
    )
    itemsize = max(1, int(value.dtype.itemsize))
    chunk_items = max(1, (1 << 20) // itemsize)
    size = int(value.size)
    for start in range(0, size, chunk_items):
        _check_deadline(deadline, stage=stage)
        stop = min(size, start + chunk_items)
        np.copyto(
            result[start:stop],
            value[start:stop],
            casting="no",
        )
    _check_deadline(deadline, stage=f"after_{stage}")
    return result


def _allocate_owned_exact_array(
    shape: Any,
    *,
    dtype: np.dtype,
    stage: str,
) -> np.ndarray:
    """Single allocation point for an audited, C-contiguous core buffer."""

    if type(stage) is not str or not stage:
        raise OperatorExactReLUPhaseCliqueError(
            "owned_buffer_stage_malformed"
        )
    return np.empty(shape, dtype=dtype, order="C")


def _assemble_owned_sparse_hz_snapshot(
    *,
    dense: Mapping[str, np.ndarray],
    sparse: Mapping[str, sp.csr_matrix],
) -> SparseHZono:
    """Assemble one core from already detached, owned exact buffers.

    ``SparseHZono.__post_init__`` normalizes each CSR via ``astype`` and thus
    copies it even when it is already canonical float64.  The snapshot path
    has just copied every dense/CSR buffer in deadline-polled chunks, so a
    second 100+ MiB copy has no isolation value.  This private constructor is
    immediately followed by the strict core-layout and semantic-digest
    audits; no unvalidated object can escape it.
    """

    if set(dense) != {"c", "b", "ub", "col_ids", "bcol_ids"}:
        raise OperatorExactReLUPhaseCliqueError(
            "owned_snapshot_dense_fields_malformed"
        )
    if set(sparse) != {"Gc", "Gb", "Ac", "Ab", "Auc", "Aub"}:
        raise OperatorExactReLUPhaseCliqueError(
            "owned_snapshot_sparse_fields_malformed"
        )
    result = object.__new__(SparseHZono)
    for name in ("c", "b", "ub", "col_ids", "bcol_ids"):
        object.__setattr__(result, name, dense[name])
    for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
        object.__setattr__(result, name, sparse[name])
    return result


def _snapshot_sparse_hz(
    hz: SparseHZono,
    *,
    caps: OperatorPhaseCliqueCaps,
    deadline: float,
    stage: str,
    prevalidated_layout: Optional[_ExactHZCoreLayout] = None,
    semantic_digest_sink: Optional[list[str]] = None,
) -> SparseHZono:
    """Take one private core snapshot and never retain a live-buffer alias."""

    if prevalidated_layout is None:
        layout = _check_parent_size(
            hz,
            caps=caps,
            deadline=deadline,
        )
    elif type(prevalidated_layout) is _ExactHZCoreLayout:
        layout = prevalidated_layout
    else:
        raise OperatorExactReLUPhaseCliqueError(
            "prevalidated_hz_layout_malformed"
        )
    if (
        semantic_digest_sink is not None
        and type(semantic_digest_sink) is not list
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "semantic_digest_sink_malformed"
        )
    live_vars = vars(hz)
    if len(live_vars) > _MAX_HZ_INSTANCE_ATTRIBUTES:
        raise OperatorExactReLUPhaseCliqueError(
            "parent_instance_attribute_cap_exceeded"
        )
    if any(
        "conditional" in name.lower() for name in live_vars
    ):
        # The first production materializer has no conditional-metadata
        # replay.  Failing closed also avoids copying opaque process-local
        # seals into a capability.
        raise OperatorExactReLUPhaseCliqueError(
            "parent_conditional_metadata_unsupported"
        )
    dense = {
        name: _copy_exact_array_with_deadline(
            value,
            deadline=deadline,
            stage=f"{stage}_{name}_copy",
        )
        for name, value in layout.dense
    }
    sparse: dict[str, sp.csr_matrix] = {}
    for name, _matrix, shape, data, indices, indptr in layout.sparse:
        copied_data = _copy_exact_array_with_deadline(
            data,
            deadline=deadline,
            stage=f"{stage}_{name}_data_copy",
        )
        copied_indices = _copy_exact_array_with_deadline(
            indices,
            deadline=deadline,
            stage=f"{stage}_{name}_indices_copy",
        )
        copied_indptr = _copy_exact_array_with_deadline(
            indptr,
            deadline=deadline,
            stage=f"{stage}_{name}_indptr_copy",
        )
        sparse[name] = sp.csr_matrix(
            (copied_data, copied_indices, copied_indptr),
            shape=shape,
            dtype=np.float64,
            copy=False,
        )
    _check_deadline(deadline, stage=f"before_{stage}_construction")
    result = _assemble_owned_sparse_hz_snapshot(
        dense=dense,
        sparse=sparse,
    )
    _check_parent_size(
        result,
        caps=caps,
        deadline=deadline,
    )
    # This recomputes CSR canonicality, finite values, stable-ID uniqueness,
    # and every semantic byte on the private buffers.
    private_digest = sparse_hz_semantic_digest(result)
    if semantic_digest_sink is not None:
        semantic_digest_sink.append(private_digest)
    _check_deadline(deadline, stage=f"after_{stage}_semantic_audit")
    return result


def _copy_exact_array_with_tail_deadline(
    value: np.ndarray,
    tail: np.ndarray,
    *,
    deadline: float,
    stage: str,
) -> np.ndarray:
    """Copy an exact one-dimensional buffer and append a bounded tail."""

    if (
        type(value) is not np.ndarray
        or type(tail) is not np.ndarray
        or value.dtype != tail.dtype
        or value.ndim != 1
        or tail.ndim != 1
        or not value.flags.c_contiguous
        or not tail.flags.c_contiguous
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "owned_tail_buffer_malformed"
        )
    _check_deadline(deadline, stage=f"before_{stage}")
    result = _allocate_owned_exact_array(
        (int(value.size) + int(tail.size),),
        dtype=value.dtype,
        stage=stage,
    )
    itemsize = max(1, int(value.dtype.itemsize))
    chunk_items = max(1, (1 << 20) // itemsize)
    for start in range(0, int(value.size), chunk_items):
        _check_deadline(deadline, stage=stage)
        stop = min(int(value.size), start + chunk_items)
        np.copyto(
            result[start:stop],
            value[start:stop],
            casting="no",
        )
    offset = int(value.size)
    for start in range(0, int(tail.size), chunk_items):
        _check_deadline(deadline, stage=stage)
        stop = min(int(tail.size), start + chunk_items)
        np.copyto(
            result[offset + start : offset + stop],
            tail[start:stop],
            casting="no",
        )
    _check_deadline(deadline, stage=f"after_{stage}")
    return result


def _owned_csr_from_exact_buffers(
    *,
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: Tuple[int, int],
    stage: str,
) -> sp.csr_matrix:
    """Wrap owned exact CSR buffers without normalizing or copying them."""

    result = sp.csr_matrix(
        (data, indices, indptr),
        shape=shape,
        dtype=np.float64,
        copy=False,
    )
    for expected, actual in (
        (data, result.data),
        (indices, result.indices),
        (indptr, result.indptr),
    ):
        if int(expected.size) and not np.shares_memory(
            expected, actual
        ):
            raise OperatorExactReLUPhaseCliqueError(
                f"owned_csr_buffer_reallocated_{stage}"
            )
    return result


def _copy_parent_with_clique_cuts(
    hz: SparseHZono,
    clique_literals: Tuple[Tuple[PhaseLiteral, ...], ...],
    *,
    caps: OperatorPhaseCliqueCaps,
    deadline: float,
) -> SparseHZono:
    """Append all clique rows in one detached, single-copy HZ core.

    The legacy constructor first created a complete sparse cut core with
    SciPy ``vstack`` and then copied all six CSR matrices again in
    ``SparseHZono.__post_init__``.  Building directly from owned buffers keeps
    the live peak to the parent plus one result core.  Multiple cliques are
    appended together so an earlier full cut copy is never retained while a
    later one is constructed.
    """

    deadline_value = _normalize_deadline(deadline)
    _check_deadline(deadline_value, stage="before_owned_clique_cut")
    if (
        type(hz) is not SparseHZono
        or type(caps) is not OperatorPhaseCliqueCaps
        or type(clique_literals) is not tuple
        or not clique_literals
        or len(clique_literals) > caps.max_cliques
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "owned_clique_cut_input_malformed"
        )
    # Re-normalize every cap before any allocation.  This rejects forged
    # dataclass payloads as well as values outside the hard implementation
    # limits.
    _caps_payload(caps)
    layout = _check_parent_size(
        hz,
        caps=caps,
        deadline=deadline_value,
    )
    parent_digest = sparse_hz_semantic_digest(hz)
    _check_deadline(
        deadline_value, stage="after_owned_clique_parent_digest"
    )

    bcol_ids = dict(layout.dense)["bcol_ids"]
    positions = {
        int(stable_id): position
        for position, stable_id in enumerate(bcol_ids.tolist())
    }
    normalized_rows: list[Tuple[Tuple[int, float], ...]] = []
    seen_rows: set[Tuple[Tuple[int, int, str], ...]] = set()
    total_cut_nonzeros = 0
    for row_number, literals in enumerate(clique_literals):
        _check_deadline(
            deadline_value,
            stage=f"owned_clique_cut_row_{row_number}",
        )
        if (
            type(literals) is not tuple
            or len(literals) < 2
            or len(literals) > caps.max_top_literals
        ):
            raise OperatorExactReLUPhaseCliqueError(
                "owned_clique_literal_row_malformed"
            )
        row_positions: list[Tuple[int, float]] = []
        row_binding: list[Tuple[int, int, str]] = []
        seen_stable_ids: set[int] = set()
        for literal in literals:
            if type(literal) is not PhaseLiteral:
                raise OperatorExactReLUPhaseCliqueError(
                    "owned_clique_literal_wrong_type"
                )
            stable_id = literal.stable_bcol_id
            phase = literal.phase
            if (
                type(stable_id) is not int
                or stable_id < 0
                or type(phase) is not int
                or phase not in {-1, 1}
                or not _valid_sha256(literal.binding_digest)
                or stable_id in seen_stable_ids
                or stable_id not in positions
            ):
                raise OperatorExactReLUPhaseCliqueError(
                    "owned_clique_literal_invalid"
                )
            seen_stable_ids.add(stable_id)
            row_positions.append(
                (positions[stable_id], float(phase))
            )
            row_binding.append(
                (stable_id, phase, literal.binding_digest)
            )
        binding = tuple(row_binding)
        if binding in seen_rows:
            raise OperatorExactReLUPhaseCliqueError(
                "owned_clique_row_duplicate"
            )
        seen_rows.add(binding)
        normalized_rows.append(
            tuple(sorted(row_positions, key=lambda item: item[0]))
        )
        total_cut_nonzeros += len(row_positions)

    added_rows = len(normalized_rows)
    metadata_items = _conditional_metadata_buffer_items(
        hz,
        maximum=(
            caps.max_parent_buffer_items - layout.buffer_items
        ),
        deadline=deadline_value,
    )
    # Per added row: one Auc indptr, one Aub indptr, and one ub item.
    # Per literal: one Aub index and one Aub data item.
    output_buffer_items = (
        layout.buffer_items
        + metadata_items
        + 3 * added_rows
        + 2 * total_cut_nonzeros
    )
    if (
        layout.n_out + layout.n_eq + layout.n_ub + added_rows
        > caps.max_parent_rows
        or layout.nonzeros + total_cut_nonzeros
        > caps.max_parent_nonzeros
        or output_buffer_items > caps.max_parent_buffer_items
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "owned_clique_cut_output_cap_exceeded"
        )
    _check_deadline(
        deadline_value, stage="before_owned_clique_allocations"
    )

    dense_source = dict(layout.dense)
    dense = {
        name: _copy_exact_array_with_deadline(
            dense_source[name],
            deadline=deadline_value,
            stage=f"owned_clique_{name}_copy",
        )
        for name in ("c", "b", "col_ids", "bcol_ids")
    }
    rhs_tail = np.asarray(
        [2 - len(literals) for literals in clique_literals],
        dtype=np.float64,
    )
    dense["ub"] = _copy_exact_array_with_tail_deadline(
        dense_source["ub"],
        rhs_tail,
        deadline=deadline_value,
        stage="owned_clique_ub_append",
    )

    sparse_source = {item[0]: item for item in layout.sparse}
    sparse: dict[str, sp.csr_matrix] = {}
    for name in ("Gc", "Gb", "Ac", "Ab"):
        _, _matrix, shape, data, indices, indptr = sparse_source[
            name
        ]
        owned_data = _copy_exact_array_with_deadline(
            data,
            deadline=deadline_value,
            stage=f"owned_clique_{name}_data_copy",
        )
        owned_indices = _copy_exact_array_with_deadline(
            indices,
            deadline=deadline_value,
            stage=f"owned_clique_{name}_indices_copy",
        )
        owned_indptr = _copy_exact_array_with_deadline(
            indptr,
            deadline=deadline_value,
            stage=f"owned_clique_{name}_indptr_copy",
        )
        sparse[name] = _owned_csr_from_exact_buffers(
            data=owned_data,
            indices=owned_indices,
            indptr=owned_indptr,
            shape=shape,
            stage=name,
        )

    (
        _,
        _auc_matrix,
        auc_shape,
        auc_data,
        auc_indices,
        auc_indptr,
    ) = sparse_source["Auc"]
    owned_auc_data = _copy_exact_array_with_deadline(
        auc_data,
        deadline=deadline_value,
        stage="owned_clique_Auc_data_copy",
    )
    owned_auc_indices = _copy_exact_array_with_deadline(
        auc_indices,
        deadline=deadline_value,
        stage="owned_clique_Auc_indices_copy",
    )
    auc_indptr_tail = np.full(
        added_rows,
        int(auc_indptr[-1]),
        dtype=auc_indptr.dtype,
    )
    owned_auc_indptr = _copy_exact_array_with_tail_deadline(
        auc_indptr,
        auc_indptr_tail,
        deadline=deadline_value,
        stage="owned_clique_Auc_indptr_append",
    )
    sparse["Auc"] = _owned_csr_from_exact_buffers(
        data=owned_auc_data,
        indices=owned_auc_indices,
        indptr=owned_auc_indptr,
        shape=(auc_shape[0] + added_rows, auc_shape[1]),
        stage="Auc",
    )

    (
        _,
        _aub_matrix,
        aub_shape,
        aub_data,
        aub_indices,
        aub_indptr,
    ) = sparse_source["Aub"]
    aub_data_tail = np.asarray(
        [phase for row in normalized_rows for _position, phase in row],
        dtype=np.float64,
    )
    aub_indices_tail = np.asarray(
        [
            position
            for row in normalized_rows
            for position, _phase in row
        ],
        dtype=aub_indices.dtype,
    )
    running_nnz = int(aub_indptr[-1])
    appended_indptr = []
    for row in normalized_rows:
        running_nnz += len(row)
        appended_indptr.append(running_nnz)
    aub_indptr_tail = np.asarray(
        appended_indptr,
        dtype=aub_indptr.dtype,
    )
    owned_aub_data = _copy_exact_array_with_tail_deadline(
        aub_data,
        aub_data_tail,
        deadline=deadline_value,
        stage="owned_clique_Aub_data_append",
    )
    owned_aub_indices = _copy_exact_array_with_tail_deadline(
        aub_indices,
        aub_indices_tail,
        deadline=deadline_value,
        stage="owned_clique_Aub_indices_append",
    )
    owned_aub_indptr = _copy_exact_array_with_tail_deadline(
        aub_indptr,
        aub_indptr_tail,
        deadline=deadline_value,
        stage="owned_clique_Aub_indptr_append",
    )
    sparse["Aub"] = _owned_csr_from_exact_buffers(
        data=owned_aub_data,
        indices=owned_aub_indices,
        indptr=owned_aub_indptr,
        shape=(aub_shape[0] + added_rows, aub_shape[1]),
        stage="Aub",
    )

    _check_deadline(
        deadline_value, stage="before_owned_clique_construction"
    )
    result = _assemble_owned_sparse_hz_snapshot(
        dense=dense,
        sparse=sparse,
    )
    for name, value in vars(hz).items():
        if "conditional" in name.lower():
            setattr(result, name, value)
    _check_parent_size(
        result,
        caps=caps,
        deadline=deadline_value,
    )
    # This is the strict independent audit for shape, CSR canonicality,
    # finite values, stable-ID uniqueness, conditional metadata, and every
    # semantic byte of the final cut core.
    sparse_hz_semantic_digest(result)
    if sparse_hz_semantic_digest(hz) != parent_digest:
        raise OperatorExactReLUPhaseCliqueError(
            "owned_clique_parent_mutated_during_copy"
        )
    for name in ("c", "b", "ub", "col_ids", "bcol_ids"):
        if np.shares_memory(getattr(hz, name), getattr(result, name)):
            raise OperatorExactReLUPhaseCliqueError(
                "owned_clique_dense_alias"
            )
    for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
        source_matrix = getattr(hz, name)
        result_matrix = getattr(result, name)
        for buffer_name in ("data", "indices", "indptr"):
            if np.shares_memory(
                getattr(source_matrix, buffer_name),
                getattr(result_matrix, buffer_name),
            ):
                raise OperatorExactReLUPhaseCliqueError(
                    "owned_clique_sparse_alias"
                )
    _check_deadline(
        deadline_value, stage="after_owned_clique_semantic_audit"
    )
    return result


def _copy_parent_with_clique_cut(
    hz: SparseHZono,
    literals: Tuple[PhaseLiteral, ...],
    *,
    caps: OperatorPhaseCliqueCaps,
    deadline: float,
) -> SparseHZono:
    """Single-row compatibility wrapper over the bulk owned constructor."""

    return _copy_parent_with_clique_cuts(
        hz,
        (literals,),
        caps=caps,
        deadline=deadline,
    )


def _snapshot_exact_vector(
    value: Any,
    *,
    dtype: np.dtype,
    expected_length: int,
    name: str,
    deadline: float,
) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(dtype)
        or value.ndim != 1
        or int(value.size) != expected_length
        or not value.flags.c_contiguous
    ):
        raise OperatorExactReLUPhaseCliqueError(
            f"operator_snapshot_{name}_malformed"
        )
    result = _copy_exact_array_with_deadline(
        value,
        deadline=deadline,
        stage=f"operator_snapshot_{name}",
    )
    if (
        np.issubdtype(result.dtype, np.floating)
        and not np.all(np.isfinite(result))
    ):
        raise OperatorExactReLUPhaseCliqueError(
            f"operator_snapshot_{name}_nonfinite"
        )
    return result


def _require_exact_string_dict_keys(
    value: Any,
    *,
    maximum_keys: int,
    maximum_key_bytes: int,
    deadline: float,
    stage: str,
) -> dict[Any, Any]:
    if type(value) is not dict or len(value) > maximum_keys:
        raise OperatorExactReLUPhaseCliqueError(
            f"{stage}_mapping_malformed"
        )
    for index, key in enumerate(value):
        if index % 256 == 0:
            _check_deadline(deadline, stage=stage)
        if (
            type(key) is not str
            or len(key) > maximum_key_bytes
            or len(key.encode("utf-8")) > maximum_key_bytes
        ):
            raise OperatorExactReLUPhaseCliqueError(
                f"{stage}_mapping_key_malformed"
            )
    _check_deadline(deadline, stage=f"after_{stage}")
    return value


def _snapshot_operator_build(
    build: OperatorHZBuild,
    *,
    caps: OperatorPhaseCliqueCaps,
    deadline: float,
    require_materializer_source: bool,
) -> _OperatorBuildSnapshot:
    """Snapshot the core, tags, provenance, and closed source modes."""

    _check_deadline(deadline, stage="before_operator_build_snapshot")
    if type(build) is not OperatorHZBuild:
        raise OperatorExactReLUPhaseCliqueError(
            "operator_phase_clique_build_wrong_type"
        )
    build_vars = vars(build)
    live_hz = build_vars.get("hz")
    if type(live_hz) is not SparseHZono:
        raise OperatorExactReLUPhaseCliqueError(
            "operator_phase_clique_parent_wrong_type"
        )
    # Gate the exact core before consulting any live HZ property, then reduce
    # the process-local construction token to an immutable snapshot primitive.
    layout = _check_parent_size(
        live_hz,
        caps=caps,
        deadline=deadline,
    )
    producer_seal = build_vars.get(
        "constructive_nonempty_seal"
    )
    producer_core_identity = None
    if require_materializer_source and producer_seal is not None:
        producer_core_identity = _exact_hz_core_identity(
            live_hz, layout
        )
    source_constructively_nonempty = bool(
        hz_constructively_nonempty(live_hz)
    )
    live_hz_vars = vars(live_hz)

    raw_tags = live_hz_vars.get("_solver_constraint_row_tags")
    if (
        type(raw_tags) is not tuple
        or len(raw_tags) != layout.n_eq + layout.n_ub
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "operator_snapshot_row_tags_malformed"
        )
    tags = []
    tag_bytes = 0
    for index, tag in enumerate(raw_tags):
        if index % 256 == 0:
            _check_deadline(deadline, stage="operator_row_tag_copy")
        if type(tag) is not str or "\x00" in tag:
            raise OperatorExactReLUPhaseCliqueError(
                "operator_snapshot_row_tags_malformed"
            )
        try:
            encoded = tag.encode("ascii")
        except UnicodeEncodeError as exc:
            raise OperatorExactReLUPhaseCliqueError(
                "operator_snapshot_row_tag_nonascii"
            ) from exc
        if len(encoded) > 4096:
            raise OperatorExactReLUPhaseCliqueError(
                "operator_snapshot_row_tag_too_large"
            )
        tag_bytes += len(encoded)
        if tag_bytes > caps.max_parent_buffer_items:
            raise OperatorExactReLUPhaseCliqueError(
                "operator_snapshot_row_tag_cap_exceeded"
            )
        tags.append(tag)
    parent_row_tags = tuple(tags)
    if any(
        tag.startswith("property_micro_rlt:")
        for tag in parent_row_tags
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "operator_snapshot_micro_rlt_rows_unsupported"
        )

    raw_build_input_ids = build_vars.get("input_col_ids")
    if (
        type(raw_build_input_ids) is not np.ndarray
        or raw_build_input_ids.dtype != np.dtype(np.int64)
        or raw_build_input_ids.ndim != 1
        or not raw_build_input_ids.flags.c_contiguous
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "operator_snapshot_build_input_col_ids_malformed"
        )
    input_length = int(raw_build_input_ids.size)
    required_provenance_items = (
        layout.n_cont + 4 * input_length
        if require_materializer_source
        else input_length
    )
    if (
        layout.buffer_items
        + tag_bytes
        + required_provenance_items
        > caps.max_parent_buffer_items
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "operator_snapshot_provenance_cap_exceeded"
        )
    build_input_ids = _snapshot_exact_vector(
        raw_build_input_ids,
        dtype=np.int64,
        expected_length=input_length,
        name="build_input_col_ids",
        deadline=deadline,
    )
    layer_ids = []
    for name in (
        "input_layer_id",
        "output_layer_id",
        "assert_layer_id",
    ):
        value = build_vars.get(name)
        if type(value) is not int or value < 0:
            raise OperatorExactReLUPhaseCliqueError(
                f"operator_snapshot_{name}_malformed"
            )
        layer_ids.append(value)
    if build_vars.get("property_upper_output") is not False:
        raise OperatorExactReLUPhaseCliqueError(
            "operator_snapshot_property_upper_unsupported"
        )
    if not require_materializer_source:
        hz = _snapshot_sparse_hz(
            live_hz,
            caps=caps,
            deadline=deadline,
            stage="parent",
            prevalidated_layout=layout,
        )
        private_layout = _exact_hz_core_layout(hz)
        if (
            private_layout.n_out,
            private_layout.n_cont,
            private_layout.n_bin,
            private_layout.n_eq,
            private_layout.n_ub,
        ) != (
            layout.n_out,
            layout.n_cont,
            layout.n_bin,
            layout.n_eq,
            layout.n_ub,
        ):
            raise OperatorExactReLUPhaseCliqueError(
                "operator_snapshot_core_frame_dimension_race"
            )
        setattr(hz, "_solver_constraint_row_tags", parent_row_tags)
        snapshot_build = OperatorHZBuild(
            hz=hz,
            input_col_ids=build_input_ids,
            input_layer_id=layer_ids[0],
            output_layer_id=layer_ids[1],
            assert_layer_id=layer_ids[2],
            metadata={},
            property_upper_output=False,
            property_upper_row_groups=(),
            verified_preactivation_frame=None,
        )
        empty_i64 = np.zeros(0, dtype=np.int64)
        empty_f64 = np.zeros(0, dtype=np.float64)
        return _OperatorBuildSnapshot(
            build=snapshot_build,
            parent_row_tags=parent_row_tags,
            continuous_layer_ids=empty_i64,
            full_col_ids=empty_i64.copy(),
            input_center=empty_f64,
            input_radius=empty_f64.copy(),
            build_input_col_ids=build_input_ids,
            source_modes=(),
            producer_nonempty_seal_verified=False,
            private_parent_semantic_digest=None,
        )
    continuous_layer_ids = _snapshot_exact_vector(
        live_hz_vars.get("_solver_continuous_column_layer_ids"),
        dtype=np.int64,
        expected_length=layout.n_cont,
        name="continuous_layer_ids",
        deadline=deadline,
    )
    full_col_ids = _snapshot_exact_vector(
        live_hz_vars.get("full_col_ids"),
        dtype=np.int64,
        expected_length=input_length,
        name="full_col_ids",
        deadline=deadline,
    )
    input_center = _snapshot_exact_vector(
        live_hz_vars.get("operator_input_center"),
        dtype=np.float64,
        expected_length=input_length,
        name="input_center",
        deadline=deadline,
    )
    input_radius = _snapshot_exact_vector(
        live_hz_vars.get("operator_input_radius"),
        dtype=np.float64,
        expected_length=input_length,
        name="input_radius",
        deadline=deadline,
    )
    if (
        not np.array_equal(full_col_ids, build_input_ids)
        or np.any(full_col_ids < 0)
        or len(set(int(item) for item in full_col_ids)) != input_length
        or np.any(input_radius < 0.0)
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "operator_snapshot_input_provenance_invalid"
        )
    provenance_items = (
        int(continuous_layer_ids.size)
        + int(full_col_ids.size)
        + int(input_center.size)
        + int(input_radius.size)
        + int(build_input_ids.size)
    )
    if (
        layout.buffer_items + provenance_items + tag_bytes
        > caps.max_parent_buffer_items
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "operator_snapshot_provenance_cap_exceeded"
        )

    if (
        type(build_vars.get("property_upper_row_groups")) is not tuple
        or build_vars.get("property_upper_row_groups")
        or build_vars.get("verified_preactivation_frame") is not None
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "operator_snapshot_property_mode_unsupported"
        )
    if any(
        "conditional" in name.lower() for name in live_hz_vars
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "operator_snapshot_conditional_metadata_unsupported"
        )
    prefix_frames = live_hz_vars.get(
        "_solver_row_constraint_prefix_frames"
    )
    if type(prefix_frames) is not dict or prefix_frames:
        raise OperatorExactReLUPhaseCliqueError(
            "operator_snapshot_row_prefix_frames_not_empty"
        )
    if (
        live_hz_vars.get("_property_full_input_replay_result")
        is not None
        or "_property_micro_rlt_receipt" in live_hz_vars
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "operator_snapshot_parent_receipt_unsupported"
        )
    metadata = _require_exact_string_dict_keys(
        build_vars.get("metadata"),
        maximum_keys=65536,
        maximum_key_bytes=4096,
        deadline=deadline,
        stage="operator_snapshot_metadata",
    )
    if any(
        key.startswith("verified_query_dual") for key in metadata
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "operator_snapshot_query_dual_metadata_unsupported"
        )
    micro_rlt = _require_exact_string_dict_keys(
        metadata.get("property_micro_rlt"),
        maximum_keys=256,
        maximum_key_bytes=4096,
        deadline=deadline,
        stage="operator_snapshot_micro_rlt_metadata",
    )
    if (
        type(micro_rlt) is not dict
        or type(micro_rlt.get("schema")) is not str
        or micro_rlt.get("schema")
        != "operator_hz_property_micro_rlt_v1"
        or type(micro_rlt.get("enabled")) is not bool
        or micro_rlt.get("enabled") is not False
        or type(micro_rlt.get("status")) is not str
        or micro_rlt.get("status") != "no_op_disabled"
        or type(micro_rlt.get("proof_authority")) is not bool
        or micro_rlt.get("proof_authority") is not False
        or type(micro_rlt.get("live_result_validation_passed"))
        is not bool
        or micro_rlt.get("live_result_validation_passed") is not False
        or micro_rlt.get("property_micro_rlt_receipt_sha256")
        is not None
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "operator_snapshot_micro_rlt_metadata_not_closed"
        )
    property_tail = _require_exact_string_dict_keys(
        metadata.get("property_tail_upper"),
        maximum_keys=256,
        maximum_key_bytes=4096,
        deadline=deadline,
        stage="operator_snapshot_property_tail_metadata",
    )
    if (
        type(property_tail) is not dict
        or type(property_tail.get("schema")) is not str
        or property_tail.get("schema")
        != "operator_hz_property_tail_fraction_v1"
        or type(property_tail.get("enabled")) is not bool
        or property_tail.get("enabled") is not False
        or type(property_tail.get("proof_authority")) is not bool
        or property_tail.get("proof_authority") is not False
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "operator_snapshot_property_tail_metadata_not_closed"
        )
    if not source_constructively_nonempty:
        raise OperatorExactReLUPhaseCliqueError(
            "operator_snapshot_source_not_constructively_nonempty"
        )
    source_modes = tuple(
        sorted(
            {
                "conditional_metadata_absent": True,
                "full_input_replay_absent": True,
                "micro_rlt_metadata_closed": True,
                "micro_rlt_parent_receipt_absent": True,
                "property_tail_metadata_closed": True,
                "property_upper_output": False,
                "property_upper_row_groups_empty": True,
                "row_prefix_frames_empty": True,
                "source_constructively_nonempty": True,
                "verified_preactivation_frame_absent": True,
                "verified_query_dual_metadata_absent": True,
            }.items()
        )
    )

    private_digest_sink: Optional[list[str]] = (
        [] if producer_seal is not None else None
    )
    hz = _snapshot_sparse_hz(
        live_hz,
        caps=caps,
        deadline=deadline,
        stage="parent",
        prevalidated_layout=layout,
        semantic_digest_sink=private_digest_sink,
    )
    private_layout = _exact_hz_core_layout(hz)
    if (
        private_layout.n_out,
        private_layout.n_cont,
        private_layout.n_bin,
        private_layout.n_eq,
        private_layout.n_ub,
    ) != (
        layout.n_out,
        layout.n_cont,
        layout.n_bin,
        layout.n_eq,
        layout.n_ub,
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "operator_snapshot_core_frame_dimension_race"
        )
    producer_nonempty_seal_verified = False
    private_parent_semantic_digest = None
    if producer_seal is not None:
        if (
            private_digest_sink is None
            or len(private_digest_sink) != 1
            or producer_core_identity is None
        ):
            raise OperatorExactReLUPhaseCliqueError(
                "operator_snapshot_seal_digest_missing"
            )
        private_parent_semantic_digest = (
            private_digest_sink[0]
        )
        producer_nonempty_seal_verified = (
            validate_operator_hz_constructive_nonempty_seal(
                producer_seal,
                owner_build=build,
                owner_hz=live_hz,
                owner_core_identity=producer_core_identity,
                private_parent_semantic_digest=(
                    private_parent_semantic_digest
                ),
            )
        )
        if not producer_nonempty_seal_verified:
            raise OperatorExactReLUPhaseCliqueError(
                "operator_snapshot_constructive_seal_invalid"
            )
    setattr(hz, "_solver_constraint_row_tags", parent_row_tags)
    snapshot_build = OperatorHZBuild(
        hz=hz,
        input_col_ids=build_input_ids,
        input_layer_id=layer_ids[0],
        output_layer_id=layer_ids[1],
        assert_layer_id=layer_ids[2],
        metadata={},
        property_upper_output=False,
        property_upper_row_groups=(),
        verified_preactivation_frame=None,
    )
    _check_deadline(deadline, stage="after_operator_build_snapshot")
    return _OperatorBuildSnapshot(
        build=snapshot_build,
        parent_row_tags=parent_row_tags,
        continuous_layer_ids=continuous_layer_ids,
        full_col_ids=full_col_ids,
        input_center=input_center,
        input_radius=input_radius,
        build_input_col_ids=build_input_ids,
        source_modes=source_modes,
        producer_nonempty_seal_verified=(
            producer_nonempty_seal_verified
        ),
        private_parent_semantic_digest=(
            private_parent_semantic_digest
        ),
    )


def _snapshot_rivals(
    focused_rivals: Sequence[RivalSpec],
    *,
    output_width: int,
    maximum: int,
    deadline: float,
) -> Tuple[RivalSpec, ...]:
    if (
        type(focused_rivals) is not tuple
        or not focused_rivals
        or len(focused_rivals) > maximum
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "focused_rivals_noncanonical"
        )
    result = []
    for index, rival in enumerate(focused_rivals):
        if index % 64 == 0:
            _check_deadline(deadline, stage="focused_rival_snapshot")
        if type(rival) is not RivalSpec:
            raise OperatorExactReLUPhaseCliqueError(
                "focused_rival_wrong_type"
            )
        rival_vars = vars(rival)
        rival_id = rival_vars.get("rival_id")
        objective = rival_vars.get("objective")
        threshold = rival_vars.get("threshold")
        assert_digest = rival_vars.get("assert_digest")
        if (
            type(rival_id) is not int
            or rival_id < 0
            or rival_id > _MAX_INT64
            or type(objective) is not tuple
            or len(objective) != output_width
            or any(type(value) is not float for value in objective)
            or any(not math.isfinite(value) for value in objective)
            or type(threshold) is not float
            or not math.isfinite(threshold)
            or not _valid_sha256(assert_digest)
        ):
            raise OperatorExactReLUPhaseCliqueError(
                "focused_rival_noncanonical"
            )
        result.append(
            RivalSpec(
                rival_id=rival_id,
                objective=tuple(float(value) for value in objective),
                threshold=float(threshold),
                assert_digest=assert_digest,
            )
        )
    return tuple(result)


def _verify_live_selection(
    build: OperatorHZBuild,
    focused_rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    *,
    deadline: float,
    selection_max_rivals: int,
    selection_max_binaries: int,
    selection_max_work_items: int,
    selection_timeout_seconds: float,
) -> OperatorExactReLUPhaseSelection:
    """Re-derive once on the private parent and bind the caller digest.

    The old path called the public verifier (which itself performs a full
    trusted re-derivation) and then derived the same selection a second time.
    The candidate never consumes caller-owned mapping fields: the sole live
    input binding is the canonical selection SHA, and all later work uses the
    freshly derived private object.  One re-derivation is therefore both
    sufficient and strictly less exposed to a candidate-time deadline.
    """

    timeout = _selection_timeout_value(
        selection_timeout_seconds
    )
    _check_deadline(deadline, stage="before_selection_audit")
    if deadline - time.monotonic() < timeout:
        raise OperatorExactReLUPhaseCliqueError(
            "deadline_cannot_cover_selection_audit"
        )
    _check_deadline(deadline, stage="before_trusted_selection_derivation")
    trusted = derive_operator_exact_relu_property_phase_literals(
        build,
        focused_rivals,
        max_rivals=selection_max_rivals,
        max_binaries=selection_max_binaries,
        max_work_items=selection_max_work_items,
        timeout_seconds=selection_timeout_seconds,
    )
    candidate_vars = vars(selection)
    candidate_digest = candidate_vars.get("selection_digest")
    if (
        type(candidate_digest) is not str
        or not _valid_sha256(candidate_digest)
        or candidate_digest != trusted.selection_digest
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "operator_phase_selection_digest_mismatch"
        )
    _check_deadline(deadline, stage="after_selection_audit")
    return trusted


def _mapping_score(
    mapping: OperatorExactReLUPhaseMapping,
) -> Fraction:
    """Minimum exact effect; the live selection already checked its shape."""

    values = tuple(
        abs(Fraction(item.numerator, item.denominator))
        for item in mapping.rival_coefficients
    )
    if not values or mapping.selected_phase not in {-1, 1}:
        raise OperatorExactReLUPhaseCliqueError(
            "selected_mapping_score_invalid"
        )
    score = min(values)
    if score <= 0:
        raise OperatorExactReLUPhaseCliqueError(
            "selected_mapping_score_not_positive"
        )
    return score


def _ranked_subset(
    selection: OperatorExactReLUPhaseSelection,
    *,
    caps: OperatorPhaseCliqueCaps,
    deadline: float,
) -> Tuple[
    Tuple[RankedOperatorPhase, ...],
    Tuple[int, ...],
    Tuple[int, ...],
]:
    selected = []
    omitted = []
    for index, mapping in enumerate(selection.mappings):
        if index % 256 == 0:
            _check_deadline(
                deadline, stage="focused_mapping_ranking"
            )
        if mapping.selected_phase is None:
            omitted.append(mapping.stable_bcol_id)
        else:
            score = _mapping_score(mapping)
            if (
                score.numerator.bit_length()
                > caps.max_exact_bits
                or score.denominator.bit_length()
                > caps.max_exact_bits
            ):
                raise OperatorExactReLUPhaseCliqueError(
                    "ranking_score_exact_bit_cap_exceeded"
                )
            selected.append((mapping, score))
    selected.sort(
        key=lambda item: (
            -item[1],
            item[0].stable_bcol_id,
        )
    )
    chosen = selected[: caps.max_top_literals]
    excluded = tuple(
        item[0].stable_bcol_id
        for item in selected[caps.max_top_literals :]
    )
    pair_count = len(chosen) * (len(chosen) - 1) // 2
    if pair_count > caps.max_total_pairs:
        raise OperatorExactReLUPhaseCliqueError(
            "top_k_pair_cap_exceeded"
        )
    _check_deadline(deadline, stage="after_focused_ranking")
    ranked = tuple(
        RankedOperatorPhase(
            rank=rank,
            stable_bcol_id=mapping.stable_bcol_id,
            phase=int(mapping.selected_phase),
            score_numerator=score.numerator,
            score_denominator=score.denominator,
        )
        for rank, (mapping, score) in enumerate(chosen)
    )
    return ranked, tuple(omitted), excluded


def _ranked_payload(
    ranked: Sequence[RankedOperatorPhase],
) -> Tuple[Tuple[int, int, int, int, int], ...]:
    return tuple(
        (
            item.rank,
            item.stable_bcol_id,
            item.phase,
            item.score_numerator,
            item.score_denominator,
        )
        for item in ranked
    )


def _subset_binding_digest(
    *,
    selection: OperatorExactReLUPhaseSelection,
    caps: OperatorPhaseCliqueCaps,
    ranked: Tuple[RankedOperatorPhase, ...],
    omitted_zero_bcol_ids: Tuple[int, ...],
    excluded_selected_bcol_ids: Tuple[int, ...],
    deadline: float,
) -> str:
    _check_deadline(deadline, stage="before_subset_binding")
    result = _canonical_sha256(
        {
            "schema": (
                "act.operator_exact_relu_phase_clique_subset.v1"
            ),
            "parent_semantic_digest": (
                selection.parent_semantic_digest
            ),
            "property_digest": selection.property_digest,
            "operator_row_tag_digest": (
                selection.operator_row_tag_digest
            ),
            "selection_digest": selection.selection_digest,
            "caps": _caps_payload(caps),
            "ranked_phases": _ranked_payload(ranked),
            "omitted_zero_bcol_ids": omitted_zero_bcol_ids,
            "excluded_selected_bcol_ids": (
                excluded_selected_bcol_ids
            ),
            "proof_authority": False,
        }
    )
    _check_deadline(deadline, stage="after_subset_binding")
    return result


def _make_bound_literals(
    *,
    parent_digest: str,
    subset_digest: str,
    ranked: Tuple[RankedOperatorPhase, ...],
) -> Tuple[PhaseLiteral, ...]:
    return tuple(
        PhaseLiteral(
            stable_bcol_id=item.stable_bcol_id,
            phase=item.phase,
            binding_digest=_literal_binding_digest(
                parent_digest=parent_digest,
                property_digest=subset_digest,
                stable_bcol_id=item.stable_bcol_id,
                phase=item.phase,
            ),
        )
        for item in ranked
    )


def _literal_payload(
    literal: Any,
) -> Optional[Tuple[int, int, str]]:
    if (
        type(literal) is not PhaseLiteral
        or type(literal.stable_bcol_id) is not int
        or literal.stable_bcol_id < 0
        or literal.stable_bcol_id > _MAX_INT64
        or type(literal.phase) is not int
        or literal.phase not in {-1, 1}
        or not _valid_sha256(literal.binding_digest)
    ):
        return None
    return (
        literal.stable_bcol_id,
        literal.phase,
        literal.binding_digest,
    )


def _literal_tuple_payload(
    literals: Any,
    *,
    expected_length: Optional[int] = None,
    maximum_length: int = 64,
    deadline: Optional[float] = None,
) -> Optional[Tuple[Tuple[int, int, str], ...]]:
    if (
        type(literals) is not tuple
        or len(literals) > maximum_length
        or (
            expected_length is not None
            and len(literals) != expected_length
        )
    ):
        return None
    result = []
    for index, item in enumerate(literals):
        if (
            deadline is not None
            and index % 64 == 0
            and time.monotonic() >= deadline
        ):
            return None
        result.append(_literal_payload(item))
    if any(item is None for item in result):
        return None
    return tuple(result)  # type: ignore[return-value]


def _all_pairs(
    literals: Tuple[PhaseLiteral, ...],
) -> Tuple[Tuple[PhaseLiteral, PhaseLiteral], ...]:
    return tuple(
        _ordered_pair(left, right)
        for left_index, left in enumerate(literals)
        for right in literals[left_index + 1 :]
    )


def _new_candidate_progress() -> dict[str, Any]:
    """Return an authority-free builtin-only partial-progress frame."""

    return {
        "schema": _PROGRESS_SCHEMA,
        "status": "initialized",
        "candidate_only": True,
        "proof_authority": False,
        "verdict_authority": False,
        "model_load_started": False,
        "model_loaded": False,
        "oracle_backend": None,
        "oracle_presolve": None,
        "candidate_load_mode": None,
        "binary_change_coefficient_cap": None,
        "candidate_rows": None,
        "candidate_columns": None,
        "candidate_nonzeros": None,
        "pair_target_count": 0,
        "pair_attempted_count": 0,
        "pair_completed_count": 0,
        "certified_conflict_count": 0,
        "last_pair_index": None,
        "terminal_complete": False,
        "candidate_cut_hz_emitted": False,
        "partial_never_authorizes_edge": True,
        "materializer_reached": False,
    }


def _publish_candidate_progress(
    progress: dict[str, Any],
    sink: Optional[dict[str, Any]],
) -> None:
    """Overwrite a caller sink; the proof path never reads it back."""

    if sink is None:
        return
    sink.clear()
    sink.update(progress)


def _pair_key_from_literals(
    literals: Tuple[PhaseLiteral, PhaseLiteral],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    return tuple(
        (literal.stable_bcol_id, literal.phase)
        for literal in literals
    )  # type: ignore[return-value]


def _probe_all_pairs(
    hz: SparseHZono,
    *,
    literals: Tuple[PhaseLiteral, ...],
    parent_digest: str,
    subset_digest: str,
    source_frame_digest: str,
    deadline: float,
    caps: OperatorPhaseCliqueCaps,
    progress: dict[str, Any],
    progress_sink: Optional[dict[str, Any]],
) -> Tuple[
    Tuple[PersistentPairRecord, ...],
    Tuple[ExactDualRayConflictCertificate, ...],
    Optional[Mapping[str, Any]],
]:
    pairs = _all_pairs(literals)
    progress["pair_target_count"] = len(pairs)
    _publish_candidate_progress(progress, progress_sink)
    if not pairs:
        return (), (), None
    _check_deadline(deadline, stage="before_persistent_model_build")
    # Candidate-first mode avoids solving the huge unfixed relaxation merely
    # to seed a basis.  The first run is made only after fixing a requested
    # pair.  HiGHS remains proposal-only; every accepted edge below still
    # requires full-parent exact Fraction replay.
    progress["status"] = "model_load_started"
    progress["model_load_started"] = True
    _publish_candidate_progress(progress, progress_sink)
    oracle = _PersistentHighsPairLP(
        hz,
        deadline=deadline,
        solve_base_relaxation=False,
        candidate_presolve=True,
    )
    records = []
    certificates = []
    try:
        initial_oracle_telemetry = dict(oracle.telemetry)
        progress.update({
            "status": "model_loaded",
            "model_loaded": True,
            "oracle_backend": initial_oracle_telemetry.get(
                "backend"
            ),
            "oracle_presolve": initial_oracle_telemetry.get(
                "presolve"
            ),
            "candidate_load_mode": initial_oracle_telemetry.get(
                "candidate_load_mode"
            ),
            "binary_change_coefficient_cap": (
                initial_oracle_telemetry.get(
                    "binary_change_coefficient_cap"
                )
            ),
            "candidate_rows": initial_oracle_telemetry.get(
                "candidate_rows"
            ),
            "candidate_columns": initial_oracle_telemetry.get(
                "candidate_columns"
            ),
            "candidate_nonzeros": initial_oracle_telemetry.get(
                "candidate_nonzeros"
            ),
        })
        _publish_candidate_progress(progress, progress_sink)
        _check_deadline(
            deadline, stage="after_persistent_model_build"
        )
        for pair_index, pair in enumerate(pairs):
            _check_deadline(deadline, stage="before_pair_probe")
            progress["status"] = "pair_probe"
            progress["pair_attempted_count"] = pair_index + 1
            progress["last_pair_index"] = pair_index
            _publish_candidate_progress(progress, progress_sink)
            candidate_status, raw_ray = oracle.probe(
                pair, deadline=deadline
            )
            certificate = None
            if candidate_status == "infeasible_with_ray":
                if raw_ray is None:
                    raise OperatorExactReLUPhaseCliqueError(
                        "dual_ray_missing"
                    )
                ray_nonzeros = int(np.count_nonzero(raw_ray))
                certificate = (
                    exact_certificate_from_highs_dual_ray_candidate(
                        hz,
                        pair,
                        raw_ray,
                        parent_digest=parent_digest,
                        property_digest=subset_digest,
                        source_frame_digest=source_frame_digest,
                        deadline=deadline,
                        max_source_terms=caps.max_source_terms,
                        max_multiplier_bits=caps.max_multiplier_bits,
                        max_exact_bits=caps.max_exact_bits,
                        max_exact_nonzeros=caps.max_exact_nonzeros,
                    )
                )
                status = (
                    "certified_conflict"
                    if certificate is not None
                    else "exact_replay_rejected"
                )
            else:
                ray_nonzeros = 0
                status = candidate_status
            if certificate is not None:
                certificates.append(certificate)
            records.append(
                PersistentPairRecord(
                    literals=pair,
                    status=status,
                    ray_nonzero_rows=ray_nonzeros,
                    certificate_sha256=(
                        None
                        if certificate is None
                        else certificate.certificate_sha256
                    ),
                    rationalization=(
                        None
                        if certificate is None
                        else certificate.rationalization
                    ),
                )
            )
            progress["pair_completed_count"] = pair_index + 1
            progress["certified_conflict_count"] = len(
                certificates
            )
            _publish_candidate_progress(progress, progress_sink)
    finally:
        # The native model is owned solely by this scope.  Closing in one
        # unconditional location prevents success, deadline, solver, and
        # exact-replay exits from relying on frame teardown or ``__del__``.
        oracle.close()
    oracle_telemetry = MappingProxyType(dict(oracle.telemetry))
    return (
        tuple(records),
        tuple(certificates),
        oracle_telemetry,
    )


def _maximal_weighted_cliques(
    *,
    ranked: Tuple[RankedOperatorPhase, ...],
    literals: Tuple[PhaseLiteral, ...],
    pair_records: Tuple[PersistentPairRecord, ...],
    subset_digest: str,
    max_cliques: int,
    max_search_nodes: int,
    max_exact_bits: int,
    deadline: float,
) -> Tuple[
    Tuple[OperatorCertifiedPhaseClique, ...],
    int,
    bool,
]:
    """Bounded exact maximal-clique search with greedy warm starts."""

    index_by_stable = {
        item.stable_bcol_id: index
        for index, item in enumerate(ranked)
    }
    certificate_by_edge: dict[Tuple[int, int], str] = {}
    adjacency = [set() for _ in ranked]
    for record_index, record in enumerate(pair_records):
        if record_index % 64 == 0:
            _check_deadline(
                deadline, stage="certified_graph_build"
            )
        if record.status != "certified_conflict":
            continue
        left, right = record.literals
        left_index = index_by_stable[left.stable_bcol_id]
        right_index = index_by_stable[right.stable_bcol_id]
        edge = (
            min(left_index, right_index),
            max(left_index, right_index),
        )
        adjacency[edge[0]].add(edge[1])
        adjacency[edge[1]].add(edge[0])
        if record.certificate_sha256 is None:
            raise OperatorExactReLUPhaseCliqueError(
                "certified_edge_missing_digest"
            )
        certificate_by_edge[edge] = (
            record.certificate_sha256
        )

    def ranking_key(
        members: Tuple[int, ...],
    ) -> Tuple[int, Fraction, Tuple[int, ...]]:
        total = sum(
            (ranked[index].score for index in members),
            Fraction(0),
        )
        stable_ids = tuple(
            sorted(
                ranked[index].stable_bcol_id
                for index in members
            )
        )
        return (-len(members), -total, stable_ids)

    best: dict[
        Tuple[int, ...],
        Tuple[int, Fraction, Tuple[int, ...]],
    ] = {}

    def consider(members: Tuple[int, ...]) -> None:
        if len(members) < 2 or members in best:
            return
        key = ranking_key(members)
        if len(best) < max_cliques:
            best[members] = key
            return
        worst = max(best, key=lambda item: best[item])
        if key < best[worst]:
            del best[worst]
            best[members] = key

    for left, right in sorted(certificate_by_edge):
        _check_deadline(
            deadline, stage="weighted_clique_extraction"
        )
        clique = {left, right}
        for candidate in range(len(ranked)):
            if candidate in clique:
                continue
            if all(
                candidate in adjacency[member]
                for member in clique
            ):
                clique.add(candidate)
        if len(clique) >= 2:
            consider(tuple(sorted(clique)))

    # Enumerate maximal cliques with deterministic Bron--Kerbosch pivoting.
    # Greedy maximal cliques warm the bounded top-K set.  Hitting the search
    # node cap still fails closed so the advertised size/weight ordering is
    # never based on a silently partial maximal-clique enumeration.
    adjacency_bits = []
    for neighbours in adjacency:
        bits = 0
        for neighbour in neighbours:
            bits |= 1 << neighbour
        adjacency_bits.append(bits)
    all_bits = (1 << len(ranked)) - 1
    search_nodes = [0]
    search_truncated = [False]

    def bit_members(bits: int) -> Tuple[int, ...]:
        members = []
        while bits:
            least = bits & -bits
            members.append(least.bit_length() - 1)
            bits ^= least
        return tuple(members)

    def search(
        clique_bits: int,
        possible_bits: int,
        excluded_bits: int,
    ) -> None:
        if search_truncated[0]:
            return
        if search_nodes[0] >= max_search_nodes:
            search_truncated[0] = True
            return
        search_nodes[0] += 1
        if search_nodes[0] % 256 == 1:
            _check_deadline(
                deadline, stage="bron_kerbosch_search"
            )
        if possible_bits == 0 and excluded_bits == 0:
            if clique_bits.bit_count() >= 2:
                consider(bit_members(clique_bits))
            return
        pivot_pool = possible_bits | excluded_bits
        if pivot_pool:
            pivots = bit_members(pivot_pool)
            pivot = max(
                pivots,
                key=lambda item: (
                    (
                        possible_bits
                        & adjacency_bits[item]
                    ).bit_count(),
                    -item,
                ),
            )
            branch_bits = (
                possible_bits
                & ~adjacency_bits[pivot]
                & all_bits
            )
        else:
            branch_bits = possible_bits
        while branch_bits:
            least = branch_bits & -branch_bits
            vertex = least.bit_length() - 1
            search(
                clique_bits | least,
                possible_bits & adjacency_bits[vertex],
                excluded_bits & adjacency_bits[vertex],
            )
            possible_bits &= ~least
            excluded_bits |= least
            branch_bits &= ~least
            if search_truncated[0]:
                return

    search(0, all_bits, 0)
    if search_truncated[0]:
        raise OperatorExactReLUPhaseCliqueError(
            "clique_search_node_cap_exceeded"
        )
    _check_deadline(deadline, stage="after_bron_kerbosch_search")
    ordered = tuple(
        sorted(best, key=lambda item: best[item])
    )
    result = []
    for members in ordered:
        _check_deadline(
            deadline, stage="weighted_clique_materialization"
        )
        clique_literals = tuple(literals[index] for index in members)
        edge_digests = tuple(
            certificate_by_edge[(left, right)]
            for left_offset, left in enumerate(members)
            for right in members[left_offset + 1 :]
        )
        total_score = sum(
            (ranked[index].score for index in members),
            Fraction(0),
        )
        if (
            total_score.numerator.bit_length()
            > max_exact_bits
            or total_score.denominator.bit_length()
            > max_exact_bits
        ):
            raise OperatorExactReLUPhaseCliqueError(
                "clique_score_exact_bit_cap_exceeded"
            )
        clique_id = _canonical_sha256(
            {
                "schema": (
                    "act.operator_exact_relu_certified_clique.v1"
                ),
                "subset_binding_digest": subset_digest,
                "literals": tuple(
                    _literal_payload(item)
                    for item in clique_literals
                ),
                "edge_certificate_sha256s": edge_digests,
                "total_score": (
                    total_score.numerator,
                    total_score.denominator,
                ),
                "proof_authority": False,
            }
        )
        result.append(
            OperatorCertifiedPhaseClique(
                clique_id=clique_id,
                literals=clique_literals,
                edge_certificate_sha256s=edge_digests,
                total_score_numerator=total_score.numerator,
                total_score_denominator=total_score.denominator,
            )
        )
    _check_deadline(deadline, stage="after_weighted_cliques")
    return (
        tuple(result),
        search_nodes[0],
        search_truncated[0],
    )


def _telemetry(
    *,
    caps: OperatorPhaseCliqueCaps,
    ranked_count: int,
    omitted_count: int,
    excluded_count: int,
    records: Tuple[PersistentPairRecord, ...],
    certificates: Tuple[
        ExactDualRayConflictCertificate, ...
    ],
    cliques: Tuple[OperatorCertifiedPhaseClique, ...],
    clique_search_nodes: int,
    clique_search_truncated: bool,
    oracle_telemetry: Optional[Mapping[str, Any]],
    emit_cut_hz: bool,
) -> Mapping[str, Any]:
    return {
        "schema": (
            _LEGACY_TELEMETRY_SCHEMA
            if emit_cut_hz
            else _COMPACT_TELEMETRY_SCHEMA
        ),
        "caps": dict(_caps_payload(caps)),
        "ranked_literal_count": ranked_count,
        "omitted_zero_count": omitted_count,
        "excluded_selected_count": excluded_count,
        "pair_count": len(records),
        "certified_edge_count": sum(
            record.status == "certified_conflict"
            for record in records
        ),
        "exact_certificate_count": len(certificates),
        "clique_count": len(cliques),
        "clique_search_nodes": clique_search_nodes,
        "clique_search_truncated": clique_search_truncated,
        "model_builds": 0 if oracle_telemetry is None else 1,
        "oracle": (
            {}
            if oracle_telemetry is None
            else dict(oracle_telemetry)
        ),
        "proof_authority": False,
    }


def run_operator_exact_relu_phase_cliques_candidate(
    build: OperatorHZBuild,
    focused_rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    *,
    deadline: float,
    selection_max_rivals: int = 128,
    selection_max_binaries: int = 16384,
    selection_max_work_items: int = 5_000_000,
    selection_timeout_seconds: float = 5.0,
    max_parent_variables: int = _DEFAULT_MAX_PARENT_VARIABLES,
    max_parent_rows: int = _DEFAULT_MAX_PARENT_ROWS,
    max_parent_nonzeros: int = _DEFAULT_MAX_PARENT_NONZEROS,
    max_parent_buffer_items: int = (
        _DEFAULT_MAX_PARENT_BUFFER_ITEMS
    ),
    max_top_literals: int = _DEFAULT_MAX_TOP_LITERALS,
    max_total_pairs: int = _DEFAULT_MAX_TOTAL_PAIRS,
    max_cliques: int = _DEFAULT_MAX_CLIQUES,
    max_clique_search_nodes: int = (
        _DEFAULT_MAX_CLIQUE_SEARCH_NODES
    ),
    max_source_terms: int = _DEFAULT_MAX_SOURCE_TERMS,
    max_multiplier_bits: int = _DEFAULT_MAX_MULTIPLIER_BITS,
    max_exact_bits: int = _DEFAULT_MAX_EXACT_BITS,
    max_exact_nonzeros: int = _DEFAULT_MAX_EXACT_NONZEROS,
    emit_cut_hz: bool = True,
    diagnostic_progress: Optional[dict[str, Any]] = None,
) -> OperatorExactReLUPhaseCliqueResult:
    """Close one explicit common-sign focused rival selection.

    ``focused_rivals`` may be a strict ordered subset of the full property,
    including a single rival.  The returned result claims no selection
    coverage outside that subset.  Any emitted cut is still globally valid
    because every phase-conflict edge is replayed against the full parent set.
    """

    if type(emit_cut_hz) is not bool:
        raise OperatorExactReLUPhaseCliqueError(
            "emit_cut_hz_not_builtin_bool"
        )
    if (
        diagnostic_progress is not None
        and (
            type(diagnostic_progress) is not dict
            or len(diagnostic_progress) != 0
        )
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "diagnostic_progress_must_be_empty_builtin_dict_or_none"
        )
    # This mutable frame is diagnostic-only.  No proof, graph, cut, or
    # return-value decision below reads it back from the caller-owned sink.
    progress = _new_candidate_progress()
    _publish_candidate_progress(progress, diagnostic_progress)
    deadline_value = _normalize_deadline(deadline)
    caps = _normalize_caps(
        max_parent_variables=max_parent_variables,
        max_parent_rows=max_parent_rows,
        max_parent_nonzeros=max_parent_nonzeros,
        max_parent_buffer_items=max_parent_buffer_items,
        max_top_literals=max_top_literals,
        max_total_pairs=max_total_pairs,
        max_cliques=max_cliques,
        max_clique_search_nodes=max_clique_search_nodes,
        max_source_terms=max_source_terms,
        max_multiplier_bits=max_multiplier_bits,
        max_exact_bits=max_exact_bits,
        max_exact_nonzeros=max_exact_nonzeros,
    )
    if type(selection) is not OperatorExactReLUPhaseSelection:
        raise OperatorExactReLUPhaseCliqueError(
            "operator_phase_clique_input_wrong_type"
        )
    source = _snapshot_operator_build(
        build,
        caps=caps,
        deadline=deadline_value,
        require_materializer_source=False,
    )
    hz = source.build.hz
    hz_layout = _exact_hz_core_layout(hz)
    rival_snapshot = _snapshot_rivals(
        focused_rivals,
        output_width=hz_layout.n_out,
        maximum=selection_max_rivals,
        deadline=deadline_value,
    )
    trusted_selection = _verify_live_selection(
        source.build,
        rival_snapshot,
        selection,
        deadline=deadline_value,
        selection_max_rivals=selection_max_rivals,
        selection_max_binaries=selection_max_binaries,
        selection_max_work_items=selection_max_work_items,
        selection_timeout_seconds=selection_timeout_seconds,
    )
    _check_deadline(
        deadline_value, stage="before_parent_semantic_seal"
    )
    parent_digest = sparse_hz_semantic_digest(hz)
    if parent_digest != trusted_selection.parent_semantic_digest:
        raise OperatorExactReLUPhaseCliqueError(
            "operator_parent_digest_stale"
        )
    _check_deadline(deadline_value, stage="after_parent_seal")
    ranked, omitted, excluded = _ranked_subset(
        trusted_selection, caps=caps, deadline=deadline_value
    )
    subset_digest = _subset_binding_digest(
        selection=trusted_selection,
        caps=caps,
        ranked=ranked,
        omitted_zero_bcol_ids=omitted,
        excluded_selected_bcol_ids=excluded,
        deadline=deadline_value,
    )
    literals = _make_bound_literals(
        parent_digest=parent_digest,
        subset_digest=subset_digest,
        ranked=ranked,
    )
    pairs = _all_pairs(literals)
    if len(pairs) > caps.max_total_pairs:
        raise OperatorExactReLUPhaseCliqueError(
            "top_k_pair_cap_exceeded"
        )
    progress["status"] = "pair_plan_ready"
    progress["pair_target_count"] = len(pairs)
    _publish_candidate_progress(progress, diagnostic_progress)
    source_frame_digest = None
    if pairs:
        source_frame_digest = _ordered_source_frame_digest(
            hz,
            parent_digest=parent_digest,
            deadline=deadline_value,
        )
        _check_deadline(
            deadline_value, stage="after_source_frame_seal"
        )
    records, certificates, oracle_telemetry = _probe_all_pairs(
        hz,
        literals=literals,
        parent_digest=parent_digest,
        subset_digest=subset_digest,
        source_frame_digest=(
            ""
            if source_frame_digest is None
            else source_frame_digest
        ),
        deadline=deadline_value,
        caps=caps,
        progress=progress,
        progress_sink=diagnostic_progress,
    )
    (
        cliques,
        clique_search_nodes,
        clique_search_truncated,
    ) = _maximal_weighted_cliques(
        ranked=ranked,
        literals=literals,
        pair_records=records,
        subset_digest=subset_digest,
        max_cliques=caps.max_cliques,
        max_search_nodes=caps.max_clique_search_nodes,
        max_exact_bits=caps.max_exact_bits,
        deadline=deadline_value,
    )
    cut_hz: Optional[SparseHZono] = None
    if emit_cut_hz:
        if cliques:
            _check_deadline(
                deadline_value, stage="before_clique_cut"
            )
            literal_rows = tuple(
                clique.literals for clique in cliques
            )
            if len(literal_rows) == 1:
                cut_hz = _copy_parent_with_clique_cut(
                    hz,
                    literal_rows[0],
                    caps=caps,
                    deadline=deadline_value,
                )
            else:
                cut_hz = _copy_parent_with_clique_cuts(
                    hz,
                    literal_rows,
                    caps=caps,
                    deadline=deadline_value,
                )
    _check_deadline(deadline_value, stage="after_clique_cuts")
    if cut_hz is not None:
        _check_parent_size(
            cut_hz, caps=caps, deadline=deadline_value
        )
    _check_parent_size(
        hz, caps=caps, deadline=deadline_value
    )
    if sparse_hz_semantic_digest(hz) != parent_digest:
        raise OperatorExactReLUPhaseCliqueError(
            "private_parent_mutated_during_clique_closure"
        )
    _check_deadline(deadline_value, stage="after_terminal_parent_seal")
    progress.update(
        {
            "status": "complete",
            "pair_completed_count": len(records),
            "certified_conflict_count": len(certificates),
            "terminal_complete": True,
            "candidate_cut_hz_emitted": cut_hz is not None,
        }
    )
    _publish_candidate_progress(progress, diagnostic_progress)
    return OperatorExactReLUPhaseCliqueResult(
        status=(
            (
                _LEGACY_SUCCESS_STATUS
                if emit_cut_hz
                else _COMPACT_SUCCESS_STATUS
            )
            if cliques
            else (
                _LEGACY_EMPTY_STATUS
                if emit_cut_hz
                else _COMPACT_EMPTY_STATUS
            )
        ),
        hz=cut_hz,
        parent_semantic_digest=parent_digest,
        focused_property_digest=trusted_selection.property_digest,
        operator_row_tag_digest=(
            trusted_selection.operator_row_tag_digest
        ),
        selection_digest=trusted_selection.selection_digest,
        subset_binding_digest=subset_digest,
        ordered_source_frame_sha256=source_frame_digest,
        caps=caps,
        ranked_phases=ranked,
        literals=literals,
        omitted_zero_bcol_ids=omitted,
        excluded_selected_bcol_ids=excluded,
        pair_records=records,
        certificates=certificates,
        cliques=cliques,
        telemetry=_telemetry(
            caps=caps,
            ranked_count=len(ranked),
            omitted_count=len(omitted),
            excluded_count=len(excluded),
            records=records,
            certificates=certificates,
            cliques=cliques,
            clique_search_nodes=clique_search_nodes,
            clique_search_truncated=(
                clique_search_truncated
            ),
            oracle_telemetry=oracle_telemetry,
            emit_cut_hz=emit_cut_hz,
        ),
    )


def _exact_ranked_payload(
    ranked: Any,
    *,
    expected_length: int,
    max_exact_bits: int,
    deadline: float,
) -> Optional[Tuple[Tuple[int, int, int, int, int], ...]]:
    if (
        type(ranked) is not tuple
        or len(ranked) != expected_length
        or len(ranked) > _HARD_LIMITS["max_top_literals"]
    ):
        return None
    payload = []
    for index, item in enumerate(ranked):
        if index % 64 == 0 and time.monotonic() >= deadline:
            return None
        if (
            type(item) is not RankedOperatorPhase
            or item.proof_authority is not False
            or type(item.rank) is not int
            or item.rank < 0
            or item.rank > _MAX_INT64
            or type(item.stable_bcol_id) is not int
            or item.stable_bcol_id < 0
            or item.stable_bcol_id > _MAX_INT64
            or type(item.phase) is not int
            or item.phase not in {-1, 1}
            or type(item.score_numerator) is not int
            or item.score_numerator <= 0
            or type(item.score_denominator) is not int
            or item.score_denominator <= 0
            or item.score_numerator.bit_length()
            > max_exact_bits
            or item.score_denominator.bit_length()
            > max_exact_bits
            or math.gcd(
                item.score_numerator,
                item.score_denominator,
            )
            != 1
        ):
            return None
        payload.append(
            (
                item.rank,
                item.stable_bcol_id,
                item.phase,
                item.score_numerator,
                item.score_denominator,
            )
        )
    return tuple(payload)


def _exact_certificate_shape(
    certificate: Any,
    *,
    caps: OperatorPhaseCliqueCaps,
    deadline: float,
) -> bool:
    if (
        type(certificate)
        is not ExactDualRayConflictCertificate
        or certificate.proof_authority is not False
        or type(certificate.arithmetic) is not str
        or certificate.arithmetic
        != "sparse_Fraction_exact_replay_v2"
        or not _valid_sha256(
            certificate.parent_semantic_digest
        )
        or not _valid_sha256(certificate.property_digest)
        or not _valid_sha256(
            certificate.ordered_source_frame_sha256
        )
        or not _valid_sha256(
            certificate.certificate_sha256
        )
        or type(certificate.rationalization) is not str
        or type(certificate.contradiction_numerator) is not int
        or certificate.contradiction_numerator >= 0
        or type(certificate.contradiction_denominator) is not int
        or certificate.contradiction_denominator <= 0
        or abs(
            certificate.contradiction_numerator
        ).bit_length()
        > caps.max_exact_bits
        or certificate.contradiction_denominator.bit_length()
        > caps.max_exact_bits
        or math.gcd(
            abs(certificate.contradiction_numerator),
            certificate.contradiction_denominator,
        )
        != 1
        or _literal_tuple_payload(
            certificate.literals,
            expected_length=2,
            maximum_length=2,
            deadline=deadline,
        )
        is None
        or type(certificate.source_terms) is not tuple
        or not certificate.source_terms
        or len(certificate.source_terms)
        > caps.max_source_terms
    ):
        return False
    for term in certificate.source_terms:
        if time.monotonic() >= deadline:
            return False
        if (
            type(term) is not ExactSourceTermV2
            or type(term.global_row_index) is not int
            or term.global_row_index < 0
            or term.global_row_index > _MAX_INT64
            or type(term.kind) is not str
            or term.kind not in _VALID_SOURCE_KINDS
            or type(term.local_row_index) is not int
            or term.local_row_index < 0
            or term.local_row_index > _MAX_INT64
            or type(term.numerator) is not int
            or term.numerator <= 0
            or type(term.denominator) is not int
            or term.denominator <= 0
            or term.numerator.bit_length()
            > caps.max_multiplier_bits
            or term.denominator.bit_length()
            > caps.max_multiplier_bits
            or math.gcd(term.numerator, term.denominator) != 1
            or not _valid_sha256(term.source_row_sha256)
        ):
            return False
    return True


def _snapshot_exact_certificate(
    certificate: Any,
    *,
    caps: OperatorPhaseCliqueCaps,
    deadline: float,
) -> Optional[ExactDualRayConflictCertificate]:
    """Copy a candidate certificate before any exact replay consumes it."""

    if type(certificate) is not ExactDualRayConflictCertificate:
        return None
    live = vars(certificate)
    raw_literals = live.get("literals")
    raw_terms = live.get("source_terms")
    if (
        type(raw_literals) is not tuple
        or len(raw_literals) != 2
        or type(raw_terms) is not tuple
        or not raw_terms
        or len(raw_terms) > caps.max_source_terms
    ):
        return None
    literals = []
    for literal in raw_literals:
        payload = _literal_payload(literal)
        if payload is None:
            return None
        literals.append(
            PhaseLiteral(
                stable_bcol_id=payload[0],
                phase=payload[1],
                binding_digest=payload[2],
            )
        )
    terms = []
    for index, term in enumerate(raw_terms):
        if index % 64 == 0 and time.monotonic() >= deadline:
            return None
        if type(term) is not ExactSourceTermV2:
            return None
        term_live = vars(term)
        terms.append(
            ExactSourceTermV2(
                global_row_index=term_live.get(
                    "global_row_index"
                ),
                kind=term_live.get("kind"),
                local_row_index=term_live.get(
                    "local_row_index"
                ),
                numerator=term_live.get("numerator"),
                denominator=term_live.get("denominator"),
                source_row_sha256=term_live.get(
                    "source_row_sha256"
                ),
            )
        )
    private = ExactDualRayConflictCertificate(
        literals=(literals[0], literals[1]),
        parent_semantic_digest=live.get(
            "parent_semantic_digest"
        ),
        property_digest=live.get("property_digest"),
        ordered_source_frame_sha256=live.get(
            "ordered_source_frame_sha256"
        ),
        source_terms=tuple(terms),
        contradiction_numerator=live.get(
            "contradiction_numerator"
        ),
        contradiction_denominator=live.get(
            "contradiction_denominator"
        ),
        rationalization=live.get("rationalization"),
        certificate_sha256=live.get("certificate_sha256"),
        arithmetic=live.get("arithmetic"),
        proof_authority=live.get("proof_authority"),
    )
    if not _exact_certificate_shape(
        private,
        caps=caps,
        deadline=deadline,
    ):
        return None
    return private


def _exact_clique_payload(
    clique: Any,
    *,
    caps: OperatorPhaseCliqueCaps,
    deadline: float,
) -> Optional[
    Tuple[
        str,
        Tuple[Tuple[int, int, str], ...],
        Tuple[str, ...],
        int,
        int,
    ]
]:
    if (
        type(clique) is not OperatorCertifiedPhaseClique
        or clique.proof_authority is not False
        or clique.cut_applied is not True
        or not _valid_sha256(clique.clique_id)
        or type(clique.edge_certificate_sha256s) is not tuple
        or len(clique.edge_certificate_sha256s)
        > caps.max_total_pairs
        or type(clique.total_score_numerator) is not int
        or clique.total_score_numerator <= 0
        or type(clique.total_score_denominator) is not int
        or clique.total_score_denominator <= 0
        or clique.total_score_numerator.bit_length()
        > caps.max_exact_bits
        or clique.total_score_denominator.bit_length()
        > caps.max_exact_bits
        or math.gcd(
            clique.total_score_numerator,
            clique.total_score_denominator,
        )
        != 1
    ):
        return None
    literals = _literal_tuple_payload(
        clique.literals,
        maximum_length=caps.max_top_literals,
        deadline=deadline,
    )
    if literals is None or len(literals) < 2:
        return None
    expected_edges = len(literals) * (len(literals) - 1) // 2
    if len(clique.edge_certificate_sha256s) != expected_edges:
        return None
    for index, digest in enumerate(
        clique.edge_certificate_sha256s
    ):
        if index % 64 == 0 and time.monotonic() >= deadline:
            return None
        if not _valid_sha256(digest):
            return None
    return (
        clique.clique_id,
        literals,
        clique.edge_certificate_sha256s,
        clique.total_score_numerator,
        clique.total_score_denominator,
    )


def _exact_integer_tuple(
    value: Any,
    *,
    expected_length: int,
    maximum_length: int,
    deadline: float,
) -> Optional[Tuple[int, ...]]:
    if (
        type(value) is not tuple
        or len(value) != expected_length
        or len(value) > maximum_length
    ):
        return None
    for index, item in enumerate(value):
        if index % 256 == 0 and time.monotonic() >= deadline:
            return None
        if (
            type(item) is not int
            or item < 0
            or item > _MAX_INT64
        ):
            return None
    if time.monotonic() >= deadline:
        return None
    if (
        len(set(value)) != len(value)
    ):
        return None
    if time.monotonic() >= deadline:
        return None
    return value


def _telemetry_is_exact(
    telemetry: Any,
    *,
    hz: SparseHZono,
    caps: OperatorPhaseCliqueCaps,
    ranked_count: int,
    omitted_count: int,
    excluded_count: int,
    records: Tuple[PersistentPairRecord, ...],
    certificate_count: int,
    clique_count: int,
    clique_search_nodes: int,
    clique_search_truncated: bool,
    expected_schema: str,
    deadline: float,
) -> bool:
    expected_keys = {
        "schema",
        "caps",
        "ranked_literal_count",
        "omitted_zero_count",
        "excluded_selected_count",
        "pair_count",
        "certified_edge_count",
        "exact_certificate_count",
        "clique_count",
        "clique_search_nodes",
        "clique_search_truncated",
        "model_builds",
        "oracle",
        "proof_authority",
    }
    if (
        type(telemetry) is not dict
        or len(telemetry) != len(expected_keys)
        or any(type(key) is not str for key in telemetry)
        or set(telemetry) != expected_keys
        or type(telemetry["schema"]) is not str
        or telemetry["schema"]
        != expected_schema
        or telemetry["proof_authority"] is not False
        or type(telemetry["clique_search_truncated"])
        is not bool
        or telemetry["clique_search_truncated"]
        is not clique_search_truncated
        or type(telemetry["caps"]) is not dict
    ):
        return False
    live_caps = telemetry["caps"]
    expected_caps = dict(_caps_payload(caps))
    if (
        len(live_caps) != len(expected_caps)
        or
        any(type(key) is not str for key in live_caps)
        or set(live_caps) != set(expected_caps)
        or any(
            type(live_caps[name]) is not int
            or live_caps[name] != expected_caps[name]
            for name in expected_caps
        )
    ):
        return False
    expected_counts = {
        "ranked_literal_count": ranked_count,
        "omitted_zero_count": omitted_count,
        "excluded_selected_count": excluded_count,
        "pair_count": len(records),
        "certified_edge_count": sum(
            record.status == "certified_conflict"
            for record in records
        ),
        "exact_certificate_count": certificate_count,
        "clique_count": clique_count,
        "clique_search_nodes": clique_search_nodes,
        "model_builds": 0 if not records else 1,
    }
    if any(
        type(telemetry[name]) is not int
        or telemetry[name] != expected
        for name, expected in expected_counts.items()
    ):
        return False
    oracle = telemetry["oracle"]
    if not records:
        return type(oracle) is dict and not oracle
    expected_oracle_keys = {
        "backend",
        "highs_version",
        "row_order",
        "presolve",
        "threads",
        "model_builds",
        "candidate_rows",
        "candidate_columns",
        "candidate_nonzeros",
        "candidate_load_mode",
        "binary_change_coefficient_cap",
        "base_solve_calls",
        "solve_calls",
        "bound_update_calls",
        "dual_ray_calls",
        "highs_cumulative_run_time_seconds",
    }
    if (
        type(oracle) is not dict
        or len(oracle) != len(expected_oracle_keys)
        or any(type(key) is not str for key in oracle)
        or set(oracle) != expected_oracle_keys
        or type(oracle["backend"]) is not str
        or type(oracle["highs_version"]) is not str
        or type(oracle["row_order"]) is not str
        or oracle["row_order"] != "upper_then_equality"
        or type(oracle["presolve"]) is not str
        or (
            oracle["presolve"], oracle["backend"]
        )
        not in {
            (
                "off",
                "highspy_persistent_simplex_dual_ray_v1",
            ),
            (
                "on",
                "highspy_persistent_simplex_presolve_lazy_dual_ray_v2",
            ),
        }
        or type(oracle["candidate_load_mode"]) is not str
        or oracle["candidate_load_mode"]
        not in {
            "split_continuous_rows_binary_change_coeff_v1",
            "single_merged_csr_binary_cap_fallback_v1",
        }
        or type(oracle["binary_change_coefficient_cap"])
        is not int
        or oracle["binary_change_coefficient_cap"]
        != _MAX_BINARY_CHANGE_COEFFICIENTS
    ):
        return False
    integer_fields = (
        "threads",
        "model_builds",
        "candidate_rows",
        "candidate_columns",
        "candidate_nonzeros",
        "base_solve_calls",
        "solve_calls",
        "bound_update_calls",
        "dual_ray_calls",
    )
    if any(
        type(oracle[name]) is not int or oracle[name] < 0
        for name in integer_fields
    ):
        return False
    runtime = oracle["highs_cumulative_run_time_seconds"]
    def kept_nonzeros(matrix: sp.csr_matrix) -> int:
        total = 0
        data = matrix.data
        for start in range(0, int(data.size), 1 << 18):
            _check_deadline(
                deadline, stage="candidate_telemetry_nnz_replay"
            )
            chunk = data[start : start + (1 << 18)]
            total += int(
                np.count_nonzero(
                    np.abs(chunk) > _CANDIDATE_DUST_ABS
                )
            )
        return total

    kept_counts = {
        name: kept_nonzeros(getattr(hz, name))
        for name in ("Auc", "Aub", "Ac", "Ab")
    }
    expected_nonzeros = sum(kept_counts.values())
    kept_binary_nonzeros = kept_counts["Aub"] + kept_counts["Ab"]
    expected_load_mode = (
        "split_continuous_rows_binary_change_coeff_v1"
        if kept_binary_nonzeros
        <= _MAX_BINARY_CHANGE_COEFFICIENTS
        else "single_merged_csr_binary_cap_fallback_v1"
    )
    return bool(
        oracle["threads"] >= 1
        and oracle["model_builds"] == 1
        and oracle["candidate_rows"] == hz.n_ub + hz.n_eq
        and oracle["candidate_columns"] == hz.n_cont + hz.n_bin
        # Replay the loader's exact dust rule in bounded chunks.  This field
        # is route evidence; accepted rays still replay against the unmodified
        # full parent.
        and oracle["candidate_nonzeros"] == expected_nonzeros
        and oracle["candidate_load_mode"] == expected_load_mode
        and oracle["base_solve_calls"] == 0
        and oracle["solve_calls"] == len(records)
        and oracle["bound_update_calls"] == 2 * len(records)
        and oracle["dual_ray_calls"] <= len(records)
        and type(runtime) is float
        and math.isfinite(runtime)
        and runtime >= 0.0
    )


def _array_bytes_equal_with_deadline(
    left: Any,
    right: Any,
    *,
    deadline: float,
) -> bool:
    """Bitwise compare trusted-size contiguous arrays in bounded chunks."""

    if (
        type(left) is not np.ndarray
        or type(right) is not np.ndarray
        or left.dtype != right.dtype
        or left.shape != right.shape
        or left.ndim != right.ndim
        or not left.flags.c_contiguous
        or not right.flags.c_contiguous
    ):
        return False
    left_bytes = left.view(np.uint8).reshape(-1)
    right_bytes = right.view(np.uint8).reshape(-1)
    chunk_bytes = 1 << 20
    for start in range(0, int(left_bytes.size), chunk_bytes):
        if time.monotonic() >= deadline:
            return False
        stop = min(start + chunk_bytes, int(left_bytes.size))
        if not np.array_equal(
            left_bytes[start:stop], right_bytes[start:stop]
        ):
            return False
    return time.monotonic() < deadline


def _cut_hz_exactly_matches_reconstruction(
    candidate: SparseHZono,
    expected: SparseHZono,
    *,
    deadline: float,
) -> bool:
    """Deadline-aware exact comparison without hashing candidate payloads."""

    if (
        type(candidate) is not SparseHZono
        or type(expected) is not SparseHZono
    ):
        return False
    candidate_vars = vars(candidate)
    expected_vars = vars(expected)
    if (
        len(candidate_vars) != len(expected_vars)
        or any(name not in candidate_vars for name in expected_vars)
    ):
        return False
    for name in ("c", "b", "ub", "col_ids", "bcol_ids"):
        if not _array_bytes_equal_with_deadline(
            getattr(candidate, name, None),
            getattr(expected, name, None),
            deadline=deadline,
        ):
            return False
    for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
        left = getattr(candidate, name, None)
        right = getattr(expected, name, None)
        if (
            type(left) is not sp.csr_matrix
            or type(right) is not sp.csr_matrix
            or left.shape != right.shape
            or left.dtype != right.dtype
        ):
            return False
        for buffer_name in ("indptr", "indices", "data"):
            if not _array_bytes_equal_with_deadline(
                getattr(left, buffer_name, None),
                getattr(right, buffer_name, None),
                deadline=deadline,
            ):
                return False
    if time.monotonic() >= deadline:
        return False
    expected_conditional_names = tuple(
        name
        for name in expected_vars
        if "conditional" in name.lower()
    )
    # `_copy_parent_with_clique_cut` preserves these live opaque semantics by
    # reference.  Requiring the same objects rejects injected/deep-copied
    # candidate metadata without traversing or hashing attacker payloads.
    for name in expected_conditional_names:
        if (
            time.monotonic() >= deadline
            or getattr(candidate, name)
            is not getattr(expected, name)
        ):
            return False
    return time.monotonic() < deadline


def _verify_operator_exact_relu_phase_cliques_impl(
    build: OperatorHZBuild,
    focused_rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    result: OperatorExactReLUPhaseCliqueResult,
    *,
    deadline: Optional[float] = None,
    selection_max_rivals: int = 128,
    selection_max_binaries: int = 16384,
    selection_max_work_items: int = 5_000_000,
    selection_timeout_seconds: float = 5.0,
    max_parent_variables: int = _DEFAULT_MAX_PARENT_VARIABLES,
    max_parent_rows: int = _DEFAULT_MAX_PARENT_ROWS,
    max_parent_nonzeros: int = _DEFAULT_MAX_PARENT_NONZEROS,
    max_parent_buffer_items: int = (
        _DEFAULT_MAX_PARENT_BUFFER_ITEMS
    ),
    max_top_literals: int = _DEFAULT_MAX_TOP_LITERALS,
    max_total_pairs: int = _DEFAULT_MAX_TOTAL_PAIRS,
    max_cliques: int = _DEFAULT_MAX_CLIQUES,
    max_clique_search_nodes: int = (
        _DEFAULT_MAX_CLIQUE_SEARCH_NODES
    ),
    max_source_terms: int = _DEFAULT_MAX_SOURCE_TERMS,
    max_multiplier_bits: int = _DEFAULT_MAX_MULTIPLIER_BITS,
    max_exact_bits: int = _DEFAULT_MAX_EXACT_BITS,
    max_exact_nonzeros: int = _DEFAULT_MAX_EXACT_NONZEROS,
    _verified_core_sink: Optional[list[_VerifiedCliqueCore]] = None,
) -> bool:
    """Replay a focused selection and globally valid full-parent cuts."""

    try:
        deadline_value = _normalize_deadline(
            time.monotonic() + 60.0
            if deadline is None
            else deadline
        )
        caps = _normalize_caps(
            max_parent_variables=max_parent_variables,
            max_parent_rows=max_parent_rows,
            max_parent_nonzeros=max_parent_nonzeros,
            max_parent_buffer_items=max_parent_buffer_items,
            max_top_literals=max_top_literals,
            max_total_pairs=max_total_pairs,
            max_cliques=max_cliques,
            max_clique_search_nodes=max_clique_search_nodes,
            max_source_terms=max_source_terms,
            max_multiplier_bits=max_multiplier_bits,
            max_exact_bits=max_exact_bits,
            max_exact_nonzeros=max_exact_nonzeros,
        )
        if (
            type(build) is not OperatorHZBuild
            or type(selection)
            is not OperatorExactReLUPhaseSelection
            or type(result)
            is not OperatorExactReLUPhaseCliqueResult
        ):
            return False
        source = _snapshot_operator_build(
            build,
            caps=caps,
            deadline=deadline_value,
            require_materializer_source=(
                _verified_core_sink is not None
            ),
        )
        hz = source.build.hz
        hz_layout = _exact_hz_core_layout(hz)
        result_vars = vars(result)
        live_candidate_hz = result_vars.get("hz")
        if live_candidate_hz is None:
            candidate_hz = None
        elif type(live_candidate_hz) is SparseHZono:
            candidate_hz = _snapshot_sparse_hz(
                live_candidate_hz,
                caps=caps,
                deadline=deadline_value,
                stage="candidate_result",
            )
        else:
            return False
        compact_result = result.status in {
            _COMPACT_SUCCESS_STATUS,
            _COMPACT_EMPTY_STATUS,
        }
        expected_telemetry_schema = (
            _COMPACT_TELEMETRY_SCHEMA
            if compact_result
            else _LEGACY_TELEMETRY_SCHEMA
        )
        if (
            type(selection.mappings) is not tuple
            or len(selection.mappings) > _MAX_SELECTION_MAPPINGS
            or result.proof_authority is not False
            or type(result.status) is not str
            or result.status
            not in {
                _LEGACY_SUCCESS_STATUS,
                _LEGACY_EMPTY_STATUS,
                _COMPACT_SUCCESS_STATUS,
                _COMPACT_EMPTY_STATUS,
            }
            or not _valid_sha256(
                result.parent_semantic_digest
            )
            or not _valid_sha256(
                result.focused_property_digest
            )
            or not _valid_sha256(
                result.operator_row_tag_digest
            )
            or not _valid_sha256(result.selection_digest)
            or not _valid_sha256(
                result.subset_binding_digest
            )
            or (
                result.ordered_source_frame_sha256 is not None
                and not _valid_sha256(
                    result.ordered_source_frame_sha256
                )
            )
            or type(result.pair_records) is not tuple
            or len(result.pair_records) > caps.max_total_pairs
            or type(result.certificates) is not tuple
            or len(result.certificates) > caps.max_total_pairs
            or type(result.cliques) is not tuple
            or len(result.cliques) > caps.max_cliques
            or type(result.ranked_phases) is not tuple
            or len(result.ranked_phases)
            > caps.max_top_literals
            or type(result.literals) is not tuple
            or len(result.literals) > caps.max_top_literals
            or len(result.literals)
            != len(result.ranked_phases)
            or type(result.omitted_zero_bcol_ids) is not tuple
            or len(result.omitted_zero_bcol_ids)
            > len(selection.mappings)
            or type(result.excluded_selected_bcol_ids)
            is not tuple
            or len(result.excluded_selected_bcol_ids)
            > len(selection.mappings)
            or (
                len(result.ranked_phases)
                + len(result.omitted_zero_bcol_ids)
                + len(result.excluded_selected_bcol_ids)
                > len(selection.mappings)
            )
            or type(result.telemetry) is not dict
            or len(result.telemetry) != _TELEMETRY_KEY_COUNT
        ):
            return False
        if _caps_payload(result.caps) != _caps_payload(caps):
            return False
        rival_snapshot = _snapshot_rivals(
            focused_rivals,
            output_width=hz_layout.n_out,
            maximum=selection_max_rivals,
            deadline=deadline_value,
        )
        trusted_selection = _verify_live_selection(
            source.build,
            rival_snapshot,
            selection,
            deadline=deadline_value,
            selection_max_rivals=selection_max_rivals,
            selection_max_binaries=selection_max_binaries,
            selection_max_work_items=selection_max_work_items,
            selection_timeout_seconds=selection_timeout_seconds,
        )
        _check_deadline(
            deadline_value, stage="before_parent_semantic_seal"
        )
        if source.private_parent_semantic_digest is None:
            parent_digest = sparse_hz_semantic_digest(hz)
        else:
            parent_digest = (
                source.private_parent_semantic_digest
            )
        _check_deadline(deadline_value, stage="after_parent_seal")
        if (
            result.parent_semantic_digest != parent_digest
            or result.parent_semantic_digest
            != trusted_selection.parent_semantic_digest
            or result.focused_property_digest
            != trusted_selection.property_digest
            or result.operator_row_tag_digest
            != trusted_selection.operator_row_tag_digest
            or result.selection_digest
            != trusted_selection.selection_digest
        ):
            return False
        ranked, omitted, excluded = _ranked_subset(
            trusted_selection, caps=caps, deadline=deadline_value
        )
        ranked_payload = _exact_ranked_payload(
            result.ranked_phases,
            expected_length=len(ranked),
            max_exact_bits=caps.max_exact_bits,
            deadline=deadline_value,
        )
        expected_ranked_payload = _exact_ranked_payload(
            ranked,
            expected_length=len(ranked),
            max_exact_bits=caps.max_exact_bits,
            deadline=deadline_value,
        )
        actual_omitted = _exact_integer_tuple(
            result.omitted_zero_bcol_ids,
            expected_length=len(omitted),
            maximum_length=len(trusted_selection.mappings),
            deadline=deadline_value,
        )
        actual_excluded = _exact_integer_tuple(
            result.excluded_selected_bcol_ids,
            expected_length=len(excluded),
            maximum_length=len(trusted_selection.mappings),
            deadline=deadline_value,
        )
        if (
            ranked_payload is None
            or expected_ranked_payload is None
            or ranked_payload != expected_ranked_payload
            or actual_omitted is None
            or actual_omitted != omitted
            or actual_excluded is None
            or actual_excluded != excluded
        ):
            return False
        subset_digest = _subset_binding_digest(
            selection=trusted_selection,
            caps=caps,
            ranked=ranked,
            omitted_zero_bcol_ids=omitted,
            excluded_selected_bcol_ids=excluded,
            deadline=deadline_value,
        )
        expected_literals = _make_bound_literals(
            parent_digest=parent_digest,
            subset_digest=subset_digest,
            ranked=ranked,
        )
        actual_literal_payload = _literal_tuple_payload(
            result.literals,
            expected_length=len(expected_literals),
            maximum_length=caps.max_top_literals,
            deadline=deadline_value,
        )
        expected_literal_payload = _literal_tuple_payload(
            expected_literals,
            expected_length=len(expected_literals),
            maximum_length=caps.max_top_literals,
            deadline=deadline_value,
        )
        if (
            result.subset_binding_digest != subset_digest
            or actual_literal_payload is None
            or expected_literal_payload is None
            or actual_literal_payload
            != expected_literal_payload
        ):
            return False
        expected_pairs = _all_pairs(expected_literals)
        candidate_pair_records = result.pair_records
        candidate_certificates = result.certificates
        if (
            type(candidate_pair_records) is not tuple
            or type(candidate_certificates) is not tuple
            or len(expected_pairs) > caps.max_total_pairs
            or len(candidate_pair_records) != len(expected_pairs)
            or len(candidate_certificates) > caps.max_total_pairs
        ):
            return False
        source_frame_digest = None
        if expected_pairs:
            source_frame_digest = _ordered_source_frame_digest(
                hz,
                parent_digest=parent_digest,
                deadline=deadline_value,
            )
            _check_deadline(
                deadline_value, stage="after_source_frame_seal"
            )
        if (
            result.ordered_source_frame_sha256
            != source_frame_digest
        ):
            return False

        certificate_index = 0
        seen_pair_bindings = set()
        seen_certificate_digests = set()
        trusted_records = []
        trusted_certificates = []
        for expected_pair, record in zip(
            expected_pairs, candidate_pair_records
        ):
            _check_deadline(
                deadline_value, stage="before_edge_replay"
            )
            if type(record) is not PersistentPairRecord:
                return False
            record_live = vars(record)
            record_literals = record_live.get("literals")
            record_status = record_live.get("status")
            record_ray_nonzero_rows = record_live.get(
                "ray_nonzero_rows"
            )
            record_certificate_sha256 = record_live.get(
                "certificate_sha256"
            )
            record_rationalization = record_live.get(
                "rationalization"
            )
            expected_pair_payload = _literal_tuple_payload(
                expected_pair,
                expected_length=2,
                maximum_length=2,
                deadline=deadline_value,
            )
            record_pair_payload = _literal_tuple_payload(
                record_literals,
                expected_length=2,
                maximum_length=2,
                deadline=deadline_value,
            )
            if (
                expected_pair_payload is None
                or record_pair_payload is None
                or record_pair_payload != expected_pair_payload
                or type(record_status) is not str
                or record_status not in _VALID_PAIR_STATUSES
                or type(record_ray_nonzero_rows) is not int
                or record_ray_nonzero_rows < 0
                or record_ray_nonzero_rows
                > hz.n_ub + hz.n_eq
            ):
                return False
            pair_binding = (
                subset_digest,
                expected_pair_payload,
            )
            if pair_binding in seen_pair_bindings:
                return False
            seen_pair_bindings.add(pair_binding)
            if record_status == "certified_conflict":
                if certificate_index >= len(
                    candidate_certificates
                ):
                    return False
                certificate = _snapshot_exact_certificate(
                    candidate_certificates[certificate_index],
                    caps=caps,
                    deadline=deadline_value,
                )
                certificate_index += 1
                if certificate is None:
                    return False
                certificate_pair_payload = (
                    _literal_tuple_payload(
                        certificate.literals,
                        expected_length=2,
                        maximum_length=2,
                        deadline=deadline_value,
                    )
                )
                if (
                    certificate_pair_payload is None
                    or certificate_pair_payload
                    != expected_pair_payload
                    or not _valid_sha256(
                        record_certificate_sha256
                    )
                    or record_certificate_sha256
                    != certificate.certificate_sha256
                    or type(record_rationalization) is not str
                    or record_rationalization
                    != certificate.rationalization
                    or certificate.certificate_sha256
                    in seen_certificate_digests
                    or source_frame_digest is None
                    or not _verify_exact_certificate_with_source_frame(
                        hz,
                        certificate,
                        property_digest=subset_digest,
                        parent_digest=parent_digest,
                        source_frame_digest=source_frame_digest,
                        deadline=deadline_value,
                        max_source_terms=caps.max_source_terms,
                        max_multiplier_bits=caps.max_multiplier_bits,
                        max_exact_bits=caps.max_exact_bits,
                        max_exact_nonzeros=(
                            caps.max_exact_nonzeros
                        ),
                    )
                ):
                    return False
                seen_certificate_digests.add(
                    certificate.certificate_sha256
                )
                trusted_certificates.append(certificate)
            elif (
                record_certificate_sha256 is not None
                or record_rationalization is not None
            ):
                return False
            trusted_records.append(
                PersistentPairRecord(
                    literals=expected_pair,
                    status=record_status,
                    ray_nonzero_rows=record_ray_nonzero_rows,
                    certificate_sha256=(
                        record_certificate_sha256
                    ),
                    rationalization=record_rationalization,
                )
            )
        if certificate_index != len(candidate_certificates):
            return False

        (
            expected_cliques,
            expected_clique_search_nodes,
            expected_clique_search_truncated,
        ) = _maximal_weighted_cliques(
            ranked=ranked,
            literals=expected_literals,
            pair_records=tuple(trusted_records),
            subset_digest=subset_digest,
            max_cliques=caps.max_cliques,
            max_search_nodes=caps.max_clique_search_nodes,
            max_exact_bits=caps.max_exact_bits,
            deadline=deadline_value,
        )
        if len(result.cliques) != len(expected_cliques):
            return False
        seen_clique_ids = set()
        for actual, expected in zip(
            result.cliques, expected_cliques
        ):
            actual_payload = _exact_clique_payload(
                actual,
                caps=caps,
                deadline=deadline_value,
            )
            expected_payload = _exact_clique_payload(
                expected,
                caps=caps,
                deadline=deadline_value,
            )
            if (
                actual_payload is None
                or expected_payload is None
                or actual_payload != expected_payload
                or actual.clique_id in seen_clique_ids
            ):
                return False
            seen_clique_ids.add(actual.clique_id)

        reconstructed: Optional[SparseHZono] = None
        if expected_cliques:
            _check_deadline(
                deadline_value, stage="before_cut_reconstruction"
            )
            literal_rows = tuple(
                clique.literals for clique in expected_cliques
            )
            if len(literal_rows) == 1:
                reconstructed = _copy_parent_with_clique_cut(
                    hz,
                    literal_rows[0],
                    caps=caps,
                    deadline=deadline_value,
                )
            else:
                reconstructed = _copy_parent_with_clique_cuts(
                    hz,
                    literal_rows,
                    caps=caps,
                    deadline=deadline_value,
                )
        expected_status = (
            (
                _COMPACT_SUCCESS_STATUS
                if compact_result
                else _LEGACY_SUCCESS_STATUS
            )
            if reconstructed is not None
            else (
                _COMPACT_EMPTY_STATUS
                if compact_result
                else _LEGACY_EMPTY_STATUS
            )
        )
        if result.status != expected_status:
            return False
        if reconstructed is None:
            if candidate_hz is not None:
                return False
        else:
            _check_parent_size(
                reconstructed,
                caps=caps,
                deadline=deadline_value,
            )
            if compact_result:
                if candidate_hz is not None:
                    return False
            else:
                if type(candidate_hz) is not SparseHZono:
                    return False
                _check_parent_size(
                    candidate_hz,
                    caps=caps,
                    deadline=deadline_value,
                )
                if not _cut_hz_exactly_matches_reconstruction(
                    candidate_hz,
                    reconstructed,
                    deadline=deadline_value,
                ):
                    return False
        if not _telemetry_is_exact(
            result.telemetry,
            hz=hz,
            caps=caps,
            ranked_count=len(ranked),
            omitted_count=len(omitted),
            excluded_count=len(excluded),
            records=tuple(trusted_records),
            certificate_count=len(trusted_certificates),
            clique_count=len(expected_cliques),
            clique_search_nodes=(
                expected_clique_search_nodes
            ),
            clique_search_truncated=(
                expected_clique_search_truncated
            ),
            expected_schema=expected_telemetry_schema,
            deadline=deadline_value,
        ):
            return False
        _check_parent_size(
            hz, caps=caps, deadline=deadline_value
        )
        if sparse_hz_semantic_digest(hz) != parent_digest:
            return False
        verified_cliques = tuple(
            (
                clique.clique_id,
                tuple(
                    (
                        literal.stable_bcol_id,
                        literal.phase,
                        literal.binding_digest,
                    )
                    for literal in clique.literals
                ),
            )
            for clique in expected_cliques
        )
        cut_semantic_digest = (
            None
            if reconstructed is None
            else sparse_hz_semantic_digest(reconstructed)
        )
        caps_tuple = tuple(sorted(_caps_payload(caps).items()))
        verified_result_digest = _canonical_sha256(
            {
                "schema": (
                    "act.operator_exact_relu_phase_clique_verified_result.v1"
                ),
                "status": expected_status,
                "parent_semantic_digest": parent_digest,
                "focused_property_digest": (
                    trusted_selection.property_digest
                ),
                "selection_digest": (
                    trusted_selection.selection_digest
                ),
                "subset_binding_digest": subset_digest,
                "ordered_source_frame_sha256": (
                    source_frame_digest
                ),
                "ranked_phases": _ranked_payload(ranked),
                "pair_statuses": tuple(
                    (
                        record.status,
                        record.certificate_sha256,
                    )
                    for record in trusted_records
                ),
                "certificate_sha256s": tuple(
                    certificate.certificate_sha256
                    for certificate in trusted_certificates
                ),
                "verified_cliques": verified_cliques,
                "cut_semantic_digest": cut_semantic_digest,
                "caps": caps_tuple,
                "proof_authority": False,
            }
        )
        if _verified_core_sink is not None:
            if (
                type(_verified_core_sink) is not list
                or _verified_core_sink
            ):
                return False
            _verified_core_sink.append(
                _VerifiedCliqueCore(
                    cut_hz=reconstructed,
                    verified_cliques=verified_cliques,
                    parent_row_tags=source.parent_row_tags,
                    continuous_layer_ids=(
                        source.continuous_layer_ids
                    ),
                    full_col_ids=source.full_col_ids,
                    input_center=source.input_center,
                    input_radius=source.input_radius,
                    build_input_col_ids=(
                        source.build_input_col_ids
                    ),
                    input_layer_id=source.build.input_layer_id,
                    output_layer_id=source.build.output_layer_id,
                    assert_layer_id=source.build.assert_layer_id,
                    original_parent_n_ub=hz_layout.n_ub,
                    parent_semantic_digest=parent_digest,
                    ordered_source_frame_sha256=(
                        source_frame_digest
                    ),
                    focused_property_digest=(
                        trusted_selection.property_digest
                    ),
                    selection_digest=(
                        trusted_selection.selection_digest
                    ),
                    subset_binding_digest=subset_digest,
                    verified_result_digest=(
                        verified_result_digest
                    ),
                    caps_payload=caps_tuple,
                    materializer_source_modes=(
                        source.source_modes
                    ),
                    producer_nonempty_seal_verified=(
                        source.producer_nonempty_seal_verified
                    ),
                )
            )
        _check_deadline(deadline_value, stage="verification_complete")
        return True
    except (
        OperatorExactReLUPhaseCliqueError,
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


def verify_operator_exact_relu_phase_cliques_result(
    build: OperatorHZBuild,
    focused_rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    result: OperatorExactReLUPhaseCliqueResult,
    *,
    deadline: Optional[float] = None,
    selection_max_rivals: int = 128,
    selection_max_binaries: int = 16384,
    selection_max_work_items: int = 5_000_000,
    selection_timeout_seconds: float = 5.0,
    max_parent_variables: int = _DEFAULT_MAX_PARENT_VARIABLES,
    max_parent_rows: int = _DEFAULT_MAX_PARENT_ROWS,
    max_parent_nonzeros: int = _DEFAULT_MAX_PARENT_NONZEROS,
    max_parent_buffer_items: int = (
        _DEFAULT_MAX_PARENT_BUFFER_ITEMS
    ),
    max_top_literals: int = _DEFAULT_MAX_TOP_LITERALS,
    max_total_pairs: int = _DEFAULT_MAX_TOTAL_PAIRS,
    max_cliques: int = _DEFAULT_MAX_CLIQUES,
    max_clique_search_nodes: int = (
        _DEFAULT_MAX_CLIQUE_SEARCH_NODES
    ),
    max_source_terms: int = _DEFAULT_MAX_SOURCE_TERMS,
    max_multiplier_bits: int = _DEFAULT_MAX_MULTIPLIER_BITS,
    max_exact_bits: int = _DEFAULT_MAX_EXACT_BITS,
    max_exact_nonzeros: int = _DEFAULT_MAX_EXACT_NONZEROS,
) -> bool:
    """Diagnostic replay over private snapshots; never issues authority."""

    return _verify_operator_exact_relu_phase_cliques_impl(
        build,
        focused_rivals,
        selection,
        result,
        deadline=deadline,
        selection_max_rivals=selection_max_rivals,
        selection_max_binaries=selection_max_binaries,
        selection_max_work_items=selection_max_work_items,
        selection_timeout_seconds=selection_timeout_seconds,
        max_parent_variables=max_parent_variables,
        max_parent_rows=max_parent_rows,
        max_parent_nonzeros=max_parent_nonzeros,
        max_parent_buffer_items=max_parent_buffer_items,
        max_top_literals=max_top_literals,
        max_total_pairs=max_total_pairs,
        max_cliques=max_cliques,
        max_clique_search_nodes=max_clique_search_nodes,
        max_source_terms=max_source_terms,
        max_multiplier_bits=max_multiplier_bits,
        max_exact_bits=max_exact_bits,
        max_exact_nonzeros=max_exact_nonzeros,
    )


def _freeze_exact_hz(
    hz: SparseHZono,
    *,
    deadline: float,
) -> None:
    layout = _exact_hz_core_layout(hz)
    for index, (_name, value) in enumerate(layout.dense):
        if index % 4 == 0:
            _check_deadline(deadline, stage="freeze_snapshot_dense")
        value.setflags(write=False)
    for index, item in enumerate(layout.sparse):
        if index % 2 == 0:
            _check_deadline(deadline, stage="freeze_snapshot_sparse")
        for value in item[3:]:
            value.setflags(write=False)
    _check_deadline(deadline, stage="after_snapshot_freeze")


def _hash_snapshot_array(
    digest: "hashlib._Hash",
    *,
    name: str,
    value: np.ndarray,
    deadline: float,
) -> None:
    digest.update(name.encode("ascii") + b"\0")
    digest.update(value.dtype.str.encode("ascii") + b"\0")
    digest.update(
        _canonical_bytes(
            {
                "shape": tuple(int(item) for item in value.shape),
                "writeable": bool(value.flags.writeable),
            }
        )
    )
    raw = value.view(np.uint8).reshape(-1)
    for start in range(0, int(raw.size), 1 << 20):
        _check_deadline(deadline, stage="snapshot_digest_array")
        digest.update(
            memoryview(raw[start : start + (1 << 20)])
        )


def _verified_snapshot_digest(
    core: _VerifiedCliqueCore,
    *,
    cut_semantic_digest: str,
    deadline: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        b"act.operator_exact_relu_phase_clique_snapshot.v1\0"
    )
    digest.update(
        _canonical_bytes(
            {
                "verified_cliques": core.verified_cliques,
                "parent_row_tags": core.parent_row_tags,
                "input_layer_id": core.input_layer_id,
                "output_layer_id": core.output_layer_id,
                "assert_layer_id": core.assert_layer_id,
                "original_parent_n_ub": (
                    core.original_parent_n_ub
                ),
                "parent_semantic_digest": (
                    core.parent_semantic_digest
                ),
                "ordered_source_frame_sha256": (
                    core.ordered_source_frame_sha256
                ),
                "focused_property_digest": (
                    core.focused_property_digest
                ),
                "selection_digest": core.selection_digest,
                "subset_binding_digest": (
                    core.subset_binding_digest
                ),
                "verified_result_digest": (
                    core.verified_result_digest
                ),
                "caps_payload": core.caps_payload,
                "materializer_source_modes": (
                    core.materializer_source_modes
                ),
                "producer_nonempty_seal_verified": (
                    core.producer_nonempty_seal_verified
                ),
                "cut_semantic_digest": cut_semantic_digest,
                "proof_authority": False,
            }
        )
    )
    for name, value in (
        ("continuous_layer_ids", core.continuous_layer_ids),
        ("full_col_ids", core.full_col_ids),
        ("input_center", core.input_center),
        ("input_radius", core.input_radius),
        ("build_input_col_ids", core.build_input_col_ids),
    ):
        _hash_snapshot_array(
            digest,
            name=name,
            value=value,
            deadline=deadline,
        )
    _check_deadline(deadline, stage="after_snapshot_digest")
    return digest.hexdigest()


def verify_consumed_operator_phase_clique_snapshot_integrity(
    snapshot: Any,
    *,
    expected_snapshot_digest: Any,
    deadline: float,
) -> bool:
    """Recompute a consumed snapshot seal from its live private buffers.

    The expected digest must be copied from the registry-validated one-use
    capability *before* consumption.  In particular, this routine never
    treats the mutable/frozen-dataclass fields on ``snapshot`` as an authority:
    ``object.__setattr__`` and NumPy ``setflags`` can change those fields after
    the registry entry has been popped.
    """

    try:
        deadline_value = _normalize_deadline(deadline)
        _check_deadline(
            deadline_value,
            stage="before_consumed_snapshot_integrity_replay",
        )
        if (
            type(snapshot) is not VerifiedOperatorPhaseCliqueSnapshot
            or not _valid_sha256(expected_snapshot_digest)
            or snapshot.snapshot_digest != expected_snapshot_digest
            or type(snapshot.producer_nonempty_seal_verified) is not bool
        ):
            return False
        live_core = _VerifiedCliqueCore(
            cut_hz=snapshot.cut_hz,
            verified_cliques=snapshot.verified_cliques,
            parent_row_tags=snapshot.parent_row_tags,
            continuous_layer_ids=snapshot.continuous_layer_ids,
            full_col_ids=snapshot.full_col_ids,
            input_center=snapshot.input_center,
            input_radius=snapshot.input_radius,
            build_input_col_ids=snapshot.build_input_col_ids,
            input_layer_id=snapshot.input_layer_id,
            output_layer_id=snapshot.output_layer_id,
            assert_layer_id=snapshot.assert_layer_id,
            original_parent_n_ub=snapshot.original_parent_n_ub,
            parent_semantic_digest=snapshot.parent_semantic_digest,
            ordered_source_frame_sha256=(
                snapshot.ordered_source_frame_sha256
            ),
            focused_property_digest=snapshot.focused_property_digest,
            selection_digest=snapshot.selection_digest,
            subset_binding_digest=snapshot.subset_binding_digest,
            verified_result_digest=snapshot.verified_result_digest,
            caps_payload=snapshot.caps_payload,
            materializer_source_modes=(
                snapshot.materializer_source_modes
            ),
            producer_nonempty_seal_verified=(
                snapshot.producer_nonempty_seal_verified
            ),
        )
        if type(live_core.cut_hz) is not SparseHZono:
            return False
        observed_cut_digest = sparse_hz_semantic_digest(
            live_core.cut_hz
        )
        observed_snapshot_digest = _verified_snapshot_digest(
            live_core,
            cut_semantic_digest=observed_cut_digest,
            deadline=deadline_value,
        )
        _check_deadline(
            deadline_value,
            stage="after_consumed_snapshot_integrity_replay",
        )
        return observed_snapshot_digest == expected_snapshot_digest
    except (
        OperatorExactReLUPhaseCliqueError,
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


def _sweep_snapshot_registry_locked(now: float) -> None:
    process_id = os.getpid()
    stale = tuple(
        token
        for token, record in _SNAPSHOT_REGISTRY.items()
        if (
            record.process_id != process_id
            or record.expires_monotonic <= now
            or record.capability_ref() is None
        )
    )
    for token in stale:
        _SNAPSHOT_REGISTRY.pop(token, None)


def _exact_original_hz_feasible_candidate(
    hz: SparseHZono,
    base_candidate: np.ndarray,
    *,
    caps: OperatorPhaseCliqueCaps,
    deadline: float,
) -> Tuple[bool, str]:
    """Replay a generated ``[xi_c, z]`` point in original HZ semantics."""

    _check_deadline(deadline, stage="before_original_hz_witness_replay")
    try:
        layout = _exact_hz_core_layout(hz)
    except OperatorExactReLUPhaseCliqueError:
        return False, "layout"
    dense = dict(layout.dense)
    sparse = {item[0]: item for item in layout.sparse}
    constraint_names = ("Ac", "Ab", "Auc", "Aub")
    constraint_nonzeros = sum(
        int(sparse[name][3].size)
        for name in constraint_names
    )
    if constraint_nonzeros > caps.max_exact_nonzeros:
        return (
            False,
            "exact_budget:"
            f"{constraint_nonzeros}>{caps.max_exact_nonzeros}",
        )

    try:
        candidate = np.asarray(
            base_candidate, dtype=np.float64
        ).reshape(-1).copy()
    except (TypeError, ValueError):
        return False, "candidate_conversion"
    n_variables = layout.n_cont + layout.n_bin
    if (
        candidate.size != n_variables
        or not np.all(np.isfinite(candidate))
    ):
        return False, "candidate_shape_or_finiteness"
    continuous = candidate[: layout.n_cont]
    if np.any(continuous < -1.0) or np.any(continuous > 1.0):
        return False, "continuous_bounds"
    binary_z = np.rint(candidate[layout.n_cont :])
    if np.any((binary_z != 0.0) & (binary_z != 1.0)):
        return False, "binary_integrality"
    binary = 2.0 * binary_z - 1.0

    for name in constraint_names:
        (
            _matrix_name,
            _matrix,
            shape,
            data,
            indices,
            indptr,
        ) = sparse[name]
        if (
            (data.size and not np.all(np.isfinite(data)))
            or int(indptr[0]) != 0
            or int(indptr[-1]) != int(data.size)
            or np.any(indptr[1:] < indptr[:-1])
            or (
                indices.size
                and (
                    np.any(indices < 0)
                    or np.any(indices >= shape[1])
                )
            )
        ):
            return False, f"{name}_csr_invalid"
        _check_deadline(
            deadline, stage="original_hz_witness_csr_audit"
        )
    if (
        not np.all(np.isfinite(dense["b"]))
        or not np.all(np.isfinite(dense["ub"]))
    ):
        return False, "rhs_nonfinite"

    try:
        continuous_fraction = tuple(
            Fraction.from_float(float(value))
            for value in continuous
        )
        binary_fraction = tuple(
            Fraction(int(value), 1)
            for value in binary
        )
    except (OverflowError, ValueError, ZeroDivisionError):
        return False, "candidate_exact_arithmetic"

    def within_bit_cap(value: Fraction) -> bool:
        return (
            abs(value.numerator).bit_length()
            <= caps.max_exact_bits
            and value.denominator.bit_length()
            <= caps.max_exact_bits
        )

    if any(
        not within_bit_cap(value)
        for value in continuous_fraction
    ):
        return False, "candidate_exact_bit_cap"

    terms_seen = 0

    def replay_rows(
        continuous_name: str,
        binary_name: str,
        rhs_name: str,
        *,
        equality: bool,
    ) -> Tuple[bool, str]:
        nonlocal terms_seen
        continuous_item = sparse[continuous_name]
        binary_item = sparse[binary_name]
        continuous_data = continuous_item[3]
        continuous_indices = continuous_item[4]
        continuous_indptr = continuous_item[5]
        binary_data = binary_item[3]
        binary_indices = binary_item[4]
        binary_indptr = binary_item[5]
        rhs = dense[rhs_name]
        for row in range(int(rhs.size)):
            if row % 128 == 0:
                _check_deadline(
                    deadline,
                    stage="original_hz_witness_row_replay",
                )
            total = Fraction(0)
            c_start = int(continuous_indptr[row])
            c_stop = int(continuous_indptr[row + 1])
            for position in range(c_start, c_stop):
                coefficient = Fraction.from_float(
                    float(continuous_data[position])
                )
                term = (
                    coefficient
                    * continuous_fraction[
                        int(continuous_indices[position])
                    ]
                )
                total += term
                terms_seen += 1
                if terms_seen % 4096 == 0:
                    _check_deadline(
                        deadline,
                        stage="original_hz_witness_term_replay",
                    )
                if (
                    not within_bit_cap(coefficient)
                    or not within_bit_cap(term)
                    or not within_bit_cap(total)
                ):
                    return False, "row_exact_bit_cap"
            b_start = int(binary_indptr[row])
            b_stop = int(binary_indptr[row + 1])
            for position in range(b_start, b_stop):
                coefficient = Fraction.from_float(
                    float(binary_data[position])
                )
                term = (
                    coefficient
                    * binary_fraction[
                        int(binary_indices[position])
                    ]
                )
                total += term
                terms_seen += 1
                if terms_seen % 4096 == 0:
                    _check_deadline(
                        deadline,
                        stage="original_hz_witness_term_replay",
                    )
                if (
                    not within_bit_cap(coefficient)
                    or not within_bit_cap(term)
                    or not within_bit_cap(total)
                ):
                    return False, "row_exact_bit_cap"
            target = Fraction.from_float(float(rhs[row]))
            if not within_bit_cap(target):
                return False, "rhs_exact_bit_cap"
            if equality:
                if total != target:
                    return False, f"eq_row_{row}"
            elif total > target:
                return False, f"ub_row_{row}"
        return True, "exact"

    try:
        equality_ok, reason = replay_rows(
            "Ac", "Ab", "b", equality=True
        )
        if not equality_ok:
            return False, reason
        upper_ok, reason = replay_rows(
            "Auc", "Aub", "ub", equality=False
        )
    except (OverflowError, ValueError, ZeroDivisionError, IndexError):
        return False, "row_exact_arithmetic"
    _check_deadline(deadline, stage="after_original_hz_witness_replay")
    return upper_ok, reason


def _fixed_binary_exact_private_witness(
    witness_hz: SparseHZono,
    binary_assignment: np.ndarray,
    *,
    caps: OperatorPhaseCliqueCaps,
    deadline: float,
) -> bool:
    """Generate an LP point, then authorize it only by exact dyadic replay."""

    _check_deadline(deadline, stage="before_fixed_binary_witness")
    try:
        layout = _exact_hz_core_layout(witness_hz)
    except OperatorExactReLUPhaseCliqueError:
        return False
    dense = dict(layout.dense)
    sparse = {
        item[0]: item[1] for item in layout.sparse
    }
    n_continuous = layout.n_cont
    n_binary = layout.n_bin
    binary = np.asarray(
        binary_assignment, dtype=np.float64
    ).reshape(-1)
    if (
        binary.size != n_binary
        or not np.all(np.isfinite(binary))
        or np.any((binary != 0.0) & (binary != 1.0))
    ):
        return False

    zero_continuous = np.zeros(
        n_continuous, dtype=np.float64
    )
    direct = np.concatenate((zero_continuous, binary))
    exact, reason = _exact_original_hz_feasible_candidate(
        witness_hz,
        direct,
        caps=caps,
        deadline=deadline,
    )
    if exact:
        return True
    if (
        type(reason) is str
        and reason.startswith("exact_budget:")
    ):
        return False
    if n_continuous == 0:
        return False

    _check_deadline(deadline, stage="before_fixed_binary_lp_build")
    binary_xi = 2.0 * binary - 1.0
    equality_continuous = sparse["Ac"].tocsr()
    upper_continuous = sparse["Auc"].tocsr()
    equality_rhs = (
        dense["b"]
        - np.asarray(
            sparse["Ab"] @ binary_xi,
            dtype=np.float64,
        ).reshape(-1)
    )
    upper_rhs = (
        dense["ub"]
        - np.asarray(
            sparse["Aub"] @ binary_xi,
            dtype=np.float64,
        ).reshape(-1)
    )
    if (
        not np.all(np.isfinite(equality_rhs))
        or not np.all(np.isfinite(upper_rhs))
    ):
        return False
    _check_deadline(
        deadline, stage="after_fixed_binary_rhs_build"
    )

    inequality_blocks = []
    inequality_rhs = []
    if layout.n_ub:
        inequality_blocks.append(
            sp.hstack(
                (
                    upper_continuous,
                    sp.csr_matrix(
                        np.ones(
                            (layout.n_ub, 1),
                            dtype=np.float64,
                        )
                    ),
                ),
                format="csr",
            )
        )
        inequality_rhs.append(upper_rhs)

    identity = sp.eye(
        n_continuous, dtype=np.float64, format="csr"
    )
    bound_slack = sp.csr_matrix(
        np.ones((n_continuous, 1), dtype=np.float64)
    )
    inequality_blocks.extend(
        (
            sp.hstack(
                (identity, bound_slack), format="csr"
            ),
            sp.hstack(
                (-identity, bound_slack), format="csr"
            ),
        )
    )
    inequality_rhs.extend(
        (
            np.ones(n_continuous, dtype=np.float64),
            np.ones(n_continuous, dtype=np.float64),
        )
    )
    A_ub = sp.vstack(inequality_blocks, format="csr")
    b_ub = np.concatenate(inequality_rhs)
    _check_deadline(deadline, stage="after_fixed_binary_ub_build")

    A_eq = None
    b_eq = None
    if layout.n_eq:
        A_eq = sp.hstack(
            (
                equality_continuous,
                sp.csr_matrix(
                    (layout.n_eq, 1), dtype=np.float64
                ),
            ),
            format="csr",
        )
        b_eq = equality_rhs
    _check_deadline(deadline, stage="after_fixed_binary_eq_build")

    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        return False
    objective = np.zeros(
        n_continuous + 1, dtype=np.float64
    )
    objective[-1] = -1.0
    augmented_lb = np.concatenate(
        (
            -np.ones(n_continuous, dtype=np.float64),
            np.array([0.0]),
        )
    )
    augmented_ub = np.concatenate(
        (
            np.ones(n_continuous, dtype=np.float64),
            np.array([np.inf]),
        )
    )
    candidate = linprog(
        objective,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=np.column_stack((augmented_lb, augmented_ub)),
        method="highs",
        options={"time_limit": remaining},
    )
    _check_deadline(deadline, stage="after_fixed_binary_lp")
    if (
        not bool(candidate.success)
        or candidate.x is None
    ):
        return False
    repaired = np.concatenate(
        (
            np.asarray(
                candidate.x[:n_continuous],
                dtype=np.float64,
            ),
            binary,
        )
    )
    exact, _reason = _exact_original_hz_feasible_candidate(
        witness_hz,
        repaired,
        caps=caps,
        deadline=deadline,
    )
    _check_deadline(deadline, stage="after_fixed_binary_exact_replay")
    return exact


def _generated_exact_private_witness(
    witness_hz: SparseHZono,
    *,
    caps: OperatorPhaseCliqueCaps,
    deadline: float,
) -> bool:
    """Generate a bounded candidate; exact replay is the only authority."""

    _check_deadline(deadline, stage="before_exact_witness_generation")
    try:
        A, rl, ru, lb, ub, integrality = (
            _base_milp_matrices(witness_hz)
        )
    except Exception:
        return False
    A = sp.csr_matrix(A, dtype=np.float64)
    integrality = np.asarray(
        integrality, dtype=int
    ).reshape(-1)
    n_binary = int(
        np.count_nonzero(
            np.asarray(integrality, dtype=bool)
        )
    )
    if int(A.nnz) > caps.max_exact_nonzeros:
        return False

    if n_binary <= 8:
        for assignment_index in range(1 << n_binary):
            _check_deadline(
                deadline, stage="exact_witness_enumeration"
            )
            binary = np.fromiter(
                (
                    float((assignment_index >> bit) & 1)
                    for bit in range(n_binary)
                ),
                dtype=np.float64,
                count=n_binary,
            )
            if _fixed_binary_exact_private_witness(
                witness_hz,
                binary,
                caps=caps,
                deadline=deadline,
            ):
                return True
        return False

    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        return False
    try:
        candidate = milp(
            np.zeros(A.shape[1], dtype=np.float64),
            integrality=integrality,
            bounds=ScipyBounds(lb, ub),
            constraints=ScipyLinearConstraint(A, rl, ru),
            options={
                "presolve": True,
                "time_limit": remaining,
            },
        )
    except (TypeError, ValueError, RuntimeError):
        return False
    _check_deadline(deadline, stage="after_exact_witness_milp")
    if candidate.x is None:
        return False
    generated = np.asarray(
        candidate.x, dtype=np.float64
    ).reshape(-1)
    exact, reason = _exact_original_hz_feasible_candidate(
        witness_hz,
        generated,
        caps=caps,
        deadline=deadline,
    )
    if exact:
        return True
    if (
        type(reason) is str
        and reason.startswith("exact_budget:")
    ):
        return False
    integer_mask = integrality.astype(bool)
    n_continuous = int(A.shape[1]) - n_binary
    if (
        np.any(integer_mask[:n_continuous])
        or np.any(~integer_mask[n_continuous:])
    ):
        return False
    binary = np.rint(generated[n_continuous:])
    if np.any((binary != 0.0) & (binary != 1.0)):
        return False
    return _fixed_binary_exact_private_witness(
        witness_hz,
        binary,
        caps=caps,
        deadline=deadline,
    )


def _cut_has_exact_private_nonempty_witness(
    cut_hz: SparseHZono,
    *,
    caps: OperatorPhaseCliqueCaps,
    deadline: float,
) -> bool:
    """Independently prove the immutable cut snapshot is not vacuous."""

    _check_deadline(deadline, stage="before_cut_nonempty_witness")
    witness_hz = _snapshot_sparse_hz(
        cut_hz,
        caps=caps,
        deadline=deadline,
        stage="cut_nonempty_witness",
    )
    return _generated_exact_private_witness(
        witness_hz,
        caps=caps,
        deadline=deadline,
    )


def verify_and_issue_operator_phase_clique_snapshot(
    build: OperatorHZBuild,
    focused_rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    result: OperatorExactReLUPhaseCliqueResult,
    *,
    deadline: Optional[float] = None,
    capability_ttl_seconds: float = (
        _SNAPSHOT_DEFAULT_TTL_SECONDS
    ),
    selection_max_rivals: int = 128,
    selection_max_binaries: int = 16384,
    selection_max_work_items: int = 5_000_000,
    selection_timeout_seconds: float = 5.0,
    max_parent_variables: int = _DEFAULT_MAX_PARENT_VARIABLES,
    max_parent_rows: int = _DEFAULT_MAX_PARENT_ROWS,
    max_parent_nonzeros: int = _DEFAULT_MAX_PARENT_NONZEROS,
    max_parent_buffer_items: int = (
        _DEFAULT_MAX_PARENT_BUFFER_ITEMS
    ),
    max_top_literals: int = _DEFAULT_MAX_TOP_LITERALS,
    max_total_pairs: int = _DEFAULT_MAX_TOTAL_PAIRS,
    max_cliques: int = _DEFAULT_MAX_CLIQUES,
    max_clique_search_nodes: int = (
        _DEFAULT_MAX_CLIQUE_SEARCH_NODES
    ),
    max_source_terms: int = _DEFAULT_MAX_SOURCE_TERMS,
    max_multiplier_bits: int = _DEFAULT_MAX_MULTIPLIER_BITS,
    max_exact_bits: int = _DEFAULT_MAX_EXACT_BITS,
    max_exact_nonzeros: int = _DEFAULT_MAX_EXACT_NONZEROS,
) -> Optional[VerifiedOperatorPhaseCliqueCapability]:
    """Verify once and issue an owner-bound handle to the private cut."""

    try:
        deadline_value = _normalize_deadline(
            time.monotonic() + 60.0
            if deadline is None
            else deadline
        )
        if (
            type(capability_ttl_seconds) not in {int, float}
            or type(capability_ttl_seconds) is bool
        ):
            return None
        ttl = float(capability_ttl_seconds)
        if (
            not math.isfinite(ttl)
            or ttl <= 0.0
            or ttl > _SNAPSHOT_HARD_TTL_SECONDS
        ):
            return None
        sink: list[_VerifiedCliqueCore] = []
        if not _verify_operator_exact_relu_phase_cliques_impl(
            build,
            focused_rivals,
            selection,
            result,
            deadline=deadline_value,
            selection_max_rivals=selection_max_rivals,
            selection_max_binaries=selection_max_binaries,
            selection_max_work_items=selection_max_work_items,
            selection_timeout_seconds=selection_timeout_seconds,
            max_parent_variables=max_parent_variables,
            max_parent_rows=max_parent_rows,
            max_parent_nonzeros=max_parent_nonzeros,
            max_parent_buffer_items=max_parent_buffer_items,
            max_top_literals=max_top_literals,
            max_total_pairs=max_total_pairs,
            max_cliques=max_cliques,
            max_clique_search_nodes=max_clique_search_nodes,
            max_source_terms=max_source_terms,
            max_multiplier_bits=max_multiplier_bits,
            max_exact_bits=max_exact_bits,
            max_exact_nonzeros=max_exact_nonzeros,
            _verified_core_sink=sink,
        ):
            return None
        if len(sink) != 1:
            return None
        core = sink[0]
        cut_hz = core.cut_hz
        if type(cut_hz) is not SparseHZono:
            return None
        cut_layout = _exact_hz_core_layout(cut_hz)
        if (
            not core.verified_cliques
            or cut_layout.n_ub
            != core.original_parent_n_ub
            + len(core.verified_cliques)
            or len(core.parent_row_tags)
            != cut_layout.n_eq + core.original_parent_n_ub
        ):
            return None
        caps_object = OperatorPhaseCliqueCaps(
            **dict(core.caps_payload)
        )
        if core.producer_nonempty_seal_verified is not True:
            if not _cut_has_exact_private_nonempty_witness(
                cut_hz,
                caps=caps_object,
                deadline=deadline_value,
            ):
                return None
        # A validated producer seal proves the exact private parent nonempty.
        # Every accepted edge was replayed over that complete parent, every
        # clique contains all of its exact edges, and ``cut_hz`` was rebuilt
        # as that parent plus only the resulting globally redundant rows.
        # Hence the cut and parent feasible sets are equal; no Fraction base
        # witness is needed (or permitted) on the sealed large-model path.
        for value in (
            core.continuous_layer_ids,
            core.full_col_ids,
            core.input_center,
            core.input_radius,
            core.build_input_col_ids,
        ):
            value.setflags(write=False)
        _freeze_exact_hz(cut_hz, deadline=deadline_value)
        cut_semantic_digest = sparse_hz_semantic_digest(cut_hz)
        snapshot_digest = _verified_snapshot_digest(
            core,
            cut_semantic_digest=cut_semantic_digest,
            deadline=deadline_value,
        )
        snapshot = VerifiedOperatorPhaseCliqueSnapshot(
            cut_hz=cut_hz,
            verified_cliques=core.verified_cliques,
            parent_row_tags=core.parent_row_tags,
            continuous_layer_ids=core.continuous_layer_ids,
            full_col_ids=core.full_col_ids,
            input_center=core.input_center,
            input_radius=core.input_radius,
            build_input_col_ids=core.build_input_col_ids,
            input_layer_id=core.input_layer_id,
            output_layer_id=core.output_layer_id,
            assert_layer_id=core.assert_layer_id,
            original_parent_n_ub=core.original_parent_n_ub,
            parent_semantic_digest=core.parent_semantic_digest,
            ordered_source_frame_sha256=(
                core.ordered_source_frame_sha256
            ),
            focused_property_digest=(
                core.focused_property_digest
            ),
            selection_digest=core.selection_digest,
            subset_binding_digest=core.subset_binding_digest,
            verified_result_digest=core.verified_result_digest,
            caps_payload=core.caps_payload,
            materializer_source_modes=(
                core.materializer_source_modes
            ),
            producer_nonempty_seal_verified=(
                core.producer_nonempty_seal_verified
            ),
            snapshot_digest=snapshot_digest,
        )
        now = time.monotonic()
        _check_deadline(deadline_value, stage="before_snapshot_issue")
        expires = min(now + ttl, deadline_value)
        if expires <= now:
            return None
        token = secrets.token_hex(32)
        capability = VerifiedOperatorPhaseCliqueCapability(
            token=token,
            snapshot_digest=snapshot_digest,
            expires_monotonic=expires,
        )
        with _SNAPSHOT_REGISTRY_LOCK:
            locked_now = time.monotonic()
            _sweep_snapshot_registry_locked(locked_now)
            if (
                locked_now >= deadline_value
                or locked_now >= expires
                or len(_SNAPSHOT_REGISTRY)
                >= _SNAPSHOT_REGISTRY_CAPACITY
                or token in _SNAPSHOT_REGISTRY
            ):
                return None
            _SNAPSHOT_REGISTRY[token] = _VerifiedSnapshotRecord(
                capability_ref=weakref.ref(capability),
                snapshot=snapshot,
                expires_monotonic=expires,
                process_id=os.getpid(),
            )
        return capability
    except (
        OperatorExactReLUPhaseCliqueError,
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return None


def consume_verified_operator_phase_clique_snapshot(
    capability: VerifiedOperatorPhaseCliqueCapability,
    *,
    deadline: float,
) -> VerifiedOperatorPhaseCliqueSnapshot:
    """Atomically consume exactly one owner-bound process-local snapshot."""

    deadline_value = _normalize_deadline(deadline)
    _check_deadline(deadline_value, stage="before_snapshot_consume")
    if (
        type(capability)
        is not VerifiedOperatorPhaseCliqueCapability
        or capability.proof_authority is not False
        or type(capability.token) is not str
        or len(capability.token) != 64
        or any(
            character not in "0123456789abcdef"
            for character in capability.token
        )
        or not _valid_sha256(capability.snapshot_digest)
        or type(capability.expires_monotonic) is not float
        or not math.isfinite(capability.expires_monotonic)
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "verified_snapshot_capability_malformed"
        )
    with _SNAPSHOT_REGISTRY_LOCK:
        locked_now = time.monotonic()
        _sweep_snapshot_registry_locked(locked_now)
        record = _SNAPSHOT_REGISTRY.get(capability.token)
        if (
            record is None
            or record.process_id != os.getpid()
            or record.capability_ref() is not capability
            or record.expires_monotonic
            != capability.expires_monotonic
            or record.snapshot.snapshot_digest
            != capability.snapshot_digest
            or record.expires_monotonic <= locked_now
            or deadline_value <= locked_now
        ):
            raise OperatorExactReLUPhaseCliqueError(
                "verified_snapshot_capability_invalid"
            )
        # Pop while holding the lock: two consumers can never receive aliases.
        _SNAPSHOT_REGISTRY.pop(capability.token)
        snapshot = record.snapshot
    final_now = time.monotonic()
    if (
        final_now >= deadline_value
        or final_now >= capability.expires_monotonic
    ):
        raise OperatorExactReLUPhaseCliqueError(
            "verified_snapshot_capability_expired_during_consume"
        )
    return snapshot


__all__ = [
    "OperatorCertifiedPhaseClique",
    "OperatorExactReLUPhaseCliqueError",
    "OperatorExactReLUPhaseCliqueResult",
    "OperatorPhaseCliqueCaps",
    "RankedOperatorPhase",
    "VerifiedOperatorPhaseCliqueCapability",
    "VerifiedOperatorPhaseCliqueSnapshot",
    "consume_verified_operator_phase_clique_snapshot",
    "run_operator_exact_relu_phase_cliques_candidate",
    "verify_consumed_operator_phase_clique_snapshot_integrity",
    "verify_and_issue_operator_phase_clique_snapshot",
    "verify_operator_exact_relu_phase_cliques_result",
]
