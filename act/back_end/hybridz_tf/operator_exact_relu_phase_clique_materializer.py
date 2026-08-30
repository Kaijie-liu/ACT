#!/usr/bin/env python3
"""Fresh, default-off materialization of verified Operator-HZ clique cuts.

The exact phase-clique search is deliberately candidate-only.  Its returned
``hz`` object is useful for controlled tightness experiments, but it is not a
production object: arbitrary Python attributes on that candidate must never
be allowed to carry solver caches, receipts, or verdict capabilities into a
later solve.

This module is the narrow bridge between that candidate and a fresh
``OperatorHZBuild``:

* the public hardened clique-result verifier must pass under the caller's
  original absolute deadline and exact caps;
* cut rows are reconstructed from verified stable binary IDs and phases, not
  copied from ``result.hz``;
* the uniquely consumed, fully verified, read-only cut snapshot becomes the
  public candidate without another full-core copy;
* parent replay uses a strict read-only row-prefix view of that snapshot;
* only an explicit witness/provenance whitelist is bound from the private
  verifier snapshot;
* incompatible conditional, prefix, micro-RLT, query-dual, full-input replay,
  and property-upper modes fail closed in this first version; and
* constructive non-emptiness is reissued only on an unexposed, frozen solver
  copy after verification, because exact conflict-clique rows are redundant
  for integer binary assignments; the public candidate carries no such token.

The materialization receipt remains candidate-only and permanently has
``proof_authority=False``.  It cannot authorize SAFE.  Only the one-use
private solver handoff may be decided by
:func:`act.back_end.solver.solver_hz.hz_objbound_decide`.
The default-off wrapper returns the original ``OperatorHZBuild`` by identity
before inspecting the selection, candidate, deadline, or caps.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
import secrets
import threading
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple, Union
import weakref

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    RivalSpec,
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_cliques import (
    OperatorExactReLUPhaseCliqueResult,
    OperatorPhaseCliqueCaps,
    VerifiedOperatorPhaseCliqueSnapshot,
    _freeze_exact_hz,
    _snapshot_sparse_hz,
    consume_verified_operator_phase_clique_snapshot,
    verify_consumed_operator_phase_clique_snapshot_integrity,
    verify_and_issue_operator_phase_clique_snapshot,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
    OperatorExactReLUPhaseSelection,
    OperatorExactReLUPhaseSelectionCaps,
)
from act.back_end.hybridz_tf.operator_hz import (
    OperatorHZBuild,
    OperatorHZConstructiveNonemptySeal,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    hz_constructively_nonempty,
    hz_mark_constructively_nonempty,
)


class OperatorPhaseCliqueMaterializationError(ValueError):
    """A verified cut could not be safely rebound to a fresh Operator-HZ."""


_SOLVER_HANDOFF_PRODUCER = object()
_SOLVER_HANDOFF_REGISTRY_LOCK = threading.Lock()
_SOLVER_HANDOFF_REGISTRY_CAPACITY = 64
_SOLVER_HANDOFF_TTL_SECONDS = 60.0


class OperatorPhaseCliqueMaterializationSolverCapability:
    """Opaque one-use handle for an unexposed, frozen solver snapshot."""

    __slots__ = (
        "_token",
        "_process_id",
        "_expires_monotonic",
        "_fresh_semantic_digest",
        "__weakref__",
    )

    def __init__(
        self,
        *,
        token: str,
        process_id: int,
        expires_monotonic: float,
        fresh_semantic_digest: str,
        _producer_capability: Any,
    ) -> None:
        if _producer_capability is not _SOLVER_HANDOFF_PRODUCER:
            raise PermissionError(
                "solver handoff capability requires its materializer"
            )
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_process_id", process_id)
        object.__setattr__(
            self, "_expires_monotonic", expires_monotonic
        )
        object.__setattr__(
            self,
            "_fresh_semantic_digest",
            fresh_semantic_digest,
        )

    @property
    def token(self) -> str:
        return self._token

    @property
    def process_id(self) -> int:
        return self._process_id

    @property
    def expires_monotonic(self) -> float:
        return self._expires_monotonic

    @property
    def fresh_semantic_digest(self) -> str:
        return self._fresh_semantic_digest

    @property
    def proof_authority(self) -> bool:
        # This capability authorizes an ownership transfer, never a verdict.
        return False

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise TypeError("solver handoff capabilities are immutable")

    def __copy__(self):
        # A copied object deliberately has no registry identity.
        copied = object.__new__(type(self))
        for name in (
            "_token",
            "_process_id",
            "_expires_monotonic",
            "_fresh_semantic_digest",
        ):
            object.__setattr__(
                copied, name, object.__getattribute__(self, name)
            )
        return copied

    def __deepcopy__(self, _memo):
        return self.__copy__()


@dataclass(frozen=True)
class OperatorPhaseCliqueMaterialization:
    """A fresh build and a diagnostic, non-authoritative receipt."""

    build: OperatorHZBuild
    parent_semantic_digest: str
    fresh_semantic_digest: str
    source_frame_digest: str
    fresh_frame_digest: str
    clique_ids: Tuple[str, ...]
    cut_row_tags: Tuple[str, ...]
    receipt: Mapping[str, Any]
    solver_handoff_capability: (
        OperatorPhaseCliqueMaterializationSolverCapability
    )
    proof_authority: bool = False

    def __post_init__(self) -> None:
        if self.proof_authority is not False:
            raise ValueError(
                "clique materialization never has proof authority"
            )
        if (
            type(self.solver_handoff_capability)
            is not OperatorPhaseCliqueMaterializationSolverCapability
        ):
            raise ValueError(
                "clique materialization solver handoff is malformed"
            )
        object.__setattr__(
            self,
            "receipt",
            MappingProxyType(copy.deepcopy(dict(self.receipt))),
        )


@dataclass(frozen=True)
class _SolverHandoffRecord:
    capability_ref: "weakref.ReferenceType[OperatorPhaseCliqueMaterializationSolverCapability]"
    materialization_ref: "weakref.ReferenceType[OperatorPhaseCliqueMaterialization]"
    public_build_ref: "weakref.ReferenceType[OperatorHZBuild]"
    public_hz_ref: "weakref.ReferenceType[SparseHZono]"
    public_owned_identity: Tuple[int, ...]
    private_build: OperatorHZBuild
    private_owned_identity: Tuple[int, ...]
    fresh_semantic_digest: str
    expires_monotonic: float
    process_id: int


_SOLVER_HANDOFF_REGISTRY: dict[str, _SolverHandoffRecord] = {}


_CAP_FIELDS = (
    "max_parent_variables",
    "max_parent_rows",
    "max_parent_nonzeros",
    "max_parent_buffer_items",
    "max_top_literals",
    "max_total_pairs",
    "max_cliques",
    "max_clique_search_nodes",
    "max_source_terms",
    "max_multiplier_bits",
    "max_exact_bits",
    "max_exact_nonzeros",
)
_PROVENANCE_ATTRS = (
    "full_col_ids",
    "operator_input_center",
    "operator_input_radius",
    "_solver_continuous_column_layer_ids",
)
_CONSTRUCTIVE_REASON = (
    "operator_hz_redundant_exact_integer_phase_clique_cuts_v1"
)
_CUT_TAG_PREFIX = "operator_exact_relu_phase_clique_cut:v1"
_MAX_TAG_BYTES = 4096
_SPARSE_HZ_CORE_NAMES = frozenset(
    {
        "c",
        "Gc",
        "Gb",
        "Ac",
        "Ab",
        "b",
        "Auc",
        "Aub",
        "ub",
        "col_ids",
        "bcol_ids",
    }
)
_SPARSE_HZ_DENSE_NAMES = (
    "c",
    "b",
    "ub",
    "col_ids",
    "bcol_ids",
)
_SPARSE_HZ_CSR_NAMES = (
    "Gc",
    "Gb",
    "Ac",
    "Ab",
    "Auc",
    "Aub",
)
_EXPECTED_SOURCE_MODES = tuple(
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


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _normalize_deadline(value: Any) -> float:
    if type(value) not in {int, float} or type(value) is bool:
        raise OperatorPhaseCliqueMaterializationError(
            "deadline_not_builtin_numeric"
        )
    deadline = float(value)
    if not math.isfinite(deadline):
        raise OperatorPhaseCliqueMaterializationError(
            "deadline_nonfinite"
        )
    return deadline


def _check_deadline(deadline: float, *, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise OperatorPhaseCliqueMaterializationError(
            f"deadline_expired_{stage}"
        )


def _caps_payload(caps: Any) -> Mapping[str, int]:
    if type(caps) is not OperatorPhaseCliqueCaps:
        raise OperatorPhaseCliqueMaterializationError(
            "clique_caps_wrong_type"
        )
    payload = {}
    for name in _CAP_FIELDS:
        value = getattr(caps, name)
        if type(value) is not int or value < 1:
            raise OperatorPhaseCliqueMaterializationError(
                f"clique_cap_invalid_{name}"
            )
        payload[name] = value
    return payload


def _selection_caps(
    selection: Any,
) -> OperatorExactReLUPhaseSelectionCaps:
    if type(selection) is not OperatorExactReLUPhaseSelection:
        raise OperatorPhaseCliqueMaterializationError(
            "selection_wrong_type"
        )
    caps = selection.caps
    if type(caps) is not OperatorExactReLUPhaseSelectionCaps:
        raise OperatorPhaseCliqueMaterializationError(
            "selection_caps_wrong_type"
        )
    if (
        type(caps.max_rivals) is not int
        or caps.max_rivals < 1
        or type(caps.max_binaries) is not int
        or caps.max_binaries < 1
        or type(caps.max_work_items) is not int
        or caps.max_work_items < 1
        or type(caps.timeout_seconds) is not float
        or not math.isfinite(caps.timeout_seconds)
        or caps.timeout_seconds <= 0.0
    ):
        raise OperatorPhaseCliqueMaterializationError(
            "selection_caps_invalid"
        )
    return caps


def _exact_array(
    value: Any,
    *,
    name: str,
    dtype: np.dtype,
    length: int,
) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(dtype)
        or value.ndim != 1
        or int(value.size) != int(length)
        or not value.flags.c_contiguous
    ):
        raise OperatorPhaseCliqueMaterializationError(
            f"source_array_malformed_{name}"
        )
    if np.issubdtype(value.dtype, np.floating) and not np.all(
        np.isfinite(value)
    ):
        raise OperatorPhaseCliqueMaterializationError(
            f"source_array_nonfinite_{name}"
        )
    return value


def _exact_csr(
    value: Any,
    *,
    name: str,
    shape: Tuple[int, int],
) -> sp.csr_matrix:
    if (
        type(value) is not sp.csr_matrix
        or value.dtype != np.dtype(np.float64)
        or value.shape != shape
        or not value.has_canonical_format
        or (
            value.nnz
            and not np.all(np.isfinite(value.data))
        )
    ):
        raise OperatorPhaseCliqueMaterializationError(
            f"source_csr_malformed_{name}"
        )
    return value


def _validate_core(hz: Any, *, deadline: float) -> None:
    _check_deadline(deadline, stage="before_core_validation")
    if type(hz) is not SparseHZono:
        raise OperatorPhaseCliqueMaterializationError(
            "source_not_exact_sparse_hz"
        )
    _exact_array(
        hz.c, name="c", dtype=np.float64, length=hz.n_out
    )
    _exact_array(
        hz.b, name="b", dtype=np.float64, length=hz.n_eq
    )
    _exact_array(
        hz.ub, name="ub", dtype=np.float64, length=hz.n_ub
    )
    _exact_array(
        hz.col_ids,
        name="col_ids",
        dtype=np.int64,
        length=hz.n_cont,
    )
    _exact_array(
        hz.bcol_ids,
        name="bcol_ids",
        dtype=np.int64,
        length=hz.n_bin,
    )
    _exact_csr(
        hz.Gc,
        name="Gc",
        shape=(hz.n_out, hz.n_cont),
    )
    _exact_csr(
        hz.Gb,
        name="Gb",
        shape=(hz.n_out, hz.n_bin),
    )
    _exact_csr(
        hz.Ac,
        name="Ac",
        shape=(hz.n_eq, hz.n_cont),
    )
    _exact_csr(
        hz.Ab,
        name="Ab",
        shape=(hz.n_eq, hz.n_bin),
    )
    _exact_csr(
        hz.Auc,
        name="Auc",
        shape=(hz.n_ub, hz.n_cont),
    )
    _exact_csr(
        hz.Aub,
        name="Aub",
        shape=(hz.n_ub, hz.n_bin),
    )
    _check_deadline(deadline, stage="after_core_validation")


def _require_readonly_core(
    hz: SparseHZono,
    *,
    stage: str,
    deadline: float,
) -> None:
    """Fail closed unless every semantic core buffer is immutable."""

    _check_deadline(deadline, stage=f"before_{stage}_readonly_audit")
    for name in _SPARSE_HZ_DENSE_NAMES:
        value = getattr(hz, name)
        if value.flags.writeable:
            raise OperatorPhaseCliqueMaterializationError(
                f"{stage}_dense_buffer_writeable_{name}"
            )
    for name in _SPARSE_HZ_CSR_NAMES:
        matrix = getattr(hz, name)
        for buffer_name in ("data", "indices", "indptr"):
            if getattr(matrix, buffer_name).flags.writeable:
                raise OperatorPhaseCliqueMaterializationError(
                    f"{stage}_csr_buffer_writeable_{name}_{buffer_name}"
                )
    _check_deadline(deadline, stage=f"after_{stage}_readonly_audit")


def _audit_caps(
    hz: SparseHZono,
    *,
    caps: OperatorPhaseCliqueCaps,
    provenance_items: int,
    tag_bytes: int,
    deadline: float,
) -> None:
    _check_deadline(deadline, stage="before_size_audit")
    payload = _caps_payload(caps)
    matrices = (
        hz.Gc,
        hz.Gb,
        hz.Ac,
        hz.Ab,
        hz.Auc,
        hz.Aub,
    )
    arrays = (hz.c, hz.b, hz.ub, hz.col_ids, hz.bcol_ids)
    variables = int(hz.n_cont) + int(hz.n_bin)
    rows = int(hz.n_out) + int(hz.n_eq) + int(hz.n_ub)
    nonzeros = sum(int(matrix.nnz) for matrix in matrices)
    buffer_items = (
        sum(
            int(matrix.data.size)
            + int(matrix.indices.size)
            + int(matrix.indptr.size)
            for matrix in matrices
        )
        + sum(int(array.size) for array in arrays)
        + int(provenance_items)
        + int(tag_bytes)
    )
    if (
        variables > payload["max_parent_variables"]
        or rows > payload["max_parent_rows"]
        or nonzeros > payload["max_parent_nonzeros"]
        or buffer_items > payload["max_parent_buffer_items"]
    ):
        raise OperatorPhaseCliqueMaterializationError(
            "materialization_size_cap_exceeded"
        )
    _check_deadline(deadline, stage="after_size_audit")


def _verified_snapshot_cliques(
    snapshot: VerifiedOperatorPhaseCliqueSnapshot,
    *,
    caps: OperatorPhaseCliqueCaps,
    deadline: float,
) -> Tuple[Tuple[str, Tuple[Tuple[int, int, str], ...]], ...]:
    """Validate verifier-owned builtin cut primitives without equality hooks."""

    raw = snapshot.verified_cliques
    if (
        type(raw) is not tuple
        or not raw
        or len(raw) > caps.max_cliques
    ):
        raise OperatorPhaseCliqueMaterializationError(
            "verified_snapshot_cliques_malformed"
        )
    trusted = []
    seen_clique_ids = set()
    for index, item in enumerate(raw):
        _check_deadline(
            deadline, stage="verified_snapshot_clique_scan"
        )
        if (
            type(item) is not tuple
            or len(item) != 2
            or not _valid_sha256(item[0])
            or item[0] in seen_clique_ids
            or type(item[1]) is not tuple
            or not 2 <= len(item[1]) <= caps.max_top_literals
        ):
            raise OperatorPhaseCliqueMaterializationError(
                "verified_snapshot_clique_malformed"
            )
        literals = []
        seen_stable_ids = set()
        for literal in item[1]:
            if (
                type(literal) is not tuple
                or len(literal) != 3
                or type(literal[0]) is not int
                or literal[0] < 0
                or type(literal[1]) is not int
                or literal[1] not in {-1, 1}
                or not _valid_sha256(literal[2])
                or literal[0] in seen_stable_ids
            ):
                raise OperatorPhaseCliqueMaterializationError(
                    "verified_snapshot_literal_malformed"
                )
            seen_stable_ids.add(literal[0])
            literals.append((literal[0], literal[1], literal[2]))
        seen_clique_ids.add(item[0])
        trusted.append((item[0], tuple(literals)))
        if index % 64 == 0:
            _check_deadline(
                deadline, stage="verified_snapshot_clique_scan"
            )
    return tuple(trusted)


def _validate_consumed_snapshot(
    snapshot: Any,
    *,
    caps: OperatorPhaseCliqueCaps,
    deadline: float,
) -> Tuple[
    SparseHZono,
    Tuple[Tuple[str, Tuple[Tuple[int, int, str], ...]], ...],
    Tuple[str, ...],
    Mapping[str, np.ndarray],
    int,
    int,
]:
    """Audit the unique one-use snapshot before constructing any fresh HZ."""

    _check_deadline(deadline, stage="before_consumed_snapshot_audit")
    if (
        type(snapshot) is not VerifiedOperatorPhaseCliqueSnapshot
        or snapshot.proof_authority is not False
        or type(snapshot.snapshot_digest) is not str
        or not _valid_sha256(snapshot.snapshot_digest)
        or not _valid_sha256(snapshot.parent_semantic_digest)
        or not _valid_sha256(snapshot.focused_property_digest)
        or not _valid_sha256(snapshot.selection_digest)
        or not _valid_sha256(snapshot.subset_binding_digest)
        or not _valid_sha256(snapshot.verified_result_digest)
        or (
            snapshot.ordered_source_frame_sha256 is not None
            and not _valid_sha256(
                snapshot.ordered_source_frame_sha256
            )
        )
        or type(snapshot.original_parent_n_ub) is not int
        or snapshot.original_parent_n_ub < 0
        or type(snapshot.materializer_source_modes) is not tuple
        or snapshot.materializer_source_modes
        != _EXPECTED_SOURCE_MODES
    ):
        raise OperatorPhaseCliqueMaterializationError(
            "verified_snapshot_header_malformed"
        )
    if snapshot.producer_nonempty_seal_verified is not True:
        raise OperatorPhaseCliqueMaterializationError(
            "verified_snapshot_producer_nonempty_seal_not_verified"
        )
    expected_caps = tuple(sorted(_caps_payload(caps).items()))
    if (
        type(snapshot.caps_payload) is not tuple
        or snapshot.caps_payload != expected_caps
    ):
        raise OperatorPhaseCliqueMaterializationError(
            "verified_snapshot_caps_mismatch"
        )
    cut_hz = snapshot.cut_hz
    _validate_core(cut_hz, deadline=deadline)
    _require_readonly_core(
        cut_hz,
        stage="consumed_verified_cut",
        deadline=deadline,
    )
    if set(vars(cut_hz)) != _SPARSE_HZ_CORE_NAMES:
        raise OperatorPhaseCliqueMaterializationError(
            "verified_snapshot_cut_hz_has_extra_attributes"
        )
    cliques = _verified_snapshot_cliques(
        snapshot, caps=caps, deadline=deadline
    )
    if cut_hz.n_ub != snapshot.original_parent_n_ub + len(cliques):
        raise OperatorPhaseCliqueMaterializationError(
            "verified_snapshot_cut_row_count_mismatch"
        )
    tags = snapshot.parent_row_tags
    if (
        type(tags) is not tuple
        or len(tags) != cut_hz.n_eq + snapshot.original_parent_n_ub
    ):
        raise OperatorPhaseCliqueMaterializationError(
            "verified_snapshot_parent_tags_malformed"
        )
    tag_bytes = 0
    for index, tag in enumerate(tags):
        if index % 256 == 0:
            _check_deadline(
                deadline, stage="verified_snapshot_tag_scan"
            )
        if type(tag) is not str or "\x00" in tag:
            raise OperatorPhaseCliqueMaterializationError(
                "verified_snapshot_parent_tag_malformed"
            )
        try:
            encoded = tag.encode("ascii")
        except UnicodeEncodeError as exc:
            raise OperatorPhaseCliqueMaterializationError(
                "verified_snapshot_parent_tag_nonascii"
            ) from exc
        if len(encoded) > _MAX_TAG_BYTES:
            raise OperatorPhaseCliqueMaterializationError(
                "verified_snapshot_parent_tag_too_large"
            )
        tag_bytes += len(encoded)
        if tag_bytes > caps.max_parent_buffer_items:
            raise OperatorPhaseCliqueMaterializationError(
                "verified_snapshot_parent_tag_cap_exceeded"
            )
    if any(tag.startswith("property_micro_rlt:") for tag in tags):
        raise OperatorPhaseCliqueMaterializationError(
            "verified_snapshot_micro_rlt_tag_unsupported"
        )

    for name in (
        "input_layer_id",
        "output_layer_id",
        "assert_layer_id",
    ):
        value = getattr(snapshot, name)
        if type(value) is not int or value < 0:
            raise OperatorPhaseCliqueMaterializationError(
                f"verified_snapshot_{name}_malformed"
            )
    build_input_ids = _exact_array(
        snapshot.build_input_col_ids,
        name="snapshot_build_input_col_ids",
        dtype=np.int64,
        length=int(snapshot.build_input_col_ids.size)
        if type(snapshot.build_input_col_ids) is np.ndarray
        else -1,
    )
    input_length = int(build_input_ids.size)
    full_col_ids = _exact_array(
        snapshot.full_col_ids,
        name="snapshot_full_col_ids",
        dtype=np.int64,
        length=input_length,
    )
    input_center = _exact_array(
        snapshot.input_center,
        name="snapshot_input_center",
        dtype=np.float64,
        length=input_length,
    )
    input_radius = _exact_array(
        snapshot.input_radius,
        name="snapshot_input_radius",
        dtype=np.float64,
        length=input_length,
    )
    continuous_layer_ids = _exact_array(
        snapshot.continuous_layer_ids,
        name="snapshot_continuous_layer_ids",
        dtype=np.int64,
        length=cut_hz.n_cont,
    )
    for name, value in (
        ("build_input_col_ids", build_input_ids),
        ("full_col_ids", full_col_ids),
        ("input_center", input_center),
        ("input_radius", input_radius),
        ("continuous_layer_ids", continuous_layer_ids),
    ):
        if value.flags.writeable:
            raise OperatorPhaseCliqueMaterializationError(
                f"verified_snapshot_provenance_writeable_{name}"
            )
    if (
        not np.array_equal(build_input_ids, full_col_ids)
        or np.any(input_radius < 0.0)
    ):
        raise OperatorPhaseCliqueMaterializationError(
            "verified_snapshot_input_provenance_mismatch"
        )
    provenance = {
        "full_col_ids": full_col_ids,
        "operator_input_center": input_center,
        "operator_input_radius": input_radius,
        "_solver_continuous_column_layer_ids": (
            continuous_layer_ids
        ),
    }
    provenance_items = (
        int(build_input_ids.size)
        + sum(int(value.size) for value in provenance.values())
    )
    _audit_caps(
        cut_hz,
        caps=caps,
        provenance_items=provenance_items,
        tag_bytes=tag_bytes,
        deadline=deadline,
    )
    _check_deadline(deadline, stage="after_consumed_snapshot_audit")
    return (
        cut_hz,
        cliques,
        tags,
        provenance,
        provenance_items,
        tag_bytes,
    )


def _buffers_alias_or_are_empty(
    source: np.ndarray,
    view: np.ndarray,
) -> bool:
    return (
        bool(np.shares_memory(source, view))
        or (int(source.size) == 0 and int(view.size) == 0)
    )


def _readonly_csr_row_prefix_view(
    matrix: sp.csr_matrix,
    *,
    rows: int,
    name: str,
) -> sp.csr_matrix:
    """Construct one CSR row-prefix view without copying any CSR buffer."""

    if (
        type(matrix) is not sp.csr_matrix
        or type(rows) is not int
        or rows < 0
        or rows > int(matrix.shape[0])
        or any(
            getattr(matrix, buffer_name).flags.writeable
            for buffer_name in ("data", "indices", "indptr")
        )
    ):
        raise OperatorPhaseCliqueMaterializationError(
            f"readonly_prefix_source_malformed_{name}"
        )
    stop = int(matrix.indptr[rows])
    data = matrix.data[:stop]
    indices = matrix.indices[:stop]
    indptr = matrix.indptr[: rows + 1]
    # ``copy=False`` is necessary but not trusted: exact alias checks below
    # make a SciPy coercion/copy a fail-closed event.
    view = sp.csr_matrix(
        (data, indices, indptr),
        shape=(rows, int(matrix.shape[1])),
        dtype=np.float64,
        copy=False,
    )
    for buffer_name, source in (
        ("data", matrix.data),
        ("indices", matrix.indices),
        ("indptr", matrix.indptr),
    ):
        value = getattr(view, buffer_name)
        if (
            value.flags.writeable
            or not _buffers_alias_or_are_empty(source, value)
        ):
            raise OperatorPhaseCliqueMaterializationError(
                f"readonly_prefix_copy_or_writeable_{name}_{buffer_name}"
            )
    return view


def _parent_prefix_from_verified_cut(
    cut_hz: SparseHZono,
    *,
    original_parent_n_ub: int,
    deadline: float,
) -> SparseHZono:
    """Return a semantic parent view over the unique read-only cut core."""

    _check_deadline(deadline, stage="before_parent_prefix_view")
    _validate_core(cut_hz, deadline=deadline)
    _require_readonly_core(
        cut_hz,
        stage="verified_cut",
        deadline=deadline,
    )
    if (
        type(original_parent_n_ub) is not int
        or original_parent_n_ub < 0
        or original_parent_n_ub > cut_hz.n_ub
    ):
        raise OperatorPhaseCliqueMaterializationError(
            "parent_prefix_row_count_malformed"
        )

    # Bypass SparseHZono.__post_init__: its CSR normalization copies an
    # already canonical matrix.  Every field is immediately subjected to the
    # full exact-core, read-only, alias, cap, digest, and terminal-seal gates.
    parent = object.__new__(SparseHZono)
    for name in ("c", "Gc", "Gb", "Ac", "Ab", "b", "col_ids", "bcol_ids"):
        object.__setattr__(parent, name, getattr(cut_hz, name))
    object.__setattr__(
        parent,
        "Auc",
        _readonly_csr_row_prefix_view(
            cut_hz.Auc,
            rows=original_parent_n_ub,
            name="Auc",
        ),
    )
    object.__setattr__(
        parent,
        "Aub",
        _readonly_csr_row_prefix_view(
            cut_hz.Aub,
            rows=original_parent_n_ub,
            name="Aub",
        ),
    )
    object.__setattr__(
        parent,
        "ub",
        cut_hz.ub[:original_parent_n_ub],
    )
    _validate_core(parent, deadline=deadline)
    _require_readonly_core(
        parent,
        stage="parent_prefix",
        deadline=deadline,
    )
    for name in ("c", "b", "col_ids", "bcol_ids", "ub"):
        if not _buffers_alias_or_are_empty(
            getattr(cut_hz, name), getattr(parent, name)
        ):
            raise OperatorPhaseCliqueMaterializationError(
                f"parent_prefix_dense_copy_detected_{name}"
            )
    for name in _SPARSE_HZ_CSR_NAMES:
        source = getattr(cut_hz, name)
        view = getattr(parent, name)
        for buffer_name in ("data", "indices", "indptr"):
            if not _buffers_alias_or_are_empty(
                getattr(source, buffer_name),
                getattr(view, buffer_name),
            ):
                raise OperatorPhaseCliqueMaterializationError(
                    "parent_prefix_csr_copy_detected_"
                    f"{name}_{buffer_name}"
                )
    _check_deadline(deadline, stage="after_parent_prefix_view")
    return parent


def _verify_snapshot_cut_rows(
    cut_hz: SparseHZono,
    *,
    original_parent_n_ub: int,
    cliques: Tuple[
        Tuple[str, Tuple[Tuple[int, int, str], ...]], ...
    ],
    deadline: float,
) -> None:
    stable_positions = {
        int(stable_id): position
        for position, stable_id in enumerate(
            cut_hz.bcol_ids.tolist()
        )
    }
    if len(stable_positions) != cut_hz.n_bin:
        raise OperatorPhaseCliqueMaterializationError(
            "verified_snapshot_binary_ids_not_unique"
        )
    for offset, (_clique_id, literals) in enumerate(cliques):
        _check_deadline(
            deadline, stage="verified_snapshot_cut_row_replay"
        )
        row = original_parent_n_ub + offset
        try:
            expected = tuple(
                sorted(
                    (
                        stable_positions[stable_id],
                        float(phase),
                    )
                    for stable_id, phase, _binding in literals
                )
            )
        except KeyError as exc:
            raise OperatorPhaseCliqueMaterializationError(
                "verified_snapshot_literal_missing_from_cut_hz"
            ) from exc
        if (
            cut_hz.Auc.indptr[row] != cut_hz.Auc.indptr[row + 1]
            or int(cut_hz.Aub.indptr[row + 1])
            - int(cut_hz.Aub.indptr[row])
            != len(expected)
            or float(cut_hz.ub[row])
            != float(2 - len(literals))
        ):
            raise OperatorPhaseCliqueMaterializationError(
                "verified_snapshot_cut_row_semantics_mismatch"
            )
        start = int(cut_hz.Aub.indptr[row])
        stop = int(cut_hz.Aub.indptr[row + 1])
        if (
            not np.array_equal(
                cut_hz.Aub.indices[start:stop],
                np.asarray(
                    [position for position, _phase in expected],
                    dtype=cut_hz.Aub.indices.dtype,
                ),
            )
            or not np.array_equal(
                cut_hz.Aub.data[start:stop],
                np.asarray(
                    [phase for _position, phase in expected],
                    dtype=np.float64,
                ),
            )
        ):
            raise OperatorPhaseCliqueMaterializationError(
                "verified_snapshot_cut_row_semantics_mismatch"
            )


def _source_frame_digest(
    build: OperatorHZBuild,
    *,
    tags: Tuple[str, ...],
    provenance: Mapping[str, np.ndarray],
    parent_semantic_digest: str,
    deadline: float,
) -> str:
    _check_deadline(deadline, stage="before_materializer_frame_digest")
    digest = hashlib.sha256()
    digest.update(
        b"act.operator_exact_relu_phase_clique_source_frame.v2\0"
    )
    digest.update(parent_semantic_digest.encode("ascii"))
    digest.update(
        np.asarray(
            [
                int(build.input_layer_id),
                int(build.output_layer_id),
                int(build.assert_layer_id),
            ],
            dtype=np.int64,
        ).tobytes()
    )
    digest.update(
        len(tags).to_bytes(8, "little", signed=False)
    )
    for index, tag in enumerate(tags):
        if index % 256 == 0:
            _check_deadline(
                deadline, stage="materializer_frame_tags"
            )
        encoded = tag.encode("ascii")
        digest.update(
            len(encoded).to_bytes(8, "little", signed=False)
        )
        digest.update(encoded)
    for name in _PROVENANCE_ATTRS:
        _check_deadline(
            deadline, stage="materializer_frame_provenance"
        )
        value = provenance[name]
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(
            np.asarray(value.shape, dtype=np.int64).tobytes()
        )
        raw = value.view(np.uint8).reshape(-1)
        for start in range(0, int(raw.size), 1 << 20):
            _check_deadline(
                deadline,
                stage="materializer_frame_provenance_bytes",
            )
            digest.update(
                memoryview(raw[start : start + (1 << 20)])
            )
    _check_deadline(deadline, stage="after_materializer_frame_digest")
    return digest.hexdigest()


def _fresh_frame_digest(
    build: OperatorHZBuild,
    *,
    tags: Tuple[str, ...],
    semantic_digest: str,
    deadline: float,
) -> str:
    provenance = {
        name: getattr(build.hz, name) for name in _PROVENANCE_ATTRS
    }
    return _source_frame_digest(
        build,
        tags=tags,
        provenance=provenance,
        parent_semantic_digest=semantic_digest,
        deadline=deadline,
    )


def _solver_owned_objects(build: Any) -> Tuple[Any, ...]:
    """Return every live object whose identity one handoff owns."""

    if (
        type(build) is not OperatorHZBuild
        or type(build.hz) is not SparseHZono
        or type(build.input_col_ids) is not np.ndarray
    ):
        raise OperatorPhaseCliqueMaterializationError(
            "solver_handoff_build_malformed"
        )
    hz = build.hz
    live = vars(hz)
    objects = [build, build.input_col_ids, hz]
    for name in ("c", "b", "ub", "col_ids", "bcol_ids"):
        value = live.get(name)
        if type(value) is not np.ndarray:
            raise OperatorPhaseCliqueMaterializationError(
                f"solver_handoff_dense_malformed_{name}"
            )
        objects.append(value)
    for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
        matrix = live.get(name)
        if type(matrix) is not sp.csr_matrix:
            raise OperatorPhaseCliqueMaterializationError(
                f"solver_handoff_csr_malformed_{name}"
            )
        objects.append(matrix)
        for buffer_name in ("data", "indices", "indptr"):
            value = vars(matrix).get(buffer_name)
            if type(value) is not np.ndarray:
                raise OperatorPhaseCliqueMaterializationError(
                    "solver_handoff_csr_buffer_malformed_"
                    f"{name}_{buffer_name}"
                )
            objects.append(value)
    for name in _PROVENANCE_ATTRS:
        value = live.get(name)
        if type(value) is not np.ndarray:
            raise OperatorPhaseCliqueMaterializationError(
                f"solver_handoff_provenance_malformed_{name}"
            )
        objects.append(value)
    tags = live.get("_solver_constraint_row_tags")
    prefix = live.get("_solver_row_constraint_prefix_frames")
    if (
        type(tags) is not tuple
        or type(prefix) is not dict
        or prefix
    ):
        raise OperatorPhaseCliqueMaterializationError(
            "solver_handoff_row_metadata_malformed"
        )
    objects.extend((tags, prefix))
    return tuple(objects)


def _solver_owned_arrays(build: OperatorHZBuild) -> Tuple[np.ndarray, ...]:
    return tuple(
        value
        for value in _solver_owned_objects(build)
        if type(value) is np.ndarray
    )


def _private_solver_build(
    public_build: OperatorHZBuild,
    *,
    caps: OperatorPhaseCliqueCaps,
    expected_fresh_semantic_digest: str,
    expected_fresh_frame_digest: str,
    constructive_reason: str | None,
    deadline: float,
) -> OperatorHZBuild:
    """Create the sole solver-owned HZ; no public result retains an alias."""

    if (
        not _valid_sha256(expected_fresh_semantic_digest)
        or not _valid_sha256(expected_fresh_frame_digest)
    ):
        raise OperatorPhaseCliqueMaterializationError(
            "solver_handoff_expected_digest_malformed"
        )
    public_hz = public_build.hz
    digest_sink: list[str] = []
    try:
        private_hz = _snapshot_sparse_hz(
            public_hz,
            caps=caps,
            deadline=deadline,
            stage="solver_handoff",
            semantic_digest_sink=digest_sink,
        )
    except Exception as exc:
        raise OperatorPhaseCliqueMaterializationError(
            "solver_handoff_core_snapshot_failed"
        ) from exc
    if (
        len(digest_sink) != 1
        or digest_sink[0] != expected_fresh_semantic_digest
    ):
        raise OperatorPhaseCliqueMaterializationError(
            "solver_handoff_private_digest_mismatch"
        )

    input_length = int(public_build.input_col_ids.size)
    provenance_lengths = {
        "full_col_ids": input_length,
        "operator_input_center": input_length,
        "operator_input_radius": input_length,
        "_solver_continuous_column_layer_ids": private_hz.n_cont,
    }
    for name, length in provenance_lengths.items():
        dtype = (
            np.int64
            if name
            in {
                "full_col_ids",
                "_solver_continuous_column_layer_ids",
            }
            else np.float64
        )
        source = _exact_array(
            getattr(public_hz, name, None),
            name=f"solver_handoff_{name}",
            dtype=np.dtype(dtype),
            length=length,
        )
        copied = source.copy(order="C")
        copied.setflags(write=False)
        setattr(private_hz, name, copied)
    tags = getattr(public_hz, "_solver_constraint_row_tags", None)
    if (
        type(tags) is not tuple
        or len(tags) != private_hz.n_eq + private_hz.n_ub
        or any(type(tag) is not str for tag in tags)
    ):
        raise OperatorPhaseCliqueMaterializationError(
            "solver_handoff_constraint_tags_malformed"
        )
    setattr(private_hz, "_solver_constraint_row_tags", tuple(tags))
    setattr(private_hz, "_solver_row_constraint_prefix_frames", {})

    private_input_ids = _exact_array(
        public_build.input_col_ids,
        name="solver_handoff_build_input_col_ids",
        dtype=np.dtype(np.int64),
        length=input_length,
    ).copy(order="C")
    if not np.array_equal(
        private_input_ids, private_hz.full_col_ids
    ):
        raise OperatorPhaseCliqueMaterializationError(
            "solver_handoff_build_input_provenance_mismatch"
        )
    private_input_ids.setflags(write=False)
    private_build = OperatorHZBuild(
        hz=private_hz,
        input_col_ids=private_input_ids,
        input_layer_id=public_build.input_layer_id,
        output_layer_id=public_build.output_layer_id,
        assert_layer_id=public_build.assert_layer_id,
        # Solver semantics and witness decoding need no arbitrary producer
        # metadata.  Keep the private ownership boundary builtin-only.
        metadata={
            "schema": "operator_phase_clique_private_solver_handoff_v1",
            "candidate_only": False,
            "proof_authority": False,
            "semantic_digest": expected_fresh_semantic_digest,
        },
        property_upper_output=False,
        property_upper_row_groups=(),
        verified_preactivation_frame=None,
        constructive_nonempty_seal=None,
    )
    observed_frame_digest = _fresh_frame_digest(
        private_build,
        tags=tuple(tags),
        semantic_digest=expected_fresh_semantic_digest,
        deadline=deadline,
    )
    if observed_frame_digest != expected_fresh_frame_digest:
        raise OperatorPhaseCliqueMaterializationError(
            "solver_handoff_private_frame_digest_mismatch"
        )

    # The caller must already own the relevant non-emptiness theorem.  Issue
    # its token only on the private semantic clone; never copy a public token,
    # cache, statistic, or SAFE capability.
    if constructive_reason is not None:
        if type(constructive_reason) is not str or not constructive_reason:
            raise OperatorPhaseCliqueMaterializationError(
                "solver_handoff_constructive_reason_invalid"
            )
        hz_mark_constructively_nonempty(
            private_hz, constructive_reason
        )
    _freeze_exact_hz(private_hz, deadline=deadline)
    for name in _PROVENANCE_ATTRS:
        getattr(private_hz, name).setflags(write=False)

    public_arrays = _solver_owned_arrays(public_build)
    private_arrays = _solver_owned_arrays(private_build)
    if any(
        np.shares_memory(public_value, private_value)
        for public_value in public_arrays
        for private_value in private_arrays
    ):
        raise OperatorPhaseCliqueMaterializationError(
            "solver_handoff_public_private_buffer_alias"
        )
    if any(value.flags.writeable for value in private_arrays):
        raise OperatorPhaseCliqueMaterializationError(
            "solver_handoff_private_buffer_not_readonly"
        )
    if (
        sparse_hz_semantic_digest(private_hz)
        != expected_fresh_semantic_digest
    ):
        raise OperatorPhaseCliqueMaterializationError(
            "solver_handoff_private_terminal_digest_mismatch"
        )
    _check_deadline(deadline, stage="solver_handoff_private_complete")
    return private_build


def _strip_public_solver_authority(hz: SparseHZono) -> None:
    """Leave the returned fresh HZ candidate-only and non-authoritative."""

    live = vars(hz)
    for name in (
        "_solver_known_nonempty",
        "_solver_known_nonempty_reason",
        "_solver_constructive_nonempty_token",
        "_solver_constructive_nonempty_reason",
        "_solver_base_feas_cache",
        "_solver_base_feas_exact",
        "_solver_base_witness_cache",
        "_solver_objbound_stats",
        "_solver_objbound_safe_capability",
    ):
        live.pop(name, None)


def _make_solver_handoff_capability(
    *, fresh_semantic_digest: str
) -> OperatorPhaseCliqueMaterializationSolverCapability:
    now = time.monotonic()
    return OperatorPhaseCliqueMaterializationSolverCapability(
        token=secrets.token_hex(32),
        process_id=os.getpid(),
        expires_monotonic=now + _SOLVER_HANDOFF_TTL_SECONDS,
        fresh_semantic_digest=fresh_semantic_digest,
        _producer_capability=_SOLVER_HANDOFF_PRODUCER,
    )


def _sweep_solver_handoffs_locked(now: float) -> None:
    process_id = os.getpid()
    stale = tuple(
        token
        for token, record in _SOLVER_HANDOFF_REGISTRY.items()
        if (
            record.process_id != process_id
            or record.expires_monotonic <= now
            or record.capability_ref() is None
            or record.materialization_ref() is None
            or record.public_build_ref() is None
            or record.public_hz_ref() is None
        )
    )
    for token in stale:
        _SOLVER_HANDOFF_REGISTRY.pop(token, None)


def _register_solver_handoff(
    materialization: OperatorPhaseCliqueMaterialization,
    private_build: OperatorHZBuild,
) -> None:
    capability = materialization.solver_handoff_capability
    public_build = materialization.build
    public_hz = public_build.hz
    public_objects = _solver_owned_objects(public_build)
    private_objects = _solver_owned_objects(private_build)
    record = _SolverHandoffRecord(
        capability_ref=weakref.ref(capability),
        materialization_ref=weakref.ref(materialization),
        public_build_ref=weakref.ref(public_build),
        public_hz_ref=weakref.ref(public_hz),
        public_owned_identity=tuple(id(value) for value in public_objects),
        private_build=private_build,
        private_owned_identity=tuple(id(value) for value in private_objects),
        fresh_semantic_digest=materialization.fresh_semantic_digest,
        expires_monotonic=capability.expires_monotonic,
        process_id=capability.process_id,
    )
    with _SOLVER_HANDOFF_REGISTRY_LOCK:
        now = time.monotonic()
        _sweep_solver_handoffs_locked(now)
        if (
            now >= capability.expires_monotonic
            or len(_SOLVER_HANDOFF_REGISTRY)
            >= _SOLVER_HANDOFF_REGISTRY_CAPACITY
            or capability.token in _SOLVER_HANDOFF_REGISTRY
        ):
            raise OperatorPhaseCliqueMaterializationError(
                "solver_handoff_registry_issue_failed"
            )
        _SOLVER_HANDOFF_REGISTRY[capability.token] = record


def validate_operator_phase_clique_materialization_solver_handoff(
    materialization: Any,
) -> bool:
    """Read-only audit of a live materializer handoff registration."""

    try:
        if type(materialization) is not OperatorPhaseCliqueMaterialization:
            return False
        capability = materialization.solver_handoff_capability
        if (
            type(capability)
            is not OperatorPhaseCliqueMaterializationSolverCapability
            or capability.proof_authority is not False
            or capability.process_id != os.getpid()
            or not _valid_sha256(capability.fresh_semantic_digest)
            or capability.fresh_semantic_digest
            != materialization.fresh_semantic_digest
        ):
            return False
        with _SOLVER_HANDOFF_REGISTRY_LOCK:
            now = time.monotonic()
            _sweep_solver_handoffs_locked(now)
            record = _SOLVER_HANDOFF_REGISTRY.get(capability.token)
            return bool(
                record is not None
                and record.process_id == os.getpid()
                and record.capability_ref() is capability
                and record.materialization_ref() is materialization
                and record.public_build_ref() is materialization.build
                and record.public_hz_ref() is materialization.build.hz
                and record.expires_monotonic
                == capability.expires_monotonic
                and record.expires_monotonic > now
                and record.fresh_semantic_digest
                == materialization.fresh_semantic_digest
            )
    except (
        AttributeError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


def consume_operator_phase_clique_materialization_solver_handoff(
    materialization: Any,
    capability: Any,
    *,
    deadline: float,
) -> OperatorHZBuild:
    """Atomically transfer the sole private frozen HZ to the verifier."""

    deadline_value = _normalize_deadline(deadline)
    _check_deadline(deadline_value, stage="before_solver_handoff_consume")
    if (
        type(materialization) is not OperatorPhaseCliqueMaterialization
        or type(capability)
        is not OperatorPhaseCliqueMaterializationSolverCapability
        or materialization.solver_handoff_capability is not capability
        or capability.proof_authority is not False
        or capability.process_id != os.getpid()
        or type(capability.token) is not str
        or len(capability.token) != 64
        or any(
            character not in "0123456789abcdef"
            for character in capability.token
        )
        or not _valid_sha256(capability.fresh_semantic_digest)
        or type(capability.expires_monotonic) is not float
        or not math.isfinite(capability.expires_monotonic)
    ):
        raise OperatorPhaseCliqueMaterializationError(
            "solver_handoff_capability_malformed"
        )
    with _SOLVER_HANDOFF_REGISTRY_LOCK:
        now = time.monotonic()
        _sweep_solver_handoffs_locked(now)
        record = _SOLVER_HANDOFF_REGISTRY.get(capability.token)
        if (
            record is None
            or record.process_id != os.getpid()
            or record.capability_ref() is not capability
            or record.materialization_ref() is not materialization
            or record.public_build_ref() is not materialization.build
            or record.public_hz_ref() is not materialization.build.hz
            or record.expires_monotonic != capability.expires_monotonic
            or record.expires_monotonic <= now
            or deadline_value <= now
            or record.fresh_semantic_digest
            != capability.fresh_semantic_digest
        ):
            raise OperatorPhaseCliqueMaterializationError(
                "solver_handoff_capability_invalid"
            )
        try:
            public_identity = tuple(
                id(value)
                for value in _solver_owned_objects(
                    materialization.build
                )
            )
            private_identity = tuple(
                id(value)
                for value in _solver_owned_objects(
                    record.private_build
                )
            )
        except OperatorPhaseCliqueMaterializationError:
            public_identity = ()
            private_identity = ()
        if (
            public_identity != record.public_owned_identity
            or private_identity != record.private_owned_identity
            or any(
                value.flags.writeable
                for value in _solver_owned_arrays(
                    record.private_build
                )
            )
        ):
            _SOLVER_HANDOFF_REGISTRY.pop(capability.token, None)
            raise OperatorPhaseCliqueMaterializationError(
                "solver_handoff_owner_identity_changed"
            )
        # Pop under the lock: copies, clones, threads, and replays cannot get
        # a second reference to the private build.
        _SOLVER_HANDOFF_REGISTRY.pop(capability.token)
        private_build = record.private_build
    _check_deadline(deadline_value, stage="after_solver_handoff_consume")
    return private_build


def materialize_verified_operator_phase_clique_cuts(
    build: OperatorHZBuild,
    focused_rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    result: OperatorExactReLUPhaseCliqueResult,
    *,
    deadline: float,
    caps: OperatorPhaseCliqueCaps,
) -> OperatorPhaseCliqueMaterialization:
    """Consume a verifier-owned cut snapshot and build a fresh Operator-HZ.

    The live ``build`` and ``result`` are consulted only inside the issuing
    verifier.  Once the one-use snapshot has been atomically consumed, this
    function never reads either live object again.
    """

    deadline_value = _normalize_deadline(deadline)
    caps_payload = _caps_payload(caps)
    frozen_caps = OperatorPhaseCliqueCaps(**caps_payload)
    selection_caps = _selection_caps(selection)
    selection_cap_values = (
        selection_caps.max_rivals,
        selection_caps.max_binaries,
        selection_caps.max_work_items,
        selection_caps.timeout_seconds,
    )
    if (
        type(build) is not OperatorHZBuild
        or type(build.constructive_nonempty_seal)
        is not OperatorHZConstructiveNonemptySeal
    ):
        # The large-model route must never fall through to the verifier's
        # bounded witness-generation fallback: that path takes another full
        # HZ snapshot and may invoke MILP.  A present-but-invalid seal still
        # fails inside the owner/digest-bound verifier below.
        raise OperatorPhaseCliqueMaterializationError(
            "producer_nonempty_seal_required"
        )
    _check_deadline(deadline_value, stage="before_snapshot_issue")
    capability = verify_and_issue_operator_phase_clique_snapshot(
        build,
        focused_rivals,
        selection,
        result,
        deadline=deadline_value,
        selection_max_rivals=selection_cap_values[0],
        selection_max_binaries=selection_cap_values[1],
        selection_max_work_items=selection_cap_values[2],
        selection_timeout_seconds=selection_cap_values[3],
        **caps_payload,
    )
    if capability is None:
        raise OperatorPhaseCliqueMaterializationError(
            "hardened_clique_snapshot_issue_failed"
        )
    # Copy the registry-bound value before the one-use pop.  The returned
    # snapshot is a Python object whose frozen fields and NumPy write flags are
    # defensive aids, not an authority boundary.
    expected_snapshot_digest = capability.snapshot_digest
    _check_deadline(deadline_value, stage="before_snapshot_consume")
    try:
        snapshot = consume_verified_operator_phase_clique_snapshot(
            capability, deadline=deadline_value
        )
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ) as exc:
        raise OperatorPhaseCliqueMaterializationError(
            "verified_clique_snapshot_consume_failed"
        ) from exc
    if not verify_consumed_operator_phase_clique_snapshot_integrity(
        snapshot,
        expected_snapshot_digest=expected_snapshot_digest,
        deadline=deadline_value,
    ):
        raise OperatorPhaseCliqueMaterializationError(
            "verified_clique_snapshot_integrity_replay_failed"
        )

    # From here onward the unique private snapshot is the sole source.
    (
        verified_cut,
        verified_cliques,
        source_tags,
        source_provenance,
        provenance_items,
        source_tag_bytes,
    ) = _validate_consumed_snapshot(
        snapshot, caps=frozen_caps, deadline=deadline_value
    )
    _verify_snapshot_cut_rows(
        verified_cut,
        original_parent_n_ub=snapshot.original_parent_n_ub,
        cliques=verified_cliques,
        deadline=deadline_value,
    )
    parent_snapshot = _parent_prefix_from_verified_cut(
        verified_cut,
        original_parent_n_ub=snapshot.original_parent_n_ub,
        deadline=deadline_value,
    )
    _validate_core(parent_snapshot, deadline=deadline_value)
    _audit_caps(
        parent_snapshot,
        caps=frozen_caps,
        provenance_items=provenance_items,
        tag_bytes=source_tag_bytes,
        deadline=deadline_value,
    )
    parent_digest = sparse_hz_semantic_digest(parent_snapshot)
    if parent_digest != snapshot.parent_semantic_digest:
        raise OperatorPhaseCliqueMaterializationError(
            "verified_snapshot_parent_digest_mismatch"
        )
    verified_cut_digest = sparse_hz_semantic_digest(verified_cut)

    source_snapshot_build = OperatorHZBuild(
        hz=parent_snapshot,
        input_col_ids=snapshot.build_input_col_ids,
        input_layer_id=snapshot.input_layer_id,
        output_layer_id=snapshot.output_layer_id,
        assert_layer_id=snapshot.assert_layer_id,
        metadata={},
        property_upper_output=False,
        property_upper_row_groups=(),
        verified_preactivation_frame=None,
    )
    source_frame_digest = _source_frame_digest(
        source_snapshot_build,
        tags=source_tags,
        provenance=source_provenance,
        parent_semantic_digest=parent_digest,
        deadline=deadline_value,
    )
    _check_deadline(deadline_value, stage="after_source_frame_digest")

    # Ownership of the one-use verifier snapshot has transferred to this
    # materializer.  It is already detached, exact, semantic-digest checked,
    # and frozen, so it becomes the public candidate by identity.  The only
    # subsequent full-core allocation is the independent private handoff.
    fresh = verified_cut
    for name in _PROVENANCE_ATTRS:
        setattr(fresh, name, source_provenance[name])
    clique_ids = tuple(item[0] for item in verified_cliques)
    cut_tags = tuple(
        f"{_CUT_TAG_PREFIX}:{ordinal}:{clique_id}"
        for ordinal, clique_id in enumerate(clique_ids)
    )
    fresh_tags = source_tags + cut_tags
    setattr(fresh, "_solver_constraint_row_tags", fresh_tags)
    setattr(fresh, "_solver_row_constraint_prefix_frames", {})

    # This is a new token issued after exact verification and one-use
    # consumption.  No parent token or candidate attribute is copied.
    hz_mark_constructively_nonempty(fresh, _CONSTRUCTIVE_REASON)
    _validate_core(fresh, deadline=deadline_value)
    _require_readonly_core(
        fresh,
        stage="public_fresh",
        deadline=deadline_value,
    )
    fresh_tag_bytes = sum(
        len(tag.encode("ascii")) for tag in fresh_tags
    )
    _audit_caps(
        fresh,
        caps=frozen_caps,
        provenance_items=provenance_items,
        tag_bytes=fresh_tag_bytes,
        deadline=deadline_value,
    )
    fresh_digest = sparse_hz_semantic_digest(fresh)
    if fresh_digest != verified_cut_digest:
        raise OperatorPhaseCliqueMaterializationError(
            "fresh_cut_digest_mismatch"
        )

    materialized_metadata = {
        "schema": (
            "operator_hz_exact_relu_phase_clique_materialized_v1"
        ),
        "candidate_only": True,
        "proof_authority": False,
        "parent_semantic_digest": parent_digest,
        "fresh_semantic_digest": fresh_digest,
        "verified_snapshot_digest": snapshot.snapshot_digest,
        "exact_clique_count": len(verified_cliques),
        "exact_clique_cut_rows": len(verified_cliques),
        "verdict_path": "hz_objbound_decide_only",
        "constructive_nonempty_reason": _CONSTRUCTIVE_REASON,
        "public_core_source": "consumed_verified_cut_zero_copy",
        "materializer_full_core_copy_count": 1,
    }
    fresh_build = OperatorHZBuild(
        hz=fresh,
        input_col_ids=snapshot.build_input_col_ids,
        input_layer_id=snapshot.input_layer_id,
        output_layer_id=snapshot.output_layer_id,
        assert_layer_id=snapshot.assert_layer_id,
        metadata=materialized_metadata,
        property_upper_output=False,
        property_upper_row_groups=(),
        verified_preactivation_frame=None,
    )
    fresh_frame_digest = _fresh_frame_digest(
        fresh_build,
        tags=fresh_tags,
        semantic_digest=fresh_digest,
        deadline=deadline_value,
    )

    # Terminal seals touch only the consumed snapshot and its read-only parent
    # view.  They run before the sole independent full-core solver snapshot.
    _check_deadline(deadline_value, stage="before_terminal_seals")
    terminal_parent_digest = sparse_hz_semantic_digest(
        parent_snapshot
    )
    terminal_cut_digest = sparse_hz_semantic_digest(verified_cut)
    terminal_fresh_digest = sparse_hz_semantic_digest(fresh)
    terminal_source_frame_digest = _source_frame_digest(
        source_snapshot_build,
        tags=source_tags,
        provenance=source_provenance,
        parent_semantic_digest=terminal_parent_digest,
        deadline=deadline_value,
    )
    terminal_fresh_frame_digest = _fresh_frame_digest(
        fresh_build,
        tags=tuple(fresh._solver_constraint_row_tags),
        semantic_digest=terminal_fresh_digest,
        deadline=deadline_value,
    )
    if (
        terminal_parent_digest != parent_digest
        or terminal_cut_digest != verified_cut_digest
        or terminal_fresh_digest != fresh_digest
        or terminal_source_frame_digest != source_frame_digest
        or terminal_fresh_frame_digest != fresh_frame_digest
    ):
        raise OperatorPhaseCliqueMaterializationError(
            "terminal_materialization_seal_mismatch"
        )
    _require_readonly_core(
        parent_snapshot,
        stage="terminal_parent_prefix",
        deadline=deadline_value,
    )
    _require_readonly_core(
        fresh,
        stage="terminal_public_fresh",
        deadline=deadline_value,
    )
    _audit_caps(
        parent_snapshot,
        caps=frozen_caps,
        provenance_items=provenance_items,
        tag_bytes=source_tag_bytes,
        deadline=deadline_value,
    )
    _audit_caps(
        fresh,
        caps=frozen_caps,
        provenance_items=provenance_items,
        tag_bytes=fresh_tag_bytes,
        deadline=deadline_value,
    )
    private_solver_build = _private_solver_build(
        fresh_build,
        caps=frozen_caps,
        expected_fresh_semantic_digest=fresh_digest,
        expected_fresh_frame_digest=fresh_frame_digest,
        constructive_reason=_CONSTRUCTIVE_REASON,
        deadline=deadline_value,
    )
    # ``fresh_build`` is returned inside a candidate-only public result.  It
    # must never independently satisfy the solver's constructive base gate;
    # only the unexposed one-use private copy above receives that authority.
    _strip_public_solver_authority(fresh)
    if hz_constructively_nonempty(fresh):
        raise OperatorPhaseCliqueMaterializationError(
            "public_materialization_retained_solver_authority"
        )
    solver_handoff_capability = _make_solver_handoff_capability(
        fresh_semantic_digest=fresh_digest
    )
    _check_deadline(deadline_value, stage="before_receipt")

    receipt_body = {
        "schema": (
            "act.operator_exact_relu_phase_clique_materialization.v2"
        ),
        "status": "fresh_verified_clique_cuts_materialized",
        "candidate_only": True,
        "proof_authority": False,
        "hardened_exact_result_verifier_passed": True,
        "one_use_snapshot_consumed": True,
        "producer_nonempty_seal_verified": (
            snapshot.producer_nonempty_seal_verified
        ),
        "verified_snapshot_digest": snapshot.snapshot_digest,
        "verified_result_digest": snapshot.verified_result_digest,
        "parent_semantic_digest": parent_digest,
        "verified_cut_semantic_digest": verified_cut_digest,
        "fresh_semantic_digest": fresh_digest,
        "ordered_source_frame_sha256": (
            snapshot.ordered_source_frame_sha256
        ),
        "source_frame_digest": source_frame_digest,
        "fresh_frame_digest": fresh_frame_digest,
        "selection_digest": snapshot.selection_digest,
        "focused_property_digest": (
            snapshot.focused_property_digest
        ),
        "subset_binding_digest": snapshot.subset_binding_digest,
        "clique_ids": clique_ids,
        "cut_row_tags": cut_tags,
        "cut_row_count": len(verified_cliques),
        "source_upper_rows": snapshot.original_parent_n_ub,
        "fresh_upper_rows": fresh.n_ub,
        "copied_parent_attributes": _PROVENANCE_ATTRS,
        "public_core_source": "consumed_verified_cut_zero_copy",
        "parent_prefix_core": "strict_readonly_zero_copy_view",
        "parent_prefix_readonly": True,
        "parent_prefix_aliases_public_cut": True,
        "public_core_readonly": True,
        "materializer_full_core_copy_count": 1,
        "private_solver_core": "single_independent_snapshot",
        "public_private_core_no_alias": True,
        "row_prefix_frames": "fresh_empty",
        "incompatible_receipts": "rejected_not_copied",
        "solver_caches_stats_safe_tokens": "not_copied",
        "constructive_nonempty_reissued": True,
        "constructive_nonempty_scope": "private_solver_handoff_only",
        "public_constructive_nonempty_token": "absent",
        "solver_handoff_one_use": True,
        "solver_handoff_owner_bound": True,
        "solver_handoff_pid_bound": True,
        "solver_handoff_private_core_readonly": True,
        "constructive_nonempty_reason": _CONSTRUCTIVE_REASON,
        "constructive_rule": (
            "full_parent_exact_pair_conflicts_imply_redundant_"
            "integer_clique_rows"
        ),
        "verdict_path": "hz_objbound_decide_only",
        "caps": dict(caps_payload),
    }
    receipt = dict(receipt_body)
    receipt["receipt_sha256"] = _canonical_sha256(receipt_body)
    materialized = OperatorPhaseCliqueMaterialization(
        build=fresh_build,
        parent_semantic_digest=parent_digest,
        fresh_semantic_digest=fresh_digest,
        source_frame_digest=source_frame_digest,
        fresh_frame_digest=fresh_frame_digest,
        clique_ids=clique_ids,
        cut_row_tags=cut_tags,
        receipt=receipt,
        solver_handoff_capability=solver_handoff_capability,
        proof_authority=False,
    )
    _check_deadline(deadline_value, stage="materialization_complete")
    _register_solver_handoff(materialized, private_solver_build)
    return materialized


def maybe_materialize_verified_operator_phase_clique_cuts(
    build: OperatorHZBuild,
    focused_rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    result: OperatorExactReLUPhaseCliqueResult,
    *,
    enabled: bool = False,
    deadline: Any = None,
    caps: Any = None,
) -> Union[OperatorHZBuild, OperatorPhaseCliqueMaterialization]:
    """Default-off wrapper with a zero-touch, identity-preserving no-op."""

    if type(enabled) is not bool:
        raise OperatorPhaseCliqueMaterializationError(
            "enabled_not_builtin_bool"
        )
    if not enabled:
        return build
    return materialize_verified_operator_phase_clique_cuts(
        build,
        focused_rivals,
        selection,
        result,
        deadline=deadline,
        caps=caps,
    )


__all__ = [
    "OperatorPhaseCliqueMaterialization",
    "OperatorPhaseCliqueMaterializationError",
    "OperatorPhaseCliqueMaterializationSolverCapability",
    "consume_operator_phase_clique_materialization_solver_handoff",
    "materialize_verified_operator_phase_clique_cuts",
    "maybe_materialize_verified_operator_phase_clique_cuts",
    "validate_operator_phase_clique_materialization_solver_handoff",
]
