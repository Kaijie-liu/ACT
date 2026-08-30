#!/usr/bin/env python3
"""Default-off Operator top-2 adapter for one localized phase edge.

The adapter reuses the production-audited Operator snapshot, selection
re-derivation, exact ranking, and subset binding from
``operator_exact_relu_phase_cliques``.  It selects exactly the deterministic
top two nonzero phase candidates and delegates only that bound pair to the
isolated localized oracle.  It never materializes a cut and has no verifier,
BaB, pipeline, Gate, or ground-truth integration.

Neither this adapter nor the localized solver has proof authority.  An edge is
reported only when the localized result is structurally intact, contains an
existing full-parent exact certificate, and that certificate independently
replays against the adapter's private Operator-HZ snapshot.  The live caller
parent and row-tag frame are sealed again before return.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    RivalSpec,
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf import (
    localized_phase_conflict_oracle as localized_oracle,
)
from act.back_end.hybridz_tf import (
    operator_exact_relu_phase_cliques as clique_module,
)
from act.back_end.hybridz_tf.localized_phase_conflict_oracle import (
    LocalizedPhaseConflictOracleResult,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_cliques import (
    RankedOperatorPhase,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
    OperatorExactReLUPhaseSelection,
)
from act.back_end.hybridz_tf.operator_hz import OperatorHZBuild
from act.back_end.hybridz_tf.persistent_phase_conflict_oracle import (
    ExactDualRayConflictCertificate,
    _ordered_source_frame_digest,
    verify_exact_dual_ray_conflict_certificate,
)
from act.back_end.hybridz_tf.property_phase_conflict_clique import (
    PhaseLiteral,
)
from act.back_end.solver.solver_hz import SparseHZono


class OperatorLocalizedPhaseEdgeError(ValueError):
    """Malformed adapter invocation or failed trusted Operator audit."""


@dataclass(frozen=True)
class OperatorLocalizedPhaseEdgeCaps:
    """Every caller-controlled and fixed stop-loss bound for one E2 run."""

    selection_max_rivals: int
    selection_max_binaries: int
    selection_max_work_items: int
    selection_timeout_seconds: float
    max_parent_variables: int
    max_parent_rows: int
    max_parent_nonzeros: int
    max_parent_buffer_items: int
    max_top_literals: int
    max_total_pairs: int
    max_source_terms: int
    max_multiplier_bits: int
    max_exact_bits: int
    max_exact_nonzeros: int
    localized_row_tiers: Tuple[int, ...]
    localized_max_selected_nnz: int
    localized_max_source_terms: int


@dataclass(frozen=True)
class OperatorLocalizedPhaseEdgeCandidateResult:
    """Frozen, checksummed top-2 edge proposal; never a materialized cut."""

    status: str
    reason: str
    enabled: bool
    build_binding_sha256: Optional[str]
    parent_semantic_digest: Optional[str]
    terminal_parent_semantic_digest: Optional[str]
    focused_property_digest: Optional[str]
    operator_row_tag_digest: Optional[str]
    terminal_operator_row_tag_digest: Optional[str]
    selection_digest: Optional[str]
    subset_binding_digest: Optional[str]
    ordered_source_frame_sha256: Optional[str]
    producer_nonempty_seal_verified: bool
    source_modes: Tuple[Tuple[str, bool], ...]
    source_modes_sha256: str
    ranked_phases: Tuple[RankedOperatorPhase, ...]
    literals: Tuple[PhaseLiteral, ...]
    omitted_zero_bcol_ids: Tuple[int, ...]
    excluded_selected_bcol_ids: Tuple[int, ...]
    localized_result: Optional[LocalizedPhaseConflictOracleResult]
    localized_result_sha256: Optional[str]
    certificate: Optional[ExactDualRayConflictCertificate]
    edge_accepted: bool
    parent_unchanged: bool
    caps: Optional[OperatorLocalizedPhaseEdgeCaps]
    result_sha256: str
    proof_authority: bool = False


_DEFAULT_MAX_PARENT_VARIABLES = 2_000_000
_DEFAULT_MAX_PARENT_ROWS = 2_000_000
_DEFAULT_MAX_PARENT_NONZEROS = 50_000_000
_DEFAULT_MAX_PARENT_BUFFER_ITEMS = 120_000_000
_DEFAULT_MAX_SOURCE_TERMS = 128
_DEFAULT_MAX_MULTIPLIER_BITS = 256
_DEFAULT_MAX_EXACT_BITS = 4096
_DEFAULT_MAX_EXACT_NONZEROS = 200_000
_STRICT_TOP_LITERALS = 2
_STRICT_TOTAL_PAIRS = 1
_STRICT_LOCALIZED_ROW_TIERS = (64, 256, 1024, 4096)
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


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    if type(value) is not str or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _strict_int(value: Any, *, name: str) -> int:
    if type(value) is not int:
        raise OperatorLocalizedPhaseEdgeError(f"{name}_not_builtin_integer")
    return value


def _normalize_caps(
    *,
    selection_max_rivals: int,
    selection_max_binaries: int,
    selection_max_work_items: int,
    selection_timeout_seconds: float,
    max_parent_variables: int,
    max_parent_rows: int,
    max_parent_nonzeros: int,
    max_parent_buffer_items: int,
    max_top_literals: int,
    max_total_pairs: int,
    max_source_terms: int,
    max_multiplier_bits: int,
    max_exact_bits: int,
    max_exact_nonzeros: int,
    localized_row_tiers: Tuple[int, ...],
    localized_max_selected_nnz: int,
    localized_max_source_terms: int,
) -> Tuple[OperatorLocalizedPhaseEdgeCaps, Any]:
    if type(selection_timeout_seconds) not in {int, float} or type(
        selection_timeout_seconds
    ) is bool:
        raise OperatorLocalizedPhaseEdgeError("selection_timeout_not_numeric")
    timeout = float(selection_timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0.0 or timeout > 60.0:
        raise OperatorLocalizedPhaseEdgeError("selection_timeout_out_of_range")
    tiers = tuple(
        _strict_int(value, name="localized_row_tier")
        for value in localized_row_tiers
    )
    if tiers != _STRICT_LOCALIZED_ROW_TIERS:
        raise OperatorLocalizedPhaseEdgeError("localized_row_tiers_not_strict")
    if max_top_literals != _STRICT_TOP_LITERALS:
        raise OperatorLocalizedPhaseEdgeError("adapter_requires_exact_top2")
    if max_total_pairs != _STRICT_TOTAL_PAIRS:
        raise OperatorLocalizedPhaseEdgeError("adapter_requires_one_pair")
    # Reuse the clique implementation's strict hard-limit validation.  The
    # clique/search values are inert here but are fixed so they enter the exact
    # subset-binding digest without creating a caller-controlled side channel.
    clique_caps = clique_module._normalize_caps(
        max_parent_variables=max_parent_variables,
        max_parent_rows=max_parent_rows,
        max_parent_nonzeros=max_parent_nonzeros,
        max_parent_buffer_items=max_parent_buffer_items,
        max_top_literals=max_top_literals,
        max_total_pairs=max_total_pairs,
        max_cliques=1,
        max_clique_search_nodes=1,
        max_source_terms=max_source_terms,
        max_multiplier_bits=max_multiplier_bits,
        max_exact_bits=max_exact_bits,
        max_exact_nonzeros=max_exact_nonzeros,
    )
    selection_values = (
        _strict_int(selection_max_rivals, name="selection_max_rivals"),
        _strict_int(selection_max_binaries, name="selection_max_binaries"),
        _strict_int(selection_max_work_items, name="selection_max_work_items"),
    )
    if (
        selection_values[0] < 1
        or selection_values[0] > 256
        or selection_values[1] < 1
        or selection_values[1] > 65_536
        or selection_values[2] < 1
        or selection_values[2] > 50_000_000
    ):
        raise OperatorLocalizedPhaseEdgeError("selection_cap_out_of_range")
    localized_nnz = _strict_int(
        localized_max_selected_nnz, name="localized_max_selected_nnz"
    )
    localized_terms = _strict_int(
        localized_max_source_terms, name="localized_max_source_terms"
    )
    if localized_nnz < 1 or localized_nnz > 1_000_000:
        raise OperatorLocalizedPhaseEdgeError("localized_nnz_cap_out_of_range")
    if localized_terms < 1 or localized_terms > 256:
        raise OperatorLocalizedPhaseEdgeError("localized_source_cap_out_of_range")
    result = OperatorLocalizedPhaseEdgeCaps(
        selection_max_rivals=selection_values[0],
        selection_max_binaries=selection_values[1],
        selection_max_work_items=selection_values[2],
        selection_timeout_seconds=timeout,
        max_parent_variables=clique_caps.max_parent_variables,
        max_parent_rows=clique_caps.max_parent_rows,
        max_parent_nonzeros=clique_caps.max_parent_nonzeros,
        max_parent_buffer_items=clique_caps.max_parent_buffer_items,
        max_top_literals=clique_caps.max_top_literals,
        max_total_pairs=clique_caps.max_total_pairs,
        max_source_terms=clique_caps.max_source_terms,
        max_multiplier_bits=clique_caps.max_multiplier_bits,
        max_exact_bits=clique_caps.max_exact_bits,
        max_exact_nonzeros=clique_caps.max_exact_nonzeros,
        localized_row_tiers=tiers,
        localized_max_selected_nnz=localized_nnz,
        localized_max_source_terms=localized_terms,
    )
    return result, clique_caps


def _caps_payload(caps: OperatorLocalizedPhaseEdgeCaps) -> dict[str, Any]:
    return {
        "selection_max_rivals": caps.selection_max_rivals,
        "selection_max_binaries": caps.selection_max_binaries,
        "selection_max_work_items": caps.selection_max_work_items,
        "selection_timeout_seconds_f64_hex": caps.selection_timeout_seconds.hex(),
        "max_parent_variables": caps.max_parent_variables,
        "max_parent_rows": caps.max_parent_rows,
        "max_parent_nonzeros": caps.max_parent_nonzeros,
        "max_parent_buffer_items": caps.max_parent_buffer_items,
        "max_top_literals": caps.max_top_literals,
        "max_total_pairs": caps.max_total_pairs,
        "max_source_terms": caps.max_source_terms,
        "max_multiplier_bits": caps.max_multiplier_bits,
        "max_exact_bits": caps.max_exact_bits,
        "max_exact_nonzeros": caps.max_exact_nonzeros,
        "localized_row_tiers": list(caps.localized_row_tiers),
        "localized_max_selected_nnz": caps.localized_max_selected_nnz,
        "localized_max_source_terms": caps.localized_max_source_terms,
    }


def _source_modes_payload(
    source_modes: Tuple[Tuple[str, bool], ...],
) -> list[list[Any]]:
    return [[name, enabled] for name, enabled in source_modes]


def _validate_source_modes(value: Any) -> Tuple[Tuple[str, bool], ...]:
    if (
        type(value) is not tuple
        or not value
        or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not bool
            for item in value
        )
        or tuple(value) != _EXPECTED_SOURCE_MODES
    ):
        raise OperatorLocalizedPhaseEdgeError(
            "operator_materializer_source_modes_not_closed"
        )
    return tuple(value)


def _source_modes_digest(
    source_modes: Tuple[Tuple[str, bool], ...],
) -> str:
    return _sha256(
        {
            "schema": "act.operator_localized_phase_edge.source_modes.v1",
            "source_modes": _source_modes_payload(source_modes),
        }
    )


def _row_tag_digest(hz: SparseHZono) -> str:
    raw = vars(hz).get("_solver_constraint_row_tags")
    if (
        type(raw) is not tuple
        or len(raw) != hz.n_eq + hz.n_ub
        or any(type(tag) is not str for tag in raw)
    ):
        raise OperatorLocalizedPhaseEdgeError("operator_row_tags_malformed")
    return _sha256(
        {
            "schema": "act.operator_exact_relu_row_tags.v1",
            "n_eq": int(hz.n_eq),
            "n_ub": int(hz.n_ub),
            "tags": raw,
        }
    )


def _build_binding(
    build: OperatorHZBuild,
    *,
    parent_digest: str,
    row_tag_digest: str,
    producer_nonempty_seal_verified: bool,
    source_modes: Tuple[Tuple[str, bool], ...],
    source_modes_sha256: str,
) -> str:
    values = vars(build)
    hz = values.get("hz")
    ids = values.get("input_col_ids")
    if (
        type(hz) is not SparseHZono
        or type(ids) is not np.ndarray
        or ids.dtype != np.dtype(np.int64)
        or ids.ndim != 1
        or not ids.flags.c_contiguous
        or any(type(values.get(name)) is not int for name in (
            "input_layer_id", "output_layer_id", "assert_layer_id"
        ))
        or values.get("property_upper_output") is not False
        or producer_nonempty_seal_verified is not True
        or source_modes != _EXPECTED_SOURCE_MODES
        or not _valid_sha256(source_modes_sha256)
        or source_modes_sha256 != _source_modes_digest(source_modes)
    ):
        raise OperatorLocalizedPhaseEdgeError("operator_build_binding_malformed")
    return _sha256(
        {
            "schema": "act.operator_localized_phase_edge.build.v1",
            "parent_semantic_digest": parent_digest,
            "operator_row_tag_digest": row_tag_digest,
            "input_col_ids": [int(value) for value in ids.tolist()],
            "input_layer_id": values["input_layer_id"],
            "output_layer_id": values["output_layer_id"],
            "assert_layer_id": values["assert_layer_id"],
            "dimensions": [hz.n_out, hz.n_cont, hz.n_bin, hz.n_eq, hz.n_ub],
            "property_upper_output": False,
            "producer_nonempty_seal_verified": True,
            "source_modes": _source_modes_payload(source_modes),
            "source_modes_sha256": source_modes_sha256,
        }
    )


def _ranked_payload(ranked: Sequence[RankedOperatorPhase]) -> list[list[Any]]:
    return [
        [
            item.rank,
            item.stable_bcol_id,
            item.phase,
            item.score_numerator,
            item.score_denominator,
        ]
        for item in ranked
    ]


def _literal_payload(literals: Sequence[PhaseLiteral]) -> list[list[Any]]:
    return [
        [item.stable_bcol_id, item.phase, item.binding_digest]
        for item in literals
    ]


def _validate_localized_result(
    result: Any,
    *,
    hz: SparseHZono,
    literals: Tuple[PhaseLiteral, PhaseLiteral],
    subset_digest: str,
    parent_digest: str,
    source_frame_digest: str,
    caps: OperatorLocalizedPhaseEdgeCaps,
    deadline: float,
) -> Tuple[bool, Optional[ExactDualRayConflictCertificate]]:
    if type(result) is not LocalizedPhaseConflictOracleResult:
        raise OperatorLocalizedPhaseEdgeError("localized_result_wrong_type")
    try:
        live_result_digest = localized_oracle._sha256(
            localized_oracle._result_payload(result, include_digest=False)
        )
    except Exception as exc:
        raise OperatorLocalizedPhaseEdgeError("localized_result_payload_invalid") from exc
    if (
        result.proof_authority is not False
        or not _valid_sha256(result.result_sha256)
        or live_result_digest != result.result_sha256
        or result.literals != literals
        or result.property_digest != subset_digest
        or result.parent_semantic_digest != parent_digest
        or result.ordered_source_frame_sha256 != source_frame_digest
        or result.row_tiers != caps.localized_row_tiers
        or result.max_selected_nnz != caps.localized_max_selected_nnz
        or result.max_source_terms != caps.localized_max_source_terms
    ):
        raise OperatorLocalizedPhaseEdgeError("localized_result_binding_mismatch")
    if result.edge_accepted:
        certificate = result.certificate
        if (
            type(certificate) is not ExactDualRayConflictCertificate
            or result.status != "certified_conflict"
            or not result.parent_unchanged
            or result.terminal_parent_semantic_digest != parent_digest
            or result.terminal_source_frame_sha256 != source_frame_digest
            or certificate.literals != literals
            or certificate.parent_semantic_digest != parent_digest
            or certificate.property_digest != subset_digest
            or certificate.ordered_source_frame_sha256 != source_frame_digest
            or not verify_exact_dual_ray_conflict_certificate(
                hz,
                certificate,
                property_digest=subset_digest,
                deadline=deadline,
                max_source_terms=caps.localized_max_source_terms,
                max_multiplier_bits=caps.max_multiplier_bits,
                max_exact_bits=caps.max_exact_bits,
                max_exact_nonzeros=caps.max_exact_nonzeros,
            )
        ):
            raise OperatorLocalizedPhaseEdgeError("localized_edge_exact_replay_rejected")
        return True, certificate
    if result.certificate is not None:
        raise OperatorLocalizedPhaseEdgeError("localized_nonedge_retained_certificate")
    return False, None


def _result_payload(
    result: OperatorLocalizedPhaseEdgeCandidateResult,
    *,
    include_digest: bool,
) -> dict[str, Any]:
    payload = {
        "schema": "act.operator_localized_phase_edge_candidate.v1",
        "status": result.status,
        "reason": result.reason,
        "enabled": result.enabled,
        "build_binding_sha256": result.build_binding_sha256,
        "parent_semantic_digest": result.parent_semantic_digest,
        "terminal_parent_semantic_digest": result.terminal_parent_semantic_digest,
        "focused_property_digest": result.focused_property_digest,
        "operator_row_tag_digest": result.operator_row_tag_digest,
        "terminal_operator_row_tag_digest": result.terminal_operator_row_tag_digest,
        "selection_digest": result.selection_digest,
        "subset_binding_digest": result.subset_binding_digest,
        "ordered_source_frame_sha256": result.ordered_source_frame_sha256,
        "producer_nonempty_seal_verified": (
            result.producer_nonempty_seal_verified
        ),
        "source_modes": _source_modes_payload(result.source_modes),
        "source_modes_sha256": result.source_modes_sha256,
        "ranked_phases": _ranked_payload(result.ranked_phases),
        "literals": _literal_payload(result.literals),
        "omitted_zero_bcol_ids": list(result.omitted_zero_bcol_ids),
        "excluded_selected_bcol_ids": list(result.excluded_selected_bcol_ids),
        "localized_result_sha256": result.localized_result_sha256,
        "certificate_sha256": (
            None if result.certificate is None else result.certificate.certificate_sha256
        ),
        "edge_accepted": result.edge_accepted,
        "parent_unchanged": result.parent_unchanged,
        "caps": (
            None if result.caps is None else _caps_payload(result.caps)
        ),
        "proof_authority": result.proof_authority,
    }
    if include_digest:
        payload["result_sha256"] = result.result_sha256
    return payload


def _finish(
    *,
    status: str,
    reason: str,
    enabled: bool,
    build_binding: str,
    parent_digest: str,
    property_digest: str,
    row_tag_digest: str,
    selection_digest: str,
    subset_digest: str,
    source_frame_digest: Optional[str],
    producer_nonempty_seal_verified: bool,
    source_modes: Tuple[Tuple[str, bool], ...],
    source_modes_sha256: str,
    ranked: Tuple[RankedOperatorPhase, ...],
    literals: Tuple[PhaseLiteral, ...],
    omitted: Tuple[int, ...],
    excluded: Tuple[int, ...],
    localized_result: Optional[LocalizedPhaseConflictOracleResult],
    certificate: Optional[ExactDualRayConflictCertificate],
    edge: bool,
    caps: OperatorLocalizedPhaseEdgeCaps,
    live_build: OperatorHZBuild,
) -> OperatorLocalizedPhaseEdgeCandidateResult:
    live_hz = vars(live_build).get("hz")
    if type(live_hz) is not SparseHZono:
        raise OperatorLocalizedPhaseEdgeError("live_parent_disappeared")
    terminal_parent = sparse_hz_semantic_digest(live_hz)
    terminal_tags = _row_tag_digest(live_hz)
    terminal_build = _build_binding(
        live_build,
        parent_digest=terminal_parent,
        row_tag_digest=terminal_tags,
        producer_nonempty_seal_verified=(
            producer_nonempty_seal_verified
        ),
        source_modes=source_modes,
        source_modes_sha256=source_modes_sha256,
    )
    unchanged = (
        terminal_parent == parent_digest
        and terminal_tags == row_tag_digest
        and terminal_build == build_binding
    )
    if not unchanged:
        status = "parent_mutated"
        reason = "live_operator_build_terminal_seal_mismatch"
        certificate = None
        edge = False
    if edge is not (certificate is not None):
        raise OperatorLocalizedPhaseEdgeError("adapter_edge_certificate_invariant_failed")
    localized_digest = (
        None if localized_result is None else localized_result.result_sha256
    )
    placeholder = OperatorLocalizedPhaseEdgeCandidateResult(
        status=status,
        reason=reason,
        enabled=enabled,
        build_binding_sha256=build_binding,
        parent_semantic_digest=parent_digest,
        terminal_parent_semantic_digest=terminal_parent,
        focused_property_digest=property_digest,
        operator_row_tag_digest=row_tag_digest,
        terminal_operator_row_tag_digest=terminal_tags,
        selection_digest=selection_digest,
        subset_binding_digest=subset_digest,
        ordered_source_frame_sha256=source_frame_digest,
        producer_nonempty_seal_verified=(
            producer_nonempty_seal_verified
        ),
        source_modes=source_modes,
        source_modes_sha256=source_modes_sha256,
        ranked_phases=ranked,
        literals=literals,
        omitted_zero_bcol_ids=omitted,
        excluded_selected_bcol_ids=excluded,
        localized_result=localized_result,
        localized_result_sha256=localized_digest,
        certificate=certificate,
        edge_accepted=edge,
        parent_unchanged=unchanged,
        caps=caps,
        result_sha256="",
    )
    return OperatorLocalizedPhaseEdgeCandidateResult(
        **{
            **placeholder.__dict__,
            "result_sha256": _sha256(_result_payload(placeholder, include_digest=False)),
        }
    )


def _make_static_disabled_result(
) -> OperatorLocalizedPhaseEdgeCandidateResult:
    """Construct the process-independent disabled receipt without inputs."""

    source_modes: Tuple[Tuple[str, bool], ...] = ()
    modes_digest = _source_modes_digest(source_modes)
    placeholder = OperatorLocalizedPhaseEdgeCandidateResult(
        status="disabled",
        reason="operator_localized_phase_edge_default_off_static",
        enabled=False,
        build_binding_sha256=None,
        parent_semantic_digest=None,
        terminal_parent_semantic_digest=None,
        focused_property_digest=None,
        operator_row_tag_digest=None,
        terminal_operator_row_tag_digest=None,
        selection_digest=None,
        subset_binding_digest=None,
        ordered_source_frame_sha256=None,
        producer_nonempty_seal_verified=False,
        source_modes=source_modes,
        source_modes_sha256=modes_digest,
        ranked_phases=(),
        literals=(),
        omitted_zero_bcol_ids=(),
        excluded_selected_bcol_ids=(),
        localized_result=None,
        localized_result_sha256=None,
        certificate=None,
        edge_accepted=False,
        parent_unchanged=False,
        caps=None,
        result_sha256="",
    )
    return OperatorLocalizedPhaseEdgeCandidateResult(
        **{
            **placeholder.__dict__,
            "result_sha256": _sha256(
                _result_payload(placeholder, include_digest=False)
            ),
        }
    )


_STATIC_DISABLED_RESULT = _make_static_disabled_result()


def run_operator_localized_phase_edge_candidate(
    build: OperatorHZBuild,
    focused_rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    *,
    deadline: float,
    enabled: bool = False,
    selection_max_rivals: int = 128,
    selection_max_binaries: int = 16_384,
    selection_max_work_items: int = 5_000_000,
    selection_timeout_seconds: float = 5.0,
    max_parent_variables: int = _DEFAULT_MAX_PARENT_VARIABLES,
    max_parent_rows: int = _DEFAULT_MAX_PARENT_ROWS,
    max_parent_nonzeros: int = _DEFAULT_MAX_PARENT_NONZEROS,
    max_parent_buffer_items: int = _DEFAULT_MAX_PARENT_BUFFER_ITEMS,
    max_top_literals: int = _STRICT_TOP_LITERALS,
    max_total_pairs: int = _STRICT_TOTAL_PAIRS,
    max_source_terms: int = _DEFAULT_MAX_SOURCE_TERMS,
    max_multiplier_bits: int = _DEFAULT_MAX_MULTIPLIER_BITS,
    max_exact_bits: int = _DEFAULT_MAX_EXACT_BITS,
    max_exact_nonzeros: int = _DEFAULT_MAX_EXACT_NONZEROS,
    localized_row_tiers: Tuple[int, ...] = _STRICT_LOCALIZED_ROW_TIERS,
    localized_max_selected_nnz: int = 1_000_000,
    localized_max_source_terms: int = 128,
) -> OperatorLocalizedPhaseEdgeCandidateResult:
    """Run one deterministic Operator E2 proposal, default-off and isolated."""

    if type(enabled) is not bool:
        raise OperatorLocalizedPhaseEdgeError("enabled_not_bool")
    if not enabled:
        # This branch intentionally precedes every read or validation of all
        # caller-owned objects, deadlines, caps, digests, and snapshots.
        return _STATIC_DISABLED_RESULT
    if type(selection) is not OperatorExactReLUPhaseSelection:
        raise OperatorLocalizedPhaseEdgeError("selection_wrong_type")
    try:
        deadline_value = clique_module._normalize_deadline(deadline)
        caps, clique_caps = _normalize_caps(
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
            max_source_terms=max_source_terms,
            max_multiplier_bits=max_multiplier_bits,
            max_exact_bits=max_exact_bits,
            max_exact_nonzeros=max_exact_nonzeros,
            localized_row_tiers=localized_row_tiers,
            localized_max_selected_nnz=localized_max_selected_nnz,
            localized_max_source_terms=localized_max_source_terms,
        )
        if caps.localized_max_source_terms != caps.max_source_terms:
            raise OperatorLocalizedPhaseEdgeError(
                "localized_and_exact_source_term_caps_mismatch"
            )
        if type(build) is not OperatorHZBuild:
            raise OperatorLocalizedPhaseEdgeError("build_wrong_type")
        live_hz = vars(build).get("hz")
        if type(live_hz) is not SparseHZono:
            raise OperatorLocalizedPhaseEdgeError("build_parent_wrong_type")
        initial_parent = sparse_hz_semantic_digest(live_hz)
        initial_tags = _row_tag_digest(live_hz)
        source = clique_module._snapshot_operator_build(
            build,
            caps=clique_caps,
            deadline=deadline_value,
            require_materializer_source=True,
        )
        private_hz = source.build.hz
        private_parent_digest = sparse_hz_semantic_digest(private_hz)
        if (
            source.producer_nonempty_seal_verified is not True
            or source.private_parent_semantic_digest
            != private_parent_digest
        ):
            raise OperatorLocalizedPhaseEdgeError(
                "operator_constructive_nonempty_source_seal_invalid"
            )
        source_modes = _validate_source_modes(source.source_modes)
        source_modes_sha256 = _source_modes_digest(source_modes)
        build_binding = _build_binding(
            build,
            parent_digest=initial_parent,
            row_tag_digest=initial_tags,
            producer_nonempty_seal_verified=True,
            source_modes=source_modes,
            source_modes_sha256=source_modes_sha256,
        )
        layout = clique_module._exact_hz_core_layout(private_hz)
        rivals = clique_module._snapshot_rivals(
            focused_rivals,
            output_width=layout.n_out,
            maximum=caps.selection_max_rivals,
            deadline=deadline_value,
        )
        trusted = clique_module._verify_live_selection(
            source.build,
            rivals,
            selection,
            deadline=deadline_value,
            selection_max_rivals=caps.selection_max_rivals,
            selection_max_binaries=caps.selection_max_binaries,
            selection_max_work_items=caps.selection_max_work_items,
            selection_timeout_seconds=caps.selection_timeout_seconds,
        )
        parent_digest = sparse_hz_semantic_digest(private_hz)
        if (
            parent_digest != trusted.parent_semantic_digest
            or parent_digest != initial_parent
            or trusted.operator_row_tag_digest != initial_tags
        ):
            raise OperatorLocalizedPhaseEdgeError("trusted_selection_build_binding_stale")
        ranked, omitted, excluded = clique_module._ranked_subset(
            trusted,
            caps=clique_caps,
            deadline=deadline_value,
        )
        subset_digest = clique_module._subset_binding_digest(
            selection=trusted,
            caps=clique_caps,
            ranked=ranked,
            omitted_zero_bcol_ids=omitted,
            excluded_selected_bcol_ids=excluded,
            deadline=deadline_value,
        )
        literals = clique_module._make_bound_literals(
            parent_digest=parent_digest,
            subset_digest=subset_digest,
            ranked=ranked,
        )
        source_frame: Optional[str] = None
        if len(literals) == 2:
            source_frame = _ordered_source_frame_digest(
                private_hz,
                parent_digest=parent_digest,
                deadline=deadline_value,
            )
        if len(literals) != 2 or source_frame is None:
            return _finish(
                status="insufficient_ranked_literals",
                reason="operator_selection_has_fewer_than_two_nonzero_literals",
                enabled=True,
                build_binding=build_binding,
                parent_digest=parent_digest,
                property_digest=trusted.property_digest,
                row_tag_digest=trusted.operator_row_tag_digest,
                selection_digest=trusted.selection_digest,
                subset_digest=subset_digest,
                source_frame_digest=None,
                producer_nonempty_seal_verified=True,
                source_modes=source_modes,
                source_modes_sha256=source_modes_sha256,
                ranked=ranked,
                literals=literals,
                omitted=omitted,
                excluded=excluded,
                localized_result=None,
                certificate=None,
                edge=False,
                caps=caps,
                live_build=build,
            )
        localized = localized_oracle.run_localized_phase_conflict_oracle_candidate(
            private_hz,
            literals,
            property_digest=subset_digest,
            parent_digest=parent_digest,
            source_frame_digest=source_frame,
            deadline=deadline_value,
            enabled=True,
            row_tiers=caps.localized_row_tiers,
            max_selected_nnz=caps.localized_max_selected_nnz,
            max_source_terms=caps.localized_max_source_terms,
        )
        accepted, certificate = _validate_localized_result(
            localized,
            hz=private_hz,
            literals=literals,
            subset_digest=subset_digest,
            parent_digest=parent_digest,
            source_frame_digest=source_frame,
            caps=caps,
            deadline=deadline_value,
        )
        if sparse_hz_semantic_digest(private_hz) != parent_digest:
            raise OperatorLocalizedPhaseEdgeError("private_parent_mutated")
        return _finish(
            status=(
                "certified_localized_phase_edge"
                if accepted
                else "no_certified_localized_phase_edge"
            ),
            reason=(
                "localized_full_parent_exact_certificate_accepted"
                if accepted
                else "localized_candidate_returned_no_exact_edge"
            ),
            enabled=True,
            build_binding=build_binding,
            parent_digest=parent_digest,
            property_digest=trusted.property_digest,
            row_tag_digest=trusted.operator_row_tag_digest,
            selection_digest=trusted.selection_digest,
            subset_digest=subset_digest,
            source_frame_digest=source_frame,
            producer_nonempty_seal_verified=True,
            source_modes=source_modes,
            source_modes_sha256=source_modes_sha256,
            ranked=ranked,
            literals=literals,
            omitted=omitted,
            excluded=excluded,
            localized_result=localized,
            certificate=certificate,
            edge=accepted,
            caps=caps,
            live_build=build,
        )
    except OperatorLocalizedPhaseEdgeError:
        raise
    except Exception as exc:
        raise OperatorLocalizedPhaseEdgeError(
            type(exc).__name__ + ":" + str(exc)
        ) from exc


__all__ = [
    "OperatorLocalizedPhaseEdgeCandidateResult",
    "OperatorLocalizedPhaseEdgeCaps",
    "OperatorLocalizedPhaseEdgeError",
    "run_operator_localized_phase_edge_candidate",
]
