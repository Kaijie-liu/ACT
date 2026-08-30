#!/usr/bin/env python3
# ===- property_causal_block_integration.py - PC-CBDE integration ---===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------===#
"""Proof-neutral integration of incidence selection and causal block duals.

This module deliberately stops before solver dispatch.  It:

* selects a stored-CSR packet-to-property ordinary cone;
* builds a local original-frame LP from that cone, an explicit generated
  optimization packet, and explicit source rows;
* optimizes the complete bridge/generated/source semantic families under four
  causal ablations; and
* expands every local multiplier back into the unmodified full row frame.

The returned ``row_dual`` values are candidates only.  In particular,
``proof_authority`` is permanently false here: a caller must submit each
full-frame candidate to the independent long-double checker before it can
affect a verification verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.gpu_dual_candidates import (
    ConstraintRowTag,
    OriginalFrameLP,
    property_conditioned_incidence_cone_rows,
)
from act.back_end.hybridz_tf.property_causal_block_dual import (
    CausalDirectionUnion,
    CausalDualBlock,
    PropertyCausalBlockDualCandidates,
    property_causal_block_duals,
)


_FAMILIES = ("bridge", "generated", "source")
_ABLATIONS = (
    ("full", ("bridge", "generated", "source")),
    ("without_generated", ("bridge", "source")),
    ("without_bridge", ("generated", "source")),
    ("without_both", ("source",)),
)


@dataclass(frozen=True)
class PropertyCausalBlockAblation:
    """One expanded full-frame candidate for a declared causal ablation."""

    name: str
    enabled_families: Tuple[str, ...]
    d: np.ndarray
    row_dual: np.ndarray
    candidate_support: np.ndarray
    local_candidate_support: np.ndarray
    optimizer: PropertyCausalBlockDualCandidates
    proof_authority: bool = False


@dataclass(frozen=True)
class PropertyCausalBlockIntegrationCandidates:
    """Candidate-only selector/optimizer integration result.

    ``success=False`` is the fail-closed outcome.  In that case no partial
    ablation candidate is exposed.
    """

    success: bool
    status: str
    diagnostic: str
    property_columns: np.ndarray
    cone_rows: np.ndarray
    local_rows: np.ndarray
    bridge_rows: np.ndarray
    generated_rows: np.ndarray
    source_rows: np.ndarray
    ignored_source_rows: np.ndarray
    ablations: Tuple[PropertyCausalBlockAblation, ...]
    elapsed_seconds: float
    deadline_reached: bool
    proof_authority: bool = False
    method: str = "property_causal_block_integration_v1"

    def ablation(self, name: str) -> PropertyCausalBlockAblation:
        """Return one named ablation, rejecting absent/partial results."""

        for candidate in self.ablations:
            if candidate.name == name:
                return candidate
        raise KeyError(name)


class _FailClosed(ValueError):
    def __init__(self, status: str, diagnostic: str) -> None:
        super().__init__(diagnostic)
        self.status = str(status)
        self.diagnostic = str(diagnostic)


def _empty_i64() -> np.ndarray:
    return np.zeros(0, dtype=np.int64)


def _failure(
    *,
    status: str,
    diagnostic: str,
    started: float,
    deadline_reached: bool = False,
) -> PropertyCausalBlockIntegrationCandidates:
    return PropertyCausalBlockIntegrationCandidates(
        success=False,
        status=str(status),
        diagnostic=str(diagnostic),
        property_columns=_empty_i64(),
        cone_rows=_empty_i64(),
        local_rows=_empty_i64(),
        bridge_rows=_empty_i64(),
        generated_rows=_empty_i64(),
        source_rows=_empty_i64(),
        ignored_source_rows=_empty_i64(),
        ablations=(),
        elapsed_seconds=float(time.monotonic() - started),
        deadline_reached=bool(deadline_reached),
    )


def _canonical_csr(value: Any) -> sp.csr_matrix:
    matrix = sp.csr_matrix(value, dtype=np.float64)
    if not matrix.has_sorted_indices:
        matrix.sort_indices()
    if not matrix.has_canonical_format:
        raise _FailClosed(
            "invalid_frame",
            "original-frame matrix contains duplicate coefficients",
        )
    if matrix.nnz and not np.all(np.isfinite(matrix.data)):
        raise _FailClosed(
            "invalid_frame",
            "original-frame matrix contains non-finite coefficients",
        )
    return matrix


def _strict_indices(
    values: Sequence[int],
    *,
    upper: int,
    name: str,
    allow_empty: bool,
) -> np.ndarray:
    raw = np.asarray(values)
    if allow_empty and raw.ndim == 1 and raw.size == 0:
        return np.zeros(0, dtype=np.int64)
    if (
        raw.ndim != 1
        or raw.dtype.kind not in {"i", "u"}
        or raw.dtype == np.dtype(np.bool_)
        or (not allow_empty and raw.size == 0)
    ):
        raise _FailClosed("invalid_rows", f"{name} must be a 1-D integer list")
    if raw.dtype.kind == "i" and np.any(raw < 0):
        raise _FailClosed("invalid_rows", f"{name} contains a negative index")
    if np.any(raw >= int(upper)):
        raise _FailClosed("invalid_rows", f"{name} contains an out-of-range index")
    result = raw.astype(np.int64, copy=False)
    if np.unique(result).size != result.size:
        raise _FailClosed("invalid_rows", f"{name} contains duplicate indices")
    return result.copy()


def _tag_text(tag: Any, *, expected_global_row: int) -> str:
    if isinstance(tag, ConstraintRowTag):
        if (
            isinstance(tag.global_row, (bool, np.bool_))
            or not isinstance(tag.global_row, (int, np.integer))
            or int(tag.global_row) != int(expected_global_row)
            or not isinstance(tag.block_tag, str)
            or not tag.block_tag
        ):
            raise _FailClosed(
                "invalid_tags",
                "ConstraintRowTag is not aligned with the full row frame",
            )
        return tag.block_tag
    if isinstance(tag, str) and tag:
        return tag
    raise _FailClosed("invalid_tags", "row tags must be non-empty strings or tags")


def _project_full_warm(
    warm: np.ndarray,
    *,
    rl: np.ndarray,
    ru: np.ndarray,
) -> np.ndarray:
    result = np.asarray(warm, dtype=np.float64).copy()
    upper_only = np.isneginf(rl) & np.isfinite(ru)
    lower_only = np.isfinite(rl) & np.isposinf(ru)
    free = np.isneginf(rl) & np.isposinf(ru)
    if np.any(upper_only):
        result[:, upper_only] = np.maximum(result[:, upper_only], 0.0)
    if np.any(lower_only):
        result[:, lower_only] = np.minimum(result[:, lower_only], 0.0)
    if np.any(free):
        result[:, free] = 0.0
    return result


def _full_support(
    *,
    A: sp.csr_matrix,
    rl: np.ndarray,
    ru: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    q: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    values = np.empty(q.shape[0], dtype=np.float64)
    for rival in range(q.shape[0]):
        multiplier = np.asarray(d[rival], dtype=np.float64)
        nonzero = np.flatnonzero(multiplier != 0.0)
        if nonzero.size:
            sides = np.where(
                multiplier[nonzero] >= 0.0,
                ru[nonzero],
                rl[nonzero],
            )
            if not np.all(np.isfinite(sides)):
                values[rival] = np.inf
                continue
            row_support = float(np.dot(multiplier[nonzero], sides))
        else:
            row_support = 0.0
        residual = np.asarray(
            q[rival] - A.transpose() @ multiplier,
            dtype=np.float64,
        ).reshape(-1)
        box_sides = np.where(residual >= 0.0, ub, lb)
        value = row_support + float(np.dot(residual, box_sides))
        values[rival] = value if np.isfinite(value) else np.inf
    return values


def _stable_row_keys(
    *,
    A: sp.csr_matrix,
    rows: np.ndarray,
    tag_texts: Tuple[str, ...],
    rl: np.ndarray,
    ru: np.ndarray,
) -> Tuple[str, ...]:
    """Build semantic/content keys without embedding the mutable row position."""

    bases: list[str] = []
    for raw_row in rows:
        row = int(raw_row)
        start = int(A.indptr[row])
        stop = int(A.indptr[row + 1])
        digest = hashlib.sha256()
        digest.update(tag_texts[row].encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(A.indices[start:stop], dtype=np.int64).tobytes())
        for value in A.data[start:stop]:
            digest.update(float(value).hex().encode("ascii"))
            digest.update(b"\0")
        digest.update(float(rl[row]).hex().encode("ascii"))
        digest.update(b"\0")
        digest.update(float(ru[row]).hex().encode("ascii"))
        bases.append(f"{tag_texts[row]}:{digest.hexdigest()}")

    totals: dict[str, int] = {}
    for base in bases:
        totals[base] = totals.get(base, 0) + 1
    seen: dict[str, int] = {}
    result: list[str] = []
    for base in bases:
        occurrence = seen.get(base, 0)
        seen[base] = occurrence + 1
        result.append(
            base if totals[base] == 1 else f"{base}:duplicate:{occurrence}"
        )
    return tuple(result)


def _incident_columns(A: sp.csr_matrix, rows: np.ndarray) -> Tuple[int, ...]:
    columns: set[int] = set()
    for raw_row in rows:
        row = int(raw_row)
        columns.update(
            int(value)
            for value in A.indices[A.indptr[row]:A.indptr[row + 1]]
        )
    return tuple(sorted(columns))


def property_causal_block_integration(
    frame: OriginalFrameLP,
    q: np.ndarray,
    warm_d_full: np.ndarray,
    *,
    incidence_packet_rows: Sequence[int],
    optimization_packet_rows: Sequence[int],
    source_rows: Sequence[int],
    allowed_row_mask: Any,
    row_tags: Sequence[Any],
    property_columns: Optional[Sequence[int]] = None,
    deadline: Optional[float] = None,
    selector_max_rows: int = 4096,
    selector_max_selected_nnz: int = 1_000_000,
    selector_max_depth: int = 6,
    optimizer_max_updates: int = 64,
    optimizer_max_zero_gain_updates: int = 16,
    optimizer_face_visit_cap: int = 2,
    optimizer_frontier_topk: int = 64,
    optimizer_nnz_cap: int = 96,
) -> PropertyCausalBlockIntegrationCandidates:
    """Select, optimize four ablations, and expand into the full row frame.

    ``deadline`` is one absolute ``time.monotonic()`` deadline shared by the
    selector and all four optimizer calls.  No partial result is returned after
    a deadline or structural failure.
    """

    started = time.monotonic()
    try:
        if deadline is not None:
            if (
                isinstance(deadline, (bool, np.bool_))
                or not isinstance(
                    deadline,
                    (int, float, np.integer, np.floating),
                )
                or not math.isfinite(float(deadline))
            ):
                raise _FailClosed(
                    "invalid_deadline",
                    "deadline must be a finite absolute monotonic time",
                )
            if time.monotonic() >= float(deadline):
                raise _FailClosed("deadline", "deadline expired before selection")

        numeric_caps = (
            ("optimizer_max_updates", optimizer_max_updates, 0, 4096),
            (
                "optimizer_max_zero_gain_updates",
                optimizer_max_zero_gain_updates,
                0,
                1024,
            ),
            ("optimizer_face_visit_cap", optimizer_face_visit_cap, 1, 32),
            ("optimizer_frontier_topk", optimizer_frontier_topk, 1, 4096),
            ("optimizer_nnz_cap", optimizer_nnz_cap, 1, 8192),
        )
        for name, value, lower, upper in numeric_caps:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < int(lower)
                or int(value) > int(upper)
            ):
                raise _FailClosed(
                    "invalid_caps",
                    f"{name} must be an integer in [{lower}, {upper}]",
                )

        A = _canonical_csr(frame.A)
        rl = np.asarray(frame.rl, dtype=np.float64).reshape(-1)
        ru = np.asarray(frame.ru, dtype=np.float64).reshape(-1)
        lb = np.asarray(frame.lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(frame.ub, dtype=np.float64).reshape(-1)
        n_rows, n_variables = map(int, A.shape)
        if (
            A.shape != (rl.size, lb.size)
            or ru.size != rl.size
            or ub.size != lb.size
            or len(frame.row_tags) != n_rows
            or n_rows <= 0
            or n_variables <= 0
            or not np.all(np.isfinite(lb))
            or not np.all(np.isfinite(ub))
            or np.any(lb > ub)
            or np.any(np.isnan(rl))
            or np.any(np.isnan(ru))
            or np.any(rl > ru)
            or np.any(np.isposinf(rl))
            or np.any(np.isneginf(ru))
        ):
            raise _FailClosed("invalid_frame", "original-frame LP shape/data mismatch")

        q64 = np.asarray(q, dtype=np.float64)
        warm64 = np.asarray(warm_d_full, dtype=np.float64)
        if (
            q64.ndim != 2
            or q64.shape[0] != 1
            or q64.shape[1] != n_variables
            or not np.all(np.isfinite(q64))
        ):
            raise _FailClosed(
                "invalid_objective",
                "q must be finite with shape [1, frame variables]",
            )
        if (
            warm64.shape != (q64.shape[0], n_rows)
            or not np.all(np.isfinite(warm64))
        ):
            raise _FailClosed(
                "invalid_warm_start",
                "warm_d_full must be finite with shape [objectives, frame rows]",
            )

        incidence_packet = _strict_indices(
            incidence_packet_rows,
            upper=n_rows,
            name="incidence_packet_rows",
            allow_empty=False,
        )
        optimization_packet = _strict_indices(
            optimization_packet_rows,
            upper=n_rows,
            name="optimization_packet_rows",
            allow_empty=False,
        )
        sources = _strict_indices(
            source_rows,
            upper=n_rows,
            name="source_rows",
            allow_empty=True,
        )
        if optimization_packet.size > 64:
            raise _FailClosed(
                "optimization_packet_cap",
                "optimization_packet_rows exceeds the hard 64-row cap",
            )
        if not set(optimization_packet.tolist()).issubset(
            set(incidence_packet.tolist())
        ):
            raise _FailClosed(
                "packet_mismatch",
                "optimization packet must be a subset of the incidence packet",
            )

        allowed = np.asarray(allowed_row_mask)
        if (
            allowed.shape != (n_rows,)
            or allowed.dtype != np.dtype(np.bool_)
        ):
            raise _FailClosed(
                "invalid_allowed_mask",
                "allowed_row_mask must be bool with one entry per full row",
            )
        tags_input = tuple(row_tags)
        if len(tags_input) != n_rows:
            raise _FailClosed(
                "invalid_tags",
                "row_tags must contain one entry per full row",
            )
        tag_texts = tuple(
            _tag_text(tag, expected_global_row=row)
            for row, tag in enumerate(tags_input)
        )
        frame_tags_input = tuple(frame.row_tags)
        frame_tag_texts = tuple(
            _tag_text(tag, expected_global_row=row)
            for row, tag in enumerate(frame_tags_input)
        )
        if tag_texts != frame_tag_texts:
            raise _FailClosed(
                "invalid_tags",
                "row_tags do not match the OriginalFrameLP row metadata",
            )
        for supplied, stored in zip(tags_input, frame_tags_input):
            if (
                isinstance(supplied, ConstraintRowTag)
                and isinstance(stored, ConstraintRowTag)
                and (
                    supplied.sense != stored.sense
                    or int(supplied.block_local_row)
                    != int(stored.block_local_row)
                )
            ):
                raise _FailClosed(
                    "invalid_tags",
                    "row_tags do not match the OriginalFrameLP row metadata",
                )
        if any(
            not tag_texts[int(row)].startswith("property_micro_rlt:")
            for row in incidence_packet
        ):
            raise _FailClosed(
                "invalid_packet_tags",
                "incidence packet rows must be generated property_micro_rlt rows",
            )

        if property_columns is None:
            properties = np.flatnonzero(
                np.any(q64 != 0.0, axis=0)
            ).astype(np.int64, copy=False)
            if properties.size == 0:
                raise _FailClosed(
                    "invalid_property_columns",
                    "cannot derive property columns from a zero objective",
                )
        else:
            properties = _strict_indices(
                property_columns,
                upper=n_variables,
                name="property_columns",
                allow_empty=False,
            )

        cone = property_conditioned_incidence_cone_rows(
            A,
            property_columns=properties,
            packet_rows=incidence_packet,
            row_tags=tags_input,
            allowed_row_mask=allowed,
            max_rows=selector_max_rows,
            max_selected_nnz=selector_max_selected_nnz,
            max_depth=selector_max_depth,
            deadline=deadline,
        )
        cone = np.asarray(cone, dtype=np.int64).reshape(-1)
        if cone.size == 0:
            if deadline is not None and time.monotonic() >= float(deadline):
                raise _FailClosed("deadline", "selector exhausted the deadline")
            raise _FailClosed(
                "incidence_path_unavailable",
                "no complete ordinary packet-to-property path was selected",
            )
        if (
            np.unique(cone).size != cone.size
            or np.any(cone < 0)
            or np.any(cone >= n_rows)
            or not np.all(allowed[cone])
        ):
            raise _FailClosed(
                "selector_contract_violation",
                "selector returned malformed or disallowed rows",
            )
        cone_set = set(cone.tolist())
        requested_source_set = set(sources.tolist())
        source_set = requested_source_set.intersection(cone_set)
        selected_sources = np.asarray(sorted(source_set), dtype=np.int64)
        ignored_sources = np.asarray(
            sorted(requested_source_set - cone_set),
            dtype=np.int64,
        )
        packet_set = set(optimization_packet.tolist())
        if source_set.intersection(packet_set):
            raise _FailClosed(
                "family_overlap",
                "source and generated optimization rows must be disjoint",
            )
        if any(
            tag_texts[int(row)].startswith("add_materialize:")
            for row in selected_sources
        ):
            raise _FailClosed(
                "source_splits_atomic_add",
                "source rows cannot split an atomic ADD bridge block",
            )
        bridge = np.asarray(
            sorted(cone_set - source_set),
            dtype=np.int64,
        )
        if bridge.size == 0:
            raise _FailClosed(
                "empty_bridge",
                "selected causal path contains no bridge rows",
            )
        if cone_set.intersection(packet_set):
            raise _FailClosed(
                "selector_contract_violation",
                "selector returned one of its generated packet rows",
            )

        local_rows = np.asarray(
            sorted(cone_set | source_set | packet_set),
            dtype=np.int64,
        )
        upper_only = np.isneginf(rl[local_rows]) & np.isfinite(ru[local_rows])
        if not np.all(upper_only):
            raise _FailClosed(
                "unsupported_row_domain",
                "PC-CBDE v1 local families must contain upper-only rows",
            )
        if deadline is not None and time.monotonic() >= float(deadline):
            raise _FailClosed("deadline", "deadline expired after selection")

        projected_warm = _project_full_warm(warm64, rl=rl, ru=ru)
        outside_warm = projected_warm.copy()
        outside_warm[:, local_rows] = 0.0
        local_q = q64 - np.asarray(
            A.transpose() @ outside_warm.transpose(),
            dtype=np.float64,
        ).transpose()

        global_to_local = {
            int(global_row): local_row
            for local_row, global_row in enumerate(local_rows.tolist())
        }
        local_A = A[local_rows, :].tocsr()
        local_tag_texts = tuple(tag_texts[int(row)] for row in local_rows)
        local_frame = OriginalFrameLP(
            A=local_A,
            rl=rl[local_rows].copy(),
            ru=ru[local_rows].copy(),
            lb=lb.copy(),
            ub=ub.copy(),
            row_tags=tuple(
                ConstraintRowTag(
                    global_row=local_row,
                    sense="ub",
                    block_tag=local_tag_texts[local_row],
                    block_local_row=0,
                )
                for local_row in range(local_rows.size)
            ),
        )
        local_warm = projected_warm[:, local_rows].copy()

        family_global_rows = {
            "bridge": bridge,
            "generated": optimization_packet,
            "source": selected_sources,
        }
        blocks: list[CausalDualBlock] = []
        block_id_by_family: dict[str, str] = {}
        for family in _FAMILIES:
            global_rows = family_global_rows[family]
            if global_rows.size == 0:
                continue
            local_family_rows = np.asarray(
                [global_to_local[int(row)] for row in global_rows],
                dtype=np.int64,
            )
            block_id = f"{family}_family"
            block_id_by_family[family] = block_id
            blocks.append(
                CausalDualBlock(
                    block_id=block_id,
                    family_id=family,
                    global_rows=tuple(int(row) for row in local_family_rows),
                    stable_row_keys=_stable_row_keys(
                        A=A,
                        rows=global_rows,
                        tag_texts=tag_texts,
                        rl=rl,
                        ru=ru,
                    ),
                    incident_columns=_incident_columns(A, global_rows),
                )
            )

        unions = [
            CausalDirectionUnion(
                union_id=f"{family}_family",
                block_ids=(block_id,),
            )
            for family, block_id in block_id_by_family.items()
        ]
        unions.append(
            CausalDirectionUnion(
                union_id="full_causal_path",
                block_ids=tuple(
                    block_id_by_family[family]
                    for family in _FAMILIES
                    if family in block_id_by_family
                ),
            )
        )

        ablations: list[PropertyCausalBlockAblation] = []
        available_families = set(block_id_by_family)
        for ablation_name, requested_families in _ABLATIONS:
            if deadline is not None and time.monotonic() >= float(deadline):
                raise _FailClosed(
                    "deadline",
                    f"deadline expired before {ablation_name}",
                )
            enabled = tuple(
                family
                for family in requested_families
                if family in available_families
            )
            optimized = property_causal_block_duals(
                local_frame,
                local_q,
                local_warm,
                blocks=tuple(blocks),
                direction_unions=tuple(unions),
                enabled_families=enabled,
                max_updates=int(optimizer_max_updates),
                max_zero_gain_updates=int(optimizer_max_zero_gain_updates),
                face_visit_cap=int(optimizer_face_visit_cap),
                frontier_topk=int(optimizer_frontier_topk),
                nnz_cap=int(optimizer_nnz_cap),
                deadline=deadline,
            )
            if optimized.deadline_reached:
                raise _FailClosed(
                    "deadline",
                    f"optimizer deadline reached during {ablation_name}",
                )
            expanded = projected_warm.copy()
            expanded[:, local_rows] = optimized.d
            support = _full_support(
                A=A,
                rl=rl,
                ru=ru,
                lb=lb,
                ub=ub,
                q=q64,
                d=expanded,
            )
            if not np.all(np.isfinite(support)):
                raise _FailClosed(
                    "nonfinite_candidate",
                    f"{ablation_name} expanded support is non-finite",
                )
            ablations.append(
                PropertyCausalBlockAblation(
                    name=ablation_name,
                    enabled_families=enabled,
                    d=expanded,
                    row_dual=-expanded,
                    candidate_support=support,
                    local_candidate_support=(
                        optimized.candidate_support.copy()
                    ),
                    optimizer=optimized,
                )
            )

        return PropertyCausalBlockIntegrationCandidates(
            success=True,
            status="candidate_ready_unchecked",
            diagnostic=(
                "full-frame candidates require the independent long-double "
                "checker"
            ),
            property_columns=properties.copy(),
            cone_rows=cone.copy(),
            local_rows=local_rows,
            bridge_rows=bridge,
            generated_rows=optimization_packet.copy(),
            source_rows=selected_sources,
            ignored_source_rows=ignored_sources,
            ablations=tuple(ablations),
            elapsed_seconds=float(time.monotonic() - started),
            deadline_reached=False,
        )
    except _FailClosed as exc:
        return _failure(
            status=exc.status,
            diagnostic=exc.diagnostic,
            started=started,
            deadline_reached=(exc.status == "deadline"),
        )
    except Exception as exc:
        return _failure(
            status=f"invalid:{type(exc).__name__}",
            diagnostic=str(exc)[:240],
            started=started,
            deadline_reached=(
                deadline is not None
                and isinstance(deadline, (int, float, np.integer, np.floating))
                and not isinstance(deadline, (bool, np.bool_))
                and math.isfinite(float(deadline))
                and time.monotonic() >= float(deadline)
            ),
        )


__all__ = [
    "PropertyCausalBlockAblation",
    "PropertyCausalBlockIntegrationCandidates",
    "property_causal_block_integration",
]
