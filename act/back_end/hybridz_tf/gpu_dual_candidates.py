#!/usr/bin/env python3
# ===- gpu_dual_candidates.py - GPU LP-dual candidate generation -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------====#
"""Untrusted batched GPU candidates for the original operator-HZ LP frame.

This module deliberately has no proof authority.  It minimizes the same
Lagrangian support expression which
``solver_hz._hz_independent_lp_lagrangian_upper`` later reconstructs in
long-double arithmetic::

    support_[rl,ru](d) + support_[lb,ub](q - A.T @ d).

The important interface convention is explicit:

* ``d`` is the maximization-certificate multiplier.
* the existing independent checker accepts a HiGHS/minimization-style
  ``row_dual`` and internally computes ``d = -row_dual``.
* consequently this module returns ``row_dual = -d``.

Candidates are optimized directly against the *exported* operator-HZ rows.
There is therefore no attempt to reinterpret DualSolver's layerwise ``nu`` as
an operator-HZ row multiplier.  Such a reinterpretation is currently
ambiguous because operator-HZ materializes ReLU/ADD values in normalized
factor coordinates, represents numerical equalities as two outward bands,
and filters structurally empty rows during assembly.

The helper supports CUDA and CPU.  CUDA is the intended large-class path;
CPU keeps the deterministic toy audit runnable on machines without a GPU.
Every returned multiplier must still be checked independently before it can
affect a verification verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch


@dataclass(frozen=True)
class ConstraintRowTag:
    """Stable location of one row in the exported original-frame matrix."""

    global_row: int
    sense: str
    block_tag: str
    block_local_row: int


@dataclass(frozen=True)
class OriginalFrameLP:
    """The exact stored-float row/column frame consumed by the checker."""

    A: sp.csr_matrix
    rl: np.ndarray
    ru: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    row_tags: Tuple[ConstraintRowTag, ...]

    @property
    def n_rows(self) -> int:
        return int(self.A.shape[0])

    @property
    def n_variables(self) -> int:
        return int(self.A.shape[1])


@dataclass(frozen=True)
class BatchedDualCandidates:
    """Candidate-only result; ``row_dual`` has no proof authority."""

    row_dual: np.ndarray
    initial_support: np.ndarray
    candidate_support: np.ndarray
    selected_rows: np.ndarray
    device: str
    dtype: str
    steps_requested: int
    steps_completed: int
    elapsed_seconds: float
    deadline_reached: bool
    proof_authority: bool = False
    optimization_method: str = (
        "projected_adam_smooth_absolute_continuation_v2"
    )
    smooth_abs_start: float = 0.05
    smooth_abs_end: float = 0.0
    wavefront_updates: int = 0
    wavefront_support_improved_rows: int = 0
    wavefront_best_improvement: float = 0.0
    wavefront_elapsed_seconds: float = 0.0
    wavefront_selected_constraint_count: int = 0


@dataclass(frozen=True)
class CoordinateWavefrontCandidates:
    """Candidate-only sparse duals from exact one-coordinate line searches.

    ``d`` uses the maximization-certificate convention.  The routine has no
    proof authority; it is only a property-conditioned initializer for the
    independently checked original-frame certificate.
    """

    d: np.ndarray
    initial_support: np.ndarray
    candidate_support: np.ndarray
    updates: int
    selected_constraint_count: int
    elapsed_seconds: float
    deadline_reached: bool
    proof_authority: bool = False
    method: str = "property_conditioned_l1_coordinate_wavefront_v1"


@dataclass(frozen=True)
class ReluSensitivity:
    """One heuristic property-sensitivity entry for preactivation scheduling."""

    layer_id: int
    neuron: int
    score: float
    rival: int


def _canonical_csr(value: Any, *, shape: Optional[Tuple[int, int]] = None) -> sp.csr_matrix:
    matrix = sp.csr_matrix(value, dtype=np.float64, shape=shape)
    if not matrix.has_sorted_indices:
        # Sorting is an exact permutation of stored coefficients.
        matrix.sort_indices()
    if not matrix.has_canonical_format:
        # Summing duplicate coefficients would introduce an extra rounded
        # transformation between the candidate and checker coordinates.
        raise ValueError("original-frame matrix has duplicate coefficients")
    if matrix.nnz and not np.all(np.isfinite(matrix.data)):
        raise ValueError("original-frame matrix has non-finite coefficients")
    return matrix


def property_conditioned_incidence_cone_rows(
    A: Any,
    *,
    property_columns: Sequence[int],
    packet_rows: Sequence[int],
    row_tags: Sequence[Any],
    allowed_row_mask: Any,
    max_rows: int = 4096,
    max_selected_nnz: int = 1_000_000,
    max_depth: int = 6,
    deadline: Optional[float] = None,
) -> np.ndarray:
    """Select one bounded packet-to-property path in the actual CSR graph.

    This is a proof-neutral scheduling helper.  Columns are connected only
    when they co-occur in an allowed stored CSR row; neither layer numbers nor
    tag proximity create graph edges.  When an ``add_materialize`` layer is
    first reached, its complete forward/reverse signature multisets are
    audited as exact stored-float negatives.  Search expansion then uses only
    the reached equality-coordinate pair (or its exact duplicate group), not
    the entire layer.

    ``packet_rows`` seed the reachable columns but are never returned.
    Generated ``property_micro_rlt:*`` rows are also never traversed, even
    when their entry in ``allowed_row_mask`` is true.  Malformed metadata,
    incomplete ADD blocks, expired deadlines, and any search/candidate cap
    violation fail closed to an empty candidate.
    """

    empty = np.zeros(0, dtype=np.int64)
    hard_max_rows = 4096
    hard_max_selected_nnz = 1_000_000
    hard_max_depth = 6

    try:
        cap_values = (
            (max_rows, hard_max_rows),
            (max_selected_nnz, hard_max_selected_nnz),
            (max_depth, hard_max_depth),
        )
        for value, hard_limit in cap_values:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) <= 0
                or int(value) > hard_limit
            ):
                return empty
        row_cap = int(max_rows)
        nnz_cap = int(max_selected_nnz)
        depth_cap = int(max_depth)

        absolute_deadline: Optional[float]
        if deadline is None:
            absolute_deadline = None
        elif (
            isinstance(deadline, (bool, np.bool_))
            or not isinstance(
                deadline,
                (int, float, np.integer, np.floating),
            )
        ):
            return empty
        else:
            absolute_deadline = float(deadline)
            if not math.isfinite(absolute_deadline):
                return empty

        def deadline_reached() -> bool:
            return (
                absolute_deadline is not None
                and time.monotonic() >= absolute_deadline
            )

        if deadline_reached():
            return empty

        matrix = _canonical_csr(A)
        n_rows, n_columns = map(int, matrix.shape)
        if (
            n_rows <= 0
            or n_rows > 131_072
            or n_columns <= 0
            or int(matrix.nnz) <= 0
            or int(matrix.nnz) > 12_000_000
            or deadline_reached()
        ):
            return empty
        # Copying a production float64 CSR solely to eliminate explicit zeros
        # would exceed the selector's memory budget.  They cannot be graph
        # edges, so fail closed instead.
        if np.any(matrix.data == 0.0) or deadline_reached():
            return empty

        def strict_indices(
            values: Sequence[int],
            *,
            upper: int,
        ) -> Optional[np.ndarray]:
            raw = np.asarray(values)
            if (
                raw.ndim != 1
                or raw.size == 0
                or raw.dtype.kind not in {"i", "u"}
                or raw.dtype == np.dtype(np.bool_)
                or np.any(raw >= upper)
            ):
                return None
            if raw.dtype.kind == "i" and np.any(raw < 0):
                return None
            result = raw.astype(np.int64, copy=False)
            if np.unique(result).size != result.size:
                return None
            return result

        properties = strict_indices(property_columns, upper=n_columns)
        packets = strict_indices(packet_rows, upper=n_rows)
        if properties is None or packets is None:
            return empty

        allowed = np.asarray(allowed_row_mask)
        if (
            allowed.ndim != 1
            or allowed.shape != (n_rows,)
            or allowed.dtype != np.dtype(np.bool_)
        ):
            return empty
        allowed = allowed.copy()

        tags_input = tuple(row_tags)
        if len(tags_input) != n_rows:
            return empty
        tag_texts: list[str] = []
        for row, tag in enumerate(tags_input):
            if row % 256 == 0 and deadline_reached():
                return empty
            if isinstance(tag, ConstraintRowTag):
                if (
                    isinstance(tag.global_row, (bool, np.bool_))
                    or not isinstance(tag.global_row, (int, np.integer))
                    or int(tag.global_row) != row
                    or not isinstance(tag.block_tag, str)
                ):
                    return empty
                text = tag.block_tag
            elif isinstance(tag, str):
                text = tag
            else:
                return empty
            if not text:
                return empty
            tag_texts.append(text)

        packet_set = {int(row) for row in packets}
        eligible = allowed
        for row, text in enumerate(tag_texts):
            if row in packet_set or text.startswith("property_micro_rlt:"):
                eligible[row] = False

        add_groups: dict[int, dict[str, list[int]]] = {}
        add_group_by_row = np.full(n_rows, -1, dtype=np.int32)
        add_group_index: dict[int, int] = {}
        for row, text in enumerate(tag_texts):
            if row % 256 == 0 and deadline_reached():
                return empty
            if not text.startswith("add_materialize:"):
                continue
            parts = text.split(":")
            if (
                len(parts) != 3
                or parts[0] != "add_materialize"
                or not parts[1]
                or not parts[1].isascii()
                or not parts[1].isdigit()
                or parts[2] not in {"forward", "reverse"}
            ):
                return empty
            layer_id = int(parts[1])
            group_index = add_group_index.get(layer_id)
            if group_index is None:
                group_index = len(add_group_index)
                add_group_index[layer_id] = group_index
            group = add_groups.setdefault(
                layer_id,
                {"forward": [], "reverse": []},
            )
            group[parts[2]].append(row)
            add_group_by_row[row] = int(group_index)

        def row_signature(
            row: int,
            *,
            negate: bool = False,
        ) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
            start = int(matrix.indptr[row])
            stop = int(matrix.indptr[row + 1])
            indices = tuple(int(value) for value in matrix.indices[start:stop])
            if negate:
                values = tuple(
                    float(-float(value)).hex()
                    for value in matrix.data[start:stop]
                )
            else:
                values = tuple(
                    float(value).hex()
                    for value in matrix.data[start:stop]
                )
            return indices, values

        add_layers_by_index = [0] * len(add_group_index)
        for layer_id, group_index in add_group_index.items():
            add_layers_by_index[group_index] = layer_id

        # A node is materialized only after the CSC frontier reaches one of
        # its rows.  In particular, disconnected ADD blocks never pay the
        # potentially expensive exact negative-row audit.
        node_cache: dict[
            Tuple[Any, ...],
            Tuple[
                Tuple[int, ...],
                Tuple[int, ...],
                int,
                Tuple[Any, ...],
            ],
        ] = {}
        oversized_nodes: set[Tuple[Any, ...]] = set()
        audited_add_groups: set[int] = set()
        add_atom_key_by_row: dict[int, Tuple[Any, ...]] = {}

        def audit_add_group(group_index: int) -> None:
            """Audit one full tag block, then atomize exact coordinates."""

            if group_index in audited_add_groups:
                return
            layer_id = add_layers_by_index[group_index]
            group = add_groups[layer_id]
            forward = tuple(group["forward"])
            reverse = tuple(group["reverse"])
            all_rows = forward + reverse
            if (
                not forward
                or not reverse
                or len(forward) != len(reverse)
                or not all(bool(eligible[row]) for row in all_rows)
            ):
                raise ValueError("incomplete allowed ADD tag block")

            forward_by_signature: dict[
                Tuple[Tuple[int, ...], Tuple[str, ...]],
                list[int],
            ] = {}
            reverse_by_signature: dict[
                Tuple[Tuple[int, ...], Tuple[str, ...]],
                list[int],
            ] = {}
            for offset, row in enumerate(forward):
                if offset % 256 == 0 and deadline_reached():
                    raise TimeoutError
                signature = row_signature(row)
                forward_by_signature.setdefault(signature, []).append(row)
            for offset, row in enumerate(reverse):
                if offset % 256 == 0 and deadline_reached():
                    raise TimeoutError
                signature = row_signature(row, negate=True)
                reverse_by_signature.setdefault(signature, []).append(row)
            if (
                forward_by_signature.keys()
                != reverse_by_signature.keys()
                or any(
                    len(rows)
                    != len(reverse_by_signature[signature])
                    for signature, rows
                    in forward_by_signature.items()
                )
            ):
                raise ValueError("ADD reverse rows are not exact negatives")

            # The whole layer is audited above, but scheduling granularity is
            # one equality coordinate.  Identical stored signatures form one
            # small duplicate atom because no row-order pairing is canonical.
            for atom_index, signature in enumerate(
                sorted(forward_by_signature)
            ):
                atom_rows = tuple(
                    sorted(
                        forward_by_signature[signature]
                        + reverse_by_signature[signature]
                    )
                )
                atom_nnz = sum(
                    int(
                        matrix.indptr[row + 1]
                        - matrix.indptr[row]
                    )
                    for row in atom_rows
                )
                key = ("add_atom", group_index, atom_index)
                for row in atom_rows:
                    add_atom_key_by_row[row] = key
                if (
                    len(atom_rows) > row_cap
                    or atom_nnz <= 0
                    or atom_nnz > nnz_cap
                ):
                    oversized_nodes.add(key)
                    continue
                node_cache[key] = (
                    atom_rows,
                    signature[0],
                    atom_nnz,
                    (
                        "add_materialize_atom",
                        str(layer_id),
                        signature,
                        len(forward_by_signature[signature]),
                    ),
                )
            if any(row not in add_atom_key_by_row for row in all_rows):
                raise ValueError("ADD atomization did not cover its tag block")
            audited_add_groups.add(group_index)

        def materialize_node(
            key: Tuple[Any, ...],
        ) -> Optional[
            Tuple[
                Tuple[int, ...],
                Tuple[int, ...],
                int,
                Tuple[Any, ...],
            ]
        ]:
            kind = key[0]
            if deadline_reached():
                raise TimeoutError
            if kind == "ordinary":
                identity = key[1]
                row = int(identity)
                start = int(matrix.indptr[row])
                stop = int(matrix.indptr[row + 1])
                row_nnz = stop - start
                if row_nnz <= 0 or row_nnz > nnz_cap:
                    return None
                columns = tuple(
                    int(column) for column in matrix.indices[start:stop]
                )
                return (
                    (row,),
                    columns,
                    row_nnz,
                    (
                        "ordinary_row",
                        tag_texts[row],
                        row_signature(row),
                    ),
                )
            if kind != "add_atom":
                raise ValueError("unknown incidence-cone node kind")
            if key in oversized_nodes:
                return None
            node = node_cache.get(key)
            if node is None:
                raise ValueError("ADD atom was not audited before expansion")
            return node

        packet_nnz = sum(
            int(
                matrix.indptr[int(row) + 1]
                - matrix.indptr[int(row)]
            )
            for row in packets
        )
        if packet_nnz <= 0 or packet_nnz > nnz_cap:
            return empty
        initial_columns = {
            int(column)
            for row in packets
            for column in matrix.indices[
                matrix.indptr[int(row)]:matrix.indptr[int(row) + 1]
            ]
        }
        property_set = {int(column) for column in properties}
        if not initial_columns or initial_columns & property_set:
            return empty

        # Build only the row-incidence payload of a CSC.  Copying all float64
        # coefficients would consume about 112 MiB at the 9.3M-nnz production
        # shape before any search state existed.  The int8 pattern peaks near
        # 55 MiB including its temporary CSR payload.
        pattern_data = np.ones(int(matrix.nnz), dtype=np.int8)
        pattern_csr = sp.csr_matrix(
            (
                pattern_data,
                matrix.indices,
                matrix.indptr,
            ),
            shape=matrix.shape,
            copy=False,
        )
        pattern_csc = pattern_csr.tocsc(copy=True)
        column_indptr = pattern_csc.indptr
        column_rows = pattern_csc.indices
        del pattern_csc
        del pattern_csr
        del pattern_data
        if deadline_reached():
            return empty

        # Each column stores its canonical cheapest path.  Path keys refer to
        # frontier-materialized nodes, not to all rows in the full frame.
        best_by_column: dict[
            int,
            Tuple[Tuple[Any, ...], ...],
        ] = {
            column: () for column in initial_columns
        }
        metrics_cache: dict[
            Tuple[Tuple[Any, ...], ...],
            Tuple[int, int, int, Tuple[Any, ...]],
        ] = {(): (0, 0, 0, ())}

        def path_metrics(
            path: Tuple[Tuple[Any, ...], ...],
        ) -> Tuple[int, int, int, Tuple[Any, ...]]:
            cached = metrics_cache.get(path)
            if cached is not None:
                return cached
            row_count = sum(len(node_cache[key][0]) for key in path)
            selected_nnz = sum(node_cache[key][2] for key in path)
            result = (
                len(path),
                row_count,
                selected_nnz,
                tuple(node_cache[key][3] for key in path),
            )
            metrics_cache[path] = result
            return result

        frontier_columns = tuple(sorted(initial_columns))
        processed_nodes: set[Tuple[Any, ...]] = set()
        adjacent_rows_scanned = 0
        for _depth in range(1, depth_cap + 1):
            if deadline_reached():
                return empty
            snapshot = dict(best_by_column)
            candidate_keys: set[Tuple[Any, ...]] = set()
            for column in frontier_columns:
                start = int(column_indptr[column])
                stop = int(column_indptr[column + 1])
                for raw_row in column_rows[start:stop]:
                    adjacent_rows_scanned += 1
                    if (
                        adjacent_rows_scanned % 4096 == 0
                        and deadline_reached()
                    ):
                        return empty
                    row = int(raw_row)
                    if not bool(eligible[row]):
                        continue
                    group_index = int(add_group_by_row[row])
                    if group_index >= 0:
                        audit_add_group(group_index)
                        key = add_atom_key_by_row.get(row)
                        if key is None:
                            raise ValueError(
                                "frontier ADD row has no audited atom"
                            )
                    else:
                        key = ("ordinary", row)
                    if key not in processed_nodes:
                        candidate_keys.add(key)

            active_keys: list[Tuple[Any, ...]] = []
            for key in candidate_keys:
                processed_nodes.add(key)
                node = materialize_node(key)
                if node is None:
                    continue
                node_cache[key] = node
                active_keys.append(key)
            active_keys.sort(
                key=lambda key: (node_cache[key][3], key),
            )

            proposals: dict[
                int,
                Tuple[Tuple[Any, ...], ...],
            ] = {}
            for key_index, key in enumerate(active_keys):
                if key_index % 256 == 0 and deadline_reached():
                    return empty
                node = node_cache[key]
                base_paths = {
                    snapshot[column]
                    for column in node[1]
                    if column in snapshot
                }
                if not base_paths:
                    continue
                ordered_bases = sorted(
                    base_paths,
                    key=lambda path: (path_metrics(path), path),
                )
                candidate = None
                for base in ordered_bases:
                    possible = base + (key,)
                    possible_metrics = path_metrics(possible)
                    if (
                        possible_metrics[1] <= row_cap
                        and possible_metrics[2] <= nnz_cap
                    ):
                        candidate = possible
                        break
                if candidate is None:
                    continue
                for column in node[1]:
                    if column in snapshot:
                        continue
                    current = proposals.get(column)
                    if (
                        current is None
                        or (path_metrics(candidate), candidate)
                        < (path_metrics(current), current)
                    ):
                        proposals[column] = candidate

            if not proposals:
                break
            best_by_column.update(proposals)
            frontier_columns = tuple(sorted(proposals))

            goal_paths = [
                best_by_column[column]
                for column in property_set
                if column in best_by_column
            ]
            if not goal_paths:
                continue
            chosen = min(
                goal_paths,
                key=lambda path: (path_metrics(path), path),
            )
            selected = np.asarray(
                sorted(
                    {
                        row
                        for key in chosen
                        for row in node_cache[key][0]
                    }
                ),
                dtype=np.int64,
            )
            if (
                selected.size == 0
                or selected.size > row_cap
                or any(int(row) in packet_set for row in selected)
                or any(
                    tag_texts[int(row)].startswith("property_micro_rlt:")
                    for row in selected
                )
                or sum(
                    int(
                        matrix.indptr[int(row) + 1]
                        - matrix.indptr[int(row)]
                    )
                    for row in selected
                )
                > nnz_cap
                or deadline_reached()
            ):
                return empty
            return selected
        return empty
    except (
        TypeError,
        ValueError,
        IndexError,
        OverflowError,
        TimeoutError,
    ):
        return empty


def _expand_tag_blocks(
    blocks: Sequence[Mapping[str, Any]],
    *,
    sense: str,
    start: int,
) -> Tuple[ConstraintRowTag, ...]:
    result = []
    cursor = int(start)
    for item in blocks:
        if not {"tag", "rows"} <= set(item):
            raise ValueError(f"malformed operator-HZ {sense} tag block")
        count = int(item["rows"])
        if count < 0:
            raise ValueError(f"negative operator-HZ {sense} tag row count")
        tag = str(item["tag"])
        result.extend(
            ConstraintRowTag(
                global_row=cursor + local,
                sense=sense,
                block_tag=tag,
                block_local_row=local,
            )
            for local in range(count)
        )
        cursor += count
    return tuple(result)


def original_frame_from_operator_hz(
    hz: Any,
    metadata: Optional[Mapping[str, Any]] = None,
) -> OriginalFrameLP:
    """Extract the checker's row order from one exported ``SparseHZono``.

    The row order is exact and intentionally simple:

    ``[Ac|Ab]`` equality rows first, then ``[Auc|Aub]`` upper rows.

    ``constraint_tags_eq`` and ``constraint_tags_ub`` are expanded in their
    stored block order and audited against the actual matrix sizes.  This maps
    every candidate tensor column to one original operator-HZ row without
    relying on graph reconstruction.
    """

    n_cont = int(hz.Gc.shape[1])
    n_bin = int(hz.Gb.shape[1])
    eq_c = _canonical_csr(hz.Ac, shape=(int(hz.Ac.shape[0]), n_cont))
    eq_b = _canonical_csr(hz.Ab, shape=(int(hz.Ab.shape[0]), n_bin))
    ub_c = _canonical_csr(hz.Auc, shape=(int(hz.Auc.shape[0]), n_cont))
    ub_b = _canonical_csr(hz.Aub, shape=(int(hz.Aub.shape[0]), n_bin))
    if eq_c.shape[0] != eq_b.shape[0] or ub_c.shape[0] != ub_b.shape[0]:
        raise ValueError("operator-HZ continuous/binary row counts disagree")

    eq_A = sp.hstack((eq_c, eq_b), format="csr")
    ub_A = sp.hstack((ub_c, ub_b), format="csr")
    A = _canonical_csr(sp.vstack((eq_A, ub_A), format="csr"))
    eq_rhs = np.asarray(hz.b, dtype=np.float64).reshape(-1)
    ub_rhs = np.asarray(hz.ub, dtype=np.float64).reshape(-1)
    if eq_rhs.size != eq_A.shape[0] or ub_rhs.size != ub_A.shape[0]:
        raise ValueError("operator-HZ constraint RHS size mismatch")
    if not np.all(np.isfinite(eq_rhs)) or not np.all(np.isfinite(ub_rhs)):
        raise ValueError("operator-HZ constraint RHS is non-finite")

    meta = (
        metadata
        if metadata is not None
        else getattr(hz, "operator_hz_metadata", None)
    )
    if not isinstance(meta, Mapping):
        raise ValueError("operator-HZ row tags are required for audited mapping")
    eq_tags = _expand_tag_blocks(
        meta.get("constraint_tags_eq", ()),
        sense="eq",
        start=0,
    )
    ub_tags = _expand_tag_blocks(
        meta.get("constraint_tags_ub", ()),
        sense="ub",
        start=eq_A.shape[0],
    )
    if len(eq_tags) != eq_A.shape[0] or len(ub_tags) != ub_A.shape[0]:
        raise ValueError(
            "operator-HZ tag counts do not match exported constraint rows"
        )

    rl = np.concatenate(
        (
            eq_rhs,
            np.full(ub_rhs.size, -np.inf, dtype=np.float64),
        )
    )
    ru = np.concatenate((eq_rhs, ub_rhs))
    n_variables = n_cont + n_bin
    return OriginalFrameLP(
        A=A,
        rl=rl,
        ru=ru,
        lb=np.full(n_variables, -1.0, dtype=np.float64),
        ub=np.full(n_variables, 1.0, dtype=np.float64),
        row_tags=eq_tags + ub_tags,
    )


def output_frame_objectives(
    hz: Any,
    C: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map output rows to ``kappa + q @ v`` in the original HZ frame."""

    C64 = np.asarray(C, dtype=np.float64)
    if C64.ndim != 2 or C64.shape[1] != int(hz.c.size):
        raise ValueError("output objective matrix must have shape [R, n_out]")
    if not np.all(np.isfinite(C64)):
        raise ValueError("output objective matrix is non-finite")
    center = np.asarray(hz.c, dtype=np.float64).reshape(-1)
    G = sp.hstack(
        (
            _canonical_csr(hz.Gc),
            _canonical_csr(hz.Gb),
        ),
        format="csr",
    )
    # Sparse-left multiplication avoids NumPy's object-array behavior for
    # dense @ sparse.
    q = np.asarray(G.transpose() @ C64.transpose(), dtype=np.float64).transpose()
    kappa = C64 @ center
    if thresholds is not None:
        t = np.asarray(thresholds, dtype=np.float64).reshape(-1)
        if t.size != C64.shape[0] or not np.all(np.isfinite(t)):
            raise ValueError("objective thresholds must be one finite value per row")
        kappa = kappa - t
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(kappa)):
        raise ValueError("mapped frame objective is non-finite")
    return np.asarray(kappa, dtype=np.float64), np.asarray(q, dtype=np.float64)


def _validate_frame(frame: OriginalFrameLP, q: np.ndarray) -> np.ndarray:
    A = _canonical_csr(frame.A)
    rl = np.asarray(frame.rl, dtype=np.float64).reshape(-1)
    ru = np.asarray(frame.ru, dtype=np.float64).reshape(-1)
    lb = np.asarray(frame.lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(frame.ub, dtype=np.float64).reshape(-1)
    q64 = np.asarray(q, dtype=np.float64)
    if q64.ndim != 2 or q64.shape[1] != A.shape[1]:
        raise ValueError("q must have shape [R, original_frame_variables]")
    if (
        A.shape != (rl.size, lb.size)
        or ru.size != rl.size
        or ub.size != lb.size
        or len(frame.row_tags) != rl.size
    ):
        raise ValueError("original-frame LP shape mismatch")
    if (
        not np.all(np.isfinite(q64))
        or not np.all(np.isfinite(lb))
        or not np.all(np.isfinite(ub))
        or np.any(lb > ub)
        or np.any(np.isnan(rl))
        or np.any(np.isnan(ru))
        or np.any(rl > ru)
        or np.any(np.isposinf(rl))
        or np.any(np.isneginf(ru))
    ):
        raise ValueError("original-frame LP contains invalid bounds")
    return q64


def _frame_support_value(
    *,
    A: sp.csr_matrix,
    rl: np.ndarray,
    ru: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    q: np.ndarray,
    d: np.ndarray,
) -> float:
    """Evaluate the unguarded original-frame Lagrangian support."""

    finite_l = np.isfinite(rl)
    finite_u = np.isfinite(ru)
    upper_only = (~finite_l) & finite_u
    lower_only = finite_l & (~finite_u)
    free = (~finite_l) & (~finite_u)
    if (
        np.any(d[upper_only] < -1e-12)
        or np.any(d[lower_only] > 1e-12)
        or np.any(np.abs(d[free]) > 1e-12)
    ):
        return float("inf")
    row_side = np.where(d >= 0.0, ru, rl)
    row_terms = np.zeros_like(d)
    finite_side = np.isfinite(row_side)
    row_terms[finite_side] = d[finite_side] * row_side[finite_side]
    residual = np.asarray(q - A.transpose() @ d, dtype=np.float64).reshape(-1)
    box_side = np.where(residual >= 0.0, ub, lb)
    value = float(np.sum(row_terms) + np.dot(residual, box_side))
    return value if np.isfinite(value) else float("inf")


def _coordinate_line_candidate(
    *,
    indices: np.ndarray,
    coefficients: np.ndarray,
    residual: np.ndarray,
    d_value: float,
    row_lower: float,
    row_upper: float,
    box_lower: np.ndarray,
    box_upper: np.ndarray,
) -> Tuple[float, float]:
    """Return an exact piecewise-linear coordinate candidate and local gain.

    The minimizer is a weighted median of the residual breakpoints
    ``residual[j] / A[i,j]`` plus the row-support kink at ``d_i = 0``.
    Only the affected sparse row is inspected.  The returned point is still
    heuristic because all arithmetic here is binary64; final authority stays
    with the independent long-double checker.
    """

    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    a = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if (
        idx.size == 0
        or idx.size != a.size
        or not np.all(np.isfinite(a))
        or np.any(a == 0.0)
    ):
        return 0.0, 0.0
    local_r = np.asarray(residual[idx], dtype=np.float64)
    local_lb = np.asarray(box_lower[idx], dtype=np.float64)
    local_ub = np.asarray(box_upper[idx], dtype=np.float64)
    midpoint = 0.5 * (local_lb + local_ub)
    radius = 0.5 * (local_ub - local_lb)
    if (
        not np.all(np.isfinite(local_r))
        or not np.all(np.isfinite(midpoint))
        or not np.all(np.isfinite(radius))
        or np.any(radius < 0.0)
    ):
        return 0.0, 0.0

    finite_l = np.isfinite(row_lower)
    finite_u = np.isfinite(row_upper)
    if not finite_l and not finite_u:
        return 0.0, 0.0
    if finite_l and finite_u and row_lower > row_upper:
        return 0.0, 0.0

    lower_domain = -np.inf
    upper_domain = np.inf
    row_left_slope: float
    row_kink = None
    row_jump = 0.0
    if not finite_l:
        # Upper-only rows require d_i >= 0.
        lower_domain = -float(d_value)
        row_left_slope = float(row_upper)
    elif not finite_u:
        # Lower-only rows require d_i <= 0.
        upper_domain = -float(d_value)
        row_left_slope = float(row_lower)
    elif float(row_lower) == float(row_upper):
        row_left_slope = float(row_lower)
    else:
        row_left_slope = float(row_lower)
        row_kink = -float(d_value)
        row_jump = float(row_upper) - float(row_lower)

    breakpoints = local_r / a
    jumps = 2.0 * radius * np.abs(a)
    keep = np.isfinite(breakpoints) & np.isfinite(jumps) & (jumps > 0.0)
    points = breakpoints[keep].tolist()
    point_jumps = jumps[keep].tolist()
    if row_kink is not None and np.isfinite(row_kink) and row_jump > 0.0:
        points.append(float(row_kink))
        point_jumps.append(float(row_jump))
    if not points:
        return 0.0, 0.0

    order = np.argsort(np.asarray(points, dtype=np.float64), kind="mergesort")
    sorted_points = np.asarray(points, dtype=np.float64)[order]
    sorted_jumps = np.asarray(point_jumps, dtype=np.float64)[order]
    base_slope = (
        row_left_slope
        - float(np.dot(midpoint, a))
        - float(np.dot(radius, np.abs(a)))
    )
    if not np.isfinite(base_slope):
        return 0.0, 0.0

    # Aggregate identical binary64 breakpoints.  This makes the subgradient
    # crossing deterministic for duplicate generator coefficients.
    unique_points, first = np.unique(sorted_points, return_index=True)
    unique_jumps = np.add.reduceat(sorted_jumps, first)
    slope = float(base_slope)
    position = None

    if np.isfinite(lower_domain):
        before = unique_points < lower_domain
        if np.any(before):
            slope += float(np.sum(unique_jumps[before]))
        at = unique_points == lower_domain
        if np.any(at):
            slope += float(np.sum(unique_jumps[at]))
        if slope > 1e-12:
            position = float(lower_domain)
        scan = np.flatnonzero(unique_points > lower_domain)
    else:
        # A positive slope at -infinity would make the candidate dual
        # unbounded below.  A constructively feasible primal excludes that in
        # exact arithmetic; binary64 disagreement is discarded.
        if slope > 1e-10:
            return 0.0, 0.0
        scan = np.arange(unique_points.size, dtype=np.int64)

    if position is None:
        for raw_index in scan:
            point = float(unique_points[int(raw_index)])
            if point > upper_domain:
                break
            slope += float(unique_jumps[int(raw_index)])
            if slope > 1e-12:
                position = point
                break
    if position is None:
        if np.isfinite(upper_domain):
            position = float(upper_domain)
        else:
            return 0.0, 0.0
    position = min(max(position, lower_domain), upper_domain)
    if not np.isfinite(position) or abs(position) <= 1e-15:
        return 0.0, 0.0

    def row_support(value: float) -> float:
        if value >= 0.0:
            if not finite_u:
                return float("inf")
            return float(value * row_upper)
        if not finite_l:
            return float("inf")
        return float(value * row_lower)

    old_row = row_support(float(d_value))
    new_value = float(d_value + position)
    # Remove tiny domain drift introduced by the addition.
    if not finite_l and new_value < 0.0 and new_value > -1e-12:
        new_value = 0.0
        position = -float(d_value)
    if not finite_u and new_value > 0.0 and new_value < 1e-12:
        new_value = 0.0
        position = -float(d_value)
    new_row = row_support(new_value)
    old_box = np.where(local_r >= 0.0, local_ub, local_lb)
    new_r = local_r - a * position
    new_box = np.where(new_r >= 0.0, local_ub, local_lb)
    old_local = old_row + float(np.dot(local_r, old_box))
    new_local = new_row + float(np.dot(new_r, new_box))
    gain = old_local - new_local
    scale = 1.0 + abs(old_local) + abs(new_local)
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    if not np.isfinite(gain) or gain < -tolerance:
        return 0.0, 0.0
    return float(position), float(max(gain, 0.0))


def property_conditioned_coordinate_wavefront_duals(
    frame: OriginalFrameLP,
    q: np.ndarray,
    *,
    max_updates: int = 64,
    frontier_topk: int = 64,
    refresh_batch: int = 4,
    deadline: Optional[float] = None,
) -> CoordinateWavefrontCandidates:
    """Generate sparse duals by property-conditioned coordinate wavefronts.

    The current residual ``q - A.T @ d`` supplies a box-support subgradient.
    Rows with the largest KKT violation form a small frontier; each chosen row
    then receives the exact one-coordinate piecewise-linear minimizer.  A
    refresh propagates relevance from the property generators through adjacent
    constraint rows.  This avoids constructing or solving a generic LP and
    keeps the candidate dual sparse by design.
    """

    q64 = _validate_frame(frame, q)
    if int(max_updates) < 0:
        raise ValueError("max_updates must be nonnegative")
    if int(frontier_topk) <= 0 or int(refresh_batch) <= 0:
        raise ValueError("wavefront sizes must be positive")
    if deadline is not None and not math.isfinite(float(deadline)):
        raise ValueError("deadline must be finite")

    started = time.monotonic()
    if deadline is not None and started >= float(deadline):
        return CoordinateWavefrontCandidates(
            d=np.zeros(
                (q64.shape[0], frame.n_rows), dtype=np.float64
            ),
            initial_support=np.zeros(q64.shape[0], dtype=np.float64),
            candidate_support=np.zeros(q64.shape[0], dtype=np.float64),
            updates=0,
            selected_constraint_count=0,
            elapsed_seconds=float(time.monotonic() - started),
            deadline_reached=True,
        )
    A = _canonical_csr(frame.A)
    rl = np.asarray(frame.rl, dtype=np.float64).reshape(-1)
    ru = np.asarray(frame.ru, dtype=np.float64).reshape(-1)
    lb = np.asarray(frame.lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(frame.ub, dtype=np.float64).reshape(-1)
    initial = np.asarray(
        [
            _frame_support_value(
                A=A, rl=rl, ru=ru, lb=lb, ub=ub, q=row,
                d=np.zeros(A.shape[0], dtype=np.float64),
            )
            for row in q64
        ],
        dtype=np.float64,
    )
    result_d = np.zeros((q64.shape[0], A.shape[0]), dtype=np.float64)
    candidate = initial.copy()
    updates_total = 0
    selected_constraints: set[int] = set()
    deadline_reached = False

    finite_l = np.isfinite(rl)
    finite_u = np.isfinite(ru)
    upper_only = (~finite_l) & finite_u
    lower_only = finite_l & (~finite_u)
    bounded = finite_l & finite_u
    free = (~finite_l) & (~finite_u)
    midpoint = 0.5 * (lb + ub)
    radius = 0.5 * (ub - lb)

    for rival in range(q64.shape[0]):
        d = result_d[rival]
        residual = q64[rival].copy()
        updates = 0
        while updates < int(max_updates):
            if deadline is not None and time.monotonic() >= float(deadline):
                deadline_reached = True
                break
            sign = np.sign(residual)
            box_side = midpoint + radius * sign
            projected = np.asarray(A @ box_side, dtype=np.float64).reshape(-1)
            score = np.zeros(A.shape[0], dtype=np.float64)

            equal = bounded & (rl == ru)
            positive = bounded & (rl != ru) & (d > 1e-14)
            negative = bounded & (rl != ru) & (d < -1e-14)
            at_zero = bounded & (rl != ru) & ~(positive | negative)
            score[equal] = np.abs(rl[equal] - projected[equal])
            score[positive] = np.abs(ru[positive] - projected[positive])
            score[negative] = np.abs(rl[negative] - projected[negative])
            score[at_zero] = np.maximum(
                projected[at_zero] - ru[at_zero],
                rl[at_zero] - projected[at_zero],
            )

            upper_positive = upper_only & (d > 1e-14)
            upper_zero = upper_only & ~upper_positive
            score[upper_positive] = np.abs(
                ru[upper_positive] - projected[upper_positive]
            )
            score[upper_zero] = np.maximum(
                projected[upper_zero] - ru[upper_zero], 0.0
            )
            lower_negative = lower_only & (d < -1e-14)
            lower_zero = lower_only & ~lower_negative
            score[lower_negative] = np.abs(
                rl[lower_negative] - projected[lower_negative]
            )
            score[lower_zero] = np.maximum(
                rl[lower_zero] - projected[lower_zero], 0.0
            )
            score[free] = 0.0
            score[~np.isfinite(score)] = 0.0
            nonzero = np.flatnonzero(score > 1e-12)
            if nonzero.size == 0:
                break
            keep = min(int(frontier_topk), int(nonzero.size))
            if keep < nonzero.size:
                chosen = nonzero[
                    np.argpartition(score[nonzero], -keep)[-keep:]
                ]
            else:
                chosen = nonzero
            chosen = chosen[
                np.argsort(-score[chosen], kind="mergesort")
            ]

            refresh_updates = 0
            available = chosen.tolist()
            while (
                available
                and refresh_updates < int(refresh_batch)
                and updates < int(max_updates)
            ):
                if deadline is not None and time.monotonic() >= float(deadline):
                    deadline_reached = True
                    break
                best = None
                best_gain = -1.0
                best_delta = 0.0
                for raw_row in available:
                    row = int(raw_row)
                    start, end = A.indptr[row], A.indptr[row + 1]
                    delta, gain = _coordinate_line_candidate(
                        indices=A.indices[start:end],
                        coefficients=A.data[start:end],
                        residual=residual,
                        d_value=float(d[row]),
                        row_lower=float(rl[row]),
                        row_upper=float(ru[row]),
                        box_lower=lb,
                        box_upper=ub,
                    )
                    if (
                        abs(delta) > 1e-15
                        and (
                            gain > best_gain
                            or (
                                gain == best_gain
                                and best is not None
                                and row < best
                            )
                        )
                    ):
                        best = row
                        best_gain = float(gain)
                        best_delta = float(delta)
                if best is None:
                    break
                start, end = A.indptr[best], A.indptr[best + 1]
                residual[A.indices[start:end]] -= (
                    A.data[start:end] * best_delta
                )
                d[best] += best_delta
                selected_constraints.add(int(best))
                updates += 1
                updates_total += 1
                refresh_updates += 1
                available.remove(best)
            if refresh_updates == 0 or deadline_reached:
                break

        if deadline_reached:
            # The zero candidate is the fail-closed fallback.  Do not launch
            # another sparse support evaluation after the absolute deadline.
            d.fill(0.0)
            candidate[rival] = initial[rival]
            break
        candidate[rival] = _frame_support_value(
            A=A,
            rl=rl,
            ru=ru,
            lb=lb,
            ub=ub,
            q=q64[rival],
            d=d,
        )
        # Candidate generation must never lose the zero-dual fallback.
        if (
            not np.isfinite(candidate[rival])
            or candidate[rival] > initial[rival]
        ):
            d.fill(0.0)
            candidate[rival] = initial[rival]
        if deadline_reached:
            break

    return CoordinateWavefrontCandidates(
        d=result_d,
        initial_support=initial,
        candidate_support=candidate,
        updates=int(updates_total),
        selected_constraint_count=int(len(selected_constraints)),
        elapsed_seconds=float(time.monotonic() - started),
        deadline_reached=bool(deadline_reached),
    )


def _torch_sparse_coo(
    matrix: sp.csr_matrix,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coo = matrix.tocoo(copy=False)
    indices = torch.as_tensor(
        np.vstack((coo.row, coo.col)),
        dtype=torch.int64,
        device=device,
    )
    values = torch.as_tensor(coo.data, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=coo.shape,
        dtype=dtype,
        device=device,
    ).coalesce()


def batched_original_frame_row_duals(
    frame: OriginalFrameLP,
    q: np.ndarray,
    *,
    device: str | torch.device = "cuda",
    steps: int = 40,
    learning_rate: float = 0.08,
    candidate_rows: Optional[Sequence[int]] = None,
    deadline: Optional[float] = None,
) -> BatchedDualCandidates:
    """Generate batched untrusted row duals in the checker's exact row order.

    ``q[r]`` is the frame objective for rival ``r``.  The dense candidate
    tensor has shape ``[R, number_of_original_rows]`` and entry ``[r, k]``
    corresponds exactly to ``frame.row_tags[k]`` / ``frame.A[k]``.

    The optimization is candidate-only projected Adam.  Zero is retained as a
    fallback, so the reported unguarded support never exceeds the cube support.
    The independent checker remains mandatory even when
    ``candidate_support < 0``.
    """

    q64 = _validate_frame(frame, q)
    if int(steps) < 0:
        raise ValueError("steps must be nonnegative")
    if not np.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if deadline is not None and not math.isfinite(float(deadline)):
        raise ValueError("deadline must be a finite absolute monotonic time")
    started = time.monotonic()
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA dual candidates requested but CUDA is unavailable")
    dtype = torch.float64

    n_rows = frame.n_rows
    if candidate_rows is None:
        selected = np.arange(n_rows, dtype=np.int64)
    else:
        selected = np.asarray(candidate_rows, dtype=np.int64).reshape(-1)
        if selected.size and (
            int(selected.min()) < 0 or int(selected.max()) >= n_rows
        ):
            raise IndexError("candidate row index out of range")
        if np.unique(selected).size != selected.size:
            raise ValueError("candidate rows must be unique")
    A_selected = _canonical_csr(frame.A[selected, :])
    rl_selected = np.asarray(frame.rl, dtype=np.float64)[selected]
    ru_selected = np.asarray(frame.ru, dtype=np.float64)[selected]
    scale_np = np.maximum(np.max(np.abs(q64), axis=1), 1.0)
    wavefront = CoordinateWavefrontCandidates(
        d=np.zeros((q64.shape[0], selected.size), dtype=np.float64),
        initial_support=np.zeros(q64.shape[0], dtype=np.float64),
        candidate_support=np.zeros(q64.shape[0], dtype=np.float64),
        updates=0,
        selected_constraint_count=0,
        elapsed_seconds=0.0,
        deadline_reached=False,
    )
    if int(steps) > 0 and selected.size:
        remaining = (
            1.5
            if deadline is None
            else max(0.0, float(deadline) - time.monotonic())
        )
        wave_budget = min(1.5, 0.30 * remaining)
        if wave_budget > 1e-3:
            wave_frame = OriginalFrameLP(
                A=A_selected,
                rl=rl_selected,
                ru=ru_selected,
                lb=np.asarray(frame.lb, dtype=np.float64),
                ub=np.asarray(frame.ub, dtype=np.float64),
                row_tags=tuple(frame.row_tags[int(row)] for row in selected),
            )
            wavefront = property_conditioned_coordinate_wavefront_duals(
                wave_frame,
                q64 / scale_np[:, None],
                max_updates=max(8, min(128, 4 * int(steps))),
                frontier_topk=64,
                refresh_batch=4,
                deadline=time.monotonic() + wave_budget,
            )

    A_t = _torch_sparse_coo(
        A_selected.transpose().tocsr(),
        device=target,
        dtype=dtype,
    )
    q_t = torch.as_tensor(q64, dtype=dtype, device=target)
    lb_t = torch.as_tensor(frame.lb, dtype=dtype, device=target)
    ub_t = torch.as_tensor(frame.ub, dtype=dtype, device=target)
    rl_t = torch.as_tensor(rl_selected, dtype=dtype, device=target)
    ru_t = torch.as_tensor(ru_selected, dtype=dtype, device=target)

    finite_l = torch.isfinite(rl_t)
    finite_u = torch.isfinite(ru_t)
    upper_only = (~finite_l) & finite_u
    lower_only = finite_l & (~finite_u)
    bounded = finite_l & finite_u
    free = (~finite_l) & (~finite_u)

    # The support objective is positively homogeneous in (q, d).  Optimizing
    # normalized rivals gives Adam a common scale, then d is rescaled exactly
    # in real arithmetic for the candidate handoff.
    scale = q_t.abs().amax(dim=1).clamp(min=1.0)
    q_normalized = q_t / scale[:, None]
    d = torch.nn.Parameter(
        torch.as_tensor(
            wavefront.d,
            dtype=dtype,
            device=target,
        )
    )
    optimizer = torch.optim.Adam([d], lr=float(learning_rate))

    def project() -> None:
        with torch.no_grad():
            if bool(upper_only.any()):
                # Boolean indexing returns a copy in PyTorch; assign the
                # projected values back explicitly.
                d[:, upper_only] = d[:, upper_only].clamp(min=0.0)
            if bool(lower_only.any()):
                d[:, lower_only] = d[:, lower_only].clamp(max=0.0)
            if bool(free.any()):
                d[:, free] = 0.0

    def exact_support(candidate: torch.Tensor) -> torch.Tensor:
        row_support = torch.zeros(
            candidate.shape[0], dtype=dtype, device=target
        )
        if bool(upper_only.any()):
            row_support = row_support + (
                candidate[:, upper_only] * ru_t[upper_only]
            ).sum(dim=1)
        if bool(lower_only.any()):
            row_support = row_support + (
                candidate[:, lower_only] * rl_t[lower_only]
            ).sum(dim=1)
        if bool(bounded.any()):
            local = candidate[:, bounded]
            side = torch.where(
                local >= 0.0,
                ru_t[bounded].expand_as(local),
                rl_t[bounded].expand_as(local),
            )
            row_support = row_support + (local * side).sum(dim=1)
        residual = q_normalized - torch.sparse.mm(
            A_t, candidate.transpose(0, 1)
        ).transpose(0, 1)
        box_side = torch.where(
            residual >= 0.0,
            ub_t.expand_as(residual),
            lb_t.expand_as(residual),
        )
        return row_support + (residual * box_side).sum(dim=1)

    def smooth_support(
        candidate: torch.Tensor,
        *,
        smooth_abs: float,
    ) -> torch.Tensor:
        """Differentiable candidate objective with symmetric zero gradient.

        Exact box/row support is piecewise linear.  At the many coordinates
        where ``q-A.T@d == 0``, ``torch.where(residual >= 0, ...)`` chooses a
        one-sided subgradient which can make projected Adam bounce away from
        every useful multiplier and then fall back to zero.  A smooth absolute
        value supplies the central subgradient at zero.  This function has no
        proof authority: every iterate is still selected by ``exact_support``
        below and independently rechecked after the CUDA handoff.
        """

        tau = torch.as_tensor(
            max(float(smooth_abs), 1e-12),
            dtype=dtype,
            device=target,
        )
        row_support = torch.zeros(
            candidate.shape[0], dtype=dtype, device=target
        )
        if bool(upper_only.any()):
            row_support = row_support + (
                candidate[:, upper_only] * ru_t[upper_only]
            ).sum(dim=1)
        if bool(lower_only.any()):
            row_support = row_support + (
                candidate[:, lower_only] * rl_t[lower_only]
            ).sum(dim=1)
        if bool(bounded.any()):
            local = candidate[:, bounded]
            midpoint = 0.5 * (
                ru_t[bounded] + rl_t[bounded]
            )
            radius = 0.5 * (
                ru_t[bounded] - rl_t[bounded]
            )
            row_support = row_support + (
                midpoint.expand_as(local) * local
                + radius.expand_as(local)
                * torch.sqrt(local.square() + tau.square())
            ).sum(dim=1)
        residual = q_normalized - torch.sparse.mm(
            A_t, candidate.transpose(0, 1)
        ).transpose(0, 1)
        midpoint = 0.5 * (ub_t + lb_t)
        radius = 0.5 * (ub_t - lb_t)
        box_support = (
            midpoint.expand_as(residual) * residual
            + radius.expand_as(residual)
            * torch.sqrt(residual.square() + tau.square())
        ).sum(dim=1)
        return row_support + box_support

    project()
    with torch.no_grad():
        zero = torch.zeros_like(d)
        initial = exact_support(zero).detach()
        warm_value = exact_support(d).detach()
        use_warm = warm_value < initial
        best_value = torch.where(use_warm, warm_value, initial)
        best_d = d.detach().clone()
        best_d[~use_warm] = 0.0
    steps_completed = 0
    deadline_reached = False
    smooth_start = 0.05
    smooth_end = smooth_start
    for step_index in range(int(steps)):
        # A shared deadline is observed between optimizer iterations.  A
        # single already-launched CUDA sparse kernel cannot be preempted here.
        if deadline is not None and time.monotonic() >= float(deadline):
            deadline_reached = True
            break
        # Verification normally runs under an outer ``torch.no_grad()``.
        # Candidate optimization is the one deliberately differentiable
        # island; without this explicit override, production would fail
        # before the first step while standalone toys happened to work.
        with torch.enable_grad():
            optimizer.zero_grad(set_to_none=True)
            progress = (
                float(step_index) / float(max(1, int(steps) - 1))
            )
            # Geometric continuation approaches the exact kink while keeping
            # the first iterations well-conditioned in normalized q units.
            smooth_abs = smooth_start * (
                (1.0e-3 / smooth_start) ** progress
            )
            smooth_end = float(smooth_abs)
            values = smooth_support(d, smooth_abs=smooth_abs)
            # Keep an autograd edge even when ``candidate_rows`` is empty.
            (values.sum() + 0.0 * d.sum()).backward()
        optimizer.step()
        project()
        steps_completed += 1
        with torch.no_grad():
            current = exact_support(d)
            improved = current < best_value
            if bool(improved.any()):
                best_value = torch.where(improved, current, best_value)
                best_d[improved] = d.detach()[improved]

    d_full = torch.zeros(
        (q_t.shape[0], n_rows), dtype=dtype, device=target
    )
    if selected.size:
        d_full[:, torch.as_tensor(selected, device=target)] = (
            best_d * scale[:, None]
        )
    # Checker convention: it immediately computes d = -raw_row_dual.
    row_dual = (-d_full).detach().cpu().numpy()
    initial_np = (initial * scale).detach().cpu().numpy()
    candidate_np = (best_value * scale).detach().cpu().numpy()
    if (
        not np.all(np.isfinite(row_dual))
        or not np.all(np.isfinite(initial_np))
        or not np.all(np.isfinite(candidate_np))
    ):
        raise RuntimeError("GPU dual candidate optimization produced non-finite data")
    return BatchedDualCandidates(
        row_dual=row_dual,
        initial_support=initial_np,
        candidate_support=candidate_np,
        selected_rows=selected.copy(),
        device=str(target),
        dtype=str(dtype),
        steps_requested=int(steps),
        steps_completed=int(steps_completed),
        elapsed_seconds=float(time.monotonic() - started),
        deadline_reached=bool(deadline_reached),
        optimization_method=(
            "property_conditioned_l1_coordinate_wavefront+"
            "projected_adam_smooth_absolute_continuation_v3"
        ),
        smooth_abs_start=float(smooth_start),
        smooth_abs_end=float(smooth_end),
        wavefront_updates=int(wavefront.updates),
        wavefront_support_improved_rows=int(np.count_nonzero(
            wavefront.candidate_support < wavefront.initial_support
        )),
        wavefront_best_improvement=float(np.max(
            wavefront.initial_support - wavefront.candidate_support
        ) if wavefront.initial_support.size else 0.0),
        wavefront_elapsed_seconds=float(wavefront.elapsed_seconds),
        wavefront_selected_constraint_count=int(
            wavefront.selected_constraint_count
        ),
    )


def rank_relu_property_sensitivities(
    nu_per_layer: Mapping[int, torch.Tensor],
    bounds_by_layer: Mapping[int, Any],
    *,
    M: int,
    top_k: int,
) -> Tuple[ReluSensitivity, ...]:
    """Rank ReLU preactivations using batched DualSolver ``nu`` heuristically.

    DualSolver packs rivals sample-major: tensor row ``b*M + j`` is rival
    ``j`` for sample ``b``.  The operator-HZ large-class path has one input
    sample, so this helper deliberately requires ``B == 1``.  For an unstable
    interval ``l < 0 < u``, the score is

    ``max_j |nu[j, i]| * (-l_i*u_i/(u_i-l_i))``.

    The second factor is the ReLU triangle's maximum vertical gap.  This is
    only a scheduling score for certified preactivation LPs or exact-neuron
    selection; it must never authorize a bound or verdict.
    """

    if int(M) <= 0:
        raise ValueError("M must be positive")
    if int(top_k) < 0:
        raise ValueError("top_k must be nonnegative")
    ranked = []
    for layer_id in sorted(nu_per_layer):
        if layer_id not in bounds_by_layer:
            raise ValueError(f"missing bounds for ReLU layer {layer_id}")
        nu = nu_per_layer[layer_id].detach()
        if nu.dim() < 2 or nu.shape[0] != int(M):
            raise ValueError(
                f"layer {layer_id}: expected one-sample nu [M, ...], "
                f"got {tuple(nu.shape)} for M={M}"
            )
        bounds = bounds_by_layer[layer_id]
        lower = bounds.lb.detach().flatten()
        upper = bounds.ub.detach().flatten()
        nu_flat = nu.flatten(start_dim=1)
        if lower.numel() != nu_flat.shape[1] or upper.numel() != lower.numel():
            raise ValueError(f"layer {layer_id}: nu/bounds neuron count mismatch")
        lower = lower.to(device=nu.device, dtype=nu.dtype)
        upper = upper.to(device=nu.device, dtype=nu.dtype)
        unstable = (lower < 0.0) & (upper > 0.0)
        gap = torch.zeros_like(lower)
        gap[unstable] = (
            -lower[unstable] * upper[unstable]
            / (upper[unstable] - lower[unstable])
        )
        per_rival = nu_flat.abs() * gap.unsqueeze(0)
        score, rival = per_rival.max(dim=0)
        for neuron in torch.where(unstable)[0].tolist():
            ranked.append(
                ReluSensitivity(
                    layer_id=int(layer_id),
                    neuron=int(neuron),
                    score=float(score[neuron].item()),
                    rival=int(rival[neuron].item()),
                )
            )
    ranked.sort(
        key=lambda item: (
            -item.score,
            item.layer_id,
            item.neuron,
            item.rival,
        )
    )
    return tuple(ranked[: int(top_k)])


__all__ = [
    "BatchedDualCandidates",
    "CoordinateWavefrontCandidates",
    "ConstraintRowTag",
    "OriginalFrameLP",
    "ReluSensitivity",
    "batched_original_frame_row_duals",
    "original_frame_from_operator_hz",
    "output_frame_objectives",
    "property_conditioned_incidence_cone_rows",
    "property_conditioned_coordinate_wavefront_duals",
    "rank_relu_property_sensitivities",
]
