#!/usr/bin/env python3
# ===- query_dual_candidates.py - frozen-alpha dual queries --------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===----------------------------------------------------------------===#
"""Candidate-only DualSolver queries with frozen alpha/query descriptors.

This module has deliberately no proof authority.  It proposes tighter bounds
for two query families:

* unstable neurons of one target ReLU, queried in blocks as ``[+e_i, -e_i]``
  against the ReLU's unique preactivation predecessor;
* optional final property rows.  The production default queries only ``-C_j``
  because the strict path needs ``UB(C_j y) = -LB(-C_j y)``.  A diagnostic
  bidirectional mode can still query ``[C_j, -C_j]``.

The legacy V1 API freezes each optimized alpha state and performs a second
``optimize=False`` GPU replay whose margins become non-authoritative candidate
bounds.  V1 remains the default for compatibility.

Explicit ``descriptor_only=True`` selects V2.  V2 calls the optimizer exactly
once per block, freezes and hashes only the alpha/objective descriptor, never
exports optimizer margins, and leaves all candidate bounds at their baseline.
Supplying ``selected_target_rows`` alongside descriptor-only mode selects the
independent V3 protocol.  V3 constructs ``[+e_i, -e_i]`` only for those rows
at the source; it never optimizes all unstable rows and slices afterward.
Final property rows remain fully covered.  Only the separate CPU independent
replayer may derive a bound or improvement from a V2/V3 descriptor.

All solver calls receive private clones of the caller's frozen bounds.  A
shared absolute monotonic deadline is checked before and after every
optimization, alpha transfer, and frozen replay.  Errors and deadline
crossings discard the entire batch and return the original target/property
bounds as an explicit fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from act.back_end.core import Bounds


_V2_NON_AUTHORITATIVE_AUDIT_FIELDS = (
    "lr_alpha",
    "lr_decay",
    "solver",
    "elapsed_seconds",
    "timings",
)


class _QueryDeadline(RuntimeError):
    """Internal signal used to discard an incomplete query batch."""


@dataclass(frozen=True)
class QueryDescriptor:
    """One independently replayable objective block."""

    block_id: int
    query_kind: str
    start_lid: Optional[int]
    target_relu_lid: Optional[int]
    row_ids: Tuple[int, ...]
    objective_order: str
    objectives: np.ndarray
    objective_sha256: str
    M: int
    alpha_tree_index: int
    alpha_sha256: str


@dataclass(frozen=True)
class QueryDualCandidates:
    """Candidate bounds and the frozen alpha/query replay material."""

    target_bounds: Bounds
    property_lower: np.ndarray
    property_upper: np.ndarray
    query_descriptors: Tuple[QueryDescriptor, ...]
    alpha_trees: Tuple[Mapping[int, Any], ...]
    improved_target_indices: np.ndarray
    improved_property_indices: np.ndarray
    timings: Tuple[Mapping[str, Any], ...]
    receipt: Dict[str, Any]
    proof_authority: bool = False

    @property
    def status(self) -> str:
        return str(self.receipt.get("status", "unknown"))


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: Any, *, dtype: Any = np.float64) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _ordered_rows_sha256(rows: Sequence[int]) -> str:
    return _canonical_sha256([int(value) for value in rows])


def _indexed_bounds_sha256(
    lower: np.ndarray,
    upper: np.ndarray,
    rows: Sequence[int],
) -> str:
    index = np.asarray(tuple(int(value) for value in rows), dtype=np.int64)
    return _array_sha256(
        np.stack([lower[index], upper[index]])
        if index.size
        else np.zeros((2, 0), dtype=np.float64)
    )


def _normalize_selected_target_rows(value: Any) -> Tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError("selected_target_rows must be an integer sequence")
    try:
        raw_rows = list(value)
    except TypeError as exc:
        raise ValueError(
            "selected_target_rows must be an integer sequence"
        ) from exc
    rows = []
    for raw in raw_rows:
        if (
            isinstance(raw, (bool, np.bool_))
            or not isinstance(raw, (Integral, np.integer))
            or int(raw) < 0
        ):
            raise ValueError(
                "selected_target_rows must contain nonnegative integers"
            )
        rows.append(int(raw))
    if len(set(rows)) != len(rows):
        raise ValueError("selected_target_rows must not contain duplicates")
    return tuple(rows)


def _tensor_cpu_f64(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    stored = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(stored).all()):
        raise ValueError(f"{name} contains non-finite values")
    return stored.clone()


def _bounds_sha256(bounds_dict: Mapping[int, Bounds]) -> str:
    digest = hashlib.sha256()
    keys = sorted(int(key) for key in bounds_dict)
    digest.update(np.asarray([len(keys)], dtype=np.int64).tobytes())
    for lid in keys:
        bounds = bounds_dict[lid]
        if not isinstance(bounds, Bounds):
            raise TypeError(f"bounds_dict[{lid}] must be Bounds")
        lb = _tensor_cpu_f64(bounds.lb, name=f"bounds[{lid}].lb")
        ub = _tensor_cpu_f64(bounds.ub, name=f"bounds[{lid}].ub")
        if lb.shape != ub.shape or lb.dim() < 2:
            raise ValueError(
                f"bounds_dict[{lid}] must have matching batched bounds"
            )
        if bool((lb > ub).any()):
            raise ValueError(f"bounds_dict[{lid}] has lb > ub")
        digest.update(np.asarray([lid], dtype=np.int64).tobytes())
        digest.update(np.asarray(lb.shape, dtype=np.int64).tobytes())
        digest.update(lb.numpy().tobytes())
        digest.update(ub.numpy().tobytes())
    return digest.hexdigest()


def _clone_bounds_dict(
    bounds_dict: Mapping[int, Bounds],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[int, Bounds]:
    result: Dict[int, Bounds] = {}
    for raw_lid, bounds in bounds_dict.items():
        lid = int(raw_lid)
        if not isinstance(bounds, Bounds):
            raise TypeError(f"bounds_dict[{lid}] must be Bounds")
        lb = bounds.lb.detach().clone().to(device=device, dtype=dtype)
        ub = bounds.ub.detach().clone().to(device=device, dtype=dtype)
        if (
            lb.shape != ub.shape
            or lb.dim() < 2
            or int(lb.shape[0]) != 1
            or not bool(torch.isfinite(lb).all())
            or not bool(torch.isfinite(ub).all())
            or bool((lb > ub).any())
        ):
            raise ValueError(
                f"bounds_dict[{lid}] must be one finite batched box"
            )
        result[lid] = Bounds(lb=lb, ub=ub)
    if not result:
        raise ValueError("bounds_dict must not be empty")
    return result


def _alpha_to_cpu_f64(tree: Any, *, path: str = "alpha") -> Any:
    if tree is None:
        return None
    if isinstance(tree, torch.Tensor):
        stored = _tensor_cpu_f64(tree, name=path)
        if bool((stored < 0.0).any()) or bool((stored > 1.0).any()):
            raise ValueError(f"{path} lies outside [0, 1]")
        return stored
    if isinstance(tree, Mapping):
        return {
            key: _alpha_to_cpu_f64(value, path=f"{path}.{key}")
            for key, value in tree.items()
        }
    if isinstance(tree, list):
        return [
            _alpha_to_cpu_f64(value, path=f"{path}[{index}]")
            for index, value in enumerate(tree)
        ]
    if isinstance(tree, tuple):
        return tuple(
            _alpha_to_cpu_f64(value, path=f"{path}[{index}]")
            for index, value in enumerate(tree)
        )
    raise TypeError(f"{path}: unsupported alpha node {type(tree)!r}")


def _alpha_to_device(
    tree: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    if tree is None:
        return None
    if isinstance(tree, torch.Tensor):
        return tree.detach().clone().to(device=device, dtype=dtype)
    if isinstance(tree, Mapping):
        return {
            key: _alpha_to_device(value, device=device, dtype=dtype)
            for key, value in tree.items()
        }
    if isinstance(tree, list):
        return [
            _alpha_to_device(value, device=device, dtype=dtype)
            for value in tree
        ]
    if isinstance(tree, tuple):
        return tuple(
            _alpha_to_device(value, device=device, dtype=dtype)
            for value in tree
        )
    raise TypeError(f"unsupported stored alpha node {type(tree)!r}")


def _alpha_sha256(tree: Any) -> str:
    digest = hashlib.sha256()

    def update(node: Any) -> None:
        if node is None:
            digest.update(b"N")
        elif isinstance(node, torch.Tensor):
            value = _tensor_cpu_f64(node, name="stored alpha")
            digest.update(b"T")
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.numpy().tobytes())
        elif isinstance(node, Mapping):
            digest.update(b"D")
            keys = sorted(node, key=lambda key: (type(key).__name__, repr(key)))
            digest.update(np.asarray([len(keys)], dtype=np.int64).tobytes())
            for key in keys:
                encoded = json.dumps(
                    [type(key).__name__, repr(key)],
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("ascii")
                digest.update(np.asarray([len(encoded)], dtype=np.int64).tobytes())
                digest.update(encoded)
                update(node[key])
        elif isinstance(node, list):
            digest.update(b"L")
            digest.update(np.asarray([len(node)], dtype=np.int64).tobytes())
            for value in node:
                update(value)
        elif isinstance(node, tuple):
            digest.update(b"Q")
            digest.update(np.asarray([len(node)], dtype=np.int64).tobytes())
            for value in node:
                update(value)
        else:
            raise TypeError(f"unsupported stored alpha node {type(node)!r}")

    update(tree)
    return digest.hexdigest()


def _receipt_with_sha256(payload: Mapping[str, Any]) -> Dict[str, Any]:
    receipt = dict(payload)
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def verify_query_dual_candidates_receipt(
    receipt: Mapping[str, Any],
) -> bool:
    try:
        expected = receipt["receipt_sha256"]
        if not isinstance(expected, str) or len(expected) != 64:
            return False
        payload = dict(receipt)
        del payload["receipt_sha256"]
        return _canonical_sha256(payload) == expected
    except (KeyError, TypeError, ValueError):
        return False


def query_dual_stored_alpha_sha256(tree: Any) -> str:
    """Hash the exact CPU-binary64 alpha tree consumed by a descriptor.

    The helper deliberately does not grant proof authority.  It is public so
    the independent orchestration layer can bind an in-memory candidate tree
    to its descriptor before passing that tree to the CPU proof replayer.
    """

    return _alpha_sha256(tree)


def validate_query_dual_candidates(
    result: QueryDualCandidates,
) -> bool:
    """Validate the complete in-memory candidate object against its receipt.

    ``verify_query_dual_candidates_receipt`` checks only JSON integrity.  This
    stronger validator additionally binds every descriptor objective and
    alpha tree, both returned bound arrays, improvement-index metadata, and
    fallback/generated status semantics.  It still does *not* make a Dual
    candidate authoritative; only independent replay may do that.
    """

    try:
        if not isinstance(result, QueryDualCandidates):
            return False
        if result.proof_authority is not False:
            return False
        receipt = result.receipt
        schema = receipt.get("schema") if isinstance(receipt, Mapping) else None
        v1 = schema == "act.query_dual_candidates.v1"
        v2 = schema == "act.query_dual_candidates.v2"
        v3 = schema == "act.query_dual_candidates.v3"
        descriptor_schema = v2 or v3
        if (
            not isinstance(receipt, Mapping)
            or not verify_query_dual_candidates_receipt(receipt)
            or not (v1 or v2 or v3)
            or receipt.get("candidate_only") is not True
            or receipt.get("proof_authority") is not False
            or receipt.get("optimizer_best_margins_used_as_bounds") is not False
        ):
            return False
        if descriptor_schema and (
            receipt.get("protocol")
            != (
                "property_sparse_descriptor_only_v3"
                if v3
                else "descriptor_only_v2"
            )
            or receipt.get("non_authoritative_audit_fields")
            != list(_V2_NON_AUTHORITATIVE_AUDIT_FIELDS)
            or receipt.get("candidate_bound_source")
            != "none_descriptor_only"
            or receipt.get("optimizer_margins_exported") is not False
            or receipt.get("optimizer_margins_used_for_improvement") is not False
            or receipt.get("gpu_frozen_alpha_replay") is not False
            or receipt.get("cpu_independent_replay_required") is not True
            or receipt.get("all_candidate_updates_replayed_with_stored_alpha")
            is not False
            or receipt.get("all_bounds_replayed_with_stored_alpha") is not False
            or receipt.get("property_lower_dual_replayed") is not False
        ):
            return False
        if v3 and (
            receipt.get("selector_authoritative") is not False
            or receipt.get("selected_rows_source")
            != "caller_supplied_non_authoritative_selector"
            or receipt.get("property_coverage_policy")
            != "all_property_rows"
            or receipt.get("unselected_policy")
            != "bit_identical_immutable_parent_target_bounds"
        ):
            return False

        target_lb = _tensor_cpu_f64(
            result.target_bounds.lb, name="result target lower"
        ).numpy().reshape(-1)
        target_ub = _tensor_cpu_f64(
            result.target_bounds.ub, name="result target upper"
        ).numpy().reshape(-1)
        property_lower = np.ascontiguousarray(
            np.asarray(result.property_lower, dtype=np.float64).reshape(-1)
        )
        property_upper = np.ascontiguousarray(
            np.asarray(result.property_upper, dtype=np.float64).reshape(-1)
        )
        if (
            target_lb.shape != target_ub.shape
            or property_lower.shape != property_upper.shape
            or not np.all(np.isfinite(target_lb))
            or not np.all(np.isfinite(target_ub))
            or not np.all(np.isfinite(property_lower))
            or not np.all(np.isfinite(property_upper))
            or np.any(target_lb > target_ub)
            or np.any(property_lower > property_upper)
            or receipt.get("candidate_target_bounds_sha256")
            != _array_sha256(np.stack([target_lb, target_ub]))
            or receipt.get("candidate_property_bounds_sha256")
            != _array_sha256(
                np.stack([property_lower, property_upper])
                if property_lower.size
                else np.zeros((2, 0), dtype=np.float64)
            )
        ):
            return False
        if descriptor_schema and receipt.get("candidate_generated") is True and (
            receipt.get("candidate_target_bounds_sha256")
            != receipt.get("target_bounds_sha256")
            or receipt.get("candidate_property_bounds_sha256")
            != receipt.get("property_baseline_sha256")
        ):
            return False
        if v3:
            sparse_bound = receipt.get("sparse_selection_bound") is True
            if not sparse_bound:
                if (
                    receipt.get("status")
                    != "error_fallback_frozen_bounds"
                    or receipt.get("candidate_generated") is not False
                ):
                    return False
            else:
                def receipt_rows(name: str) -> Tuple[int, ...]:
                    values = receipt.get(name)
                    if not isinstance(values, list):
                        raise ValueError(f"{name} must be a list")
                    normalized = []
                    for value in values:
                        if (
                            isinstance(value, (bool, np.bool_))
                            or not isinstance(
                                value, (Integral, np.integer)
                            )
                            or int(value) < 0
                        ):
                            raise ValueError(f"{name} is malformed")
                        normalized.append(int(value))
                    if len(set(normalized)) != len(normalized):
                        raise ValueError(f"{name} contains duplicates")
                    return tuple(normalized)

                eligible_rows = receipt_rows(
                    "eligible_target_row_ids"
                )
                selected_rows = receipt_rows(
                    "selected_target_row_ids"
                )
                omitted_rows = receipt_rows(
                    "omitted_target_row_ids"
                )
                actual_eligible = tuple(
                    int(value)
                    for value in np.flatnonzero(
                        (target_lb < 0.0) & (target_ub > 0.0)
                    )
                )
                selected_set = set(selected_rows)
                expected_omitted = tuple(
                    row
                    for row in actual_eligible
                    if row not in selected_set
                )
                if (
                    eligible_rows != actual_eligible
                    or not selected_set.issubset(actual_eligible)
                    or omitted_rows != expected_omitted
                    or receipt.get("eligible_target_count")
                    != len(eligible_rows)
                    or receipt.get("selected_target_count")
                    != len(selected_rows)
                    or receipt.get("omitted_target_count")
                    != len(omitted_rows)
                    or receipt.get("eligible_target_rows_sha256")
                    != _ordered_rows_sha256(eligible_rows)
                    or receipt.get("selected_target_rows_sha256")
                    != _ordered_rows_sha256(selected_rows)
                    or receipt.get("omitted_target_rows_sha256")
                    != _ordered_rows_sha256(omitted_rows)
                    or receipt.get("target_partition_complete") is not True
                    or receipt.get("target_partition_disjoint") is not True
                    or not set(selected_rows).isdisjoint(omitted_rows)
                    or set(eligible_rows)
                    != set(selected_rows).union(omitted_rows)
                    or receipt.get(
                        "selected_parent_target_bounds_sha256"
                    )
                    != _indexed_bounds_sha256(
                        target_lb, target_ub, selected_rows
                    )
                    or receipt.get(
                        "unselected_parent_target_bounds_sha256"
                    )
                    != _indexed_bounds_sha256(
                        target_lb, target_ub, omitted_rows
                    )
                    or receipt.get(
                        "unselected_candidate_target_bounds_sha256"
                    )
                    != _indexed_bounds_sha256(
                        target_lb, target_ub, omitted_rows
                    )
                    or receipt.get(
                        "unselected_bounds_bit_identical_parent"
                    )
                    is not True
                ):
                    return False
                property_rows = receipt_rows(
                    "eligible_property_row_ids"
                )
                selected_property_rows = receipt_rows(
                    "selected_property_row_ids"
                )
                expected_property_rows = tuple(
                    range(int(property_lower.size))
                )
                if (
                    property_rows != expected_property_rows
                    or selected_property_rows != expected_property_rows
                    or receipt.get("property_rows")
                    != len(expected_property_rows)
                    or receipt.get("selected_property_count")
                    != len(expected_property_rows)
                    or receipt.get("eligible_property_rows_sha256")
                    != _ordered_rows_sha256(expected_property_rows)
                    or receipt.get("selected_property_rows_sha256")
                    != _ordered_rows_sha256(expected_property_rows)
                ):
                    return False

        improved_target = np.asarray(
            result.improved_target_indices, dtype=np.int64
        ).reshape(-1)
        improved_property = np.asarray(
            result.improved_property_indices, dtype=np.int64
        ).reshape(-1)
        if (
            np.unique(improved_target).size != improved_target.size
            or np.unique(improved_property).size != improved_property.size
            or np.any(improved_target < 0)
            or np.any(improved_target >= target_lb.size)
            or np.any(improved_property < 0)
            or np.any(improved_property >= property_lower.size)
            or list(map(int, improved_target))
            != list(receipt.get("improved_target_indices", []))
            or list(map(int, improved_property))
            != list(receipt.get("improved_property_indices", []))
            or int(receipt.get("strict_target_improvements", -1))
            != int(improved_target.size)
            or int(receipt.get("strict_property_improvements", -1))
            != int(improved_property.size)
            or (
                descriptor_schema
                and (improved_target.size or improved_property.size)
            )
        ):
            return False

        descriptors = tuple(result.query_descriptors)
        alpha_trees = tuple(result.alpha_trees)
        records = receipt.get("descriptor_records", [])
        alpha_hashes = receipt.get("alpha_hashes", [])
        status = str(receipt.get("status", ""))
        generated = status == (
            "descriptors_generated"
            if descriptor_schema
            else "generated"
        )
        expected_descriptor_layout = []
        if descriptor_schema and generated:
            chunk = receipt.get("block_size")
            property_count = receipt.get("property_rows")
            property_width = receipt.get("property_width")
            if (
                isinstance(chunk, (bool, np.bool_))
                or not isinstance(chunk, (Integral, np.integer))
                or int(chunk) <= 0
                or isinstance(property_count, (bool, np.bool_))
                or not isinstance(property_count, (Integral, np.integer))
                or int(property_count) < 0
                or isinstance(property_width, (bool, np.bool_))
                or not isinstance(property_width, (Integral, np.integer))
                or int(property_width) < 0
            ):
                return False
            chunk = int(chunk)
            property_count = int(property_count)
            property_width = int(property_width)
            if (property_count == 0) != (property_width == 0):
                return False
            target_lid = receipt.get("target_relu_lid")
            if target_lid is not None:
                unstable = tuple(
                    int(value)
                    for value in np.flatnonzero(
                        (target_lb < 0.0) & (target_ub > 0.0)
                    )
                )
                if receipt.get("unstable_target_neurons") != len(unstable):
                    return False
                target_rows = (
                    tuple(
                        int(value)
                        for value in receipt.get(
                            "selected_target_row_ids", []
                        )
                    )
                    if v3
                    else unstable
                )
                for offset in range(0, len(target_rows), chunk):
                    block_rows = target_rows[offset : offset + chunk]
                    expected_objective = np.zeros(
                        (len(block_rows), target_lb.size),
                        dtype=np.float64,
                    )
                    expected_objective[
                        np.arange(len(block_rows)),
                        np.asarray(block_rows, dtype=np.int64),
                    ] = 1.0
                    expected_descriptor_layout.append(
                        (
                            "relu_unstable_plus_minus_one_hot",
                            receipt.get("target_start_lid"),
                            target_lid,
                            block_rows,
                            "positive_rows_then_negated_rows",
                            np.vstack(
                                [expected_objective, -expected_objective]
                            ),
                        )
                    )
            elif target_lb.size != 0:
                return False
            if property_count:
                upper_only = receipt.get("property_upper_only") is True
                for offset in range(0, property_count, chunk):
                    block_rows = tuple(
                        range(offset, min(offset + chunk, property_count))
                    )
                    expected_descriptor_layout.append(
                        (
                            (
                                "final_property_negative_c_upper_only"
                                if upper_only
                                else "final_property_c_minus_c"
                            ),
                            None,
                            None,
                            block_rows,
                            (
                                "negated_rows_only_for_property_upper_bounds"
                                if upper_only
                                else "positive_rows_then_negated_rows"
                            ),
                            None,
                        )
                    )
            if (
                receipt.get("planned_query_blocks")
                != len(expected_descriptor_layout)
            ):
                return False
        if generated:
            if (
                receipt.get("candidate_generated") is not True
                or receipt.get("whole_batch_complete") is not True
                or not descriptors
                or len(descriptors) != len(alpha_trees)
                or not isinstance(records, list)
                or len(records) != len(descriptors)
                or not isinstance(alpha_hashes, list)
                or len(alpha_hashes) != len(alpha_trees)
                or int(receipt.get("query_blocks", -1)) != len(descriptors)
                or int(receipt.get("alpha_trees", -1)) != len(alpha_trees)
                or (
                    v1
                    and receipt.get(
                        "all_candidate_updates_replayed_with_stored_alpha"
                    )
                    is not True
                )
                or (
                    descriptor_schema
                    and (
                        receipt.get("descriptor_coverage_complete") is not True
                        or receipt.get("descriptor_coverage_sha256")
                        != receipt.get("descriptor_records_sha256")
                    )
                )
                or (
                    v3
                    and (
                        receipt.get("selected_coverage_complete")
                        is not True
                        or receipt.get("property_coverage_complete")
                        is not True
                    )
                )
            ):
                return False
        else:
            if (
                receipt.get("candidate_generated") is not False
                or descriptors
                or alpha_trees
                or int(receipt.get("query_blocks", -1)) != 0
                or int(receipt.get("alpha_trees", -1)) != 0
                or improved_target.size
                or improved_property.size
            ):
                return False
            if descriptor_schema and (
                status
                not in {
                    "disabled",
                    "no_queries_fallback",
                    "deadline_fallback_frozen_bounds",
                    "error_fallback_frozen_bounds",
                }
                or records != []
                or alpha_hashes != []
                or receipt.get("descriptor_records_sha256")
                != _canonical_sha256([])
                or receipt.get("descriptor_coverage_sha256")
                != _canonical_sha256([])
                or receipt.get("alpha_hashes_sha256")
                != _canonical_sha256([])
                or (
                    v3
                    and (
                        receipt.get(
                            "selected_descriptor_rows_sha256"
                        )
                        != _ordered_rows_sha256(())
                        or receipt.get(
                            "property_descriptor_rows_sha256"
                        )
                        != _ordered_rows_sha256(())
                    )
                )
            ):
                return False
            if descriptor_schema:
                whole_complete = receipt.get("whole_batch_complete")
                coverage_complete = receipt.get(
                    "descriptor_coverage_complete"
                )
                completed = receipt.get("completed_blocks_discarded")
                if (
                    isinstance(completed, (bool, np.bool_))
                    or not isinstance(completed, (Integral, np.integer))
                    or int(completed) < 0
                ):
                    return False
                if status == "disabled":
                    if v3:
                        expected_queries = int(
                            receipt.get("selected_target_count", 0)
                        ) + int(receipt.get("property_rows", 0))
                        valid_fallback = bool(
                            receipt.get("steps_requested") == 0
                            and whole_complete is True
                            and coverage_complete
                            is (expected_queries == 0)
                            and receipt.get(
                                "selected_coverage_complete"
                            )
                            is (
                                int(
                                    receipt.get(
                                        "selected_target_count", 0
                                    )
                                )
                                == 0
                            )
                            and receipt.get(
                                "property_coverage_complete"
                            )
                            is (
                                int(receipt.get("property_rows", 0))
                                == 0
                            )
                            and int(completed) == 0
                        )
                    else:
                        valid_fallback = bool(
                            receipt.get("steps_requested") == 0
                            and whole_complete is True
                            and coverage_complete is True
                            and int(completed) == 0
                        )
                elif status == "no_queries_fallback":
                    if v3:
                        valid_fallback = bool(
                            receipt.get("steps_requested", 0) > 0
                            and receipt.get("property_rows") == 0
                            and receipt.get("selected_target_count") == 0
                            and receipt.get("planned_query_blocks") == 0
                            and whole_complete is True
                            and coverage_complete is True
                            and receipt.get(
                                "selected_coverage_complete"
                            )
                            is True
                            and receipt.get(
                                "property_coverage_complete"
                            )
                            is True
                            and int(completed) == 0
                        )
                    else:
                        actual_unstable = int(
                            np.count_nonzero(
                                (target_lb < 0.0) & (target_ub > 0.0)
                            )
                        )
                        valid_fallback = bool(
                            receipt.get("steps_requested", 0) > 0
                            and receipt.get("property_rows") == 0
                            and receipt.get(
                                "unstable_target_neurons"
                            )
                            == 0
                            and actual_unstable == 0
                            and receipt.get("planned_query_blocks") == 0
                            and whole_complete is True
                            and coverage_complete is True
                            and int(completed) == 0
                        )
                else:
                    valid_fallback = bool(
                        status
                        in {
                            "deadline_fallback_frozen_bounds",
                            "error_fallback_frozen_bounds",
                        }
                        and whole_complete is False
                        and coverage_complete is False
                    )
                if not valid_fallback:
                    return False
            return True

        seen_rows: Dict[str, set[int]] = {}
        for index, descriptor in enumerate(descriptors):
            if (
                not isinstance(descriptor, QueryDescriptor)
                or descriptor.block_id != index
                or descriptor.alpha_tree_index != index
                or descriptor.M <= 0
                or descriptor.M != int(np.asarray(descriptor.objectives).shape[0])
                or descriptor.alpha_tree_index < 0
                or descriptor.alpha_tree_index >= len(alpha_trees)
            ):
                return False
            objectives = np.ascontiguousarray(
                np.asarray(descriptor.objectives, dtype=np.float64)
            )
            if (
                objectives.ndim != 2
                or not np.all(np.isfinite(objectives))
                or descriptor.objective_sha256 != _array_sha256(objectives)
            ):
                return False
            rows = tuple(int(value) for value in descriptor.row_ids)
            if (
                not rows
                or any(value < 0 for value in rows)
                or len(set(rows)) != len(rows)
            ):
                return False
            kind_rows = seen_rows.setdefault(descriptor.query_kind, set())
            if kind_rows.intersection(rows):
                return False
            kind_rows.update(rows)
            alpha_hash = _alpha_sha256(alpha_trees[index])
            if (
                descriptor.alpha_sha256 != alpha_hash
                or alpha_hashes[index] != alpha_hash
            ):
                return False
            record = records[index]
            if not isinstance(record, Mapping):
                return False
            expected_record_fields = {
                "block_id": int(descriptor.block_id),
                "query_kind": descriptor.query_kind,
                "start_lid": descriptor.start_lid,
                "target_relu_lid": descriptor.target_relu_lid,
                "row_ids": list(rows),
                "objective_order": descriptor.objective_order,
                "objective_sha256": descriptor.objective_sha256,
                "M": int(descriptor.M),
                "alpha_tree_index": int(descriptor.alpha_tree_index),
                "alpha_sha256": descriptor.alpha_sha256,
                "bound_source": (
                    "none_descriptor_only"
                    if descriptor_schema
                    else "frozen_alpha_replay_only"
                ),
            }
            if any(
                record.get(key) != value
                for key, value in expected_record_fields.items()
            ):
                return False
            if descriptor_schema and (
                "optimizer_margin_sha256" in record
                or "replay_margin_sha256" in record
            ):
                return False
            if descriptor_schema:
                if index >= len(expected_descriptor_layout):
                    return False
                (
                    expected_kind,
                    expected_start,
                    expected_target,
                    expected_rows,
                    expected_order,
                    expected_objective,
                ) = expected_descriptor_layout[index]
                if (
                    descriptor.query_kind != expected_kind
                    or descriptor.start_lid != expected_start
                    or descriptor.target_relu_lid != expected_target
                    or rows != expected_rows
                    or descriptor.objective_order != expected_order
                ):
                    return False
                if expected_objective is not None:
                    if not np.array_equal(objectives, expected_objective):
                        return False
                elif expected_kind == "final_property_c_minus_c":
                    count = len(expected_rows)
                    if (
                        objectives.shape[0] != 2 * count
                        or objectives.shape[1] != property_width
                        or not np.array_equal(
                            objectives[count:], -objectives[:count]
                        )
                    ):
                        return False
                elif (
                    objectives.shape[0] != len(expected_rows)
                    or objectives.shape[1] != property_width
                ):
                    return False

        if descriptor_schema:
            property_blocks = []
            for descriptor in descriptors:
                objective = np.asarray(
                    descriptor.objectives, dtype=np.float64
                )
                if descriptor.query_kind == "final_property_c_minus_c":
                    property_blocks.append(
                        objective[: len(descriptor.row_ids)].copy()
                    )
                elif (
                    descriptor.query_kind
                    == "final_property_negative_c_upper_only"
                ):
                    property_blocks.append((-objective).copy())
            property_count = int(receipt.get("property_rows", 0))
            if property_count:
                if not property_blocks:
                    return False
                reconstructed_property = np.vstack(property_blocks)
                if (
                    reconstructed_property.shape
                    != (
                        property_count,
                        int(receipt.get("property_width", -1)),
                    )
                    or receipt.get("property_rows_sha256")
                    != _array_sha256(reconstructed_property)
                ):
                    return False
            elif property_blocks or receipt.get("property_rows_sha256") is not None:
                return False

        if v3:
            descriptor_selected_rows = tuple(
                int(row)
                for descriptor in descriptors
                if descriptor.query_kind
                == "relu_unstable_plus_minus_one_hot"
                for row in descriptor.row_ids
            )
            descriptor_property_rows = tuple(
                int(row)
                for descriptor in descriptors
                if descriptor.query_kind
                in {
                    "final_property_c_minus_c",
                    "final_property_negative_c_upper_only",
                }
                for row in descriptor.row_ids
            )
            if (
                descriptor_selected_rows
                != tuple(
                    int(value)
                    for value in receipt.get(
                        "selected_target_row_ids", []
                    )
                )
                or descriptor_property_rows
                != tuple(
                    int(value)
                    for value in receipt.get(
                        "selected_property_row_ids", []
                    )
                )
                or receipt.get("selected_descriptor_rows_sha256")
                != _ordered_rows_sha256(descriptor_selected_rows)
                or receipt.get("property_descriptor_rows_sha256")
                != _ordered_rows_sha256(descriptor_property_rows)
            ):
                return False

        if (
            (
                descriptor_schema
                and len(descriptors) != len(expected_descriptor_layout)
            )
            or
            receipt.get("descriptor_records_sha256")
            != _canonical_sha256(records)
            or receipt.get("alpha_hashes_sha256")
            != _canonical_sha256(alpha_hashes)
            or (
                descriptor_schema
                and receipt.get("descriptor_coverage_sha256")
                != _canonical_sha256(records)
            )
        ):
            return False
        return True
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return False


def _layer_kind(layer: Any) -> str:
    kind = getattr(layer, "kind", "")
    return str(kind).upper()


def _layer_by_id(net: Any) -> Dict[int, Any]:
    layers = getattr(net, "layers", None)
    if not isinstance(layers, Sequence):
        raise ValueError("net.layers must be a sequence")
    result = {int(layer.id): layer for layer in layers}
    if len(result) != len(layers):
        raise ValueError("net layer ids must be unique")
    return result


def _property_baseline(
    *,
    net: Any,
    bounds_dict: Mapping[int, Bounds],
    property_rows: Optional[Any],
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, Optional[int]]:
    if property_rows is None:
        return None, np.zeros(0, dtype=np.float64), np.zeros(
            0, dtype=np.float64
        ), None
    if isinstance(property_rows, torch.Tensor):
        rows = property_rows.detach().to(
            device="cpu", dtype=torch.float64
        ).numpy()
    else:
        rows = np.asarray(property_rows, dtype=np.float64)
    if (
        rows.ndim != 2
        or rows.shape[0] <= 0
        or rows.shape[1] <= 0
        or not np.all(np.isfinite(rows))
    ):
        raise ValueError("property_rows must be a finite nonempty matrix")
    by_id = _layer_by_id(net)
    assert_layers = [
        layer for layer in by_id.values() if _layer_kind(layer) == "ASSERT"
    ]
    if len(assert_layers) != 1:
        raise ValueError("property queries require exactly one ASSERT layer")
    assert_lid = int(assert_layers[0].id)
    preds = getattr(net, "preds", {}).get(assert_lid, [])
    if len(preds) != 1:
        raise ValueError("ASSERT layer must have exactly one predecessor")
    output_lid = int(preds[0])
    if output_lid not in bounds_dict:
        raise ValueError("frozen bounds omit the ASSERT predecessor")
    output = bounds_dict[output_lid]
    lb = _tensor_cpu_f64(output.lb, name="property output lb").numpy()
    ub = _tensor_cpu_f64(output.ub, name="property output ub").numpy()
    if lb.shape[0] != 1 or ub.shape != lb.shape:
        raise ValueError("property output bounds must have batch size one")
    lb_flat = lb.reshape(1, -1)[0]
    ub_flat = ub.reshape(1, -1)[0]
    if rows.shape[1] != lb_flat.size:
        raise ValueError("property row width disagrees with output bounds")
    positive = np.maximum(rows, 0.0)
    negative = np.minimum(rows, 0.0)
    lower = positive @ lb_flat + negative @ ub_flat
    upper = positive @ ub_flat + negative @ lb_flat
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("property interval baseline is non-finite")
    return rows.copy(), lower, upper, output_lid


def _cpu_bounds(bounds: Bounds) -> Bounds:
    return Bounds(
        lb=_tensor_cpu_f64(bounds.lb, name="fallback target lb"),
        ub=_tensor_cpu_f64(bounds.ub, name="fallback target ub"),
    )


def _fallback_result(
    *,
    status: str,
    started: float,
    receipt: Dict[str, Any],
    target_bounds: Bounds,
    property_lower: np.ndarray,
    property_upper: np.ndarray,
    timings: Sequence[Mapping[str, Any]] = (),
    completed_blocks_discarded: int = 0,
    error: Optional[BaseException] = None,
) -> QueryDualCandidates:
    target = _cpu_bounds(target_bounds)
    payload = {
        **receipt,
        "status": str(status),
        "candidate_generated": False,
        "whole_batch_complete": bool(
            status
            in {
                "disabled",
                "no_queries_fallback",
                "no_improvement_fallback",
            }
        ),
        "completed_blocks_discarded": int(completed_blocks_discarded),
        "query_blocks": 0,
        "alpha_trees": 0,
        "strict_target_improvements": 0,
        "strict_property_improvements": 0,
        "caller_bounds_unchanged": True,
        "optimizer_best_margins_used_as_bounds": False,
        "elapsed_seconds": float(max(0.0, time.monotonic() - started)),
        "timings": [dict(value) for value in timings],
        "candidate_target_bounds_sha256": _array_sha256(
            np.stack(
                [
                    target.lb.numpy().reshape(-1),
                    target.ub.numpy().reshape(-1),
                ]
            )
        ),
        "candidate_property_bounds_sha256": _array_sha256(
            np.stack([property_lower, property_upper])
            if property_lower.size
            else np.zeros((2, 0), dtype=np.float64)
        ),
    }
    if error is not None:
        payload.update(
            {
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
            }
        )
    descriptor_schema = receipt.get("schema") in {
        "act.query_dual_candidates.v2",
        "act.query_dual_candidates.v3",
    }
    if descriptor_schema:
        empty_records: List[Dict[str, Any]] = []
        empty_hashes: List[str] = []
        v3 = receipt.get("schema") == "act.query_dual_candidates.v3"
        selected_count = int(receipt.get("selected_target_count", 0))
        property_count = int(receipt.get("property_rows", 0))
        vacuous_coverage = selected_count == 0 and property_count == 0
        payload.update(
            {
                "descriptor_records": empty_records,
                "descriptor_records_sha256": _canonical_sha256(empty_records),
                "descriptor_coverage_sha256": _canonical_sha256(
                    empty_records
                ),
                "descriptor_coverage_complete": bool(
                    (
                        status in {"disabled", "no_queries_fallback"}
                        if not v3
                        else (
                            status == "no_queries_fallback"
                            or (
                                status == "disabled"
                                and vacuous_coverage
                            )
                        )
                    )
                ),
                "alpha_hashes": empty_hashes,
                "alpha_hashes_sha256": _canonical_sha256(empty_hashes),
                "improved_target_indices": [],
                "improved_property_indices": [],
                "optimizer_margins_exported": False,
                "optimizer_margins_used_for_improvement": False,
                "gpu_frozen_alpha_replay": False,
                "cpu_independent_replay_required": True,
                "all_candidate_updates_replayed_with_stored_alpha": False,
                "all_bounds_replayed_with_stored_alpha": False,
                "property_lower_dual_replayed": False,
            }
        )
        if v3:
            omitted_rows = tuple(
                int(value)
                for value in receipt.get("omitted_target_row_ids", [])
            )
            target_lower = target.lb.numpy().reshape(-1)
            target_upper = target.ub.numpy().reshape(-1)
            selection_bound = receipt.get("sparse_selection_bound") is True
            unselected_candidate_hash = (
                _indexed_bounds_sha256(
                    target_lower, target_upper, omitted_rows
                )
                if selection_bound
                else None
            )
            payload.update(
                {
                    "selected_descriptor_rows_sha256":
                        _ordered_rows_sha256(()),
                    "property_descriptor_rows_sha256":
                        _ordered_rows_sha256(()),
                    "selected_coverage_complete": bool(
                        selected_count == 0
                        and status
                        in {"disabled", "no_queries_fallback"}
                    ),
                    "property_coverage_complete": bool(
                        property_count == 0
                        and status
                        in {"disabled", "no_queries_fallback"}
                    ),
                    "unselected_bounds_bit_identical_parent": bool(
                        selection_bound
                        and unselected_candidate_hash
                        == receipt.get(
                            "unselected_parent_target_bounds_sha256"
                        )
                    ),
                    "unselected_candidate_target_bounds_sha256":
                        unselected_candidate_hash,
                }
            )
    return QueryDualCandidates(
        target_bounds=target,
        property_lower=np.asarray(property_lower, dtype=np.float64).copy(),
        property_upper=np.asarray(property_upper, dtype=np.float64).copy(),
        query_descriptors=(),
        alpha_trees=(),
        improved_target_indices=np.zeros(0, dtype=np.int64),
        improved_property_indices=np.zeros(0, dtype=np.int64),
        timings=tuple(dict(value) for value in timings),
        receipt=_receipt_with_sha256(payload),
    )


def generate_query_dual_candidates(
    *,
    net: Any,
    bounds_dict: Mapping[int, Bounds],
    target_relu_lid: Optional[int] = None,
    property_rows: Optional[Any] = None,
    property_upper_only: bool = True,
    steps: int = 8,
    block_size: int = 64,
    lr_alpha: float = 0.25,
    lr_decay: float = 0.98,
    deadline: Optional[float] = None,
    solver_factory: Optional[Callable[[], Any]] = None,
    descriptor_only: bool = False,
    selected_target_rows: Optional[Sequence[int]] = None,
) -> QueryDualCandidates:
    """Generate frozen-alpha query material without mutating ``bounds_dict``.

    ``selected_target_rows is None`` preserves the V1/V2 APIs exactly.  An
    explicit sequence (including ``()`` for a property-only stage) requests
    sparse descriptor-only V3.
    """

    started = time.monotonic()
    source_hash: Optional[str] = None
    target_fallback = Bounds(
        lb=torch.zeros((1, 0), dtype=torch.float64),
        ub=torch.zeros((1, 0), dtype=torch.float64),
    )
    property_lower = np.zeros(0, dtype=np.float64)
    property_upper = np.zeros(0, dtype=np.float64)
    timings: List[Mapping[str, Any]] = []
    completed_blocks = 0
    descriptor_mode = bool(descriptor_only) if isinstance(
        descriptor_only, (bool, np.bool_)
    ) else False
    sparse_mode = selected_target_rows is not None
    base_receipt: Dict[str, Any] = {
        "schema": (
            "act.query_dual_candidates.v3"
            if sparse_mode
            else "act.query_dual_candidates.v2"
            if descriptor_mode
            else "act.query_dual_candidates.v1"
        ),
        "candidate_only": True,
        "proof_authority": False,
        "return_optimized_required": True,
        "refresh_forward": False,
        "bounds_source": "caller_frozen_bounds_private_clone",
        "alpha_storage": "cpu_stored_binary64_tree",
        "candidate_bound_source": "optimize_false_frozen_alpha_replay",
        "optimizer_best_margins_used_as_bounds": False,
        "shared_absolute_deadline": deadline is not None,
        "deadline_monotonic": (
            float(deadline)
            if deadline is not None
            and isinstance(deadline, Real)
            and not isinstance(deadline, (bool, np.bool_))
            and math.isfinite(float(deadline))
            else None if deadline is None else repr(deadline)
        ),
        "lr_alpha": (
            float(lr_alpha)
            if isinstance(lr_alpha, Real)
            and not isinstance(lr_alpha, (bool, np.bool_))
            and math.isfinite(float(lr_alpha))
            else repr(lr_alpha)
        ),
        "lr_decay": (
            float(lr_decay)
            if isinstance(lr_decay, Real)
            and not isinstance(lr_decay, (bool, np.bool_))
            and math.isfinite(float(lr_decay))
            else repr(lr_decay)
        ),
        "steps_requested": (
            int(steps)
            if isinstance(steps, (Integral, np.integer))
            and not isinstance(steps, (bool, np.bool_))
            else repr(steps)
        ),
        "block_size": (
            int(block_size)
            if isinstance(block_size, (Integral, np.integer))
            and not isinstance(block_size, (bool, np.bool_))
            else repr(block_size)
        ),
        "target_relu_lid": (
            int(target_relu_lid)
            if isinstance(target_relu_lid, (Integral, np.integer))
            and not isinstance(target_relu_lid, (bool, np.bool_))
            else None if target_relu_lid is None else repr(target_relu_lid)
        ),
        "property_only": target_relu_lid is None,
        "property_upper_only": (
            bool(property_upper_only)
            if isinstance(property_upper_only, (bool, np.bool_))
            else repr(property_upper_only)
        ),
    }
    if descriptor_mode:
        base_receipt.update(
            {
                "protocol": (
                    "property_sparse_descriptor_only_v3"
                    if sparse_mode
                    else "descriptor_only_v2"
                ),
                # These fields aid performance/debug forensics.  They neither
                # supply a bound nor grant authority; objective, alpha and
                # every consumed value are independently rebound downstream.
                "non_authoritative_audit_fields": list(
                    _V2_NON_AUTHORITATIVE_AUDIT_FIELDS
                ),
                "candidate_bound_source": "none_descriptor_only",
                "optimizer_margins_exported": False,
                "optimizer_margins_used_for_improvement": False,
                "gpu_frozen_alpha_replay": False,
                "cpu_independent_replay_required": True,
            }
        )
    if sparse_mode:
        base_receipt.update(
            {
                "selector_authoritative": False,
                "selected_rows_source":
                    "caller_supplied_non_authoritative_selector",
                "sparse_selection_bound": False,
                "eligible_target_row_ids": [],
                "eligible_target_rows_sha256": _ordered_rows_sha256(()),
                "eligible_target_count": 0,
                "selected_target_row_ids": [],
                "selected_target_rows_sha256": _ordered_rows_sha256(()),
                "selected_target_count": 0,
                "omitted_target_row_ids": [],
                "omitted_target_rows_sha256": _ordered_rows_sha256(()),
                "omitted_target_count": 0,
                "target_partition_complete": False,
                "target_partition_disjoint": False,
                "selected_coverage_complete": False,
                "property_coverage_policy": "all_property_rows",
                "unselected_policy":
                    "bit_identical_immutable_parent_target_bounds",
                "unselected_bounds_bit_identical_parent": False,
            }
        )
    try:
        if not isinstance(descriptor_only, (bool, np.bool_)):
            raise ValueError("descriptor_only must be a boolean")
        if sparse_mode and not descriptor_mode:
            raise ValueError(
                "selected_target_rows requires descriptor_only=True"
            )
        if target_relu_lid is None:
            if property_rows is None:
                raise ValueError(
                    "target_relu_lid=None requires nonempty property_rows"
                )
            target_lid: Optional[int] = None
        else:
            if (
                isinstance(target_relu_lid, (bool, np.bool_))
                or not isinstance(
                    target_relu_lid, (Integral, np.integer)
                )
            ):
                raise ValueError(
                    "target_relu_lid must be an integer or None"
                )
            target_lid = int(target_relu_lid)
        if not isinstance(property_upper_only, (bool, np.bool_)):
            raise ValueError("property_upper_only must be a boolean")
        upper_only = bool(property_upper_only)
        if target_lid is not None:
            early_target = bounds_dict.get(target_lid)
            if isinstance(early_target, Bounds):
                try:
                    target_fallback = _cpu_bounds(early_target)
                except (TypeError, ValueError):
                    # A malformed target cannot itself be a safe fallback.
                    # Keep the explicit empty target while preserving caller
                    # state and reporting the validation error below.
                    pass
        if (
            isinstance(steps, (bool, np.bool_))
            or not isinstance(steps, (Integral, np.integer))
            or int(steps) < 0
        ):
            raise ValueError("steps must be a nonnegative integer")
        step_count = int(steps)
        if (
            isinstance(block_size, (bool, np.bool_))
            or not isinstance(block_size, (Integral, np.integer))
            or int(block_size) <= 0
        ):
            raise ValueError("block_size must be a positive integer")
        chunk = int(block_size)
        if (
            isinstance(lr_alpha, (bool, np.bool_))
            or not isinstance(lr_alpha, Real)
            or not math.isfinite(float(lr_alpha))
            or float(lr_alpha) <= 0.0
        ):
            raise ValueError("lr_alpha must be finite and positive")
        if (
            isinstance(lr_decay, (bool, np.bool_))
            or not isinstance(lr_decay, Real)
            or not math.isfinite(float(lr_decay))
            or not 0.0 < float(lr_decay) <= 1.0
        ):
            raise ValueError("lr_decay must lie in (0, 1]")
        if deadline is not None and (
            isinstance(deadline, (bool, np.bool_))
            or not isinstance(deadline, Real)
            or not math.isfinite(float(deadline))
        ):
            raise ValueError("deadline must be a finite absolute time")

        source_hash = _bounds_sha256(bounds_dict)
        base_receipt["input_bounds_sha256"] = source_hash
        by_id = _layer_by_id(net)
        rows, property_lower, property_upper, property_output_lid = (
            _property_baseline(
                net=net,
                bounds_dict=bounds_dict,
                property_rows=property_rows,
            )
        )
        target_source: Optional[Bounds]
        if target_lid is None:
            target_source = None
            start_lid = None
            target_lb = np.zeros(0, dtype=np.float64)
            target_ub = np.zeros(0, dtype=np.float64)
            if property_output_lid is None:
                raise ValueError(
                    "property-only queries require an ASSERT predecessor"
                )
            execution_source = bounds_dict[property_output_lid]
        else:
            if (
                target_lid not in by_id
                or _layer_kind(by_id[target_lid]) != "RELU"
            ):
                raise ValueError(
                    "target_relu_lid must identify a ReLU layer"
                )
            preds = getattr(net, "preds", {}).get(target_lid, [])
            if len(preds) != 1:
                raise ValueError(
                    "target ReLU must have exactly one predecessor"
                )
            start_lid = int(preds[0])
            if (
                target_lid not in bounds_dict
                or start_lid not in bounds_dict
            ):
                raise ValueError(
                    "frozen bounds omit target ReLU/predecessor"
                )
            target_source = bounds_dict[target_lid]
            target_fallback = _cpu_bounds(target_source)
            target_lb = (
                target_fallback.lb.numpy().reshape(1, -1)[0].copy()
            )
            target_ub = (
                target_fallback.ub.numpy().reshape(1, -1)[0].copy()
            )
            if (
                target_fallback.lb.shape[0] != 1
                or target_fallback.ub.shape != target_fallback.lb.shape
            ):
                raise ValueError(
                    "target ReLU bounds must have batch size one"
                )
            pred_bounds = bounds_dict[start_lid]
            if pred_bounds.lb.numel() != target_source.lb.numel():
                raise ValueError(
                    "target ReLU/predecessor widths disagree"
                )
            execution_source = target_source
        unstable = (
            tuple(
                int(value)
                for value in np.flatnonzero(
                    (target_lb < 0.0) & (target_ub > 0.0)
                )
            )
            if target_lid is not None
            else ()
        )
        if sparse_mode:
            selected_rows = _normalize_selected_target_rows(
                selected_target_rows
            )
            if target_lid is None and selected_rows:
                raise ValueError(
                    "property-only V3 requires selected_target_rows=()"
                )
            unstable_set = set(unstable)
            ineligible = [
                row for row in selected_rows if row not in unstable_set
            ]
            if ineligible:
                raise ValueError(
                    "selected_target_rows contain non-unstable rows: "
                    f"{ineligible}"
                )
            selected_set = set(selected_rows)
            omitted_rows = tuple(
                row for row in unstable if row not in selected_set
            )
            property_row_ids = tuple(
                range(int(rows.shape[0])) if rows is not None else ()
            )
            base_receipt.update(
                {
                    "sparse_selection_bound": True,
                    "eligible_target_row_ids": list(unstable),
                    "eligible_target_rows_sha256":
                        _ordered_rows_sha256(unstable),
                    "eligible_target_count": int(len(unstable)),
                    "selected_target_row_ids": list(selected_rows),
                    "selected_target_rows_sha256":
                        _ordered_rows_sha256(selected_rows),
                    "selected_target_count": int(len(selected_rows)),
                    "omitted_target_row_ids": list(omitted_rows),
                    "omitted_target_rows_sha256":
                        _ordered_rows_sha256(omitted_rows),
                    "omitted_target_count": int(len(omitted_rows)),
                    "target_partition_complete": (
                        unstable_set
                        == selected_set.union(omitted_rows)
                    ),
                    "target_partition_disjoint":
                        selected_set.isdisjoint(omitted_rows),
                    "selected_parent_target_bounds_sha256":
                        _indexed_bounds_sha256(
                            target_lb, target_ub, selected_rows
                        ),
                    "unselected_parent_target_bounds_sha256":
                        _indexed_bounds_sha256(
                            target_lb, target_ub, omitted_rows
                        ),
                    "eligible_property_row_ids":
                        list(property_row_ids),
                    "eligible_property_rows_sha256":
                        _ordered_rows_sha256(property_row_ids),
                    "selected_property_row_ids":
                        list(property_row_ids),
                    "selected_property_rows_sha256":
                        _ordered_rows_sha256(property_row_ids),
                    "selected_property_count":
                        int(len(property_row_ids)),
                }
            )
            query_target_rows = selected_rows
        else:
            query_target_rows = unstable
        base_receipt.update(
            {
                "target_start_lid": start_lid,
                "target_width": int(target_lb.size),
                "target_bounds_sha256": _array_sha256(
                    np.stack([target_lb, target_ub])
                ),
                "property_rows": int(rows.shape[0]) if rows is not None else 0,
                "property_width": int(rows.shape[1]) if rows is not None else 0,
                "property_rows_sha256": (
                    _array_sha256(rows) if rows is not None else None
                ),
                "property_output_lid": property_output_lid,
                "property_baseline_sha256": _array_sha256(
                    np.stack([property_lower, property_upper])
                    if property_lower.size
                    else np.zeros((2, 0), dtype=np.float64)
                ),
                "property_lower_bound_source": (
                    "frozen_interval_baseline_not_dual_replayed"
                    if rows is not None and (upper_only or descriptor_mode)
                    else (
                        "frozen_alpha_replay"
                        if rows is not None
                        else "not_requested"
                    )
                ),
                "property_upper_bound_source": (
                    (
                        "baseline_placeholder_no_candidate_bound"
                        if descriptor_mode
                        else "frozen_alpha_replay"
                    )
                    if rows is not None
                    else "not_requested"
                ),
            }
        )
        base_receipt["unstable_target_neurons"] = int(len(unstable))
        if step_count == 0:
            return _fallback_result(
                status="disabled",
                started=started,
                receipt=base_receipt,
                target_bounds=target_fallback,
                property_lower=property_lower,
                property_upper=property_upper,
            )
        if deadline is not None and time.monotonic() >= float(deadline):
            raise _QueryDeadline("query deadline already exhausted")

        device = execution_source.lb.device
        dtype = execution_source.lb.dtype
        if not dtype.is_floating_point:
            raise ValueError("frozen bounds must use a floating dtype")
        private_bounds = _clone_bounds_dict(
            bounds_dict, device=device, dtype=dtype
        )
        private_hash = _bounds_sha256(private_bounds)
        if private_hash != source_hash:
            raise RuntimeError("private frozen-bounds clone changed values")
        planned: List[Tuple[str, Optional[int], Tuple[int, ...], np.ndarray]] = []
        for offset in range(0, len(query_target_rows), chunk):
            indices = query_target_rows[offset : offset + chunk]
            eye = np.zeros(
                (len(indices), target_lb.size), dtype=np.float64
            )
            eye[
                np.arange(len(indices)),
                np.asarray(indices, dtype=np.int64),
            ] = 1.0
            planned.append(
                (
                    "relu_unstable_plus_minus_one_hot",
                    start_lid,
                    tuple(int(value) for value in indices),
                    np.vstack([eye, -eye]),
                )
            )
        if rows is not None:
            for offset in range(0, int(rows.shape[0]), chunk):
                row_ids = tuple(
                    range(offset, min(offset + chunk, int(rows.shape[0])))
                )
                block = rows[np.asarray(row_ids, dtype=np.int64), :]
                if upper_only:
                    planned.append(
                        (
                            "final_property_negative_c_upper_only",
                            None,
                            tuple(int(value) for value in row_ids),
                            -block,
                        )
                    )
                else:
                    planned.append(
                        (
                            "final_property_c_minus_c",
                            None,
                            tuple(int(value) for value in row_ids),
                            np.vstack([block, -block]),
                        )
                    )
        base_receipt["planned_query_blocks"] = int(len(planned))
        if not planned:
            return _fallback_result(
                status="no_queries_fallback",
                started=started,
                receipt=base_receipt,
                target_bounds=target_fallback,
                property_lower=property_lower,
                property_upper=property_upper,
            )

        if solver_factory is None:
            from act.back_end.solver.solver_dual import DualSolver

            solver = DualSolver()
            solver_name = "DualSolver"
        else:
            solver = solver_factory()
            solver_name = type(solver).__name__
        compute = getattr(solver, "compute_certified_bound", None)
        if not callable(compute):
            raise TypeError("solver_factory must return a DualSolver-like object")
        base_receipt["solver"] = solver_name

        candidate_lb = target_lb.copy()
        candidate_ub = target_ub.copy()
        candidate_property_lower = property_lower.copy()
        candidate_property_upper = property_upper.copy()
        descriptors: List[QueryDescriptor] = []
        stored_alphas: List[Mapping[int, Any]] = []
        descriptor_records: List[Dict[str, Any]] = []
        alpha_hashes: List[str] = []

        for block_id, (kind, query_start, row_ids, objectives) in enumerate(
            planned
        ):
            if deadline is not None and time.monotonic() >= float(deadline):
                raise _QueryDeadline("deadline before query optimization")
            objective = torch.as_tensor(
                objectives, device=device, dtype=dtype
            ).contiguous()
            M = int(objective.shape[0])
            optimize_started = time.monotonic()
            optimized = compute(
                net,
                private_bounds,
                objective,
                M=M,
                optimize=True,
                n_iters=step_count,
                lr_alpha=float(lr_alpha),
                lr_decay=float(lr_decay),
                per_class_alpha=True,
                refresh_forward=False,
                return_optimized=True,
                start_lid=query_start,
            )
            optimize_elapsed = time.monotonic() - optimize_started
            if deadline is not None and time.monotonic() >= float(deadline):
                raise _QueryDeadline("deadline after query optimization")
            alpha_state = getattr(optimized, "alpha_state", None)
            if not isinstance(alpha_state, Mapping) or not alpha_state:
                raise RuntimeError(
                    "DualSolver return_optimized produced no alpha state"
                )
            optimizer_margin_cpu: Optional[torch.Tensor] = None
            if not descriptor_mode:
                optimizer_margins = getattr(optimized, "margins", None)
                if not isinstance(optimizer_margins, torch.Tensor):
                    raise RuntimeError(
                        "DualSolver optimized result has no margins"
                    )
                optimizer_margin_cpu = _tensor_cpu_f64(
                    optimizer_margins, name="optimizer margins"
                ).reshape(-1)
                if optimizer_margin_cpu.numel() != M:
                    raise RuntimeError(
                        "optimizer margin count disagrees with query"
                    )

            stored_alpha = _alpha_to_cpu_f64(alpha_state)
            if not isinstance(stored_alpha, Mapping) or not stored_alpha:
                raise RuntimeError("stored optimized alpha tree is empty")
            alpha_hash = _alpha_sha256(stored_alpha)
            if (
                descriptor_mode
                and deadline is not None
                and time.monotonic() >= float(deadline)
            ):
                raise _QueryDeadline("deadline after alpha freeze")

            replay_elapsed = 0.0
            replay_values: Optional[np.ndarray] = None
            if not descriptor_mode:
                replay_alpha = _alpha_to_device(
                    stored_alpha, device=device, dtype=dtype
                )
                if _alpha_sha256(
                    _alpha_to_cpu_f64(replay_alpha)
                ) != alpha_hash:
                    raise RuntimeError(
                        "stored alpha changed during replay transfer"
                    )
                if deadline is not None and time.monotonic() >= float(deadline):
                    raise _QueryDeadline("deadline after alpha freeze")
                replay_started = time.monotonic()
                replayed = compute(
                    net,
                    private_bounds,
                    objective,
                    M=M,
                    optimize=False,
                    alpha=replay_alpha,
                    per_class_alpha=True,
                    refresh_forward=False,
                    return_optimized=False,
                    start_lid=query_start,
                )
                replay_margins = getattr(replayed, "margins", None)
                if not isinstance(replay_margins, torch.Tensor):
                    raise RuntimeError(
                        "DualSolver replay result has no margins"
                    )
                replay_cpu = _tensor_cpu_f64(
                    replay_margins, name="frozen-alpha replay margins"
                ).reshape(-1)
                replay_elapsed = time.monotonic() - replay_started
                if replay_cpu.numel() != M:
                    raise RuntimeError(
                        "replay margin count disagrees with query"
                    )
                if deadline is not None and time.monotonic() >= float(deadline):
                    raise _QueryDeadline(
                        "deadline after frozen-alpha replay"
                    )
                replay_values = replay_cpu.numpy()
            row_count = len(row_ids)
            if kind == "relu_unstable_plus_minus_one_hot":
                if M != 2 * row_count:
                    raise RuntimeError("ReLU objective ordering is malformed")
                objective_order = "positive_rows_then_negated_rows"
                if replay_values is not None:
                    raw_lower = replay_values[:row_count]
                    raw_upper = -replay_values[row_count:]
                    if np.any(raw_lower > raw_upper):
                        raise RuntimeError(
                            "replayed ReLU lower bound exceeds upper bound"
                        )
                    index = np.asarray(row_ids, dtype=np.int64)
                    candidate_lb[index] = np.maximum(
                        candidate_lb[index], raw_lower
                    )
                    candidate_ub[index] = np.minimum(
                        candidate_ub[index], raw_upper
                    )
                    if np.any(candidate_lb[index] > candidate_ub[index]):
                        raise RuntimeError(
                            "candidate target intersection is inconsistent"
                        )
            elif kind == "final_property_c_minus_c":
                if M != 2 * row_count:
                    raise RuntimeError(
                        "bidirectional property ordering is malformed"
                    )
                objective_order = "positive_rows_then_negated_rows"
                if replay_values is not None:
                    raw_lower = replay_values[:row_count]
                    raw_upper = -replay_values[row_count:]
                    if np.any(raw_lower > raw_upper):
                        raise RuntimeError(
                            "replayed property lower bound exceeds upper bound"
                        )
                    index = np.asarray(row_ids, dtype=np.int64)
                    candidate_property_lower[index] = np.maximum(
                        candidate_property_lower[index], raw_lower
                    )
                    candidate_property_upper[index] = np.minimum(
                        candidate_property_upper[index], raw_upper
                    )
                    if np.any(
                        candidate_property_lower[index]
                        > candidate_property_upper[index]
                    ):
                        raise RuntimeError(
                            "candidate property intersection is inconsistent"
                        )
            elif kind == "final_property_negative_c_upper_only":
                if M != row_count:
                    raise RuntimeError(
                        "upper-only property ordering is malformed"
                    )
                objective_order = (
                    "negated_rows_only_for_property_upper_bounds"
                )
                if replay_values is not None:
                    raw_upper = -replay_values
                    index = np.asarray(row_ids, dtype=np.int64)
                    candidate_property_upper[index] = np.minimum(
                        candidate_property_upper[index], raw_upper
                    )
                    if np.any(
                        candidate_property_lower[index]
                        > candidate_property_upper[index]
                    ):
                        raise RuntimeError(
                            "candidate property intersection is inconsistent"
                        )
            else:
                raise RuntimeError(f"unknown query kind: {kind}")

            objective_stored = np.asarray(
                objectives, dtype=np.float64
            ).copy()
            objective_hash = _array_sha256(objective_stored)
            descriptor = QueryDescriptor(
                block_id=int(block_id),
                query_kind=kind,
                start_lid=query_start,
                target_relu_lid=(
                    int(target_lid)
                    if kind == "relu_unstable_plus_minus_one_hot"
                    else None
                ),
                row_ids=row_ids,
                objective_order=objective_order,
                objectives=objective_stored,
                objective_sha256=objective_hash,
                M=M,
                alpha_tree_index=len(stored_alphas),
                alpha_sha256=alpha_hash,
            )
            descriptors.append(descriptor)
            stored_alphas.append(stored_alpha)
            alpha_hashes.append(alpha_hash)
            timing = {
                "block_id": int(block_id),
                "query_kind": kind,
                "rows": int(row_count),
                "M": M,
                "optimize_seconds": float(optimize_elapsed),
            }
            if not descriptor_mode:
                timing["replay_seconds"] = float(replay_elapsed)
            timings.append(timing)
            record = {
                "block_id": int(block_id),
                "query_kind": kind,
                "start_lid": query_start,
                "target_relu_lid": descriptor.target_relu_lid,
                "row_ids": [int(value) for value in row_ids],
                "objective_order": descriptor.objective_order,
                "objective_sha256": objective_hash,
                "M": M,
                "alpha_tree_index": descriptor.alpha_tree_index,
                "alpha_sha256": alpha_hash,
                "bound_source": (
                    "none_descriptor_only"
                    if descriptor_mode
                    else "frozen_alpha_replay_only"
                ),
            }
            if not descriptor_mode:
                if optimizer_margin_cpu is None or replay_values is None:
                    raise RuntimeError(
                        "V1 margin material unexpectedly missing"
                    )
                record.update(
                    {
                        "optimizer_margin_sha256": _array_sha256(
                            optimizer_margin_cpu.numpy()
                        ),
                        "replay_margin_sha256": _array_sha256(replay_values),
                    }
                )
            descriptor_records.append(record)
            completed_blocks += 1

        if _bounds_sha256(bounds_dict) != source_hash:
            raise RuntimeError("caller bounds changed during candidate queries")
        if _bounds_sha256(private_bounds) != private_hash:
            raise RuntimeError("private frozen bounds changed during queries")
        if deadline is not None and time.monotonic() >= float(deadline):
            raise _QueryDeadline("deadline before final candidate return")

        if descriptor_mode:
            candidate_target = _cpu_bounds(target_fallback)
            target_result_lb = (
                candidate_target.lb.numpy().reshape(-1).copy()
            )
            target_result_ub = (
                candidate_target.ub.numpy().reshape(-1).copy()
            )
            descriptor_hash = _canonical_sha256(descriptor_records)
            complete_receipt = {
                **base_receipt,
                "status": "descriptors_generated",
                "candidate_generated": True,
                "whole_batch_complete": True,
                "caller_bounds_unchanged": True,
                "query_blocks": int(len(descriptors)),
                "alpha_trees": int(len(stored_alphas)),
                "descriptor_records": descriptor_records,
                "descriptor_records_sha256": descriptor_hash,
                "descriptor_coverage_sha256": descriptor_hash,
                "descriptor_coverage_complete": True,
                "alpha_hashes": alpha_hashes,
                "alpha_hashes_sha256": _canonical_sha256(alpha_hashes),
                "strict_target_improvements": 0,
                "strict_property_improvements": 0,
                "improved_target_indices": [],
                "improved_property_indices": [],
                "candidate_target_bounds_sha256": _array_sha256(
                    np.stack([target_result_lb, target_result_ub])
                ),
                "candidate_property_bounds_sha256": _array_sha256(
                    np.stack([property_lower, property_upper])
                    if property_lower.size
                    else np.zeros((2, 0), dtype=np.float64)
                ),
                "optimizer_best_margins_used_as_bounds": False,
                "optimizer_margins_exported": False,
                "optimizer_margins_used_for_improvement": False,
                "gpu_frozen_alpha_replay": False,
                "cpu_independent_replay_required": True,
                "all_candidate_updates_replayed_with_stored_alpha": False,
                "all_bounds_replayed_with_stored_alpha": False,
                "property_lower_dual_replayed": False,
                "completed_blocks_discarded": 0,
                "elapsed_seconds": float(
                    max(0.0, time.monotonic() - started)
                ),
                "timings": [dict(value) for value in timings],
            }
            if sparse_mode:
                descriptor_selected_rows = tuple(
                    int(row)
                    for descriptor in descriptors
                    if descriptor.query_kind
                    == "relu_unstable_plus_minus_one_hot"
                    for row in descriptor.row_ids
                )
                descriptor_property_rows = tuple(
                    int(row)
                    for descriptor in descriptors
                    if descriptor.query_kind
                    in {
                        "final_property_c_minus_c",
                        "final_property_negative_c_upper_only",
                    }
                    for row in descriptor.row_ids
                )
                omitted_rows = tuple(
                    int(value)
                    for value in base_receipt[
                        "omitted_target_row_ids"
                    ]
                )
                complete_receipt.update(
                    {
                        "selected_descriptor_rows_sha256":
                            _ordered_rows_sha256(
                                descriptor_selected_rows
                            ),
                        "selected_coverage_complete":
                            descriptor_selected_rows
                            == tuple(
                                int(value)
                                for value in base_receipt[
                                    "selected_target_row_ids"
                                ]
                            ),
                        "property_descriptor_rows_sha256":
                            _ordered_rows_sha256(
                                descriptor_property_rows
                            ),
                        "property_coverage_complete":
                            descriptor_property_rows
                            == tuple(
                                int(value)
                                for value in base_receipt[
                                    "selected_property_row_ids"
                                ]
                            ),
                        "unselected_candidate_target_bounds_sha256":
                            _indexed_bounds_sha256(
                                target_result_lb,
                                target_result_ub,
                                omitted_rows,
                            ),
                        "unselected_bounds_bit_identical_parent":
                            _indexed_bounds_sha256(
                                target_result_lb,
                                target_result_ub,
                                omitted_rows,
                            )
                            == base_receipt[
                                "unselected_parent_target_bounds_sha256"
                            ],
                    }
                )
            finalized = _receipt_with_sha256(complete_receipt)
            if deadline is not None and time.monotonic() >= float(deadline):
                raise _QueryDeadline(
                    "deadline crossed during descriptor receipt finalization"
                )
            return QueryDualCandidates(
                target_bounds=candidate_target,
                property_lower=property_lower.copy(),
                property_upper=property_upper.copy(),
                query_descriptors=tuple(descriptors),
                alpha_trees=tuple(stored_alphas),
                improved_target_indices=np.zeros(0, dtype=np.int64),
                improved_property_indices=np.zeros(0, dtype=np.int64),
                timings=tuple(dict(value) for value in timings),
                receipt=finalized,
            )

        improved_target = np.flatnonzero(
            (candidate_lb > target_lb) | (candidate_ub < target_ub)
        ).astype(np.int64, copy=False)
        improved_property = np.flatnonzero(
            (candidate_property_lower > property_lower)
            | (candidate_property_upper < property_upper)
        ).astype(np.int64, copy=False)
        if improved_target.size == 0 and improved_property.size == 0:
            base_receipt.update(
                {
                    "attempted_query_blocks": int(len(descriptors)),
                    "attempted_descriptor_sha256": _canonical_sha256(
                        descriptor_records
                    ),
                    "attempted_alpha_sha256": _canonical_sha256(
                        alpha_hashes
                    ),
                }
            )
            return _fallback_result(
                status="no_improvement_fallback",
                started=started,
                receipt=base_receipt,
                target_bounds=target_fallback,
                property_lower=property_lower,
                property_upper=property_upper,
                timings=timings,
                completed_blocks_discarded=completed_blocks,
            )

        target_shape = tuple(
            int(value) for value in target_fallback.lb.shape
        )
        candidate_target = Bounds(
            lb=torch.from_numpy(candidate_lb.copy())
            .to(dtype=torch.float64)
            .reshape(target_shape),
            ub=torch.from_numpy(candidate_ub.copy())
            .to(dtype=torch.float64)
            .reshape(target_shape),
        )
        complete_receipt = {
            **base_receipt,
            "status": "generated",
            "candidate_generated": True,
            "whole_batch_complete": True,
            "caller_bounds_unchanged": True,
            "query_blocks": int(len(descriptors)),
            "alpha_trees": int(len(stored_alphas)),
            "descriptor_records": descriptor_records,
            "descriptor_records_sha256": _canonical_sha256(
                descriptor_records
            ),
            "alpha_hashes": alpha_hashes,
            "alpha_hashes_sha256": _canonical_sha256(alpha_hashes),
            "strict_target_improvements": int(improved_target.size),
            "strict_property_improvements": int(improved_property.size),
            "improved_target_indices": [
                int(value) for value in improved_target
            ],
            "improved_property_indices": [
                int(value) for value in improved_property
            ],
            "candidate_target_bounds_sha256": _array_sha256(
                np.stack([candidate_lb, candidate_ub])
            ),
            "candidate_property_bounds_sha256": _array_sha256(
                np.stack(
                    [candidate_property_lower, candidate_property_upper]
                )
                if candidate_property_lower.size
                else np.zeros((2, 0), dtype=np.float64)
            ),
            "optimizer_best_margins_used_as_bounds": False,
            "all_candidate_updates_replayed_with_stored_alpha": True,
            "all_bounds_replayed_with_stored_alpha": not (
                rows is not None and upper_only
            ),
            "property_lower_dual_replayed": bool(
                rows is not None and not upper_only
            ),
            "completed_blocks_discarded": 0,
            "elapsed_seconds": float(max(0.0, time.monotonic() - started)),
            "timings": [dict(value) for value in timings],
        }
        finalized = _receipt_with_sha256(complete_receipt)
        if deadline is not None and time.monotonic() >= float(deadline):
            raise _QueryDeadline("deadline crossed during receipt finalization")
        return QueryDualCandidates(
            target_bounds=candidate_target,
            property_lower=candidate_property_lower.copy(),
            property_upper=candidate_property_upper.copy(),
            query_descriptors=tuple(descriptors),
            alpha_trees=tuple(stored_alphas),
            improved_target_indices=improved_target.copy(),
            improved_property_indices=improved_property.copy(),
            timings=tuple(dict(value) for value in timings),
            receipt=finalized,
        )
    except _QueryDeadline as exc:
        return _fallback_result(
            status="deadline_fallback_frozen_bounds",
            started=started,
            receipt=base_receipt,
            target_bounds=target_fallback,
            property_lower=property_lower,
            property_upper=property_upper,
            timings=timings,
            completed_blocks_discarded=completed_blocks,
            error=exc,
        )
    except Exception as exc:
        return _fallback_result(
            status="error_fallback_frozen_bounds",
            started=started,
            receipt=base_receipt,
            target_bounds=target_fallback,
            property_lower=property_lower,
            property_upper=property_upper,
            timings=timings,
            completed_blocks_discarded=completed_blocks,
            error=exc,
        )


__all__ = [
    "QueryDescriptor",
    "QueryDualCandidates",
    "generate_query_dual_candidates",
    "query_dual_stored_alpha_sha256",
    "validate_query_dual_candidates",
    "verify_query_dual_candidates_receipt",
]
