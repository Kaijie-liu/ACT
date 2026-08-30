# ===- act/back_end/bab/bab.py - BaB Verification Engine -----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   BaB loop on a single-spec instance.  Subproblems explored in K-batched
#   waves via solve_batch; CE validation per SAT lane.  Solver-agnostic.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

import inspect
import itertools
import logging
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Union,
    cast,
)

import torch

from act.back_end.config import BaBConfig, VALID_SOLVER_TIERS
from act.back_end.bab.node import (
    BabNode,
    SubproblemBatch,
    _infer_spec_axis_size,
    concat_children,
    rederive_embedding_block_eps,
    split_input,
    split_input_nary,
    split_neuron_subproblems,
    split_subproblems,
)
from act.back_end.bab.branching.branching import (
    BranchingStrategy,
    RandomBranching,
    SplitDecision,
    _build_branching_strategy as _build_branching_strategy_impl,
    _collect_neuron_candidates,
    _multi_split_from_decision,
    _multi_split_from_groups,
    _propose_joint_split_groups,
    enumerate_unstable_candidates,
)
from act.back_end.bab.branching.bounding import (
    BoundingStrategy,
    RandomBounding,
    TopKBounding,
    DepthLowerBoundOrder,
    GreedyOrder,
    SAOrder,
)

from act.back_end.core import Bounds, Layer, Net, ParamValue
from act.back_end.solver.solver_base import BatchLPSolution, Solver, SolveStatus
from act.back_end.verifier import (
    gather_input_spec_layers,
    get_assert_layer,
    get_input_ids,
    seed_from_input_specs,
    setup_and_solve_batch,
)
from act.front_end.specs import OutKind, OutputSpec
from act.front_end.specs import InKind, normalize_position_mask
from act.util.model_inference import infer_single_model
from act.util.stats import VerifyStatus, VerifyResult

if TYPE_CHECKING:
    from act.back_end.interval_tf.tf_attention import LinearBounds

log = logging.getLogger(__name__)


@dataclass
class DualSolveResult:
    solution: BatchLPSolution
    bounds_dict: Optional[Dict[int, Bounds]] = None
    nu_per_layer: Optional[Dict[int, torch.Tensor]] = None
    row_slack: Optional[torch.Tensor] = None
    row_certified: Optional[torch.Tensor] = None
    """Per-spec-row slack ``[K, m]``.

    For ALL-row kinds, a row is certified only when its slack is finite and
    strictly above a dtype-aware numerical tolerance. Zero is deliberately
    unresolved because TOP1/MARGIN ASSERT semantics treat an exact tie as a
    concrete violation. ``row_certified`` stores the exact mask used for the
    solve verdict so root pruning cannot recompute it with a different scale.
    """
    refine_audit: Optional[Dict[str, Any]] = None
    """Non-authoritative child-local bound-refinement telemetry."""


def _strictly_certified_slack(
    slack: torch.Tensor,
    reference: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return the fail-closed per-row certification mask.

    A proof bound at exactly zero does not exclude a boundary violation for
    strict robustness properties, and a non-finite proof value carries no
    certification authority.  The positive band matches the ordinary dual
    verifier's accumulated-rounding guard.
    """

    if not slack.is_floating_point():
        raise TypeError("certified slack must use a floating-point dtype")
    scale_source = slack if reference is None else reference
    if scale_source.shape != slack.shape:
        raise ValueError(
            "certification reference must have the same shape as slack"
        )
    cert_eps = max(
        100.0 * torch.finfo(slack.dtype).eps,
        1e-11,
    )
    cert_tol = cert_eps * scale_source.abs().clamp(min=1.0)
    return (
        torch.isfinite(slack)
        & torch.isfinite(scale_source)
        & (slack > cert_tol)
    )


def _select_spec_rows(
    state: Optional[Dict[int, torch.Tensor]],
    keep_rows: torch.Tensor,
) -> Optional[Dict[int, torch.Tensor]]:
    """Slice the spec axis (dim 1 of ``[N, M, n]``) of per-layer dual state."""
    if state is None:
        return None
    return {
        lid: t.index_select(1, keep_rows.to(t.device)) if t.dim() >= 3 else t
        for lid, t in state.items()
    }


def _expand_property_forest_root(
    root: SubproblemBatch,
    original_row_ids: torch.Tensor,
) -> SubproblemBatch:
    """Duplicate one full-domain root into one immutable tree per conjunct."""

    if root.batch_size != 1:
        raise ValueError(
            "property-separable BaB requires a single-instance root"
        )
    row_ids = original_row_ids.to(
        device=root.lb.device, dtype=torch.long
    ).reshape(-1)
    n_rows = int(row_ids.numel())
    if n_rows < 1:
        raise ValueError("property forest requires at least one row")
    if bool((row_ids < 0).any().item()) or int(torch.unique(row_ids).numel()) != n_rows:
        raise ValueError(
            "property forest row ids must be nonnegative and unique"
        )

    def _expand_state(
        state: Optional[Dict[int, torch.Tensor]],
    ) -> Optional[Dict[int, torch.Tensor]]:
        if state is None:
            return None
        expanded: Dict[int, torch.Tensor] = {}
        for layer_id, tensor in state.items():
            if tensor.shape[0] != 1:
                raise ValueError(
                    "property forest root state must have batch size one"
                )
            if tensor.dim() >= 3:
                if tensor.shape[1] != n_rows:
                    raise ValueError(
                        f"property forest state layer {layer_id} has "
                        f"{tensor.shape[1]} rows, expected {n_rows}"
                    )
                expanded[layer_id] = tensor.transpose(0, 1).contiguous()
            else:
                expanded[layer_id] = tensor.repeat(
                    n_rows, *([1] * (tensor.dim() - 1))
                )
        return expanded

    def _repeat_optional(
        value: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if value.shape[0] != 1:
            raise ValueError(
                "property forest root tensor state must have batch size one"
            )
        return value.repeat(n_rows, *([1] * (value.dim() - 1)))

    return SubproblemBatch(
        lb=root.lb.repeat(n_rows, 1),
        ub=root.ub.repeat(n_rows, 1),
        depths=root.depths.repeat(n_rows),
        incremental_alpha=_expand_state(root.incremental_alpha),
        incremental_eta=_expand_state(root.incremental_eta),
        split_signs=_expand_state(root.split_signs),
        parent_margins=_repeat_optional(root.parent_margins),
        lower_bound=_repeat_optional(root.lower_bound),
        node_id=_repeat_optional(root.node_id),
        parent_id=_repeat_optional(root.parent_id),
        spec_row_ids=row_ids,
    )


_PROPERTY_FOREST_RECEIPT_SCHEMA = (
    "act.property_forest_node_conservation.v1"
)


def _is_nonnegative_int(value: object) -> bool:
    """Receipt counters are JSON integers, never bools or float lookalikes."""

    return type(value) is int and cast(int, value) >= 0


def _validate_property_forest_receipt(
    receipt: object,
    *,
    expected_row_ids: tuple[int, ...],
    expected_processed: int,
    expected_pool_remaining: int,
) -> tuple[bool, tuple[str, ...]]:
    """Independently validate a property-forest node-conservation receipt.

    The receipt is deliberately non-authoritative: certification continues to
    come from the live dual UNSAT solves.  This validator is an omission and
    duplication firewall around that proof search.  It consumes only plain
    receipt data plus independently observed terminal totals.
    """

    errors: list[str] = []
    if type(receipt) is not dict:
        return False, ("receipt_not_dict",)
    data = cast(dict[str, object], receipt)
    expected_top_keys = {
        "schema",
        "proof_authority",
        "complete",
        "root_rows",
        "root_count",
        "rows",
        "totals",
        "runtime_integrity_errors",
    }
    if set(data) != expected_top_keys:
        errors.append("top_level_schema_mismatch")
    if data.get("schema") != _PROPERTY_FOREST_RECEIPT_SCHEMA:
        errors.append("schema_mismatch")
    if data.get("proof_authority") is not False:
        errors.append("receipt_must_not_have_proof_authority")

    expected_rows = tuple(int(row_id) for row_id in expected_row_ids)
    if (
        not expected_rows
        or any(row_id < 0 for row_id in expected_rows)
        or len(set(expected_rows)) != len(expected_rows)
    ):
        errors.append("invalid_expected_root_rows")
    root_rows = data.get("root_rows")
    if (
        type(root_rows) is not list
        or any(type(row_id) is not int for row_id in cast(list[object], root_rows))
        or tuple(cast(list[int], root_rows)) != expected_rows
    ):
        errors.append("root_rows_mismatch")
    if (
        not _is_nonnegative_int(data.get("root_count"))
        or data.get("root_count") != len(expected_rows)
    ):
        errors.append("root_count_mismatch")

    runtime_errors = data.get("runtime_integrity_errors")
    if (
        type(runtime_errors) is not list
        or any(type(item) is not str for item in cast(list[object], runtime_errors))
    ):
        errors.append("runtime_integrity_errors_malformed")
    elif runtime_errors:
        errors.append("runtime_integrity_error")

    rows_raw = data.get("rows")
    expected_row_keys = {str(row_id) for row_id in expected_rows}
    if type(rows_raw) is not dict:
        errors.append("rows_not_dict")
        rows: dict[str, object] = {}
    else:
        rows = cast(dict[str, object], rows_raw)
        if set(rows) != expected_row_keys:
            errors.append("row_key_set_mismatch")

    counter_names = (
        "roots",
        "children_expected",
        "children_minted",
        "processed",
        "certified",
        "branched",
        "active_pool",
    )
    summed = {name: 0 for name in counter_names}
    summed_dropped = {"frontier_cap": 0, "max_depth": 0}
    rows_complete = True
    expected_row_schema = set(counter_names) | {
        "dropped",
        "terminal_reasons",
        "integrity_errors",
    }
    for row_id in expected_rows:
        key = str(row_id)
        item_raw = rows.get(key)
        if type(item_raw) is not dict:
            errors.append(f"row_{key}_not_dict")
            rows_complete = False
            continue
        item = cast(dict[str, object], item_raw)
        if set(item) != expected_row_schema:
            errors.append(f"row_{key}_schema_mismatch")
        counters: dict[str, int] = {}
        for name in counter_names:
            value = item.get(name)
            if not _is_nonnegative_int(value):
                errors.append(f"row_{key}_{name}_invalid")
                rows_complete = False
                counters[name] = 0
            else:
                counters[name] = cast(int, value)
                summed[name] += cast(int, value)

        dropped_raw = item.get("dropped")
        if (
            type(dropped_raw) is not dict
            or set(cast(dict[object, object], dropped_raw))
            != {"frontier_cap", "max_depth"}
        ):
            errors.append(f"row_{key}_dropped_schema_mismatch")
            rows_complete = False
            dropped = {"frontier_cap": 0, "max_depth": 0}
        else:
            dropped = {}
            for reason in ("frontier_cap", "max_depth"):
                value = cast(dict[str, object], dropped_raw).get(reason)
                if not _is_nonnegative_int(value):
                    errors.append(
                        f"row_{key}_dropped_{reason}_invalid"
                    )
                    rows_complete = False
                    dropped[reason] = 0
                else:
                    dropped[reason] = cast(int, value)
                    summed_dropped[reason] += cast(int, value)

        integrity_raw = item.get("integrity_errors")
        if (
            type(integrity_raw) is not list
            or any(
                type(entry) is not str
                for entry in cast(list[object], integrity_raw)
            )
        ):
            errors.append(f"row_{key}_integrity_errors_malformed")
            rows_complete = False
            integrity_errors: list[str] = []
        else:
            integrity_errors = cast(list[str], integrity_raw)
            if integrity_errors:
                errors.append(f"row_{key}_integrity_error")
                rows_complete = False

        terminal_raw = item.get("terminal_reasons")
        expected_terminal = {
            "certified": counters["certified"],
            "dropped_max_depth": dropped["max_depth"],
            "dropped_frontier_cap": dropped["frontier_cap"],
            "active_pool": counters["active_pool"],
        }
        if (
            type(terminal_raw) is not dict
            or set(cast(dict[object, object], terminal_raw))
            != set(expected_terminal)
            or any(
                not _is_nonnegative_int(value)
                for value in cast(dict[object, object], terminal_raw).values()
            )
            or cast(dict[str, object], terminal_raw) != expected_terminal
        ):
            errors.append(f"row_{key}_terminal_reasons_mismatch")
            rows_complete = False

        # General (also meaningful for an incomplete run) conservation:
        # frontier eviction removes an unprocessed minted node; max-depth
        # termination happens after processing it.
        if (
            counters["roots"]
            + counters["children_minted"]
            != counters["processed"]
            + counters["active_pool"]
            + dropped["frontier_cap"]
        ):
            errors.append(f"row_{key}_creation_conservation_failed")
            rows_complete = False
        if (
            counters["processed"]
            != counters["certified"]
            + counters["branched"]
            + dropped["max_depth"]
        ):
            errors.append(f"row_{key}_outcome_conservation_failed")
            rows_complete = False
        if counters["children_expected"] != counters["children_minted"]:
            errors.append(f"row_{key}_child_partition_failed")
            rows_complete = False

        # Complete SAFE requires the two stronger equalities requested by the
        # audit, exactly one root, and no unproved terminal disposition.
        row_complete = (
            counters["roots"] == 1
            and counters["processed"]
            == counters["certified"] + counters["branched"]
            and counters["roots"] + counters["children_minted"]
            == counters["processed"]
            and counters["active_pool"] == 0
            and dropped["frontier_cap"] == 0
            and dropped["max_depth"] == 0
            and not integrity_errors
        )
        if not row_complete:
            errors.append(f"row_{key}_incomplete")
            rows_complete = False

    totals_raw = data.get("totals")
    expected_totals_schema = set(counter_names) | {"dropped"}
    if (
        type(totals_raw) is not dict
        or set(cast(dict[object, object], totals_raw))
        != expected_totals_schema
    ):
        errors.append("totals_schema_mismatch")
        totals: dict[str, object] = {}
    else:
        totals = cast(dict[str, object], totals_raw)
    for name in counter_names:
        value = totals.get(name)
        if not _is_nonnegative_int(value) or value != summed[name]:
            errors.append(f"total_{name}_mismatch")
    totals_dropped_raw = totals.get("dropped")
    if (
        type(totals_dropped_raw) is not dict
        or cast(dict[str, object], totals_dropped_raw) != summed_dropped
        or any(
            not _is_nonnegative_int(value)
            for value in (
                cast(dict[object, object], totals_dropped_raw).values()
                if type(totals_dropped_raw) is dict
                else ()
            )
        )
    ):
        errors.append("total_dropped_mismatch")

    if summed["processed"] != expected_processed:
        errors.append("processed_total_mismatch")
    if summed["active_pool"] != expected_pool_remaining:
        errors.append("active_pool_total_mismatch")
    if (
        summed["roots"] + summed["children_minted"]
        != summed["processed"]
    ):
        errors.append("global_creation_conservation_failed")
        rows_complete = False
    if (
        summed["processed"]
        != summed["certified"] + summed["branched"]
    ):
        errors.append("global_outcome_conservation_failed")
        rows_complete = False
    if summed["roots"] != len(expected_rows):
        errors.append("global_root_partition_failed")
        rows_complete = False
    if (
        summed["active_pool"] != 0
        or summed_dropped["frontier_cap"] != 0
        or summed_dropped["max_depth"] != 0
    ):
        errors.append("global_unproved_terminal_nodes")
        rows_complete = False

    declared_complete = data.get("complete")
    if type(declared_complete) is not bool:
        errors.append("complete_flag_invalid")
    elif declared_complete is not rows_complete:
        errors.append("complete_flag_mismatch")
    return not errors and rows_complete, tuple(sorted(set(errors)))


def _build_property_forest_receipt(
    *,
    row_ids: tuple[int, ...],
    counters: dict[str, Dict[int, int]],
    dropped: dict[str, Dict[int, int]],
    integrity_errors_by_row: Dict[int, list[str]],
    runtime_integrity_errors: list[str],
    processed: int,
    pool_remaining: int,
) -> dict[str, object]:
    """Serialize counters without granting the serialization proof authority."""

    counter_names = (
        "roots",
        "children_expected",
        "children_minted",
        "processed",
        "certified",
        "branched",
        "active_pool",
    )
    rows: dict[str, object] = {}
    provisional_complete = (
        bool(row_ids)
        and not runtime_integrity_errors
        and int(pool_remaining) == 0
    )
    for row_id in row_ids:
        values = {
            name: int(counters[name].get(row_id, 0))
            for name in counter_names
        }
        row_dropped = {
            reason: int(dropped[reason].get(row_id, 0))
            for reason in ("frontier_cap", "max_depth")
        }
        row_errors = sorted(
            set(integrity_errors_by_row.get(row_id, []))
        )
        row_complete = (
            values["roots"] == 1
            and values["children_expected"]
            == values["children_minted"]
            and values["processed"]
            == values["certified"] + values["branched"]
            and values["roots"] + values["children_minted"]
            == values["processed"]
            and values["active_pool"] == 0
            and row_dropped["frontier_cap"] == 0
            and row_dropped["max_depth"] == 0
            and not row_errors
        )
        provisional_complete = provisional_complete and row_complete
        rows[str(row_id)] = {
            **values,
            "dropped": row_dropped,
            "terminal_reasons": {
                "certified": values["certified"],
                "dropped_max_depth": row_dropped["max_depth"],
                "dropped_frontier_cap": row_dropped["frontier_cap"],
                "active_pool": values["active_pool"],
            },
            "integrity_errors": row_errors,
        }

    totals = {
        name: sum(
            int(counters[name].get(row_id, 0)) for row_id in row_ids
        )
        for name in counter_names
    }
    totals["dropped"] = {
        reason: sum(
            int(dropped[reason].get(row_id, 0)) for row_id in row_ids
        )
        for reason in ("frontier_cap", "max_depth")
    }
    provisional_complete = bool(
        provisional_complete
        and totals["processed"] == int(processed)
        and totals["active_pool"] == int(pool_remaining)
        and totals["roots"] == len(row_ids)
        and totals["roots"] + totals["children_minted"]
        == totals["processed"]
        and totals["processed"]
        == totals["certified"] + totals["branched"]
    )
    return {
        "schema": _PROPERTY_FOREST_RECEIPT_SCHEMA,
        "proof_authority": False,
        "complete": provisional_complete,
        "root_rows": list(row_ids),
        "root_count": len(row_ids),
        "rows": rows,
        "totals": totals,
        "runtime_integrity_errors": sorted(
            set(runtime_integrity_errors)
        ),
    }


def _validate_property_forest_child_partition(
    parent: SubproblemBatch,
    children: SubproblemBatch,
    parent_index: torch.Tensor,
    *,
    expected_children_per_parent: int,
) -> tuple[bool, tuple[str, ...]]:
    """Validate the live geometric/phase partition before accepting children.

    Node counts alone cannot distinguish a complete split from replacing one
    child with a duplicate of its sibling.  Property-forest promotion therefore
    checks the actual proof domains at the split boundary:

    * input children must form one exact, contiguous axis partition of their
      parent box and inherit every phase choice unchanged;
    * phase children must keep the parent box, preserve all existing choices,
      and enumerate one complete ``{-1,+1}^k`` cube of newly fixed ReLUs.

    The check is deliberately exact on stored tensors.  All split constructors
    create shared endpoints and copied state from the same tensor operations,
    so accepting a tolerance here would only enlarge the tamper surface.
    """

    errors: list[str] = []
    expected = int(expected_children_per_parent)
    if expected < 2:
        return False, ("expected_child_count_below_two",)
    if (
        parent_index.ndim != 1
        or parent_index.dtype != torch.long
        or int(parent_index.numel()) != children.batch_size
    ):
        return False, ("malformed_parent_index",)
    if (
        bool((parent_index < 0).any().item())
        or bool((parent_index >= parent.batch_size).any().item())
    ):
        return False, ("parent_index_out_of_range",)
    if (
        parent.lb.ndim != 2
        or parent.ub.shape != parent.lb.shape
        or children.lb.ndim != 2
        or children.ub.shape != children.lb.shape
        or children.lb.shape[1:] != parent.lb.shape[1:]
    ):
        return False, ("box_shape_mismatch",)

    def _lane_signs(
        state: Optional[Dict[int, torch.Tensor]],
        lane: int,
    ) -> dict[int, torch.Tensor]:
        if state is None:
            return {}
        return {
            int(layer_id): tensor[lane].detach()
            for layer_id, tensor in state.items()
        }

    def _input_signs_unchanged(
        parent_lane: int,
        child_indices: torch.Tensor,
    ) -> bool:
        parent_signs = _lane_signs(parent.split_signs, parent_lane)
        child_layers = (
            set() if children.split_signs is None
            else {int(layer_id) for layer_id in children.split_signs}
        )
        if child_layers != set(parent_signs):
            return False
        for child_index in child_indices.tolist():
            child_signs = _lane_signs(
                children.split_signs, int(child_index)
            )
            if any(
                not torch.equal(
                    child_signs[layer_id],
                    parent_signs[layer_id],
                )
                for layer_id in parent_signs
            ):
                return False
        return True

    for parent_lane in range(parent.batch_size):
        child_indices = torch.where(parent_index == parent_lane)[0]
        if int(child_indices.numel()) != expected:
            errors.append(
                f"parent_{parent_lane}_child_count:"
                f"expected={expected},actual={int(child_indices.numel())}"
            )
            continue
        parent_lb = parent.lb[parent_lane]
        parent_ub = parent.ub[parent_lane]
        child_lb = children.lb.index_select(
            0, child_indices.to(children.lb.device)
        )
        child_ub = children.ub.index_select(
            0, child_indices.to(children.ub.device)
        )
        parent_lb_rows = parent_lb.unsqueeze(0).expand_as(child_lb)
        parent_ub_rows = parent_ub.unsqueeze(0).expand_as(child_ub)
        boxes_unchanged = bool(
            torch.equal(child_lb, parent_lb_rows)
            and torch.equal(child_ub, parent_ub_rows)
        )

        depth_delta = int(math.ceil(math.log2(expected)))
        expected_depth = parent.depths[parent_lane] + depth_delta
        if not torch.equal(
            children.depths.index_select(
                0, child_indices.to(children.depths.device)
            ),
            expected_depth.expand(int(child_indices.numel())),
        ):
            errors.append(f"parent_{parent_lane}_depth_partition")

        if not boxes_unchanged:
            if not _input_signs_unchanged(parent_lane, child_indices):
                errors.append(
                    f"parent_{parent_lane}_input_split_changed_phase"
                )
                continue
            if not bool(
                torch.isfinite(parent_lb).all().item()
                and torch.isfinite(parent_ub).all().item()
                and torch.isfinite(child_lb).all().item()
                and torch.isfinite(child_ub).all().item()
            ):
                errors.append(
                    f"parent_{parent_lane}_nonfinite_input_partition"
                )
                continue
            if bool((parent_lb > parent_ub).any().item()) or bool(
                (child_lb > child_ub).any().item()
            ):
                errors.append(
                    f"parent_{parent_lane}_invalid_input_box"
                )
                continue
            changed = (
                (child_lb != parent_lb_rows)
                | (child_ub != parent_ub_rows)
            ).any(dim=0)
            changed_dims = torch.where(changed)[0]
            if int(changed_dims.numel()) != 1:
                errors.append(
                    f"parent_{parent_lane}_input_partition_axes"
                )
                continue
            split_dim = int(changed_dims[0].item())
            other = torch.ones(
                parent_lb.numel(),
                dtype=torch.bool,
                device=child_lb.device,
            )
            other[split_dim] = False
            if (
                not torch.equal(
                    child_lb[:, other], parent_lb_rows[:, other]
                )
                or not torch.equal(
                    child_ub[:, other], parent_ub_rows[:, other]
                )
            ):
                errors.append(
                    f"parent_{parent_lane}_input_partition_other_axes"
                )
                continue
            intervals = sorted(
                (
                    float(child_lb[index, split_dim].item()),
                    float(child_ub[index, split_dim].item()),
                )
                for index in range(int(child_indices.numel()))
            )
            if (
                intervals[0][0] != float(parent_lb[split_dim].item())
                or intervals[-1][1] != float(parent_ub[split_dim].item())
                or any(lower >= upper for lower, upper in intervals)
                or any(
                    intervals[index][1] != intervals[index + 1][0]
                    for index in range(len(intervals) - 1)
                )
            ):
                errors.append(
                    f"parent_{parent_lane}_input_partition_not_exact"
                )
            continue

        # An unchanged box must be a complete ReLU phase cube.  A power-of-two
        # child count is necessary, and the number of newly fixed coordinates
        # must be log2(child_count).
        if expected & (expected - 1):
            errors.append(
                f"parent_{parent_lane}_phase_count_not_power_of_two"
            )
            continue
        phase_depth = int(math.log2(expected))
        parent_signs = _lane_signs(parent.split_signs, parent_lane)
        child_layers = (
            set() if children.split_signs is None
            else {int(layer_id) for layer_id in children.split_signs}
        )
        if not set(parent_signs).issubset(child_layers):
            errors.append(
                f"parent_{parent_lane}_phase_state_removed"
            )
            continue
        layer_shapes: dict[int, torch.Size] = {}
        malformed_state = False
        for layer_id in child_layers:
            child_state = cast(
                Dict[int, torch.Tensor], children.split_signs
            )[layer_id]
            if (
                child_state.shape[0] != children.batch_size
                or child_state.ndim < 2
            ):
                malformed_state = True
                break
            lane_shape = child_state.shape[1:]
            layer_shapes[layer_id] = lane_shape
            if (
                layer_id in parent_signs
                and parent_signs[layer_id].shape != lane_shape
            ):
                malformed_state = True
                break
        if malformed_state:
            errors.append(
                f"parent_{parent_lane}_phase_state_shape"
            )
            continue

        changed_coordinates: Optional[tuple[tuple[int, int], ...]] = None
        assignments: list[tuple[int, ...]] = []
        for raw_child_index in child_indices.tolist():
            child_index = int(raw_child_index)
            child_changed: list[tuple[int, int]] = []
            child_assignment: list[int] = []
            for layer_id in sorted(child_layers):
                child_lane = cast(
                    Dict[int, torch.Tensor], children.split_signs
                )[layer_id][child_index].reshape(-1)
                parent_lane_state = (
                    parent_signs[layer_id].reshape(-1)
                    if layer_id in parent_signs
                    else torch.zeros_like(child_lane)
                )
                if not bool(
                    torch.isfinite(child_lane).all().item()
                    and torch.isfinite(parent_lane_state).all().item()
                ):
                    malformed_state = True
                    break
                inherited = parent_lane_state != 0
                if not torch.equal(
                    child_lane[inherited],
                    parent_lane_state[inherited],
                ):
                    malformed_state = True
                    break
                newly_fixed = (~inherited) & (child_lane != 0)
                if bool(
                    (
                        (child_lane[newly_fixed] != 1)
                        & (child_lane[newly_fixed] != -1)
                    ).any().item()
                ):
                    malformed_state = True
                    break
                for coordinate in torch.where(newly_fixed)[0].tolist():
                    child_changed.append((layer_id, int(coordinate)))
                    child_assignment.append(
                        int(child_lane[int(coordinate)].item())
                    )
            if malformed_state:
                break
            zipped = sorted(zip(child_changed, child_assignment))
            coordinates = tuple(item[0] for item in zipped)
            assignment = tuple(item[1] for item in zipped)
            if changed_coordinates is None:
                changed_coordinates = coordinates
            elif coordinates != changed_coordinates:
                malformed_state = True
                break
            assignments.append(assignment)
        if malformed_state:
            errors.append(
                f"parent_{parent_lane}_phase_assignment_malformed"
            )
            continue
        if (
            changed_coordinates is None
            or len(changed_coordinates) != phase_depth
            or len(set(assignments)) != expected
            or set(assignments)
            != set(
                itertools.product((-1, 1), repeat=phase_depth)
            )
        ):
            errors.append(
                f"parent_{parent_lane}_phase_cube_incomplete"
            )

    return not errors, tuple(sorted(set(errors)))


def _presplit_root(
    root: SubproblemBatch,
    bounds_dict: Dict[int, Bounds],
    nu_per_layer: Dict[int, torch.Tensor],
    k: int,
) -> Optional[SubproblemBatch]:
    """Materialize the 2^k descendants of the root's top-k scored neurons.

    Score = triangle relaxation area x |nu| (BaBSR essence); the layer with
    the strongest top score wins. The 2^k sign assignments exactly partition
    the root region (each unstable neuron is either >=0 or <=0), so replacing
    the root by these children is sound.
    """
    best: Optional[tuple[int, torch.Tensor, torch.Tensor]] = None
    for lid, nu in nu_per_layer.items():
        b = bounds_dict.get(lid)
        if b is None:
            continue
        lb = b.lb.flatten(start_dim=1)[0]
        ub = b.ub.flatten(start_dim=1)[0]
        n = min(lb.shape[-1], nu.shape[-1])
        lb, ub = lb[:n], ub[:n]
        amb = (lb < 0) & (ub > 0)
        if not bool(amb.any().item()):
            continue
        area = (-lb * ub / (ub - lb).clamp(min=1e-12)).clamp(min=0)
        score = area * nu.reshape(-1, nu.shape[-1])[:, :n].abs().sum(dim=0)
        score = torch.where(amb, score, torch.zeros_like(score))
        if best is None or float(score.max()) > float(best[1].max()):
            best = (lid, score, lb)
    if best is None:
        return None
    lid, score, _ = best
    k = min(k, int((score > 0).sum().item()))
    if k < 1:
        return None
    top_idx = torch.topk(score, k=k).indices
    n_children = 2 ** k
    n_layer = score.shape[-1]
    m = _infer_spec_axis_size(root)

    signs = torch.zeros(n_children, m, n_layer, device=root.lb.device, dtype=root.lb.dtype)
    for j in range(n_children):
        for bit, neuron in enumerate(top_idx.tolist()):
            signs[j, :, neuron] = 1.0 if (j >> bit) & 1 else -1.0

    def _rep(state: Optional[Dict[int, torch.Tensor]]) -> Optional[Dict[int, torch.Tensor]]:
        if state is None:
            return None
        return {l: t.repeat(n_children, *([1] * (t.dim() - 1))) for l, t in state.items()}

    merged_signs = _rep(root.split_signs) or {}
    if lid in merged_signs:
        merged_signs[lid] = merged_signs[lid] + signs
    else:
        merged_signs[lid] = signs
    return SubproblemBatch(
        lb=root.lb.repeat(n_children, 1),
        ub=root.ub.repeat(n_children, 1),
        depths=torch.full((n_children,), k, dtype=torch.long, device=root.lb.device),
        incremental_alpha=_rep(root.incremental_alpha),
        incremental_eta=_rep(root.incremental_eta),
        split_signs=merged_signs,
        spec_row_ids=(
            root.spec_row_ids.repeat(n_children)
            if root.spec_row_ids is not None
            else None
        ),
    )


def _interval_refresh_bounds(
    net: Net,
    base: Dict[int, Bounds],
    split_signs: Dict[int, torch.Tensor],
) -> Optional[Dict[int, Bounds]]:
    """Cheap batched IBP re-propagation of split phases, intersected with base.

    Frozen root bounds lose the downstream effect of hardened splits; a plain
    interval pass (no LinearBound A matrices, milliseconds per batch) restores
    it. Every entry is intersected with the base bounds, so the result can
    only tighten valid over-approximations (sound). Returns None when an
    unsupported layer kind is encountered - the caller keeps the base dict.
    """
    from act.back_end.dual_tf.tf_forward import _fwd_conv2d_interval

    out = dict(base)
    vals: Dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in net.layers:
        k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
        lid = layer.id
        if k == "ASSERT":
            continue
        if k in ("INPUT", "INPUT_SPEC"):
            b = out.get(lid)
            if b is None:
                return None
            vals[lid] = (b.lb.flatten(start_dim=1), b.ub.flatten(start_dim=1))
            continue
        preds = net.preds.get(lid, [])
        try:
            if k == "CONV2D":
                plb, pub = vals[preds[0]]
                lb, ub = _fwd_conv2d_interval(layer, plb, pub)
                lb, ub = lb.flatten(start_dim=1), ub.flatten(start_dim=1)
            elif k == "DENSE":
                w = layer.params["weight"]
                bias = layer.params.get("bias")
                if not isinstance(w, torch.Tensor):
                    return None
                plb, pub = vals[preds[0]]
                w_pos, w_neg = w.clamp(min=0), w.clamp(max=0)
                lb = plb @ w_pos.T + pub @ w_neg.T
                ub = pub @ w_pos.T + plb @ w_neg.T
                if isinstance(bias, torch.Tensor):
                    lb, ub = lb + bias, ub + bias
            elif k == "ADD":
                (alb, aub), (blb, bub) = vals[preds[0]], vals[preds[1]]
                lb, ub = alb + blb, aub + bub
            elif k in ("FLATTEN", "RESHAPE"):
                lb, ub = vals[preds[0]]
            elif k == "RELU":
                lb, ub = vals[preds[0]]
            else:
                return None
        except (KeyError, IndexError, ValueError):
            return None

        b = out.get(lid)
        if b is not None:
            lb = torch.maximum(lb, b.lb.flatten(start_dim=1))
            ub = torch.minimum(ub, b.ub.flatten(start_dim=1))
            ub = torch.maximum(ub, lb)
        if k == "RELU":
            s = split_signs.get(lid)
            if s is not None:
                sl = s[:, 0, :] if s.dim() == 3 else s
                n = min(lb.shape[-1], sl.shape[-1])
                sn = sl[..., :n].to(lb.device)
                lb, ub = lb.clone(), ub.clone()
                lb[..., :n] = torch.where(sn > 0, lb[..., :n].clamp(min=0.0), lb[..., :n])
                ub[..., :n] = torch.where(sn < 0, ub[..., :n].clamp(max=0.0), ub[..., :n])
                ub[..., :n] = torch.maximum(ub[..., :n], lb[..., :n])
        if b is not None:
            out[lid] = Bounds(lb.view_as(b.lb).clone(), ub.view_as(b.ub).clone())
        if k == "RELU":
            vals[lid] = (lb.clamp(min=0.0), ub.clamp(min=0.0))
        else:
            vals[lid] = (lb, ub)
    return out


def _want_babsr_neuron_branching(config: BaBConfig) -> bool:
    return (
        getattr(config, "branching_method", "random") in ("babsr", "fsb", "gain")
        and getattr(config, "solver_tier", "lp") in ("dual_alpha", "dual_alpha_eta")
    )


def _branch_layers_with_unstable_successors(
    net: Net,
    bounds_dict: Optional[Dict[int, Bounds]],
    nu_per_layer: Optional[Dict[int, torch.Tensor]],
) -> set[int]:
    """ReLU proposal layers that can affect another unstable ReLU relaxation."""

    if bounds_dict is None or nu_per_layer is None:
        return set()
    unstable: set[int] = set()
    for layer in net.layers:
        kind = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
        bounds = bounds_dict.get(layer.id)
        if kind != "RELU" or bounds is None or layer.id not in nu_per_layer:
            continue
        lower = bounds.lb.flatten(start_dim=1)
        upper = bounds.ub.flatten(start_dim=1)
        if bool(((lower < 0) & (upper > 0)).any().item()):
            unstable.add(int(layer.id))
    eligible: set[int] = set()
    for source in unstable:
        seen: set[int] = set()
        work = list(net.succs.get(source, []))
        while work:
            layer_id = int(work.pop())
            if layer_id in seen:
                continue
            seen.add(layer_id)
            if layer_id in unstable:
                eligible.add(source)
                break
            work.extend(net.succs.get(layer_id, []))
    return eligible


def _filter_branching_state_to_unstable_successors(
    branch_batch: SubproblemBatch,
    net: Net,
    bounds_dict: Optional[Dict[int, Bounds]],
    nu_per_layer: Optional[Dict[int, torch.Tensor]],
) -> tuple[
    Optional[Dict[int, Bounds]],
    Optional[Dict[int, torch.Tensor]],
    set[int],
    bool,
]:
    """Apply the long-horizon layer filter only when every lane can branch.

    The filter is proposal-only: it cannot remove any proof obligation.
    Applying a batch-wide layer subset when even one lane has no remaining
    finite candidate would, however, let later ``topk`` code pick ``-inf``
    entries.  In that case the complete batch fails safe to its original
    branching state.
    """

    eligible = _branch_layers_with_unstable_successors(
        net, bounds_dict, nu_per_layer
    )
    if not eligible or bounds_dict is None or nu_per_layer is None:
        return bounds_dict, nu_per_layer, eligible, False
    filtered_bounds = {
        layer_id: bounds
        for layer_id, bounds in bounds_dict.items()
        if layer_id in eligible
    }
    filtered_nu = {
        layer_id: nu
        for layer_id, nu in nu_per_layer.items()
        if layer_id in eligible
    }
    candidates = _collect_neuron_candidates(
        branch_batch, filtered_bounds, filtered_nu
    )
    if candidates is None:
        return bounds_dict, nu_per_layer, eligible, False
    finite_per_lane = torch.isfinite(candidates[0]).sum(dim=1)
    if bool((finite_per_lane < 1).any().item()):
        return bounds_dict, nu_per_layer, eligible, False
    return filtered_bounds, filtered_nu, eligible, True


def _survival_controlled_split_depth(
    max_levels: int,
    survivor_rate: float,
    target_multiplier: float,
) -> int:
    """Largest full phase-split depth predicted not to grow the frontier."""

    if max_levels < 1:
        raise ValueError("max_levels must be positive")
    if not 0.0 <= survivor_rate <= 1.0:
        raise ValueError("survivor_rate must be in [0, 1]")
    if not 0.0 < target_multiplier <= 1.0:
        raise ValueError("target_multiplier must be in (0, 1]")
    for levels in range(int(max_levels), 0, -1):
        if (2 ** levels) * float(survivor_rate) <= target_multiplier:
            return levels
    return 1


def _gain_tested_decision(
    branch_batch: SubproblemBatch,
    net: Net,
    assert_layer: Layer,
    config: BaBConfig,
    keep_rows: Optional[torch.Tensor],
    root_bounds_dict: Optional[Dict[int, Bounds]],
    bounds_dict: Optional[Dict[int, Bounds]],
    nu_per_layer: Optional[Dict[int, torch.Tensor]],
    input_shape: tuple[int, ...],
    n_candidates: int = 3,
    spec_row_index: Optional[torch.Tensor] = None,
) -> Optional[SplitDecision]:
    """Pick each lane's split by measured child bounds, not by score proxy.

    BaBSR scores can rank a regression (-0.07) above the true best split
    (+0.07); evaluating the top candidates' actual children with one cheap
    non-optimized dual batch restores monotone progress (kfsb-style).
    """
    if bounds_dict is None or nu_per_layer is None:
        return None
    kb = branch_batch.batch_size
    device = branch_batch.lb.device

    cand = _collect_neuron_candidates(
        branch_batch,
        bounds_dict,
        nu_per_layer,
        spec_row_index=spec_row_index,
    )
    if cand is None:
        return None
    all_scores, all_layers, all_neurons = cand
    finite_per_lane = torch.isfinite(all_scores).sum(dim=1)
    n_c = min(n_candidates, int(finite_per_lane.min().item()))
    if n_c < 1:
        return None
    top = torch.topk(all_scores, k=n_c, dim=1).indices
    top_layers = all_layers.gather(1, top)
    top_neurons = all_neurons.gather(1, top)

    rep_idx = torch.arange(kb, device=device).repeat_interleave(2 * n_c)

    def _rep_state(state):
        if state is None:
            return None
        return {l: t.index_select(0, rep_idx.to(t.device)) for l, t in state.items()}

    m_specs = _infer_spec_axis_size(branch_batch)
    signs = _rep_state(branch_batch.split_signs) or {}
    for lid_val in torch.unique(top_layers).tolist():
        lid_int = int(lid_val)
        layer = net.by_id[lid_int]
        n_neurons = int(layer.out_vars[-1] - layer.out_vars[0] + 1)
        if lid_int not in signs:
            signs[lid_int] = torch.zeros(
                2 * n_c * kb, m_specs, n_neurons, device=device, dtype=branch_batch.lb.dtype,
            )
        else:
            signs[lid_int] = signs[lid_int].clone()
        for lane in range(kb):
            for c in range(n_c):
                if int(top_layers[lane, c]) != lid_int:
                    continue
                row = lane * 2 * n_c + 2 * c
                neuron = int(top_neurons[lane, c])
                signs[lid_int][row, :, neuron] = 1.0
                signs[lid_int][row + 1, :, neuron] = -1.0

    probe = SubproblemBatch(
        lb=branch_batch.lb.index_select(0, rep_idx),
        ub=branch_batch.ub.index_select(0, rep_idx),
        depths=branch_batch.depths.index_select(0, rep_idx),
        incremental_alpha=_rep_state(branch_batch.incremental_alpha),
        incremental_eta=_rep_state(branch_batch.incremental_eta),
        split_signs=signs,
        spec_row_ids=(
            branch_batch.spec_row_ids.index_select(
                0, rep_idx.to(branch_batch.spec_row_ids.device)
            )
            if branch_batch.spec_row_ids is not None
            else None
        ),
    )
    n_probe = probe.batch_size
    probe_bounds = Bounds(
        probe.lb.reshape(n_probe, *input_shape) if input_shape else probe.lb,
        probe.ub.reshape(n_probe, *input_shape) if input_shape else probe.ub,
    )
    res = _dispatch_dual_solve(
        net=net,
        assert_layer=assert_layer,
        batched_bounds=probe_bounds,
        k_actual=n_probe,
        batch=probe,
        config=config,
        optimize=False,
        keep_rows=keep_rows,
        root_bounds_dict=root_bounds_dict,
    )
    child_lbs = (-res.solution.max_viol).view(kb, n_c, 2)
    pair_gain = child_lbs.min(dim=2).values
    best_c = pair_gain.argmax(dim=1)
    lane_idx = torch.arange(kb, device=device)
    return SplitDecision(
        kind="neuron",
        layer_id=top_layers[lane_idx, best_c],
        neuron_idx=top_neurons[lane_idx, best_c],
    )


def _gain_tested_multi_split(
    branch_batch: SubproblemBatch,
    net: Net,
    assert_layer: Layer,
    config: BaBConfig,
    keep_rows: Optional[torch.Tensor],
    root_bounds_dict: Optional[Dict[int, Bounds]],
    bounds_dict: Optional[Dict[int, Bounds]],
    nu_per_layer: Optional[Dict[int, torch.Tensor]],
    input_shape: tuple[int, ...],
    *,
    k_levels: int,
    max_groups: int,
    max_probe_batch: int,
    audit: Optional[Dict[str, Any]] = None,
    spec_row_index: Optional[torch.Tensor] = None,
) -> Optional[tuple[SubproblemBatch, torch.Tensor]]:
    """Measure several joint groups through their complete child partitions.

    Candidate scores only choose a small finite proposal pool.  Each proposed
    group is expanded into every one of its ``2^k`` sign combinations and all
    groups are evaluated in one non-optimizing dual batch.  The group with the
    largest worst-child lower bound is selected independently for every lane.
    This affects search order only; ``_multi_split_from_groups`` still creates
    the complete selected partition used by the proof search.
    """

    if (
        bounds_dict is None
        or nu_per_layer is None
        or k_levels < 2
        or max_groups < 2
    ):
        return None
    candidates = _collect_neuron_candidates(
        branch_batch,
        bounds_dict,
        nu_per_layer,
        spec_row_index=spec_row_index,
    )
    if candidates is None:
        return None
    all_scores, all_layers, all_neurons = candidates
    children_per_group = 2 ** int(k_levels)
    group_cap = min(
        int(max_groups),
        int(max_probe_batch)
        // max(1, int(branch_batch.batch_size) * children_per_group),
    )
    if group_cap < 2:
        return None
    proposed = _propose_joint_split_groups(
        all_scores,
        all_layers,
        all_neurons,
        k_levels=int(k_levels),
        max_groups=group_cap,
    )
    if proposed is None:
        return None
    group_layers, group_neurons = proposed
    n_lanes, n_groups, _ = group_layers.shape
    if n_groups < 2:
        return None
    device = branch_batch.lb.device
    parent_index = torch.arange(n_lanes, device=device).repeat(
        n_groups * children_per_group
    )

    def _gather(
        state: Optional[Dict[int, torch.Tensor]],
    ) -> Optional[Dict[int, torch.Tensor]]:
        if state is None:
            return None
        return {
            layer_id: tensor.index_select(
                0, parent_index.to(tensor.device)
            )
            for layer_id, tensor in state.items()
        }

    m_specs = _infer_spec_axis_size(branch_batch)
    signs = _gather(branch_batch.split_signs) or {}
    for layer_value in torch.unique(group_layers).tolist():
        layer_id = int(layer_value)
        layer = net.by_id[layer_id]
        n_neurons = int(layer.out_vars[-1] - layer.out_vars[0] + 1)
        if layer_id not in signs:
            signs[layer_id] = torch.zeros(
                n_groups * children_per_group * n_lanes,
                m_specs,
                n_neurons,
                device=device,
                dtype=branch_batch.lb.dtype,
            )
        else:
            signs[layer_id] = signs[layer_id].clone()
        for group_index in range(n_groups):
            for bit in range(k_levels):
                lanes = torch.where(
                    group_layers[:, group_index, bit] == layer_value
                )[0]
                if int(lanes.numel()) == 0:
                    continue
                neurons = group_neurons[
                    lanes, group_index, bit
                ].to(device=device, dtype=torch.long)
                for assignment in range(children_per_group):
                    sign = 1.0 if (assignment >> bit) & 1 else -1.0
                    rows = (
                        (group_index * children_per_group + assignment)
                        * n_lanes
                        + lanes
                    )
                    signs[layer_id][rows, :, neurons] = sign

    probe = SubproblemBatch(
        lb=branch_batch.lb.index_select(0, parent_index),
        ub=branch_batch.ub.index_select(0, parent_index),
        depths=branch_batch.depths.index_select(0, parent_index),
        incremental_alpha=_gather(branch_batch.incremental_alpha),
        incremental_eta=_gather(branch_batch.incremental_eta),
        split_signs=signs,
        spec_row_ids=(
            branch_batch.spec_row_ids.index_select(
                0, parent_index.to(branch_batch.spec_row_ids.device)
            )
            if branch_batch.spec_row_ids is not None
            else None
        ),
    )
    probe_count = int(probe.batch_size)
    probe_bounds = Bounds(
        (
            probe.lb.reshape(probe_count, *input_shape)
            if input_shape
            else probe.lb
        ),
        (
            probe.ub.reshape(probe_count, *input_shape)
            if input_shape
            else probe.ub
        ),
    )
    result = _dispatch_dual_solve(
        net=net,
        assert_layer=assert_layer,
        batched_bounds=probe_bounds,
        k_actual=probe_count,
        batch=probe,
        config=config,
        optimize=False,
        keep_rows=keep_rows,
        root_bounds_dict=root_bounds_dict,
    )
    child_lower = (-result.solution.max_viol).reshape(
        n_groups, children_per_group, n_lanes
    ).permute(2, 0, 1)
    best_group = child_lower.min(dim=2).values.argmax(dim=1)
    lane_index = torch.arange(n_lanes, device=device)
    selected_layers = group_layers[lane_index, best_group]
    selected_neurons = group_neurons[lane_index, best_group]
    if audit is not None:
        baseline_diversity = torch.tensor(
            [
                len(set(group_layers[lane, 0].tolist()))
                for lane in range(n_lanes)
            ],
            device=device,
            dtype=torch.long,
        )
        selected_diversity = torch.tensor(
            [
                len(set(selected_layers[lane].tolist()))
                for lane in range(n_lanes)
            ],
            device=device,
            dtype=torch.long,
        )
        audit.update(
            {
                "lanes": int(n_lanes),
                "groups": int(n_groups),
                "k_levels": int(k_levels),
                "probe_nodes": int(probe_count),
                "selected_nonbaseline_lanes": int(
                    torch.count_nonzero(best_group != 0).item()
                ),
                "selected_more_diverse_lanes": int(
                    torch.count_nonzero(
                        selected_diversity > baseline_diversity
                    ).item()
                ),
                "selected_group_ids": [
                    int(value) for value in best_group.tolist()
                ],
                "baseline_worst_child_lb": [
                    float(value)
                    for value in child_lower[:, 0].min(dim=1).values.tolist()
                ],
                "selected_worst_child_lb": [
                    float(value)
                    for value in child_lower[
                        lane_index, best_group
                    ].min(dim=1).values.tolist()
                ],
            }
        )
    return _multi_split_from_groups(
        branch_batch,
        net,
        selected_layers,
        selected_neurons,
        int(k_levels),
    )


def _input_axis_decision_tensor(
    decision: SplitDecision,
    batch: SubproblemBatch,
) -> torch.Tensor:
    if decision.input_axis is None:
        raise ValueError("input-axis decision missing input_axis")
    input_axis = torch.as_tensor(decision.input_axis, device=batch.lb.device, dtype=torch.long).reshape(-1)
    if input_axis.numel() == 1:
        input_axis = input_axis.expand(batch.batch_size)
    if input_axis.numel() != batch.batch_size:
        raise ValueError(
            f"input-axis decision has {input_axis.numel()} lanes for batch size {batch.batch_size}"
        )
    return input_axis.contiguous()


def _split_from_decision(
    batch: SubproblemBatch,
    decision: SplitDecision,
    net: Net,
) -> tuple[SubproblemBatch, torch.Tensor]:
    fanout = max(2, int(getattr(decision, "fanout", 2)))
    if decision.kind == "input_axis":
        dims = (
            _input_axis_decision_tensor(SplitDecision(kind="input_axis", input_axis=decision.cut_dim), batch)
            if decision.cut_dim is not None
            else _input_axis_decision_tensor(decision, batch)
        )
        if fanout == 2:
            return split_input(batch, dims)
        return split_input_nary(batch, dims, fanout)

    if decision.kind == "neuron":
        if decision.layer_id is None or decision.neuron_idx is None:
            raise ValueError("neuron decision missing layer_id or neuron_idx")

        layer_id_tensor = decision.layer_id.reshape(-1)
        neuron_idx_tensor = decision.neuron_idx.reshape(-1)
        if layer_id_tensor.numel() == 0 or neuron_idx_tensor.numel() == 0:
            raise ValueError("neuron decision tensors must be non-empty")

        rep_lid = int(layer_id_tensor[0].item())
        rep_idx = int(neuron_idx_tensor[0].item())
        if rep_lid < 0:
            fallback_dims = (batch.ub - batch.lb).argmax(dim=-1)
            if fanout == 2:
                return split_input(batch, fallback_dims)
            return split_input_nary(batch, fallback_dims, fanout)

        n_lanes = batch.batch_size
        lids = layer_id_tensor.expand(n_lanes) if layer_id_tensor.numel() == 1 else layer_id_tensor
        idxs = neuron_idx_tensor.expand(n_lanes) if neuron_idx_tensor.numel() == 1 else neuron_idx_tensor
        if lids.numel() != n_lanes or idxs.numel() != n_lanes:
            raise ValueError(
                f"neuron decision has {lids.numel()}/{idxs.numel()} entries "
                f"for batch size {n_lanes}"
            )

        # Per-lane split: lane i hardens ITS OWN (layer, neuron); collapsing
        # to lane 0's choice makes the other K-1 lanes split an irrelevant
        # neuron and stalls deep convergence.
        device = batch.lb.device
        parent_index = torch.arange(n_lanes, device=device).repeat(2)

        def _gather(state: Optional[Dict[int, torch.Tensor]]) -> Optional[Dict[int, torch.Tensor]]:
            if state is None:
                return None
            return {
                l: t.index_select(0, parent_index.to(t.device)) for l, t in state.items()
            }

        m_specs = _infer_spec_axis_size(batch)

        signs = _gather(batch.split_signs) or {}
        for lid_val in torch.unique(lids).tolist():
            lid_int = int(lid_val)
            layer = net.by_id[lid_int]
            n_neurons = int(layer.out_vars[-1] - layer.out_vars[0] + 1)
            if lid_int not in signs:
                signs[lid_int] = torch.zeros(
                    2 * n_lanes, m_specs, n_neurons, device=device, dtype=batch.lb.dtype,
                )
            else:
                signs[lid_int] = signs[lid_int].clone()
            lane_sel = torch.where(lids == lid_val)[0]
            neuron_sel = idxs[lane_sel].to(device=device, dtype=torch.long)
            signs[lid_int][lane_sel, :, neuron_sel] = 1.0
            signs[lid_int][lane_sel + n_lanes, :, neuron_sel] = -1.0

        children = SubproblemBatch(
            lb=batch.lb.index_select(0, parent_index),
            ub=batch.ub.index_select(0, parent_index),
            depths=batch.depths.index_select(0, parent_index) + 1,
            incremental_alpha=_gather(batch.incremental_alpha),
            incremental_eta=_gather(batch.incremental_eta),
            split_signs=signs,
            parent_margins=(
                batch.parent_margins.index_select(0, parent_index)
                if batch.parent_margins is not None
                else None
            ),
            lower_bound=(
                batch.lower_bound.index_select(0, parent_index)
                if batch.lower_bound is not None
                else None
            ),
            spec_row_ids=(
                batch.spec_row_ids.index_select(
                    0, parent_index.to(batch.spec_row_ids.device)
                )
                if batch.spec_row_ids is not None
                else None
            ),
        )
        return children, parent_index

    raise ValueError(f"Unknown SplitDecision.kind: {decision.kind!r}")


def _slice_branching_state(
    bounds_dict: Optional[Dict[int, Bounds]],
    nu_per_layer: Optional[Dict[int, torch.Tensor]],
    lane_idx: torch.Tensor,
    k_actual: int,
) -> tuple[Optional[Dict[int, Bounds]], Optional[Dict[int, torch.Tensor]]]:
    # ν/bounds are computed over the full k_actual wave; the brancher runs on the
    # sub-batch actually being split. Bounds are [k_actual, *]; ν is [k_actual*M, n]
    # packed sample-major (row b*M+j), so ν rows expand per selected lane.
    bd_out: Optional[Dict[int, Bounds]] = None
    if bounds_dict is not None:
        bd_out = {
            lid: Bounds(
                b.lb.index_select(0, lane_idx.to(b.lb.device)),
                b.ub.index_select(0, lane_idx.to(b.ub.device)),
            )
            for lid, b in bounds_dict.items()
        }
    nu_out: Optional[Dict[int, torch.Tensor]] = None
    if nu_per_layer is not None:
        nu_out = {}
        for lid, t in nu_per_layer.items():
            total = int(t.shape[0])
            if k_actual > 0 and total != k_actual and total % k_actual == 0:
                m = total // k_actual
                rows = (
                    lane_idx.to(t.device).unsqueeze(1) * m
                    + torch.arange(m, device=t.device)
                ).reshape(-1)
            else:
                rows = lane_idx.to(t.device)
            nu_out[lid] = t.index_select(0, rows)
    return bd_out, nu_out



def _unbatch_field(val: Any) -> Any:
    """Strip lazy-M broadcast batch dim when a field is shared by one sample.

    BaB dual dispatch rebuilds an ``OutputSpec`` from ASSERT parameters while
    subproblem lanes live in the leading lazy-M dimension. If a parameter is a
    tensor with a singleton leading batch axis, remove that axis so
    ``OutputSpec.encode_linear`` can re-broadcast it to the current K lanes.
    """
    if isinstance(val, torch.Tensor) and val.dim() >= 2 and val.shape[0] == 1:
        return val[0]
    return val




def _as_batched_vector(
    value: object,
    n_batch: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    t = t.to(device=device, dtype=dtype)
    if t.dim() == 0:
        return t.expand(n_batch, width).contiguous()
    if t.dim() == 1:
        if t.numel() == width:
            return t.unsqueeze(0).expand(n_batch, -1).contiguous()
        if width == 1 and t.numel() == n_batch:
            return t.reshape(n_batch, 1).contiguous()
    if t.dim() == 2:
        if t.shape == (1, width):
            return t.expand(n_batch, -1).contiguous()
        if t.shape == (n_batch, width):
            return t.contiguous()
    raise ValueError(
        f"{name}: expected scalar, ({width},), (1,{width}), or "
        f"({n_batch},{width}); got {tuple(t.shape)}"
    )


def _as_batched_index(
    value: object,
    n_batch: int,
    n_out: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    t = t.to(device=device, dtype=torch.long).reshape(-1)
    if t.numel() == 1:
        t = t.expand(n_batch)
    if t.numel() != n_batch:
        raise ValueError(
            f"{name}: expected 1 or {n_batch} indices, got {t.numel()}"
        )
    if bool(((t < 0) | (t >= n_out)).any().item()):
        raise ValueError(f"{name}: index out of range for n_out={n_out}: {t.tolist()}")
    return t.contiguous()


# Per-net cache of reconstructed PyTorch nn.Module used for CE validation.
# Without this, every check_violations_batched call rebuilds the module from
# scratch via ACTToTorch.run() — costly under K-batched BaB which can invoke
# CE validation dozens of times for a single net.  Cleared per top-level
# verify-all dispatch via clear_violation_check_module_cache().
_VIOLATION_CHECK_MODULE_CACHE: dict[int, torch.nn.Module] = {}


def clear_violation_check_module_cache() -> None:
    _VIOLATION_CHECK_MODULE_CACHE.clear()


def _module_float_dtype(module: torch.nn.Module, default: torch.dtype) -> torch.dtype:
    for tensor in module.parameters():
        if tensor.is_floating_point():
            return tensor.dtype
    for tensor in module.buffers():
        if tensor.is_floating_point():
            return tensor.dtype
    return default


def _forward_for_violation_check(net: object, x_batch: torch.Tensor) -> torch.Tensor:
    if isinstance(net, torch.nn.Module):
        module = net
    else:
        key = id(net)
        cached = _VIOLATION_CHECK_MODULE_CACHE.get(key)
        if cached is None:
            from act.pipeline.verification.act2torch import ACTToTorch

            cached = ACTToTorch(cast(Net, net)).run()
            # ACTToTorch emits mixed float32 weights + float64 buffers; unify to
            # the analysis dtype or the internal forward clashes float32/float64.
            if x_batch.is_floating_point():
                cached = cached.to(dtype=x_batch.dtype)
            _VIOLATION_CHECK_MODULE_CACHE[key] = cached
        module = cached
    _ = module.eval()
    target_dtype = _module_float_dtype(module, x_batch.dtype)
    if x_batch.dtype != target_dtype:
        x_batch = x_batch.to(dtype=target_dtype)
    success, output, error = infer_single_model("ce_validate_batched", module, x_batch)
    if not success or output is None:
        raise RuntimeError(f"check_violations_batched: model forward failed: {error}")
    if output.dim() < 2:
        raise ValueError(
            f"check_violations_batched: model output must be batched, got "
            f"shape={tuple(output.shape)}"
        )
    return output.reshape(output.shape[0], -1)


@torch.no_grad()
def check_violations_batched(net: object, x_batch: torch.Tensor, assert_layer: Layer) -> torch.Tensor:
    """[BATCHED-API] Return a ``[N]`` bool tensor for concrete ASSERT violations.

    ``x_batch`` is always treated as a tensor-view batch ``[N, *input_shape]``;
    N=1 is represented by a length-one leading dimension. ASSERT parameters are
    read directly from ``assert_layer.params`` in their batch-native form.
    """
    if x_batch.dim() < 2:
        raise ValueError(
            f"check_violations_batched: x_batch must be [N, *input_shape], "
            f"got shape={tuple(x_batch.shape)}"
        )
    y_batch = _forward_for_violation_check(net, x_batch)
    n_batch = int(x_batch.shape[0])
    if int(y_batch.shape[0]) != n_batch:
        raise ValueError(
            f"check_violations_batched: output batch {int(y_batch.shape[0])} "
            f"!= input batch {n_batch}"
        )
    n_out = int(y_batch.shape[1])
    device = y_batch.device
    dtype = y_batch.dtype
    params = assert_layer.params
    kind = params.get("kind")
    eps = 1e-8

    if kind == OutKind.TOP1_ROBUST:
        y_true = _as_batched_index(params["y_true"], n_batch, n_out, device, "y_true")
        y_true_scores = y_batch.gather(1, y_true.unsqueeze(1)).squeeze(1)
        mask = torch.ones_like(y_batch, dtype=torch.bool)
        _ = mask.scatter_(1, y_true.unsqueeze(1), False)
        other_scores = y_batch.masked_fill(~mask, -float("inf"))
        return (other_scores.max(dim=1).values - y_true_scores) >= 0

    if kind == OutKind.MARGIN_ROBUST:
        y_true = _as_batched_index(params["y_true"], n_batch, n_out, device, "y_true")
        margin = _as_batched_vector(
            params["margin"], n_batch, 1, device, dtype, "margin"
        ).reshape(n_batch)
        y_true_scores = y_batch.gather(1, y_true.unsqueeze(1)).squeeze(1)
        mask = torch.ones_like(y_batch, dtype=torch.bool)
        _ = mask.scatter_(1, y_true.unsqueeze(1), False)
        other_scores = y_batch.masked_fill(~mask, -float("inf"))
        return (other_scores.max(dim=1).values - y_true_scores) >= margin

    if kind == OutKind.LINEAR_LE:
        c_raw = params["c"]
        c_t = (c_raw if isinstance(c_raw, torch.Tensor) else torch.as_tensor(c_raw)).to(device=device, dtype=dtype)
        if c_t.dim() <= 1:
            rows = 1
        elif c_t.dim() == 2:
            if c_t.shape[1] != n_out:
                raise ValueError(f"LINEAR_LE: c cols {c_t.shape[1]} != n_out {n_out}")
            rows = int(c_t.shape[0])
        else:
            rows = int(c_t.shape[1])
        if rows <= 1:
            coeff = _as_batched_vector(params["c"], n_batch, n_out, device, dtype, "c")
            bound = _as_batched_vector(params["d"], n_batch, 1, device, dtype, "d").reshape(n_batch)
            return (coeff * y_batch).sum(dim=1) >= bound + eps
        if c_t.dim() == 2:
            c_view = c_t.unsqueeze(0).expand(n_batch, -1, -1)
        else:
            c_view = c_t if c_t.shape[0] == n_batch else c_t.expand(n_batch, -1, -1)
        d_raw = params["d"]
        d_t = (d_raw if isinstance(d_raw, torch.Tensor) else torch.as_tensor(d_raw)).to(device=device, dtype=dtype).flatten()
        if d_t.numel() == rows:
            d_view = d_t.unsqueeze(0).expand(n_batch, -1)
        elif d_t.numel() == n_batch * rows:
            d_view = d_t.reshape(n_batch, rows)
        else:
            raise ValueError(f"LINEAR_LE: d numel {d_t.numel()} incompatible with rows {rows}")
        lhs = torch.einsum("bmo,bo->bm", c_view.contiguous(), y_batch)
        return (lhs > d_view + eps).any(dim=1)

    if kind == OutKind.RANGE:
        result = torch.zeros(n_batch, dtype=torch.bool, device=device)
        lb_raw = params.get("lb")
        ub_raw = params.get("ub")
        if lb_raw is not None:
            lb = _as_batched_vector(lb_raw, n_batch, n_out, device, dtype, "lb")
            result = result | (y_batch < lb - eps).any(dim=1)
        if ub_raw is not None:
            ub = _as_batched_vector(ub_raw, n_batch, n_out, device, dtype, "ub")
            result = result | (y_batch > ub + eps).any(dim=1)
        return result

    if kind == OutKind.UNSAFE_LINEAR:
        m_raw = params.get("M", 1)
        if isinstance(m_raw, torch.Tensor):
            m_rows = int(m_raw.item())
        elif isinstance(m_raw, int):
            m_rows = m_raw
        else:
            raise ValueError(f"UNSAFE_LINEAR: M must be int, got {m_raw!r}")
        c_raw = params.get("C", params.get("c"))
        if c_raw is None:
            raise ValueError("UNSAFE_LINEAR requires C or c params")
        c_tensor = c_raw if isinstance(c_raw, torch.Tensor) else torch.as_tensor(c_raw)
        c_tensor = c_tensor.to(device=device, dtype=dtype)
        if c_tensor.dim() == 2:
            if c_tensor.shape == (m_rows, n_out):
                c_view = c_tensor.unsqueeze(0).expand(n_batch, -1, -1).contiguous()
            elif c_tensor.shape == (n_batch * m_rows, n_out):
                c_view = c_tensor.reshape(n_batch, m_rows, n_out).contiguous()
            else:
                raise ValueError(
                    f"UNSAFE_LINEAR: C shape {tuple(c_tensor.shape)} incompatible "
                    f"with N={n_batch}, M={m_rows}, n_out={n_out}"
                )
        elif c_tensor.dim() == 3:
            if c_tensor.shape == (1, m_rows, n_out):
                c_view = c_tensor.expand(n_batch, -1, -1).contiguous()
            elif c_tensor.shape == (n_batch, m_rows, n_out):
                c_view = c_tensor.contiguous()
            else:
                raise ValueError(
                    f"UNSAFE_LINEAR: c shape {tuple(c_tensor.shape)} incompatible "
                    f"with N={n_batch}, M={m_rows}, n_out={n_out}"
                )
        else:
            raise ValueError(f"UNSAFE_LINEAR: unsupported C dim {c_tensor.dim()}")
        d_raw = params.get("thresholds", params.get("d"))
        if d_raw is None:
            raise ValueError("UNSAFE_LINEAR requires thresholds or d params")
        d_tensor = d_raw if isinstance(d_raw, torch.Tensor) else torch.as_tensor(d_raw)
        d_tensor = d_tensor.to(device=device, dtype=dtype)
        if d_tensor.dim() == 1 and d_tensor.numel() == m_rows:
            d_view = d_tensor.unsqueeze(0).expand(n_batch, -1).contiguous()
        elif d_tensor.shape == (1, m_rows):
            d_view = d_tensor.expand(n_batch, -1).contiguous()
        elif d_tensor.shape == (n_batch, m_rows):
            d_view = d_tensor.contiguous()
        else:
            raise ValueError(
                f"UNSAFE_LINEAR: d shape {tuple(d_tensor.shape)} incompatible "
                f"with N={n_batch}, M={m_rows}"
            )
        lhs = torch.einsum("bmo,bo->bm", c_view, y_batch)
        return (lhs <= d_view + eps).all(dim=1)

    raise NotImplementedError(f"ASSERT kind not supported: {kind}")


def _check_input_specs_batched(x_batch: torch.Tensor, spec_layers: List[Layer]) -> torch.Tensor:
    """Check every concrete input constraint without a fail-open kind.

    This is a counterexample authority boundary: unsupported or incomplete
    constraints must never be silently ignored.
    """

    result = torch.ones(x_batch.shape[0], device=x_batch.device, dtype=torch.bool)
    n_batch = int(x_batch.shape[0])
    x_flat = x_batch.flatten(start_dim=1)

    def _as_float_tensor(value: Any) -> Optional[torch.Tensor]:
        if isinstance(value, torch.Tensor):
            return value.to(device=x_batch.device, dtype=x_batch.dtype)
        if isinstance(value, (int, float, bool)):
            return x_batch.new_tensor(value)
        return None

    def _batch_scalar(value: Any) -> Optional[torch.Tensor]:
        tensor = _as_float_tensor(value)
        if tensor is None:
            return None
        if tensor.numel() == 1:
            return tensor.reshape(1).expand(n_batch)
        if tensor.numel() == n_batch:
            return tensor.reshape(n_batch)
        return None

    for layer in spec_layers:
        kind = layer.params.get("kind")
        if kind == InKind.BOX:
            lb_t = _as_float_tensor(layer.params.get("lb"))
            ub_t = _as_float_tensor(layer.params.get("ub"))
            if lb_t is None or ub_t is None:
                result &= torch.zeros_like(result)
                continue
            result &= (
                (x_batch >= lb_t) & (x_batch <= ub_t)
            ).flatten(start_dim=1).all(dim=1)
            continue

        if kind == InKind.LINF_BALL:
            center_t = _as_float_tensor(layer.params.get("center"))
            eps_b = _batch_scalar(layer.params.get("eps"))
            if center_t is not None and eps_b is not None:
                delta_linf = (
                    x_batch - center_t
                ).abs().flatten(start_dim=1).amax(dim=1)
                result &= torch.isfinite(delta_linf) & (
                    delta_linf <= eps_b
                )
                continue
            # Older synthesized layers may materialize only the equivalent
            # box.  It is authoritative only when both sides are present.
            lb_t = _as_float_tensor(layer.params.get("lb"))
            ub_t = _as_float_tensor(layer.params.get("ub"))
            if lb_t is None or ub_t is None:
                result &= torch.zeros_like(result)
                continue
            result &= (
                (x_batch >= lb_t) & (x_batch <= ub_t)
            ).flatten(start_dim=1).all(dim=1)
            continue

        if kind == InKind.LIN_POLY:
            A_t = _as_float_tensor(layer.params.get("A"))
            b_t = _as_float_tensor(layer.params.get("b"))
            if A_t is None or b_t is None:
                result &= torch.zeros_like(result)
                continue
            if A_t.dim() == 2:
                A_b = A_t.unsqueeze(0).expand(n_batch, -1, -1)
            elif A_t.dim() == 3 and A_t.shape[0] in (1, n_batch):
                A_b = (
                    A_t.expand(n_batch, -1, -1)
                    if A_t.shape[0] == 1
                    else A_t
                )
            else:
                result &= torch.zeros_like(result)
                continue
            if int(A_b.shape[-1]) != int(x_flat.shape[-1]):
                result &= torch.zeros_like(result)
                continue
            n_rows = int(A_b.shape[1])
            if b_t.numel() == n_rows:
                b_b = b_t.reshape(1, n_rows).expand(n_batch, -1)
            elif b_t.numel() == n_batch * n_rows:
                b_b = b_t.reshape(n_batch, n_rows)
            else:
                result &= torch.zeros_like(result)
                continue
            lhs = torch.bmm(
                A_b, x_flat.unsqueeze(-1)
            ).squeeze(-1)
            result &= (
                torch.isfinite(lhs)
                & torch.isfinite(b_b)
                & (lhs <= b_b)
            ).all(dim=1)
            continue

        if kind != InKind.LP_EMBEDDING:
            raise NotImplementedError(
                f"unsupported INPUT_SPEC kind in counterexample replay: "
                f"{kind!r}"
            )

        lb_t = _as_float_tensor(layer.params.get("lb"))
        ub_t = _as_float_tensor(layer.params.get("ub"))
        if (lb_t is None) != (ub_t is None):
            result &= torch.zeros_like(result)
            continue
        if lb_t is not None and ub_t is not None:
            result &= (
                (x_batch >= lb_t) & (x_batch <= ub_t)
            ).flatten(start_dim=1).all(dim=1)

        center_t = _as_float_tensor(layer.params.get("center"))
        eps_b = _batch_scalar(layer.params.get("eps"))
        p_norm_t = _as_float_tensor(layer.params.get("p_norm"))
        if (
            center_t is None
            or eps_b is None
            or p_norm_t is None
            or p_norm_t.numel() != 1
        ):
            result &= torch.zeros_like(result)
            continue
        p_value = float(p_norm_t.reshape(-1)[0].item())
        mask = normalize_position_mask(
            layer.params.get("perturbed_positions"),
            int(center_t.shape[-2]),
            batch_shape=tuple(center_t.shape[:-2]),
            device=x_batch.device,
        )
        if mask.shape[0] == 1 and x_batch.shape[0] != 1:
            mask_b = mask.expand(x_batch.shape[0], *mask.shape[1:])
            center_b = center_t.expand_as(x_batch)
        else:
            mask_b = mask
            center_b = center_t.expand_as(x_batch)
        delta = x_batch - center_b
        clean = (~mask_b).unsqueeze(-1).expand_as(delta)
        if bool(clean.any().item()):
            clean_ok = (
                torch.where(
                    clean, delta.abs(), torch.zeros_like(delta)
                )
                .flatten(start_dim=1)
                .amax(dim=1)
                == 0
            )
            result &= clean_ok
        perturbed_delta = delta[mask_b.unsqueeze(-1).expand_as(delta)].reshape(x_batch.shape[0], -1, center_t.shape[-1])
        if perturbed_delta.numel() == 0:
            continue
        if p_value == float("inf"):
            norms = perturbed_delta.abs().amax(dim=-1)
        elif p_value == 1.0:
            norms = perturbed_delta.abs().sum(dim=-1)
        elif p_value == 2.0:
            norms = torch.linalg.vector_norm(perturbed_delta, ord=2, dim=-1)
        else:
            norms = torch.linalg.vector_norm(perturbed_delta, ord=p_value, dim=-1)
        result &= (
            torch.isfinite(norms)
            & (norms <= eps_b.unsqueeze(1))
        ).all(dim=1)
    return result


# ---------------------------------------------------------------------------
# Strategy factories
# ---------------------------------------------------------------------------


def _build_branching_strategy(method: str, *, dual_solver: Any = None) -> BranchingStrategy:
    return _build_branching_strategy_impl(method, dual_solver=dual_solver)


def _build_bounding(
    method: str,
    *,
    depth_weight: float = 1.0,
    bound_weight: float = 1.0,
    order_name: str = "depth_lb",
    cooling_rate: float = 0.99,
) -> BoundingStrategy:
    if method == "random":
        return RandomBounding()
    if method == "topk":
        if order_name == "depth_lb":
            order = DepthLowerBoundOrder(depth_weight=depth_weight, bound_weight=bound_weight)
        elif order_name == "greedy":
            order = GreedyOrder()
        elif order_name == "sa":
            order = SAOrder(cooling_rate=cooling_rate)
        else:
            raise ValueError(f"unknown bounding_order {order_name!r}")

        return TopKBounding(order)
    raise ValueError(f"Unknown bounding method: {method!r}")


def _groups_to_tensors(groups: Dict[int, Any], batch: SubproblemBatch):
    bb = batch.batch_size
    if len(groups) != bb:
        return None, None, 0
    k_eff = len(groups.get(0, []))
    if k_eff < 1:
        return None, None, 0
    device = batch.lb.device
    top_layers = torch.zeros(bb, k_eff, dtype=torch.long, device=device)
    top_neurons = torch.zeros(bb, k_eff, dtype=torch.long, device=device)
    for lane in range(bb):
        entries = groups.get(lane, [])
        if len(entries) != k_eff:
            return None, None, 0
        for j, (lid, nidx) in enumerate(entries):
            top_layers[lane, j] = int(lid)
            top_neurons[lane, j] = int(nidx)
    return top_layers, top_neurons, k_eff


def _dispatch_dual_solve(
    *,
    net: Net,
    assert_layer: Layer,
    batched_bounds: Bounds,
    k_actual: int,
    batch: SubproblemBatch,
    config: BaBConfig,
    optimize: bool,
    keep_rows: Optional[torch.Tensor] = None,
    root_bounds_dict: Optional[Dict[int, Bounds]] = None,
    round_policy: Optional[Any] = None,
) -> DualSolveResult:
    """Run one dual-family BaB bound pass and decode lane statuses.

    ``keep_rows`` restricts the encoded spec to the given row indices
    (ALL-rows kinds only). ``root_bounds_dict`` replaces the per-node forward
    pass with the root box's bounds (input-layer entries overridden by each
    lane's sub-box). Both are sound by bound monotonicity: certified rows and
    per-layer bounds of an ancestor box remain valid on every descendant.
    """
    from act.back_end.dual_tf.tf_forward import compute_forward_bounds
    from act.back_end.solver.solver_dual import DualSolver, expand_bounds_dict

    solver_tier = getattr(config, "solver_tier", "lp")
    refine_audit: Optional[Dict[str, Any]] = None
    block_eps_updates = _install_embedding_child_block_eps(net, batched_bounds, batch)
    if root_bounds_dict is not None:
        bounds_dict_dual = expand_bounds_dict(root_bounds_dict, k_actual)
        lane_box = Bounds(batched_bounds.lb, batched_bounds.ub)
        for layer in net.layers:
            kind_up = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if kind_up in ("INPUT", "INPUT_SPEC") and layer.id in bounds_dict_dual:
                bounds_dict_dual[layer.id] = lane_box
        if batch.split_signs:
            refreshed = _interval_refresh_bounds(net, bounds_dict_dual, batch.split_signs)
            if refreshed is not None:
                bounds_dict_dual = refreshed
            psr_mode = getattr(config, "per_subproblem_refine", "none")
            psr_rows_cap = getattr(config, "per_subproblem_refine_rows_cap", 64)
            psr_iters = getattr(config, "per_subproblem_refine_iters", 0)
            psr_layer_cap = getattr(
                config, "per_subproblem_refine_layer_cap", 2
            )
            if round_policy is not None:
                if round_policy.refine_mode is not None:
                    psr_mode = round_policy.refine_mode
                if round_policy.refine_rows_cap is not None:
                    psr_rows_cap = round_policy.refine_rows_cap
                if round_policy.refine_iters is not None:
                    psr_iters = round_policy.refine_iters
            if psr_mode != "none":
                refine_started = time.monotonic()
                before_refine = bounds_dict_dual
                selector_audit: Dict[str, Any] = {}
                bounds_dict_dual = DualSolver().refine_intermediate_bounds_batched(
                    net,
                    bounds_dict_dual,
                    split_signs=batch.split_signs,
                    mode=psr_mode,
                    rows_cap=psr_rows_cap,
                    optimize_iters=psr_iters,
                    layer_cap=psr_layer_cap,
                    audit=selector_audit,
                )
                strict_lower = 0
                strict_upper = 0
                changed_layers = 0
                for layer_id, old_bounds in before_refine.items():
                    new_bounds = bounds_dict_dual.get(layer_id)
                    if new_bounds is None:
                        continue
                    lower_count = int(
                        torch.count_nonzero(
                            new_bounds.lb > old_bounds.lb
                        ).item()
                    )
                    upper_count = int(
                        torch.count_nonzero(
                            new_bounds.ub < old_bounds.ub
                        ).item()
                    )
                    strict_lower += lower_count
                    strict_upper += upper_count
                    changed_layers += int(
                        lower_count > 0 or upper_count > 0
                    )
                refine_audit = {
                    "mode": str(psr_mode),
                    "rows_cap": int(psr_rows_cap),
                    "optimize_iters": int(psr_iters),
                    "elapsed_seconds": (
                        time.monotonic() - refine_started
                    ),
                    "strict_lower_entries": strict_lower,
                    "strict_upper_entries": strict_upper,
                    "changed_layers": changed_layers,
                    "proof_authority": False,
                    "selected_layer_ids": selector_audit.get(
                        "selected_layer_ids", []
                    ),
                    "queried_objective_rows": int(
                        selector_audit.get("queried_objective_rows", 0)
                    ),
                }
    else:
        bounds_dict_dual = compute_forward_bounds(net, batched_bounds.lb, batched_bounds.ub)
    out_kind_raw = assert_layer.params["kind"]
    if not isinstance(out_kind_raw, str):
        raise TypeError(f"ASSERT kind must be str, got {type(out_kind_raw).__name__}")

    out_spec_fields: dict[str, torch.Tensor] = {}
    for key in OutputSpec.SLICEABLE_PARAM_KEYS:
        if key in assert_layer.params and assert_layer.params[key] is not None:
            value = assert_layer.params[key]
            tensor_value = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            out_spec_fields[key] = _unbatch_field(tensor_value)

    out_spec = OutputSpec(
        kind=out_kind_raw,
        y_true=out_spec_fields.get("y_true"),
        margin=out_spec_fields.get("margin"),
        c=out_spec_fields.get("c"),
        d=out_spec_fields.get("d"),
        lb=out_spec_fields.get("lb"),
        ub=out_spec_fields.get("ub"),
    )
    sample_bounds = next(iter(bounds_dict_dual.values()))
    device = sample_bounds.lb.device
    dtype = sample_bounds.lb.dtype
    assert_preds = net.preds.get(assert_layer.id, [])
    if len(assert_preds) != 1:
        raise ValueError(
            f"ASSERT layer {assert_layer.id} must have exactly 1 predecessor, "
            f"got {len(assert_preds)}"
        )
    output_bounds = bounds_dict_dual[assert_preds[0]]
    n_out = int(output_bounds.lb.flatten(start_dim=1).shape[-1])
    encoded_spec = out_spec.encode_linear(B=k_actual, n_out=n_out, device=device, dtype=dtype)
    m_specs = int(encoded_spec["M"])
    dual = DualSolver()

    if out_spec.kind == OutKind.UNSAFE_LINEAR:
        if batch.spec_row_ids is not None:
            raise ValueError(
                "property-separable BaB is invalid for UNSAFE_LINEAR "
                "(OR-row semantics)"
            )
        c_rows = cast(torch.Tensor, encoded_spec["C"]).contiguous()
        thresholds = cast(torch.Tensor, encoded_spec["thresholds"]).contiguous()
    else:
        c_rows = -cast(torch.Tensor, encoded_spec["C"]).contiguous()
        thresholds = -cast(torch.Tensor, encoded_spec["thresholds"]).contiguous()
        if batch.spec_row_ids is not None:
            if keep_rows is not None:
                raise ValueError(
                    "lane-specific spec rows and global keep_rows cannot "
                    "be active together"
                )
            row_ids = batch.spec_row_ids.to(
                device=device, dtype=torch.long
            ).reshape(-1)
            if row_ids.numel() != k_actual:
                raise ValueError(
                    "spec_row_ids must contain one row per BaB lane"
                )
            if bool(
                ((row_ids < 0) | (row_ids >= m_specs)).any().item()
            ):
                raise ValueError(
                    "spec_row_ids contains an ASSERT row outside the "
                    f"encoded range [0, {m_specs})"
                )
            lane_index = torch.arange(k_actual, device=device)
            c_rows = (
                c_rows.view(k_actual, m_specs, n_out)[
                    lane_index, row_ids
                ]
                .reshape(k_actual, n_out)
                .contiguous()
            )
            thresholds = thresholds[lane_index, row_ids].reshape(
                k_actual, 1
            )
            m_specs = 1
        elif keep_rows is not None:
            idx = keep_rows.to(device=device, dtype=torch.long)
            c_rows = (
                c_rows.view(k_actual, m_specs, n_out)
                .index_select(1, idx)
                .reshape(k_actual * int(idx.numel()), n_out)
                .contiguous()
            )
            thresholds = thresholds.index_select(1, idx).contiguous()
            m_specs = int(idx.numel())
    active_mask = torch.ones(k_actual, m_specs, dtype=torch.bool, device=device)

    return_nu = _want_babsr_neuron_branching(config)
    supports_return_nu = "return_nu_per_layer" in inspect.signature(
        dual.compute_certified_bound
    ).parameters

    compute_certified_bound = cast(Any, dual.compute_certified_bound)

    is_child_batch = bool(batch.depths.min().item() > 0) if batch.depths.numel() else False
    if optimize:
        dual_result = compute_certified_bound(
            net,
            bounds_dict_dual,
            c_rows,
            M=m_specs,
            optimize=True,
            optimize_alpha=not (
                getattr(config, "eta_only_children", False) and is_child_batch
            ),
            refresh_forward=root_bounds_dict is None,
            n_iters=config.dual_n_iters,
            lr_alpha=config.lr_alpha,
            lr_beta=config.lr_beta,
            lr_decay=config.lr_decay,
            eta=batch.incremental_eta if solver_tier == "dual_alpha_eta" else None,
            incremental_alphas=batch.incremental_alpha if getattr(config, "incremental_start_enabled", True) else None,
            incremental_etas=(
                batch.incremental_eta
                if solver_tier == "dual_alpha_eta"
                and getattr(config, "incremental_start_enabled", True)
                else None
            ),
            split_signs=batch.split_signs if solver_tier == "dual_alpha_eta" else None,
            return_optimized=True,
            return_sce=True,
            per_class_alpha=config.per_class_alpha,
            **({"return_nu_per_layer": True} if return_nu and supports_return_nu else {}),
        )
        margins_flat = dual_result.margins
        sce = cast(Optional[torch.Tensor], dual_result.sce)
        batch.incremental_alpha = dual_result.alpha_state
        if solver_tier == "dual_alpha_eta":
            batch.incremental_eta = dual_result.eta_state
    else:
        dual_result = compute_certified_bound(
            net,
            bounds_dict_dual,
            c_rows,
            M=m_specs,
            return_sce=True,
            **({"return_nu_per_layer": True} if return_nu and supports_return_nu else {}),
        )
        margins_flat = dual_result.margins
        sce = cast(Optional[torch.Tensor], dual_result.sce)

    margins = margins_flat.view(k_actual, m_specs)
    slack = margins - thresholds
    strictly_certified = _strictly_certified_slack(slack, margins)
    if out_spec.kind == OutKind.UNSAFE_LINEAR:
        certified = (strictly_certified & active_mask).any(dim=-1)
        candidate_rows = torch.zeros(k_actual, dtype=torch.long, device=device)
    else:
        unresolved = (~strictly_certified) & active_mask
        certified = ~unresolved.any(dim=-1)
        candidate_rows = torch.where(
            unresolved.any(dim=1),
            unresolved.to(torch.int64).argmax(dim=1),
            torch.zeros(k_actual, dtype=torch.long, device=device),
        )

    statuses = tuple(
        SolveStatus.UNSAT if bool(is_certified.item()) else SolveStatus.SAT
        for is_certified in certified
    )
    nvars = max((max(layer.out_vars) for layer in net.layers if layer.out_vars), default=-1) + 1
    x_candidate = torch.zeros(k_actual, nvars, device=device, dtype=dtype)
    if sce is not None:
        sce_flat = sce.flatten(start_dim=1).to(device=device)
        row_offsets = torch.arange(k_actual, device=device) * m_specs + candidate_rows.to(device=device)
        chosen_sce = sce_flat.index_select(0, row_offsets)
        input_ids = torch.tensor(get_input_ids(net), device=device, dtype=torch.long)
        x_candidate[:, input_ids] = chosen_sce.to(device=device, dtype=dtype)
    else:
        # TODO: extend CE-candidate generation for dual paths that do not return SCE.
        statuses = tuple(
            SolveStatus.UNSAT if status == SolveStatus.UNSAT else SolveStatus.UNKNOWN
            for status in statuses
        )
    solution = BatchLPSolution(
        statuses=statuses,
        x=x_candidate,
        max_viol=-slack.min(dim=1).values.detach(),
    )
    branch_bounds: Optional[Dict[int, Bounds]] = None
    branch_nu: Optional[Dict[int, torch.Tensor]] = None
    if return_nu and root_bounds_dict is not None:
        # Heuristic-only consumer (BaBSR/FSB scores). The optimize path does
        # not emit nu, and nu=None silently degrades neuron branching to
        # input-axis splits - so run one grad-free backward at the converged
        # alpha/eta to extract per-layer nu on the same (reused) bounds.
        branch_bounds = bounds_dict_dual
        branch_nu = getattr(dual_result, "nu_per_layer", None)
        if branch_nu is None:
            nu_pass = dual.compute_certified_bound(
                net,
                bounds_dict_dual,
                c_rows,
                M=m_specs,
                alpha=getattr(dual_result, "alpha_state", None),
                eta=(
                    getattr(dual_result, "eta_state", None)
                    if solver_tier == "dual_alpha_eta"
                    else None
                ),
                split_signs=(
                    batch.split_signs if solver_tier == "dual_alpha_eta" else None
                ),
                return_nu_per_layer=True,
            )
            branch_nu = nu_pass.nu_per_layer
    elif return_nu:
        branch_bounds, branch_nu = dual.recompute_bounds_and_nu(
            net,
            bounds_dict_dual,
            c_rows,
            m_specs,
            alpha_state=getattr(dual_result, "alpha_state", None),
            eta_state=(
                getattr(dual_result, "eta_state", None)
                if solver_tier == "dual_alpha_eta"
                else None
            ),
            split_signs=(
                batch.split_signs if solver_tier == "dual_alpha_eta" else None
            ),
            per_class_alpha=config.per_class_alpha,
        )
    try:
        return DualSolveResult(
            solution=solution,
            bounds_dict=branch_bounds,
            nu_per_layer=branch_nu,
            row_slack=slack.detach(),
            row_certified=strictly_certified.detach(),
            refine_audit=refine_audit,
        )
    finally:
        _restore_embedding_child_block_eps(block_eps_updates)


def _finite_embedding_spec(net: Net) -> Optional[Layer]:
    for layer in net.layers:
        if layer.kind == "INPUT_SPEC" and layer.params.get("kind") == InKind.LP_EMBEDDING:
            p_norm = layer.params.get("p_norm", float("inf"))
            if isinstance(p_norm, torch.Tensor):
                p_value = float(p_norm.reshape(-1)[0].item())
            elif isinstance(p_norm, (int, float, bool)):
                p_value = float(p_norm)
            else:
                continue
            if p_value != float("inf"):
                return layer
    return None


def _install_embedding_child_block_eps(
    net: Net,
    batched_bounds: Bounds,
    batch: SubproblemBatch,
) -> list[tuple[Layer, ParamValue]]:
    spec = _finite_embedding_spec(net)
    if spec is None or batch.depths.numel() == 0 or int(batch.depths.max().item()) == 0:
        return []
    p_raw = spec.params.get("p_norm", float("inf"))
    if isinstance(p_raw, torch.Tensor):
        p_norm = float(p_raw.reshape(-1)[0].item())
    elif isinstance(p_raw, (int, float, bool)):
        p_norm = float(p_raw)
    else:
        return []
    input_shape = tuple(batched_bounds.lb.shape[1:])
    positions_raw = spec.params.get("perturbed_positions")
    positions = positions_raw if isinstance(positions_raw, torch.Tensor) else None
    block_eps = rederive_embedding_block_eps(
        batched_bounds.lb.flatten(start_dim=1),
        batched_bounds.ub.flatten(start_dim=1),
        input_shape,
        positions,
        p_norm,
    )
    old_values: list[tuple[Layer, ParamValue]] = []
    for layer in net.layers:
        kind_up = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
        if kind_up in ("INPUT", "INPUT_SPEC"):
            old_values.append((layer, layer.params.get("bab_block_eps", None)))
            layer.params["bab_block_eps"] = block_eps
    return old_values


def _restore_embedding_child_block_eps(updates: list[tuple[Layer, ParamValue]]) -> None:
    for layer, old in updates:
        if old is None:
            layer.params.pop("bab_block_eps", None)
        else:
            layer.params["bab_block_eps"] = old


# ---------------------------------------------------------------------------
# BaB engine
# ---------------------------------------------------------------------------


def _net_bound_elements(net: Net) -> int:
    """Total bound-carrying variables; a proxy for per-lane memory cost."""
    return sum(len(l.out_vars) for l in net.layers if getattr(l, "out_vars", None))


def _auto_batch_budget_bytes(safety: float) -> float:
    """Memory the auto sizer may use: min(safety*total, 90% of what this
    process can reclaim), so it shares the GPU with other processes."""
    free, total = torch.cuda.mem_get_info()
    reclaimable = free + torch.cuda.memory_reserved()
    return min(float(total) * safety, float(reclaimable) * 0.9)


def _auto_initial_batch(net: Net, config: BaBConfig) -> int:
    """Conservative first batch from net size; the loop recalibrates it from
    the measured per-lane peak after the first real round."""
    safety = float(getattr(config, "auto_batch_safety", 0.55))
    cap = int(getattr(config, "auto_batch_cap", 2048))
    floor = int(getattr(config, "auto_batch_floor", 8))
    per_lane = 4.0 * max(1, _net_bound_elements(net)) * 256.0
    k = int(_auto_batch_budget_bytes(safety) / per_lane)
    return max(floor, min(cap, k))


def _auto_recalibrate_batch(peak_bytes: float, max_k_seen: int, config: BaBConfig) -> int:
    """Batch for the next round = budget / measured bytes-per-lane.

    ``peak_bytes / max_k_seen`` over-estimates the marginal per-lane cost (it
    folds in the one-time root/presolve peak), so the sizer errs toward fewer
    lanes - safe against OOM while still ramping up on small nets with spare
    memory."""
    safety = float(getattr(config, "auto_batch_safety", 0.55))
    cap = int(getattr(config, "auto_batch_cap", 2048))
    floor = int(getattr(config, "auto_batch_floor", 8))
    bpl = max(peak_bytes / max(1, max_k_seen), 1.0)
    k = int(_auto_batch_budget_bytes(safety) / bpl)
    return max(floor, min(cap, k))


@torch.no_grad()
def verify_bab_batched(
    net: Net,
    solver_factory: Callable[[], Solver],
    config: Optional[BaBConfig] = None,
    *,
    max_batch_size: Optional[Union[int, str]] = None,
    time_budget_s: Optional[float] = None,
    verbose: bool = False,
    _k_log: Optional[List[int]] = None,
    _property_forest_run_token: Optional[str] = None,
    _property_forest_source_digests: Optional[
        Mapping[str, str]
    ] = None,
) -> VerifyResult:
    """[BATCHED-API] K-batched Branch-and-Bound verification (single instance).

    Per iteration::

        K       = min(len(pool), max_batch_size, max_nodes - processed)
        batch   = pool.pop(K)                       # [K, D_flat]
        sol     = setup_and_solve_batch(net, [K,*input_shape] bounds, solver_factory())
        # decode per-lane:
        #   UNSAT       -> prune (region certified)
        #   SAT + violation (check_violations_batched) -> FALSIFIED (terminate)
        #   SAT spurious / UNKNOWN -> branch (or drop at max_depth)

    Soundness: returns CERTIFIED only when the pool drains via UNSAT pruning
    with every processed sub-box resolved (``all_resolved_unsat`` and
    ``pool.empty``). If the time/node budget exhausts with unproven sub-boxes
    remaining (branched-then-never-revisited, or dropped at ``max_depth``),
    returns UNKNOWN with
    ``metadata['reason'] == 'budget_exhausted_with_unproven_subboxes'``.

    Args:
        net: ACT network with a single-instance INPUT_SPEC (B=1 seed).
        solver_factory: callable returning a fresh ``Solver`` per iteration
            (no state leakage across iterations).
        config: ``BaBConfig``; ``bab_max_batch_size`` (if present, otherwise 8)
            caps K. ``max_depth`` and ``max_nodes`` cap the search tree.
        max_batch_size: explicit override for K cap; takes precedence over
            ``config.bab_max_batch_size``.
        time_budget_s: wall-clock budget (default 300 s).
        verbose: reserved.
        _k_log: diagnostic only — if supplied, the actual K used per iteration
            is appended. Tests use this to verify K fluctuates per D4.
    """
    if config is None:
        config = BaBConfig()
    auto_batch = isinstance(max_batch_size, str) and max_batch_size == "auto"
    if auto_batch:
        effective_batch = (
            _auto_initial_batch(net, config)
            if torch.cuda.is_available()
            else int(getattr(config, "auto_batch_cap", 512))
        )
    elif max_batch_size is None:
        effective_batch = int(getattr(config, "bab_max_batch_size", 8))
    else:
        effective_batch = int(cast(int, max_batch_size))
    if effective_batch < 1:
        raise ValueError(f"max_batch_size must be >= 1, got {effective_batch}")
    initial_effective_batch = int(effective_batch)
    requested_max_batch_size: object = (
        max_batch_size
        if isinstance(max_batch_size, (int, str))
        else None
    )
    solver_factory_binding = (
        f"{getattr(solver_factory, '__module__', type(solver_factory).__module__)}."
        f"{getattr(solver_factory, '__qualname__', type(solver_factory).__qualname__)}"
    )
    max_k_seen = 0

    budget_s = time_budget_s if time_budget_s is not None else 300.0
    property_forest_live_bindings: Optional[dict[str, str]] = None
    if _property_forest_run_token is not None:
        if not bool(getattr(config, "property_separable_bab", False)):
            raise ValueError(
                "a property-forest run token requires "
                "property_separable_bab=true"
            )
        if not math.isfinite(float(budget_s)) or float(budget_s) <= 0.0:
            raise ValueError(
                "authoritative property-forest runs require a finite "
                "positive time budget"
            )
        if (
            type(_property_forest_source_digests) is not dict
            or set(_property_forest_source_digests)
            != {"onnx", "vnnlib"}
            or any(
                type(value) is not str
                or len(value) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in value
                )
                for value in _property_forest_source_digests.values()
            )
        ):
            raise ValueError(
                "authoritative property-forest runs require exact "
                "pre-run onnx/vnnlib SHA-256 digests"
            )
        from act.back_end.bab.property_forest_authority import (
            property_forest_binding_digests,
        )

        property_forest_live_bindings = (
            property_forest_binding_digests(net, config)
        )
    elif _property_forest_source_digests is not None:
        raise ValueError(
            "property-forest source digests require a run token"
        )

    fsb_dual_solver = None
    if config.branching_method == "fsb":
        from act.back_end.solver.solver_dual import DualSolver

        fsb_dual_solver = DualSolver()
    # "gain" measures child bounds directly; its fallback (when no measured
    # decision is available) is BaBSR — it reuses the dual ν scores and
    # degrades to width-based only when ν/bounds are absent, which is strictly
    # better than a random fallback.
    brancher_method = (
        "babsr" if config.branching_method == "gain" else config.branching_method
    )
    brancher = _build_branching_strategy(brancher_method, dual_solver=fsb_dual_solver)
    pool = _build_bounding(
        config.bounding_method,
        depth_weight=getattr(config, "bounding_depth_weight", 1.0),
        bound_weight=getattr(config, "bounding_bound_weight", 1.0),
        order_name=getattr(config, "bounding_order", "depth_lb"),
        cooling_rate=getattr(config, "sa_cooling_rate", 0.99),
    )
    llm_probe: Any = None
    _llm: Any = None
    _wave_index = 0
    if getattr(config, "llm_probe_enabled", False):
        from act.pipeline.verification import llm_probe as _llm
        llm_probe = _llm.build_llm_probe(config)

    provenance = bool(getattr(config, "provenance_enabled", False))
    if provenance and not isinstance(pool, TopKBounding):
        raise ValueError("provenance_enabled requires bounding_method='topk'")
    node_counter = 0
    fanout = max(2, int(getattr(config, "input_split_fanout", 2)))
    frontier_cap = int(getattr(config, "frontier_cap", 0))
    joint_gain_probe_calls = 0
    joint_gain_probe_nodes = 0
    joint_gain_groups_tested = 0
    joint_gain_nonbaseline_lanes = 0
    joint_gain_more_diverse_lanes = 0
    joint_gain_worst_child_improvement_max = 0.0
    split_depth_histogram: Dict[int, int] = {}
    child_refine_calls = 0
    child_refine_seconds = 0.0
    child_refine_strict_lower_entries = 0
    child_refine_strict_upper_entries = 0
    child_refine_changed_layers = 0
    child_refine_queried_objective_rows = 0
    child_refine_selected_layer_ids: set[int] = set()
    nonterminal_filter_calls = 0
    nonterminal_filter_applied_calls = 0
    nonterminal_filter_fallbacks = 0
    nonterminal_filter_selected_layer_ids: set[int] = set()
    property_forest_enabled = bool(
        getattr(config, "property_separable_bab", False)
    )
    property_forest_row_ids: tuple[int, ...] = ()
    property_forest_counters: dict[str, Dict[int, int]] = {
        name: {}
        for name in (
            "roots",
            "children_expected",
            "children_minted",
            "processed",
            "certified",
            "branched",
            "active_pool",
        )
    }
    property_forest_processed_by_row = property_forest_counters[
        "processed"
    ]
    property_forest_certified_nodes_by_row = property_forest_counters[
        "certified"
    ]
    property_forest_dropped: dict[str, Dict[int, int]] = {
        "frontier_cap": {},
        "max_depth": {},
    }
    property_forest_integrity_errors_by_row: Dict[int, list[str]] = {}
    property_forest_runtime_integrity_errors: list[str] = []
    property_forest_total_rows = 0
    property_forest_root_certified_rows: tuple[int, ...] = ()
    property_forest_verification_dtype: Optional[str] = None
    property_forest_verification_device: Optional[str] = None

    def _property_forest_live_facts(
        *,
        mode: str,
        processed_nodes: int,
        pool_nodes: int,
        dropped_frontier: bool,
        dropped_depth: bool,
    ) -> dict[str, object]:
        return {
            "mode": mode,
            "verifier_status": VerifyStatus.CERTIFIED.value,
            "spec_rows_total": int(property_forest_total_rows),
            "root_certified_rows": list(
                property_forest_root_certified_rows
            ),
            "forest_rows": list(property_forest_row_ids),
            "processed_nodes": int(processed_nodes),
            "pool_remaining": int(pool_nodes),
            "any_dropped_frontier_cap": bool(dropped_frontier),
            "any_dropped_max_depth": bool(dropped_depth),
            "requested_max_batch_size": requested_max_batch_size,
            "initial_effective_max_batch_size": (
                initial_effective_batch
            ),
            "time_budget_seconds": float(budget_s),
            "solver_tier": str(getattr(config, "solver_tier", "lp")),
            "solver_backend": (
                "act.back_end.solver.solver_dual.DualSolver"
            ),
            "solver_factory": solver_factory_binding,
            "verification_dtype": property_forest_verification_dtype,
            "verification_device": property_forest_verification_device,
            "torch_default_dtype": str(torch.get_default_dtype()),
        }

    def _seal_property_forest_live(
        result: VerifyResult,
        node_receipt: Mapping[str, object],
        live_facts: Mapping[str, object],
    ) -> tuple[Optional[dict[str, object]], Optional[object]]:
        if (
            _property_forest_run_token is None
            or property_forest_live_bindings is None
        ):
            return None, None
        from act.back_end.bab.property_forest_authority import (
            _issue_property_forest_live_result,
        )

        return _issue_property_forest_live_result(
            result=result,
            run_token=_property_forest_run_token,
            binding_digests=property_forest_live_bindings,
            source_digests=cast(
                Mapping[str, str],
                _property_forest_source_digests,
            ),
            node_receipt=node_receipt,
            live_facts=live_facts,
        )

    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)
    if property_forest_enabled:
        if assert_layer.params.get("kind") == OutKind.UNSAFE_LINEAR:
            raise ValueError(
                "property-separable BaB requires ALL-rows output semantics"
            )
        if getattr(config, "solver_tier", "lp") not in (
            "dual",
            "dual_alpha",
            "dual_alpha_eta",
        ):
            raise ValueError(
                "property-separable BaB requires a dual solver tier"
            )
        if not isinstance(pool, TopKBounding):
            raise ValueError(
                "property-separable BaB requires lossless topk bounding"
            )
        if int(getattr(config, "presplit_levels", 0)) != 0:
            raise ValueError(
                "property-separable BaB and presplit_levels cannot be "
                "combined"
            )
    root_bounds = seed_from_input_specs(spec_layers)
    property_forest_verification_dtype = str(root_bounds.lb.dtype)
    property_forest_verification_device = root_bounds.lb.device.type
    if (
        root_bounds.ub.dtype != root_bounds.lb.dtype
        or root_bounds.ub.device.type
        != property_forest_verification_device
    ):
        raise ValueError(
            "root lower/upper bounds must share dtype and device"
        )
    input_shape: tuple[int, ...] = tuple(root_bounds.lb.shape[1:])

    per_lane_dim = int(root_bounds.lb[0].numel())
    n_input_vars = len(get_input_ids(net))
    if n_input_vars != per_lane_dim:
        raise ValueError(
            f"verify_bab_batched: INPUT layer declares {n_input_vars} variables "
            f"but the per-lane input dim is {per_lane_dim}. The net was likely "
            f"converted with a batched input shape (B baked into INPUT vars); "
            f"synthesize per-instance (B=1) models before BaB."
        )

    root_batch = SubproblemBatch.from_bounds(root_bounds)
    if provenance:
        n = root_batch.batch_size
        root_batch.node_id = torch.arange(
            node_counter,
            node_counter + n,
            device=root_batch.lb.device,
            dtype=torch.long,
        )
        root_batch.parent_id = torch.full(
            (n,), -1, device=root_batch.lb.device, dtype=torch.long
        )
        node_counter += n

    # Root spec-pruning presolve (ALL-rows kinds, dual tiers): rows certified
    # on the root box stay certified on every sub-box, so descendants only
    # carry the unproven rows.
    spec_keep_rows: Optional[torch.Tensor] = None
    presolve_tier = getattr(config, "solver_tier", "lp")
    root_fwd: Optional[Dict[int, Bounds]] = None
    refine_mode = getattr(config, "intermediate_refine", "none")
    if presolve_tier in ("dual", "dual_alpha", "dual_alpha_eta") and (
        getattr(config, "reuse_root_bounds", False) or refine_mode != "none"
    ):
        from act.back_end.dual_tf.tf_forward import compute_forward_bounds
        from act.back_end.solver.solver_dual import DualSolver

        root_fwd = compute_forward_bounds(net, root_bounds.lb, root_bounds.ub)
        if refine_mode != "none":
            root_fwd = DualSolver().refine_intermediate_bounds(
                net,
                root_fwd,
                mode=refine_mode,
                blowup_ratio=getattr(config, "intermediate_refine_ratio", 10.0),
            )
    # Per-node bound reuse is governed solely by reuse_root_bounds: root_fwd may
    # exist just for the root presolve/refine above, and passing it to descendant
    # solves would freeze every child's intermediate bounds at root tightness
    # (fatal for input-split BaB, where the whole gain comes from recomputing
    # intermediates on the smaller box).
    node_root_fwd: Optional[Dict[int, Bounds]] = (
        root_fwd if getattr(config, "reuse_root_bounds", False) else None
    )
    if (
        presolve_tier in ("dual", "dual_alpha", "dual_alpha_eta")
        and assert_layer.params.get("kind") != OutKind.UNSAFE_LINEAR
    ):
        presolve = _dispatch_dual_solve(
            net=net,
            assert_layer=assert_layer,
            batched_bounds=Bounds(root_bounds.lb, root_bounds.ub),
            k_actual=root_batch.batch_size,
            batch=root_batch,
            config=config,
            optimize=presolve_tier in ("dual_alpha", "dual_alpha_eta"),
            root_bounds_dict=root_fwd,
        )
        if presolve.row_slack is not None:
            if (
                presolve.row_certified is None
                or presolve.row_certified.shape
                != presolve.row_slack.shape
            ):
                raise RuntimeError(
                    "root presolve did not return its exact row "
                    "certification mask"
                )
            unproven = (~presolve.row_certified).any(dim=0)
            total_rows = int(unproven.numel())
            if property_forest_enabled:
                property_forest_total_rows = total_rows
                property_forest_root_certified_rows = tuple(
                    int(value)
                    for value in torch.where(~unproven)[0].tolist()
                )
            if not bool(unproven.any().item()):
                root_receipt: dict[str, object] = {
                    "schema": (
                        "act.property_forest_root_presolve_receipt.v1"
                    ),
                    "proof_authority": False,
                    "spec_rows_total": total_rows,
                    "strictly_certified_rows": list(
                        property_forest_root_certified_rows
                    ),
                    "forest_rows": [],
                    "complete": bool(
                        not property_forest_enabled
                        or len(
                            property_forest_root_certified_rows
                        )
                        == total_rows
                    ),
                }
                root_live_facts = _property_forest_live_facts(
                    mode="root_presolve",
                    processed_nodes=root_batch.batch_size,
                    pool_nodes=0,
                    dropped_frontier=False,
                    dropped_depth=False,
                )
                root_result = VerifyResult(
                    VerifyStatus.CERTIFIED,
                    metadata={
                        "nodes": root_batch.batch_size,
                        "pool_remaining": 0,
                        "spec_rows_total": total_rows,
                        "spec_rows_kept": 0,
                        "resolved_by": "root_presolve",
                        "property_separable_bab": property_forest_enabled,
                        "property_forest_root_rows": [],
                        "property_forest_root_certified_rows": list(
                            property_forest_root_certified_rows
                        ),
                        "property_forest_root_presolve_receipt": (
                            root_receipt
                        ),
                        "property_forest_live_facts": root_live_facts,
                        "property_forest_live_seal": None,
                        "_property_forest_live_capability": None,
                    },
                )
                if property_forest_enabled:
                    (
                        root_live_seal,
                        root_live_capability,
                    ) = _seal_property_forest_live(
                        root_result,
                        root_receipt,
                        root_live_facts,
                    )
                    root_result.metadata[
                        "property_forest_live_seal"
                    ] = root_live_seal
                    root_result.metadata[
                        "_property_forest_live_capability"
                    ] = root_live_capability
                return root_result
            keep = torch.where(unproven)[0]
            if int(keep.numel()) < total_rows:
                spec_keep_rows = keep
                root_batch.incremental_alpha = _select_spec_rows(
                    root_batch.incremental_alpha, keep,
                )
                root_batch.incremental_eta = _select_spec_rows(
                    root_batch.incremental_eta, keep,
                )
                root_batch.split_signs = _select_spec_rows(
                    root_batch.split_signs, keep,
                )
            if property_forest_enabled:
                property_forest_row_ids = tuple(
                    int(value) for value in keep.tolist()
                )
                root_batch = _expand_property_forest_root(
                    root_batch, keep
                )
                # Each lane now selects its immutable original ASSERT row;
                # no global row slice may remain active.
                spec_keep_rows = None
                property_forest_counters = {
                    name: {
                        row_id: 0
                        for row_id in property_forest_row_ids
                    }
                    for name in property_forest_counters
                }
                property_forest_processed_by_row = (
                    property_forest_counters["processed"]
                )
                property_forest_certified_nodes_by_row = (
                    property_forest_counters["certified"]
                )
                property_forest_dropped = {
                    reason: {
                        row_id: 0
                        for row_id in property_forest_row_ids
                    }
                    for reason in property_forest_dropped
                }
                property_forest_integrity_errors_by_row = {
                    row_id: [] for row_id in property_forest_row_ids
                }
                if root_batch.spec_row_ids is None:
                    property_forest_runtime_integrity_errors.append(
                        "expanded_root_missing_row_ids"
                    )
                else:
                    for raw_row_id in root_batch.spec_row_ids.tolist():
                        row_id = int(raw_row_id)
                        if row_id not in property_forest_counters["roots"]:
                            property_forest_runtime_integrity_errors.append(
                                "expanded_root_contains_unknown_row"
                            )
                            continue
                        property_forest_counters["roots"][row_id] += 1
                        property_forest_counters["active_pool"][row_id] += 1
                if provenance:
                    n = root_batch.batch_size
                    root_batch.node_id = torch.arange(
                        node_counter,
                        node_counter + n,
                        device=root_batch.lb.device,
                        dtype=torch.long,
                    )
                    root_batch.parent_id = torch.full(
                        (n,),
                        -1,
                        device=root_batch.lb.device,
                        dtype=torch.long,
                    )
                    node_counter += n
        presplit_k = int(getattr(config, "presplit_levels", 0))
        if (
            presplit_k > 0
            and root_batch.batch_size == 1
            and presolve.bounds_dict is not None
            and presolve.nu_per_layer is not None
        ):
            presplit = _presplit_root(
                root_batch, presolve.bounds_dict, presolve.nu_per_layer, presplit_k,
            )
            if presplit is not None:
                root_batch = presplit
                node_counter += root_batch.batch_size

    def _property_pool_row_counts() -> Dict[int, int]:
        counts = {
            row_id: 0 for row_id in property_forest_row_ids
        }
        if not property_forest_enabled:
            return counts
        stored_ids = getattr(pool, "_spec_row_ids", None)
        if len(pool) == 0:
            if stored_ids is not None and int(stored_ids.numel()) != 0:
                property_forest_runtime_integrity_errors.append(
                    "empty_pool_retained_row_ids"
                )
            return counts
        if (
            not isinstance(stored_ids, torch.Tensor)
            or int(stored_ids.numel()) != len(pool)
        ):
            property_forest_runtime_integrity_errors.append(
                "pool_row_id_storage_mismatch"
            )
            return counts
        for raw_row_id in stored_ids.tolist():
            row_id = int(raw_row_id)
            if row_id not in counts:
                property_forest_runtime_integrity_errors.append(
                    "pool_contains_unknown_row"
                )
                continue
            counts[row_id] += 1
        return counts

    def _evict_to_frontier_cap() -> int:
        if frontier_cap <= 0 or len(pool) <= frontier_cap:
            return 0
        before = _property_pool_row_counts()
        evicted = int(pool.evict_to(frontier_cap))
        if property_forest_enabled:
            after = _property_pool_row_counts()
            attributed = 0
            for row_id in property_forest_row_ids:
                removed = before[row_id] - after[row_id]
                if removed < 0:
                    property_forest_runtime_integrity_errors.append(
                        "frontier_eviction_increased_a_row"
                    )
                    continue
                attributed += removed
                property_forest_dropped["frontier_cap"][
                    row_id
                ] += removed
                property_forest_counters["active_pool"][
                    row_id
                ] -= removed
            if attributed != evicted:
                property_forest_runtime_integrity_errors.append(
                    "frontier_eviction_attribution_mismatch"
                )
        return evicted

    pool.push(root_batch)
    any_dropped_frontier_cap = False
    if _evict_to_frontier_cap() > 0:
        any_dropped_frontier_cap = True

    start = time.time()
    processed = 0
    any_dropped_max_depth = False
    _last_input_widths: Optional[list[float]] = None

    while not pool.empty:
        elapsed = time.time() - start
        if elapsed >= budget_s or processed >= config.max_nodes:
            break

        remaining_nodes = config.max_nodes - processed
        k_requested = min(len(pool), effective_batch, remaining_nodes)
        if k_requested <= 0:
            break

        _wave_t0 = time.time()
        _pool_before = len(pool)
        _wave_policy = None
        _wave_split_used = None
        if llm_probe is not None and _llm is not None:
            _wave_policy = llm_probe.begin_wave(_llm.build_frontier_stats(
                wave_index=_wave_index,
                pool_size=len(pool),
                effective_batch=effective_batch,
                remaining_nodes=remaining_nodes,
                elapsed_s=elapsed,
                remaining_s=max(0.0, budget_s - elapsed),
                input_widths=_last_input_widths,
            ))
            if _wave_policy.k_requested is not None:
                k_requested = max(1, min(_wave_policy.k_requested, len(pool), effective_batch, remaining_nodes))

        batch = pool.pop(batch_size=k_requested)
        k_actual = batch.batch_size
        if property_forest_enabled:
            if batch.spec_row_ids is None:
                raise RuntimeError(
                    "property forest pool lost its ASSERT-row identities"
                )
            row_values = [
                int(value) for value in batch.spec_row_ids.tolist()
            ]
            allowed_rows = set(property_forest_row_ids)
            if any(row_id not in allowed_rows for row_id in row_values):
                raise RuntimeError(
                    "property forest pool contains an unknown ASSERT row"
                )
            for row_id in row_values:
                property_forest_counters["active_pool"][row_id] -= 1
                if (
                    property_forest_counters["active_pool"][row_id]
                    < 0
                ):
                    property_forest_integrity_errors_by_row[
                        row_id
                    ].append("processed_node_was_not_active")
                property_forest_processed_by_row[row_id] += 1
        if _k_log is not None:
            _k_log.append(k_actual)

        if input_shape:
            k_lb = batch.lb.reshape(k_actual, *input_shape)
            k_ub = batch.ub.reshape(k_actual, *input_shape)
        else:
            k_lb = batch.lb
            k_ub = batch.ub
        batched_bounds = Bounds(k_lb, k_ub)

        solver_tier = getattr(config, "solver_tier", "lp")
        want_neuron_branching = _want_babsr_neuron_branching(config)
        bounds_dict_for_branching: Optional[Dict[int, Bounds]] = None
        nu_per_layer_for_branching: Optional[Dict[int, torch.Tensor]] = None
        row_slack_for_branching: Optional[torch.Tensor] = None
        if solver_tier == "lp":
            solver = solver_factory()
            solution = setup_and_solve_batch(
                net, batched_bounds, solver, timelimit=None,
            )
        elif solver_tier == "dual":
            dual_solve_result = _dispatch_dual_solve(
                net=net,
                assert_layer=assert_layer,
                batched_bounds=batched_bounds,
                k_actual=k_actual,
                batch=batch,
                config=config,
                optimize=False,
                keep_rows=spec_keep_rows,
                root_bounds_dict=node_root_fwd,
                round_policy=_wave_policy,
            )
            solution = dual_solve_result.solution
        elif solver_tier in ("dual_alpha", "dual_alpha_eta"):
            dual_solve_result = _dispatch_dual_solve(
                net=net,
                assert_layer=assert_layer,
                batched_bounds=batched_bounds,
                k_actual=k_actual,
                batch=batch,
                config=config,
                optimize=True,
                keep_rows=spec_keep_rows,
                root_bounds_dict=node_root_fwd,
                round_policy=_wave_policy,
            )
            solution = dual_solve_result.solution
            bounds_dict_for_branching = dual_solve_result.bounds_dict
            nu_per_layer_for_branching = dual_solve_result.nu_per_layer
            row_slack_for_branching = dual_solve_result.row_slack
            if dual_solve_result.refine_audit is not None:
                child_refine_calls += 1
                child_refine_seconds += float(
                    dual_solve_result.refine_audit["elapsed_seconds"]
                )
                child_refine_strict_lower_entries += int(
                    dual_solve_result.refine_audit[
                        "strict_lower_entries"
                    ]
                )
                child_refine_strict_upper_entries += int(
                    dual_solve_result.refine_audit[
                        "strict_upper_entries"
                    ]
                )
                child_refine_changed_layers += int(
                    dual_solve_result.refine_audit["changed_layers"]
                )
                child_refine_queried_objective_rows += int(
                    dual_solve_result.refine_audit[
                        "queried_objective_rows"
                    ]
                )
                child_refine_selected_layer_ids.update(
                    int(layer_id)
                    for layer_id in dual_solve_result.refine_audit[
                        "selected_layer_ids"
                    ]
                )
        else:
            raise ValueError(
                f"Unknown solver_tier={solver_tier!r}. Valid: {VALID_SOLVER_TIERS}."
            )

        node_lower_bound = (-solution.max_viol).detach()
        if batch.lower_bound is not None:
            # Bound inheritance: a child region is a subset of its parent, so
            # the parent's certified lower bound stays valid; clamping removes
            # per-subproblem optimization regressions (observed: re-optimized
            # children reporting bounds below their parent's).
            node_lower_bound = torch.maximum(
                node_lower_bound, batch.lower_bound.to(node_lower_bound.device)
            )
        if property_forest_enabled:
            assert batch.spec_row_ids is not None
            for lane, status in enumerate(solution.statuses):
                if status == SolveStatus.UNSAT:
                    row_id = int(batch.spec_row_ids[lane].item())
                    property_forest_certified_nodes_by_row[row_id] += 1

        sat_lane_idx = [
            i for i, s in enumerate(solution.statuses) if s == SolveStatus.SAT
        ]
        if sat_lane_idx:
            input_ids = get_input_ids(net)
            input_index = torch.tensor(
                input_ids, device=solution.x.device, dtype=torch.long,
            )
            sat_idx_t = torch.tensor(
                sat_lane_idx, device=solution.x.device, dtype=torch.long,
            )
            x_full = solution.x.index_select(0, sat_idx_t)
            x_input_flat = x_full.index_select(1, input_index)
            x_input_shaped = (
                x_input_flat.reshape(len(sat_lane_idx), *input_shape)
                if input_shape
                else x_input_flat
            )
            in_region = _check_input_specs_batched(x_input_shaped, spec_layers)
            raw_violations = check_violations_batched(
                net, x_input_shaped, assert_layer
            )
            violations = raw_violations & in_region
            for j, lane in enumerate(sat_lane_idx):
                if bool(violations[j].item()):
                    forest_row_id = None
                    if batch.spec_row_ids is not None:
                        forest_row_id = int(
                            batch.spec_row_ids[lane].item()
                        )
                    return VerifyResult(
                        VerifyStatus.FALSIFIED,
                        counterexample=x_input_shaped[j].detach().cpu().clone(),
                        metadata={
                            "nodes": processed + k_actual,
                            "lane": lane,
                            "K": k_actual,
                            "nodes_minted": node_counter,
                            "any_dropped_frontier_cap": any_dropped_frontier_cap,
                            "any_dropped_max_depth": any_dropped_max_depth,
                            "property_separable_bab": (
                                property_forest_enabled
                            ),
                            "property_forest_root_rows": list(
                                property_forest_row_ids
                            ),
                            "counterexample_input_spec_valid": bool(
                                in_region[j].item()
                            ),
                            "counterexample_full_assert_violated": bool(
                                raw_violations[j].item()
                            ),
                            "counterexample_replayed": True,
                            "counterexample_spec_row_id": forest_row_id,
                        },
                    )

        unresolved_idx = torch.tensor(
            [i for i, status in enumerate(solution.statuses) if status != SolveStatus.UNSAT],
            device=batch.lb.device,
            dtype=torch.long,
        )
        if int(unresolved_idx.numel()) > 0:
            def _select_incremental_state(
                state: Optional[dict[int, torch.Tensor]],
                indices: torch.Tensor,
            ) -> Optional[dict[int, torch.Tensor]]:
                if state is None:
                    return None
                return {
                    layer_id: tensor.index_select(0, indices.to(tensor.device))
                    for layer_id, tensor in state.items()
                }

            unresolved = SubproblemBatch(
                lb=batch.lb.index_select(0, unresolved_idx.to(batch.lb.device)),
                ub=batch.ub.index_select(0, unresolved_idx.to(batch.ub.device)),
                depths=batch.depths.index_select(0, unresolved_idx.to(batch.depths.device)),
                incremental_alpha=_select_incremental_state(batch.incremental_alpha, unresolved_idx),
                incremental_eta=_select_incremental_state(batch.incremental_eta, unresolved_idx),
                split_signs=_select_incremental_state(batch.split_signs, unresolved_idx),
                parent_margins=(
                    batch.parent_margins.index_select(0, unresolved_idx.to(batch.parent_margins.device))
                    if batch.parent_margins is not None
                    else None
                ),
                lower_bound=node_lower_bound.index_select(
                    0, unresolved_idx.to(node_lower_bound.device)
                ),
                node_id=(
                    batch.node_id.index_select(0, unresolved_idx.to(batch.node_id.device))
                    if batch.node_id is not None
                    else None
                ),
                parent_id=(
                    batch.parent_id.index_select(0, unresolved_idx.to(batch.parent_id.device))
                    if batch.parent_id is not None
                    else None
                ),
                spec_row_ids=(
                    batch.spec_row_ids.index_select(
                        0, unresolved_idx.to(batch.spec_row_ids.device)
                    )
                    if batch.spec_row_ids is not None
                    else None
                ),
            )
            branch_mask = unresolved.depths < int(config.max_depth)
            if bool((~branch_mask).any().item()):
                any_dropped_max_depth = True
                if property_forest_enabled:
                    if unresolved.spec_row_ids is None:
                        property_forest_runtime_integrity_errors.append(
                            "max_depth_nodes_missing_row_ids"
                        )
                    else:
                        for terminal_index in torch.where(
                            ~branch_mask
                        )[0].tolist():
                            row_id = int(
                                unresolved.spec_row_ids[
                                    terminal_index
                                ].item()
                            )
                            property_forest_dropped["max_depth"][
                                row_id
                            ] += 1
            branch_idx = torch.where(branch_mask)[0]
            if int(branch_idx.numel()) > 0:
                branch_batch = SubproblemBatch(
                    lb=unresolved.lb.index_select(0, branch_idx.to(unresolved.lb.device)),
                    ub=unresolved.ub.index_select(0, branch_idx.to(unresolved.ub.device)),
                    depths=unresolved.depths.index_select(0, branch_idx),
                    incremental_alpha=_select_incremental_state(unresolved.incremental_alpha, branch_idx),
                    incremental_eta=_select_incremental_state(unresolved.incremental_eta, branch_idx),
                    split_signs=_select_incremental_state(unresolved.split_signs, branch_idx),
                    parent_margins=(
                        unresolved.parent_margins.index_select(0, branch_idx)
                        if unresolved.parent_margins is not None
                        else None
                    ),
                    lower_bound=(
                        unresolved.lower_bound.index_select(0, branch_idx)
                        if unresolved.lower_bound is not None
                        else None
                    ),
                    node_id=(
                        unresolved.node_id.index_select(0, branch_idx.to(unresolved.node_id.device))
                        if unresolved.node_id is not None
                        else None
                    ),
                    parent_id=(
                        unresolved.parent_id.index_select(0, branch_idx.to(unresolved.parent_id.device))
                        if unresolved.parent_id is not None
                        else None
                    ),
                    spec_row_ids=(
                        unresolved.spec_row_ids.index_select(
                            0,
                            branch_idx.to(
                                unresolved.spec_row_ids.device
                            ),
                        )
                        if unresolved.spec_row_ids is not None
                        else None
                    ),
                )
                expected_children_per_parent = 2
                if want_neuron_branching:
                    full_branch_idx = unresolved_idx.index_select(
                        0, branch_idx.to(unresolved_idx.device)
                    )
                    bd_branch, nu_branch = _slice_branching_state(
                        bounds_dict_for_branching,
                        nu_per_layer_for_branching,
                        full_branch_idx,
                        k_actual,
                    )
                    if getattr(
                        config,
                        "branch_requires_unstable_successor",
                        False,
                    ):
                        nonterminal_filter_calls += 1
                        (
                            bd_branch,
                            nu_branch,
                            eligible_layers,
                            filter_applied,
                        ) = _filter_branching_state_to_unstable_successors(
                            branch_batch,
                            net,
                            bd_branch,
                            nu_branch,
                        )
                        if filter_applied:
                            nonterminal_filter_applied_calls += 1
                            nonterminal_filter_selected_layer_ids.update(
                                eligible_layers
                            )
                        else:
                            nonterminal_filter_fallbacks += 1
                    spec_row_focus = None
                    if (
                        getattr(config, "property_branch_focus", "sum")
                        == "worst"
                        and row_slack_for_branching is not None
                    ):
                        branch_slack = row_slack_for_branching.index_select(
                            0,
                            full_branch_idx.to(
                                row_slack_for_branching.device
                            ),
                        )
                        if branch_slack.ndim == 2 and branch_slack.shape[1] > 0:
                            spec_row_focus = branch_slack.argmin(dim=1)
                    multi = None
                    multi_k = int(getattr(config, "multi_split_levels", 1))
                    if llm_probe is not None and _llm is not None and llm_probe.wants_neuron:
                        # neuron_topk>0 => never bail on candidate count: enumerate the
                        # full set (limit=None) and let advise_neuron_groups truncate to
                        # the top-K by score, so the LLM always decides (with a bounded
                        # view) instead of falling back to FSB. neuron_topk==0 keeps the
                        # legacy "bail to FSB when > max_candidates_total" behavior.
                        _neuron_topk = int(getattr(config, "llm_probe_neuron_topk", 0))
                        _cand_dicts = enumerate_unstable_candidates(
                            branch_batch, bd_branch, nu_branch,
                            limit=None if _neuron_topk > 0
                            else getattr(config, "llm_probe_max_candidates_total", 1024),
                            spec_row_index=spec_row_focus,
                        )
                        if _cand_dicts:
                            _ngroups = llm_probe.advise_neuron_groups(_llm.build_frontier_stats(
                                wave_index=_wave_index,
                                pool_size=len(pool),
                                effective_batch=effective_batch,
                                remaining_nodes=remaining_nodes,
                                elapsed_s=elapsed,
                                branch_batch_size=branch_batch.batch_size,
                                candidates=[_llm.CandidateSummary(**_d) for _d in _cand_dicts],
                            ))
                            if _ngroups is not None:
                                _tl, _tn, _keff = _groups_to_tensors(_ngroups, branch_batch)
                                if _tl is not None and _tn is not None:
                                    multi = _multi_split_from_groups(branch_batch, net, _tl, _tn, _keff)
                                    _wave_split_used = _keff
                                    if multi is not None:
                                        expected_children_per_parent = (
                                            2 ** int(_keff)
                                        )
                    if multi is None and config.branching_method == "gain" and multi_k > 1:
                        # Adaptive split depth: fan out so children roughly
                        # fill one bounding batch; n_branch lanes x 2^k <=
                        # max_batch_size keeps the frontier from flooding
                        # the pool.
                        k_adaptive = max(
                            1,
                            min(
                                multi_k,
                                int(math.log2(max(2, effective_batch // max(1, branch_batch.batch_size)))),
                            ),
                        )
                        if _wave_policy is not None and _wave_policy.split_k is not None and _llm is not None:
                            k_adaptive = _llm.clip_split_k(
                                _wave_policy.split_k,
                                branch_batch_size=branch_batch.batch_size,
                                effective_batch=effective_batch,
                                multi_split_levels=multi_k,
                            )
                        contraction_target = float(
                            getattr(
                                config,
                                "frontier_contraction_target",
                                0.0,
                            )
                        )
                        if (
                            contraction_target > 0.0
                            and int(branch_batch.depths.min().item()) > 0
                        ):
                            survivor_rate = (
                                float(branch_batch.batch_size)
                                / float(max(1, k_actual))
                            )
                            k_adaptive = min(
                                k_adaptive,
                                _survival_controlled_split_depth(
                                    multi_k,
                                    survivor_rate,
                                    contraction_target,
                                ),
                            )
                        _wave_split_used = k_adaptive
                        split_depth_histogram[k_adaptive] = (
                            split_depth_histogram.get(k_adaptive, 0)
                            + int(branch_batch.batch_size)
                        )
                        if k_adaptive > 1:
                            joint_gain_groups = int(
                                getattr(config, "joint_gain_groups", 1)
                            )
                            if joint_gain_groups > 1:
                                joint_audit: Dict[str, Any] = {}
                                multi = _gain_tested_multi_split(
                                    branch_batch,
                                    net,
                                    assert_layer,
                                    config,
                                    spec_keep_rows,
                                    node_root_fwd,
                                    bd_branch,
                                    nu_branch,
                                    input_shape,
                                    k_levels=k_adaptive,
                                    max_groups=joint_gain_groups,
                                    max_probe_batch=effective_batch,
                                    audit=joint_audit,
                                    spec_row_index=spec_row_focus,
                                )
                                if multi is not None:
                                    expected_children_per_parent = (
                                        2 ** int(k_adaptive)
                                    )
                                    joint_gain_probe_calls += 1
                                    joint_gain_probe_nodes += int(
                                        joint_audit["probe_nodes"]
                                    )
                                    joint_gain_groups_tested += int(
                                        joint_audit["groups"]
                                    ) * int(joint_audit["lanes"])
                                    joint_gain_nonbaseline_lanes += int(
                                        joint_audit[
                                            "selected_nonbaseline_lanes"
                                        ]
                                    )
                                    joint_gain_more_diverse_lanes += int(
                                        joint_audit[
                                            "selected_more_diverse_lanes"
                                        ]
                                    )
                                    improvements = [
                                        selected - baseline
                                        for selected, baseline in zip(
                                            joint_audit[
                                                "selected_worst_child_lb"
                                            ],
                                            joint_audit[
                                                "baseline_worst_child_lb"
                                            ],
                                        )
                                    ]
                                    if improvements:
                                        joint_gain_worst_child_improvement_max = max(
                                            joint_gain_worst_child_improvement_max,
                                            max(improvements),
                                        )
                            if multi is None:
                                multi = _multi_split_from_decision(
                                    branch_batch,
                                    net,
                                    bd_branch,
                                    nu_branch,
                                    k_adaptive,
                                    spec_row_index=spec_row_focus,
                                )
                                if multi is not None:
                                    expected_children_per_parent = (
                                        2 ** int(k_adaptive)
                                    )
                    if multi is not None:
                        children, parent_index = multi
                    else:
                        decision = None
                        if config.branching_method == "gain":
                            decision = _gain_tested_decision(
                                branch_batch,
                                net,
                                assert_layer,
                                config,
                                spec_keep_rows,
                                node_root_fwd,
                                bd_branch,
                                nu_branch,
                                input_shape,
                                spec_row_index=spec_row_focus,
                            )
                        if decision is None:
                            scores = cast(Any, brancher).compute_scores(
                                branch_batch,
                                net,
                                bounds_dict=bd_branch,
                                nu_per_layer=nu_branch,
                            )
                            decision = cast(SplitDecision, cast(Any, brancher).select(scores))
                        if decision.kind == "input_axis":
                            decision.fanout = fanout
                            expected_children_per_parent = int(fanout)
                        else:
                            expected_children_per_parent = 2
                        children, parent_index = _split_from_decision(branch_batch, decision, net)
                else:
                    scores = brancher.compute_scores(branch_batch, net)
                    legacy_decision = cast(Any, brancher).select(scores)
                    split_fanout = fanout
                    if isinstance(legacy_decision, SplitDecision):
                        if legacy_decision.cut_dim is not None:
                            split_dims = _input_axis_decision_tensor(
                                SplitDecision(kind="input_axis", input_axis=legacy_decision.cut_dim),
                                branch_batch,
                            )
                        else:
                            if legacy_decision.input_axis is None:
                                raise ValueError("input-axis decision missing input_axis")
                            split_dims = _input_axis_decision_tensor(
                                legacy_decision,
                                branch_batch,
                            )
                        split_fanout = max(2, int(getattr(legacy_decision, "fanout", fanout)))
                    else:
                        split_dims = torch.as_tensor(
                            legacy_decision,
                            device=branch_batch.lb.device,
                            dtype=torch.long,
                        ).reshape(-1)
                    widths = branch_batch.widths()
                    _last_input_widths = (
                        widths.mean(dim=0).tolist() if widths.shape[1] <= 32 else None
                    )
                    if _wave_policy is not None and getattr(_wave_policy, "input_split_dim", None) is not None:
                        # LLM-advised input dimension (already range-clipped). Lanes where
                        # the advised dim has zero width keep the brancher's choice:
                        # splitting a zero-width dim yields identical children (livelock).
                        advised = torch.full_like(split_dims, int(_wave_policy.input_split_dim))
                        has_width = widths.gather(1, advised.unsqueeze(1)).squeeze(1) > 0
                        split_dims = torch.where(has_width, advised, split_dims)
                    if _wave_policy is not None and getattr(_wave_policy, "input_split_fanout", None) is not None:
                        split_fanout = int(_wave_policy.input_split_fanout)
                    if split_fanout == 2:
                        children, parent_index = split_input(branch_batch, split_dims)
                    else:
                        children, parent_index = split_input_nary(branch_batch, split_dims, split_fanout)
                    expected_children_per_parent = int(split_fanout)

                accept_children = True
                if property_forest_enabled:
                    partition_errors: list[str] = []
                    branch_rows = (
                        []
                        if branch_batch.spec_row_ids is None
                        else [
                            int(value)
                            for value
                            in branch_batch.spec_row_ids.tolist()
                        ]
                    )
                    if len(branch_rows) != branch_batch.batch_size:
                        partition_errors.append(
                            "split_parent_missing_assert_row_identity"
                        )
                    partition_valid, structural_errors = (
                        _validate_property_forest_child_partition(
                            branch_batch,
                            children,
                            parent_index,
                            expected_children_per_parent=(
                                expected_children_per_parent
                            ),
                        )
                    )
                    if not partition_valid:
                        partition_errors.extend(structural_errors)
                    if children.spec_row_ids is None:
                        partition_errors.append(
                            "split_children_missing_assert_row_identity"
                        )
                    elif (
                        parent_index.ndim == 1
                        and parent_index.dtype == torch.long
                        and int(parent_index.numel())
                        == children.batch_size
                        and not bool((parent_index < 0).any().item())
                        and not bool(
                            (
                                parent_index
                                >= branch_batch.batch_size
                            ).any().item()
                        )
                        and branch_batch.spec_row_ids is not None
                    ):
                        expected_rows = (
                            branch_batch.spec_row_ids.index_select(
                                0,
                                parent_index.to(
                                    branch_batch.spec_row_ids.device
                                ),
                            )
                        )
                        if not torch.equal(
                            children.spec_row_ids.to(
                                expected_rows.device
                            ),
                            expected_rows,
                        ):
                            partition_errors.append(
                                "split_changed_assert_row_identity"
                            )
                    else:
                        partition_errors.append(
                            "split_parent_index_not_row_bindable"
                        )

                    for row_id in branch_rows:
                        property_forest_counters["branched"][
                            row_id
                        ] += 1
                        property_forest_counters[
                            "children_expected"
                        ][row_id] += int(
                            expected_children_per_parent
                        )
                    if partition_errors:
                        accept_children = False
                        for row_id in branch_rows:
                            property_forest_integrity_errors_by_row[
                                row_id
                            ].extend(partition_errors)
                    else:
                        assert children.spec_row_ids is not None
                        for raw_row_id in (
                            children.spec_row_ids.tolist()
                        ):
                            row_id = int(raw_row_id)
                            property_forest_counters[
                                "children_minted"
                            ][row_id] += 1
                            property_forest_counters[
                                "active_pool"
                            ][row_id] += 1
                if accept_children and provenance:
                    pid = branch_batch.node_id
                    assert pid is not None
                    children.parent_id = pid.index_select(0, parent_index.to(pid.device))
                    nc = children.batch_size
                    children.node_id = torch.arange(
                        node_counter,
                        node_counter + nc,
                        device=children.lb.device,
                        dtype=torch.long,
                    )
                    node_counter += nc
                if accept_children:
                    pool.push(children)
                    if _evict_to_frontier_cap() > 0:
                        any_dropped_frontier_cap = True

        processed += k_actual

        if auto_batch and torch.cuda.is_available():
            max_k_seen = max(max_k_seen, k_actual)
            effective_batch = _auto_recalibrate_batch(
                torch.cuda.max_memory_allocated(), max_k_seen, config,
            )

        if llm_probe is not None and _llm is not None:
            llm_probe.end_wave(_llm.WaveOutcome(
                wave_index=_wave_index,
                pool_before=_pool_before,
                pool_after=len(pool),
                k_requested_used=k_actual,
                split_k_used=_wave_split_used if _wave_split_used is not None else 1,
                refine_iters_used=(_wave_policy.refine_iters if (_wave_policy is not None and _wave_policy.refine_iters is not None) else 0),
                certified_count=0,
                falsified_found=False,
                branched_count=0,
                best_lb_before=None,
                best_lb_after=None,
                wave_time_s=time.time() - _wave_t0,
                fallback_used=False,
            ))
            _wave_index += 1

    pool_remaining = len(pool)
    elapsed_total = time.time() - start
    exhausted_time = elapsed_total >= budget_s
    exhausted_nodes = processed >= config.max_nodes
    property_forest_receipt: Optional[dict[str, object]] = None
    property_forest_receipt_errors: tuple[str, ...] = ()
    property_forest_receipt_valid = False
    if property_forest_enabled:
        property_forest_receipt = _build_property_forest_receipt(
            row_ids=property_forest_row_ids,
            counters=property_forest_counters,
            dropped=property_forest_dropped,
            integrity_errors_by_row=(
                property_forest_integrity_errors_by_row
            ),
            runtime_integrity_errors=(
                property_forest_runtime_integrity_errors
            ),
            processed=processed,
            pool_remaining=pool_remaining,
        )
        (
            property_forest_receipt_valid,
            property_forest_receipt_errors,
        ) = _validate_property_forest_receipt(
            property_forest_receipt,
            expected_row_ids=property_forest_row_ids,
            expected_processed=processed,
            expected_pool_remaining=pool_remaining,
        )

    def _row_conservation_complete(row_id: int) -> bool:
        if property_forest_receipt is None:
            return False
        item = cast(
            dict[str, object],
            cast(dict[str, object], property_forest_receipt["rows"])[
                str(row_id)
            ],
        )
        dropped = cast(dict[str, int], item["dropped"])
        return bool(
            item["roots"] == 1
            and item["children_expected"] == item["children_minted"]
            and item["processed"]
            == cast(int, item["certified"]) + cast(int, item["branched"])
            and cast(int, item["roots"])
            + cast(int, item["children_minted"])
            == item["processed"]
            and item["active_pool"] == 0
            and dropped["frontier_cap"] == 0
            and dropped["max_depth"] == 0
            and not item["integrity_errors"]
        )

    property_forest_coverage_by_row = {
        str(row_id): {
            "processed": int(
                property_forest_processed_by_row.get(row_id, 0)
            ),
            "certified_nodes": int(
                property_forest_certified_nodes_by_row.get(row_id, 0)
            ),
            "covered": bool(
                _row_conservation_complete(row_id)
            ),
        }
        for row_id in property_forest_row_ids
    }
    # This is an omission/duplication firewall, not a serialized proof.
    # Certification still comes exclusively from the live UNSAT solves.
    property_forest_coverage_complete = bool(
        not property_forest_enabled
        or property_forest_receipt_valid
    )
    property_forest_incomplete_terminal = bool(
        property_forest_enabled
        and pool_remaining == 0
        and not property_forest_coverage_complete
    )
    property_forest_live_facts: Optional[dict[str, object]] = None
    property_forest_live_seal: Optional[dict[str, object]] = None
    property_forest_live_capability: Optional[object] = None
    if (
        property_forest_enabled
        and property_forest_receipt_valid
        and property_forest_receipt is not None
        and not any_dropped_max_depth
        and not any_dropped_frontier_cap
        and pool_remaining == 0
    ):
        property_forest_live_facts = _property_forest_live_facts(
            mode="complete_forest",
            processed_nodes=processed,
            pool_nodes=pool_remaining,
            dropped_frontier=any_dropped_frontier_cap,
            dropped_depth=any_dropped_max_depth,
        )

    spec_rows_kept = (
        len(property_forest_row_ids)
        if property_forest_enabled
        else (
            int(spec_keep_rows.numel())
            if spec_keep_rows is not None
            else None
        )
    )

    if (
        not any_dropped_max_depth
        and not any_dropped_frontier_cap
        and pool_remaining == 0
        and property_forest_coverage_complete
    ):
        certified_result = VerifyResult(
            VerifyStatus.CERTIFIED,
            metadata={
                "nodes": processed,
                "spec_rows_total": (
                    property_forest_total_rows
                    if property_forest_enabled
                    else None
                ),
                "spec_rows_kept": spec_rows_kept,
                "pool_remaining": 0,
                "exhausted_budget_time": exhausted_time,
                "exhausted_budget_nodes": exhausted_nodes,
                "nodes_minted": node_counter,
                "any_dropped_frontier_cap": any_dropped_frontier_cap,
                "any_dropped_max_depth": any_dropped_max_depth,
                "joint_gain_groups": int(
                    getattr(config, "joint_gain_groups", 1)
                ),
                "property_branch_focus": getattr(
                    config, "property_branch_focus", "sum"
                ),
                "branch_requires_unstable_successor": bool(
                    getattr(
                        config,
                        "branch_requires_unstable_successor",
                        False,
                    )
                ),
                "property_separable_bab": property_forest_enabled,
                "property_forest_root_rows": list(
                    property_forest_row_ids
                ),
                "property_forest_root_count": len(
                    property_forest_row_ids
                ),
                "property_forest_all_solves_single_row": bool(
                    property_forest_enabled
                ),
                "property_forest_coverage_complete": (
                    property_forest_coverage_complete
                ),
                "property_forest_coverage_by_row": (
                    property_forest_coverage_by_row
                ),
                "property_forest_node_conservation_receipt": (
                    property_forest_receipt
                ),
                "property_forest_node_conservation_valid": (
                    property_forest_receipt_valid
                ),
                "property_forest_node_conservation_errors": list(
                    property_forest_receipt_errors
                ),
                "property_forest_root_certified_rows": list(
                    property_forest_root_certified_rows
                ),
                "property_forest_live_facts": (
                    property_forest_live_facts
                ),
                "property_forest_live_seal": (
                    property_forest_live_seal
                ),
                "_property_forest_live_capability": (
                    property_forest_live_capability
                ),
                "property_forest_processed_by_row": {
                    str(row_id): int(count)
                    for row_id, count in sorted(
                        property_forest_processed_by_row.items()
                    )
                },
                "property_forest_certified_nodes_by_row": {
                    str(row_id): int(count)
                    for row_id, count in sorted(
                        property_forest_certified_nodes_by_row.items()
                    )
                },
                "nonterminal_filter_calls": nonterminal_filter_calls,
                "nonterminal_filter_applied_calls": (
                    nonterminal_filter_applied_calls
                ),
                "nonterminal_filter_fallbacks": (
                    nonterminal_filter_fallbacks
                ),
                "nonterminal_filter_selected_layer_ids": sorted(
                    nonterminal_filter_selected_layer_ids
                ),
                "frontier_contraction_target": float(
                    getattr(config, "frontier_contraction_target", 0.0)
                ),
                "split_depth_histogram": {
                    str(depth): int(count)
                    for depth, count in sorted(
                        split_depth_histogram.items()
                    )
                },
                "child_refine_calls": child_refine_calls,
                "child_refine_seconds": child_refine_seconds,
                "child_refine_strict_lower_entries": (
                    child_refine_strict_lower_entries
                ),
                "child_refine_strict_upper_entries": (
                    child_refine_strict_upper_entries
                ),
                "child_refine_changed_layers": (
                    child_refine_changed_layers
                ),
                "child_refine_queried_objective_rows": (
                    child_refine_queried_objective_rows
                ),
                "child_refine_selected_layer_ids": sorted(
                    child_refine_selected_layer_ids
                ),
                "joint_gain_probe_calls": joint_gain_probe_calls,
                "joint_gain_probe_nodes": joint_gain_probe_nodes,
                "joint_gain_groups_tested": joint_gain_groups_tested,
                "joint_gain_nonbaseline_lanes": (
                    joint_gain_nonbaseline_lanes
                ),
                "joint_gain_more_diverse_lanes": (
                    joint_gain_more_diverse_lanes
                ),
                "joint_gain_worst_child_improvement_max": (
                    joint_gain_worst_child_improvement_max
                ),
            },
        )
        if (
            property_forest_enabled
            and property_forest_receipt is not None
            and property_forest_live_facts is not None
        ):
            (
                property_forest_live_seal,
                property_forest_live_capability,
            ) = _seal_property_forest_live(
                certified_result,
                property_forest_receipt,
                property_forest_live_facts,
            )
            certified_result.metadata[
                "property_forest_live_seal"
            ] = property_forest_live_seal
            certified_result.metadata[
                "_property_forest_live_capability"
            ] = property_forest_live_capability
        return certified_result

    return VerifyResult(
        VerifyStatus.UNKNOWN,
        metadata={
            "reason": (
                "property_forest_incomplete_coverage"
                if property_forest_incomplete_terminal
                else "budget_exhausted_with_unproven_subboxes"
            ),
            "nodes": processed,
            "spec_rows_total": (
                property_forest_total_rows
                if property_forest_enabled
                else None
            ),
            "spec_rows_kept": spec_rows_kept,
            "pool_remaining": pool_remaining,
            "exhausted_budget_time": exhausted_time,
            "exhausted_budget_nodes": exhausted_nodes,
            "nodes_minted": node_counter,
            "any_dropped_frontier_cap": any_dropped_frontier_cap,
            "any_dropped_max_depth": any_dropped_max_depth,
            "joint_gain_groups": int(
                getattr(config, "joint_gain_groups", 1)
            ),
            "property_branch_focus": getattr(
                config, "property_branch_focus", "sum"
            ),
            "branch_requires_unstable_successor": bool(
                getattr(
                    config,
                    "branch_requires_unstable_successor",
                    False,
                )
            ),
            "property_separable_bab": property_forest_enabled,
            "property_forest_root_rows": list(property_forest_row_ids),
            "property_forest_root_count": len(property_forest_row_ids),
            "property_forest_all_solves_single_row": bool(
                property_forest_enabled
            ),
            "property_forest_coverage_complete": (
                property_forest_coverage_complete
            ),
            "property_forest_coverage_by_row": (
                property_forest_coverage_by_row
            ),
            "property_forest_node_conservation_receipt": (
                property_forest_receipt
            ),
            "property_forest_node_conservation_valid": (
                property_forest_receipt_valid
            ),
            "property_forest_node_conservation_errors": list(
                property_forest_receipt_errors
            ),
            "property_forest_root_certified_rows": list(
                property_forest_root_certified_rows
            ),
            "property_forest_live_facts": None,
            "property_forest_live_seal": None,
            "_property_forest_live_capability": None,
            "property_forest_processed_by_row": {
                str(row_id): int(count)
                for row_id, count in sorted(
                    property_forest_processed_by_row.items()
                )
            },
            "property_forest_certified_nodes_by_row": {
                str(row_id): int(count)
                for row_id, count in sorted(
                    property_forest_certified_nodes_by_row.items()
                )
            },
            "nonterminal_filter_calls": nonterminal_filter_calls,
            "nonterminal_filter_applied_calls": (
                nonterminal_filter_applied_calls
            ),
            "nonterminal_filter_fallbacks": nonterminal_filter_fallbacks,
            "nonterminal_filter_selected_layer_ids": sorted(
                nonterminal_filter_selected_layer_ids
            ),
            "frontier_contraction_target": float(
                getattr(config, "frontier_contraction_target", 0.0)
            ),
            "split_depth_histogram": {
                str(depth): int(count)
                for depth, count in sorted(split_depth_histogram.items())
            },
            "child_refine_calls": child_refine_calls,
            "child_refine_seconds": child_refine_seconds,
            "child_refine_strict_lower_entries": (
                child_refine_strict_lower_entries
            ),
            "child_refine_strict_upper_entries": (
                child_refine_strict_upper_entries
            ),
            "child_refine_changed_layers": child_refine_changed_layers,
            "child_refine_queried_objective_rows": (
                child_refine_queried_objective_rows
            ),
            "child_refine_selected_layer_ids": sorted(
                child_refine_selected_layer_ids
            ),
            "joint_gain_probe_calls": joint_gain_probe_calls,
            "joint_gain_probe_nodes": joint_gain_probe_nodes,
            "joint_gain_groups_tested": joint_gain_groups_tested,
            "joint_gain_nonbaseline_lanes": joint_gain_nonbaseline_lanes,
            "joint_gain_more_diverse_lanes": (
                joint_gain_more_diverse_lanes
            ),
            "joint_gain_worst_child_improvement_max": (
                joint_gain_worst_child_improvement_max
            ),
        },
    )


@torch.no_grad()
def verify_bab(
    net: Net,
    solver: Solver,
    config: Optional[BaBConfig] = None,
    *,
    max_depth: Optional[int] = None,
    max_nodes: Optional[int] = None,
    max_subproblems: Optional[int] = None,
    time_budget_s: Optional[float] = None,
    timelimit: Optional[float] = None,
    verbose: bool = False,
) -> VerifyResult:
    """Single-solver Branch-and-Bound entry: one subproblem per iteration.

    Thin wrapper over ``verify_bab_batched`` with K=1. Constructs a solver factory
    from the supplied solver instance's type so each BaB iteration gets a fresh
    instance. Prefer ``verify_bab_batched`` directly for batched (K>1) solving.
    """
    if config is None:
        config = BaBConfig(
            max_depth=max_depth if max_depth is not None else 20,
            max_nodes=(max_nodes or max_subproblems or 2000),
            verbose=verbose,
        )
    budget = (
        time_budget_s if time_budget_s is not None
        else (timelimit if timelimit is not None else 300.0)
    )
    solver_tier = getattr(config, "solver_tier", "lp")
    if solver_tier not in VALID_SOLVER_TIERS:
        raise ValueError(
            f"Unknown solver_tier={solver_tier!r}. Valid: {VALID_SOLVER_TIERS}."
        )
    solver_type = type(solver)
    return verify_bab_batched(
        net=net,
        solver_factory=lambda: solver_type(),
        config=config,
        max_batch_size=1,
        time_budget_s=budget,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Module tests
# ---------------------------------------------------------------------------


class _StubNet:  # pragma: no cover
    layers = []


def test_imports():  # pragma: no cover
    for sym in (
        verify_bab,
        BaBConfig,
        BabNode,
        SubproblemBatch,
        split_subproblems,
        BranchingStrategy,
        BoundingStrategy,
        RandomBranching,
        RandomBounding,
    ):
        assert sym is not None


def test_config_yaml_roundtrip():  # pragma: no cover
    c1 = BaBConfig()
    assert c1.max_depth == 20

    c2 = BaBConfig.from_yaml()
    assert c2.branching_method == "random"

    c3 = BaBConfig.from_yaml(max_depth=50, branching_method="kfsb")
    assert c3.max_depth == 50 and c3.branching_method == "kfsb"

    # Round-trip through a standalone BaB YAML (uses top-level "bab" key)
    tmp = tempfile.mktemp(suffix=".yaml")
    try:
        c3.to_yaml(tmp)
        c4 = BaBConfig.from_yaml(tmp)
        assert c4.max_depth == 50
        assert c4.branching_method == "kfsb"
    finally:
        os.unlink(tmp)

    # BaBConfig must not expose a time_budget_s attribute.
    assert not hasattr(c1, "time_budget_s")


def test_subproblem_batch():  # pragma: no cover
    lb = torch.tensor([[-1.0, -2.0, -3.0]])
    ub = torch.tensor([[1.0, 2.0, 3.0]])
    batch = SubproblemBatch(lb=lb, ub=ub, depths=torch.tensor([0]))

    assert batch.batch_size == 1
    assert batch.input_dim == 3
    assert batch.total_width().item() == 12.0

    bounds = Bounds(lb.squeeze(0), ub.squeeze(0))
    batch2 = SubproblemBatch.from_bounds(bounds)
    assert torch.equal(batch2.lb, lb)

    back = batch2.to_bounds_list()
    assert len(back) == 1
    assert torch.equal(back[0].lb, bounds.lb)


def test_split_subproblems():  # pragma: no cover
    lb = torch.tensor([[-1.0, -2.0, -3.0]])
    ub = torch.tensor([[1.0, 2.0, 3.0]])
    batch = SubproblemBatch(lb=lb, ub=ub, depths=torch.tensor([0]))
    split_dim = torch.tensor([1])

    left, right = split_subproblems(batch, split_dim)

    mid = (lb[0, 1] + ub[0, 1]) / 2
    assert torch.isclose(left.ub[0, 1], mid)
    assert torch.isclose(right.lb[0, 1], mid)
    assert left.depths[0] == 1
    assert right.depths[0] == 1

    assert torch.equal(left.lb[0, 0], lb[0, 0])
    assert torch.equal(right.ub[0, 2], ub[0, 2])


def test_random_branching():  # pragma: no cover
    lb = torch.tensor([[-1.0, -2.0, -3.0]])
    ub = torch.tensor([[1.0, 2.0, 3.0]])
    batch = SubproblemBatch(lb=lb, ub=ub, depths=torch.tensor([0]))

    brancher = RandomBranching()
    scores = brancher.compute_scores(batch, cast(Net, cast(object, _StubNet())))
    assert scores.shape == (1, 3)
    assert (scores >= 0).all()

    dims = cast(torch.Tensor, brancher.select(scores))
    assert dims.shape == (1,)
    assert 0 <= dims.item() <= 2


def test_random_branching_with_mask():  # pragma: no cover
    lb = torch.tensor([[-1.0, -2.0, -3.0]])
    ub = torch.tensor([[1.0, 2.0, 3.0]])
    batch = SubproblemBatch(lb=lb, ub=ub, depths=torch.tensor([0]))
    mask = torch.tensor([False, True, False])

    brancher = RandomBranching()
    scores = brancher.compute_scores(batch, cast(Net, cast(object, _StubNet())), unstable_mask=mask)
    assert scores[0, 0].item() == 0.0
    assert scores[0, 2].item() == 0.0
    assert cast(torch.Tensor, brancher.select(scores)).item() == 1


def test_random_bounding():  # pragma: no cover
    lb = torch.tensor([[-1.0, -2.0], [0.0, 0.0]])
    ub = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    batch = SubproblemBatch(lb=lb, ub=ub, depths=torch.tensor([0, 1]))

    pool = RandomBounding()
    assert pool.empty

    pool.push(batch)
    assert len(pool) == 2

    popped = pool.pop(1)
    assert popped.batch_size == 1
    assert len(pool) == 1

    pool.pop(1)
    assert pool.empty


def test_babnode_compat():  # pragma: no cover
    bounds = Bounds(torch.tensor([-1.0, -2.0]), torch.tensor([1.0, 2.0]))
    node = BabNode(box=bounds, depth=3, score=0.5)
    batch = node.to_batch()
    assert batch.batch_size == 1
    assert batch.depths[0].item() == 3


class _IdentityOutput(torch.nn.Module):  # pragma: no cover
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)


def _make_assert_layer(kind: str, params: dict[str, ParamValue], n_out: int) -> Layer:  # pragma: no cover
    from act.back_end.layer_schema import LayerKind

    merged: dict[str, ParamValue] = {"kind": kind}
    merged.update(params)
    if "C" not in merged or "thresholds" not in merged or "M" not in merged:
        batch_size = 1
        for key in ("y_true", "margin", "c", "d", "lb", "ub"):
            value = merged.get(key)
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                batch_size = max(batch_size, int(value.shape[0]))
        if kind == OutKind.UNSAFE_LINEAR:
            c_value = merged.get("c")
            d_value = merged.get("d")
            if not isinstance(c_value, torch.Tensor) or not isinstance(d_value, torch.Tensor):
                raise ValueError("UNSAFE_LINEAR test layer requires tensor c and d")
            if c_value.dim() == 3:
                batch_size = int(c_value.shape[0])
                m_rows = int(c_value.shape[1])
                merged["C"] = c_value.reshape(batch_size * m_rows, n_out)
            elif c_value.dim() == 2:
                m_rows = int(c_value.shape[0])
                merged["C"] = c_value
            else:
                raise ValueError(f"UNSAFE_LINEAR test c dim {c_value.dim()} unsupported")
            merged["thresholds"] = d_value.reshape(batch_size, m_rows)
            merged["M"] = m_rows
        else:
            merged["C"] = torch.zeros(batch_size, n_out)
            merged["thresholds"] = torch.zeros(batch_size, 1)
            merged["M"] = 1
    return Layer(
        id=99,
        kind=LayerKind.ASSERT.value,
        params=merged,
        in_vars=list(range(n_out)),
        out_vars=list(range(n_out)),
    )


def _test_check_violations_batched_per_kind():  # pragma: no cover
    y = torch.tensor(
        [
            [3.0, 1.0, 0.0, -1.0],
            [0.0, 2.0, 1.0, -1.0],
            [0.0, 3.0, 1.0, -1.0],
            [0.0, 1.0, 3.0, -1.0],
            [0.0, 1.0, 4.0, -1.0],
            [0.0, 1.0, 2.0, 5.0],
            [0.0, 1.0, 2.0, 6.0],
            [4.0, 1.0, 2.0, 3.0],
        ],
        dtype=torch.float64,
    )
    net = _IdentityOutput()
    n_batch, n_out = y.shape

    top1 = _make_assert_layer(
        OutKind.TOP1_ROBUST,
        {"y_true": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])},
        n_out,
    )
    y_true_top1 = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    expected_top1 = y.argmax(dim=1) != y_true_top1
    assert torch.equal(check_violations_batched(net, y, top1), expected_top1)

    margin = _make_assert_layer(
        OutKind.MARGIN_ROBUST,
        {
            "y_true": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
            "margin": torch.full((n_batch,), 1.5, dtype=y.dtype),
        },
        n_out,
    )
    y_true = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    true_scores = y.gather(1, y_true.unsqueeze(1)).squeeze(1)
    mask = torch.ones_like(y, dtype=torch.bool)
    _ = mask.scatter_(1, y_true.unsqueeze(1), False)
    expected_margin = (y.masked_fill(~mask, -float("inf")).max(dim=1).values - true_scores) >= 1.5
    assert torch.equal(check_violations_batched(net, y, margin), expected_margin)

    linear = _make_assert_layer(
        OutKind.LINEAR_LE,
        {"c": torch.ones(n_batch, n_out, dtype=y.dtype), "d": torch.full((n_batch,), 4.0, dtype=y.dtype)},
        n_out,
    )
    expected_linear = y.sum(dim=1) >= 4.0 + 1e-8
    assert torch.equal(check_violations_batched(net, y, linear), expected_linear)

    range_layer = _make_assert_layer(
        OutKind.RANGE,
        {
            "lb": torch.full((n_batch, n_out), -0.5, dtype=y.dtype),
            "ub": torch.full((n_batch, n_out), 4.5, dtype=y.dtype),
        },
        n_out,
    )
    expected_range = ((y < -0.5 - 1e-8) | (y > 4.5 + 1e-8)).any(dim=1)
    assert torch.equal(check_violations_batched(net, y, range_layer), expected_range)

    c = torch.eye(n_out, dtype=y.dtype).unsqueeze(0).expand(n_batch, -1, -1).contiguous()
    d = torch.full((n_batch, n_out), 3.5, dtype=y.dtype)
    unsafe = _make_assert_layer(
        OutKind.UNSAFE_LINEAR,
        {"c": c, "d": d, "C": c.reshape(n_batch * n_out, n_out), "thresholds": d, "M": n_out},
        n_out,
    )
    expected_unsafe = (y <= 3.5 + 1e-8).all(dim=1)
    assert torch.equal(check_violations_batched(net, y, unsafe), expected_unsafe)



def _test_check_violations_batched_b1_scalar_params():  # pragma: no cover
    net = _IdentityOutput()
    x = torch.tensor([[0.0, 2.0, 1.0]], dtype=torch.float64)
    assert_layer = _make_assert_layer(
        OutKind.TOP1_ROBUST,
        {"y_true": torch.tensor([0], dtype=torch.long)},
        n_out=3,
    )
    result = check_violations_batched(net, x, assert_layer)
    assert tuple(result.shape) == (1,)
    assert bool(result[0].item()) is True



# ---------------------------------------------------------------------------
# C12: K-batched verify_bab_batched test fixtures
# ---------------------------------------------------------------------------


def _load_bab_deep_net() -> Optional[Net]:  # pragma: no cover
    """Load layer_testing_bab_deep.json from examples/nets, or None if absent.

    Returns None silently when the fixture is missing so tests can skip rather
    than hard-fail in isolated environments. Forces CPU device for hermetic
    test execution: the BaB integration tests must not depend on GPU
    availability or device-manager global state.
    """
    from pathlib import Path

    from act.back_end.serialization.serialization import load_net_from_file
    from act.util.device_manager import initialize_device

    here = Path(__file__).resolve()
    candidate = here.parents[1] / "examples" / "nets" / "layer_testing_bab_deep.json"
    if not candidate.exists():
        return None
    initialize_device("cpu", "float64")
    return load_net_from_file(str(candidate), target_device="cpu")


class _UnknownSolver(Solver):  # pragma: no cover
    """Mock solver: returns UNKNOWN on every lane (forces BaB to branch)."""

    def solve_batch(self, problem, timelimit=None):
        from act.back_end.solver.solver_base import BatchLPSolution

        n = problem.N
        return BatchLPSolution(
            statuses=tuple([SolveStatus.UNKNOWN] * n),
            x=torch.zeros(
                (n, problem.nvars), device=problem.lb.device, dtype=problem.lb.dtype,
            ),
            max_viol=torch.full(
                (n,), float("nan"), device=problem.lb.device, dtype=problem.lb.dtype,
            ),
        )


class _OOMSolver(Solver):  # pragma: no cover
    """Mock solver: raises an OOM-like exception on every solve_batch call."""

    def solve_batch(self, problem, timelimit=None):
        raise RuntimeError("CUDA out of memory: mocked for OOM-fails-loud test")


def _test_bab_kbatch_status_parity():  # pragma: no cover
    net = _load_bab_deep_net()
    if net is None:
        print("  SKIP _test_bab_kbatch_status_parity: layer_testing_bab_deep.json absent")
        return
    from act.back_end.solver.solver_torchlp import TorchLPSolver

    config = BaBConfig(max_depth=6, max_nodes=32, verbose=False)
    statuses_by_k: dict[int, VerifyStatus] = {}
    for k in (1, 2, 4, 8):
        result = verify_bab_batched(
            net=net,
            solver_factory=lambda: TorchLPSolver(),
            config=config,
            max_batch_size=k,
            time_budget_s=60.0,
        )
        statuses_by_k[k] = result.status
    distinct = set(statuses_by_k.values())
    assert len(distinct) == 1, (
        f"K-batch status parity violated: {statuses_by_k}"
    )


def _test_bab_budget_exhaustion_returns_unknown():  # pragma: no cover
    net = _load_bab_deep_net()
    if net is None:
        print("  SKIP _test_bab_budget_exhaustion_returns_unknown: fixture absent")
        return
    config = BaBConfig(max_depth=10, max_nodes=2, verbose=False)
    result = verify_bab_batched(
        net=net,
        solver_factory=lambda: _UnknownSolver(),
        config=config,
        max_batch_size=1,
        time_budget_s=30.0,
    )
    assert result.status == VerifyStatus.UNKNOWN, (
        f"Expected UNKNOWN under-budget with mock-UNKNOWN solver, got "
        f"{result.status}; metadata={result.metadata}"
    )
    assert result.metadata.get("reason") == "budget_exhausted_with_unproven_subboxes", (
        f"Missing soundness-reason metadata: {result.metadata}"
    )


def _test_bab_oom_fails_loud():  # pragma: no cover
    net = _load_bab_deep_net()
    if net is None:
        print("  SKIP _test_bab_oom_fails_loud: fixture absent")
        return
    config = BaBConfig(max_depth=5, max_nodes=10, verbose=False)
    raised = False
    try:
        verify_bab_batched(
            net=net,
            solver_factory=lambda: _OOMSolver(),
            config=config,
            max_batch_size=4,
            time_budget_s=10.0,
        )
    except RuntimeError as e:
        msg = str(e).lower()
        assert "out of memory" in msg, f"Unexpected RuntimeError message: {e}"
        raised = True
    assert raised, "OOM exception was swallowed — silent fallback present"


def _test_bab_k_fluctuates():  # pragma: no cover
    net = _load_bab_deep_net()
    if net is None:
        print("  SKIP _test_bab_k_fluctuates: fixture absent")
        return
    config = BaBConfig(max_depth=8, max_nodes=20, verbose=False)
    k_log: List[int] = []
    _ = verify_bab_batched(
        net=net,
        solver_factory=lambda: _UnknownSolver(),
        config=config,
        max_batch_size=8,
        time_budget_s=30.0,
        _k_log=k_log,
    )
    distinct = set(k_log)
    assert len(distinct) >= 2, (
        f"K did not fluctuate across iterations (got {k_log}); dynamic K-batching "
        f"requires at least 2 distinct K values per D4."
    )


_TESTS = [  # pragma: no cover
    test_imports,
    test_config_yaml_roundtrip,
    test_subproblem_batch,
    test_split_subproblems,
    test_random_branching,
    test_random_branching_with_mask,
    test_random_bounding,
    test_babnode_compat,
    _test_check_violations_batched_per_kind,
    _test_check_violations_batched_b1_scalar_params,
    _test_bab_kbatch_status_parity,
    _test_bab_budget_exhaustion_returns_unknown,
    _test_bab_oom_fails_loud,
    _test_bab_k_fluctuates,
]


def run_all_tests() -> int:
    passed = failed = 0
    for fn in _TESTS:
        try:
            fn()
            passed += 1
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    print("Running BaB module tests\n")
    sys.exit(run_all_tests())
