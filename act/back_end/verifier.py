#===- act/back_end/verifier.py - Spec-free Verification Engine ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Spec-free, input-free verification. Assumes the ACT Net already encodes
#   both input and output specifications via INPUT_SPEC and ASSERT layers
#   (produced by torch2act.TorchToACT).
#
# Architecture — verify_once:
#   1. Seed [B, *input_shape] bounds from INPUT_SPEC layers (no CSP).
#   2. analyze() propagates batched bounds through every TF op.
#   3. Read pre-encoded [B*M, n_out] linear-form C / [B, M] thresholds / M
#      from the ASSERT layer params (produced upstream by
#      OutputSpec.encode_linear at FE construction time).
#   4. INTERVAL CERTIFICATION: one tensor pass computes margin_max under
#      output bounds; sample b is CERTIFIED iff every M lane passes.
#   5. FALSIFICATION EVIDENCE: either the optional concrete-model replay or a
#      verifier-owned exact phase projection can establish a counterexample.
#      The latter proves raw-BOX membership and the property with a zero-width
#      forward interval plus exact stored-float arithmetic; it does not sample
#      inputs, execute ONNX, or invoke PGD.
#   6. Return List[VerifyResult] of length B (one per input lane).
#
#===---------------------------------------------------------------------===#

# Public API:
#   - verify_once(net, *, model_fn=None, collect_facts=False)
#       Pure-tensor batched single-shot verifier. Returns List[VerifyResult]
#       by default, or (results, facts_or_none) when collect_facts=True.
#   - setup_and_solve_batch(net, input_bounds_per_b, solver, timelimit=None)
#       Batch-native CSP setup helper used by LP and BaB refinement.
#   - find_entry_layer_id / get_input_ids / get_output_ids /
#     gather_input_spec_layers / get_assert_layer / seed_from_input_specs /
#     add_all_input_specs (helpers).
#
# Notes:
#   * Spec-free verification: all constraints extracted from ACT Net layers.
#   * verify_once returns one VerifyResult per lane (len(result) == B).
#   * INPUT_SPEC constraints (including LIN_POLY) are propagated through
#     analyze(); they enter via add_all_input_specs into entry_fact.cons.
#     LIN_POLY constraints are not consumed by verify_once's interval
#     certification; they are preserved for the batch-native solver path.

from __future__ import annotations
from typing import (
    Optional,
    List,
    Callable,
    Dict,
    Any,
    TYPE_CHECKING,
    Tuple,
    Literal,
    Mapping,
    overload,
)

import torch
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import time
import numpy as np

# ACT backend imports
from act.back_end.core import Bounds, Con, ConSet, Fact, Net
from act.back_end.solver.solver_base import Solver, SolveStatus, BatchLPSolution
from act.back_end.layer_schema import LayerKind
from act.back_end.utils import validate_constraints

if TYPE_CHECKING:
    from act.back_end.analyze import AnalyzeCache

# Front-end enums (kinds)
from act.front_end.specs import InKind, OutKind, OutputSpec, normalize_position_mask

# Verification types (canonical location: act/util/stats.py)
from act.util.stats import VerifyStatus, VerifyResult


# -----------------------------------------------------------------------------
# Exact binary-phase cover firewall
# -----------------------------------------------------------------------------

def _audit_sparse_binary_phase_cover(
    parent: Any,
    cover: Any,
    *,
    phase_depth: int,
    deadline: Optional[float] = None,
) -> Dict[str, Any]:
    """Validate the complete private contract behind a phase-cover SAFE.

    The enumerator is an internal proof-producing routine, but the verifier is
    the component that promotes all child SAFE results to a parent SAFE.  This
    independent audit therefore checks completeness, uniqueness, canonical
    assignment/child binding, and the unforgeable exact-cover capability before
    any child solver is called.
    """

    from act.back_end.solver.solver_hz import (
        SparseHZono,
        _hz_exact_phase_cover_member,
        hz_constructively_nonempty,
        hz_verify_sparse_binary_phase_child,
    )

    if (
        isinstance(phase_depth, (bool, np.bool_))
        or not isinstance(phase_depth, (int, np.integer))
    ):
        raise TypeError("phase-cover depth must be an integer")
    phase_depth = int(phase_depth)
    if not isinstance(parent, SparseHZono):
        raise TypeError("phase-cover parent must be SparseHZono")
    if phase_depth <= 0 or phase_depth != int(parent.n_bin):
        raise ValueError(
            "phase-cover depth must equal the parent binary-factor count"
        )
    if not isinstance(cover, tuple):
        raise TypeError("phase cover must be a tuple")
    if deadline is not None and time.monotonic() >= float(deadline):
        raise TimeoutError(
            "binary phase deadline expired before cover audit"
        )

    expected_count = 1 << phase_depth
    if len(cover) != expected_count:
        raise ValueError(
            "phase cover is incomplete: "
            f"expected={expected_count}, actual={len(cover)}"
        )

    expected_positions = tuple(range(phase_depth))
    if parent.bcol_ids is None:
        expected_bcol_ids: Tuple[Optional[int], ...] = (
            (None,) * phase_depth
        )
    else:
        parent_bcol_ids = np.asarray(
            parent.bcol_ids, dtype=np.int64
        ).reshape(-1)
        if parent_bcol_ids.size != phase_depth:
            raise ValueError(
                "phase-cover parent binary-column IDs are malformed"
            )
        expected_bcol_ids = tuple(
            int(value) for value in parent_bcol_ids.tolist()
        )

    parent_constructive = bool(hz_constructively_nonempty(parent))
    assignments: List[Tuple[int, ...]] = []
    for index, raw_item in enumerate(cover):
        if deadline is not None and time.monotonic() >= float(deadline):
            raise TimeoutError(
                "binary phase deadline expired before auditing "
                f"child {index}"
            )
        if not isinstance(raw_item, tuple) or len(raw_item) != 2:
            raise TypeError(
                f"phase-cover item {index} must be an assignment/child tuple"
            )
        raw_assignment, child = raw_item
        if not isinstance(raw_assignment, tuple):
            raise TypeError(
                f"phase-cover assignment {index} must be a tuple"
            )
        if len(raw_assignment) != phase_depth:
            raise ValueError(
                f"phase-cover assignment {index} has the wrong depth"
            )

        positions: List[int] = []
        signs: List[int] = []
        for raw_pair in raw_assignment:
            if not isinstance(raw_pair, tuple) or len(raw_pair) != 2:
                raise TypeError(
                    "phase-cover assignment entries must be position/value "
                    "tuples"
                )
            raw_position, raw_sign = raw_pair
            if (
                isinstance(raw_position, (bool, np.bool_))
                or not isinstance(raw_position, (int, np.integer))
                or isinstance(raw_sign, (bool, np.bool_))
                or not isinstance(raw_sign, (int, np.integer))
            ):
                raise TypeError(
                    "phase-cover positions and values must be strict integers"
                )
            position = int(raw_position)
            sign = int(raw_sign)
            if sign not in {-1, 1}:
                raise ValueError(
                    "phase-cover values must be exactly -1 or +1"
                )
            positions.append(position)
            signs.append(sign)
        if tuple(positions) != expected_positions:
            raise ValueError(
                f"phase-cover assignment {index} has non-canonical positions"
            )
        sign_tuple = tuple(signs)
        assignments.append(sign_tuple)

        if not isinstance(child, SparseHZono):
            raise TypeError(
                f"phase-cover child {index} must be SparseHZono"
            )
        if hz_verify_sparse_binary_phase_child(
            parent,
            raw_assignment,
            child,
            deadline=deadline,
        ) is not True:
            raise ValueError(
                f"phase-cover child {index} failed live projection audit"
            )
        if int(child.n_bin) != 0:
            raise ValueError(
                f"phase-cover child {index} retained binary factors"
            )
        child_bcol_ids = (
            ()
            if child.bcol_ids is None
            else tuple(
                int(value)
                for value in np.asarray(
                    child.bcol_ids, dtype=np.int64
                ).reshape(-1).tolist()
            )
        )
        if child_bcol_ids:
            raise ValueError(
                f"phase-cover child {index} retained binary-column IDs"
            )

        fix = getattr(child, "_solver_binary_phase_fix", None)
        if not isinstance(fix, Mapping):
            raise ValueError(
                f"phase-cover child {index} has no phase-fix receipt"
            )
        if (
            fix.get("schema") != "sparse_hz_binary_phase_fix_v2"
            or fix.get("proof_authority") is not True
            or fix.get("proof_rule")
            != (
                "exact_fraction_fixed_binary_substitution_with_explicit_"
                "center_and_equality_roundoff_generators_and_upper_rhs_"
                "rounding_toward_positive_infinity;all_sign_assignments_"
                "form_sound_parent_cover"
            )
            or fix.get("projection_relation")
            != "exact_fixed_phase_projection_subset_of_child"
            or fix.get("arithmetic")
            != "Fraction.from_float_exact_dyadic"
            or isinstance(fix.get("parent_n_bin"), (bool, np.bool_))
            or not isinstance(
                fix.get("parent_n_bin"), (int, np.integer)
            )
            or int(fix["parent_n_bin"]) != phase_depth
            or isinstance(fix.get("child_n_bin"), (bool, np.bool_))
            or not isinstance(
                fix.get("child_n_bin"), (int, np.integer)
            )
            or int(fix["child_n_bin"]) != 0
            or isinstance(fix.get("parent_n_cont"), (bool, np.bool_))
            or not isinstance(
                fix.get("parent_n_cont"), (int, np.integer)
            )
            or int(fix["parent_n_cont"]) != int(parent.n_cont)
            or isinstance(fix.get("child_n_cont"), (bool, np.bool_))
            or not isinstance(
                fix.get("child_n_cont"), (int, np.integer)
            )
            or not (
                int(parent.n_cont)
                <= int(fix["child_n_cont"])
                <= int(child.n_cont)
            )
        ):
            raise ValueError(
                f"phase-cover child {index} has an invalid phase-fix receipt"
            )

        def strict_int_tuple(value: Any, field: str) -> Tuple[int, ...]:
            if not isinstance(value, list):
                raise TypeError(
                    f"phase-cover child {index} {field} must be a list"
                )
            if any(
                isinstance(item, (bool, np.bool_))
                or not isinstance(item, (int, np.integer))
                for item in value
            ):
                raise TypeError(
                    f"phase-cover child {index} {field} must contain strict "
                    "integers"
                )
            return tuple(int(item) for item in value)

        if strict_int_tuple(
            fix.get("fixed_positions"), "fixed_positions"
        ) != expected_positions:
            raise ValueError(
                f"phase-cover child {index} fixed positions disagree"
            )
        if strict_int_tuple(
            fix.get("fixed_values"), "fixed_values"
        ) != sign_tuple:
            raise ValueError(
                f"phase-cover child {index} is bound to another assignment"
            )
        raw_fixed_ids = fix.get("fixed_bcol_ids")
        if not isinstance(raw_fixed_ids, list) or len(
            raw_fixed_ids
        ) != phase_depth:
            raise TypeError(
                f"phase-cover child {index} fixed_bcol_ids are malformed"
            )
        normalized_fixed_ids: List[Optional[int]] = []
        for value in raw_fixed_ids:
            if value is None:
                normalized_fixed_ids.append(None)
            elif (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
            ):
                raise TypeError(
                    f"phase-cover child {index} fixed_bcol_ids are malformed"
                )
            else:
                normalized_fixed_ids.append(int(value))
        if tuple(normalized_fixed_ids) != expected_bcol_ids:
            raise ValueError(
                f"phase-cover child {index} fixed binary IDs disagree"
            )

        child_cover_capability = bool(
            _hz_exact_phase_cover_member(child)
        )
        if child_cover_capability != parent_constructive:
            raise ValueError(
                f"phase-cover child {index} exact-cover capability disagrees "
                "with its parent"
            )

    expected_assignments = {
        tuple(
            1 if ((bits >> position) & 1) else -1
            for position in expected_positions
        )
        for bits in range(expected_count)
    }
    if (
        len(set(assignments)) != expected_count
        or set(assignments) != expected_assignments
    ):
        raise ValueError(
            "phase cover assignments are duplicate or incomplete"
        )

    return {
        "schema": "verifier_sparse_binary_phase_cover_audit_v1",
        "proof_authority": True,
        "expected_child_count": int(expected_count),
        "actual_child_count": int(len(cover)),
        "canonical_positions": [
            int(position) for position in expected_positions
        ],
        "unique_assignment_count": int(len(set(assignments))),
        "all_assignments_enumerated": True,
        "all_children_assignment_bound": True,
        "all_child_capabilities_valid": True,
        "all_children_live_projection_valid": True,
        "parent_constructively_nonempty": bool(parent_constructive),
    }


def _audit_live_operator_property_micro_rlt(
    hz: Any,
    operator_receipt: Any,
) -> bool:
    """Bind the operator's outer receipt to the live lifted sparse HZ."""

    try:
        from act.back_end.hybridz_tf.property_micro_rlt import (
            PropertyMicroRLTResult,
            verify_property_micro_rlt_result,
        )

        if (
            not isinstance(operator_receipt, Mapping)
            or operator_receipt.get("status") != "applied"
            or operator_receipt.get("proof_authority") is not True
            or operator_receipt.get("live_result_validation_passed")
            is not True
            or operator_receipt.get("scope") != "parent_pre_phase_fix"
        ):
            return False
        outer_hash = operator_receipt.get("receipt_sha256")
        if (
            not isinstance(outer_hash, str)
            or len(outer_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in outer_hash
            )
        ):
            return False
        outer_payload = dict(operator_receipt)
        del outer_payload["receipt_sha256"]
        if _canonical_receipt_sha256(outer_payload) != outer_hash:
            return False

        attached = getattr(hz, "_property_micro_rlt_receipt", None)
        if not isinstance(attached, Mapping):
            return False
        attached = dict(attached)
        if (
            operator_receipt.get(
                "property_micro_rlt_receipt_sha256"
            )
            != attached.get("receipt_sha256")
            or verify_property_micro_rlt_result(
                PropertyMicroRLTResult(hz=hz, receipt=attached)
            )
            is not True
        ):
            return False

        base_counts = operator_receipt.get("base_counts")
        result_counts = operator_receipt.get("result_counts")
        if not isinstance(base_counts, Mapping) or not isinstance(
            result_counts, Mapping
        ):
            return False

        def strict_int(mapping: Mapping[str, Any], key: str) -> int:
            value = mapping[key]
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(key)
            return int(value)

        packet_mode = operator_receipt.get("requested_packet_mode")
        expected_packet_indices = {
            "both": [0, 1],
            "first": [0],
            "second": [1],
        }.get(packet_mode)
        exact_records = operator_receipt.get("exact_relu_records")
        selected_packet_indices = operator_receipt.get(
            "selected_packet_record_indices"
        )
        outer_selection = operator_receipt.get(
            "source_rows_by_binary"
        )
        inner_selection = attached.get("selection")
        if (
            expected_packet_indices is None
            or selected_packet_indices != expected_packet_indices
            or strict_int(
                operator_receipt, "selected_packet_count"
            )
            != len(expected_packet_indices)
            or not isinstance(exact_records, list)
            or len(exact_records) != 2
            or any(
                not isinstance(record, Mapping)
                for record in exact_records
            )
            or not isinstance(outer_selection, list)
            or outer_selection != inner_selection
        ):
            return False
        expected_selection = []
        for index in expected_packet_indices:
            own = exact_records[index]
            other = exact_records[1 - index]
            expected_selection.append(
                {
                    "binary_position": strict_int(
                        own, "binary_position"
                    ),
                    "source_upper_rows": sorted(
                        [
                            strict_int(own, "lower_upper_row"),
                            strict_int(own, "x_branch_upper_row"),
                            strict_int(own, "zero_branch_upper_row"),
                            strict_int(other, "lower_upper_row"),
                        ]
                    ),
                }
            )
        expected_selection.sort(
            key=lambda entry: entry["binary_position"]
        )
        if outer_selection != expected_selection:
            return False

        dimensions = ("n_out", "n_cont", "n_bin", "n_eq", "n_ub")
        for name in dimensions:
            if strict_int(result_counts, name) != strict_int(
                attached, f"result_{name}"
            ):
                return False
            if strict_int(base_counts, name) != strict_int(
                attached, f"base_{name}"
            ):
                return False
        if strict_int(
            operator_receipt, "new_product_factors"
        ) != strict_int(attached, "new_product_factors"):
            return False
        if (
            operator_receipt.get("requirement_count_complete") is not True
            or operator_receipt.get(
                "selected_source_nnz_cap_exceeded"
            )
            is not False
            or operator_receipt.get("product_factor_cap_exceeded")
            is not False
            or operator_receipt.get("primary_cap_failure") is not None
            or strict_int(
                operator_receipt,
                "required_selected_source_row_nnz",
            )
            != strict_int(attached, "selected_source_row_nnz")
            or strict_int(
                operator_receipt, "required_product_factors"
            )
            != strict_int(attached, "new_product_factors")
        ):
            return False
        if strict_int(
            operator_receipt, "new_upper_rows"
        ) != strict_int(attached, "new_upper_rows"):
            return False
        generated_tags = operator_receipt.get(
            "generated_upper_row_tags"
        )
        if (
            not isinstance(generated_tags, list)
            or any(
                not isinstance(value, str)
                or not value.startswith("property_micro_rlt:")
                for value in generated_tags
            )
            or len(generated_tags)
            != strict_int(operator_receipt, "new_upper_rows")
        ):
            return False
        return True
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return False


# -----------------------------------------------------------------------------
# Sequential per-sample slicing (for B>1 BaB)
# -----------------------------------------------------------------------------

def _slice_first_dim(value: Any, sample_idx: int, expected_b: int) -> Any:
    if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == expected_b:
        return value[sample_idx:sample_idx + 1]
    return value


def slice_net_to_sample(net: Net, sample_idx: int) -> Net:
    from act.front_end.spec_creator_base import LabeledInputTensor

    mutable_kinds = {
        LayerKind.INPUT.value,
        LayerKind.INPUT_SPEC.value,
        LayerKind.ASSERT.value,
    }
    layers = []
    for layer in net.layers:
        if layer.kind not in mutable_kinds:
            layers.append(layer)
            continue
        layer2 = copy.copy(layer)
        layer2.params = dict(layer.params)
        layer2.in_vars = list(layer.in_vars)
        layer2.out_vars = list(layer.out_vars)
        layer2.cache = dict(layer.cache)
        layers.append(layer2)
    net2 = copy.copy(net)
    net2.layers = layers
    net2.preds = net.preds
    net2.succs = net.succs
    net2.by_id = {layer.id: layer for layer in layers}

    entry_id = find_entry_layer_id(net2)
    input_layer = net2.by_id[entry_id]
    shape = input_layer.params.get("shape") or []
    shape_t = tuple(shape) if isinstance(shape, (list, tuple)) else ()
    B = int(shape_t[0]) if shape_t else 1
    if shape_t and int(shape_t[0]) == B:
        input_layer.params["shape"] = (1,) + tuple(shape_t[1:])
    li = input_layer.params.get("labeled_input")
    if isinstance(li, LabeledInputTensor):
        new_tensor = _slice_first_dim(li.tensor, sample_idx, B)
        new_label = _slice_first_dim(li.label, sample_idx, B) if li.label is not None else None
        input_layer.__dict__["params"]["labeled_input"] = LabeledInputTensor(
            tensor=new_tensor, label=new_label,
        )

    for spec_layer in gather_input_spec_layers(net2):
        for key in ("center", "eps", "lb", "ub", "A", "b"):
            val = spec_layer.params.get(key)
            if val is not None:
                spec_layer.params[key] = _slice_first_dim(val, sample_idx, B)

    assert_layer = get_assert_layer(net2)
    m_raw = assert_layer.params.get("M", 1)
    if isinstance(m_raw, torch.Tensor):
        m_rows = int(m_raw.item())
    elif isinstance(m_raw, int):
        m_rows = m_raw
    else:
        raise ValueError(f"ASSERT M must be int or tensor, got {m_raw!r}")
    for key in ("y_true", "margin", "c", "d", "lb", "ub"):
        val = assert_layer.params.get(key)
        if val is not None:
            assert_layer.params[key] = _slice_first_dim(val, sample_idx, B)
    # C is [B*M, n_out] — first dim is B*M not B, so slice rows manually
    c_big = assert_layer.params.get("C")
    if isinstance(c_big, torch.Tensor) and c_big.shape[0] == B * m_rows:
        assert_layer.params["C"] = c_big[sample_idx * m_rows:(sample_idx + 1) * m_rows]
    thresholds = assert_layer.params.get("thresholds")
    if isinstance(thresholds, torch.Tensor) and thresholds.shape[0] == B:
        assert_layer.params["thresholds"] = thresholds[sample_idx:sample_idx + 1]

    return net2


# -----------------------------------------------------------------------------
# ACT Net extraction helpers
# -----------------------------------------------------------------------------

def find_entry_layer_id(net) -> int:
    """Return the id of the single INPUT layer."""
    candidates = [L.id for L in net.layers if L.kind == LayerKind.INPUT.value]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one INPUT layer, found {len(candidates)}.")
    return candidates[0]

def get_input_ids(net) -> List[int]:
    """Return input variable IDs (out_vars of INPUT layer)."""
    entry = find_entry_layer_id(net)
    return list(net.by_id[entry].out_vars)

def get_output_ids(net) -> List[int]:
    """Return output variable IDs (in_vars of ASSERT layer)."""
    assert_layer = net.layers[-1]
    if assert_layer.kind != LayerKind.ASSERT.value:
        raise ValueError("Expected last layer to be ASSERT.")
    return list(assert_layer.in_vars)

def gather_input_spec_layers(net):
    """Return list of INPUT_SPEC layers."""
    return [L for L in net.layers if L.kind == LayerKind.INPUT_SPEC.value]

def get_assert_layer(net):
    """Return the ASSERT layer (must be last)."""
    assert_layer = net.layers[-1]
    if assert_layer.kind != LayerKind.ASSERT.value:
        raise ValueError("Expected last layer to be ASSERT.")
    return assert_layer

# -----------------------------------------------------------------------------
# Seed and input spec helpers
# -----------------------------------------------------------------------------

def seed_from_input_specs(spec_layers) -> Bounds:
    """
    Create seed Bounds from INPUT_SPEC layers.
    Prefers BOX, then LINF_BALL, raises if only LIN_POLY exists.

    Note: This extracts only box bounds for seeding abstract interpretation.
    All constraints (including LIN_POLY) are added via add_all_input_specs().
    """
    # BOX first
    for spec_layer in spec_layers:
        if spec_layer.params.get("kind") == InKind.BOX and "lb" in spec_layer.params and "ub" in spec_layer.params:
            return Bounds(spec_layer.params["lb"].clone(), spec_layer.params["ub"].clone())

    # LINF_BALL next
    for spec_layer in spec_layers:
        if spec_layer.params.get("kind") == InKind.LINF_BALL:
            if "lb" in spec_layer.params and "ub" in spec_layer.params:
                return Bounds(spec_layer.params["lb"].clone(), spec_layer.params["ub"].clone())
            center = spec_layer.params.get("center")
            eps = spec_layer.params.get("eps")
            if center is not None and eps is not None:
                e = eps.to(device=center.device, dtype=center.dtype) if torch.is_tensor(eps) else center.new_tensor(eps)
                return Bounds(center - e, center + e)

    # LP_EMBEDDING seeds the enclosing box; finite-p precision is recovered by
    # the dual input contribution, which reads p_norm/perturbed_positions.
    for spec_layer in spec_layers:
        if spec_layer.params.get("kind") == InKind.LP_EMBEDDING:
            if "lb" in spec_layer.params and "ub" in spec_layer.params:
                return Bounds(spec_layer.params["lb"].clone(), spec_layer.params["ub"].clone())
            center = spec_layer.params.get("center")
            eps = spec_layer.params.get("eps")
            if center is None or eps is None:
                raise ValueError("LP_EMBEDDING requires center/eps or lb/ub for seeding.")
            e = eps.to(device=center.device, dtype=center.dtype) if torch.is_tensor(eps) else center.new_tensor(eps)
            lb = center.clone()
            ub = center.clone()
            mask = normalize_position_mask(
                spec_layer.params.get("perturbed_positions"),
                int(center.shape[-2]),
                batch_shape=tuple(center.shape[:-2]),
                device=center.device,
            )
            expanded = mask.unsqueeze(-1).expand_as(center)
            return Bounds(torch.where(expanded, center - e, lb), torch.where(expanded, center + e, ub))
    
    # LIN_POLY only -> error
    if any(spec_layer.params.get("kind") == InKind.LIN_POLY for spec_layer in spec_layers):
        raise ValueError("LIN_POLY requires a seed box (BOX or LINF_BALL).")

    raise ValueError("No valid input specification found for seeding.")

def add_all_input_specs(globalC: ConSet, input_ids: List[int], spec_layers) -> None:
    """
    Add all INPUT_SPEC constraints to constraint set.

    This function adds:
    - BOX constraints (box bounds)
    - LINF_BALL constraints (converted to box)
    - LP_EMBEDDING/LIN_POLY constraints (box seed or linear polytope A·x ≤ b)
    
    The LIN_POLY constraints are tagged with "in:linpoly" and will be
    exported by export_to_batch_problem() in cons_exportor.py.
    """
    for L in spec_layers:
        k = L.params.get("kind")
        if k == InKind.BOX:
            globalC.add_box(-1, input_ids, Bounds(L.params["lb"], L.params["ub"]))
        elif k == InKind.LINF_BALL:
            if "lb" in L.params and "ub" in L.params:
                globalC.add_box(-1, input_ids, Bounds(L.params["lb"], L.params["ub"]))
            else:
                center = L.params["center"]
                eps = L.params["eps"]
                e = eps.to(device=center.device, dtype=center.dtype) if torch.is_tensor(eps) else center.new_tensor(eps)
                globalC.add_box(-1, input_ids, Bounds(center - e, center + e))
        elif k == InKind.LP_EMBEDDING:
            if "lb" in L.params and "ub" in L.params:
                globalC.add_box(-1, input_ids, Bounds(L.params["lb"], L.params["ub"]))
            else:
                center = L.params["center"]
                eps = L.params["eps"]
                e = eps.to(device=center.device, dtype=center.dtype) if torch.is_tensor(eps) else center.new_tensor(eps)
                globalC.add_box(-1, input_ids, Bounds(center - e, center + e))
        elif k == InKind.LIN_POLY:
            A, b = L.params["A"], L.params["b"]
            globalC.replace(Con("INEQ", tuple(input_ids), {"tag": "in:linpoly", "A": A, "b": b}))
        else:
            raise NotImplementedError(f"Unsupported INPUT_SPEC kind: {k}")




@torch.no_grad()
def setup_and_solve_batch(
    net,
    input_bounds_per_b: Bounds,
    solver: Solver,
    timelimit: Optional[float] = None,
    *,
    cache: Optional["AnalyzeCache"] = None,
) -> BatchLPSolution:
    """[BATCHED-API] Orchestrate analyze → export_to_batch_problem → solve_batch.

    ``input_bounds_per_b`` must already be a tensor-view batch
    ``[B, *input_shape]``; B=1 is just
    the length-one batch case, not a scalar special case.
    """
    from act.back_end.analyze import analyze
    from act.back_end.cons_exportor import export_to_batch_problem

    if input_bounds_per_b.lb.dim() < 2 or input_bounds_per_b.ub.dim() < 2:
        raise ValueError(
            f"setup_and_solve_batch: input_bounds_per_b must be batched "
            f"[B, *input_shape], got lb={tuple(input_bounds_per_b.lb.shape)} "
            f"ub={tuple(input_bounds_per_b.ub.shape)}"
        )

    entry_id = find_entry_layer_id(net)
    input_ids = get_input_ids(net)
    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)

    entry_fact = Fact(bounds=input_bounds_per_b, cons=ConSet())
    add_all_input_specs(entry_fact.cons, input_ids, spec_layers)

    _before, after, globalC = analyze(net, entry_id, entry_fact, cache=cache)
    validate_constraints(globalC, after, net)

    problem = export_to_batch_problem(
        net=net,
        globalC=globalC,
        assert_layer=assert_layer,
        input_box_per_b=input_bounds_per_b,
    )
    solution = solver.solve_batch(problem, timelimit=timelimit)

    expected_n = int(input_bounds_per_b.lb.shape[0])
    if len(solution.statuses) != expected_n:
        raise ValueError(
            f"setup_and_solve_batch: solver returned {len(solution.statuses)} "
            f"statuses for B={expected_n}"
        )
    valid_statuses = {SolveStatus.SAT, SolveStatus.UNSAT, SolveStatus.UNKNOWN}
    unexpected = [status for status in solution.statuses if status not in valid_statuses]
    if unexpected:
        raise ValueError(
            f"setup_and_solve_batch: unexpected solver statuses {unexpected}"
        )
    if solution.max_viol.shape != (expected_n,):
        raise ValueError(
            f"setup_and_solve_batch: max_viol shape "
            f"{tuple(solution.max_viol.shape)} != ({expected_n},)"
        )
    return solution


@torch.no_grad()
def verify_lp_batched(
    net,
    solver_factory: Callable[[], Solver],
    timelimit: Optional[float] = None,
) -> List[VerifyResult]:
    """[BATCHED-API] Run one native batched LP verification pass.

    The ACT net supplies a batched INPUT_SPEC seed ``[B, *input_shape]`` and a
    batched ASSERT layer. ``setup_and_solve_batch`` solves all B LPs at once;
    this function decodes each solver lane to a ``VerifyResult`` and validates
    SAT candidates concretely before reporting FALSIFIED.
    """
    import importlib

    spec_layers = gather_input_spec_layers(net)
    seed_bounds = seed_from_input_specs(spec_layers)
    if seed_bounds.lb.dim() < 2 or seed_bounds.ub.dim() < 2:
        raise ValueError(
            f"verify_lp_batched: seed bounds must be [B, *input_shape], "
            f"got lb={tuple(seed_bounds.lb.shape)} ub={tuple(seed_bounds.ub.shape)}"
        )
    batch_size = int(seed_bounds.lb.shape[0])
    solver = solver_factory()
    solution = setup_and_solve_batch(
        net,
        Bounds(seed_bounds.lb.clone(), seed_bounds.ub.clone()),
        solver,
        timelimit=timelimit,
    )
    if len(solution.statuses) != batch_size:
        raise ValueError(
            f"verify_lp_batched: solver returned {len(solution.statuses)} "
            f"statuses for B={batch_size}"
        )
    if solution.x.dim() != 2 or solution.x.shape[0] != batch_size:
        raise ValueError(
            f"verify_lp_batched: solution.x must be [B, nvars], got "
            f"shape={tuple(solution.x.shape)} for B={batch_size}"
        )

    input_ids = get_input_ids(net)
    input_index = torch.tensor(input_ids, device=solution.x.device, dtype=torch.long)
    x_candidates = solution.x.index_select(1, input_index).reshape_as(seed_bounds.lb)
    assert_layer = get_assert_layer(net)

    sat_mask = torch.tensor(
        [status in (SolveStatus.SAT, "FEASIBLE") for status in solution.statuses],
        device=x_candidates.device,
        dtype=torch.bool,
    )
    violations = torch.zeros(batch_size, device=x_candidates.device, dtype=torch.bool)
    if bool(sat_mask.any().item()):
        bab_module = importlib.import_module("act.back_end.bab.bab")
        sat_idx = torch.where(sat_mask)[0]
        checked_sat = bab_module.check_violations_batched(
            net, x_candidates.index_select(0, sat_idx), assert_layer,
        )
        if checked_sat.shape != (int(sat_idx.numel()),):
            raise ValueError(
                f"verify_lp_batched: check_violations_batched returned "
                f"shape={tuple(checked_sat.shape)} expected ({int(sat_idx.numel())},)"
            )
        violations.scatter_(
            0, sat_idx, checked_sat.to(device=x_candidates.device, dtype=torch.bool),
        )

    results: List[VerifyResult] = []
    x_cpu = x_candidates.detach().cpu()
    max_viol_cpu = solution.max_viol.detach().cpu()
    for lane, status in enumerate(solution.statuses):
        metadata: Dict[str, Any] = {
            "lane": lane,
            "B": batch_size,
            "solver_status": status,
            "max_viol": float(max_viol_cpu[lane].item()),
        }
        if status in (SolveStatus.SAT, "FEASIBLE"):
            if bool(violations[lane].item()):
                results.append(
                    VerifyResult(
                        VerifyStatus.FALSIFIED,
                        counterexample=x_cpu[lane].clone(),
                        metadata=metadata,
                    )
                )
            else:
                metadata["validation"] = "no_verified_violation"
                results.append(VerifyResult(VerifyStatus.UNKNOWN, metadata=metadata))
        elif status in (SolveStatus.UNSAT, "INFEASIBLE"):
            results.append(VerifyResult(VerifyStatus.CERTIFIED, metadata=metadata))
        elif status == "TIMEOUT":
            results.append(VerifyResult(VerifyStatus.TIMEOUT, metadata=metadata))
        elif status == SolveStatus.UNKNOWN:
            results.append(VerifyResult(VerifyStatus.UNKNOWN, metadata=metadata))
        else:
            raise ValueError(f"verify_lp_batched: unexpected solver status {status!r}")
    return results


# -----------------------------------------------------------------------------
# Single-shot verification
# -----------------------------------------------------------------------------


def _get_output_layer_id(net) -> int:
    """Return the unique predecessor layer id of the ASSERT layer."""

    assert_layer = get_assert_layer(net)
    pred_ids = net.preds.get(assert_layer.id, [])
    if len(pred_ids) != 1:
        raise ValueError(
            f"ASSERT layer {assert_layer.id} must have exactly one "
            f"predecessor (the network output), got predecessors={pred_ids}"
        )
    return int(pred_ids[0])


def _get_output_layer_bounds(net, after: Dict[int, Fact]) -> Bounds:
    """Return the Bounds tensor produced by the network's output layer.

    The output layer is the unique predecessor of the ASSERT layer; the
    returned Bounds is shaped ``[B, n_out]``.
    """
    return after[_get_output_layer_id(net)].bounds


def _hybridz_witness_input(
    hz: Any,
    witness: Any,
    seed_bounds: Bounds,
    active_tf: Any,
) -> tuple[Optional[torch.Tensor], str]:
    """Decode output-HZ continuous factors back to the original input box.

    Generator positions are not stable across nonlinear branches, reduction,
    and sparse joins.  Decoding is therefore allowed only when both the input
    factors and the final representation carry matching stable ids.  Missing
    provenance fails closed instead of guessing a positional prefix.
    """

    if witness is None:
        return None, "missing_witness"
    col_ids = getattr(hz, "col_ids", None)
    input_ids = getattr(active_tf, "_input_ids", None)
    if input_ids is None:
        input_ids = getattr(hz, "full_col_ids", None)
    if col_ids is None or input_ids is None:
        return None, "missing_input_generator_ids"

    if isinstance(col_ids, torch.Tensor):
        col_ids_np = col_ids.detach().cpu().numpy().reshape(-1)
    else:
        col_ids_np = np.asarray(col_ids, dtype=np.int64).reshape(-1)
    if isinstance(input_ids, torch.Tensor):
        input_ids_np = input_ids.detach().cpu().numpy().reshape(-1)
    else:
        input_ids_np = np.asarray(input_ids, dtype=np.int64).reshape(-1)
    witness_np = np.asarray(witness, dtype=np.float64).reshape(-1)
    if witness_np.size < col_ids_np.size:
        return None, "short_continuous_witness"

    lb = seed_bounds.lb.detach().cpu().double().numpy().reshape(-1)
    ub = seed_bounds.ub.detach().cpu().double().numpy().reshape(-1)
    operator_center = getattr(hz, "operator_input_center", None)
    operator_radius = getattr(hz, "operator_input_radius", None)
    if operator_center is not None or operator_radius is not None:
        if operator_center is None or operator_radius is None:
            return None, "incomplete_operator_input_normalization"
        center = np.asarray(operator_center, dtype=np.float64).reshape(-1)
        radius = np.asarray(operator_radius, dtype=np.float64).reshape(-1)
        if (
            center.size != lb.size
            or radius.size != lb.size
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(radius))
            or np.any(radius < 0.0)
        ):
            return None, "invalid_operator_input_normalization"
    else:
        center = 0.5 * (lb + ub)
        radius = 0.5 * (ub - lb)
    if input_ids_np.size != center.size:
        return None, "input_generator_id_shape_mismatch"

    col_by_id = {int(gid): pos for pos, gid in enumerate(col_ids_np.tolist())}
    x = center.copy()
    for input_dim in np.flatnonzero(radius > 0.0):
        pos = col_by_id.get(int(input_ids_np[input_dim]))
        if pos is None:
            return None, "input_generator_missing_after_transform"
        xi = float(witness_np[pos])
        if not np.isfinite(xi) or xi < -1.0 - 1e-7 or xi > 1.0 + 1e-7:
            return None, "input_generator_out_of_range"
        x[input_dim] = center[input_dim] + radius[input_dim] * np.clip(xi, -1.0, 1.0)

    shape = tuple(int(d) for d in seed_bounds.lb.shape)
    x_batch = torch.as_tensor(
        x.reshape(shape),
        dtype=seed_bounds.lb.dtype,
        device=seed_bounds.lb.device,
    )
    if not bool(
        torch.all(x_batch >= seed_bounds.lb - 1e-8).item()
        and torch.all(x_batch <= seed_bounds.ub + 1e-8).item()
    ):
        return None, "decoded_input_outside_seed_box"
    return x_batch, "stable_generator_ids"


def _hybridz_model_candidate_check(
    *,
    x_batch: torch.Tensor,
    model_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    C: torch.Tensor,
    thresholds: torch.Tensor,
    M: int,
    n_out: int,
    is_unsafe_linear: bool,
) -> tuple[Optional[bool], str]:
    """Diagnostic concrete check against ACT's canonical output rows.

    This is not sufficient for a VNN-LIB FALSIFIED verdict: materialisation may
    conservatively discard coupled input assertions or original Boolean
    structure.  The independent ``counterexample_replay_fn`` remains the sole
    authority for accepting a HybridZ candidate.
    """

    if model_fn is None:
        return None, "model_fn_not_provided"
    try:
        y = model_fn(x_batch)
        if isinstance(y, dict):
            y = y.get("output")
        if not isinstance(y, torch.Tensor):
            return None, "model_fn_no_tensor_output"
        y = y.detach()
        if y.dim() == 1:
            y = y.view(1, -1)
        else:
            y = y.reshape(y.shape[0], -1)
        if tuple(y.shape) != (1, n_out):
            return None, f"model_fn_output_shape:{tuple(y.shape)}"
        y = y.to(device=C.device, dtype=C.dtype)
    except Exception as exc:
        return None, f"model_fn_failed:{type(exc).__name__}"

    scores = C.view(1, M, n_out)[0].matmul(y[0])
    if is_unsafe_linear:
        unsafe = bool(torch.all(scores <= thresholds.view(1, M)[0] + 1e-8).item())
    else:
        unsafe = bool(torch.any(scores >= thresholds.view(1, M)[0] - 1e-8).item())
    return unsafe, "canonical_output_rows"


def _hybridz_independent_replay(
    replay_fn: Optional[Callable[[torch.Tensor], Any]],
    x_batch: torch.Tensor,
) -> tuple[bool, str, Any]:
    """Run the caller-provided original-model + raw-property replay gate."""

    if replay_fn is None:
        return False, "independent_replay_not_provided", None
    try:
        receipt = replay_fn(x_batch.detach().cpu())
    except Exception as exc:
        return False, f"independent_replay_failed:{type(exc).__name__}", None

    if isinstance(receipt, bool):
        accepted = receipt
    elif isinstance(receipt, dict):
        marker = next(
            (
                receipt[key]
                for key in ("valid_counterexample", "valid", "accepted")
                if key in receipt
            ),
            None,
        )
        if not isinstance(marker, (bool, np.bool_)):
            return False, "independent_replay_receipt_missing_boolean", receipt
        accepted = bool(marker)
    else:
        marker = getattr(receipt, "valid_counterexample", None)
        if marker is None:
            marker = getattr(receipt, "valid", None)
        if not isinstance(marker, (bool, np.bool_)):
            return False, "independent_replay_bad_receipt", receipt
        accepted = bool(marker)
    return accepted, (
        "independent_replay_accepted" if accepted else "independent_replay_rejected"
    ), receipt


_MISSING_ATTR = object()


def _apply_hybridz_tf_config(
    active_tf: Any,
    hz_cfg: Optional[Any] = None,
) -> list[tuple[str, Any]]:
    """Apply explicit HybridZ forward knobs to one TF instance."""

    if hz_cfg is None:
        return []
    updates: Dict[str, Any] = {}
    override_map = {
        "compressed_relu": "_relu_compressed",
        "relu_valid_cuts": "_relu_valid_cuts",
        "sigmoid_k": "_sigmoid_K",
        "tanh_k": "_tanh_K",
        "scurve_domain_cuts": "_scurve_domain_cuts",
        "scurve_graph_cuts": "_scurve_graph_cuts",
        "cell_budget": "_hz_cell_budget",
    }
    for cfg_name, attr_name in override_map.items():
        val = getattr(hz_cfg, cfg_name, None)
        if val is None:
            continue
        if cfg_name in {"sigmoid_k", "tanh_k"}:
            val = max(1, int(val))
        elif cfg_name == "cell_budget":
            val = int(val)
        elif cfg_name in {
            "compressed_relu",
            "relu_valid_cuts",
            "scurve_domain_cuts",
            "scurve_graph_cuts",
        }:
            val = bool(val)
        updates[attr_name] = val

    old_values: list[tuple[str, Any]] = []
    for name, value in updates.items():
        old_values.append((name, getattr(active_tf, "__dict__", {}).get(name, _MISSING_ATTR)))
        setattr(active_tf, name, value)
    return old_values


def _restore_hybridz_tf_profile(active_tf: Any, old_values: list[tuple[str, Any]]) -> None:
    for name, old in reversed(old_values):
        if old is _MISSING_ATTR:
            try:
                delattr(active_tf, name)
            except AttributeError:
                pass
        else:
            setattr(active_tf, name, old)


def _hybridz_timeout(hz_cfg: Optional[Any], fallback_timeout: Optional[float]) -> float:
    if hz_cfg is not None and hasattr(hz_cfg, "verdict_timeout"):
        return float(hz_cfg.verdict_timeout(fallback_timeout=fallback_timeout))
    if fallback_timeout is not None:
        return float(fallback_timeout)
    return 30.0


def _hybridz_config_metadata(hz_cfg: Optional[Any]) -> Dict[str, Any]:
    if hz_cfg is None:
        return {}
    keys = (
        "sigmoid_k",
        "tanh_k",
        "scurve_domain_cuts",
        "scurve_graph_cuts",
        "compressed_relu",
        "relu_valid_cuts",
        "cell_budget",
        "operator_exact_budget",
        "operator_phase_projection_time_limit",
        "operator_phase_clique_time_limit",
        "operator_materialize_add",
        "preactivation_lp_budget",
        "preactivation_lp_time_limit",
        "property_correlation_budget",
        "property_correlation_time_limit",
        "residual_phase_screen",
        "residual_bound_screen",
        "property_residual_budget",
        "property_residual_time_limit",
        "property_residual_max_adjoint_cells",
        "property_residual_pool_per_rival",
        "property_tail_upper",
        "property_micro_rlt_product_cap",
        "property_micro_rlt_packet_mode",
        "property_micro_rlt_parent_prefilter_seconds",
        "property_micro_rlt_parent_only_diagnostic",
        "property_tail_add_source_planes",
        "property_tail_mixture_grid_bits",
        "property_tail_alpha_steps",
        "property_tail_alpha_time_limit",
        "property_tail_alpha_learning_rate",
        "property_tail_alpha_max_cells",
        "property_tail_alpha_device",
        "property_tail_pairhull_budget",
        "property_tail_pairhull_time_limit",
        "property_tail_suffix_blocks",
        "property_tail_suffix_alpha_steps",
        "property_tail_suffix_alpha_time_limit",
        "property_tail_suffix_alpha_device",
        "query_dual_feedback_targets",
        "query_dual_feedback_steps",
        "query_dual_feedback_time_limit",
        "query_dual_feedback_block_size",
        "query_dual_feedback_device",
        "gpu_dual_steps",
        "gpu_dual_time_limit",
        "gpu_dual_row_topk",
        "gpu_dual_learning_rate",
        "lp_prefilter_fraction",
        "lp_prefilter_max_seconds",
    )
    result = {
        f"cfg_{key}": getattr(hz_cfg, key, None)
        for key in keys
        if getattr(hz_cfg, key, None) is not None
    }
    targets_key = "cfg_query_dual_feedback_targets"
    if targets_key in result:
        result[targets_key] = [
            int(value) for value in result[targets_key]
        ]
    return result


def _operator_phase_clique_disabled_receipt() -> Dict[str, Any]:
    """Canonical no-op receipt without importing any candidate module."""

    body = {
        "schema": "act.operator_phase_clique_pipeline.v1",
        "enabled": False,
        "status": "no_op_disabled",
        "candidate_attempted": False,
        "candidate_only": True,
        "proof_authority": False,
        "identity_preserved": True,
        "materialized": False,
        "materialization_receipt_sha256": None,
        "verdict_path": "hz_objbound_decide_only",
        "candidate_budget_fraction": 0.40,
        "materializer_reserve_fraction": 0.60,
        "timings": {"total_seconds": 0.0},
    }
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return {
        **body,
        "receipt_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _operator_phase_clique_receipt_copy(value: Any) -> Any:
    """Copy an already-verified receipt into a JSON-native metadata tree."""

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError(
                "operator phase-clique receipt contains a non-finite float"
            )
        return value
    if isinstance(value, Mapping):
        copied: Dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(
                    "operator phase-clique receipt key is not a string"
                )
            copied[key] = _operator_phase_clique_receipt_copy(item)
        return copied
    if type(value) in {list, tuple}:
        return [
            _operator_phase_clique_receipt_copy(item)
            for item in value
        ]
    raise TypeError(
        "operator phase-clique receipt contains a non-JSON value"
    )


def _operator_phase_clique_handoff_receipt(
    *,
    pipeline_receipt_sha256: str,
    semantic_digest: str,
    materialized: bool,
) -> Dict[str, Any]:
    """Checksummed audit record for the private one-use solver transfer."""

    body = {
        "schema": "verifier_operator_phase_clique_solver_handoff_v1",
        "status": "consumed_private",
        "proof_authority": False,
        "one_use_consumed": True,
        "owner_bound": True,
        "pid_bound": True,
        "private_core_readonly": True,
        "solver_hz_is_public_result_hz": False,
        "materialized": bool(materialized),
        "pipeline_receipt_sha256": str(
            pipeline_receipt_sha256
        ),
        "semantic_digest": str(semantic_digest),
        "verdict_path": "hz_objbound_decide_only",
    }
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return {
        **body,
        "receipt_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _query_dual_disabled_metadata(
    *,
    targets: Tuple[int, ...],
    steps: int,
    time_limit: float,
    block_size: int,
    device: str,
) -> Dict[str, Any]:
    """Canonical verifier receipt for an explicitly disabled mechanism."""

    return {
        "schema": "verifier_query_dual_feedback_transaction_v1",
        "status": "disabled",
        "proof_authority": False,
        "source": "configuration",
        "reason": "steps_zero",
        "targets": [int(value) for value in targets],
        "steps": int(steps),
        "time_limit": float(time_limit),
        "block_size": int(block_size),
        "device": str(device),
    }


def _query_dual_error_metadata(
    *,
    targets: Tuple[int, ...],
    steps: int,
    time_limit: float,
    block_size: int,
    device: str,
    elapsed_seconds: float,
    exc: BaseException,
    deadline: bool,
    source: str = "built_in_verify_once",
) -> Dict[str, Any]:
    """Fail-closed transaction receipt before reverting to baseline HZ."""

    code = getattr(exc, "code", None)
    return {
        "schema": "verifier_query_dual_feedback_transaction_v1",
        "status": (
            "deadline_fallback_baseline"
            if deadline
            else "error_fallback_baseline"
        ),
        "proof_authority": False,
        "source": str(source),
        "targets": [int(value) for value in targets],
        "steps": int(steps),
        "time_limit": float(time_limit),
        "block_size": int(block_size),
        "device": str(device),
        "elapsed_seconds": float(elapsed_seconds),
        "error_type": type(exc).__name__,
        "error_code": None if code is None else str(code),
        "error": str(exc)[:1000],
        "rollback": "complete_query_dual_feature",
    }


_QUERY_DUAL_PIPELINE_SCHEMA = "act.verified_query_dual_feedback.v2"
_QUERY_DUAL_STAGE_SCHEMA = "act.verified_query_dual_stage.v2"
_QUERY_DUAL_PROPERTY_SCHEMA = "act.verified_query_dual_property.v2"
_QUERY_DUAL_CANDIDATE_SCHEMA = "act.query_dual_candidates.v2"
_QUERY_DUAL_CANDIDATE_PROTOCOL = "descriptor_only_v2"
_QUERY_DUAL_CANDIDATE_AUDIT_FIELDS = [
    "lr_alpha",
    "lr_decay",
    "solver",
    "elapsed_seconds",
    "timings",
]
_QUERY_DUAL_PIPELINE_AUDIT_FIELDS = [
    "candidate_generator",
    "candidate_solver_factory",
    "dual_solver_default_device",
    "dual_solver_default_dtype",
    "candidate_cuda_device_name",
]


def _query_dual_self_hashed_receipt_valid(
    receipt: Any,
    *,
    schema: str,
) -> bool:
    if not isinstance(receipt, Mapping):
        return False
    claimed = receipt.get("receipt_sha256")
    return bool(
        _is_lower_sha256(claimed)
        and receipt.get("schema") == schema
        and _canonical_receipt_sha256(
            receipt,
            checksum_field="receipt_sha256",
        )
        == claimed
    )


def _query_dual_v2_candidate_receipt_valid(
    receipt: Any,
    *,
    expected_status: str,
    expected_blocks: int,
    property_candidate: bool,
) -> bool:
    """Independently enforce the descriptor-only V2 candidate envelope."""

    if not _query_dual_self_hashed_receipt_valid(
        receipt,
        schema=_QUERY_DUAL_CANDIDATE_SCHEMA,
    ):
        return False
    assert isinstance(receipt, Mapping)
    records = receipt.get("descriptor_records")
    coverage_sha = receipt.get("descriptor_coverage_sha256")
    if (
        receipt.get("protocol") != _QUERY_DUAL_CANDIDATE_PROTOCOL
        or receipt.get("non_authoritative_audit_fields")
        != _QUERY_DUAL_CANDIDATE_AUDIT_FIELDS
        or receipt.get("candidate_only") is not True
        or receipt.get("proof_authority") is not False
        or receipt.get("candidate_bound_source")
        != "none_descriptor_only"
        or receipt.get("optimizer_best_margins_used_as_bounds") is not False
        or receipt.get("optimizer_margins_exported") is not False
        or receipt.get("optimizer_margins_used_for_improvement") is not False
        or receipt.get("gpu_frozen_alpha_replay") is not False
        or receipt.get("cpu_independent_replay_required") is not True
        or receipt.get("all_candidate_updates_replayed_with_stored_alpha")
        is not False
        or receipt.get("all_bounds_replayed_with_stored_alpha") is not False
        or receipt.get("property_lower_dual_replayed") is not False
        or receipt.get("strict_target_improvements") != 0
        or receipt.get("strict_property_improvements") != 0
        or receipt.get("improved_target_indices") != []
        or receipt.get("improved_property_indices") != []
        or receipt.get("candidate_target_bounds_sha256")
        != receipt.get("target_bounds_sha256")
        or receipt.get("candidate_property_bounds_sha256")
        != receipt.get("property_baseline_sha256")
        or receipt.get("status") != expected_status
        or not isinstance(records, list)
        or not _is_lower_sha256(coverage_sha)
        or coverage_sha != _canonical_receipt_sha256(records)
        or receipt.get("descriptor_records_sha256") != coverage_sha
        or receipt.get("descriptor_coverage_complete") is not True
        or isinstance(receipt.get("query_blocks"), bool)
        or not isinstance(receipt.get("query_blocks"), int)
        or int(receipt["query_blocks"]) != expected_blocks
        or len(records) != expected_blocks
        or (
            (receipt.get("target_relu_lid") is None)
            is not property_candidate
        )
        or receipt.get("property_upper_only") is not True
    ):
        return False
    if expected_status == "descriptors_generated":
        return bool(
            expected_blocks > 0
            and receipt.get("candidate_generated") is True
            and receipt.get("whole_batch_complete") is True
        )
    return bool(
        expected_status == "no_queries_fallback"
        and expected_blocks == 0
        and receipt.get("candidate_generated") is False
        and receipt.get("whole_batch_complete") is True
        and receipt.get("property_rows") == 0
    )


def _validate_query_dual_v2_pending_bindings(
    feedback: Any,
    *,
    pipeline_receipt: Mapping[str, Any],
    target_stages: Tuple[Any, ...],
    property_stage: Any,
    targets: Tuple[int, ...],
    steps: int,
    block_size: int,
    device: str,
    bind_effective_config: bool,
) -> None:
    """Reject V1/mixed receipt chains before Operator-HZ construction."""

    def reject(reason: str) -> None:
        raise ValueError(
            "query-dual descriptor-only V2 pending receipt rejected: "
            f"{reason}"
        )

    pipeline_steps = pipeline_receipt.get("steps")
    pipeline_block_size = pipeline_receipt.get("block_size")
    pipeline_replay_chunk_size = pipeline_receipt.get(
        "replay_chunk_size"
    )
    pipeline_device = pipeline_receipt.get("candidate_device")
    if (
        feedback.proof_authority is not True
        or not _query_dual_self_hashed_receipt_valid(
            pipeline_receipt,
            schema=_QUERY_DUAL_PIPELINE_SCHEMA,
        )
        or pipeline_receipt.get("status") != "verified"
        or pipeline_receipt.get("proof_authority") is not True
        or pipeline_receipt.get("candidate_schema")
        != _QUERY_DUAL_CANDIDATE_SCHEMA
        or pipeline_receipt.get("candidate_protocol")
        != _QUERY_DUAL_CANDIDATE_PROTOCOL
        or pipeline_receipt.get("non_authoritative_audit_fields")
        != _QUERY_DUAL_PIPELINE_AUDIT_FIELDS
        or pipeline_receipt.get("target_relu_ids") != list(targets)
        or isinstance(pipeline_steps, bool)
        or not isinstance(pipeline_steps, int)
        or pipeline_steps <= 0
        or isinstance(pipeline_block_size, bool)
        or not isinstance(pipeline_block_size, int)
        or pipeline_block_size <= 0
        or isinstance(pipeline_replay_chunk_size, bool)
        or not isinstance(pipeline_replay_chunk_size, int)
        or pipeline_replay_chunk_size <= 0
        or pipeline_device not in {"cpu", "cuda"}
        or len(target_stages) != len(targets)
    ):
        reject("pipeline identity or target coverage mismatch")
    if bind_effective_config and (
        pipeline_steps != int(steps)
        or pipeline_block_size != int(block_size)
        or pipeline_replay_chunk_size != int(block_size)
        or pipeline_device != str(device)
    ):
        reject("built-in pipeline disagrees with effective configuration")

    stage_receipts = [stage.receipt for stage in target_stages]
    candidate_receipts = [
        stage.candidate_receipt for stage in target_stages
    ]
    if (
        pipeline_receipt.get("stage_receipt_sha256")
        != [receipt.get("receipt_sha256") for receipt in stage_receipts]
        or pipeline_receipt.get("target_candidate_receipt_sha256")
        != [
            receipt.get("receipt_sha256")
            for receipt in candidate_receipts
        ]
        or pipeline_receipt.get(
            "target_candidate_descriptor_coverage_sha256"
        )
        != [
            receipt.get("descriptor_coverage_sha256")
            for receipt in candidate_receipts
        ]
    ):
        reject("pipeline/target candidate hash binding mismatch")

    for index, (target, stage, receipt, candidate) in enumerate(
        zip(targets, target_stages, stage_receipts, candidate_receipts)
    ):
        blocks = tuple(stage.blocks)
        block_count = len(blocks)
        expected_candidate_status = (
            "descriptors_generated"
            if block_count
            else "no_queries_fallback"
        )
        strict = stage.strict_improvements
        expected_stage_status = (
            "verified"
            if isinstance(strict, int)
            and not isinstance(strict, bool)
            and strict > 0
            else "verified_no_improvement"
        )
        if (
            not _query_dual_self_hashed_receipt_valid(
                receipt,
                schema=_QUERY_DUAL_STAGE_SCHEMA,
            )
            or receipt.get("stage_index") != index
            or receipt.get("target_relu_lid") != target
            or receipt.get("status") != expected_stage_status
            or receipt.get("candidate_schema")
            != _QUERY_DUAL_CANDIDATE_SCHEMA
            or receipt.get("candidate_protocol")
            != _QUERY_DUAL_CANDIDATE_PROTOCOL
            or receipt.get("candidate_status")
            != expected_candidate_status
            or receipt.get("candidate_receipt_sha256")
            != candidate.get("receipt_sha256")
            or receipt.get("candidate_descriptor_coverage_sha256")
            != candidate.get("descriptor_coverage_sha256")
            or receipt.get("block_receipt_sha256")
            != [
                block.replay_receipt.get("receipt_sha256")
                for block in blocks
            ]
            or isinstance(strict, bool)
            or not isinstance(strict, int)
            or strict < 0
            or receipt.get("strict_improvements") != strict
            or (not block_count and strict != 0)
            or not _query_dual_v2_candidate_receipt_valid(
                candidate,
                expected_status=expected_candidate_status,
                expected_blocks=block_count,
                property_candidate=False,
            )
        ):
            reject(f"target stage {index} V2 binding mismatch")

    property_receipt = property_stage.receipt
    property_candidate = property_stage.candidate_receipt
    property_blocks = tuple(property_stage.blocks)
    if (
        not property_blocks
        or not _query_dual_self_hashed_receipt_valid(
            property_receipt,
            schema=_QUERY_DUAL_PROPERTY_SCHEMA,
        )
        or property_receipt.get("status") != "verified"
        or property_receipt.get("proof_authority") is not True
        or property_receipt.get("direction") != "UPPER"
        or property_receipt.get("quantity") != "C_y_minus_threshold"
        or property_receipt.get("objective") != "-C"
        or property_receipt.get("replay_query_bias") != "+threshold"
        or property_receipt.get("upper_reconstruction")
        != "-LB(-C_y+threshold)"
        or property_receipt.get("coverage_complete") is not True
        or property_receipt.get("candidate_schema")
        != _QUERY_DUAL_CANDIDATE_SCHEMA
        or property_receipt.get("candidate_protocol")
        != _QUERY_DUAL_CANDIDATE_PROTOCOL
        or property_receipt.get("candidate_status")
        != "descriptors_generated"
        or property_receipt.get("candidate_receipt_sha256")
        != property_candidate.get("receipt_sha256")
        or property_receipt.get(
            "candidate_descriptor_coverage_sha256"
        )
        != property_candidate.get("descriptor_coverage_sha256")
        or property_receipt.get("block_receipt_sha256")
        != [
            block.replay_receipt.get("receipt_sha256")
            for block in property_blocks
        ]
        or pipeline_receipt.get("property_receipt_sha256")
        != property_receipt.get("receipt_sha256")
        or pipeline_receipt.get("property_candidate_receipt_sha256")
        != property_candidate.get("receipt_sha256")
        or pipeline_receipt.get(
            "property_candidate_descriptor_coverage_sha256"
        )
        != property_candidate.get("descriptor_coverage_sha256")
        or not _query_dual_v2_candidate_receipt_valid(
            property_candidate,
            expected_status="descriptors_generated",
            expected_blocks=len(property_blocks),
            property_candidate=True,
        )
    ):
        reject("property-stage V2 binding mismatch")


def _query_dual_pending_metadata(
    feedback: Any,
    *,
    source: str,
    targets: Tuple[int, ...],
    steps: int,
    time_limit: float,
    block_size: int,
    device: str,
    elapsed_seconds: float,
    bind_effective_config: bool,
) -> Dict[str, Any]:
    """Snapshot a live transaction before Operator-HZ consumes it."""

    pipeline_receipt = copy.deepcopy(dict(feedback.receipt))
    property_upper = np.ascontiguousarray(
        np.asarray(feedback.property_upper, dtype=np.float64).reshape(-1)
    )
    target_stages = tuple(feedback.stages)
    property_stage = feedback.property_stage
    _validate_query_dual_v2_pending_bindings(
        feedback,
        pipeline_receipt=pipeline_receipt,
        target_stages=target_stages,
        property_stage=property_stage,
        targets=targets,
        steps=steps,
        block_size=block_size,
        device=device,
        bind_effective_config=bind_effective_config,
    )
    transaction_steps = (
        int(steps)
        if bind_effective_config
        else int(pipeline_receipt["steps"])
    )
    transaction_block_size = (
        int(block_size)
        if bind_effective_config
        else int(pipeline_receipt["block_size"])
    )
    transaction_device = (
        str(device)
        if bind_effective_config
        else str(pipeline_receipt["candidate_device"])
    )
    return {
        # The transaction envelope and Operator-HZ consumption contract remain
        # V1; their nested receipt identities below make the V2 upgrade
        # explicit while preserving existing consumers.
        "schema": "verifier_query_dual_feedback_transaction_v1",
        "status": "pipeline_verified_pending_operator",
        "proof_authority": False,
        "pipeline_proof_authority": (
            feedback.proof_authority is True
            and pipeline_receipt.get("proof_authority") is True
        ),
        "source": str(source),
        "targets": [int(value) for value in targets],
        "steps": transaction_steps,
        "time_limit": float(time_limit),
        "block_size": transaction_block_size,
        "device": transaction_device,
        "elapsed_seconds": float(elapsed_seconds),
        "pipeline_schema": _QUERY_DUAL_PIPELINE_SCHEMA,
        "target_stage_schema": _QUERY_DUAL_STAGE_SCHEMA,
        "property_stage_schema": _QUERY_DUAL_PROPERTY_SCHEMA,
        "candidate_schema": _QUERY_DUAL_CANDIDATE_SCHEMA,
        "candidate_protocol": _QUERY_DUAL_CANDIDATE_PROTOCOL,
        "candidate_non_authoritative_audit_fields": list(
            _QUERY_DUAL_CANDIDATE_AUDIT_FIELDS
        ),
        "pipeline_non_authoritative_audit_fields": list(
            _QUERY_DUAL_PIPELINE_AUDIT_FIELDS
        ),
        "replay_chunk_size": int(pipeline_receipt["replay_chunk_size"]),
        "pipeline_receipt": pipeline_receipt,
        "target_stage_receipts": [
            copy.deepcopy(dict(stage.receipt)) for stage in target_stages
        ],
        "property_stage_receipt": copy.deepcopy(
            dict(property_stage.receipt)
        ),
        "root_bounds_count": int(len(feedback.root_certificate.bounds)),
        "target_stage_count": int(len(target_stages)),
        "target_block_count": int(
            sum(len(stage.blocks) for stage in target_stages)
        ),
        "property_block_count": int(len(property_stage.blocks)),
        "strict_improvements_total": int(
            sum(stage.strict_improvements for stage in target_stages)
        ),
        "property_rows": int(property_upper.size),
        "property_upper_sha256": pipeline_receipt.get(
            "property_upper_sha256"
        ),
        "property_upper_hex": [
            float(value).hex() for value in property_upper
        ],
    }


def _query_dual_mark_applied(
    metadata: Mapping[str, Any],
    *,
    operator_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    """Bind a verified pipeline receipt to the exact Operator-HZ build."""

    result = copy.deepcopy(dict(metadata))
    pipeline_receipt = result.get("pipeline_receipt")
    operator_receipt = operator_metadata.get(
        "verified_query_dual_feedback"
    )
    if (
        not isinstance(pipeline_receipt, Mapping)
        or not isinstance(operator_receipt, Mapping)
    ):
        raise ValueError(
            "query-dual pipeline/operator receipt is missing at commit"
        )
    transaction_sha256 = pipeline_receipt.get("receipt_sha256")
    if (
        not _is_lower_sha256(transaction_sha256)
        or result.get("pipeline_schema") != _QUERY_DUAL_PIPELINE_SCHEMA
        or result.get("target_stage_schema") != _QUERY_DUAL_STAGE_SCHEMA
        or result.get("property_stage_schema")
        != _QUERY_DUAL_PROPERTY_SCHEMA
        or result.get("candidate_schema") != _QUERY_DUAL_CANDIDATE_SCHEMA
        or result.get("candidate_protocol")
        != _QUERY_DUAL_CANDIDATE_PROTOCOL
        or result.get("candidate_non_authoritative_audit_fields")
        != _QUERY_DUAL_CANDIDATE_AUDIT_FIELDS
        or result.get("pipeline_non_authoritative_audit_fields")
        != _QUERY_DUAL_PIPELINE_AUDIT_FIELDS
        or pipeline_receipt.get("schema") != _QUERY_DUAL_PIPELINE_SCHEMA
        or pipeline_receipt.get("candidate_schema")
        != _QUERY_DUAL_CANDIDATE_SCHEMA
        or pipeline_receipt.get("candidate_protocol")
        != _QUERY_DUAL_CANDIDATE_PROTOCOL
        or pipeline_receipt.get("non_authoritative_audit_fields")
        != _QUERY_DUAL_PIPELINE_AUDIT_FIELDS
        or operator_receipt.get("schema")
        != "operator_hz_verified_query_dual_feedback_v1"
        or operator_receipt.get("transaction_receipt_sha256")
        != transaction_sha256
        or operator_receipt.get("proof_authority") is not True
        or operator_receipt.get("process_local_validation") is not True
        or operator_receipt.get("receipt_rehydration_authority") is not False
        or operator_receipt.get("target_relu_ids")
        != result.get("targets")
    ):
        raise ValueError(
            "query-dual operator receipt disagrees with the live transaction"
        )
    for key in (
        "root_boxes_sha256",
        "final_boxes_sha256",
        "property_spec_sha256",
        "property_upper_sha256",
    ):
        if operator_receipt.get(key) != pipeline_receipt.get(key):
            raise ValueError(
                f"query-dual operator receipt disagrees on {key}"
            )
    result.update(
        {
            "status": "applied",
            "proof_authority": True,
            "operator_transaction_receipt_sha256": transaction_sha256,
        }
    )
    return result


def _query_dual_mark_property_only_applied(
    metadata: Mapping[str, Any],
    *,
    operator_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    """Bind property-only replay to the exact live C5 frame export."""

    result = copy.deepcopy(dict(metadata))
    pipeline = result.get("pipeline_receipt")
    exported = operator_metadata.get(
        "verified_preactivation_frame"
    )
    if not isinstance(pipeline, Mapping) or not isinstance(
        exported, Mapping
    ):
        raise ValueError(
            "property-only query replay lacks its pipeline/frame receipt"
        )
    initial = pipeline.get("initial_preactivation_frame")
    transaction_sha256 = pipeline.get("receipt_sha256")
    if (
        result.get("targets") != []
        or result.get("target_stage_count") != 0
        or result.get("target_stage_receipts") != []
        or not _is_lower_sha256(transaction_sha256)
        or not isinstance(initial, Mapping)
        or initial.get("schema")
        != "query_dual_operator_hz_bound_frame_v1"
        or initial.get("enabled") is not True
        or initial.get("proof_authority") is not True
        or initial.get("source")
        != "live_operator_hz_preactivation_frame"
        or initial.get("intersection_only") is not True
        or initial.get("target_replay_stages_required") is not False
        or exported.get("schema")
        != "operator_hz_verified_preactivation_frame_v1"
        or exported.get("proof_authority") is not True
        or exported.get("process_local_validation_required") is not True
        or initial.get("source_receipt_sha256")
        != exported.get("receipt_sha256")
        or initial.get("source_bounds_sha256")
        != exported.get("bounds_sha256")
        or initial.get("source_network_sha256")
        != exported.get("network_sha256")
    ):
        raise ValueError(
            "property-only query replay disagrees with the live C5 frame"
        )
    result.update(
        {
            "status": "applied",
            "proof_authority": True,
            "application_mode": (
                "property_only_post_operator_bound_frame"
            ),
            "operator_transaction_receipt_sha256": (
                transaction_sha256
            ),
            "operator_bound_frame_receipt_sha256": (
                exported["receipt_sha256"]
            ),
        }
    )
    return result


def _is_lower_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_receipt_sha256(
    value: Any,
    *,
    checksum_field: Optional[str] = None,
) -> Optional[str]:
    """Hash one JSON-friendly receipt using the candidate module convention."""

    try:
        payload = dict(value) if isinstance(value, Mapping) else value
        if checksum_field is not None:
            if not isinstance(payload, dict) or checksum_field not in payload:
                return None
            payload = dict(payload)
            del payload[checksum_field]
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError):
        return None
    return hashlib.sha256(encoded).hexdigest()


def _validate_property_tail_pairhull_receipt(
    receipt: Any,
    *,
    requested_budget: Any,
    requested_time_limit: Any,
    alternative_rivals: Any,
    alternative_kinds: Any,
    rival_count: Any,
) -> bool:
    """Fail-closed validation for exported PairHull property alternatives.

    Candidate selection is explicitly non-authoritative.  This check binds an
    applied operator receipt to the configured request and to the actual
    alternative-row mapping, and validates the outer batch checksum plus every
    nested exact PairHull receipt before the grouped property rows may be used.
    """

    try:
        if (
            isinstance(requested_budget, (bool, np.bool_))
            or not isinstance(requested_budget, (int, np.integer))
            or not 0 <= int(requested_budget) <= 8
            or isinstance(requested_time_limit, (bool, np.bool_))
            or not isinstance(
                requested_time_limit,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(float(requested_time_limit))
            or not 0.0 <= float(requested_time_limit) <= 1.5
            or (int(requested_budget) > 0)
            != (float(requested_time_limit) > 0.0)
            or isinstance(rival_count, (bool, np.bool_))
            or not isinstance(rival_count, (int, np.integer))
            or int(rival_count) <= 0
            or not isinstance(receipt, Mapping)
            or not isinstance(alternative_rivals, list)
            or not isinstance(alternative_kinds, list)
            or len(alternative_rivals) != len(alternative_kinds)
        ):
            return False

        budget = int(requested_budget)
        time_limit = float(requested_time_limit)
        requested = bool(budget > 0 and time_limit > 0.0)
        if (
            receipt.get("schema")
            != "operator_hz_property_tail_pairhull_v1"
            or receipt.get("enabled") is not requested
            or isinstance(receipt.get("pair_budget"), (bool, np.bool_))
            or not isinstance(
                receipt.get("pair_budget"), (int, np.integer)
            )
            or int(receipt["pair_budget"]) != budget
            or isinstance(
                receipt.get("time_limit_seconds"), (bool, np.bool_)
            )
            or not isinstance(
                receipt.get("time_limit_seconds"),
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(float(receipt["time_limit_seconds"]))
            or float(receipt["time_limit_seconds"]) != time_limit
            or receipt.get("safe_only") is not True
            or receipt.get("selection_candidate_only") is not True
            or receipt.get("selection_proof_authority") is not False
            or receipt.get("error_included") is not True
            or receipt.get("compact_sparse_projection") is not True
            or receipt.get("baseline_fallback_retained_per_rival") is not True
            or receipt.get("foundation_slopes_reused") is not True
            or receipt.get(
                "foundation_intercept_outward_slack_inherited"
            )
            is not True
            or receipt.get("prunes_prefix_frame") is not False
            or receipt.get("budget_semantics")
            != "global_unique_disjoint_pairs_v1"
            or receipt.get("max_rows_per_rival") != 1
            or not _is_lower_sha256(
                receipt.get("candidate_rows_sha256")
            )
            or not _is_lower_sha256(
                receipt.get("candidate_intercepts_sha256")
            )
            or not _is_lower_sha256(receipt.get("receipt_sha256"))
        ):
            return False

        # The candidate module deliberately uses the same canonical receipt
        # checksum for its outer and inner mappings.
        from act.back_end.hybridz_tf.property_pairhull_candidates import (
            verify_property_pairhull_candidates_receipt,
        )

        if not verify_property_pairhull_candidates_receipt(receipt):
            return False

        selected_rivals = receipt.get("selected_rivals")
        selected_rival_ids = receipt.get("selected_rival_ids")
        global_pair_count = receipt.get("global_pair_count")
        if (
            isinstance(selected_rivals, (bool, np.bool_))
            or not isinstance(selected_rivals, (int, np.integer))
            or int(selected_rivals) < 0
            or not isinstance(selected_rival_ids, list)
            or len(selected_rival_ids) != int(selected_rivals)
            or any(
                isinstance(rival, (bool, np.bool_))
                or not isinstance(rival, (int, np.integer))
                or not 0 <= int(rival) < int(rival_count)
                for rival in selected_rival_ids
            )
            or len({int(rival) for rival in selected_rival_ids})
            != len(selected_rival_ids)
            or isinstance(global_pair_count, (bool, np.bool_))
            or not isinstance(global_pair_count, (int, np.integer))
            or not 0 <= int(global_pair_count) <= budget
        ):
            return False

        pair_rivals = []
        for rival, kind in zip(alternative_rivals, alternative_kinds):
            if kind != "pairhull_joint_materialized":
                continue
            if (
                isinstance(rival, (bool, np.bool_))
                or not isinstance(rival, (int, np.integer))
                or not 0 <= int(rival) < int(rival_count)
            ):
                return False
            pair_rivals.append(int(rival))
        if (
            len(pair_rivals) != len(set(pair_rivals))
            or pair_rivals != [int(rival) for rival in selected_rival_ids]
            or len(pair_rivals) != int(selected_rivals)
        ):
            return False

        status = receipt.get("status")
        if not isinstance(status, str) or not status.strip():
            return False
        normalized_status = status.strip().lower()
        inner = receipt.get("candidate_receipt")
        if inner is None and not requested:
            return bool(
                normalized_status == "disabled"
                and not pair_rivals
                and int(selected_rivals) == 0
                and not selected_rival_ids
                and receipt.get("proof_authority") is False
            )
        if (
            not isinstance(inner, Mapping)
            or inner.get("schema") != "act.property_pairhull.candidates.v1"
            or not verify_property_pairhull_candidates_receipt(inner)
            or inner.get("candidate_only") is not True
            or inner.get("pair_selector_proof_authority") is not False
            or inner.get("foundation_rows_must_remain_retained") is not True
            or isinstance(
                inner.get("requested_pair_budget"), (bool, np.bool_)
            )
            or not isinstance(
                inner.get("requested_pair_budget"), (int, np.integer)
            )
            or int(inner["requested_pair_budget"]) != budget
            or isinstance(
                inner.get("time_limit_seconds"), (bool, np.bool_)
            )
            or not isinstance(
                inner.get("time_limit_seconds"),
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(float(inner["time_limit_seconds"]))
            or float(inner["time_limit_seconds"]) != time_limit
            or isinstance(inner.get("selected_candidates"), (bool, np.bool_))
            or not isinstance(
                inner.get("selected_candidates"), (int, np.integer)
            )
            or int(inner["selected_candidates"]) < int(selected_rivals)
            or isinstance(inner.get("global_pair_count"), (bool, np.bool_))
            or not isinstance(
                inner.get("global_pair_count"), (int, np.integer)
            )
            or int(inner["global_pair_count"]) != int(global_pair_count)
        ):
            return False

        candidate_records = inner.get("candidate_records", [])
        if not isinstance(candidate_records, list):
            return False
        if int(inner["selected_candidates"]) != len(candidate_records):
            return False
        if candidate_records:
            if (
                not _is_lower_sha256(
                    inner.get("candidate_records_sha256")
                )
                or _canonical_receipt_sha256(candidate_records)
                != inner.get("candidate_records_sha256")
            ):
                return False
        inner_rival_ids = []
        from act.back_end.hybridz_tf.property_pairhull import (
            verify_pairhull_receipt,
        )

        for record in candidate_records:
            if not isinstance(record, Mapping):
                return False
            rival = record.get("rival_id")
            exact_receipt = record.get("exact_pairhull_receipt")
            if (
                isinstance(rival, (bool, np.bool_))
                or not isinstance(rival, (int, np.integer))
                or not 0 <= int(rival) < int(rival_count)
                or record.get("candidate_selection_proof_authority")
                is not False
                or record.get("outward_intercept_validated") is not True
                or not _is_lower_sha256(
                    record.get("candidate_plane_sha256")
                )
                or not _is_lower_sha256(
                    record.get("source_affine_sha256")
                )
                or not _is_lower_sha256(
                    record.get("constraints_sha256")
                )
                or not _is_lower_sha256(record.get("record_sha256"))
                or _canonical_receipt_sha256(
                    record, checksum_field="record_sha256"
                )
                != record.get("record_sha256")
                or not isinstance(exact_receipt, Mapping)
                or exact_receipt.get("schema")
                != "act.property_pairhull.exact.v1"
                or exact_receipt.get("proof_authority")
                != "exact_fraction_phase_vertex_enumeration"
                or exact_receipt.get(
                    "projection_uses_outward_stored_supports"
                )
                is not True
                or exact_receipt.get("candidate_slope_proof_authority")
                is not False
                or exact_receipt.get("float_lp_proof_authority") is not False
                or exact_receipt.get("phases_total") != 4
                or not verify_pairhull_receipt(exact_receipt)
            ):
                return False
            inner_rival_ids.append(int(rival))
        if (
            len(inner_rival_ids) != len(set(inner_rival_ids))
            or any(
                rival not in set(inner_rival_ids)
                for rival in selected_rival_ids
            )
        ):
            return False
        selected_pairs = receipt.get("selected_pair_indices", [])
        selected_foundations = receipt.get(
            "selected_foundation_indices", []
        )
        if int(selected_rivals):
            if (
                not isinstance(selected_pairs, list)
                or not isinstance(selected_foundations, list)
                or len(selected_pairs) != int(selected_rivals)
                or len(selected_foundations) != int(selected_rivals)
            ):
                return False
            records_by_rival = {
                int(record["rival_id"]): record
                for record in candidate_records
            }
            for offset, rival in enumerate(selected_rival_ids):
                pair = selected_pairs[offset]
                foundation = selected_foundations[offset]
                record = records_by_rival[int(rival)]
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or any(
                        isinstance(value, (bool, np.bool_))
                        or not isinstance(value, (int, np.integer))
                        or int(value) < 0
                        for value in pair
                    )
                    or int(pair[0]) == int(pair[1])
                    or isinstance(foundation, (bool, np.bool_))
                    or not isinstance(
                        foundation, (int, np.integer)
                    )
                    or int(foundation) < 0
                    or [int(pair[0]), int(pair[1])]
                    != [int(value) for value in record.get("pair", [])]
                    or int(foundation)
                    != int(record.get("foundation_index", -1))
                ):
                    return False
        elif selected_pairs or selected_foundations:
            return False

        applied = normalized_status == "applied"
        if applied:
            return bool(
                requested
                and receipt.get("proof_authority") is True
                and receipt.get("exact_search_complete") is True
                and receipt.get("full_row_outward_affine") is True
                and int(selected_rivals) > 0
                and int(global_pair_count) > 0
                and inner.get("status") == "generated"
                and inner.get("whole_batch_complete") is True
                and inner.get("at_most_one_candidate_per_rival") is True
                and inner.get("foundation_rows_retained_by_caller") is True
            )

        # No non-applied path may smuggle a PairHull row into the exported
        # alternative mapping.  Enabled requests must report a completed
        # no-improvement or an explicit fail-closed fallback, never pending.
        if (
            pair_rivals
            or int(selected_rivals) != 0
            or selected_rival_ids
            or receipt.get("proof_authority") is not False
        ):
            return False
        if not requested:
            return normalized_status == "disabled"
        return normalized_status not in {"pending", "disabled", "applied"}
    except (
        ImportError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return False


def _f64_array_sha256(value: Any) -> str:
    """Hash exact binary64 storage using the operator-HZ convention."""

    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_verified_query_dual_property_actual_object(
    feedback: Any,
    *,
    net: Any,
    property_rows: Any,
    thresholds: Any,
    operator_build: Any,
    tail_receipt: Any,
    property_row_groups: Any,
    alternative_rivals: Any,
    alternative_kinds: Any,
) -> bool:
    """Bind live query-dual authority to the actual exported constant rows.

    Receipt checks alone are deliberately insufficient: this calls the
    process-local transaction validator again, then compares every expected
    binary64 constant against the final HZ center and proves its complete
    continuous/binary generator row is empty.  Thus no output-roundoff
    generator or hidden affine dependence can be smuggled behind a valid
    self-hash.
    """

    try:
        from act.back_end.hybridz_tf.query_dual_pipeline import (
            validate_verified_query_dual_feedback,
        )

        C = np.ascontiguousarray(
            np.asarray(property_rows, dtype=np.float64)
        )
        d = np.ascontiguousarray(
            np.asarray(thresholds, dtype=np.float64).reshape(-1)
        )
        targets = tuple(int(value) for value in feedback.target_relu_ids)
        if (
            C.ndim != 2
            or C.shape[0] == 0
            or d.shape != (C.shape[0],)
            or feedback.proof_authority is not True
            or validate_verified_query_dual_feedback(
                feedback,
                net=net,
                property_rows=C,
                thresholds=d,
                expected_target_relu_ids=targets,
            )
            is not True
            or not isinstance(tail_receipt, Mapping)
            or not isinstance(alternative_rivals, list)
            or not isinstance(alternative_kinds, list)
            or len(alternative_rivals) != len(alternative_kinds)
        ):
            return False

        expected = np.ascontiguousarray(
            np.asarray(feedback.property_upper, dtype=np.float64).reshape(-1)
        )
        rival_count = int(C.shape[0])
        if (
            expected.shape != (rival_count,)
            or not np.all(np.isfinite(expected))
            or len(property_row_groups) != rival_count
        ):
            return False

        constant_offsets = [
            int(index)
            for index, kind in enumerate(alternative_kinds)
            if kind == "verified_query_dual_property_constant"
        ]
        if (
            len(constant_offsets) != rival_count
            or constant_offsets
            != list(
                range(
                    len(alternative_kinds) - rival_count,
                    len(alternative_kinds),
                )
            )
            or [
                int(alternative_rivals[index])
                for index in constant_offsets
            ]
            != list(range(rival_count))
        ):
            return False
        actual_rows = np.asarray(
            [rival_count + index for index in constant_offsets],
            dtype=np.int64,
        )
        if any(
            not property_row_groups[rival]
            or int(property_row_groups[rival][-1])
            != int(actual_rows[rival])
            for rival in range(rival_count)
        ):
            return False

        receipt = tail_receipt.get(
            "verified_query_dual_property_constants"
        )
        bundle_receipt = feedback.receipt
        top_receipt = operator_build.metadata.get(
            "verified_query_dual_feedback"
        )
        by_layer_id = {
            int(layer.id): layer for layer in net.layers
        }
        relu_layer_id = int(tail_receipt["relu_layer_id"])
        relu_width = len(by_layer_id[relu_layer_id].out_vars)
        expected_zero_planes = np.zeros(
            (rival_count, relu_width), dtype=np.float64
        )
        if (
            not isinstance(receipt, Mapping)
            or not isinstance(bundle_receipt, Mapping)
            or not isinstance(top_receipt, Mapping)
            or receipt.get("schema")
            != "operator_hz_verified_query_dual_property_constant_v1"
            or receipt.get("status") != "applied"
            or receipt.get("proof_authority") is not True
            or receipt.get("safe_only") is not True
            or receipt.get("baseline_fallback_retained_per_rival") is not True
            or receipt.get("no_output_error_generators") is not True
            or receipt.get("constant_row_count") != rival_count
            or receipt.get("constant_row_indices")
            != [int(value) for value in actual_rows]
            or receipt.get("constant_rival_ids")
            != list(range(rival_count))
            or receipt.get("constant_values_hex")
            != [float(value).hex() for value in expected]
            or receipt.get("constant_values_sha256")
            != _f64_array_sha256(expected)
            or receipt.get("zero_envelope_planes_sha256")
            != _f64_array_sha256(expected_zero_planes)
            or receipt.get("constant_row_indices_sha256")
            != hashlib.sha256(
                np.ascontiguousarray(
                    actual_rows, dtype=np.int64
                ).tobytes()
            ).hexdigest()
            or top_receipt.get("proof_authority") is not True
            or top_receipt.get("target_relu_ids")
            != [int(value) for value in targets]
            or top_receipt.get("process_local_validation") is not True
            or top_receipt.get("receipt_rehydration_authority") is not False
        ):
            return False

        for key in (
            "root_boxes_sha256",
            "final_boxes_sha256",
            "property_spec_sha256",
            "property_upper_sha256",
        ):
            value = bundle_receipt.get(key)
            if (
                not _is_lower_sha256(value)
                or receipt.get(key) != value
                or top_receipt.get(key) != value
            ):
                return False
        transaction_sha256 = bundle_receipt.get("receipt_sha256")
        if (
            not _is_lower_sha256(transaction_sha256)
            or receipt.get("transaction_receipt_sha256")
            != transaction_sha256
            or top_receipt.get("transaction_receipt_sha256")
            != transaction_sha256
        ):
            return False

        layer_metadata = operator_build.metadata.get("layers")
        if not isinstance(layer_metadata, list):
            return False
        metadata_by_id = {
            int(item["layer_id"]): item
            for item in layer_metadata
            if isinstance(item, Mapping) and "layer_id" in item
        }
        if len(metadata_by_id) != len(layer_metadata):
            return False
        for target in targets:
            if (
                target not in feedback.certified_bounds
                or target not in metadata_by_id
            ):
                return False
            certified = feedback.certified_bounds[target]
            lower = np.ascontiguousarray(
                torch.as_tensor(certified.lb)
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .numpy()
                .reshape(-1)
            )
            upper = np.ascontiguousarray(
                torch.as_tensor(certified.ub)
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .numpy()
                .reshape(-1)
            )
            bound_receipt = metadata_by_id[target].get(
                "verified_query_dual_bound"
            )
            if (
                not isinstance(bound_receipt, Mapping)
                or metadata_by_id[target].get(
                    "preactivation_bound_source"
                )
                != "verified_query_dual_replay_intersection"
                or bound_receipt.get("schema")
                != "operator_hz_verified_query_dual_relu_bound_v1"
                or bound_receipt.get("proof_authority") is not True
                or bound_receipt.get("layer_id") != target
                or bound_receipt.get("bound_sha256")
                != _f64_array_sha256(np.stack([lower, upper]))
                or bound_receipt.get("root_boxes_sha256")
                != bundle_receipt["root_boxes_sha256"]
                or bound_receipt.get("final_boxes_sha256")
                != bundle_receipt["final_boxes_sha256"]
                or bound_receipt.get("property_spec_sha256")
                != bundle_receipt["property_spec_sha256"]
                or bound_receipt.get("property_upper_sha256")
                != bundle_receipt["property_upper_sha256"]
                or bound_receipt.get("transaction_receipt_sha256")
                != transaction_sha256
            ):
                return False

        final_hz = operator_build.hz
        if (
            np.any(actual_rows < 0)
            or np.any(actual_rows >= int(final_hz.n_out))
        ):
            return False
        actual_center = np.ascontiguousarray(
            np.asarray(final_hz.c, dtype=np.float64)[actual_rows]
        )
        if (
            actual_center.shape != expected.shape
            or not np.array_equal(
                actual_center.view(np.uint64),
                expected.view(np.uint64),
            )
            or final_hz.Gc[actual_rows, :].nnz != 0
            or (
                final_hz.Gb is not None
                and final_hz.Gb[actual_rows, :].nnz != 0
            )
        ):
            return False
        return True
    except (
        AttributeError,
        ImportError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return False


def _first_batched_value(value: Any, B: int) -> Any:
    """Return sample-0 high-level ASSERT value when a legacy net pre-batched it."""

    if not isinstance(value, torch.Tensor) or B <= 1 or value.dim() < 2:
        return value
    if int(value.shape[0]) == int(B):
        return value[0].contiguous()
    return value


def _ensure_assert_linear_encoding(
    assert_layer: Any,
    B: int,
    n_out: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Encode legacy high-level ASSERT params for ``verify_once``.

    New front-end paths already store ``C`` / ``thresholds`` / ``M`` on ASSERT.
    Some CI example nets still carry only high-level fields such as ``c``/``d``
    or ``y_true``.  This compatibility shim uses the same ``OutputSpec`` encoder
    rather than duplicating row semantics.
    """

    params = assert_layer.params
    if all(k in params for k in ("C", "thresholds", "M")):
        return

    kind = params.get("kind")
    kwargs: Dict[str, Any] = {}
    if kind in (OutKind.LINEAR_LE, OutKind.RANGE, OutKind.UNSAFE_LINEAR):
        for key in ("c", "d", "lb", "ub"):
            if key in params:
                kwargs[key] = _first_batched_value(params[key], B)
    elif kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
        for key in ("y_true", "margin"):
            if key in params:
                kwargs[key] = params[key]
    else:
        raise ValueError(f"verify_once: unsupported ASSERT kind without C encoding: {kind!r}")

    encoded = OutputSpec(kind=kind, **kwargs).encode_linear(
        B=B,
        n_out=n_out,
        device=device,
        dtype=dtype,
    )
    params.update(encoded)


@overload
def verify_once(
    net,
    *,
    model_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    counterexample_replay_fn: Optional[Callable[[torch.Tensor], Any]] = None,
    backend_cfg: Optional[Any] = None,
    verified_query_dual_feedback: Optional[Any] = None,
    fail_fast_on_query_dual_fallback: bool = False,
    raw_vnnlib_path: Optional[Any] = None,
    expected_raw_vnnlib_sha256: Optional[str] = None,
) -> List[VerifyResult]:
    ...


@overload
def verify_once(
    net,
    *,
    model_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    counterexample_replay_fn: Optional[Callable[[torch.Tensor], Any]] = None,
    backend_cfg: Optional[Any] = None,
    verified_query_dual_feedback: Optional[Any] = None,
    fail_fast_on_query_dual_fallback: bool = False,
    raw_vnnlib_path: Optional[Any] = None,
    expected_raw_vnnlib_sha256: Optional[str] = None,
    collect_facts: Literal[False],
) -> List[VerifyResult]:
    ...


@overload
def verify_once(
    net,
    *,
    model_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    counterexample_replay_fn: Optional[Callable[[torch.Tensor], Any]] = None,
    backend_cfg: Optional[Any] = None,
    verified_query_dual_feedback: Optional[Any] = None,
    fail_fast_on_query_dual_fallback: bool = False,
    raw_vnnlib_path: Optional[Any] = None,
    expected_raw_vnnlib_sha256: Optional[str] = None,
    collect_facts: Literal[True],
) -> Tuple[List[VerifyResult], Optional[Dict[int, Any]]]:
    ...


@torch.no_grad()
def verify_once(
    net,
    *,
    model_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    counterexample_replay_fn: Optional[Callable[[torch.Tensor], Any]] = None,
    backend_cfg: Optional[Any] = None,
    verified_query_dual_feedback: Optional[Any] = None,
    fail_fast_on_query_dual_fallback: bool = False,
    raw_vnnlib_path: Optional[Any] = None,
    expected_raw_vnnlib_sha256: Optional[str] = None,
    collect_facts: bool = False,
) -> List[VerifyResult] | Tuple[List[VerifyResult], Optional[Dict[int, Any]]]:
    """Single-shot, pure-tensor batched verifier.

    Pipeline:

      1. Seed bounds from INPUT_SPEC layers (already shaped ``[B, *input_shape]``).
      2. ``analyze`` propagates batched bounds through every layer.
      3. Read pre-encoded ``C`` / ``thresholds`` / ``M`` from the ASSERT
         layer params (encoding lives in ``OutputSpec.encode_linear`` on the
         front-end; verify_once does no kind-dispatch).
      4. INTERVAL CERTIFICATION: in one tensor pass, compute the
         per-row interval upper bound of ``C @ y`` and compare to the
         per-lane threshold; ALL of a sample's M lanes must pass for that
         sample to be CERTIFIED.
      5. FALSIFICATION EVIDENCE: optionally evaluate ``model_fn`` at the box
         centre, or (when explicitly enabled) run the verifier-owned exact
         phase-projection rule.  The latter is a formal forward proof over a
         solver-produced candidate, not input sampling or ONNX execution.
      6. Remaining samples are UNKNOWN.

    Args:
        net: an ACT ``Net`` whose first layer is INPUT, last layer is ASSERT,
            and whose INPUT_SPEC layers carry already-batchified
            ``[B, *input_shape]`` lb/ub.
        model_fn: optional callable mapping ``x: [B, *input_shape] ->
            [B, n_out]`` for concrete falsification.  If omitted, FALSIFIED
            still requires evidence and is available only through an enabled
            verifier-owned formal rule such as exact phase projection.
        counterexample_replay_fn: independent authority for HybridZ witness
            acceptance. It receives one CPU-batched decoded input and must
            replay the original model and raw property, returning ``bool`` or
            a receipt mapping with ``valid_counterexample``/``valid``/
            ``accepted``. Without this gate, a HybridZ UNSAFE candidate is
            reported as UNKNOWN, never FALSIFIED.
        backend_cfg: optional ``BackendConfig``. The strict HybridZ solver path
            reads its ``hybridz`` sub-config when present.
        verified_query_dual_feedback: Optional live, process-local transaction
            produced by the independent query-dual pipeline.  Serialized
            receipts cannot substitute for the capability-bearing object.
        fail_fast_on_query_dual_fallback: when true, an enabled query-dual
            pipeline that forms a proofless fallback receipt returns UNKNOWN
            before residual selection or Operator-HZ construction.  The
            default false preserves the general verifier's baseline fallback.
        collect_facts: when true, return the verifier results together with
            the fact map used by validation: analyze() ``after`` facts for the
            interval/hybridz path, or dual pre-activation forward bounds for the
            dual path.

    Returns:
        ``List[VerifyResult]`` of length ``B`` (one per input lane), or
        ``(results, facts_or_none)`` when ``collect_facts`` is true. Each
        result carries ``status`` plus a ``metadata['lane'] = i`` and any
        ``counterexample`` (a ``torch.Tensor`` of shape ``[*input_shape]``)
        for FALSIFIED lanes.
    """
    from act.back_end.analyze import analyze
    from act.back_end.transfer_functions import get_transfer_function

    if not isinstance(fail_fast_on_query_dual_fallback, bool):
        raise TypeError(
            "fail_fast_on_query_dual_fallback must be boolean"
        )

    # 1. Extract structure and seed.
    entry_id = find_entry_layer_id(net)
    input_ids = get_input_ids(net)
    output_ids = get_output_ids(net)
    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)

    seed_bounds = seed_from_input_specs(spec_layers)
    if seed_bounds.lb.dim() < 2:
        raise ValueError(
            f"verify_once: INPUT_SPEC seed must be batched [B, *input_shape], "
            f"got dim={seed_bounds.lb.dim()} shape={tuple(seed_bounds.lb.shape)}. "
            f"Use VerifiableModel._merge_specs_to_batch (front-end) or manually "
            f"expand INPUT_SPEC lb/ub to [B, ...] before calling verify_once."
        )
    B = seed_bounds.lb.shape[0]

    # Dual standalone dispatch: when ``--solver dual`` is set (dual moved
    # dual from the --tf-mode axis to the --solver axis), route through
    # DualSolver.evaluate_spec instead of analyze() + interval cert. LP/Gurobi
    # path remains authoritative for the LP-feeding TFs (interval/hybridz).
    # ``ensure_active_tf`` still self-heals the TF default for interval/hybridz
    # callers; ``is_dual_solver_active`` reads the orthogonal solver-mode global.
    from act.back_end.transfer_functions import (
        ensure_active_tf,
        is_dual_solver_active,
        is_hybridz_solver_active,
        set_transfer_function_mode,
    )
    hybridz_solver = is_hybridz_solver_active()
    hz_cfg = getattr(backend_cfg, "hybridz", None) if hybridz_solver else None
    hz_engine = getattr(hz_cfg, "engine", "dense_hz_objbound") if hybridz_solver else None
    operator_hz_engine = hz_engine == "operator_hz_objbound"
    hybridz_started = time.monotonic()
    hybridz_total_timeout = None
    hybridz_deadline = None
    if hybridz_solver:
        fallback_timeout = (
            getattr(backend_cfg, "timeout", None)
            if backend_cfg is not None
            else None
        )
        hybridz_total_timeout = max(
            0.0,
            _hybridz_timeout(hz_cfg, fallback_timeout),
        )
        hybridz_deadline = hybridz_started + hybridz_total_timeout
    phase_clique_progress_enabled = bool(
        hybridz_solver
        and operator_hz_engine
        and float(
            getattr(
                hz_cfg,
                "operator_phase_clique_time_limit",
                0.0,
            )
        )
        > 0.0
    )

    def _phase_clique_progress(stage: str) -> None:
        if not phase_clique_progress_enabled:
            return
        now = time.monotonic()
        remaining = (
            max(0.0, float(hybridz_deadline) - now)
            if hybridz_deadline is not None
            else 0.0
        )
        print(
            "[verifier-phase-clique] "
            f"stage={stage} "
            f"elapsed={max(0.0, now - hybridz_started):.6f}s "
            f"remaining={remaining:.6f}s",
            flush=True,
        )
    if hybridz_solver:
        # The operator-backed path needs interval facts but constructs its HZ
        # graph afterwards.  Avoid paying for (and then discarding) ordinary
        # dense/sparse HZ propagation.
        set_transfer_function_mode("interval" if operator_hz_engine else "hybridz")
    active_tf = ensure_active_tf(
        "interval"
        if (not hybridz_solver or operator_hz_engine)
        else "hybridz"
    )
    sparse_hz_engine = hz_engine in {"sparse_hz", "sparse_hz_objbound"}
    if hybridz_solver and hasattr(active_tf, "enable_sparse_hz"):
        active_tf.enable_sparse_hz(bool(sparse_hz_engine and B == 1))

    if is_dual_solver_active():
        from act.back_end.solver.solver_dual import DualSolver
        from act.front_end.specs import OutputSpec

        def _unbatch(val: Any) -> Any:
            # ASSERT params are pre-batchified ([B, ...]) by FE; OutputSpec
            # constructor expects unbroadcasted scalar/1-D form. Single-property
            # batch verification: all rows share the same spec, so row 0 is the
            # canonical form. Per-sample-varying spec support is a future task.
            if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.shape[0] == B:
                return val[0]
            return val

        out_spec = OutputSpec(
            kind=assert_layer.params.get("kind"),
            c=_unbatch(assert_layer.params.get("c")),
            d=_unbatch(assert_layer.params.get("d")),
            y_true=assert_layer.params.get("y_true"),
            margin=_unbatch(assert_layer.params.get("margin")),
            lb=_unbatch(assert_layer.params.get("lb")),
            ub=_unbatch(assert_layer.params.get("ub")),
        )
        num_classes = len(output_ids)
        # DualSolver is now self-contained: no tf parameter, evaluate_spec
        # computes its own pre-activation forward bounds internally from the net.
        solver = DualSolver()
        result = solver.evaluate_spec(
            net,
            out_spec,
            num_classes=num_classes,
            collect_bounds=collect_facts,
        )
        results = result.to_verify_results()
        if collect_facts:
            return results, solver.last_forward_bounds
        return results

    # 2. Build entry_fact (with all INPUT_SPEC constraints) and analyze.
    entry_fact = Fact(bounds=seed_bounds, cons=ConSet())
    add_all_input_specs(entry_fact.cons, input_ids, spec_layers)
    config_old_values = (
        _apply_hybridz_tf_config(active_tf, hz_cfg)
        if hybridz_solver and not operator_hz_engine else []
    )
    _phase_clique_progress("interval_analyze_start")
    try:
        _before, after, _globalC = analyze(net, entry_id, entry_fact)
    finally:
        if config_old_values:
            _restore_hybridz_tf_profile(active_tf, config_old_values)
    _phase_clique_progress("interval_analyze_done")

    # 3. Pull output bounds (pre-ASSERT layer's Fact).
    output_bounds = _get_output_layer_bounds(net, after)
    output_lb = output_bounds.lb
    output_ub = output_bounds.ub
    if output_lb.dim() != 2 or output_lb.shape[0] != B:
        raise ValueError(
            f"verify_once: output bounds must be [B={B}, n_out], got "
            f"shape={tuple(output_lb.shape)}. Some TF op on this network's "
            f"path collapsed the leading batch dimension."
        )
    n_out = output_lb.shape[1]
    if n_out != len(output_ids):
        raise ValueError(
            f"verify_once: output_lb has n_out={n_out} but ASSERT.in_vars "
            f"has length {len(output_ids)}"
        )
    device = output_lb.device
    dtype = output_lb.dtype

    # 4. Read pre-encoded ASSERT params (produced by OutputSpec.encode_linear
    # at FE construction time). Dispatch on ``kind`` because UNSAFE_LINEAR
    # has EXISTS-row safety semantics while the four other kinds (LINEAR_LE,
    # TOP1_ROBUST, MARGIN_ROBUST, RANGE) share an ALL-rows form.
    _ensure_assert_linear_encoding(
        assert_layer,
        B=B,
        n_out=n_out,
        device=device,
        dtype=dtype,
    )
    C = assert_layer.params["C"].to(device=device, dtype=dtype)
    thresholds = assert_layer.params["thresholds"].to(device=device, dtype=dtype)
    M = int(assert_layer.params["M"])
    kind = assert_layer.params.get("kind")
    is_unsafe_linear = kind == OutKind.UNSAFE_LINEAR
    assert C.dim() == 2 and C.shape == (B * M, n_out), (
        f"verify_once: ASSERT params['C'].shape={tuple(C.shape)} "
        f"expected ({B * M}, {n_out})"
    )
    assert thresholds.shape == (B, M), (
        f"verify_once: ASSERT params['thresholds'].shape="
        f"{tuple(thresholds.shape)} expected ({B}, {M})"
    )

    if hybridz_solver:
        meta: Dict[str, Any] = {
            "solver": "hybridz",
            "B": B,
            "M": M,
            "engine": hz_engine,
        }
        meta.update(_hybridz_config_metadata(hz_cfg))
        meta["operator_phase_clique_materialization"] = (
            _operator_phase_clique_disabled_receipt()
        )
        if B != 1:
            return [
                VerifyResult(
                    VerifyStatus.UNKNOWN,
                    metadata={**meta, "lane": i, "reason": "hybridz_batched_not_supported"},
                )
                for i in range(B)
            ]
        output_layer_id = _get_output_layer_id(net)
        C_np = np.ascontiguousarray(
            C.detach().cpu().double().numpy().reshape(M, n_out),
            dtype=np.float64,
        )
        thresholds_np = np.ascontiguousarray(
            thresholds.detach().cpu().double().numpy().reshape(M),
            dtype=np.float64,
        )
        phase_projection_seconds = float(
            getattr(
                hz_cfg,
                "operator_phase_projection_time_limit",
                0.0,
            )
        )
        meta["operator_phase_projection"] = {
            "schema": "verifier_operator_phase_projection_v1",
            "enabled": bool(
                operator_hz_engine and phase_projection_seconds > 0.0
            ),
            "status": "not_run",
            "configured_seconds": phase_projection_seconds,
            "verifier_owned_proof_authority": False,
            "input_sampling_used": False,
            "pgd_used": False,
            "concrete_onnx_execution_used": False,
            "bab_used": False,
            "backward_used": False,
            "dual_tightening_used": False,
        }
        if operator_hz_engine and phase_projection_seconds > 0.0:
            from act.back_end.hybridz_tf.forward_exact_relu_phase_projection_candidate import (
                ExactReLUPhaseProjectionUnknown,
                build_forward_exact_relu_phase_projection_candidate,
            )

            def _finish_phase_projection(
                result: VerifyResult,
            ) -> (
                List[VerifyResult]
                | Tuple[List[VerifyResult], Optional[Dict[int, Any]]]
            ):
                results = [result]
                if collect_facts:
                    return results, after
                return results

            projection_started = time.monotonic()
            projection_deadline = min(
                float(hybridz_deadline),
                projection_started + phase_projection_seconds,
            )
            try:
                projection = (
                    build_forward_exact_relu_phase_projection_candidate(
                        net,
                        int(entry_id),
                        _before,
                        after,
                        deadline=projection_deadline,
                        lp_time_limit=min(
                            30.0, phase_projection_seconds
                        ),
                    )
                )
            except ExactReLUPhaseProjectionUnknown as exc:
                meta["operator_phase_projection"].update(
                    {
                        "status": "UNKNOWN",
                        "reason": str(exc),
                        "elapsed_seconds": (
                            time.monotonic() - projection_started
                        ),
                    }
                )
                return _finish_phase_projection(
                    VerifyResult(
                        VerifyStatus.UNKNOWN,
                        metadata=meta,
                    )
                )
            except Exception as exc:
                meta["operator_phase_projection"].update(
                    {
                        "status": "UNKNOWN",
                        "reason": (
                            "unexpected_fail_closed:"
                            f"{type(exc).__name__}"
                        ),
                        "elapsed_seconds": (
                            time.monotonic() - projection_started
                        ),
                    }
                )
                return _finish_phase_projection(
                    VerifyResult(
                        VerifyStatus.UNKNOWN,
                        metadata=meta,
                    )
                )
            else:
                projection_receipt = dict(vars(projection.receipt))
                meta["operator_phase_projection"].update(
                    {
                        "status": "FALSIFIED",
                        "elapsed_seconds": (
                            time.monotonic() - projection_started
                        ),
                        "verifier_owned_proof_authority": True,
                        "proof_rule": (
                            "decoded_input_in_raw_BOX;"
                            "verifier_owned_zero_width_forward_interval;"
                            "exact_Fraction_property_lower_bound_positive"
                        ),
                        "candidate_receipt": projection_receipt,
                    }
                )
                counterexample = torch.from_numpy(
                    np.array(
                        projection.decoded_input,
                        dtype=np.float64,
                        order="C",
                        copy=True,
                    )
                )[0]
                return _finish_phase_projection(
                    VerifyResult(
                        VerifyStatus.FALSIFIED,
                        counterexample=counterexample,
                        metadata=meta,
                    )
                )
        query_targets = tuple(
            int(value)
            for value in getattr(
                hz_cfg, "query_dual_feedback_targets", ()
            )
        )
        query_steps = int(
            getattr(hz_cfg, "query_dual_feedback_steps", 0)
        )
        query_time_limit = float(
            getattr(hz_cfg, "query_dual_feedback_time_limit", 0.0)
        )
        query_block_size = int(
            getattr(hz_cfg, "query_dual_feedback_block_size", 1024)
        )
        query_device = str(
            getattr(hz_cfg, "query_dual_feedback_device", "cuda")
        ).lower()
        property_only_bound_replay = bool(
            query_steps > 0
            and not query_targets
            and getattr(hz_cfg, "residual_bound_screen", False)
        )
        effective_query_feedback = verified_query_dual_feedback
        query_transaction_metadata = _query_dual_disabled_metadata(
            targets=query_targets,
            steps=query_steps,
            time_limit=query_time_limit,
            block_size=query_block_size,
            device=query_device,
        )
        meta["query_dual_feedback_transaction"] = (
            query_transaction_metadata
        )
        hz = None
        operator_phase_clique_pipeline_result = None
        operator_phase_clique_source_build = None
        operator_phase_clique_solver_build = None
        sparse_drop_reason = None
        property_upper_output = False
        property_upper_row_groups = ()
        property_only_query_upper = None
        phase_split_mode = False
        phase_focus_rival_ids: Tuple[int, ...] = ()
        phase_focus_rivals_by_target: Dict[
            Tuple[int, int], Tuple[int, ...]
        ] = {}
        if operator_hz_engine:
            from act.back_end.hybridz_tf.operator_hz import (
                OperatorHZBuildError,
                OperatorHZBuildTimeout,
                _ConstraintBlock,
                _constraint_blocks_sha256,
                _csr_sha256,
                build_operator_hz,
            )
            residual_targets = None
            correlation_targets = None
            residual_selector_receipt = None
            property_tail_requested = bool(
                getattr(hz_cfg, "property_tail_upper", False)
            )
            property_tail_enabled = bool(
                property_tail_requested and not is_unsafe_linear
            )
            operator_exact_budget = int(
                getattr(hz_cfg, "operator_exact_budget", 0)
            )
            residual_budget = int(
                getattr(hz_cfg, "property_residual_budget", 0)
            )
            residual_time_limit = float(
                getattr(hz_cfg, "property_residual_time_limit", 0.0)
            )
            phase_clique_time_limit = float(
                getattr(
                    hz_cfg,
                    "operator_phase_clique_time_limit",
                    0.0,
                )
            )
            phase_clique_enabled = phase_clique_time_limit > 0.0
            _phase_clique_progress("operator_preselection_start")
            phase_split_mode = bool(
                property_tail_enabled
                and 1 <= operator_exact_budget <= 2
                and residual_budget == operator_exact_budget
                and residual_time_limit > 0.0
            )
            meta["property_phase_split_request"] = {
                "schema": "verifier_property_phase_split_request_v1",
                "enabled": phase_split_mode,
                "depth": (
                    int(operator_exact_budget) if phase_split_mode else 0
                ),
                "selector_budget": int(residual_budget),
                "selector_time_limit": float(residual_time_limit),
                "proof_rule": (
                    "property_selected_exact_relu_binary_factors;"
                    "enumerate_all_signs;"
                    "every_continuous_child_must_independently_certify"
                    if phase_split_mode
                    else None
                ),
                "proof_authority": False,
            }
            meta["property_tail_upper_request"] = {
                "requested": property_tail_requested,
                "enabled": property_tail_enabled,
                "status": (
                    "enabled"
                    if property_tail_enabled
                    else "unsupported_joint_unsafe"
                    if property_tail_requested
                    else "disabled"
                ),
                "proof_authority": False,
            }
            if effective_query_feedback is not None:
                explicit_started = time.monotonic()
                explicit_targets = query_targets
                try:
                    explicit_targets = tuple(
                        int(value)
                        for value in effective_query_feedback.target_relu_ids
                    )
                    if (
                        query_steps > 0
                        and explicit_targets != query_targets
                    ):
                        raise ValueError(
                            "explicit query-dual target schedule differs "
                            "from the enabled HybridZ configuration"
                        )
                    query_transaction_metadata = (
                        _query_dual_pending_metadata(
                            effective_query_feedback,
                            source="explicit_live_object",
                            targets=explicit_targets,
                            steps=query_steps,
                            time_limit=query_time_limit,
                            block_size=query_block_size,
                            device=query_device,
                            elapsed_seconds=(
                                time.monotonic() - explicit_started
                            ),
                            bind_effective_config=False,
                        )
                    )
                except Exception as exc:
                    query_transaction_metadata = (
                        _query_dual_error_metadata(
                            targets=explicit_targets,
                            steps=query_steps,
                            time_limit=query_time_limit,
                            block_size=query_block_size,
                            device=query_device,
                            elapsed_seconds=(
                                time.monotonic() - explicit_started
                            ),
                            exc=exc,
                            deadline=False,
                            source="explicit_live_object",
                        )
                    )
                    effective_query_feedback = None
                meta["query_dual_feedback_transaction"] = (
                    query_transaction_metadata
                )
            elif query_steps > 0 and not property_only_bound_replay:
                query_started = time.monotonic()
                try:
                    if not query_targets:
                        raise ValueError(
                            "enabled query-dual feedback has no target ReLUs"
                        )
                    if not property_tail_enabled:
                        raise ValueError(
                            "query-dual feedback requires an ALL-rows "
                            "property-tail upper encoding"
                        )
                    if query_time_limit <= 0.0:
                        raise ValueError(
                            "enabled query-dual feedback has no time budget"
                        )
                    from act.back_end.hybridz_tf.query_dual_pipeline import (
                        build_verified_query_dual_feedback,
                    )

                    effective_query_feedback = (
                        build_verified_query_dual_feedback(
                            net,
                            C_np,
                            thresholds_np,
                            target_relu_ids=query_targets,
                            steps=query_steps,
                            block_size=query_block_size,
                            replay_chunk_size=query_block_size,
                            candidate_device=query_device,
                            deadline=hybridz_deadline,
                            timeout_s=query_time_limit,
                        )
                    )
                    query_transaction_metadata = (
                        _query_dual_pending_metadata(
                            effective_query_feedback,
                            source="built_in_verify_once",
                            targets=query_targets,
                            steps=query_steps,
                            time_limit=query_time_limit,
                            block_size=query_block_size,
                            device=query_device,
                            elapsed_seconds=(
                                time.monotonic() - query_started
                            ),
                            bind_effective_config=True,
                        )
                    )
                except Exception as exc:
                    try:
                        from act.back_end.hybridz_tf.query_dual_pipeline import (
                            QueryDualPipelineTimeout,
                        )

                        deadline_failure = isinstance(
                            exc, QueryDualPipelineTimeout
                        )
                    except ImportError:
                        deadline_failure = isinstance(exc, TimeoutError)
                    query_transaction_metadata = (
                        _query_dual_error_metadata(
                            targets=query_targets,
                            steps=query_steps,
                            time_limit=query_time_limit,
                            block_size=query_block_size,
                            device=query_device,
                            elapsed_seconds=(
                                time.monotonic() - query_started
                            ),
                            exc=exc,
                            deadline=deadline_failure,
                        )
                    )
                    effective_query_feedback = None
                meta["query_dual_feedback_transaction"] = (
                    query_transaction_metadata
                )
            elif property_only_bound_replay and is_unsafe_linear:
                query_transaction_metadata = (
                    _query_dual_error_metadata(
                        targets=query_targets,
                        steps=query_steps,
                        time_limit=query_time_limit,
                        block_size=query_block_size,
                        device=query_device,
                        elapsed_seconds=0.0,
                        exc=ValueError(
                            "property-only residual-bound replay supports "
                            "ALL-rows properties only"
                        ),
                        deadline=False,
                    )
                )
                meta["query_dual_feedback_transaction"] = (
                    query_transaction_metadata
                )
            if (
                fail_fast_on_query_dual_fallback
                and query_steps > 0
                and query_transaction_metadata.get("proof_authority")
                is False
                and query_transaction_metadata.get("status")
                in {
                    "deadline_fallback_baseline",
                    "error_fallback_baseline",
                }
            ):
                results = [
                    VerifyResult(
                        VerifyStatus.UNKNOWN,
                        metadata={
                            **meta,
                            "lane": 0,
                            "reason": "query_dual_feedback_not_applied",
                        },
                    )
                ]
                return (results, after) if collect_facts else results
            correlation_budget = int(
                getattr(hz_cfg, "property_correlation_budget", 0)
            )
            correlation_time_limit = float(
                getattr(hz_cfg, "property_correlation_time_limit", 0.0)
            )
            if (
                not property_tail_enabled
                and correlation_budget > 0
                and correlation_time_limit > 0.0
            ):
                try:
                    from act.back_end.hybridz_tf.property_residual_targets import (
                        property_correlation_layer_quotas,
                        select_property_sparse_query_rows,
                    )

                    correlation_quotas = property_correlation_layer_quotas(
                        net,
                        budget=correlation_budget,
                        per_layer_cap=int(
                            getattr(
                                hz_cfg,
                                "property_residual_pool_per_rival",
                                8,
                            )
                        ),
                        before=_before,
                    )
                    correlation_plan = select_property_sparse_query_rows(
                        net=net,
                        before=_before,
                        after=after,
                        C=C_np,
                        thresholds=thresholds_np,
                        kind=kind,
                        output_layer_id=output_layer_id,
                        layer_quotas=correlation_quotas,
                        time_limit=correlation_time_limit,
                        deadline=hybridz_deadline,
                        max_adjoint_cells=int(
                            getattr(
                                hz_cfg,
                                "property_residual_max_adjoint_cells",
                                30_000_000,
                            )
                        ),
                        pool_per_rival=int(
                            getattr(
                                hz_cfg,
                                "property_residual_pool_per_rival",
                                8,
                            )
                        ),
                    )
                    correlation_targets = tuple(
                        (int(target.layer_id), int(target.row))
                        for target in correlation_plan.targets
                    )
                    selector_receipt = dict(correlation_plan.receipt)
                    selector_receipt.update(
                        {
                            "schema": (
                                "property_correlation_selector_v1"
                            ),
                            "source_selector_schema": (
                                correlation_plan.receipt.get("schema")
                            ),
                            "candidate_only": True,
                            "proof_authority": False,
                            "layer_quotas": [
                                [int(layer_id), int(quota)]
                                for layer_id, quota in sorted(
                                    correlation_quotas.items()
                                )
                            ],
                        }
                    )
                    meta.update(
                        {
                            "property_correlation_selector": (
                                selector_receipt
                            ),
                            "property_correlation_property_sha256": (
                                correlation_plan.property_sha256
                            ),
                            "property_correlation_targets_sha256": (
                                correlation_plan.selection_sha256
                            ),
                        }
                    )
                except Exception as exc:
                    meta["property_correlation_selector"] = {
                        "schema": "property_correlation_selector_v1",
                        "status": "error_fallback_baseline",
                        "proof_authority": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    }
                    correlation_targets = None
            else:
                meta["property_correlation_selector"] = {
                    "schema": "property_correlation_selector_v1",
                    "status": (
                        "disabled_by_property_tail"
                        if property_tail_enabled
                        else "disabled"
                    ),
                    "proof_authority": False,
                    "targets_selected": 0,
                }

            if (
                (phase_split_mode or not property_tail_enabled)
                and correlation_budget == 0
                and residual_budget > 0
                and residual_time_limit > 0.0
            ):
                try:
                    from act.back_end.hybridz_tf.property_residual_targets import (
                        select_property_residual_targets,
                    )

                    allowed_relu_layer_ids = None
                    phase_stop_layer_id = None
                    phase_stop_candidates = ()
                    if phase_split_mode:
                        from act.back_end.hybridz_tf.operator_hz import (
                            operator_hz_property_suffix_stop_layer_id,
                        )

                        phase_stop_layer_id, phase_stop_candidates = (
                            operator_hz_property_suffix_stop_layer_id(
                                net,
                                output_layer_id=int(output_layer_id),
                                suffix_blocks=int(
                                    getattr(
                                        hz_cfg,
                                        "property_tail_suffix_blocks",
                                        0,
                                    )
                                ),
                            )
                        )
                        topology_position = {
                            int(layer.id): index
                            for index, layer in enumerate(net.layers)
                        }
                        stop_position = topology_position[
                            int(phase_stop_layer_id)
                        ]
                        output_position = topology_position[
                            int(output_layer_id)
                        ]
                        suffix_relu_layer_ids = tuple(
                            int(layer.id)
                            for layer in net.layers[
                                stop_position + 1 : output_position
                            ]
                            if str(
                                getattr(layer.kind, "value", layer.kind)
                            ).upper()
                            == "RELU"
                        )
                        # The last ReLU is the property-tail snapshot itself;
                        # its exact graph is intentionally pruned by the
                        # safe-only tail.  Retained interior suffix ReLUs can
                        # instead guard branch-conditioned affine replays.
                        conditional_suffix_selection = bool(
                            suffix_relu_layer_ids
                        )
                        prefix_relu_layer_ids = tuple(
                            int(layer.id)
                            for layer in net.layers[:stop_position]
                            if str(
                                getattr(
                                    layer.kind,
                                    "value",
                                    layer.kind,
                                )
                            ).upper()
                            == "RELU"
                        )
                        allowed_relu_layer_ids = (
                            suffix_relu_layer_ids
                            if conditional_suffix_selection
                            else prefix_relu_layer_ids
                        )
                        if not allowed_relu_layer_ids:
                            raise ValueError(
                                "phase split has no retained interior suffix "
                                "ReLU layer"
                            )
                    residual_plan = select_property_residual_targets(
                        net=net,
                        before=_before,
                        after=after,
                        C=C_np,
                        thresholds=thresholds_np,
                        kind=kind,
                        output_layer_id=output_layer_id,
                        budget=residual_budget,
                        time_limit=residual_time_limit,
                        deadline=hybridz_deadline,
                        max_adjoint_cells=int(
                            getattr(
                                hz_cfg,
                                "property_residual_max_adjoint_cells",
                                30_000_000,
                            )
                        ),
                        pool_per_rival=int(
                            getattr(
                                hz_cfg,
                                "property_residual_pool_per_rival",
                                8,
                            )
                        ),
                        allowed_relu_layer_ids=allowed_relu_layer_ids,
                        phase_joint_focus_after_first=bool(
                            phase_clique_enabled
                            or (
                                phase_split_mode
                                and conditional_suffix_selection
                                and operator_exact_budget > 1
                            )
                        ),
                    )
                    if (
                        phase_split_mode
                        and conditional_suffix_selection
                        and not residual_plan.targets
                        and prefix_relu_layer_ids
                    ):
                        # A suffix can be entirely phase-stable (the scalar
                        # cancellation toy is one example).  Preserve the
                        # exact-cover fallback rather than turning an absent
                        # conditional candidate into a build failure.
                        allowed_relu_layer_ids = prefix_relu_layer_ids
                        conditional_suffix_selection = False
                        residual_plan = select_property_residual_targets(
                            net=net,
                            before=_before,
                            after=after,
                            C=C_np,
                            thresholds=thresholds_np,
                            kind=kind,
                            output_layer_id=output_layer_id,
                            budget=residual_budget,
                            time_limit=residual_time_limit,
                            deadline=hybridz_deadline,
                            max_adjoint_cells=int(
                                getattr(
                                    hz_cfg,
                                    "property_residual_max_adjoint_cells",
                                    30_000_000,
                                )
                            ),
                            pool_per_rival=int(
                                getattr(
                                    hz_cfg,
                                    "property_residual_pool_per_rival",
                                    8,
                                )
                            ),
                            allowed_relu_layer_ids=allowed_relu_layer_ids,
                            phase_joint_focus_after_first=False,
                        )
                    residual_targets = residual_plan.builder_targets
                    if phase_split_mode:
                        phase_focus_rival_ids = tuple(
                            dict.fromkeys(
                                int(target.dominant_rival)
                                for target in residual_plan.targets
                            )
                        )
                        phase_focus_rivals_by_target = {
                            (int(target.layer_id), int(target.row)): (
                                int(target.dominant_rival),
                            )
                            for target in residual_plan.targets
                        }
                    selector_receipt = dict(residual_plan.receipt)
                    residual_selector_receipt = selector_receipt
                    if phase_split_mode:
                        selector_receipt.update(
                            {
                                "schema": (
                                    "property_phase_split_selector_v1"
                                ),
                                "source_selector_schema": (
                                    residual_plan.receipt.get("schema")
                                ),
                                "role": (
                                    "candidate_only_exact_relu_phase_"
                                    "selection"
                                ),
                                "suffix_stop_layer_id": int(
                                    phase_stop_layer_id
                                ),
                                "suffix_stop_candidates_nearest_first": [
                                    int(value)
                                    for value in phase_stop_candidates
                                ],
                                "selected_layers_strictly_after_stop": bool(
                                    conditional_suffix_selection
                                    and
                                    all(
                                        int(target.layer_id)
                                        in set(allowed_relu_layer_ids)
                                        for target in residual_plan.targets
                                    )
                                ),
                                "selected_layers_strictly_before_stop": bool(
                                    not conditional_suffix_selection
                                    and all(
                                        int(target.layer_id)
                                        in set(allowed_relu_layer_ids)
                                        for target in residual_plan.targets
                                    )
                                ),
                                "conditional_suffix_replay_requested": bool(
                                    conditional_suffix_selection
                                ),
                                "proof_authority": False,
                            }
                        )
                    meta.update(
                        {
                            "property_residual_selector": dict(
                                selector_receipt
                            ),
                            "property_residual_property_sha256": (
                                residual_plan.property_sha256
                            ),
                            "property_residual_targets_sha256": (
                                residual_plan.targets_sha256
                            ),
                            "property_phase_split_targets": (
                                [
                                    {
                                        "layer_id": int(target.layer_id),
                                        "row": int(target.row),
                                        "score": float(target.score),
                                        "dominant_rival": int(
                                            target.dominant_rival
                                        ),
                                    }
                                    for target in residual_plan.targets
                                ]
                                if phase_split_mode
                                else []
                            ),
                        }
                    )
                except Exception as exc:
                    # Scheduling has no proof authority.  A selector failure
                    # therefore falls back to the unchanged triangle builder;
                    # the shared outer deadline still accounts for its time.
                    meta["property_residual_selector"] = {
                        "schema": "property_residual_selector_v1",
                        "status": "error_fallback_baseline",
                        "proof_authority": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    }
                    residual_targets = None
            else:
                meta["property_residual_selector"] = {
                    "schema": "property_residual_selector_v1",
                    "status": (
                        "disabled_by_property_tail"
                        if property_tail_enabled
                        else "disabled"
                    ),
                    "proof_authority": False,
                    "targets_selected": 0,
                }
            _phase_clique_progress("operator_preselection_done")
            _phase_clique_progress("operator_build_start")
            try:
                operator_build = build_operator_hz(
                    net,
                    _before,
                    after,
                    exact_budget=int(
                        getattr(hz_cfg, "operator_exact_budget", 0)
                    ),
                    materialize_add=bool(
                        getattr(hz_cfg, "operator_materialize_add", True)
                    ),
                    issue_constructive_nonempty_seal=(
                        phase_clique_enabled
                    ),
                    preactivation_lp_budget=int(
                        getattr(hz_cfg, "preactivation_lp_budget", 0)
                    ),
                    preactivation_lp_time_limit=float(
                        getattr(hz_cfg, "preactivation_lp_time_limit", 0.0)
                    ),
                    correlation_targets=correlation_targets,
                    residual_phase_screen=bool(
                        getattr(
                            hz_cfg, "residual_phase_screen", False
                        )
                    ),
                    residual_bound_screen=bool(
                        getattr(
                            hz_cfg, "residual_bound_screen", False
                        )
                    ),
                    residual_targets=residual_targets,
                    property_phase_focus_rivals=(
                        phase_focus_rivals_by_target
                        if phase_split_mode
                        else None
                    ),
                    property_micro_rlt_product_cap=int(
                        getattr(
                            hz_cfg,
                            "property_micro_rlt_product_cap",
                            0,
                        )
                    ),
                    property_micro_rlt_packet_mode=str(
                        getattr(
                            hz_cfg,
                            "property_micro_rlt_packet_mode",
                            "both",
                        )
                    ),
                    property_upper_C=(
                        C_np if property_tail_enabled else None
                    ),
                    property_upper_thresholds=(
                        thresholds_np if property_tail_enabled else None
                    ),
                    property_tail_add_source_planes=bool(
                        getattr(
                            hz_cfg,
                            "property_tail_add_source_planes",
                            False,
                        )
                    ),
                    property_tail_alpha_steps=int(
                        getattr(
                            hz_cfg, "property_tail_alpha_steps", 0
                        )
                    ),
                    property_tail_alpha_time_limit=float(
                        getattr(
                            hz_cfg,
                            "property_tail_alpha_time_limit",
                            0.0,
                        )
                    ),
                    property_tail_alpha_learning_rate=float(
                        getattr(
                            hz_cfg,
                            "property_tail_alpha_learning_rate",
                            0.08,
                        )
                    ),
                    property_tail_alpha_max_cells=int(
                        getattr(
                            hz_cfg,
                            "property_tail_alpha_max_cells",
                            50_000_000,
                        )
                    ),
                    property_tail_alpha_device=str(
                        getattr(
                            hz_cfg,
                            "property_tail_alpha_device",
                            "auto",
                        )
                    ),
                    property_tail_pairhull_budget=int(
                        getattr(
                            hz_cfg,
                            "property_tail_pairhull_budget",
                            0,
                        )
                    ),
                    property_tail_pairhull_time_limit=float(
                        getattr(
                            hz_cfg,
                            "property_tail_pairhull_time_limit",
                            0.0,
                        )
                    ),
                    property_tail_suffix_blocks=int(
                        getattr(
                            hz_cfg,
                            "property_tail_suffix_blocks",
                            0,
                        )
                    ),
                    property_tail_suffix_alpha_steps=int(
                        getattr(
                            hz_cfg,
                            "property_tail_suffix_alpha_steps",
                            0,
                        )
                    ),
                    property_tail_suffix_alpha_time_limit=float(
                        getattr(
                            hz_cfg,
                            "property_tail_suffix_alpha_time_limit",
                            0.0,
                        )
                    ),
                    property_tail_suffix_alpha_device=str(
                        getattr(
                            hz_cfg,
                            "property_tail_suffix_alpha_device",
                            "auto",
                        )
                    ),
                    verified_query_dual_feedback=(
                        effective_query_feedback
                    ),
                    deadline=hybridz_deadline,
                )
                _phase_clique_progress("operator_build_done")
                if phase_clique_progress_enabled:
                    build_diagnostic = operator_build.metadata
                    print(
                        "[verifier-operator-build-outcome] "
                        + json.dumps(
                            {
                                name: build_diagnostic.get(name)
                                for name in (
                                    "build_seconds",
                                    "n_layers",
                                    "n_cont",
                                    "n_bin",
                                    "n_eq",
                                    "n_ub",
                                    "value_nnz",
                                    "constraint_nnz",
                                    "exact_budget_requested",
                                    "exact_budget_used",
                                    "residual_target_count",
                                    "residual_targets_applied",
                                )
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                        flush=True,
                    )
                hz = operator_build.hz
                property_upper_output = bool(
                    operator_build.property_upper_output
                )
                property_upper_row_groups = tuple(
                    tuple(int(row) for row in group)
                    for group in operator_build.property_upper_row_groups
                )
                if property_only_bound_replay:
                    query_started = time.monotonic()
                    try:
                        frame = (
                            operator_build.verified_preactivation_frame
                        )
                        if frame is None:
                            raise ValueError(
                                "residual-bound Operator-HZ build did not "
                                "export a live preactivation frame"
                            )
                        if query_time_limit <= 0.0:
                            raise ValueError(
                                "property-only query replay has no time "
                                "budget"
                            )
                        from act.back_end.hybridz_tf.query_dual_pipeline import (
                            build_verified_query_dual_feedback,
                            validate_verified_query_dual_feedback,
                        )

                        effective_query_feedback = (
                            build_verified_query_dual_feedback(
                                net,
                                C_np,
                                thresholds_np,
                                target_relu_ids=(),
                                steps=query_steps,
                                block_size=query_block_size,
                                replay_chunk_size=query_block_size,
                                candidate_device=query_device,
                                deadline=hybridz_deadline,
                                timeout_s=query_time_limit,
                                verified_preactivation_frame=frame,
                            )
                        )
                        if not validate_verified_query_dual_feedback(
                            effective_query_feedback,
                            net=net,
                            property_rows=C_np,
                            thresholds=thresholds_np,
                            expected_target_relu_ids=(),
                        ):
                            raise ValueError(
                                "fresh property-only query transaction "
                                "failed live validation"
                            )
                        query_transaction_metadata = (
                            _query_dual_pending_metadata(
                                effective_query_feedback,
                                source="built_in_verify_once",
                                targets=(),
                                steps=query_steps,
                                time_limit=query_time_limit,
                                block_size=query_block_size,
                                device=query_device,
                                elapsed_seconds=(
                                    time.monotonic() - query_started
                                ),
                                bind_effective_config=True,
                            )
                        )
                        query_transaction_metadata = (
                            _query_dual_mark_property_only_applied(
                                query_transaction_metadata,
                                operator_metadata=operator_build.metadata,
                            )
                        )
                        property_only_query_upper = (
                            np.ascontiguousarray(
                                effective_query_feedback.property_upper,
                                dtype=np.float64,
                            ).reshape(-1)
                        )
                    except Exception as exc:
                        try:
                            from act.back_end.hybridz_tf.query_dual_pipeline import (
                                QueryDualPipelineTimeout,
                            )

                            deadline_failure = isinstance(
                                exc, QueryDualPipelineTimeout
                            )
                        except ImportError:
                            deadline_failure = isinstance(
                                exc, TimeoutError
                            )
                        query_transaction_metadata = (
                            _query_dual_error_metadata(
                                targets=(),
                                steps=query_steps,
                                time_limit=query_time_limit,
                                block_size=query_block_size,
                                device=query_device,
                                elapsed_seconds=(
                                    time.monotonic() - query_started
                                ),
                                exc=exc,
                                deadline=deadline_failure,
                            )
                        )
                        effective_query_feedback = None
                    meta["query_dual_feedback_transaction"] = (
                        query_transaction_metadata
                    )
                if property_upper_output:
                    tail_receipt = operator_build.metadata.get(
                        "property_tail_upper", {}
                    )
                    flattened_group_rows = [
                        int(row)
                        for group in property_upper_row_groups
                        for row in group
                    ]
                    property_digest = hashlib.sha256()
                    property_digest.update(
                        np.asarray(
                            C_np.shape, dtype=np.int64
                        ).tobytes()
                    )
                    property_digest.update(C_np.tobytes())
                    property_digest.update(
                        np.asarray(
                            thresholds_np.shape, dtype=np.int64
                        ).tobytes()
                    )
                    property_digest.update(thresholds_np.tobytes())
                    group_digest = hashlib.sha256(
                        repr(property_upper_row_groups).encode("ascii")
                    ).hexdigest()
                    baseline_plane_count = tail_receipt.get(
                        "baseline_plane_count"
                    )
                    alternative_plane_count = tail_receipt.get(
                        "alternative_plane_count"
                    )
                    exported_plane_count = tail_receipt.get(
                        "exported_plane_count"
                    )
                    alternative_rivals = tail_receipt.get(
                        "alternative_plane_rival_ids"
                    )
                    alternative_kinds = tail_receipt.get(
                        "alternative_plane_kinds"
                    )
                    alpha_receipt = tail_receipt.get(
                        "negative_alpha_candidates", {}
                    )
                    add_source_receipt = tail_receipt.get(
                        "add_source_planes", {}
                    )
                    pairhull_receipt = tail_receipt.get(
                        "pairhull_candidates", {}
                    )
                    suffix_receipt = tail_receipt.get(
                        "shared_suffix_replay", {}
                    )
                    allowed_alternative_kinds = {
                        "negative_alpha_materialized",
                        "add_source_alpha0",
                        "pairhull_joint_materialized",
                        "query_dual_shared_suffix_add_projection",
                        "query_dual_full_input_property_constant",
                        "verified_query_dual_property_constant",
                    }
                    alternative_kinds_valid = bool(
                        isinstance(alternative_kinds, list)
                        and isinstance(alternative_plane_count, int)
                        and len(alternative_kinds)
                        == alternative_plane_count
                        and all(
                            isinstance(value, str)
                            and value in allowed_alternative_kinds
                            for value in alternative_kinds
                        )
                    )
                    has_alpha_alternative = bool(
                        alternative_kinds_valid
                        and "negative_alpha_materialized"
                        in alternative_kinds
                    )
                    has_add_source_alternative = bool(
                        alternative_kinds_valid
                        and "add_source_alpha0" in alternative_kinds
                    )
                    has_verified_query_dual_alternative = bool(
                        alternative_kinds_valid
                        and "verified_query_dual_property_constant"
                        in alternative_kinds
                    )
                    has_suffix_alternative = bool(
                        alternative_kinds_valid
                        and any(
                            value
                            in {
                                "query_dual_shared_suffix_add_projection",
                                "query_dual_full_input_property_constant",
                            }
                            for value in alternative_kinds
                        )
                    )
                    verified_query_dual_actual_object_valid = bool(
                        (
                            effective_query_feedback is None
                            and not has_verified_query_dual_alternative
                        )
                        or (
                            effective_query_feedback is not None
                            and has_verified_query_dual_alternative
                            and _validate_verified_query_dual_property_actual_object(
                                effective_query_feedback,
                                net=net,
                                property_rows=C_np,
                                thresholds=thresholds_np,
                                operator_build=operator_build,
                                tail_receipt=tail_receipt,
                                property_row_groups=(
                                    property_upper_row_groups
                                ),
                                alternative_rivals=alternative_rivals,
                                alternative_kinds=alternative_kinds,
                            )
                        )
                    )
                    add_source_requested = bool(
                        getattr(
                            hz_cfg,
                            "property_tail_add_source_planes",
                            False,
                        )
                    )
                    pairhull_receipt_valid = (
                        _validate_property_tail_pairhull_receipt(
                            pairhull_receipt,
                            requested_budget=getattr(
                                hz_cfg,
                                "property_tail_pairhull_budget",
                                0,
                            ),
                            requested_time_limit=getattr(
                                hz_cfg,
                                "property_tail_pairhull_time_limit",
                                0.0,
                            ),
                            alternative_rivals=alternative_rivals,
                            alternative_kinds=alternative_kinds,
                            rival_count=M,
                        )
                    )
                    suffix_requested = int(
                        getattr(
                            hz_cfg,
                            "property_tail_suffix_blocks",
                            0,
                        )
                    )
                    suffix_alpha_steps_requested = int(
                        getattr(
                            hz_cfg,
                            "property_tail_suffix_alpha_steps",
                            0,
                        )
                    )
                    suffix_alpha_time_requested = float(
                        getattr(
                            hz_cfg,
                            "property_tail_suffix_alpha_time_limit",
                            0.0,
                        )
                    )
                    suffix_alpha_device_requested = str(
                        getattr(
                            hz_cfg,
                            "property_tail_suffix_alpha_device",
                            "auto",
                        )
                    )
                    suffix_rivals = (
                        [
                            int(alternative_rivals[index])
                            for index, value in enumerate(
                                alternative_kinds
                            )
                            if value
                            in {
                                "query_dual_shared_suffix_add_projection",
                                "query_dual_full_input_property_constant",
                            }
                        ]
                        if (
                            alternative_kinds_valid
                            and isinstance(alternative_rivals, list)
                            and len(alternative_rivals)
                            == len(alternative_kinds)
                        )
                        else []
                    )
                    suffix_mapping_valid = bool(
                        (
                            not has_suffix_alternative
                            and not suffix_rivals
                        )
                        or (
                            has_suffix_alternative
                            and len(suffix_rivals) == M
                            and sorted(suffix_rivals)
                            == list(range(M))
                        )
                    )
                    suffix_receipt_valid = False
                    try:
                        suffix_status = str(
                            suffix_receipt["status"]
                        )
                        suffix_common = bool(
                            suffix_receipt.get("schema")
                            == (
                                "operator_hz_property_suffix_"
                                "replay_v1"
                            )
                            and int(
                                suffix_receipt[
                                    "requested_earlier_blocks"
                                ]
                            )
                            == suffix_requested
                            and int(
                                suffix_receipt[
                                    "requested_alpha_steps"
                                ]
                            )
                            == suffix_alpha_steps_requested
                            and float(
                                suffix_receipt[
                                    "requested_alpha_time_limit"
                                ]
                            )
                            == suffix_alpha_time_requested
                            and str(
                                suffix_receipt[
                                    "requested_alpha_device"
                                ]
                            )
                            == suffix_alpha_device_requested
                            and bool(
                                suffix_receipt.get(
                                    "baseline_fallback_retained_per_rival"
                                )
                            )
                        )
                        if suffix_requested == 0:
                            suffix_receipt_valid = bool(
                                suffix_common
                                and suffix_status == "disabled"
                                and not has_suffix_alternative
                                and not bool(
                                    suffix_receipt.get(
                                        "proof_authority"
                                    )
                                )
                            )
                        elif not has_suffix_alternative:
                            suffix_receipt_valid = bool(
                                suffix_common
                                and suffix_status
                                == "error_fallback_baseline"
                                and not bool(
                                    suffix_receipt.get(
                                        "proof_authority"
                                    )
                                )
                            )
                        else:
                            by_id = {
                                int(layer.id): layer
                                for layer in net.layers
                            }
                            stop_lid = int(
                                suffix_receipt["stop_layer_id"]
                            )
                            stop_layer = by_id[stop_lid]
                            stop_kind = str(
                                getattr(
                                    stop_layer.kind,
                                    "value",
                                    stop_layer.kind,
                                )
                            ).upper()
                            candidates = [
                                int(value)
                                for value in suffix_receipt[
                                    "dominating_add_candidates_nearest_first"
                                ]
                            ]
                            deep_optimized_only = bool(
                                suffix_receipt.get("replay_strategy")
                                == "optimized_only_deep_suffix"
                            )
                            full_input_only = bool(
                                suffix_receipt.get("replay_strategy")
                                == "optimized_only_full_input"
                            )
                            optimized_single_replay = bool(
                                deep_optimized_only or full_input_only
                            )
                            hashes_valid = all(
                                isinstance(
                                    suffix_receipt.get(key), str
                                )
                                and len(suffix_receipt[key]) == 64
                                for key in (
                                    "coefficient_sha256",
                                    "scalar_sha256",
                                    "replay_net_sha256",
                                    "replay_bounds_sha256",
                                    "replay_query_sha256",
                                )
                            ) and (
                                (
                                    optimized_single_replay
                                    and suffix_receipt.get(
                                        "alpha_zero_replay_receipt_sha256"
                                    )
                                    is None
                                    and suffix_receipt.get(
                                        "alpha_one_replay_receipt_sha256"
                                    )
                                    is None
                                )
                                or (
                                    not deep_optimized_only
                                    and all(
                                        isinstance(
                                            suffix_receipt.get(key), str
                                        )
                                        and len(suffix_receipt[key]) == 64
                                        for key in (
                                            "alpha_zero_replay_receipt_sha256",
                                            "alpha_one_replay_receipt_sha256",
                                        )
                                    )
                                )
                            )
                            optimized_alpha = suffix_receipt.get(
                                "optimized_alpha"
                            )
                            optimized_selected = suffix_receipt.get(
                                "optimized_alpha_selected_rows"
                            )
                            optimized_alpha_valid = bool(
                                isinstance(optimized_alpha, Mapping)
                                and isinstance(optimized_selected, int)
                                and not isinstance(
                                    optimized_selected, bool
                                )
                                and 0 <= optimized_selected <= M
                                and (
                                    (
                                        suffix_alpha_steps_requested == 0
                                        and optimized_alpha.get("status")
                                        == "disabled"
                                        and optimized_selected == 0
                                    )
                                    or (
                                        suffix_alpha_steps_requested > 0
                                        and optimized_alpha.get("status")
                                        in {
                                            "replayed",
                                            "error_fallback_extremes",
                                        }
                                        and (
                                            optimized_alpha.get("status")
                                            == "replayed"
                                            or optimized_selected == 0
                                        )
                                        and (
                                            optimized_alpha.get("status")
                                            != "replayed"
                                            or all(
                                                isinstance(
                                                    optimized_alpha.get(
                                                        key
                                                    ),
                                                    str,
                                                )
                                                and len(
                                                    optimized_alpha[key]
                                                )
                                                == 64
                                                for key in (
                                                    "candidate_receipt_sha256",
                                                    "candidate_alpha_sha256",
                                                    "replay_receipt_sha256",
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            full_input_actual_valid = True
                            if full_input_only:
                                from act.back_end.hybridz_tf.query_dual_replay import (
                                    validate_query_dual_replay_result,
                                )

                                live_full_input = getattr(
                                    operator_build.hz,
                                    "_property_full_input_replay_result",
                                    None,
                                )
                                row_start = int(
                                    suffix_receipt["row_start"]
                                )
                                row_count = int(
                                    suffix_receipt["row_count"]
                                )
                                lower_values = np.asarray(
                                    getattr(
                                        live_full_input,
                                        "lower_bounds",
                                        np.zeros(0, dtype=np.float64),
                                    ),
                                    dtype=np.float64,
                                ).reshape(-1)
                                final_hz = operator_build.hz
                                full_input_actual_valid = bool(
                                    validate_query_dual_replay_result(
                                        live_full_input,
                                        expected_net_sha256=(
                                            suffix_receipt[
                                                "replay_net_sha256"
                                            ]
                                        ),
                                        expected_bounds_sha256=(
                                            suffix_receipt[
                                                "replay_bounds_sha256"
                                            ]
                                        ),
                                        expected_query_sha256=(
                                            suffix_receipt[
                                                "replay_query_sha256"
                                            ]
                                        ),
                                    )
                                    and row_count == M
                                    and lower_values.size == M
                                    and 0 <= row_start
                                    and row_start + row_count
                                    <= int(final_hz.n_out)
                                    and np.array_equal(
                                        final_hz.c[
                                            row_start:
                                            row_start + row_count
                                        ],
                                        -lower_values,
                                    )
                                    and final_hz.Gc[
                                        row_start:
                                        row_start + row_count,
                                        :,
                                    ].nnz
                                    == 0
                                    and final_hz.Gb[
                                        row_start:
                                        row_start + row_count,
                                        :,
                                    ].nnz
                                    == 0
                                    and optimized_alpha.get(
                                        "replay_receipt_sha256"
                                    )
                                    == live_full_input.receipt.get(
                                        "receipt_sha256"
                                    )
                                )
                            suffix_receipt_valid = bool(
                                suffix_common
                                and suffix_status == "applied"
                                and bool(
                                    suffix_receipt.get(
                                        "proof_authority"
                                    )
                                )
                                and suffix_mapping_valid
                                and (
                                    (
                                        full_input_only
                                        and stop_kind == "INPUT_SPEC"
                                        and suffix_receipt.get(
                                            "stop_layer_kind"
                                        )
                                        == "INPUT_SPEC"
                                        and suffix_requested == 8
                                        and suffix_receipt.get(
                                            "output_form"
                                        )
                                        == "full_input_property_constant"
                                        and suffix_receipt.get(
                                            "crosses_all_dominating_adds"
                                        )
                                        is True
                                    )
                                    or (
                                        not full_input_only
                                        and stop_kind == "ADD"
                                        and suffix_receipt.get(
                                            "stop_layer_kind"
                                        )
                                        == "ADD"
                                        and len(candidates)
                                        > suffix_requested
                                        and candidates[suffix_requested]
                                        == stop_lid
                                    )
                                )
                                and len(set(candidates))
                                == len(candidates)
                                and all(
                                    str(
                                        getattr(
                                            by_id[lid].kind,
                                            "value",
                                            by_id[lid].kind,
                                        )
                                    ).upper()
                                    == "ADD"
                                    for lid in candidates
                                )
                                and (
                                    (
                                        optimized_single_replay
                                        and suffix_requested >= 1
                                        and suffix_alpha_steps_requested > 0
                                        and suffix_receipt.get(
                                            "alpha_extremes"
                                        )
                                        == []
                                        and suffix_receipt.get(
                                            "uniform_endpoint_replays_omitted"
                                        )
                                        is True
                                        and optimized_selected == M
                                    )
                                    or (
                                        not deep_optimized_only
                                        and suffix_receipt.get(
                                            "alpha_extremes"
                                        )
                                        == [0.0, 1.0]
                                    )
                                )
                                and isinstance(
                                    suffix_receipt.get(
                                        "alpha_one_selected_rows"
                                    ),
                                    int,
                                )
                                and 0
                                <= suffix_receipt[
                                    "alpha_one_selected_rows"
                                ]
                                <= M
                                and suffix_receipt.get(
                                    "query_count"
                                )
                                == M
                                and suffix_receipt.get(
                                    "row_count"
                                )
                                == M
                                and hashes_valid
                                and optimized_alpha_valid
                                and full_input_actual_valid
                            )
                    except (
                        KeyError,
                        TypeError,
                        ValueError,
                        OverflowError,
                    ):
                        suffix_receipt_valid = False
                    add_source_rivals = (
                        [
                            int(alternative_rivals[index])
                            for index, value in enumerate(
                                alternative_kinds
                            )
                            if value == "add_source_alpha0"
                        ]
                        if (
                            alternative_kinds_valid
                            and isinstance(alternative_rivals, list)
                            and len(alternative_rivals)
                            == len(alternative_kinds)
                        )
                        else []
                    )
                    add_source_mapping_valid = bool(
                        (
                            not has_add_source_alternative
                            and not add_source_rivals
                        )
                        or (
                            has_add_source_alternative
                            and len(add_source_rivals) == M
                            and sorted(add_source_rivals)
                            == list(range(M))
                        )
                    )
                    relation_rows = add_source_receipt.get(
                        "materialized_relation_block_rows"
                    )
                    relation_tags = add_source_receipt.get(
                        "materialized_relation_block_tags"
                    )
                    relation_digest = add_source_receipt.get(
                        "materialized_relation_blocks_sha256"
                    )
                    add_source_receipt_valid = bool(
                        not has_add_source_alternative
                        or (
                            isinstance(relation_rows, list)
                            and isinstance(relation_tags, list)
                            and len(relation_rows) == len(relation_tags)
                            and all(
                                isinstance(value, int)
                                and not isinstance(value, bool)
                                and value >= 0
                                for value in relation_rows
                            )
                            and all(
                                isinstance(value, str)
                                and value.startswith("add_materialize:")
                                for value in relation_tags
                            )
                            and sum(relation_rows)
                            == add_source_receipt.get(
                                "materialized_new_ub"
                            )
                            and isinstance(relation_digest, str)
                            and len(relation_digest) == 64
                            and add_source_receipt.get(
                                "materialized_eq_block_count_before"
                            )
                            == add_source_receipt.get(
                                "materialized_eq_block_count_after"
                            )
                            and add_source_receipt.get(
                                "materialized_ub_block_count_after"
                            )
                            == (
                                add_source_receipt.get(
                                    "materialized_ub_block_count_before"
                                )
                                + len(relation_rows)
                            )
                        )
                    )
                    property_proof_object_valid = False
                    try:
                        prefix_n_cont = int(
                            tail_receipt["prefix_n_cont"]
                        )
                        final_hz = operator_build.hz
                        property_frame_n_cont = int(final_hz.n_cont)
                        micro_rlt_build_receipt = (
                            operator_build.metadata.get(
                                "property_micro_rlt", {}
                            )
                        )
                        if (
                            isinstance(
                                micro_rlt_build_receipt, Mapping
                            )
                            and micro_rlt_build_receipt.get("status")
                            == "applied"
                        ):
                            micro_result_counts = (
                                micro_rlt_build_receipt.get(
                                    "result_counts", {}
                                )
                            )
                            micro_base_counts = (
                                micro_rlt_build_receipt.get(
                                    "base_counts", {}
                                )
                            )
                            micro_new_factors = int(
                                micro_rlt_build_receipt[
                                    "new_product_factors"
                                ]
                            )
                            if not (
                                micro_rlt_build_receipt.get(
                                    "proof_authority"
                                )
                                is True
                                and micro_rlt_build_receipt.get(
                                    "live_result_validation_passed"
                                )
                                is True
                                and micro_rlt_build_receipt.get("scope")
                                == "parent_pre_phase_fix"
                                and _audit_live_operator_property_micro_rlt(
                                    final_hz,
                                    micro_rlt_build_receipt,
                                )
                                and micro_new_factors > 0
                                and int(
                                    micro_result_counts.get("n_cont", -1)
                                )
                                == int(final_hz.n_cont)
                                and int(
                                    micro_base_counts.get("n_cont", -1)
                                )
                                + micro_new_factors
                                == int(final_hz.n_cont)
                            ):
                                raise ValueError(
                                    "invalid property micro-RLT extension "
                                    "dimensions"
                                )
                            property_frame_n_cont -= micro_new_factors
                        if not (
                            0 <= prefix_n_cont <= property_frame_n_cont
                        ):
                            raise ValueError(
                                "invalid property-tail prefix width"
                            )
                        center_digest = hashlib.sha256(
                            np.ascontiguousarray(
                                final_hz.c, dtype=np.float64
                            ).tobytes()
                        ).hexdigest()
                        prefix_generators = final_hz.Gc[
                            :, :prefix_n_cont
                        ].tocsr()
                        trailing_generators = final_hz.Gc[
                            :, prefix_n_cont:property_frame_n_cont
                        ].tocoo()
                        trailing_count = int(
                            property_frame_n_cont - prefix_n_cont
                        )
                        error_vector = np.zeros(
                            int(final_hz.n_out), dtype=np.float64
                        )
                        diagonal_valid = bool(
                            trailing_generators.nnz == trailing_count
                            and (
                                trailing_count == 0
                                or (
                                    np.unique(
                                        trailing_generators.col
                                    ).size
                                    == trailing_count
                                    and np.unique(
                                        trailing_generators.row
                                    ).size
                                    == trailing_count
                                    and np.all(
                                        np.isfinite(
                                            trailing_generators.data
                                        )
                                    )
                                    and np.all(
                                        trailing_generators.data > 0.0
                                    )
                                )
                            )
                        )
                        if diagonal_valid and trailing_count:
                            error_vector[trailing_generators.row] = (
                                trailing_generators.data
                            )
                        error_digest = hashlib.sha256(
                            np.ascontiguousarray(
                                error_vector, dtype=np.float64
                            ).tobytes()
                        ).hexdigest()
                        constraint_tail_zero = bool(
                            final_hz.Ac[
                                :,
                                prefix_n_cont:property_frame_n_cont,
                            ].nnz
                            == 0
                            and (
                                final_hz.Auc is None
                                or final_hz.Auc[
                                    :,
                                    prefix_n_cont:property_frame_n_cont,
                                ].nnz
                                == 0
                            )
                        )
                        property_proof_object_valid = bool(
                            center_digest
                            == tail_receipt.get(
                                "upper_expression_center_sha256"
                            )
                            and _csr_sha256(prefix_generators)
                            == tail_receipt.get(
                                "upper_expression_generator_sha256"
                            )
                            and error_digest
                            == tail_receipt.get(
                                "upper_expression_error_sha256"
                            )
                            and diagonal_valid
                            and constraint_tail_zero
                            and trailing_count
                            == int(
                                operator_build.metadata.get(
                                    "output_roundoff_generator_count",
                                    -1,
                                )
                            )
                        )
                    except (
                        KeyError,
                        TypeError,
                        ValueError,
                        OverflowError,
                    ):
                        property_proof_object_valid = False

                    add_source_actual_object_valid = bool(
                        not has_add_source_alternative
                    )
                    if has_add_source_alternative:
                        try:
                            relation_before = int(
                                add_source_receipt[
                                    "materialized_ub_block_count_before"
                                ]
                            )
                            relation_after = int(
                                add_source_receipt[
                                    "materialized_ub_block_count_after"
                                ]
                            )
                            materialized_n_cont = int(
                                add_source_receipt[
                                    "materialized_n_cont_after"
                                ]
                            )
                            materialized_n_bin = int(
                                add_source_receipt[
                                    "materialized_n_bin"
                                ]
                            )
                            constraint_meta = operator_build.metadata[
                                "constraint_tags_ub"
                            ]
                            if not (
                                0 <= relation_before <= relation_after
                                <= len(constraint_meta)
                                and materialized_n_cont
                                <= int(tail_receipt["prefix_n_cont"])
                                and materialized_n_bin
                                <= int(final_hz.n_bin)
                            ):
                                raise ValueError(
                                    "invalid retained ADD relation frame"
                                )
                            actual_relation_meta = constraint_meta[
                                relation_before:relation_after
                            ]
                            if [
                                int(item["rows"])
                                for item in actual_relation_meta
                            ] != relation_rows or [
                                str(item["tag"])
                                for item in actual_relation_meta
                            ] != relation_tags:
                                raise ValueError(
                                    "retained ADD relation metadata mismatch"
                                )
                            relation_row_start = sum(
                                int(item["rows"])
                                for item in constraint_meta[
                                    :relation_before
                                ]
                            )
                            relation_blocks = []
                            relation_cursor = relation_row_start
                            for item in actual_relation_meta:
                                block_rows = int(item["rows"])
                                block_end = (
                                    relation_cursor + block_rows
                                )
                                if (
                                    final_hz.Auc[
                                        relation_cursor:block_end,
                                        materialized_n_cont:,
                                    ].nnz
                                    or final_hz.Aub[
                                        relation_cursor:block_end,
                                        materialized_n_bin:,
                                    ].nnz
                                ):
                                    raise ValueError(
                                        "retained ADD relation uses future "
                                        "columns"
                                    )
                                relation_blocks.append(
                                    _ConstraintBlock(
                                        Ac=final_hz.Auc[
                                            relation_cursor:block_end,
                                            :materialized_n_cont,
                                        ].tocsr(),
                                        Ab=final_hz.Aub[
                                            relation_cursor:block_end,
                                            :materialized_n_bin,
                                        ].tocsr(),
                                        rhs=np.asarray(
                                            final_hz.ub[
                                                relation_cursor:block_end
                                            ],
                                            dtype=np.float64,
                                        ),
                                        tag=str(item["tag"]),
                                    )
                                )
                                relation_cursor = block_end
                            relation_object_valid = bool(
                                relation_cursor
                                == relation_row_start
                                + sum(relation_rows)
                                and _constraint_blocks_sha256(
                                    relation_blocks
                                )
                                == relation_digest
                                and bool(
                                    add_source_receipt.get(
                                        "materialized_relation_"
                                        "revalidated_at_export"
                                    )
                                )
                            )

                            def layer_kind(layer):
                                value = getattr(
                                    layer.kind, "value", layer.kind
                                )
                                return str(value).upper()

                            by_layer_id = {
                                int(layer.id): layer
                                for layer in net.layers
                            }
                            add_layer_id = int(
                                add_source_receipt["add_layer_id"]
                            )
                            relu_layer_id = int(
                                tail_receipt["relu_layer_id"]
                            )
                            bridge_ids = [
                                int(value)
                                for value in add_source_receipt[
                                    "bridge_layer_ids"
                                ]
                            ]
                            bridge_kinds = list(
                                add_source_receipt[
                                    "bridge_layer_kinds"
                                ]
                            )
                            parameter_receipts = list(
                                add_source_receipt[
                                    "bridge_parameter_receipts"
                                ]
                            )
                            chain = [
                                add_layer_id,
                                *bridge_ids,
                                relu_layer_id,
                            ]
                            chain_valid = bool(
                                add_layer_id in by_layer_id
                                and relu_layer_id in by_layer_id
                                and layer_kind(
                                    by_layer_id[add_layer_id]
                                )
                                == "ADD"
                                and layer_kind(
                                    by_layer_id[relu_layer_id]
                                )
                                == "RELU"
                                and all(
                                    list(
                                        map(
                                            int,
                                            net.preds.get(
                                                int(right), []
                                            ),
                                        )
                                    )
                                    == [int(left)]
                                    and list(
                                        map(
                                            int,
                                            net.succs.get(
                                                int(left), []
                                            ),
                                        )
                                    )
                                    == [int(right)]
                                    for left, right in zip(
                                        chain[:-1], chain[1:]
                                    )
                                )
                            )
                            if bridge_ids:
                                chain_valid = bool(
                                    chain_valid
                                    and len(bridge_ids) == 2
                                    and bridge_kinds
                                    == ["FLATTEN", "DENSE"]
                                    and [
                                        layer_kind(
                                            by_layer_id[layer_id]
                                        )
                                        for layer_id in bridge_ids
                                    ]
                                    == bridge_kinds
                                    and len(parameter_receipts) == 2
                                )
                            else:
                                chain_valid = bool(
                                    chain_valid
                                    and bridge_kinds == []
                                    and parameter_receipts == []
                                )
                            parameter_valid = True
                            if bridge_ids:
                                from act.back_end.hybridz_tf.tf_mlp import (
                                    sparse_dense_matrix_from_layer,
                                )

                                flatten_layer = by_layer_id[
                                    bridge_ids[0]
                                ]
                                dense_layer = by_layer_id[
                                    bridge_ids[1]
                                ]
                                dense_matrix, dense_bias = (
                                    sparse_dense_matrix_from_layer(
                                        dense_layer
                                    )
                                )
                                parameter_valid = bool(
                                    parameter_receipts[0]
                                    == {
                                        "layer_id": int(
                                            flatten_layer.id
                                        ),
                                        "kind": "FLATTEN",
                                        "input_size": int(
                                            add_source_receipt[
                                                "source_expression_size"
                                            ]
                                        ),
                                        "output_size": int(
                                            len(
                                                flatten_layer.out_vars
                                            )
                                        ),
                                    }
                                    and parameter_receipts[1]
                                    == {
                                        "layer_id": int(
                                            dense_layer.id
                                        ),
                                        "kind": "DENSE",
                                        "matrix_shape": [
                                            int(value)
                                            for value in dense_matrix.shape
                                        ],
                                        "matrix_sha256": _csr_sha256(
                                            dense_matrix
                                        ),
                                        "bias_sha256": hashlib.sha256(
                                            np.ascontiguousarray(
                                                dense_bias,
                                                dtype=np.float64,
                                            ).tobytes()
                                        ).hexdigest(),
                                    }
                                    and len(flatten_layer.out_vars)
                                    == int(
                                        add_source_receipt[
                                            "source_expression_size"
                                        ]
                                    )
                                    and len(dense_layer.out_vars)
                                    == int(
                                        add_source_receipt[
                                            "source_preactivation_size"
                                        ]
                                    )
                                )
                            add_source_actual_object_valid = bool(
                                relation_object_valid
                                and chain_valid
                                and parameter_valid
                            )
                        except (
                            KeyError,
                            TypeError,
                            ValueError,
                            OverflowError,
                        ):
                            add_source_actual_object_valid = False
                    alternative_mapping_valid = bool(
                        isinstance(alternative_rivals, list)
                        and isinstance(alternative_plane_count, int)
                        and len(alternative_rivals)
                        == alternative_plane_count
                        and all(
                            isinstance(rival, int)
                            and 0 <= rival < M
                            and int(M + offset)
                            in property_upper_row_groups[int(rival)]
                            for offset, rival in enumerate(
                                alternative_rivals
                            )
                        )
                    )
                    groups_valid = bool(
                        len(property_upper_row_groups) == M
                        and all(
                            int(rival)
                            in property_upper_row_groups[int(rival)]
                            for rival in range(M)
                        )
                        and len(flattened_group_rows)
                        == int(operator_build.hz.n_out)
                        and len(set(flattened_group_rows))
                        == int(operator_build.hz.n_out)
                        and set(flattened_group_rows)
                        == set(range(int(operator_build.hz.n_out)))
                        and bool(tail_receipt.get("proof_authority"))
                        and bool(tail_receipt.get("safe_only"))
                        and baseline_plane_count == M
                        and exported_plane_count
                        == int(operator_build.hz.n_out)
                        and alternative_plane_count
                        == int(operator_build.hz.n_out) - M
                        and alternative_mapping_valid
                        and alternative_kinds_valid
                        and property_proof_object_valid
                        and verified_query_dual_actual_object_valid
                        and pairhull_receipt_valid
                        and suffix_mapping_valid
                        and suffix_receipt_valid
                        and add_source_mapping_valid
                        and add_source_receipt_valid
                        and add_source_actual_object_valid
                        and bool(
                            add_source_receipt.get("enabled", False)
                        )
                        == add_source_requested
                        and (
                            not has_alpha_alternative
                            or bool(
                                alpha_receipt.get(
                                    "exact_candidate_audit", {}
                                ).get("proof_authority")
                            )
                        )
                        and (
                            not has_add_source_alternative
                            or (
                                add_source_requested
                                and bool(
                                    add_source_receipt.get(
                                        "proof_authority"
                                    )
                                )
                                and add_source_receipt.get("status")
                                == "applied"
                                and bool(
                                    add_source_receipt.get(
                                        "materialized_relation_retained"
                                    )
                                )
                                and not bool(
                                    add_source_receipt.get(
                                        "prunes_materialized_frame"
                                    )
                                )
                                and bool(
                                    add_source_receipt.get(
                                        "materialized_baseline_retained_per_rival"
                                    )
                                )
                                and add_source_receipt.get(
                                    "source_row_count"
                                )
                                == M
                            )
                        )
                        and tail_receipt.get("property_sha256")
                        == property_digest.hexdigest()
                        and tail_receipt.get(
                            "property_row_groups_sha256"
                        )
                        == group_digest
                    )
                    if not groups_valid:
                        raise OperatorHZBuildError(
                            "property-tail grouped upper-plane receipt "
                            "failed verifier-side validation"
                        )
                # The K4 pipeline is an explicitly enabled, candidate-only
                # transaction.  Import it only on the positive-budget path;
                # the default-off verifier does not import or invoke any of
                # its candidate modules.  Its fresh HZ, when present, is
                # independently replayed before it can reach the verdict
                # engine.  Candidate receipts never carry proof authority.
                if phase_clique_enabled:
                    _phase_clique_progress("k4_pipeline_start")
                    from act.back_end.hybridz_tf.operator_phase_clique_pipeline import (
                        consume_operator_phase_clique_pipeline_solver_handoff,
                        maybe_run_operator_phase_clique_pipeline,
                    )

                    phase_deadline = min(
                        float(hybridz_deadline),
                        time.monotonic() + phase_clique_time_limit,
                    )
                    live_assert_params = {
                        "kind": str(kind),
                        "C": C,
                        "thresholds": thresholds,
                        "M": int(M),
                        "y_true": assert_layer.params["y_true"],
                    }
                    output_lower_np = np.ascontiguousarray(
                        output_lb.detach().cpu().double().numpy(),
                        dtype=np.float64,
                    )
                    output_upper_np = np.ascontiguousarray(
                        output_ub.detach().cpu().double().numpy(),
                        dtype=np.float64,
                    )
                    pipeline_result = (
                        maybe_run_operator_phase_clique_pipeline(
                            operator_build,
                            enabled=True,
                            vnnlib_path=raw_vnnlib_path,
                            expected_vnnlib_sha256=(
                                expected_raw_vnnlib_sha256
                            ),
                            live_assert_params=live_assert_params,
                            output_lower=output_lower_np,
                            output_upper=output_upper_np,
                            residual_selector_receipt=(
                                residual_selector_receipt
                            ),
                            residual_selector_property_sha256=(
                                meta.get(
                                    "property_residual_property_sha256"
                                )
                            ),
                            deadline=phase_deadline,
                            caps=None,
                        )
                    )
                    try:
                        phase_solver_build = (
                            consume_operator_phase_clique_pipeline_solver_handoff(
                                operator_build,
                                pipeline_result,
                                deadline=phase_deadline,
                            )
                        )
                    except Exception as exc:
                        raise OperatorHZBuildError(
                            "operator phase-clique private solver handoff "
                            "failed verifier-side transaction replay"
                        ) from exc
                    operator_phase_clique_pipeline_result = (
                        pipeline_result
                    )
                    operator_phase_clique_source_build = operator_build
                    operator_phase_clique_solver_build = (
                        phase_solver_build
                    )
                    meta["operator_phase_clique_materialization"] = (
                        _operator_phase_clique_receipt_copy(
                            pipeline_result.receipt
                        )
                    )
                    meta["operator_phase_clique_solver_handoff"] = (
                        _operator_phase_clique_handoff_receipt(
                            pipeline_receipt_sha256=(
                                pipeline_result.receipt[
                                    "receipt_sha256"
                                ]
                            ),
                            semantic_digest=(
                                pipeline_result
                                .solver_handoff_capability
                                .semantic_digest
                            ),
                            materialized=pipeline_result.materialized,
                        )
                    )
                    diagnostic_receipt = pipeline_result.receipt
                    print(
                        "[verifier-phase-clique-outcome] "
                        + json.dumps(
                            {
                                "status": pipeline_result.status,
                                "materialized": bool(
                                    pipeline_result.materialized
                                ),
                                "certified_edge_count": (
                                    diagnostic_receipt.get(
                                        "certified_edge_count"
                                    )
                                ),
                                "clique_count": diagnostic_receipt.get(
                                    "clique_count"
                                ),
                                "cut_row_count": diagnostic_receipt.get(
                                    "cut_row_count"
                                ),
                                "source_upper_rows": (
                                    diagnostic_receipt.get(
                                        "source_upper_rows"
                                    )
                                ),
                                "fresh_upper_rows": (
                                    diagnostic_receipt.get(
                                        "fresh_upper_rows"
                                    )
                                ),
                                "fallback_reason": (
                                    diagnostic_receipt.get(
                                        "fallback_reason"
                                    )
                                ),
                                "failed_stage": diagnostic_receipt.get(
                                    "failed_stage"
                                ),
                                "error_type": diagnostic_receipt.get(
                                    "error_type"
                                ),
                                "timings": (
                                    _operator_phase_clique_receipt_copy(
                                        diagnostic_receipt.get("timings")
                                    )
                                ),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                        flush=True,
                    )
                    hz = phase_solver_build.hz
                    _phase_clique_progress("k4_pipeline_done")

                # ``full_col_ids`` is one stable id per seed coordinate,
                # including point dimensions which allocate no HZ column.
                if phase_clique_enabled:
                    live_full_col_ids = getattr(
                        hz, "full_col_ids", None
                    )
                    if (
                        type(live_full_col_ids) is not np.ndarray
                        or live_full_col_ids.dtype
                        != np.dtype(np.int64)
                        or not np.array_equal(
                            live_full_col_ids,
                            operator_build.input_col_ids,
                        )
                    ):
                        raise OperatorHZBuildError(
                            "operator phase-clique private input provenance "
                            "mismatch"
                        )
                else:
                    hz.full_col_ids = operator_build.input_col_ids.copy()
                operator_meta = operator_build.metadata
                if (
                    effective_query_feedback is not None
                    and not property_only_bound_replay
                ):
                    try:
                        query_transaction_metadata = (
                            _query_dual_mark_applied(
                                query_transaction_metadata,
                                operator_metadata=operator_meta,
                            )
                        )
                    except ValueError as exc:
                        raise OperatorHZBuildError(
                            "query-dual transaction/operator commit failed: "
                            f"{exc}"
                        ) from exc
                    meta["query_dual_feedback_transaction"] = (
                        query_transaction_metadata
                    )
                meta.update(
                    {
                        "engine": "operator_hz_objbound",
                        "operator_hz": operator_meta,
                        "operator_n_cont": operator_meta.get("n_cont"),
                        "operator_n_bin": operator_meta.get("n_bin"),
                        "operator_n_eq": operator_meta.get("n_eq"),
                        "operator_n_ub": int(hz.n_ub),
                        "operator_source_n_ub": operator_meta.get("n_ub"),
                        "operator_value_nnz": operator_meta.get("value_nnz"),
                        "operator_constraint_nnz": operator_meta.get(
                            "constraint_nnz"
                        ),
                        "operator_build_seconds": operator_meta.get(
                            "build_seconds"
                        ),
                        "property_upper_row_groups": [
                            [int(row) for row in group]
                            for group in property_upper_row_groups
                        ],
                    }
                )
                if property_only_bound_replay:
                    if property_only_query_upper is None:
                        return [
                            VerifyResult(
                                VerifyStatus.UNKNOWN,
                                metadata={
                                    **meta,
                                    "lane": 0,
                                    "reason": (
                                        "property_only_query_dual_"
                                        "not_applied"
                                    ),
                                    "hybridz_elapsed_s": (
                                        time.monotonic()
                                        - hybridz_started
                                    ),
                                },
                            )
                        ]
                    if property_only_query_upper.shape != (M,):
                        raise OperatorHZBuildError(
                            "property-only query result width mismatch"
                        )
                    property_only_certified = bool(
                        np.all(property_only_query_upper < 0.0)
                    )
                    meta.update(
                        {
                            "lane": 0,
                            "property_only_query_dual": True,
                            "property_only_query_dual_rows": int(M),
                            "property_only_query_dual_negative_rows": int(
                                np.count_nonzero(
                                    property_only_query_upper < 0.0
                                )
                            ),
                            "property_only_query_dual_upper_min": float(
                                np.min(property_only_query_upper)
                            ),
                            "property_only_query_dual_upper_max": float(
                                np.max(property_only_query_upper)
                            ),
                            "property_only_query_dual_all_negative": (
                                property_only_certified
                            ),
                            "hybridz_elapsed_s": (
                                time.monotonic() - hybridz_started
                            ),
                        }
                    )
                    if property_only_certified:
                        return [
                            VerifyResult(
                                VerifyStatus.CERTIFIED,
                                metadata=meta,
                            )
                        ]
                    meta["reason"] = (
                        "property_only_query_dual_incomplete"
                    )
                    return [
                        VerifyResult(
                            VerifyStatus.UNKNOWN,
                            metadata=meta,
                        )
                    ]
            except OperatorHZBuildTimeout as exc:
                if (
                    meta.get("query_dual_feedback_transaction", {}).get(
                        "status"
                    )
                    == "pipeline_verified_pending_operator"
                ):
                    failed_query_metadata = copy.deepcopy(
                        dict(meta["query_dual_feedback_transaction"])
                    )
                    failed_query_metadata.update(
                        {
                            "status": "operator_timeout_no_application",
                            "proof_authority": False,
                            "operator_error_type": type(exc).__name__,
                            "operator_error": str(exc)[:1000],
                        }
                    )
                    meta["query_dual_feedback_transaction"] = (
                        failed_query_metadata
                    )
                return [
                    VerifyResult(
                        VerifyStatus.TIMEOUT,
                        metadata={
                            **meta,
                            "lane": 0,
                            "reason": "hybridz_total_deadline",
                            "timeout_stage": "operator_build",
                            "hz_timeout_s": hybridz_total_timeout,
                            "hybridz_elapsed_s": (
                                time.monotonic() - hybridz_started
                            ),
                            "operator_error": str(exc)[:1000],
                        },
                    )
                ]
            except Exception as exc:
                if (
                    meta.get("query_dual_feedback_transaction", {}).get(
                        "status"
                    )
                    == "pipeline_verified_pending_operator"
                ):
                    failed_query_metadata = copy.deepcopy(
                        dict(meta["query_dual_feedback_transaction"])
                    )
                    failed_query_metadata.update(
                        {
                            "status": "operator_error_no_application",
                            "proof_authority": False,
                            "operator_error_type": type(exc).__name__,
                            "operator_error": str(exc)[:1000],
                        }
                    )
                    meta["query_dual_feedback_transaction"] = (
                        failed_query_metadata
                    )
                return [
                    VerifyResult(
                        VerifyStatus.UNKNOWN,
                        metadata={
                            **meta,
                            "lane": 0,
                            "reason": "hybridz_operator_build_failed",
                            "operator_error_type": type(exc).__name__,
                            "operator_error": str(exc)[:1000],
                        },
                    )
                ]
        else:
            if sparse_hz_engine and hasattr(active_tf, "get_sparse_hz"):
                hz = active_tf.get_sparse_hz(output_layer_id)
                if hz is not None:
                    meta.update({
                        "engine": "sparse_hz_objbound",
                        "sparse_source": "propagated",
                        "sparse_value_nnz": getattr(hz, "value_nnz", None),
                        "sparse_constraint_nnz": getattr(hz, "constraint_nnz", None),
                    })
                if hz is None and hasattr(active_tf, "sparse_drop_reason"):
                    sparse_drop_reason = active_tf.sparse_drop_reason(output_layer_id)
            dense_hz = active_tf.get_hz(output_layer_id) if hasattr(active_tf, "get_hz") else None
            if hz is None:
                hz = dense_hz
            if hz is None:
                reason = "hybridz_representation_drop"
                if sparse_hz_engine and sparse_drop_reason:
                    reason = f"hybridz_sparse_drop:{sparse_drop_reason}"
                return [
                    VerifyResult(
                        VerifyStatus.UNKNOWN,
                        metadata={**meta, "lane": 0, "reason": reason},
                    )
                ]
            if sparse_hz_engine and hz is dense_hz:
                from act.back_end.solver.solver_hz import SparseHZono
                hz = SparseHZono.from_dense_hz(hz)
                meta.update({
                    "engine": "sparse_hz_objbound",
                    "sparse_source": "dense_conversion",
                    "sparse_drop_reason": sparse_drop_reason,
                })
        from act.back_end.solver.solver_hz import (
            hz_enumerate_sparse_binary_phase_cover,
            hz_objbound_decide,
            hz_objbound_safe_capability_receipt,
        )

        hz_timeout = max(
            0.0,
            float(hybridz_deadline) - time.monotonic(),
        )
        if hz_timeout <= 0.0:
            return [
                VerifyResult(
                    VerifyStatus.TIMEOUT,
                    metadata={
                        **meta,
                        "lane": 0,
                        "reason": "hybridz_total_deadline",
                        "timeout_stage": "before_solver",
                        "hz_timeout_s": hybridz_total_timeout,
                        "hz_solver_budget_s": 0.0,
                        "hybridz_elapsed_s": (
                            time.monotonic() - hybridz_started
                        ),
                    },
                )
            ]
        solver_C = (
            np.eye(hz.n_out, dtype=np.float64)
            if property_upper_output
            else C_np
        )
        solver_thresholds = (
            np.zeros(hz.n_out, dtype=np.float64)
            if property_upper_output
            else thresholds_np
        )
        solver_kwargs = dict(
            is_unsafe_linear=(
                False if property_upper_output else is_unsafe_linear
            ),
            lp_prefilter_fraction=getattr(
                hz_cfg, "lp_prefilter_fraction", None
            ),
            lp_prefilter_max_seconds=getattr(
                hz_cfg, "lp_prefilter_max_seconds", None
            ),
            gpu_dual_steps=int(
                getattr(hz_cfg, "gpu_dual_steps", 0)
            ),
            gpu_dual_time_limit=float(
                getattr(hz_cfg, "gpu_dual_time_limit", 0.0)
            ),
            gpu_dual_row_topk=int(
                getattr(hz_cfg, "gpu_dual_row_topk", 0)
            ),
            gpu_dual_learning_rate=float(
                getattr(hz_cfg, "gpu_dual_learning_rate", 0.08)
            ),
            safe_row_groups=(
                property_upper_row_groups
                if property_upper_output else None
            ),
            expected_safe_group_count=(
                M if property_upper_output else None
            ),
            safe_group_mixture_grid_bits=int(
                getattr(
                    hz_cfg,
                    "property_tail_mixture_grid_bits",
                    0,
                )
                if property_upper_output
                else 0
            ),
        )
        phase_cover_enabled = bool(
            phase_split_mode
            and property_upper_output
            and int(getattr(hz, "n_bin", 0)) > 0
        )
        micro_rlt_parent_only_diagnostic = bool(
            getattr(
                hz_cfg,
                "property_micro_rlt_parent_only_diagnostic",
                False,
            )
        )

        def return_micro_rlt_parent_only_diagnostic(
            *,
            parent_receipt: Mapping[str, Any],
            operator_receipt: Any,
        ) -> List[VerifyResult]:
            """Stop after parent observation with no verdict authority."""

            phase_depth = int(getattr(hz, "n_bin", 0))
            stop_reason = str(
                parent_receipt.get(
                    "status",
                    "parent_only_diagnostic_stop",
                )
            )
            shared_deadline_expired = bool(
                time.monotonic() >= float(hybridz_deadline)
                or stop_reason.startswith("shared_deadline_")
            )
            diagnostic_receipt: Dict[str, Any] = {
                "schema": (
                    "verifier_property_micro_rlt_"
                    "parent_only_diagnostic_v1"
                ),
                "enabled": True,
                "diagnostic_only": True,
                "proof_authority": False,
                "verdict_forced_unknown": True,
                "operator_receipt_status": (
                    operator_receipt.get("status")
                    if isinstance(operator_receipt, Mapping)
                    else None
                ),
                "operator_receipt_sha256": (
                    operator_receipt.get("receipt_sha256")
                    if isinstance(operator_receipt, Mapping)
                    else None
                ),
                "operator_live_validation_passed": (
                    operator_receipt.get(
                        "live_result_validation_passed"
                    )
                    if isinstance(operator_receipt, Mapping)
                    else None
                ),
                "parent_prefilter_status": stop_reason,
                "parent_call_count": int(
                    parent_receipt.get("parent_call_count", 0)
                ),
                "parent_solver_verdict": parent_receipt.get(
                    "solver_verdict"
                ),
                "parent_safe_contract_observed": bool(
                    parent_receipt.get("safe_contract_valid") is True
                ),
                "shared_deadline_expired": shared_deadline_expired,
                "phase_cover_attempted": False,
                "phase_children_created": 0,
                "baseline_solver_attempted": False,
                "stop_reason": stop_reason,
            }
            diagnostic_receipt["receipt_sha256"] = (
                _canonical_receipt_sha256(diagnostic_receipt)
            )
            meta.update(
                {
                    "base_feasibility_status": (
                        "SHARED_DEADLINE_IN_PARENT_ONLY_DIAGNOSTIC"
                        if shared_deadline_expired
                        else "PARENT_ONLY_DIAGNOSTIC"
                    ),
                    "all_rivals_covered": False,
                    "property_micro_rlt_parent_only_diagnostic": (
                        diagnostic_receipt
                    ),
                    "property_phase_split": {
                        "schema": "verifier_property_phase_split_v1",
                        "status": stop_reason,
                        "proof_authority": False,
                        "diagnostic_only": True,
                        "binary_depth": phase_depth,
                        "expected_child_count": (
                            int(1 << phase_depth)
                            if 1 <= phase_depth <= 2
                            else 0
                        ),
                        "actual_child_count": 0,
                        "all_assignments_enumerated": False,
                        "phase_enumeration_skipped": True,
                        "children_run_in_parallel": False,
                        "parallel_workers": 0,
                        "children": [],
                    },
                    "lane": 0,
                    "reason": (
                        "property_micro_rlt_parent_only_diagnostic"
                    ),
                    "timeout_stage": (
                        "property_micro_rlt_parent_only_diagnostic"
                        if shared_deadline_expired
                        else None
                    ),
                    "hz_verdict": "UNKNOWN",
                    "hz_timeout_s": hybridz_total_timeout,
                    "hz_solver_budget_s": float(
                        parent_receipt.get(
                            "actual_budget_seconds",
                            0.0,
                        )
                    ),
                    "hybridz_elapsed_s": (
                        time.monotonic() - hybridz_started
                    ),
                    "hz_has_witness": False,
                }
            )
            return [
                VerifyResult(
                    (
                        VerifyStatus.TIMEOUT
                        if shared_deadline_expired
                        else VerifyStatus.UNKNOWN
                    ),
                    metadata=meta,
                )
            ]

        if phase_cover_enabled:
            parent_gpu_candidates_enabled = bool(
                micro_rlt_parent_only_diagnostic
                and int(solver_kwargs.get("gpu_dual_steps", 0)) > 0
                and float(
                    solver_kwargs.get("gpu_dual_time_limit", 0.0)
                )
                > 0.0
            )
            micro_rlt_parent_seconds = float(
                getattr(
                    hz_cfg,
                    "property_micro_rlt_parent_prefilter_seconds",
                    0.0,
                )
            )
            micro_rlt_operator_receipt = operator_meta.get(
                "property_micro_rlt", {}
            )
            micro_rlt_receipt_valid = bool(
                isinstance(micro_rlt_operator_receipt, Mapping)
                and micro_rlt_operator_receipt.get("status") == "applied"
                and micro_rlt_operator_receipt.get("proof_authority") is True
                and micro_rlt_operator_receipt.get(
                    "live_result_validation_passed"
                )
                is True
                and micro_rlt_operator_receipt.get("scope")
                == "parent_pre_phase_fix"
                and _audit_live_operator_property_micro_rlt(
                    hz,
                    micro_rlt_operator_receipt,
                )
            )
            parent_prefilter_receipt: Dict[str, Any] = {
                "schema": "verifier_property_micro_rlt_parent_prefilter_v1",
                "enabled": bool(micro_rlt_parent_seconds > 0.0),
                "status": (
                    "disabled"
                    if micro_rlt_parent_seconds <= 0.0
                    else (
                        "operator_receipt_ineligible_diagnostic_stop"
                        if micro_rlt_parent_only_diagnostic
                        else "operator_receipt_ineligible"
                    )
                    if not micro_rlt_receipt_valid
                    else "pending"
                ),
                "proof_authority": False,
                "scope": "parent_pre_phase_fix",
                "configured_seconds": float(micro_rlt_parent_seconds),
                "actual_budget_seconds": 0.0,
                "elapsed_seconds": 0.0,
                "parent_call_count": 0,
                "solver_verdict": None,
                "safe_only": True,
                "base_witness_precheck": False,
                "lp_prefilter_fraction": 1.0,
                "gpu_candidates_enabled": bool(
                    parent_gpu_candidates_enabled
                ),
                "safe_group_mixture_grid_bits": 0,
                "identity_objective_rows": int(solver_C.shape[0]),
                "zero_thresholds": bool(
                    np.array_equal(
                        solver_thresholds,
                        np.zeros_like(solver_thresholds),
                    )
                ),
                "safe_row_group_count": int(
                    len(property_upper_row_groups)
                ),
                "operator_receipt_status": (
                    micro_rlt_operator_receipt.get("status")
                    if isinstance(micro_rlt_operator_receipt, Mapping)
                    else None
                ),
                "operator_receipt_live_validation": (
                    micro_rlt_operator_receipt.get(
                        "live_result_validation_passed"
                    )
                    if isinstance(micro_rlt_operator_receipt, Mapping)
                    else None
                ),
                "stats": {},
                "error": None,
            }
            meta["property_micro_rlt_parent_prefilter"] = (
                parent_prefilter_receipt
            )
            if (
                micro_rlt_parent_seconds > 0.0
                and micro_rlt_receipt_valid
            ):
                parent_budget = min(
                    micro_rlt_parent_seconds,
                    max(
                        0.0,
                        float(hybridz_deadline) - time.monotonic(),
                    ),
                )
                parent_prefilter_receipt["actual_budget_seconds"] = float(
                    parent_budget
                )
                if parent_budget <= 0.0:
                    parent_prefilter_receipt["status"] = (
                        "shared_deadline_before_parent"
                    )
                else:
                    parent_started = time.monotonic()
                    parent_solver_kwargs = dict(solver_kwargs)
                    parent_solver_kwargs.update(
                        {
                            "is_unsafe_linear": False,
                            "lp_prefilter_fraction": 1.0,
                            "lp_prefilter_max_seconds": float(parent_budget),
                            "safe_group_mixture_grid_bits": 0,
                        }
                    )
                    if not parent_gpu_candidates_enabled:
                        parent_solver_kwargs.update(
                            {
                                "gpu_dual_steps": 0,
                                "gpu_dual_time_limit": 0.0,
                                "gpu_dual_row_topk": 0,
                            }
                        )
                    parent_prefilter_receipt["parent_call_count"] = 1
                    _parent_witness = None
                    try:
                        parent_verdict, _parent_witness = (
                            hz_objbound_decide(
                                hz,
                                solver_C,
                                solver_thresholds,
                                time_limit=float(parent_budget),
                                base_witness_precheck=False,
                                **parent_solver_kwargs,
                            )
                        )
                        parent_prefilter_receipt["solver_verdict"] = str(
                            parent_verdict
                        )
                    except Exception as exc:
                        parent_verdict = "UNKNOWN"
                        parent_prefilter_receipt.update(
                            {
                                "status": (
                                    "parent_error_diagnostic_stop"
                                    if micro_rlt_parent_only_diagnostic
                                    else "error_fallback_phase_cover"
                                ),
                                "error": (
                                    f"{type(exc).__name__}:"
                                    f"{str(exc)[:500]}"
                                ),
                            }
                        )
                    parent_elapsed = time.monotonic() - parent_started
                    parent_prefilter_receipt["elapsed_seconds"] = float(
                        parent_elapsed
                    )
                    parent_stats = getattr(
                        hz, "_solver_objbound_stats", {}
                    )
                    if not isinstance(parent_stats, dict):
                        parent_stats = {}
                    parent_stat_keys = (
                        "base_feasibility_status",
                        "base_feasibility_reason",
                        "cube_min_upper",
                        "cube_max_upper",
                        "cube_pruned_rows",
                        "cube_survivor_rows",
                        "cube_elapsed_s",
                        "parent_stage_timing_schema",
                        "parent_stage_timings_diagnostic_only",
                        "parent_stage_timings_proof_authority",
                        "parent_last_stage",
                        "parent_exit_reason",
                        "parent_elapsed_s",
                        "parent_cube_complete_elapsed_s",
                        "parent_base_matrix_materialization_status",
                        "parent_base_matrix_materialization_elapsed_s",
                        "parent_base_matrix_rows",
                        "parent_base_matrix_columns",
                        "parent_base_matrix_nnz",
                        "parent_base_matrix_error_type",
                        "parent_persistent_lp_status",
                        "parent_persistent_lp_elapsed_s",
                        "parent_persistent_lp_input_rows",
                        "parent_persistent_lp_output_rows",
                        "parent_persistent_lp_budget_s",
                        "parent_persistent_lp_error_type",
                        "lp_status",
                        "lp_input_rows",
                        "lp_safe_certificate_eligible",
                        "lp_binary_relaxation_certificate_eligible",
                        "lp_candidate_witness_eligible",
                        "lp_binary_factor_count",
                        "lp_certificate_factor_columns",
                        "lp_certified_rows",
                        "lp_uncertified_rows",
                        "lp_survivor_rows",
                        "lp_cert_max_upper",
                        "lp_cert_min_gap_to_cutoff",
                        "lp_cert_center_transform_guard_max",
                        "lp_elapsed_s",
                        "lp_completed_rows",
                        "lp_full_resolve_rows",
                        "lp_optimal_certificate_candidates",
                        "lp_nonoptimal_certificate_candidates",
                        "lp_certificate_attempted_rows",
                        "lp_zero_dual_certificate_skips",
                        "lp_certificate_failures",
                        "lp_last_run_status",
                        "lp_last_model_status",
                        "lp_row_time_slice_min_s",
                        "lp_row_time_slice_max_s",
                        "lp_model_reused",
                        "lp_persistent_model_builds",
                        "lp_basis_warmup_attempted",
                        "lp_basis_warmup_seconds",
                        "lp_basis_warmup_run_status",
                        "lp_basis_warmup_model_status",
                        "lp_deadline_exhausted_stage",
                        "lp_matrix_input_nnz",
                        "lp_matrix_loaded_nnz",
                        "lp_matrix_dropped_nnz",
                        "lp_matrix_load_status",
                        "lp_input_validation_s",
                        "lp_binary_frame_s",
                        "lp_candidate_csr_s",
                        "lp_highs_setup_s",
                        "lp_highs_add_columns_s",
                        "lp_highs_add_rows_s",
                        "lp_model_build_elapsed_s",
                        "lp_proof_authority",
                        "gpu_dual_status",
                        "gpu_dual_elapsed_s",
                        "gpu_dual_steps_requested",
                        "gpu_dual_steps_completed",
                        "gpu_dual_deadline_reached",
                        "gpu_dual_deadline_stage",
                        "gpu_dual_device_requested",
                        "gpu_dual_device",
                        "gpu_dual_packet_core_cpu_fallback",
                        "gpu_dual_total_input_rows",
                        "gpu_dual_objective_scope",
                        "gpu_dual_objective_rows_scheduled",
                        "gpu_dual_objective_rows_deferred",
                        "gpu_dual_first_scheduled_objective_row",
                        "gpu_dual_objective_focus_rival_id",
                        "gpu_dual_objective_focus_plane_kind",
                        "gpu_dual_objective_focus_mapping_valid",
                        "gpu_dual_candidate_constraint_scope",
                        "gpu_dual_candidate_constraint_rows_total",
                        "gpu_dual_candidate_constraint_rows_selected",
                        "gpu_dual_candidate_constraint_rows_deferred",
                        "gpu_dual_packet_generated_rows_selected",
                        "gpu_dual_packet_source_rows_selected",
                        "gpu_dual_packet_bridge_rows_selected",
                        "gpu_dual_bridge_base_updates",
                        "gpu_dual_bridge_packet_updates",
                        "gpu_dual_bridge_base_nnz",
                        "gpu_dual_bridge_packet_nnz",
                        "gpu_dual_bridge_base_support_improvement",
                        "gpu_dual_bridge_combined_support_improvement",
                        "gpu_dual_pc_cbde_status",
                        "gpu_dual_pc_cbde_elapsed_s",
                        "gpu_dual_pc_cbde_budget_s",
                        "gpu_dual_pc_cbde_deadline_reached",
                        "gpu_dual_pc_cbde_cone_rows",
                        "gpu_dual_pc_cbde_cone_row_count",
                        "gpu_dual_pc_cbde_local_row_count",
                        "gpu_dual_pc_cbde_bridge_row_count",
                        "gpu_dual_pc_cbde_generated_row_count",
                        "gpu_dual_pc_cbde_generated_warm_nonzero_count",
                        "gpu_dual_pc_cbde_generated_warm_truncated_count",
                        "gpu_dual_pc_cbde_source_row_count",
                        "gpu_dual_pc_cbde_ignored_source_row_count",
                        "gpu_dual_pc_cbde_full_nnz",
                        "gpu_dual_pc_cbde_updates",
                        "gpu_dual_pc_cbde_checked_upper_full",
                        "gpu_dual_pc_cbde_checked_upper_without_generated",
                        "gpu_dual_pc_cbde_checked_upper_without_bridge",
                        "gpu_dual_pc_cbde_checked_upper_without_both",
                        "gpu_dual_pc_cbde_all_ablations_verified",
                        "gpu_dual_pc_cbde_strict_family_ablation",
                        "gpu_dual_pc_cbde_strict_family_ablation_tol",
                        "gpu_dual_pc_cbde_full_vs_without_generated_gap",
                        "gpu_dual_pc_cbde_full_vs_without_bridge_gap",
                        "gpu_dual_pc_cbde_old_support",
                        "gpu_dual_pc_cbde_full_support",
                        "gpu_dual_pc_cbde_support_improvement",
                        "gpu_dual_pc_cbde_support_improvement_tol",
                        "gpu_dual_pc_cbde_replaced_old_candidate",
                        "gpu_dual_pc_cbde_error_type",
                        "gpu_dual_pc_cbde_error_message",
                        "gpu_dual_pc_cbde_proof_authority",
                        "gpu_dual_initial_support_min",
                        "gpu_dual_initial_support_max",
                        "gpu_dual_candidate_support_min",
                        "gpu_dual_candidate_support_max",
                        "gpu_dual_support_improved_rows",
                        "gpu_dual_support_best_improvement",
                        "gpu_dual_certificate_attempted_rows",
                        "gpu_dual_certificate_errors",
                        "gpu_dual_candidate_dual_nnz_total",
                        "gpu_dual_candidate_dual_nnz_max",
                        "gpu_dual_support_attribution_elapsed_s",
                        "gpu_dual_independent_certificate_elapsed_s",
                        "gpu_dual_checked_upper_min",
                        "gpu_dual_checked_upper_max",
                        "gpu_dual_cert_upper_max",
                        "gpu_dual_certified_rows",
                        "gpu_dual_uncertified_rows",
                        "gpu_dual_checked_dual_nnz_total",
                        "gpu_dual_checked_dual_nnz_max",
                        "gpu_dual_checked_generated_nnz_total",
                        "gpu_dual_checked_generated_nnz_max",
                        "gpu_dual_checked_source_nnz_total",
                        "gpu_dual_checked_source_nnz_max",
                        "gpu_dual_checked_bridge_nnz_total",
                        "gpu_dual_checked_bridge_nnz_max",
                        "gpu_dual_checked_other_nnz_total",
                        "gpu_dual_checked_other_nnz_max",
                        "gpu_dual_wavefront_updates",
                        "gpu_dual_wavefront_support_improved_rows",
                        "gpu_dual_wavefront_best_improvement",
                        "gpu_dual_wavefront_elapsed_s",
                        "gpu_dual_wavefront_selected_constraint_count",
                        "gpu_dual_constraint_generation_attempted_rows",
                        "gpu_dual_constraint_generation_improved_rows",
                        "gpu_dual_constraint_generation_best_improvement",
                        "gpu_dual_constraint_generation_elapsed_s",
                        "gpu_dual_constraint_generation_status",
                        "gpu_dual_cert_center_transform_guard_max",
                        "gpu_dual_binary_factor_count",
                        "gpu_dual_binary_relaxation_enabled",
                        "gpu_dual_candidate_witness_eligible",
                        "gpu_dual_coverage_ok",
                        "gpu_dual_errors",
                        "gpu_dual_error_type",
                        "gpu_dual_error_stage",
                        "gpu_dual_error_message",
                        "safe_row_groups_resolved",
                        "safe_row_groups_unresolved",
                        "all_rivals_covered",
                    )
                    bounded_parent_stats = {
                        key: parent_stats.get(key)
                        for key in parent_stat_keys
                        if key in parent_stats
                    }
                    parent_prefilter_receipt["stats"] = (
                        bounded_parent_stats
                    )
                    shared_deadline_respected = bool(
                        time.monotonic() <= float(hybridz_deadline)
                    )
                    parent_prefilter_receipt[
                        "shared_deadline_respected"
                    ] = shared_deadline_respected
                    parent_safe_capability = (
                        hz_objbound_safe_capability_receipt(
                            hz,
                            solver_C,
                            solver_thresholds,
                            is_unsafe_linear=False,
                            tol=float(
                                parent_solver_kwargs.get(
                                    "tol", 1e-9
                                )
                            ),
                            require_base_feasible=True,
                            base_witness_precheck=False,
                            safe_row_groups=(
                                property_upper_row_groups
                            ),
                            expected_safe_group_count=M,
                            require_binary_relaxation_lp=False,
                        )
                    )
                    binary_lp_safe_capability = (
                        hz_objbound_safe_capability_receipt(
                            hz,
                            solver_C,
                            solver_thresholds,
                            is_unsafe_linear=False,
                            tol=float(
                                parent_solver_kwargs.get(
                                    "tol", 1e-9
                                )
                            ),
                            require_base_feasible=True,
                            base_witness_precheck=False,
                            safe_row_groups=(
                                property_upper_row_groups
                            ),
                            expected_safe_group_count=M,
                            require_binary_relaxation_lp=True,
                        )
                    )
                    parent_prefilter_receipt["safe_capability"] = (
                        parent_safe_capability
                    )
                    parent_prefilter_receipt[
                        "binary_relaxation_attributed"
                    ] = bool(binary_lp_safe_capability is not None)
                    parent_safe_contract = bool(
                        parent_verdict == "SAFE"
                        and _parent_witness is None
                        and parent_safe_capability is not None
                    )
                    parent_prefilter_receipt[
                        "safe_contract_valid"
                    ] = parent_safe_contract
                    if (
                        parent_safe_contract
                        and shared_deadline_respected
                    ):
                        parent_proof_stage = str(
                            parent_safe_capability["proof_stage"]
                        )
                        parent_safe_reason = (
                            "parent_binary_relaxation_safe"
                            if binary_lp_safe_capability is not None
                            else f"parent_{parent_proof_stage}_safe"
                        )
                        if micro_rlt_parent_only_diagnostic:
                            parent_prefilter_receipt.update(
                                {
                                    "status": (
                                        "parent_safe_observed_"
                                        "diagnostic_stop"
                                    ),
                                    "proof_authority": False,
                                    "proof_stage": parent_proof_stage,
                                    "phase_enumeration_skipped": True,
                                    "phase_children_created": 0,
                                }
                            )
                        else:
                            parent_prefilter_receipt.update(
                                {
                                    "status": parent_safe_reason,
                                    "proof_authority": True,
                                    "proof_stage": parent_proof_stage,
                                    "phase_enumeration_skipped": True,
                                    "phase_children_created": 0,
                                }
                            )
                            phase_depth = int(getattr(hz, "n_bin", 0))
                            meta.update(
                                {
                                    "base_feasibility_status": (
                                        bounded_parent_stats.get(
                                            "base_feasibility_status"
                                        )
                                    ),
                                    "all_rivals_covered": bool(
                                        bounded_parent_stats.get(
                                            "all_rivals_covered", False
                                        )
                                    ),
                                    "certification_reason": (
                                        parent_safe_reason
                                    ),
                                    "reason": parent_safe_reason,
                                    "property_phase_split": {
                                        "schema": (
                                            "verifier_property_phase_split_v1"
                                        ),
                                        "status": parent_safe_reason,
                                        "proof_authority": True,
                                        "proof_rule": (
                                            "the_sound_parent_HZ_or_its_"
                                            "checked_continuous_relaxation_"
                                            "contains_all_binary_phase_slices;"
                                            "private_live_SAFE_capability_"
                                            "certifies_every_property_group"
                                        ),
                                        "parent_proof_stage": (
                                            parent_proof_stage
                                        ),
                                        "binary_relaxation_attributed": bool(
                                            binary_lp_safe_capability
                                            is not None
                                        ),
                                        "binary_depth": int(phase_depth),
                                        "expected_child_count": int(
                                            1 << phase_depth
                                        ),
                                        "actual_child_count": 0,
                                        "all_assignments_enumerated": False,
                                        "phase_enumeration_skipped": True,
                                        "children_run_in_parallel": False,
                                        "parallel_workers": 0,
                                        "children": [],
                                    },
                                    "lane": 0,
                                    "hz_verdict": "SAFE",
                                    "hz_timeout_s": hybridz_total_timeout,
                                    "hz_solver_budget_s": float(
                                        parent_budget
                                    ),
                                    "hybridz_elapsed_s": (
                                        time.monotonic()
                                        - hybridz_started
                                    ),
                                    "hz_has_witness": False,
                                }
                            )
                            return [
                                VerifyResult(
                                    VerifyStatus.CERTIFIED,
                                    metadata=meta,
                                )
                            ]
                    if (
                        parent_verdict == "SAFE"
                        and not parent_safe_contract
                    ):
                        parent_prefilter_receipt["status"] = (
                            "contract_mismatch_diagnostic_stop"
                            if micro_rlt_parent_only_diagnostic
                            else "contract_mismatch_fallback_phase_cover"
                        )
                    if parent_prefilter_receipt["status"] == "pending":
                        parent_prefilter_receipt["status"] = (
                            "parent_unknown_diagnostic_stop"
                            if (
                                micro_rlt_parent_only_diagnostic
                                and shared_deadline_respected
                            )
                            else "unknown_fallback_phase_cover"
                            if shared_deadline_respected
                            else "shared_deadline_after_parent"
                        )
            if micro_rlt_parent_only_diagnostic:
                return return_micro_rlt_parent_only_diagnostic(
                    parent_receipt=parent_prefilter_receipt,
                    operator_receipt=micro_rlt_operator_receipt,
                )
            phase_started = time.monotonic()
            phase_depth = int(getattr(hz, "n_bin", 0))
            if not 1 <= phase_depth <= 2:
                verdict, witness = "UNKNOWN", None
                phase_children_receipt = []
                meta["property_phase_split"] = {
                    "schema": "verifier_property_phase_split_v1",
                    "status": "invalid_binary_depth",
                    "proof_authority": False,
                    "binary_depth": phase_depth,
                    "max_depth": 2,
                }
            else:
                if time.monotonic() >= float(hybridz_deadline):
                    meta.update(
                        {
                            "base_feasibility_status": (
                                "SHARED_DEADLINE_BEFORE_PHASE_COVER"
                            ),
                            "all_rivals_covered": False,
                            "property_phase_split": {
                                "schema": (
                                    "verifier_property_phase_split_v1"
                                ),
                                "status": (
                                    "shared_deadline_before_phase_cover"
                                ),
                                "proof_authority": False,
                                "binary_depth": int(phase_depth),
                                "expected_child_count": int(
                                    1 << phase_depth
                                ),
                                "actual_child_count": 0,
                                "all_assignments_enumerated": False,
                                "children_run_in_parallel": False,
                                "parallel_workers": 0,
                                "children": [],
                                "elapsed_seconds": float(
                                    time.monotonic() - phase_started
                                ),
                            },
                            "lane": 0,
                            "reason": (
                                "shared_deadline_before_phase_cover"
                            ),
                            "timeout_stage": (
                                "before_binary_phase_enumeration"
                            ),
                            "hz_verdict": "UNKNOWN",
                            "hz_timeout_s": hybridz_total_timeout,
                            "hz_solver_budget_s": 0.0,
                            "hybridz_elapsed_s": (
                                time.monotonic() - hybridz_started
                            ),
                            "hz_has_witness": False,
                        }
                    )
                    return [
                        VerifyResult(
                            VerifyStatus.TIMEOUT,
                            metadata=meta,
                        )
                    ]
                phase_cover_audit: Dict[str, Any]
                cover: Tuple[Any, ...] = ()
                phase_cover_stage = "enumeration"
                phase_cover_segment_started = time.monotonic()
                phase_enumeration_seconds = 0.0
                phase_audit_seconds = 0.0
                try:
                    cover = hz_enumerate_sparse_binary_phase_cover(
                        hz,
                        max_children=1 << phase_depth,
                        deadline=float(hybridz_deadline),
                    )
                    phase_enumeration_seconds = float(
                        time.monotonic() - phase_cover_segment_started
                    )
                    phase_cover_stage = "live_audit"
                    phase_cover_segment_started = time.monotonic()
                    phase_cover_audit = (
                        _audit_sparse_binary_phase_cover(
                            hz,
                            cover,
                            phase_depth=phase_depth,
                            deadline=float(hybridz_deadline),
                        )
                    )
                    phase_audit_seconds = float(
                        time.monotonic() - phase_cover_segment_started
                    )
                    phase_cover_audit.update(
                        {
                            "enumeration_seconds": (
                                phase_enumeration_seconds
                            ),
                            "live_audit_seconds": phase_audit_seconds,
                        }
                    )
                except Exception as exc:
                    phase_cover_timed_out = isinstance(exc, TimeoutError)
                    if phase_cover_stage == "enumeration":
                        phase_enumeration_seconds = float(
                            time.monotonic()
                            - phase_cover_segment_started
                        )
                    else:
                        phase_audit_seconds = float(
                            time.monotonic()
                            - phase_cover_segment_started
                        )
                    phase_cover_audit = {
                        "schema": (
                            "verifier_sparse_binary_phase_cover_audit_v1"
                        ),
                        "proof_authority": False,
                        "expected_child_count": int(1 << phase_depth),
                        "actual_child_count": int(len(cover)),
                        "all_assignments_enumerated": False,
                        "all_children_assignment_bound": False,
                        "all_child_capabilities_valid": False,
                        "timed_out": phase_cover_timed_out,
                        "timeout_stage": (
                            phase_cover_stage
                            if phase_cover_timed_out
                            else None
                        ),
                        "enumeration_seconds": (
                            phase_enumeration_seconds
                        ),
                        "live_audit_seconds": phase_audit_seconds,
                        "error": (
                            f"{type(exc).__name__}:{str(exc)[:500]}"
                        ),
                    }
                    meta.update(
                        {
                            "base_feasibility_status": (
                                "SHARED_DEADLINE_DURING_PHASE_COVER"
                                if phase_cover_timed_out
                                else "INVALID_EXACT_PHASE_COVER"
                            ),
                            "all_rivals_covered": False,
                            "property_phase_split": {
                                "schema": (
                                    "verifier_property_phase_split_v1"
                                ),
                                "status": (
                                    "shared_deadline_during_phase_cover"
                                    if phase_cover_timed_out
                                    else "invalid_exact_phase_cover"
                                ),
                                "proof_authority": False,
                                "binary_depth": int(phase_depth),
                                "expected_child_count": int(
                                    1 << phase_depth
                                ),
                                "actual_child_count": int(
                                    phase_cover_audit[
                                        "actual_child_count"
                                    ]
                                ),
                                "all_assignments_enumerated": False,
                                "children_run_in_parallel": False,
                                "parallel_workers": 0,
                                "phase_cover_audit": (
                                    phase_cover_audit
                                ),
                                "children": [],
                                "elapsed_seconds": float(
                                    time.monotonic() - phase_started
                                ),
                            },
                            "lane": 0,
                            "hz_verdict": "UNKNOWN",
                            "hz_timeout_s": hybridz_total_timeout,
                            "hz_solver_budget_s": 0.0,
                            "hybridz_elapsed_s": (
                                time.monotonic() - hybridz_started
                            ),
                            "hz_has_witness": False,
                            "reason": (
                                "shared_deadline_during_phase_cover"
                                if phase_cover_timed_out
                                else "invalid_exact_phase_cover"
                            ),
                            "timeout_stage": (
                                f"binary_phase_{phase_cover_stage}"
                                if phase_cover_timed_out
                                else None
                            ),
                        }
                    )
                    return [
                        VerifyResult(
                            (
                                VerifyStatus.TIMEOUT
                                if phase_cover_timed_out
                                else VerifyStatus.UNKNOWN
                            ),
                            metadata=meta,
                        )
                    ]
                phase_property_row_groups = tuple(
                    tuple(int(row) for row in group)
                    for group in property_upper_row_groups
                )
                phase_output_rows = int(hz.n_out)
                conditional_maps = []
                for _assignment, child in cover:
                    applied = getattr(
                        child,
                        "_solver_conditional_property_rows_applied",
                        None,
                    )
                    # The cover audit immediately above already performed the
                    # complete live parent/assignment/child reconstruction.
                    # No user or solver callback runs between that audit and
                    # this metadata read, so repeating the same multi-million
                    # nnz reconstruction here adds cost but no proof boundary.
                    conditional_live = bool(
                        phase_cover_audit.get(
                            "all_children_live_projection_valid"
                        )
                        is True
                    )
                    if (
                        conditional_live
                        and isinstance(applied, Mapping)
                        and applied.get("schema")
                        == (
                            "hz_exact_phase_conditional_property_rows_"
                            "child_v2"
                        )
                        and applied.get("proof_authority") is True
                        and isinstance(
                            applied.get("live_content_sha256"), str
                        )
                        and len(applied["live_content_sha256"]) == 64
                    ):
                        conditional_maps.append(
                            {
                                int(rival): tuple(
                                    int(row) for row in rows
                                )
                                for rival, rows in applied.get(
                                    "rival_to_output_rows", {}
                                ).items()
                            }
                        )
                    else:
                        conditional_maps.append({})
                if conditional_maps and all(
                    mapping == conditional_maps[0]
                    for mapping in conditional_maps
                ):
                    shared_conditional = conditional_maps[0]
                    if shared_conditional:
                        phase_output_rows = int(cover[0][1].n_out)
                        if any(
                            int(child.n_out) != phase_output_rows
                            for _assignment, child in cover
                        ):
                            raise ValueError(
                                "conditional phase children changed output "
                                "width across the sound phase cover"
                            )
                        phase_property_row_groups = tuple(
                            (
                                *phase_property_row_groups[rival],
                                *shared_conditional.get(rival, ()),
                            )
                            for rival in range(M)
                        )
                phase_solver_C = np.eye(
                    phase_output_rows, dtype=np.float64
                )
                phase_solver_thresholds = np.zeros(
                    phase_output_rows, dtype=np.float64
                )
                phase_solver_kwargs = dict(solver_kwargs)
                phase_solver_kwargs["safe_row_groups"] = (
                    phase_property_row_groups
                )
                phase_children_receipt: List[Dict[str, Any]] = []

                def solve_phase_child(
                    index,
                    assignment,
                    child,
                    local_C,
                    local_thresholds,
                    local_solver_kwargs,
                    budget_cap=None,
                ):
                    child_started = time.monotonic()
                    child_budget = max(
                        0.0,
                        float(hybridz_deadline) - child_started,
                    )
                    if budget_cap is not None:
                        child_budget = min(
                            child_budget, max(0.0, float(budget_cap))
                        )
                    if child_budget <= 0.0:
                        return {
                            "index": int(index),
                            "assignment": [
                                [int(position), int(value)]
                                for position, value in assignment
                            ],
                            "verdict": "UNKNOWN",
                            "witness": None,
                            "elapsed_seconds": 0.0,
                            "solver_budget_seconds": 0.0,
                            "stats": {},
                            "safe_capability": None,
                            "safe_contract_valid": False,
                            "shared_deadline_respected": False,
                            "error": "shared_deadline_before_child",
                        }
                    try:
                        child_verdict, child_witness = hz_objbound_decide(
                            child,
                            local_C,
                            local_thresholds,
                            time_limit=child_budget,
                            require_base_feasible=False,
                            **local_solver_kwargs,
                        )
                        child_stats = getattr(
                            child, "_solver_objbound_stats", {}
                        )
                        if not isinstance(child_stats, dict):
                            child_stats = {}
                        child_safe_capability = (
                            hz_objbound_safe_capability_receipt(
                                child,
                                local_C,
                                local_thresholds,
                                is_unsafe_linear=bool(
                                    local_solver_kwargs.get(
                                        "is_unsafe_linear", False
                                    )
                                ),
                                tol=float(
                                    local_solver_kwargs.get(
                                        "tol", 1e-9
                                    )
                                ),
                                require_base_feasible=False,
                                base_witness_precheck=bool(
                                    local_solver_kwargs.get(
                                        "base_witness_precheck", True
                                    )
                                ),
                                safe_row_groups=(
                                    local_solver_kwargs.get(
                                        "safe_row_groups"
                                    )
                                ),
                                expected_safe_group_count=(
                                    local_solver_kwargs.get(
                                        "expected_safe_group_count"
                                    )
                                ),
                                require_binary_relaxation_lp=False,
                            )
                        )
                        shared_deadline_respected = bool(
                            time.monotonic()
                            <= float(hybridz_deadline)
                        )
                        child_safe_contract = bool(
                            child_verdict == "SAFE"
                            and child_witness is None
                            and child_safe_capability is not None
                            and shared_deadline_respected
                        )
                        keep_stats = {
                            key: child_stats.get(key)
                            for key in (
                                "base_feasibility_status",
                                "base_feasibility_reason",
                                "cube_min_upper",
                                "cube_max_upper",
                                "cube_pruned_rows",
                                "cube_survivor_rows",
                                "row_prefix_lp_status",
                                "row_prefix_lp_eligible_rows",
                                "row_prefix_lp_certified_rows",
                                "row_prefix_lp_elapsed_s",
                                "row_prefix_lp_model_receipts",
                                "row_prefix_gpu_dual_status",
                                "row_prefix_gpu_dual_certified_rows",
                                "row_prefix_gpu_dual_steps_completed",
                                "row_prefix_gpu_dual_initial_support_min",
                                "row_prefix_gpu_dual_initial_support_max",
                                "row_prefix_gpu_dual_candidate_support_min",
                                "row_prefix_gpu_dual_candidate_support_max",
                                "row_prefix_gpu_dual_support_improved_rows",
                                "row_prefix_gpu_dual_support_best_improvement",
                                "row_prefix_gpu_dual_candidate_dual_nnz_total",
                                "row_prefix_gpu_dual_checked_dual_nnz_total",
                                "row_prefix_gpu_dual_cert_upper_max",
                                "row_prefix_gpu_dual_model_receipts",
                                "gpu_dual_pc_cbde_status",
                                "gpu_dual_pc_cbde_elapsed_s",
                                "gpu_dual_pc_cbde_budget_s",
                                "gpu_dual_pc_cbde_deadline_reached",
                                "gpu_dual_pc_cbde_cone_rows",
                                "gpu_dual_pc_cbde_cone_row_count",
                                "gpu_dual_pc_cbde_local_row_count",
                                "gpu_dual_pc_cbde_bridge_row_count",
                                "gpu_dual_pc_cbde_generated_row_count",
                                "gpu_dual_pc_cbde_generated_warm_nonzero_count",
                                "gpu_dual_pc_cbde_generated_warm_truncated_count",
                                "gpu_dual_pc_cbde_source_row_count",
                                "gpu_dual_pc_cbde_ignored_source_row_count",
                                "gpu_dual_pc_cbde_full_nnz",
                                "gpu_dual_pc_cbde_updates",
                                "gpu_dual_pc_cbde_checked_upper_full",
                                "gpu_dual_pc_cbde_checked_upper_without_generated",
                                "gpu_dual_pc_cbde_checked_upper_without_bridge",
                                "gpu_dual_pc_cbde_checked_upper_without_both",
                                "gpu_dual_pc_cbde_all_ablations_verified",
                                "gpu_dual_pc_cbde_strict_family_ablation",
                                "gpu_dual_pc_cbde_strict_family_ablation_tol",
                                "gpu_dual_pc_cbde_full_vs_without_generated_gap",
                                "gpu_dual_pc_cbde_full_vs_without_bridge_gap",
                                "gpu_dual_pc_cbde_old_support",
                                "gpu_dual_pc_cbde_full_support",
                                "gpu_dual_pc_cbde_support_improvement",
                                "gpu_dual_pc_cbde_support_improvement_tol",
                                "gpu_dual_pc_cbde_replaced_old_candidate",
                                "gpu_dual_pc_cbde_error_type",
                                "gpu_dual_pc_cbde_error_message",
                                "gpu_dual_pc_cbde_proof_authority",
                                "lp_status",
                                "lp_certified_rows",
                                "lp_cert_max_upper",
                                "lp_elapsed_s",
                                "safe_row_groups_resolved",
                                "safe_row_groups_unresolved",
                                "all_rivals_covered",
                                "exact_phase_cover_member",
                                "exact_phase_cover_vacuous_child_allowed",
                            )
                            if key in child_stats
                        }
                        return {
                            "index": int(index),
                            "assignment": [
                                [int(position), int(value)]
                                for position, value in assignment
                            ],
                            "verdict": str(child_verdict),
                            "witness": child_witness,
                            "elapsed_seconds": float(
                                time.monotonic() - child_started
                            ),
                            "solver_budget_seconds": float(child_budget),
                            "stats": keep_stats,
                            "safe_capability": child_safe_capability,
                            "safe_contract_valid": child_safe_contract,
                            "shared_deadline_respected": (
                                shared_deadline_respected
                            ),
                            "error": None,
                        }
                    except Exception as exc:
                        return {
                            "index": int(index),
                            "assignment": [
                                [int(position), int(value)]
                                for position, value in assignment
                            ],
                            "verdict": "UNKNOWN",
                            "witness": None,
                            "elapsed_seconds": float(
                                time.monotonic() - child_started
                            ),
                            "solver_budget_seconds": float(child_budget),
                            "stats": {},
                            "safe_capability": None,
                            "safe_contract_valid": False,
                            "shared_deadline_respected": False,
                            "error": (
                                f"{type(exc).__name__}:{str(exc)[:500]}"
                            ),
                        }

                def run_phase_cover(
                    local_cover,
                    local_C,
                    local_thresholds,
                    local_solver_kwargs,
                    *,
                    budget_cap=None,
                    thread_name="hybridz_phase",
                ):
                    receipts: List[Dict[str, Any]] = []

                    def failed_receipt(index: int, exc: Exception):
                        assignment = local_cover[int(index)][0]
                        return {
                            "index": int(index),
                            "assignment": [
                                [int(position), int(value)]
                                for position, value in assignment
                            ],
                            "verdict": "UNKNOWN",
                            "witness": None,
                            "elapsed_seconds": 0.0,
                            "solver_budget_seconds": 0.0,
                            "stats": {},
                            "safe_capability": None,
                            "safe_contract_valid": False,
                            "shared_deadline_respected": False,
                            "error": (
                                f"{type(exc).__name__}:"
                                f"{str(exc)[:500]}"
                            ),
                        }

                    # One HiGHS model uses the fixed five-thread gate
                    # allocation.  Depth one therefore consumes ten solver
                    # threads; depth two consumes the declared twenty-thread
                    # cap without nesting a second process pool.
                    try:
                        with ThreadPoolExecutor(
                            max_workers=len(local_cover),
                            thread_name_prefix=thread_name,
                        ) as executor:
                            pending = {
                                executor.submit(
                                    solve_phase_child,
                                    index,
                                    assignment,
                                    child,
                                    local_C,
                                    local_thresholds,
                                    local_solver_kwargs,
                                    budget_cap,
                                ): int(index)
                                for index, (assignment, child) in enumerate(
                                    local_cover
                                )
                            }
                            for future in as_completed(pending):
                                index = int(pending[future])
                                try:
                                    result = future.result()
                                    if (
                                        not isinstance(result, Mapping)
                                        or result.get("index") != index
                                    ):
                                        raise ValueError(
                                            "phase child returned a malformed "
                                            "or misbound receipt"
                                        )
                                    receipts.append(dict(result))
                                except Exception as exc:
                                    receipts.append(
                                        failed_receipt(index, exc)
                                    )
                    except Exception as exc:
                        return [
                            failed_receipt(index, exc)
                            for index in range(len(local_cover))
                        ]
                    receipts.sort(key=lambda item: int(item["index"]))
                    return receipts

                focused_preflight = None
                focused_all_safe = False
                focused_is_complete_property = False
                if phase_focus_rival_ids:
                    focus_rival = int(phase_focus_rival_ids[0])
                    if not 0 <= focus_rival < len(
                        phase_property_row_groups
                    ):
                        raise ValueError(
                            "phase focus rival is outside property groups"
                        )
                    focus_output_rows = tuple(
                        int(row)
                        for row in phase_property_row_groups[focus_rival]
                    )
                    focus_C = np.zeros(
                        (len(focus_output_rows), phase_output_rows),
                        dtype=np.float64,
                    )
                    for local_row, output_row in enumerate(
                        focus_output_rows
                    ):
                        focus_C[local_row, output_row] = 1.0
                    focus_thresholds = np.zeros(
                        len(focus_output_rows), dtype=np.float64
                    )
                    focus_kwargs = dict(solver_kwargs)
                    focus_kwargs.update(
                        {
                            "safe_row_groups": (
                                tuple(range(len(focus_output_rows))),
                            ),
                            "expected_safe_group_count": 1,
                            "lp_prefilter_fraction": 1.0,
                            "lp_prefilter_max_seconds": min(
                                25.0,
                                float(
                                    getattr(
                                        hz_cfg,
                                        "lp_prefilter_max_seconds",
                                        8.0,
                                    )
                                ),
                            ),
                        }
                    )
                    focus_cover = []
                    for assignment, child in cover:
                        focus_child = copy.copy(child)
                        original_prefix = getattr(
                            child,
                            "_solver_row_constraint_prefix_frames",
                            {},
                        )
                        remapped_prefix = {}
                        if isinstance(original_prefix, dict):
                            for local_row, output_row in enumerate(
                                focus_output_rows
                            ):
                                entry = original_prefix.get(output_row)
                                if not isinstance(entry, dict):
                                    continue
                                remapped = dict(entry)
                                remapped["spec_row"] = int(local_row)
                                remapped["output_row"] = int(output_row)
                                remapped_prefix[int(local_row)] = remapped
                        setattr(
                            focus_child,
                            "_solver_row_constraint_prefix_frames",
                            remapped_prefix,
                        )
                        focus_cover.append((assignment, focus_child))
                    focus_receipts = run_phase_cover(
                        tuple(focus_cover),
                        focus_C,
                        focus_thresholds,
                        focus_kwargs,
                        budget_cap=25.0,
                        thread_name="hybridz_phase_focus",
                    )
                    focused_all_safe = bool(
                        phase_cover_audit.get("proof_authority") is True
                        and len(focus_receipts) == len(focus_cover)
                        and all(
                            item.get("safe_contract_valid") is True
                            for item in focus_receipts
                        )
                    )
                    focused_is_complete_property = bool(M == 1)
                    focused_preflight = {
                        "schema": (
                            "verifier_property_phase_split_focus_v1"
                        ),
                        "rival_id": int(focus_rival),
                        "output_rows": [
                            int(row) for row in focus_output_rows
                        ],
                        "all_phase_children_safe": focused_all_safe,
                        "certifies_complete_property": bool(
                            focused_all_safe
                            and focused_is_complete_property
                        ),
                        "proof_authority_for_rival": focused_all_safe,
                        "budget_cap_per_child_seconds": 25.0,
                        "children": [
                            {
                                key: value
                                for key, value in item.items()
                                if key != "witness"
                            }
                            for item in focus_receipts
                        ],
                    }

                full_cover_attempted = bool(
                    focused_preflight is None
                    or focused_all_safe
                )
                if focused_all_safe and focused_is_complete_property:
                    phase_children_receipt = focus_receipts
                elif full_cover_attempted:
                    phase_children_receipt = run_phase_cover(
                        cover,
                        phase_solver_C,
                        phase_solver_thresholds,
                        phase_solver_kwargs,
                    )
                else:
                    phase_children_receipt = focus_receipts
                child_verdicts = [
                    str(item["verdict"])
                    for item in phase_children_receipt
                ]
                all_safe = bool(
                    phase_cover_audit.get("proof_authority") is True
                    and (
                        (
                            focused_all_safe
                            and focused_is_complete_property
                        )
                        or (
                            full_cover_attempted
                            and len(child_verdicts) == len(cover)
                            and all(
                                item.get("safe_contract_valid") is True
                                for item in phase_children_receipt
                            )
                        )
                    )
                )
                verdict = "SAFE" if all_safe else "UNKNOWN"
                witness = None
                all_base_feasible = bool(
                    all(
                        item.get("stats", {}).get(
                            "base_feasibility_status"
                        )
                        in {
                            "FEASIBLE",
                            "EXACT_COVER_MEMBER_NOT_REQUIRED",
                        }
                        for item in phase_children_receipt
                    )
                )
                all_rivals_covered = bool(
                    all(
                        item.get("stats", {}).get(
                            "all_rivals_covered", False
                        )
                        for item in phase_children_receipt
                    )
                )
                meta.update(
                    {
                        "base_feasibility_status": (
                            "EXACT_PHASE_COVER_AUTHORIZED"
                            if all_base_feasible
                            else "CHILD_UNKNOWN"
                        ),
                        "all_rivals_covered": all_rivals_covered,
                        "property_phase_split": {
                            "schema": (
                                "verifier_property_phase_split_v1"
                            ),
                            "status": (
                                "all_children_safe"
                                if all_safe
                                else "focused_rival_unresolved"
                                if (
                                    focused_preflight is not None
                                    and not focused_all_safe
                                )
                                else "child_unresolved"
                            ),
                            "proof_authority": bool(all_safe),
                            "proof_rule": (
                                "every_exact_binary_phase_slice_is_a_"
                                "subset_of_its_Fraction_audited_outward_"
                                "child;private_live_SAFE_capability_is_"
                                "required_for_every_child"
                            ),
                            "binary_depth": int(phase_depth),
                            "expected_child_count": int(1 << phase_depth),
                            "actual_child_count": int(len(cover)),
                            "all_assignments_enumerated": bool(
                                phase_cover_audit.get(
                                    "all_assignments_enumerated"
                                )
                                is True
                            ),
                            "phase_cover_audit": phase_cover_audit,
                            "children_run_in_parallel": True,
                            "parallel_workers": int(len(cover)),
                            "focused_rival_preflight": focused_preflight,
                            "full_cover_attempted": bool(
                                full_cover_attempted
                                and not focused_is_complete_property
                            ),
                            "children": [
                                {
                                    key: value
                                    for key, value in item.items()
                                    if key != "witness"
                                }
                                for item in phase_children_receipt
                            ],
                            "elapsed_seconds": float(
                                time.monotonic() - phase_started
                            ),
                        },
                    }
                )
        elif micro_rlt_parent_only_diagnostic:
            micro_rlt_operator_receipt = operator_meta.get(
                "property_micro_rlt", {}
            )
            parent_prefilter_receipt = {
                "schema": (
                    "verifier_property_micro_rlt_parent_prefilter_v1"
                ),
                "enabled": True,
                "status": "no_selected_binary_diagnostic_stop",
                "proof_authority": False,
                "scope": "parent_pre_phase_fix",
                "configured_seconds": float(
                    getattr(
                        hz_cfg,
                        "property_micro_rlt_parent_prefilter_seconds",
                        0.0,
                    )
                ),
                "actual_budget_seconds": 0.0,
                "elapsed_seconds": 0.0,
                "parent_call_count": 0,
                "solver_verdict": None,
                "safe_only": True,
                "base_witness_precheck": False,
                "operator_receipt_status": (
                    micro_rlt_operator_receipt.get("status")
                    if isinstance(
                        micro_rlt_operator_receipt, Mapping
                    )
                    else None
                ),
                "operator_receipt_live_validation": (
                    micro_rlt_operator_receipt.get(
                        "live_result_validation_passed"
                    )
                    if isinstance(
                        micro_rlt_operator_receipt, Mapping
                    )
                    else None
                ),
                "stats": {},
                "error": None,
            }
            meta["property_micro_rlt_parent_prefilter"] = (
                parent_prefilter_receipt
            )
            return return_micro_rlt_parent_only_diagnostic(
                parent_receipt=parent_prefilter_receipt,
                operator_receipt=micro_rlt_operator_receipt,
            )
        else:
            if operator_phase_clique_pipeline_result is not None:
                from act.back_end.hybridz_tf.operator_phase_clique_pipeline import (
                    validate_consumed_operator_phase_clique_solver_build,
                )

                if (
                    operator_phase_clique_source_build is None
                    or operator_phase_clique_solver_build is None
                    or hz is not operator_phase_clique_solver_build.hz
                    or not validate_consumed_operator_phase_clique_solver_build(
                        operator_phase_clique_pipeline_result,
                        operator_phase_clique_solver_build,
                    )
                ):
                    meta.update(
                        {
                            "lane": 0,
                            "reason": (
                                "operator_phase_clique_terminal_"
                                "transaction_rejected"
                            ),
                            "hz_verdict": "UNKNOWN",
                            "hz_has_witness": False,
                            "hybridz_elapsed_s": (
                                time.monotonic() - hybridz_started
                            ),
                        }
                    )
                    return [
                        VerifyResult(
                            VerifyStatus.UNKNOWN,
                            metadata=meta,
                        )
                    ]
                hz_timeout = max(
                    0.0,
                    float(hybridz_deadline) - time.monotonic(),
                )
                if hz_timeout <= 0.0:
                    return [
                        VerifyResult(
                            VerifyStatus.TIMEOUT,
                            metadata={
                                **meta,
                                "lane": 0,
                                "reason": "hybridz_total_deadline",
                                "timeout_stage": (
                                    "operator_phase_clique_terminal_"
                                    "transaction"
                                ),
                                "hz_timeout_s": hybridz_total_timeout,
                                "hz_solver_budget_s": 0.0,
                                "hybridz_elapsed_s": (
                                    time.monotonic() - hybridz_started
                                ),
                            },
                        )
                    ]
            _phase_clique_progress("hz_objbound_decide_start")
            verdict, witness = hz_objbound_decide(
                hz,
                solver_C,
                solver_thresholds,
                time_limit=hz_timeout,
                **solver_kwargs,
            )
            _phase_clique_progress("hz_objbound_decide_done")
            objbound_stats = getattr(hz, "_solver_objbound_stats", None)
            if isinstance(objbound_stats, dict):
                meta.update(objbound_stats)
            if phase_split_mode:
                meta["property_phase_split"] = {
                    "schema": "verifier_property_phase_split_v1",
                    "status": "no_unstable_selected_binary",
                    "proof_authority": False,
                    "binary_depth": int(getattr(hz, "n_bin", 0)),
                    "baseline_solver_used": True,
                }
        meta.update({
            "lane": 0,
            "hz_verdict": verdict,
            "hz_timeout_s": hybridz_total_timeout,
            "hz_solver_budget_s": hz_timeout,
            "hybridz_elapsed_s": time.monotonic() - hybridz_started,
            "hz_has_witness": witness is not None,
        })
        if verdict == "SAFE":
            return [VerifyResult(VerifyStatus.CERTIFIED, metadata=meta)]
        if verdict == "UNSAFE":
            if property_upper_output:
                # A violating affine upper plane is not evidence that the
                # original network violates the property.  Tail folding is a
                # one-sided SAFE certificate path and can never emit a
                # counterexample candidate.
                meta["reason"] = "property_tail_upper_is_safe_only"
                meta["property_tail_unsafe_demoted"] = True
                meta["hz_has_witness"] = False
                return [VerifyResult(VerifyStatus.UNKNOWN, metadata=meta)]
            x_batch, decode_reason = _hybridz_witness_input(
                hz,
                witness,
                seed_bounds,
                active_tf,
            )
            meta["hz_candidate_decode"] = decode_reason
            if x_batch is None:
                meta["reason"] = "hybridz_unsafe_candidate_not_decodable"
                return [VerifyResult(VerifyStatus.UNKNOWN, metadata=meta)]

            model_unsafe, model_reason = _hybridz_model_candidate_check(
                x_batch=x_batch,
                model_fn=model_fn,
                C=C,
                thresholds=thresholds,
                M=M,
                n_out=n_out,
                is_unsafe_linear=is_unsafe_linear,
            )
            meta["hz_candidate_model_check"] = model_reason
            if model_unsafe is not None:
                meta["hz_candidate_model_unsafe"] = bool(model_unsafe)

            replay_ok, replay_reason, receipt = _hybridz_independent_replay(
                counterexample_replay_fn,
                x_batch,
            )
            meta["hz_independent_replay"] = replay_reason
            if isinstance(receipt, dict):
                meta["hz_replay_receipt"] = {
                    str(key): value
                    for key, value in receipt.items()
                    if isinstance(value, (str, int, float, bool, type(None)))
                }
            elif receipt is not None:
                meta["hz_replay_receipt"] = repr(receipt)

            # Any disagreement with the optional canonical ACT-model check is
            # treated as an audit conflict.  Only independently replayed,
            # concrete evidence may leave the verifier as FALSIFIED.
            if replay_ok and model_unsafe is not False:
                return [
                    VerifyResult(
                        VerifyStatus.FALSIFIED,
                        counterexample=x_batch.detach().cpu()[0].clone(),
                        metadata=meta,
                    )
                ]
            meta["reason"] = (
                "hybridz_replay_conflict"
                if replay_ok and model_unsafe is False
                else "hybridz_unsafe_candidate_not_replayed"
            )
            return [VerifyResult(VerifyStatus.UNKNOWN, metadata=meta)]
        meta["reason"] = "hybridz_verdict_unknown"
        return [VerifyResult(VerifyStatus.UNKNOWN, metadata=meta)]

    C_pos = C.clamp(min=0)
    C_neg = C.clamp(max=0)
    lb_exp = output_lb.repeat_interleave(M, dim=0)
    ub_exp = output_ub.repeat_interleave(M, dim=0)

    if is_unsafe_linear:
        # UNSAFE polytope = {y : C y <= d}. Property is SAFE iff for all y in
        # the box, EXISTS row i with c_i @ y > d_i (i.e. y leaves the polytope
        # on row i). Sound under-approximation: EXISTS row i such that
        # min_{y in box} (c_i @ y) > d_i. min(c_i @ y) = c_i_pos @ lb + c_i_neg @ ub.
        margin_min = (C_pos * lb_exp + C_neg * ub_exp).sum(dim=-1)
        certified = (margin_min.view(B, M) > thresholds).any(dim=-1)
    else:
        # LINEAR_LE / TOP1_ROBUST / MARGIN_ROBUST / RANGE: certified iff for
        # all y in the box, ALL rows max_y (c_i @ y) < d_i.
        margin_max = (C_pos * ub_exp + C_neg * lb_exp).sum(dim=-1)
        certified = (margin_max.view(B, M) < thresholds).all(dim=-1)

    # 5. Concrete falsification (optional).
    falsified = torch.zeros(B, dtype=torch.bool, device=device)
    counterexamples: List[Optional[torch.Tensor]] = [None] * B
    if model_fn is not None:
        x_center = 0.5 * (seed_bounds.lb + seed_bounds.ub)
        y_concrete = model_fn(x_center)
        if y_concrete.dim() != 2 or y_concrete.shape != (B, n_out):
            raise ValueError(
                f"verify_once: model_fn returned shape "
                f"{tuple(y_concrete.shape)}, expected ({B}, {n_out})"
            )
        y_concrete = y_concrete.to(device=device, dtype=dtype)
        C_view = C.view(B, M, n_out)
        concrete_violation = torch.einsum("bmn,bn->bm", C_view, y_concrete)
        if is_unsafe_linear:
            # Concrete y is in the UNSAFE polytope iff ALL rows c_i @ y <= d_i;
            # that is the violation condition for UNSAFE_LINEAR.
            falsified = (~certified) & (
                (concrete_violation <= thresholds).all(dim=-1)
            )
        else:
            # ALL-rows kinds: FALSIFIED iff ANY lane's concrete margin
            # meets-or-exceeds threshold.
            falsified = (~certified) & (
                (concrete_violation >= thresholds).any(dim=-1)
            )
        if falsified.any():
            x_center_cpu = x_center.detach().cpu()
            # B1 (oracle-verified): single sync via .tolist() replaces B per-element .item() syncs.
            # torch.where returns ascending indices; lane order is preserved.
            for i in torch.where(falsified)[0].tolist():
                counterexamples[i] = x_center_cpu[i].clone()

    # 6. Assemble per-lane results.
    results: List[VerifyResult] = []
    cert_list = certified.tolist()
    fals_list = falsified.tolist()
    for i in range(B):
        meta: Dict[str, Any] = {"lane": i, "B": B, "M": M}
        if cert_list[i]:
            results.append(
                VerifyResult(VerifyStatus.CERTIFIED, metadata=meta)
            )
        elif fals_list[i]:
            results.append(
                VerifyResult(
                    VerifyStatus.FALSIFIED,
                    counterexample=counterexamples[i],
                    metadata=meta,
                )
            )
        else:
            results.append(
                VerifyResult(VerifyStatus.UNKNOWN, metadata=meta)
            )
    return (results, after) if collect_facts else results


#===---------------------------------------------------------------------===#
# Self-contained ASSERT-encoding + verify_once test battery.
# Run via: python -m act.back_end.verifier
#===---------------------------------------------------------------------===#





def _make_dense_net_box_test(  # pragma: no cover
    B: int,
    n_in: int,
    n_out: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    lb_in: torch.Tensor,
    ub_in: torch.Tensor,
    assert_params: Dict[str, Any],
):
    # assert_params is high-level (kind + y_true/margin/c/d/lb/ub); lift to
    # encoded form via OutputSpec.encode_linear to match the production
    # OutputSpecLayer.to_act_layers path.
    from act.back_end.core import Layer, Net
    from act.front_end.specs import OutputSpec

    in_v = list(range(n_in))
    out_v = list(range(n_in, n_in + n_out))

    spec_kwargs = {
        k: assert_params[k] for k in ("y_true", "margin", "c", "d", "lb", "ub")
        if k in assert_params
    }
    out_spec = OutputSpec(kind=assert_params["kind"], **spec_kwargs)
    encoded = out_spec.encode_linear(
        B=B, n_out=n_out, device=weight.device, dtype=weight.dtype,
    )

    layers = [
        Layer(
            id=0,
            kind=LayerKind.INPUT.value,
            params={"shape": (B, n_in), "dtype": str(weight.dtype)},
            in_vars=[],
            out_vars=in_v,
        ),
        Layer(
            id=1,
            kind=LayerKind.INPUT_SPEC.value,
            params={"kind": "BOX", "lb": lb_in, "ub": ub_in},
            in_vars=in_v,
            out_vars=in_v,
        ),
        Layer(
            id=2,
            kind=LayerKind.DENSE.value,
            params={
                "weight": weight,
                "in_features": n_in,
                "out_features": n_out,
                "weight_pos": weight.clamp(min=0),
                "weight_neg": weight.clamp(max=0),
                "bias": bias,
                "input_shape": (n_in,),
            },
            in_vars=in_v,
            out_vars=out_v,
        ),
        Layer(
            id=3,
            kind=LayerKind.ASSERT.value,
            params=encoded,
            in_vars=out_v,
            out_vars=out_v,
        ),
    ]
    preds = {0: [], 1: [0], 2: [1], 3: [2]}
    succs = {0: [1], 1: [2], 2: [3], 3: []}
    return Net(layers=layers, preds=preds, succs=succs)


def _make_attn_dual_planar_net(  # pragma: no cover
    B: int, L: int, D: int, H: int,
    center: torch.Tensor, eps: float,
    assert_d: float,
    *,
    mask: "torch.Tensor | None" = None,
    clamp_alpha: bool = False,
) -> "tuple[Net, dict[str, Any]]":
    """Build INPUT -> Q/K DENSE projections -> ATT_SCORES(dual_planar) -> ASSERT.

    Exercises the real ``analyze()`` -> ``tf_att_scores`` ->
    ``att_scores_dual_planar``/``LinearBounds`` -> ``cons_exportor``'s
    ``att_dual_planar:`` export path end-to-end, not direct unit calls into
    ``interval_tf/tf_attention.py``. The ``q_lb``/``k_lb`` baked onto the
    ATT_SCORES layer are seeded from the same box as INPUT_SPEC and pushed
    through the same ``Wq``/``Wk`` as the DENSE Q/K layers, so the result is
    a faithful (not synthetic) attention-score relaxation of this network.
    """
    from act.back_end.core import Layer
    from act.back_end.interval_tf.tf_attention import LinearBounds
    from act.front_end.specs import OutputSpec

    n_in = L * D
    in_v = list(range(n_in))
    lb_in = center - eps
    ub_in = center + eps

    Wq = torch.randn(H, D, dtype=center.dtype, generator=torch.Generator().manual_seed(11)) * 0.3
    Wk = torch.randn(H, D, dtype=center.dtype, generator=torch.Generator().manual_seed(12)) * 0.3

    eye = torch.eye(D, dtype=center.dtype)
    center3 = center.reshape(B, L, D)
    radius3 = torch.full((B, L, D), eps, dtype=center.dtype)
    seed_w = radius3.unsqueeze(-1) * eye
    emb_lb = LinearBounds(
        seed_w, seed_w.clone(), center3.clone(), center3.clone(),
        p=float("inf"), eps=1.0, perturbed_words=1,
    )
    q_lb = emb_lb.matmul(Wq)
    k_lb = emb_lb.matmul(Wk)

    q_vars = list(range(n_in, n_in + L * H))
    k_vars = list(range(n_in + L * H, n_in + 2 * L * H))
    score_vars = list(range(n_in + 2 * L * H, n_in + 2 * L * H + L * L))

    def block_diag_proj(W: torch.Tensor) -> torch.Tensor:
        full = torch.zeros(L * H, n_in, dtype=center.dtype)
        for t in range(L):
            full[t * H:(t + 1) * H, t * D:(t + 1) * D] = W
        return full

    Wq_full, Wk_full = block_diag_proj(Wq), block_diag_proj(Wk)

    layers = [
        Layer(
            id=0, kind=LayerKind.INPUT.value,
            params={"shape": (B, n_in), "dtype": str(center.dtype)},
            in_vars=[], out_vars=in_v,
        ),
        Layer(
            id=1, kind=LayerKind.INPUT_SPEC.value,
            params={"kind": "BOX", "lb": lb_in, "ub": ub_in},
            in_vars=in_v, out_vars=in_v,
        ),
        Layer(
            id=2, kind=LayerKind.DENSE.value,
            params={
                "weight": Wq_full, "in_features": n_in, "out_features": L * H,
                "weight_pos": Wq_full.clamp(min=0), "weight_neg": Wq_full.clamp(max=0),
                "bias": torch.zeros(L * H, dtype=center.dtype), "input_shape": (n_in,),
            },
            in_vars=in_v, out_vars=q_vars,
        ),
        Layer(
            id=3, kind=LayerKind.DENSE.value,
            params={
                "weight": Wk_full, "in_features": n_in, "out_features": L * H,
                "weight_pos": Wk_full.clamp(min=0), "weight_neg": Wk_full.clamp(max=0),
                "bias": torch.zeros(L * H, dtype=center.dtype), "input_shape": (n_in,),
            },
            in_vars=in_v, out_vars=k_vars,
        ),
        Layer(
            id=4, kind=LayerKind.ATT_SCORES.value,
            params={
                "dk": float(H) ** 0.5, "q_vars": tuple(q_vars), "k_vars": tuple(k_vars),
                "q_src": 2, "k_src": 3,
                "attn_mode": "dual_planar", "q_lb": q_lb, "k_lb": k_lb, "head_size": H,
                "mask": mask, "clamp_alpha": clamp_alpha,
            },
            in_vars=q_vars + k_vars, out_vars=score_vars,
        ),
    ]
    n_scores = len(score_vars)
    out_spec = OutputSpec(
        kind="LINEAR_LE", c=torch.ones(n_scores, dtype=center.dtype),
        d=torch.tensor(assert_d, dtype=center.dtype),
    )
    encoded = out_spec.encode_linear(B=B, n_out=n_scores, device=center.device, dtype=center.dtype)
    layers.append(
        Layer(id=5, kind=LayerKind.ASSERT.value, params=encoded, in_vars=score_vars, out_vars=score_vars)
    )

    preds = {0: [], 1: [0], 2: [1], 3: [1], 4: [2, 3], 5: [4]}
    succs = {0: [1], 1: [2, 3], 2: [4], 3: [4], 4: [5], 5: []}
    net = Net(layers=layers, preds=preds, succs=succs)
    info: "dict[str, Any]" = {
        "Wq": Wq, "Wk": Wk, "lb_in": lb_in, "ub_in": ub_in,
        "score_id": 4, "n_in": n_in, "L": L, "D": D, "H": H,
    }
    return net, info


def _test_att_scores_dual_planar_analyze_soundness() -> None:  # pragma: no cover
    # Real `analyze()` worklist (not a direct LinearBounds unit call): the
    # propagated box for the ATT_SCORES(dual_planar) layer must bracket the
    # true concrete scaled-Q.K^T value for every sampled point in the box.
    from act.back_end.analyze import analyze
    from act.util.device_manager import get_default_dtype

    dtype = get_default_dtype()
    B, L, D, H = 1, 3, 4, 2
    torch.manual_seed(20)
    center = torch.randn(B, L * D, dtype=dtype) * 0.1
    eps = 0.05
    net, info = _make_attn_dual_planar_net(B, L, D, H, center, eps, assert_d=100.0)

    entry_fact = Fact(bounds=Bounds(info["lb_in"].clone(), info["ub_in"].clone()), cons=ConSet())
    _before, after, _globalC = analyze(net, 0, entry_fact)
    bounds = after[info["score_id"]].bounds

    Wq, Wk = info["Wq"], info["Wk"]
    l_box, u_box = info["lb_in"], info["ub_in"]
    n_samples = 100

    def concrete_scores(x: torch.Tensor) -> torch.Tensor:
        x3 = x.reshape(B, L, D)
        s = (x3 @ Wq.t()) @ (x3 @ Wk.t()).transpose(-1, -2) / (H ** 0.5)
        return s.reshape(B, -1)

    true_min = concrete_scores(l_box).clone()
    true_max = true_min.clone()
    for _ in range(n_samples):
        x = l_box + torch.rand_like(l_box) * (u_box - l_box)
        s = concrete_scores(x)
        true_min = torch.minimum(true_min, s)
        true_max = torch.maximum(true_max, s)
    assert (bounds.lb <= true_min + 1e-6).all(), "analyze(): unsound lower bound on ATT_SCORES(dual_planar)"
    assert (bounds.ub >= true_max - 1e-6).all(), "analyze(): unsound upper bound on ATT_SCORES(dual_planar)"


def _test_att_scores_dual_planar_verify_once_certified() -> None:  # pragma: no cover
    # End-to-end `verify_once()` through the dual-planar attention path with
    # a threshold far above the true score range -> CERTIFIED.
    from act.util.device_manager import get_default_dtype
    from act.util.stats import VerifyStatus

    dtype = get_default_dtype()
    B, L, D, H = 1, 3, 4, 2
    torch.manual_seed(21)
    center = torch.randn(B, L * D, dtype=dtype) * 0.1
    eps = 0.05
    net, _info = _make_attn_dual_planar_net(B, L, D, H, center, eps, assert_d=100.0)

    results = verify_once(net)
    assert len(results) == B
    assert results[0].status == VerifyStatus.CERTIFIED, f"expected CERTIFIED, got {results[0].status}"


def _test_att_scores_dual_planar_lp_export_solve() -> None:  # pragma: no cover
    # End-to-end LP export+solve through `cons_exportor`'s
    # `att_dual_planar:` handler (not reachable from any other test): a
    # tight threshold near the true score range exercises a real SAT/UNKNOWN
    # decision from TorchLPSolver, proving the export glue round-trips.
    from act.back_end.solver.solver_torchlp import TorchLPSolver
    from act.util.device_manager import get_default_dtype

    dtype = get_default_dtype()
    B, L, D, H = 1, 3, 4, 2
    torch.manual_seed(22)
    center = torch.randn(B, L * D, dtype=dtype) * 0.1
    eps = 0.05
    net, info = _make_attn_dual_planar_net(B, L, D, H, center, eps, assert_d=0.0)

    solution = setup_and_solve_batch(
        net, Bounds(info["lb_in"].clone(), info["ub_in"].clone()), TorchLPSolver(),
    )
    assert solution.statuses[0] in (SolveStatus.SAT, SolveStatus.UNKNOWN), (
        f"unexpected solver status {solution.statuses[0]!r}"
    )
    assert tuple(solution.x.shape)[0] == B
    assert float(solution.max_viol[0].item()) < 1.0, (
        f"LP residual too large: {float(solution.max_viol[0].item())}"
    )


def _test_att_scores_dual_planar_masked_and_clamp_alpha_soundness() -> None:  # pragma: no cover
    # Real `analyze()` with an additive mask and the clamp_alpha warm-start
    # variant both engaged -- exercises `fuse_attention_planes`'s
    # `clamp_alpha` branch and `att_scores_dual_planar`'s `mask is not None`
    # branch, neither hit by the unmasked/default tests above.
    from act.back_end.analyze import analyze
    from act.util.device_manager import get_default_dtype

    dtype = get_default_dtype()
    B, L, D, H = 1, 3, 4, 2
    torch.manual_seed(23)
    center = torch.randn(B, L * D, dtype=dtype) * 0.1
    eps = 0.05
    mask = torch.zeros(B, L, L, dtype=dtype)
    mask[0, 0, 1] = -5.0
    net, info = _make_attn_dual_planar_net(
        B, L, D, H, center, eps, assert_d=100.0, mask=mask, clamp_alpha=True,
    )

    entry_fact = Fact(bounds=Bounds(info["lb_in"].clone(), info["ub_in"].clone()), cons=ConSet())
    _before, after, _globalC = analyze(net, 0, entry_fact)
    bounds = after[info["score_id"]].bounds

    Wq, Wk = info["Wq"], info["Wk"]
    l_box, u_box = info["lb_in"], info["ub_in"]
    n_samples = 100

    def concrete_masked_scores(x: torch.Tensor) -> torch.Tensor:
        x3 = x.reshape(B, L, D)
        s = (x3 @ Wq.t()) @ (x3 @ Wk.t()).transpose(-1, -2) / (H ** 0.5)
        return (s + mask).reshape(B, -1)

    true_min = concrete_masked_scores(l_box).clone()
    true_max = true_min.clone()
    for _ in range(n_samples):
        x = l_box + torch.rand_like(l_box) * (u_box - l_box)
        s = concrete_masked_scores(x)
        true_min = torch.minimum(true_min, s)
        true_max = torch.maximum(true_max, s)
    assert (bounds.lb <= true_min + 1e-6).all(), "masked/clamp_alpha: unsound lower bound"
    assert (bounds.ub >= true_max - 1e-6).all(), "masked/clamp_alpha: unsound upper bound"


def _make_mini_transformer_block_net(  # pragma: no cover
    B: int, L: int, D: int, center: torch.Tensor, eps: float,
) -> "tuple[Net, dict[str, Any]]":
    """Build a real explicit-attention block: MHA_SPLIT(Q/K/V) -> ATT_SCORES
    (plain McCormick box mode, not dual_planar) -> CONCAT -> SOFTMAX ->
    ATT_MIX -> MHA_JOIN -> LAYERNORM(variant='no_var', broadcast gamma).

    Mirrors the per-position/per-feature decomposition torch2act's BERT
    graph builder uses (one MHA_SPLIT per query/key position, one ATT_MIX
    per value feature), at the smallest size (L=2 positions) that still
    requires the CONCAT-of-two-scores -> SOFTMAX -> two-feature ATT_MIX/
    MHA_JOIN path. None of these layer kinds have any other producer in
    the codebase (no NetFactory family, no torch2act path on this branch),
    so this is the only real (non-direct-unit-call) exercise of them.
    """
    from act.back_end.core import Layer
    from act.front_end.specs import OutputSpec

    n_in = L * D
    in_v = list(range(n_in))
    lb_in = center - eps
    ub_in = center + eps

    gen = torch.Generator().manual_seed(40)
    Wq = torch.randn(D, D, dtype=center.dtype, generator=gen) * 0.3
    Wk = torch.randn(D, D, dtype=center.dtype, generator=torch.Generator().manual_seed(41)) * 0.3
    Wv = torch.randn(D, D, dtype=center.dtype, generator=torch.Generator().manual_seed(42)) * 0.3

    layers = [
        Layer(id=0, kind=LayerKind.INPUT.value, params={"shape": (B, n_in), "dtype": str(center.dtype)}, in_vars=[], out_vars=in_v),
        Layer(id=1, kind=LayerKind.INPUT_SPEC.value, params={"kind": "BOX", "lb": lb_in, "ub": ub_in}, in_vars=in_v, out_vars=in_v),
    ]
    preds: "dict[int, list[int]]" = {0: [], 1: [0]}
    succs: "dict[int, list[int]]" = {0: [1], 1: []}
    next_id = 2
    next_var = n_in

    def alloc(n: int) -> "list[int]":
        nonlocal next_var
        v = list(range(next_var, next_var + n))
        next_var += n
        return v

    def add_layer(kind: str, params: "dict[str, Any]", in_vars: "list[int]", out_vars: "list[int]", pred_ids: "list[int]") -> int:
        nonlocal next_id
        layers.append(Layer(id=next_id, kind=kind, params=params, in_vars=in_vars, out_vars=out_vars))
        lid = next_id
        next_id += 1
        preds[lid] = pred_ids
        succs.setdefault(lid, [])
        for p in pred_ids:
            succs[p].append(lid)
        return lid

    mha_split = lambda W, role, **extra: add_layer(  # noqa: E731 - local convenience, not module API
        LayerKind.MHA_SPLIT.value,
        {"weight": W, "input_shape": (B, L, D), "hidden_size": D, "role": role, **extra},
        in_v, alloc(D if role != "value" else L), [1],
    )

    q_id = mha_split(Wq, "query", position=0)
    q_vars = layers[q_id].out_vars
    k_ids = [mha_split(Wk, "key", position=p) for p in range(L)]
    k_vars_per_pos = [layers[kid].out_vars for kid in k_ids]

    score_ids = []
    score_vars_flat: "list[int]" = []
    for kid, kv in zip(k_ids, k_vars_per_pos):
        sv = alloc(1)
        sid = add_layer(
            LayerKind.ATT_SCORES.value,
            {"dk": float(D) ** 0.5, "q_vars": q_vars, "k_vars": kv, "q_src": q_id, "k_src": kid},
            q_vars + kv, sv, [q_id, kid],
        )
        score_ids.append(sid)
        score_vars_flat += sv
    cat_vars = alloc(L)
    cat_id = add_layer(LayerKind.CONCAT.value, {"concat_dim": -1}, score_vars_flat, cat_vars, score_ids)
    sm_vars = alloc(L)
    sm_id = add_layer(LayerKind.SOFTMAX.value, {"axis": -1}, cat_vars, sm_vars, [cat_id])

    v_ids = [mha_split(Wv, "value", feature=f) for f in range(D)]
    v_vars_per_feature = [layers[vid].out_vars for vid in v_ids]

    mix_ids = []
    mix_vars_flat: "list[int]" = []
    for vid, vv in zip(v_ids, v_vars_per_feature):
        mv = alloc(1)
        mid = add_layer(
            LayerKind.ATT_MIX.value,
            {"rowsize": L, "w_vars": sm_vars, "v_vars": vv, "w_src": sm_id, "v_src": vid},
            sm_vars + vv, mv, [sm_id, vid],
        )
        mix_ids.append(mid)
        mix_vars_flat += mv
    join_vars = alloc(D)
    join_id = add_layer(LayerKind.MHA_JOIN.value, {}, mix_vars_flat, join_vars, mix_ids)

    # gamma.numel()==1 != D forces the broadcast-repeat branch.
    gamma = torch.tensor([1.5], dtype=center.dtype)
    beta = torch.tensor([0.1], dtype=center.dtype)
    ln_vars = alloc(D)
    ln_id = add_layer(
        LayerKind.LAYERNORM.value, {"gamma": gamma, "beta": beta, "variant": "no_var"},
        join_vars, ln_vars, [join_id],
    )

    out_spec = OutputSpec(kind="LINEAR_LE", c=torch.ones(D, dtype=center.dtype), d=torch.tensor(100.0, dtype=center.dtype))
    encoded = out_spec.encode_linear(B=B, n_out=D, device=center.device, dtype=center.dtype)
    assert_id = add_layer(LayerKind.ASSERT.value, encoded, ln_vars, ln_vars, [ln_id])

    net = Net(layers=layers, preds=preds, succs=succs)
    info: "dict[str, Any]" = {
        "Wq": Wq, "Wk": Wk, "Wv": Wv, "gamma": gamma, "beta": beta,
        "lb_in": lb_in, "ub_in": ub_in, "ln_id": ln_id, "n_in": n_in,
    }
    return net, info


def _test_mini_transformer_block_analyze_soundness() -> None:  # pragma: no cover
    # Real `analyze()` through MHA_SPLIT -> ATT_SCORES(box) -> SOFTMAX ->
    # ATT_MIX -> MHA_JOIN -> LAYERNORM(no_var, broadcast gamma): the
    # propagated box must bracket the true concrete forward pass.
    from act.back_end.analyze import analyze
    from act.util.device_manager import get_default_dtype

    dtype = get_default_dtype()
    B, L, D = 1, 2, 2
    torch.manual_seed(43)
    center = torch.randn(B, L * D, dtype=dtype) * 0.1
    eps = 0.05
    net, info = _make_mini_transformer_block_net(B, L, D, center, eps)

    entry_fact = Fact(bounds=Bounds(info["lb_in"].clone(), info["ub_in"].clone()), cons=ConSet())
    _before, after, _globalC = analyze(net, 0, entry_fact)
    bounds = after[info["ln_id"]].bounds

    Wq, Wk, Wv = info["Wq"], info["Wk"], info["Wv"]
    gamma, beta = info["gamma"], info["beta"]
    l_box, u_box = info["lb_in"], info["ub_in"]

    def concrete_forward(x: torch.Tensor) -> torch.Tensor:
        x3 = x.reshape(B, L, D)
        q = (x3 @ Wq.t())[:, 0, :]
        scores = torch.cat(
            [(q * (x3 @ Wk.t())[:, p, :]).sum(-1, keepdim=True) / (D ** 0.5) for p in range(L)], dim=-1,
        )
        probs = torch.softmax(scores, dim=-1)
        v_all = x3 @ Wv.t()
        mixed = torch.cat([(probs * v_all[:, :, f]).sum(-1, keepdim=True) for f in range(D)], dim=-1)
        centered = mixed - mixed.mean(dim=-1, keepdim=True)
        return centered * gamma.repeat(D) + beta.repeat(D)

    n_samples = 150
    true_min = concrete_forward(l_box).clone()
    true_max = true_min.clone()
    for _ in range(n_samples):
        x = l_box + torch.rand_like(l_box) * (u_box - l_box)
        y = concrete_forward(x)
        true_min = torch.minimum(true_min, y)
        true_max = torch.maximum(true_max, y)
    assert (bounds.lb <= true_min + 1e-6).all(), "mini transformer block: unsound lower bound"
    assert (bounds.ub >= true_max - 1e-6).all(), "mini transformer block: unsound upper bound"


def _test_mha_split_edge_cases_and_mask_add() -> None:  # pragma: no cover
    # Direct calls to the production transfer functions for the branches
    # the full-block Net above can't reach structurally: MHA_SPLIT with no
    # "weight" param (passthrough), MHA_SPLIT with no "role" (flatten), and
    # MASK_ADD (an unrelated single-layer op with no other test coverage).
    from act.back_end.core import Layer
    from act.back_end.interval_tf.tf_transformer import tf_mha_split, tf_mask_add
    from act.util.device_manager import get_default_dtype

    dtype = get_default_dtype()
    Bin = Bounds(torch.tensor([[-1.0, 2.0]], dtype=dtype), torch.tensor([[1.0, 3.0]], dtype=dtype))

    passthrough = tf_mha_split(Layer(id=0, kind=LayerKind.MHA_SPLIT.value, params={}, in_vars=[0, 1], out_vars=[0, 1]), Bin)
    assert torch.equal(passthrough.bounds.lb, Bin.lb) and torch.equal(passthrough.bounds.ub, Bin.ub), (
        "MHA_SPLIT with no weight must passthrough Bin unchanged"
    )

    W = torch.eye(2, dtype=dtype)
    flat = tf_mha_split(
        Layer(
            id=1, kind=LayerKind.MHA_SPLIT.value,
            params={"weight": W, "input_shape": (1, 1, 2), "hidden_size": 2}, in_vars=[0, 1], out_vars=[0, 1],
        ),
        Bin,
    )
    assert flat.bounds.lb.shape == (1, 2) and flat.bounds.ub.shape == (1, 2), "MHA_SPLIT flatten-role output shape"

    M = torch.tensor([[0.5, -0.5]], dtype=dtype)
    masked = tf_mask_add(Layer(id=2, kind=LayerKind.MASK_ADD.value, params={"M": M}, in_vars=[0, 1], out_vars=[0, 1]), Bin)
    assert torch.allclose(masked.bounds.lb, Bin.lb + M) and torch.allclose(masked.bounds.ub, Bin.ub + M), (
        "MASK_ADD must shift both bounds by M"
    )


def _test_new_elementwise_tf_soundness() -> None:  # pragma: no cover
    # Direct calls to the 5 new interval_tf/tf_mlp.py transfer functions
    # (ERF, SQRT, SIN, COS, QUANTIZE) -- no NetFactory family or other
    # producer generates these layer kinds, so this is their only exercise.
    # Each assertion samples the true concrete function over the box and
    # checks the propagated interval brackets it.
    from act.back_end.core import Layer
    from act.back_end.interval_tf.tf_mlp import tf_erf, tf_sqrt, tf_sin, tf_cos, tf_quantize
    from act.util.device_manager import get_default_dtype

    dtype = get_default_dtype()

    def assert_sound(name: str, lo: torch.Tensor, hi: torch.Tensor, l_box: torch.Tensor, u_box: torch.Tensor, fn, n: int = 150) -> None:
        true_min = fn(l_box).clone()
        true_max = true_min.clone()
        for _ in range(n):
            x = l_box + torch.rand_like(l_box) * (u_box - l_box)
            y = fn(x)
            true_min = torch.minimum(true_min, y)
            true_max = torch.maximum(true_max, y)
        assert (lo <= true_min + 1e-6).all(), f"{name}: unsound lower bound"
        assert (hi >= true_max - 1e-6).all(), f"{name}: unsound upper bound"

    l_erf = torch.tensor([[-1.0, 0.5]], dtype=dtype)
    u_erf = torch.tensor([[1.0, 2.0]], dtype=dtype)
    erf_out = tf_erf(Layer(id=0, kind=LayerKind.ERF.value, params={}, in_vars=[0, 1], out_vars=[0, 1]), Bounds(l_erf, u_erf))
    assert_sound("erf", erf_out.bounds.lb, erf_out.bounds.ub, l_erf, u_erf, torch.erf)

    # Box straddles negative -> exercises the min-clamp in tf_sqrt.
    l_sqrt = torch.tensor([[-1.0, 0.5]], dtype=dtype)
    u_sqrt = torch.tensor([[2.0, 3.0]], dtype=dtype)
    sqrt_out = tf_sqrt(Layer(id=1, kind=LayerKind.SQRT.value, params={}, in_vars=[0, 1], out_vars=[0, 1]), Bounds(l_sqrt, u_sqrt))
    assert_sound(
        "sqrt", sqrt_out.bounds.lb, sqrt_out.bounds.ub, l_sqrt, u_sqrt,
        lambda x: torch.sqrt(torch.clamp(x, min=0.0)),
    )

    # SIN/COS: narrow (no critical point), has-max, has-min, full-period(>=2pi).
    sin_cases = {"narrow": (0.1, 0.5), "has_max": (1.0, 2.0), "has_min": (-2.0, -1.0), "full_period": (0.0, 7.0)}
    for name, (lv, uv) in sin_cases.items():
        lb = torch.tensor([[lv]], dtype=dtype)
        ub = torch.tensor([[uv]], dtype=dtype)
        out = tf_sin(Layer(id=2, kind=LayerKind.SIN.value, params={}, in_vars=[0], out_vars=[0]), Bounds(lb, ub))
        assert_sound(f"sin[{name}]", out.bounds.lb, out.bounds.ub, lb, ub, torch.sin)

    cos_cases = {"narrow": (0.1, 0.5), "has_max": (-0.5, 0.5), "has_min": (2.5, 3.5), "full_period": (0.0, 7.0)}
    for name, (lv, uv) in cos_cases.items():
        lb = torch.tensor([[lv]], dtype=dtype)
        ub = torch.tensor([[uv]], dtype=dtype)
        out = tf_cos(Layer(id=3, kind=LayerKind.COS.value, params={}, in_vars=[0], out_vars=[0]), Bounds(lb, ub))
        assert_sound(f"cos[{name}]", out.bounds.lb, out.bounds.ub, lb, ub, torch.cos)

    scale = torch.tensor([0.1], dtype=dtype)
    zero_point = torch.tensor([0.0], dtype=dtype)
    l_q = torch.tensor([[-1.0, 0.5]], dtype=dtype)
    u_q = torch.tensor([[1.0, 2.0]], dtype=dtype)
    q_out = tf_quantize(
        Layer(
            id=4, kind=LayerKind.QUANTIZE.value,
            params={"scale": scale, "zero_point": zero_point, "qmin": -128, "qmax": 127},
            in_vars=[0, 1], out_vars=[0, 1],
        ),
        Bounds(l_q, u_q),
    )

    def quantize_concrete(x: torch.Tensor) -> torch.Tensor:
        code = torch.clamp(torch.round(x / scale), min=-128 - zero_point, max=127 - zero_point)
        return scale * code

    assert_sound("quantize", q_out.bounds.lb, q_out.bounds.ub, l_q, u_q, quantize_concrete)


def _make_dual_att_cores_net(  # pragma: no cover
    B: int, L: int, D: int, center: torch.Tensor, eps: float, assert_d: float,
) -> "tuple[Net, dict[str, Any]]":
    """DENSE Q/K/V -> ATT_SCORES -> SOFTMAX -> ATT_MIX -> CONCAT -> LAYERNORM -> GELU.

    The dual attention path (dual_tf/tf_transformer.py) consumes the bilinear
    cores ATT_SCORES (Q.Kt) / ATT_MIX (probs.V) with q_src/k_src/w_src/v_src
    reading predecessor boxes; it stubs MHA_SPLIT/MHA_JOIN. So Q/K/V come from
    DENSE (which dual supports) rather than the interval MHA_SPLIT decomposition,
    giving a net the DualSolver can run end to end. Non-degenerate dims (L,D>1)
    avoid the size-1 shape class.
    """
    from act.back_end.core import Layer
    from act.front_end.specs import OutputSpec

    n_in = L * D
    in_v = list(range(n_in))
    lb_in, ub_in = center - eps, center + eps
    Wq = torch.randn(D, D, dtype=center.dtype, generator=torch.Generator().manual_seed(71)) * 0.2
    Wk = torch.randn(D, D, dtype=center.dtype, generator=torch.Generator().manual_seed(72)) * 0.2
    Wv = torch.randn(D, D, dtype=center.dtype, generator=torch.Generator().manual_seed(73)) * 0.2

    layers = [
        Layer(id=0, kind=LayerKind.INPUT.value, params={"shape": (B, n_in), "dtype": str(center.dtype)}, in_vars=[], out_vars=in_v),
        Layer(id=1, kind=LayerKind.INPUT_SPEC.value, params={"kind": "BOX", "lb": lb_in, "ub": ub_in}, in_vars=in_v, out_vars=in_v),
    ]
    preds: "dict[int, list[int]]" = {0: [], 1: [0]}
    succs: "dict[int, list[int]]" = {0: [1], 1: []}
    next_id, next_var = 2, n_in

    def alloc(n: int) -> "list[int]":
        nonlocal next_var
        v = list(range(next_var, next_var + n)); next_var += n
        return v

    def add(kind, params, in_vars, out_vars, pred_ids) -> int:
        nonlocal next_id
        layers.append(Layer(id=next_id, kind=kind, params=params, in_vars=in_vars, out_vars=out_vars))
        lid = next_id; next_id += 1
        preds[lid] = pred_ids
        succs.setdefault(lid, [])
        for p in pred_ids:
            succs[p].append(lid)
        return lid

    def dense_pos(W, pos) -> int:
        full = torch.zeros(D, n_in, dtype=center.dtype)
        full[:, pos * D:(pos + 1) * D] = W
        return add(LayerKind.DENSE.value, {
            "weight": full, "in_features": n_in, "out_features": D,
            "weight_pos": full.clamp(min=0), "weight_neg": full.clamp(max=0),
            "bias": torch.zeros(D, dtype=center.dtype), "input_shape": (n_in,),
        }, in_v, alloc(D), [1])

    def dense_value_feature(W, feat) -> int:
        full = torch.zeros(L, n_in, dtype=center.dtype)
        for p in range(L):
            full[p, p * D:(p + 1) * D] = W[feat]
        return add(LayerKind.DENSE.value, {
            "weight": full, "in_features": n_in, "out_features": L,
            "weight_pos": full.clamp(min=0), "weight_neg": full.clamp(max=0),
            "bias": torch.zeros(L, dtype=center.dtype), "input_shape": (n_in,),
        }, in_v, alloc(L), [1])

    q_ids = [dense_pos(Wq, p) for p in range(L)]
    k_ids = [dense_pos(Wk, p) for p in range(L)]
    v_ids = [dense_value_feature(Wv, f) for f in range(D)]

    score_ids = []
    score_vars: "list[int]" = []
    for kp in range(L):
        sv = alloc(1)
        sid = add(LayerKind.ATT_SCORES.value, {
            "dk": float(D) ** 0.5,
            "q_vars": layers[q_ids[0]].out_vars, "k_vars": layers[k_ids[kp]].out_vars,
            "q_src": q_ids[0], "k_src": k_ids[kp],
        }, layers[q_ids[0]].out_vars + layers[k_ids[kp]].out_vars, sv, [q_ids[0], k_ids[kp]])
        score_ids.append(sid); score_vars += sv
    cat_id = add(LayerKind.CONCAT.value, {"concat_dim": -1}, score_vars, alloc(L), score_ids)
    sm_id = add(LayerKind.SOFTMAX.value, {"axis": -1}, layers[cat_id].out_vars, alloc(L), [cat_id])
    mix_ids = []
    mix_vars: "list[int]" = []
    for f in range(D):
        mv = alloc(1)
        mid = add(LayerKind.ATT_MIX.value, {
            "rowsize": L, "w_vars": layers[sm_id].out_vars, "v_vars": layers[v_ids[f]].out_vars,
            "w_src": sm_id, "v_src": v_ids[f],
        }, layers[sm_id].out_vars + layers[v_ids[f]].out_vars, mv, [sm_id, v_ids[f]])
        mix_ids.append(mid); mix_vars += mv
    join_id = add(LayerKind.CONCAT.value, {"concat_dim": -1}, mix_vars, alloc(D), mix_ids)
    gamma = torch.ones(D, dtype=center.dtype)
    beta = torch.zeros(D, dtype=center.dtype)
    ln_id = add(LayerKind.LAYERNORM.value, {"gamma": gamma, "beta": beta, "variant": "no_var"}, layers[join_id].out_vars, alloc(D), [join_id])
    gelu_id = add(LayerKind.GELU.value, {}, layers[ln_id].out_vars, alloc(D), [ln_id])

    out_spec = OutputSpec(kind="LINEAR_LE", c=torch.ones(D, dtype=center.dtype), d=torch.tensor(assert_d, dtype=center.dtype))
    enc = out_spec.encode_linear(B=B, n_out=D, device=center.device, dtype=center.dtype)
    add(LayerKind.ASSERT.value, enc, layers[gelu_id].out_vars, layers[gelu_id].out_vars, [gelu_id])

    net = Net(layers=layers, preds=preds, succs=succs)
    info: "dict[str, Any]" = {"Wq": Wq, "Wk": Wk, "Wv": Wv, "lb_in": lb_in, "ub_in": ub_in, "out_id": gelu_id, "L": L, "D": D}
    return net, info


def _make_dual_matmul_net(  # pragma: no cover
    B: int, I: int, K: int, J: int, center: torch.Tensor, eps: float, assert_d: float,
) -> "tuple[Net, dict[str, Any]]":
    """DENSE X [I,K] + DENSE Y [K,J] -> MATMUL -> SOFTMAX -> LAYERNORM -> GELU.

    The ONNX import lowers attention Q.Kt / probs.V to a generic var x var
    MATMUL, a distinct dual kernel (forward_matmul / backward_matmul) from the
    scalar ATT_SCORES/ATT_MIX cores. This net exercises that batched-bilinear
    path end to end through the DualSolver.
    """
    from act.back_end.core import Layer
    from act.front_end.specs import OutputSpec

    n_in = center.shape[1]
    in_v = list(range(n_in))
    lb_in, ub_in = center - eps, center + eps
    Wx = torch.randn(I * K, n_in, dtype=center.dtype, generator=torch.Generator().manual_seed(81)) * 0.2
    Wy = torch.randn(K * J, n_in, dtype=center.dtype, generator=torch.Generator().manual_seed(82)) * 0.2

    layers = [
        Layer(id=0, kind=LayerKind.INPUT.value, params={"shape": (B, n_in), "dtype": str(center.dtype)}, in_vars=[], out_vars=in_v),
        Layer(id=1, kind=LayerKind.INPUT_SPEC.value, params={"kind": "BOX", "lb": lb_in, "ub": ub_in}, in_vars=in_v, out_vars=in_v),
    ]
    x_vars = list(range(n_in, n_in + I * K))
    y_vars = list(range(n_in + I * K, n_in + I * K + K * J))
    z_vars = list(range(n_in + I * K + K * J, n_in + I * K + K * J + I * J))

    def dense(W, out_vars, n_out):
        return {
            "weight": W, "in_features": n_in, "out_features": n_out,
            "weight_pos": W.clamp(min=0), "weight_neg": W.clamp(max=0),
            "bias": torch.zeros(n_out, dtype=center.dtype), "input_shape": (n_in,),
        }

    layers.append(Layer(id=2, kind=LayerKind.DENSE.value, params=dense(Wx, x_vars, I * K), in_vars=in_v, out_vars=x_vars))
    layers.append(Layer(id=3, kind=LayerKind.DENSE.value, params=dense(Wy, y_vars, K * J), in_vars=in_v, out_vars=y_vars))
    layers.append(Layer(id=4, kind=LayerKind.MATMUL.value, params={"x_vars": x_vars, "y_vars": y_vars, "x_shape": (I, K), "y_shape": (K, J)}, in_vars=x_vars + y_vars, out_vars=z_vars))
    sm_vars = list(range(z_vars[-1] + 1, z_vars[-1] + 1 + I * J))
    layers.append(Layer(id=5, kind=LayerKind.SOFTMAX.value, params={"axis": -1}, in_vars=z_vars, out_vars=sm_vars))
    gamma = torch.ones(I * J, dtype=center.dtype)
    beta = torch.zeros(I * J, dtype=center.dtype)
    ln_vars = list(range(sm_vars[-1] + 1, sm_vars[-1] + 1 + I * J))
    layers.append(Layer(id=6, kind=LayerKind.LAYERNORM.value, params={"gamma": gamma, "beta": beta, "variant": "no_var"}, in_vars=sm_vars, out_vars=ln_vars))
    gelu_vars = list(range(ln_vars[-1] + 1, ln_vars[-1] + 1 + I * J))
    layers.append(Layer(id=7, kind=LayerKind.GELU.value, params={}, in_vars=ln_vars, out_vars=gelu_vars))
    out_spec = OutputSpec(kind="LINEAR_LE", c=torch.ones(I * J, dtype=center.dtype), d=torch.tensor(assert_d, dtype=center.dtype))
    enc = out_spec.encode_linear(B=B, n_out=I * J, device=center.device, dtype=center.dtype)
    layers.append(Layer(id=8, kind=LayerKind.ASSERT.value, params=enc, in_vars=gelu_vars, out_vars=gelu_vars))
    preds = {0: [], 1: [0], 2: [1], 3: [1], 4: [2, 3], 5: [4], 6: [5], 7: [6], 8: [7]}
    succs = {0: [1], 1: [2, 3], 2: [4], 3: [4], 4: [5], 5: [6], 6: [7], 7: [8], 8: []}
    net = Net(layers=layers, preds=preds, succs=succs)
    info: "dict[str, Any]" = {"Wx": Wx, "Wy": Wy, "lb_in": lb_in, "ub_in": ub_in, "z_id": 4, "I": I, "K": K, "J": J}
    return net, info


def _dual_forward_box(net, lb_in, ub_in, layer_id):  # pragma: no cover
    """Run the dual forward pass and return the (lb, ub) box at ``layer_id``."""
    from act.back_end.dual_tf.tf_forward import compute_forward_bounds

    bounds_dict = compute_forward_bounds(net, lb_in.clone(), ub_in.clone(), post_activation=False)
    box = bounds_dict[layer_id]
    return box.lb, box.ub


def _test_dual_transformer_att_cores() -> None:  # pragma: no cover
    # Dual attention scalar cores end to end: the dual FORWARD pass
    # (forward_attention/softmax/layernorm/gelu) box must bracket the concrete
    # attention output, and the dual BACKWARD pass (DualSolver.evaluate_spec)
    # must CERTIFY a loose bound yet NOT certify a bound below the true range
    # (proving the certified bound is used, not vacuous).
    from act.back_end.transfer_functions import set_solver_mode, get_solver_mode
    from act.util.device_manager import get_default_dtype
    from act.util.stats import VerifyStatus

    dtype = get_default_dtype()
    B, L, D = 1, 2, 2
    torch.manual_seed(90)
    center = torch.randn(B, L * D, dtype=dtype) * 0.05
    eps = 0.02
    net, info = _make_dual_att_cores_net(B, L, D, center, eps, assert_d=100.0)
    Wq, Wk, Wv = info["Wq"], info["Wk"], info["Wv"]
    l_box, u_box = info["lb_in"], info["ub_in"]

    def concrete_gelu_out(x: torch.Tensor) -> torch.Tensor:
        x3 = x.reshape(B, L, D)
        q0 = x3[:, 0, :] @ Wq.t()
        scores = torch.cat([(q0 * (x3[:, kp, :] @ Wk.t())).sum(-1, keepdim=True) / (D ** 0.5) for kp in range(L)], dim=-1)
        probs = torch.softmax(scores, dim=-1)
        v = torch.stack([x3[:, p, :] @ Wv.t() for p in range(L)], dim=1)
        ctx = torch.cat([(probs * v[:, :, f]).sum(-1, keepdim=True) for f in range(D)], dim=-1)
        normed = ctx - ctx.mean(dim=-1, keepdim=True)
        return torch.nn.functional.gelu(normed)

    lb, ub = _dual_forward_box(net, l_box, u_box, info["out_id"])
    assert torch.isfinite(lb).all() and torch.isfinite(ub).all(), "dual att-cores forward box must be finite"
    assert (lb <= ub + 1e-9).all(), "dual att-cores forward box lb must not exceed ub"
    concrete_sum_max = float(concrete_gelu_out(l_box).sum(-1).item())
    for _ in range(120):
        x = l_box + torch.rand_like(l_box) * (u_box - l_box)
        concrete_sum_max = max(concrete_sum_max, float(concrete_gelu_out(x).sum(-1).item()))

    prev = get_solver_mode()
    try:
        set_solver_mode("dual")
        loose = verify_once(net)
        assert loose[0].status == VerifyStatus.CERTIFIED, f"dual att-cores: loose bound expected CERTIFIED, got {loose[0].status}"
        assert concrete_sum_max <= 100.0 + 1e-6, (
            f"dual att-cores: certified d=100 contradicted by concrete sum {concrete_sum_max}"
        )
        net_tight, _ = _make_dual_att_cores_net(B, L, D, center, eps, assert_d=concrete_sum_max - 1.0)
        tight = verify_once(net_tight)
        assert tight[0].status != VerifyStatus.CERTIFIED, (
            f"dual att-cores: threshold below range must NOT certify, got {tight[0].status}"
        )
    finally:
        set_solver_mode(prev)


def _test_dual_transformer_matmul() -> None:  # pragma: no cover
    # Dual batched-bilinear MATMUL core (the ONNX attention lowering) end to
    # end: forward_matmul box brackets the concrete X@Y (through softmax/
    # layernorm/gelu), and backward_matmul via DualSolver certifies a loose
    # bound but not a below-range one.
    from act.back_end.transfer_functions import set_solver_mode, get_solver_mode
    from act.util.device_manager import get_default_dtype
    from act.util.stats import VerifyStatus

    dtype = get_default_dtype()
    B, I, K, J = 1, 2, 2, 2
    torch.manual_seed(91)
    center = torch.randn(B, 3, dtype=dtype) * 0.05
    eps = 0.02
    net, info = _make_dual_matmul_net(B, I, K, J, center, eps, assert_d=100.0)
    Wx, Wy = info["Wx"], info["Wy"]
    l_box, u_box = info["lb_in"], info["ub_in"]

    def concrete_matmul_z(x: torch.Tensor) -> torch.Tensor:
        X = (x @ Wx.t()).reshape(B, I, K)
        Y = (x @ Wy.t()).reshape(B, K, J)
        return (X @ Y).reshape(B, I * J)

    lb, ub = _dual_forward_box(net, l_box, u_box, info["z_id"])
    n_samples = 120
    true_min = concrete_matmul_z(l_box).clone()
    true_max = true_min.clone()
    for _ in range(n_samples):
        x = l_box + torch.rand_like(l_box) * (u_box - l_box)
        z = concrete_matmul_z(x)
        true_min = torch.minimum(true_min, z)
        true_max = torch.maximum(true_max, z)
    # MATMUL forward box is the sound four-corner McCormick envelope; it must
    # bracket the concrete X@Y (the layernorm/gelu that follow are checked via
    # the end-to-end certified bound below, not this pre-softmax box).
    assert (lb <= true_min + 1e-6).all(), "dual MATMUL forward: unsound lower bound"
    assert (ub >= true_max - 1e-6).all(), "dual MATMUL forward: unsound upper bound"

    prev = get_solver_mode()
    try:
        set_solver_mode("dual")
        loose = verify_once(net)
        assert loose[0].status == VerifyStatus.CERTIFIED, f"dual MATMUL: loose expected CERTIFIED, got {loose[0].status}"
        net_tight, info_t = _make_dual_matmul_net(B, I, K, J, center, eps, assert_d=-50.0)
        tight = verify_once(net_tight)
        assert tight[0].status != VerifyStatus.CERTIFIED, (
            f"dual MATMUL: threshold below range must NOT certify, got {tight[0].status}"
        )
    finally:
        set_solver_mode(prev)


def _test_dual_lp_embedding_finite_p() -> None:  # pragma: no cover
    # Finite-p LP_EMBEDDING input spec (p_norm=2) verified through the dual
    # solver: exercises seed_from_input_specs' LP_EMBEDDING center/eps/
    # perturbed_positions seeding AND solver_dual's exact per-word Lp-ball
    # dual-norm input contribution (_resolve_perturbation_norm ->
    # _dual_norm_exponent -> _dual_norm_contribution / _perturbed_block_slices),
    # the finite-p path box/L-inf specs never reach.
    from act.back_end.core import Layer
    from act.back_end.transfer_functions import set_solver_mode, get_solver_mode
    from act.front_end.specs import OutputSpec, InKind
    from act.util.device_manager import get_default_dtype
    from act.util.stats import VerifyStatus

    dtype = get_default_dtype()
    B, L, D = 1, 2, 2
    n_in = L * D
    torch.manual_seed(97)
    center3 = torch.randn(B, L, D, dtype=dtype) * 0.1
    eps = 0.05
    in_v = list(range(n_in))
    d_v = list(range(n_in, n_in + 2))
    W = torch.randn(2, n_in, dtype=dtype) * 0.2

    def build(assert_d: float) -> Net:
        layers = [
            Layer(id=0, kind=LayerKind.INPUT.value, params={"shape": (B, L, D), "dtype": str(dtype)}, in_vars=[], out_vars=in_v),
            Layer(id=1, kind=LayerKind.INPUT_SPEC.value, params={"kind": InKind.LP_EMBEDDING, "center": center3, "eps": torch.tensor([eps], dtype=dtype), "p_norm": 2.0, "perturbed_positions": torch.tensor([0])}, in_vars=in_v, out_vars=in_v),
            Layer(id=2, kind=LayerKind.DENSE.value, params={"weight": W, "in_features": n_in, "out_features": 2, "weight_pos": W.clamp(min=0), "weight_neg": W.clamp(max=0), "bias": torch.zeros(2, dtype=dtype), "input_shape": (n_in,)}, in_vars=in_v, out_vars=d_v),
        ]
        enc = OutputSpec(kind="LINEAR_LE", c=torch.ones(2, dtype=dtype), d=torch.tensor(assert_d, dtype=dtype)).encode_linear(B=B, n_out=2, device=torch.device("cpu"), dtype=dtype)
        layers.append(Layer(id=3, kind=LayerKind.ASSERT.value, params=enc, in_vars=d_v, out_vars=d_v))
        return Net(layers=layers, preds={0: [], 1: [0], 2: [1], 3: [2]}, succs={0: [1], 1: [2], 2: [3], 3: []})

    prev = get_solver_mode()
    try:
        set_solver_mode("dual")
        loose = verify_once(build(100.0))
        assert loose[0].status == VerifyStatus.CERTIFIED, f"dual LP_EMBEDDING: loose expected CERTIFIED, got {loose[0].status}"
        tight = verify_once(build(-100.0))
        assert tight[0].status != VerifyStatus.CERTIFIED, (
            f"dual LP_EMBEDDING: threshold below range must NOT certify, got {tight[0].status}"
        )
    finally:
        set_solver_mode(prev)


def _test_dual_smooth_activations() -> None:  # pragma: no cover
    # Dual backward for the new smooth activations (ERF/SQRT/SIN/COS/QUANTIZE
    # in dual_tf/tf_smooth.py): a DENSE -> activation -> ASSERT net run through
    # the DualSolver must CERTIFY a loose bound, exercising each activation's
    # forward relaxation + backward routing.
    from act.back_end.core import Layer
    from act.back_end.transfer_functions import set_solver_mode, get_solver_mode
    from act.util.device_manager import get_default_dtype
    from act.util.stats import VerifyStatus

    dtype = get_default_dtype()
    B, n = 1, 3

    def build(act_kind: str, act_params: "dict[str, Any]") -> Net:
        center = torch.full((B, n), 0.7, dtype=dtype)
        lb_in, ub_in = center - 0.05, center + 0.05
        in_v = list(range(n))
        d_v = list(range(n, 2 * n))
        o_v = list(range(2 * n, 3 * n))
        W = torch.eye(n, dtype=dtype)
        layers = [
            Layer(id=0, kind=LayerKind.INPUT.value, params={"shape": (B, n), "dtype": str(dtype)}, in_vars=[], out_vars=in_v),
            Layer(id=1, kind=LayerKind.INPUT_SPEC.value, params={"kind": "BOX", "lb": lb_in, "ub": ub_in}, in_vars=in_v, out_vars=in_v),
            Layer(id=2, kind=LayerKind.DENSE.value, params={"weight": W, "in_features": n, "out_features": n, "weight_pos": W, "weight_neg": W * 0, "bias": torch.zeros(n, dtype=dtype), "input_shape": (n,)}, in_vars=in_v, out_vars=d_v),
            Layer(id=3, kind=act_kind, params=act_params, in_vars=d_v, out_vars=o_v),
        ]
        from act.front_end.specs import OutputSpec
        enc = OutputSpec(kind="LINEAR_LE", c=torch.ones(n, dtype=dtype), d=torch.tensor(100.0, dtype=dtype)).encode_linear(B=B, n_out=n, device=torch.device("cpu"), dtype=dtype)
        layers.append(Layer(id=4, kind=LayerKind.ASSERT.value, params=enc, in_vars=o_v, out_vars=o_v))
        return Net(layers=layers, preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3]}, succs={0: [1], 1: [2], 2: [3], 3: [4], 4: []})

    cases = [
        (LayerKind.ERF.value, {}),
        (LayerKind.SQRT.value, {}),
        (LayerKind.SIN.value, {}),
        (LayerKind.COS.value, {}),
        (LayerKind.QUANTIZE.value, {"scale": torch.tensor([0.1], dtype=dtype), "zero_point": torch.tensor([0.0], dtype=dtype), "qmin": -128, "qmax": 127}),
    ]
    prev = get_solver_mode()
    try:
        set_solver_mode("dual")
        for kind, params in cases:
            r = verify_once(build(kind, params))
            assert r[0].status == VerifyStatus.CERTIFIED, f"dual {kind}: expected CERTIFIED, got {r[0].status}"
    finally:
        set_solver_mode(prev)


def _test_dual_mha_split_join_not_implemented() -> None:  # pragma: no cover
    # The dual path deliberately stubs the MHA split/join reshape family
    # (only the ATT_SCORES/ATT_MIX scalar cores + MATMUL are relaxed); the
    # stubs must raise NotImplementedError so a mis-lowered net fails loudly.
    from act.back_end.core import Layer
    from act.back_end.dual_tf.tf_transformer import forward_mha, backward_mha

    dummy = Layer(id=0, kind=LayerKind.MHA_SPLIT.value, params={}, in_vars=[0], out_vars=[0])
    raised_fwd = False
    try:
        forward_mha(dummy, [], [], [], [], False, torch.device("cpu"), torch.get_default_dtype())
    except NotImplementedError:
        raised_fwd = True
    assert raised_fwd, "forward_mha must raise NotImplementedError"
    raised_bwd = False
    try:
        backward_mha(dummy, torch.zeros(1, 1), {}, [])
    except NotImplementedError:
        raised_bwd = True
    assert raised_bwd, "backward_mha must raise NotImplementedError"








def _test_act2torch_smooth_activation_reconstruction() -> None:  # pragma: no cover
    from act.back_end.core import Layer
    from act.pipeline.verification.act2torch import ACTToTorch
    from act.front_end.specs import OutputSpec, OutKind
    from act.util.device_manager import get_default_dtype

    dtype = get_default_dtype()
    B, n = 1, 2
    in_v = list(range(n))
    layer_vars = [list(range((i + 1) * n, (i + 2) * n)) for i in range(5)]
    x = torch.tensor([[0.64, 0.81]], dtype=dtype)
    eps = torch.tensor([[0.01, 0.01]], dtype=dtype)
    encoded = OutputSpec(
        kind=OutKind.LINEAR_LE,
        c=torch.ones(n, dtype=dtype),
        d=torch.tensor(10.0, dtype=dtype),
    ).encode_linear(B=B, n_out=n, device=torch.device("cpu"), dtype=dtype)
    layers = [
        Layer(id=0, kind=LayerKind.INPUT.value, params={"shape": (B, n), "dtype": str(dtype)}, in_vars=[], out_vars=in_v),
        Layer(id=1, kind=LayerKind.INPUT_SPEC.value, params={"kind": InKind.BOX, "lb": x - eps, "ub": x + eps}, in_vars=in_v, out_vars=in_v),
        Layer(id=2, kind=LayerKind.ERF.value, params={}, in_vars=in_v, out_vars=layer_vars[0]),
        Layer(id=3, kind=LayerKind.SQRT.value, params={}, in_vars=layer_vars[0], out_vars=layer_vars[1]),
        Layer(id=4, kind=LayerKind.SIN.value, params={}, in_vars=layer_vars[1], out_vars=layer_vars[2]),
        Layer(id=5, kind=LayerKind.COS.value, params={}, in_vars=layer_vars[2], out_vars=layer_vars[3]),
        Layer(
            id=6,
            kind=LayerKind.QUANTIZE.value,
            params={"scale": torch.tensor([0.05], dtype=dtype), "zero_point": torch.tensor([0.0], dtype=dtype), "qmin": -128, "qmax": 127},
            in_vars=layer_vars[3],
            out_vars=layer_vars[4],
        ),
        Layer(id=7, kind=LayerKind.ASSERT.value, params=encoded, in_vars=layer_vars[4], out_vars=layer_vars[4]),
    ]
    net = Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4], 6: [5], 7: [6]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: [6], 6: [7], 7: []},
    )

    restored = ACTToTorch(net).run()
    y = restored(x)["output"]
    expected = torch.erf(x)
    expected = torch.sqrt(torch.clamp(expected, min=0.0))
    expected = torch.sin(expected)
    expected = torch.cos(expected)
    expected = 0.05 * torch.clamp(torch.round(expected / 0.05), min=-128.0, max=127.0)
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6), (
        f"smooth ACTToTorch reconstruction mismatch: got={y.tolist()} want={expected.tolist()}"
    )


def _test_torch2act_minimal_vit_fixture_soundness() -> None:  # pragma: no cover
    import torch.nn as nn
    from act.back_end.analyze import analyze
    from act.back_end.core import Fact, ConSet
    from act.front_end.spec_creator_base import LabeledInputTensor
    from act.front_end.specs import InputSpec, OutputSpec, OutKind
    from act.front_end.verifiable_model import InputLayer, InputSpecLayer, OutputSpecLayer, VerifiableModel
    from act.pipeline.verification.torch2act import TorchToACT
    from act.util.device_manager import get_default_dtype

    dtype = get_default_dtype()

    class TinyRegressionBertLayerNorm(nn.LayerNorm):

        def __init__(self, hidden_size: int) -> None:
            super().__init__(hidden_size, eps=1e-5)
            self.variance_epsilon = self.eps


    class TinyRegressionBertSelfAttention(nn.Module):

        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.num_attention_heads = 1
            self.attention_head_size = hidden_size
            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            query_layer = self.query(hidden_states)
            key_layer = self.key(hidden_states)
            value_layer = self.value(hidden_states)
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / (self.attention_head_size ** 0.5)
            attention_probs = torch.softmax(attention_scores, dim=-1)
            return torch.matmul(attention_probs, value_layer)


    class TinyRegressionBertSelfOutput(nn.Module):

        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.dense = nn.Linear(hidden_size, hidden_size)
            self.LayerNorm = TinyRegressionBertLayerNorm(hidden_size)

        def forward(
            self,
            hidden_states: torch.Tensor,
            input_tensor: torch.Tensor,
        ) -> torch.Tensor:
            return self.LayerNorm(self.dense(hidden_states) + input_tensor)


    class TinyRegressionBertAttention(nn.Module):

        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.self = TinyRegressionBertSelfAttention(hidden_size)
            self.output = TinyRegressionBertSelfOutput(hidden_size)

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            return self.output(self.self(input_tensor), input_tensor)


    class TinyRegressionBertIntermediate(nn.Module):

        def __init__(self, hidden_size: int, intermediate_size: int) -> None:
            super().__init__()
            self.dense = nn.Linear(hidden_size, intermediate_size)
            self.intermediate_act_fn = nn.GELU()

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.intermediate_act_fn(self.dense(hidden_states))


    class TinyRegressionBertOutput(nn.Module):

        def __init__(self, hidden_size: int, intermediate_size: int) -> None:
            super().__init__()
            self.dense = nn.Linear(intermediate_size, hidden_size)
            self.LayerNorm = TinyRegressionBertLayerNorm(hidden_size)

        def forward(
            self,
            hidden_states: torch.Tensor,
            input_tensor: torch.Tensor,
        ) -> torch.Tensor:
            return self.LayerNorm(self.dense(hidden_states) + input_tensor)


    class TinyRegressionBertLayer(nn.Module):

        def __init__(self, hidden_size: int, intermediate_size: int) -> None:
            super().__init__()
            self.attention = TinyRegressionBertAttention(hidden_size)
            self.intermediate = TinyRegressionBertIntermediate(
                hidden_size,
                intermediate_size,
            )
            self.output = TinyRegressionBertOutput(hidden_size, intermediate_size)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            attention_output = self.attention(hidden_states)
            intermediate_output = self.intermediate(attention_output)
            return self.output(intermediate_output, attention_output)

    class PatchEmbed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Conv2d(1, 2, kernel_size=2, stride=2, bias=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    class TinyDuckViT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embed = PatchEmbed()
            self.block = TinyRegressionBertLayer(hidden_size=2, intermediate_size=4)
            self.norm = TinyRegressionBertLayerNorm(2)
            self.head = nn.Linear(2, 2)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 2, dtype=dtype))
            self.pos_embed = nn.Parameter(torch.zeros(1, 2, 2, dtype=dtype))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            patch = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            hidden = torch.cat([cls, patch], dim=1) + self.pos_embed
            hidden = self.block(hidden)
            hidden = self.norm(hidden)
            return self.head(hidden[:, 0, :])

    torch.manual_seed(23)
    body = TinyDuckViT().to(dtype=dtype).eval()
    center = torch.tensor([[[[0.1, -0.2], [0.3, 0.4]]]], dtype=dtype)
    eps = torch.full_like(center, 1e-4)
    wrapped = VerifiableModel(
        input_layer=InputLayer(
            labeled_input=LabeledInputTensor(tensor=center, label=None),
            shape=tuple(center.shape),
            dtype=dtype,
        ),
        input_spec=InputSpecLayer(InputSpec(kind=InKind.BOX, lb=center - eps, ub=center + eps)),
        model=body,
        output_spec=OutputSpecLayer(
            OutputSpec(kind=OutKind.LINEAR_LE, c=torch.ones(2, dtype=dtype), d=torch.tensor(100.0, dtype=dtype))
        ),
    ).eval()

    net = TorchToACT(wrapped).run()
    kinds = [layer.kind for layer in net.layers]
    assert LayerKind.CONV2D.value in kinds, "ViT fixture must emit patch Conv2d"
    assert kinds.count(LayerKind.CONSTANT.value) >= 2, "ViT fixture must emit cls/pos constants"
    assert LayerKind.ATT_SCORES.value in kinds and LayerKind.ATT_MIX.value in kinds, (
        "ViT fixture must lower the block through attention layers"
    )

    entry_id = find_entry_layer_id(net)
    seed = seed_from_input_specs(gather_input_spec_layers(net))
    entry_fact = Fact(bounds=seed, cons=ConSet())
    add_all_input_specs(entry_fact.cons, get_input_ids(net), gather_input_spec_layers(net))
    _before, after, _global_c = analyze(net, entry_id, entry_fact)
    conv_layer = next(layer for layer in net.layers if layer.kind == LayerKind.CONV2D.value)
    conv_bounds = after[conv_layer.id].bounds
    concrete_patch = body.patch_embed.proj(center).reshape(1, -1)
    assert torch.isfinite(conv_bounds.lb).all() and torch.isfinite(conv_bounds.ub).all(), (
        "ViT fixture patch Conv2d bounds must be finite"
    )
    assert (conv_bounds.lb <= concrete_patch + 1e-6).all(), "ViT patch lower bound must cover concrete output"
    assert (conv_bounds.ub >= concrete_patch - 1e-6).all(), "ViT patch upper bound must cover concrete output"
    results = verify_once(net)
    assert len(results) == 1 and results[0].status in {
        VerifyStatus.CERTIFIED, VerifyStatus.FALSIFIED, VerifyStatus.UNKNOWN
    }, f"ViT fixture verify_once returned unexpected result {results}"






def _test_verify_once_b3_all_certified() -> None:  # pragma: no cover
    # Zero DENSE -> abstract output is singleton {0}, well below d=10.
    # End-to-end check that the [B*M, n_out] cert pass folds to per-sample.
    from act.util.device_manager import get_default_device, get_default_dtype
    from act.util.stats import VerifyStatus

    device = get_default_device()
    dtype = get_default_dtype()

    B, n_in, n_out = 3, 4, 2
    W = torch.zeros(n_out, n_in, device=device, dtype=dtype)
    b = torch.zeros(n_out, device=device, dtype=dtype)
    lb_in = torch.full((B, n_in), -1.0, device=device, dtype=dtype)
    ub_in = torch.full((B, n_in), 1.0, device=device, dtype=dtype)

    net = _make_dense_net_box_test(
        B=B, n_in=n_in, n_out=n_out, weight=W, bias=b,
        lb_in=lb_in, ub_in=ub_in,
        assert_params={
            "kind": "LINEAR_LE",
            "c": torch.tensor([1.0, 1.0], device=device, dtype=dtype),
            "d": 10.0,
        },
    )

    results = verify_once(net)
    assert len(results) == B, f"expected {B} results, got {len(results)}"
    for i, r in enumerate(results):
        assert r.status == VerifyStatus.CERTIFIED, (
            f"sample {i}: expected CERTIFIED, got {r.status}"
        )


def _test_verify_once_b8_mixed_outcomes() -> None:  # pragma: no cover
    # 8 input boxes designed to produce CERT/FALS/UNK mix in one run,
    # proving the cert pass + concrete falsification operate sample-wise
    # rather than collapsing the batch.
    from act.util.device_manager import get_default_device, get_default_dtype
    from act.util.stats import VerifyStatus

    device = get_default_device()
    dtype = get_default_dtype()

    B, n_in, n_out = 8, 2, 2
    W = torch.eye(n_out, device=device, dtype=dtype)
    b = torch.zeros(n_out, device=device, dtype=dtype)
    lb_in = torch.tensor(
        [
            [2.0, -2.0],
            [1.0, -2.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
            [-2.0, -1.0],
            [1.0, -1.0],
            [-1.0, 0.0],
        ],
        device=device, dtype=dtype,
    )
    ub_in = torch.tensor(
        [
            [3.0, -1.0],
            [2.0, -1.5],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 0.5],
            [2.0, 0.5],
            [2.0, 0.0],
            [1.0, 1.0],
        ],
        device=device, dtype=dtype,
    )
    net = _make_dense_net_box_test(
        B=B, n_in=n_in, n_out=n_out, weight=W, bias=b,
        lb_in=lb_in, ub_in=ub_in,
        assert_params={
            "kind": "TOP1_ROBUST",
            "y_true": torch.zeros(B, dtype=torch.long, device=device),
        },
    )

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        return x

    results = verify_once(net, model_fn=model_fn)
    assert len(results) == B, f"expected {B} results, got {len(results)}"

    valid = {
        VerifyStatus.CERTIFIED, VerifyStatus.FALSIFIED, VerifyStatus.UNKNOWN,
    }
    statuses = [r.status for r in results]
    assert all(s in valid for s in statuses), (
        f"unexpected status enum value in {statuses}"
    )
    assert any(s == VerifyStatus.CERTIFIED for s in statuses), (
        f"no CERTIFIED lane in {statuses}"
    )
    assert any(s == VerifyStatus.FALSIFIED for s in statuses), (
        f"no FALSIFIED lane in {statuses}"
    )
    assert any(s == VerifyStatus.UNKNOWN for s in statuses), (
        f"no UNKNOWN lane in {statuses}"
    )

_TESTS = [  # pragma: no cover
    _test_verify_once_b3_all_certified,
    _test_verify_once_b8_mixed_outcomes,
    _test_att_scores_dual_planar_analyze_soundness,
    _test_att_scores_dual_planar_verify_once_certified,
    _test_att_scores_dual_planar_lp_export_solve,
    _test_att_scores_dual_planar_masked_and_clamp_alpha_soundness,
    _test_mini_transformer_block_analyze_soundness,
    _test_mha_split_edge_cases_and_mask_add,
    _test_new_elementwise_tf_soundness,
    _test_dual_transformer_att_cores,
    _test_dual_transformer_matmul,
    _test_dual_lp_embedding_finite_p,
    _test_dual_smooth_activations,
    _test_dual_mha_split_join_not_implemented,
    _test_act2torch_smooth_activation_reconstruction,
    _test_torch2act_minimal_vit_fixture_soundness,
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
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


def main() -> int:
    # Pin device/dtype to CPU/float64 so hosts where CUDA is visible but
    # no kernel matches the runtime's compute capability don't raise on
    # the default GPU init path in act.util.device_manager.
    from act.util.device_manager import initialize_device

    initialize_device("cpu", "float64")
    print("Running verifier self-tests (act.back_end.verifier)\n")
    return run_all_tests()


if __name__ == "__main__":
    import sys

    sys.exit(main())
