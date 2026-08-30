#!/usr/bin/env python3
"""Bounded F-prime persistent low-rank update sentinel.

This disconnected scratch program starts from the frozen screened composed
base LP already loaded in the one F-prime ``highspy.Highs`` owner.  It applies
the complete frozen selector set once by appending signed ReLU-correction
columns and topologically triangular native RANGE definitions.  It never
forms the updated ``target_pre``, ``target_output``, triangularly eliminated
updated matrix, or a second solver model on the proposed path.

The LP, row marginals, dual ray, correction variables, and all phase rows are
candidate-only.  A positive candidate is accepted only by the unchanged raw
BOX, verifier-owned zero-width interval, and stored-binary64 Fraction
terminal.  There is no sampling, ONNX input-point execution, PGD, BaB/split,
backward bound, dual tightening, retry, menu, or second solver.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import highspy
import numpy as np
import scipy.sparse as sp
import torch


ROOT = Path(__file__).resolve().parent
FPRIME_PATH = ROOT / "scratch_phase_projection_fprime_single_owner_probe.py"
FPRIME_SHA256 = "6ce6ab6b208a1224ce70cfc0e4f183f5acf07fbc6d9db65ef34a8f4694602a15"
SMALL = 1.0e-12
LARGE = 1.0e15
INFINITE_BOUND = 1.0e20
LP_TIME_LIMIT = 30.0

CASES = (
    "cifar100_large_iid153",
    "tinyimagenet_medium_iid153",
    "tinyimagenet_medium_iid143",
)


class SentinelStop(RuntimeError):
    def __init__(self, reason: str, **details: Any):
        super().__init__(reason)
        self.reason = reason
        self.details = details


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def import_fprime() -> Any:
    if sha256(FPRIME_PATH) != FPRIME_SHA256:
        raise SentinelStop("frozen F-prime source hash changed")
    spec = importlib.util.spec_from_file_location("persistent_lowrank_fprime", FPRIME_PATH)
    if spec is None or spec.loader is None:
        raise SentinelStop("could not import frozen F-prime source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def restrictions() -> Dict[str, Any]:
    return {
        "production_modified": False,
        "input_center_boundary_or_random_sampling_used": False,
        "onnx_input_point_execution_used": False,
        "pgd_used": False,
        "bab_split_or_enumeration_used": False,
        "backward_bounds_used": False,
        "dual_tightening_used": False,
        "candidate_lp_marginal_or_ray_has_authority": False,
        "second_solver_used": False,
        "fallback_or_runtime_menu": False,
        "parameter_scan": False,
        "phase_updates_max": 1,
        "resolves_after_base_max": 1,
        "updated_full_target_materialized_on_proposed_path": False,
    }


def fraction_down(value: Fraction) -> float:
    rounded = float(value)
    if not math.isfinite(rounded):
        raise SentinelStop("outward lower conversion overflowed")
    if Fraction.from_float(rounded) > value:
        rounded = float(np.nextafter(rounded, -np.inf))
    if not math.isfinite(rounded) or Fraction.from_float(rounded) > value:
        raise SentinelStop("outward lower conversion failed")
    return rounded


def fraction_up(value: Fraction) -> float:
    rounded = float(value)
    if not math.isfinite(rounded):
        raise SentinelStop("outward upper conversion overflowed")
    if Fraction.from_float(rounded) < value:
        rounded = float(np.nextafter(rounded, np.inf))
    if not math.isfinite(rounded) or Fraction.from_float(rounded) < value:
        raise SentinelStop("outward upper conversion failed")
    return rounded


def _canonical_csr(matrix: sp.spmatrix, *, columns: Optional[int] = None) -> sp.csr_matrix:
    result = sp.csr_matrix(matrix, dtype=np.float64)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    result.indptr = np.ascontiguousarray(result.indptr, dtype=np.int32)
    result.indices = np.ascontiguousarray(result.indices, dtype=np.int32)
    result.data = np.ascontiguousarray(result.data, dtype=np.float64)
    if columns is not None and result.shape[1] != columns:
        raise SentinelStop("CSR column width mismatch")
    if (
        not result.has_sorted_indices
        or not result.has_canonical_format
        or np.any(result.data == 0.0)
        or not np.all(np.isfinite(result.data))
        or np.any(np.abs(result.data) >= LARGE)
    ):
        raise SentinelStop("CSR canonical/finiteness/threshold contract failed")
    return result


def filter_tiny_ranged_rows(
    matrix: sp.csr_matrix,
    row_lower: np.ndarray,
    row_upper: np.ndarray,
    col_lower: np.ndarray,
    col_upper: np.ndarray,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Drop unavoidable HiGHS-tiny terms with exact outward projection.

    For a logical RANGE row ``L <= a*v <= U`` and deleted contribution
    ``d in [m, M]``, the loaded outward row is
    ``L-M <= a_keep*v <= U-m``.  All sums use exact stored-binary64 dyadics
    and each finite result is rounded once in its outward direction.
    """

    logical = _canonical_csr(matrix)
    lower = np.ascontiguousarray(row_lower, dtype=np.float64).reshape(-1)
    upper = np.ascontiguousarray(row_upper, dtype=np.float64).reshape(-1)
    col_lower = np.ascontiguousarray(col_lower, dtype=np.float64).reshape(-1)
    col_upper = np.ascontiguousarray(col_upper, dtype=np.float64).reshape(-1)
    if (
        logical.shape != (lower.size, col_lower.size)
        or upper.shape != lower.shape
        or col_upper.shape != col_lower.shape
        or np.any(lower > upper)
        or not np.all(np.isfinite(col_lower))
        or not np.all(np.isfinite(col_upper))
        or np.any(col_lower > col_upper)
        or np.any(np.abs(col_lower) >= INFINITE_BOUND)
        or np.any(np.abs(col_upper) >= INFINITE_BOUND)
    ):
        raise SentinelStop("ranged tiny-filter shape/bound contract failed")
    finite_bounds = np.isfinite(lower) | np.isfinite(upper)
    if np.any(np.abs(lower[np.isfinite(lower)]) >= INFINITE_BOUND) or np.any(
        np.abs(upper[np.isfinite(upper)]) >= INFINITE_BOUND
    ):
        raise SentinelStop("logical row bound reaches HiGHS infinity threshold")

    tiny = (logical.data != 0.0) & (np.abs(logical.data) <= SMALL)
    loaded_lower = lower.copy()
    loaded_upper = upper.copy()
    deleted = int(np.count_nonzero(tiny))
    affected_rows = 0
    maximum_widening = 0.0
    if deleted:
        row_of = np.repeat(np.arange(logical.shape[0], dtype=np.int64), np.diff(logical.indptr))
        affected = np.unique(row_of[tiny])
        affected_rows = int(affected.size)
        for row in affected:
            start, stop = int(logical.indptr[row]), int(logical.indptr[row + 1])
            local = tiny[start:stop]
            exact_min = Fraction(0)
            exact_max = Fraction(0)
            for column, value in zip(logical.indices[start:stop][local], logical.data[start:stop][local]):
                coefficient = Fraction.from_float(float(value))
                lo = coefficient * Fraction.from_float(float(col_lower[column]))
                hi = coefficient * Fraction.from_float(float(col_upper[column]))
                exact_min += min(lo, hi)
                exact_max += max(lo, hi)
            if math.isfinite(lower[row]):
                old = float(lower[row])
                loaded_lower[row] = fraction_down(Fraction.from_float(old) - exact_max)
                maximum_widening = max(maximum_widening, old - float(loaded_lower[row]))
            if math.isfinite(upper[row]):
                old = float(upper[row])
                loaded_upper[row] = fraction_up(Fraction.from_float(old) - exact_min)
                maximum_widening = max(maximum_widening, float(loaded_upper[row]) - old)

        kept = ~tiny
        row_counts = np.bincount(row_of[kept], minlength=logical.shape[0]).astype(np.int64)
        indptr64 = np.empty(logical.shape[0] + 1, dtype=np.int64)
        indptr64[0] = 0
        np.cumsum(row_counts, out=indptr64[1:])
        if int(indptr64[-1]) > np.iinfo(np.int32).max:
            raise SentinelStop("filtered CSR exceeds int32 capacity")
        logical = sp.csr_matrix(
            (
                np.ascontiguousarray(logical.data[kept], dtype=np.float64),
                np.ascontiguousarray(logical.indices[kept], dtype=np.int32),
                np.ascontiguousarray(indptr64, dtype=np.int32),
            ),
            shape=logical.shape,
        )
        logical = _canonical_csr(logical)
    if np.any(loaded_lower > loaded_upper) or not np.all(
        np.isfinite(loaded_lower[np.isfinite(loaded_lower)])
    ) or not np.all(np.isfinite(loaded_upper[np.isfinite(loaded_upper)])):
        raise SentinelStop("outward ranged tiny filter produced invalid bounds")
    return logical, loaded_lower, loaded_upper, {
        "deleted_tiny_nnz": deleted,
        "affected_rows": affected_rows,
        "maximum_bound_widening": maximum_widening,
        "formula": "L'=down(L-sum(max(a*l,a*u))); U'=up(U-sum(min(a*l,a*u)))",
        "logical_rows_with_any_finite_bound": int(np.count_nonzero(finite_bounds)),
    }


def full_phase_state(c: Mapping[str, Any]) -> Tuple[Tuple[Tuple[int, int, int], ...], np.ndarray, np.ndarray, sp.csr_matrix]:
    mapping = []
    selected_parts = []
    center_parts = []
    for layer in c["order"]:
        layer_id = int(layer.id)
        original = c["original_frames"].get(layer_id)
        if original is None or not original.exact.size:
            continue
        rows = np.asarray(original.stream_rows, dtype=np.int64)
        selected = np.asarray(c["target_assign"][layer_id], dtype=bool)
        centers = np.asarray(c["target_pre_center"][layer_id], dtype=np.float64)
        if not rows.size == selected.size == centers.size:
            raise SentinelStop("full phase state width mismatch", layer_id=layer_id)
        mapping.extend((layer_id, int(position), int(row)) for position, row in enumerate(rows))
        selected_parts.append(selected)
        center_parts.append(centers)
    mapping_tuple = tuple(mapping)
    if len(mapping_tuple) != len({(a, row) for a, _p, row in mapping_tuple}):
        raise SentinelStop("duplicate physical phase row")
    selected = np.concatenate(selected_parts)
    centers = np.concatenate(center_parts)
    full_oriented = _canonical_csr(c["A"])
    if full_oriented.shape[0] != len(mapping_tuple):
        raise SentinelStop("captured full A disagrees with phase map")
    return mapping_tuple, selected, centers, full_oriented


def build_signed_delta(
    helper: Any,
    c: Mapping[str, Any],
    selected_rows: Sequence[Tuple[int, int, int]],
) -> Tuple[Dict[int, np.ndarray], np.ndarray, Dict[int, np.ndarray], float]:
    phase = helper.phase
    assignments = {key: np.asarray(value, dtype=bool).copy() for key, value in c["target_assign"].items()}
    for layer_id, position, _row in selected_rows:
        assignments[layer_id][position] = ~assignments[layer_id][position]

    changes = []
    for layer_id, position, row in selected_rows:
        base_active = bool(c["target_assign"][layer_id][position])
        changes.append((layer_id, row, base_active, not base_active))
    if len(changes) == 0 or len(changes) > 64:
        raise SentinelStop("sentinel requires one nonempty at-most-64 flip set")

    updated_frames = {}
    active_live = {}
    exact_device = {}
    changed_by_layer: Dict[int, list[Tuple[int, int]]] = {}
    for layer in c["order"]:
        layer_id = int(layer.id)
        original = c["original_frames"].get(layer_id)
        if original is None:
            continue
        frame = phase._fixed_frame(original, assignments[layer_id])
        updated_frames[layer_id] = frame
        active_live[layer_id] = torch.as_tensor(
            np.intersect1d(c["live_rows"][layer_id], frame.active, assume_unique=True),
            dtype=torch.int64,
            device="cuda",
        )
        exact_device[layer_id] = torch.as_tensor(
            original.stream_rows, dtype=torch.int64, device="cuda"
        )
        changed_by_layer[layer_id] = [
            (row, index)
            for index, (local_layer, row, _base, _target) in enumerate(changes)
            if local_layer == layer_id
        ]

    torch.cuda.synchronize()
    started = time.monotonic()
    width = len(changes)
    values: Dict[int, torch.Tensor] = {}
    pre: Dict[int, np.ndarray] = {}
    for layer in c["order"]:
        layer_id = int(layer.id)
        kind = phase._oh._kind(layer.kind)
        predecessors = tuple(int(v) for v in c["net"].preds.get(layer_id, []))
        if kind == "INPUT":
            values[layer_id] = torch.zeros((len(layer.out_vars), width), dtype=torch.float64, device="cuda")
        elif kind in {"INPUT_SPEC", "FLATTEN", "ASSERT"}:
            values[layer_id] = values[predecessors[0]]
        elif kind in {"CONV2D", "DENSE"}:
            selected_value = helper.live._ordered_csr_dense(
                c["device_matrices"][layer_id], values[predecessors[0]]
            )
            value = torch.zeros((len(layer.out_vars), width), dtype=torch.float64, device="cuda")
            if c["device_rows"][layer_id].numel():
                value[c["device_rows"][layer_id]] = selected_value
            values[layer_id] = value
        elif kind == "SCALE":
            parameter = torch.as_tensor(c["pointwise"][layer_id], dtype=torch.float64, device="cuda").reshape(-1, 1)
            values[layer_id] = values[predecessors[0]] * parameter
        elif kind == "BIAS":
            values[layer_id] = values[predecessors[0]]
        elif kind == "ADD":
            values[layer_id] = values[predecessors[0]] + values[predecessors[1]]
        elif kind == "RELU":
            source = values[predecessors[0]]
            pre[layer_id] = source[exact_device[layer_id]].detach().cpu().numpy()
            value = torch.zeros_like(source)
            active = active_live[layer_id]
            if active.numel():
                value[active] = source[active]
            for row, column in changed_by_layer[layer_id]:
                value[row] = 0.0
                value[row, column] = 1.0
            values[layer_id] = value
        else:
            raise SentinelStop("unsupported graph kind in delta pass", kind=kind)
    output = values[int(c["assert_layer"].id)].detach().cpu().numpy()
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    if output.shape != (int(c["output_width"]), width) or not np.all(np.isfinite(output)):
        raise SentinelStop("delta output malformed")
    for value in pre.values():
        if value.shape[1] != width or not np.all(np.isfinite(value)):
            raise SentinelStop("delta preactivation malformed")
    return pre, output, assignments, seconds


def recursive_aux_bounds(
    q: sp.csr_matrix,
    centers: np.ndarray,
    delta: np.ndarray,
    selected_ordinals: np.ndarray,
    base_selected: np.ndarray,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k = selected_ordinals.size
    signs = np.where(base_selected[selected_ordinals], -1.0, 1.0)
    earlier = (signs + 1.0) * 0.5
    lower = np.empty(k, dtype=np.float64)
    upper = np.empty(k, dtype=np.float64)
    for i, ordinal in enumerate(selected_ordinals):
        exact_low = Fraction.from_float(float(signs[i] * centers[ordinal]))
        exact_high = exact_low
        row = q.getrow(int(ordinal))
        for column, value in zip(row.indices, row.data):
            coefficient = Fraction.from_float(float(signs[i] * value))
            lo = coefficient * Fraction.from_float(float(x_lower[column]))
            hi = coefficient * Fraction.from_float(float(x_upper[column]))
            exact_low += min(lo, hi)
            exact_high += max(lo, hi)
        if earlier[i] != 0.0:
            for j in range(i):
                value = float(delta[ordinal, j])
                if value == 0.0:
                    continue
                coefficient = Fraction.from_float(value)
                lo = coefficient * Fraction.from_float(float(lower[j]))
                hi = coefficient * Fraction.from_float(float(upper[j]))
                exact_low += min(lo, hi)
                exact_high += max(lo, hi)
        lower[i] = fraction_down(exact_low)
        upper[i] = fraction_up(exact_high)
        if (
            not math.isfinite(lower[i])
            or not math.isfinite(upper[i])
            or lower[i] > upper[i]
            or abs(lower[i]) >= INFINITE_BOUND
            or abs(upper[i]) >= INFINITE_BOUND
        ):
            raise SentinelStop("finite recursive auxiliary bound failed", index=i)
    return lower, upper, signs, earlier


def definition_matrix(
    q: sp.csr_matrix,
    centers: np.ndarray,
    delta: np.ndarray,
    selected_ordinals: np.ndarray,
    signs: np.ndarray,
    earlier: np.ndarray,
    n_x: int,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    rows = []
    rhs = []
    for i, ordinal in enumerate(selected_ordinals):
        base = q.getrow(int(ordinal)).multiply(-float(signs[i]))
        prior = np.zeros((1, selected_ordinals.size), dtype=np.float64)
        if earlier[i] != 0.0 and i:
            prior[0, :i] = -delta[ordinal, :i]
        prior[0, i] = 1.0
        rows.append(sp.hstack((base, sp.csr_matrix(prior)), format="csr"))
        rhs.append(float(signs[i] * centers[ordinal]))
    return _canonical_csr(sp.vstack(rows, format="csr"), columns=n_x + selected_ordinals.size), np.asarray(rhs, dtype=np.float64)


def build_update_plan(
    fprime: Any,
    c: Mapping[str, Any],
    base: Any,
    selected_rows: Sequence[Tuple[int, int, int]],
    delta_pre: Mapping[int, np.ndarray],
    delta_output: np.ndarray,
) -> Dict[str, Any]:
    started = time.monotonic()
    mapping, base_selected, centers, base_oriented = full_phase_state(c)
    ordinal_of = {key: index for index, key in enumerate(mapping)}
    selected_ordinals = np.asarray([ordinal_of[tuple(row)] for row in selected_rows], dtype=np.int64)
    if not np.all(selected_ordinals[1:] > selected_ordinals[:-1]):
        raise SentinelStop("selected rows are not in strict topological order")
    keep = np.asarray(c["keep"], dtype=bool)
    if keep.shape != (len(mapping),) or not np.all(keep[selected_ordinals]):
        raise SentinelStop("selector contains a row absent from the loaded base model")
    screened_to_full = np.flatnonzero(keep)
    full_to_screened = np.full(len(mapping), -1, dtype=np.int64)
    full_to_screened[screened_to_full] = np.arange(screened_to_full.size)

    delta = np.concatenate(
        [
            delta_pre[int(layer.id)]
            for layer in c["order"]
            if int(layer.id) in c["original_frames"]
            and c["original_frames"][int(layer.id)].exact.size
        ],
        axis=0,
    )
    if delta.shape != (len(mapping), len(selected_rows)):
        raise SentinelStop("delta/full-row shape mismatch")
    for i, ordinal in enumerate(selected_ordinals):
        if np.any(delta[ordinal, i:] != 0.0):
            raise SentinelStop("delta violates topological triangularity", index=i)

    orientation = np.where(base_selected, -1.0, 1.0)
    q = _canonical_csr(base_oriented.multiply(orientation[:, None]))
    x_lower = np.asarray(c["factor_lower"], dtype=np.float64)
    x_upper = np.asarray(c["factor_upper"], dtype=np.float64)
    y_lower, y_upper, signs, earlier = recursive_aux_bounds(
        q, centers, delta, selected_ordinals, base_selected, x_lower, x_upper
    )
    n_x = x_lower.size
    k = len(selected_rows)
    col_lower = np.concatenate((x_lower, y_lower))
    col_upper = np.concatenate((x_upper, y_upper))

    oriented_delta = orientation[:, None] * delta
    combined = _canonical_csr(
        sp.hstack((base_oriented, sp.csr_matrix(oriented_delta)), format="csr"),
        columns=n_x + k,
    )
    logical_rhs = np.asarray(c["b"], dtype=np.float64)
    logical_lower = np.full(len(mapping), -np.inf, dtype=np.float64)
    logical_upper = logical_rhs.copy()
    logical_lower[selected_ordinals] = logical_rhs[selected_ordinals]
    logical_upper[selected_ordinals] = np.inf

    existing_logical = combined[keep].tocsr()
    existing_lower = logical_lower[keep]
    existing_upper = logical_upper[keep]
    existing_loaded, existing_loaded_lower, existing_loaded_upper, existing_tiny = filter_tiny_ranged_rows(
        existing_logical, existing_lower, existing_upper, col_lower, col_upper
    )
    existing_x = _canonical_csr(existing_loaded[:, :n_x], columns=n_x)
    if not (
        existing_x.shape == base.loaded_matrix.shape
        and existing_x.nnz == base.loaded_matrix.nnz
        and np.array_equal(existing_x.indptr, base.loaded_matrix.indptr)
        and np.array_equal(existing_x.indices, base.loaded_matrix.indices)
        and np.array_equal(existing_x.data, base.loaded_matrix.data)
    ):
        raise SentinelStop("persistent base x block changed under ranged update")
    existing_aux = _canonical_csr(existing_loaded[:, n_x:], columns=k)

    omitted = np.flatnonzero(~keep)
    missing = np.empty(0, dtype=np.int64)
    missing_loaded = sp.csr_matrix((0, n_x + k), dtype=np.float64)
    missing_lower = np.empty(0, dtype=np.float64)
    missing_upper = np.empty(0, dtype=np.float64)
    missing_tiny: Dict[str, Any] = {
        "deleted_tiny_nnz": 0,
        "affected_rows": 0,
        "maximum_bound_widening": 0.0,
        "formula": "not invoked",
    }
    if omitted.size:
        omitted_matrix = combined[omitted].tocsr()
        contribution = omitted_matrix.data * np.where(
            omitted_matrix.data >= 0.0,
            col_upper[omitted_matrix.indices],
            col_lower[omitted_matrix.indices],
        )
        upper_box = np.zeros(omitted.size, dtype=np.float64)
        nonempty = np.diff(omitted_matrix.indptr) > 0
        upper_box[nonempty] = np.add.reduceat(contribution, omitted_matrix.indptr[:-1][nonempty])
        if not np.all(np.isfinite(upper_box)):
            raise SentinelStop("augmented missing-row screen overflowed")
        missing = omitted[upper_box > logical_rhs[omitted]]
        if missing.size:
            missing_loaded, missing_lower, missing_upper, missing_tiny = filter_tiny_ranged_rows(
                combined[missing].tocsr(),
                logical_lower[missing],
                logical_upper[missing],
                col_lower,
                col_upper,
            )

    definitions, definition_rhs = definition_matrix(
        q, centers, delta, selected_ordinals, signs, earlier, n_x
    )
    definitions_loaded, definitions_lower, definitions_upper, definition_tiny = filter_tiny_ranged_rows(
        definitions, definition_rhs, definition_rhs, col_lower, col_upper
    )

    objective_aux = np.asarray(c["C"][[c["rival"]]] @ delta_output, dtype=np.float64).reshape(-1)
    objective_nonzero = objective_aux != 0.0
    objective_tiny = objective_nonzero & (np.abs(objective_aux) <= SMALL)
    if np.any(objective_tiny) or not np.all(np.isfinite(objective_aux)) or np.any(
        np.abs(objective_aux) >= INFINITE_BOUND
    ):
        raise SentinelStop(
            "auxiliary objective is nonfinite/tiny/out-of-range",
            tiny_count=int(np.count_nonzero(objective_tiny)),
        )

    appended = _canonical_csr(
        sp.vstack((missing_loaded, definitions_loaded), format="csr"),
        columns=n_x + k,
    )
    appended_lower = np.concatenate((missing_lower, definitions_lower))
    appended_upper = np.concatenate((missing_upper, definitions_upper))
    return {
        "mapping": mapping,
        "selected_ordinals": selected_ordinals,
        "selected_solver_rows": full_to_screened[selected_ordinals].astype(np.int32),
        "existing_aux": existing_aux,
        "existing_lower": existing_loaded_lower,
        "existing_upper": existing_loaded_upper,
        "appended": appended,
        "appended_lower": appended_lower,
        "appended_upper": appended_upper,
        "missing_ordinals": missing,
        "definition_count": k,
        "objective_aux": objective_aux,
        "col_lower": col_lower,
        "col_upper": col_upper,
        "y_lower": y_lower,
        "y_upper": y_upper,
        "definitions": definitions,
        "definition_rhs": definition_rhs,
        "combined": combined,
        "logical_lower": logical_lower,
        "logical_upper": logical_upper,
        "delta": delta,
        "q": q,
        "centers": centers,
        "signs": signs,
        "earlier": earlier,
        "tiny": {
            "existing": existing_tiny,
            "missing": missing_tiny,
            "definitions": definition_tiny,
        },
        "assembly_seconds": time.monotonic() - started,
    }


def mutate_and_solve(owner: Any, base: Any, plan: Mapping[str, Any]) -> Dict[str, Any]:
    if owner.state != "BASE_SOLVED" or owner.model_loads != 1 or not owner.model_loaded:
        owner._poison("persistent update requires one solved loaded base")
    if owner.highs.getNumRow() != base.loaded_matrix.shape[0] or owner.highs.getNumCol() != base.loaded_matrix.shape[1]:
        owner._poison("persistent base dimensions drifted")
    owner.state = "UPDATED_LOADING"
    owner.check_deadline("persistent_update:before_mutation")
    h = owner.highs
    n_x = base.loaded_matrix.shape[1]
    k = int(plan["objective_aux"].size)
    aux_csc = plan["existing_aux"].tocsc()
    aux_csc.sort_indices()
    mutation_started = time.monotonic()
    owner._require_ok(
        h.addCols(
            k,
            -np.asarray(plan["objective_aux"], dtype=np.float64),
            np.asarray(plan["y_lower"], dtype=np.float64),
            np.asarray(plan["y_upper"], dtype=np.float64),
            int(aux_csc.nnz),
            np.ascontiguousarray(aux_csc.indptr, dtype=np.int32),
            np.ascontiguousarray(aux_csc.indices, dtype=np.int32),
            np.ascontiguousarray(aux_csc.data, dtype=np.float64),
        ),
        "persistent add auxiliary columns",
    )
    row_ids = np.arange(base.loaded_matrix.shape[0], dtype=np.int32)
    owner._require_ok(
        h.changeRowsBounds(
            row_ids.size,
            row_ids,
            np.asarray(plan["existing_lower"], dtype=np.float64),
            np.asarray(plan["existing_upper"], dtype=np.float64),
        ),
        "persistent change phase row sides",
    )
    appended = plan["appended"]
    owner._require_ok(
        h.addRows(
            appended.shape[0],
            np.asarray(plan["appended_lower"], dtype=np.float64),
            np.asarray(plan["appended_upper"], dtype=np.float64),
            int(appended.nnz),
            np.ascontiguousarray(appended.indptr, dtype=np.int32),
            np.ascontiguousarray(appended.indices, dtype=np.int32),
            np.ascontiguousarray(appended.data, dtype=np.float64),
        ),
        "persistent append missing phase and definition rows",
    )
    mutation_seconds = time.monotonic() - mutation_started
    expected_rows = base.loaded_matrix.shape[0] + appended.shape[0]
    expected_cols = n_x + k
    expected_nnz = base.loaded_matrix.nnz + aux_csc.nnz + appended.nnz
    if h.getNumRow() != expected_rows or h.getNumCol() != expected_cols or h.getNumNz() != expected_nnz:
        owner._poison("persistent mutation row/column/nnz postcondition failed")

    basis_before_run = bool(h.getBasis().valid)
    owner.check_deadline("persistent_update:before_run")
    remaining = owner._remaining(LP_TIME_LIMIT)
    owner._require_ok(h.setOptionValue("time_limit", remaining), "persistent latest set time_limit")
    status, observed = h.getOptionValue("time_limit")
    owner._require_ok(status, "persistent latest get time_limit")
    if observed != remaining:
        owner._poison("persistent latest time_limit did not round-trip")
    solve_started = time.monotonic()
    owner._require_ok(h.run(), "persistent warm run")
    solve_seconds = time.monotonic() - solve_started
    owner.solve_count += 1
    model_status = h.getModelStatus()
    readback_started = time.monotonic()
    factors = None
    aux = None
    objective_value = None
    iterations = int(h.getInfo().simplex_iteration_count)
    if model_status == highspy.HighsModelStatus.kOptimal:
        solution = h.getSolution()
        values = np.asarray(solution.col_value, dtype=np.float64)
        if not solution.value_valid or values.shape != (expected_cols,) or not np.all(np.isfinite(values)):
            owner._poison("persistent optimal primal is malformed")
        factors = values[:n_x].copy()
        aux = values[n_x:].copy()
        objective_value = float(h.getObjectiveValue())
        if not math.isfinite(objective_value):
            owner._poison("persistent objective is nonfinite")
    readback_seconds = time.monotonic() - readback_started
    owner.model_loads += 1
    owner.state = "UPDATED_SOLVED"
    return {
        "model_status": str(model_status),
        "factors": factors,
        "aux": aux,
        "objective_value": objective_value,
        "mutation_seconds": mutation_seconds,
        "warm_solve_seconds": solve_seconds,
        "readback_seconds": readback_seconds,
        "simplex_iterations": iterations,
        "basis_valid_before_warm_run": basis_before_run,
        "rows": expected_rows,
        "columns": expected_cols,
        "nnz": expected_nnz,
    }


def lowrank_diagnostics(c: Mapping[str, Any], plan: Mapping[str, Any], solved: Mapping[str, Any]) -> Dict[str, Any]:
    factors = solved["factors"]
    aux = solved["aux"]
    if factors is None or aux is None:
        return {"optimal": False}
    logical_def = np.asarray(plan["definitions"] @ np.concatenate((factors, aux))).reshape(-1)
    rhs = np.asarray(plan["definition_rhs"])
    margin = float(
        c["objective_center"]
        + np.asarray(c["objective_coeff"], dtype=np.float64) @ factors
        + np.asarray(plan["objective_aux"], dtype=np.float64) @ aux
    )
    return {
        "optimal": True,
        "margin": margin,
        "definition_max_abs_residual": float(np.max(np.abs(logical_def - rhs))),
        "aux_min_bound_slack": float(
            min(
                np.min(aux - np.asarray(plan["y_lower"])),
                np.min(np.asarray(plan["y_upper"]) - aux),
            )
        ),
    }


def oracle_at_lowrank_solution(
    helper: Any,
    c: Dict[str, Any],
    assignments: Mapping[int, np.ndarray],
    factors: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Diagnostic full-rebuild oracle after proposed timing; never solved."""
    if factors is None:
        return {"invoked": False}
    started = time.monotonic()
    rebuilt = helper.rebuild_one_cell(c, assignments)
    matrix, rhs, coeff, center, keep = import_fprime().assemble_updated_cell(helper, c, rebuilt)
    values = np.asarray(matrix @ factors).reshape(-1)
    violation = values - rhs
    margin = float(center + coeff @ factors)
    return {
        "invoked": True,
        "second_solver_used": False,
        "full_target_materialization_excluded_from_proposed_timing": True,
        "rows": int(matrix.shape[0]),
        "nnz": int(matrix.nnz),
        "margin_at_lowrank_factors": margin,
        "max_upper_row_violation": float(max(0.0, np.max(violation))) if violation.size else 0.0,
        "violated_rows": int(np.count_nonzero(violation > 1.0e-9 * (1.0 + np.abs(rhs)))),
        "keep_count": int(np.count_nonzero(keep)),
        "seconds_nonauthority": time.monotonic() - started,
        "rebuilt_stage_seconds": {
            "center": float(rebuilt["center_seconds"]),
            "delta": float(rebuilt["delta_seconds"]),
            "expansion": float(rebuilt["expansion_seconds"]),
        },
    }


def run_case(case: str) -> Dict[str, Any]:
    fprime = import_fprime()
    helper = fprime.import_frozen_helpers(case)
    helper.initialize_device(device="cuda", dtype="float64")
    helper.set_solver_mode("hybridz")
    helper.set_transfer_function_mode("interval")
    category, onnx, vnnlib = fprime.CONTROLS[case]
    sr = helper.create_specs_from_paths(onnx, vnnlib, category=category)
    vm = next(iter(helper.synthesize_models_from_specs([sr]).values()))
    net = helper.TorchToACT(vm).run()
    entry = helper.find_entry_layer_id(net)
    specs = helper.gather_input_spec_layers(net)
    seed = helper.seed_from_input_specs(specs)
    fact = helper.Fact(bounds=seed, cons=helper.ConSet())
    helper.add_all_input_specs(fact.cons, helper.get_input_ids(net), specs)
    before, after, _ = helper.analyze(net, entry, fact)

    owner = fprime.SafeHighsOwner(deadline=time.monotonic() + LP_TIME_LIMIT)
    cleanup = None
    primary: Optional[BaseException] = None
    try:
        captured, base_result, base_error, base_seconds = fprime.capture_base_cell(
            helper, owner, net, entry, before, after
        )
        if base_result is not None or not owner.records:
            raise SentinelStop("control did not expose an update-eligible base LP")
        base = owner.records[0]
        captured["net"] = net
        captured["objective_center"] = float(captured["objective_center"])
        captured["objective_coeff"] = np.asarray(captured["objective_coeff"], dtype=np.float64)
        rule, selected, selection = fprime.select_update_rows(captured, base)
        mapping, _base_selected, _centers, _A = full_phase_state(captured)
        ordinal = {row: index for index, row in enumerate(mapping)}
        selected = sorted(selected, key=lambda row: ordinal[tuple(row)])

        delta_pre, delta_output, assignments, delta_seconds = build_signed_delta(
            helper, captured, selected
        )
        plan = build_update_plan(
            fprime, captured, base, selected, delta_pre, delta_output
        )
        solved = mutate_and_solve(owner, base, plan)
        diagnostics = lowrank_diagnostics(captured, plan, solved)
        terminal = None
        if diagnostics.get("optimal") and diagnostics["margin"] > 0.0:
            owner.check_deadline("persistent_update:before_terminal")
            terminal = fprime.unchanged_terminal_with_deadline(
                helper, captured, solved["factors"], owner.deadline
            )
            owner.check_deadline("persistent_update:after_terminal")
        oracle = oracle_at_lowrank_solution(
            helper, captured, assignments, solved["factors"]
        )
        component_seconds = (
            delta_seconds
            + solved["mutation_seconds"]
            + solved["warm_solve_seconds"]
            + solved["readback_seconds"]
        )
        full_incremental_seconds = component_seconds + plan["assembly_seconds"]
        return {
            "schema": "act.scratch.fprime_persistent_lowrank_sentinel.v1",
            "case": case,
            "status": (
                "UPDATED_TERMINAL_VERIFIED"
                if terminal and terminal.get("verified")
                else "UPDATED_POSITIVE_TERMINAL_REJECTED"
                if terminal
                else "UPDATED_NONPOSITIVE_OR_UNSOLVED"
            ),
            "selection_rule": rule,
            "selection_diagnostics": selection,
            "selected_flip_count": len(selected),
            "selected_rows_sha256": fprime.selected_digest(selected),
            "base": {
                "model_status": str(base.model_status),
                "seconds_instrumented": base_seconds,
                "rows": base.logical_rows,
                "logical_nnz": base.logical_nnz,
                "loaded_nnz": base.loaded_nnz,
                "simplex_iterations": base.simplex_iterations,
                "builder_error_type": None if base_error is None else type(base_error).__name__,
            },
            "representation": {
                "formula_0_to_1": "y_i=c_i+q_i*x+sum_{j<i}D_ij*y_j",
                "formula_1_to_0": "y_i=-(c_i+q_i*x)",
                "target_pre": "c+Q*x+D*y",
                "objective": "c_obj+q_obj*x+D_obj*y",
                "existing_flipped_phase_operation": "same oriented row; upper bound becomes lower bound",
                "full_target_pre_or_output_materialized_on_proposed_path": False,
                "same_highs_owner": True,
                "second_solver": False,
            },
            "plan": {
                "base_screened_rows": base.logical_rows,
                "missing_phase_rows_appended": int(plan["missing_ordinals"].size),
                "definition_range_rows_appended": int(plan["definition_count"]),
                "auxiliary_columns_appended": int(len(selected)),
                "appended_rows_total": int(plan["appended"].shape[0]),
                "appended_nnz": int(plan["appended"].nnz),
                "existing_aux_nnz": int(plan["existing_aux"].nnz),
                "aux_bound_min": float(np.min(plan["y_lower"])),
                "aux_bound_max": float(np.max(plan["y_upper"])),
                "tiny": plan["tiny"],
            },
            "updated": {
                key: value
                for key, value in solved.items()
                if key not in {"factors", "aux"}
            },
            "candidate": diagnostics,
            "terminal": terminal,
            "oracle": oracle,
            "timing": {
                "timing_authority": False,
                "delta_seconds": delta_seconds,
                "plan_assembly_seconds": plan["assembly_seconds"],
                "mutation_seconds": solved["mutation_seconds"],
                "warm_solve_seconds": solved["warm_solve_seconds"],
                "readback_seconds": solved["readback_seconds"],
                "component_delta_mutation_warm_readback_seconds": component_seconds,
                "full_incremental_through_readback_seconds": full_incremental_seconds,
                "component_screening_target_seconds": 0.097170,
                "component_screening_target_pass": bool(component_seconds <= 0.097170),
                "component_target_is_not_fprime_promotion_authority": True,
            },
            "owner": {
                "instances": 1,
                "model_loads": owner.model_loads,
                "solves": owner.solve_count,
                "dual_ray_exist_calls": owner.dual_ray_exist_calls,
                "dual_ray_calls": owner.dual_ray_calls,
                "state_before_close": owner.state,
                "poisoned": owner.poisoned,
            },
            "restrictions": restrictions(),
            "source_locks": {
                "fprime": FPRIME_SHA256,
                "phase": sha256(ROOT / "act/back_end/hybridz_tf/forward_exact_relu_phase_projection_candidate.py"),
                "live": sha256(ROOT / "act/back_end/hybridz_tf/forward_exact_relu_live_row_stream_candidate.py"),
            },
        }
    except BaseException as exc:
        primary = exc
        if not isinstance(exc, Exception):
            raise
        if isinstance(exc, SentinelStop):
            reason = exc.reason
            details = exc.details
        elif isinstance(exc, fprime.ProbeStop):
            reason = exc.reason
            details = exc.details
        else:
            reason = "unexpected fail-closed exception"
            details = {"exception_type": type(exc).__name__}
        return {
            "schema": "act.scratch.fprime_persistent_lowrank_sentinel.v1",
            "case": case,
            "status": "STOP_LOSS_UNKNOWN",
            "reason": reason,
            "details": details,
            "restrictions": restrictions(),
        }
    finally:
        try:
            cleanup = owner.close()
        except BaseException as cleanup_error:
            if primary is not None:
                try:
                    primary.add_note("secondary owner cleanup failure type=" + type(cleanup_error).__name__)
                except BaseException:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=CASES)
    args = parser.parse_args()
    receipt = run_case(args.case)
    receipt["scratch_sha256"] = sha256(Path(__file__).resolve())
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
