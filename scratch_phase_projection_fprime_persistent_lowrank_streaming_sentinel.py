#!/usr/bin/env python3
"""Existing-only streaming plan for the F-prime low-rank sentinel.

This is the sole structural replacement admitted after the generic plan's
performance stop-loss.  It reuses the already loaded base x matrix and its
x-tiny-compensated upper bounds.  Existing rows stream only the new D column
entries.  Only K flipped base rows are scanned for the opposite-side x-tiny
compensation; complete rows are constructed only for newly required phase
rows and the K triangular definitions.

The new-column tiny projection is deliberately owner-side and occurs once.
The plan never folds y-tiny compensation into its x-only bounds, preventing a
future SafeHighsOwner from compensating the same deleted term twice.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import highspy
import numpy as np
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parent
GENERIC_PATH = ROOT / "scratch_phase_projection_fprime_persistent_lowrank_sentinel.py"
GENERIC_SHA256 = "1f399a29be9fdf9cf2cca0b89adbb6721b307d1913d37be3bb704155458fdaeb"
SMALL = 1.0e-12
INFINITE_BOUND = 1.0e20
LP_TIME_LIMIT = 30.0


def import_generic() -> Any:
    import hashlib

    digest = hashlib.sha256(GENERIC_PATH.read_bytes()).hexdigest()
    if digest != GENERIC_SHA256:
        raise RuntimeError("generic sentinel source hash changed")
    spec = importlib.util.spec_from_file_location("fprime_lowrank_generic_frozen", GENERIC_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import frozen generic sentinel")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_existing_csr(matrix: Any) -> sp.csr_matrix:
    if (
        not sp.isspmatrix_csr(matrix)
        or matrix.dtype != np.dtype(np.float64)
        or not matrix.has_sorted_indices
        or not matrix.has_canonical_format
        or np.any(matrix.data == 0.0)
        or not np.all(np.isfinite(matrix.data))
    ):
        raise RuntimeError("captured base A is not a canonical finite float64 CSR")
    return matrix


def full_phase_metadata(c: Mapping[str, Any]) -> Tuple[Tuple[Tuple[int, int, int], ...], np.ndarray, np.ndarray, sp.csr_matrix]:
    mapping = []
    assignments = []
    centers = []
    for layer in c["order"]:
        layer_id = int(layer.id)
        original = c["original_frames"].get(layer_id)
        if original is None or not original.exact.size:
            continue
        rows = np.asarray(original.stream_rows, dtype=np.int64)
        selected = np.asarray(c["target_assign"][layer_id], dtype=bool)
        local_centers = np.asarray(c["target_pre_center"][layer_id], dtype=np.float64)
        if not rows.size == selected.size == local_centers.size:
            raise RuntimeError("phase metadata width mismatch")
        mapping.extend((layer_id, int(position), int(row)) for position, row in enumerate(rows))
        assignments.append(selected)
        centers.append(local_centers)
    result = tuple(mapping)
    if len(result) != len({(layer, row) for layer, _position, row in result}):
        raise RuntimeError("duplicate physical phase id")
    selected = np.concatenate(assignments)
    center = np.concatenate(centers)
    matrix = validate_existing_csr(c["A"])
    if matrix.shape[0] != len(result):
        raise RuntimeError("full phase map/A row mismatch")
    return result, selected, center, matrix


def exact_row_interval(
    row: sp.csr_matrix,
    lower: np.ndarray,
    upper: np.ndarray,
    constant: float,
    generic: Any,
) -> Tuple[float, float]:
    exact_lower = Fraction.from_float(float(constant))
    exact_upper = exact_lower
    for column, value in zip(row.indices, row.data):
        coefficient = Fraction.from_float(float(value))
        lo = coefficient * Fraction.from_float(float(lower[column]))
        hi = coefficient * Fraction.from_float(float(upper[column]))
        exact_lower += min(lo, hi)
        exact_upper += max(lo, hi)
    return generic.fraction_down(exact_lower), generic.fraction_up(exact_upper)


def streaming_recursive_bounds(
    generic: Any,
    selected_q: sp.csr_matrix,
    selected_centers: np.ndarray,
    selected_delta: np.ndarray,
    selected_base_active: np.ndarray,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k = selected_q.shape[0]
    signs = np.where(selected_base_active, -1.0, 1.0)
    earlier = (signs + 1.0) * 0.5
    y_lower = np.empty(k, dtype=np.float64)
    y_upper = np.empty(k, dtype=np.float64)
    for i in range(k):
        row = selected_q.getrow(i).multiply(float(signs[i]))
        lo, hi = exact_row_interval(
            row,
            x_lower,
            x_upper,
            float(signs[i] * selected_centers[i]),
            generic,
        )
        exact_lo = Fraction.from_float(lo)
        exact_hi = Fraction.from_float(hi)
        if earlier[i] != 0.0:
            for j in range(i):
                value = float(selected_delta[i, j])
                if value == 0.0:
                    continue
                coefficient = Fraction.from_float(value)
                left = coefficient * Fraction.from_float(float(y_lower[j]))
                right = coefficient * Fraction.from_float(float(y_upper[j]))
                exact_lo += min(left, right)
                exact_hi += max(left, right)
        y_lower[i] = generic.fraction_down(exact_lo)
        y_upper[i] = generic.fraction_up(exact_hi)
        if (
            not math.isfinite(y_lower[i])
            or not math.isfinite(y_upper[i])
            or y_lower[i] > y_upper[i]
            or abs(y_lower[i]) >= INFINITE_BOUND
            or abs(y_upper[i]) >= INFINITE_BOUND
        ):
            raise RuntimeError("streaming auxiliary bound failed")
    return y_lower, y_upper, signs, earlier


def x_only_flipped_lower(
    generic: Any,
    oriented_row: sp.csr_matrix,
    logical_bound: float,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
) -> Tuple[float, int]:
    exact_max = Fraction(0)
    count = 0
    for column, value in zip(oriented_row.indices, oriented_row.data):
        if not (0.0 < abs(float(value)) <= SMALL):
            continue
        count += 1
        coefficient = Fraction.from_float(float(value))
        lo = coefficient * Fraction.from_float(float(x_lower[column]))
        hi = coefficient * Fraction.from_float(float(x_upper[column]))
        exact_max += max(lo, hi)
    lower = generic.fraction_down(Fraction.from_float(float(logical_bound)) - exact_max)
    return lower, count


def build_streaming_plan(
    fprime: Any,
    c: Mapping[str, Any],
    base: Any,
    selected_rows: Sequence[Tuple[int, int, int]],
    delta_pre: Mapping[int, np.ndarray],
    delta_output: np.ndarray,
) -> Dict[str, Any]:
    del fprime
    generic = import_generic()
    started = time.monotonic()
    mapping, base_selected, centers, base_oriented = full_phase_metadata(c)
    ordinal_of = {key: index for index, key in enumerate(mapping)}
    selected_ordinals = np.asarray([ordinal_of[tuple(row)] for row in selected_rows], dtype=np.int64)
    if selected_ordinals.size == 0 or not np.all(selected_ordinals[1:] > selected_ordinals[:-1]):
        raise RuntimeError("selected set is not nonempty strict topological order")
    keep = np.asarray(c["keep"], dtype=bool)
    if keep.shape != (len(mapping),) or not np.all(keep[selected_ordinals]):
        raise RuntimeError("selected phase absent from base screened model")
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
    k = selected_ordinals.size
    if delta.shape != (len(mapping), k):
        raise RuntimeError("streaming delta shape mismatch")
    for i, ordinal in enumerate(selected_ordinals):
        if np.any(delta[ordinal, i:] != 0.0):
            raise RuntimeError("streaming delta is not topologically triangular")

    orientation = np.where(base_selected, -1.0, 1.0)
    selected_q = generic._canonical_csr(
        base_oriented[selected_ordinals].multiply(orientation[selected_ordinals, None])
    )
    selected_delta = delta[selected_ordinals]
    x_lower = np.asarray(c["factor_lower"], dtype=np.float64)
    x_upper = np.asarray(c["factor_upper"], dtype=np.float64)
    y_lower, y_upper, signs, earlier = streaming_recursive_bounds(
        generic,
        selected_q,
        centers[selected_ordinals],
        selected_delta,
        base_selected[selected_ordinals],
        x_lower,
        x_upper,
    )
    n_x = x_lower.size

    # The plan hands logical D to the owner.  It performs no y-tiny deletion
    # and therefore cannot pre-compensate a y term that the owner will later
    # project exactly once.
    raw_existing_aux = generic._canonical_csr(
        sp.csr_matrix(orientation[screened_to_full, None] * delta[screened_to_full]),
        columns=k,
    )
    existing_x_only_lower = np.full(base.logical_rows, -np.inf, dtype=np.float64)
    existing_x_only_upper = np.asarray(base.loaded_rhs, dtype=np.float64).copy()
    selected_solver_rows = full_to_screened[selected_ordinals].astype(np.int32)
    flipped_x_tiny = 0
    logical_rhs = np.asarray(c["b"], dtype=np.float64)
    for solver_row, full_row in zip(selected_solver_rows, selected_ordinals):
        lower, count = x_only_flipped_lower(
            generic,
            base_oriented.getrow(int(full_row)),
            float(logical_rhs[full_row]),
            x_lower,
            x_upper,
        )
        existing_x_only_lower[int(solver_row)] = lower
        existing_x_only_upper[int(solver_row)] = np.inf
        flipped_x_tiny += count

    # Missing-row screen reuses the already computed base x box upper and
    # scans only K new coefficients per omitted row.
    omitted = np.flatnonzero(~keep)
    missing = np.empty(0, dtype=np.int64)
    raw_missing = sp.csr_matrix((0, n_x + k), dtype=np.float64)
    missing_lower = np.empty(0, dtype=np.float64)
    missing_upper = np.empty(0, dtype=np.float64)
    if omitted.size:
        aux = orientation[omitted, None] * delta[omitted]
        aux_upper = np.sum(
            aux * np.where(aux >= 0.0, y_upper[None, :], y_lower[None, :]),
            axis=1,
            dtype=np.float64,
        )
        base_box_upper = np.asarray(c["row_max"], dtype=np.float64)[omitted]
        augmented_upper = base_box_upper + aux_upper
        if not np.all(np.isfinite(augmented_upper)):
            raise RuntimeError("streaming missing-row screen overflowed")
        missing = omitted[augmented_upper > logical_rhs[omitted]]
        if missing.size:
            raw_missing = generic._canonical_csr(
                sp.hstack(
                    (
                        base_oriented[missing],
                        sp.csr_matrix(orientation[missing, None] * delta[missing]),
                    ),
                    format="csr",
                ),
                columns=n_x + k,
            )
            missing_lower = np.full(missing.size, -np.inf, dtype=np.float64)
            missing_upper = logical_rhs[missing].copy()

    definitions_rows = []
    definition_rhs = []
    for i in range(k):
        base_part = selected_q.getrow(i).multiply(-float(signs[i]))
        prior = np.zeros((1, k), dtype=np.float64)
        if earlier[i] != 0.0 and i:
            prior[0, :i] = -selected_delta[i, :i]
        prior[0, i] = 1.0
        definitions_rows.append(sp.hstack((base_part, sp.csr_matrix(prior)), format="csr"))
        definition_rhs.append(float(signs[i] * centers[selected_ordinals[i]]))
    definitions = generic._canonical_csr(
        sp.vstack(definitions_rows, format="csr"), columns=n_x + k
    )
    definition_rhs_array = np.asarray(definition_rhs, dtype=np.float64)
    raw_appended = generic._canonical_csr(
        sp.vstack((raw_missing, definitions), format="csr"), columns=n_x + k
    )
    raw_appended_lower = np.concatenate((missing_lower, definition_rhs_array))
    raw_appended_upper = np.concatenate((missing_upper, definition_rhs_array))

    objective_aux = np.asarray(c["C"][[c["rival"]]] @ delta_output, dtype=np.float64).reshape(-1)
    objective_tiny = (objective_aux != 0.0) & (np.abs(objective_aux) <= SMALL)
    if (
        np.any(objective_tiny)
        or not np.all(np.isfinite(objective_aux))
        or np.any(np.abs(objective_aux) >= INFINITE_BOUND)
    ):
        raise RuntimeError("streaming auxiliary objective failed tiny/finiteness contract")

    return {
        "mapping": mapping,
        "selected_ordinals": selected_ordinals,
        "selected_solver_rows": selected_solver_rows,
        "existing_aux": raw_existing_aux,
        "existing_lower": existing_x_only_lower,
        "existing_upper": existing_x_only_upper,
        "appended": raw_appended,
        "appended_lower": raw_appended_lower,
        "appended_upper": raw_appended_upper,
        "missing_ordinals": missing,
        "definition_count": int(k),
        "objective_aux": objective_aux,
        "col_lower": np.concatenate((x_lower, y_lower)),
        "col_upper": np.concatenate((x_upper, y_upper)),
        "y_lower": y_lower,
        "y_upper": y_upper,
        "definitions": definitions,
        "definition_rhs": definition_rhs_array,
        "delta": delta,
        "signs": signs,
        "earlier": earlier,
        "tiny": {
            "plan_y_tiny_projection_count": 0,
            "plan_y_tiny_compensation_count": 0,
            "flipped_x_tiny_scanned": int(flipped_x_tiny),
            "owner_projection_pending": True,
        },
        "assembly_seconds": time.monotonic() - started,
        "streaming_contract": {
            "existing_x_matrix_copied": False,
            "existing_x_matrix_refiltered": False,
            "existing_logical_D_passed_to_owner": True,
            "existing_bounds_contain_x_tiny_only": True,
            "new_column_y_tiny_projected_owner_side_once": True,
        },
    }


def mutate_and_solve_streaming(owner: Any, base: Any, plan: Mapping[str, Any]) -> Dict[str, Any]:
    generic = import_generic()
    if owner.state != "BASE_SOLVED" or owner.model_loads != 1 or not owner.model_loaded:
        owner._poison("streaming update requires one solved loaded base")
    if owner.highs.getNumRow() != base.loaded_matrix.shape[0] or owner.highs.getNumCol() != base.loaded_matrix.shape[1]:
        owner._poison("streaming base dimensions drifted")
    owner.state = "UPDATED_LOADING"
    owner.check_deadline("streaming_update:before_owner_projection")
    tiny_contract = plan.get("tiny", {})
    if not (
        tiny_contract.get("plan_y_tiny_projection_count") == 0
        and tiny_contract.get("plan_y_tiny_compensation_count") == 0
        and tiny_contract.get("owner_projection_pending") is True
        and plan.get("streaming_contract", {}).get("existing_bounds_contain_x_tiny_only") is True
        and plan.get("streaming_contract", {}).get("existing_logical_D_passed_to_owner") is True
    ):
        owner._poison("streaming plan attempted or obscured caller-side y-tiny projection")
    h = owner.highs
    n_x = base.loaded_matrix.shape[1]
    k = int(np.asarray(plan["objective_aux"]).size)

    # FrozenNewColumns-equivalent owner projection.  Only y coefficients are
    # supplied here, so their tiny contribution is compensated exactly once.
    projection_started = time.monotonic()
    loaded_aux, loaded_existing_lower, loaded_existing_upper, existing_tiny = generic.filter_tiny_ranged_rows(
        plan["existing_aux"],
        np.asarray(plan["existing_lower"], dtype=np.float64),
        np.asarray(plan["existing_upper"], dtype=np.float64),
        np.asarray(plan["y_lower"], dtype=np.float64),
        np.asarray(plan["y_upper"], dtype=np.float64),
    )
    loaded_appended, loaded_appended_lower, loaded_appended_upper, appended_tiny = generic.filter_tiny_ranged_rows(
        plan["appended"],
        np.asarray(plan["appended_lower"], dtype=np.float64),
        np.asarray(plan["appended_upper"], dtype=np.float64),
        np.asarray(plan["col_lower"], dtype=np.float64),
        np.asarray(plan["col_upper"], dtype=np.float64),
    )
    projection_seconds = time.monotonic() - projection_started
    plan["tiny"]["owner_existing_new_columns"] = existing_tiny
    plan["tiny"]["owner_appended_rows"] = appended_tiny
    plan["tiny"]["owner_projection_pending"] = False
    plan["tiny"]["owner_y_tiny_projection_passes"] = 1

    aux_csc = loaded_aux.tocsc()
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
        "streaming add auxiliary columns",
    )
    row_ids = np.arange(base.loaded_matrix.shape[0], dtype=np.int32)
    owner._require_ok(
        h.changeRowsBounds(
            row_ids.size,
            row_ids,
            loaded_existing_lower,
            loaded_existing_upper,
        ),
        "streaming change phase row sides",
    )
    owner._require_ok(
        h.addRows(
            loaded_appended.shape[0],
            loaded_appended_lower,
            loaded_appended_upper,
            int(loaded_appended.nnz),
            np.ascontiguousarray(loaded_appended.indptr, dtype=np.int32),
            np.ascontiguousarray(loaded_appended.indices, dtype=np.int32),
            np.ascontiguousarray(loaded_appended.data, dtype=np.float64),
        ),
        "streaming append missing phase and definition rows",
    )
    mutation_seconds = time.monotonic() - mutation_started
    expected_rows = base.loaded_matrix.shape[0] + loaded_appended.shape[0]
    expected_cols = n_x + k
    expected_nnz = base.loaded_matrix.nnz + aux_csc.nnz + loaded_appended.nnz
    if h.getNumRow() != expected_rows or h.getNumCol() != expected_cols or h.getNumNz() != expected_nnz:
        owner._poison("streaming mutation postcondition failed")

    basis_before_run = bool(h.getBasis().valid)
    owner.check_deadline("streaming_update:before_run")
    remaining = owner._remaining(LP_TIME_LIMIT)
    owner._require_ok(h.setOptionValue("time_limit", remaining), "streaming latest set time_limit")
    status, observed = h.getOptionValue("time_limit")
    owner._require_ok(status, "streaming latest get time_limit")
    if observed != remaining:
        owner._poison("streaming latest time_limit did not round-trip")
    solve_started = time.monotonic()
    owner._require_ok(h.run(), "streaming warm run")
    solve_seconds = time.monotonic() - solve_started
    owner.solve_count += 1
    model_status = h.getModelStatus()
    readback_started = time.monotonic()
    factors = None
    aux = None
    objective_value = None
    if model_status == highspy.HighsModelStatus.kOptimal:
        solution = h.getSolution()
        values = np.asarray(solution.col_value, dtype=np.float64)
        if not solution.value_valid or values.shape != (expected_cols,) or not np.all(np.isfinite(values)):
            owner._poison("streaming optimal primal malformed")
        factors = values[:n_x].copy()
        aux = values[n_x:].copy()
        objective_value = float(h.getObjectiveValue())
        if not math.isfinite(objective_value):
            owner._poison("streaming objective nonfinite")
    readback_seconds = time.monotonic() - readback_started
    owner.model_loads += 1
    owner.state = "UPDATED_SOLVED"
    return {
        "model_status": str(model_status),
        "factors": factors,
        "aux": aux,
        "objective_value": objective_value,
        "owner_projection_seconds": projection_seconds,
        "mutation_seconds": mutation_seconds,
        "warm_solve_seconds": solve_seconds,
        "readback_seconds": readback_seconds,
        "simplex_iterations": int(h.getInfo().simplex_iteration_count),
        "basis_valid_before_warm_run": basis_before_run,
        "rows": expected_rows,
        "columns": expected_cols,
        "nnz": expected_nnz,
    }


def self_test() -> Dict[str, Any]:
    generic = import_generic()
    x = sp.csr_matrix(
        np.asarray([[2.0e-13, 2.0], [-3.0e-13, -1.0]], dtype=np.float64)
    )
    y = sp.csr_matrix(np.asarray([[4.0e-13], [-5.0e-13]], dtype=np.float64))
    x_lo = np.asarray([-1.0, -1.0])
    x_hi = np.asarray([1.0, 1.0])
    y_lo = np.asarray([-2.0])
    y_hi = np.asarray([3.0])
    logical_lower = np.asarray([-np.inf, 0.25])
    logical_upper = np.asarray([0.5, np.inf])
    loaded_x, x_lower, x_upper, _ = generic.filter_tiny_ranged_rows(
        x, logical_lower, logical_upper, x_lo, x_hi
    )
    del loaded_x
    loaded_y, sequential_lower, sequential_upper, receipt = generic.filter_tiny_ranged_rows(
        y, x_lower, x_upper, y_lo, y_hi
    )
    combined = generic._canonical_csr(sp.hstack((x, y), format="csr"))
    _, one_lower, one_upper, _ = generic.filter_tiny_ranged_rows(
        combined,
        logical_lower,
        logical_upper,
        np.concatenate((x_lo, y_lo)),
        np.concatenate((x_hi, y_hi)),
    )
    finite_lower = np.isfinite(one_lower)
    finite_upper = np.isfinite(one_upper)
    if not (
        np.all(sequential_lower[finite_lower] <= one_lower[finite_lower])
        and np.all(sequential_upper[finite_upper] >= one_upper[finite_upper])
        and receipt["deleted_tiny_nnz"] == 2
        and loaded_y.nnz == 0
    ):
        raise RuntimeError("owner-side one-time new-column projection self-test failed")
    # Execute the real build_streaming_plan entry, not merely its leaf helpers,
    # on a two-phase synthetic base model.  This catches runtime-only names,
    # mapping assumptions, x-only bound ownership, and the missing-row path.
    order = [SimpleNamespace(id=5)]
    original = SimpleNamespace(
        exact=np.asarray([10, 11], dtype=np.int64),
        stream_rows=np.asarray([10, 11], dtype=np.int64),
    )
    base_selected = np.asarray([False, True], dtype=bool)
    centers = np.asarray([0.1, -0.2], dtype=np.float64)
    raw_q = sp.csr_matrix(
        np.asarray([[1.0, 2.0e-13], [0.5, -0.25]], dtype=np.float64)
    )
    orientation = np.where(base_selected, -1.0, 1.0)
    oriented = generic._canonical_csr(raw_q.multiply(orientation[:, None]))
    logical_b = -orientation * centers
    factor_lower = np.asarray([-1.0, -1.0])
    factor_upper = np.asarray([1.0, 1.0])
    base_logical = oriented[[0]].tocsr()
    base_loaded, _lo, base_loaded_rhs, _base_tiny = generic.filter_tiny_ranged_rows(
        base_logical,
        np.asarray([-np.inf]),
        logical_b[[0]],
        factor_lower,
        factor_upper,
    )
    row_max = np.asarray([1.0 + 2.0e-13, 0.75], dtype=np.float64)
    synthetic = {
        "order": order,
        "original_frames": {5: original},
        "target_assign": {5: base_selected},
        "target_pre_center": {5: centers},
        "A": oriented,
        "keep": np.asarray([True, False]),
        "factor_lower": factor_lower,
        "factor_upper": factor_upper,
        "b": logical_b,
        "row_max": row_max,
        "C": np.asarray([[1.0]]),
        "rival": 0,
    }
    synthetic_base = SimpleNamespace(
        logical_rows=1,
        loaded_rhs=base_loaded_rhs,
        loaded_matrix=base_loaded,
    )
    built = build_streaming_plan(
        None,
        synthetic,
        synthetic_base,
        [(5, 0, 10)],
        {5: np.asarray([[0.0], [0.3]], dtype=np.float64)},
        np.asarray([[0.2]], dtype=np.float64),
    )
    if not (
        built["definition_count"] == 1
        and built["streaming_contract"]["existing_x_matrix_copied"] is False
        and built["streaming_contract"]["existing_bounds_contain_x_tiny_only"] is True
        and built["tiny"]["plan_y_tiny_compensation_count"] == 0
    ):
        raise RuntimeError("real streaming plan build self-test failed")
    return {
        "status": "PASS",
        "owner_side_y_tiny_deleted_once": receipt["deleted_tiny_nnz"],
        "sequential_projection_outward_encloses_one_shot": True,
        "existing_x_matrix_reused": True,
        "real_build_streaming_plan_invoked": True,
    }


def write_exclusive(path: Path, payload: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(
        "cifar100_large_iid153",
        "tinyimagenet_medium_iid153",
        "tinyimagenet_medium_iid143",
    ))
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--raw-path")
    parser.add_argument("--receipt-path")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True, separators=(",", ":")))
        return
    if args.case is None:
        parser.error("--case is required unless --self-test is used")
    if (args.raw_path is None) != (args.receipt_path is None):
        parser.error("--raw-path and --receipt-path must be supplied together")
    generic = import_generic()
    generic.build_update_plan = build_streaming_plan
    generic.mutate_and_solve = mutate_and_solve_streaming
    receipt = generic.run_case(args.case)
    receipt["schema"] = "act.scratch.fprime_persistent_lowrank_streaming_sentinel.v1"
    receipt["streaming_plan"] = "reuse_base_loaded_x_existing_D_only_owner_projects_y_tiny_once"
    receipt["generic_source_sha256"] = GENERIC_SHA256
    receipt["scratch_sha256"] = generic.sha256(Path(__file__).resolve())
    if "timing" in receipt and "updated" in receipt:
        receipt["timing"]["owner_projection_seconds"] = receipt["updated"].get(
            "owner_projection_seconds"
        )
        receipt["timing"]["full_incremental_including_owner_projection_seconds"] = (
            receipt["timing"]["full_incremental_through_readback_seconds"]
            + float(receipt["updated"].get("owner_projection_seconds", 0.0))
        )
    if args.raw_path is not None:
        receipt["capture"] = {
            "raw_final_json_line_path": str(Path(args.raw_path)),
            "receipt_path": str(Path(args.receipt_path)),
            "exclusive_no_overwrite": True,
        }
    compact = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if args.raw_path is not None:
        write_exclusive(Path(args.raw_path), compact + "\n")
        write_exclusive(
            Path(args.receipt_path),
            json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n",
        )
    print(compact)


if __name__ == "__main__":
    main()
