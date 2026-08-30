#!/usr/bin/env python3
"""Frozen iid2 transaction sentinel for a chunked direct HiGHS target model.

Scratch only.  The current side reruns the frozen target-center, delta,
triangular expansion, full target_pre/target_output materialization, CSR model,
solve, and readback transaction.  The proposed side builds one request-local
device program, preserves the retained legacy centers(projected) pass,
performs coefficient pre-elimination on CUDA, and composes and box-screens phase rows in
fixed chunks, and loads those chunks directly into one request-local HiGHS
model before one solve/readback.  It never materializes target_pre,
target_output, blocks, or A on the host.

No input sampling or point ONNX execution, PGD, BaB/splitting/enumeration,
backward bounds, dual tightening, retry, fallback, or parameter menu is used.
Candidate arithmetic has no verdict authority.  The unchanged terminal is run
once after timing only as a mechanical oracle.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Dict, Mapping, Sequence, Tuple

import highspy
import numpy as np
import scipy
import scipy.sparse as sp
import torch

import scratch_device_affine_program_sentinel as device


ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "artifacts/hybridz_largecls_gates/phase_projection_chunked_target_transaction_iid2_20260814.json"
WORKERS = 4
PAIRS = 5
ROW_CHUNK = 128
HIGHS_SMALL = 1.0e-12
SINGLE_BUDGET_SECONDS = 0.642884481
FOUR_BUDGET_SECONDS = 0.367045639

# The dependency reads these at import time.  They name the same fixed iid2
# files already frozen above; no external result label is read.
os.environ.setdefault("ACT_PHASE_PROJECTION_ONNX", str(device.ONNX))
os.environ.setdefault("ACT_PHASE_PROJECTION_VNNLIB", str(device.VNNLIB))
os.environ.setdefault("ACT_PHASE_PROJECTION_CATEGORY", "cifar100_2024")
os.environ.setdefault("ACT_PHASE_PROJECTION_CASE", "cifar100_medium_iid2")
ONE_PATH = ROOT / "scratch_phase_projection_one_multi_flip_probe.py"
SPEC = importlib.util.spec_from_file_location("transaction_current", ONE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("one-multi-flip dependency cannot be loaded")
current_impl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(current_impl)
phase = current_impl.phase
live = current_impl.live


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def restrictions() -> Dict[str, Any]:
    return {
        "scope": "fixed_cifar100_medium_iid2_target_transaction_component",
        "production_modified": False,
        "input_sampling_used": False,
        "onnx_input_point_execution_used": False,
        "pgd_used": False,
        "bab_split_or_enumeration_used": False,
        "backward_bounds_used": False,
        "dual_tightening_used": False,
        "candidate_or_lp_has_authority": False,
        "runtime_retry_or_fallback_menu": False,
        "parameter_scan": False,
        "phase_updates": 0,
    }


def lane_context(c: Mapping[str, Any], lane: device.Lane) -> Dict[str, Any]:
    result = dict(c)
    result["device_matrices"] = lane.matrices
    result["device_rows"] = lane.rows
    return result


def current_transaction(c: Mapping[str, Any], lane: device.Lane) -> Dict[str, Any]:
    local = lane_context(c, lane)
    assignments = {
        key: np.asarray(value, dtype=bool).copy()
        for key, value in c["target_assign"].items()
    }
    started = time.monotonic()
    rebuilt = current_impl.rebuild_one_cell(local, assignments)
    solved = current_impl.solve_cell(local, rebuilt)
    total = time.monotonic() - started
    if not solved["success"] or solved.get("factors") is None:
        raise RuntimeError("current target transaction did not solve")
    return {
        "total_seconds": total,
        "program_build_seconds": 0.0,
        "center_seconds": float(rebuilt["center_seconds"]),
        "delta_seconds": float(rebuilt["delta_seconds"]),
        "preelimination_seconds": float(rebuilt["expansion_seconds"]),
        "chunk_screen_load_seconds": float(solved["model_seconds"]),
        "solve_readback_seconds": float(solved["lp_seconds"]),
        "margin": float(solved["margin"]),
        "rows": int(solved["rows"]),
        "nnz": int(solved["nnz"]),
        "factors": np.asarray(solved["factors"], dtype=np.float64),
    }


def target_frames_and_changes(c: Mapping[str, Any]):
    frames = {}
    changes = []
    positions = {}
    for layer in c["order"]:
        layer_id = int(layer.id)
        original = c["original_frames"].get(layer_id)
        if original is None:
            continue
        selected = np.asarray(c["target_assign"][layer_id], dtype=bool)
        frames[layer_id] = phase._fixed_frame(original, selected)
        if not original.exact.size:
            continue
        rows = np.asarray(original.stream_rows, dtype=np.int64)
        positions[layer_id] = {int(row): pos for pos, row in enumerate(rows)}
        for pos in np.flatnonzero(selected != c["first_assign"][layer_id]):
            changes.append(
                (
                    layer_id,
                    int(rows[pos]),
                    bool(c["first_assign"][layer_id][pos]),
                    bool(selected[pos]),
                )
            )
    return frames, changes, positions


def device_delta(
    c: Mapping[str, Any],
    lane: device.Lane,
    program: device.DeviceProgram,
    target_frames: Mapping[int, Any],
    changes: Sequence[Tuple[int, int, bool, bool]],
):
    width_total = len(changes)
    change_index = {
        (layer_id, row): index
        for index, (layer_id, row, _base, _target) in enumerate(changes)
    }
    active = {
        layer_id: torch.as_tensor(
            np.intersect1d(
                c["live_rows"][layer_id], frame.active, assume_unique=True
            ),
            dtype=torch.int64,
            device="cuda",
        )
        for layer_id, frame in target_frames.items()
    }
    changed = {
        layer_id: [
            (row, change_index[(layer_id, row)])
            for local_layer, row, _base, _target in changes
            if local_layer == layer_id
        ]
        for layer_id in target_frames
    }
    exact_rows = {
        layer_id: torch.as_tensor(
            c["original_frames"][layer_id].stream_rows,
            dtype=torch.int64,
            device="cuda",
        )
        for layer_id in target_frames
    }
    delta_pre = {
        layer_id: torch.empty(
            (c["original_frames"][layer_id].exact.size, width_total),
            dtype=torch.float64,
            device="cuda",
        )
        for layer_id in target_frames
    }
    delta_output = torch.empty(
        (int(c["output_width"]), width_total),
        dtype=torch.float64,
        device="cuda",
    )

    torch.cuda.synchronize()
    started = time.monotonic()
    for start in range(0, width_total, 64):
        stop = min(width_total, start + 64)
        width = stop - start
        values: Dict[int, torch.Tensor] = {}
        for layer in c["order"]:
            layer_id = int(layer.id)
            kind = phase._oh._kind(layer.kind)
            predecessors = tuple(int(v) for v in c["net"].preds.get(layer_id, []))
            if kind == "INPUT":
                values[layer_id] = torch.zeros(
                    (len(layer.out_vars), width),
                    dtype=torch.float64,
                    device="cuda",
                )
            elif kind in {"INPUT_SPEC", "FLATTEN", "ASSERT"}:
                values[layer_id] = values[predecessors[0]]
            elif kind in {"CONV2D", "DENSE"}:
                selected_value = live._ordered_csr_dense(
                    lane.matrices[layer_id], values[predecessors[0]]
                )
                value = torch.zeros(
                    (len(layer.out_vars), width),
                    dtype=torch.float64,
                    device="cuda",
                )
                if lane.rows[layer_id].numel():
                    value[lane.rows[layer_id]] = selected_value
                values[layer_id] = value
            elif kind == "SCALE":
                values[layer_id] = (
                    values[predecessors[0]]
                    * program.pointwise[layer_id].reshape(-1, 1)
                )
            elif kind == "BIAS":
                values[layer_id] = values[predecessors[0]]
            elif kind == "ADD":
                values[layer_id] = values[predecessors[0]] + values[predecessors[1]]
            elif kind == "RELU":
                source = values[predecessors[0]]
                delta_pre[layer_id][:, start:stop] = source[exact_rows[layer_id]]
                value = torch.zeros_like(source)
                active_rows = active[layer_id]
                if active_rows.numel():
                    value[active_rows] = source[active_rows]
                for row, column in changed[layer_id]:
                    value[row] = 0.0
                    if start <= column < stop:
                        value[row, column - start] = 1.0
                values[layer_id] = value
            else:
                raise RuntimeError(f"unsupported target delta graph kind {kind}")
        delta_output[:, start:stop] = values[int(c["assert_layer"].id)]
    torch.cuda.synchronize()
    return delta_pre, delta_output, time.monotonic() - started


def device_preelimination(
    c: Mapping[str, Any],
    changes: Sequence[Tuple[int, int, bool, bool]],
    positions: Mapping[int, Mapping[int, int]],
    delta_pre: Mapping[int, torch.Tensor],
):
    width_total = len(changes)
    augmented_width = int(c["input_rows"].size)
    U = torch.zeros(
        (width_total, augmented_width), dtype=torch.float64, device="cuda"
    )
    torch.cuda.synchronize()
    started = time.monotonic()
    for index, (layer_id, row, base_active, target_active) in enumerate(changes):
        position = positions[layer_id][row]
        base_row = np.asarray(c["first_pre"][layer_id][position], dtype=np.float64)
        base_device = torch.as_tensor(base_row, dtype=torch.float64, device="cuda")
        if (not base_active) and target_active:
            U[index] = base_device
            if index:
                U[index] += torch.matmul(
                    delta_pre[layer_id][position, :index], U[:index]
                )
        elif base_active and (not target_active):
            U[index] = -base_device
        else:
            raise RuntimeError("invalid target phase change")
    torch.cuda.synchronize()
    return U, time.monotonic() - started


def make_highs() -> highspy.Highs:
    h = highspy.Highs()
    for key, value in (
        ("output_flag", False),
        ("solver", "simplex"),
        ("presolve", "off"),
        ("threads", 1),
        ("small_matrix_value", HIGHS_SMALL),
        ("primal_feasibility_tolerance", phase._SOLVER_TOLERANCE),
        ("dual_feasibility_tolerance", phase._SOLVER_TOLERANCE),
    ):
        if h.setOptionValue(key, value) != highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS rejected {key}")
    return h


def require_ok(status, operation: str) -> None:
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS {operation} failed: {status}")


def fraction_upper(value: Fraction) -> float:
    """Least binary64 at or above one exact dyadic sum."""
    rounded = float(value)
    if Fraction.from_float(rounded) < value:
        rounded = float(np.nextafter(rounded, np.inf))
    return rounded


def filter_tiny_rows(
    dense: np.ndarray,
    rhs: np.ndarray,
    factor_lower: np.ndarray,
    factor_upper: np.ndarray,
) -> Tuple[sp.csr_matrix, np.ndarray, int, float]:
    """Fixed fail-closed loader filter after logical full-row screening."""
    dense = np.asarray(dense, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    tiny = (dense != 0.0) & (np.abs(dense) <= HIGHS_SMALL)
    deleted = int(np.count_nonzero(tiny))
    maximum_relaxation = 0.0
    loaded_rhs = rhs.copy()
    if deleted:
        for row in np.flatnonzero(np.any(tiny, axis=1)):
            exact = Fraction.from_float(float(rhs[row]))
            for column in np.flatnonzero(tiny[row]):
                coefficient = Fraction.from_float(float(dense[row, column]))
                lower = Fraction.from_float(float(factor_lower[column]))
                upper = Fraction.from_float(float(factor_upper[column]))
                exact += max(coefficient * lower, coefficient * upper)
            loaded_rhs[row] = fraction_upper(exact)
            maximum_relaxation = max(
                maximum_relaxation, float(loaded_rhs[row] - rhs[row])
            )
        dense = dense.copy()
        dense[tiny] = 0.0
    matrix = sp.csr_matrix(dense)
    matrix.eliminate_zeros()
    return matrix, loaded_rhs, deleted, maximum_relaxation


def proposed_transaction(
    c: Mapping[str, Any],
    lane: device.Lane,
    *,
    oracle: bool = False,
    use_augmented_center: bool = False,
) -> Dict[str, Any]:
    if use_augmented_center:
        raise RuntimeError("augmented constant center is structurally retired")
    total_started = time.monotonic()
    program = device.build_program(c)
    frame_started = time.monotonic()
    target_frames, changes, positions = target_frames_and_changes(c)
    frame_seconds = time.monotonic() - frame_started
    current_center_started = time.monotonic()
    (
        current_assign,
        current_pre_center,
        current_output_center,
        target_frames,
    ) = c["centers"](c["target_assign"])
    current_center_seconds = time.monotonic() - current_center_started
    if any(
        not np.array_equal(current_assign[key], c["target_assign"][key])
        for key in current_assign
    ):
        raise RuntimeError("current projected center changed phase assignment")
    pre_center_bitwise_equal = bool(
        all(
            np.array_equal(current_pre_center[key], c["target_pre_center"][key])
            for key in current_pre_center
        )
    )
    pre_center_max_abs_difference = float(
        max(
            np.max(
                np.abs(
                    np.asarray(current_pre_center[key])
                    - np.asarray(c["target_pre_center"][key])
                )
            )
            if np.asarray(current_pre_center[key]).size
            else 0.0
            for key in current_pre_center
        )
    )
    if len(changes) != int(c["width_total"]):
        raise RuntimeError("target change map disagrees with frozen transaction")
    delta_pre, delta_output, delta_seconds = device_delta(
        c, lane, program, target_frames, changes
    )
    U, preelimination_seconds = device_preelimination(
        c, changes, positions, delta_pre
    )

    objective_started = time.monotonic()
    first_output_device = torch.as_tensor(
        c["first_output"], dtype=torch.float64, device="cuda"
    )
    target_output_device = first_output_device + torch.matmul(delta_output, U)
    C_row = torch.as_tensor(
        c["C"][c["rival"]], dtype=torch.float64, device="cuda"
    )
    objective_device = torch.matmul(C_row, target_output_device)
    active_output_center = np.asarray(current_output_center, dtype=np.float64)
    objective_center = float(
        c["C"][c["rival"]] @ active_output_center
        - c["thresholds"][c["rival"]]
    )
    objective_coeff = objective_device.detach().cpu().numpy()
    torch.cuda.synchronize()
    objective_seconds = time.monotonic() - objective_started

    h = make_highs()
    n = int(c["input_rows"].size)
    require_ok(
        h.addCols(
            n,
            -objective_coeff,
            np.asarray(c["factor_lower"], dtype=np.float64),
            np.asarray(c["factor_upper"], dtype=np.float64),
            0,
            np.zeros(n + 1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        ),
        "addCols",
    )
    factor_lower = torch.as_tensor(
        c["factor_lower"], dtype=torch.float64, device="cuda"
    )
    factor_upper = torch.as_tensor(
        c["factor_upper"], dtype=torch.float64, device="cuda"
    )
    frozen_keep = np.asarray(c["keep"], dtype=bool)
    frozen_A = c["screened_A"].tocsr(copy=False)
    frozen_b = np.asarray(c["screened_b"], dtype=np.float64)
    frozen_cursor = 0
    keep_digest = hashlib.sha256()
    matrix_indices_equal = True
    matrix_data_bitwise_equal = True
    rhs_bitwise_equal = True
    matrix_max_abs_difference = 0.0
    objective_bitwise_equal = bool(
        np.array_equal(objective_coeff, np.asarray(c["objective_coeff"]))
    )
    objective_max_abs_difference = float(
        np.max(np.abs(objective_coeff - np.asarray(c["objective_coeff"])))
    )
    total_rows = 0
    logical_total_nnz = 0
    loaded_total_nnz = 0
    full_offset = 0
    keep_bitwise_equal = True
    deleted_tiny_nnz = 0
    maximum_rhs_relaxation = 0.0

    torch.cuda.synchronize()
    model_started = time.monotonic()
    for layer in c["order"]:
        layer_id = int(layer.id)
        original = c["original_frames"].get(layer_id)
        if original is None or not original.exact.size:
            continue
        first_G = np.asarray(c["first_pre"][layer_id], dtype=np.float64)
        selected = np.asarray(c["target_assign"][layer_id], dtype=bool)
        influence = delta_pre[layer_id]
        for start in range(0, int(original.exact.size), ROW_CHUNK):
            stop = min(int(original.exact.size), start + ROW_CHUNK)
            first_chunk = torch.as_tensor(
                first_G[start:stop], dtype=torch.float64, device="cuda"
            )
            target_G = first_chunk + torch.matmul(influence[start:stop], U)
            target_center = torch.as_tensor(
                current_pre_center[layer_id][start:stop],
                dtype=torch.float64,
                device="cuda",
            )
            orientation = torch.as_tensor(
                np.where(selected[start:stop], -1.0, 1.0),
                dtype=torch.float64,
                device="cuda",
            )
            oriented = target_G * orientation.reshape(-1, 1)
            rhs = -orientation * target_center
            contribution = oriented * torch.where(
                oriented >= 0.0,
                factor_upper.reshape(1, -1),
                factor_lower.reshape(1, -1),
            )
            keep_device = torch.sum(contribution, dim=1) > rhs
            keep = keep_device.detach().cpu().numpy().astype(bool, copy=False)
            keep_digest.update(keep.tobytes())
            expected_keep = frozen_keep[
                full_offset + start : full_offset + stop
            ]
            chunk_keep_equal = bool(np.array_equal(keep, expected_keep))
            keep_bitwise_equal &= chunk_keep_equal
            if not chunk_keep_equal:
                matrix_indices_equal = False
                matrix_data_bitwise_equal = False
                rhs_bitwise_equal = False
            if np.any(keep):
                kept_dense = oriented[keep_device].detach().cpu().numpy()
                kept_rhs = rhs[keep_device].detach().cpu().numpy()
                logical_chunk = sp.csr_matrix(kept_dense)
                chunk, loaded_rhs, deleted, relaxation = filter_tiny_rows(
                    kept_dense,
                    kept_rhs,
                    np.asarray(c["factor_lower"], dtype=np.float64),
                    np.asarray(c["factor_upper"], dtype=np.float64),
                )
                deleted_tiny_nnz += deleted
                maximum_rhs_relaxation = max(maximum_rhs_relaxation, relaxation)
                add_status = h.addRows(
                        chunk.shape[0],
                        np.full(chunk.shape[0], -np.inf),
                        loaded_rhs,
                        int(chunk.nnz),
                        chunk.indptr.astype(np.int32, copy=False),
                        chunk.indices.astype(np.int32, copy=False),
                        chunk.data,
                    )
                require_ok(add_status, "chunk addRows")
                if oracle and np.array_equal(keep, expected_keep):
                    expected = frozen_A[
                        frozen_cursor : frozen_cursor + logical_chunk.shape[0]
                    ].tocsr(copy=False)
                    expected_rhs = frozen_b[
                        frozen_cursor : frozen_cursor + logical_chunk.shape[0]
                    ]
                    matrix_indices_equal &= bool(
                        np.array_equal(logical_chunk.indptr, expected.indptr)
                        and np.array_equal(logical_chunk.indices, expected.indices)
                    )
                    if logical_chunk.shape == expected.shape and matrix_indices_equal:
                        matrix_data_bitwise_equal &= bool(
                            np.array_equal(logical_chunk.data, expected.data)
                        )
                        if logical_chunk.data.size:
                            matrix_max_abs_difference = max(
                                matrix_max_abs_difference,
                                float(
                                    np.max(
                                        np.abs(logical_chunk.data - expected.data)
                                    )
                                ),
                            )
                    else:
                        matrix_data_bitwise_equal = False
                    rhs_bitwise_equal &= bool(np.array_equal(kept_rhs, expected_rhs))
                frozen_cursor += int(logical_chunk.shape[0])
                total_rows += int(logical_chunk.shape[0])
                logical_total_nnz += int(logical_chunk.nnz)
                loaded_total_nnz += int(chunk.nnz)
        full_offset += int(original.exact.size)
    torch.cuda.synchronize()
    model_seconds = time.monotonic() - model_started
    if h.getNumRow() != total_rows or h.getNumCol() != n:
        raise RuntimeError("HiGHS post-load row/column count mismatch")
    if h.getNumNz() != loaded_total_nnz:
        raise RuntimeError("HiGHS post-load nnz count mismatch")

    solve_started = time.monotonic()
    run_status = h.run()
    solve_seconds = time.monotonic() - solve_started
    readback_started = time.monotonic()
    model_status = h.getModelStatus()
    solution = h.getSolution()
    factors = np.asarray(solution.col_value, dtype=np.float64)
    row_values = np.asarray(solution.row_value, dtype=np.float64)
    readback_seconds = time.monotonic() - readback_started
    success = bool(
        run_status == highspy.HighsStatus.kOk
        and model_status == highspy.HighsModelStatus.kOptimal
        and solution.value_valid
    )
    if not success:
        raise RuntimeError(f"proposed transaction failed: {model_status}")
    margin = float(objective_center - h.getObjectiveValue())
    total_seconds = time.monotonic() - total_started
    result = {
        "total_seconds": total_seconds,
        "program_build_seconds": float(program.build_seconds),
        "frame_seconds": frame_seconds,
        "center_seconds": current_center_seconds,
        "center_source": (
            "frozen_centers_projected"
        ),
        "delta_seconds": delta_seconds,
        "preelimination_seconds": preelimination_seconds,
        "objective_seconds": objective_seconds,
        "chunk_screen_load_seconds": model_seconds,
        "solve_seconds": solve_seconds,
        "readback_seconds": readback_seconds,
        "solve_readback_seconds": solve_seconds + readback_seconds,
        "margin": margin,
        "rows": total_rows,
        "nnz": logical_total_nnz,
        "loaded_nnz": loaded_total_nnz,
        "simplex_iterations": int(h.getInfo().simplex_iteration_count),
        "factors": factors,
        "row_values_count": int(row_values.size),
        "program_cuda_bytes": int(program.cuda_bytes),
        "deleted_tiny_nnz": deleted_tiny_nnz,
        "maximum_rhs_relaxation": maximum_rhs_relaxation,
    }
    if oracle:
        result["streaming_oracle"] = {
            "keep_sha256": keep_digest.hexdigest(),
            "keep_bitwise_equal": keep_bitwise_equal,
            "rows_equal": total_rows == int(frozen_A.shape[0]),
            "nnz_equal": logical_total_nnz == int(frozen_A.nnz),
            "loaded_nnz_postcheck_equal": h.getNumNz() == loaded_total_nnz,
            "matrix_indices_equal": matrix_indices_equal,
            "matrix_data_bitwise_equal": matrix_data_bitwise_equal,
            "matrix_max_abs_difference": matrix_max_abs_difference,
            "rhs_bitwise_equal": rhs_bitwise_equal,
            "objective_bitwise_equal": objective_bitwise_equal,
            "objective_max_abs_difference": objective_max_abs_difference,
            "active_pre_center_bitwise_equal": pre_center_bitwise_equal,
            "active_pre_center_max_abs_difference": pre_center_max_abs_difference,
            "active_output_center_bitwise_equal": bool(
                np.array_equal(current_output_center, c["target_output_center"])
            ),
            "active_output_center_max_abs_difference": float(
                np.max(
                    np.abs(
                        np.asarray(current_output_center)
                        - np.asarray(c["target_output_center"])
                    )
                )
            ),
            "active_center_source": "frozen_centers_projected",
            "frozen_rows_consumed": frozen_cursor,
        }
    del h, program, delta_pre, delta_output, U, target_output_device
    return result


def strip_arrays(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: item for key, item in value.items() if key != "factors"}


def run_group(function, lanes: Sequence[device.Lane]):
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        values = tuple(pool.map(function, lanes))
    return time.monotonic() - started, values


def main() -> None:
    c, frozen_result = device.load_context()
    lanes = tuple(device.clone_lane(c) for _ in range(WORKERS))

    oracle_current = current_transaction(c, lanes[0])
    # Independent audit closed the augmented-constant center route: selected
    # CSR omits constant contributions and its reduction order is not the
    # retained centers(projected) semantics.  The only timed route is legacy.
    oracle_proposed = proposed_transaction(
        c, lanes[0], oracle=True, use_augmented_center=False
    )
    current_factors = np.asarray(oracle_current["factors"])
    proposed_factors = np.asarray(oracle_proposed["factors"])
    current_terminal = current_impl.terminal(c, current_factors)
    proposed_terminal = current_impl.terminal(c, proposed_factors)
    mechanical = {
        "current_vs_frozen_margin_abs_difference": abs(
            float(oracle_current["margin"]) - float(c["candidate_margin"])
        ),
        "proposed_vs_frozen_margin_abs_difference": abs(
            float(oracle_proposed["margin"]) - float(c["candidate_margin"])
        ),
        "current_vs_frozen_factor_max_abs_difference": float(
            np.max(np.abs(current_factors - np.asarray(c["solved"].x)))
        ),
        "proposed_vs_frozen_factor_max_abs_difference": float(
            np.max(np.abs(proposed_factors - np.asarray(c["solved"].x)))
        ),
        "current_vs_proposed_factor_max_abs_difference": float(
            np.max(np.abs(current_factors - proposed_factors))
        ),
        "current_terminal": current_terminal,
        "proposed_terminal": proposed_terminal,
        "both_terminal_verified": bool(
            current_terminal["verified"] and proposed_terminal["verified"]
        ),
        "fraction_margin_lower_abs_difference": abs(
            float(current_terminal["fraction_margin_lower"])
            - float(proposed_terminal["fraction_margin_lower"])
        ),
        "streaming_matrix": oracle_proposed["streaming_oracle"],
    }
    mechanical_pass = bool(
        mechanical["both_terminal_verified"]
        and oracle_proposed["rows"] == oracle_current["rows"]
        and oracle_proposed["streaming_oracle"]["keep_bitwise_equal"]
        and oracle_proposed["streaming_oracle"]["matrix_indices_equal"]
    )
    if not mechanical_pass:
        raise RuntimeError("mechanical matrix/solution/terminal oracle failed")

    current_single = lambda: current_transaction(c, lanes[0])
    proposed_single = lambda: proposed_transaction(
        c, lanes[0], use_augmented_center=False
    )
    current_group = lambda lane: current_transaction(c, lane)
    proposed_group = lambda lane: proposed_transaction(
        c, lane, use_augmented_center=False
    )

    # One fixed warmup for each path and concurrency shape.
    current_single()
    proposed_single()
    run_group(current_group, lanes)
    run_group(proposed_group, lanes)

    records = []
    for pair in range(PAIRS):
        order = ("current", "proposed") if pair % 2 == 0 else ("proposed", "current")
        record: Dict[str, Any] = {"pair": pair, "order": list(order)}
        for name in order:
            if name == "current":
                single = current_single()
                group_wall, group_values = run_group(current_group, lanes)
            else:
                single = proposed_single()
                group_wall, group_values = run_group(proposed_group, lanes)
            record[f"{name}_single"] = strip_arrays(single)
            record[f"{name}_group_wall_seconds"] = group_wall
            record[f"{name}_group_requests"] = [strip_arrays(v) for v in group_values]
        record["single_saving_seconds"] = (
            record["current_single"]["total_seconds"]
            - record["proposed_single"]["total_seconds"]
        )
        record["group_saving_seconds"] = (
            record["current_group_wall_seconds"]
            - record["proposed_group_wall_seconds"]
        )
        records.append(record)

    proposed_single_values = [
        item["proposed_single"]["total_seconds"] for item in records
    ]
    proposed_group_values = [
        item["proposed_group_wall_seconds"] for item in records
    ]
    single_median = statistics.median(proposed_single_values)
    group_median = statistics.median(proposed_group_values)
    single_pass = bool(single_median <= SINGLE_BUDGET_SECONDS)
    group_pass = bool(group_median <= FOUR_BUDGET_SECONDS)
    hard_pass = bool(single_pass and group_pass and mechanical_pass)
    status = (
        "COMPONENT_SENTINEL_PASSES_REOPEN_NEGATIVE_AUX"
        if hard_pass
        else "STOP_LOSS_TRANSACTION_BUDGET_OR_MECHANICAL_FAILURE"
    )
    artifact = {
        "schema": "act.scratch.phase_projection_chunked_target_transaction_iid2.v1",
        "created_at": "2026-08-14",
        "status": status,
        "audit_complete": True,
        "formal_fixed400_unchanged": 59,
        "hard_gate": {
            "single_budget_seconds": SINGLE_BUDGET_SECONDS,
            "single_observed_median_seconds": single_median,
            "single_pass": single_pass,
            "four_concurrent_budget_seconds": FOUR_BUDGET_SECONDS,
            "four_concurrent_observed_median_seconds": group_median,
            "four_concurrent_pass": group_pass,
            "mechanical_pass": mechanical_pass,
            "pass": hard_pass,
            "failure_means": "stop without aux, threshold tuning, or another representation",
        },
        "method": {
            "pairs": PAIRS,
            "workers": WORKERS,
            "alternating_order": True,
            "warmup_single_per_path": 1,
            "warmup_group_per_path": 1,
            "row_chunk": ROW_CHUNK,
            "request_local_device_program_build_included_in_every_proposed_timing": True,
            "request_local_highs_load_solve_readback_included": True,
            "target_center_second_pass_used": True,
            "active_center_source": "frozen_centers_projected",
            "augmented_constant_column_used": False,
            "augmented_constant_column_status": "RETIRED_STRUCTURALLY_NOT_EQUIVALENT",
            "target_pre_host_materialized": False,
            "target_output_host_materialized": False,
            "blocks_host_materialized": False,
            "A_host_materialized": False,
            "full_promotion_gate_claimed": False,
        },
        "mechanical_oracle": mechanical,
        "center_semantics": {
            "timed_center_source": "frozen_centers_projected",
            "pre_center_bitwise_equal": oracle_proposed["streaming_oracle"][
                "active_pre_center_bitwise_equal"
            ],
            "pre_center_max_abs_difference": oracle_proposed["streaming_oracle"][
                "active_pre_center_max_abs_difference"
            ],
            "output_center_bitwise_equal": oracle_proposed["streaming_oracle"][
                "active_output_center_bitwise_equal"
            ],
            "output_center_max_abs_difference": oracle_proposed[
                "streaming_oracle"
            ]["active_output_center_max_abs_difference"],
            "augmented_constant_column": "retired before timed gate by structural audit",
        },
        "oracle_current": strip_arrays(oracle_current),
        "oracle_proposed": strip_arrays(oracle_proposed),
        "measurements": records,
        "restrictions": restrictions(),
        "sources": {
            "scratch": str(Path(__file__).resolve().relative_to(ROOT)),
            "scratch_sha256": sha256(Path(__file__).resolve()),
            "device_program_scratch": str(device.ROOT / Path(device.__file__).name),
            "device_program_scratch_sha256": sha256(Path(device.__file__).resolve()),
            "current_transaction_dependency": str(ONE_PATH),
            "current_transaction_dependency_sha256": sha256(ONE_PATH),
            "phase_source_sha256": sha256(Path(phase.__file__).resolve()),
            "live_source_sha256": sha256(Path(live.__file__).resolve()),
            "scipy_version_and_bundled_highs_path": scipy.__version__,
            "highspy_version": highspy.Highs().version(),
        },
        "loader_safety": {
            "small_matrix_value": HIGHS_SMALL,
            "logical_screen_uses_unfiltered_rows": True,
            "deleted_coefficient_rule": "0 < abs(a) <= 1e-12",
            "rhs_relaxation": "b + exact Fraction sum(max(a*l,a*u)), rounded toward +inf",
            "objective_coefficients_filtered": False,
            "every_addRows_requires_strict_kOk": True,
            "post_load_row_column_nnz_check": True,
        },
        "production_files_modified": [],
    }
    ARTIFACT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
