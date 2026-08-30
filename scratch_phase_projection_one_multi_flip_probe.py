#!/usr/bin/env python3
"""Disposable one-shot multi-flip diagnostic for the frozen 59-path.

The selector is pre-registered and has no parameter menu: after the frozen
path's first exact-cell LP, flip every screened phase row whose HiGHS upper-
inequality marginal is strictly negative and whose residual is within the
already-frozen primal feasibility tolerance.  Rebuild that one cell using the
same triangular phase-delta representation, solve exactly one more LP, and
stop.  External benchmark labels are not read by this program.

No input sampling, ONNX point execution, PGD, BaB/splitting, backward bounds,
or dual tightening is performed.  LP marginals select a candidate cell only
and have no verdict authority.  Any positive candidate is subjected to the
unchanged raw-BOX, local zero-width interval, and stored-binary64 Fraction
property terminal.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import torch

from act.back_end.analyze import analyze
from act.back_end.core import ConSet, Fact
from act.back_end.hybridz_tf import forward_exact_relu_live_row_stream_candidate as live
from act.back_end.hybridz_tf import forward_exact_relu_phase_projection_candidate as phase
from act.back_end.transfer_functions import set_solver_mode, set_transfer_function_mode
from act.back_end.verifier import (
    add_all_input_specs,
    find_entry_layer_id,
    gather_input_spec_layers,
    get_input_ids,
    seed_from_input_specs,
)
from act.front_end.model_synthesis import synthesize_models_from_specs
from act.front_end.vnnlib_loader.create_specs import create_specs_from_paths
from act.pipeline.verification.torch2act import TorchToACT
from act.util.device_manager import initialize_device


ONNX = os.environ["ACT_PHASE_PROJECTION_ONNX"]
VNNLIB = os.environ["ACT_PHASE_PROJECTION_VNNLIB"]
CATEGORY = os.environ["ACT_PHASE_PROJECTION_CATEGORY"]
CASE = os.environ["ACT_PHASE_PROJECTION_CASE"]


def fail(reason: str, **extra) -> None:
    receipt = {
        "schema": "act.scratch.phase_projection_one_multi_flip.v1",
        "case": CASE,
        "status": "STOP_LOSS",
        "reason": reason,
        "production_modified": False,
        "external_labels_read_by_rule": False,
        "restrictions": restrictions(),
    }
    receipt.update(extra)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False))
    raise SystemExit(0)


def restrictions():
    return {
        "input_sampling_used": False,
        "onnx_point_execution_used": False,
        "pgd_used": False,
        "bab_or_split_used": False,
        "backward_bounds_used": False,
        "dual_tightening_used": False,
        "candidate_or_marginal_has_authority": False,
        "phase_updates": 1,
        "phase_or_property_retries": 0,
        "fallback_menu": False,
    }


def capture_negative_path(net, entry, before, after):
    captured = {}
    target_code = phase.build_forward_exact_relu_phase_projection_candidate.__code__

    def tracer(frame, event, arg):
        if frame.f_code is target_code and event == "exception":
            exc = arg[1]
            if "float screen rejected candidate margin=" in str(exc):
                captured.update(frame.f_locals)
        return tracer

    started = time.monotonic()
    sys.settrace(tracer)
    try:
        result = phase.build_forward_exact_relu_phase_projection_candidate(
            net,
            int(entry),
            before,
            after,
            deadline=time.monotonic() + 30.0,
            lp_time_limit=30.0,
        )
    except phase.ExactReLUPhaseProjectionUnknown as exc:
        error = str(exc)
        result = None
    finally:
        sys.settrace(None)
    elapsed = time.monotonic() - started
    if result is not None:
        fail("frozen path unexpectedly already produced a terminal candidate", frozen_seconds=elapsed)
    if not captured:
        fail("frozen path did not reach a negative-margin LP", frozen_reason=error, frozen_seconds=elapsed)
    return captured, error, elapsed


def select_all_active_improving_phase_rows(c):
    solved = c["solved"]
    keep = np.asarray(c["keep"], dtype=bool)
    screened_b = np.asarray(c["screened_b"], dtype=np.float64)
    try:
        residual = np.asarray(solved.ineqlin.residual, dtype=np.float64)
        marginal = np.asarray(solved.ineqlin.marginals, dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as exc:
        fail("HiGHS did not expose finite inequality residuals and marginals", detail=str(exc))
    if (
        residual.shape != screened_b.shape
        or marginal.shape != screened_b.shape
        or not np.all(np.isfinite(residual))
        or not np.all(np.isfinite(marginal))
    ):
        fail("HiGHS inequality diagnostic shape or finiteness contract failed")
    tolerance = float(phase._SOLVER_TOLERANCE)
    tight = residual <= tolerance * (1.0 + np.abs(screened_b))
    eligible_screened = tight & (marginal < 0.0)
    screened_to_full = np.flatnonzero(keep)
    eligible_full = screened_to_full[np.flatnonzero(eligible_screened)]

    phase_rows = []
    for layer in c["order"]:
        layer_id = int(layer.id)
        original = c["original_frames"].get(layer_id)
        if original is None or not original.exact.size:
            continue
        for position, relu_row in enumerate(np.asarray(original.stream_rows, dtype=np.int64)):
            phase_rows.append((layer_id, position, int(relu_row)))
    if len(phase_rows) != keep.size:
        fail("flattened phase-row map disagrees with frozen LP")
    selected = [phase_rows[int(index)] for index in eligible_full]
    return selected, residual, marginal, tight, eligible_screened


def rebuild_one_cell(c, assignments):
    update_started = time.monotonic()
    (
        target_assign,
        target_pre_center,
        target_output_center,
        target_frames,
    ) = c["centers"](assignments)
    center_seconds = time.monotonic() - update_started

    changes = []
    positions = {}
    for layer in c["order"]:
        layer_id = int(layer.id)
        original = c["original_frames"].get(layer_id)
        if original is None or not original.exact.size:
            continue
        rows = np.asarray(original.stream_rows, dtype=np.int64)
        positions[layer_id] = {int(row): pos for pos, row in enumerate(rows)}
        for position in np.flatnonzero(target_assign[layer_id] != c["first_assign"][layer_id]):
            changes.append(
                (
                    layer_id,
                    int(rows[position]),
                    bool(c["first_assign"][layer_id][position]),
                    bool(target_assign[layer_id][position]),
                )
            )

    width_total = len(changes)
    change_index = {
        (layer_id, row): index
        for index, (layer_id, row, _base, _target) in enumerate(changes)
    }
    target_active_live = {}
    changed_by_layer = {}
    exact_device = {}
    for layer in c["order"]:
        layer_id = int(layer.id)
        if layer_id not in target_frames:
            continue
        target_active_live[layer_id] = torch.as_tensor(
            np.intersect1d(
                c["live_rows"][layer_id],
                target_frames[layer_id].active,
                assume_unique=True,
            ),
            dtype=torch.int64,
            device="cuda",
        )
        changed_by_layer[layer_id] = [
            (row, change_index[(layer_id, row)])
            for local_layer, row, _base, _target in changes
            if local_layer == layer_id
        ]
        exact_device[layer_id] = torch.as_tensor(
            c["original_frames"][layer_id].stream_rows,
            dtype=torch.int64,
            device="cuda",
        )

    torch.cuda.synchronize()
    delta_started = time.monotonic()
    delta_pre_parts = {layer_id: [] for layer_id in c["original_frames"]}
    delta_output_parts = []
    for start in range(0, width_total, 64):
        stop = min(width_total, start + 64)
        width = stop - start
        values = {}
        for layer in c["order"]:
            layer_id = int(layer.id)
            kind = phase._oh._kind(layer.kind)
            predecessors = tuple(int(value) for value in c["net"].preds.get(layer_id, []))
            if kind == "INPUT":
                values[layer_id] = torch.zeros(
                    (len(layer.out_vars), width), dtype=torch.float64, device="cuda"
                )
            elif kind in {"INPUT_SPEC", "FLATTEN", "ASSERT"}:
                values[layer_id] = values[predecessors[0]]
            elif kind in {"CONV2D", "DENSE"}:
                selected_value = live._ordered_csr_dense(
                    c["device_matrices"][layer_id], values[predecessors[0]]
                )
                value = torch.zeros(
                    (len(layer.out_vars), width), dtype=torch.float64, device="cuda"
                )
                if c["device_rows"][layer_id].numel():
                    value[c["device_rows"][layer_id]] = selected_value
                values[layer_id] = value
            elif kind == "SCALE":
                parameter = torch.tensor(
                    c["pointwise"][layer_id], dtype=torch.float64, device="cuda"
                ).reshape(-1, 1)
                values[layer_id] = values[predecessors[0]] * parameter
            elif kind == "BIAS":
                values[layer_id] = values[predecessors[0]]
            elif kind == "ADD":
                values[layer_id] = values[predecessors[0]] + values[predecessors[1]]
            elif kind == "RELU":
                source = values[predecessors[0]]
                delta_pre_parts[layer_id].append(
                    source[exact_device[layer_id]].detach().cpu().numpy()
                )
                value = torch.zeros_like(source)
                active = target_active_live[layer_id]
                if active.numel():
                    value[active] = source[active]
                for row, column in changed_by_layer[layer_id]:
                    value[row] = 0.0
                    if start <= column < stop:
                        value[row, column - start] = 1.0
                values[layer_id] = value
            else:
                fail("unsupported graph kind during one update", graph_kind=kind)
        delta_output_parts.append(values[int(c["assert_layer"].id)].detach().cpu().numpy())
    torch.cuda.synchronize()
    delta_pre = {
        layer_id: (
            np.concatenate(parts, axis=1)
            if parts
            else np.empty((c["original_frames"][layer_id].exact.size, 0), dtype=np.float64)
        )
        for layer_id, parts in delta_pre_parts.items()
    }
    delta_output = (
        np.concatenate(delta_output_parts, axis=1)
        if delta_output_parts
        else np.empty((c["output_width"], 0), dtype=np.float64)
    )
    delta_seconds = time.monotonic() - delta_started

    expansion_started = time.monotonic()
    U = phase._triangular_input_expansion(
        changes,
        positions,
        c["first_pre"],
        delta_pre,
        input_width=int(c["input_rows"].size),
    )
    target_pre = {
        layer_id: np.asarray(c["first_pre"][layer_id], dtype=np.float64)
        + np.asarray(delta_pre[layer_id] @ U, dtype=np.float64)
        for layer_id in c["first_pre"]
    }
    target_output = np.asarray(c["first_output"], dtype=np.float64) + np.asarray(
        delta_output @ U, dtype=np.float64
    )
    expansion_seconds = time.monotonic() - expansion_started
    return {
        "assign": target_assign,
        "pre_center": target_pre_center,
        "output_center": target_output_center,
        "pre": target_pre,
        "output": target_output,
        "changes_from_first": width_total,
        "center_seconds": center_seconds,
        "delta_seconds": delta_seconds,
        "expansion_seconds": expansion_seconds,
    }


def solve_cell(c, rebuilt):
    model_started = time.monotonic()
    blocks = []
    rhs = []
    total_phases = 0
    for layer in c["order"]:
        layer_id = int(layer.id)
        original = c["original_frames"].get(layer_id)
        if original is None or not original.exact.size:
            continue
        matrix = sp.csr_matrix(rebuilt["pre"][layer_id])
        selected = rebuilt["assign"][layer_id]
        blocks.append(matrix.multiply(np.where(selected, -1.0, 1.0)[:, None]).tocsr())
        center = rebuilt["pre_center"][layer_id]
        rhs.append(np.where(selected, center, -center))
        total_phases += int(original.exact.size)
    A = sp.vstack(blocks, format="csr")
    b = np.ascontiguousarray(np.concatenate(rhs), dtype=np.float64)
    row_max = phase._csr_box_upper(A, c["factor_lower"], c["factor_upper"])
    keep = row_max > b
    screened_A = A[keep].tocsr()
    screened_b = b[keep]
    objective_coeff = np.asarray(c["C"][[c["rival"]]] @ rebuilt["output"], dtype=np.float64).reshape(-1)
    objective_center = float(
        c["C"][c["rival"]] @ rebuilt["output_center"] - c["thresholds"][c["rival"]]
    )
    model_seconds = time.monotonic() - model_started
    lp_started = time.monotonic()
    solved = linprog(
        -objective_coeff,
        A_ub=screened_A,
        b_ub=screened_b,
        bounds=c["factor_bounds"],
        method="highs-ds",
        options={
            "presolve": False,
            "time_limit": 30.0,
            "primal_feasibility_tolerance": phase._SOLVER_TOLERANCE,
        },
    )
    lp_seconds = time.monotonic() - lp_started
    if not solved.success or solved.x is None:
        return {
            "success": False,
            "message": str(solved.message),
            "status": int(solved.status),
            "rows": int(screened_A.shape[0]),
            "nnz": int(screened_A.nnz),
            "model_seconds": model_seconds,
            "lp_seconds": lp_seconds,
        }
    factors = np.asarray(solved.x, dtype=np.float64).reshape(-1)
    margin = float(objective_center + objective_coeff @ factors)
    consistent = 0
    inconsistent = 0
    max_phase_sign_violation = 0.0
    for layer in c["order"]:
        layer_id = int(layer.id)
        original = c["original_frames"].get(layer_id)
        if original is None or not original.exact.size:
            continue
        value = rebuilt["pre_center"][layer_id] + rebuilt["pre"][layer_id] @ factors
        selected = rebuilt["assign"][layer_id]
        ok = (selected & (value >= 0.0)) | (~selected & (value <= 0.0))
        signed_value = np.where(selected, value, -value)
        if np.any(signed_value < 0.0):
            max_phase_sign_violation = max(
                max_phase_sign_violation,
                float(np.max(-signed_value[signed_value < 0.0])),
            )
        consistent += int(np.count_nonzero(ok))
        inconsistent += int(np.count_nonzero(~ok))
    return {
        "success": True,
        "factors": factors,
        "margin": margin,
        "rows": int(screened_A.shape[0]),
        "nnz": int(screened_A.nnz),
        "model_seconds": model_seconds,
        "lp_seconds": lp_seconds,
        "phase_consistent": consistent,
        "phase_inconsistent": inconsistent,
        "max_phase_sign_violation": max_phase_sign_violation,
        "all_phase_sign_violations_within_frozen_primal_tolerance": bool(
            max_phase_sign_violation <= phase._SOLVER_TOLERANCE
        ),
    }


def terminal(c, factors):
    decoded = np.asarray(c["raw_lower"], dtype=np.float64).copy()
    for column, raw_row in enumerate(c["input_rows"]):
        row = int(raw_row)
        exact_value = Fraction.from_float(float(c["input_center"][row]))
        exact_value += Fraction.from_float(float(c["input_radius"][row])) * Fraction.from_float(
            float(factors[column])
        )
        decoded[row] = float(exact_value)
    in_box = bool(
        np.all(np.isfinite(decoded))
        and np.all(decoded >= c["raw_lower"])
        and np.all(decoded <= c["raw_upper"])
    )
    if not in_box:
        return {"raw_box": False, "verified": False, "seconds": 0.0}
    started = time.monotonic()
    lower, upper = phase._singleton_interval_forward(
        c["net"],
        c["order"],
        c["affines"],
        decoded.reshape(c["input_shape"]),
        c["output_layer_id"],
        pointwise=c["pointwise"],
        deadline=None,
    )
    exact = phase._exact_singleton_margin_lower(
        c["C"][c["rival"]], c["thresholds"][c["rival"]], lower, upper
    )
    return {
        "raw_box": True,
        "zero_width_interval": True,
        "fraction_margin_lower": float(exact),
        "verified": bool(exact > 0),
        "seconds": time.monotonic() - started,
    }


def main():
    initialize_device(device="cuda", dtype="float64")
    set_solver_mode("hybridz")
    set_transfer_function_mode("interval")
    total_started = time.monotonic()
    sr = create_specs_from_paths(ONNX, VNNLIB, category=CATEGORY)
    vm = next(iter(synthesize_models_from_specs([sr]).values()))
    net = TorchToACT(vm).run()
    entry = find_entry_layer_id(net)
    specs = gather_input_spec_layers(net)
    seed = seed_from_input_specs(specs)
    fact = Fact(bounds=seed, cons=ConSet())
    add_all_input_specs(fact.cons, get_input_ids(net), specs)
    before, after, _ = analyze(net, entry, fact)

    c, frozen_error, frozen_seconds = capture_negative_path(net, entry, before, after)
    c["net"] = net
    selected, residual, marginal, tight, eligible_screened = select_all_active_improving_phase_rows(c)
    if not selected:
        fail(
            "the fixed selector found no active improving phase row",
            frozen_margin=float(c["candidate_margin"]),
            frozen_seconds=frozen_seconds,
        )

    assignments = {key: np.asarray(value, dtype=bool).copy() for key, value in c["target_assign"].items()}
    for layer_id, position, _relu_row in selected:
        assignments[layer_id][position] = ~assignments[layer_id][position]

    update_started = time.monotonic()
    rebuilt = rebuild_one_cell(c, assignments)
    solved = solve_cell(c, rebuilt)
    update_seconds = time.monotonic() - update_started
    old_margin = float(c["candidate_margin"])
    new_margin = solved.get("margin")
    strict_improvement = bool(new_margin is not None and new_margin > old_margin)
    terminal_receipt = None
    if solved["success"] and strict_improvement and new_margin > 0.0:
        terminal_receipt = terminal(c, solved["factors"])

    magnitudes = -marginal[eligible_screened]
    by_layer = {}
    for layer_id, _position, _relu_row in selected:
        by_layer[str(layer_id)] = by_layer.get(str(layer_id), 0) + 1
    selected_rows_payload = [
        [int(layer_id), int(position), int(row)]
        for layer_id, position, row in selected
    ]
    selected_rows_sha256 = hashlib.sha256(
        json.dumps(
            selected_rows_payload,
            sort_keys=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    receipt = {
        "schema": "act.scratch.phase_projection_one_multi_flip.v1",
        "case": CASE,
        "status": (
            "TERMINAL_VERIFIED"
            if terminal_receipt and terminal_receipt["verified"]
            else "UPDATED_POSITIVE_TERMINAL_REJECTED"
            if terminal_receipt
            else "UPDATED_NEGATIVE"
            if solved["success"] and strict_improvement
            else "STOP_LOSS"
        ),
        "rule": {
            "name": "flip_all_tight_strictly_negative_marginal_phase_rows_once",
            "tightness": "residual <= frozen_primal_tolerance * (1 + abs(rhs))",
            "marginal": "strictly < 0; no magnitude threshold",
            "top_k": None,
            "threshold_scan": False,
            "retry": False,
            "strict_margin_improvement_required": True,
        },
        "selected_property_row": int(c["rival"]),
        "frozen_reason": frozen_error,
        "old_margin": old_margin,
        "selected_flip_count": len(selected),
        "selected_by_layer": by_layer,
        "selected_rows_first_ten": [
            {"layer_id": layer_id, "position": position, "relu_row": row}
            for layer_id, position, row in selected[:10]
        ],
        "selected_rows_canonical_sha256": selected_rows_sha256,
        "selected_marginal_magnitude_min": float(np.min(magnitudes)),
        "selected_marginal_magnitude_max": float(np.max(magnitudes)),
        "tight_screened_rows": int(np.count_nonzero(tight)),
        "negative_marginal_screened_rows": int(np.count_nonzero(marginal < 0.0)),
        "new_cell_changes_from_first": rebuilt["changes_from_first"],
        "new_lp": {key: value for key, value in solved.items() if key != "factors"},
        "new_margin": new_margin,
        "margin_gain": None if new_margin is None else float(new_margin - old_margin),
        "strict_improvement": strict_improvement,
        "terminal": terminal_receipt,
        "timing": {
            "frozen_negative_path_seconds": frozen_seconds,
            "update_center_seconds": rebuilt["center_seconds"],
            "update_delta_seconds": rebuilt["delta_seconds"],
            "update_expansion_seconds": rebuilt["expansion_seconds"],
            "update_model_seconds": solved["model_seconds"],
            "update_lp_seconds": solved["lp_seconds"],
            "one_update_total_seconds": update_seconds,
            "end_to_end_seconds": time.monotonic() - total_started,
        },
        "production_modified": False,
        "external_labels_read_by_rule": False,
        "restrictions": restrictions(),
    }
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
