#!/usr/bin/env python3
"""Disposable one-update exact phase-cell projection probe.

No ONNX execution, input sampling, random points, PGD, BaB, backward pass, or
dual tightening is performed.  One verifier-derived analytic box optimum
updates one phase assignment; one continuous input-factor LP is then solved
and replayed against every stored affine phase envelope.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from fractions import Fraction

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import torch

from act.back_end.analyze import analyze
from act.back_end.config import BackendConfig, HybridZConfig
from act.back_end.core import Bounds, ConSet, Fact
from act.back_end.hybridz_tf import forward_exact_relu_live_row_stream_candidate as cand
from act.back_end.hybridz_tf.forward_exact_relu_phase_projection_candidate import (
    ExactReLUPhaseProjectionUnknown,
    build_forward_exact_relu_phase_projection_candidate,
)
from act.back_end.hybridz_tf import operator_hz as oh
from act.back_end.transfer_functions import set_solver_mode, set_transfer_function_mode
from act.back_end.verifier import (
    _ensure_assert_linear_encoding,
    _get_output_layer_id,
    add_all_input_specs,
    find_entry_layer_id,
    gather_input_spec_layers,
    get_assert_layer,
    get_input_ids,
    seed_from_input_specs,
)
from act.front_end.model_synthesis import synthesize_models_from_specs
from act.front_end.vnnlib_loader.create_specs import create_specs_from_paths
from act.pipeline.verification.torch2act import TorchToACT
from act.util.device_manager import initialize_device


ONNX = os.environ.get(
    "ACT_PHASE_PROJECTION_ONNX",
    "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024/onnx/CIFAR100_resnet_medium.onnx",
)
VNNLIB = os.environ.get(
    "ACT_PHASE_PROJECTION_VNNLIB",
    "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024/vnnlib/CIFAR100_resnet_medium_prop_idx_6232_sidx_3020_eps_0.0039.vnnlib",
)
CATEGORY = os.environ.get("ACT_PHASE_PROJECTION_CATEGORY", "cifar100_2024")


def inward_factor_bounds(lb, ub, center, radius, rows, tolerance):
    """Return binary64 factor bounds strictly inside the exact input box."""
    bounds = []
    for row in rows:
        c = Fraction.from_float(float(center[row]))
        r = Fraction.from_float(float(radius[row]))
        if r <= 0:
            raise RuntimeError("active input factor has nonpositive radius")
        exact_lower = (Fraction.from_float(float(lb[row])) - c) / r
        exact_upper = (Fraction.from_float(float(ub[row])) - c) / r
        lower = float(exact_lower)
        upper = float(exact_upper)
        while Fraction.from_float(lower) < exact_lower:
            lower = float(np.nextafter(lower, np.inf))
        while Fraction.from_float(upper) > exact_upper:
            upper = float(np.nextafter(upper, -np.inf))
        guard = 16.0 * tolerance * (1.0 + max(abs(lower), abs(upper)))
        lower = float(np.nextafter(lower + guard, np.inf))
        upper = float(np.nextafter(upper - guard, -np.inf))
        if not (np.isfinite(lower) and np.isfinite(upper) and lower <= upper):
            raise RuntimeError("input factor interval vanished after inward guard")
        bounds.append((lower, upper))
    return bounds


def radius(matrix, center, inherited, name):
    matrix = cand._canonical(sp.csr_matrix(matrix), name=f"{name}.matrix")
    center = np.asarray(center, dtype=np.float64).reshape(-1)
    inherited = np.asarray(inherited, dtype=np.float64).reshape(-1)
    mass = oh._row_l1_upper(matrix, name=f"{name}.l1")
    nnz = np.diff(matrix.indptr).astype(np.float64)
    gamma = np.asarray(
        [oh._gamma_ops(2.0 * float(value) + 2.0, name=f"{name}.gamma") for value in nnz],
        dtype=np.float64,
    )
    arithmetic = oh._inflate_nonnegative(
        gamma * (np.abs(center) + mass),
        4,
        active=(mass > 0.0) | (center != 0.0),
        name=f"{name}.arithmetic",
    )
    return oh._nonnegative_sum_upper(inherited, arithmetic, name=f"{name}.radius")


def interval(matrix, xi, center, inherited, name):
    matrix = cand._canonical(sp.csr_matrix(matrix), name=f"{name}.matrix")
    center = np.asarray(center, dtype=np.float64).reshape(-1)
    rad = radius(matrix, center, inherited, f"{name}.eval")
    value = center + np.asarray(matrix @ xi, dtype=np.float64).reshape(-1)
    lower = np.nextafter(value - rad, -np.inf)
    upper = np.nextafter(value + rad, np.inf)
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise RuntimeError(f"{name} overflow")
    return value, lower, upper, rad


def main() -> None:
    initialize_device(device="cuda", dtype="float64")
    set_solver_mode("hybridz")
    set_transfer_function_mode("interval")
    started = time.monotonic()
    sr = create_specs_from_paths(ONNX, VNNLIB, category=CATEGORY)
    vm = next(iter(synthesize_models_from_specs([sr]).values()))
    net = TorchToACT(vm).run()
    entry = find_entry_layer_id(net)
    specs = gather_input_spec_layers(net)
    seed = seed_from_input_specs(specs)
    if os.environ.get("ACT_PHASE_PROJECTION_VERIFY_ONCE") == "1":
        from act.back_end.verifier import verify_once

        verified_started = time.monotonic()
        verified = verify_once(
            net,
            backend_cfg=BackendConfig(
                solver="hybridz",
                device="cuda",
                dtype="float64",
                timeout=30.0,
                hybridz=HybridZConfig(
                    timeout=20.0,
                    engine="operator_hz_objbound",
                    operator_exact_budget=-1,
                    operator_phase_projection_time_limit=10.0,
                    operator_materialize_add=True,
                    gpu_dual_steps=0,
                    gpu_dual_time_limit=0.0,
                    gpu_dual_row_topk=0,
                ),
            ),
        )[0]
        projection_meta = verified.metadata.get(
            "operator_phase_projection", {}
        )
        print(
            json.dumps(
                {
                    "schema": "act.hybridz.phase_projection_verify_once_probe.v1",
                    "status": str(verified.status),
                    "has_counterexample": verified.counterexample is not None,
                    "phase_projection": projection_meta,
                    "total_seconds": time.monotonic() - verified_started,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return
    fact = Fact(bounds=seed, cons=ConSet())
    add_all_input_specs(fact.cons, get_input_ids(net), specs)
    before, after, _ = analyze(net, entry, fact)
    try:
        promoted = build_forward_exact_relu_phase_projection_candidate(
            net,
            int(entry),
            before,
            after,
            deadline=time.monotonic() + 30.0,
        )
    except ExactReLUPhaseProjectionUnknown as exc:
        print(
            json.dumps(
                {
                    "schema": "act.hybridz.forward_exact_relu_phase_projection_probe.v1",
                    "status": "UNKNOWN",
                    "reason": str(exc),
                    "model": Path(ONNX).name,
                    "spec": Path(VNNLIB).name,
                    "total_seconds": time.monotonic() - started,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return
    print(
        json.dumps(
            {
                "schema": promoted.receipt.schema,
                "status": promoted.receipt.status,
                "model": Path(ONNX).name,
                "spec": Path(VNNLIB).name,
                "selected_property_row": promoted.receipt.selected_property_row,
                "phase_rows": promoted.receipt.phase_rows,
                "replay_consistent": promoted.receipt.replay_consistent,
                "projected_margin_lower": promoted.receipt.projected_margin_lower,
                "singleton_margin_lower": promoted.receipt.singleton_margin_lower,
                "candidate_seconds": promoted.receipt.total_seconds,
                "decoded_input_shape": list(promoted.decoded_input.shape),
                "decoded_input_readonly": not promoted.decoded_input.flags.writeable,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return

    order, by_id = cand._topological(net)
    input_layer = next(layer for layer in order if oh._kind(layer.kind) == "INPUT")
    assert_layer = get_assert_layer(net)
    output_layer_id = _get_output_layer_id(net)
    lb, ub = cand._facts_box(
        after, int(input_layer.id), len(input_layer.out_vars), name="phase_projection.input"
    )
    input_center, input_radius = oh._enclosing_center_radius(
        lb, ub, name="phase_projection.input"
    )
    input_rows = np.flatnonzero(input_radius > 0.0).astype(np.int64)
    solver_tolerance = 1.0e-9
    factor_bounds = inward_factor_bounds(
        lb,
        ub,
        input_center,
        input_radius,
        input_rows,
        solver_tolerance,
    )
    original_frames, _, _ = cand._make_phase_frames(
        order, before, first_continuous_column=int(input_rows.size)
    )
    affines = {}
    for layer in order:
        if oh._kind(layer.kind) in {"CONV2D", "DENSE"}:
            predecessor = cand._preds(net, layer, 1)[0]
            affines[int(layer.id)] = cand._affine_snapshot(
                layer, input_size=len(by_id[predecessor].out_vars)
            )
    live, possible = cand._live_rows(
        net, order, affines, original_frames, input_rows, output_layer_id
    )
    matrices = {}
    for layer_id, snapshot in affines.items():
        predecessor = cand._preds(net, by_id[layer_id], 1)[0]
        matrices[layer_id] = cand._selected_affine_matrix(
            snapshot,
            live[layer_id],
            possible[predecessor],
            name=f"phase_projection.stream[{layer_id}]",
        )
    device_matrices = {key: cand._device_csr(value) for key, value in matrices.items()}
    device_rows = {
        key: torch.as_tensor(value, dtype=torch.int64, device="cuda")
        for key, value in live.items()
    }
    input_mass = oh._nonnegative_sum_upper(
        np.abs(input_center), input_radius, name="phase_projection.input_mass"
    )
    empty = np.zeros(0, dtype=np.int64)
    empty_bool = np.zeros(0, dtype=bool)

    def build_cell(assignments):
        shadows = {}
        pre_shadows = {}
        frames = {}
        selected_map = {}
        for layer in order:
            layer_id = int(layer.id)
            kind = oh._kind(layer.kind)
            if kind == "INPUT":
                shadows[layer_id] = cand._Shadow(
                    input_center.copy(), np.zeros(input_center.size), input_mass
                )
            elif kind in {"INPUT_SPEC", "FLATTEN", "ASSERT"}:
                shadows[layer_id] = shadows[cand._preds(net, layer, 1)[0]]
            elif kind in {"CONV2D", "DENSE"}:
                shadows[layer_id] = cand._gpu_affine_shadow(
                    shadows[cand._preds(net, layer, 1)[0]],
                    affines[layer_id],
                    layer_id=layer_id,
                )
            elif kind == "ADD":
                left, right = cand._preds(net, layer, 2)
                shadows[layer_id] = cand._add_shadow(
                    shadows[left], shadows[right], layer_id=layer_id
                )
            elif kind == "RELU":
                source = shadows[cand._preds(net, layer, 1)[0]]
                pre_shadows[layer_id] = source
                original = original_frames[layer_id]
                selected = (
                    source.center[original.exact] >= 0.0
                    if assignments is None
                    else np.asarray(
                        assignments.get(
                            layer_id,
                            source.center[original.exact] >= 0.0,
                        ),
                        dtype=bool,
                    )
                )
                selected_map[layer_id] = selected.copy()
                active = np.sort(
                    np.concatenate((original.active, original.exact[selected]))
                ).astype(np.int64)
                inactive = np.sort(
                    np.concatenate((original.inactive, original.exact[~selected]))
                ).astype(np.int64)
                frame = cand._PhaseFrame(
                    original.lower,
                    original.upper,
                    active,
                    inactive,
                    empty,
                    empty,
                    empty,
                    np.zeros(0, dtype=np.float64),
                    original.exact.copy(),
                    empty,
                    empty,
                    empty_bool,
                    empty,
                    empty,
                    empty,
                )
                frames[layer_id] = frame
                shadows[layer_id] = cand._relu_shadow(source, frame, layer_id=layer_id)
            else:
                raise RuntimeError(f"unsupported {kind}")
        pre, out, stream_seconds = cand._stream_generators(
            net,
            order,
            frames,
            live,
            device_matrices,
            device_rows,
            input_rows=input_rows,
            input_radius=input_radius,
            n_cont=int(input_rows.size),
            assert_layer=assert_layer,
            deadline=None,
            stage_prefix="phase_projection",
            collect_output=True,
        )
        if out is None:
            raise RuntimeError("missing output stream")
        return selected_map, pre_shadows, pre, shadows[output_layer_id], out, stream_seconds

    first_assign, first_pre_shadows, first_pre, first_out_shadow, first_out, first_s = build_cell(None)
    n_out = len(by_id[output_layer_id].out_vars)
    _ensure_assert_linear_encoding(
        assert_layer, B=1, n_out=n_out, device=torch.device("cuda"), dtype=torch.float64
    )
    C = np.asarray(
        assert_layer.params["C"].detach().cpu().double().numpy().reshape(-1, n_out),
        dtype=np.float64,
    )
    thresholds = np.asarray(
        assert_layer.params["thresholds"].detach().cpu().double().numpy().reshape(-1),
        dtype=np.float64,
    )
    first_output_G = cand._canonical(sp.csr_matrix(first_out), name="phase_projection.first_G")
    first_objective_G = cand._canonical(
        sp.csr_matrix(C) @ first_output_G, name="phase_projection.first_objective"
    )
    first_center = np.asarray(C @ first_out_shadow.center - thresholds, dtype=np.float64)
    first_error = np.asarray(np.abs(C) @ first_out_shadow.error, dtype=np.float64)
    first_upper = first_center + oh._row_l1_upper(
        first_objective_G, name="phase_projection.first_l1"
    ) + first_error
    rival = int(np.argmax(first_upper))
    first_coeff = np.asarray(first_objective_G.getrow(rival).toarray()[0])
    factor_lower = np.asarray([bound[0] for bound in factor_bounds])
    factor_upper = np.asarray([bound[1] for bound in factor_bounds])
    first_xi = np.where(first_coeff >= 0.0, factor_upper, factor_lower)

    projected = {}
    initial_changes = 0
    first_inconsistent = 0
    for layer in order:
        layer_id = int(layer.id)
        original = original_frames.get(layer_id)
        if original is None or not original.exact.size:
            continue
        matrix = cand._canonical(
            sp.csr_matrix(first_pre[layer_id]), name=f"phase_projection.first_pre[{layer_id}]"
        )
        value, lower, upper, _ = interval(
            matrix,
            first_xi,
            first_pre_shadows[layer_id].center[original.exact],
            first_pre_shadows[layer_id].error[original.exact],
            f"phase_projection.first_phase[{layer_id}]",
        )
        if np.any((lower <= 0.0) & (upper >= 0.0)):
            raise RuntimeError("first projection contains an ambiguous phase")
        selected = value >= 0.0
        initial_changes += int(np.count_nonzero(selected != first_assign[layer_id]))
        first_inconsistent += int(np.count_nonzero(selected != first_assign[layer_id]))
        projected[layer_id] = selected

    second_assign, second_pre_shadows, second_pre, second_out_shadow, second_out, second_s = build_cell(projected)
    second_output_G = cand._canonical(sp.csr_matrix(second_out), name="phase_projection.second_G")
    objective_G = cand._canonical(
        sp.csr_matrix(C[[rival]]) @ second_output_G,
        name="phase_projection.second_objective",
    )
    objective_center = np.asarray(
        C[[rival]] @ second_out_shadow.center - thresholds[[rival]], dtype=np.float64
    )
    objective_error = np.asarray(
        np.abs(C[[rival]]) @ second_out_shadow.error, dtype=np.float64
    )
    objective_coeff = np.asarray(objective_G.toarray()[0], dtype=np.float64)

    blocks = []
    rhs = []
    total_phases = 0
    for layer in order:
        layer_id = int(layer.id)
        original = original_frames.get(layer_id)
        if original is None or not original.exact.size:
            continue
        matrix = cand._canonical(
            sp.csr_matrix(second_pre[layer_id]), name=f"phase_projection.lp_pre[{layer_id}]"
        )
        center = second_pre_shadows[layer_id].center[original.exact]
        rad = radius(
            matrix,
            center,
            second_pre_shadows[layer_id].error[original.exact],
            f"phase_projection.lp_radius[{layer_id}]",
        )
        selected = second_assign[layer_id]
        blocks.append(matrix.multiply(np.where(selected, -1.0, 1.0)[:, None]).tocsr())
        row_scale = oh._row_l1_upper(
            matrix, name=f"phase_projection.lp_scale[{layer_id}]"
        )
        interior = 16.0 * solver_tolerance * (
            1.0 + np.abs(center) + row_scale + rad
        )
        rhs.append(
            np.where(selected, center - rad, -center - rad) - interior
        )
        total_phases += int(original.exact.size)
    A = cand._canonical(sp.vstack(blocks, format="csr"), name="phase_projection.lp_A")
    b = np.ascontiguousarray(np.concatenate(rhs), dtype=np.float64)
    lp_started = time.monotonic()
    result = linprog(
        -objective_coeff,
        A_ub=A,
        b_ub=b,
        bounds=factor_bounds,
        method="highs",
        options={
            "time_limit": 30.0,
            "presolve": True,
            "primal_feasibility_tolerance": solver_tolerance,
        },
    )
    lp_seconds = time.monotonic() - lp_started

    replay_consistent = 0
    replay_inconsistent = 0
    replay_ambiguous = 0
    margin_lower = None
    margin_upper = None
    if result.x is not None:
        xi = np.asarray(result.x, dtype=np.float64).reshape(-1)
        for layer in order:
            layer_id = int(layer.id)
            original = original_frames.get(layer_id)
            if original is None or not original.exact.size:
                continue
            matrix = cand._canonical(
                sp.csr_matrix(second_pre[layer_id]), name=f"phase_projection.replay_pre[{layer_id}]"
            )
            _, lower, upper, _ = interval(
                matrix,
                xi,
                second_pre_shadows[layer_id].center[original.exact],
                second_pre_shadows[layer_id].error[original.exact],
                f"phase_projection.replay_phase[{layer_id}]",
            )
            selected = second_assign[layer_id]
            consistent = (selected & (lower >= 0.0)) | (~selected & (upper <= 0.0))
            inconsistent = (selected & (upper < 0.0)) | (~selected & (lower > 0.0))
            ambiguous = ~(consistent | inconsistent)
            replay_consistent += int(np.count_nonzero(consistent))
            replay_inconsistent += int(np.count_nonzero(inconsistent))
            replay_ambiguous += int(np.count_nonzero(ambiguous))
        _, lower, upper, _ = interval(
            objective_G,
            xi,
            objective_center,
            objective_error,
            "phase_projection.replay_margin",
        )
        margin_lower = float(lower[0])
        margin_upper = float(upper[0])

    success = bool(
        result.success
        and replay_consistent == total_phases
        and margin_lower is not None
        and margin_lower > 0.0
    )
    decoded_input_in_box = False
    singleton_margin_lower = None
    singleton_verified = False
    if success and result.x is not None:
        xi = np.asarray(result.x, dtype=np.float64).reshape(-1)
        decoded = np.asarray(lb, dtype=np.float64).copy()
        for column, row in enumerate(input_rows):
            exact_value = Fraction.from_float(float(input_center[row]))
            exact_value += Fraction.from_float(float(input_radius[row])) * Fraction.from_float(
                float(xi[column])
            )
            decoded[row] = float(exact_value)
        decoded_input_in_box = bool(
            np.all(decoded >= lb) and np.all(decoded <= ub)
        )
        if decoded_input_in_box:
            point = torch.as_tensor(
                decoded.reshape(tuple(seed.lb.shape)),
                dtype=torch.float64,
                device="cuda",
            )
            point_fact = Fact(bounds=Bounds(point, point), cons=ConSet())
            _point_before, point_after, _point_global = analyze(
                net, entry, point_fact
            )
            point_bounds = point_after[output_layer_id].bounds
            point_lower = point_bounds.lb.detach().cpu().double().numpy().reshape(-1)
            point_upper = point_bounds.ub.detach().cpu().double().numpy().reshape(-1)
            exact_lower = -Fraction.from_float(float(thresholds[rival]))
            for coefficient, lower_value, upper_value in zip(
                C[rival], point_lower, point_upper
            ):
                coefficient = float(coefficient)
                if coefficient > 0.0:
                    exact_lower += Fraction.from_float(coefficient) * Fraction.from_float(
                        float(lower_value)
                    )
                elif coefficient < 0.0:
                    exact_lower += Fraction.from_float(coefficient) * Fraction.from_float(
                        float(upper_value)
                    )
            singleton_margin_lower = float(exact_lower)
            singleton_verified = exact_lower > 0
    print(
        json.dumps(
            {
                "schema": "act.hybridz.one_update_phase_projection_probe.v1",
                "status": (
                    "singleton_verified"
                    if singleton_verified
                    else "closed_by_stop_loss"
                ),
                "scope": {
                    "production_integrated": False,
                    "verdict_authority": False,
                    "input_sampling_used": False,
                    "pgd_used": False,
                    "concrete_onnx_execution_used": False,
                    "phase_updates": 1,
                    "phase_retries": 0,
                    "property_rows_selected": 1,
                },
                "selected_property_row": rival,
                "initial_phase_changes": initial_changes,
                "initial_phase_inconsistent": first_inconsistent,
                "input_factors": int(input_rows.size),
                "phase_rows": total_phases,
                "lp_rows": int(A.shape[0]),
                "lp_nnz": int(A.nnz),
                "lp_success": bool(result.success),
                "lp_status": int(result.status),
                "lp_message": str(result.message),
                "lp_seconds": lp_seconds,
                "replay_consistent": replay_consistent,
                "replay_inconsistent": replay_inconsistent,
                "replay_ambiguous": replay_ambiguous,
                "margin_lower": margin_lower,
                "margin_upper": margin_upper,
                "decoded_input_in_box": decoded_input_in_box,
                "singleton_margin_lower": singleton_margin_lower,
                "singleton_verified": singleton_verified,
                "first_stream_seconds": first_s,
                "second_stream_seconds": second_s,
                "total_seconds": time.monotonic() - started,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
