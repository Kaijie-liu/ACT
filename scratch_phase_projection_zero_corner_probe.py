#!/usr/bin/env python3
"""One-rule audit of the first-stream analytic corner.

The non-authoritative candidate is the unique inward BOX corner selected by
the first affine generator stream.  Candidate construction stops there: no
target-cell update, phase-delta expansion, LP model, LP solve, retry, or
fallback is executed.  Authority remains exclusively with raw-BOX membership,
the verifier-owned zero-width interval forward pass, and the exact
stored-binary64 Fraction property lower bound.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from act.back_end.analyze import analyze
from act.back_end.core import ConSet, Fact
from act.back_end.hybridz_tf import forward_exact_relu_phase_projection_candidate as phase
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


ONNX = os.environ["ACT_PHASE_PROJECTION_ONNX"]
VNNLIB = os.environ["ACT_PHASE_PROJECTION_VNNLIB"]
CATEGORY = os.environ["ACT_PHASE_PROJECTION_CATEGORY"]
CASE = os.environ["ACT_PHASE_PROJECTION_CASE"]


class _FirstCornerReady(RuntimeError):
    pass


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scope() -> dict[str, object]:
    return {
        "candidate_rule": "unique_first_stream_analytic_inward_box_corner",
        "candidate_authority": False,
        "terminal_authority": (
            "raw_BOX;verifier_owned_zero_width_interval;"
            "stored_binary64_Fraction_property"
        ),
        "target_cell_update_used": False,
        "phase_delta_or_expansion_used": False,
        "lp_model_built": False,
        "lp_solve_used": False,
        "input_sampling_used": False,
        "onnx_point_execution_used": False,
        "pgd_used": False,
        "bab_or_split_used": False,
        "backward_bounds_used": False,
        "dual_tightening_used": False,
        "fallback_or_retry_menu": False,
        "production_modified": False,
    }


def _extract_first_corner(net, entry, before, after):
    captured: dict[str, object] = {}
    target_code = phase.build_forward_exact_relu_phase_projection_candidate.__code__

    def tracer(frame, event, _arg):
        if frame.f_code is target_code and event == "line":
            local = frame.f_locals
            if "first_factors" in local and "projected" not in local:
                captured.update(local)
                raise _FirstCornerReady
        return tracer

    started = time.monotonic()
    sys.settrace(tracer)
    try:
        phase.build_forward_exact_relu_phase_projection_candidate(
            net,
            int(entry),
            before,
            after,
            deadline=started + 30.0,
            lp_time_limit=30.0,
        )
    except _FirstCornerReady:
        pass
    finally:
        sys.settrace(None)
    if not captured:
        raise RuntimeError("first analytic corner was not reached")
    return captured, time.monotonic() - started


def main() -> None:
    initialize_device(device="cuda", dtype="float64")
    set_solver_mode("hybridz")
    set_transfer_function_mode("interval")

    request_started = time.monotonic()
    spec = create_specs_from_paths(ONNX, VNNLIB, category=CATEGORY)
    wrapped = next(iter(synthesize_models_from_specs([spec]).values())).to(
        device=torch.device("cuda"), dtype=torch.float64
    )
    net = TorchToACT(wrapped).run()
    entry = int(find_entry_layer_id(net))
    specs = gather_input_spec_layers(net)
    fact = Fact(bounds=seed_from_input_specs(specs), cons=ConSet())
    add_all_input_specs(fact.cons, get_input_ids(net), specs)
    before, after, _ = analyze(net, entry, fact)
    assert_layer = get_assert_layer(net)
    output_layer_id = _get_output_layer_id(net)
    output_width = len(
        next(
            layer
            for layer in net.layers
            if int(layer.id) == int(output_layer_id)
        ).out_vars
    )
    _ensure_assert_linear_encoding(
        assert_layer,
        B=1,
        n_out=output_width,
        device=torch.device("cuda"),
        dtype=torch.float64,
    )
    analysis_seconds = time.monotonic() - request_started

    local, corner_seconds = _extract_first_corner(net, entry, before, after)
    factors = np.asarray(local["first_factors"], dtype=np.float64)
    raw_lower = np.asarray(local["raw_lower"], dtype=np.float64)
    raw_upper = np.asarray(local["raw_upper"], dtype=np.float64)
    input_center = np.asarray(local["input_center"], dtype=np.float64)
    input_radius = np.asarray(local["input_radius"], dtype=np.float64)
    input_rows = np.asarray(local["input_rows"], dtype=np.int64)
    decoded = raw_lower.copy()
    for column, raw_row in enumerate(input_rows):
        row = int(raw_row)
        value = Fraction.from_float(float(input_center[row]))
        value += Fraction.from_float(float(input_radius[row])) * Fraction.from_float(
            float(factors[column])
        )
        decoded[row] = float(value)
    raw_box_verified = bool(
        np.all(np.isfinite(decoded))
        and np.all(decoded >= raw_lower)
        and np.all(decoded <= raw_upper)
    )

    terminal_started = time.monotonic()
    point_lower, point_upper = phase._singleton_interval_forward(
        net,
        local["order"],
        local["affines"],
        decoded.reshape(local["input_shape"]),
        int(local["output_layer_id"]),
        pointwise=local["pointwise"],
        deadline=None,
    )
    exact_margin = phase._exact_singleton_margin_lower(
        local["C"][int(local["rival"])],
        local["thresholds"][int(local["rival"])],
        point_lower,
        point_upper,
    )
    terminal_seconds = time.monotonic() - terminal_started
    terminal_verified = bool(raw_box_verified and exact_margin > 0)
    first_affine_margin = float(
        local["first_objective_center"][int(local["rival"])]
        + local["first_coeff"] @ factors
    )

    print(
        json.dumps(
            {
                "schema": "act.scratch.phase_projection_zero_corner.v1",
                "case": CASE,
                "status": "TERMINAL_VERIFIED" if terminal_verified else "STOP_LOSS",
                "selected_property_row": int(local["rival"]),
                "input_factors": int(factors.size),
                "first_cell_affine_margin": first_affine_margin,
                "raw_box_verified": raw_box_verified,
                "terminal_exact_margin_lower": float(exact_margin),
                "terminal_verified": terminal_verified,
                "timing": {
                    "analysis_seconds": analysis_seconds,
                    "corner_path_seconds_instrumented": corner_seconds,
                    "setup_seconds_instrumented": float(local["setup_seconds"]),
                    "first_center_seconds_instrumented": float(
                        local["first_center_seconds"]
                    ),
                    "first_stream_seconds_instrumented": float(
                        local["first_stream_seconds"]
                    ),
                    "terminal_seconds": terminal_seconds,
                    "candidate_plus_terminal_seconds_instrumented": (
                        corner_seconds + terminal_seconds
                    ),
                },
                "source_sha256": {
                    "probe": _sha256(__file__),
                    "phase_projection": _sha256(phase.__file__),
                    "onnx": _sha256(ONNX),
                    "vnnlib": _sha256(VNNLIB),
                },
                "scope": _scope(),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
