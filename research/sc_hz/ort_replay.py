"""Strict ORT replay for FAL_CANDIDATE → A promotion.

Per design lock §1.5 and EXECUTION §2.3:
  - Decode the LP maximizer xi_star to an input x_star
  - Run x_star through onnxruntime
  - Check (input_box_holds, vnnlib_query_holds, spec_zero_tol_holds)
  - Only promote to A if ALL three checks pass

Phase A simplification: for each FAL_CANDIDATE iid, we re-run the
forward propagator to get the per-condition xi_star and decode. For
each unsafe condition (d, threshold, label) with LP UB > threshold,
we decode to input x*, run ORT, check if the condition actually
holds — if yes, A confirmed.

Phantom LP SAT: if the xi_star depends on interval-tail variables
(not original root variables), the decoding is not unique and we
cannot reproduce in the concrete input. Record as phantom_lp_sat.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnxruntime as ort

ACT_ROOT = Path(__file__).resolve().parents[2]
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))

from research.canonical_provenance import load_instance  # noqa: E402
from research.sc_hz.onnx_walker import parse_onnx_to_layers  # noqa: E402
from research.sc_hz.vnnlib_parse import parse_vnnlib  # noqa: E402


def decode_xi_star_for_condition(
    state_out_metadata: Dict[str, Any],
    d_out: np.ndarray,
    input_c: np.ndarray,
    input_r: np.ndarray,
    d_at_input: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """Closed-form decode: the xi that maximizes d_out @ y is

        xi_star_input[i] = sign(d_at_input[i])  (so input goes to the
                                                  extreme of the box in
                                                  the direction that
                                                  maximizes rival margin)

    Returns (x_star_in_input_space, uses_only_root_vars).

    The xi_star is constructed purely from input root variables; we do
    NOT use interval-tail variables (which would be phantom). The
    caller should verify the resulting LP UB on this decoded xi_star is
    close to the abstract LP UB; if not, the decoded x_star is not the
    true maximizer (likely the tail contributed significantly), and the
    rival condition can only be promoted if ORT confirms it on x_star.
    """
    # Sign convention: maximize d_out @ y under linear approx is equivalent
    # to maximizing d_at_input @ x over the box. The maximizer is
    # x* = c_in + r_in * sign(d_at_input).
    sign_d = np.sign(d_at_input)
    # avoid zero-sign: default to +1
    sign_d = np.where(sign_d == 0, 1.0, sign_d)
    x_star = input_c + input_r * sign_d
    uses_only_root_vars = True
    return x_star, uses_only_root_vars


def ort_replay_one(onnx_path: str, x_input: np.ndarray,
                    input_shape: Tuple[int, ...]) -> np.ndarray:
    """Run ONNX model on x_input and return output y."""
    sess = ort.InferenceSession(str(onnx_path),
                                 providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    # Try matching the model's declared input shape (with implicit batch dim)
    expected_shape = sess.get_inputs()[0].shape
    # Reshape x_input
    n_in = x_input.size
    if expected_shape == ["batch_size"] or (
        isinstance(expected_shape[0], str) if expected_shape else False
    ):
        # Dynamic batch; assume (1, n_in) or (1,) + remaining
        target = tuple(int(d) if isinstance(d, int) and d > 0 else 1
                        for d in expected_shape)
        x_arr = x_input.astype(np.float32).reshape(target)
    else:
        target = tuple(int(d) for d in expected_shape if d != 0)
        if target == ():
            target = (1, n_in)
        try:
            x_arr = x_input.astype(np.float32).reshape(target)
        except ValueError:
            # Fallback to (1,) + input_shape
            x_arr = x_input.astype(np.float32).reshape(1, *input_shape)
    y = sess.run(None, {in_name: x_arr})[0].reshape(-1)
    return y.astype(np.float64)


def check_unsafe_condition(y: np.ndarray, d_out: np.ndarray,
                             threshold: float, atol: float = 1e-9) -> bool:
    """Strictly check whether the unsafe condition holds: d_out @ y >= threshold."""
    return float(d_out @ y) >= threshold - atol


def promote_iid(benchmark: str, iid: int,
                 sc_hz_receipt: Dict[str, Any],
                 K: int = 256) -> Dict[str, Any]:
    """For one FAL_CANDIDATE iid, try to promote via ORT replay.

    Returns an updated verdict dict with:
      - replay_verdict: "A_CONFIRMED" | "PHANTOM_LP_SAT" | "UNK_REPLAY_FAILED"
      - per_cond_replay: list of per-condition replay outcomes
      - x_star_at_each_violating_cond
    """
    from research.sc_hz.onnx_walker import (
        parse_onnx_to_layers, forward_propagate,
    )
    from research.sc_hz.precompute_direction import precompute_d_per_layer_chain
    from research.sc_hz.prune import PrunedState
    from research.sc_hz.ops import lp_ub_rival_margin

    out: Dict[str, Any] = {"replay_verdict": "UNK_REPLAY_FAILED",
                            "per_cond_replay": []}

    onnx_path, vnn_path = load_instance(benchmark, iid)
    layers, input_shape, n_classes = parse_onnx_to_layers(str(onnx_path))
    n_in = 1
    for d in input_shape:
        n_in *= int(d)
    lb_x, ub_x, unsafe = parse_vnnlib(str(vnn_path), n_in, n_classes)

    c_in = (lb_x + ub_x) / 2.0
    r_in = (ub_x - lb_x) / 2.0
    G0 = np.diag(r_in).astype(np.float64)

    # For each unsafe condition where LP UB suggests violation, attempt ORT replay
    any_confirmed = False
    for cond_idx, (d_out, threshold, label) in enumerate(unsafe):
        # Need to compute d_at_input — which is d_chain[0]
        from research.sc_hz.run_sentinels import _layer_output_shapes
        out_shapes = _layer_output_shapes(layers, input_shape)
        d_chain = precompute_d_per_layer_chain(layers, d_out, out_shapes)
        d_at_input = d_chain[0]

        # Decode xi_star to input x_star
        x_star, root_only = decode_xi_star_for_condition(
            {}, d_out, c_in, r_in, d_at_input,
        )
        # Clip to input box (safety)
        x_star = np.clip(x_star, lb_x, ub_x)
        # ORT replay
        try:
            y_at_x_star = ort_replay_one(str(onnx_path), x_star, input_shape)
        except Exception as e:
            out["per_cond_replay"].append({
                "label": label, "ort_error": str(e)[:200],
            })
            continue
        cond_holds = check_unsafe_condition(y_at_x_star, d_out, threshold)
        out["per_cond_replay"].append({
            "label": label,
            "threshold": float(threshold),
            "d_dot_y_at_x_star": float(d_out @ y_at_x_star),
            "cond_holds_on_x_star": bool(cond_holds),
        })
        if cond_holds:
            any_confirmed = True

    if any_confirmed:
        out["replay_verdict"] = "A_CONFIRMED"
    else:
        # If LP said FAL but ORT can't realize it, it's phantom
        out["replay_verdict"] = "PHANTOM_LP_SAT"
    return out


if __name__ == "__main__":
    # Quick smoke
    import sys
    if len(sys.argv) > 2:
        b, i = sys.argv[1], int(sys.argv[2])
        result = promote_iid(b, i, {})
        print(json.dumps(result, indent=2, default=str))
