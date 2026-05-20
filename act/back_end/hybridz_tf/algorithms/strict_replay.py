"""Strict zero-tolerance witness replay against the ground-truth model.

ACT-native port of HyZor's ``strict_replay_for_act`` (HZ __init__.py
:2034-:2117). Used by HyZorSolver Phase 4 to confirm an LP-feasibility
witness corresponds to a real model output that violates the spec, vs
a spurious LP relaxation point. Strict ZERO-TOLERANCE comparison —
mirrors VNN-COMP's referee.

The replay path:
  1. If ``net`` carries an ``onnx_path`` attribute pointing to an
     existing file, run onnxruntime forward (preferred — same numerics
     as the official scorer).
  2. Otherwise convert ACT Net → torch ``nn.Module`` via ACTToTorch and
     run float64 forward.
  3. Evaluate the unsafe predicate (``TOP1_ROBUST`` / ``MARGIN_ROBUST``
     / ``LINEAR_LE`` / ``UNSAFE_LINEAR`` / ``RANGE``) on the output.

Returns ``True`` iff the witness concretely violates the safety
specification.
"""
from __future__ import annotations
import os
from typing import Any

import numpy as np
import torch


__all__ = ["strict_replay_for_act"]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _to_np(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _unwrap_int(x) -> int:
    if torch.is_tensor(x):
        return int(x.item() if x.numel() == 1 else x.flatten()[0].item())
    if hasattr(x, "__len__"):
        return int(x[0])
    return int(x)


def _unwrap_float(x) -> float:
    if torch.is_tensor(x):
        return float(x.item() if x.numel() == 1 else x.flatten()[0].item())
    if hasattr(x, "__len__"):
        return float(x[0])
    return float(x)


def _eval_unsafe_strict(y: np.ndarray, assert_layer) -> bool:
    """Return True iff ``y`` violates the safety spec (i.e. is in the
    unsafe set). ZERO tolerance — exact comparisons.
    """
    kind = assert_layer.params.get("kind")
    kstr = str(kind).split(".")[-1] if kind is not None else ""

    if kstr == "TOP1_ROBUST":
        t = int(_unwrap_int(assert_layer.params["y_true"]))
        return any(y[j] >= y[t] for j in range(len(y)) if j != t)

    if kstr == "MARGIN_ROBUST":
        t = int(_unwrap_int(assert_layer.params["y_true"]))
        m = float(_unwrap_float(assert_layer.params["margin"]))
        return any(y[j] >= y[t] - m for j in range(len(y)) if j != t)

    if kstr == "LINEAR_LE":
        coef = _to_np(assert_layer.params["c"]).reshape(-1)
        d = float(_unwrap_float(assert_layer.params["d"]))
        return float(coef @ y) > d

    if kstr == "UNSAFE_LINEAR":
        C = _to_np(assert_layer.params["c"])
        d_vec = _to_np(assert_layer.params["d"]).reshape(-1)
        if C.ndim == 1:
            C = C.reshape(1, -1)
        return bool(np.all(C @ y <= d_vec))

    if kstr == "RANGE":
        lb_t = assert_layer.params.get("lb")
        ub_t = assert_layer.params.get("ub")
        if lb_t is not None and np.any(y < _to_np(lb_t).reshape(-1)):
            return True
        if ub_t is not None and np.any(y > _to_np(ub_t).reshape(-1)):
            return True
        return False

    return False


def _ort_replay(onnx_path: str, x_t: torch.Tensor, assert_layer) -> bool:
    """Run onnxruntime forward and evaluate unsafe predicate.

    Reshapes ``x_t`` to match the ONNX input shape (overrides batch dim
    to 1) and casts to float32 (ONNX models default precision).
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    in_name = sess.get_inputs()[0].name
    in_shape = list(sess.get_inputs()[0].shape)
    in_shape[0] = 1
    x_in = x_t.numpy().reshape(in_shape).astype(np.float32)
    y = sess.run(None, {in_name: x_in})[0].ravel()
    return _eval_unsafe_strict(y, assert_layer)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def strict_replay_for_act(*, net, x_star, assert_layer) -> bool:
    """Strict (zero-tol) witness replay for the ACT verifier.

    Args:
        net: ACT ``Net`` — may carry ``.onnx_path`` for ORT fast path.
        x_star: witness in input space, ``np.ndarray`` of length ``n_in``.
        assert_layer: ACT ASSERT layer carrying spec params.

    Returns:
        ``True`` iff the model's output at ``x_star`` violates the spec
        (i.e. is in the unsafe set). Used to confirm SAT witnesses; if
        False, the LP cert is spurious and the verdict downgrades.
    """
    x_arr = np.asarray(x_star, dtype=np.float64)

    # Path 1: ORT replay (preferred — matches VNN-COMP scorer).
    onnx_path = getattr(net, "onnx_path", None)
    if onnx_path is not None and os.path.exists(onnx_path):
        try:
            x_t = torch.from_numpy(x_arr.astype(np.float32))
            return _ort_replay(onnx_path, x_t, assert_layer)
        except Exception:
            # Fall through to torch path.
            pass

    # Path 2: ACTToTorch conversion + torch forward.
    try:
        from act.pipeline.verification.act2torch import ACTToTorch
        from act.back_end.layer_schema import LayerKind
    except Exception:
        # Pipeline unavailable → can't replay. Sound: reject witness.
        return False

    try:
        torch_model = ACTToTorch(net).run()
        torch_model.eval()
        input_layer = next(
            L for L in net.layers if L.kind == LayerKind.INPUT.value
        )
        in_shape = input_layer.params.get("shape")
        if in_shape is None:
            x_t = torch.from_numpy(x_arr.astype(np.float64)).unsqueeze(0)
        else:
            x_t = torch.from_numpy(x_arr.astype(np.float64)).reshape(in_shape)
        with torch.no_grad():
            y = torch_model(x_t)
            if isinstance(y, dict):
                y = y["output"]
            y_np = y.detach().cpu().numpy().reshape(-1)
        return _eval_unsafe_strict(y_np, assert_layer)
    except Exception:
        return False
