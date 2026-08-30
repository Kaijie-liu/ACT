"""Standalone soundness audit for raw VNNLIB evaluation and strict replay.

Run directly (no pytest):

    python -m act.pipeline.verification.strict_replay_toy_audit
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Callable

import numpy as np
import torch

from act.front_end.specs import OutKind
from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.vnnlib_loader.vnnlib_parser import (
    UnsupportedSpecError,
    evaluate_vnnlib_2_concrete,
    evaluate_vnnlib_concrete,
    extract_vnnlib_concrete_layout,
    parse_vnnlib_queries,
    parse_vnnlib_to_tensors,
)
from act.pipeline.verification.strict_replay import make_strict_replay


def _write(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def _expect_unsupported(action: Callable[[], object], needle: str) -> None:
    try:
        action()
    except UnsupportedSpecError as exc:
        assert needle.lower() in str(exc).lower(), (needle, str(exc))
    else:
        raise AssertionError("expected UnsupportedSpecError")


def _make_identity_onnx(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    x_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, ["batch", 2]
    )
    y_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, ["batch", 2]
    )
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "strict_replay_identity",
        [x_info],
        [y_info],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="act-strict-replay-audit",
    )
    model.ir_version = min(int(model.ir_version), 10)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _case_raw_boolean_and_dialects(root: Path) -> None:
    legacy = _write(
        root / "raw_legacy.vnnlib",
        """
        ; dialect markers in comments must not change recognition:
        ; (vnnlib-version 2.0)
        (declare-const X_0 Real)
        (declare-const X_1 Real)
        (declare-const Y_0 Real)
        (declare-const Y_1 Real)
        (assert (>= X_0 0))
        (assert (<= X_0 1))
        (assert (or
            (and (> (+ X_0 X_1) 0) (>= Y_1 Y_0))
            (= (+ Y_0 X_1) 9)))
        """,
    )
    accepted = evaluate_vnnlib_concrete(
        legacy, np.array([0.25, 0.5]), np.array([1.0, 2.0])
    )
    assert accepted["evaluated"] and accepted["holds"], accepted
    assert accepted["dialect"] == "vnnlib-1.0-flat"
    assert [atom["op"] for atom in accepted["atoms"]] == [
        ">=", "<=", ">", ">=", "="
    ]
    assert len(accepted["assertions"]) == 3
    json.dumps(accepted, allow_nan=False)

    rejected = evaluate_vnnlib_concrete(
        legacy, np.array([0.0, -1.0]), np.array([1.0, 2.0])
    )
    assert rejected["evaluated"] and not rejected["holds"], rejected
    wrong_api = evaluate_vnnlib_2_concrete(
        legacy, np.array([0.25, 0.5]), np.array([1.0, 2.0])
    )
    assert not wrong_api["evaluated"] and not wrong_api["holds"]

    v2 = _write(
        root / "raw_v2.vnnlib",
        """
        (vnnlib-version 2.0)
        (declare-network N)
        (declare-input X Real [2])
        (declare-output Y Real [2])
        (assert (and
            (>= X[0] 0)
            (or (< Y[0] Y[1]) (= (+ X[1] Y[0]) 2))))
        """,
    )
    v2_result = evaluate_vnnlib_2_concrete(
        v2, np.array([0.0, 1.0]), np.array([1.0, 1.0])
    )
    assert v2_result["evaluated"] and v2_result["holds"], v2_result
    assert v2_result["dialect"] == "vnnlib-2.0"
    assert extract_vnnlib_concrete_layout(v2)["input_shape"] == [2]

    strict = _write(
        root / "strict.vnnlib",
        """
        (declare-const X_0 Real)
        (declare-const Y_0 Real)
        (assert (> X_0 0))
        """,
    )
    boundary = evaluate_vnnlib_concrete(strict, [0.0], [0.0], tol=0.0)
    positive = evaluate_vnnlib_concrete(strict, [1e-12], [0.0], tol=0.0)
    assert boundary["evaluated"] and not boundary["holds"]
    assert positive["evaluated"] and positive["holds"]

    missing = _write(
        root / "missing.vnnlib",
        """
        (declare-const X_0 Real)
        (declare-const Y_0 Real)
        (assert (<= X_1 Y_0))
        """,
    )
    missing_result = evaluate_vnnlib_concrete(missing, [0.0], [0.0])
    assert not missing_result["evaluated"] and not missing_result["holds"]
    assert "undeclared X" in missing_result["error"]["message"]

    nan_result = evaluate_vnnlib_concrete(strict, [float("nan")], [0.0])
    assert not nan_result["evaluated"] and not nan_result["holds"]

    mixed_dialect = _write(
        root / "mixed_dialect.vnnlib",
        """
        (vnnlib-version 2.0)
        (declare-network N)
        (declare-input X Real [1])
        (declare-output Y Real [1])
        (declare-const X_0 Real)
        (declare-const Y_0 Real)
        (assert (<= X_0 Y_0))
        """,
    )
    mixed_result = evaluate_vnnlib_concrete(
        mixed_dialect, [0.0], [0.0]
    )
    assert not mixed_result["evaluated"] and not mixed_result["holds"]
    assert "mixed VNNLIB" in mixed_result["error"]["message"]


def _case_legacy_frontend_soundness(root: Path) -> None:
    valid = _write(
        root / "legacy_box_top1.vnnlib",
        """
        ; toy classification property with label: 1.
        (declare-const X_0 Real)
        (declare-const X_1 Real)
        (declare-const Y_0 Real)
        (declare-const Y_1 Real)
        (declare-const Y_2 Real)
        (assert (>= X_0 -1))
        (assert (<= X_0 1))
        (assert (>= X_1 0))
        (assert (<= X_1 2))
        (assert (or
            (and (>= Y_0 Y_1))
            (and (>= Y_2 Y_1))))
        """,
    )
    center, metadata = parse_vnnlib_to_tensors(valid)
    queries = parse_vnnlib_queries(valid)
    assert center.tolist() == [0.0, 1.0]
    assert metadata["dialect"] == "vnnlib-1.0-flat"
    assert len(queries) == 1
    input_spec, output_spec = queries[0]
    assert torch.equal(input_spec.lb, torch.tensor([-1.0, 0.0]))
    assert torch.equal(input_spec.ub, torch.tensor([1.0, 2.0]))
    assert output_spec.kind == OutKind.TOP1_ROBUST
    encoded = output_spec.encode_linear(
        1, 3, torch.device("cpu"), torch.float32
    )
    assert encoded["M"] == 2

    coupled = _write(
        root / "coupled_x.vnnlib",
        """
        (declare-const X_0 Real)
        (declare-const X_1 Real)
        (declare-const Y_0 Real)
        (assert (>= X_0 0))
        (assert (<= X_0 1))
        (assert (>= X_1 0))
        (assert (<= X_1 1))
        (assert (<= (+ X_0 X_1) 1))
        (assert (>= Y_0 0))
        """,
    )
    raw_coupled = evaluate_vnnlib_concrete(coupled, [0.25, 0.25], [1.0])
    assert raw_coupled["evaluated"] and raw_coupled["holds"]
    _expect_unsupported(
        lambda: parse_vnnlib_queries(coupled), "non-rectangular"
    )

    mixed = _write(
        root / "mixed_xy.vnnlib",
        """
        (declare-const X_0 Real)
        (declare-const Y_0 Real)
        (assert (>= X_0 0))
        (assert (<= X_0 1))
        (assert (<= (+ X_0 Y_0) 2))
        """,
    )
    raw_mixed = evaluate_vnnlib_concrete(mixed, [0.5], [1.0])
    assert raw_mixed["evaluated"] and raw_mixed["holds"]
    _expect_unsupported(lambda: parse_vnnlib_queries(mixed), "mixed X/Y")

    equality = _write(
        root / "frontend_equality.vnnlib",
        """
        (declare-const X_0 Real)
        (declare-const Y_0 Real)
        (assert (>= X_0 0))
        (assert (<= X_0 1))
        (assert (= Y_0 0))
        """,
    )
    _expect_unsupported(
        lambda: parse_vnnlib_queries(equality), "equalities"
    )

    missing_bound = _write(
        root / "missing_bound.vnnlib",
        """
        (declare-const X_0 Real)
        (declare-const Y_0 Real)
        (assert (>= X_0 0))
        (assert (>= Y_0 0))
        """,
    )
    _expect_unsupported(
        lambda: parse_vnnlib_queries(missing_bound), "finite lower and upper"
    )

    empty_box = _write(
        root / "empty_box.vnnlib",
        """
        (declare-const X_0 Real)
        (declare-const Y_0 Real)
        (assert (>= X_0 2))
        (assert (<= X_0 1))
        (assert (>= Y_0 0))
        """,
    )
    try:
        parse_vnnlib_queries(empty_box)
    except Exception as exc:
        assert "empty at X_0" in str(exc)
    else:
        raise AssertionError("empty input box was accepted")

    reverse_top1 = _write(
        root / "reverse_top1.vnnlib",
        """
        ; classification property with label: 0.
        (declare-const X_0 Real)
        (declare-const Y_0 Real)
        (declare-const Y_1 Real)
        (assert (>= X_0 0))
        (assert (<= X_0 1))
        (assert (or (and (>= Y_0 Y_1))))
        """,
    )
    reverse_queries = parse_vnnlib_queries(reverse_top1)
    assert len(reverse_queries) == 1
    assert reverse_queries[0][1].kind == OutKind.UNSAFE_LINEAR

    bad_multi_label = LabeledInputTensor(
        tensor=torch.zeros(2),
        label=torch.tensor([0, 1], dtype=torch.int64),
    )
    try:
        parse_vnnlib_queries(valid, labeled_tensor=bad_multi_label)
    except Exception as exc:
        assert "exactly one element" in str(exc)
    else:
        raise AssertionError("multi-element label was accepted")

    bad_range_label = LabeledInputTensor(
        tensor=torch.zeros(2),
        label=torch.tensor([3], dtype=torch.int64),
    )
    try:
        parse_vnnlib_queries(valid, labeled_tensor=bad_range_label)
    except Exception as exc:
        assert "outside [0, 3)" in str(exc)
    else:
        raise AssertionError("out-of-range label was accepted")


def _case_strict_ort_replay(root: Path) -> None:
    model = root / "identity.onnx"
    _make_identity_onnx(model)
    legacy = _write(
        root / "replay_legacy.vnnlib",
        """
        (declare-const X_0 Real)
        (declare-const X_1 Real)
        (declare-const Y_0 Real)
        (declare-const Y_1 Real)
        (assert (>= X_0 0))
        (assert (<= X_0 1))
        (assert (>= X_1 0))
        (assert (<= X_1 1))
        (assert (or (and (>= Y_0 Y_1))))
        """,
    )
    replay = make_strict_replay(model, legacy)
    accepted = replay(np.array([0.75, 0.25], dtype=np.float64))
    assert accepted["valid_counterexample"], accepted
    assert accepted["replay_completed"]
    assert accepted["ort_executed"]
    assert accepted["raw_spec_evaluated"]
    assert accepted["zero_tolerance_holds"]
    assert accepted["vnnlib_dialect"] == "vnnlib-1.0-flat"
    assert accepted["session_config"]["providers"] == ["CPUExecutionProvider"]
    assert accepted["session_config"]["intra_op_num_threads"] == 1
    assert accepted["session_config"]["inter_op_num_threads"] == 1
    assert accepted["session_config"]["use_deterministic_compute"] is True
    assert accepted["input"]["cast_from_dtype"] == "float64"
    assert accepted["input"]["cast_to_dtype"] == "float32"
    assert accepted["input"]["cast_performed"] is True
    assert len(accepted["model_sha256"]) == 64
    assert len(accepted["vnnlib_sha256"]) == 64
    json.dumps(accepted, allow_nan=False)

    rejected = replay(np.array([0.25, 0.75], dtype=np.float32))
    assert not rejected["valid_counterexample"]
    assert rejected["session_was_cached"]
    assert rejected["replay_completed"] and rejected["ort_executed"]
    assert rejected["raw_spec_evaluated"]
    assert not rejected["zero_tolerance_holds"]

    wrong_shape = replay(np.array([0.5], dtype=np.float32))
    assert not wrong_shape["valid_counterexample"]
    assert not wrong_shape["ort_executed"]
    nonfinite = replay(np.array([np.nan, 0.0], dtype=np.float32))
    assert not nonfinite["valid_counterexample"]
    assert not nonfinite["ort_executed"]

    v2 = _write(
        root / "replay_v2.vnnlib",
        """
        (vnnlib-version 2.0)
        (declare-network N)
        (declare-input X Real [2])
        (declare-output Y Real [2])
        (assert (and
            (>= X[0] 0) (<= X[0] 1)
            (>= X[1] 0) (<= X[1] 1)
            (> Y[0] Y[1])))
        """,
    )
    v2_replay = make_strict_replay(model, v2)
    strict_boundary = v2_replay(np.array([0.5, 0.5], dtype=np.float32))
    assert strict_boundary["replay_completed"]
    assert not strict_boundary["valid_counterexample"]
    assert strict_boundary["vnnlib_dialect"] == "vnnlib-2.0"
    strict_positive = v2_replay(np.array([0.5001, 0.5], dtype=np.float32))
    assert strict_positive["valid_counterexample"], strict_positive

    model.write_bytes(model.read_bytes() + b"\x00")
    changed = replay(np.array([0.75, 0.25], dtype=np.float32))
    assert not changed["valid_counterexample"]
    assert "changed after strict replay" in changed["error"]["message"]


def _case_real_large_classification_smoke() -> None:
    benchmark_root = Path(
        "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
    )
    cases = [
        (
            benchmark_root
            / "cifar100_2024/vnnlib/"
            "CIFAR100_resnet_large_prop_idx_1426_sidx_3855_eps_0.0039.vnnlib",
            3072,
            100,
            99,
        ),
        (
            benchmark_root
            / "tinyimagenet_2024/vnnlib/"
            "TinyImageNet_resnet_medium_prop_idx_1496_sidx_9465_eps_0.0039.vnnlib",
            9408,
            200,
            199,
        ),
    ]
    for path, num_inputs, num_outputs, rival_rows in cases:
        if not path.is_file():
            continue
        layout = extract_vnnlib_concrete_layout(path)
        center, metadata = parse_vnnlib_to_tensors(path)
        queries = parse_vnnlib_queries(path)
        assert layout["dialect"] == "vnnlib-1.0-flat"
        assert layout["num_inputs"] == num_inputs
        assert layout["num_outputs"] == num_outputs
        assert center.numel() == num_inputs
        assert metadata["num_outputs"] == num_outputs
        assert len(queries) == 1
        assert queries[0][1].kind == OutKind.TOP1_ROBUST
        encoded = queries[0][1].encode_linear(
            1, num_outputs, torch.device("cpu"), torch.float32
        )
        assert encoded["M"] == rival_rows


def run_all() -> None:
    cases = [
        ("raw Boolean/dialects", _case_raw_boolean_and_dialects),
        ("legacy frontend soundness", _case_legacy_frontend_soundness),
        ("strict ORT replay", _case_strict_ort_replay),
    ]
    with tempfile.TemporaryDirectory(prefix="act-strict-replay-") as temp:
        root = Path(temp)
        for label, case in cases:
            case(root)
            print(f"PASS  {label}")
    _case_real_large_classification_smoke()
    print("PASS  CIFAR100/TinyImageNet parser smoke")


if __name__ == "__main__":
    run_all()
