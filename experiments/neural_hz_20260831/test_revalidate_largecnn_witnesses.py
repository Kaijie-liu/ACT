from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from freeze_family_manifest import build_manifest, render_manifest
from revalidate_largecnn_witnesses import (
    build_evidence_ledger,
    problem_content_key,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _fixture(tmp_path: Path):
    bench = tmp_path / "bench"
    family = bench / "demo"
    (family / "onnx").mkdir(parents=True)
    (family / "vnnlib").mkdir()
    graph = helper.make_graph(
        [helper.make_node("Identity", ["X"], ["Y"])],
        "identity",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8
    model_path = family / "onnx" / "m.onnx"
    onnx.save(model, model_path)
    spec = """
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 0))
(assert (<= X_0 1))
(assert (>= X_1 0))
(assert (<= X_1 1))
(assert (or (>= Y_1 Y_0)))
"""
    spec_path = family / "vnnlib" / "a.vnnlib"
    spec_path.write_text(spec, encoding="utf-8")
    (family / "instances.csv").write_text(
        "onnx/m.onnx,vnnlib/a.vnnlib,10\n", encoding="utf-8"
    )
    universe_path = tmp_path / "universe.json"
    universe_path.write_bytes(render_manifest(build_manifest(bench, "demo")))

    sidecar = tmp_path / "sidecars"
    artifact_dir = sidecar / "run_demo_0"
    artifact_dir.mkdir(parents=True)
    x = np.asarray([0.0, 1.0], dtype=np.float64)
    y = np.asarray([0.0, 1.0], dtype=np.float64)
    x_path = artifact_dir / "x.npy"
    y_path = artifact_dir / "y.npy"
    np.save(x_path, x)
    np.save(y_path, y)
    artifact = {
        "sidecar_verdict": "sat_zero_tol",
        "spec_result_zero_tol": {"ast_holds": True},
        "witness_id": {
            "benchmark": "demo",
            "instance_id": 0,
            "source": "unit_test",
        },
        "model_path": str(model_path),
        "spec_path": str(spec_path),
        "model_sha256": _sha(model_path),
        "spec_sha256": _sha(spec_path),
        "x_star_npy": x_path.name,
        "x_star_sha256": _array_sha(x),
        "y_ort_npy": y_path.name,
        "repair_applied": False,
        "repair_method": None,
    }
    artifact_path = artifact_dir / "artifact.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    (artifact_dir / "MANIFEST.json").write_text("{}", encoding="utf-8")
    return universe_path, sidecar, artifact_path, x_path


def test_problem_content_key_ignores_iid_and_path():
    first = problem_content_key("f", "a" * 64, "b" * 64)
    second = problem_content_key("f", "a" * 64, "b" * 64)
    assert first == second
    assert first != problem_content_key("f", "c" * 64, "b" * 64)


def test_independent_ort_vnnlib_replay_builds_evidence_anchor(tmp_path):
    universe, sidecar, _, _ = _fixture(tmp_path)
    ledger = build_evidence_ledger(universe, sidecar, expected_witnesses=1)
    assert ledger["validated_adv_count"] == 1
    assert ledger["unknown_count"] == 0
    assert ledger["invalid_adv_count"] == 0
    assert ledger["neural_hz_baseline_claim"] is False
    assert len(ledger["validator_sources"]) == 3
    assert ledger["ort_session_config"]["providers"] == [
        "CPUExecutionProvider"
    ]
    row = ledger["rows"][0]
    assert row["baseline_verdict"] == "VALIDATED_ADV"
    assert row["neural_hz_gain_credit"] is False
    assert row["evidence"]["replay_status"] == "ORT_VNNLIB_PASS_ZERO_TOL"
    assert row["evidence"]["input_min_slack"] == 0.0
    assert row["evidence"]["unsafe_min_slack"] == 1.0
    assert row["evidence"]["source_witness_file_sha256"] != row["evidence"][
        "source_witness_array_sha256"
    ]


def test_witness_outside_box_fails_without_ledger(tmp_path):
    universe, sidecar, artifact_path, x_path = _fixture(tmp_path)
    np.save(x_path, np.asarray([-0.1, 1.0], dtype=np.float64))
    artifact = json.loads(artifact_path.read_text())
    artifact["x_star_sha256"] = _array_sha(
        np.asarray([-0.1, 1.0], dtype=np.float64)
    )
    artifact_path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="input-domain replay failed"):
        build_evidence_ledger(universe, sidecar, expected_witnesses=1)


def test_artifact_content_hash_mismatch_fails_closed(tmp_path):
    universe, sidecar, artifact_path, _ = _fixture(tmp_path)
    artifact = json.loads(artifact_path.read_text())
    artifact["spec_sha256"] = "0" * 64
    artifact_path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="historical spec hash mismatch"):
        build_evidence_ledger(universe, sidecar, expected_witnesses=1)


def test_expected_witness_count_is_a_hard_gate(tmp_path):
    universe, sidecar, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="expected 2 witness artifacts"):
        build_evidence_ledger(universe, sidecar, expected_witnesses=2)
