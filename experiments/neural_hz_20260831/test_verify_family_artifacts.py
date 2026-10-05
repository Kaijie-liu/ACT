from __future__ import annotations

import json
from pathlib import Path

import pytest

from convert_family_vnnlib_v2 import convert_family
from freeze_family_manifest import build_manifest, render_manifest
from verify_family_artifacts import verify_family_artifacts


def _fixture(tmp_path: Path):
    bench = tmp_path / "bench"
    family = bench / "demo"
    (family / "onnx").mkdir(parents=True)
    (family / "vnnlib").mkdir()
    (family / "onnx" / "m.onnx").write_bytes(b"model")
    source = (
        "(declare-const X_0 Real)\n"
        "(declare-const X_1 Real)\n"
        "(declare-const Y_0 Real)\n"
        "(assert (<= X_0 X_1))\n"
        "(assert (>= Y_0 0))\n"
    )
    (family / "vnnlib" / "a.vnnlib").write_text(source, encoding="utf-8")
    (family / "instances.csv").write_text(
        "onnx/m.onnx,vnnlib/a.vnnlib,10\n", encoding="utf-8"
    )
    universe = tmp_path / "universe.json"
    universe.write_bytes(render_manifest(build_manifest(bench, "demo")))
    output = tmp_path / "out"
    target, _ = convert_family(
        universe,
        output,
        input_shape=(1, 2),
        output_shape=(1, 1),
        allowed_root=tmp_path,
    )
    return universe, target


def test_complete_artifact_closure_verifies(tmp_path):
    universe, target = _fixture(tmp_path)
    result = verify_family_artifacts(universe, target)
    assert result["instance_count"] == 1
    assert result["verified_source_assets"] == 2
    assert result["verified_converted_specs"] == 1
    assert result["baseline_vector_status"] == "UNFROZEN"


def test_complete_artifact_parses_with_act_v2_parser(tmp_path):
    universe, target = _fixture(tmp_path)
    result = verify_family_artifacts(
        universe,
        target,
        parse_vnnlib=True,
        act_root=Path(__file__).resolve().parents[2],
    )
    assert result["parsed_queries"] == 1


def test_invalid_act_root_fails_before_parser_import(tmp_path):
    universe, target = _fixture(tmp_path)
    with pytest.raises(ValueError, match="does not contain the ACT package"):
        verify_family_artifacts(
            universe,
            target,
            parse_vnnlib=True,
            act_root=tmp_path,
        )


def test_converted_tamper_is_detected(tmp_path):
    universe, target = _fixture(tmp_path)
    converted = target / "vnnlib" / "a.vnnlib"
    converted.write_text(converted.read_text() + "; tamper\n")
    with pytest.raises(ValueError, match="converted size mismatch"):
        verify_family_artifacts(universe, target)


def test_extra_converted_spec_is_detected(tmp_path):
    universe, target = _fixture(tmp_path)
    (target / "vnnlib" / "extra.vnnlib").write_text("extra")
    with pytest.raises(ValueError, match="file-set mismatch"):
        verify_family_artifacts(universe, target)


def test_conversion_manifest_tamper_is_detected(tmp_path):
    universe, target = _fixture(tmp_path)
    path = target / "CONVERSION_MANIFEST.json"
    payload = json.loads(path.read_text())
    payload["converted_spec_count"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="payload hash mismatch"):
        verify_family_artifacts(universe, target)


def test_source_asset_tamper_is_detected(tmp_path):
    universe, target = _fixture(tmp_path)
    payload = json.loads(universe.read_text())
    root = Path(payload["source_benchmark_root"]) / "demo"
    (root / "onnx" / "m.onnx").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source asset size mismatch"):
        verify_family_artifacts(universe, target)
