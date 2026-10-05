from __future__ import annotations

import json
from pathlib import Path

import pytest

from convert_family_vnnlib_v2 import convert_family
from freeze_family_manifest import build_manifest, render_manifest


def _flat(bound: int) -> str:
    return "\n".join(
        ["; synthetic"]
        + [f"(declare-const X_{index} Real)" for index in range(2)]
        + ["(declare-const Y_0 Real)"]
        + [f"(assert (<= X_0 {bound}))", "(assert (>= Y_0 0))", ""]
    )


def _universe(tmp_path: Path) -> tuple[Path, Path, Path]:
    bench = tmp_path / "bench"
    family = bench / "demo"
    (family / "onnx").mkdir(parents=True)
    (family / "vnnlib").mkdir()
    (family / "onnx" / "m.onnx").write_bytes(b"model")
    (family / "vnnlib" / "a.vnnlib").write_text(_flat(1), encoding="utf-8")
    (family / "vnnlib" / "b.vnnlib").write_text(_flat(2), encoding="utf-8")
    (family / "instances.csv").write_text(
        "onnx/m.onnx,vnnlib/a.vnnlib,10\n"
        "onnx/m.onnx,vnnlib/b.vnnlib,10\n",
        encoding="utf-8",
    )
    universe = tmp_path / "universe.json"
    universe.write_bytes(render_manifest(build_manifest(bench, "demo")))
    output = tmp_path / "output"
    return universe, output, family


def test_complete_family_is_atomically_published(tmp_path):
    universe, output, _ = _universe(tmp_path)
    target, manifest = convert_family(
        universe,
        output,
        input_shape=(1, 2),
        output_shape=(1, 1),
        allowed_root=tmp_path,
    )
    assert target == output / "demo"
    assert manifest["instance_count"] == 2
    assert manifest["converted_spec_count"] == 2
    assert all(
        file["round_trip"] == "EXACT_TOKEN_IDENTITY"
        for file in manifest["files"]
    )
    assert (target / "vnnlib" / "a.vnnlib").read_text().startswith(
        "(vnnlib-version 2.0)\n"
    )
    persisted = json.loads((target / "CONVERSION_MANIFEST.json").read_text())
    assert persisted == manifest
    assert not list(output.glob(".*.tmp.*"))
    assert not list(output.glob("*.publish.lock"))


def test_existing_target_is_never_replaced(tmp_path):
    universe, output, _ = _universe(tmp_path)
    convert_family(
        universe,
        output,
        input_shape=(1, 2),
        output_shape=(1, 1),
        allowed_root=tmp_path,
    )
    marker = output / "demo" / "marker"
    marker.write_text("keep")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        convert_family(
            universe,
            output,
            input_shape=(1, 2),
            output_shape=(1, 1),
            allowed_root=tmp_path,
        )
    assert marker.read_text() == "keep"


def test_source_change_after_freeze_fails_without_publication(tmp_path):
    universe, output, family = _universe(tmp_path)
    (family / "vnnlib" / "a.vnnlib").write_text(_flat(99), encoding="utf-8")
    with pytest.raises(ValueError, match="source spec hash mismatch"):
        convert_family(
            universe,
            output,
            input_shape=(1, 2),
            output_shape=(1, 1),
            allowed_root=tmp_path,
        )
    assert not (output / "demo").exists()
    assert not list(output.glob(".*.tmp.*"))


def test_tampered_universe_fails_closed(tmp_path):
    universe, output, _ = _universe(tmp_path)
    payload = json.loads(universe.read_text())
    payload["instance_count"] = 3
    universe.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="payload hash mismatch"):
        convert_family(
            universe,
            output,
            input_shape=(1, 2),
            output_shape=(1, 1),
            allowed_root=tmp_path,
        )


def test_output_root_must_remain_isolated(tmp_path):
    universe, _, _ = _universe(tmp_path)
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    with pytest.raises(ValueError, match="output root must remain below"):
        convert_family(
            universe,
            outside,
            input_shape=(1, 2),
            output_shape=(1, 1),
            allowed_root=allowed,
        )
