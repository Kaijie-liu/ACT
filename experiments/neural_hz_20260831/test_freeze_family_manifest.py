from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from freeze_family_manifest import (
    build_manifest,
    exclusive_atomic_write,
    render_manifest,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _family(tmp_path: Path) -> tuple[Path, Path]:
    bench = tmp_path / "bench"
    family = bench / "demo"
    (family / "onnx").mkdir(parents=True)
    (family / "vnnlib").mkdir()
    (family / "onnx" / "model.onnx").write_bytes(b"model-v1")
    (family / "vnnlib" / "a.vnnlib").write_bytes(b"spec-a")
    (family / "vnnlib" / "b.vnnlib").write_bytes(b"spec-b")
    (family / "instances.csv").write_text(
        "./onnx/model.onnx,./vnnlib/a.vnnlib,100\n"
        "onnx/model.onnx,vnnlib/b.vnnlib,100.0\n",
        encoding="utf-8",
    )
    return bench, family


def test_manifest_is_deterministic_and_content_addressed(tmp_path):
    bench, family = _family(tmp_path)
    first = build_manifest(bench, "demo")
    second = build_manifest(bench, "demo")
    assert first == second
    assert first["instance_count"] == 2
    assert first["baseline_vector_status"] == "UNFROZEN"
    assert first["baseline_verdict_counts"] is None
    assert first["instances_csv"]["sha256"] == _sha(
        (family / "instances.csv").read_bytes()
    )
    assert len(first["assets"]) == 3
    assert first["instances"][0]["timeout_seconds"] == "100"
    assert first["instances"][1]["timeout_seconds"] == "100"
    assert first["instances"][0]["instance_key"] != first["instances"][1][
        "instance_key"
    ]
    assert json.loads(render_manifest(first)) == first


def test_content_change_changes_instance_and_universe_hash(tmp_path):
    bench, family = _family(tmp_path)
    before = build_manifest(bench, "demo")
    (family / "vnnlib" / "a.vnnlib").write_bytes(b"spec-a-changed")
    after = build_manifest(bench, "demo")
    assert before["instances"][0]["instance_key"] != after["instances"][0][
        "instance_key"
    ]
    assert before["ordered_universe_sha256"] != after["ordered_universe_sha256"]
    assert before["assets_sha256"] != after["assets_sha256"]


def test_atomic_write_refuses_overwrite(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    output = allowed / "manifest.json"
    exclusive_atomic_write(output, b"first", allowed_root=allowed)
    assert output.read_bytes() == b"first"
    with pytest.raises(FileExistsError):
        exclusive_atomic_write(output, b"second", allowed_root=allowed)
    assert output.read_bytes() == b"first"
    assert not list(allowed.glob(".*.tmp.*"))


def test_output_must_stay_in_isolated_root(tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    with pytest.raises(ValueError, match="output must remain below"):
        exclusive_atomic_write(
            outside / "manifest.json", b"no", allowed_root=allowed
        )


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf", "nope"])
def test_invalid_timeout_fails_closed(tmp_path, timeout):
    bench, family = _family(tmp_path)
    (family / "instances.csv").write_text(
        f"onnx/model.onnx,vnnlib/a.vnnlib,{timeout}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="timeout"):
        build_manifest(bench, "demo")


def test_path_traversal_and_escaping_symlink_fail_closed(tmp_path):
    bench, family = _family(tmp_path)
    outside = tmp_path / "outside.vnnlib"
    outside.write_bytes(b"outside")
    (family / "vnnlib" / "escape.vnnlib").symlink_to(outside)
    (family / "instances.csv").write_text(
        "onnx/model.onnx,vnnlib/escape.vnnlib,100\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="escapes family root"):
        build_manifest(bench, "demo")
    (family / "instances.csv").write_text(
        "onnx/model.onnx,../outside.vnnlib,100\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="contained relative path"):
        build_manifest(bench, "demo")


def test_malformed_or_blank_csv_fails_closed(tmp_path):
    bench, family = _family(tmp_path)
    (family / "instances.csv").write_text(
        "onnx/model.onnx,vnnlib/a.vnnlib\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="expected 3"):
        build_manifest(bench, "demo")
    (family / "instances.csv").write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="blank row"):
        build_manifest(bench, "demo")
