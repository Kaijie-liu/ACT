#!/usr/bin/env python3
"""Atomically convert one frozen family universe to tensor-indexed VNNLIB 2.0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
from pathlib import Path

from convert_flat_vnnlib_v2 import _shape, convert_text
from freeze_family_manifest import EXPERIMENT_ROOT


FORMAT_VERSION = "neural_hz_vnnlib_v2_family_conversion_v1"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    )


def _verify_universe(manifest: dict[str, object]) -> None:
    if manifest.get("format_version") != "neural_hz_family_universe_v1":
        raise ValueError("unsupported universe manifest format")
    claimed = manifest.get("manifest_payload_sha256")
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    actual = _sha256_json(payload)
    if claimed != actual:
        raise ValueError("universe manifest payload hash mismatch")
    instances = manifest.get("instances")
    if not isinstance(instances, list) or len(instances) != manifest.get(
        "instance_count"
    ):
        raise ValueError("universe manifest instance count mismatch")


def _contained_file(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"invalid source relative path: {relative!r}")
    resolved_root = root.resolve(strict=True)
    candidate = (resolved_root / path).resolve(strict=True)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"source path escapes family root: {relative}") from exc
    if not candidate.is_file():
        raise ValueError(f"source path is not a regular file: {relative}")
    return candidate


def _isolated_output_root(path: Path, allowed_root: Path) -> Path:
    root = allowed_root.resolve(strict=True)
    output = path.resolve(strict=False)
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"output root must remain below {root}: {output}") from exc
    output.mkdir(parents=True, exist_ok=True)
    return output


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def convert_family(
    universe_manifest_path: Path,
    output_root: Path,
    *,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    allowed_root: Path = EXPERIMENT_ROOT,
) -> tuple[Path, dict[str, object]]:
    """Convert and atomically publish one complete family directory."""

    universe_path = universe_manifest_path.resolve(strict=True)
    universe_bytes = universe_path.read_bytes()
    manifest = json.loads(universe_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("universe manifest must be a JSON object")
    _verify_universe(manifest)

    family = manifest.get("family")
    if not isinstance(family, str) or Path(family).name != family:
        raise ValueError("invalid family in universe manifest")
    bench_root = Path(str(manifest["source_benchmark_root"])).resolve(strict=True)
    family_root = (bench_root / family).resolve(strict=True)
    try:
        family_root.relative_to(bench_root)
    except ValueError as exc:
        raise ValueError("family root escapes benchmark root") from exc

    output = _isolated_output_root(output_root, allowed_root)
    target = output / family
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    temporary = output / f".{family}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    lock = output / f".{family}.publish.lock"
    lock_descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    os.close(lock_descriptor)

    converter_path = Path(__file__).with_name("convert_flat_vnnlib_v2.py")
    records: list[dict[str, object]] = []
    seen_specs: set[str] = set()
    published = False
    try:
        temporary.mkdir(mode=0o755)
        rows = sorted(manifest["instances"], key=lambda row: int(row["row_index"]))
        for expected_index, row in enumerate(rows):
            if int(row["row_index"]) != expected_index:
                raise ValueError("universe rows are not one contiguous order")
            spec_relative = str(row["spec_relative_path"])
            if spec_relative in seen_specs:
                raise ValueError(f"duplicate spec path: {spec_relative}")
            seen_specs.add(spec_relative)
            source = _contained_file(family_root, spec_relative)
            source_bytes = source.read_bytes()
            source_sha256 = _sha256_bytes(source_bytes)
            if source_sha256 != row["spec_sha256"]:
                raise ValueError(f"source spec hash mismatch: {spec_relative}")
            try:
                original = source_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"source spec is not UTF-8: {spec_relative}") from exc
            converted, proof = convert_text(
                original,
                input_shape=input_shape,
                output_shape=output_shape,
            )
            converted_bytes = converted.encode("utf-8")
            destination = temporary / spec_relative
            _write_new(destination, converted_bytes)
            records.append(
                {
                    "row_index": expected_index,
                    "instance_key": row["instance_key"],
                    "relative_path": spec_relative,
                    "source_size_bytes": len(source_bytes),
                    "source_sha256": source_sha256,
                    "converted_size_bytes": len(converted_bytes),
                    "converted_sha256": _sha256_bytes(converted_bytes),
                    "round_trip": proof["round_trip"],
                }
            )

        conversion_manifest: dict[str, object] = {
            "format_version": FORMAT_VERSION,
            "family": family,
            "input_shape": [int(value) for value in input_shape],
            "output_shape": [int(value) for value in output_shape],
            "source_universe_file_sha256": _sha256_bytes(universe_bytes),
            "source_universe_payload_sha256": manifest[
                "manifest_payload_sha256"
            ],
            "source_instances_csv_sha256": manifest["instances_csv"]["sha256"],
            "converter_sha256": _sha256_file(converter_path),
            "semantic_check": "EXACT_FLAT_TENSOR_TOKEN_ROUND_TRIP",
            "instance_count": len(rows),
            "converted_spec_count": len(records),
            "files": records,
            "files_payload_sha256": _sha256_json(records),
        }
        conversion_manifest["manifest_payload_sha256"] = _sha256_json(
            conversion_manifest
        )
        _write_new(
            temporary / "CONVERSION_MANIFEST.json",
            (
                json.dumps(conversion_manifest, indent=2, sort_keys=True)
                + "\n"
            ).encode("ascii"),
        )

        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        os.rename(temporary, target)
        published = True
        directory_fd = os.open(output, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return target, conversion_manifest
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-shape", type=_shape, required=True)
    parser.add_argument("--output-shape", type=_shape, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    target, manifest = convert_family(
        args.universe_manifest,
        args.output_root,
        input_shape=args.input_shape,
        output_shape=args.output_shape,
    )
    print(
        json.dumps(
            {
                "target": str(target),
                "family": manifest["family"],
                "instance_count": manifest["instance_count"],
                "converted_spec_count": manifest["converted_spec_count"],
                "files_payload_sha256": manifest["files_payload_sha256"],
                "manifest_payload_sha256": manifest[
                    "manifest_payload_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
