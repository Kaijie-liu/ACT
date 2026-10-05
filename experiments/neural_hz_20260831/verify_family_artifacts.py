#!/usr/bin/env python3
"""Read-only verification of a frozen universe and its converted VNNLIB set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


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


def _load_hashed_json(path: Path, *, expected_format: str) -> tuple[dict, bytes]:
    payload = path.resolve(strict=True).read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if value.get("format_version") != expected_format:
        raise ValueError(f"{path} has an unsupported format")
    claimed = value.get("manifest_payload_sha256")
    unhashed = dict(value)
    unhashed.pop("manifest_payload_sha256", None)
    if claimed != _sha256_json(unhashed):
        raise ValueError(f"{path} payload hash mismatch")
    return value, payload


def _contained_file(root: Path, relative: str) -> Path:
    resolved_root = root.resolve(strict=True)
    candidate = (resolved_root / relative).resolve(strict=True)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"asset escapes family root: {relative}") from exc
    if not candidate.is_file():
        raise ValueError(f"missing regular asset: {relative}")
    return candidate


def verify_family_artifacts(
    universe_path: Path,
    converted_family_dir: Path,
    *,
    parse_vnnlib: bool = False,
    act_root: Path | None = None,
) -> dict[str, object]:
    universe, universe_bytes = _load_hashed_json(
        universe_path,
        expected_format="neural_hz_family_universe_v1",
    )
    converted_root = converted_family_dir.resolve(strict=True)
    conversion_path = converted_root / "CONVERSION_MANIFEST.json"
    conversion, _ = _load_hashed_json(
        conversion_path,
        expected_format="neural_hz_vnnlib_v2_family_conversion_v1",
    )
    family = universe["family"]
    if conversion["family"] != family or converted_root.name != family:
        raise ValueError("family identity mismatch")
    if conversion["source_universe_file_sha256"] != _sha256_bytes(
        universe_bytes
    ):
        raise ValueError("conversion references a different universe file")
    if conversion["source_universe_payload_sha256"] != universe[
        "manifest_payload_sha256"
    ]:
        raise ValueError("conversion references a different universe payload")
    if conversion["source_instances_csv_sha256"] != universe["instances_csv"][
        "sha256"
    ]:
        raise ValueError("conversion references a different instances.csv")

    bench_root = Path(universe["source_benchmark_root"]).resolve(strict=True)
    family_root = (bench_root / family).resolve(strict=True)
    verified_assets = 0
    for asset in universe["assets"]:
        source = _contained_file(family_root, asset["relative_path"])
        if int(source.stat().st_size) != int(asset["size_bytes"]):
            raise ValueError(f"source asset size mismatch: {asset['relative_path']}")
        if _sha256_file(source) != asset["sha256"]:
            raise ValueError(f"source asset hash mismatch: {asset['relative_path']}")
        verified_assets += 1

    rows = sorted(universe["instances"], key=lambda row: int(row["row_index"]))
    records = sorted(conversion["files"], key=lambda row: int(row["row_index"]))
    if len(records) != len(rows):
        raise ValueError("converted record count mismatch")
    if conversion["instance_count"] != len(rows) or conversion[
        "converted_spec_count"
    ] != len(records):
        raise ValueError("conversion manifest count mismatch")
    if conversion["files_payload_sha256"] != _sha256_json(conversion["files"]):
        raise ValueError("conversion files payload hash mismatch")

    expected_paths: set[str] = set()
    converted_paths: list[Path] = []
    for index, (row, record) in enumerate(zip(rows, records, strict=True)):
        if int(row["row_index"]) != index or int(record["row_index"]) != index:
            raise ValueError("row order is not contiguous")
        if record["instance_key"] != row["instance_key"]:
            raise ValueError(f"instance key mismatch at row {index}")
        relative = str(row["spec_relative_path"])
        if record["relative_path"] != relative:
            raise ValueError(f"spec path mismatch at row {index}")
        if record["source_sha256"] != row["spec_sha256"]:
            raise ValueError(f"source spec hash mismatch at row {index}")
        converted = _contained_file(converted_root, relative)
        if int(converted.stat().st_size) != int(record["converted_size_bytes"]):
            raise ValueError(f"converted size mismatch: {relative}")
        if _sha256_file(converted) != record["converted_sha256"]:
            raise ValueError(f"converted hash mismatch: {relative}")
        header = converted.read_text(encoding="utf-8")[:512]
        if f"source_sha256={record['source_sha256']}" not in header:
            raise ValueError(f"converted source header mismatch: {relative}")
        if record["round_trip"] != "EXACT_TOKEN_IDENTITY":
            raise ValueError(f"missing exact token proof: {relative}")
        expected_paths.add(relative)
        converted_paths.append(converted)

    actual_paths = {
        path.relative_to(converted_root).as_posix()
        for path in converted_root.rglob("*.vnnlib")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise ValueError(f"converted file-set mismatch: missing={missing}, extra={extra}")

    parsed_queries = None
    if parse_vnnlib:
        repository = (
            Path(__file__).resolve().parents[2]
            if act_root is None
            else act_root.resolve(strict=True)
        )
        if not (repository / "act" / "__init__.py").is_file():
            raise ValueError(f"act_root does not contain the ACT package: {repository}")
        repository_text = str(repository)
        if repository_text not in sys.path:
            sys.path.insert(0, repository_text)
        import torch

        from act.front_end.spec_creator_base import LabeledInputTensor
        from act.front_end.vnnlib_loader.vnnlib_parser import parse_vnnlib_queries

        shape = tuple(int(value) for value in conversion["input_shape"])
        labeled = LabeledInputTensor(
            tensor=torch.zeros(shape, dtype=torch.float64),
            label=torch.tensor([0]),
        )
        parsed_queries = 0
        for path in converted_paths:
            queries = parse_vnnlib_queries(path, labeled_tensor=labeled)
            if not queries:
                raise ValueError(f"parser returned no query: {path}")
            parsed_queries += len(queries)

    return {
        "family": family,
        "instance_count": len(rows),
        "verified_source_assets": verified_assets,
        "verified_converted_specs": len(records),
        "parsed_queries": parsed_queries,
        "universe_file_sha256": _sha256_bytes(universe_bytes),
        "ordered_universe_sha256": universe["ordered_universe_sha256"],
        "conversion_manifest_file_sha256": _sha256_file(conversion_path),
        "conversion_manifest_payload_sha256": conversion[
            "manifest_payload_sha256"
        ],
        "baseline_vector_status": universe["baseline_vector_status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-manifest", type=Path, required=True)
    parser.add_argument("--converted-family-dir", type=Path, required=True)
    parser.add_argument("--parse-vnnlib", action="store_true")
    parser.add_argument(
        "--act-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    result = verify_family_artifacts(
        args.universe_manifest,
        args.converted_family_dir,
        parse_vnnlib=args.parse_vnnlib,
        act_root=args.act_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
