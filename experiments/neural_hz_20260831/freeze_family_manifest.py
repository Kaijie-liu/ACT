#!/usr/bin/env python3
"""Freeze a content-addressed benchmark-family universe without verdict claims.

The output is deterministic for a fixed source tree and uses atomic,
no-overwrite creation.  The CLI deliberately restricts outputs to this
isolated experiment directory.  It never writes to the benchmark source.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable


FORMAT_VERSION = "neural_hz_family_universe_v1"
EXPERIMENT_ROOT = Path(__file__).resolve().parent


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _strict_relative(raw: str, *, field: str) -> Path:
    value = raw.strip()
    if not value:
        raise ValueError(f"{field} is empty")
    path = Path(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"{field} must be a contained relative path: {raw!r}")
    normalized = Path(*[part for part in path.parts if part not in ("", ".")])
    if not normalized.parts:
        raise ValueError(f"{field} is empty after normalization")
    return normalized


def _contained_file(root: Path, relative: Path, *, field: str) -> Path:
    resolved_root = root.resolve(strict=True)
    candidate = (resolved_root / relative).resolve(strict=True)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes family root: {relative}") from exc
    if not candidate.is_file():
        raise ValueError(f"{field} is not a regular file: {relative}")
    return candidate


def _canonical_timeout(raw: str) -> str:
    value = raw.strip()
    try:
        timeout = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"invalid timeout: {raw!r}") from exc
    if not timeout.is_finite() or timeout <= 0:
        raise ValueError(f"timeout must be finite and positive: {raw!r}")
    normalized = timeout.normalize()
    rendered = format(normalized, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


@dataclass(frozen=True)
class _Asset:
    relative_path: str
    size_bytes: int
    sha256: str

    def record(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def _asset(
    family_root: Path,
    relative: Path,
    *,
    field: str,
    cache: dict[str, _Asset],
) -> _Asset:
    key = relative.as_posix()
    cached = cache.get(key)
    if cached is not None:
        return cached
    path = _contained_file(family_root, relative, field=field)
    result = _Asset(
        relative_path=key,
        size_bytes=int(path.stat().st_size),
        sha256=_sha256_file(path),
    )
    cache[key] = result
    return result


def build_manifest(bench_root: Path, family: str) -> dict[str, object]:
    """Return a deterministic family-universe manifest.

    This freezes only the instance universe.  It intentionally makes no
    baseline or candidate verdict claim.
    """

    family_name = family.strip()
    if not family_name or Path(family_name).name != family_name:
        raise ValueError(f"family must be one path component: {family!r}")
    resolved_bench_root = bench_root.resolve(strict=True)
    family_root = (resolved_bench_root / family_name).resolve(strict=True)
    try:
        family_root.relative_to(resolved_bench_root)
    except ValueError as exc:
        raise ValueError("family escapes benchmark root") from exc
    instances_path = _contained_file(
        family_root, Path("instances.csv"), field="instances.csv"
    )

    assets: dict[str, _Asset] = {}
    rows: list[dict[str, object]] = []
    with instances_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        for row_index, row in enumerate(reader):
            if not row or all(not value.strip() for value in row):
                raise ValueError(f"blank row at instances.csv row {row_index}")
            if len(row) != 3:
                raise ValueError(
                    f"instances.csv row {row_index} has {len(row)} fields, expected 3"
                )
            model_relative = _strict_relative(row[0], field="model path")
            spec_relative = _strict_relative(row[1], field="spec path")
            timeout_seconds = _canonical_timeout(row[2])
            model = _asset(
                family_root,
                model_relative,
                field="model path",
                cache=assets,
            )
            spec = _asset(
                family_root,
                spec_relative,
                field="spec path",
                cache=assets,
            )
            identity_payload = {
                "format": "neural_hz_instance_key_v1",
                "family": family_name,
                "model_sha256": model.sha256,
                "spec_sha256": spec.sha256,
                "timeout_seconds": timeout_seconds,
            }
            rows.append(
                {
                    "row_index": row_index,
                    "instance_key": _sha256_json(identity_payload),
                    "model_relative_path": model.relative_path,
                    "model_sha256": model.sha256,
                    "spec_relative_path": spec.relative_path,
                    "spec_sha256": spec.sha256,
                    "timeout_seconds": timeout_seconds,
                    "baseline_verdict": None,
                }
            )

    if not rows:
        raise ValueError("instances.csv contains no instances")
    duplicate_keys = sorted(
        key
        for key in {str(row["instance_key"]) for row in rows}
        if sum(row["instance_key"] == key for row in rows) > 1
    )
    asset_records = [assets[key].record() for key in sorted(assets)]
    row_identity = [
        {
            "instance_key": row["instance_key"],
            "model_relative_path": row["model_relative_path"],
            "spec_relative_path": row["spec_relative_path"],
            "timeout_seconds": row["timeout_seconds"],
        }
        for row in rows
    ]
    manifest = {
        "format_version": FORMAT_VERSION,
        "family": family_name,
        "source_benchmark_root": str(resolved_bench_root),
        "instances_csv": {
            "relative_path": f"{family_name}/instances.csv",
            "size_bytes": int(instances_path.stat().st_size),
            "sha256": _sha256_file(instances_path),
        },
        "instance_count": len(rows),
        "baseline_vector_status": "UNFROZEN",
        "baseline_verdict_counts": None,
        "duplicate_instance_keys": duplicate_keys,
        "assets": asset_records,
        "assets_sha256": _sha256_json(asset_records),
        "instances": rows,
        "ordered_universe_sha256": _sha256_json(row_identity),
    }
    manifest["manifest_payload_sha256"] = _sha256_json(manifest)
    return manifest


def _require_output_below(output: Path, allowed_root: Path) -> Path:
    root = allowed_root.resolve(strict=True)
    parent = output.parent.resolve(strict=True)
    candidate = parent / output.name
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"output must remain below {root}: {output}") from exc
    return candidate


def exclusive_atomic_write(
    output: Path,
    payload: bytes,
    *,
    allowed_root: Path = EXPERIMENT_ROOT,
) -> None:
    """Atomically publish payload without replacing an existing output."""

    target = _require_output_below(output, allowed_root)
    temporary = target.with_name(
        f".{target.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    )
    descriptor = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def render_manifest(manifest: dict[str, object]) -> bytes:
    return (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_manifest(args.bench_root, args.family)
    exclusive_atomic_write(args.output, render_manifest(manifest))
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "family": manifest["family"],
                "instance_count": manifest["instance_count"],
                "instances_csv_sha256": manifest["instances_csv"]["sha256"],
                "ordered_universe_sha256": manifest["ordered_universe_sha256"],
                "manifest_payload_sha256": manifest["manifest_payload_sha256"],
                "baseline_vector_status": manifest["baseline_vector_status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
