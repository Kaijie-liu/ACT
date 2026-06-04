"""Canonical-root provenance guard for research scripts.

Per advisor 2026-06-03 REC2: every research script that reads
benchmark instances MUST go through this module. It hard-codes the
canonical VNN-COMP 2025 root, fail-closes on any drift, and emits the
hash bundle every receipt is required to record.

Concrete contract:
  - Single source of truth: ``CANONICAL_ROOT`` = vnncomp2025_benchmarks/benchmarks.
  - ``CANONICAL_INSTANCES_CSV(benchmark)`` reads the canonical
    instances.csv for the named benchmark and returns (rows, sha256).
  - ``load_instance(benchmark, iid)`` returns (onnx_path, vnnlib_path)
    derived from the canonical instances.csv. Raises on missing files
    so silent fallback to ``/data1/Kane/ACT/data/vnnlib/`` (the LOCAL
    pool that 2026-06-03 dispatch tripped on) cannot happen.
  - ``sha256_file(path)`` returns hex digest.
  - ``build_provenance(benchmark, iid)`` returns a dict you embed in
    every receipt:
        canonical_root, benchmark, iid,
        instances_csv_path, instances_csv_sha256,
        onnx_path, onnx_sha256,
        vnnlib_path, vnnlib_sha256

History note: the prior 2026-06-03 P0 dispatch silently read
``/data1/Kane/ACT/data/vnnlib/cifar100_2024/instances.csv``. The two
files have zero vnnlib-file overlap, so the result was on a different
pool than baseline. This module exists to make that mistake
impossible.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

CANONICAL_ROOT = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks")

# Block the prior LOCAL pool from ever being used silently. Any path
# whose absolute form starts with one of these prefixes is rejected by
# ``assert_canonical_path``.
FORBIDDEN_PREFIXES: Tuple[str, ...] = (
    "/data1/Kane/ACT/data/vnnlib",
    "/data1/Kane/HyZor/data/vnnlib",
)


class ProvenanceError(RuntimeError):
    """Raised on any provenance violation. Always fail-closed."""


def assert_canonical_root_exists() -> None:
    if not CANONICAL_ROOT.exists():
        raise ProvenanceError(
            f"canonical root {CANONICAL_ROOT} does not exist; "
            f"refuse to proceed on a non-standard tree"
        )


def assert_canonical_path(path: str | os.PathLike[str]) -> Path:
    """Resolve to an absolute path AND check it is under the canonical
    root and not in any forbidden prefix. Raises on violation."""
    p = Path(path).resolve()
    for bad in FORBIDDEN_PREFIXES:
        if str(p).startswith(bad):
            raise ProvenanceError(
                f"path {p} is in forbidden LOCAL pool ({bad}); "
                f"only canonical-root files are allowed"
            )
    # Path doesn't have to be UNDER canonical root (output dirs are
    # outside), but ONNX/VNNLIB inputs are strict.
    return p


def assert_canonical_input_path(path: str | os.PathLike[str]) -> Path:
    """Strict variant: the path must be under CANONICAL_ROOT."""
    p = assert_canonical_path(path)
    if not str(p).startswith(str(CANONICAL_ROOT)):
        raise ProvenanceError(
            f"input path {p} is not under canonical root "
            f"{CANONICAL_ROOT}; this guard exists because the 2026-06-03 "
            f"P0 dispatch silently used a different vnnlib pool"
        )
    return p


def sha256_file(path: str | os.PathLike[str]) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=64)
def canonical_instances_rows(benchmark: str) -> Tuple[Tuple[str, str, str], ...]:
    """Return the rows of canonical instances.csv as
    (onnx_rel, vnnlib_rel, timeout) tuples. Cached because the file
    must be read many times.
    """
    assert_canonical_root_exists()
    csv_path = CANONICAL_ROOT / benchmark / "instances.csv"
    if not csv_path.exists():
        raise ProvenanceError(
            f"canonical instances.csv not found at {csv_path} for "
            f"benchmark={benchmark!r}; refusing to fall back"
        )
    rows: List[Tuple[str, str, str]] = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                raise ProvenanceError(
                    f"malformed instances.csv row in {csv_path}: {line!r}"
                )
            onnx_rel = parts[0]
            vnn_rel = parts[1]
            timeout = parts[2] if len(parts) > 2 else ""
            rows.append((onnx_rel, vnn_rel, timeout))
    return tuple(rows)


@lru_cache(maxsize=64)
def canonical_instances_csv_sha256(benchmark: str) -> str:
    csv_path = CANONICAL_ROOT / benchmark / "instances.csv"
    return sha256_file(csv_path)


def canonical_instances_csv_path(benchmark: str) -> Path:
    return CANONICAL_ROOT / benchmark / "instances.csv"


def load_instance(benchmark: str, iid: int) -> Tuple[Path, Path]:
    """Return (onnx_path, vnnlib_path) for the iid-th row of the
    canonical instances.csv for ``benchmark``. Raises ProvenanceError
    on missing files. Iid is the row index, 0-based.
    """
    rows = canonical_instances_rows(benchmark)
    if iid < 0 or iid >= len(rows):
        raise ProvenanceError(
            f"iid {iid} out of range for benchmark {benchmark!r} "
            f"(rows={len(rows)})"
        )
    onnx_rel, vnn_rel, _ = rows[iid]
    bench_dir = CANONICAL_ROOT / benchmark
    onnx_path = (bench_dir / onnx_rel).resolve()
    vnn_path = (bench_dir / vnn_rel).resolve()
    if not onnx_path.exists():
        raise ProvenanceError(
            f"onnx file missing for {benchmark}/iid={iid}: {onnx_path}"
        )
    if not vnn_path.exists():
        raise ProvenanceError(
            f"vnnlib file missing for {benchmark}/iid={iid}: {vnn_path}"
        )
    # Belt-and-braces: the resolved paths must still be under canonical root.
    assert_canonical_input_path(onnx_path)
    assert_canonical_input_path(vnn_path)
    return onnx_path, vnn_path


@dataclass(frozen=True)
class Provenance:
    canonical_root: str
    benchmark: str
    iid: int
    instances_csv_path: str
    instances_csv_sha256: str
    onnx_path: str
    onnx_sha256: str
    vnnlib_path: str
    vnnlib_sha256: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "canonical_root": self.canonical_root,
            "benchmark": self.benchmark,
            "iid": self.iid,
            "instances_csv_path": self.instances_csv_path,
            "instances_csv_sha256": self.instances_csv_sha256,
            "onnx_path": self.onnx_path,
            "onnx_sha256": self.onnx_sha256,
            "vnnlib_path": self.vnnlib_path,
            "vnnlib_sha256": self.vnnlib_sha256,
        }


def build_provenance(benchmark: str, iid: int) -> Provenance:
    onnx_path, vnn_path = load_instance(benchmark, iid)
    return Provenance(
        canonical_root=str(CANONICAL_ROOT),
        benchmark=benchmark,
        iid=int(iid),
        instances_csv_path=str(canonical_instances_csv_path(benchmark)),
        instances_csv_sha256=canonical_instances_csv_sha256(benchmark),
        onnx_path=str(onnx_path),
        onnx_sha256=sha256_file(onnx_path),
        vnnlib_path=str(vnn_path),
        vnnlib_sha256=sha256_file(vnn_path),
    )


def smoke_self_check() -> None:
    """Quick check this module is properly wired:
    - canonical root exists
    - cifar100_2024 instances.csv loads with 200 rows
    - iid 0 resolves to a file that exists
    - sha256 produces a 64-char hex
    """
    assert_canonical_root_exists()
    rows = canonical_instances_rows("cifar100_2024")
    assert len(rows) == 200, f"expected 200 rows, got {len(rows)}"
    onnx_p, vnn_p = load_instance("cifar100_2024", 0)
    assert onnx_p.exists() and vnn_p.exists()
    prov = build_provenance("cifar100_2024", 0)
    assert len(prov.onnx_sha256) == 64
    assert len(prov.vnnlib_sha256) == 64
    print(f"[provenance] OK  canonical_root={CANONICAL_ROOT}")
    print(f"[provenance] cifar100_2024 instances.csv sha256="
          f"{prov.instances_csv_sha256[:16]}...")
    print(f"[provenance] iid 0 vnnlib: {prov.vnnlib_path}")
    # Confirm the LOCAL pool would be rejected.
    try:
        assert_canonical_input_path(
            "/data1/Kane/ACT/data/vnnlib/cifar100_2024/instances.csv"
        )
    except ProvenanceError as e:
        print(f"[provenance] LOCAL pool correctly rejected: {type(e).__name__}")
    else:
        raise RuntimeError("guard did not reject LOCAL pool path!")


if __name__ == "__main__":
    smoke_self_check()
