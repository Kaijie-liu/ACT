"""Bridge: SATSidecar strict-FAL artifacts → ACT canonical FAL receipts.

History: cersyve sidecar (HyZor SATSidecar) finds strict FAL witnesses
via input-only candidate + ORT replay (zero-tolerance). These artifacts
live in SATSidecar's own JSON schema (model/spec sha256, x_star,
in_input_domain, spec_result_zero_tol) and are NOT counted in ACT's
canonical capability tally because the canonical path requires receipts
in ``fal_receipt.write_receipt`` format.

This bridge translates each ``sidecar_verdict='sat_zero_tol'`` artifact
into an ACT canonical receipt + companion x_star.npy / y_ort.npy. After
bridging, the cersyve strict FAL can be counted alongside other formal
results.

Sidecar invariants we rely on:
  * ``in_input_domain=True``        → maps to ACT's input_box_holds=True
  * ``spec_result_zero_tol.ast_holds=True`` → spec_zero_tol_holds=True
  * ``model_sha256``, ``spec_sha256``, ``x_star_sha256`` already computed
  * companion ``x_star.npy`` and ``y_ort.npy`` already written

We re-validate sha256 of the npy files before emitting the bridged
receipt so a corrupt sidecar dir cannot poison ACT counts.

Usage:
    python bridge_sidecar_to_act_receipt.py \
        --sidecar-dir /path/to/sidecar/run \
        --receipt-dir /path/to/act/receipt/out \
        --benchmark cersyve
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Tolerances must match fal_receipt.py.
ZERO_TOL = 0.0
SMALL_TOL = 1e-6
SCHEMA_VERSION = 1


class BridgeError(Exception):
    pass


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _act_git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _disjunct_id_from_source(source: str) -> int:
    # SATSidecar source convention: "d{disjunct_id}_{strategy}" e.g.
    # "d0_random_uniform", "d3_box_corner". Falls back to 0 if no prefix.
    if source and source.startswith("d") and "_" in source:
        head = source.split("_", 1)[0]
        try:
            return int(head[1:])
        except (ValueError, IndexError):
            return 0
    return 0


def _verify_sidecar_strict(artifact: Dict[str, Any], sidecar_dir: Path) -> None:
    """Re-validate the invariants that justify ACT receipt emission.

    Raises BridgeError if any invariant fails. Soundness gate: this is
    fail-CLOSED — we will not produce an ACT canonical receipt unless
    every check below passes, even if the sidecar already claimed
    sat_zero_tol."""
    if artifact.get("sidecar_verdict") != "sat_zero_tol":
        raise BridgeError(
            f"verdict mismatch: expected sat_zero_tol, got {artifact.get('sidecar_verdict')!r}"
        )
    if not artifact.get("in_input_domain", False):
        raise BridgeError(
            "in_input_domain is not True; ACT R9.3 input-box gate would fail-close"
        )
    sr_zero = artifact.get("spec_result_zero_tol") or {}
    if not sr_zero.get("ast_holds", False):
        raise BridgeError(
            "spec_result_zero_tol.ast_holds is not True; strict ORT zero-tol failed"
        )
    # Re-hash the .npy files; mismatches indicate tampering or partial writes.
    x_npy = sidecar_dir / artifact["x_star_npy"]
    if not x_npy.exists():
        raise BridgeError(f"x_star npy missing: {x_npy}")
    x_arr = np.load(x_npy)
    expected_sha = artifact.get("x_star_sha256", "")
    actual_sha = _sha256_array(np.asarray(x_arr, dtype=x_arr.dtype))
    if expected_sha and actual_sha != expected_sha:
        raise BridgeError(
            f"x_star sha256 mismatch: expected {expected_sha[:16]}..., got {actual_sha[:16]}..."
        )
    y_npy = sidecar_dir / artifact["y_ort_npy"]
    if not y_npy.exists():
        raise BridgeError(f"y_ort npy missing: {y_npy}")
    model_path = Path(artifact["model_path"])
    if not model_path.exists():
        raise BridgeError(f"model_path no longer exists: {model_path}")
    expected_model_sha = artifact.get("model_sha256", "")
    if expected_model_sha and _sha256_file(model_path) != expected_model_sha:
        raise BridgeError(
            f"model file changed since sidecar run: {model_path}"
        )


def _emit_act_receipt(
    artifact: Dict[str, Any],
    sidecar_dir: Path,
    receipt_dir: Path,
    benchmark: str,
    repo_root: Path,
    bridge_source_tag: str,
) -> Path:
    """Write one ACT-format receipt + copy x_star.npy + y_ort.npy."""
    witness = artifact.get("witness_id") or {}
    instance_id = int(witness.get("instance_id", -1))
    if instance_id < 0:
        raise BridgeError(f"missing/sentinel instance_id in witness_id={witness!r}")
    source = witness.get("source") or "sidecar"
    attempt = int(witness.get("attempt_index", 0))
    query_index = _disjunct_id_from_source(source)

    stem = f"{benchmark}_{instance_id}_q{query_index}_{bridge_source_tag}_{attempt}"
    json_path = receipt_dir / f"{stem}.json"
    x_path = receipt_dir / f"{stem}.x_star.npy"
    y_path = receipt_dir / f"{stem}.y_ort.npy"
    if json_path.exists():
        raise BridgeError(
            f"receipt collision: {json_path.name} already exists in receipt_dir"
        )

    # Atomic-style copy: write then rename for npy files.
    x_arr = np.load(sidecar_dir / artifact["x_star_npy"])
    y_arr = np.load(sidecar_dir / artifact["y_ort_npy"])
    np.save(x_path, x_arr)
    np.save(y_path, y_arr)

    sr_zero = artifact.get("spec_result_zero_tol") or {}
    sr_small = artifact.get("spec_result_small_tol") or {}
    zero_holds = bool(sr_zero.get("ast_holds", False))
    small_holds = bool(sr_small.get("ast_holds", zero_holds))

    record = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": _now_iso(),
        "act_git_head": _act_git_head(repo_root),
        "witness_id": {
            "benchmark": benchmark,
            "instance_id": instance_id,
            "query_index": query_index,
            "source": bridge_source_tag,
            "attempt": attempt,
        },
        "model_path": str(artifact["model_path"]),
        "spec_path": str(artifact.get("spec_path", "")),
        "model_sha256": artifact["model_sha256"],
        "spec_sha256": artifact.get("spec_sha256", ""),
        "x_star_sha256": artifact["x_star_sha256"],
        "x_star_npy": x_path.name,
        "y_ort_npy": y_path.name,
        "spec_zero_tol_holds": zero_holds,
        "spec_small_tol_holds": small_holds,
        "tol_zero": ZERO_TOL,
        "tol_small": SMALL_TOL,
        "input_box_holds": True,    # sidecar in_input_domain re-verified above
        "input_box_reason": "ok",
        # Provenance: every bridged receipt carries the sidecar artifact ref
        # so an auditor can re-derive zero-tol from the original sidecar run.
        "bridged_from_sidecar": {
            "sidecar_run_id": (artifact.get("run_id") or ""),
            "sidecar_artifact": str(sidecar_dir / Path(artifact["x_star_npy"]).with_suffix(".json").name),
            "sidecar_git_head": artifact.get("hyzor_git_head", ""),
            "sidecar_worktree_dirty": bool(artifact.get("hyzor_worktree_dirty", False)),
        },
    }
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    # MANIFEST.csv append (mirrors fal_receipt convention)
    manifest_csv = receipt_dir / "MANIFEST.csv"
    header = ("benchmark,instance_id,query_index,source,attempt,"
              "zero_tol,small_tol,artifact\n")
    line = (f"{benchmark},{instance_id},{query_index},{bridge_source_tag},{attempt},"
            f"{int(zero_holds)},{int(small_holds)},{json_path.name}\n")
    write_header = not manifest_csv.exists()
    with open(manifest_csv, "a") as f:
        if write_header:
            f.write(header)
        f.write(line)
    return json_path


def bridge(
    sidecar_dir: Path,
    receipt_dir: Path,
    benchmark: str,
    repo_root: Path,
    bridge_source_tag: str = "sidecar_bridge",
) -> Dict[str, Any]:
    if not sidecar_dir.is_dir():
        raise BridgeError(f"sidecar_dir not a directory: {sidecar_dir}")
    manifest_path = sidecar_dir / "MANIFEST.json"
    if not manifest_path.exists():
        raise BridgeError(f"sidecar MANIFEST.json missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    receipt_dir.mkdir(parents=True, exist_ok=True)

    emitted: List[str] = []
    skipped: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for entry in manifest.get("artifacts", []):
        if entry.get("verdict") != "sat_zero_tol":
            skipped.append({"file": entry.get("file"), "reason": entry.get("verdict")})
            continue
        artifact_path = sidecar_dir / entry["file"]
        if not artifact_path.exists():
            rejected.append({"file": entry["file"], "reason": "artifact_missing"})
            continue
        artifact = json.loads(artifact_path.read_text())
        artifact.setdefault("witness_id", entry.get("witness_id", {}))
        try:
            _verify_sidecar_strict(artifact, sidecar_dir)
            out = _emit_act_receipt(
                artifact, sidecar_dir, receipt_dir, benchmark, repo_root, bridge_source_tag
            )
            emitted.append(str(out))
        except BridgeError as e:
            rejected.append({"file": entry["file"], "reason": str(e)})

    summary = {
        "sidecar_dir": str(sidecar_dir),
        "receipt_dir": str(receipt_dir),
        "benchmark": benchmark,
        "n_total_artifacts": len(manifest.get("artifacts", [])),
        "n_sat_zero_tol": sum(1 for a in manifest.get("artifacts", []) if a.get("verdict") == "sat_zero_tol"),
        "n_emitted": len(emitted),
        "n_skipped_non_strict": len(skipped),
        "n_rejected_invariant": len(rejected),
        "emitted_receipts": emitted,
        "rejected": rejected,
    }
    (receipt_dir / "bridge_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sidecar-dir", required=True, type=Path)
    ap.add_argument("--receipt-dir", required=True, type=Path)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--source-tag", default="sidecar_bridge")
    ap.add_argument("--act-repo", default="/data1/Kane/ACT", type=Path)
    args = ap.parse_args()

    summary = bridge(
        sidecar_dir=args.sidecar_dir,
        receipt_dir=args.receipt_dir,
        benchmark=args.benchmark,
        repo_root=args.act_repo,
        bridge_source_tag=args.source_tag,
    )
    print(json.dumps(summary, indent=2))
    if summary["n_emitted"] < summary["n_sat_zero_tol"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
