"""Probe α,β-CROWN's actual ONNX conversion and bound-graph front end."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch


MOE_ROOT = Path("/data1/Kane/MOE")
CROWN_REPO = MOE_ROOT / "baselines/alpha-beta-CROWN"
CROWN_COMMIT = "e5c7e17bf0488843acb77b7519f59876717a49f4"


def _inside(path: Path) -> Path:
    value = path.resolve()
    if not value.is_relative_to(MOE_ROOT):
        raise RuntimeError(f"parser probe path escapes {MOE_ROOT}: {path}")
    return value


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=CROWN_REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def run(onnx_path: Path, probes_path: Path) -> dict[str, Any]:
    onnx_path = _inside(onnx_path)
    probes_path = _inside(probes_path)
    if _git("rev-parse", "HEAD") != CROWN_COMMIT or _git("status", "--porcelain"):
        raise RuntimeError("α,β-CROWN repository identity/cleanliness gate failed")
    started = time.monotonic()
    result: dict[str, Any] = {
        "schema_version": 1,
        "crown_repository": str(CROWN_REPO),
        "crown_commit": CROWN_COMMIT,
        "onnx": str(onnx_path),
        "probes": str(probes_path),
        "onnx2pytorch": {"status": "NOT_RUN"},
        "auto_lirpa": {"status": "NOT_RUN"},
    }
    with np.load(probes_path, allow_pickle=False) as probes:
        inputs = torch.from_numpy(probes["inputs"].copy()).float()
        expected = torch.from_numpy(probes["outputs"].copy()).float()
    try:
        import onnx
        import onnx2pytorch

        graph = onnx.load(str(onnx_path))
        converted = onnx2pytorch.ConvertModel(
            graph, experimental=True, quirks={}
        ).eval()
        with torch.no_grad():
            converted_output = converted(inputs)
        maximum_error = float((converted_output - expected).abs().max().item())
        conversion_matches = bool(
            torch.allclose(converted_output, expected, atol=1e-4, rtol=1e-5)
        )
        result["onnx2pytorch"] = {
            "status": (
                "ACCEPTED_SEMANTICS_MATCH"
                if conversion_matches
                else "ACCEPTED_SILENT_SEMANTIC_MISMATCH"
            ),
            "maximum_abs_error": maximum_error,
            "probes": int(inputs.shape[0]),
        }
    except Exception as error:
        result["onnx2pytorch"] = {
            "status": "REJECTED",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        result["overall_status"] = "EXISTING_VERIFIER_CANNOT_CONSUME"
        result["runtime_seconds"] = time.monotonic() - started
        return result

    if result["onnx2pytorch"]["status"] != "ACCEPTED_SEMANTICS_MATCH":
        result["overall_status"] = "SILENT_SEMANTIC_MISMATCH"
        result["runtime_seconds"] = time.monotonic() - started
        return result
    try:
        from auto_LiRPA import BoundedModule

        bounded = BoundedModule(converted, inputs[:1], device="cpu")
        with torch.no_grad():
            bounded_output = bounded(inputs[:1])
        maximum_error = float((bounded_output - expected[:1]).abs().max().item())
        matches = bool(torch.allclose(bounded_output, expected[:1], atol=1e-4, rtol=1e-5))
        result["auto_lirpa"] = {
            "status": "ACCEPTED" if matches else "ACCEPTED_SILENT_SEMANTIC_MISMATCH",
            "maximum_abs_error": maximum_error,
        }
        result["overall_status"] = (
            "EXISTING_VERIFIER_FRONTEND_ACCEPTS"
            if matches
            else "SILENT_SEMANTIC_MISMATCH"
        )
    except Exception as error:
        result["auto_lirpa"] = {
            "status": "REJECTED",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        result["overall_status"] = "EXISTING_VERIFIER_CANNOT_CONSUME"
    result["runtime_seconds"] = time.monotonic() - started
    if _git("status", "--porcelain"):
        raise RuntimeError("α,β-CROWN repository became dirty during parser probe")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = _inside(args.output)
    if output.exists():
        raise RuntimeError(f"parser worker refuses to overwrite {output}")
    result = run(args.onnx, args.probes)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
