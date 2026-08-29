"""Probe α,β-CROWN's actual ONNX conversion and bound-graph front end."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch


MOE_ROOT = Path("/data1/Kane/MOE")
CROWN_REPO = MOE_ROOT / "baselines/alpha-beta-CROWN"
CROWN_FRONTEND = CROWN_REPO / "complete_verifier"
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

    def finish(value: dict[str, Any]) -> dict[str, Any]:
        value["runtime_seconds"] = time.monotonic() - started
        if _git("status", "--porcelain"):
            raise RuntimeError(
                "alpha-beta-CROWN repository became dirty during parser probe"
            )
        return value

    result: dict[str, Any] = {
        "schema_version": 1,
        "crown_repository": str(CROWN_REPO),
        "crown_commit": CROWN_COMMIT,
        "onnx": str(onnx_path),
        "probes": str(probes_path),
        "onnx2pytorch": {"status": "NOT_RUN"},
        "alpha_beta_crown_loader": {"status": "NOT_RUN"},
        "auto_lirpa": {"status": "NOT_RUN"},
        "vnnlib_stage": {"status": "NOT_RUN"},
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

    sys.path.insert(0, str(CROWN_FRONTEND))
    try:
        import arguments

        arguments.Config.parse_config(["--device=cpu"], verbose=False)
        from load_model import load_model_onnx

        loaded, input_shape = load_model_onnx(str(onnx_path), x=inputs[:1])
        with torch.no_grad():
            loaded_output = loaded(inputs)
        maximum_error = float((loaded_output - expected).abs().max().item())
        loader_matches = bool(
            torch.allclose(loaded_output, expected, atol=1e-4, rtol=1e-5)
        )
        result["alpha_beta_crown_loader"] = {
            "status": (
                "ACCEPTED_SEMANTICS_MATCH"
                if loader_matches
                else "ACCEPTED_SILENT_SEMANTIC_MISMATCH"
            ),
            "input_shape": list(input_shape),
            "maximum_abs_error": maximum_error,
        }
    except Exception as error:
        result["alpha_beta_crown_loader"] = {
            "status": "REJECTED",
            "error_type": type(error).__name__,
            "error": str(error),
        }

    if result["alpha_beta_crown_loader"]["status"] == "REJECTED":
        result["vnnlib_stage"] = {
            "status": "NOT_REACHED_MODEL_FRONTEND_REJECTED"
        }
        result["overall_status"] = "EXISTING_VERIFIER_CANNOT_CONSUME"
        return finish(result)
    if result["alpha_beta_crown_loader"]["status"] != "ACCEPTED_SEMANTICS_MATCH":
        result["vnnlib_stage"] = {
            "status": "NOT_REACHED_SILENT_SEMANTIC_MISMATCH"
        }
        result["overall_status"] = "SILENT_SEMANTIC_MISMATCH"
        return finish(result)
    if result["onnx2pytorch"]["status"] != "ACCEPTED_SEMANTICS_MATCH":
        raise RuntimeError(
            "official loader accepted a graph rejected by its direct conversion component"
        )
    try:
        from auto_LiRPA import BoundedModule

        bounded = BoundedModule(loaded, inputs[:1], device="cpu")
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
        result["vnnlib_stage"] = {
            "status": "NOT_RUN_FRONTEND_PROGRAM_CONSUMPTION_SCOPE"
        }
    except Exception as error:
        result["auto_lirpa"] = {
            "status": "REJECTED",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        result["overall_status"] = "EXISTING_VERIFIER_CANNOT_CONSUME"
    return finish(result)


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
