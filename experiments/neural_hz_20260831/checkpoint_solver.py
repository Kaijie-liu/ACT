#!/usr/bin/env python3
"""Solve and concretely validate one isolated exact-HZ checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import resource
import subprocess
import sys
import time
from pathlib import Path


def _status_name(status) -> str:
    value = getattr(status, "value", None)
    return str(value if value is not None else status).upper()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bench")
    parser.add_argument("iid", type=int)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--act-root", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--vnnlib-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--solver-timeout", type=float, required=True)
    parser.add_argument("--memory-gb", type=float, default=16.0)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite solver result: {args.output}")
    if "/data1/Kane/HyZor" in str(args.output.resolve()):
        raise ValueError("historical /data1/Kane/HyZor results are read-only")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    cap = int(args.memory_gb * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    sys.path.insert(0, str(args.act_root))

    started = time.monotonic()
    with args.checkpoint.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if checkpoint.get("schema") != "neural_hz_checkpoint_v1":
        raise ValueError("unsupported HZ checkpoint schema")
    if checkpoint.get("bench") != args.bench or int(checkpoint.get("iid")) != args.iid:
        raise ValueError("HZ checkpoint benchmark identity mismatch")
    final_hz = checkpoint.get("final_hz")
    input_hz = checkpoint.get("input_hz")
    if final_hz is None or not bool(getattr(final_hz, "exact", False)):
        raise ValueError("checkpoint does not contain an exact final HZ")

    import torch
    from act.back_end.solver.solver_hz import HZSolver
    from act.front_end.spec_creator_base import LabeledInputTensor
    from act.front_end.vnnlib_loader.onnx_converter import (
        convert_onnx_to_pytorch,
        get_onnx_input_shape,
    )
    from act.front_end.vnnlib_loader.vnnlib_parser import parse_vnnlib_queries

    root = args.bench_root / args.bench
    rows = [
        line.split(",")
        for line in (root / "instances.csv").read_text().splitlines()
        if line.strip()
    ]
    model_path = root / rows[args.iid][0].replace("./", "")
    spec_path = args.vnnlib_root / args.bench / rows[args.iid][1].replace("./", "")
    input_shape = tuple(get_onnx_input_shape(model_path))
    expected_shape = tuple(int(value) for value in checkpoint["input_shape"])
    if input_shape != expected_shape:
        raise ValueError(
            f"checkpoint input shape mismatch: {expected_shape} vs {input_shape}"
        )
    labeled = LabeledInputTensor(
        tensor=torch.zeros(input_shape, dtype=torch.float64),
        label=torch.tensor([0]),
    )
    queries = parse_vnnlib_queries(spec_path, labeled_tensor=labeled)
    if len(queries) != 1:
        raise ValueError("checkpoint solver requires exactly one query")
    input_spec, output_spec = queries[0]
    input_spec.lb = input_spec.lb.double()
    input_spec.ub = input_spec.ub.double()

    solver = HZSolver(
        time_limit=args.solver_timeout,
        tolerance=1e-7,
    )
    tick = time.monotonic()
    verdicts = solver.evaluate_spec(
        final_hz,
        output_spec,
        batch_size=1,
        n_out=final_hz.n_out,
        input_hz=input_hz,
        input_shape=input_shape,
        timelimit=args.solver_timeout,
    )
    solver_s = time.monotonic() - tick

    statuses = []
    metadata = []
    concrete_validations = []
    model = None
    for verdict in verdicts:
        status = _status_name(verdict.status)
        validation = None
        if status == "FALSIFIED":
            validation = {
                "solver_status": status,
                "valid": False,
                "reason": "missing_counterexample",
            }
            counterexample = verdict.counterexample
            if counterexample is not None:
                if model is None:
                    model = convert_onnx_to_pytorch(model_path).eval()
                model_dtype = next(model.parameters()).dtype
                candidate = counterexample.detach().to(dtype=model_dtype)
                if candidate.shape != input_spec.lb.shape:
                    candidate = candidate.reshape(input_spec.lb.shape)
                input_ok = bool(
                    torch.all(candidate >= input_spec.lb.to(candidate)).item()
                    and torch.all(candidate <= input_spec.ub.to(candidate)).item()
                )
                with torch.no_grad():
                    concrete_output = model(candidate).reshape(1, -1)
                encoded = output_spec.encode_linear(
                    B=1,
                    n_out=concrete_output.shape[1],
                    device=torch.device("cpu"),
                    dtype=torch.float64,
                )
                coefficients = encoded["C"].detach().cpu().double()
                thresholds = encoded["thresholds"].detach().cpu().double()
                margins = (
                    coefficients
                    @ concrete_output.detach().cpu().double().reshape(-1)
                ).reshape(1, int(encoded["M"]))
                kind_name = getattr(encoded["kind"], "name", str(encoded["kind"]))
                if "UNSAFE_LINEAR" in kind_name.upper():
                    violates = bool(torch.all(margins <= thresholds).item())
                else:
                    violates = bool(torch.any(margins >= thresholds).item())
                valid = input_ok and violates
                validation = {
                    "solver_status": status,
                    "valid": valid,
                    "reason": (
                        "concrete_network_violation" if valid else "invalid_witness"
                    ),
                    "input_ok": input_ok,
                    "violates": violates,
                    "counterexample": candidate.detach().cpu().reshape(-1).tolist(),
                    "output": concrete_output.detach().cpu().reshape(-1).tolist(),
                    "margins": margins.reshape(-1).tolist(),
                    "thresholds": thresholds.reshape(-1).tolist(),
                }
            if not bool(validation["valid"]):
                status = "UNKNOWN"
        statuses.append(status)
        metadata.append(dict(verdict.metadata))
        if validation is not None:
            concrete_validations.append(validation)

    record = {
        "schema": "neural_hz_checkpoint_solver_v1",
        "formal_baseline_solved": 1870,
        "formal_baseline_total": 2413,
        "bench": args.bench,
        "iid": args.iid,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_provenance": checkpoint.get("provenance"),
        "solver_runner": {
            "branch": subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=args.act_root, text=True
            ).strip(),
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=args.act_root, text=True
            ).strip(),
        },
        "solver_timeout_s": args.solver_timeout,
        "solver_s": solver_s,
        "worker_wall_s": time.monotonic() - started,
        "output_hz_exact": bool(final_hz.exact),
        "input_hz_available": input_hz is not None,
        "n_cont": final_hz.n_cont,
        "n_bin": final_hz.n_bin,
        "n_eq": final_hz.n_eq,
        "n_ineq": final_hz.n_ineq,
        "statuses": statuses,
        "metadata": metadata,
        "concrete_validations": concrete_validations,
        "verdict": (
            "ADV"
            if statuses and all(status == "FALSIFIED" for status in statuses)
            else "CERT"
            if statuses and all(status == "VERIFIED" for status in statuses)
            else "UNKNOWN"
        ),
    }
    with args.output.open("x") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
