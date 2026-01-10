#!/usr/bin/env python3
#===- act/pipeline/confignet/cli.py - ConfigNet CLI --------------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Unified CLI driver for ConfigNet sampling and validation.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch

from act.pipeline.confignet import (
    ConfigNetConfig,
    build_generated_instance,
    sample_instances,
)
from act.pipeline.confignet.input_sampling import sample_feasible_inputs
from act.pipeline.confignet.jsonl import make_record, write_jsonl_records
from act.pipeline.confignet.seeds import derive_seed
from act.pipeline.confignet.factory_adapter import ConfignetFactoryAdapter
from act.pipeline.confignet.act_driver_l1l2 import run_confignet_l1l2
from act.pipeline.confignet.driver_levels import run_driver_levels



def _parse_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "float64":
        return torch.float64
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def _write_json(path: Optional[str], payload: Any) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _run_level2(
    instances,
    generated,
    *,
    device: str,
    dtype: torch.dtype,
    n_inputs: int,
    tf_modes: List[str],
):
    from act.pipeline.verification.validate_verifier import VerificationValidator

    validator = VerificationValidator(device=device, dtype=dtype)
    adapter = ConfignetFactoryAdapter(device=device, dtype=dtype)
    adapter.configure_for_validation(instances, generated)
    validator.factory = adapter
    summary = validator.validate_bounds(
        networks=adapter.list_networks(),
        tf_modes=tf_modes,
        num_samples=n_inputs,
    )
    results = [
        r for r in validator.validation_results
        if r.get("validation_type") == "bounds"
    ]
    by_name = {r["network"]: r for r in results}
    return summary, by_name


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    command = None
    if argv is not None:
        command = "confignet " + " ".join(argv)
    p = argparse.ArgumentParser("ConfigNet CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample", help="sample instances")
    p_sample.add_argument("--num", type=int, default=5)
    p_sample.add_argument("--seed", type=int, default=0)
    p_sample.add_argument("--out_json", type=str, default=None)

    p_l1 = sub.add_parser("level1", help="run level1 checks")
    p_l1.add_argument("--num", type=int, default=5)
    p_l1.add_argument("--seed", type=int, default=0)
    p_l1.add_argument("--n-inputs", type=int, default=50)
    p_l1.add_argument("--device", type=str, default="cpu")
    p_l1.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    p_l1.add_argument("--out_jsonl", type=str, default=None)
    p_l1.add_argument("--no-strict-input", action="store_true")
    p_l1.add_argument("--deterministic-algos", action="store_true")

    p_l2 = sub.add_parser("level2", help="run level2 bounds checks")
    p_l2.add_argument("--num", type=int, default=5)
    p_l2.add_argument("--seed", type=int, default=0)
    p_l2.add_argument("--n-inputs", type=int, default=5)
    p_l2.add_argument("--device", type=str, default="cpu")
    p_l2.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    p_l2.add_argument("--tf-modes", nargs="+", default=["interval"])
    p_l2.add_argument("--out_jsonl", type=str, default=None)
    p_l2.add_argument("--deterministic-algos", action="store_true")

    p_l1l2 = sub.add_parser("l1l2", help="run level1 then level2")
    p_l1l2.add_argument("--num", type=int, default=5)
    p_l1l2.add_argument("--seed", type=int, default=0)
    p_l1l2.add_argument("--n-inputs", type=int, default=50)
    p_l1l2.add_argument("--device", type=str, default="cpu")
    p_l1l2.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    p_l1l2.add_argument("--tf-modes", nargs="+", default=["interval"])
    p_l1l2.add_argument("--out_jsonl", type=str, default=None)
    p_l1l2.add_argument("--no-strict-input", action="store_true")
    p_l1l2.add_argument("--deterministic-algos", action="store_true")
    p_l1l2.add_argument("--atol", type=float, default=1e-6)
    p_l1l2.add_argument("--rtol", type=float, default=0.0)
    p_l1l2.add_argument("--topk", type=int, default=10)
    p_l1l2.add_argument("--no-strict", action="store_false", dest="strict", default=True)
    p_l1l2.add_argument("--run-solver", action="store_true")
    p_l1l2.add_argument(
        "--solver",
        type=str,
        default="torchlp",
        choices=["torchlp", "gurobi_lp", "gurobi_milp"],
    )
    p_l1l2.add_argument("--solver-timeout", type=float, default=None)
    p_l1l2.add_argument("--solver-on-failure", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "sample":
        cfg = ConfigNetConfig(num_instances=args.num, base_seed=args.seed)
        instances = sample_instances(cfg)
        payload = [inst.to_dict() for inst in instances]
        _write_json(args.out_json, payload)
        return 0

    if args.cmd == "level1":
        cfg = ConfigNetConfig(num_instances=args.num, base_seed=args.seed)
        dtype = _parse_dtype(args.dtype)
        records = run_driver_levels(
            cfg,
            device=args.device,
            dtype=dtype,
            base_seed=args.seed,
            n_inputs_l1=args.n_inputs,
            strict_input=(not args.no_strict_input),
            command=command,
            deterministic_algos=bool(args.deterministic_algos),
        )
        if args.out_jsonl:
            write_jsonl_records(args.out_jsonl, records)
        return 0

    if args.cmd == "level2":
        cfg = ConfigNetConfig(num_instances=args.num, base_seed=args.seed)
        dtype = _parse_dtype(args.dtype)
        instances = sample_instances(cfg)
        generated = [
            build_generated_instance(
                inst,
                device=args.device,
                dtype=dtype,
                deterministic_algos=bool(args.deterministic_algos),
            )
            for inst in instances
        ]
        _summary, by_name = _run_level2(
            instances,
            generated,
            device=args.device,
            dtype=dtype,
            n_inputs=args.n_inputs,
            tf_modes=list(args.tf_modes),
        )
        if args.out_jsonl:
            records = []
            for inst in instances:
                l2 = by_name.get(inst.instance_id)
                violations = l2.get("violations", []) if isinstance(l2, dict) else []
                violations_count = len(violations)
                topk = violations[:3]
                payload = {
                    "instance_spec": inst.to_dict(),
                    "level2_result": l2,
                    "level2_summary": {
                        "violations_count": int(violations_count),
                        "topk_violation_summary": topk,
                    },
                    "run_meta": {
                        "command": command,
                        "device": str(args.device),
                        "dtype": str(dtype),
                        "n_inputs": int(args.n_inputs),
                        "tf_modes": list(args.tf_modes),
                        "base_seed": int(args.seed),
                        "deterministic_algos": bool(args.deterministic_algos),
                    },
                }
                records.append(make_record(payload))
            write_jsonl_records(args.out_jsonl, records)
        return 0

    if args.cmd == "l1l2":
        dtype = _parse_dtype(args.dtype)
        if len(args.tf_modes) > 1:
            logging.warning("[confignet] l1l2 uses first tf_mode only: %s", args.tf_modes[0])
        driver_args = SimpleNamespace(
            instances=args.num,
            seed=args.seed,
            samples=args.n_inputs,
            tf_mode=args.tf_modes[0],
            tf_modes=args.tf_modes,
            strict=bool(args.strict),
            atol=float(args.atol),
            rtol=float(args.rtol),
            topk=int(args.topk),
            device=args.device,
            dtype=str(args.dtype),
            jsonl=args.out_jsonl or "confignet_l1l2.jsonl",
            no_strict_input=bool(args.no_strict_input),
            run_solver=bool(args.run_solver),
            solver=str(args.solver),
            solver_timeout=args.solver_timeout,
            solver_on_failure=bool(args.solver_on_failure),
        )
        run_confignet_l1l2(driver_args)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
