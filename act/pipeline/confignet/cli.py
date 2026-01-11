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
    ModelFamily,
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


def _parse_families(families: Optional[List[str]]) -> Optional[Tuple[ModelFamily, ...]]:
    if not families:
        return None
    mapping = {
        "mlp": ModelFamily.MLP,
        "cnn2d": ModelFamily.CNN2D,
        "template": ModelFamily.TEMPLATE,
    }
    out = []
    for name in families:
        key = str(name).lower()
        if key not in mapping:
            raise ValueError(f"Unsupported family: {name}")
        out.append(mapping[key])
    return tuple(out)


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

    p_vv = sub.add_parser("validate_verifier", help="run validate_verifier on confignet instances")
    p_vv.add_argument("--num", type=int, default=5)
    p_vv.add_argument("--seed", type=int, default=0)
    p_vv.add_argument("--device", type=str, default="cpu")
    p_vv.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    p_vv.add_argument(
        "--mode",
        type=str,
        default="comprehensive",
        choices=["counterexample", "bounds", "bounds_per_neuron", "comprehensive"],
    )
    p_vv.add_argument("--solvers", nargs="+", default=["torchlp", "gurobi"])
    p_vv.add_argument("--tf-modes", nargs="+", default=["interval"])
    p_vv.add_argument("--n-inputs", type=int, default=5)
    p_vv.add_argument("--atol", type=float, default=1e-6)
    p_vv.add_argument("--rtol", type=float, default=0.0)
    p_vv.add_argument("--topk", type=int, default=10)
    p_vv.add_argument("--no-strict", action="store_false", dest="strict", default=True)
    p_vv.add_argument("--deterministic-algos", action="store_true")
    p_vv.add_argument(
        "--families",
        nargs="+",
        choices=["mlp", "cnn2d", "template"],
        default=None,
        help="restrict confignet families for validation",
    )

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

    if args.cmd == "validate_verifier":
        from act.pipeline.verification.validate_verifier import VerificationValidator

        families = _parse_families(getattr(args, "families", None))
        if families is None:
            cfg = ConfigNetConfig(num_instances=args.num, base_seed=args.seed)
        else:
            cfg = ConfigNetConfig(
                num_instances=args.num,
                base_seed=args.seed,
                families=families,
            )
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
        adapter = ConfignetFactoryAdapter(device=args.device, dtype=dtype)
        adapter.configure_for_validation(instances, generated)
        validator = VerificationValidator(device=args.device, dtype=dtype)
        validator.factory = adapter

        if args.mode == "counterexample":
            summary = validator.validate_counterexamples(
                networks=adapter.list_networks(),
                solvers=list(args.solvers),
            )
            return 1 if (summary.get("failed", 0) > 0 or summary.get("errors", 0) > 0) else 0
        if args.mode == "bounds":
            summary = validator.validate_bounds(
                networks=adapter.list_networks(),
                tf_modes=list(args.tf_modes),
                num_samples=int(args.n_inputs),
            )
            return 1 if (summary.get("failed", 0) > 0 or summary.get("errors", 0) > 0) else 0
        if args.mode == "bounds_per_neuron":
            summary = validator.validate_bounds_per_neuron(
                networks=adapter.list_networks(),
                tf_modes=list(args.tf_modes),
                num_samples=int(args.n_inputs),
                atol=float(args.atol),
                rtol=float(args.rtol),
                topk=int(args.topk),
                strict=bool(args.strict),
            )
            return 1 if (summary.get("failed", 0) > 0 or summary.get("errors", 0) > 0) else 0

        combined = validator.validate_comprehensive(
            networks=adapter.list_networks(),
            solvers=list(args.solvers),
            tf_modes=list(args.tf_modes),
            num_samples=int(args.n_inputs),
        )
        return 1 if combined.get("overall_status") in ("FAILED", "ERROR") else 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
