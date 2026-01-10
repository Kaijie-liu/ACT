#!/usr/bin/env python3
#===- act/pipeline/confignet/act_driver_l1l2.py - Unified Driver ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Unified orchestration: ConfigNet sampling -> L1 -> L2 -> policy -> JSONL v2.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional

import torch

from act.back_end.verifier import verify_once
from act.back_end.solver.solver_gurobi import GurobiSolver
from act.back_end.solver.solver_torch import TorchLPSolver
from act.pipeline.confignet.factory_adapter import ConfignetFactoryAdapter
from act.pipeline.confignet.jsonl import write_record_jsonl
from act.pipeline.confignet.policy import apply_policy
from act.pipeline.confignet.schema import ConfigNetConfig, InstanceSpec
from act.pipeline.confignet.schema_v2 import build_record_v2, validate_record_v2
from act.pipeline.confignet.seeds import derive_seed
from act.pipeline.confignet.sampler import sample_instances
from act.pipeline.confignet.verdicts import Verdict
from act.pipeline.confignet.level1_check import _run_level1_single
from act.pipeline.verification.validate_verifier import VerificationValidator

logger = logging.getLogger(__name__)


def _emoji(final_verdict: str) -> str:
    if final_verdict == "PASS":
        return "✅"
    if final_verdict == "FAILED":
        return "🚨"
    return "❌"


def _instance_meta_from_spec(inst: InstanceSpec) -> Dict[str, Any]:
    return {
        "net_family": str(getattr(inst.family, "value", inst.family)),
        "arch": inst.model_cfg.to_dict(),
        "spec": {
            "input_spec": inst.input_spec.to_dict(),
            "output_spec": inst.output_spec.to_dict(),
        },
        "input_shape": list(inst.model_cfg.input_shape),
        "eps": float(inst.input_spec.eps or 0.0),
    }


def _derive_seeds(seed_root: int, instance_index: int, instance_id: Optional[str] = None) -> Dict[str, int]:
    seed_instance = derive_seed(int(seed_root), int(instance_index), instance_id)
    seed_inputs = derive_seed(int(seed_instance), 0, "inputs|l1l2")
    return {
        "seed_root": int(seed_root),
        "seed_instance": int(seed_instance),
        "seed_inputs": int(seed_inputs),
    }


def _run_l1(
    inst: InstanceSpec,
    model: torch.nn.Module,
    *,
    num_samples: int,
    device: str,
    dtype: torch.dtype,
    strict_input: bool,
) -> Dict[str, Any]:
    try:
        res = _run_level1_single(
            inst,
            model,
            num_samples=num_samples,
            device=device,
            dtype=dtype,
            sampling_seed_suffix="level1",
            stop_on_first_cex=True,
            strict_input=strict_input,
        )
        base_verdict = Verdict.UNKNOWN
        final_verdict, _gating = apply_policy(
            base_verdict=base_verdict,
            level1_found_cex=bool(res.found_cex),
            level1_summary={"found_cex": bool(res.found_cex), "num_cex": int(res.num_cex)},
        )
        return {
            "status": "FAILED" if res.found_cex else "PASSED",
            "counterexample_found": bool(res.found_cex),
            "verifier_status": "NOT_RUN",
            "policy_applied": True,
            "final_verdict_after_policy": "FAILED"
            if final_verdict == Verdict.FALSIFIED
            else ("ERROR" if final_verdict == Verdict.ERROR else "PASS"),
        }
    except Exception as e:
        logger.error("[CN][l1] error: %s", e)
        return {
            "status": "ERROR",
            "counterexample_found": False,
            "verifier_status": "NOT_RUN",
            "policy_applied": True,
            "final_verdict_after_policy": "ERROR",
            "details": {"error": str(e)},
        }


def _run_l2(
    inst: InstanceSpec,
    generated,
    act_model,
    *,
    device: str,
    dtype: torch.dtype,
    tf_mode: str,
    samples: int,
    strict: bool,
    atol: float,
    rtol: float,
    topk: int,
    validator: VerificationValidator,
    adapter: ConfignetFactoryAdapter,
    seed_inputs: int,
    act_build_error: Optional[str] = None,
    strict_input: bool = True,
) -> Dict[str, Any]:
    try:
        if act_build_error or generated is None or act_model is None:
            raise RuntimeError(act_build_error or "act_model unavailable")
        adapter.configure_for_validation(
            [inst],
            [generated],
            seed_inputs_by_name={inst.instance_id: int(seed_inputs)},
            act_nets_by_name={inst.instance_id: act_model},
            strict_input=bool(strict_input),
        )
        validator.factory = adapter
        validator.validate_bounds_per_neuron(
            networks=[inst.instance_id],
            tf_modes=[tf_mode],
            num_samples=int(samples),
            strict=bool(strict),
            atol=float(atol),
            rtol=float(rtol),
            topk=int(topk),
        )
        results = [
            r
            for r in validator.validation_results
            if r.get("validation_type") == "bounds_per_neuron"
            and r.get("network") == inst.instance_id
        ]
        if not results:
            raise RuntimeError("L2 bounds_per_neuron produced no results")
        r = results[-1]
        status_raw = r.get("validation_status", "ERROR")
        if r.get("status") == "ERROR":
            status_raw = "ERROR"
        if status_raw == "PASS":
            status = "PASSED"
        elif status_raw == "PASSED":
            status = "PASSED"
        elif status_raw == "FAILED":
            status = "FAILED"
        elif status_raw == "ERROR":
            status = "ERROR"
        else:
            status = "ERROR"
        worst_gap = 0.0
        worst_layer = None
        layerwise = r.get("layerwise_stats", [])
        for s in layerwise:
            gap = float(s.get("max_gap", 0.0))
            if gap >= worst_gap and int(s.get("num_violations", 0)) > 0:
                worst_gap = gap
                worst_layer = {
                    "layer_id": s.get("layer_id"),
                    "layer_kind": s.get("kind"),
                    "shape": s.get("shape"),
                    "num_violations": s.get("num_violations"),
                    "num_neurons": s.get("num_neurons"),
                    "max_gap": gap,
                }
        return {
            "status": status,
            "tf_mode": tf_mode,
            "samples": int(samples),
            "checks_total": int(r.get("total_checks", 0)),
            "violations_total": int(r.get("violations_total", 0)),
            "worst_gap": float(worst_gap),
            "worst_layer": worst_layer,
            "topk": list(r.get("violations_topk", [])),
            "layerwise_stats": list(layerwise),
            "errors": list(r.get("errors", [])),
            "warnings": list(r.get("warnings", [])),
        }
    except Exception as e:
        logger.error("[CN][l2] error: %s", e)
        return {
            "status": "ERROR",
            "tf_mode": tf_mode,
            "samples": int(samples),
            "checks_total": 0,
            "violations_total": 0,
            "worst_gap": None,
            "worst_layer": None,
            "topk": [],
            "layerwise_stats": [],
            "errors": [str(e)],
            "warnings": [],
        }


def _run_solver(
    inst: InstanceSpec,
    case: Dict[str, Any],
    *,
    solver_name: str,
    timeout_s: Optional[float],
    run_solver: bool,
    skip_reason: Optional[str] = None,
) -> Dict[str, Any]:
    if not run_solver:
        return {
            "status": "NOT_RUN",
            "solver_name": str(solver_name),
            "verdict": "NOT_RUN",
            "time_sec": None,
            "errors": [],
            "warnings": [],
            "details": {"reason": skip_reason} if skip_reason else {},
        }

    act_model = case.get("act_model")
    if act_model is None:
        return {
            "status": "ERROR",
            "solver_name": str(solver_name),
            "verdict": "ERROR",
            "time_sec": None,
            "errors": ["act_model unavailable"],
            "warnings": [],
            "details": {},
        }

    solver_key = str(solver_name).lower()
    if solver_key in ("torchlp", "torch_lp"):
        solver = TorchLPSolver()
    elif solver_key in ("gurobi", "gurobi_lp", "gurobi_milp"):
        solver = GurobiSolver()
    else:
        return {
            "status": "ERROR",
            "solver_name": str(solver_name),
            "verdict": "ERROR",
            "time_sec": None,
            "errors": [f"unknown solver: {solver_name}"],
            "warnings": [],
            "details": {},
        }

    start = time.perf_counter()
    try:
        res = verify_once(act_model, solver=solver, timelimit=timeout_s)
        return {
            "status": "PASSED",
            "solver_name": str(solver_name),
            "verdict": str(res.status),
            "time_sec": float(time.perf_counter() - start),
            "errors": [],
            "warnings": [],
            "details": dict(res.stats) if getattr(res, "stats", None) is not None else {},
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "solver_name": str(solver_name),
            "verdict": "ERROR",
            "time_sec": float(time.perf_counter() - start),
            "errors": [str(e)],
            "warnings": [],
            "details": {},
        }


def _finalize_verdict(
    l1: Dict[str, Any],
    l2: Dict[str, Any],
    solver: Dict[str, Any],
    errors: List[str],
) -> Dict[str, Any]:
    if (
        errors
        or l1.get("status") == "ERROR"
        or l2.get("status") == "ERROR"
        or solver.get("status") == "ERROR"
    ):
        return {"final_verdict": "ERROR", "exit_code": 2, "reason": "error"}
    if l1.get("final_verdict_after_policy") != "PASS":
        return {"final_verdict": "FAILED", "exit_code": 1, "reason": "l1_failed"}
    if solver.get("verdict") == "FALSIFIED":
        return {"final_verdict": "FAILED", "exit_code": 1, "reason": "solver_falsified"}
    if l2.get("status") == "FAILED":
        return {"final_verdict": "FAILED", "exit_code": 1, "reason": "l2_failed"}
    return {"final_verdict": "PASS", "exit_code": 0, "reason": "ok"}


def run_confignet_l1l2(args) -> Dict[str, Any]:
    """
    Unified driver: ConfigNet sampling -> L1 -> L2 -> JSONL v2 -> summary.
    """
    dtype = torch.float64 if str(args.dtype) == "float64" else torch.float32
    device = str(args.device)
    tf_mode = args.tf_mode if hasattr(args, "tf_mode") else args.tf_modes[0]

    cfg = ConfigNetConfig(num_instances=int(args.instances), base_seed=int(args.seed))
    instances = sample_instances(cfg)

    run_meta = {
        "seed_root": int(args.seed),
        "instances": int(args.instances),
        "samples": int(args.samples),
        "tf_mode": str(tf_mode),
        "strict": bool(args.strict),
        "device": str(device),
        "dtype": str(args.dtype),
        "atol": float(args.atol),
        "rtol": float(args.rtol),
        "topk": int(args.topk),
    }

    adapter = ConfignetFactoryAdapter(device=device, dtype=dtype)
    validator = VerificationValidator(device=device, dtype=dtype)
    run_solver = bool(getattr(args, "run_solver", False))
    solver_name = str(getattr(args, "solver", "torchlp"))
    solver_on_failure = bool(getattr(args, "solver_on_failure", False))
    solver_timeout = getattr(args, "solver_timeout", None)
    errors: List[str] = []
    warnings: List[str] = []
    passed = failed = errored = 0
    total_checks = 0
    total_violations = 0

    start = time.perf_counter()
    for idx, inst in enumerate(instances):
        inst_start = time.perf_counter()
        seeds = _derive_seeds(int(args.seed), idx, inst.instance_id)
        inst_seeded = replace(inst, seed=int(seeds["seed_instance"]))
        strict_input = not bool(getattr(args, "no_strict_input", False))
        instance_errors: List[str] = []
        case: Dict[str, Any] = {}
        l1: Dict[str, Any] = {}
        l2: Dict[str, Any] = {}
        solver: Dict[str, Any] = {}
        instance_meta = _instance_meta_from_spec(inst_seeded)

        try:
            case = adapter.build_case(
                inst_seeded,
                seed_inputs=seeds["seed_inputs"],
                num_samples=int(args.samples),
                strict_input=strict_input,
            )
            instance_meta = case.get("instance_meta", instance_meta)

            l1 = _run_l1(
                inst_seeded,
                case["torch_model"],
                num_samples=int(args.samples),
                device=device,
                dtype=dtype,
                strict_input=strict_input,
            )
            l2 = _run_l2(
                inst_seeded,
                case["generated"],
                case.get("act_model"),
                device=device,
                dtype=dtype,
                tf_mode=str(tf_mode),
                samples=int(args.samples),
                strict=bool(args.strict),
                atol=float(args.atol),
                rtol=float(args.rtol),
                topk=int(args.topk),
                validator=validator,
                adapter=adapter,
                seed_inputs=seeds["seed_inputs"],
                act_build_error=case.get("act_build_error"),
                strict_input=strict_input,
            )
            l1_failed = l1.get("status") == "FAILED"
            run_solver_this = run_solver and (solver_on_failure or not l1_failed)
            solver = _run_solver(
                inst_seeded,
                case,
                solver_name=solver_name,
                timeout_s=solver_timeout,
                run_solver=run_solver_this,
                skip_reason="skipped_l1_failed" if (run_solver and l1_failed and not solver_on_failure) else None,
            )
        except Exception as e:
            instance_errors.append(f"{type(e).__name__}: {e}")
            l1 = {
                "status": "ERROR",
                "counterexample_found": False,
                "verifier_status": "NOT_RUN",
                "policy_applied": True,
                "final_verdict_after_policy": "ERROR",
            }
            l2 = {
                "status": "ERROR",
                "tf_mode": str(tf_mode),
                "samples": int(args.samples),
                "checks_total": 0,
                "violations_total": 0,
                "worst_gap": None,
                "worst_layer": None,
                "topk": [],
                "layerwise_stats": [],
                "errors": list(instance_errors),
                "warnings": [],
            }
            solver = _run_solver(
                inst_seeded,
                case,
                solver_name=solver_name,
                timeout_s=solver_timeout,
                run_solver=run_solver,
                skip_reason="skipped_exception",
            )

        if l1.get("status") == "ERROR":
            instance_errors.append("l1_error")
        instance_errors.extend(list(l2.get("errors", [])))
        if solver.get("status") == "ERROR":
            instance_errors.extend(list(solver.get("errors", [])))

        final = _finalize_verdict(l1, l2, solver, instance_errors)
        status = final["final_verdict"]
        if status == "PASS":
            passed += 1
        elif status == "FAILED":
            failed += 1
        else:
            errored += 1

        total_checks += int(l2.get("checks_total", 0))
        total_violations += int(l2.get("violations_total", 0))

        record = build_record_v2(
            run_meta=run_meta,
            seeds=seeds,
            instance=instance_meta,
            l1=l1,
            l2=l2,
            solver=solver,
            final=final,
            timing={"wall_time": time.perf_counter() - inst_start},
            errors=instance_errors,
            warnings=list(l2.get("warnings", [])),
        )
        validate_record_v2(record)
        write_record_jsonl(str(args.jsonl), record)

        logger.info(
            "%s [CN][l1l2] idx=%d run_id=%s l1=%s l2=%s final=%s time=%.2fs",
            _emoji(status),
            idx,
            record["run_id"],
            l1.get("status"),
            l2.get("status"),
            status,
            float(record.get("timing", {}).get("wall_time", 0.0)),
        )
        if status == "FAILED":
            if l2.get("status") == "FAILED":
                logger.info(
                    "[CN][l1l2] violations_total=%s worst_gap=%s worst_layer=%s",
                    l2.get("violations_total"),
                    l2.get("worst_gap"),
                    l2.get("worst_layer"),
                )
                topk = list(l2.get("topk", []))[:3]
                logger.info("[CN][l1l2] topk:")
                for v in topk:
                    logger.info("  - %s", v)
            if l1.get("status") == "FAILED":
                logger.info(
                    "[CN][l1l2] counterexample_found=%s",
                    l1.get("counterexample_found"),
                )
                if l1.get("details"):
                    logger.info("[CN][l1l2] l1_details=%s", l1.get("details"))
        if status == "ERROR":
            logger.info("[CN][l1l2] errors:")
            for e in instance_errors[:3]:
                logger.info("  - %s", e)

    elapsed = time.perf_counter() - start
    exit_code = 2 if errored > 0 else (1 if failed > 0 else 0)
    logger.info(
        "Summary: passed=%d failed=%d errors=%d records=%d checks=%d viol=%d time=%.2fs",
        passed,
        failed,
        errored,
        len(instances),
        total_checks,
        total_violations,
        elapsed,
    )
    logger.info("ExitCode: %d", exit_code)
    logger.info("JSONL: %s", args.jsonl)

    return {
        "passed": passed,
        "failed": failed,
        "errors": errored,
        "records": len(instances),
        "total_checks": total_checks,
        "total_violations": total_violations,
        "exit_code": exit_code,
    }
