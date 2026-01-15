#!/usr/bin/env python3
#===- act/pipeline/confignet.py - ConfigNet Driver + CLI --------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Consolidated ConfigNet entrypoints:
#   - schema + sampler (imported from confignet_spec)
#   - builders + adapter (imported from confignet_adapter)
#   - drivers (L1/L2) + policy + CLI
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from act.back_end.verifier import verify_once
from act.back_end.solver.solver_gurobi import GurobiSolver
from act.back_end.solver.solver_torch import TorchLPSolver
from act.pipeline.verification.validate_verifier import VerificationValidator

from act.pipeline.confignet_spec import (
    ConfigNetConfig,
    InstanceSpec,
    ModelFamily,
    MLPConfig,
    CNN2DConfig,
    TemplateConfig,
    InputSpecConfig,
    OutputSpecConfig,
    GeneratedInstance,
    derive_seed,
    seed_everything,
    seeded,
    sample_instances,
    make_feasible_input,
    sample_feasible_inputs,
)
from act.pipeline.confignet_adapter import (
    build_wrapped_model,
    build_generated_instance,
    ConfignetFactoryAdapter,
)
from act.pipeline.confignet_io import (
    SCHEMA_VERSION_V2,
    build_record_v2,
    canonical_hash,
    canonical_hash_obj,
    compute_run_id,
    make_record,
    read_jsonl,
    tensor_digest,
    validate_record_v2,
    write_jsonl_records,
    write_record_jsonl,
)

logger = logging.getLogger(__name__)


# --------------------------
# Utility
# --------------------------


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


# --------------------------
# Policy / verdicts
# --------------------------


class Verdict(str, Enum):
    CERTIFIED = "CERTIFIED"
    FALSIFIED = "FALSIFIED"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"


@dataclass(frozen=True)
class GatingInfo:
    prohibited_certified_due_to_cex: bool = False
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prohibited_certified_due_to_cex": bool(self.prohibited_certified_due_to_cex),
            "reason": self.reason,
            "details": self.details or {},
        }


def apply_policy(
    *,
    base_verdict: Verdict,
    level1_found_cex: bool,
    level1_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[Verdict, GatingInfo]:
    if level1_found_cex:
        return (
            Verdict.FALSIFIED,
            GatingInfo(
                prohibited_certified_due_to_cex=True,
                reason="level1_concrete_counterexample",
                details={"level1": level1_summary or {}},
            ),
        )

    return (
        base_verdict,
        GatingInfo(prohibited_certified_due_to_cex=False, reason=None, details={}),
    )


# --------------------------
# Level1 validation (via validate_verifier)
# --------------------------


def _normalize_l1_solvers(solver_name: Optional[str]) -> List[str]:
    if not solver_name:
        return ["torchlp"]
    key = str(solver_name).lower()
    if "gurobi" in key:
        return ["gurobi"]
    if "torch" in key:
        return ["torchlp"]
    return ["torchlp"]


def _collect_l1_results(validator: VerificationValidator, instance_id: str) -> List[Dict[str, Any]]:
    return [
        r
        for r in validator.validation_results
        if r.get("validation_type") == "counterexample" and r.get("network") == instance_id
    ]


def _summarize_counterexample_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    found = any(r.get("concrete_counterexample") for r in results)
    num = sum(1 for r in results if r.get("concrete_counterexample"))
    return {
        "found_cex": bool(found),
        "num_cex": int(num),
        "first_cex": None,
    }


def _l1_from_validator_results(
    results: List[Dict[str, Any]],
    *,
    base_verdict: Verdict,
) -> Dict[str, Any]:
    errors = [r for r in results if r.get("status") == "ERROR"]
    counterexample_found = any(r.get("concrete_counterexample") for r in results)
    num_cex = sum(1 for r in results if r.get("concrete_counterexample"))
    verifier_status = "UNKNOWN"
    for r in results:
        if r.get("verifier_result") is not None:
            verifier_status = str(r.get("verifier_result"))
            break
    final_verdict, _gating = apply_policy(
        base_verdict=base_verdict,
        level1_found_cex=bool(counterexample_found),
        level1_summary={"found_cex": bool(counterexample_found), "num_cex": int(num_cex)},
    )
    return {
        "status": "ERROR" if errors else ("FAILED" if counterexample_found else "PASSED"),
        "counterexample_found": bool(counterexample_found),
        "verifier_status": verifier_status,
        "policy_applied": True,
        "final_verdict_after_policy": "FAILED"
        if final_verdict == Verdict.FALSIFIED
        else ("ERROR" if final_verdict == Verdict.ERROR else "PASS"),
        "details": {"results": results, "errors": [e.get("error") for e in errors if e.get("error")]},
    }


def run_level1_suite(
    cfg: ConfigNetConfig,
    *,
    out_jsonl: Optional[str],
    device: str,
    dtype: torch.dtype,
    n_inputs: int,
    seed: int,
    strict_input: bool,
    command: Optional[str] = None,
    deterministic_algos: bool = False,
    l1_solvers: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    cfg = replace(cfg, base_seed=int(seed))
    instances = sample_instances(cfg)
    generated = [
        build_generated_instance(
            inst,
            device=device,
            dtype=dtype,
            deterministic_algos=deterministic_algos,
        )
        for inst in instances
    ]
    adapter = ConfignetFactoryAdapter(device=device, dtype=dtype)
    adapter.configure_for_validation(instances, generated, strict_input=bool(strict_input))
    validator = VerificationValidator(device=device, dtype=dtype)
    validator.factory = adapter

    solvers = l1_solvers or ["torchlp"]
    validator.validate_counterexamples(networks=adapter.list_networks(), solvers=solvers)

    records: List[Dict[str, Any]] = []
    for inst in instances:
        results = _collect_l1_results(validator, inst.instance_id)
        summary = _summarize_counterexample_results(results)
        level1_result = {
            **summary,
            "verifier_status": (results[0].get("verifier_result") if results else "UNKNOWN"),
            "details": {"results": results},
        }
        payload = {
            "instance_spec": inst.to_dict(),
            "level1_result": level1_result,
            "input_sampling": {
                "input_sampling_status": "n/a",
                "reason": "validate_verifier",
            },
            "run_meta": {
                "command": command,
                "device": str(device),
                "dtype": str(dtype),
                "n_inputs": int(n_inputs),
                "strict_input": bool(strict_input),
                "base_seed": int(seed),
                "deterministic_algos": bool(deterministic_algos),
            },
        }
        records.append(make_record(payload))

    if out_jsonl:
        write_jsonl_records(out_jsonl, records)

    return records


# --------------------------
# Driver v1 (L1 only)
# --------------------------


SCHEMA_VERSION_V1 = "confignet.jsonl.v1"


@dataclass(frozen=True)
class DriverRunMeta:
    command: Optional[str]
    device: str
    dtype: str
    base_seed: int
    strict_input: bool
    n_inputs_l1: int
    deterministic_algos: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "device": self.device,
            "dtype": self.dtype,
            "base_seed": int(self.base_seed),
            "strict_input": bool(self.strict_input),
            "n_inputs_l1": int(self.n_inputs_l1),
            "deterministic_algos": bool(self.deterministic_algos),
        }


def _summarize_level1(level1_record: Dict[str, Any]) -> Dict[str, Any]:
    l1 = level1_record.get("level1_result") or {}
    return {
        "found_cex": bool(l1.get("found_cex", False)),
        "num_cex": int(l1.get("num_cex", 0) or 0),
        "first_cex": l1.get("first_cex", None),
    }


def run_driver_levels(
    cfg: ConfigNetConfig,
    *,
    device: str,
    dtype: torch.dtype,
    base_seed: int,
    n_inputs_l1: int,
    strict_input: bool,
    command: Optional[str] = None,
    deterministic_algos: bool = False,
) -> List[Dict[str, Any]]:
    l1_records = run_level1_suite(
        cfg,
        out_jsonl=None,
        device=device,
        dtype=dtype,
        n_inputs=n_inputs_l1,
        seed=base_seed,
        strict_input=strict_input,
        command=command,
        deterministic_algos=deterministic_algos,
    )

    run_meta = DriverRunMeta(
        command=command,
        device=str(device),
        dtype=str(dtype),
        base_seed=int(base_seed),
        strict_input=bool(strict_input),
        n_inputs_l1=int(n_inputs_l1),
        deterministic_algos=bool(deterministic_algos),
    ).to_dict()

    out: List[Dict[str, Any]] = []
    for r in l1_records:
        inst = r.get("instance_spec") or {}
        l1_summary = _summarize_level1(r)
        found_cex = bool(l1_summary["found_cex"])

        base_verdict = Verdict.UNKNOWN
        final_verdict, gating = apply_policy(
            base_verdict=base_verdict,
            level1_found_cex=found_cex,
            level1_summary=l1_summary,
        )

        payload = {
            "schema_version": SCHEMA_VERSION_V1,
            "instance_spec": inst,
            "level1": {
                "result": r.get("level1_result"),
                "input_sampling": r.get("input_sampling"),
                "summary": l1_summary,
            },
            "gating": gating.to_dict(),
            "final_verdict": final_verdict.value,
            "run_meta": run_meta,
        }
        out.append(make_record(payload))

    return out


# --------------------------
# Driver v2 (L1 + L2)
# --------------------------


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


def _derive_seeds(seed_root: int, inst_seed: int) -> Dict[str, int]:
    seed_instance = int(inst_seed)
    seed_inputs = derive_seed(seed_instance, 0, "inputs|l1l2")
    return {
        "seed_root": int(seed_root),
        "seed_instance": int(seed_instance),
        "seed_inputs": int(seed_inputs),
    }


def _run_l1(
    inst: InstanceSpec,
    generated,
    act_model,
    *,
    validator: VerificationValidator,
    adapter: ConfignetFactoryAdapter,
    solvers: List[str],
    strict_input: bool,
) -> Dict[str, Any]:
    try:
        if generated is None or act_model is None:
            raise RuntimeError("act_model unavailable")
        adapter.configure_for_validation(
            [inst],
            [generated],
            act_nets_by_name={inst.instance_id: act_model},
            strict_input=bool(strict_input),
        )
        validator.factory = adapter
        validator.validate_counterexamples(
            networks=[inst.instance_id],
            solvers=list(solvers),
        )
        results = _collect_l1_results(validator, inst.instance_id)
        if not results:
            raise RuntimeError("L1 counterexample validation produced no results")
        return _l1_from_validator_results(results, base_verdict=Verdict.UNKNOWN)
    except Exception as e:
        logger.error("[CN][l1] error: %s", e)
        return {
            "status": "ERROR",
            "counterexample_found": False,
            "verifier_status": "ERROR",
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
    l1_solvers = _normalize_l1_solvers(solver_name)
    solver_on_failure = bool(getattr(args, "solver_on_failure", True))
    solver_timeout = getattr(args, "solver_timeout", None)
    errors: List[str] = []
    warnings: List[str] = []
    passed = failed = errored = 0
    total_checks = 0
    total_violations = 0

    start = time.perf_counter()
    for idx, inst in enumerate(instances):
        inst_start = time.perf_counter()
        seeds = _derive_seeds(int(args.seed), int(inst.seed))
        strict_input = not bool(getattr(args, "no_strict_input", False))
        instance_errors: List[str] = []
        case: Dict[str, Any] = {}
        l1: Dict[str, Any] = {}
        l2: Dict[str, Any] = {}
        solver: Dict[str, Any] = {}
        instance_meta = _instance_meta_from_spec(inst)

        try:
            case = adapter.build_case(
                inst,
                seed_inputs=seeds["seed_inputs"],
                num_samples=int(args.samples),
                strict_input=strict_input,
            )
            instance_meta = case.get("instance_meta", instance_meta)

            l1 = _run_l1(
                inst,
                case["generated"],
                case.get("act_model"),
                validator=validator,
                adapter=adapter,
                solvers=l1_solvers,
                strict_input=strict_input,
            )
            l2 = _run_l2(
                inst,
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
                inst,
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
                inst,
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


# --------------------------
# CLI helpers
# --------------------------


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


def _run_level2(
    instances,
    generated,
    *,
    device: str,
    dtype: torch.dtype,
    n_inputs: int,
    tf_modes: List[str],
):
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


# --------------------------
# Smoke test
# --------------------------


def _assert_state_dict_equal(a: torch.nn.Module, b: torch.nn.Module) -> None:
    sa = a.state_dict()
    sb = b.state_dict()
    if sa.keys() != sb.keys():
        raise AssertionError(f"state_dict key mismatch: {sa.keys()} vs {sb.keys()}")
    for k in sa.keys():
        ta = sa[k]
        tb = sb[k]
        if torch.is_tensor(ta) and torch.is_tensor(tb):
            if ta.dtype != tb.dtype or ta.shape != tb.shape:
                raise AssertionError(f"tensor meta mismatch at '{k}': {ta.dtype}/{ta.shape} vs {tb.dtype}/{tb.shape}")
            if not torch.equal(ta, tb):
                diff = (ta - tb).abs().max().item() if ta.numel() > 0 else 0.0
                raise AssertionError(f"tensor value mismatch at '{k}', max_abs_diff={diff}")
        else:
            if ta != tb:
                raise AssertionError(f"non-tensor value mismatch at '{k}': {ta} vs {tb}")


def run_smoke(num: int, base_seed: int, device: str, dtype_str: str) -> None:
    dtype = torch.float64 if dtype_str == "float64" else torch.float32

    cfg = ConfigNetConfig(num_instances=num, base_seed=base_seed)
    specs = sample_instances(cfg)

    if len(specs) != num:
        raise AssertionError(f"sample_instances returned {len(specs)} specs, expected {num}")

    logger.info("Sampled %d instances (base_seed=%d).", len(specs), base_seed)

    for i, inst_spec in enumerate(specs):
        logger.info(
            "---- [%d/%d] instance_id=%s seed=%d family=%s ----",
            i + 1,
            len(specs),
            inst_spec.instance_id,
            inst_spec.seed,
            inst_spec.family.value,
        )

        m1 = build_wrapped_model(inst_spec, device=device, dtype=dtype)
        m2 = build_wrapped_model(inst_spec, device=device, dtype=dtype)
        _assert_state_dict_equal(m1, m2)

        x = make_feasible_input(inst_spec, device=torch.device(device), dtype=dtype)

        with torch.no_grad():
            out = m1(x)

        required_keys = ["output", "input_satisfied", "input_explanation", "output_satisfied", "output_explanation"]
        for k in required_keys:
            if k not in out:
                raise AssertionError(f"VerifiableModel output missing key '{k}'")

        if not isinstance(out["input_satisfied"], bool):
            raise AssertionError("input_satisfied must be bool")
        if not isinstance(out["output_satisfied"], bool):
            raise AssertionError("output_satisfied must be bool")

        if out["input_satisfied"] is not True:
            raise AssertionError(
                f"Expected input_satisfied=True for feasible input, got False. "
                f"explanation={out['input_explanation']}"
            )

        logger.info(
            "input_ok=%s | output_ok=%s | y_shape=%s",
            out["input_satisfied"],
            out["output_satisfied"],
            tuple(out["output"].shape),
        )

    logger.info("✅ ConfigNet smoke test PASS.")


# --------------------------
# CLI
# --------------------------


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

    p_smoke = sub.add_parser("smoke", help="run confignet smoke test")
    p_smoke.add_argument("--num", type=int, default=5)
    p_smoke.add_argument("--seed", type=int, default=0)
    p_smoke.add_argument("--device", type=str, default="cpu")
    p_smoke.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])

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

    if args.cmd == "smoke":
        try:
            run_smoke(
                num=int(args.num),
                base_seed=int(args.seed),
                device=str(args.device),
                dtype_str=str(args.dtype),
            )
            return 0
        except Exception as e:
            logger.error("❌ ConfigNet smoke test FAILED: %s", e)
            return 1

    return 1


__all__ = [
    "ConfigNetConfig",
    "InstanceSpec",
    "ModelFamily",
    "MLPConfig",
    "CNN2DConfig",
    "TemplateConfig",
    "InputSpecConfig",
    "OutputSpecConfig",
    "GeneratedInstance",
    "derive_seed",
    "seed_everything",
    "seeded",
    "sample_instances",
    "make_feasible_input",
    "sample_feasible_inputs",
    "build_wrapped_model",
    "build_generated_instance",
    "ConfignetFactoryAdapter",
    "Verdict",
    "GatingInfo",
    "apply_policy",
    "run_level1_suite",
    "run_driver_levels",
    "SCHEMA_VERSION_V1",
    "run_confignet_l1l2",
    "run_smoke",
    "_derive_seeds",
    "_run_l2",
    "_finalize_verdict",
    "canonical_hash",
    "canonical_hash_obj",
    "compute_run_id",
    "make_record",
    "read_jsonl",
    "tensor_digest",
    "SCHEMA_VERSION_V2",
    "build_record_v2",
    "validate_record_v2",
    "write_jsonl_records",
    "write_record_jsonl",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
