#!/usr/bin/env python3
#===- act/pipeline/confignet/level1_check.py - Level1 Behaviour Check ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Level 1 behavioural check (counterexample-based):
#     For sampled concrete inputs x that satisfy the input spec,
#     run the wrapped model and check OutputSpecLayer satisfaction.
#
#   If any concrete counterexample is found (output_satisfied=False),
#   Level1 must report found_cex=True, and higher-level driver must ensure
#   CERTIFIED is NOT returned for this instance.
#
# This module:
#   - imports only pipeline-side ConfigNet + front_end wrappers
#   - does NOT import back_end analyzers or solvers
#
# CLI:
#   python -m act.pipeline.confignet.level1_check --num 5 --seed 123 --samples 50 --device cpu --dtype float64
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .schema import ConfigNetConfig, InstanceSpec
from .sampler import sample_instances
from .builders import build_wrapped_model, build_generated_instance
from .input_sampling import sample_feasible_inputs
from .seeds import derive_seed
from .jsonl import make_record, write_jsonl_records, tensor_digest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Level1Cex:
    """A concrete counterexample witness for Level 1."""
    sample_idx: int
    x: torch.Tensor  # stored on CPU
    output: torch.Tensor  # stored on CPU
    output_explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_idx": int(self.sample_idx),
            "x": tensor_digest(self.x),
            "output": tensor_digest(self.output),
            "output_explanation": str(self.output_explanation),
        }


@dataclass(frozen=True)
class Level1InstanceResult:
    """Per-instance Level 1 result."""
    instance_id: str
    seed: int
    family: str
    num_samples: int
    found_cex: bool
    num_cex: int
    first_cex: Optional[Level1Cex]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "seed": int(self.seed),
            "family": self.family,
            "num_samples": int(self.num_samples),
            "found_cex": bool(self.found_cex),
            "num_cex": int(self.num_cex),
            "first_cex": self.first_cex.to_dict() if self.first_cex is not None else None,
        }


@dataclass(frozen=True)
class Level1BatchResult:
    """Batch result for a list of instances."""
    base_seed: int
    num_instances: int
    num_samples_per_instance: int
    any_cex: bool
    results: Tuple[Level1InstanceResult, ...]


def _parse_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "float64":
        return torch.float64
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def run_level1_check(
    instances: Sequence[InstanceSpec],
    *,
    num_samples: int,
    device: str,
    dtype: torch.dtype,
    deterministic_algos: bool = False,
    # derive per-instance sampling seed from base_seed + instance seed + suffix
    sampling_seed_suffix: str = "level1",
    stop_on_first_cex: bool = True,
    strict_input: bool = True,
) -> Tuple[Level1InstanceResult, ...]:
    """
    Run Level 1 behavioural check on a sequence of InstanceSpec.

    Args:
        instances: list of sampled instance specs
        num_samples: number of feasible input samples per instance
        device: torch device string
        dtype: torch dtype
        sampling_seed_suffix: included in derived sampling seed for determinism
        stop_on_first_cex: if True, stop sampling an instance after first cex

    Returns:
        Tuple of Level1InstanceResult
    """
    out: List[Level1InstanceResult] = []

    for inst in instances:
        # Build wrapped model (builders must seed from inst.seed internally for deterministic weights).
        model = build_wrapped_model(
            inst,
            device=device,
            dtype=dtype,
            deterministic_algos=deterministic_algos,
        )
        model.eval()
        out.append(
            _run_level1_single(
                inst,
                model,
                num_samples=num_samples,
                device=device,
                dtype=dtype,
                sampling_seed_suffix=sampling_seed_suffix,
                stop_on_first_cex=stop_on_first_cex,
                strict_input=strict_input,
            )
        )

    return tuple(out)


def _run_level1_single(
    inst: InstanceSpec,
    model: torch.nn.Module,
    *,
    num_samples: int,
    device: str,
    dtype: torch.dtype,
    sampling_seed_suffix: str,
    stop_on_first_cex: bool,
    strict_input: bool,
) -> Level1InstanceResult:
    model.eval()
    # Sampling seed: stable for this instance and this check mode
    samp_seed = derive_seed(inst.seed, 0, f"inputs|{sampling_seed_suffix}")

    xs = sample_feasible_inputs(
        inst,
        num_samples=num_samples,
        seed=samp_seed,
        device=device,
        dtype=dtype,
        strict_input=strict_input,
    )

    num_cex = 0
    first_cex: Optional[Level1Cex] = None

    with torch.no_grad():
        for j, x in enumerate(xs):
            # Forward output is a dict from VerifiableModel
            ret = model(x)

            # Defensive: tolerate both dict output and raw tensor (if user swaps wrapper)
            if isinstance(ret, dict):
                y = ret.get("output", None)
                ok = ret.get("output_satisfied", True)
                expl = ret.get("output_explanation", "")
                in_ok = ret.get("input_satisfied", True)
                in_expl = ret.get("input_explanation", "")
            else:
                y = ret
                ok = True
                expl = "No OutputSpecLayer / non-VerifiableModel output"
                in_ok = True
                in_expl = ""

            if strict_input and (not bool(in_ok)):
                raise RuntimeError(
                    f"[L1][BUG] sampled input violates InputSpec "
                    f"(instance_id={inst.instance_id}, sample_idx={j}). "
                    f"explanation={in_expl}"
                )

            if not bool(ok):
                num_cex += 1
                if first_cex is None:
                    first_cex = Level1Cex(
                        sample_idx=j,
                        x=x.detach().to("cpu"),
                        output=y.detach().to("cpu") if torch.is_tensor(y) else torch.tensor([]),
                        output_explanation=str(expl),
                    )
                if stop_on_first_cex:
                    break

    return Level1InstanceResult(
        instance_id=inst.instance_id,
        seed=int(inst.seed),
        family=str(getattr(inst.family, "value", inst.family)),
        num_samples=int(num_samples),
        found_cex=(num_cex > 0),
        num_cex=int(num_cex),
        first_cex=first_cex,
    )


def _evaluate_level1_with_inputs(
    inst: InstanceSpec,
    model: torch.nn.Module,
    *,
    xs: Sequence[torch.Tensor],
    num_samples: int,
    strict_input: bool,
) -> Level1InstanceResult:
    num_cex = 0
    first_cex: Optional[Level1Cex] = None

    with torch.no_grad():
        for j, x in enumerate(xs):
            ret = model(x)

            if isinstance(ret, dict):
                y = ret.get("output", None)
                ok = ret.get("output_satisfied", True)
                expl = ret.get("output_explanation", "")
                in_ok = ret.get("input_satisfied", True)
                in_expl = ret.get("input_explanation", "")
            else:
                y = ret
                ok = True
                expl = "No OutputSpecLayer / non-VerifiableModel output"
                in_ok = True
                in_expl = ""

            if strict_input and (not bool(in_ok)):
                raise RuntimeError(
                    f"[L1][BUG] sampled input violates InputSpec "
                    f"(instance_id={inst.instance_id}, sample_idx={j}). "
                    f"explanation={in_expl}"
                )

            if not bool(ok):
                num_cex += 1
                if first_cex is None:
                    first_cex = Level1Cex(
                        sample_idx=j,
                        x=x.detach().to("cpu"),
                        output=y.detach().to("cpu") if torch.is_tensor(y) else torch.tensor([]),
                        output_explanation=str(expl),
                    )
                break

    return Level1InstanceResult(
        instance_id=inst.instance_id,
        seed=int(inst.seed),
        family=str(getattr(inst.family, "value", inst.family)),
        num_samples=int(num_samples),
        found_cex=(num_cex > 0),
        num_cex=int(num_cex),
        first_cex=first_cex,
    )


def run_level1_check_from_confignet(
    *,
    num_instances: int,
    base_seed: int,
    num_samples: int,
    device: str,
    dtype: torch.dtype,
    deterministic_algos: bool = False,
) -> Level1BatchResult:
    """
    Convenience wrapper: sample instances via ConfigNetConfig then run Level1.
    """
    cfg = ConfigNetConfig(num_instances=num_instances, base_seed=base_seed)
    instances = sample_instances(cfg)
    results = run_level1_check(
        instances,
        num_samples=num_samples,
        device=device,
        dtype=dtype,
        deterministic_algos=deterministic_algos,
    )
    any_cex = any(r.found_cex for r in results)
    return Level1BatchResult(
        base_seed=int(base_seed),
        num_instances=int(num_instances),
        num_samples_per_instance=int(num_samples),
        any_cex=bool(any_cex),
        results=results,
    )


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
) -> List[Dict[str, Any]]:
    """
    Run Level1 over ConfigNet samples and optionally write JSONL audit records.
    """
    from dataclasses import replace

    cfg = replace(cfg, base_seed=int(seed))
    instances = sample_instances(cfg)
    records: List[Dict[str, Any]] = []

    for inst in instances:
        generated = build_generated_instance(
            inst,
            device=device,
            dtype=dtype,
            deterministic_algos=deterministic_algos,
        )
        samp_seed = derive_seed(inst.seed, 0, "inputs|level1")
        xs, sampling_status = sample_feasible_inputs(
            inst,
            num_samples=n_inputs,
            seed=samp_seed,
            device=device,
            dtype=dtype,
            strict_input=strict_input,
            return_status=True,
        )
        result = _evaluate_level1_with_inputs(
            inst,
            generated.wrapped_model,
            xs=xs,
            num_samples=n_inputs,
            strict_input=strict_input,
        )
        payload = {
            "instance_spec": inst.to_dict(),
            "level1_result": result.to_dict(),
            "input_sampling": sampling_status,
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


def _print_summary(batch: Level1BatchResult) -> None:
    logger.info(
        "[L1] base_seed=%d instances=%d samples/inst=%d any_cex=%s",
        batch.base_seed,
        batch.num_instances,
        batch.num_samples_per_instance,
        batch.any_cex,
    )
    for r in batch.results:
        if r.found_cex:
            ce = r.first_cex
            ce_str = ""
            if ce is not None:
                ce_str = f" | first_cex@{ce.sample_idx} expl={ce.output_explanation}"
            logger.info("[L1][FAIL] id=%s seed=%d family=%s num_cex=%d%s",
                        r.instance_id, r.seed, r.family, r.num_cex, ce_str)
        else:
            logger.info("[L1][PASS] id=%s seed=%d family=%s (no cex in %d samples)",
                        r.instance_id, r.seed, r.family, r.num_samples)


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser("ConfigNet Level1 behaviour check (counterexample-based)")
    p.add_argument("--num", type=int, default=5, help="number of instances to sample")
    p.add_argument("--seed", type=int, default=0, help="base seed for ConfigNet sampling")
    p.add_argument("--samples", type=int, default=50, help="number of feasible concrete inputs per instance")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    p.add_argument("--stop-on-first-cex", action="store_true", help="stop per instance when first cex is found")
    p.add_argument("--no-strict-input", action="store_true", help="do not error if sampled x violates InputSpec")
    args = p.parse_args()

    try:
        dtype = _parse_dtype(args.dtype)
        batch = run_level1_check_from_confignet(
            num_instances=args.num,
            base_seed=args.seed,
            num_samples=args.samples,
            device=args.device,
            dtype=dtype,
        )
        _print_summary(batch)

        # Exit code policy:
        #   - 0: the tool ran successfully (even if cex found)
        #   - 1: runtime error
        # If you want CI to fail on any cex, you can change to:
        #   return 2 if batch.any_cex else 0
        return 0
    except Exception as e:
        logger.exception("❌ Level1 check failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
