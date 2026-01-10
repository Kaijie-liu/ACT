#!/usr/bin/env python3
#===- act/pipeline/confignet/driver_levels.py - ConfigNet Driver -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from .schema import ConfigNetConfig
from .level1_check import run_level1_suite
from .jsonl import make_record
from .policy import apply_policy
from .verdicts import Verdict


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
    """
    PR1 scope:
      - sample instances (ConfigNet)
      - run level1 suite (returns per-instance record payloads)
      - apply policy -> final_verdict + gating
      - emit JSONL records with schema v1
    """
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
