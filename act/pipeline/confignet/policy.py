#!/usr/bin/env python3
#===- act/pipeline/confignet/policy.py - ConfigNet Policy --------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .verdicts import Verdict, GatingInfo


def apply_policy(
    *,
    base_verdict: Verdict,
    level1_found_cex: bool,
    level1_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[Verdict, GatingInfo]:
    """
    Central policy:
      - If Level 1 found concrete counterexample, the instance is FALSIFIED.
      - CERTIFIED must never be returned when concrete CEX exist.
    """
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
