#!/usr/bin/env python3
#===- act/pipeline/confignet/verdicts.py - Verdict Types --------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class Verdict(str, Enum):
    CERTIFIED = "CERTIFIED"
    FALSIFIED = "FALSIFIED"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"


@dataclass(frozen=True)
class GatingInfo:
    """
    Explains any post-processing / policy overrides applied to the base verdict.
    """
    prohibited_certified_due_to_cex: bool = False
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prohibited_certified_due_to_cex": bool(self.prohibited_certified_due_to_cex),
            "reason": self.reason,
            "details": self.details or {},
        }
