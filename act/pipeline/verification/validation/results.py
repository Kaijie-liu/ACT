# act/pipeline/verification/validation/results.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List


def hash_tensor(x) -> str:
    import torch  # lazy import

    x_cpu = x.detach().to("cpu").contiguous()
    return hashlib.sha256(x_cpu.numpy().tobytes()).hexdigest()


@dataclass
class Counterexample:
    input_index: int
    input_hash: str
    min_val: float
    max_val: float
    pred: int
    true_label: int
    kind: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    case_name: str
    status: str  # CERTIFIED | COUNTEREXAMPLE_FOUND | ERROR
    counterexamples: List[Counterexample]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        allowed = {"CERTIFIED", "COUNTEREXAMPLE_FOUND", "ERROR"}
        if self.status not in allowed:
            raise ValueError(f"Invalid status {self.status}; must be in {allowed}")
