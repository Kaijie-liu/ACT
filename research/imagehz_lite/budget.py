"""Phase 0 budget machinery — fail-closed when limits exceeded.

Per §9R-3 of the prototype plan: when ReLU aux generator count would
exceed the configured budget, raise `BudgetExceeded`. The driver records
the event and treats the iid as "REPRESENTATION BUDGET EXCEEDED" rather
than silently folding generators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


class BudgetExceeded(RuntimeError):
    """Raised when a Phase 0 representation budget is exhausted.

    Attributes:
        kind      : 'relu_aux' for Phase 0
        layer_id  : the operator layer index where the budget tripped
        used      : count of aux generators that would have been spent
        cap       : configured cap
    """

    def __init__(self, kind: str, layer_id: int, used: int, cap: int):
        super().__init__(
            f"ImageHZ-lite Phase 0 budget exceeded: kind={kind} "
            f"layer={layer_id} would_use={used} cap={cap}"
        )
        self.kind = kind
        self.layer_id = layer_id
        self.used = used
        self.cap = cap


@dataclass
class Budget:
    """Track per-instance Phase 0 counters and enforce caps."""

    max_relu_aux_per_image: int = 10_000_000
    relu_aux_spent: int = 0
    fail_closed_events: List[dict] = field(default_factory=list)

    def spend_relu_aux(self, layer_id: int, n: int) -> None:
        if n < 0:
            raise ValueError(f"Budget.spend_relu_aux n must be >= 0, got {n}")
        if self.relu_aux_spent + n > self.max_relu_aux_per_image:
            event = {
                "kind": "relu_aux",
                "layer_id": layer_id,
                "would_use_total": self.relu_aux_spent + n,
                "cap": self.max_relu_aux_per_image,
            }
            self.fail_closed_events.append(event)
            raise BudgetExceeded(
                kind="relu_aux",
                layer_id=layer_id,
                used=self.relu_aux_spent + n,
                cap=self.max_relu_aux_per_image,
            )
        self.relu_aux_spent += n
