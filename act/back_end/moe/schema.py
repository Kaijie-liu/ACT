# ===- act/back_end/moe/schema.py - MoE Program-Level IR --------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

from act.back_end.core import Net
from act.back_end.layer_schema import LayerKind


class GateKind(str, Enum):
    """Concrete output-level gate semantics supported by the MoE factory."""

    HARD_TOP1 = "hard_top1"
    SELECTED_SOFTMAX = "selected_softmax"
    NORMALIZED_SIGMOID = "normalized_sigmoid"
    SWITCH_PROB = "switch_prob"


class TiePolicy(str, Enum):
    """Verification semantics for equal router scores."""

    ANY_LEGAL_TOPK = "any_legal_topk"


@dataclass(frozen=True)
class OutputLevelMoESpec:
    """Semantic configuration for a final/output-level MoE block.

    Gate elimination is only applied when the MoE output is consumed directly
    by the output property, optionally through an affine map.  This IR does not
    claim gate elimination for intermediate MoE layers.
    """

    num_experts: int
    top_k: int = 1
    gate: GateKind = GateKind.HARD_TOP1
    tie_policy: TiePolicy = TiePolicy.ANY_LEGAL_TOPK
    normalized: bool = True
    nonnegative: bool = True
    affine_decoder: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate", GateKind(self.gate))
        object.__setattr__(self, "tie_policy", TiePolicy(self.tie_policy))
        if self.num_experts < 1:
            raise ValueError("num_experts must be positive")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must lie in [1, num_experts]")
        if self.gate == GateKind.HARD_TOP1 and self.top_k != 1:
            raise ValueError("hard_top1 requires top_k=1")
        if self.gate == GateKind.SWITCH_PROB and self.top_k != 1:
            raise ValueError("switch_prob requires top_k=1")
        if self.gate == GateKind.SWITCH_PROB and self.normalized:
            raise ValueError("switch_prob is an unnormalized selected-expert scale")
        if self.gate in {
            GateKind.SELECTED_SOFTMAX,
            GateKind.NORMALIZED_SIGMOID,
        } and not self.normalized:
            raise ValueError(f"{self.gate.value} is normalized by definition")

    @property
    def hard_routing(self) -> bool:
        return self.gate == GateKind.HARD_TOP1

    @property
    def gate_elimination_applicable(self) -> bool:
        """Whether arbitrary convex output sets may use gate elimination."""
        return self.affine_decoder and self.nonnegative and self.normalized


def _entry_width(net: Net) -> int:
    inputs = [layer for layer in net.layers if layer.kind == LayerKind.INPUT.value]
    if len(inputs) != 1:
        raise ValueError(f"component Net must have one INPUT layer, got {len(inputs)}")
    return len(inputs[0].out_vars)


def _output_width(net: Net) -> int:
    assertion = net.last_validation()
    if assertion is None:
        raise ValueError("component Net must end in an ASSERT layer")
    return len(assertion.in_vars)


@dataclass
class OutputLevelMoEProgram:
    """Program-level ACT IR composed of one router Net and per-expert Nets.

    Components remain separate ACT graphs on purpose: Route A never
    instantiates every expert's ReLU binaries in one monolithic model.
    """

    spec: OutputLevelMoESpec
    router: Net
    experts: Sequence[Net]
    shared_expert: Optional[Net] = None

    def __post_init__(self) -> None:
        self.experts = tuple(self.experts)
        if len(self.experts) != self.spec.num_experts:
            raise ValueError(
                f"expected {self.spec.num_experts} expert Nets, got {len(self.experts)}"
            )
        if _output_width(self.router) != self.spec.num_experts:
            raise ValueError("router output width must equal num_experts")
        input_width = _entry_width(self.router)
        output_widths = set()
        for index, expert in enumerate(self.experts):
            if _entry_width(expert) != input_width:
                raise ValueError(f"expert {index} input width differs from router")
            output_widths.add(_output_width(expert))
        if len(output_widths) != 1:
            raise ValueError("all experts must have the same output width")
        if self.shared_expert is not None:
            if not self.spec.normalized:
                raise ValueError("shared expert absorption requires normalized gates")
            if _entry_width(self.shared_expert) != input_width:
                raise ValueError("shared expert input width differs from router")
            if _output_width(self.shared_expert) not in output_widths:
                raise ValueError("shared expert output width differs from routed experts")

    @property
    def input_width(self) -> int:
        return _entry_width(self.router)

    @property
    def output_width(self) -> int:
        return _output_width(self.experts[0])
