# ===- act/back_end/moe/factory.py - Controlled MoE Model Factory ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Sequence

import torch
import torch.nn as nn

from act.back_end.moe.model import OutputLevelMoE
from act.back_end.moe.schema import (
    GateKind,
    OutputLevelMoEProgram,
    OutputLevelMoESpec,
)


@dataclass(frozen=True)
class OutputMoEFactoryConfig:
    input_shape: Sequence[int]
    num_classes: int
    num_experts: int = 4
    top_k: int = 1
    gate: GateKind = GateKind.HARD_TOP1
    router_hidden: Sequence[int] = ()
    expert_hidden: Sequence[int] = (64,)
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.input_shape or any(int(dim) <= 0 for dim in self.input_shape):
            raise ValueError("input_shape must contain positive dimensions")
        if self.num_classes < 1:
            raise ValueError("num_classes must be positive")


def _mlp(input_dim: int, hidden: Sequence[int], output_dim: int) -> nn.Sequential:
    widths = [int(input_dim), *(int(v) for v in hidden), int(output_dim)]
    layers: list[nn.Module] = [nn.Flatten(start_dim=1)]
    for index, (in_features, out_features) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(in_features, out_features))
        if index + 1 < len(widths) - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def build_output_moe(config: OutputMoEFactoryConfig) -> OutputLevelMoE:
    """Build a deterministic small ReLU MoE suitable for ACT/HyZor studies."""
    spec = OutputLevelMoESpec(
        num_experts=config.num_experts,
        top_k=config.top_k,
        gate=config.gate,
        normalized=config.gate != GateKind.SWITCH_PROB,
    )
    input_dim = int(prod(int(dim) for dim in config.input_shape))
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(config.seed))
        router = _mlp(input_dim, config.router_hidden, config.num_experts)
        experts = [
            _mlp(input_dim, config.expert_hidden, config.num_classes)
            for _ in range(config.num_experts)
        ]
    return OutputLevelMoE(router, experts, spec)


def load_output_moe_checkpoint(
    path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[OutputLevelMoE, dict]:
    """Load a checkpoint emitted by ``python -m act.pipeline.moe``."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != "act-output-moe-v1":
        raise ValueError("unsupported ACT MoE checkpoint format")
    raw = dict(payload["factory_config"])
    raw["gate"] = GateKind(raw["gate"])
    config = OutputMoEFactoryConfig(**raw)
    model = build_output_moe(config)
    model.load_state_dict(payload["state_dict"])
    return model, payload


def build_act_moe_program(
    model: OutputLevelMoE,
    *,
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    output_spec,
) -> OutputLevelMoEProgram:
    """Convert a concrete MoE into separate router/expert ACT component Nets.

    The dynamic MoE itself is deliberately not traced as one graph.  Each
    component receives the same BOX input specification, while routed experts
    receive the real output property.  The router gets a zero linear property
    that exists only to satisfy ACT's wrapper invariant; Route A reads its
    pre-ASSERT HZ directly.
    """
    from act.front_end.spec_creator_base import LabeledInputTensor
    from act.front_end.specs import InputSpec, InKind, OutKind, OutputSpec
    from act.front_end.verifiable_model import (
        InputLayer,
        InputSpecLayer,
        OutputSpecLayer,
        VerifiableModel,
    )
    from act.pipeline.verification.torch2act import TorchToACT

    if center.shape != lower.shape or center.shape != upper.shape:
        raise ValueError("center/lower/upper shapes must match")
    if center.ndim < 2 or center.shape[0] != 1:
        raise ValueError("Route A component conversion currently requires batch size 1")
    if bool((lower > upper).any()):
        raise ValueError("input lower bound exceeds upper bound")

    labeled = LabeledInputTensor(center.detach().clone(), label=None)
    input_spec = InputSpec(
        kind=InKind.BOX,
        lb=lower.detach().clone(),
        ub=upper.detach().clone(),
    )

    def convert(component: nn.Module, component_output_spec: OutputSpec):
        wrapper = VerifiableModel(
            input_layer=InputLayer(
                labeled,
                shape=tuple(center.shape),
                dtype=center.dtype,
            ),
            input_spec=InputSpecLayer(input_spec),
            model=component,
            output_spec=OutputSpecLayer(component_output_spec),
        )
        return TorchToACT(wrapper, sample_input=center).run()

    router_spec = OutputSpec(
        kind=OutKind.LINEAR_LE,
        c=torch.zeros(1, model.spec.num_experts, dtype=center.dtype),
        d=torch.zeros(1, dtype=center.dtype),
    )
    router = convert(model.router, router_spec)

    class _AbsorbedSharedExpert(nn.Module):
        def __init__(self, routed: nn.Module, shared: nn.Module) -> None:
            super().__init__()
            self.routed = routed
            self.shared = shared

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.routed(x) + self.shared(x)

    components = (
        tuple(model.experts)
        if model.shared_expert is None
        else tuple(
            _AbsorbedSharedExpert(expert, model.shared_expert)
            for expert in model.experts
        )
    )
    experts = tuple(convert(expert, output_spec) for expert in components)
    return OutputLevelMoEProgram(
        spec=model.spec,
        router=router,
        experts=experts,
        shared_expert=None,
    )
