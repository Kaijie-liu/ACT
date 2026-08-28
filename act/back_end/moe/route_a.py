# ===- act/back_end/moe/route_a.py - ACT/HyZor Route-A Engine ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from act.back_end.analyze import analyze
from act.back_end.core import ConSet, Fact
from act.back_end.hybridz_tf import HybridzTF
from act.back_end.layer_schema import LayerKind
from act.back_end.moe.hz_routing import CandidateReport, HZ, analyze_candidates
from act.back_end.moe.model import OutputLevelMoE
from act.back_end.moe.schema import OutputLevelMoEProgram
from act.back_end.moe.verifier import verify_output_gate_elimination
from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.transfer_functions import (
    get_solver_mode,
    get_transfer_function,
    set_solver_mode,
    set_transfer_function,
)
from act.back_end.verifier import (
    add_all_input_specs,
    find_entry_layer_id,
    gather_input_spec_layers,
    get_assert_layer,
    get_input_ids,
    seed_from_input_specs,
    verify_once,
)
from act.front_end.specs import OutKind, OutputSpec
from act.util.stats import VerifyResult, VerifyStatus


@dataclass(frozen=True)
class RouterAnalysis:
    router_hz: HZ
    input_hz: HZ
    candidates: CandidateReport


@dataclass(frozen=True)
class RouteAVerificationReport:
    result: VerifyResult
    router: RouterAnalysis
    expert_results: tuple[tuple[int, VerifyResult], ...]


def _component_output_hz(net, tf: HybridzTF) -> HZ:
    assertion = get_assert_layer(net)
    output_ids = net.preds.get(assertion.id, [])
    if len(output_ids) != 1:
        raise ValueError("component ASSERT must have exactly one predecessor")
    output_id = output_ids[0]
    return tf.get_sparse_hz(output_id) or tf.get_hz(output_id) or _missing_hz()


def _missing_hz():
    raise RuntimeError("HybridzTF did not retain an HZ at the component output")


def _router_and_input_hz(net, tf: HybridzTF) -> tuple[HZ, HZ]:
    router_hz = _component_output_hz(net, tf)
    specs = gather_input_spec_layers(net)
    if isinstance(router_hz, SparseHZono):
        for layer in reversed(specs):
            candidate = tf.get_sparse_hz(layer.id)
            if (
                candidate is not None
                and candidate.frame_id == router_hz.frame_id
            ):
                return router_hz, candidate
        raise RuntimeError("router sparse HZ lost its correlated input frame")
    for layer in reversed(specs):
        candidate = tf.get_hz(layer.id)
        if candidate is not None:
            return router_hz, candidate
    raise RuntimeError("router dense HZ has no correlated input state")


def _analyze_router(net, tf: HybridzTF) -> tuple[HZ, HZ]:
    entry_id = find_entry_layer_id(net)
    specs = gather_input_spec_layers(net)
    seed = seed_from_input_specs(specs)
    if seed.lb.shape[0] != 1:
        raise ValueError("Route A currently supports one verification lane at a time")
    entry = Fact(bounds=seed, cons=ConSet())
    add_all_input_specs(entry.cons, get_input_ids(net), specs)
    analyze(net, entry_id, entry)
    return _router_and_input_hz(net, tf)


def _output_spec_from_net(net) -> OutputSpec:
    assertion = get_assert_layer(net)

    def unbatch(value: Any) -> Any:
        if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == 1:
            return value[0]
        return value

    return OutputSpec(
        kind=assertion.params.get("kind"),
        c=unbatch(assertion.params.get("c")),
        d=unbatch(assertion.params.get("d")),
        y_true=assertion.params.get("y_true"),
        margin=unbatch(assertion.params.get("margin")),
        lb=unbatch(assertion.params.get("lb")),
        ub=unbatch(assertion.params.get("ub")),
    )


def _model_device_dtype(model: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(model.parameters(), None)
    if parameter is None:
        return torch.device("cpu"), torch.get_default_dtype()
    return parameter.device, parameter.dtype


def _router_net_is_exact(net) -> bool:
    exact_kinds = {
        LayerKind.INPUT.value,
        LayerKind.INPUT_SPEC.value,
        LayerKind.ASSERT.value,
        LayerKind.DENSE.value,
        LayerKind.BIAS.value,
        LayerKind.SCALE.value,
        LayerKind.RELU.value,
        LayerKind.LRELU.value,
        LayerKind.FLATTEN.value,
        LayerKind.RESHAPE.value,
        LayerKind.TRANSPOSE.value,
        LayerKind.SLICE.value,
        LayerKind.GATHER.value,
        LayerKind.EXPAND.value,
        LayerKind.CONCAT.value,
        LayerKind.ADD.value,
        LayerKind.SUB.value,
    }
    return all(layer.kind in exact_kinds for layer in net.layers)


class RouteAEngine:
    """End-to-end output-level route-conditioned verifier for ACT components."""

    def __init__(
        self,
        program: OutputLevelMoEProgram,
        *,
        concrete_model: OutputLevelMoE | None = None,
        expert_models: Sequence[torch.nn.Module] | None = None,
        time_limit_per_route: float = 30.0,
        router_exact: bool | None = None,
        hybridz_config=None,
        property_is_convex_cone: bool = False,
    ) -> None:
        self.program = program
        self.concrete_model = concrete_model
        self.expert_models = tuple(expert_models) if expert_models is not None else ()
        if self.expert_models and len(self.expert_models) != program.spec.num_experts:
            raise ValueError("expert_models count differs from program")
        self.time_limit_per_route = float(time_limit_per_route)
        self.router_exact = router_exact
        self.hybridz_config = hybridz_config
        self.property_is_convex_cone = bool(property_is_convex_cone)

    def run(self) -> RouteAVerificationReport:
        previous_solver = get_solver_mode()
        try:
            previous_tf = get_transfer_function()
        except RuntimeError:
            previous_tf = None
        tf = HybridzTF(config=self.hybridz_config)
        set_transfer_function(tf)
        set_solver_mode("hybridz")
        expert_results: list[tuple[int, VerifyResult]] = []
        try:
            router_hz, input_hz = _analyze_router(self.program.router, tf)
            representation_exact = (
                router_hz.exact
                if isinstance(router_hz, SparseHZono)
                else (
                    _router_net_is_exact(self.program.router)
                    if self.router_exact is None
                    else bool(self.router_exact)
                )
            )
            candidates = analyze_candidates(
                router_hz,
                self.program.spec.top_k,
                input_hz=input_hz,
                time_limit_per_expert=self.time_limit_per_route,
                router_exact=representation_exact,
            )

            def verify_expert(index, branch) -> VerifyResult:
                if branch.guarded_input is None:
                    result = VerifyResult(
                        VerifyStatus.UNKNOWN,
                        metadata={"source": "moe_route_a", "reason": "missing_guarded_input"},
                    )
                else:
                    tf.set_entry_hz(branch.guarded_input)
                    try:
                        model_fn = self.expert_models[index] if self.expert_models else None
                        verified = verify_once(
                            self.program.experts[index],
                            model_fn=model_fn,
                            timelimit=self.time_limit_per_route,
                        )
                        if len(verified) != 1:
                            raise ValueError("Route A expert verification requires one lane")
                        result = verified[0]
                    finally:
                        tf.clear_entry_hz()
                expert_results.append((index, result))
                return result

            def validate_counterexample(index: int, result: VerifyResult) -> bool:
                if self.concrete_model is None or result.counterexample is None:
                    return False
                device, dtype = _model_device_dtype(self.concrete_model)
                value = result.counterexample.unsqueeze(0).to(device=device, dtype=dtype)
                with torch.no_grad():
                    output, route = self.concrete_model.forward_with_routing(value)
                if int(route.indices[0, 0].item()) != int(index):
                    return False
                output_spec = _output_spec_from_net(self.program.experts[index])
                violated, _ = output_spec.violation(output)
                return bool(violated[0].item())

            aggregate = verify_output_gate_elimination(
                self.program.spec,
                candidates,
                verify_expert,
                validate_counterexample=validate_counterexample,
                property_is_convex=all(
                    get_assert_layer(net).params.get("kind") != OutKind.UNSAFE_LINEAR
                    for net in self.program.experts
                ),
                property_is_convex_cone=self.property_is_convex_cone,
            )
            return RouteAVerificationReport(
                result=aggregate,
                router=RouterAnalysis(router_hz, input_hz, candidates),
                expert_results=tuple(expert_results),
            )
        finally:
            tf.clear_entry_hz()
            set_solver_mode(previous_solver)
            if previous_tf is not None:
                set_transfer_function(previous_tf)
