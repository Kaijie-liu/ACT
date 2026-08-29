# ===- act/pipeline/moe/certificate_constants.py ----------------------====#
"""Auditable constant providers for the ICML 2025 analytic MoE formula.

This module deliberately separates *computing a number* from establishing a
formal upper bound.  In particular, sampled gradients are diagnostics and a
hard argmax router is not made Lipschitz by assigning it the local value zero.

The implemented norm is the input/output ``L_inf`` norm used by the official
CIFAR experiments.  Products of induced operator-norm upper bounds are sound,
although usually loose.  The helpers are independent of the author's source
tree so that a reproduction can record every choice made by ACT.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn


class ConstantProvider(str, Enum):
    SOUND_GLOBAL_SPECTRAL = "SOUND_GLOBAL_SPECTRAL"
    EMPIRICAL_GRADIENT_SAMPLED = "EMPIRICAL_GRADIENT_SAMPLED"
    AUTHOR_UNSPECIFIED = "AUTHOR_UNSPECIFIED"


class OutputReading(str, Enum):
    PROBABILITY = "probability"
    RAW_LOGIT = "raw_logit"


class RouterReading(str, Enum):
    CONTINUOUS_SOFTMAX = "continuous_softmax"
    HARD_ARGMAX = "hard_argmax"


class ConstantStatus(str, Enum):
    FORMAL_BOUND = "FORMAL_BOUND"
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
    NOT_FORMALLY_INSTANTIATED = "NOT_FORMALLY_INSTANTIATED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class ScalarConstant:
    value: float | None
    provider: ConstantProvider
    status: ConstantStatus
    quantity: str
    detail: str

    @property
    def formal(self) -> bool:
        return self.status == ConstantStatus.FORMAL_BOUND


@dataclass(frozen=True)
class Theorem54Constants:
    """Per-expert constants for one represented output class ``y``."""

    router_lipschitz: tuple[ScalarConstant, ...]
    expert_lipschitz: tuple[ScalarConstant, ...]
    expert_output_upper: tuple[ScalarConstant, ...]
    output_reading: OutputReading
    router_reading: RouterReading

    def __post_init__(self) -> None:
        sizes = {
            len(self.router_lipschitz),
            len(self.expert_lipschitz),
            len(self.expert_output_upper),
        }
        if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
            raise ValueError("Theorem 5.4 constants must cover the same experts")

    @property
    def formal(self) -> bool:
        return all(
            item.formal
            for group in (
                self.router_lipschitz,
                self.expert_lipschitz,
                self.expert_output_upper,
            )
            for item in group
        )


@dataclass(frozen=True)
class PaperFormulaEvaluation:
    radius: float | None
    denominator: float | None
    clean_margin: float
    status: ConstantStatus
    detail: str


def _finite_nonnegative(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def linear_linf_operator_norm(layer: nn.Linear) -> float:
    """Return the exact induced ``L_inf -> L_inf`` norm of a linear layer."""

    weight = layer.weight.detach().double()
    return float(weight.abs().sum(dim=1).max().item())


def linear_linf_row_norms(layer: nn.Linear) -> tuple[float, ...]:
    weight = layer.weight.detach().double()
    return tuple(float(v) for v in weight.abs().sum(dim=1).tolist())


def conv2d_linf_operator_upper(layer: nn.Conv2d) -> float:
    """Sound induced ``L_inf`` upper bound for a convolution.

    Padding and boundary effects can only remove terms from a receptive field,
    so the largest absolute kernel row sum is a global upper bound.  PyTorch's
    grouped-convolution weight already contains only the connected channels.
    """

    weight = layer.weight.detach().double()
    row_sums = weight.abs().flatten(start_dim=1).sum(dim=1)
    return float(row_sums.max().item())


def batchnorm_linf_operator_norm(layer: nn.modules.batchnorm._BatchNorm) -> float:
    if layer.training:
        raise ValueError("BatchNorm must be in eval mode for a fixed affine bound")
    running_var = layer.running_var
    if running_var is None:
        raise ValueError("BatchNorm without running statistics is unsupported")
    if layer.affine:
        assert layer.weight is not None
        scale = layer.weight.detach().double().abs()
    else:
        scale = torch.ones_like(running_var, dtype=torch.double)
    scale = scale / torch.sqrt(running_var.detach().double() + layer.eps)
    return float(scale.max().item())


def module_linf_lipschitz_upper(module: nn.Module) -> float:
    """Return a sound global ``L_inf`` Lipschitz upper bound.

    Only modules with an explicit compositional rule are accepted.  Refusing an
    unknown graph is intentional: silently treating it as a sequential module
    would turn a diagnostic into an invalid certificate.
    """

    if isinstance(module, nn.Linear):
        return linear_linf_operator_norm(module)
    if isinstance(module, nn.Conv2d):
        return conv2d_linf_operator_upper(module)
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return batchnorm_linf_operator_norm(module)
    if isinstance(
        module,
        (
            nn.ReLU,
            nn.LeakyReLU,
            nn.Flatten,
            nn.Identity,
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.MaxPool3d,
            nn.AvgPool1d,
            nn.AvgPool2d,
            nn.AvgPool3d,
            nn.AdaptiveAvgPool1d,
            nn.AdaptiveAvgPool2d,
            nn.AdaptiveAvgPool3d,
        ),
    ):
        if isinstance(module, nn.LeakyReLU):
            return max(1.0, abs(float(module.negative_slope)))
        return 1.0
    if isinstance(module, nn.Dropout):
        if module.training:
            raise ValueError("Dropout must be in eval mode")
        return 1.0
    if isinstance(module, nn.Sequential):
        value = 1.0
        for child in module:
            value *= module_linf_lipschitz_upper(child)
        return _finite_nonnegative(value, name="sequential Lipschitz bound")
    raise TypeError(
        "unsupported module graph for a sound compositional bound: "
        f"{type(module).__module__}.{type(module).__qualname__}"
    )


def residual_linf_lipschitz_upper(
    main_path: nn.Module,
    shortcut: nn.Module | None = None,
) -> float:
    """Bound ``main_path(x) + shortcut(x)`` by the sum of path bounds."""

    main = module_linf_lipschitz_upper(main_path)
    skip = 1.0 if shortcut is None else module_linf_lipschitz_upper(shortcut)
    return _finite_nonnegative(main + skip, name="residual Lipschitz bound")


def _official_basic_block_upper(block: nn.Module) -> float:
    required = ("conv1", "bn1", "conv2", "bn2", "shortcut")
    if not all(hasattr(block, name) for name in required):
        raise TypeError("module is not an official-code BasicBlock-shaped object")
    main = (
        module_linf_lipschitz_upper(block.conv1)
        * module_linf_lipschitz_upper(block.bn1)
        * module_linf_lipschitz_upper(block.conv2)
        * module_linf_lipschitz_upper(block.bn2)
    )
    shortcut = module_linf_lipschitz_upper(block.shortcut)
    return _finite_nonnegative(main + shortcut, name="BasicBlock bound")


def official_cifar_resnet18_logit_bounds(
    model: nn.Module,
) -> tuple[float, tuple[float, ...]]:
    """Bound the official-code CIFAR ResNet18 vector and individual logits.

    The adapter is structural rather than nominal: it accepts only the graph
    layout audited in ``TIML-Group/Robust-MoE-Dual-Model/models/resnet.py``.
    Its average-pool and ReLU operations are nonexpansive in ``L_inf``.
    """

    required = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "linear")
    if not all(hasattr(model, name) for name in required):
        raise TypeError("model does not match the audited official CIFAR ResNet")
    if model.training:
        raise ValueError("official ResNet must be in eval mode")
    prefix = (
        module_linf_lipschitz_upper(model.conv1)
        * module_linf_lipschitz_upper(model.bn1)
    )
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(model, layer_name)
        if not isinstance(layer, nn.Sequential):
            raise TypeError(f"{layer_name} must be nn.Sequential")
        for block in layer:
            prefix *= _official_basic_block_upper(block)
    rows = tuple(prefix * value for value in linear_linf_row_norms(model.linear))
    vector = max(rows)
    return (
        _finite_nonnegative(vector, name="ResNet vector bound"),
        tuple(_finite_nonnegative(v, name="ResNet row bound") for v in rows),
    )


def sound_probability_expert_constant(
    logit_vector_lipschitz: float,
    *,
    expert_index: int,
) -> tuple[ScalarConstant, ScalarConstant]:
    """Return ``(L_Ri, M_Ri)`` for a softmax probability output.

    A scalar softmax coordinate has ``L_inf -> abs`` Lipschitz constant at most
    1/2 because the L1 norm of its logit gradient is
    ``2 p_i (1-p_i) <= 1/2``.  Its value lies in ``[0, 1]``.
    """

    logit_vector_lipschitz = _finite_nonnegative(
        logit_vector_lipschitz, name="logit vector Lipschitz bound"
    )
    provider = ConstantProvider.SOUND_GLOBAL_SPECTRAL
    return (
        ScalarConstant(
            0.5 * logit_vector_lipschitz,
            provider,
            ConstantStatus.FORMAL_BOUND,
            f"L_R[{expert_index}]",
            "softmax-coordinate 1/2 bound composed with global logit bound",
        ),
        ScalarConstant(
            1.0,
            provider,
            ConstantStatus.FORMAL_BOUND,
            f"M_R[{expert_index}]",
            "softmax probability is globally in [0,1]",
        ),
    )


def raw_logit_output_upper_unspecified(*, expert_index: int) -> ScalarConstant:
    return ScalarConstant(
        None,
        ConstantProvider.AUTHOR_UNSPECIFIED,
        ConstantStatus.NOT_FORMALLY_INSTANTIATED,
        f"M_R[{expert_index}]",
        "raw logits have no disclosed global M_R <= 1 bound",
    )


def sound_softmax_router_constants(
    logit_vector_lipschitz: float,
    *,
    num_experts: int,
) -> tuple[ScalarConstant, ...]:
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    value = 0.5 * _finite_nonnegative(
        logit_vector_lipschitz, name="router-logit Lipschitz bound"
    )
    return tuple(
        ScalarConstant(
            value,
            ConstantProvider.SOUND_GLOBAL_SPECTRAL,
            ConstantStatus.FORMAL_BOUND,
            f"r_R[{index}]",
            "softmax-coordinate 1/2 bound composed with router-logit bound",
        )
        for index in range(num_experts)
    )


def hard_argmax_router_constants(*, num_experts: int) -> tuple[ScalarConstant, ...]:
    """Reject a global hard-router Lipschitz constant at reachable ties."""

    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    return tuple(
        ScalarConstant(
            None,
            ConstantProvider.SOUND_GLOBAL_SPECTRAL,
            ConstantStatus.NOT_APPLICABLE,
            f"r_R[{index}]",
            "hard argmax is discontinuous at a reachable routing tie",
        )
        for index in range(num_experts)
    )


def empirical_gradient_linf_estimates(
    module: nn.Module,
    samples: torch.Tensor,
) -> tuple[ScalarConstant, ...]:
    """Sample output-gradient L1 norms, explicitly as non-formal diagnostics."""

    if samples.ndim < 2 or samples.shape[0] == 0:
        raise ValueError("samples must contain a non-empty batch")
    was_training = module.training
    module.eval()
    estimates: torch.Tensor | None = None
    try:
        for sample in samples.detach():
            point = sample.unsqueeze(0).clone().detach().requires_grad_(True)
            output = module(point)
            if output.ndim != 2 or output.shape[0] != 1:
                raise ValueError("module must return [batch, outputs]")
            current = []
            for index in range(output.shape[1]):
                gradient = torch.autograd.grad(
                    output[0, index],
                    point,
                    retain_graph=index + 1 < output.shape[1],
                )[0]
                current.append(gradient.detach().abs().sum())
            vector = torch.stack(current).cpu().double()
            estimates = vector if estimates is None else torch.maximum(estimates, vector)
    finally:
        module.train(was_training)
    assert estimates is not None
    return tuple(
        ScalarConstant(
            float(value),
            ConstantProvider.EMPIRICAL_GRADIENT_SAMPLED,
            ConstantStatus.DIAGNOSTIC_ONLY,
            f"sampled_gradient_L[{index}]",
            "maximum sampled gradient L1 norm; not an upper bound",
        )
        for index, value in enumerate(estimates.tolist())
    )


def author_unspecified_constants(
    *,
    num_experts: int,
    quantity_prefix: str,
) -> tuple[ScalarConstant, ...]:
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    return tuple(
        ScalarConstant(
            None,
            ConstantProvider.AUTHOR_UNSPECIFIED,
            ConstantStatus.NOT_FORMALLY_INSTANTIATED,
            f"{quantity_prefix}[{index}]",
            "paper gives no computation protocol or numerical value",
        )
        for index in range(num_experts)
    )


def evaluate_theorem54_paper_formula(
    output: Sequence[float],
    predicted_class: int,
    routing_weights: Sequence[float],
    constants: Theorem54Constants,
) -> PaperFormulaEvaluation:
    """Evaluate Equation (8) with provenance-preserving status semantics.

    This is explicitly a reimplementation of the paper formula, not author
    certificate code.  A non-formal provider never yields a formal result.
    """

    values = tuple(float(v) for v in output)
    if not values or not 0 <= predicted_class < len(values):
        raise ValueError("predicted_class is outside the output vector")
    if len(values) < 2:
        raise ValueError("classification output must contain at least two classes")
    weights = tuple(float(v) for v in routing_weights)
    if len(weights) != len(constants.router_lipschitz):
        raise ValueError("routing weight count differs from constants")
    if any(not math.isfinite(v) or v < 0.0 for v in weights):
        raise ValueError("routing weights must be finite and nonnegative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
        return PaperFormulaEvaluation(
            None,
            None,
            math.nan,
            ConstantStatus.NOT_APPLICABLE,
            "Equation (8) assumes normalized routing weights",
        )
    runner_up = max(v for index, v in enumerate(values) if index != predicted_class)
    margin = values[predicted_class] - runner_up
    all_constants = (
        *constants.router_lipschitz,
        *constants.expert_lipschitz,
        *constants.expert_output_upper,
    )
    if any(item.value is None for item in all_constants):
        status = (
            ConstantStatus.NOT_APPLICABLE
            if any(item.status == ConstantStatus.NOT_APPLICABLE for item in all_constants)
            else ConstantStatus.NOT_FORMALLY_INSTANTIATED
        )
        return PaperFormulaEvaluation(
            None,
            None,
            margin,
            status,
            "one or more required constants are unavailable",
        )
    denominator = sum(
        float(router.value) * float(upper.value)
        + weight * float(expert.value)
        for router, upper, weight, expert in zip(
            constants.router_lipschitz,
            constants.expert_output_upper,
            weights,
            constants.expert_lipschitz,
        )
    )
    denominator = _finite_nonnegative(denominator, name="formula denominator")
    status = (
        ConstantStatus.FORMAL_BOUND
        if constants.formal
        else ConstantStatus.DIAGNOSTIC_ONLY
    )
    if margin <= 0.0:
        return PaperFormulaEvaluation(
            0.0,
            denominator,
            margin,
            status,
            "clean classification margin is nonpositive",
        )
    if denominator == 0.0:
        return PaperFormulaEvaluation(
            math.inf,
            denominator,
            margin,
            status,
            "paper formula denominator is zero",
        )
    return PaperFormulaEvaluation(
        margin / denominator,
        denominator,
        margin,
        status,
        "author-paper Equation (8) reimplementation",
    )
